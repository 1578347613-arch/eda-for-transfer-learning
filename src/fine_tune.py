# src/fine_tune.py
import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data_loader import get_data_and_scalers
from models import MLP, DualHeadMLP

OPAMP_TYPE = '5t_opamp'
BATCH_SIZE = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS_BIAS = 10       # 仅调bias
EPOCHS_HEAD = 60       # 仅训练B头
EPOCHS_UNFREEZE = 50   # 解冻主干末层
PATIENCE = 20          # 早停耐心
LR_BIAS = 3e-4
LR_HEAD = 1e-4
LR_UNFREEZE = 5e-5
WEIGHT_DECAY = 1e-4
L2SP_LAMBDA = 1e-4     # 让微调参数别离预训练太远

def copy_from_single_head_to_dualhead(single: MLP, dual: DualHeadMLP):
    # 把 single.model 的最后一层当作 head_A，之前的堆叠当作 backbone
    # 复制 backbone
    with torch.no_grad():
        # backbone: single.model[:-1]
        for layer_dual, layer_single in zip(dual.backbone, single.model[:-1]):
            if isinstance(layer_dual, nn.Linear) and isinstance(layer_single, nn.Linear):
                layer_dual.weight.copy_(layer_single.weight)
                layer_dual.bias.copy_(layer_single.bias)
            elif isinstance(layer_dual, nn.LayerNorm) and isinstance(layer_single, nn.LayerNorm):
                layer_dual.weight.copy_(layer_single.weight)
                layer_dual.bias.copy_(layer_single.bias)
            # GELU/Dropout无参数

        # head_A <- single.model[-1]
        last = single.model[-1]
        dual.head_A.weight.copy_(last.weight)
        dual.head_A.bias.copy_(last.bias)
        # head_B 初始 = head_A
        dual.head_B.weight.copy_(dual.head_A.weight)
        dual.head_B.bias.copy_(dual.head_A.bias)

def l2sp_regularizer(model: DualHeadMLP, pretrained_state, scale=1e-4):
    """L2-SP: sum (theta - theta*)^2，仅作用于被训练的参数。"""
    loss = 0.0
    for n, p in model.named_parameters():
        if p.requires_grad and n in pretrained_state:
            loss = loss + ((p - pretrained_state[n].to(p.device))**2).sum()
    return scale * loss

def make_loader(arrX, arrY, batch_size, shuffle):
    tensX = torch.tensor(arrX, dtype=torch.float32)
    tensY = torch.tensor(arrY, dtype=torch.float32)
    return DataLoader(TensorDataset(tensX, tensY), batch_size=batch_size, shuffle=shuffle)

def eval_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred, multioutput='raw_values')
    mae = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
    r2  = np.array([r2_score(y_true[:,i], y_pred[:,i]) for i in range(y_true.shape[1])])
    return mse, mae, r2

def run_epoch(model, loader, optimizer, loss_fn, phase, pretrained_state=None):
    is_train = optimizer is not None
    model.train(is_train)
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb, domain='B')  # 微调面向B头
        loss = loss_fn(pred, yb)
        if is_train and pretrained_state is not None:
            loss = loss + l2sp_regularizer(model, pretrained_state, scale=L2SP_LAMBDA)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += loss.item()
    return total / len(loader)

def main():
    # 载数据
    data = get_data_and_scalers(opamp_type=OPAMP_TYPE)
    X_src, y_src     = data['source']
    X_tr,  y_tr      = data['target_train']
    X_val, y_val     = data['target_val']

    train_loader = make_loader(X_tr, y_tr, BATCH_SIZE, shuffle=True)
    val_loader   = make_loader(X_val, y_val, BATCH_SIZE, shuffle=False)

    input_dim, output_dim = X_src.shape[1], y_src.shape[1]

    # 1) 加载A域预训练权重
    single = MLP(input_dim, output_dim, hidden_dim=512, num_layers=6, dropout_rate=0.1).to(DEVICE)
    ckpt_path = f'results/{OPAMP_TYPE}_baseline_model.pth'
    single.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

    # 2) 构造双头模型并拷贝权重
    model  = DualHeadMLP(input_dim, output_dim, hidden_dim=512, num_layers=6, dropout_rate=0.1).to(DEVICE)
    copy_from_single_head_to_dualhead(single, model)
    pretrained_state = copy.deepcopy(model.state_dict())  # L2-SP 参照

    # 损失函数：Huber 更稳健
    loss_fn = nn.SmoothL1Loss(beta=1.0)

    best_val = float('inf'); best_state=None; patience=PATIENCE

    # ---------- Phase 1: 仅校准 B 头 bias ----------
    for p in model.parameters():
        p.requires_grad = False
    model.head_B.bias.requires_grad = True
    opt = torch.optim.AdamW([model.head_B.bias], lr=LR_BIAS, weight_decay=0.0)

    for epoch in range(EPOCHS_BIAS):
        tr = run_epoch(model, train_loader, opt, loss_fn, 'bias', pretrained_state=None)
        vl = run_epoch(model, val_loader, None, loss_fn, 'val', pretrained_state=None)
        print(f'[Bias] Epoch {epoch+1}/{EPOCHS_BIAS} | train {tr:.4f} | val {vl:.4f}')
        if vl < best_val: best_val, best_state, patience = vl, copy.deepcopy(model.state_dict()), PATIENCE
        else:
            patience -= 1
            if patience==0: break

    # ---------- Phase 2: 仅训练 B 头 (权重+偏置) ----------
    for p in model.parameters():
        p.requires_grad = False
    for p in model.head_B.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.head_B.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)

    for epoch in range(EPOCHS_HEAD):
        tr = run_epoch(model, train_loader, opt, loss_fn, 'head', pretrained_state=pretrained_state)
        vl = run_epoch(model, val_loader, None, loss_fn, 'val', pretrained_state=None)
        print(f'[Head] Epoch {epoch+1}/{EPOCHS_HEAD} | train {tr:.4f} | val {vl:.4f}')
        if vl < best_val: best_val, best_state, patience = vl, copy.deepcopy(model.state_dict()), PATIENCE
        else:
            patience -= 1
            if patience==0: break

    # ---------- Phase 3: 解冻主干最后一层 (更强适配) ----------
    # 解冻最后一个 Linear + LN + GELU 前的 Linear（微调幅度小）
    for p in model.parameters():
        p.requires_grad = False
    # 解冻主干最后一个 Linear 层参数
    linear_layers = [m for m in model.backbone if isinstance(m, nn.Linear)]
    for p in linear_layers[-1].parameters():
        p.requires_grad = True
    for p in model.head_B.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR_UNFREEZE, weight_decay=WEIGHT_DECAY)

    patience = PATIENCE
    for epoch in range(EPOCHS_UNFREEZE):
        tr = run_epoch(model, train_loader, opt, loss_fn, 'unfreeze', pretrained_state=pretrained_state)
        vl = run_epoch(model, val_loader, None, loss_fn, 'val', pretrained_state=None)
        print(f'[Unfreeze] Epoch {epoch+1}/{EPOCHS_UNFREEZE} | train {tr:.4f} | val {vl:.4f}')
        if vl < best_val: best_val, best_state, patience = vl, copy.deepcopy(model.state_dict()), PATIENCE
        else:
            patience -= 1
            if patience==0: break

    # 保存最佳
    os.makedirs('results', exist_ok=True)
    torch.save(best_state, f'results/{OPAMP_TYPE}_dualhead_finetuned.pth')
    print(f'微调完成！最佳验证损失: {best_val:.4f} 已保存。')

if __name__ == "__main__":
    main()
