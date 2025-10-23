# fine_tune/fine_tune.py
import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_loader import get_data_and_scalers
from models.mlp import MLP
from models.dual_head_mlp import DualHeadMLP, copy_from_single_head_to_dualhead, l2sp_regularizer

import config

# 微调模型的训练过程
def run_epoch(model, loader, optimizer, loss_fn, phase, pretrained_state=None):
    is_train = optimizer is not None
    model.train(is_train)
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
        pred = model(xb, domain='B')
        loss = loss_fn(pred, yb)
        if is_train and pretrained_state is not None:
            loss += l2sp_regularizer(model, pretrained_state, scale=config.L2SP_LAMBDA)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += loss.item()
    return total / len(loader)

def main():
    # 载入数据
    data = get_data_and_scalers(opamp_type=config.OPAMP_TYPE)
    X_src, y_src = data['source']
    X_tr, y_tr = data['target_train']
    X_val, y_val = data['target_val']

    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)),
                              batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)),
                            batch_size=config.BATCH_SIZE, shuffle=False)

    # 加载预训练的模型
    single = MLP(X_src.shape[1], y_src.shape[1], hidden_dim=config.HIDDEN_DIM, num_layers=config.NUM_LAYERS).to(config.DEVICE)
    single.load_state_dict(torch.load(f'results/{config.OPAMP_TYPE}_baseline_model.pth', map_location=config.DEVICE))

    # 构建双头模型
    model = DualHeadMLP(X_src.shape[1], y_src.shape[1], hidden_dim=config.HIDDEN_DIM, num_layers=config.NUM_LAYERS).to(config.DEVICE)
    copy_from_single_head_to_dualhead(single, model)
    pretrained_state = copy.deepcopy(model.state_dict())

    # 损失函数
    loss_fn = nn.SmoothL1Loss(beta=1.0)

    best_val_loss = float('inf')
    patience = config.PATIENCE

    # 微调各个阶段：校准 B 头 bias
    for p in model.parameters():
        p.requires_grad = False
    model.head_B.bias.requires_grad = True
    opt = torch.optim.AdamW([model.head_B.bias], lr=config.LR_BIAS, weight_decay=0.0)

    for epoch in range(config.EPOCHS_BIAS):
        train_loss = run_epoch(model, train_loader, opt, loss_fn, 'bias', pretrained_state=None)
        val_loss = run_epoch(model, val_loader, None, loss_fn, 'val', pretrained_state=None)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = config.PATIENCE
        else:
            patience -= 1
            if patience == 0:
                break

    # 训练 B 头权重
    for p in model.parameters():
        p.requires_grad = False
    for p in model.head_B.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.head_B.parameters(), lr=config.LR_HEAD, weight_decay=config.WEIGHT_DECAY)

    for epoch in range(config.EPOCHS_HEAD):
        train_loss = run_epoch(model, train_loader, opt, loss_fn, 'head', pretrained_state=pretrained_state)
        val_loss = run_epoch(model, val_loader, None, loss_fn, 'val', pretrained_state=None)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = config.PATIENCE
        else:
            patience -= 1
            if patience == 0:
                break

    # 解冻主干最后一层
    for p in model.parameters():
        p.requires_grad = False
    linear_layers = [m for m in model.backbone if isinstance(m, nn.Linear)]
    for p in linear_layers[-1].parameters():
        p.requires_grad = True
    for p in model.head_B.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=config.LR_UNFREEZE, weight_decay=config.WEIGHT_DECAY)

    for epoch in range(config.EPOCHS_UNFREEZE):
        train_loss = run_epoch(model, train_loader, opt, loss_fn, 'unfreeze', pretrained_state=pretrained_state)
        val_loss = run_epoch(model, val_loader, None, loss_fn, 'val', pretrained_state=None)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = config.PATIENCE
        else:
            patience -= 1
            if patience == 0:
                break

    # 保存最佳模型
    os.makedirs('results', exist_ok=True)
    torch.save(best_state, f'results/{config.OPAMP_TYPE}_dualhead_finetuned.pth')
    print(f'微调完成！最佳验证损失: {best_val_loss:.4f} 已保存。')

if __name__ == "__main__":
    main()
