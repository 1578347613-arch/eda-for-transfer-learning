# training/train_align_coral.py
import os
import copy
import torch
from torch.utils.data import DataLoader, TensorDataset
from data_loader import get_data_and_scalers
from models import MLP, AlignHeteroMLP
from losses import heteroscedastic_nll, batch_r2, coral_loss
import config

# --- 1. 使用配置文件中的超参数 ---
OPAMP_TYPE = config.OPAMP_TYPE
BATCH_B = config.BATCH_SIZE
BATCH_A = config.BATCH_SIZE
LR = config.LEARNING_RATE
WD = 1e-4  # 可以根据需要调整
EPOCHS = config.EPOCHS
PATIENCE = config.PATIENCE
LAMBDA_CORAL = config.LAMBDA_CORAL
ALPHA_R2 = config.ALPHA_R2

DEVICE = torch.device(config.DEVICE)

def make_loader(x, y, bs, shuffle=True, drop_last=False):
    """
    创建 DataLoader
    """
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=shuffle, drop_last=drop_last)

def main():
    # 加载数据
    data = get_data_and_scalers(opamp_type=OPAMP_TYPE)
    X_src, y_src = data['source']
    X_trg_tr, y_trg_tr = data['target_train']
    X_trg_val, y_trg_val = data['target_val']

    # 初始化 baseline 模型
    baseline = MLP(input_dim=X_src.shape[1], output_dim=y_src.shape[1], hidden_dim=512, num_layers=6, dropout_rate=0.1).to(DEVICE)
    ckpt = f'results/{OPAMP_TYPE}_baseline_model.pth'
    baseline.load_state_dict(torch.load(ckpt, map_location=DEVICE))

    # 初始化 AlignHeteroMLP 模型
    model = AlignHeteroMLP(input_dim=X_src.shape[1], output_dim=y_src.shape[1], hidden_dim=512, num_layers=6, dropout_rate=0.1).to(DEVICE)
    model.backbone.load_state_dict(baseline.state_dict())

    # 优化器
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    # 创建数据加载器
    dl_B = make_loader(X_trg_tr, y_trg_tr, BATCH_B, shuffle=True)
    dl_A = make_loader(X_src, y_src, BATCH_A, shuffle=True)
    dl_A_iter = iter(dl_A)
    dl_val = make_loader(X_trg_val, y_trg_val, BATCH_B, shuffle=False)

    best_val = float('inf')
    patience = PATIENCE

    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for xb_B, yb_B in dl_B:
            xb_B, yb_B = xb_B.to(DEVICE), yb_B.to(DEVICE)

            def next_A_like_B():
                nonlocal dl_A_iter
                try:
                    xa, ya = next(dl_A_iter)
                except StopIteration:
                    dl_A_iter = iter(dl_A)
                    xa, ya = next(dl_A_iter)
                if xa.size(0) != xb_B.size(0):
                    xa = xa[:xb_B.size(0)]
                return xa.to(DEVICE)

            xa1 = next_A_like_B()
            xa2 = next_A_like_B()

            mu_B, logvar_B, feat_B = model(xb_B)
            nll = heteroscedastic_nll(mu_B, logvar_B, yb_B, reduction='mean')
            r2_vec = batch_r2(yb_B, mu_B)
            r2_loss = (1.0 - r2_vec.clamp(min=-1.0, max=1.0)).mean()

            with torch.no_grad():
                _, _, feat_A1 = model(xa1)
                _, _, feat_A2 = model(xa2)

            coral = 0.5 * (coral_loss(feat_A1, feat_B) + coral_loss(feat_A2, feat_B))

            loss = nll + ALPHA_R2 * r2_loss + LAMBDA_CORAL * coral
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                mu, logvar, _ = model(xb)
                nll = heteroscedastic_nll(mu, logvar, yb, reduction='mean')
                val_loss += nll.item()

        val_loss /= len(dl_val)

        print(f"[λ={LAMBDA_CORAL:.3f}] Epoch {epoch+1}/{EPOCHS} | train {total/len(dl_B):.4f} | valNLL {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = PATIENCE
            os.makedirs('results', exist_ok=True)
            torch.save(best_state, f'results/{OPAMP_TYPE}_align_hetero_lambda{LAMBDA_CORAL:.3f}.pth')
        else:
            patience -= 1
            if patience == 0:
                break

    print(f"完成：最佳 valNLL={best_val:.4f} 已保存。")

if __name__ == "__main__":
    main()
