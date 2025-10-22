# src/train_align_coral.py
import os, copy, math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_loader import get_data_and_scalers
from models import MLP, AlignHeteroMLP, load_backbone_from_trained_mlp
from losses import heteroscedastic_nll, batch_r2, coral_loss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参（可用命令行参数替换，这里先写死/易改）
OPAMP_TYPE     = '5t_opamp'
BATCH_B        = 256            # 每步B域监督 batch
BATCH_A        = 256            # 每步A域对齐 batch（会取两次，组成 2:1）
LR             = 1e-4
WD             = 1e-4
EPOCHS         = 80
PATIENCE       = 20
LAMBDA_CORAL   = 0.05           # 扫描 {0.02, 0.05, 0.1}
ALPHA_R2       = 1e-3           # R² 辅助项的权重（小）

def make_loader(x, y, bs, shuffle, drop_last=False):
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=shuffle, drop_last=drop_last)

def main():
    data = get_data_and_scalers(opamp_type=OPAMP_TYPE)
    X_src, y_src          = data['source']        # A（只用于对齐，不监督）
    X_trg_tr, y_trg_tr    = data['target_train']  # B train（用于监督）
    X_trg_val, y_trg_val  = data['target_val']    # B val（用于早停&评估）

    # 加载 A 上预训练的基线权重
    baseline = MLP(input_dim=X_src.shape[1], output_dim=y_src.shape[1],
                   hidden_dim=512, num_layers=6, dropout_rate=0.1).to(DEVICE)
    ckpt = f'results/{OPAMP_TYPE}_baseline_model.pth'
    baseline.load_state_dict(torch.load(ckpt, map_location=DEVICE))

    # 构造对齐模型（共享主干+异方差B头），并从baseline拷贝主干&初始化
    model = AlignHeteroMLP(input_dim=X_src.shape[1], output_dim=y_src.shape[1],
                           hidden_dim=512, num_layers=6, dropout_rate=0.1).to(DEVICE)
    load_backbone_from_trained_mlp(baseline, model)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    dl_B = make_loader(X_trg_tr, y_trg_tr, BATCH_B, shuffle=True,  drop_last=True)
    dl_A = make_loader(X_src,    y_src,    BATCH_A, shuffle=True,  drop_last=True)
    dl_A_iter = iter(dl_A)

    dl_val = make_loader(X_trg_val, y_trg_val, BATCH_B, shuffle=False, drop_last=False)

    best_val = float('inf'); best_state = None; patience = PATIENCE

    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for xb_B, yb_B in dl_B:
            xb_B, yb_B = xb_B.to(DEVICE), yb_B.to(DEVICE)

            # 2:1 混采：取两个A批次，与当前B批次做对齐（只参与CORAL）
            def next_A_like_B():
                nonlocal dl_A_iter
                try:
                    xa, ya = next(dl_A_iter)
                except StopIteration:
                    dl_A_iter = iter(dl_A)
                    xa, ya = next(dl_A_iter)
                # 采样同样大小以稳定协方差
                if xa.size(0) != xb_B.size(0):
                    xa = xa[:xb_B.size(0)]
                return xa.to(DEVICE)

            xa1 = next_A_like_B()
            xa2 = next_A_like_B()

            # 前向：B域监督
            mu_B, logvar_B, feat_B = model(xb_B)
            nll = heteroscedastic_nll(mu_B, logvar_B, yb_B, reduction='mean')

            # R² 辅助（越大越好 → 加入 (1-R²) 的小权重）
            r2_vec = batch_r2(yb_B, mu_B)          # [D]
            r2_loss = (1.0 - r2_vec.clamp(min=-1.0, max=1.0)).mean()

            # 对齐：两次A批次与B批次分别做 CORAL，然后取平均
            with torch.no_grad():
                _, _, feat_A1 = model(xa1)  # A域仅取特征，不回传监督
                _, _, feat_A2 = model(xa2)
            coral = 0.5 * (coral_loss(feat_A1, feat_B) + coral_loss(feat_A2, feat_B))

            loss = nll + ALPHA_R2 * r2_loss + LAMBDA_CORAL * coral

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += loss.item()

        # 验证（仅B监督损失，便于可比性）
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
