# training/train_align_coral.py
import os
import copy
import torch
from torch.utils.data import DataLoader, TensorDataset

from data_loader import get_data_and_scalers
from models.mlp import MLP
from models.dual_head_mlp import DualHeadMLP, copy_from_single_head_to_dualhead, l2sp_regularizer

from losses.loss_function import heteroscedastic_nll, batch_r2, coral_loss
import config

# ========== 全局超参数（来自 config） ==========
OPAMP_TYPE   = config.OPAMP_TYPE
BATCH_B      = config.BATCH_B
BATCH_A      = config.BATCH_A
LR           = config.LEARNING_RATE
WD           = 1e-4
EPOCHS       = config.EPOCHS
PATIENCE     = config.PATIENCE
LAMBDA_CORAL = config.LAMBDA_CORAL
ALPHA_R2     = config.ALPHA_R2
DEVICE       = torch.device(config.DEVICE)

# ========== DataLoader ==========
def make_loader(x, y, bs, shuffle=True, drop_last=False):
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=shuffle, drop_last=drop_last)

# ========== 稳妥的 checkpoint 加载器 ==========
def _remap_and_filter_for_mlp(ckpt_state: dict, target_state: dict):
    """
    1) 统一键名：去掉 module./backbone. 前缀；把 model.* 改成 network.*
    2) 仅保留与目标键名相同且 shape 完全一致的参数（避免 size mismatch）
    """
    remapped = {}
    for k, v in ckpt_state.items():
        if not isinstance(v, torch.Tensor):
            continue
        nk = k
        if nk.startswith("module."):
            nk = nk[7:]
        if nk.startswith("backbone."):
            nk = nk[9:]
        if nk.startswith("model."):
            nk = "network." + nk[6:]  # model.0.weight -> network.0.weight
        remapped[nk] = v

    compatible = {k: v for k, v in remapped.items()
                  if (k in target_state) and (v.shape == target_state[k].shape)}
    return compatible, remapped

def _probe_hidden_dim_from_state(state: dict):
    """尽力从 ckpt 的首层 Linear 推断 hidden_dim"""
    for probe_key in ("network.0.weight", "model.0.weight"):
        t = state.get(probe_key, None)
        if isinstance(t, torch.Tensor) and t.ndim == 2:
            return t.shape[0]   # [hidden_dim, input_dim]
    return None

def load_baseline_into_mlp(baseline: MLP, ckpt_path: str, device: torch.device):
    """
    稳妥加载 baseline：
    - 自动处理前缀/命名（module./backbone./model.）
    - 仅加载 shape 兼容的层
    - 若发现 ckpt 的 hidden_dim 与当前 baseline 不一致，则给出明确提示
    """
    raw = torch.load(ckpt_path, map_location=device)

    # 兼容可能的 { "state_dict": {...} } 结构
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        raw = raw["state_dict"]

    # 推断 ckpt 的 hidden_dim
    ckpt_hidden = _probe_hidden_dim_from_state(raw)

    # 当前 baseline 的 hidden_dim
    try:
        curr_hidden = baseline.network[0].out_features  # Linear(out_features=hidden_dim)
    except Exception:
        curr_hidden = None

    if (ckpt_hidden is not None) and (curr_hidden is not None) and (ckpt_hidden != curr_hidden):
        raise RuntimeError(
            f"[Baseline] hidden_dim 不一致：ckpt={ckpt_hidden}, 当前={curr_hidden}。\n"
            f"请把 config.HIDDEN_DIM 设为 {ckpt_hidden}，并确认 config.NUM_LAYERS 与 ckpt 相同（常见为 4）。"
        )

    compatible, remapped = _remap_and_filter_for_mlp(raw, baseline.state_dict())
    res = baseline.load_state_dict(compatible, strict=False)

    print(f"[Baseline] 加载 {ckpt_path}")
    print(f"  - 兼容加载层数: {len(compatible)}")
    if res.missing_keys:
        print(f"  - PyTorch missing_keys（信息）：{res.missing_keys}")
    if res.unexpected_keys:
        print(f"  - PyTorch unexpected_keys（信息）：{res.unexpected_keys}")

# ========== 训练入口 ==========
def main():
    # 1) 加载并标准化数据
    data = get_data_and_scalers(opamp_type=OPAMP_TYPE)
    X_src,    y_src    = data['source']
    X_trg_tr, y_trg_tr = data['target_train']
    X_trg_val, y_trg_val = data['target_val']

    # 2) 构建 baseline（结构与 ckpt 对齐：hidden_dim/num_layers 来自 config）
    baseline = MLP(
        input_dim=X_src.shape[1],
        output_dim=y_src.shape[1],
        hidden_dim=config.HIDDEN_DIM,     # 需与 ckpt 一致：常见 256
        num_layers=config.NUM_LAYERS,     # 需与 ckpt 一致：常见 4
        dropout_rate=config.DROPOUT_RATE  # 如 0.1
    ).to(DEVICE)

    # 尝试从两个常见位置找 ckpt
    candidates = [
        f"results/{OPAMP_TYPE}_baseline_model.pth",
        f"../results/{OPAMP_TYPE}_baseline_model.pth",
    ]
    ckpt = None
    for p in candidates:
        if os.path.exists(p):
            ckpt = p
            break
    if ckpt is None:
        raise FileNotFoundError(
            f"未找到 baseline ckpt，请确认存在以下任一文件：\n  - {candidates[0]}\n  - {candidates[1]}"
        )

    # 稳妥加载 baseline 参数到 baseline MLP
    load_baseline_into_mlp(baseline, ckpt, DEVICE)

    # 3) 构建 AlignHeteroMLP，并把 baseline 权重拷到 backbone
    model = AlignHeteroMLP(
        input_dim=X_src.shape[1],
        output_dim=y_src.shape[1]
    ).to(DEVICE)

    # 两个结构完全一致，严格加载
    model.backbone.load_state_dict(baseline.state_dict(), strict=True)

    # 4) 优化器
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    # 5) DataLoader
    dl_B   = make_loader(X_trg_tr, y_trg_tr, BATCH_B, shuffle=True)
    dl_A   = make_loader(X_src,    y_src,    BATCH_A, shuffle=True)
    dl_val = make_loader(X_trg_val, y_trg_val, BATCH_B, shuffle=False)
    dl_A_iter = iter(dl_A)

    best_val = float('inf')
    patience = PATIENCE

    # 6) 训练循环
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

            # B 域前向 & 损失
            mu_B, logvar_B, feat_B = model(xb_B)
            nll = heteroscedastic_nll(mu_B, logvar_B, yb_B, reduction='mean')

            r2_vec  = batch_r2(yb_B, mu_B)
            r2_loss = (1.0 - r2_vec.clamp(min=-1.0, max=1.0)).mean()

            # A 域特征（不回传梯度）
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

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                mu, logvar, _ = model(xb)
                nll = heteroscedastic_nll(mu, logvar, yb, reduction='mean')
                val_loss += nll.item()

        val_loss /= len(dl_val)

        print(f"[λ={LAMBDA_CORAL:.3f}] Epoch {epoch+1:03d}/{EPOCHS} | "
              f"train {total/len(dl_B):.4f} | valNLL {val_loss:.4f}")

        # 早停与保存
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = PATIENCE
            os.makedirs('results', exist_ok=True)
            save_path = f'results/{OPAMP_TYPE}_align_hetero_lambda{LAMBDA_CORAL:.3f}.pth'
            torch.save(best_state, save_path)
        else:
            patience -= 1
            if patience == 0:
                break

    print(f"完成：最佳 valNLL={best_val:.4f} 已保存。")

if __name__ == "__main__":
    main()
