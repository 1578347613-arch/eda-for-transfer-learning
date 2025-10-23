# src_new/training/train_target_only.py
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data_loader import get_data_and_scalers
from losses.loss_function import heteroscedastic_nll, batch_r2
from models.align_hetero import AlignHeteroMLP
import config

# 路径：results 与 src_new 同级
PROJECT_DIR = Path(__file__).resolve().parent.parent   # .../src_new
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 设备 & 超参（统一走 config，提供兜底）
DEVICE      = torch.device(getattr(config, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
OPAMP_TYPE  = getattr(config, "OPAMP_TYPE", "5t_opamp")
BATCH_SIZE  = getattr(config, "BATCH_SIZE", 256)
LR          = getattr(config, "LR", 1e-4)
WEIGHT_DECAY= getattr(config, "WEIGHT_DECAY", 1e-4)
EPOCHS      = getattr(config, "EPOCHS", 80)
PATIENCE    = getattr(config, "PATIENCE", 20)
ALPHA_R2    = getattr(config, "ALPHA_R2", 1e-3)
SEED        = getattr(config, "SEED", 42)
GRAD_CLIP   = getattr(config, "GRAD_CLIP", 1.0)

def set_seed(seed:int=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_loader(x: np.ndarray,
                y: np.ndarray,
                bs: int,
                shuffle: bool = True,
                drop_last: bool = False) -> DataLoader:
    """创建与其它训练脚本一致的 DataLoader 接口。"""
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    ds  = TensorDataset(x_t, y_t)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=drop_last)

def run_epoch(model: torch.nn.Module,
              loader: DataLoader,
              optimizer: torch.optim.Optimizer | None,
              phase: str = "train") -> Tuple[float, float]:
    """
    单个 epoch：返回 (avg_nll, avg_r2loss)
    - train：反向 + 更新
    - val：   仅前向
    """
    is_train = (optimizer is not None) and (phase == "train")
    model.train(is_train)

    total_nll = 0.0
    total_r2l = 0.0
    n_batches = max(1, len(loader))

    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        with torch.set_grad_enabled(is_train):
            mu, logv, _ = model(xb)
            nll = heteroscedastic_nll(mu, logv, yb, reduction="mean")
            r2l = (1.0 - batch_r2(yb, mu)).mean()  # R² 转为损失
            loss = nll + ALPHA_R2 * r2l

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if GRAD_CLIP and GRAD_CLIP > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

        total_nll += float(nll.detach().cpu().item())
        total_r2l += float(r2l.detach().cpu().item())

    return total_nll / n_batches, total_r2l / n_batches

def main():
    set_seed(SEED)

    # 1) 载入 Target 域数据（已 log1p + 标准化）
    data = get_data_and_scalers(opamp_type=OPAMP_TYPE)
    Xtr, Ytr = data["target_train"]
    Xva, Yva = data["target_val"]

    input_dim  = Xtr.shape[1]
    output_dim = Ytr.shape[1]

    # 2) 构建 Target-only 异方差模型（结构细节走 config；构造签名对齐当前实现）
    model = AlignHeteroMLP(input_dim, output_dim).to(DEVICE)

    # 3) 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 4) DataLoader
    train_loader = make_loader(Xtr, Ytr, BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader   = make_loader(Xva, Yva, BATCH_SIZE, shuffle=False, drop_last=False)

    # 5) 训练循环 + 早停（按验证 NLL）
    best_val = float("inf")
    best_state = None
    patience = PATIENCE

    ckpt_path = RESULTS_DIR / f"{OPAMP_TYPE}_target_only_hetero.pth"
    print(f"[Target-only] Start training on {DEVICE} | save to: {ckpt_path}")

    for ep in range(1, EPOCHS + 1):
        tr_nll, tr_r2l = run_epoch(model, train_loader, optimizer, phase="train")
        va_nll, va_r2l = run_epoch(model, val_loader,   optimizer=None, phase="val")

        print(f"[Target-only][{ep:03d}/{EPOCHS}] "
              f"train: NLL={tr_nll:.4f} R2L={tr_r2l:.4f} | "
              f"val: NLL={va_nll:.4f} R2L={va_r2l:.4f}")

        if va_nll < best_val:
            best_val = va_nll
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = PATIENCE
            torch.save(best_state, ckpt_path)
        else:
            patience -= 1
            if patience <= 0:
                print(f"[Target-only] Early stop at epoch {ep}. Best val NLL: {best_val:.4f}")
                break

    print(f"[Target-only] 最佳权重已保存: {ckpt_path}")

if __name__ == "__main__":
    main()
