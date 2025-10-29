# src/train_target_only.py

from pathlib import Path
from typing import Tuple
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data_loader import get_data_and_scalers
from loss_function import heteroscedastic_nll, batch_r2
from models.align_hetero import AlignHeteroMLP
import config

# --- 路径定义 ---
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent # <-- 获取父目录 (项目根目录)
RESULTS_DIR = PROJECT_ROOT / "results" # <-- 指向正确的 results 目录！
# RESULTS_DIR.mkdir(parents=True, exist_ok=True) # <-- 这行可以删掉或注释掉

# --- 辅助函数 ---
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_loader(x: np.ndarray, y: np.ndarray, bs: int, shuffle: bool, drop_last: bool) -> DataLoader:
    ds = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=drop_last)

def run_epoch(model, loader, optimizer, alpha_r2, device, phase="train"):
    is_train = (optimizer is not None) and (phase == "train")
    model.train(is_train)
    total_nll, total_r2l, n_batches = 0.0, 0.0, max(1, len(loader))
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.set_grad_enabled(is_train):
            mu, logv, _ = model(xb)
            nll = heteroscedastic_nll(mu, logv, yb, reduction="mean")
            r2l = (1.0 - batch_r2(yb, mu)).mean()
            loss = nll + alpha_r2 * r2l
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_nll += nll.item()
        total_r2l += r2l.item()
    return total_nll / n_batches, total_r2l / n_batches


# ========== 参数定义与解析 (适配 litian 的 config.py) ==========
def setup_args():
    """设置和解析命令行参数 (Target-Only 专属版)"""
    parser = argparse.ArgumentParser(description="Target-Only 训练脚本 (适配扁平 config)")

    # --- 核心参数 (读取 config.py 默认值) ---
    parser.add_argument("--opamp", type=str,
                        default=config.OPAMP_TYPE, help="运放类型") # <-- 和 train.py 共用 OPAMP_TYPE
    parser.add_argument("--device", type=str,
                        default=config.DEVICE, help="设备 'cuda' or 'cpu'") # <-- 和 train.py 共用 DEVICE
    parser.add_argument("--restart", action='store_true',
                        default=False, help="强制重新执行训练 (即使模型已存在)") # <-- restart 默认 False
    # save_path 在脚本后面处理，这里不用加
    parser.add_argument("--seed", type=int, default=42, help="随机种子") # <-- 和 train.py 共用 seed (如果 config.py 里有的话，没有就用42)

    # --- Target-Only 训练专属参数 (读取 config.py 默认值) ---
    # (我们假设 Target-Only 和 Fine-tune 用类似的设置)
    parser.add_argument("--epochs", type=int,
                        default=config.EPOCHS_FINETUNE, help="训练轮数")
    parser.add_argument("--patience", type=int,
                        default=config.PATIENCE_FINETUNE, help="早停耐心轮数")
    parser.add_argument("--lr", type=float,
                        default=config.LEARNING_RATE_FINETUNE, help="学习率")
    parser.add_argument("--batch_size", type=int,
                        default=config.BATCH_B, help="批次大小 (使用 B 域的 batch size)")
    parser.add_argument("--alpha_r2", type=float,
                        default=config.ALPHA_R2, help="R2 损失的权重")

    # --- 模型结构参数 (读取 config.py 默认值) ---
    parser.add_argument("--hidden_dim", type=int,
                        default=config.HIDDEN_DIM, help="隐藏层维度")
    parser.add_argument("--num_layers", type=int,
                        default=config.NUM_LAYERS, help="隐藏层数量")
    parser.add_argument("--dropout_rate", type=float,
                        default=config.DROPOUT_RATE, help="Dropout 比率")

    args = parser.parse_args()
    return args


# --- 主训练函数 ---
def main():
    args = setup_args()

    device = torch.device(args.device)
    set_seed(args.seed)

    # 超参映射（lr 优先取 lr_finetune，避免 args.lr 不存在）
    epochs     = args.epochs
    patience   = args.patience
    lr         = args.lr
    batch_size = args.batch_size
    alpha_r2   = args.alpha_r2

    data = get_data_and_scalers(opamp_type=args.opamp)
    Xtr, Ytr = data["target_train"]
    Xva, Yva = data["target_val"]
    input_dim, output_dim = Xtr.shape[1], Ytr.shape[1]

    model = AlignHeteroMLP(
        input_dim=input_dim, output_dim=output_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate
    ).to(device)

    optimizer    = torch.optim.AdamW(model.parameters(), lr=lr)
    train_loader = make_loader(Xtr, Ytr, batch_size, shuffle=True,  drop_last=True)
    val_loader   = make_loader(Xva, Yva, batch_size, shuffle=False, drop_last=False)

    ckpt_path = RESULTS_DIR / f"{args.opamp}_target_only.pth"
    print(f"[Target-Only] opamp: {args.opamp}, device: {device}, saving to: {ckpt_path.name}")

    # ========== 默认跳过：存在 ckpt 且未指定 --restart 时直接退出 ==========
    if ckpt_path.exists() and not args.restart:
        try:
            state = torch.load(ckpt_path, map_location=device)
            state_dict = state.get("state_dict", state)
            model.load_state_dict(state_dict)
            va_nll0, _ = run_epoch(model, val_loader, None, alpha_r2, device, "val")
            print(f"[Target-Only] 检测到已有 ckpt（{ckpt_path.name}）。按默认策略跳过训练并退出。"
                  f"当前 Val NLL={va_nll0:.4f}")
        except Exception as e:
            print(f"[Target-Only] 发现 ckpt 但载入失败（{e}）。将从头训练。")
        else:
            return

    # 若指定 --restart，删除旧 ckpt（若存在），然后从头训练
    if args.restart and ckpt_path.exists():
        try:
            ckpt_path.unlink()
            print("`--restart` 指定：已删除旧 checkpoint，将从头训练。")
        except Exception as e:
            print(f"删除旧 checkpoint 失败（忽略继续）：{e}")

    best_val_nll = float("inf")
    patience_counter = patience

    for ep in range(1, epochs + 1):
        tr_nll, tr_r2l = run_epoch(model, train_loader, optimizer, alpha_r2, device, "train")
        va_nll, va_r2l = run_epoch(model, val_loader, None,      alpha_r2, device, "val")
        print(f"[Target-Only][{ep:03d}/{epochs}] Train NLL={tr_nll:.4f} | Val NLL={va_nll:.4f}")

        if va_nll < best_val_nll:
            best_val_nll = va_nll
            torch.save({"state_dict": model.state_dict()}, ckpt_path)
            patience_counter = patience
            print("  -> New best model saved.")
        else:
            patience_counter -= 1
            if patience_counter <= 0:
                print(f"Early stopping at epoch {ep}.")
                break

    print(f"\n[Target-Only] Finished. Best model at: {ckpt_path}")

if __name__ == "__main__":
    main()
