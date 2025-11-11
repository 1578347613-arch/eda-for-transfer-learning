# src/train_target_only.py

from pathlib import Path
from typing import Tuple
import argparse
import ast

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data_loader import get_data_and_scalers
from loss_function import heteroscedastic_nll, batch_r2
from models.align_hetero import AlignHeteroMLP
from config import COMMON_CONFIG, TASK_CONFIGS

# --- 路径定义 ---
SRC_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SRC_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --- 通用工具函数 ---

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(x: np.ndarray, y: np.ndarray, bs: int,
                shuffle: bool, drop_last: bool) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=drop_last)


def run_epoch(model,
              loader,
              optimizer,
              alpha_r2: float,
              device,
              phase: str = "train"):
    """
    单个 epoch 的训练或验证。
    使用 heteroscedastic_nll + alpha_r2 * (1 - R2) 作为损失。
    """
    is_train = (optimizer is not None) and (phase == "train")
    model.train(is_train)

    total_nll = 0.0
    total_r2l = 0.0
    n_batches = max(1, len(loader))

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


# --- 关键：适配 AlignHeteroMLP 的 hidden_dims（支持新旧写法） ---

def _coerce_hidden_dims_from_args(args):
    """
    统一把配置转成 AlignHeteroMLP 需要的 hidden_dims: List[int]

    支持三种情况（两个工艺都适用）：
    1) config / 命令行提供 hidden_dims 为 list: [256, 256, 256]
    2) hidden_dims 为字符串: "[256, 256, 256]"
    3) 旧版本: 提供 hidden_dim + num_layers -> 展开为 [hidden_dim] * num_layers
    """
    hd = getattr(args, "hidden_dims", None)

    # case 1 / 2: hidden_dims 存在
    if isinstance(hd, str):
        try:
            hd_parsed = ast.literal_eval(hd)
            if isinstance(hd_parsed, (list, tuple)):
                return list(int(x) for x in hd_parsed)
        except Exception:
            pass  # 解析失败则继续用下面的逻辑兜底
    elif isinstance(hd, (list, tuple)):
        return list(int(x) for x in hd)

    # case 3: 兼容旧配置
    hidden_dim = getattr(args, "hidden_dim", None)
    num_layers = getattr(args, "num_layers", None)
    if hidden_dim is not None and num_layers is not None:
        return [int(hidden_dim)] * int(num_layers)

    # 兜底：避免因为配置缺失直接炸掉（一般用不到）
    print("[Target-Only][WARN] 未找到 hidden_dims / (hidden_dim + num_layers)，使用默认 [256, 256, 256].")
    return [256, 256, 256]


# --- 主训练函数 ---

def main():
    parser = argparse.ArgumentParser(description="Target-Only 训练脚本")
    parser.add_argument(
        "--opamp", type=str, required=True,
        choices=TASK_CONFIGS.keys()
    )
    parser.add_argument(
        "--restart", action="store_true",
        help="忽略已有 checkpoint，从头训练"
    )

    # 先解析拿到 opamp，再灌入默认值
    temp_args, _ = parser.parse_known_args()
    parser.set_defaults(**COMMON_CONFIG)
    if temp_args.opamp in TASK_CONFIGS:
        parser.set_defaults(**TASK_CONFIGS[temp_args.opamp])

    args = parser.parse_args()

    device = torch.device(args.device)
    set_seed(args.seed)

    # --- 超参映射 ---
    epochs = args.epochs_finetune
    patience = args.patience_finetune
    # 优先用 lr_finetune；如果老配置里有 lr 也兼容
    lr = getattr(args, "lr", None) or getattr(args, "lr_finetune", 1e-4)
    batch_size = getattr(args, "batch_b", 128)
    alpha_r2 = args.alpha_r2

    # --- 数据 ---
    data = get_data_and_scalers(opamp_type=args.opamp)
    Xtr, Ytr = data["target_train"]
    Xva, Yva = data["target_val"]
    input_dim, output_dim = Xtr.shape[1], Ytr.shape[1]

    # --- 构建模型（对两个工艺都适配） ---
    hidden_dims = _coerce_hidden_dims_from_args(args)
    print(f"[Target-Only] 使用 hidden_dims = {hidden_dims}")

    model = AlignHeteroMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        dropout_rate=args.dropout_rate,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_loader = make_loader(
        Xtr, Ytr, batch_size, shuffle=True, drop_last=True
    )
    val_loader = make_loader(
        Xva, Yva, batch_size, shuffle=False, drop_last=False
    )

    ckpt_path = RESULTS_DIR / f"{args.opamp}_target_only.pth"
    print(f"[Target-Only] opamp: {args.opamp}, device: {device}, ckpt: {ckpt_path.name}")

    # --- 若已有 ckpt 且未指定 --restart：尝试直接加载并退出 ---
    if ckpt_path.exists() and not args.restart:
        try:
            state = torch.load(ckpt_path, map_location=device)
            state_dict = state.get("state_dict", state)
            model.load_state_dict(state_dict, strict=False)
            va_nll0, _ = run_epoch(
                model, val_loader, None, alpha_r2, device, "val"
            )
            print(f"[Target-Only] 检测到已有 ckpt（{ckpt_path.name}），跳过训练。"
                  f"当前 Val NLL={va_nll0:.4f}")
            return
        except Exception as e:
            print(f"[Target-Only] 发现 ckpt 但载入失败（{e}），将从头重训。")

    # --- restart: 删除旧 ckpt ---
    if args.restart and ckpt_path.exists():
        try:
            ckpt_path.unlink()
            print("[Target-Only] `--restart` 指定：已删除旧 checkpoint，将从头训练。")
        except Exception as e:
            print(f"[Target-Only] 删除旧 checkpoint 失败（忽略继续）：{e}")

    # --- 训练循环 ---
    best_val_nll = float("inf")
    patience_counter = patience

    for ep in range(1, epochs + 1):
        tr_nll, tr_r2l = run_epoch(
            model, train_loader, optimizer, alpha_r2, device, "train"
        )
        va_nll, va_r2l = run_epoch(
            model, val_loader, None,      alpha_r2, device, "val"
        )

        print(
            f"[Target-Only][{ep:03d}/{epochs}] "
            f"Train NLL={tr_nll:.4f} | Val NLL={va_nll:.4f}"
        )

        if va_nll < best_val_nll:
            best_val_nll = va_nll
            torch.save({"state_dict": model.state_dict()}, ckpt_path)
            patience_counter = patience
            print("  -> New best model saved.")
        else:
            patience_counter -= 1
            if patience_counter <= 0:
                print(f"[Target-Only] Early stopping at epoch {ep}.")
                break

    print(f"\n[Target-Only] Finished. Best model at: {ckpt_path}")


if __name__ == "__main__":
    main()
