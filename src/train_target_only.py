from pathlib import Path
from typing import Tuple
import argparse
import ast # <-- (C3 手术) 导入 ast 来解析列表

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data_loader import get_data_and_scalers
from loss_function import heteroscedastic_nll, batch_r2
# ⬇️ (C3 手术) 我们假设 C3/models/ 里是 C1 的模型！
from models.align_hetero import AlignHeteroMLP 
# ⬇️ (C3 手术) 我们现在导入 C3 的“混合圣经”
import config 

# --- 路径定义 (来自 C2) ---
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "src" / "results"
# RESULTS_DIR.mkdir(parents=True, exist_ok=True) # (C2 的逻辑)

# --- 辅助函数 (来自 C2) ---
def set_seed(seed: int):
    # (代码 100% 来自 C2，此处省略)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_loader(x: np.ndarray, y: np.ndarray, bs: int, shuffle: bool, drop_last: bool) -> DataLoader:
    # (代码 100% 来自 C2，此处省略)
    ds = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=drop_last)

def run_epoch(model, loader, optimizer, alpha_r2, device, phase="train"):
    # (代码 100% 来自 C2，此处省略)
    is_train = (optimizer is not None) and (phase == "train")
    model.train(is_train)
    total_nll, total_r2l, n_batches = 0.0, 0.0, max(1, len(loader))
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.set_grad_enabled(is_train):
            mu, logv, _ = model(xb) # <-- (C1 的 align_hetero.py 完美兼容)
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


# ==========================================================
# ========== 👨‍⚕️ "C3 融合手术" 核心 👨‍⚕️ ==========
# ==========================================================

def setup_args():
    """
    [C3 融合版] 智能 setup_args (逻辑来自 C1 train_align_hetero.py)：
    它现在会读取 C3 "混合圣经" config.py 里的 TASK_CONFIGS 字典！
    """
    parser = argparse.ArgumentParser(description="Target-Only 训练脚本 (C3 融合版)")
    parser.add_argument("--opamp", type=str, required=True,
                        choices=config.TASK_CONFIGS.keys(), help="必须指定的电路类型")

    # Step 1: 先解析出 opamp 类型
    temp_args, other_args = parser.parse_known_args()
    opamp_type = temp_args.opamp

    # Step 2: 合并 C3 "混合圣经" 的配置作为默认值
    # (COMMON_CONFIG + TASK_CONFIGS[opamp])
    defaults = {**config.COMMON_CONFIG, **config.TASK_CONFIGS.get(opamp_type, {})}

    # Step 3: 动态为所有简单类型的默认参数创建命令行开关
    for key, value in defaults.items():
        if isinstance(value, (list, dict)):
            continue  # 跳过 C1 的 hidden_dims 和 PRETRAIN_SCHEDULER_CONFIGS
        if isinstance(value, bool):
            parser.add_argument(
                f"--{key}", action=argparse.BooleanOptionalAction, help=f"开关 '{key}'")
        else:
            # (确保 key 存在，因为 C1/C2 的 config key 可能不完全一样)
            if key in config.COMMON_CONFIG or key in config.TASK_CONFIGS[opamp_type]:
                 parser.add_argument(
                    f"--{key}", type=type(value), help=f"设置 '{key}'")

    # Step 4: 将合并后的配置设置为解析器的默认值并进行最终解析
    parser.set_defaults(**defaults)

    # --- 覆盖：手动添加在config中是列表，但希望在命令行中覆盖的参数 ---
    # (这是为了 100% 兼容 C1 的架构)
    parser.add_argument("--hidden_dims", type=str,
                        help="MLP隐藏层维度列表, e.g., '[256, 256]'")
    
    args = parser.parse_args()

    # --- (C3 手术) C1 "黄金架构" 的后处理 ---
    # (逻辑 100% 来自 C1 的 train_align_hetero.py)
    if isinstance(args.hidden_dims, str):
        try:
            # 如果从命令行传入 '[...]' 字符串，则解析它
            args.hidden_dims = ast.literal_eval(args.hidden_dims)
        except (ValueError, SyntaxError):
            print(f"错误: --hidden_dims 参数格式不正确: {args.hidden_dims}")
            sys.exit(1)
    # (如果命令行没传，args.hidden_dims 已经是来自 config.py 的 C1 黄金列表！)
    
    return args


# --- 主训练函数 (C3 融合版) ---
def main():
    args = setup_args() # <-- (C3 手术) 调用我们新的 C3 setup_args()

    device = torch.device(args.device)
    set_seed(args.seed)

    # 超参映射（现在 100% 兼容 C3 "混合圣经"）
    # (C1 的 config 里叫 'epochs_finetune', 'patience_finetune'...)
    epochs = args.epochs_finetune 
    patience = args.patience_finetune
    lr = args.lr_finetune # <-- (C1 config 里叫 lr_finetune)
    batch_size = args.batch_b
    alpha_r2 = args.alpha_r2

    data = get_data_and_scalers(opamp_type=args.opamp)
    Xtr, Ytr = data["target_train"]
    Xva, Yva = data["target_val"]
    
    # (我们假设 C3 会使用 C1 的 data_loader.py)
    try:
        input_dim = data['x_dim']
        output_dim = data['y_dim']
    except KeyError:
        # 兜底 C2 data_loader
        input_dim, output_dim = Xtr.shape[1], Ytr.shape[1]

    # ==========================================================
    # ========== 👨‍⚕️ "C3 融合手术" 核心 👨‍⚕️ ==========
    # ==========================================================
    print(f"✅ [C3 融合] 正在为 {args.opamp} 构建 C1 黄金架构...")
    
    # ⬇️ (C3 手术) 这 6 行代码是“手术”的核心！
    model = AlignHeteroMLP(
        input_dim=input_dim, 
        output_dim=output_dim,
        # ⬇️ 关键！使用 C1 的复杂列表！ ⬇️
        hidden_dims=args.hidden_dims,
        dropout_rate=args.dropout_rate
        # (我们假设 C3/models/align_hetero.py 是 C1 的版本)
    ).to(device)

    # ==========================================================
    # ========== "C3 融合手术" 结束 ==========
    # ==========================================================

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_loader = make_loader(
        Xtr, Ytr, batch_size, shuffle=True,  drop_last=True)
    val_loader = make_loader(Xva, Yva, batch_size,
                             shuffle=False, drop_last=False)

    ckpt_path = RESULTS_DIR / f"{args.opamp}_target_only.pth"
    print(
        f"[Target-Only (C3 融合版)] opamp: {args.opamp}, device: {device}, saving to: {ckpt_path.name}")

    # ========== (C2 的跳过逻辑，100% 保留) ==========
    if ckpt_path.exists() and not args.restart:
        try:
            state = torch.load(ckpt_path, map_location=device)
            state_dict = state.get("state_dict", state)
            # ⬇️ 完美！现在加载的权重 100% 兼容 C1 黄金架构！
            model.load_state_dict(state_dict) 
            va_nll0, _ = run_epoch(
                model, val_loader, None, alpha_r2, device, "val")
            print(f"[Target-Only] 检测到已有 ckpt（{ckpt_path.name}）。按默认策略跳过训练并退出。"
                  f"当前 Val NLL={va_nll0:.4f}")
        except Exception as e:
            print(f"[Target-Only] 发现 ckpt 但载入失败（{e}）。将从头训练。")
        else:
            return

    # ========== (C2 的训练循环，100% 保留) ==========
    if args.restart and ckpt_path.exists():
        # (代码 100% 来自 C2，此处省略)
        try:
            ckpt_path.unlink()
            print("`--restart` 指定：已删除旧 checkpoint，将从头训练。")
        except Exception as e:
            print(f"删除旧 checkpoint 失败（忽略继续）：{e}")

    best_val_nll = float("inf")
    patience_counter = patience

    for ep in range(1, epochs + 1):
        tr_nll, tr_r2l = run_epoch(
            model, train_loader, optimizer, alpha_r2, device, "train")
        va_nll, va_r2l = run_epoch(
            model, val_loader, None,       alpha_r2, device, "val")
        print(
            f"[Target-Only][{ep:03d}/{epochs}] Train NLL={tr_nll:.4f} | Val NLL={va_nll:.4f}")

        if va_nll < best_val_nll:
            best_val_nll = va_nll
            # ⬇️ 完美！现在保存的权重 100% 兼容 C1 黄金架构！
            torch.save({"state_dict": model.state_dict()}, ckpt_path) 
            patience_counter = patience
            print("  -> New best model saved.")
        else:
            patience_counter -= 1
            if patience_counter <= 0:
                print(f"Early stopping at epoch {ep}.")
                break

    print(f"\n[Target-Only (C3 融合版)] Finished. Best model at: {ckpt_path}")


if __name__ == "__main__":
    main()