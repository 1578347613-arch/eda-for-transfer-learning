# -*- coding: utf-8 -*-
"""
V2 反向设计：MDN 逆映射（p(x|y)），支持训练与采样
- 训练：使用 data_loader.get_data_and_scalers()（已做 log1p/标准化）
       将 (y_scaled -> x_scaled) 作为监督，训练 MDN
- 采样：给定“物理单位” y_target，内部做 log1p+标准化 -> 从 MDN 采样 x_scaled 候选
       保存为 .npy，供 inverse_opt.py 作为初值（Hybrid）
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Tuple, Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_loader.data_loader import get_data_and_scalers
import config  # 统一超参/设备

# ---------- 路径：results 与 src_new 同级 ----------
PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 指标顺序与 log1p 列（与数据预处理保持一致）
Y_ORDER = ["slewrate_pos", "dc_gain", "ugf", "phase_margin", "cmrr"]
Y_LOG1P_INDEX = [2, 4]  # ugf, cmrr


# --------------------
# MDN 模型
# --------------------
class InverseMDN(nn.Module):
    """
    p(x|y) 的混合密度网络：输入 y_scaled，输出若干高斯分量的 (pi, mu, sigma)
    - 对 x 维度使用对角协方差
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_components: int,
        hidden_dim: int = config.MDN_HIDDEN,
        num_layers: int = config.MDN_LAYERS,
    ):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        self.backbone = nn.Sequential(*layers)

        self.pi = nn.Linear(hidden_dim, n_components)
        self.mu = nn.Linear(hidden_dim, n_components * output_dim)
        self.sigma_raw = nn.Linear(hidden_dim, n_components * output_dim)

        self.n_components = n_components
        self.output_dim = output_dim
        self.softplus = nn.Softplus()  # 保证 sigma > 0

    def forward(self, y: torch.Tensor):
        """
        输入：y_scaled (B, input_dim)
        输出：
            pi    (B, K)
            mu    (B, K, D)
            sigma (B, K, D)  正值
        """
        h = self.backbone(y)
        pi = torch.softmax(self.pi(h), dim=-1)
        mu = self.mu(h).view(-1, self.n_components, self.output_dim)
        sigma = self.softplus(self.sigma_raw(h)).view(-1, self.n_components, self.output_dim) + 1e-6
        return pi, mu, sigma


def mdn_nll_loss(pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, target_x: torch.Tensor) -> torch.Tensor:
    """
    MDN 负对数似然（独立维度，对角协方差）
    target_x: (B, D)
    """
    B, K, D = mu.shape
    target = target_x.unsqueeze(1).expand(B, K, D)  # (B,K,D)
    # 对角高斯 log prob
    log_prob = -0.5 * torch.sum(((target - mu) / sigma) ** 2 + 2 * torch.log(sigma) + np.log(2 * np.pi), dim=2)  # (B,K)
    # 混合 log-sum-exp
    log_mix = torch.logsumexp(torch.log(pi + 1e-9) + log_prob, dim=1)  # (B,)
    return -torch.mean(log_mix)


# --------------------
# 数据准备
# --------------------
def prepare_inverse_dataset(opamp_type: str, device: str = config.DEVICE) -> Tuple[torch.Tensor, torch.Tensor, object, object]:
    """
    使用 data_loader 的预处理（含 log1p 与 StandardScaler）。
    返回：
        y_all (N,Dy) 作为输入（scaled）
        x_all (N,Dx) 作为目标（scaled）
        x_scaler, y_scaler
    """
    data = get_data_and_scalers(opamp_type=opamp_type)
    if "source" not in data:
        raise RuntimeError("get_data_and_scalers 返回中缺少 'source' 键，请检查 data_loader。")

    x_a, y_a = data["source"]
    x_list, y_list = [x_a], [y_a]
    if "target_train" in data and data["target_train"] is not None:
        x_b_tr, y_b_tr = data["target_train"]
        x_list.append(x_b_tr); y_list.append(y_b_tr)
    if "target_val" in data and data["target_val"] is not None:
        x_b_val, y_b_val = data["target_val"]
        x_list.append(x_b_val); y_list.append(y_b_val)

    x_all = np.vstack(x_list).astype(np.float32)
    y_all = np.vstack(y_list).astype(np.float32)

    x_tensor = torch.from_numpy(x_all).to(device)
    y_tensor = torch.from_numpy(y_all).to(device)

    # scaler：优先从 data_loader 返回；否则从 ../results/ 加载
    x_scaler = data.get("x_scaler", None)
    y_scaler = data.get("y_scaler", None)
    if x_scaler is None or y_scaler is None:
        xs_path = RESULTS_DIR / f"{opamp_type}_x_scaler.gz"
        ys_path = RESULTS_DIR / f"{opamp_type}_y_scaler.gz"
        if not xs_path.exists() or not ys_path.exists():
            raise FileNotFoundError(f"未找到标准化器：{xs_path} 或 {ys_path}")
        x_scaler = joblib.load(xs_path)
        y_scaler = joblib.load(ys_path)

    return y_tensor, x_tensor, x_scaler, y_scaler


# --------------------
# 采样（给定 y_physical）
# --------------------
def transform_y_physical_to_scaled(y_physical: np.ndarray, y_scaler) -> np.ndarray:
    """
    y(物理) -> （对指定列 log1p）-> 标准化 -> y_scaled
    """
    y = y_physical.astype(np.float64).copy()
    for idx in Y_LOG1P_INDEX:  # ugf, cmrr
        y[idx] = np.log1p(y[idx])
    y_scaled = y_scaler.transform(y.reshape(1, -1)).astype(np.float32)[0]
    return y_scaled


@torch.no_grad()
def mdn_sample(mdn: InverseMDN, y_scaled: np.ndarray, n: int, device: str = config.DEVICE) -> np.ndarray:
    """
    从 MDN 中对单个 y_scaled 采样 n 个 x_scaled
    返回 shape: (n, Dx)
    """
    y = torch.from_numpy(y_scaled.astype(np.float32)).to(device).view(1, -1)
    pi, mu, sigma = mdn(y)               # (1,K),(1,K,D),(1,K,D)
    pi, mu, sigma = pi[0], mu[0], sigma[0]
    comp_idx = torch.multinomial(pi, num_samples=n, replacement=True)  # (n,)
    mu_sel = mu[comp_idx]                 # (n,D)
    sigma_sel = sigma[comp_idx]           # (n,D)
    eps = torch.randn_like(mu_sel)
    x_samples = mu_sel + sigma_sel * eps
    return x_samples.cpu().numpy()


# --------------------
# 训练流程
# --------------------
def train_mdn(
    opamp_type: str,
    save_path: Path,
    components: int = config.MDN_COMPONENTS,
    hidden: int = config.MDN_HIDDEN,
    layers: int = config.MDN_LAYERS,
    batch_size: int = config.MDN_BATCH_SIZE,
    epochs: int = config.MDN_EPOCHS,
    lr: float = config.MDN_LR,
    weight_decay: float = config.WEIGHT_DECAY,
    device: str = config.DEVICE,
):
    y_tensor, x_tensor, _, _ = prepare_inverse_dataset(opamp_type, device=device)
    Dy = y_tensor.shape[1]
    Dx = x_tensor.shape[1]

    ds = TensorDataset(y_tensor, x_tensor)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    mdn = InverseMDN(input_dim=Dy, output_dim=Dx, n_components=components, hidden_dim=hidden, num_layers=layers).to(device)
    opt = torch.optim.AdamW(mdn.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"[MDN] 训练开始：N={len(ds)}, Dy={Dy}, Dx={Dx}, K={components}, hidden={hidden}, layers={layers}")
    mdn.train()
    for ep in range(1, epochs + 1):
        total = 0.0
        for y_b, x_b in dl:
            opt.zero_grad(set_to_none=True)
            pi, mu, sigma = mdn(y_b)
            loss = mdn_nll_loss(pi, mu, sigma, x_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdn.parameters(), 5.0)
            opt.step()
            total += float(loss.detach().cpu().item()) * y_b.size(0)
        avg = total / len(ds)
        print(f"[MDN][Epoch {ep:03d}] NLL: {avg:.4f}")

    # 保存
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": mdn.state_dict(),
            "config": {
                "components": components,
                "hidden": hidden,
                "layers": layers,
                "input_dim": Dy,
                "output_dim": Dx,
            },
            "opamp": opamp_type,
        },
        save_path,
    )
    print(f"[MDN] 已保存: {save_path}")

    # 元信息（读取 scaler 路径提示）
    meta = {
        "opamp": opamp_type,
        "x_scaler": str((RESULTS_DIR / f"{opamp_type}_x_scaler.gz").resolve()),
        "y_scaler": str((RESULTS_DIR / f"{opamp_type}_y_scaler.gz").resolve()),
    }
    meta_path = save_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[MDN] 元信息保存: {meta_path}")


# --------------------
# CLI
# --------------------
def _parse_y_target(s: str) -> np.ndarray:
    parts = [p.strip() for p in s.split(",")]
    return np.array([float(p) for p in parts], dtype=np.float64)


def _pick_existing(*candidates: Path) -> Optional[Path]:
    for p in candidates:
        if p is None:
            continue
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description="V2 反向设计：MDN 训练/采样")
    # 训练相关
    ap.add_argument("--opamp", type=str, default=config.OPAMP_TYPE, help="电路类型（影响 scaler 与数据加载）")
    ap.add_argument("--save", type=str, default="", help="模型保存路径（默认 ../results/mdn_{opamp}.pth）")
    ap.add_argument("--components", type=int, default=config.MDN_COMPONENTS, help="混合分量个数")
    ap.add_argument("--hidden", type=int, default=config.MDN_HIDDEN, help="隐藏层宽度")
    ap.add_argument("--layers", type=int, default=config.MDN_LAYERS, help="隐藏层层数")
    ap.add_argument("--batch-size", type=int, default=config.MDN_BATCH_SIZE)
    ap.add_argument("--epochs", type=int, default=config.MDN_EPOCHS)
    ap.add_argument("--lr", type=float, default=config.MDN_LR)
    ap.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    ap.add_argument("--device", type=str, default="auto", help="'cuda'/'cpu'/'auto'")
    ap.add_argument("--seed", type=int, default=42)

    # 采样相关
    ap.add_argument("--sample", action="store_true", help="进入采样模式（不训练）")
    ap.add_argument("--model", type=str, default=None, help="采样模式下，已训练 MDN 权重路径")
    ap.add_argument("--y-target", type=str, default=None, help='目标 y（物理单位，格式："SR,Gain,UGF,PM,CMRR"）')
    ap.add_argument("--n", type=int, default=64, help="采样数量")
    ap.add_argument("--out", type=str, default="", help="采样输出 .npy 路径（默认 ../results/inverse/init_64.npy）")

    args = ap.parse_args()

    # 设备与随机性
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if not args.sample:
        # 训练
        save_path = Path(args.save) if args.save else (RESULTS_DIR / f"mdn_{args.opamp}.pth")
        train_mdn(
            opamp_type=args.opamp,
            save_path=save_path,
            components=args.components,
            hidden=args.hidden,
            layers=args.layers,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
        )
    else:
        # 采样
        if args.model is None:
            raise ValueError("--sample 模式需要提供 --model")
        if args.y_target is None:
            raise ValueError("--sample 模式需要提供 --y-target")

        # scaler 路径（../results）
        x_scaler_path = RESULTS_DIR / f"{args.opamp}_x_scaler.gz"
        y_scaler_path = RESULTS_DIR / f"{args.opamp}_y_scaler.gz"
        if not x_scaler_path.exists() or not y_scaler_path.exists():
            raise FileNotFoundError(f"标准化器未找到：{x_scaler_path} 或 {y_scaler_path}")
        x_scaler = joblib.load(x_scaler_path)
        y_scaler = joblib.load(y_scaler_path)

        # 加载 MDN
        model_path = Path(args.model)
        if not model_path.exists():
            # 允许只给文件名时到 ../results 查找
            model_path = _pick_existing(RESULTS_DIR / Path(args.model).name)
        if model_path is None or not model_path.exists():
            raise FileNotFoundError(f"未找到 MDN 权重：{args.model}")

        ckpt = torch.load(model_path, map_location=device)
        cfg = ckpt.get("config", {})
        Dy = int(cfg.get("input_dim", len(Y_ORDER)))
        Dx = int(cfg.get("output_dim", x_scaler.mean_.shape[0]))

        mdn = InverseMDN(
            input_dim=Dy,
            output_dim=Dx,
            n_components=int(cfg.get("components", config.MDN_COMPONENTS)),
            hidden_dim=int(cfg.get("hidden", config.MDN_HIDDEN)),
            num_layers=int(cfg.get("layers", config.MDN_LAYERS)),
        ).to(device)
        mdn.load_state_dict(ckpt["state_dict"])
        mdn.eval()

        # y 物理 -> 标准化
        y_physical = _parse_y_target(args.y_target)
        if y_physical.shape[0] != Dy:
            raise ValueError(f"y-target 维度应为 {Dy}，而不是 {y_physical.shape}")
        y_scaled = transform_y_physical_to_scaled(y_physical, y_scaler)

        # 采样
        x_scaled_samples = mdn_sample(mdn, y_scaled, n=args.n, device=device)

        # 可选：把样本裁到 ±3σ（避免离群）
        var = getattr(x_scaler, "var_", None)
        std = np.sqrt(var).astype(np.float32) if var is not None else np.ones(Dx, dtype=np.float32)
        x_min = -3.0 * std
        x_max = +3.0 * std
        x_scaled_samples = np.clip(x_scaled_samples, x_min, x_max)

        # 输出路径（../results/inverse）
        out_path = Path(args.out) if args.out else (RESULTS_DIR / "inverse" / "init_64.npy")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, x_scaled_samples.astype(np.float32))
        print(f"[MDN] 已采样 {x_scaled_samples.shape[0]} 个初值（标准化空间），保存到：{out_path}")
        print("可配合 inverse_opt.py 的 --init-npy 使用（Hybrid）。")


if __name__ == "__main__":
    main()
