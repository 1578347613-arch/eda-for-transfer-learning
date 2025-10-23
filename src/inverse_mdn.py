# -*- coding: utf-8 -*-
"""
V2 反向设计：MDN 逆映射（p(x|y)），支持训练与采样
- 训练：使用 data_loader.get_data_and_scalers() 的 A/B 数据（已完成 log1p/标准化）
       将 (y_scaled -> x_scaled) 作为监督，训练 MDN
- 采样：给定“物理单位” y_target，内部做 log1p+标准化 -> 采样出 x_scaled 候选
       保存为 .npy，供 inverse_opt.py 作为初值（Hybrid）

训练示例：
python src/inverse_mdn.py \
  --opamp 5t_opamp \
  --epochs 60 \
  --components 10 \
  --hidden 256 \
  --layers 4 \
  --batch-size 128 \
  --lr 1e-3 \
  --save results/mdn_5t.pth

采样示例：
python src/inverse_mdn.py \
  --sample \
  --model results/mdn_5t.pth \
  --y-target "2.5e8,200,1.5e6,65,20000" \
  --n 64 \
  --opamp 5t_opamp \
  --out results/inverse/init_64.npy
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 复用与训练一致的预处理与顺序
from data_loader import get_data_and_scalers

Y_ORDER = ["slewrate_pos", "dc_gain", "ugf", "phase_margin", "cmrr"]
Y_LOG1P_INDEX = [2, 4]
X_DIM = 7
Y_DIM = 5


# --------------------
# MDN 模型
# --------------------
class InverseMDN(nn.Module):
    """
    p(x|y) 的混合密度网络：输入 y_scaled，输出若干高斯分量的 (pi, mu, sigma)
    - 对 x 维度使用对角协方差
    """
    def __init__(self, input_dim: int, output_dim: int, n_components: int, hidden_dim: int = 256, num_layers: int = 4):
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

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    MDN 负对数似然（独立维度对角协方差）
    target_x: (B, D)
    """
    B, K, D = mu.shape
    target = target_x.unsqueeze(1).expand(B, K, D)  # (B,K,D)
    # 对角高斯的 log prob
    log_prob = -0.5 * torch.sum(((target - mu) / sigma) ** 2 + 2 * torch.log(sigma) + np.log(2 * np.pi), dim=2)  # (B,K)
    # 混合：logsumexp
    log_mix = torch.logsumexp(torch.log(pi + 1e-9) + log_prob, dim=1)  # (B,)
    return -torch.mean(log_mix)


# --------------------
# 数据准备
# --------------------
# 替换 src/inverse_mdn.py 中的 prepare_inverse_dataset 函数

def prepare_inverse_dataset(opamp_type: str, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor, object, object]:
    """
    使用 data_loader 的预处理（含 log1p 与 StandardScaler）。
    返回：
        y_all (N,5) 作为输入（scaled）
        x_all (N,7) 作为目标（scaled）
        x_scaler, y_scaler 以便采样阶段使用（优先从 data_loader 返回；否则从 results/ 加载）
    """
    # 注意：不要传 test_size/random_state，与你当前 data_loader 的签名保持一致
    data = get_data_and_scalers(opamp_type=opamp_type)

    # 兼容不同返回键：必须要有 source，建议有 target_train/target_val
    if "source" not in data:
        raise RuntimeError("get_data_and_scalers 返回中缺少 'source' 键，请检查 src/data_loader.py。")

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

    # 拿 scaler：优先从 data_loader 返回；否则从 results/ 目录加载（你已经有这两个文件）
    x_scaler = data.get("x_scaler", None)
    y_scaler = data.get("y_scaler", None)
    if x_scaler is None or y_scaler is None:
        import joblib, os
        xs_path = f"results/{opamp_type}_x_scaler.gz"
        ys_path = f"results/{opamp_type}_y_scaler.gz"
        if not (os.path.exists(xs_path) and os.path.exists(ys_path)):
            raise FileNotFoundError(
                f"未在 data_loader 返回 scaler，也未在 results/ 找到标准化器：{xs_path} / {ys_path}"
            )
        x_scaler = joblib.load(xs_path)
        y_scaler = joblib.load(ys_path)

    return y_tensor, x_tensor, x_scaler, y_scaler

# --------------------
# 采样（给定 y_physical）
# --------------------
def transform_y_physical_to_scaled(y_physical: np.ndarray, y_scaler) -> np.ndarray:
    y = y_physical.astype(np.float64).copy()
    for idx in Y_LOG1P_INDEX:
        y[idx] = np.log1p(y[idx])
    y_scaled = y_scaler.transform(y.reshape(1, -1)).astype(np.float32)[0]
    return y_scaled


@torch.no_grad()
def mdn_sample(mdn: InverseMDN, y_scaled: np.ndarray, n: int, device: str = "cpu") -> np.ndarray:
    """
    从 MDN 中对单个 y_scaled 采样 n 个 x_scaled
    返回 shape: (n, X_DIM)
    """
    y = torch.from_numpy(y_scaled.astype(np.float32)).to(device).view(1, -1)
    pi, mu, sigma = mdn(y)  # (1,K),(1,K,D),(1,K,D)
    pi = pi[0]
    mu = mu[0]
    sigma = sigma[0]

    # 基于 pi 多项式采样 K 的索引
    comp_idx = torch.multinomial(pi, num_samples=n, replacement=True)  # (n,)
    mu_sel = mu[comp_idx]      # (n,D)
    sigma_sel = sigma[comp_idx]  # (n,D)
    eps = torch.randn_like(mu_sel)
    x_samples = mu_sel + sigma_sel * eps
    return x_samples.cpu().numpy()


# --------------------
# 训练流程
# --------------------
def train_mdn(
    opamp_type: str,
    save_path: Path,
    components: int = 10,
    hidden: int = 256,
    layers: int = 4,
    batch_size: int = 128,
    epochs: int = 60,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: str = "cpu",
):
    y_tensor, x_tensor, x_scaler, y_scaler = prepare_inverse_dataset(opamp_type, device=device)
    ds = TensorDataset(y_tensor, x_tensor)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    mdn = InverseMDN(input_dim=Y_DIM, output_dim=X_DIM, n_components=components, hidden_dim=hidden, num_layers=layers).to(device)
    opt = torch.optim.AdamW(mdn.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"[MDN] 训练开始：N={len(ds)}, components={components}, hidden={hidden}, layers={layers}")
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
                "input_dim": Y_DIM,
                "output_dim": X_DIM,
            },
            "opamp": opamp_type,
        },
        save_path,
    )
    print(f"[MDN] 已保存: {save_path}")

    # 同时保存 读取所需的 scaler 路径提示
    meta = {
        "opamp": opamp_type,
        "x_scaler": f"results/{opamp_type}_x_scaler.gz",
        "y_scaler": f"results/{opamp_type}_y_scaler.gz",
    }
    meta_path = save_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[MDN] 元信息保存: {meta_path}")


def main():
    ap = argparse.ArgumentParser(description="V2 反向设计：MDN 训练/采样")
    ap.add_argument("--opamp", type=str, default="5t_opamp", help="电路类型（影响 scaler 与数据加载）")
    ap.add_argument("--save", type=str, default="results/mdn_5t.pth", help="训练完成后的模型保存路径")
    ap.add_argument("--components", type=int, default=10, help="混合分量个数")
    ap.add_argument("--hidden", type=int, default=256, help="隐藏层宽度")
    ap.add_argument("--layers", type=int, default=4, help="隐藏层层数")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--device", type=str, default="auto", help="'cuda'/'cpu'/'auto'")
    ap.add_argument("--seed", type=int, default=42)

    # 采样模式
    ap.add_argument("--sample", action="store_true", help="进入采样模式（不训练）")
    ap.add_argument("--model", type=str, default=None, help="采样模式下，已训练 MDN 权重路径")
    ap.add_argument("--y-target", type=str, default=None, help="目标 y（物理单位，SR,Gain,UGF,PM,CMRR）")
    ap.add_argument("--n", type=int, default=64, help="采样数量")
    ap.add_argument("--out", type=str, default="results/inverse/init_64.npy", help="采样输出的 .npy 路径")

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
        train_mdn(
            opamp_type=args.opamp,
            save_path=Path(args.save),
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

        # 读取 scaler
        x_scaler_path = Path(f"results/{args.opamp}_x_scaler.gz")
        y_scaler_path = Path(f"results/{args.opamp}_y_scaler.gz")
        if not x_scaler_path.exists() or not y_scaler_path.exists():
            raise FileNotFoundError(f"标准化器未找到：{x_scaler_path} 或 {y_scaler_path}")
        x_scaler = joblib.load(x_scaler_path)
        y_scaler = joblib.load(y_scaler_path)

        # 加载 MDN
        ckpt = torch.load(args.model, map_location=device)
        cfg = ckpt.get("config", {})
        mdn = InverseMDN(
            input_dim=cfg.get("input_dim", Y_DIM),
            output_dim=cfg.get("output_dim", X_DIM),
            n_components=cfg.get("components", 10),
            hidden_dim=cfg.get("hidden", 256),
            num_layers=cfg.get("layers", 4),
        ).to(device)
        mdn.load_state_dict(ckpt["state_dict"])
        mdn.eval()

        # y 物理 -> 标准化
        y_physical = np.array([float(x.strip()) for x in args.y_target.split(",")], dtype=np.float64)
        if y_physical.shape[0] != Y_DIM:
            raise ValueError(f"y-target 维度应为 {Y_DIM}，而不是 {y_physical.shape}")
        y_scaled = transform_y_physical_to_scaled(y_physical, y_scaler)

        # 采样
        x_scaled_samples = mdn_sample(mdn, y_scaled, n=args.n, device=device)

        # 可选：把样本裁到 ±3σ（避免离群）
        var = getattr(x_scaler, "var_", None)
        if var is None:
            std = np.ones(X_DIM, dtype=np.float32)
        else:
            std = np.sqrt(var).astype(np.float32)
        x_min = -3.0 * std
        x_max = +3.0 * std
        x_scaled_samples = np.clip(x_scaled_samples, x_min, x_max)

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, x_scaled_samples.astype(np.float32))
        print(f"[MDN] 已采样 {x_scaled_samples.shape[0]} 个初值（标准化空间），保存到：{out_path}")
        print("可配合 inverse_opt.py 的 --init-npy 使用（Hybrid）。")


if __name__ == "__main__":
    main()
