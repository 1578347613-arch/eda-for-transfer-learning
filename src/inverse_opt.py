# src/inverse_opt.py (已完全修正)

from __future__ import annotations
import os
import math
import json
import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch

# --- 路径定义 (已修正) ---
# 这个脚本在 'src' 目录中，'results' 是 'src' 的子目录
SRC_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SRC_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==== 常量（名称保持一致，维度动态获取） ====
Y_LOG1P_INDEX = [2, 4]  # ugf(2), cmrr(4)
Y_NAMES = ["slewrate_pos", "dc_gain", "ugf", "phase_margin", "cmrr"]
X_NAMES = ["w1", "w2", "w3", "l1", "l2", "l3", "ibias"]


# ==== 通用工具 ====
def parse_floats(csv_str, n=None):
    arr = [float(x.strip()) for x in csv_str.split(",")]
    if (n is not None) and (len(arr) != n):
        raise ValueError(f"需要 {n} 个数值，实际 {len(arr)}: {csv_str}")
    return np.asarray(arr, dtype=np.float64)

def fmt_arr(a):
    return "[" + " ".join([f"{v:.6g}" for v in a]) + "]"

def ensure_dir(d: Path | str) -> str:
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


# ==== 数据与边界 ====
def get_all_scaled_X(opamp_type: str) -> np.ndarray:
    """从 data_loader 拉取全部（已标准化）的 X（A+B），用于边界估计。"""
    from data_loader import get_data_and_scalers
    data = get_data_and_scalers(opamp_type=opamp_type)
    xs = [data["source"][0]]
    if "target_train" in data and data.get("target_train") is not None:
        xs.append(data["target_train"][0])
    if "target_val" in data and data.get("target_val") is not None:
        xs.append(data["target_val"][0])
    return np.vstack(xs).astype(np.float64)

def get_bounds_and_scalers(opamp_type, x_scaler, device):
    """返回物理/标准化边界与 mean/scale 的 torch 张量。"""
    X_scaled = get_all_scaled_X(opamp_type)
    x_min_scaled = X_scaled.min(axis=0)
    x_max_scaled = X_scaled.max(axis=0)

    X_phys = x_scaler.inverse_transform(X_scaled)
    x_min_phys = X_phys.min(axis=0)
    x_max_phys = X_phys.max(axis=0)

    to_t = lambda a: torch.from_numpy(a.astype(np.float32)).to(device)
    return (
        to_t(x_min_phys), to_t(x_max_phys),
        to_t(x_min_scaled), to_t(x_max_scaled),
        to_t(x_scaler.mean_), to_t(x_scaler.scale_),
    )


# ==== 模型构建 / 加载 (已修正) ====
def load_model_from_ckpt(ckpt_path: Path | str, model_type: str, device: torch.device,
                         opamp_type: str) -> torch.nn.Module:
    """
    根据 opamp_type 从 config.py 加载配置，正确地构建和加载模型。
    """
    # 1. 导入项目配置和数据加载器
    from config import TASK_CONFIGS
    from data_loader import get_data_and_scalers

    # 2. 动态获取输入/输出维度
    data = get_data_and_scalers(opamp_type=opamp_type)
    input_dim = data['source'][0].shape[1]
    output_dim = data['source'][1].shape[1]
    
    # 3. 获取特定任务的模型超参数
    if opamp_type not in TASK_CONFIGS:
        raise KeyError(f"在 config.py 中未找到 opamp 类型 '{opamp_type}' 的配置。")
    model_config = TASK_CONFIGS[opamp_type]

    # 4. 使用正确的超参数构建模型
    mt = model_type.lower()
    if mt == "align_hetero":
        from models.align_hetero import AlignHeteroMLP
        model = AlignHeteroMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            dropout_rate=model_config['dropout_rate']
        ).to(device)
    # 你可以为其他模型类型添加类似的逻辑
    # elif mt == "dualhead_b":
    #     ...
    else:
        raise NotImplementedError(f"模型类型 '{model_type}' 的动态构建尚未实现。")

    # 5. 加载模型权重
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load warning] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    model.eval()
    return model


# ==== 目标可达性诊断 ====
def diagnose_target_feasibility(opamp_type, y_scaler, y_target_physical):
    from data_loader import get_data_and_scalers
    data = get_data_and_scalers(opamp_type=opamp_type)
    ys = [data["source"][1]]
    if "target_train" in data and data["target_train"] is not None:
        ys.append(data["target_train"][1])
    if "target_val" in data and data["target_val"] is not None:
        ys.append(data["target_val"][1])
    Y_scaled = np.vstack(ys).astype(np.float64)

    # 还原到物理便于直觉
    Y_phys = y_scaler.inverse_transform(Y_scaled.copy())
    for idx in Y_LOG1P_INDEX:
        if idx < len(Y_phys[0]):
            Y_phys[:, idx] = np.expm1(Y_phys[:, idx])
    q = np.percentile(Y_phys, [1, 5, 50, 95, 99], axis=0)

    # 目标的 z-score
    y_phys = y_target_physical.copy()
    y_log = y_phys.copy()
    for idx in Y_LOG1P_INDEX:
        if idx < len(y_log):
            y_log[idx] = np.log1p(y_log[idx])
    z = (y_log - y_scaler.mean_) / y_scaler.scale_

    print("\n[诊断] 目标在训练分布中的位置（物理单位 & z-score）")
    for j, nm in enumerate(Y_NAMES):
        if j < len(y_phys):
            print(f"{nm:14s} target={y_phys[j]:.6g} | "
                  f"q1%={q[0,j]:.6g}  q5%={q[1,j]:.6g}  median={q[2,j]:.6g}  "
                  f"q95%={q[3,j]:.6g}  q99%={q[4,j]:.6g} | z={z[j]:+.2f}")


# ==== 反向优化 ====
def optimize_x_multi_start(
    model: torch.nn.Module,
    model_type: str,
    x_dim: int,
    y_target_scaled: np.ndarray,
    x_scaler,
    y_scaler,
    n_init: int = 16,
    steps: int = 300,
    lr: float = 1e-2,
    weights: np.ndarray = None,
    init_points_scaled: np.ndarray = None,
    device: str = "cpu",
    grad_clip: float = 1.0,
    opamp_type: str = "5t_opamp",
    goal: str = "min,min,range,min,min",
    ugf_band: str = "1.2e6:3.0e6",
    pm_band: str = "60:75",
    prior: float = 1e-3,
    finish_lbfgs: int = 0,
):
    device_t = torch.device(device)
    model.eval()

    y_t = torch.from_numpy(y_target_scaled.astype(np.float32)).to(device_t).view(1, -1)

    if weights is None:
        w = torch.ones(y_t.shape[1], dtype=torch.float32, device=device_t)
    else:
        w = torch.from_numpy(weights.astype(np.float32)).to(device_t)
    w = w.view(1, -1)

    x_min_phys, x_max_phys, x_min_scaled, x_max_scaled, mean_t, scale_t = \
        get_bounds_and_scalers(opamp_type, x_scaler, device_t)

    if init_points_scaled is not None:
        init = np.asarray(init_points_scaled, dtype=np.float32)
        if init.shape[0] < n_init:
            reps = int(np.ceil(n_init / init.shape[0]))
            init = np.vstack([init] * reps)[:n_init]
        else:
            init = init[:n_init]
        x0 = torch.from_numpy(init).to(device_t)
        x0 = torch.max(torch.min(x0, x_max_scaled), x_min_scaled)
    else:
        rnd = torch.rand(n_init, x_dim, device=device_t)
        x0 = x_min_scaled + rnd * (x_max_scaled - x_min_scaled)

    x = x0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)

    best_loss = float("inf")
    best_x, best_y = None, None

    goal_list = [g.strip().lower() for g in goal.split(",")]
    Y_DIM = y_t.shape[1]
    if len(goal_list) != Y_DIM:
        raise ValueError(f"--goal 需要 {Y_DIM} 个，用逗号分隔；收到 {len(goal_list)} 个。")

    y_mean = y_scaler.mean_.astype(np.float64)
    y_scale = y_scaler.scale_.astype(np.float64)

    def phys_to_scaled_scalar(j: int, val_phys: float) -> float:
        v = math.log1p(val_phys) if j in Y_LOG1P_INDEX else float(val_phys)
        return float((v - y_mean[j]) / y_scale[j])

    ugf_lb_phys, ugf_ub_phys = [float(s) for s in ugf_band.split(":")]
    ugf_lb_scaled = phys_to_scaled_scalar(2, ugf_lb_phys)
    ugf_ub_scaled = phys_to_scaled_scalar(2, ugf_ub_phys)

    pm_lb_phys, pm_ub_phys = [float(s) for s in pm_band.split(":")]
    pm_lb_scaled = phys_to_scaled_scalar(3, pm_lb_phys)
    pm_ub_scaled = phys_to_scaled_scalar(3, pm_ub_phys)

    def scaled_constraint_loss(y_pred_scaled: torch.Tensor) -> torch.Tensor:
        total = 0.0
        y_t_scaled = y_t.expand_as(y_pred_scaled)
        for j in range(Y_DIM):
            gj, yj, wj = goal_list[j], y_pred_scaled[:, j], w[:, j]
            if gj == "target":
                total += torch.mean(wj * (yj - y_t_scaled[:, j])**2)
            elif gj == "min":
                total += torch.mean(wj * torch.relu(y_t_scaled[:, j] - yj)**2)
            elif gj == "max":
                total += torch.mean(wj * torch.relu(yj - y_t_scaled[:, j])**2)
            elif gj == "range":
                if j == 2:  # UGF
                    total += torch.mean(wj * torch.relu(ugf_lb_scaled - yj)**2)
                    total += torch.mean(wj * torch.relu(yj - ugf_ub_scaled)**2)
                elif j == 3:  # PM
                    total += torch.mean(wj * torch.relu(pm_lb_scaled - yj)**2)
                    total += torch.mean(wj * torch.relu(yj - pm_ub_scaled)**2)
                else:
                    total += torch.mean(wj * (yj - y_t_scaled[:, j])**2)
            else:
                raise ValueError(f"未知 goal: {gj}")
        if prior and prior > 0:
            total += prior * torch.mean(torch.sum(x**2, dim=1))
        return total

    for t in range(steps):
        opt.zero_grad(set_to_none=True)
        mt = model_type.lower()
        if mt == "align_hetero":
            out = model(x)
            y_pred = out[0] if isinstance(out, (tuple, list)) else out
        else:
            y_pred = model(x)
        
        loss = scaled_constraint_loss(y_pred)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([x], grad_clip)
        opt.step()

        with torch.no_grad():
            x_phys = x * scale_t + mean_t
            x_phys.clamp_(x_min_phys, x_max_phys)
            x.copy_((x_phys - mean_t) / scale_t)

        cur = float(loss.item())
        if cur < best_loss:
            best_loss = cur
            best_x = x.detach().clone()
            best_y = y_pred.detach().clone()

    if finish_lbfgs and finish_lbfgs > 0:
        x_lb = best_x.clone().detach().requires_grad_(True)
        lbfgs = torch.optim.LBFGS([x_lb], lr=1.0, max_iter=finish_lbfgs, history_size=20, line_search_fn="strong_wolfe")
        def closure():
            lbfgs.zero_grad(set_to_none=True)
            mt = model_type.lower()
            if mt == "align_hetero":
                out2 = model(x_lb); y2 = out2[0] if isinstance(out2, (tuple, list)) else out2
            else:
                y2 = model(x_lb)
            loss2 = scaled_constraint_loss(y2)
            loss2.backward()
            return loss2
        lbfgs.step(closure)
        with torch.no_grad():
            x_phys = x_lb * scale_t + mean_t
            x_phys.clamp_(x_min_phys, x_max_phys)
            x_lb.copy_((x_phys - mean_t) / scale_t)
            mt = model_type.lower()
            if mt == "align_hetero":
                out2 = model(x_lb); y2 = out2[0] if isinstance(out2, (tuple, list)) else out2
            else:
                y2 = model(x_lb)
            loss2 = float(scaled_constraint_loss(y2).item())
            if loss2 <= best_loss:
                best_loss, best_x, best_y = loss2, x_lb.detach().clone(), y2.detach().clone()

    return best_x.cpu().numpy(), best_y.cpu().numpy(), best_loss


# ==== 主程序 ====
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--opamp", type=str, default="5t_opamp", help="数据/scaler 的类型标识")
    ap.add_argument("--ckpt", type=str, required=True, help="模型权重路径")
    ap.add_argument("--model-type", type=str, default="align_hetero", choices=["align_hetero", "dualhead_b", "mlp"])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--y-target", type=str, required=True, help="物理单位目标，逗号分隔")
    ap.add_argument("--weights", type=str, default=None, help="逐指标权重（标准化空间），逗号分隔")
    ap.add_argument("--goal", type=str, default="min,min,range,range,min", help="逐维目标: min/max/target/range")
    ap.add_argument("--ugf-band", type=str, default="1.2e6:3.0e6", help="UGF 的物理区间 (goal=range)")
    ap.add_argument("--pm-band", type=str, default="60:75", help="PM 的物理区间 (goal=range)")
    ap.add_argument("--n-init", type=int, default=256)
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--finish-lbfgs", type=int, default=60)
    ap.add_argument("--prior", type=float, default=1e-3, help="x_scaled 的 L2 先验系数；0 关闭")
    ap.add_argument("--init-npy", type=str, default=None, help="初值（标准化空间）.npy")
    ap.add_argument("--save-dir", type=str, default=str(RESULTS_DIR / "inverse" / "run"), help="保存输出目录")
    args = ap.parse_args()
    device = torch.device(args.device)

    # --- 加载 scaler (从 src/results/ 下) ---
    xs_path = RESULTS_DIR / f"{args.opamp}_x_scaler.gz"
    ys_path = RESULTS_DIR / f"{args.opamp}_y_scaler.gz"
    if not xs_path.exists() or not ys_path.exists():
        raise FileNotFoundError(f"未找到标准化器：{xs_path} 或 {ys_path}")
    import joblib
    x_scaler = joblib.load(xs_path)
    y_scaler = joblib.load(ys_path)

    X_DIM = int(x_scaler.mean_.shape[0])
    Y_DIM = int(y_scaler.mean_.shape[0])

    y_target_phys = parse_floats(args.y_target, Y_DIM)
    y_target_log = y_target_phys.copy()
    for idx in Y_LOG1P_INDEX:
        if idx < len(y_target_log):
            y_target_log[idx] = np.log1p(y_target_log[idx])
    y_target_scaled = (y_target_log - y_scaler.mean_) / y_scaler.scale_

    weights = parse_floats(args.weights, Y_DIM) if args.weights else None
    init_points_scaled = np.load(args.init_npy) if args.init_npy and os.path.isfile(args.init_npy) else None

    # --- 模型加载 (已修正) ---
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute() and not ckpt_path.exists():
        alt = RESULTS_DIR / ckpt_path.name
        if alt.exists(): ckpt_path = alt
        else: raise FileNotFoundError(f"未找到 ckpt: {args.ckpt} (也尝试了 {alt})")

    model = load_model_from_ckpt(
        ckpt_path=ckpt_path,
        model_type=args.model_type,
        device=device,
        opamp_type=args.opamp  # 传递 opamp_type 以便正确构建模型
    )

    diagnose_target_feasibility(args.opamp, y_scaler, y_target_phys)

    best_x_scaled, best_y_scaled, best_loss = optimize_x_multi_start(
        model=model, model_type=args.model_type, x_dim=X_DIM,
        y_target_scaled=y_target_scaled, x_scaler=x_scaler, y_scaler=y_scaler,
        n_init=args.n_init, steps=args.steps, lr=args.lr, weights=weights,
        init_points_scaled=init_points_scaled, device=args.device, grad_clip=args.grad_clip,
        opamp_type=args.opamp, goal=args.goal, ugf_band=args.ugf_band,
        pm_band=args.pm_band, prior=args.prior, finish_lbfgs=args.finish_lbfgs,
    )

    # === 结果还原与保存 ===
    x_mean, x_scale = x_scaler.mean_.astype(np.float64), x_scaler.scale_.astype(np.float64)
    best_x_phys = best_x_scaled[0] * x_scale + x_mean
    y_mean, y_scale = y_scaler.mean_.astype(np.float64), y_scaler.scale_.astype(np.float64)
    best_y_log = best_y_scaled[0] * y_scale + y_mean
    best_y_phys = best_y_log.copy()
    for idx in Y_LOG1P_INDEX:
        if idx < len(best_y_phys):
            best_y_phys[idx] = np.expm1(best_y_phys[idx])

    ynames = Y_NAMES[:Y_DIM]
    xnames = X_NAMES[:X_DIM]
    print("\n===== 反向设计结果 =====")
    print(f"目标 y ({ynames}):\n{fmt_arr(y_target_phys)}")
    print(f"预测 y ({ynames}):\n{fmt_arr(best_y_phys)}")
    print(f"建议 x ({xnames}):\n{fmt_arr(best_x_phys)}")
    print(f"最终约束损失: {best_loss:.6f}")

    save_dir = Path(args.save_dir)
    ensure_dir(save_dir)
    np.save(save_dir / "best_x_scaled.npy", best_x_scaled)
    np.save(save_dir / "best_y_scaled.npy", best_y_scaled)
    np.save(save_dir / "best_x_phys.npy",  best_x_phys)
    np.save(save_dir / "best_y_phys.npy",  best_y_phys)
    with open(save_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("===== 反向设计结果 =====\n")
        f.write(f"目标 y ({ynames}):\n{fmt_arr(y_target_phys)}\n")
        f.write(f"预测 y ({ynames}):\n{fmt_arr(best_y_phys)}\n")
        f.write(f"建议 x ({xnames}):\n{fmt_arr(best_x_phys)}\n")
        f.write(f"最终约束损失: {best_loss:.6f}\n")
    print(f"\n结果已保存到：{save_dir.resolve()}")


if __name__ == "__main__":
    main()
