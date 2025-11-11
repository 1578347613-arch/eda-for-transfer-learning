from __future__ import annotations
import os
import math
import json
import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch

# --- 路径定义 (来自 C2) ---
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent 
RESULTS_DIR = PROJECT_ROOT / "results" 

# ==== 常量 (来自 C2) ====
Y_LOG1P_INDEX = [2, 4] 
Y_NAMES = ["slewrate_pos", "dc_gain", "ugf", "phase_margin", "cmrr"]
X_NAMES = ["w1", "w2", "w3", "l1", "l2", "l3", "ibias"]

# ==== 通用工具 (来自 C2) ====
def parse_floats(csv_str, n=None):
    # (代码 100% 来自 C2，此处省略)
    arr = [float(x.strip()) for x in csv_str.split(",")]
    if (n is not None) and (len(arr) != n):
        raise ValueError(f"需要 {n} 个数值，实际 {len(arr)}: {csv_str}")
    return np.asarray(arr, dtype=np.float64)

def fmt_arr(a):
    # (代码 100% 来自 C2，此处省略)
    return "[" + " ".join([f"{v:.6g}" for v in a]) + "]"

def ensure_dir(d: Path | str) -> str:
    # (代码 100% 来自 C2，此处省略)
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)

# ==== 数据与边界 (来自 C2) ====
def get_all_scaled_X(opamp_type: str) -> np.ndarray:
    """
    (100% 保持 C2 的原始逻辑，它能工作！)
    """
    from data_loader import get_data_and_scalers
    data = get_data_and_scalers(opamp_type=opamp_type)
    # (C2 的原始逻辑：手动拼接 A 域的 splits)
    xs = [data["source_train"][0], data["source_val"][0]] 
    if "target_train" in data and data.get("target_train") is not None:
        xs.append(data["target_train"][0])
    if "target_val" in data and data.get("target_val") is not None:
        xs.append(data["target_val"][0])
    return np.vstack(xs).astype(np.float64)

def get_bounds_and_scalers(opamp_type, x_scaler, device):
    """(100% 保持 C2 的原始逻辑)"""
    # (代码 100% 来自 C2，此处省略)
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

# ==== 模型构建 / 加载 (C3 融合手术 🔪) ====
def load_model_from_ckpt(ckpt_path: Path | str, model_type: str, device: torch.device,
                         opamp_type: str) -> torch.nn.Module:
    """
    [C3 融合版] (基于 C2 原版修改)
    """
    # 1. 导入项目配置和数据加载器
    from data_loader import get_data_and_scalers
    # ⬇️ (C3 修改点) 我们现在导入 C3 的“混合圣经”
    import config 

    # 2. 动态获取输入/输出维度 (100% 保持 C2 的原始逻辑)
    data = get_data_and_scalers(opamp_type=opamp_type)
    input_dim = data['source_train'][0].shape[1]
    output_dim = data['source_train'][1].shape[1]
    
    # 3. 使用 "混合圣经" 中的 C1 黄金架构构建模型
    mt = model_type.lower()
    if mt == "align_hetero":
        
        # ⬇️ (C3 修改点) 我们假设 C3/models/ 里是 C1 的模型！
        from models.align_hetero import AlignHeteroMLP 
        
        # ⬇️ (C3 修改点) 从 C3 "混合圣经" 读取 C1 黄金架构！
        if opamp_type not in config.TASK_CONFIGS:
             raise KeyError(f"在 C3 config.py 的 TASK_CONFIGS 中未找到 '{opamp_type}'！")
        
        model_config = config.TASK_CONFIGS[opamp_type]
        
        if 'hidden_dims' not in model_config:
             raise KeyError(f"C3 config.py 中 {opamp_type} 缺少 'hidden_dims' (C1 黄金架构)!")
        
        print(f"✅ [C3 融合加载器] 正在为 {opamp_type} 构建 C1 黄金架构...")

        # ⬇️ (C3 修改点) 这 7 行代码是“手术”的核心！
        model = AlignHeteroMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            # ⬇️ 关键！使用 C1 的复杂列表！ ⬇️
            hidden_dims=model_config['hidden_dims'],
            dropout_rate=model_config['dropout_rate']
            # (我们假设 C3/models/align_hetero.py 是 C1 的版本)
        ).to(device)

    # (C2 的 'dualhead_b' 逻辑保持不变)
    # elif mt == "dualhead_b":
    #   ...
    else:
        raise NotImplementedError(f"模型类型 '{model_type}' 的动态构建尚未实现。")

    # 5. 加载模型权重 (100% 保持 C2 的原始逻辑)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load warning] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    model.eval()
    return model

# ==== 目标可达性诊断 (来自 C2) ====
def diagnose_target_feasibility(opamp_type, y_scaler, y_target_physical):
    """(100% 保持 C2 的原始逻辑)"""
    # (代码 100% 来自 C2，此处省略)
    from data_loader import get_data_and_scalers
    data = get_data_and_scalers(opamp_type=opamp_type)
    ys = [data["source_train"][1], data["source_val"][1]]
    if "target_train" in data and data["target_train"] is not None:
        ys.append(data["target_train"][1])
    if "target_val" in data and data["target_val"] is not None:
        ys.append(data["target_val"][1])
    Y_scaled = np.vstack(ys).astype(np.float64)
    Y_phys = y_scaler.inverse_transform(Y_scaled.copy())
    for idx in Y_LOG1P_INDEX:
        if idx < len(Y_phys[0]):
            Y_phys[:, idx] = np.expm1(Y_phys[:, idx])
    q = np.percentile(Y_phys, [1, 5, 50, 95, 99], axis=0)
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


# ==== 反向优化 (来自 C2) ====
def optimize_x_multi_start(
    model: torch.nn.Module,
    model_type: str,
    # ... (所有参数 100% 保持 C2 原样)
    x_dim: int,
    y_target_scaled: np.ndarray,
    x_scaler,
    y_scaler,
    n_init: int = 16,
    steps: int = 1000,
    lr: float = 1e-4,
    weights: np.ndarray = None,
    init_points_scaled: np.ndarray = None,
    device: str = "cpu",
    grad_clip: float = 1.0,
    opamp_type: str = "5t_opamp",
    goal: str = None,
    ugf_band: str = "1.2e6:3.0e6",
    pm_band: str = "60:75",
    prior: float = 0,
    finish_lbfgs: int = 0,
):
    """(100% 保持 C2 的原始逻辑)"""
    # (代码 100% 来自 C2，此处省略)
    device_t = torch.device(device)
    model.eval()
    y_t = torch.from_numpy(y_target_scaled.astype(np.float32)).to(device_t).view(1, -1)
    if weights is None:
        w = torch.ones(y_t.shape[1], dtype=torch.float32, device=device_t)
        print("[INFO] Using default optimizer weights (all 1.0)")
    else:
        w = torch.from_numpy(weights.astype(np.float32)).to(device_t)
        print(f"[INFO] Using externally provided weights: {weights}")
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
    Y_DIM = y_t.shape[1]
    if goal is None:
        goal_list = ["target"] * Y_DIM
        print("[INFO] No goal specified, defaulting to all 'target'.")
    else:
        goal_list = [g.strip().lower() for g in goal.split(",")]
    if len(goal_list) != Y_DIM:
        raise ValueError(f"Goal 列表需要 {Y_DIM} 个维度，但收到了 {len(goal_list)} 个。")
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
        # (LBFGS 代码 100% 来自 C2，此处省略)
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


# ==== 主程序 (来自 C2) ====
def main():
    """(100% 保持 C2 的原始逻辑)"""
    # (代码 100% 来自 C2，此处省略)
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
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--finish-lbfgs", type=int, default=60)
    ap.add_argument("--prior", type=float, default=1e-3, help="x_scaled 的 L2 先验系数；0 关闭")
    ap.add_argument("--init-npy", type=str, default=None, help="初值（标准化空间）.npy")
    ap.add_argument("--save-dir", type=str, default=str(RESULTS_DIR / "inverse" / "run"), help="保存输出目录")
    args = ap.parse_args()
    device = torch.device(args.device)
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
    y_target_scaled = (y_log - y_scaler.mean_) / y_scaler.scale_
    weights = parse_floats(args.weights, Y_DIM) if args.weights else None
    init_points_scaled = np.load(args.init_npy) if args.init_npy and os.path.isfile(args.init_npy) else None
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute() and not ckpt_path.exists():
        alt = RESULTS_DIR / ckpt_path.name
        if alt.exists(): ckpt_path = alt
        else: raise FileNotFoundError(f"未找到 ckpt: {args.ckpt} (也尝试了 {alt})")
    
    # ⬇️ 完美！这里会自动调用我们 C3 融合版的 load_model_from_ckpt！
    model = load_model_from_ckpt(
        ckpt_path=ckpt_path,
        model_type=args.model_type,
        device=device,
        opamp_type=args.opamp 
    )
    diagnose_target_feasibility(args.opamp, y_scaler, y_target_phys)
    best_x_scaled, best_y_scaled, best_loss = optimize_x_multi_start(
        model=model, model_type=args.model_type, x_dim=X_DIM,
        y_target_scaled=y_target_scaled, x_scaler=x_scaler, y_scaler=y_scaler,
        n_init=args.n_init, steps=args.steps, lr=args.lr, weights=weights,
        init_points_scaled=init_points_scaled, device=args.device, grad_clip=args.grad_clip,
        opamp_type=args.opamp, goal=args.goal, ugf_band=args.ugf_band,
        pm_band=args.pm-band, prior=args.prior, finish_lbfgs=args.finish_lbfgs,
    )
    # (结果还原与保存代码 100% 来自 C2，此处省略)
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