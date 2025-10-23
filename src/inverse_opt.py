# -*- coding: utf-8 -*-
"""
inverse_opt.py
反向设计（在“标准化空间”直接优化 x，使预测 y 满足目标/区间约束）

功能要点：
1) 利用 data_loader 的预处理边界（真实物理 min/max），并在优化中做“物理盒约束投影”；
2) 约束损失在“标准化空间(z-score)”计算，稳定不爆梯度：
   - 支持 goal: min/max/target/range（逐维）
   - 支持 UGF/PM 区间（物理单位→自动换算到 z-score）
3) x 的 L2 先验（信赖域）可开关，帮助收敛回数据主体；
4) 可选 LBFGS 精修（收敛更细致）；
5) 自带目标“可达性诊断”（看目标是否远超分布边界）；
6) 兼容 align_hetero / dualhead_b 等模型类型；权重加载更健壮(strict=False)。

使用示例（与上文建议一致）：
python src/inverse_opt.py \
  --opamp 5t_opamp \
  --ckpt results/5t_opamp_align_hetero_lambda0.050.pth \
  --model-type align_hetero \
  --y-target "2.5e8,200,1.5e6,65,20000" \
  --goal "min,min,range,range,min" \
  --ugf-band "8.0e5:2.0e6" \
  --pm-band "60:75" \
  --weights "0.05,0.40,0.90,0.10,0.65" \
  --prior 1e-3 \
  --init-npy results/inverse/init_1024.npy \
  --n-init 1024 --steps 900 --lr 0.002 \
  --finish-lbfgs 80 \
  --save-dir results/inverse/try_hybrid_constrained_scaled_v2
"""

import os
import sys
import math
import json
import argparse
import numpy as np
import torch

# ==== 常量（按你仓库定义） ====
X_DIM = 7
Y_DIM = 5
# y 的第 2、4 维（0-based）是 log1p 域：ugf(2), cmrr(4)
Y_LOG1P_INDEX = [2, 4]
Y_NAMES = ["slewrate_pos", "dc_gain", "ugf", "phase_margin", "cmrr"]
X_NAMES = ["w1","w2","w3","l1","l2","l3","ibias"]


# ==== 工具函数 ====
def parse_floats(csv_str, n=None):
    arr = [float(x.strip()) for x in csv_str.split(",")]
    if (n is not None) and (len(arr) != n):
        raise ValueError(f"需要 {n} 个数值，实际 {len(arr)}: {csv_str}")
    return np.asarray(arr, dtype=np.float64)


def fmt_arr(a):
    return "[" + " ".join([f"{v:.6g}" for v in a]) + "]"


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d


# ==== 数据与边界 ====
def get_all_scaled_X(opamp_type):
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
    """基于 data_loader 的标准化数据，得到：
       - 标准化空间的 min/max（直接对 X_scaled 取 min/max）
       - 物理空间的 min/max（先 inverse 回物理单位，再取 min/max）
       并返回 torch 张量与 x 的 mean/scale（用于往返）
    """
    X_scaled = get_all_scaled_X(opamp_type)
    x_min_scaled = X_scaled.min(axis=0)
    x_max_scaled = X_scaled.max(axis=0)

    X_phys = x_scaler.inverse_transform(X_scaled)
    x_min_phys = X_phys.min(axis=0)
    x_max_phys = X_phys.max(axis=0)

    x_min_phys_t = torch.from_numpy(x_min_phys.astype(np.float32)).to(device)
    x_max_phys_t = torch.from_numpy(x_max_phys.astype(np.float32)).to(device)
    x_min_scaled_t = torch.from_numpy(x_min_scaled.astype(np.float32)).to(device)
    x_max_scaled_t = torch.from_numpy(x_max_scaled.astype(np.float32)).to(device)
    mean_t = torch.from_numpy(x_scaler.mean_.astype(np.float32)).to(device)
    scale_t = torch.from_numpy(x_scaler.scale_.astype(np.float32)).to(device)

    return x_min_phys_t, x_max_phys_t, x_min_scaled_t, x_max_scaled_t, mean_t, scale_t


# ==== 模型加载 ====
def build_model(model_type: str, input_dim=X_DIM, output_dim=Y_DIM,
                hidden_dim=512, num_layers=6, dropout=0.1):
    from models import MLP, DualHeadMLP, AlignHeteroMLP
    mt = model_type.lower()
    if mt == "align_hetero":
        return AlignHeteroMLP(input_dim, output_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout_rate=dropout)
    elif mt == "dualhead_b":
        # DualHeadMLP 前向返回 (y_a, y_b)，我们只会取 b 头
        return DualHeadMLP(input_dim, output_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout_rate=dropout)
    elif mt == "mlp":
        return MLP(input_dim, output_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout_rate=dropout)
    else:
        raise ValueError(f"未知 model_type: {model_type}")


def load_model_from_ckpt(ckpt_path, model_type, device,
                         hidden_dim=512, num_layers=6, dropout=0.1):
    model = build_model(model_type, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    ckpt = torch.load(ckpt_path, map_location=device)
    # 兼容多种保存格式
    if isinstance(ckpt, dict):
        state = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
    else:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("[load warning] missing keys:", len(missing), "unexpected keys:", len(unexpected))
    model.to(device).eval()
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
        Y_phys[:, idx] = np.expm1(Y_phys[:, idx])
    q = np.percentile(Y_phys, [1, 5, 50, 95, 99], axis=0)

    # 目标的 z-score
    y_phys = y_target_physical.copy()
    y_log = y_phys.copy()
    y_log[Y_LOG1P_INDEX] = np.log1p(y_log[Y_LOG1P_INDEX])
    z = (y_log - y_scaler.mean_) / y_scaler.scale_

    print("\n[诊断] 目标在训练分布中的位置（物理单位 & z-score）")
    for j, nm in enumerate(Y_NAMES):
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
    """
    在“标准化空间”优化 x（z-score），使 y_pred 符合“约束感知”的目标。
    返回：best_x_scaled, best_y_scaled, best_loss
    """
    device_t = torch.device(device)
    model.eval()

    # 目标 y（标准化张量）
    y_t = torch.from_numpy(y_target_scaled.astype(np.float32)).to(device_t).view(1, -1)

    # 权重（标准化空间）
    if weights is None:
        w = torch.ones(Y_DIM, dtype=torch.float32, device=device_t)
    else:
        w = torch.from_numpy(weights.astype(np.float32)).to(device_t)
    w = w.view(1, -1)

    # 真实边界（物理 & 标准化）
    x_min_phys, x_max_phys, x_min_scaled, x_max_scaled, mean_t, scale_t = \
        get_bounds_and_scalers(opamp_type, x_scaler, device_t)

    # 初值
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
    best_x = None
    best_y = None

    # === 标准化空间的不等式/区间损失（稳定） ===
    goal_list = [g.strip().lower() for g in goal.split(",")]
    if len(goal_list) != Y_DIM:
        raise ValueError(f"--goal 需要 {Y_DIM} 个，用逗号分隔；收到 {len(goal_list)} 个。")

    # 辅助：把“物理阈值”换算成标准化阈值
    y_mean = y_scaler.mean_.astype(np.float64)
    y_scale = y_scaler.scale_.astype(np.float64)

    def phys_to_scaled_scalar(j: int, val_phys: float) -> float:
        v = math.log1p(val_phys) if j in Y_LOG1P_INDEX else float(val_phys)
        return float((v - y_mean[j]) / y_scale[j])

    # UGF 区间（物理→标准化）
    ugf_lb_phys, ugf_ub_phys = [float(s) for s in ugf_band.split(":")]
    ugf_lb_scaled = phys_to_scaled_scalar(2, ugf_lb_phys)
    ugf_ub_scaled = phys_to_scaled_scalar(2, ugf_ub_phys)

    # PM 区间（物理→标准化）
    pm_lb_phys, pm_ub_phys = [float(s) for s in pm_band.split(":")]
    pm_lb_scaled = phys_to_scaled_scalar(3, pm_lb_phys)
    pm_ub_scaled = phys_to_scaled_scalar(3, pm_ub_phys)

    def scaled_constraint_loss(y_pred_scaled: torch.Tensor) -> torch.Tensor:
        # y_pred_scaled 是 z-score
        total = 0.0

        # 展开目标（z-score）
        y_t_scaled = y_t.expand_as(y_pred_scaled)

        for j in range(Y_DIM):
            gj = goal_list[j]
            yj = y_pred_scaled[:, j]
            wj = w[:, j]

            if gj == "target":
                total = total + torch.mean(wj * (yj - y_t_scaled[:, j])**2)

            elif gj == "min":
                total = total + torch.mean(wj * torch.relu(y_t_scaled[:, j] - yj)**2)

            elif gj == "max":
                total = total + torch.mean(wj * torch.relu(yj - y_t_scaled[:, j])**2)

            elif gj == "range":
                if j == 2:  # UGF
                    total = total + torch.mean(wj * torch.relu(ugf_lb_scaled - yj)**2)
                    total = total + torch.mean(wj * torch.relu(yj - ugf_ub_scaled)**2)
                elif j == 3:  # PM
                    total = total + torch.mean(wj * torch.relu(pm_lb_scaled - yj)**2)
                    total = total + torch.mean(wj * torch.relu(yj - pm_ub_scaled)**2)
                else:
                    # 其它维如设成 range，则退化为 target
                    total = total + torch.mean(wj * (yj - y_t_scaled[:, j])**2)
            else:
                raise ValueError(f"未知 goal: {gj}")

        # x 的 L2 先验（信赖域），抑制离群解
        if prior and prior > 0:
            total = total + prior * torch.mean(torch.sum(x**2, dim=1))

        return total

    # === 训练循环 ===
    for t in range(steps):
        opt.zero_grad(set_to_none=True)
        y_pred = model(x)
        if model_type.lower() == "align_hetero":
            if isinstance(y_pred, (list, tuple)):
                y_pred = y_pred[0]  # 取 mu
        elif model_type.lower() == "dualhead_b":
            if isinstance(y_pred, (list, tuple)):
                y_pred = y_pred[-1]  # 取 B 头
        # 损失
        loss = scaled_constraint_loss(y_pred)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([x], grad_clip)
        opt.step()

        # 物理盒约束投影：x_scaled → x_phys → clamp → x_scaled
        with torch.no_grad():
            x_phys = x * scale_t + mean_t
            x_phys.clamp_(x_min_phys, x_max_phys)
            x.copy_((x_phys - mean_t) / scale_t)

        cur = float(loss.detach().cpu().item())
        if cur < best_loss:
            best_loss = cur
            best_x = x.detach().clone()
            best_y = y_pred.detach().clone()

    # === 可选 LBFGS 精修 ===
    if finish_lbfgs and finish_lbfgs > 0:
        x_lb = best_x.clone().detach().requires_grad_(True)
        lbfgs = torch.optim.LBFGS([x_lb], lr=1.0, max_iter=finish_lbfgs,
                                  history_size=20, line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            y_pred2 = model(x_lb)
            if model_type.lower() == "align_hetero" and isinstance(y_pred2, (list, tuple)):
                y_pred2 = y_pred2[0]
            elif model_type.lower() == "dualhead_b" and isinstance(y_pred2, (list, tuple)):
                y_pred2 = y_pred2[-1]
            loss2 = scaled_constraint_loss(y_pred2)
            loss2.backward()
            return loss2

        lbfgs.step(closure)
        with torch.no_grad():
            x_phys = x_lb * scale_t + mean_t
            x_phys.clamp_(x_min_phys, x_max_phys)
            x_lb.copy_((x_phys - mean_t) / scale_t)

        with torch.no_grad():
            y_pred2 = model(x_lb)
            if model_type.lower() == "align_hetero" and isinstance(y_pred2, (list, tuple)):
                y_pred2 = y_pred2[0]
            elif model_type.lower() == "dualhead_b" and isinstance(y_pred2, (list, tuple)):
                y_pred2 = y_pred2[-1]
            loss2 = float(scaled_constraint_loss(y_pred2).detach().cpu().item())
            if loss2 <= best_loss:
                best_loss = loss2
                best_x = x_lb.detach().clone()
                best_y = y_pred2.detach().clone()

    return best_x.cpu().numpy(), best_y.cpu().numpy(), best_loss


# ==== 主程序 ====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opamp", type=str, default="5t_opamp",
                    help="数据/scaler 的类型标识（与 data_loader 一致）")
    ap.add_argument("--ckpt", type=str, required=True, help="模型权重路径")
    ap.add_argument("--model-type", type=str, default="align_hetero",
                    choices=["align_hetero", "dualhead_b", "mlp"])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--num-layers", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--y-target", type=str, required=True,
                    help="物理单位目标，逗号分隔 5 项（slewrate_pos,dc_gain,ugf,phase_margin,cmrr）")
    ap.add_argument("--weights", type=str, default=None,
                    help="逐指标权重（标准化空间），逗号分隔 5 项；缺省为全1")
    ap.add_argument("--goal", type=str, default="min,min,range,range,min",
                    help="逐维目标: min/max/target/range（与 y 的 5 维顺序一致）")
    ap.add_argument("--ugf-band", type=str, default="1.2e6:3.0e6",
                    help="UGF 的物理区间(Hz)，格式 lower:upper（仅当该维 goal=range 时生效）")
    ap.add_argument("--pm-band", type=str, default="60:75",
                    help="PM 的物理区间(度)，格式 lower:upper（仅当该维 goal=range 时生效）")

    ap.add_argument("--n-init", type=int, default=256)
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--finish-lbfgs", type=int, default=60)

    ap.add_argument("--prior", type=float, default=1e-3,
                    help="x_scaled 的 L2 先验系数；0 关闭")

    ap.add_argument("--init-npy", type=str, default=None,
                    help="初值（标准化空间）.npy；若为空将从边界内均匀采样")
    ap.add_argument("--save-dir", type=str, default="results/inverse/run",
                    help="保存输出的目录")

    args = ap.parse_args()

    device = torch.device(args.device)

    # 载入 scaler（直接从 results/ 目录；若你另存了路径，可做成参数）
    # 为了稳妥起见，尽量使用 data_loader 保存的同名 scaler 文件
    # 约定命名：results/{opamp}_x_scaler.gz / results/{opamp}_y_scaler.gz
    try:
        import joblib
        xs_path = f"results/{args.opamp}_x_scaler.gz"
        ys_path = f"results/{args.opamp}_y_scaler.gz"
        x_scaler = joblib.load(xs_path)
        y_scaler = joblib.load(ys_path)
    except Exception as e:
        print("[warn] 无法从 results/ 加载 scaler，退回 data_loader 计算的 scaler。", e)
        # 直接从 data_loader 重新拟合（与训练一致：仅在 A域拟合）
        from data_loader import get_data_and_scalers
        _ = get_data_and_scalers(opamp_type=args.opamp)  # 该函数会在 results/ 保存 scaler
        import joblib
        xs_path = f"results/{args.opamp}_x_scaler.gz"
        ys_path = f"results/{args.opamp}_y_scaler.gz"
        x_scaler = joblib.load(xs_path)
        y_scaler = joblib.load(ys_path)

    # 目标（物理 → log1p 某些维 → 标准化）
    y_target_phys = parse_floats(args.y_target, Y_DIM)
    y_target_log = y_target_phys.copy()
    y_target_log[Y_LOG1P_INDEX] = np.log1p(y_target_log[Y_LOG1P_INDEX])
    y_target_scaled = (y_target_log - y_scaler.mean_) / y_scaler.scale_

    # 权重
    weights = parse_floats(args.weights, Y_DIM) if args.weights else None

    # 初值
    init_points_scaled = None
    if args.init_npy and os.path.isfile(args.init_npy):
        init_points_scaled = np.load(args.init_npy)

    # 模型
    model = load_model_from_ckpt(
        args.ckpt, args.model_type, device,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout
    )

    # 目标可达性诊断（一次性打印，便于判断是否离群）
    diagnose_target_feasibility(args.opamp, y_scaler, y_target_phys)

    # 优化
    best_x_scaled, best_y_scaled, best_loss = optimize_x_multi_start(
        model=model,
        model_type=args.model_type,
        x_dim=X_DIM,
        y_target_scaled=y_target_scaled,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        n_init=args.n_init,
        steps=args.steps,
        lr=args.lr,
        weights=weights,
        init_points_scaled=init_points_scaled,
        device=args.device,
        grad_clip=args.grad_clip,
        opamp_type=args.opamp,
        goal=args.goal,
        ugf_band=args.ugf_band,
        pm_band=args.pm_band,      # ← 注意：现在 pm_band 有显式参数，不会再“找不到”
        prior=args.prior,
        finish_lbfgs=args.finish_lbfgs,
    )

    # === 结果还原：x 到物理，y 到物理（含 expm1） ===
    x_mean = x_scaler.mean_.astype(np.float64)
    x_scale = x_scaler.scale_.astype(np.float64)
    best_x_phys = best_x_scaled[0] * x_scale + x_mean

    y_mean = y_scaler.mean_.astype(np.float64)
    y_scale = y_scaler.scale_.astype(np.float64)
    best_y_log = best_y_scaled[0] * y_scale + y_mean
    best_y_phys = best_y_log.copy()
    for idx in Y_LOG1P_INDEX:
        best_y_phys[idx] = np.expm1(best_y_phys[idx])

    # 打印结果
    print("\n===== 反向设计结果 =====")
    print(f"目标 y（物理单位，{Y_NAMES}）:")
    print(fmt_arr(y_target_phys))
    print(f"预测 y（物理单位，{Y_NAMES}）:")
    print(fmt_arr(best_y_phys))
    print(f"建议 x（物理单位，顺序 {X_NAMES}）:")
    print(fmt_arr(best_x_phys))
    print(f"最终约束损失（标准化空间）: {best_loss:.6f}")

    # 保存
    save_dir = ensure_dir(args.save_dir)
    np.save(os.path.join(save_dir, "best_x_scaled.npy"), best_x_scaled)
    np.save(os.path.join(save_dir, "best_y_scaled.npy"), best_y_scaled)
    np.save(os.path.join(save_dir, "best_x_phys.npy"), best_x_phys)
    np.save(os.path.join(save_dir, "best_y_phys.npy"), best_y_phys)

    with open(os.path.join(save_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("===== 反向设计结果 =====\n")
        f.write(f"目标 y（物理单位，{Y_NAMES}）:\n{fmt_arr(y_target_phys)}\n")
        f.write(f"预测 y（物理单位，{Y_NAMES}）:\n{fmt_arr(best_y_phys)}\n")
        f.write(f"建议 x（物理单位，顺序 {X_NAMES}）:\n{fmt_arr(best_x_phys)}\n")
        f.write(f"最终约束损失（标准化空间）: {best_loss:.6f}\n")

    print(f"\n结果已保存到：{save_dir}")


if __name__ == "__main__":
    main()
