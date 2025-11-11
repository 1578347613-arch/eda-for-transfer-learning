# src/probe_target_eval.py
# 用微调后的对齐模型在目标域(train/val/all)做离线推理，输出多种诊断结果
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- 项目内模块 ---
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.align_hetero import AlignHeteroMLP
from data_loader import get_data_and_scalers
from config import TASK_CONFIGS

# 某些项目里有 LOG_TRANSFORMED_COLS；若没有则回退为空
try:
    from config import LOG_TRANSFORMED_COLS
    Y_LOG_COLS = set(LOG_TRANSFORMED_COLS)
except Exception:
    Y_LOG_COLS = set()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _coerce_hidden_dims_from_config(cfg: dict):
    hd = cfg.get("hidden_dims", None)
    if isinstance(hd, str):
        import ast
        hd = ast.literal_eval(hd)
    if isinstance(hd, (list, tuple)):
        return list(hd)
    hidden_dim = cfg.get("hidden_dim", 256)
    num_layers = cfg.get("num_layers", 3)
    return [hidden_dim] * int(num_layers)


def load_forward_model(opamp_type: str, ckpt_path: Path) -> AlignHeteroMLP:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到模型权重: {ckpt_path}")
    cfg = TASK_CONFIGS[opamp_type]
    all_data = get_data_and_scalers(opamp_type=opamp_type)
    input_dim = all_data['source'][0].shape[1]
    output_dim = all_data['source'][1].shape[1]
    model = AlignHeteroMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=_coerce_hidden_dims_from_config(cfg),
        dropout_rate=cfg['dropout_rate']
    ).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model_state = state.get('state_dict', state)
    model.load_state_dict(model_state, strict=False)
    model.eval()
    print(f"[LOAD] {ckpt_path.name} | input={input_dim} output={output_dim}")
    return model


def to_physical_y(y_std: np.ndarray, y_scaler, colnames):
    """
    标准化域 -> 训练目标域 -> 物理域（对训练时做过 log1p 的列做 expm1）。
    """
    y_unstd = y_scaler.inverse_transform(y_std)
    y_phys = y_unstd.copy()
    for i, c in enumerate(colnames):
        if c in Y_LOG_COLS:
            y_phys[:, i] = np.expm1(y_unstd[:, i])
    return y_phys


def binned_mae_by_quantile(y_true, y_pred, qbins=5):
    qs = np.linspace(0, 1, qbins + 1)
    edges = np.quantile(y_true, qs)
    # 保证边界严格递增
    edges = np.unique(edges)
    bins = np.digitize(y_true, edges[1:-1], right=True)
    rows = []
    for b in range(len(edges)-1):
        m = bins == b
        if not np.any(m):
            rows.append((float(edges[b]), float(edges[b+1]), 0, np.nan))
        else:
            mae = mean_absolute_error(y_true[m], y_pred[m])
            rows.append((float(edges[b]), float(edges[b+1]), int(m.sum()), float(mae)))
    return pd.DataFrame(rows, columns=["bin_left", "bin_right", "count", "mae"])


@torch.no_grad()
def run_probe(
    opamp_type: str,
    ckpt_path: Path,
    split: str,
    out_dir: Path
):
    out_dir.mkdir(parents=True, exist_ok=True)
    data = get_data_and_scalers(opamp_type=opamp_type)

    # --- 选择 split ---
    if split == "train":
        X_std, Y_std = data["target_train"]
        X_raw_df = data["raw_target"][0].iloc[:len(X_std)].reset_index(drop=True)
        Y_raw_cols = list(data["raw_target"][1].columns)
    elif split == "val":
        X_std, Y_std = data["target_val"]
        # raw_target[0] 通常为整份 target（train+val），这里通过长度差得到 val 的尾段
        total_X_raw = data["raw_target"][0]
        n_train = data["target_train"][0].shape[0]
        X_raw_df = total_X_raw.iloc[n_train:n_train+len(X_std)].reset_index(drop=True)
        Y_raw_cols = list(data["raw_target"][1].columns)
    elif split == "all":
        Xa, Ya = data["target_train"]
        Xb, Yb = data["target_val"]
        X_std = np.concatenate([Xa, Xb], axis=0)
        Y_std = np.concatenate([Ya, Yb], axis=0)
        X_raw_df = data["raw_target"][0].reset_index(drop=True)
        Y_raw_cols = list(data["raw_target"][1].columns)
    else:
        raise ValueError("split 仅支持 train / val / all")

    # --- 模型 ---
    model = load_forward_model(opamp_type, ckpt_path)

    # --- 标准化域前向 ---
    X_t = torch.tensor(X_std, dtype=torch.float32, device=DEVICE)
    mu_std, logv_std, _ = model(X_t)
    mu_std = mu_std.cpu().numpy()
    logv_std = logv_std.cpu().numpy()

    # --- 物理域还原 ---
    y_scaler = data["y_scaler"]
    y_true_phys = to_physical_y(Y_std,  y_scaler, Y_raw_cols)
    y_pred_phys = to_physical_y(mu_std, y_scaler, Y_raw_cols)

    # --- 逐维指标 ---
    rows = []
    for j, name in enumerate(Y_raw_cols):
        yt = y_true_phys[:, j]
        yp = y_pred_phys[:, j]
        mse = mean_squared_error(yt, yp)
        mae = mean_absolute_error(yt, yp)
        r2  = r2_score(yt, yp)
        bias = float(np.mean(yp - yt))
        rows.append((name, mse, mae, r2, bias))
    metrics_df = pd.DataFrame(rows, columns=["metric", "MSE", "MAE", "R2", "Bias"])
    metrics_df = metrics_df.sort_values("R2")
    metrics_path = out_dir / f"metrics_{split}.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # --- 样本级残差导出（便于定位样本/范围问题） ---
    res_df = pd.DataFrame(index=np.arange(len(y_true_phys)))
    for j, name in enumerate(Y_raw_cols):
        res_df[f"y_true_{name}"] = y_true_phys[:, j]
        res_df[f"y_pred_{name}"] = y_pred_phys[:, j]
        res_df[f"err_{name}"] = y_pred_phys[:, j] - y_true_phys[:, j]
        res_df[f"abs_err_{name}"] = np.abs(res_df[f"err_{name}"])
    # 拼上原始输入（物理域）
    res_df = pd.concat([res_df, X_raw_df.reset_index(drop=True)], axis=1)
    res_path = out_dir / f"residuals_{split}.csv"
    res_df.to_csv(res_path, index=False)

    # --- 诊断 1：CMRR 残差与输入特征的相关（快速定位“哪类设计最容易错”） ---
    if "cmrr" in Y_raw_cols:
        cmrr_err = res_df["err_cmrr"].to_numpy()
        corr_rows = []
        for c in X_raw_df.columns:
            x = X_raw_df[c].to_numpy()
            # 保护常数列 / NaN
            if np.allclose(np.nanstd(x), 0.0) or np.isnan(x).any():
                continue
            r = np.corrcoef(x, cmrr_err)[0, 1]
            corr_rows.append((c, float(r), float(np.std(cmrr_err[X_raw_df[c].notna()]))))
        corr_df = pd.DataFrame(corr_rows, columns=["feature", "pearson_r_with_cmrr_residual", "cmrr_residual_std"])
        corr_df["abs_r"] = corr_df["pearson_r_with_cmrr_residual"].abs()
        corr_df = corr_df.sort_values("abs_r", ascending=False)
        corr_path = out_dir / f"corr_cmrr_residual_vs_x_{split}.csv"
        corr_df.to_csv(corr_path, index=False)

        # --- 诊断 2：按 CMRR 真值分位的分箱 MAE（是否在高/低区间失真更大） ---
        cmrr_true = res_df["y_true_cmrr"].to_numpy()
        cmrr_pred = res_df["y_pred_cmrr"].to_numpy()
        cmrr_bins_df = binned_mae_by_quantile(cmrr_true, cmrr_pred, qbins=5)
        bins_path = out_dir / f"binned_mae_cmrr_by_true_{split}.csv"
        cmrr_bins_df.to_csv(bins_path, index=False)
    else:
        corr_path = None
        bins_path = None

    # --- 诊断 3：不进入物理域的“方差校准检查”（标准化域） ---
    # 理论上 E[(y-mu)^2 / var] ≈ 1，偏离太大说明 logvar 未校准
    var_std = np.exp(logv_std)
    cal_rows = []
    for j, name in enumerate(Y_raw_cols):
        num = (Y_std[:, j] - mu_std[:, j]) ** 2
        ratio = num / (var_std[:, j] + 1e-12)
        cal_rows.append((name, float(np.mean(ratio))))
    calib_df = pd.DataFrame(cal_rows, columns=["metric", "mean_resid2_over_var"])
    calib_path = out_dir / f"calibration_resid2_over_var_{split}.csv"
    calib_df.to_csv(calib_path, index=False)

    # --- 打印摘要 ---
    print("\n=== Probe Summary ===============================")
    print(f"split          : {split}")
    print(f"out dir        : {out_dir}")
    print(f"metrics csv    : {metrics_path.name}")
    print(f"residuals csv  : {res_path.name}")
    if "cmrr" in Y_raw_cols:
        print(f"corr csv       : {corr_path.name}")
        print(f"binned MAE csv : {bins_path.name}")
    print(f"calib csv      : {calib_path.name}")
    print("\n-- Metrics (sorted by R2 ascending) --")
    print(metrics_df.to_string(index=False))

    # 返回几个关键对象，方便后续交互式分析（如需要）
    return {
        "metrics_df": metrics_df,
        "residuals_df": res_df,
        "calibration_df": calib_df,
    }


def main():
    parser = argparse.ArgumentParser(
        description="在目标域(train/val/all)做离线前向推理与诊断（5t_opamp）"
    )
    parser.add_argument("--ckpt",
                        type=str,
                        default="/home/mario1578347613/eda-for-transfer-learning-1/src/results/5t_opamp_finetuned.pth",
                        help="finetuned 模型路径")
    parser.add_argument("--opamp",
                        type=str,
                        default="5t_opamp",
                        choices=["5t_opamp"],
                        help="本脚本聚焦 5t_opamp")
    parser.add_argument("--split",
                        type=str,
                        default="train",
                        choices=["train", "val", "all"],
                        help="评估哪个目标域切分")
    parser.add_argument("--out_dir",
                        type=str,
                        default="results/probe_5t",
                        help="诊断输出目录")
    args = parser.parse_args()

    run_probe(
        opamp_type=args.opamp,
        ckpt_path=Path(args.ckpt),
        split=args.split,
        out_dir=PROJECT_ROOT / args.out_dir
    )


if __name__ == "__main__":
    main()
