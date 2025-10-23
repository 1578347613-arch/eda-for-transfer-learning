# evaluation/evaluate.py
import os
import joblib
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import config
from data_loader import get_data_and_scalers
from models.dual_head_mlp import DualHeadMLP

DEVICE = torch.device(config.DEVICE)
OPAMP_TYPE = config.OPAMP_TYPE
DEFAULT_COLS = ['slewrate_pos', 'dc_gain', 'ugf', 'phase_margin', 'cmrr']


def _pick_existing_path(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def _load_state(obj):
    # 兼容 {"state_dict": ...} / 直接 state_dict
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    return obj


def main():
    # 1) 数据（target 验证集）
    data = get_data_and_scalers(opamp_type=OPAMP_TYPE)
    X_val, y_val = data['target_val']
    input_dim, output_dim = X_val.shape[1], y_val.shape[1]

    # 2) 标准化器（只需要 y_scaler）
    y_scaler_path = _pick_existing_path([
        f"results/{OPAMP_TYPE}_y_scaler.gz",
        f"../results/{OPAMP_TYPE}_y_scaler.gz",
    ])
    if y_scaler_path is None:
        raise FileNotFoundError(
            "未找到 y_scaler：请确认存在以下任一文件：\n"
            f"  - results/{OPAMP_TYPE}_y_scaler.gz\n"
            f"  - ../results/{OPAMP_TYPE}_y_scaler.gz"
        )
    y_scaler = joblib.load(y_scaler_path)

    # 3) 模型与权重
    model = DualHeadMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS
    ).to(DEVICE)

    ckpt_path = _pick_existing_path([
        f"results/{OPAMP_TYPE}_dualhead_finetuned.pth",
        f"../results/{OPAMP_TYPE}_dualhead_finetuned.pth",
    ])
    if ckpt_path is None:
        raise FileNotFoundError(
            "未找到微调后的 DualHeadMLP 权重：\n"
            f"  - results/{OPAMP_TYPE}_dualhead_finetuned.pth\n"
            f"  - ../results/{OPAMP_TYPE}_dualhead_finetuned.pth"
        )

    state_raw = torch.load(ckpt_path, map_location=DEVICE)
    state = _load_state(state_raw)
    # 对于 DualHeadMLP（backbone 是 nn.Sequential），严格加载应当可行
    model.load_state_dict(state, strict=True)
    model.eval()

    # 4) 预测（标准化空间）
    with torch.no_grad():
        x_t = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
        pred_scaled = model(x_t, domain='B').cpu().numpy()

    # 5) 反标准化到物理单位
    y_pred = y_scaler.inverse_transform(pred_scaled)
    y_true = y_scaler.inverse_transform(y_val)

    # 若训练阶段对部分指标做了 log1p，这里需要 expm1 还原
    cols = DEFAULT_COLS if output_dim == len(DEFAULT_COLS) else [f"y{i}" for i in range(output_dim)]
    for j, name in enumerate(cols):
        if name in ['ugf', 'cmrr']:
            y_pred[:, j] = np.expm1(y_pred[:, j])
            y_true[:, j] = np.expm1(y_true[:, j])

    # 6) 逐列指标
    mse = mean_squared_error(y_true, y_pred, multioutput='raw_values')
    mae = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
    r2  = np.array([r2_score(y_true[:, i], y_pred[:, i]) for i in range(output_dim)])

    print("\n=== Target(B) 验证集指标（物理单位）===")
    for i, name in enumerate(cols):
        print(f"{name:14s}  MSE={mse[i]:.4g}  MAE={mae[i]:.4g}  R2={r2[i]:.4f}")

    # 7) 可选：整体平均
    print("\nAvg  (all dims)   MSE={:.4g}  MAE={:.4g}  R2={:.4f}".format(
        mse.mean(), mae.mean(), r2.mean()
    ))


if __name__ == "__main__":
    main()
