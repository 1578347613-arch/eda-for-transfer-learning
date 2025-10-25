# evaluate.py

import config
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import List

# --- 常量定义 ---
DEFAULT_COLS = ['slewrate_pos', 'dc_gain', 'ugf', 'phase_margin', 'cmrr']

LOG_TRANSFORMED_COLS = config.LOG_TRANSFORMED_COLS


def calculate_and_print_metrics(
    y_pred_scaled: np.ndarray,
    y_true_scaled: np.ndarray,
    y_scaler,
    output_cols: List[str] = None
):
    """
    接收标准化的预测值和真实值，执行反标准化、指标计算和打印。
    这个函数是纯粹的计算器，不依赖任何模型。
    """
    print("\n--- [评估阶段] 开始计算指标 ---")

    # 1. 反标准化到物理单位
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_true = y_scaler.inverse_transform(y_true_scaled)

    # 2. 确定列名并还原对数变换
    output_dim = y_true.shape[1]
    if output_cols is None:
        output_cols = DEFAULT_COLS if output_dim == len(
            DEFAULT_COLS) else [f"y{i}" for i in range(output_dim)]

    for col_name in LOG_TRANSFORMED_COLS:
        if col_name in output_cols:
            col_idx = output_cols.index(col_name)
            y_pred[:, col_idx] = np.expm1(y_pred[:, col_idx])
            y_true[:, col_idx] = np.expm1(y_true[:, col_idx])

    # 3. 计算指标
    mse = mean_squared_error(y_true, y_pred, multioutput='raw_values')
    mae = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
    r2 = np.array([r2_score(y_true[:, i], y_pred[:, i])
                  for i in range(output_dim)])

    # 4. 打印结果
    print("\n=== 目标域验证集指标（物理单位）===")
    for i, name in enumerate(output_cols):
        print(
            f"{name:14s}  MSE={mse[i]:.4g}  MAE={mae[i]:.4g}  R2={r2[i]:.4f}")

    print("\nAvg  (all dims)   MSE={:.4g}  MAE={:.4g}  R2={:.4f}".format(
        mse.mean(), mae.mean(), r2.mean()
    ))
