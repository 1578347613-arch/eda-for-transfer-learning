#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
freeze_bounds.py

在「训练环境」运行一次：
- 使用 data_loader.get_data_and_scalers(opamp_type=...)
- 拼接全部训练/验证（source/target）的 X_scaled
- 用对应的 x_scaler.inverse_transform 得到物理空间
- 计算真实的 min/max（物理 & 标准化）
- 冻结到 src/results/bounds_{opamp}.npz

推理脚本只需读取这些 npz，不再访问训练集。
"""

from pathlib import Path
import numpy as np
import joblib

from data_loader import get_data_and_scalers
import config  # 确保和训练时同一个 config

# 假定当前脚本在项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = SRC_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OPAMPS = ["5t_opamp", "two_stage_opamp"]


def _collect_all_scaled_X(data: dict) -> np.ndarray:
    xs = []
    # 按你 unified pipeline 的约定：这些键里存的是 (X_scaled, y_scaled)
    for key in ["source_train", "source_val", "target_train", "target_val"]:
        if key in data and data[key] is not None:
            X = data[key][0]
            if X is not None and len(X) > 0:
                xs.append(X)
    if not xs:
        raise RuntimeError("get_data_and_scalers 返回的数据里没有可用的 X。")
    X_all = np.vstack(xs).astype(np.float64)
    return X_all


def main():
    print(f"[paths] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[paths] SRC_DIR      = {SRC_DIR}")
    print(f"[paths] RESULTS_DIR  = {RESULTS_DIR}")
    print("=" * 60)

    for opamp in OPAMPS:
        print(f"\n=== 冻结边界: {opamp} ===")
        data = get_data_and_scalers(opamp_type=opamp)
        x_scaler = data["x_scaler"]

        # 1) 收集所有训练/验证样本（标准化后的 X）
        X_scaled = _collect_all_scaled_X(data)
        print(f"[info] X_scaled shape = {X_scaled.shape}")

        # 2) 标准化空间 min/max
        x_min_scaled = X_scaled.min(axis=0)
        x_max_scaled = X_scaled.max(axis=0)

        # 3) 物理空间 min/max （用训练时的 x_scaler.inverse_transform）
        X_phys = x_scaler.inverse_transform(X_scaled)
        x_min_phys = X_phys.min(axis=0)
        x_max_phys = X_phys.max(axis=0)

        # 4) 保存到 npz（连同特征名，方便 sanity check）
        out_path = RESULTS_DIR / f"bounds_{opamp}.npz"
        np.savez(
            out_path,
            x_min_phys=x_min_phys,
            x_max_phys=x_max_phys,
            x_min_scaled=x_min_scaled,
            x_max_scaled=x_max_scaled,
            feature_names=np.array(getattr(x_scaler, "feature_names_in_", []), dtype=object),
        )
        print(f"[ok] saved frozen bounds to {out_path}")

    print("\n全部边界已冻结完成。")

if __name__ == "__main__":
    main()
