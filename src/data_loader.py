# src/data_loader.py (已修正路径逻辑)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path # <-- 引入 Path

# --- 关键修正：定义健壮的根目录路径 ---
# 这个脚本位于 src 目录，所以它的父目录的父目录就是项目根目录
# 为了更简单和一致，我们这样做：
# 这个脚本自身位于 'src' 目录
SRC_DIR = Path(__file__).resolve().parent
# 项目根目录是 'src' 目录的上一级
PROJECT_ROOT = SRC_DIR.parent
# --- 修正结束 ---

SKEWED_COLS_DEFAULT = ["ugf", "cmrr"]


def load_data(opamp_type: str = "5t_opamp"):
    """
    加载工艺 A/B 的原始 CSV。(使用绝对路径)
    """
    print(f"--- 开始为 {opamp_type} 加载数据 ---")
    
    # --- 关键修正：使用 PROJECT_ROOT 构建绝对路径 ---
    base_path = PROJECT_ROOT / f"data/01_train_set/{opamp_type}"
    
    source_features_path = base_path / "source/pretrain_design_features.csv"
    source_targets_path = base_path / "source/pretrain_targets.csv"
    target_features_path = base_path / "target/target_design_features.csv"
    target_targets_path = base_path / "target/target_targets.csv"
    # --- 修正结束 ---

    X_source = pd.read_csv(source_features_path)
    y_source = pd.read_csv(source_targets_path)
    X_target = pd.read_csv(target_features_path)
    y_target = pd.read_csv(target_targets_path)

    return X_source, y_source, X_target, y_target


def preprocess_data(
    X_source: pd.DataFrame,
    y_source: pd.DataFrame,
    X_target: pd.DataFrame,
    y_target: pd.DataFrame,
    skewed_cols=SKEWED_COLS_DEFAULT,
):
    """
    预处理：对目标中偏斜列做 log1p，仅在 A 域拟合 StandardScaler，并应用到 A/B。
    """
    # 1) log1p 偏斜列（仅 y）
    for col in skewed_cols:
        if col in y_source.columns:
            y_source[col] = np.log1p(y_source[col])
        if col in y_target.columns:
            y_target[col] = np.log1p(y_target[col])
    if skewed_cols:
        print(f"已对列 {list(skewed_cols)} 进行 log1p 变换。")

    # 2) 仅在 A 域拟合 scaler
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaler.fit(X_source)
    y_scaler.fit(y_source)
    print("StandardScaler 已在工艺 A(Source) 上完成 fit。")

    # 3) 应用到 A/B
    X_source_scaled = x_scaler.transform(X_source)
    y_source_scaled = y_scaler.transform(y_source)
    X_target_scaled = x_scaler.transform(X_target)
    y_target_scaled = y_scaler.transform(y_target)
    print("A/B 全部数据已完成标准化（基于 A 域的 scaler）。")

    return (
        X_source_scaled, y_source_scaled,
        X_target_scaled, y_target_scaled,
        x_scaler, y_scaler
    )


def split_target_data(
    X_target_scaled,
    y_target_scaled,
    target_val_split: float = 0.2,
    random_state: int = 42
):
    """
    划分 B 域为 train/val。
    """
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_target_scaled, y_target_scaled,
        test_size=target_val_split, random_state=random_state
    )
    print(
        f"工艺 B 数据已划分为 {1 - target_val_split:.0%} 训练集 / {target_val_split:.0%} 验证集。")
    return X_tr, X_va, y_tr, y_va


def get_data_and_scalers(
    opamp_type: str = "5t_opamp",
    target_val_split: float = 0.2,
    random_state: int = 42,
    skewed_cols=SKEWED_COLS_DEFAULT,
    process_data: bool = True # 添加一个开关，方便外部调用时只获取原始数据
):
    """
    返回字典：
      - "source": (X_source_scaled, y_source_scaled)
      - "target_train": (X_target_train, y_target_train)
      - "target_val": (X_target_val, y_target_val)
      - "x_scaler", "y_scaler"（显式提供，便于其它模块直接使用）
    """
    Xs, ys, Xt, yt = load_data(opamp_type)
    
    if not process_data:
        # 如果只是为了获取维度，加载原始数据就足够了
        return {
            "x_dim": Xs.shape[1],
            "y_dim": ys.shape[1]
        }

    (
        Xs_s, ys_s, Xt_s, yt_s, x_scaler, y_scaler
    ) = preprocess_data(Xs, ys, Xt, yt, skewed_cols=skewed_cols)

    Xb_tr, Xb_va, yb_tr, yb_va = split_target_data(
        Xt_s, yt_s, target_val_split=target_val_split, random_state=random_state
    )

    payload = {
        "source": (Xs_s, ys_s),
        "target_train": (Xb_tr, yb_tr),
        "target_val": (Xb_va, yb_va),
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        # 同时返回原始数据，以防万一
        "raw_source": (Xs, ys),
        "raw_target": (Xt, yt),
    }
    return payload

