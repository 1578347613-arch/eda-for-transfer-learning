# src/data_loader.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import warnings
import config

# --- 关键修正：定义健壮的根目录路径 ---
SRC_DIR = Path(__file__).resolve().parent
# 项目根目录是 'src' 目录的上一级
PROJECT_ROOT = SRC_DIR.parent

SKEWED_COLS_DEFAULT = config.LOG_TRANSFORMED_COLS


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


def _apply_cmrr_winsor_or_drop(
    X_df: pd.DataFrame,
    y_df: pd.DataFrame,
    cfg: dict,
    domain_label: str = "source"
):
    """
    在 log1p 之前对 'cmrr' 做线性域的钳位（winsorize）或可选删除（仅训练集的场景；本实现保留 ALL scope）。
    当前数据管线在 split 之前做预处理，故只支持 scope='all'，若设置 'train_only' 会告警并按 'all' 处理。
    """
    if "cmrr" not in y_df.columns:
        return X_df, y_df  # 没有该列，直接返回

    scope = cfg.get("cmrr_cap_scope", "all")
    if scope != "all":
        warnings.warn("[data_loader] 当前实现仅支持 cmrr_cap_scope='all'；已按 'all' 处理，以避免改动数据接口/流程。")

    mode = cfg.get("cmrr_outlier_mode", "winsor")  # 'winsor' 或 'drop'
    cap_db = cfg.get("cmrr_db_cap", None)

    # A) 线性域钳位（winsorize）
    if cap_db is not None:
        cap_lin = 10.0 ** (float(cap_db) / 20.0)
        y_vals = y_df["cmrr"].astype(float).to_numpy()
        n_capped = int((y_vals > cap_lin).sum())
        if n_capped > 0:
            y_vals = np.minimum(y_vals, cap_lin)
            y_df = y_df.copy()
            y_df.loc[:, "cmrr"] = y_vals
        print(f"[CMRR-{domain_label}] 线性域钳位至 ≤{cap_db:.1f} dB；被钳位样本数 = {n_capped}")

    # B) 可选：删除最极端尾部（仅当显式要求 drop）
    if mode == "drop":
        q = float(cfg.get("cmrr_outlier_q", 0.999))
        arr = y_df["cmrr"].astype(float).to_numpy()
        thr = np.quantile(arr, q)
        mask = arr <= thr
        n_drop = int((~mask).sum())
        if n_drop > 0:
            keep_idx = y_df.index[mask]
            y_df = y_df.loc[keep_idx].copy()
            X_df = X_df.loc[keep_idx].copy()
        print(f"[CMRR-{domain_label}] 删除上尾 q={q:.4f} 的极端样本：{n_drop} 条（阈值={thr:.6g} 线性）")

    return X_df, y_df


def preprocess_data(
    X_source: pd.DataFrame,
    y_source: pd.DataFrame,
    X_target: pd.DataFrame,
    y_target: pd.DataFrame,
    opamp_type: str = "5t_opamp",
    skewed_cols=SKEWED_COLS_DEFAULT,
):
    """
    预处理：先对 CMRR 做线性域钳位/可选删除，再对偏斜列做 log1p；
    仅在 A 域拟合 StandardScaler，并应用到 A/B。
    （不改变外部数据接口）
    """
    # 读取任务配置
    task_cfg = config.TASK_CONFIGS.get(opamp_type, {})

    # 0) 在 log1p 之前，先处理 CMRR（线性域）
# 0) 在 log1p 之前，先对 dc_gain 和 cmrr 做线性域截尾
    X_source, y_source = _apply_dc_gain_winsor(X_source, y_source, task_cfg, domain_label="A(Source)")
    X_target, y_target = _apply_dc_gain_winsor(X_target, y_target, task_cfg, domain_label="B(Target)")

    X_source, y_source = _apply_cmrr_winsor_or_drop(X_source, y_source, task_cfg, domain_label="A(Source)")
    X_target, y_target = _apply_cmrr_winsor_or_drop(X_target, y_target, task_cfg, domain_label="B(Target)")

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


def split_source_data(
    X_source_scaled,
    y_source_scaled,
    source_val_split: float = 0.2,
    random_state: int = 42
):
    """
    划分 A 域为 train/val。
    """
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_source_scaled, y_source_scaled,
        test_size=source_val_split, random_state=random_state
    )
    print(
        f"工艺 A 数据已划分为 {1 - source_val_split:.0%} 训练集 / {source_val_split:.0%} 验证集。")
    return X_tr, X_va, y_tr, y_va


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
def _apply_dc_gain_winsor(
    X_df: pd.DataFrame,
    y_df: pd.DataFrame,
    cfg: dict,
    domain_label: str = "source"
):
    if "dc_gain" not in y_df.columns:
        return X_df, y_df
    cap_lin = cfg.get("dc_gain_cap", None)
    if cap_lin is None:
        return X_df, y_df

    vals = y_df["dc_gain"].astype(float).to_numpy()
    n_capped = int((vals > cap_lin).sum())
    if n_capped > 0:
        vals = np.minimum(vals, cap_lin)
        y_df = y_df.copy()
        y_df.loc[:, "dc_gain"] = vals
    print(f"[DCGAIN-{domain_label}] 钳位 ≤{cap_lin} (线性倍数); 被钳位样本数 = {n_capped}")
    return X_df, y_df


def get_data_and_scalers(
    opamp_type: str = "5t_opamp",
    source_val_split: float = 0.2,
    target_val_split: float = 0.2,
    random_state: int = 42,
    skewed_cols=SKEWED_COLS_DEFAULT,
    process_data: bool = True  # 添加一个开关，方便外部调用时只获取原始数据
):
    """
    返回字典：
      - "source_train": (X_source_train, y_source_train)
      - "source_val": (X_source_val, y_source_val)
      - "target_train": (X_target_train, y_target_train)
      - "target_val": (X_target_val, y_target_val)
      - "x_scaler", "y_scaler"
      - "raw_source": (X_source_raw, y_source_raw)
      - "raw_target": (X_target_raw, y_target_raw)
    （外部接口保持不变）
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
    ) = preprocess_data(Xs, ys, Xt, yt, opamp_type=opamp_type, skewed_cols=skewed_cols)

    Xa_tr, Xa_va, ya_tr, ya_va = split_source_data(
        Xs_s, ys_s, source_val_split=source_val_split, random_state=random_state
    )
    Xb_tr, Xb_va, yb_tr, yb_va = split_target_data(
        Xt_s, yt_s, target_val_split=target_val_split, random_state=random_state
    )

    payload = {
        "source": (Xs_s, ys_s),
        "source_train": (Xa_tr, ya_tr),
        "source_val": (Xa_va, ya_va),
        "target_train": (Xb_tr, yb_tr),
        "target_val": (Xb_va, yb_va),
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "raw_source": (Xs, ys),  # 原始未裁剪/未log1p数据（仅用于列名/诊断）
        "raw_target": (Xt, yt),
    }
    return payload
