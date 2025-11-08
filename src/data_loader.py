import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import config

SKEWED_COLS_DEFAULT = config.LOG_TRANSFORMED_COLS


def load_data(opamp_type: str = "5t_opamp"):
    """
    加载工艺 A/B 的原始 CSV。
    返回: X_source, y_source, X_target, y_target (pandas.DataFrame)
    """
    print(f"--- 开始为 {opamp_type} 加载数据 ---")
    source_features_path = f"../data/01_train_set/{opamp_type}/source/pretrain_design_features.csv"
    source_targets_path = f"../data/01_train_set/{opamp_type}/source/pretrain_targets.csv"
    target_features_path = f"../data/01_train_set/{opamp_type}/target/target_design_features.csv"
    target_targets_path = f"../data/01_train_set/{opamp_type}/target/target_targets.csv"

    X_source = pd.read_csv(source_features_path)
    y_source = pd.read_csv(source_targets_path)
    X_target = pd.read_csv(target_features_path)
    y_target = pd.read_csv(target_targets_path)

    return X_source, y_source, X_target, y_target


def _add_physics_features(df: pd.DataFrame, opamp_type: str) -> pd.DataFrame:
    """
    在原始 DataFrame 上添加物理感知特征。
    """
    if opamp_type == '5t_opamp':
        # 假设列顺序为: w1, w2, w3, l1, l2, l3, ibias
        # 为安全起见，我们使用列名（CSV应包含表头）
        required_cols = ['w1', 'w2', 'w3', 'l1', 'l2', 'l3', 'ibias']
        if not all(col in df.columns for col in required_cols):
            print(f"警告: 5t_opamp 特征工程失败，缺少列: {required_cols}。跳过特征工程。")
            return df

        print("... 正在为 5t_opamp 添加物理感知特征 ...")
        # 创建副本以避免 SettingWithCopyWarning
        df = df.copy()

        # Epsilon for safe division
        eps = 1e-9

        # 1. W/L Ratios
        df['w1_over_l1'] = df['w1'] / (df['l1'] + eps)
        df['w2_over_l2'] = df['w2'] / (df['l2'] + eps)  # 负载
        df['w3_over_l3'] = df['w3'] / (df['l3'] + eps)  # 尾电流源

        # 2. gm1 代理 (gm ~ sqrt(W/L * I))
        # 假设 I_D1 ~ ibias
        df['g_m1_proxy'] = np.sqrt(df['w1_over_l1'] * df['ibias'])

        # 3. ro5 (尾电流) 代理 (ro ~ L / I)
        df['r_o5_proxy'] = df['l3'] / (df['ibias'] + eps)

        # 4. CMRR 代理 (CMRR ~ gm1 * ro5)
        df['cmrr_proxy'] = df['g_m1_proxy'] * df['r_o5_proxy']

        # 5. 增益 (Av) 代理
        # Av ~ gm1 * ro_load. ro_load(M2) ~ L2 / I_D2
        df['r_o_load_proxy'] = df['l2'] / (df['ibias'] + eps)
        df['gain_proxy'] = df['g_m1_proxy'] * df['r_o_load_proxy']

    elif opamp_type == 'two_stage_opamp':
        print("... two_stage_opamp 的特征工程暂未实现 ...")

    return df


def preprocess_data(
    X_source: pd.DataFrame,
    y_source: pd.DataFrame,
    X_target: pd.DataFrame,
    y_target: pd.DataFrame,
    skewed_cols=SKEWED_COLS_DEFAULT,
    opamp_type: str = "5t_opamp"
):
    """
    预处理：对目标中偏斜列做 log1p，仅在 A 域拟合 StandardScaler，并应用到 A/B。
    返回:
      X_source_scaled, y_source_scaled, X_target_scaled, y_target_scaled (np.ndarray)
      x_scaler, y_scaler (StandardScaler)
    """
    # 1) log1p 偏斜列（仅 y）
    for col in skewed_cols:
        if col in y_source.columns:
            y_source[col] = np.log1p(y_source[col])
        if col in y_target.columns:
            y_target[col] = np.log1p(y_target[col])
    if skewed_cols:
        print(f"已对列 {list(skewed_cols)} 进行 log1p 变换。")

    X_source = _add_physics_features(X_source, opamp_type)
    X_target = _add_physics_features(X_target, opamp_type)
    print(f"特征工程完毕。新 X_source 维度: {X_source.shape}")

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


def get_data_and_scalers(
    opamp_type: str = "5t_opamp",
    source_val_split: float = 0.2,
    target_val_split: float = 0.2,
    random_state: int = 42,
    skewed_cols=SKEWED_COLS_DEFAULT,
):
    """
    返回字典：
      - "source_train": (X_source_train, y_source_train)
      - "source_val": (X_source_val, y_source_val)
      - "target_train": (X_target_train, y_target_train)
      - "target_val": (X_target_val, y_target_val)
      - "x_scaler", "y_scaler"
    """
    Xs, ys, Xt, yt = load_data(opamp_type)
    (
        Xs_s, ys_s, Xt_s, yt_s, x_scaler, y_scaler
    ) = preprocess_data(Xs, ys, Xt, yt,
                        skewed_cols=skewed_cols,
                        opamp_type=opamp_type)

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
        "raw_source": (Xs, ys),  # <-- 新增：包含原始的Source DataFrame
        "raw_target": (Xt, yt)  # <-- 新增：包含原始的Target DataFrame
    }
    return payload
