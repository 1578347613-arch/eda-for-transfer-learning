# src/data_loader.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib # 用于保存scaler

def get_data_and_scalers(opamp_type='5t_opamp', target_val_split=0.2, random_state=42):
    """
    加载、预处理数据，并返回处理后的数据集和用于转换的scalers。

    该函数执行以下操作:
    1. 加载工艺A (Source) 和工艺B (Target) 的数据。
    2. 对指定的偏斜目标列（ugf, cmrr）进行对数变换。
    3. 初始化特征和目标的StandardScaler。
    4. **只在工艺A数据上fit这些scalers**。
    5. 使用fit好的scalers去transform工艺A和工艺B的数据。
    6. 将工艺B数据划分为训练集和验证集。
    7. 返回所有数据集和scalers。
    """
    print(f"--- 开始为 {opamp_type} 加载和预处理数据 ---")

    # 1. 加载数据
    # --- 修正后的代码 ---
    source_features_path = f'data/01_train_set/{opamp_type}/source/pretrain_design_features.csv'
    source_targets_path = f'data/01_train_set/{opamp_type}/source/pretrain_targets.csv'
    target_features_path = f'data/01_train_set/{opamp_type}/target/target_design_features.csv'
    target_targets_path = f'data/01_train_set/{opamp_type}/target/target_targets.csv'


    X_source = pd.read_csv(source_features_path)
    y_source = pd.read_csv(source_targets_path)
    X_target = pd.read_csv(target_features_path)
    y_target = pd.read_csv(target_targets_path)

    # 2. 对数变换处理偏斜数据
    skewed_cols = ['ugf', 'cmrr']
    for col in skewed_cols:
        if col in y_source.columns:
            y_source[col] = np.log1p(y_source[col])
            y_target[col] = np.log1p(y_target[col])
    print(f"已对列 {skewed_cols} 进行Log1p变换。")

    # 3. 初始化Scalers
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    # 4. 只在工艺A (Source) 数据上fit Scalers
    x_scaler.fit(X_source)
    y_scaler.fit(y_source)
    print("StandardScalers已在工艺A (Source) 数据上完成fit。")

    # 5. Transform所有数据
    X_source_scaled = x_scaler.transform(X_source)
    y_source_scaled = y_scaler.transform(y_source)
    X_target_scaled = x_scaler.transform(X_target)
    y_target_scaled = y_scaler.transform(y_target)
    print("所有数据已使用基于工艺A的scalers完成transform。")

    # 6. 划分工艺B (Target) 数据
    X_target_train, X_target_val, y_target_train, y_target_val = train_test_split(
        X_target_scaled, y_target_scaled, test_size=target_val_split, random_state=random_state
    )
    print(f"工艺B数据已划分为 {1-target_val_split:.0%} 训练集和 {target_val_split:.0%} 验证集。")

    # 保存scalers以备后用 (例如在预测时)
    # --- 错误的代码 ---
    joblib.dump(x_scaler, f'results/{opamp_type}_x_scaler.gz')
    joblib.dump(y_scaler, f'results/{opamp_type}_y_scaler.gz')

    print(f"Scalers已保存到 '../results/' 目录下。")

    data_payload = {
        "source": (X_source_scaled, y_source_scaled),
        "target_train": (X_target_train, y_target_train),
        "target_val": (X_target_val, y_target_val),
        "scalers": (x_scaler, y_scaler)
    }

    return data_payload

if __name__ == '__main__':
    # 这是一个如何使用该函数的例子
    # 确保你已经创建了 'results' 文件夹
    import os
    if not os.path.exists('../results'):
        os.makedirs('../results')
        
    data = get_data_and_scalers(opamp_type='5t_opamp')

    # 你可以检查一下数据的形状
    print("\n--- 数据集形状 ---")
    print("Source X:", data['source'][0].shape)
    print("Target Train X:", data['target_train'][0].shape)
    print("Target Val X:", data['target_val'][0].shape)
