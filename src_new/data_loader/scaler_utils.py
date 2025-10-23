import os
import joblib
from typing import Tuple

def get_scaler_paths(opamp_type: str) -> Tuple[str, str]:
    x_path = f"results/{opamp_type}_x_scaler.gz"
    y_path = f"results/{opamp_type}_y_scaler.gz"
    return x_path, y_path

def save_scalers(x_scaler, y_scaler, opamp_type: str):
    """
    将 x/y 的 StandardScaler 保存到 results/ 下。
    """
    os.makedirs("results", exist_ok=True)
    x_path, y_path = get_scaler_paths(opamp_type)
    joblib.dump(x_scaler, x_path)
    joblib.dump(y_scaler, y_path)
    print(f"Scalers 已保存到: {x_path} / {y_path}")

def load_scalers(opamp_type: str):
    """
    从 results/ 载入 x/y 的 StandardScaler。
    """
    x_path, y_path = get_scaler_paths(opamp_type)
    x_scaler = joblib.load(x_path)
    y_scaler = joblib.load(y_path)
    return x_scaler, y_scaler
