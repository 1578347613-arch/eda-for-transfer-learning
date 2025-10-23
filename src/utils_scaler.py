# src/utils_scaler.py
import joblib, numpy as np

Y_LOG1P_INDEX = [2, 4]  # ugf, cmrr

def load_x_scaler(path): return joblib.load(path)
def load_y_scaler(path): return joblib.load(path)

def y_scaled_to_phys(y_scaled: np.ndarray, y_scaler) -> np.ndarray:
    y = y_scaler.inverse_transform(y_scaled)
    y_phys = y.copy()
    y_phys[:, Y_LOG1P_INDEX] = np.expm1(y_phys[:, Y_LOG1P_INDEX])
    return y_phys

def y_phys_to_scaled(y_phys: np.ndarray, y_scaler) -> np.ndarray:
    y = y_phys.copy()
    y[:, Y_LOG1P_INDEX] = np.log1p(y[:, Y_LOG1P_INDEX].clip(min=0))
    return y_scaler.transform(y)
