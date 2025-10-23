# evaluation/evaluate.py
import joblib
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_loader import get_data_and_scalers
from models import DualHeadMLP
import config

def main():
    data = get_data_and_scalers(opamp_type=config.OPAMP_TYPE)
    X_val, y_val = data['target_val']

    x_scaler = joblib.load(f'results/{config.OPAMP_TYPE}_x_scaler.gz')
    y_scaler = joblib.load(f'results/{config.OPAMP_TYPE}_y_scaler.gz')

    input_dim, output_dim = X_val.shape[1], y_val.shape[1]
    model = DualHeadMLP(input_dim, output_dim, hidden_dim=config.HIDDEN_DIM, num_layers=config.NUM_LAYERS).to(config.DEVICE)
    state = torch.load(f'results/{config.OPAMP_TYPE}_dualhead_finetuned.pth', map_location=config.DEVICE)
    model.load_state_dict(state); model.eval()

    with torch.no_grad():
        pred = model(torch.tensor(X_val, dtype=torch.float32).to(config.DEVICE), domain='B').cpu().numpy()

    # 反标准化回物理单位
    y_pred = y_scaler.inverse_transform(pred)
    y_true = y_scaler.inverse_transform(y_val)

    mse = mean_squared_error(y_true, y_pred, multioutput='raw_values')
    mae = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
    r2 = np.array([r2_score(y_true[:,i], y_pred[:,i]) for i in range(y_true.shape[1])])

    print("\n=== Target(B) 验证集指标（物理单位）===")
    for i, name in enumerate(['slewrate_pos', 'dc_gain', 'ugf', 'phase_margin', 'cmrr']):
        print(f"{name:14s}  MSE={mse[i]:.4g}  MAE={mae[i]:.4g}  R2={r


