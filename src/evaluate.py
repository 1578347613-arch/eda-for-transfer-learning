# src/evaluate.py
import os, joblib, numpy as np, torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_loader import get_data_and_scalers
from models import DualHeadMLP

OPAMP_TYPE = '5t_opamp'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
COLS = ['slewrate_pos','dc_gain','ugf','phase_margin','cmrr']

def main():
    data = get_data_and_scalers(opamp_type=OPAMP_TYPE)
    X_val, y_val = data['target_val']

    x_scaler = joblib.load(f'results/{OPAMP_TYPE}_x_scaler.gz')
    y_scaler = joblib.load(f'results/{OPAMP_TYPE}_y_scaler.gz')

    input_dim, output_dim = X_val.shape[1], y_val.shape[1]
    model = DualHeadMLP(input_dim, output_dim, hidden_dim=512, num_layers=6, dropout_rate=0.1).to(DEVICE)
    state = torch.load(f'results/{OPAMP_TYPE}_dualhead_finetuned.pth', map_location=DEVICE)
    model.load_state_dict(state); model.to(DEVICE); model.eval()

    with torch.no_grad():
        pred = model(torch.tensor(X_val, dtype=torch.float32).to(DEVICE), domain='B').cpu().numpy()

    # 反标准化回物理单位（记得我们对 cmrr/ugf 做了 log1p）
    y_pred_scaled = pred
    y_val_scaled = y_val
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_true = y_scaler.inverse_transform(y_val_scaled)

    # 逆log1p
    def inv_log1p_cols(arr, cols, names):
        arr = arr.copy()
        for name in ['ugf','cmrr']:
            j = names.index(name)
            arr[:, j] = np.expm1(arr[:, j])
        return arr

    y_pred_phys = inv_log1p_cols(y_pred, COLS, COLS)
    y_true_phys = inv_log1p_cols(y_true, COLS, COLS)

    mse = [mean_squared_error(y_true_phys[:,i], y_pred_phys[:,i]) for i in range(output_dim)]
    mae = [mean_absolute_error(y_true_phys[:,i], y_pred_phys[:,i]) for i in range(output_dim)]
    r2  = [r2_score(y_true_phys[:,i], y_pred_phys[:,i]) for i in range(output_dim)]

    print("\n=== Target(B) 验证集指标（物理单位）===")
    for i, name in enumerate(COLS):
        print(f"{name:14s}  MSE={mse[i]:.4g}  MAE={mae[i]:.4g}  R2={r2[i]:.4f}")

if __name__ == "__main__":
    main()
