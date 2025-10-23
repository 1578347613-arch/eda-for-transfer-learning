# inference/infer_ensemble.py
import numpy as np
import torch
import joblib
from data_loader import get_data_and_scalers
from models import AlignHeteroMLP
from sklearn.metrics import mean_squared_error

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OPAMP_TYPE = '5t_opamp'
COLS = ['slewrate_pos', 'dc_gain', 'ugf', 'phase_margin', 'cmrr']

def load_model(path, input_dim, output_dim):
    model = AlignHeteroMLP(input_dim, output_dim, hidden_dim=512, num_layers=6, dropout_rate=0.1)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def predict_mu_logvar(model, X):
    with torch.no_grad():
        mu, logvar, _ = model(torch.tensor(X, dtype=torch.float32).to(DEVICE))
    return mu.cpu().numpy(), logvar.cpu().numpy()

def main():
    data = get_data_and_scalers(opamp_type=OPAMP_TYPE)
    X_val, y_val = data['target_val']

    input_dim, output_dim = X_val.shape[1], y_val.shape[1]

    # 加载模型
    m_align = load_model(f'results/{OPAMP_TYPE}_align_hetero_lambda0.050.pth', input_dim, output_dim)
    m_trg = load_model(f'results/{OPAMP_TYPE}_target_only_hetero.pth', input_dim, output_dim)

    # 计算模型的均值和方差
    mu_a, logv_a = predict_mu_logvar(m_align, X_val)
    mu_t, logv_t = predict_mu_logvar(m_trg, X_val)

    # 温度标定（每模型、每指标闭式解）
    def fit_temp(mu, logv, y):
        resid2 = (y - mu)**2
        var = np.exp(logv)
        c2 = np.mean(resid2 / (var + 1e-12), axis=0)
        return np.sqrt(np.maximum(c2, 1e-6))

    c_a = fit_temp(mu_a, logv_a, y_val)
    c_t = fit_temp(mu_t, logv_t, y_val)
    logv_a = logv_a + 2.0 * np.log(c_a[None, :])
    logv_t = logv_t + 2.0 * np.log(c_t[None, :])

    # 样本级 precision 权重
    tau_a = np.exp(-logv_a)
    tau_t = np.exp(-logv_t)
    clip_a = np.percentile(tau_a, 95, axis=0, keepdims=True)
    clip_t = np.percentile(tau_t, 95, axis=0, keepdims=True)
    tau_a = np.minimum(tau_a, clip_a)
    tau_t = np.minimum(tau_t, clip_t)

    w_prec_a = tau_a / (tau_a + tau_t + 1e-12)
    w_prec_t = 1.0 - w_prec_a

    # MSE 权重
    mse_a = np.array([mean_squared_error(y_val[:, i], mu_a[:, i]) for i in range(output_dim)])
    mse_t = np.array([mean_squared_error(y_val[:, i], mu_t[:, i]) for i in range(output_dim)])
    w_mse_a = 1.0 / (mse_a + 1e-12)
    w_mse_t = 1.0 / (mse_t + 1e-12)
    w_mse_sum = w_mse_a + w_mse_t
    w_mse_a /= w_mse_sum
    w_mse_t /= w_mse_sum

    # 混合模型
    ALPHA = np.array([0.7, 0.7, 0.3, 0.7, 0.85], dtype=np.float64)[None, :]
    w_a = ALPHA * w_prec_a + (1.0 - ALPHA) * w_mse_a[None, :]
    w_t = 1.0 - w_a

    # 集成均值（标准化空间内）
    mu_ens = w_a * mu_a + w_t * mu_t

    # 反标准化回物理单位
    y_scaler = joblib.load(f'results/{OPAMP_TYPE}_y_scaler.gz')
    y_pred_scaled = mu_ens
    y_true_scaled = y_val
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_true = y_scaler.inverse_transform(y_true_scaled)

    # 反 log1p
    for j, name in enumerate(COLS):
        if name in ['ugf', 'cmrr']:
            y_pred[:, j] = np.expm1(y_pred[:, j])
            y_true[:, j] = np.expm1(y_true[:, j])

    print("\n=== Ensemble on B-VAL (物理单位) ===")
    for j, name in enumerate(COLS):
        mse = mean_squared_error(y_true[:, j], y_pred[:, j])
        mae = mean_absolute_error(y_true[:, j], y_pred[:, j])
        r2 = r2_score(y_true[:, j], y_pred[:, j])
        print(f"{name:14s}  MSE={mse:.4g}  MAE={mae:.4g}  R2={r2:.4f}")

if __name__ == "__main__":
    main()
