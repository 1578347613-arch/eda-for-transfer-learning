# src/infer_ensemble.py
import numpy as np, torch, joblib
from data_loader import get_data_and_scalers
from models import AlignHeteroMLP
from sklearn.metrics import mean_squared_error

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OPAMP_TYPE = '5t_opamp'
COLS = ['slewrate_pos','dc_gain','ugf','phase_margin','cmrr']

def load_model(path, input_dim, output_dim):
    m = AlignHeteroMLP(input_dim, output_dim, hidden_dim=512, num_layers=6, dropout_rate=0.1)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.to(DEVICE).eval()
    return m

def predict_mu_logvar(model, X):
    with torch.no_grad():
        mu, logvar, _ = model(torch.tensor(X, dtype=torch.float32).to(DEVICE))
    return mu.cpu().numpy(), logvar.cpu().numpy()

def main():
    data = get_data_and_scalers(opamp_type=OPAMP_TYPE)
    X_val, y_val = data['target_val']   # 用验证集评估两个模型的误差以动态加权

    input_dim, output_dim = X_val.shape[1], y_val.shape[1]

    m_align = load_model(f'results/{OPAMP_TYPE}_align_hetero_lambda0.050.pth', input_dim, output_dim)
    m_trg   = load_model(f'results/{OPAMP_TYPE}_target_only_hetero.pth', input_dim, output_dim)

        # === 1) 计算两模型在 VAL 上的均值+方差 ===
    mu_a, logv_a = predict_mu_logvar(m_align, X_val)
    mu_t, logv_t = predict_mu_logvar(m_trg,   X_val)

    # === 2) 温度标定（每模型、每指标闭式解）：c^2 = mean( (y-mu)^2 / var )
    def fit_temp(mu, logv, y):
        resid2 = (y - mu)**2                # 标准化空间
        var    = np.exp(logv)
        c2     = np.mean(resid2 / (var + 1e-12), axis=0)   # (D,)
        return np.sqrt(np.maximum(c2, 1e-6))                # 温度 c>=0

    c_a = fit_temp(mu_a, logv_a, y_val)     # (D,)
    c_t = fit_temp(mu_t, logv_t, y_val)
    logv_a = logv_a + 2.0 * np.log(c_a[None, :])  # 方差乘 c^2 等价于 logvar 加 2log c
    logv_t = logv_t + 2.0 * np.log(c_t[None, :])

    # === 3) precision=exp(-logvar)，并做 95% 分位裁剪，避免极端权重 ===
    tau_a = np.exp(-logv_a)
    tau_t = np.exp(-logv_t)
    clip_a = np.percentile(tau_a, 95, axis=0, keepdims=True)  # (1,D)
    clip_t = np.percentile(tau_t, 95, axis=0, keepdims=True)
    tau_a  = np.minimum(tau_a, clip_a)
    tau_t  = np.minimum(tau_t, clip_t)

    # 样本级 precision 权重
    w_prec_a = tau_a / (tau_a + tau_t + 1e-12)   # (N,D)
    w_prec_t = 1.0 - w_prec_a

    # 维度级 MSE 权重（更平滑的全局偏好）
    mse_a = np.array([mean_squared_error(y_val[:,i], mu_a[:,i]) for i in range(output_dim)])
    mse_t = np.array([mean_squared_error(y_val[:,i], mu_t[:,i]) for i in range(output_dim)])
    w_mse_a = 1.0 / (mse_a + 1e-12)
    w_mse_t = 1.0 / (mse_t + 1e-12)
    w_mse_sum = w_mse_a + w_mse_t
    w_mse_a /= w_mse_sum     # (D,)
    w_mse_t /= w_mse_sum

    # === 4) 混合：α·precision(样本级) + (1-α)·MSE(维度级) ===
    #      对 ugf（索引2）适当降低 α，减少“错得自信”的影响
    ALPHA = np.array([0.7, 0.7, 0.3, 0.7, 0.85], dtype=np.float64)[None, :]  # (1,D)
    w_a = ALPHA * w_prec_a + (1.0 - ALPHA) * w_mse_a[None, :]
    w_t = 1.0 - w_a

    # 集成均值（仍在标准化空间）
    mu_ens = w_a * mu_a + w_t * mu_t

    # 反标准化回物理量（含 expm1 对 ugf/cmrr）
    y_scaler = joblib.load(f'results/{OPAMP_TYPE}_y_scaler.gz')
    y_pred_scaled = mu_ens
    y_true_scaled = y_val
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_true = y_scaler.inverse_transform(y_true_scaled)

    for j, name in enumerate(COLS):
        if name in ['ugf','cmrr']:
            y_pred[:,j] = np.expm1(y_pred[:,j])
            y_true[:,j] = np.expm1(y_true[:,j])

    # 简报
    from sklearn.metrics import r2_score, mean_absolute_error
    print("\n=== Ensemble on B-VAL (物理单位) ===")
    for j,name in enumerate(COLS):
        mse = mean_squared_error(y_true[:,j], y_pred[:,j])
        mae = mean_absolute_error(y_true[:,j], y_pred[:,j])
        r2  = r2_score(y_true[:,j], y_pred[:,j])
        print(f"{name:14s}  MSE={mse:.4g}  MAE={mae:.4g}  R2={r2:.4f}")

if __name__ == "__main__":
    main()
