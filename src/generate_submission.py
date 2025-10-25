# src/generate_submission.py (v4.3 - 适配 data_loader)

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import joblib
import sys
from tqdm import tqdm

# --- 路径设置 (保持 v4.2 的正确逻辑) ---
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
RESULTS_DIR = SRC_DIR / 'results'
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path: sys.path.insert(0, str(SRC_DIR))

from config import TASK_CONFIGS
from data_loader import get_data_and_scalers
from models.align_hetero import AlignHeteroMLP
from unified_inverse_train import InverseMDN
from inverse_opt import optimize_x_multi_start, load_model_from_ckpt as load_forward_model_for_opt

# --- 全局配置 (不变) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
FORWARD_OUTPUT_COLS = ['slewrate_pos', 'dc_gain', 'ugf', 'phase_margin', 'cmrr']
Y_LOG_COLS = ['ugf', 'cmrr']
INVERSE_OUTPUT_COLS = {
    '5t_opamp': ['w1', 'w2', 'w3', 'l1', 'l2', 'l3', 'ibias'],
    'two_stage_opamp': ['w1', 'w2', 'w3', 'w4', 'w5', 'l1', 'l2', 'l3', 'l4', 'l5', 'cc', 'cr', 'ibias']
}

# --- 模型加载函数 (已修正) ---

def load_forward_model(opamp_type: str, ckpt_path: Path) -> AlignHeteroMLP:
    """
    加载正向模型，已适配新的 data_loader.py
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"正向模型权重文件未找到: {ckpt_path}")
    
    config = TASK_CONFIGS[opamp_type]
    
    # --- 关键修正 ---
    # 1. 正确调用 get_data_and_scalers，它会返回处理好的数据
    all_data = get_data_and_scalers(opamp_type=opamp_type)
    
    # 2. 从返回的数据中获取维度信息
    #    all_data['source'][0] 是 X_source_scaled (numpy array)
    input_dim = all_data['source'][0].shape[1]
    output_dim = all_data['source'][1].shape[1]
    # --- 修正结束 ---

    model = AlignHeteroMLP(
        input_dim=input_dim, output_dim=output_dim,
        hidden_dim=config['hidden_dim'], num_layers=config['num_layers'], dropout_rate=config['dropout_rate']
    ).to(DEVICE)
    
    state = torch.load(ckpt_path, map_location=DEVICE)
    model_state = state.get('state_dict', state)
    model.load_state_dict(model_state, strict=False)
    model.eval()
    print(f"成功加载正向模型: {ckpt_path.name} (Input: {input_dim}, Output: {output_dim})")
    return model

def load_inverse_mdn_model(opamp_type: str, ckpt_path: Path) -> InverseMDN:
    """
    加载与 unified_inverse_train.py 匹配的 MDN 模型 (保持不变)
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"反向MDN模型权重文件未找到: {ckpt_path}")
    
    state = torch.load(ckpt_path, map_location=DEVICE)
    model_config = state.get('config')
    if not model_config:
        raise ValueError(f"模型文件 {ckpt_path.name} 中缺少 'config' 字段。")

    model = InverseMDN(
        input_dim=model_config['input_dim'],
        output_dim=model_config['output_dim'],
        n_components=model_config['n_components'],
        hidden_dim=model_config['hidden_dim'],
        num_layers=model_config['num_layers']
    ).to(DEVICE)
    
    model_state = state.get('state_dict', state)
    model.load_state_dict(model_state, strict=False)
    model.eval()
    print(f"成功加载反向MDN模型: {ckpt_path.name}")
    return model

# --- 核心推理函数 (保持不变) ---
@torch.no_grad()
def predict_forward_ensemble(opamp_type: str, input_csv: Path, output_csv: Path, results_dir: Path):
    print(f"\n--- 开始正向集成预测任务: {opamp_type} ---")
    all_data = get_data_and_scalers(opamp_type=opamp_type)
    X_val, y_val = all_data['target_val']
    align_model_path = results_dir / f"{opamp_type}_finetuned.pth"
    target_only_model_path = results_dir / f"{opamp_type}_target_only.pth"
    m_align = load_forward_model(opamp_type, align_model_path)
    m_trg = load_forward_model(opamp_type, target_only_model_path)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    mu_a_val, logv_a_val, _ = m_align(X_val_t)
    mu_t_val, logv_t_val, _ = m_trg(X_val_t)
    mu_a_val, logv_a_val = mu_a_val.cpu().numpy(), logv_a_val.cpu().numpy()
    mu_t_val, logv_t_val = mu_t_val.cpu().numpy(), logv_t_val.cpu().numpy()
    def fit_temp(mu, logv, y):
        resid2 = (y - mu) ** 2; var = np.exp(logv)
        c2 = np.mean(resid2 / (var + 1e-12), axis=0)
        return np.sqrt(np.maximum(c2, 1e-6))
    c_a = fit_temp(mu_a_val, logv_a_val, y_val)
    c_t = fit_temp(mu_t_val, logv_t_val, y_val)
    test_df = pd.read_csv(input_csv)
    x_scaler = all_data['x_scaler']
    x_test_scaled = x_scaler.transform(test_df.values)
    x_test_t = torch.tensor(x_test_scaled, dtype=torch.float32).to(DEVICE)
    mu_a_test, logv_a_test, _ = m_align(x_test_t)
    mu_t_test, logv_t_test, _ = m_trg(x_test_t)
    mu_a_test, logv_a_test = mu_a_test.cpu().numpy(), logv_a_test.cpu().numpy()
    mu_t_test, logv_t_test = mu_t_test.cpu().numpy(), logv_t_test.cpu().numpy()
    logv_a_calibrated = logv_a_test + 2.0 * np.log(c_a[None, :])
    logv_t_calibrated = logv_t_test + 2.0 * np.log(c_t[None, :])
    tau_a = np.exp(-logv_a_calibrated); tau_t = np.exp(-logv_t_calibrated)
    w_prec_a = tau_a / (tau_a + tau_t + 1e-12)
    output_dim = y_val.shape[1]
    mse_a = np.array([np.mean((y_val[:, i] - mu_a_val[:, i])**2) for i in range(output_dim)])
    mse_t = np.array([np.mean((y_val[:, i] - mu_t_val[:, i])**2) for i in range(output_dim)])
    inv_mse_a = 1.0 / (mse_a + 1e-12); inv_mse_t = 1.0 / (mse_t + 1e-12)
    w_mse_a = inv_mse_a / (inv_mse_a + inv_mse_t)
    ALPHA = np.array(TASK_CONFIGS[opamp_type]['ensemble_alpha'])[None, :]
    w_a = ALPHA * w_prec_a + (1.0 - ALPHA) * w_mse_a[None, :]
    w_t = 1.0 - w_a
    mu_ens = w_a * mu_a_test + w_t * mu_t_test
    y_scaler = all_data['y_scaler']
    y_pred_physical = y_scaler.inverse_transform(mu_ens)
    pred_df = pd.DataFrame(y_pred_physical, columns=FORWARD_OUTPUT_COLS[:output_dim])
    for col in Y_LOG_COLS:
        if col in pred_df.columns: pred_df[col] = np.expm1(pred_df[col])
    pred_df.to_csv(output_csv, index=False)
    print(f"成功生成集成预测文件: {output_csv.name}")

def predict_inverse_hybrid(opamp_type: str, input_csv: Path, output_csv: Path, results_dir: Path):
    print(f"\n--- 开始反向混合预测任务: {opamp_type} ---")
    print("  [Step 1/3] 使用 MDN 生成初始解...")
    mdn_model_path = results_dir / f"mdn_{opamp_type}.pth"
    x_scaler_path = results_dir / f"{opamp_type}_x_scaler.gz"
    y_scaler_path = results_dir / f"{opamp_type}_y_scaler.gz"
    x_scaler = joblib.load(x_scaler_path)
    y_scaler = joblib.load(y_scaler_path)
    mdn_model = load_inverse_mdn_model(opamp_type, mdn_model_path)
    test_df = pd.read_csv(input_csv)
    y_test_df = test_df.copy()
    for col in Y_LOG_COLS:
        if col in y_test_df.columns: y_test_df[col] = np.log1p(y_test_df[col])
    y_test_scaled = y_scaler.transform(y_test_df.values)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        pi, mu, _ = mdn_model(y_test_tensor)
        x_init_scaled = torch.sum(pi.unsqueeze(-1) * mu, dim=1).cpu().numpy()
    print("  [Step 2/3] 加载正向模型用于优化...")
    forward_model_path = results_dir / f"{opamp_type}_finetuned.pth"
    forward_model = load_forward_model_for_opt(
        ckpt_path=forward_model_path, model_type="align_hetero",
        device=DEVICE, opamp_type=opamp_type
    )
    print(f"  [Step 3/3] 对 {len(test_df)} 个目标进行优化精修...")
    final_predictions_phys = []
    for i in tqdm(range(len(test_df)), desc=f"Optimizing {opamp_type}"):
        y_target_scaled_row = y_test_scaled[i]
        x_init_scaled_row = x_init_scaled[i][np.newaxis, :]
        best_x_scaled, _, _ = optimize_x_multi_start(
            model=forward_model, model_type="align_hetero",
            x_dim=x_init_scaled_row.shape[1], y_target_scaled=y_target_scaled_row,
            x_scaler=x_scaler, y_scaler=y_scaler, n_init=1, steps=100, lr=1e-3,
            init_points_scaled=x_init_scaled_row, device=str(DEVICE),
            opamp_type=opamp_type, goal=",".join(["target"] * len(y_target_scaled_row)),
            finish_lbfgs=20
        )
        x_phys_row = best_x_scaled[0] * x_scaler.scale_ + x_scaler.mean_
        final_predictions_phys.append(x_phys_row)
    final_pred_df = pd.DataFrame(final_predictions_phys, columns=INVERSE_OUTPUT_COLS[opamp_type])
    final_pred_df.to_csv(output_csv, index=False)
    print(f"成功生成混合策略的提交文件: {output_csv.name}")

# --- main 函数 (保持不变) ---
def main():
    parser = argparse.ArgumentParser(description="为EDA比赛生成所有提交文件 (v4.3 - 路径/loader适配)")
    parser.add_argument("--data_dir", type=str, default="data/02_public_test_set", help="测试集根目录")
    parser.add_argument("--output_dir", type=str, default="submission", help="输出目录")
    parser.add_argument("--inverse_strategy", type=str, default="hybrid", choices=["mdn_only", "hybrid"], help="反向预测策略")
    args = parser.parse_args()

    data_root = PROJECT_ROOT / args.data_dir
    features_dir = data_root / "features"
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(exist_ok=True)

    print("="*50)
    print("开始生成提交文件 (v4.3)...")
    print(f"执行目录 (PWD): {Path.cwd()}")
    print(f"项目根目录 (推断): {PROJECT_ROOT}")
    print(f"SRC 目录 (推断): {SRC_DIR}")
    print(f"模型/Scaler来源: {RESULTS_DIR}")
    print(f"输入数据来源: {features_dir}")
    print(f"结果输出至: {output_dir}")
    print("="*50)

    predict_forward_ensemble('5t_opamp', features_dir / 'features_A.csv', output_dir / 'predA.csv', RESULTS_DIR)
    predict_forward_ensemble('two_stage_opamp', features_dir / 'features_B.csv', output_dir / 'predB.csv', RESULTS_DIR)

    if args.inverse_strategy == "hybrid":
        predict_inverse_hybrid('5t_opamp', features_dir / 'features_C.csv', output_dir / 'predC.csv', RESULTS_DIR)
        predict_inverse_hybrid('two_stage_opamp', features_dir / 'features_D.csv', output_dir / 'predD.csv', RESULTS_DIR)
    else:
        print("\n'mdn_only' 策略暂未实现，请使用 'hybrid' 策略。")

    print("\n" + "="*50)
    print("所有提交文件已成功生成！")
    print(f"请检查 '{output_dir.resolve()}' 目录。")
    print("="*50)

if __name__ == "__main__":
    main()
