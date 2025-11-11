# fused_project/src/generate_submission.py (C3 融合版 v7 - 终极修复版！)
#
# 手术 v5：修复了“特征增强”(13 vs 92) 和“路径” (src/results)！
# 手术 v7 (本次)：
#    1. 100% 移除了 C/D 题 (predict_inverse_hybrid) 中“垃圾”的硬编码 (steps=100)！
#    2. 100% 让 C3 "inverse_opt.py" 里的“黄金默认值” (steps=1000) 自动生效！
#    3. (v7 修复版) 100% 修复了 predD (92维) 保存为 (13维) 的列不匹配问题！
#
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import joblib
import sys
from tqdm import tqdm

# --- 路径设置 (C3 终极版 - 100% 按照您的要求) ---
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent 
RESULTS_DIR = PROJECT_ROOT / "src" / "results" # <-- 100% 指向 src 内部！

if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path: sys.path.insert(0, str(SRC_DIR))

# ⬇️ (C3 融合版) 导入 C3“混合圣经”和 C1 黄金模型
import config 
from data_loader import get_data_and_scalers
from models.align_hetero import AlignHeteroMLP 
from unified_inverse_train import InverseMDN
# ⬇️ (C3 融合版) 导入我们“手术”过的 C3 "inverse_opt.py"
from inverse_opt import optimize_x_multi_start, load_model_from_ckpt as load_forward_model_for_opt

# --- 全局配置 (100% 兼容) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
FORWARD_OUTPUT_COLS = ['slewrate_pos', 'dc_gain', 'ugf', 'phase_margin', 'cmrr']
Y_LOG_COLS = config.LOG_TRANSFORMED_COLS 
INVERSE_OUTPUT_COLS = {
    '5t_opamp': ['w1', 'w2', 'w3', 'l1', 'l2', 'l3', 'ibias'],
    'two_stage_opamp': ['w1', 'w2', 'w3', 'w4', 'w5', 'l1', 'l2', 'l3', 'l4', 'l5', 'cc', 'cr', 'ibias']
}


# --- (C3 融合版 v2) 智能加载器 (A/B 题用) ---
def load_forward_model(opamp_type: str, ckpt_path: Path) -> AlignHeteroMLP:
    # (代码 100% 来自 v5，它 100% 兼容 C3 圣经，此处省略)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"正向模型权重文件未找到: {ckpt_path}")
    try:
        data = get_data_and_scalers(opamp_type=opamp_type, process_data=False) 
        input_dim = data['x_dim']
        output_dim = data['y_dim']
    except TypeError: 
        data = get_data_and_scalers(opamp_type=opamp_type)
        input_dim = data['source_train'][0].shape[1]
        output_dim = data['source_train'][1].shape[1]
    if opamp_type not in config.TASK_CONFIGS:
        raise KeyError(f"在 C3 config.py 的 TASK_CONFIGS 中未找到 '{opamp_type}' 的配置。")
    model_config = config.TASK_CONFIGS[opamp_type]
    if 'hidden_dims' not in model_config:
         raise KeyError(f"C3 config.py 中 {opamp_type} 缺少 'hidden_dims' (C1 黄金架构)!")
    print(f"✅ [C3 融合加载器] 正在为 {opamp_type} 构建 C1 黄金架构...")
    model = AlignHeteroMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=model_config['hidden_dims'],
        dropout_rate=model_config['dropout_rate']
    ).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False) 
    model_state = state.get('state_dict', state)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing or unexpected:
        print(f"   [C3 load warning] 融合加载警告 (这可能是正常的):")
        if missing: print(f"     - 缺失的键: {len(missing)}")
        if unexpected: print(f"     - 意外的键: {len(unexpected)}")
    model.eval()
    print(f"✅ [C3] 成功加载正向模型: {ckpt_path.name}")
    return model

# --- (C2 原版) MDN 加载器 (100% 兼容) ---
def load_inverse_mdn_model(opamp_type: str, ckpt_path: Path) -> InverseMDN:
    # (代码 100% 来自 C2，它 100% 兼容 C3 圣经，此处省略)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"反向MDN模型权重文件未找到: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
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

# --- (C3 融合版 v5) C1“黄金特征增强”函数 (100% 保留) ---
_BASE_INPUT_COLS_TWO_STAGE = [
    "w1", "w2", "w3", "w4", "w5",
    "l1", "l2", "l3", "l4", "l5",
    "cc", "cr", "ibias",
]
_EPS = 1e-12
def _safe_log(x: np.ndarray) -> np.ndarray:
    # (代码 100% 来自 C1，此处省略)
    return np.log(np.clip(x, _EPS, None))
def _augment_two_stage_features_in_memory(df: pd.DataFrame) -> pd.DataFrame:
    # (代码 100% 来自 C1，此处省略)
    miss_base = [c for c in _BASE_INPUT_COLS_TWO_STAGE if c not in df.columns]
    if miss_base:
        raise ValueError(f"features_B.csv 缺少基础必需列（无法自动构造增强特征）: {miss_base}")
    out = df.copy()
    w_cols = [f"w{i}" for i in range(1, 6)]; l_cols = [f"l{i}" for i in range(1, 6)]
    W = out[w_cols].to_numpy(); L = out[l_cols].to_numpy()
    w_sum = np.clip(W.sum(axis=1), _EPS, None); l_sum = np.clip(L.sum(axis=1), _EPS, None)
    out["w_sum"] = w_sum; out["l_sum"] = l_sum
    out["log_w_sum"] = _safe_log(w_sum); out["log_l_sum"] = _safe_log(l_sum)
    out["log_ibias"] = _safe_log(out["ibias"].to_numpy())
    out["log_cc"] = _safe_log(out["cc"].to_numpy())
    out["log_cr"] = _safe_log(out["cr"].to_numpy())
    out["ibias_over_cc"] = out["ibias"] / np.clip(out["cc"], _EPS, None)
    out["ibias_over_cr"] = out["ibias"] / np.clip(out["cr"], _EPS, None)
    out["cr_over_cc"] = out["cr"] / np.clip(out["cc"], _EPS, None)
    out["log_ibias_over_cc"] = out["log_ibias"] - out["log_cc"]
    out["log_ibias_over_cr"] = out["log_ibias"] - out["log_cr"]
    out["log_cr_over_cc"] = out["log_cr"] - out["log_cc"]
    for i in range(1, 6):
        wi = out[f"w{i}"].to_numpy(); li = out[f"l{i}"].to_numpy()
        out[f"w{i}_over_l{i}"] = wi / np.clip(li, _EPS, None)
        out[f"log_w{i}_over_l{i}"] = _safe_log(wi) - _safe_log(li)
        out[f"w{i}_norm"] = wi / w_sum; out[f"l{i}_norm"] = li / l_sum
        area_sqrt = np.sqrt(np.clip(wi * li, _EPS, None))
        out[f"area{i}"] = area_sqrt; out[f"mismatch{i}"] = 1.0 / area_sqrt
    log_ibias = out["log_ibias"].to_numpy(); log_cc = out["log_cc"].to_numpy()
    for i in range(1, 6):
        wi = out[f"w{i}"].to_numpy(); li = out[f"l{i}"].to_numpy()
        lw = _safe_log(wi); ll = _safe_log(li)
        log_gm_hat = 0.5 * (lw - ll + log_ibias); log_ro_hat = ll - log_ibias
        out[f"log_gm_hat_{i}"] = log_gm_hat; out[f"log_ro_hat_{i}"] = log_ro_hat
        out[f"log_av_hat_{i}"] = log_gm_hat + log_ro_hat
        out[f"log_ugf_hat_{i}"] = 0.5 * (lw - ll) + 0.5 * log_ibias - log_cc
    out["dcgain_stage1_proxy"] = out["log_av_hat_1"]
    out["dcgain_stage2_proxy"] = out["log_av_hat_3"]
    out["dcgain_total_proxy"] = out["dcgain_stage1_proxy"] + out["dcgain_stage2_proxy"]
    out["dcgain_all_devices_proxy"] = out[[f"log_av_hat_{i}" for i in range(1, 6)]].sum(axis=1)
    def _add_cmrr_pair(i, j):
        ai = out[f"area{i}"]; aj = out[f"area{j}"]
        mr = ai / aj.replace(0, np.nan)
        out[f"cmrr_pair{i}{j}_area_ratio"] = mr.fillna(0.0)
        out[f"cmrr_pair{i}{j}_log_area_ratio"] = _safe_log(ai.to_numpy()) - _safe_log(aj.to_numpy())
        denom = (ai + aj) / 2.0
        out[f"cmrr_pair{i}{j}_area_diff_norm"] = ((ai - aj).abs() / denom.replace(0, np.nan)).fillna(0.0)
        mi = out[f"mismatch{i}"]; mj = out[f"mismatch{j}"]
        out[f"cmrr_pair{i}{j}_mismatch_diff"] = (mi - mj).abs()
    _add_cmrr_pair(1, 2); _add_cmrr_pair(3, 4); _add_cmrr_pair(4, 5)
    return out
def _ensure_two_stage_test_has_train_columns(opamp_type: str, input_csv: Path, test_df: pd.DataFrame, train_feature_cols: list) -> pd.DataFrame:
    # (代码 100% 来自 C1，此处省略)
    missing = set(train_feature_cols) - set(test_df.columns)
    if missing and (opamp_type == "two_stage_opamp") and (input_csv.name == "features_B.csv"):
        print(f"✅ [C3 特征增强] 发现 features_B.csv (13 features) 与训练集 ({len(train_feature_cols)} features) 不匹配。")
        print(f"   正在自动在内存中补齐 {len(missing)} 个增强特征列...")
        test_df = _augment_two_stage_features_in_memory(test_df)
        missing2 = set(train_feature_cols) - set(test_df.columns)
        if missing2:
            raise ValueError(f"[C3 AUTO-AUG失败] 补齐后仍缺特征列: {missing2}")
        print("   ✅ [C3 特征增强] 成功！特征数 13 -> 92！")
    elif missing:
        raise ValueError(f"{input_csv.name} 缺少特征列: {missing}")
    return test_df

# --- 核心推理函数 (C3 融合版 v5) ---
@torch.no_grad()
def predict_forward_ensemble(opamp_type: str, input_csv: Path, output_csv: Path, results_dir: Path):
    # (代码 100% 来自 v5，它 100% 兼容 C3 圣经 和 C1 特征增强，此处省略)
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
    train_feature_cols = list(all_data["x_scaler"].feature_names_in_)
    test_df = _ensure_two_stage_test_has_train_columns(
        opamp_type=opamp_type,
        input_csv=input_csv,
        test_df=test_df,
        train_feature_cols=train_feature_cols
    )
    test_df = test_df[train_feature_cols]
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
    ALPHA = np.array(config.TASK_CONFIGS[opamp_type]['ensemble_alpha'])[None, :] 
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
    """
    (C3 融合版 v7 - 终极修复版！)
    100% 移除了“垃圾硬编码”！
    100% 修复了 92维 -> 13维 的保存问题！
    """
    print(f"\n--- 开始反向混合预测任务: {opamp_type} ---")
    
    # (C2 的 MDN 加载逻辑, 100% 保留, 此处省略)
    print("   [Step 1/3] 使用 MDN 生成初始解...")
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
    
    print("   [Step 2/3] 加载正向模型用于优化...")
    forward_model_path = results_dir / f"{opamp_type}_finetuned.pth"
    
    # ⬇️ 完美！这里 100% 调用 C3 融合版(v2) 的 "inverse_opt.py" 里的智能加载器！
    forward_model = load_forward_model_for_opt(
        ckpt_path=forward_model_path, model_type="align_hetero",
        device=DEVICE, opamp_type=opamp_type
    )
    
    print(f"   [Step 3/3] 对 {len(test_df)} 个目标进行优化精修...")
    final_predictions_phys = []
    
    for i in tqdm(range(len(test_df)), desc=f"Optimizing {opamp_type}"):
        y_target_scaled_row = y_test_scaled[i]
        x_init_scaled_row = x_init_scaled[i][np.newaxis, :]
        
        # ⬇️ ========== "C3 终极手术" v7：100% 移除硬编码！ ========== ⬇️
        #
        # 我 100% 遵从您的命令！
        # 我“删除”了 C2“垃圾”的 "steps=100", "lr=1e-3", "finish_lbfgs=20"！
        #
        # 这 100% 会“自动”触发 C3/inverse_opt.py 里的
        # “黄金默认值” (steps=1000, lr=1e-4, finish_lbfgs=0)！
        #
        best_x_scaled, _, _ = optimize_x_multi_start(
            model=forward_model, model_type="align_hetero",
            x_dim=x_init_scaled_row.shape[1], 
            y_target_scaled=y_target_scaled_row,
            x_scaler=x_scaler, 
            y_scaler=y_scaler, 
            n_init=1, 
            # ⬇️ “垃圾”硬编码 100% 已移除！ ⬇️
            # steps=100, (DELETED!)
            # lr=1e-3, (DELETED!)
            # finish_lbfgs=20 (DELETED!)
            # ⬆️ “垃圾”硬编码 100% 已移除！ ⬆️
            
            init_points_scaled=x_init_scaled_row, 
            device=str(DEVICE),
            opamp_type=opamp_type, 
            goal=",".join(["target"] * len(y_target_scaled_row)),
        )
        # ⬆️ =============================================================== ⬆️
        
        # x_phys_row 现在是 92 维 (for two_stage) 或 7 维 (for 5t)
        x_phys_row = best_x_scaled[0] * x_scaler.scale_ + x_scaler.mean_
        final_predictions_phys.append(x_phys_row)
        
    
    # ⬇️ ========== (C3 v7 - 终极修复) 修复 92 -> 13 问题！ ========== ⬇️
    # 1. 从 x_scaler 获取完整的特征名称列表 (可能是 7 个或 92 个)
    #    (x_scaler 是在 Step 1 加载的)
    try:
        full_feature_names = list(x_scaler.feature_names_in_)
    except AttributeError:
        # 这是一个备用方案，以防 scaler 没有 feature_names_in_ 属性
        raise AttributeError(f"致命错误：{opamp_type}_x_scaler.gz 缺少 'feature_names_in_' 属性！"
                           "请确保用于训练和保存 scaler 的环境中 sklearn/joblib 版本支持此属性。")
    
    # 2. 获取我们“真正想提交”的目标列名 (7 个或 13 个)
    target_cols = INVERSE_OUTPUT_COLS[opamp_type]

    # 3. 先使用“完整列名”创建 DataFrame (e.g., 92 列)
    #    final_predictions_phys 里的每个元素都是 92 维的
    full_pred_df = pd.DataFrame(final_predictions_phys, columns=full_feature_names)
    
    # 4. 关键！只从完整 DataFrame 中“选择”我们需要提交的列 (e.g., 13 列)
    #    (对于 5t_opamp, 7 == 7, 这一步等于没操作)
    #    (对于 two_stage_opamp, 这一步是从 92 列中选 13 列)
    final_pred_df = full_pred_df[target_cols]
    # ⬆️ ===================== 修复完毕！===================== ⬆️

    final_pred_df.to_csv(output_csv, index=False)
    print(f"成功生成混合策略的提交文件: {output_csv.name}")

# --- main 函数 (100% 保持 C2 的原始逻辑) ---
def main():
    """(100% 保持 C2 的原始逻辑)"""
    parser = argparse.ArgumentParser(description="为EDA比赛生成所有提交文件 (C3 融合版 v7 - 终极修复版)")
    parser.add_argument("--data_dir", type=str, default="data/02_public_test_set", help="测试集根目录")
    parser.add_argument("--output_dir", type=str, default="submission", help="输出目录")
    parser.add_argument("--inverse_strategy", type=str, default="hybrid", choices=["mdn_only", "hybrid"], help="反向预测策略")
    args = parser.parse_args()

    data_root = PROJECT_ROOT / args.data_dir
    features_dir = data_root / "features"
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(exist_ok=True)

    print("="*50)
    print("开始生成提交文件 (C3 融合版 v7 - 92vs13 修复版)...")
    print(f"执行目录 (PWD): {Path.cwd()}")
    print(f"项目根目录 (推断): {PROJECT_ROOT}")
    print(f"SRC 目录 (推断): {SRC_DIR}")
    
    # ⬇️ (C3 终极版) 100% 确认 RESULTS_DIR 指向 src/results/ ！
    print(f"模型/Scaler来源: {RESULTS_DIR}")
    
    print(f"输入数据来源: {features_dir}")
    print(f"结果输出至: {output_dir}")
    print("="*50)
    
    # ⬇️ 完美！这会调用我们“手术”过的 predict_forward_ensemble！
    predict_forward_ensemble('5t_opamp', features_dir / 'features_A.csv', output_dir / 'predA.csv', RESULTS_DIR)
    predict_forward_ensemble('two_stage_opamp', features_dir / 'features_B.csv', output_dir / 'predB.csv', RESULTS_DIR)

    if args.inverse_strategy == "hybrid":
        # ⬇️ 完美！这会调用我们“手术”过的(已修复92vs13) predict_inverse_hybrid！
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