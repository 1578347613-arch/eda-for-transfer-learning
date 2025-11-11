# src/generate_submission.py (v7 - fixed log restore & eval domain + auto two_stage feature augment for predB)

from inverse_opt import optimize_x_multi_start, load_model_from_ckpt as load_forward_model_for_opt
from unified_inverse_train import InverseMDN
from models.align_hetero import AlignHeteroMLP
from data_loader import get_data_and_scalers
from config import TASK_CONFIGS, LOG_TRANSFORMED_COLS
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import joblib
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- 路径与环境 ---
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
RESULTS_DIR = SRC_DIR / "results"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# --- 全局常量 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 输出列的固定顺序（官方要求的性能列顺序）
FORWARD_OUTPUT_COLS = [
    'slewrate_pos',
    'dc_gain',
    'ugf',
    'phase_margin',
    'cmrr'
]

# 哪些列训练时做了 log1p，必须和 data_loader 完全一致
Y_LOG_COLS = LOG_TRANSFORMED_COLS

INVERSE_OUTPUT_COLS = {
    '5t_opamp': [
        'w1', 'w2', 'w3', 'l1', 'l2', 'l3', 'ibias'
    ],
    'two_stage_opamp': [
        'w1', 'w2', 'w3', 'w4', 'w5',
        'l1', 'l2', 'l3', 'l4', 'l5',
        'cc', 'cr', 'ibias'
    ]
}


# ---------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------

def _load_forward_model(opamp_type: str, ckpt_path: Path) -> AlignHeteroMLP:
    """
    加载正向模型 (AlignHeteroMLP) 并切到 eval。
    会用 data_loader 里的 scaler 推断输入输出维度。
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"正向模型权重文件未找到: {ckpt_path}")

    config = TASK_CONFIGS[opamp_type]
    all_data = get_data_and_scalers(opamp_type=opamp_type)

    # X_source_scaled shape = (N, in_dim)
    input_dim = all_data['source'][0].shape[1]
    # y_source_scaled shape = (N, out_dim)
    output_dim = all_data['source'][1].shape[1]

    model = AlignHeteroMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=config['hidden_dims'],
       # num_layers=config['num_layers'],
        dropout_rate=config['dropout_rate']
    ).to(DEVICE)

    state = torch.load(ckpt_path, map_location=DEVICE)
    model_state = state.get('state_dict', state)
    model.load_state_dict(model_state, strict=False)
    model.eval()

    print(
        f"成功加载正向模型: {ckpt_path.name} (Input: {input_dim}, Output: {output_dim})")
    return model


def _load_inverse_mdn_model(opamp_type: str, ckpt_path: Path) -> InverseMDN:
    """
    加载反向 MDN (InverseMDN) 并切到 eval。
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
        #num_layers=model_config['num_layers']
    ).to(DEVICE)

    model_state = state.get('state_dict', state)
    model.load_state_dict(model_state, strict=False)
    model.eval()
    print(f"成功加载反向MDN模型: {ckpt_path.name}")
    return model


def _to_physical_y(y_std: np.ndarray, y_scaler, colnames):
    """
    把标准化空间 y_std 反标准化到物理单位，并对训练时做过 log1p 的列做 expm1 还原。
    """
    y_unstd = y_scaler.inverse_transform(y_std)  # 回到训练时的回归目标域(也许仍是log域)
    y_phys = y_unstd.copy()
    for idx, col in enumerate(colnames):
        if col in Y_LOG_COLS:
            y_phys[:, idx] = np.expm1(y_unstd[:, idx])
    return y_phys


# ---------------------------------------------------------------------
# two_stage_opamp 的在线特征增强（仅在 predB / features_B.csv 时使用）
# ---------------------------------------------------------------------

_BASE_INPUT_COLS_TWO_STAGE = [
    "w1", "w2", "w3", "w4", "w5",
    "l1", "l2", "l3", "l4", "l5",
    "cc", "cr", "ibias",
]
_EPS = 1e-12


def _safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.clip(x, _EPS, None))


def _augment_two_stage_features_in_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于原始13列 (w1..w5, l1..l5, cc, cr, ibias) 追加我们训练使用的增强特征列。
    不修改输入 DataFrame（返回一个新副本）。
    """
    miss_base = [c for c in _BASE_INPUT_COLS_TWO_STAGE if c not in df.columns]
    if miss_base:
        raise ValueError(f"features_B.csv 缺少基础必需列（无法自动构造增强特征）: {miss_base}")

    out = df.copy()

    # --- 预计算 ---
    w_cols = [f"w{i}" for i in range(1, 6)]
    l_cols = [f"l{i}" for i in range(1, 6)]
    W = out[w_cols].to_numpy()
    L = out[l_cols].to_numpy()
    w_sum = np.clip(W.sum(axis=1), _EPS, None)
    l_sum = np.clip(L.sum(axis=1), _EPS, None)
    out["w_sum"] = w_sum
    out["l_sum"] = l_sum
    out["log_w_sum"] = _safe_log(w_sum)
    out["log_l_sum"] = _safe_log(l_sum)

    # --- 偏置/补偿 log & 比值 ---
    out["log_ibias"] = _safe_log(out["ibias"].to_numpy())
    out["log_cc"]    = _safe_log(out["cc"].to_numpy())
    out["log_cr"]    = _safe_log(out["cr"].to_numpy())
    out["ibias_over_cc"] = out["ibias"] / np.clip(out["cc"], _EPS, None)
    out["ibias_over_cr"] = out["ibias"] / np.clip(out["cr"], _EPS, None)
    out["cr_over_cc"]    = out["cr"]    / np.clip(out["cc"], _EPS, None)
    out["log_ibias_over_cc"] = out["log_ibias"] - out["log_cc"]
    out["log_ibias_over_cr"] = out["log_ibias"] - out["log_cr"]
    out["log_cr_over_cc"]    = out["log_cr"]    - out["log_cc"]

    # --- 每支管子: W/L、归一化宽/长、面积/失配、log(W/L) ---
    for i in range(1, 6):
        wi = out[f"w{i}"].to_numpy()
        li = out[f"l{i}"].to_numpy()
        out[f"w{i}_over_l{i}"] = wi / np.clip(li, _EPS, None)
        out[f"log_w{i}_over_l{i}"] = _safe_log(wi) - _safe_log(li)
        out[f"w{i}_norm"] = wi / w_sum
        out[f"l{i}_norm"] = li / l_sum
        area_sqrt = np.sqrt(np.clip(wi * li, _EPS, None))
        out[f"area{i}"] = area_sqrt
        out[f"mismatch{i}"] = 1.0 / area_sqrt

    # --- 每支管子: gm/ro/Av/UGF 的 log 代理 ---
    log_ibias = out["log_ibias"].to_numpy()
    log_cc    = out["log_cc"].to_numpy()
    for i in range(1, 6):
        wi = out[f"w{i}"].to_numpy()
        li = out[f"l{i}"].to_numpy()
        lw = _safe_log(wi)
        ll = _safe_log(li)
        log_gm_hat = 0.5 * (lw - ll + log_ibias)
        log_ro_hat = ll - log_ibias
        out[f"log_gm_hat_{i}"] = log_gm_hat
        out[f"log_ro_hat_{i}"] = log_ro_hat
        out[f"log_av_hat_{i}"] = log_gm_hat + log_ro_hat
        out[f"log_ugf_hat_{i}"] = 0.5 * (lw - ll) + 0.5 * log_ibias - log_cc

    # --- DCGAIN 相关 proxy（命名与训练脚本保持一致） ---
    out["dcgain_stage1_proxy"] = out["log_av_hat_1"]
    out["dcgain_stage2_proxy"] = out["log_av_hat_3"]
    out["dcgain_total_proxy"]  = out["dcgain_stage1_proxy"] + out["dcgain_stage2_proxy"]
    out["dcgain_all_devices_proxy"] = out[[f"log_av_hat_{i}" for i in range(1, 6)]].sum(axis=1)

    # --- CMRR 相关（配对：12 / 34 / 45） ---
    def _add_cmrr_pair(i, j):
        ai = out[f"area{i}"]
        aj = out[f"area{j}"]
        mr = ai / aj.replace(0, np.nan)
        out[f"cmrr_pair{i}{j}_area_ratio"] = mr.fillna(0.0)
        out[f"cmrr_pair{i}{j}_log_area_ratio"] = _safe_log(ai.to_numpy()) - _safe_log(aj.to_numpy())
        denom = (ai + aj) / 2.0
        out[f"cmrr_pair{i}{j}_area_diff_norm"] = ((ai - aj).abs() / denom.replace(0, np.nan)).fillna(0.0)
        mi = out[f"mismatch{i}"]
        mj = out[f"mismatch{j}"]
        out[f"cmrr_pair{i}{j}_mismatch_diff"] = (mi - mj).abs()

    _add_cmrr_pair(1, 2)
    _add_cmrr_pair(3, 4)
    _add_cmrr_pair(4, 5)

    return out


def _ensure_two_stage_test_has_train_columns(opamp_type: str,
                                             input_csv: Path,
                                             test_df: pd.DataFrame,
                                             train_feature_cols: list) -> pd.DataFrame:
    """
    仅在 two_stage_opamp + features_B.csv 时：
    若 test_df 缺少训练用列，则用内存增强函数补齐列，再次校验。
    其他场景保持原逻辑（缺就报错）。
    """
    missing = set(train_feature_cols) - set(test_df.columns)
    if missing and (opamp_type == "two_stage_opamp") and (input_csv.name == "features_B.csv"):
        print(f"[AUTO-AUG] 发现 features_B.csv 缺少训练特征列 {len(missing)} 个，"
              f"自动在内存中补齐增强特征列…")
        test_df = _augment_two_stage_features_in_memory(test_df)
        missing2 = set(train_feature_cols) - set(test_df.columns)
        if missing2:
            raise ValueError(f"[AUTO-AUG失败] 补齐后仍缺特征列: {missing2}")
    elif missing:
        # 非 two_stage/predB 的场景，保持严格检查
        raise ValueError(f"{input_csv.name} 缺少特征列: {missing}")
    return test_df


# ---------------------------------------------------------------------
# A/B: 正向推理（两种模式）
# ---------------------------------------------------------------------

@torch.no_grad()
def predict_forward_simple(
    opamp_type: str,
    input_csv: Path,
    output_csv: Path,
    results_dir: Path,
    model_choice: str = "align"
):
    """
    极简正向推理：单模型直出，不做温度标定、不做双模型融合。
    - model_choice="align"   -> {opamp_type}_finetuned.pth
    - model_choice="target"  -> {opamp_type}_target_only.pth

    输出：
    output_csv: 物理域 (slewrate_pos≈十几, dc_gain≈上百等)，列顺序为官方 FORWARD_OUTPUT_COLS
    """

    print(f"\n--- 开始正向简单预测任务: {opamp_type} [{model_choice}] ---")

    all_data = get_data_and_scalers(opamp_type=opamp_type)

    # 载模型
    ckpt_name = f"{opamp_type}_{'finetuned' if model_choice=='align' else 'target_only'}.pth"
    model = _load_forward_model(opamp_type, results_dir / ckpt_name)

    # 读取 test csv，并按训练时特征列顺序重排（防止列错位）
    train_feature_cols = list(all_data["raw_target"][0].columns)  # X(B) 原始列顺序
    test_df = pd.read_csv(input_csv)

    # 新逻辑：two_stage + features_B.csv 时自动补齐增强特征列，其它情况维持原有严格校验
    test_df = _ensure_two_stage_test_has_train_columns(
        opamp_type=opamp_type,
        input_csv=input_csv,
        test_df=test_df,
        train_feature_cols=train_feature_cols
    )

    # 只按训练列顺序取列，丢弃多余列
    test_df = test_df[train_feature_cols].copy()

    # 标准化到模型输入域
    x_scaler = all_data['x_scaler']
    x_test_scaled = x_scaler.transform(test_df.values)

    # 模型前向 (标准化空间输出)
    x_test_t = torch.tensor(x_test_scaled, dtype=torch.float32, device=DEVICE)
    # mu_std: [N, D_out] in standardized space
    mu_std, _, _ = model(x_test_t)
    mu_std = mu_std.cpu().numpy()

    # 回到物理域
    y_scaler = all_data['y_scaler']
    # y_cols 是训练时 y(B) 的真实列名顺序
    y_cols = list(all_data["raw_target"][1].columns)
    y_pred_phys = _to_physical_y(mu_std, y_scaler, y_cols)

    # 输出列顺序对齐官方要求
    pred_df = pd.DataFrame(y_pred_phys, columns=y_cols)
    pred_df = pred_df.reindex(columns=FORWARD_OUTPUT_COLS[:pred_df.shape[1]])

    pred_df.to_csv(output_csv, index=False)
    print(f"成功生成简单预测文件: {output_csv.name}")


@torch.no_grad()
def predict_forward_ensemble(
    opamp_type: str,
    input_csv: Path,
    output_csv: Path,
    results_dir: Path
):
    """
    集成推理：两个模型 + 温度标定 + 精度/方差加权融合。
    注意：最后仍然通过 _to_physical_y() 输出物理域。
    """
    print(f"\n--- 开始正向集成预测任务: {opamp_type} ---")

    all_data = get_data_and_scalers(opamp_type=opamp_type)
    X_val, y_val = all_data['target_val']  # 用 val 来拟合温度
    y_scaler = all_data['y_scaler']

    # 加载两个模型
    align_model_path = results_dir / f"{opamp_type}_finetuned.pth"
    target_only_model_path = results_dir / f"{opamp_type}_target_only.pth"
    m_align = _load_forward_model(opamp_type, align_model_path)
    m_trg = _load_forward_model(opamp_type, target_only_model_path)

    # 在 val split 上计算温度系数、MSE，用于融合
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    mu_a_val, logv_a_val, _ = m_align(X_val_t)
    mu_t_val, logv_t_val, _ = m_trg(X_val_t)
    mu_a_val = mu_a_val.cpu().numpy()
    logv_a_val = logv_a_val.cpu().numpy()
    mu_t_val = mu_t_val.cpu().numpy()
    logv_t_val = logv_t_val.cpu().numpy()

    def fit_temp(mu, logv, y):
        resid2 = (y - mu) ** 2
        var = np.exp(logv)
        c2 = np.mean(resid2 / (var + 1e-12), axis=0)
        return np.sqrt(np.maximum(c2, 1e-12))

    c_a = fit_temp(mu_a_val, logv_a_val, y_val)
    c_t = fit_temp(mu_t_val, logv_t_val, y_val)

    # 读取 test csv，并按训练列顺序重排
    train_feature_cols = list(all_data["raw_target"][0].columns)
    test_df = pd.read_csv(input_csv)

    # 同样在 ensemble 路径下支持 predB 自动补列
    test_df = _ensure_two_stage_test_has_train_columns(
        opamp_type=opamp_type,
        input_csv=input_csv,
        test_df=test_df,
        train_feature_cols=train_feature_cols
    )

    test_df = test_df[train_feature_cols].copy()

    # 标准化
    x_scaler = all_data['x_scaler']
    x_test_scaled = x_scaler.transform(test_df.values)
    x_test_t = torch.tensor(x_test_scaled, dtype=torch.float32, device=DEVICE)

    # 两个模型各自推理
    mu_a_test, logv_a_test, _ = m_align(x_test_t)
    mu_t_test, logv_t_test, _ = m_trg(x_test_t)
    mu_a_test = mu_a_test.cpu().numpy()
    logv_a_test = logv_a_test.cpu().numpy()
    mu_t_test = mu_t_test.cpu().numpy()
    logv_t_test = logv_t_test.cpu().numpy()

    # 温度校准 logvar
    logv_a_cal = logv_a_test + 2.0 * np.log(c_a[None, :])
    logv_t_cal = logv_t_test + 2.0 * np.log(c_t[None, :])

    tau_a = np.exp(-logv_a_cal)
    tau_t = np.exp(-logv_t_cal)
    w_prec_a = tau_a / (tau_a + tau_t + 1e-12)

    output_dim = y_val.shape[1]
    mse_a = np.array([
        np.mean((y_val[:, i] - mu_a_val[:, i]) ** 2)
        for i in range(output_dim)
    ])
    mse_t = np.array([
        np.mean((y_val[:, i] - mu_t_val[:, i]) ** 2)
        for i in range(output_dim)
    ])
    inv_mse_a = 1.0 / (mse_a + 1e-12)
    inv_mse_t = 1.0 / (mse_t + 1e-12)
    w_mse_a = inv_mse_a / (inv_mse_a + inv_mse_t)

    # 来自 TASK_CONFIGS 的 alpha，逐维混权
    ALPHA = np.array(TASK_CONFIGS[opamp_type]['ensemble_alpha'])
    assert ALPHA.shape[0] == output_dim, \
        f"ensemble_alpha 维度不匹配: {ALPHA.shape[0]} vs {output_dim}"
    ALPHA = ALPHA[None, :]  # [1, D]

    w_a = ALPHA * w_prec_a + (1.0 - ALPHA) * w_mse_a[None, :]
    w_t = 1.0 - w_a

    mu_ens = w_a * mu_a_test + w_t * mu_t_test  # 标准化空间的均值融合

    # 回到物理域
    y_cols = list(all_data["raw_target"][1].columns)
    y_pred_phys = _to_physical_y(mu_ens, y_scaler, y_cols)

    pred_df = pd.DataFrame(y_pred_phys, columns=y_cols)
    pred_df = pred_df.reindex(columns=FORWARD_OUTPUT_COLS[:pred_df.shape[1]])

    pred_df.to_csv(output_csv, index=False)
    print(f"成功生成集成预测文件: {output_csv.name}")


# ---------------------------------------------------------------------
# C/D: 反向推理 (hybrid)
# ---------------------------------------------------------------------

def predict_inverse_hybrid(
    opamp_type: str,
    input_csv: Path,
    output_csv: Path,
    results_dir: Path
):
    """
    hybrid 反向策略：
    1) 用反向 MDN (y->x 分布) 生成初始解 (无梯度)
    2) 用正向模型 + 梯度下降(+LBFGS) 优化 x，使 f(x) 贴近指定目标 y
    3) 输出物理域下的 x
    """
    print(f"\n--- 开始反向混合预测任务: {opamp_type} ---")

    # == Step 1: MDN 给初值（无梯度） ==
    mdn_model_path = results_dir / f"mdn_{opamp_type}.pth"
    x_scaler_path = results_dir / f"{opamp_type}_x_scaler.gz"
    y_scaler_path = results_dir / f"{opamp_type}_y_scaler.gz"

    x_scaler = joblib.load(x_scaler_path)
    y_scaler = joblib.load(y_scaler_path)
    mdn_model = _load_inverse_mdn_model(opamp_type, mdn_model_path)

    # 读取目标规格，并对做过 log1p 的列执行同样的 log1p 变换
    y_target_df = pd.read_csv(input_csv).copy()
    for col in Y_LOG_COLS:
        if col in y_target_df.columns:
            y_target_df[col] = np.log1p(y_target_df[col])

    # 缩放到 y_scaler 域
    y_target_scaled = y_scaler.transform(y_target_df)

    # MDN 推理 (无梯度)
    with torch.inference_mode():
        y_target_t = torch.tensor(
            y_target_scaled, dtype=torch.float32, device=DEVICE)
        # pi: [B,K], mu: [B,K,Dx]
        pi, mu, _ = mdn_model(y_target_t)
        # 期望作为初值
        x_init_scaled = torch.sum(
            pi.unsqueeze(-1) * mu, dim=1).cpu().numpy()  # [B,Dx]

    # == Step 2: 用前向模型作为 cost，开启梯度优化 ==
    forward_model_path = results_dir / f"{opamp_type}_finetuned.pth"
    forward_model = load_forward_model_for_opt(
        ckpt_path=forward_model_path,
        model_type="align_hetero",
        device=DEVICE,
        opamp_type=opamp_type
    )

    print(f"开始逐样本优化 {len(y_target_df)} 个目标点 ...")
    final_predictions_phys = []

    # 重要：优化阶段必须开 grad
    torch.set_grad_enabled(True)

    for i in tqdm(range(len(y_target_df)), desc=f"Optimizing {opamp_type}"):
        y_goal_i_scaled = y_target_scaled[i]
        x_init_i_scaled = x_init_scaled[i][np.newaxis, :]

        best_x_scaled, _, _ = optimize_x_multi_start(
            model=forward_model,
            model_type="align_hetero",
            x_dim=x_init_i_scaled.shape[1],
            y_target_scaled=y_goal_i_scaled,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            n_init=1,
            steps=100,
            lr=1e-3,
            init_points_scaled=x_init_i_scaled,
            device=str(DEVICE),
            opamp_type=opamp_type,
            goal=",".join(["target"] * len(y_goal_i_scaled)),
            finish_lbfgs=20
        )

        # 逆标准化回物理 X 域
        # StandardScaler: x = x_scaled * scale_ + mean_
        x_phys = best_x_scaled[0] * x_scaler.scale_ + x_scaler.mean_
        final_predictions_phys.append(x_phys)

    # == Step 3: 输出 CSV (物理域的设计参数) ==
    final_pred_df = pd.DataFrame(
        final_predictions_phys, columns=INVERSE_OUTPUT_COLS[opamp_type])
    final_pred_df.to_csv(output_csv, index=False)
    print(f"成功生成混合策略的提交文件: {output_csv.name}")


# ---------------------------------------------------------------------
# 离线评测：在 target split 上跑 simple 推理逻辑并打分
# ---------------------------------------------------------------------

@torch.no_grad()
def offline_eval_forward_simple(
    opamp_type: str,
    results_dir: Path,
    model_choice: str = "align",
    split: str = "val"
):
    """
    用单模型 (align 或 target_only) 直出，
    在 target_* split 上评估 MSE/MAE/R2，全部在物理域。
    """
    data = get_data_and_scalers(opamp_type=opamp_type)

    if split == "train":
        X_std, Y_std = data["target_train"]
    elif split == "val":
        X_std, Y_std = data["target_val"]
    elif split == "all":
        Xa, Ya = data["target_train"]
        Xb, Yb = data["target_val"]
        X_std = np.concatenate([Xa, Xb], axis=0)
        Y_std = np.concatenate([Ya, Yb], axis=0)
    else:
        raise ValueError("split 仅支持 train / val / all")

    # 模型
    ckpt_path = results_dir / \
        f"{opamp_type}_{'finetuned' if model_choice=='align' else 'target_only'}.pth"
    model = _load_forward_model(opamp_type, ckpt_path)

    # 前向 (标准化空间)
    X_t = torch.tensor(X_std, dtype=torch.float32, device=DEVICE)
    mu_std, _, _ = model(X_t)
    mu_std = mu_std.cpu().numpy()

    # 还原到物理域
    y_scaler = data["y_scaler"]
    y_cols = list(data["raw_target"][1].columns)
    y_pred_phys = _to_physical_y(mu_std, y_scaler, y_cols)
    y_true_phys = _to_physical_y(Y_std,  y_scaler, y_cols)

    # 逐列指标
    mse_list = []
    mae_list = []
    r2_list = []
    print(
        f"\n=== Offline Eval (simple forward) [{opamp_type} | model={model_choice} | split={split}] ===")
    for j, name in enumerate(y_cols):
        yt = y_true_phys[:, j]
        yp = y_pred_phys[:, j]
        mse_j = mean_squared_error(yt, yp)
        mae_j = mean_absolute_error(yt, yp)
        r2_j = r2_score(yt, yp)
        mse_list.append(mse_j)
        mae_list.append(mae_j)
        r2_list.append(r2_j)
        print(f"{name:14s}  MSE={mse_j:.4g}  MAE={mae_j:.4g}  R2={r2_j:.4f}")

    print("\nAvg  (all dims)   MSE={:.4g}  MAE={:.4g}  R2={:.4f}".format(
        float(np.mean(mse_list)),
        float(np.mean(mae_list)),
        float(np.mean(r2_list)),
    ))


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="为EDA比赛生成提交文件 (v7 - simple/ensemble, unified log restore, offline eval, auto-augment two_stage features for predB)"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/02_public_test_set",
        help="测试集根目录 (包含 features/*.csv)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="submission",
        help="预测输出目录"
    )

    # forward 推理策略
    parser.add_argument(
        "--forward_mode",
        type=str,
        default="simple",
        choices=["simple", "ensemble"],
        help="simple=单模型直出；ensemble=双模型+温度+加权融合"
    )
    parser.add_argument(
        "--forward_model",
        type=str,
        default="align",
        choices=["align", "target"],
        help="当 forward_mode=simple 时，用哪个模型：align->*_finetuned.pth / target->*_target_only.pth"
    )

    # inverse 策略
    parser.add_argument(
        "--inverse_strategy",
        type=str,
        default="hybrid",
        choices=["mdn_only", "hybrid"],
        help="反向预测策略（当前仅实现hybrid）"
    )

    # offline eval 开关
    parser.add_argument(
        "--offline_eval",
        action="store_true",
        help="是否在 target split 上做离线评测(只对 forward simple 做影子榜单)"
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="val",
        choices=["train", "val", "all"],
        help="offline_eval 用哪个 split"
    )

    args = parser.parse_args()

    data_root = PROJECT_ROOT / args.data_dir
    features_dir = data_root / "features"

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("开始生成提交文件 (v7)")
    print(f"执行目录 (PWD): {Path.cwd()}")
    print(f"项目根目录 (推断): {PROJECT_ROOT}")
    print(f"SRC 目录 (推断):  {SRC_DIR}")
    print(f"模型/Scaler来源: {RESULTS_DIR}")
    print(f"输入数据来源:   {features_dir}")
    print(f"结果输出至:     {output_dir}")
    print("=" * 60)

    # ---------- 正向推理：A / B ----------
    if args.forward_mode == "simple":
        predict_forward_simple(
            "5t_opamp",
            features_dir / "features_A.csv",
            output_dir / "predA.csv",
            RESULTS_DIR,
            model_choice=args.forward_model
        )
        predict_forward_simple(
            "two_stage_opamp",
            features_dir / "features_B.csv",
            output_dir / "predB.csv",
            RESULTS_DIR,
            model_choice=args.forward_model
        )
    else:
        predict_forward_ensemble(
            "5t_opamp",
            features_dir / "features_A.csv",
            output_dir / "predA.csv",
            RESULTS_DIR
        )
        predict_forward_ensemble(
            "two_stage_opamp",
            features_dir / "features_B.csv",
            output_dir / "predB.csv",
            RESULTS_DIR
        )

    # ---------- 反向推理：C / D ----------
    if args.inverse_strategy == "hybrid":
        predict_inverse_hybrid(
            "5t_opamp",
            features_dir / "features_C.csv",
            output_dir / "predC.csv",
            RESULTS_DIR
        )
        predict_inverse_hybrid(
            "two_stage_opamp",
            features_dir / "features_D.csv",
            output_dir / "predD.csv",
            RESULTS_DIR
        )
    else:
        print("\n'mdn_only' 策略暂未实现，请使用 'hybrid' 策略。")

    # ---------- 离线评测（影子榜单） ----------
    if args.offline_eval:
        print("\n--- 运行离线评测（simple forward 影子榜单，物理域口径） ---")
        offline_eval_forward_simple(
            "5t_opamp",
            RESULTS_DIR,
            model_choice=args.forward_model,
            split=args.eval_split
        )
        offline_eval_forward_simple(
            "two_stage_opamp",
            RESULTS_DIR,
            model_choice=args.forward_model,
            split=args.eval_split
        )

    print("\n" + "=" * 60)
    print("所有提交文件已成功生成！")
    print(f"请检查 '{output_dir.resolve()}' 目录。")
    print("=" * 60)


if __name__ == "__main__":
    main()