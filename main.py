#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
单文件纯推理 main.py：
- 不读取 data/01_train_set/*
- 只加载已训练好的 scaler(.gz) 与模型(.pth)
- A/B 正向：joblib 的 x/y scaler + finetuned/target_only 模型
- C/D 反向：joblib 的 y/x scaler + mdn_{opamp}.pth + finetuned 前向模型做 hybrid 优化
"""
# models/mlp.py
import torch.nn as nn
from typing import List  # <-- 导入 List
import argparse
from pathlib import Path
import sys, os, time, platform, hashlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from tqdm import tqdm

# =============== 环境/工具 ===============

def _sha8(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]

def _fp(p: Path, tag: str):
    print(f"[fingerprint] {tag}: {p}  size={p.stat().st_size}  sha256[:8]={_sha8(p)}")

def _print_env():
    import numpy, sklearn
    print(f"[env] python={platform.python_version()}  torch={torch.__version__}  "
          f"numpy={numpy.__version__}  sklearn={sklearn.__version__}")
    if torch.cuda.is_available():
        print(f"[env] cuda_available=True  device_name={torch.cuda.get_device_name(0)}")
    # 关闭 TF32，避免不同硬件路径差异
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

def _set_runtime(force_cpu: bool, deterministic: bool):
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if deterministic:
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float32)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============== 与训练一致的配置（硬编码版） ===============

LOG_TRANSFORMED_COLS = [
    "ugf",
    "cmrr",
    "dc_gain",
    "slewrate_pos",
]

TASK_CONFIGS = {
    '5t_opamp': {
        # 模型设置
        'hidden_dims': [128, 256, 256, 512],
        'num_layers': 4,
        'dropout_rate': 0.2,
        # 反向模型
        'mdn_components': 20,
        'mdn_hidden_dim': 128,
        'mdn_num_layers': 3,
    },
    'two_stage_opamp': {
        # 模型设置
        'hidden_dims': [256, 256, 256, 256],
        'num_layers': 4,
        'dropout_rate': 0.2,
        # 反向模型
        'mdn_components': 20,
        'mdn_hidden_dim': 128,
        'mdn_num_layers': 3,
    },
}

FORWARD_OUTPUT_COLS = ['slewrate_pos', 'dc_gain', 'ugf', 'phase_margin', 'cmrr']

INVERSE_OUTPUT_COLS = {
    '5t_opamp': ['w1', 'w2', 'w3', 'l1', 'l2', 'l3', 'ibias'],
    'two_stage_opamp': ['w1','w2','w3','w4','w5','l1','l2','l3','l4','l5','cc','cr','ibias']
}

_BASE_INPUT_COLS_TWO_STAGE = ["w1","w2","w3","w4","w5","l1","l2","l3","l4","l5","cc","cr","ibias"]
_EPS = 1e-12

# =============== 与训练一致的模型实现（内联） ===============


class MLP(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: List[int],    # <-- 改为列表
                 dropout_rate: float
                 ):
        """
        构造函数，用于初始化可变隐藏层的 MLP 模型。

        参数:
        - input_dim (int): 输入特征的维度。
        - output_dim (int): 输出预测的维度。
        - hidden_dims (List[int]): 一个整数列表，定义每个隐藏层的维度。
        - dropout_rate (float): Dropout 比率。
        """
        super().__init__()

        if not hidden_dims:
            raise ValueError("hidden_dims 列表不能为空")

        layers = []
        current_dim = input_dim

        # --- 动态构建网络层 ---
        # 1. 循环构建所有隐藏层
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout_rate and dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim  # 更新当前维度为本层输出维度

        # 2. 输出层: 将最后一个隐藏层维度映射到最终的输出维度
        layers.append(nn.Linear(current_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class AlignHeteroMLP(nn.Module):
    """
    与训练一致：MLP 作为 backbone，另加一层线性 head 预测 logvar（异方差）。
    最终返回 (mu, logvar, features)，其中 mu=features。
    """
    def __init__(self, input_dim, output_dim, hidden_dims, dropout_rate: float):
        super().__init__()
        self.backbone = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        )
        self.hetero_head = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        features = self.backbone(x)
        mu = features
        logvar = self.hetero_head(features)
        return mu, logvar, features

class InverseMDN(nn.Module):
    def __init__(self, input_dim, output_dim, n_components, hidden_dim, num_layers):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        self.backbone = nn.Sequential(*layers)
        self.pi = nn.Linear(hidden_dim, n_components)
        self.mu = nn.Linear(hidden_dim, n_components * output_dim)
        self.sigma_raw = nn.Linear(hidden_dim, n_components * output_dim)
        self.n_components = n_components
        self.output_dim = output_dim
        self.softplus = nn.Softplus()
    def forward(self, y):
        h = self.backbone(y)
        pi = torch.softmax(self.pi(h), dim=-1)
        mu = self.mu(h).view(-1, self.n_components, self.output_dim)
        sigma = self.softplus(self.sigma_raw(h)).view(-1, self.n_components, self.output_dim) + 1e-6
        return pi, mu, sigma

# =============== 工具：特征增强/对齐/变换 ===============

def _safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.clip(x, _EPS, None))

def _augment_two_stage_features_in_memory(df: pd.DataFrame) -> pd.DataFrame:
    miss_base = [c for c in _BASE_INPUT_COLS_TWO_STAGE if c not in df.columns]
    if miss_base:
        raise ValueError(f"features_B.csv 缺少基础必需列: {miss_base}")
    out = df.copy()
    w_cols = [f"w{i}" for i in range(1, 6)]
    l_cols = [f"l{i}" for i in range(1, 6)]
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
    log_ibias = out["log_ibias"].to_numpy(); log_cc = out["log_cc"].to_numpy()
    for i in range(1, 6):
        wi = out[f"w{i}"].to_numpy(); li = out[f"l{i}"].to_numpy()
        out[f"w{i}_over_l{i}"] = wi / np.clip(li, _EPS, None)
        out[f"log_w{i}_over_l{i}"] = _safe_log(wi) - _safe_log(li)
        out[f"w{i}_norm"] = wi / w_sum; out[f"l{i}_norm"] = li / l_sum
        area_sqrt = np.sqrt(np.clip(wi * li, _EPS, None))
        out[f"area{i}"] = area_sqrt; out[f"mismatch{i}"] = 1.0 / area_sqrt
        lw = _safe_log(wi); ll = _safe_log(li)
        log_gm_hat = 0.5 * (lw - ll + log_ibias); log_ro_hat = ll - log_ibias
        out[f"log_gm_hat_{i}"] = log_gm_hat; out[f"log_ro_hat_{i}"] = log_ro_hat
        out[f"log_av_hat_{i}"] = log_gm_hat + log_ro_hat
        out[f"log_ugf_hat_{i}"] = 0.5*(lw-ll) + 0.5*log_ibias - log_cc
    out["dcgain_stage1_proxy"] = out["log_av_hat_1"]
    out["dcgain_stage2_proxy"] = out["log_av_hat_3"]
    out["dcgain_total_proxy"]  = out["dcgain_stage1_proxy"] + out["dcgain_stage2_proxy"]
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

def _ensure_two_stage_test_has_train_columns(opamp_type: str,
                                             input_csv: Path,
                                             test_df: pd.DataFrame,
                                             train_feature_cols: list) -> pd.DataFrame:
    missing = set(train_feature_cols) - set(test_df.columns)
    if missing and (opamp_type == "two_stage_opamp") and (input_csv.name == "features_B.csv"):
        print(f"[AUTO-AUG] features_B.csv 缺少 {len(missing)} 个训练特征，执行在线增强…")
        test_df = _augment_two_stage_features_in_memory(test_df)
        missing2 = set(train_feature_cols) - set(test_df.columns)
        if missing2:
            raise ValueError(f"[AUTO-AUG失败] 仍缺特征列: {missing2}")
    elif missing:
        raise ValueError(f"{input_csv.name} 缺少特征列: {missing}")
    return test_df

def _to_physical_y(y_std: np.ndarray, y_scaler, y_colnames):
    y_unstd = y_scaler.inverse_transform(y_std)
    y_phys = y_unstd.copy()
    for idx, col in enumerate(y_colnames):
        if col in LOG_TRANSFORMED_COLS:
            y_phys[:, idx] = np.expm1(y_unstd[:, idx])
    return y_phys

def _get_train_feature_cols_from_scaler(x_scaler, opamp_type: str):
    if hasattr(x_scaler, "feature_names_in_"):
        return list(x_scaler.feature_names_in_)
    raise RuntimeError(
        f"{opamp_type}_x_scaler.gz 缺少 feature_names_in_。"
        "请用与保存时相同版本的 scikit-learn 重新导出，或在 results/ 中提供列名清单。"
    )

# =============== 加载器（scaler / 前向 / 反向） ===============

def _load_scalers(opamp_type: str, SCALER_DIR: Path):
    xs_path = SCALER_DIR / f"{opamp_type}_x_scaler.gz"
    ys_path = SCALER_DIR / f"{opamp_type}_y_scaler.gz"
    if not xs_path.exists() or not ys_path.exists():
        raise FileNotFoundError(f"缺少 scaler: {xs_path} 或 {ys_path}")
    _fp(xs_path, f"{opamp_type}_x_scaler.gz")
    _fp(ys_path, f"{opamp_type}_y_scaler.gz")
    xs = joblib.load(xs_path)
    ys = joblib.load(ys_path)
    return xs, ys

def _load_forward_model(opamp_type: str, x_scaler, y_scaler, MODEL_DIR: Path) -> AlignHeteroMLP:
    cfg = TASK_CONFIGS[opamp_type]
    input_dim = int(x_scaler.mean_.shape[0])
    output_dim = int(y_scaler.mean_.shape[0])
    model = AlignHeteroMLP(input_dim, output_dim, cfg['hidden_dims'], cfg['dropout_rate']).to(DEVICE)
    ckpt_path = MODEL_DIR / f"{opamp_type}_finetuned.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"缺少模型权重: {ckpt_path}")
    _fp(ckpt_path, ckpt_path.name)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model_state = state.get('state_dict', state)
    # 结构必须严格一致，否则直接报错，避免“部分加载”
    model.load_state_dict(model_state, strict=True)
    model.eval()
    return model

def _load_forward_model_target_only(opamp_type: str, x_scaler, y_scaler, MODEL_DIR: Path) -> AlignHeteroMLP:
    cfg = TASK_CONFIGS[opamp_type]
    input_dim = int(x_scaler.mean_.shape[0])
    output_dim = int(y_scaler.mean_.shape[0])
    model = AlignHeteroMLP(input_dim, output_dim, cfg['hidden_dims'], cfg['dropout_rate']).to(DEVICE)
    ckpt_path = MODEL_DIR / f"{opamp_type}_target_only.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"缺少模型权重: {ckpt_path}")
    _fp(ckpt_path, ckpt_path.name)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model_state = state.get('state_dict', state)
    model.load_state_dict(model_state, strict=True)
    model.eval()
    return model

def _load_inverse_mdn(opamp_type: str, MODEL_DIR: Path) -> InverseMDN:
    ckpt = MODEL_DIR / f"mdn_{opamp_type}.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"缺少反向 MDN 权重: {ckpt}")
    _fp(ckpt, ckpt.name)
    blob = torch.load(ckpt, map_location=DEVICE)
    cfg = blob.get("config")
    if not cfg:
        raise ValueError(f"{ckpt.name} 中缺少 config")
    m = InverseMDN(cfg["input_dim"], cfg["output_dim"], cfg["n_components"], cfg["hidden_dim"], cfg["num_layers"]).to(DEVICE)
    m.load_state_dict(blob.get("state_dict", blob), strict=True)
    m.eval()
    return m

def _forward_y_std(model, x_scaled_tensor: torch.Tensor):
    mu, _, _ = model(x_scaled_tensor)
    return mu

# =============== 反向优化（hybrid） ===============

def optimize_x_multi_start_simple(model: nn.Module,
                                  x_scaler,
                                  y_target_scaled: np.ndarray,
                                  x_init_scaled: np.ndarray,
                                  steps: int = 1000,
                                  lr: float = 1e-4,
                                  n_init: int = 1):
    device = next(model.parameters()).device
    y_t = torch.from_numpy(y_target_scaled.astype(np.float32)).to(device).view(1, -1)
    init = np.asarray(x_init_scaled, dtype=np.float32)
    if init.shape[0] < n_init:
        reps = int(np.ceil(n_init / init.shape[0])); init = np.vstack([init]*reps)[:n_init]
    else:
        init = init[:n_init]
    x = torch.from_numpy(init).to(device).clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    mean_t = torch.from_numpy(x_scaler.mean_.astype(np.float32)).to(device)
    scale_t = torch.from_numpy(x_scaler.scale_.astype(np.float32)).to(device)
    x_min_phys = mean_t - 5.0 * scale_t
    x_max_phys = mean_t + 5.0 * scale_t
    best_loss = float("inf"); best_x = None
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        y_pred = _forward_y_std(model, x)
        loss = torch.mean((y_pred - y_t)**2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([x], 1.0)
        opt.step()
        with torch.no_grad():
            x_phys = x * scale_t + mean_t
            x_phys.clamp_(x_min_phys, x_max_phys)
            x.copy_((x_phys - mean_t) / scale_t)
        cur = float(loss.item())
        if cur < best_loss: best_loss, best_x = cur, x.detach().clone()
    return best_x.cpu().numpy(), best_loss

# =============== A/B 正向推理 ===============

def predict_forward_simple(opamp_type: str, input_csv: Path, output_csv: Path,
                           MODEL_DIR: Path, SCALER_DIR: Path, model_choice: str = "align"):
    print(f"\n=== 正向简单预测: {opamp_type} [{model_choice}] ===")
    x_scaler, y_scaler = _load_scalers(opamp_type, SCALER_DIR)
    train_feature_cols = _get_train_feature_cols_from_scaler(x_scaler, opamp_type)

    test_df = pd.read_csv(input_csv)
    _fp(input_csv, input_csv.name)
    test_df = _ensure_two_stage_test_has_train_columns(opamp_type, input_csv, test_df, train_feature_cols)

    # 严格对齐列顺序（DataFrame 保留列名，便于 sklearn 做检查）
    test_df = test_df[train_feature_cols].copy()
    assert list(test_df.columns) == list(train_feature_cols), "特征列顺序不一致！"

    x_test_scaled = x_scaler.transform(test_df)

    if model_choice == "align":
        model = _load_forward_model(opamp_type, x_scaler, y_scaler, MODEL_DIR)
    else:
        model = _load_forward_model_target_only(opamp_type, x_scaler, y_scaler, MODEL_DIR)

    t0 = time.perf_counter()
    with torch.no_grad():
        x_t = torch.tensor(x_test_scaled, dtype=torch.float32, device=DEVICE)
        y_std = _forward_y_std(model, x_t).cpu().numpy()
    print(f"[forward] 模型前向 {time.perf_counter()-t0:.3f}s")

    y_cols = FORWARD_OUTPUT_COLS[:y_std.shape[1]]
    y_phys = _to_physical_y(y_std, y_scaler, y_cols)

    pd.DataFrame(y_phys, columns=y_cols).to_csv(output_csv, index=False)
    print(f"[forward] 写出 {output_csv.name}")

# =============== C/D 反向（MDN + 前向优化） ===============

def predict_inverse_hybrid(opamp_type: str, input_csv: Path, output_csv: Path,
                           MODEL_DIR: Path, SCALER_DIR: Path):
    print(f"\n=== 反向混合预测: {opamp_type} ===")
    x_scaler, y_scaler = _load_scalers(opamp_type, SCALER_DIR)
    mdn = _load_inverse_mdn(opamp_type, MODEL_DIR)

    y_df = pd.read_csv(input_csv).copy()
    _fp(input_csv, input_csv.name)
    for col in LOG_TRANSFORMED_COLS:
        if col in y_df.columns:
            y_df[col] = np.log1p(y_df[col])
    y_scaled = y_scaler.transform(y_df.values).astype(np.float32)

    with torch.no_grad():
        y_t = torch.tensor(y_scaled, dtype=torch.float32, device=DEVICE)
        pi, mu, _ = mdn(y_t)
        x_init_scaled = torch.sum(pi.unsqueeze(-1) * mu, dim=1).cpu().numpy()

    fwd_model = _load_forward_model(opamp_type, x_scaler, y_scaler, MODEL_DIR)

    final_x_phys = []
    for i in tqdm(range(len(y_scaled)), desc=f"Optimizing {opamp_type}"):
        best_x_scaled, _ = optimize_x_multi_start_simple(
            model=fwd_model, x_scaler=x_scaler,
            y_target_scaled=y_scaled[i],
            x_init_scaled=x_init_scaled[i][np.newaxis, :],
            steps=1000, lr=1e-4, n_init=1
        )
        x_phys = best_x_scaled[0] * x_scaler.scale_ + x_scaler.mean_
        final_x_phys.append(x_phys)

    train_feature_cols = _get_train_feature_cols_from_scaler(x_scaler, opamp_type)
    full_df = pd.DataFrame(final_x_phys, columns=train_feature_cols)
    submit_cols = INVERSE_OUTPUT_COLS[opamp_type]
    full_df[submit_cols].to_csv(output_csv, index=False)
    print(f"[inverse] 写出 {output_csv.name}")

# =============== main ===============

def main():
    parser = argparse.ArgumentParser(description="单文件纯推理：A/B 正向 + C/D 反向（只用已保存的 scaler 与模型）")
    parser.add_argument("--inputDir", type=str, required=True, help="包含 features_A/B/C/D.csv 的目录或其父目录")
    parser.add_argument("--outputDir", type=str, required=True, help="输出目录（必须存在或可创建）")
    parser.add_argument("--model_dir", type=str, required=True, help="*.pth 所在目录（必须）")
    parser.add_argument("--scaler_dir", type=str, required=True, help="*_x/y_scaler.gz 所在目录（必须）")
    parser.add_argument("--forward_model", type=str, default="align", choices=["align", "target"],
                        help="simple 模式下使用的前向模型：align->*_finetuned.pth / target->*_target_only.pth")
    parser.add_argument("--force_cpu", action="store_true", help="强制用 CPU 推理（确保可复现）")
    parser.add_argument("--deterministic", action="store_true", help="开启确定性（禁 TF32）")
    args = parser.parse_args()

    _set_runtime(force_cpu=args.force_cpu, deterministic=args.deterministic)
    _print_env()

    input_dir = Path(args.inputDir).resolve()
    output_dir = Path(args.outputDir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    MODEL_DIR  = Path(args.model_dir).resolve()
    SCALER_DIR = Path(args.scaler_dir).resolve()
    assert MODEL_DIR.exists(),  f"MODEL_DIR 不存在: {MODEL_DIR}"
    assert SCALER_DIR.exists(), f"SCALER_DIR 不存在: {SCALER_DIR}"
    print("="*60)
    print(f"[paths] input_dir={input_dir}")
    print(f"[paths] output_dir={output_dir}")
    print(f"[paths] MODEL_DIR={MODEL_DIR}")
    print(f"[paths] SCALER_DIR={SCALER_DIR}")
    print("="*60)

    def _csv(name: str) -> Path:
        p1 = input_dir / name
        if p1.exists():
            _fp(p1, name)
            return p1
        p2 = input_dir / "features" / name
        if p2.exists():
            _fp(p2, f"features/{name}")
            return p2
        raise FileNotFoundError(f"找不到 {name}（在 {input_dir} 或 {input_dir/'features'}）")

    fa = _csv("features_A.csv")
    fb = _csv("features_B.csv")
    fc = _csv("features_C.csv")
    fd = _csv("features_D.csv")

    t_all = time.perf_counter()

    # A
    tA = time.perf_counter()
    predict_forward_simple("5t_opamp", fa, output_dir / "pred_A.csv",
                           MODEL_DIR, SCALER_DIR, model_choice=args.forward_model)
    print(f"[timer] A 总耗时 {time.perf_counter()-tA:.3f}s")

    # B
    tB = time.perf_counter()
    predict_forward_simple("two_stage_opamp", fb, output_dir / "pred_B.csv",
                           MODEL_DIR, SCALER_DIR, model_choice=args.forward_model)
    print(f"[timer] B 总耗时 {time.perf_counter()-tB:.3f}s")

    # C
    tC = time.perf_counter()
    predict_inverse_hybrid("5t_opamp", fc, output_dir / "pred_C.csv",
                           MODEL_DIR, SCALER_DIR)
    print(f"[timer] C 总耗时 {time.perf_counter()-tC:.3f}s")

    # D
    tD = time.perf_counter()
    predict_inverse_hybrid("two_stage_opamp", fd, output_dir / "pred_D.csv",
                           MODEL_DIR, SCALER_DIR)
    print(f"[timer] D 总耗时 {time.perf_counter()-tD:.3f}s")

    print("\n" + "="*60)
    print("[main] 全部完成，pred_A/B/C/D.csv 已生成于：", output_dir)
    print(f"[main] 总耗时 {time.perf_counter()-t_all:.3f}s")
    print("="*60)

if __name__ == "__main__":
    main()
