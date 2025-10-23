# inference/infer_ensemble.py
import os
import re
import numpy as np
import torch
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import config
from data_loader import get_data_and_scalers
from models.align_hetero import AlignHeteroMLP

DEVICE = torch.device(config.DEVICE)
OPAMP_TYPE = config.OPAMP_TYPE
DEFAULT_COLS = ['slewrate_pos', 'dc_gain', 'ugf', 'phase_margin', 'cmrr']


# ---------- 小工具 ----------
def _pick_existing_path(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

def _load_state(obj):
    # 兼容 {"state_dict": ...} / 直接 state_dict
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    return obj

def _strip_prefixes(name: str):
    # 去除常见前缀
    if name.startswith("module."):
        name = name[7:]
    return name

def _int_between_dots(name: str):
    # 提取诸如 backbone.network.12.weight -> 12
    nums = re.findall(r'\.(\d+)\.', name)
    return int(nums[-1]) if nums else -1

def _ordered_linear_keys(state_dict, prefix: str):
    """
    从 state_dict 中取出以 prefix 开头并以 .weight 结尾且是2D权重的键，
    按索引号升序返回（Linear 层顺序）。
    """
    items = []
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if (k.startswith(prefix) and k.endswith(".weight") and v.ndim == 2):
            items.append(k)
    items.sort(key=_int_between_dots)
    return items


# ---------- 适配式加载器 ----------
def load_align_hetero_adapt(path, input_dim, output_dim):
    """
    加载 AlignHeteroMLP，如果严格加载失败，则尝试从
    {backbone.* + head_B_mu.*, head_B_logvar.*} 自动映射到
    {backbone.network.* + hetero_head.*}
    """
    model = AlignHeteroMLP(input_dim, output_dim).to(DEVICE)
    raw = torch.load(path, map_location=DEVICE)
    src = { _strip_prefixes(k): v for k, v in _load_state(raw).items() if isinstance(v, torch.Tensor) }

    # 1) 先尝试严格加载（已是对齐命名）
    try:
        model.load_state_dict(src, strict=True)
        print(f"[load] 严格加载成功: {path}")
        model.eval()
        return model
    except Exception as e:
        print(f"[load] 严格加载失败，进入自适配: {e}")

    # 2) 自适配：构造一个与 model.state_dict() 匹配的新字典
    dst = model.state_dict()
    new_state = {}

    # 2.1 backbone 映射
    # 情况A：ckpt 使用 backbone.network.*
    src_backbone_net = _ordered_linear_keys(src, "backbone.network.")
    # 情况B：ckpt 使用 backbone.*（无 network）
    src_backbone_plain = _ordered_linear_keys(src, "backbone.")

    # 目标模型 backbone 的线性层（按顺序）
    dst_backbone = _ordered_linear_keys(dst, "backbone.network.")

    if src_backbone_net:
        # 直接同名拷贝（只要形状匹配）
        for k in src_backbone_net:
            if k in dst and src[k].shape == dst[k].shape:
                new_state[k] = src[k]
                b_key = k.replace(".weight", ".bias")
                if b_key in src and b_key in dst and src[b_key].shape == dst[b_key].shape:
                    new_state[b_key] = src[b_key]
        print(f"[map] backbone: 采用 backbone.network.* 直接拷贝 {len(src_backbone_net)} 层。")
    elif src_backbone_plain:
        # 需要把 backbone.* → backbone.network.*，并将最后一层来自 head_B_mu
        # 先复制除最后一层外的线性层
        copy_count = min(len(src_backbone_plain), max(0, len(dst_backbone)-1))
        for i in range(copy_count):
            s_w = src_backbone_plain[i]
            d_w = dst_backbone[i]
            s_b = s_w.replace(".weight", ".bias")
            d_b = d_w.replace(".weight", ".bias")
            if src[s_w].shape == dst[d_w].shape and src[s_b].shape == dst[d_b].shape:
                new_state[d_w] = src[s_w]
                new_state[d_b] = src[s_b]
            else:
                print(f"[warn] 跳过 backbone 第{i}层：shape 不匹配 {tuple(src[s_w].shape)} vs {tuple(dst[d_w].shape)}")

        # 最后一层（μ），来自 head_B_mu
        head_mu_w = "head_B_mu.weight"
        head_mu_b = "head_B_mu.bias"
        if len(dst_backbone) >= 1 and head_mu_w in src and head_mu_b in src:
            last_w = dst_backbone[-1]
            last_b = last_w.replace(".weight", ".bias")
            if src[head_mu_w].shape == dst[last_w].shape and src[head_mu_b].shape == dst[last_b].shape:
                new_state[last_w] = src[head_mu_w]
                new_state[last_b] = src[head_mu_b]
                print("[map] backbone 最后一层由 head_B_mu.* 映射完成。")
            else:
                print(f"[warn] head_B_mu.* 与目标最后一层 shape 不匹配，"
                      f"{tuple(src[head_mu_w].shape)} vs {tuple(dst[last_w].shape)}")
        else:
            print("[warn] 未找到 head_B_mu.*，无法映射 backbone 最后一层（μ）。")
    else:
        print("[warn] 未发现可用的 backbone 权重键（backbone.* / backbone.network.*）。")

    # 2.2 hetero_head（logvar）映射
    if "hetero_head.weight" in dst and "hetero_head.bias" in dst:
        if "hetero_head.weight" in src and "hetero_head.bias" in src:
            if src["hetero_head.weight"].shape == dst["hetero_head.weight"].shape:
                new_state["hetero_head.weight"] = src["hetero_head.weight"]
                new_state["hetero_head.bias"] = src["hetero_head.bias"]
                print("[map] 直接使用 hetero_head.* 权重。")
        elif "head_B_logvar.weight" in src and "head_B_logvar.bias" in src:
            if src["head_B_logvar.weight"].shape == dst["hetero_head.weight"].shape:
                new_state["hetero_head.weight"] = src["head_B_logvar.weight"]
                new_state["hetero_head.bias"] = src["head_B_logvar.bias"]
                print("[map] 由 head_B_logvar.* → hetero_head.* 映射完成。")
        else:
            print("[warn] 未找到异方差头（hetero/logvar）权重，使用随机初始化。")

    # 2.3 合并到目标
    merged = {**dst, **new_state}
    missing = [k for k in dst.keys() if k not in merged or merged[k].shape != dst[k].shape]
    print(f"[load] 合并后缺失参数数: {len(missing)}")
    model.load_state_dict(merged, strict=False)
    model.eval()
    return model


@torch.no_grad()
def predict_mu_logvar(model, X):
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    mu, logvar, _ = model(X_t)
    return mu.cpu().numpy(), logvar.cpu().numpy()


def fit_temp(mu, logv, y):
    resid2 = (y - mu) ** 2
    var = np.exp(logv)
    c2 = np.mean(resid2 / (var + 1e-12), axis=0)
    return np.sqrt(np.maximum(c2, 1e-6))


def main():
    # 1) 数据
    data = get_data_and_scalers(opamp_type=OPAMP_TYPE)
    X_val, y_val = data['target_val']
    input_dim, output_dim = X_val.shape[1], y_val.shape[1]

    # 2) 模型路径
    align_candidates = [
        f"results/{OPAMP_TYPE}_align_hetero_lambda{config.LAMBDA_CORAL:.3f}.pth",
        f"results/{OPAMP_TYPE}_align_hetero.pth",
        f"../results/{OPAMP_TYPE}_align_hetero_lambda{config.LAMBDA_CORAL:.3f}.pth",
    ]
    target_only_candidates = [
        f"results/{OPAMP_TYPE}_target_only_hetero.pth",
        f"../results/{OPAMP_TYPE}_target_only_hetero.pth",
    ]
    align_ckpt = _pick_existing_path(align_candidates)
    target_only_ckpt = _pick_existing_path(target_only_candidates)
    if align_ckpt is None:
        raise FileNotFoundError("未找到对齐模型权重：\n  - " + "\n  - ".join(align_candidates))
    if target_only_ckpt is None:
        raise FileNotFoundError("未找到 target-only 模型权重：\n  - " + "\n  - ".join(target_only_candidates))

    # 3) 加载两种模型（自动适配不同命名/结构）
    m_align = load_align_hetero_adapt(align_ckpt, input_dim, output_dim)
    m_trg   = load_align_hetero_adapt(target_only_ckpt, input_dim, output_dim)

    # 4) 预测 μ / logv
    mu_a, logv_a = predict_mu_logvar(m_align, X_val)
    mu_t, logv_t = predict_mu_logvar(m_trg,  X_val)

    # 5) 温度标定
    c_a = fit_temp(mu_a, logv_a, y_val)
    c_t = fit_temp(mu_t, logv_t, y_val)
    logv_a = logv_a + 2.0 * np.log(c_a[None, :])
    logv_t = logv_t + 2.0 * np.log(c_t[None, :])

    # 6) precision 权重
    tau_a = np.exp(-logv_a)
    tau_t = np.exp(-logv_t)
    clip_a = np.percentile(tau_a, 95, axis=0, keepdims=True)
    clip_t = np.percentile(tau_t, 95, axis=0, keepdims=True)
    tau_a = np.minimum(tau_a, clip_a)
    tau_t = np.minimum(tau_t, clip_t)
    w_prec_a = tau_a / (tau_a + tau_t + 1e-12)  # [N, D]

    # 7) MSE 权重（维度级）
    mse_a = np.array([mean_squared_error(y_val[:, i], mu_a[:, i]) for i in range(output_dim)])
    mse_t = np.array([mean_squared_error(y_val[:, i], mu_t[:, i]) for i in range(output_dim)])
    w_mse_a = 1.0 / (mse_a + 1e-12)
    w_mse_t = 1.0 / (mse_t + 1e-12)
    s = w_mse_a + w_mse_t
    w_mse_a /= s
    w_mse_t /= s

    # 8) 融合权重
    if output_dim == 5:
        ALPHA = np.array([0.7, 0.7, 0.3, 0.7, 0.85], dtype=np.float64)[None, :]
    else:
        ALPHA = np.full((1, output_dim), 0.5, dtype=np.float64)

    w_a = ALPHA * w_prec_a + (1.0 - ALPHA) * w_mse_a[None, :]
    w_t = 1.0 - w_a

    # 9) 集成均值（标准化空间）
    mu_ens = w_a * mu_a + w_t * mu_t

    # 10) 反标准化回物理单位
    y_scaler_path = _pick_existing_path([
        f"results/{OPAMP_TYPE}_y_scaler.gz",
        f"../results/{OPAMP_TYPE}_y_scaler.gz",
    ])
    if y_scaler_path is None:
        raise FileNotFoundError("未找到 y_scaler：results/ 或 ../results/ 下的 *_y_scaler.gz")
    y_scaler = joblib.load(y_scaler_path)

    y_pred = y_scaler.inverse_transform(mu_ens)
    y_true = y_scaler.inverse_transform(y_val)

    # 11) 反 log1p
    cols = DEFAULT_COLS if output_dim == len(DEFAULT_COLS) else [f"y{i}" for i in range(output_dim)]
    for j, name in enumerate(cols):
        if name in ['ugf', 'cmrr']:
            y_pred[:, j] = np.expm1(y_pred[:, j])
            y_true[:, j] = np.expm1(y_true[:, j])

    # 12) 评估
    print("\n=== Ensemble on B-VAL (物理单位) ===")
    for j, name in enumerate(cols):
        mse = mean_squared_error(y_true[:, j], y_pred[:, j])
        mae = mean_absolute_error(y_true[:, j], y_pred[:, j])
        r2  = r2_score(y_true[:, j], y_pred[:, j])
        print(f"{name:14s}  MSE={mse:.4g}  MAE={mae:.4g}  R2={r2:.4f}")


if __name__ == "__main__":
    main()
