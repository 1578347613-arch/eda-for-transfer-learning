#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_B.py

只把 hidden_dim / num_layers 当作列表做网格扫描，其他所有参数一律从 config 读取：
- 训练策略：对齐(align) 两阶段（source 预训练 + 对齐微调），不跑 target_only
- 数据与维度：从 get_data_and_scalers(opamp_type) 读取
- 评估：在 target 验证集（物理域）计算逐列与平均 MSE/MAE/R2，并打印本次试验用到的参数
- 结果：src/results/sweeps_B/<opamp>/HD{hd}_L{nl}/ 下保存 ckpt 与单次指标；
        汇总写到 src/results/sweeps_B/<opamp>/sweep_summary.csv
"""

import argparse
import itertools
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 项目内模块（请在 src/ 目录运行本脚本）
from data_loader import get_data_and_scalers
from models.align_hetero import AlignHeteroMLP
from loss_function import heteroscedastic_nll, batch_r2, coral_loss
from config import COMMON_CONFIG, TASK_CONFIGS, LOG_TRANSFORMED_COLS
from evaluate import calculate_and_print_metrics

SRC_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SRC_DIR / "results"
SWEEP_ROOT = RESULTS_DIR / "sweeps_B"


# ----------------------------
# 小工具
# ----------------------------
def parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    return [int(p.strip()) for p in s.split(",") if p.strip()]


def make_loader(x, y, bs, shuffle=True, drop_last=False):
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=shuffle, drop_last=drop_last)


def build_device() -> torch.device:
    dev_str = COMMON_CONFIG.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if dev_str.lower().startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，已回退到 CPU。")
        return torch.device("cpu")
    return torch.device(dev_str)


# ----------------------------
# 阶段一：source 预训练（HuberLoss）
# ----------------------------
def run_pretraining(model, train_loader, val_loader, device, save_path,
                    lr_pretrain: float, epochs_pretrain: int, patience_pretrain: int):
    print("\n--- [阶段一] Backbone 预训练 (source_train / source_val, HuberLoss) ---")
    optimizer = torch.optim.AdamW(model.backbone.parameters(), lr=lr_pretrain)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=1, eta_min=1e-6)
    criterion = torch.nn.HuberLoss(delta=1.0)

    best_val = float("inf")
    patience_cnt = patience_pretrain

    for epoch in range(epochs_pretrain):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            mu, _, _ = model(xb)
            loss = criterion(mu, yb)
            loss.backward()
            optimizer.step()
            total += loss.item()
        train_loss = total / max(1, len(train_loader))

        model.eval()
        total = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                mu, _, _ = model(xb)
                total += criterion(mu, yb).item()
        val_loss = total / max(1, len(val_loader))
        scheduler.step()

        print(f"Pretrain [{epoch + 1}/{epochs_pretrain}]  train={train_loss:.6f}  val={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
            patience_cnt = patience_pretrain
            print(f"  -> 预训练验证改进，保存 {save_path}")
        else:
            patience_cnt -= 1
            if patience_cnt == 0:
                print(f"  -> 验证未改进 {patience_pretrain} 次，早停。")
                break

    print(f"[PRETRAIN] 最佳 val={best_val:.6f} 已保存。")


# ----------------------------
# 阶段二：对齐微调（NLL + α·(1−R2) + λ·CORAL）
# ----------------------------
def run_finetuning_align(model, loaders, device, save_path,
                         lr_finetune: float, epochs_finetune: int, patience_finetune: int,
                         alpha_r2: float, lambda_coral: float):
    print("\n--- [阶段二] 对齐微调 (NLL + α·(1−R2) + λ·CORAL) ---")
    opt = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": lr_finetune / 10.0},
            {"params": model.hetero_head.parameters(), "lr": lr_finetune},
        ],
        weight_decay=1e-4,
    )

    best_val = float("inf")
    patience_cnt = patience_finetune

    dl_A, dl_B, dl_val = loaders["source"], loaders["target_train"], loaders["target_val"]
    dl_A_iter = iter(dl_A)

    for epoch in range(epochs_finetune):
        model.train()
        for xb_B, yb_B in dl_B:
            xb_B, yb_B = xb_B.to(device), yb_B.to(device)

            try:
                xa_A, _ = next(dl_A_iter)
            except StopIteration:
                dl_A_iter = iter(dl_A)
                xa_A, _ = next(dl_A_iter)

            if xa_A.size(0) != xb_B.size(0):
                xa_A = xa_A[: xb_B.size(0)]
            xa_A = xa_A.to(device)

            mu_B, logvar_B, feat_B = model(xb_B)
            with torch.no_grad():
                _, _, feat_A = model(xa_A)

            nll = heteroscedastic_nll(mu_B, logvar_B, yb_B)
            r2_loss = (1.0 - batch_r2(yb_B, mu_B).clamp(min=-1.0, max=1.0)).mean()
            coral = coral_loss(feat_A, feat_B)
            loss = nll + alpha_r2 * r2_loss + lambda_coral * coral

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # 验证（仅看 NLL）
        model.eval()
        total = 0.0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                mu, logvar, _ = model(xb)
                total += heteroscedastic_nll(mu, logvar, yb).item()
        val_loss = total / len(dl_val)

        print(f"Fine-tune [{epoch + 1}/{epochs_finetune}]  val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
            patience_cnt = patience_finetune
            print(f"  -> 验证改进，保存 {save_path}")
        else:
            patience_cnt -= 1
            if patience_cnt == 0:
                print(f"  -> 验证未改进 {patience_finetune} 次，早停。")
                break

    print(f"[FINETUNE] 最佳验证损失={best_val:.6f} 已保存。")


# ----------------------------
# 评估：target 验证集（物理域）
# ----------------------------
@torch.no_grad()
def eval_and_log(model: torch.nn.Module,
                 device: torch.device,
                 ckpt_path: Path,
                 data,
                 run_meta: Dict) -> Dict[str, float]:
    """
    打印“本次试验参数 + 逐列指标”，并返回 metrics 字典用于保存。
    """
    X_val, Y_val = data["target_val"]
    y_scaler = data["y_scaler"]
    y_cols = list(data["raw_target"][1].columns)

    # 载入权重并预测（标准化空间）
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    X_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_pred_scaled, _, _ = model(X_t)
    y_pred_scaled = y_pred_scaled.cpu().numpy()

    # —— 打印本次试验使用的关键参数 ——
    print("\n=== 本次试验参数 (Run Config) ===")
    for k in [
        "opamp", "hidden_dim", "num_layers",
        "lr_pretrain", "epochs_pretrain", "patience_pretrain",
        "lr_finetune", "epochs_finetune", "patience_finetune",
        "batch_a", "batch_b", "dropout_rate",
        "alpha_r2", "lambda_coral",
        "seed", "device"
    ]:
        if k in run_meta:
            print(f"{k:18s}: {run_meta[k]}")

    # —— 打印逐列指标（使用你给的 evaluate.py） ——
    calculate_and_print_metrics(
        y_pred_scaled,
        Y_val,
        y_scaler,
        output_cols=y_cols
    )

    # —— 返回 metrics（也在物理空间计算，用于落盘） ——
    # 反标准化 -> 物理空间，并对 log1p 列做 expm1
    y_pred_unstd = y_scaler.inverse_transform(y_pred_scaled)
    y_true_unstd = y_scaler.inverse_transform(Y_val)
    y_pred_phy = y_pred_unstd.copy()
    y_true_phy = y_true_unstd.copy()
    for j, col in enumerate(y_cols):
        if col in LOG_TRANSFORMED_COLS:
            y_pred_phy[:, j] = np.expm1(y_pred_unstd[:, j])
            y_true_phy[:, j] = np.expm1(y_true_unstd[:, j])

    mse_list, mae_list, r2_list = [], [], []
    for j in range(len(y_cols)):
        yt, yp = y_true_phy[:, j], y_pred_phy[:, j]
        mse_list.append(mean_squared_error(yt, yp))
        mae_list.append(mean_absolute_error(yt, yp))
        r2_list.append(r2_score(yt, yp))

    metrics = {}
    for j, name in enumerate(y_cols):
        metrics[f"mse_{name}"] = float(mse_list[j])
        metrics[f"mae_{name}"] = float(mae_list[j])
        metrics[f"r2_{name}"] = float(r2_list[j])
    metrics["mse_avg"] = float(np.mean(mse_list))
    metrics["mae_avg"] = float(np.mean(mae_list))
    metrics["r2_avg"] = float(np.mean(r2_list))
    return metrics


# ----------------------------
# 主流程
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="扫描 hidden_dim / num_layers，其他参数从 config 读取，评估 target 验证集")
    parser.add_argument("--opamp", type=str, required=True, help="电路类型，如 5t_opamp / two_stage_opamp")
    parser.add_argument("--hidden_dims", type=str, required=True, help="如 '128,256,512' 或 '[128,256,512]'")
    parser.add_argument("--num_layers", type=str, required=True, help="如 '2,3,4' 或 '[2,3,4]'")
    args = parser.parse_args()

    opamp = args.opamp
    if opamp not in TASK_CONFIGS:
        raise KeyError(f"未知 opamp: {opamp}. 可选: {list(TASK_CONFIGS.keys())}")

    hidden_dims = parse_int_list(args.hidden_dims)
    num_layers_list = parse_int_list(args.num_layers)

    # 读取 config
    cfg = TASK_CONFIGS[opamp]
    lr_pre = float(cfg.get("lr_pretrain", 3e-3))
    ep_pre = int(cfg.get("epochs_pretrain", 1000))
    pa_pre = int(cfg.get("patience_pretrain", 200))

    lr_ft = float(cfg.get("lr_finetune", 1e-3))
    ep_ft = int(cfg.get("epochs_finetune", 200))
    pa_ft = int(cfg.get("patience_finetune", 50))

    batch_a = int(cfg.get("batch_a", 128))
    batch_b = int(cfg.get("batch_b", 64))
    dropout = float(cfg.get("dropout_rate", 0.2))
    alpha_r2 = float(cfg.get("alpha_r2", 1.0))
    lambda_coral = float(cfg.get("lambda_coral", 0.1))

    # 随机种子（来自 COMMON_CONFIG，否则默认 42）
    seed = int(COMMON_CONFIG.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = build_device()

    # 数据与维度
    data = get_data_and_scalers(opamp_type=opamp)
    X_src, y_src = data["source"]
    in_dim, out_dim = X_src.shape[1], y_src.shape[1]

    loaders = {
        "pretrain_train": make_loader(data["source_train"][0], data["source_train"][1], batch_a, shuffle=True),
        "pretrain_val":   make_loader(data["source_val"][0],   data["source_val"][1],   batch_a, shuffle=False),
        "source":         make_loader(data["source"][0],       data["source"][1],       batch_a, shuffle=True, drop_last=True),
        "target_train":   make_loader(data["target_train"][0], data["target_train"][1], batch_b, shuffle=True),
        "target_val":     make_loader(data["target_val"][0],   data["target_val"][1],   batch_b, shuffle=False),
    }

    # 目录
    sweep_dir = SWEEP_ROOT / opamp
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print("=============================================")
    print(f"== SWEEP_B on {opamp}")
    print(f"== hidden_dims: {hidden_dims}")
    print(f"== num_layers : {num_layers_list}")
    print("== 其它参数全部来自 config；策略：对齐(align) 两阶段，不训练 target_only")
    print("=============================================\n")

    summary_rows = []

    # 网格扫描（遍历每种超参组合）
    for hd, nl in itertools.product(hidden_dims, num_layers_list):
        run_tag = f"HD{hd}_L{nl}"
        run_dir = sweep_dir / run_tag
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n===== [{run_tag}] 训练开始 =====")

        # 构建模型（仅这两个超参覆盖，其它按 config）
        model = AlignHeteroMLP(
            input_dim=in_dim,
            output_dim=out_dim,
            hidden_dim=hd,
            num_layers=nl,
            dropout_rate=dropout,
        ).to(device)

        # 阶段一：预训练
        pre_ckpt = run_dir / "pretrained.pth"
        run_pretraining(
            model,
            loaders["pretrain_train"], loaders["pretrain_val"],
            device, pre_ckpt,
            lr_pre, ep_pre, pa_pre,
        )

        # 明确载入最佳预训练权重
        model.load_state_dict(torch.load(pre_ckpt, map_location=device), strict=False)

        # 阶段二：对齐微调
        ft_ckpt = run_dir / "finetuned.pth"
        run_finetuning_align(
            model,
            {"source": loaders["source"], "target_train": loaders["target_train"], "target_val": loaders["target_val"]},
            device, ft_ckpt,
            lr_ft, ep_ft, pa_ft,
            alpha_r2, lambda_coral,
        )

        # 本次运行的 meta（会打印也会写文件）
        meta = {
            "opamp": opamp,
            "hidden_dim": hd,
            "num_layers": nl,
            "lr_pretrain": lr_pre,
            "epochs_pretrain": ep_pre,
            "patience_pretrain": pa_pre,
            "lr_finetune": lr_ft,
            "epochs_finetune": ep_ft,
            "patience_finetune": pa_ft,
            "batch_a": batch_a,
            "batch_b": batch_b,
            "dropout_rate": dropout,
            "alpha_r2": alpha_r2,
            "lambda_coral": lambda_coral,
            "seed": seed,
            "device": str(device),
        }

        # —— 评估：打印参数 + 每列指标，并拿到 metrics 用于保存 ——
        print(f"\n--- 评估 [{run_tag}] ---")
        metrics = eval_and_log(model, device, ft_ckpt, data, meta)

        # 保存 meta 与单次指标
        (run_dir / "ckpts").mkdir(exist_ok=True)
        # 备份权重
        state_pre = torch.load(pre_ckpt, map_location="cpu")
        torch.save(state_pre, run_dir / "ckpts" / "pretrained.pth")

        state_ft = torch.load(ft_ckpt, map_location="cpu")
        torch.save(state_ft, run_dir / "ckpts" / "finetuned.pth")
        with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        row = {"run_tag": run_tag, "hidden_dim": hd, "num_layers": nl, **metrics}
        pd.DataFrame([row]).to_csv(run_dir / "val_metrics.csv", index=False)

        summary_rows.append(row)
        print(f"[OK] {run_tag} -> r2_avg={metrics['r2_avg']:.4f}, "
              f"mae_avg={metrics['mae_avg']:.4g}, mse_avg={metrics['mse_avg']:.4g}")
        print(f"===== [{run_tag}] 训练完成 =====")

    # 汇总
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = sweep_dir / "sweep_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n[SUMMARY] 共 {len(summary_rows)} 次试验，已写入: {summary_path}")

        top_mae = summary_df.sort_values("mae_avg").head(3)
        top_r2 = summary_df.sort_values("r2_avg", ascending=False).head(3)

        print("\n=== Top-3 by MAE_avg (lower is better) ===")
        print(top_mae[["run_tag", "hidden_dim", "num_layers", "mae_avg", "r2_avg"]].to_string(index=False))

        print("\n=== Top-3 by R2_avg (higher is better) ===")
        print(top_r2[["run_tag", "hidden_dim", "num_layers", "r2_avg", "mae_avg"]].to_string(index=False))
    else:
        print("\n[SUMMARY] 没有可汇总的结果。请检查训练是否成功。")


if __name__ == "__main__":
    main()
