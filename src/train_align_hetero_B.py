#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# 项目内模块
from data_loader import get_data_and_scalers
from models.align_hetero import AlignHeteroMLP
from loss_function import batch_r2, coral_loss  # 用自带 coral；NLL 我们在本脚本内实现
from evaluate import calculate_and_print_metrics
from config import COMMON_CONFIG, TASK_CONFIGS, LOG_TRANSFORMED_COLS

# -------------------------
# 参数解析：完全沿用 unified_train 的风格（从 config 取）
# -------------------------
def setup_args():
    import argparse
    parser = argparse.ArgumentParser(description="B版：尾部采样 + Student-t + CORAL衰减 + logvar稳定化")

    # 先只解析出 opamp
    parser.add_argument("--opamp", type=str, required=True,
                        choices=TASK_CONFIGS.keys(), help="电路类型")
    tmp, _ = parser.parse_known_args()
    chosen = TASK_CONFIGS[tmp.opamp]

    # 追加 COMMON_CONFIG 的参数定义
    for key, value in COMMON_CONFIG.items():
        if isinstance(value, bool):
            if value is False:
                parser.add_argument(f"--{key}", action="store_true", help=f"启用 {key}")
            else:
                parser.add_argument(f"--no-{key}", dest=key, action="store_false", help=f"禁用 {key}")
        else:
            parser.add_argument(f"--{key}", type=type(value), help=f"设置 {key}")

    # 仅为“当前选择的 opamp”添加任务专属参数（避免访问其它任务没有的键）
    for key, sample_val in chosen.items():
        if key in COMMON_CONFIG:
            continue
        parser.add_argument(f"--{key}", type=type(sample_val), help=f"任务参数: {key}")

    # 默认值：COMMON + 当前任务
    parser.set_defaults(**COMMON_CONFIG)
    parser.set_defaults(**chosen)

    # 最终解析
    args = parser.parse_args()
    return args

# -------------------------
# DataLoader 便捷函数
# -------------------------
def _make_loader(x, y, bs, shuffle=True, drop_last=False, sampler=None):
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    if sampler is not None:
        return DataLoader(TensorDataset(x_t, y_t), batch_size=bs, sampler=sampler, drop_last=drop_last)
    return DataLoader(TensorDataset(x_t, y_t), batch_size=bs, shuffle=shuffle, drop_last=drop_last)


# -------------------------
# 尾部过采样（按目标列的分位打分）
# -------------------------
def _build_tail_sampler(y_array, y_cols, focus_cols: str, enabled: bool, gamma: float = 1.0):
    if not enabled:
        print("[TAIL-B] use_tail_sampler_B=False -> 不做尾部过采样")
        return None, None

    focus = [c.strip() for c in str(focus_cols).split(",") if c.strip()]
    df = None
    try:
        import pandas as pd
        df = pd.DataFrame(y_array, columns=y_cols)
    except Exception:
        print("[TAIL-B] 无法构造 DataFrame（可能缺列名），跳过尾部过采样")
        return None, None

    valid = [c for c in focus if c in df.columns]
    if not valid:
        print("[TAIL-B] 提供的 tail_focus_cols_B 未命中任何目标列，跳过尾部过采样")
        return None, None

    score = np.zeros(len(df), dtype=np.float32)
    for col in valid:
        score += df[col].rank(pct=True).values.astype(np.float32)
    score /= float(len(valid))
    score = np.power(score, max(1.0, float(gamma)))

    weights = 0.2 + 0.8 * score
    n = len(weights)
    top_pct = 5 if n >= 20 else max(1, int(100 / max(1, n)))
    thr = np.percentile(score, 100 - top_pct)
    hi = score >= thr
    lo = score < np.median(score)

    print("\n[TAIL-B] 过采样诊断")
    print(f"  focus_cols          : {valid}")
    print(f"  gamma               : {gamma}")
    print(f"  top {top_pct}% avg w : {float(weights[hi].mean()):.4f}")
    print(f"  <50% avg w          : {float(weights[lo].mean()):.4f}")
    print(f"  weight min/max      : {float(weights.min()):.4f} / {float(weights.max()):.4f}")

    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True
    )
    return sampler, weights


# -------------------------
# Student-t / Gaussian NLL（逐维版，便于加权）
# -------------------------
def _gauss_nll_perdim(mu, logv, y, eps=1e-8):
    var = (logv).exp().clamp_min(eps)
    return 0.5 * ((y - mu) ** 2 / var + logv)   # [B,D]

def _student_nll_perdim(mu, logv, y, nu=2.5, eps=1e-8):
    var = (logv).exp().clamp_min(eps)
    resid2 = (y - mu) ** 2 + eps
    term = torch.log1p(resid2 / (nu * var))
    return 0.5 * (nu + 1.0) * term + 0.5 * logv  # [B,D]


# -------------------------
# CORAL 衰减日程
# -------------------------
def _schedule_coral_weight(epoch_idx, total_epochs, base_lambda_coral,
                           use_decay, start_B, end_B, mode):
    if base_lambda_coral == 0.0:
        return 0.0
    if not use_decay:
        return float(base_lambda_coral)

    start_val, end_val = float(start_B), float(end_B)
    t, T = float(epoch_idx), max(1.0, float(total_epochs) - 1.0)
    if mode == "cosine":
        cos_t = (1 + math.cos(math.pi * min(max(t / T, 0.0), 1.0))) / 2.0
        return float(end_val + (start_val - end_val) * cos_t)
    return float(start_val + (end_val - start_val) * (min(max(t / T, 0.0), 1.0)))


# -------------------------
# 预训练（源域，仅 backbone）
# -------------------------
def run_pretraining(model, dl_tr, dl_va, device, save_path, args):
    print("\n--- [阶段一(B)] Backbone 预训练 @source ---")
    opt = torch.optim.AdamW(model.backbone.parameters(), lr=args.lr_pretrain)
    sch = CosineAnnealingWarmRestarts(opt, T_0=200, T_mult=1, eta_min=1e-6)
    crit = nn.HuberLoss(delta=1.0)

    best = float("inf"); counter = args.patience_pretrain
    for ep in range(args.epochs_pretrain):
        model.train(); tr=0.0
        for xb,yb in dl_tr:
            xb,yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            mu,_,_ = model(xb)
            loss = crit(mu,yb)
            loss.backward(); opt.step()
            tr += float(loss.item())
        tr /= max(1,len(dl_tr))

        model.eval(); va=0.0
        with torch.no_grad():
            for xb,yb in dl_va:
                xb,yb = xb.to(device), yb.to(device)
                mu,_,_ = model(xb)
                va += float(crit(mu,yb).item())
        va /= max(1,len(dl_va))
        sch.step()

        print(f"[Pretrain-B] {ep+1}/{args.epochs_pretrain}  TrainHuber={tr:.6f}  ValHuber={va:.6f}  lr={opt.param_groups[0]['lr']:.2e}")

        if va < best:
            best = va; counter = args.patience_pretrain
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ 保存预训练最佳 -> {save_path}")
        else:
            counter -= 1
            if counter == 0:
                print(f"  ⏹ 预训练早停（{args.patience_pretrain} 轮无提升）")
                break
    print("--- [阶段一(B)] 完成 ---")


# -------------------------
# 评估辅助：把标准化空间还原到物理域（与 evaluate 一致）
# -------------------------
def _inverse_to_physical(y_std: np.ndarray, y_scaler, y_cols):
    y_unstd = y_scaler.inverse_transform(y_std)  # 回到 log/线性混合域
    col2idx = {n:i for i,n in enumerate(y_cols)}
    for name in LOG_TRANSFORMED_COLS:
        if name in col2idx:
            j = col2idx[name]
            y_unstd[:, j] = np.expm1(y_unstd[:, j])
    return y_unstd


# -------------------------
# 微调（目标域；尾采样 + Student-t + CORAL 衰减 + 稳定化）
# -------------------------
def run_finetuning(model, data_loaders, device, final_save_path, args, y_cols):
    print("\n--- [阶段二(B)] 整体微调 @target ---")

    # 两个 param group：0->backbone, 1->head
    opt = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(),     "lr": args.lr_finetune / 10.0},
            {"params": model.hetero_head.parameters(),  "lr": args.lr_finetune},
        ],
        weight_decay=1e-4
    )

    dl_A = data_loaders.get("source")
    dl_B = data_loaders["target_train"]
    dl_V = data_loaders["target_val"]

    best_val = float("inf"); counter = args.patience_finetune

    # NLL 模式
    use_student = bool(args.use_student_t_B)
    nu = float(args.student_t_nu_B)

    # CORAL（基础值来自 config 的 lambda_coral）
    base_coral = float(args.lambda_coral)

    # 维度加权（专盯 cmrr / dc_gain）
    dim_w = torch.ones(len(y_cols), dtype=torch.float32, device=device)
    col2idx = {n:i for i,n in enumerate(y_cols)}
    if "cmrr" in col2idx:
        dim_w[col2idx["cmrr"]] = float(args.cmrr_loss_boost_B)
    if "dc_gain" in col2idx:
        dim_w[col2idx["dc_gain"]] = float(args.dcgain_loss_boost_B)

    print(f"[Finetune-B] 模式={'Student-t' if use_student else 'Gaussian'} (nu={nu if use_student else 'NA'})")
    print(f"[Finetune-B] base lambda_coral={base_coral}, use_coral_decay_B={args.use_coral_decay_B}")
    print(f"[Finetune-B] freeze_backbone_epochs_B={args.freeze_backbone_epochs_B}")
    print(f"[Finetune-B] loss weights: cmrr={float(dim_w[col2idx['cmrr']]) if 'cmrr' in col2idx else 'NA'}, "
          f"dc_gain={float(dim_w[col2idx['dc_gain']]) if 'dc_gain' in col2idx else 'NA'}")

    # 源域迭代器（供 CORAL 使用）
    if base_coral != 0.0 and dl_A is not None:
        dl_A_iter = iter(dl_A)

    for ep in range(args.epochs_finetune):
        model.train()

        # 冻结/解冻骨干
        if ep == 0 and args.freeze_backbone_epochs_B > 0:
            opt.param_groups[0]["lr_backup"] = opt.param_groups[0]["lr"]
            opt.param_groups[0]["lr"] = 0.0
        if ep == args.freeze_backbone_epochs_B and "lr_backup" in opt.param_groups[0]:
            opt.param_groups[0]["lr"] = opt.param_groups[0].pop("lr_backup")
        backbone_frozen = (ep < args.freeze_backbone_epochs_B)

        # CORAL 当前权重
        coral_now = _schedule_coral_weight(
            ep, args.epochs_finetune, base_coral,
            bool(args.use_coral_decay_B), args.lambda_coral_start_B, args.lambda_coral_end_B,
            str(args.lambda_coral_decay_mode_B)
        )

        for xb_B, yb_B in dl_B:
            xb_B, yb_B = xb_B.to(device), yb_B.to(device)

            # 目标域前向
            mu_B, logv_B, feat_B = model(xb_B)

            # ---- 稳定化：对 logvar 做钳位，避免“抬方差逃课” ----
            logv_B = logv_B.clamp(min=args.logvar_min, max=args.logvar_max)

            # 逐维 NLL + 维度加权
            if use_student:
                nll_dim = _student_nll_perdim(mu_B, logv_B, yb_B, nu=nu)  # [B,D]
            else:
                nll_dim = _gauss_nll_perdim(mu_B, logv_B, yb_B)          # [B,D]
            nll = (nll_dim * dim_w).mean()

            # R² 惩罚（温和）
            r2_term = (1.0 - batch_r2(yb_B, mu_B).clamp(-1.0, 1.0)).mean()

            # μ 的 Huber 锚点 + logvar L2 正则
            anchor = F.smooth_l1_loss(mu_B, yb_B, beta=1.0)
            logvar_reg = (logv_B ** 2).mean()

            loss = nll + args.alpha_r2 * r2_term + args.anchor_huber * anchor + args.logvar_reg * logvar_reg

            # CORAL（对齐源-目标的 backbone 特征）
            if coral_now > 0.0 and base_coral != 0.0 and dl_A is not None:
                try:
                    xa_A, _ = next(dl_A_iter)
                except StopIteration:
                    dl_A_iter = iter(dl_A)
                    xa_A, _ = next(dl_A_iter)
                if xa_A.size(0) != xb_B.size(0):
                    xa_A = xa_A[: xb_B.size(0)]
                xa_A = xa_A.to(device)
                with torch.no_grad():
                    _, _, feat_A = model(xa_A)
                c_loss = coral_loss(feat_A, feat_B)
                loss = loss + coral_now * c_loss

            # 反传
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # 验证 NLL（带维度加权；logvar 同样钳位）
        model.eval()
        val_nll = 0.0
        with torch.no_grad():
            for xb_V, yb_V in dl_V:
                xb_V, yb_V = xb_V.to(device), yb_V.to(device)
                mu_V, logv_V, _ = model(xb_V)
                logv_V = logv_V.clamp(min=args.logvar_min, max=args.logvar_max)
                if use_student:
                    vdim = _student_nll_perdim(mu_V, logv_V, yb_V, nu=nu)
                else:
                    vdim = _gauss_nll_perdim(mu_V, logv_V, yb_V)
                val_nll += float((vdim * dim_w).mean().item())
        val_nll /= max(1, len(dl_V))

        print(f"[Finetune-B] Epoch {ep+1}/{args.epochs_finetune} | ValNLL={val_nll:.6f} "
              f"| backbone_frozen={backbone_frozen} | coral_now={coral_now:.4g}")

        if val_nll < best_val:
            best_val = val_nll; counter = args.patience_finetune
            torch.save(model.state_dict(), final_save_path)
            print(f"  ✓ 保存微调最佳 -> {final_save_path}")
        else:
            counter -= 1
            if counter == 0:
                print(f"  ⏹ 微调早停（{args.patience_finetune} 轮无提升）")
                break

    print("--- [阶段二(B)] 完成 ---")


# -------------------------
# 统一评估：打印总表 + Focus(dc_gain/cmrr)
# -------------------------
def eval_and_print_B(model, dl_val, device, y_scaler, y_cols):
    model.eval()
    preds_std, trues_std = [], []
    with torch.no_grad():
        for xb, yb in dl_val:
            xb = xb.to(device)
            mu, _, _ = model(xb)
            preds_std.append(mu.cpu().numpy())
            trues_std.append(yb.numpy())
    preds_std = np.concatenate(preds_std, axis=0)
    trues_std = np.concatenate(trues_std, axis=0)

    print("\n[Evaluate-B] 统一评估(调用calculate_and_print_metrics进行物理域比较)")
    calculate_and_print_metrics(preds_std, trues_std, y_scaler)

    # Focus
    try:
        from sklearn.metrics import r2_score, mean_absolute_error
        y_pred_phys = _inverse_to_physical(preds_std, y_scaler, y_cols)
        y_true_phys = _inverse_to_physical(trues_std, y_scaler, y_cols)
        name_to_idx = {n:i for i,n in enumerate(y_cols)}
        for name in ["dc_gain", "cmrr"]:
            if name in name_to_idx:
                j = name_to_idx[name]
                r2 = r2_score(y_true_phys[:,j], y_pred_phys[:,j])
                mae = mean_absolute_error(y_true_phys[:,j], y_pred_phys[:,j])
                print(f"[Evaluate-B Focus] {name:<8s}  R2={r2:.4f}  MAE={mae:.6g}")
    except Exception as e:
        print(f"[Evaluate-B] Focus 打印失败：{e}")


# -------------------------
# 主函数
# -------------------------
def main():
    args = setup_args()
    DEVICE = torch.device(args.device)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    if args.opamp != "two_stage_opamp":
        print(f"[WARN] 本 B 脚本主要用于 two_stage_opamp；当前为 '{args.opamp}'。请确认 TASK_CONFIGS 已为该任务补齐 B 键。")

    os.makedirs(args.save_path, exist_ok=True)
    pretrained_path = os.path.join(args.save_path, f"{args.opamp}_pretrained.pth")
    finetuned_path  = os.path.join(args.save_path, f"{args.opamp}_finetuned.pth")

    # 取数据
    data = get_data_and_scalers(opamp_type=args.opamp)
    X_src, y_src = data["source"]
    Xs_tr, ys_tr = data["source_train"]; Xs_va, ys_va = data["source_val"]
    Xt_tr, yt_tr = data["target_train"]; Xt_va, yt_va = data["target_val"]

    # 目标列名（用于尾采样与 Focus 输出）
    if "raw_target" in data and isinstance(data["raw_target"], (list, tuple)):
        y_cols = list(data["raw_target"][1].columns)
    else:
        y_cols = [f"y{i}" for i in range(y_src.shape[1])]

    input_dim, output_dim = X_src.shape[1], y_src.shape[1]
    print(f"--- [B] 动态检测到 {args.opamp} 维度: Input={input_dim}, Output={output_dim} ---")
    print(f"[B] y_cols={y_cols}")
    print(f"[B] lambda_coral(from config)={args.lambda_coral}")

    # 模型
    model = AlignHeteroMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate
    ).to(DEVICE)

    # 阶段一：预训练
    if args.restart or not os.path.exists(pretrained_path):
        pre_tr = _make_loader(Xs_tr, ys_tr, args.batch_a, shuffle=True)
        pre_va = _make_loader(Xs_va, ys_va, args.batch_a, shuffle=False)
        run_pretraining(model, pre_tr, pre_va, DEVICE, pretrained_path, args)
    else:
        print(f"--- [阶段一(B)] 跳过预训练，加载已有权重: {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE))

    # 目标域 Loader（训练：可带尾部采样；验证：顺序）
    sampler_B, _ = _build_tail_sampler(
        yt_tr, y_cols,
        focus_cols=args.tail_focus_cols_B,
        enabled=bool(args.use_tail_sampler_B),
        gamma=float(args.tail_gamma_B)
    )
    dl_src_full = _make_loader(X_src, y_src, args.batch_a, shuffle=True, drop_last=True)
    dl_t_tr = _make_loader(Xt_tr, yt_tr, args.batch_b, shuffle=(sampler_B is None), sampler=sampler_B)
    dl_t_va = _make_loader(Xt_va, yt_va, args.batch_b, shuffle=False)

    loaders = {"source": dl_src_full, "target_train": dl_t_tr, "target_val": dl_t_va}

    # 阶段二：微调
    if os.path.exists(finetuned_path) and not args.restart:
        print(f"--- [阶段二(B)] 检测到已有微调模型: {finetuned_path}，跳过微调并直接载入该权重 ---")
        model.load_state_dict(torch.load(finetuned_path, map_location=DEVICE))
    else:
        run_finetuning(model, loaders, DEVICE, finetuned_path, args, y_cols)

    print("\n训练(B)流程全部完成。")

    # 评估
    if args.evaluate:
        print("\n--- [Evaluate-B] 启动 ---")
        if not os.path.exists(finetuned_path):
            print(f"错误：未找到已训练的模型文件 {finetuned_path}。跳过评估。")
            return
        model.load_state_dict(torch.load(finetuned_path, map_location=DEVICE))
        eval_and_print_B(model, dl_t_va, DEVICE, data["y_scaler"], y_cols)
        print("\n============================================================")
        print("=== PIPELINE FINISHED ===")
        print("============================================================")


if __name__ == "__main__":
    main()
