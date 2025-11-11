#!/usr/bin/env python3
# train_align_hetero.py (two_stage-only: warmup+cosine & dc_gain boost; safe defaults for others)

import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import copy
import ast
import json

from data_loader import get_data_and_scalers
from models.align_hetero import AlignHeteroMLP
from loss_function import batch_r2, coral_loss
from evaluate import calculate_and_print_metrics
from config import COMMON_CONFIG, TASK_CONFIGS


# ========== 1) 参数解析 ==========

def setup_args():
    parser = argparse.ArgumentParser(description="统一的多任务训练脚本")
    parser.add_argument("--opamp", type=str, required=True,
                        choices=TASK_CONFIGS.keys(), help="必须指定的电路类型")

    # 先解析 opamp
    tmp_args, _ = parser.parse_known_args()
    opamp_type = tmp_args.opamp

    # 合并通用 & 任务配置作为默认
    defaults = {**COMMON_CONFIG, **TASK_CONFIGS.get(opamp_type, {})}

    # 自动为简单类型添加命令行开关
    for key, value in defaults.items():
        if isinstance(value, (list, dict)):
            continue
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", action=argparse.BooleanOptionalAction,
                                help=f"开关 '{key}'")
        else:
            parser.add_argument(f"--{key}", type=type(value),
                                help=f"设置 '{key}'")

    # 复杂类型（可覆盖）
    parser.set_defaults(**defaults)
    parser.add_argument("--hidden_dims", type=str,
                        help="MLP隐藏层维度列表, e.g., '[256, 256, 256]'")

    # 统一提供两个开关（若 config 未提供，也能用）：
    parser.add_argument("--enable_warmup_cosine", action=argparse.BooleanOptionalAction,
                        help="是否启用 finetune 的 warmup+cosine 调度")
    parser.add_argument("--enable_dcgain_boost", action=argparse.BooleanOptionalAction,
                        help="是否对 dc_gain 维度加权提升")

    # 评估参数（如未在 defaults 提供）
    if 'evaluate' not in defaults:
        parser.add_argument("--evaluate", action='store_true',
                            help="训练结束后评估目标域表现")
    if 'results_file' not in defaults:
        parser.add_argument("--results_file", type=str, default=None,
                            help="将评估结果追加保存到此 JSON 文件")

    args = parser.parse_args()
    return args


# ========== 2) 工具函数 ==========

def make_loader(x, y, bs, shuffle=True, drop_last=False):
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(x, y), batch_size=bs,
                      shuffle=shuffle, drop_last=drop_last)


def run_pretraining(model, train_loader, val_loader, device, args, scheduler_config):
    print(f"\n--- [子运行] 使用配置: {scheduler_config} ---")
    optimizer = torch.optim.AdamW(model.backbone.parameters(), lr=args.lr_pretrain)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=scheduler_config['T_0'],
        T_mult=scheduler_config['T_mult'],
        eta_min=1e-6
    )
    criterion = torch.nn.HuberLoss(delta=1.0)

    best_val_loss_this_run = float('inf')
    best_state_dict_this_run = None
    epochs = scheduler_config['epochs_pretrain']

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            mu, _, _ = model(xb)
            loss = criterion(mu, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                mu, _, _ = model(xb)
                loss = criterion(mu, yb)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Pre-train Epoch [{epoch+1}/{epochs}], "
            f"Train Loss: {avg_train_loss:.6f}, "
            f"Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.2e}"
        )

        if avg_val_loss < best_val_loss_this_run:
            best_val_loss_this_run = avg_val_loss
            best_state_dict_this_run = copy.deepcopy(model.state_dict())
            print(f"    - 本次运行最佳损失更新: {best_val_loss_this_run:.6f}")

    return best_val_loss_this_run, best_state_dict_this_run


def _weighted_hetero_nll(mu, logvar, y, dim_weights):
    """
    可加权的 heteroscedastic NLL（高斯，忽略常数项）
    mu, logvar, y: [B, D]
    dim_weights: [D]
    """
    inv_var = torch.exp(-logvar)
    nll_dim = 0.5 * (logvar + (y - mu) ** 2 * inv_var)  # [B, D]
    return (nll_dim * dim_weights.view(1, -1)).mean()


# ========== 3) 微调（含 two_stage 专属开关） ==========

def run_finetuning(model, data_loaders, device, final_save_path, args, dim_weights):
    """
    - 若 enable_warmup_cosine=True：使用线性 warmup + cosine 衰减
    - 否则：恒定学习率
    - CORAL 对齐 & 可选的 per-dim NLL 加权
    """
    print("\n--- [阶段二] 开始整体模型微调 ---")

    # 解析开关（默认：仅 two_stage 启用）
    enable_warmup_cosine = getattr(args, "enable_warmup_cosine",
                                   args.opamp == "two_stage_opamp")
    # 学习率下限比例 / warmup 比例：提供安全默认
    finetune_warmup_ratio = float(getattr(args, "finetune_warmup_ratio", 0.1 if enable_warmup_cosine else 0.0))
    finetune_min_lr_factor = float(getattr(args, "finetune_min_lr_factor", 0.1))

    # 两个 param group：backbone / hetero_head
    base_backbone_lr = args.lr_finetune / 10.0
    base_head_lr = args.lr_finetune

    opt = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": base_backbone_lr},
            {"params": model.hetero_head.parameters(), "lr": base_head_lr},
        ],
        weight_decay=1e-4,
    )

    dl_A = data_loaders['source']
    dl_B = data_loaders['target_train']
    dl_val = data_loaders['target_val']

    dl_A_iter = iter(dl_A)

    total_epochs = args.epochs_finetune
    warmup_epochs = max(1, int(finetune_warmup_ratio * total_epochs)) if enable_warmup_cosine else 0

    def set_lr(epoch_idx: int):
        # 不启用 warmup+cosine：恒定 LR
        if not enable_warmup_cosine or total_epochs <= 0:
            opt.param_groups[0]['lr'] = base_backbone_lr
            opt.param_groups[1]['lr'] = base_head_lr
            return base_head_lr

        # 启用 warmup+cosine
        if epoch_idx < warmup_epochs:
            lr_mult = float(epoch_idx + 1) / float(warmup_epochs)
        else:
            if total_epochs == warmup_epochs:
                lr_mult = 1.0
            else:
                t = (epoch_idx + 1 - warmup_epochs) / float(total_epochs - warmup_epochs)
                lr_mult = finetune_min_lr_factor + 0.5 * (1.0 - finetune_min_lr_factor) * (1.0 + np.cos(np.pi * t))

        opt.param_groups[0]['lr'] = base_backbone_lr * lr_mult
        opt.param_groups[1]['lr'] = base_head_lr * lr_mult
        return opt.param_groups[1]['lr']

    best_val = float('inf')
    patience_counter = args.patience_finetune
    temp_save_path = final_save_path + ".tmp"

    for epoch in range(total_epochs):
        current_lr = set_lr(epoch)

        model.train()
        total_train_loss = 0.0

        for xb_B, yb_B in dl_B:
            xb_B, yb_B = xb_B.to(device), yb_B.to(device)

            try:
                xa_A, _ = next(dl_A_iter)
            except StopIteration:
                dl_A_iter = iter(dl_A)
                xa_A, _ = next(dl_A_iter)
            if xa_A.size(0) != xb_B.size(0):
                xa_A = xa_A[:xb_B.size(0)]
            xa_A = xa_A.to(device)

            mu_B, logvar_B, feat_B = model(xb_B)
            with torch.no_grad():
                _, _, feat_A = model(xa_A)

            nll = _weighted_hetero_nll(mu_B, logvar_B, yb_B, dim_weights)
            r2_loss = (1.0 - batch_r2(yb_B, mu_B).clamp(min=-1.0, max=1.0)).mean()
            coral = coral_loss(feat_A, feat_B)

            loss = nll + args.alpha_r2 * r2_loss + args.lambda_coral * coral

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(dl_B)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                mu, logvar, _ = model(xb)
                val_loss += _weighted_hetero_nll(mu, logvar, yb, dim_weights).item()
        val_loss /= len(dl_val)

        print(
            f"Fine-tune Epoch [{epoch+1}/{total_epochs}], "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val NLL: {val_loss:.4f}, "
            f"LR_head: {current_lr:.2e}"
        )

        if val_loss < best_val:
            print(f"  - 微调验证损失改善 ({best_val:.4f} -> {val_loss:.4f})，保存模型...")
            best_val = val_loss
            torch.save(model.state_dict(), temp_save_path)
            os.replace(temp_save_path, final_save_path)
            patience_counter = args.patience_finetune
        else:
            patience_counter -= 1
            if patience_counter == 0:
                print(f"验证损失连续 {args.patience_finetune} 轮未改善，触发早停。")
                break

    print(f"--- [阶段二] 微调完成，最佳模型已保存至 {final_save_path} ---")
    return best_val


def get_predictions(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            mu, _, _ = model(xb)
            all_preds.append(mu.cpu().numpy())
            all_labels.append(yb.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


# ========== 4) 主流程 ==========

def main():
    args = setup_args()

    device = torch.device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.save_path, exist_ok=True)

    print(f"--- 任务启动: {args.opamp} | 设备: {device} ---")

    pretrained_path = os.path.join(args.save_path, f'{args.opamp}_pretrained.pth')
    finetuned_path = os.path.join(args.save_path, f'{args.opamp}_finetuned.pth')

    data = get_data_and_scalers(opamp_type=args.opamp)
    input_dim = data['source'][0].shape[1]
    output_dim = data['source'][1].shape[1]

    # hidden_dims 解析（list 或字符串）
    if isinstance(args.hidden_dims, str):
        hidden_dims_list = ast.literal_eval(args.hidden_dims)
    else:
        hidden_dims_list = args.hidden_dims

    # === 阶段一：预训练（多次重启取最佳） ===
    global_best_val_loss = float('inf')

    if args.restart or not os.path.exists(pretrained_path):
        dl_src_train = make_loader(
            data['source_train'][0], data['source_train'][1],
            args.batch_a, shuffle=True
        )
        dl_src_val = make_loader(
            data['source_val'][0], data['source_val'][1],
            args.batch_a, shuffle=False
        )

        sched_cfgs = args.PRETRAIN_SCHEDULER_CONFIGS
        print(f"--- [阶段一] 启动预训练流程 ({len(sched_cfgs)} 次独立实验) ---")

        for i, sched in enumerate(sched_cfgs):
            print(f"\n{'='*30} 独立实验 {i+1}/{len(sched_cfgs)} {'='*30}")
            model = AlignHeteroMLP(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dims_list,
                dropout_rate=args.dropout_rate
            ).to(device)

            best_run_loss, best_state = run_pretraining(
                model, dl_src_train, dl_src_val, device, args, sched
            )

            if best_state is not None and best_run_loss < global_best_val_loss:
                global_best_val_loss = best_run_loss
                print(f"[BEST] 新的全局最佳损失 {global_best_val_loss:.6f}，保存至 {pretrained_path}")
                torch.save(best_state, pretrained_path)

            del model, best_state
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"\n所有独立实验完成，最终最佳预训练模型损失: {global_best_val_loss:.6f}")
    else:
        print(f"跳过预训练，使用已有 {pretrained_path}")

    # === 加载预训练权重 ===
    print(f"--- [阶段一完成] 加载最佳预训练模型: {pretrained_path} ---")
    model = AlignHeteroMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims_list,
        dropout_rate=args.dropout_rate
    ).to(device)

    if os.path.exists(pretrained_path):
        state = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state, strict=False)
    else:
        print(f"警告：未找到预训练模型 {pretrained_path}，将从随机初始化开始微调。")

    # === 阶段二：微调 ===
    finetune_loaders = {
        'source': make_loader(
            data['source'][0], data['source'][1],
            args.batch_a, shuffle=True, drop_last=True
        ),
        'target_train': make_loader(
            data['target_train'][0], data['target_train'][1],
            args.batch_b, shuffle=True
        ),
        'target_val': make_loader(
            data['target_val'][0], data['target_val'][1],
            args.batch_b, shuffle=False
        ),
    }

    # 只有 two_stage 才默认启用 dc_gain boost；其它模型全 1.0
    enable_dcgain_boost = getattr(args, "enable_dcgain_boost",
                                  args.opamp == "two_stage_opamp")
    if args.opamp == "two_stage_opamp" and enable_dcgain_boost:
        dcw = float(getattr(args, "dcgain_loss_weight", 2.0))
        cmw = float(getattr(args, "cmrr_loss_weight", 1.0))
        dim_weights = torch.tensor([1.0, dcw, 1.0, 1.0, cmw], device=device)
        print(f"[Finetune] two_stage: 启用 dc_gain 加权 (dc:{dcw}, cmrr:{cmw})")
    else:
        dim_weights = torch.ones(output_dim, device=device)
        print(f"[Finetune] {args.opamp}: 使用等权重 NLL。")

    if os.path.exists(finetuned_path) and not args.restart:
        print(f"检测到已存在微调模型 {finetuned_path}，直接加载用于评估/推理。")
        model.load_state_dict(torch.load(finetuned_path, map_location=device))
        best_finetune_nll = float('nan')
    else:
        best_finetune_nll = run_finetuning(
            model, finetune_loaders, device, finetuned_path, args, dim_weights
        )

    # === 可选评估 ===
    if args.evaluate:
        print("\n--- [评估流程启动] ---")
        if not os.path.exists(finetuned_path):
            print(f"错误：未找到已训练的模型文件 {finetuned_path}，跳过评估。")
            return

        model.load_state_dict(torch.load(finetuned_path, map_location=device))

        preds_scaled, true_scaled = get_predictions(
            model, finetune_loaders['target_val'], device
        )
        eval_metrics = calculate_and_print_metrics(
            preds_scaled, true_scaled, data['y_scaler']
        )

        final_results = {
            'opamp': args.opamp,
            'best_finetune_val_nll': best_finetune_nll,
            'evaluation_metrics': eval_metrics,
        }

        if args.results_file:
            try:
                with open(args.results_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(final_results, indent=4) + "\n")
                print(f"评估结果已追加保存到 {args.results_file}")
            except Exception as e:
                print(f"保存结果文件失败: {e}")


if __name__ == "__main__":
    main()
