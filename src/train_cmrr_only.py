# src/train.py (已更新：支持 --pretrain-only 模式)
import os
import torch
import torch.nn as nn
import argparse
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import copy
import ast
import json
from sklearn.preprocessing import StandardScaler

# --- 从项目模块中导入 ---
from data_loader import get_data_and_scalers
from models.align_hetero import AlignHeteroMLP
from loss_function import heteroscedastic_nll, batch_r2, coral_loss
from evaluate import calculate_and_print_metrics
import config

# ========== 1. 参数定义与解析 ==========


def setup_args():
    parser = argparse.ArgumentParser(description="[CMRR 专用] 统一的预训练与微调脚本")

    # --- 核心参数 ---
    parser.add_argument("--opamp", type=str,
                        default=config.OPAMP_TYPE, help="运放类型")
    parser.add_argument("--device", type=str,
                        default=config.DEVICE, help="设备 'cuda' or 'cpu'")
    parser.add_argument("--save_path", type=str,
                        default="../results", help="模型存放地址")
    parser.add_argument("--results_file", type=str,
                        default=None, help="如果提供，则将最终评估结果保存到此JSON文件")

    # --- 行为控制标志 (核心改动) ---
    parser.add_argument("--restart", action='store_true',
                        help="强制重新执行所有阶段 (删除所有缓存的模型)")
    parser.add_argument("--finetune", action='store_true',
                        help="强制重新执行微调阶段 (删除缓存的微调模型)")
    parser.add_argument("--evaluate", action='store_true',
                        help="评估最终模型。如果单独使用, 则只评估不训练。")
    parser.add_argument("--pretrain", action='store_true',
                        help="只运行预训练阶段并保存模型")

    # --- 模型结构参数 ---
    parser.add_argument("--hidden_dims", type=str, default=str(
        config.CMRR_HIDDEN_DIMS), help="MLP隐藏层维度列表, e.g., '[128, 128, 128]'")
    parser.add_argument("--dropout_rate", type=float,
                        default=config.CMRR_DROPOUT_RATE, help="Dropout比率")

    # --- 训练超参数 ---
    parser.add_argument("--lr_pretrain", type=float,
                        default=config.CMRR_LEARNING_RATE_PRETRAIN, help="学习率")
    parser.add_argument("--lr_hetero", type=float,
                        default=config.CMRR_LEARNING_RATE_HETERO,
                        help="微调阶段 Hetero Head 的学习率 (基础)")
    parser.add_argument("--backbone_lr_scale", type=float,
                        default=config.CMRR_BACKBONE_LR_SCALE,
                        help="Backbone 相对于 Hetero Head 的学习率比例 (例如 0.1)")
    parser.add_argument("--epochs_finetune", type=int,
                        default=config.EPOCHS_FINETUNE, help="微调阶段的总轮数")
    parser.add_argument("--batch_a", type=int,
                        default=config.BATCH_A, help="源域 Batch Size")
    parser.add_argument("--batch_b", type=int,
                        default=config.BATCH_B, help="目标域 Batch Size")
    parser.add_argument("--patience_pretrain", type=int,
                        default=config.PATIENCE_PRETRAIN, help="预训练早停的耐心轮数")
    parser.add_argument("--patience_finetune", type=int,
                        default=config.PATIENCE_FINETUNE, help="微调早停的耐心轮数")

    # --- 损失函数权重 ---
    parser.add_argument("--lambda_nll", type=float,
                        default=config.LAMBDA_NLL, help="NLL 损失的权重")
    parser.add_argument("--lambda_coral", type=float,
                        default=config.LAMBDA_CORAL, help="CORAL 损失的权重")
    parser.add_argument("--alpha_r2", type=float,
                        default=config.ALPHA_R2, help="R2 损失的权重")

    args = parser.parse_args()
    return args

# ========== 2. 辅助函数 (DataLoader, 预训练, 微调, 预测) ==========


def make_loader(x, y, bs, shuffle=True, drop_last=False):
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=shuffle, drop_last=drop_last)


def run_pretraining(model, train_loader, val_loader, device, args, scheduler_config):
    print(
        f"\n--- [子运行] 使用配置: T_0={scheduler_config['T_0']}, T_mult={scheduler_config['T_mult']} ---")
    optimizer = torch.optim.AdamW(
        model.backbone.parameters(), lr=args.lr_pretrain)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=scheduler_config['T_0'], T_mult=scheduler_config['T_mult'], eta_min=1e-6)
    criterion = torch.nn.HuberLoss(delta=1)
    best_val_loss_this_run = float('inf')
    patience = args.patience_pretrain
    patience_counter = patience
    best_state_dict_this_run = None
    T_0 = scheduler_config['T_0']
    T_mult = scheduler_config['T_mult']
    current_T = T_0
    restart_epochs_list = [current_T]
    max_cycles = int(np.log(scheduler_config['epochs_pretrain'] / T_0) / np.log(T_mult)
                     ) + 2 if T_mult > 1 else scheduler_config['epochs_pretrain'] // T_0
    for _ in range(max_cycles):
        current_T *= T_mult
        next_restart = restart_epochs_list[-1] + current_T
        if next_restart <= scheduler_config['epochs_pretrain']:
            restart_epochs_list.append(next_restart)
        else:
            break
    restart_epochs = set(restart_epochs_list)
    print(f"优化器将在以下 epoch 结束后重置: {sorted(restart_epochs_list)}")
    for epoch in range(scheduler_config['epochs_pretrain']):
        model.train()
        total_train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            mu, _, _ = model(inputs)
            loss = criterion(mu, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                mu, _, _ = model(inputs)
                loss = criterion(mu, labels)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Pre-train Epoch [{epoch+1}/{scheduler_config['epochs_pretrain']}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.2e}")
        if avg_val_loss < best_val_loss_this_run:
            best_val_loss_this_run = avg_val_loss
            best_state_dict_this_run = copy.deepcopy(model.state_dict())
            print(f"    - 新的本次运行最佳损失: {best_val_loss_this_run:.6f}")
        if (epoch + 1) in restart_epochs and (epoch + 1) < scheduler_config['epochs_pretrain']:
            print(f"--- Epoch {epoch+1} 是一个重启点。重置 AdamW 优化器状态！ ---")
            optimizer = torch.optim.AdamW(
                model.backbone.parameters(), lr=args.lr_pretrain)
            scheduler.optimizer = optimizer
    return best_val_loss_this_run, best_state_dict_this_run


def run_finetuning(model, data_loaders, device, final_save_path, args):
    # ... (函数体与您文件中的完全一致)
    print("\n--- [阶段二] 开始整体模型微调 ---")
    lr_hetero_base = args.lr_hetero
    lr_backbone = lr_hetero_base * args.backbone_lr_scale

    print(f"--- [Optimizer] 应用 \"十分之一\" (2-Group) 策略 ---")
    print(f"    - Hetero Head LR (G0): {lr_hetero_base:.2e}")
    print(
        f"    - Backbone LR (G1): {lr_backbone:.2e} (Base * {args.backbone_lr_scale})")

    optimizer_params = [
        {"params": model.hetero_head.parameters(), "lr": lr_hetero_base},
        {"params": model.backbone.parameters(), "lr": lr_backbone}
    ]
    opt = torch.optim.AdamW(optimizer_params, weight_decay=1e-4)

    dl_A, dl_B, dl_val = data_loaders['source_full'], data_loaders['target_train'], data_loaders['target_val']
    dl_A_iter = iter(dl_A)
    best_val = float('inf')
    patience = args.patience_finetune
    patience_counter = patience
    temp_save_path = final_save_path + ".tmp"
    for epoch in range(args.epochs_finetune):
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
            nll = heteroscedastic_nll(mu_B, logvar_B, yb_B)
            r2_loss = (
                1.0 - batch_r2(yb_B, mu_B).clamp(min=-1.0, max=1.0)).mean()
            coral = coral_loss(feat_A, feat_B)
            loss = args.lambda_nll * nll + args.alpha_r2 * \
                r2_loss + args.lambda_coral * coral
            total_train_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        avg_train_loss = total_train_loss / len(dl_B)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                mu, logvar, _ = model(xb)
                val_loss += heteroscedastic_nll(mu, logvar, yb).item()
        val_loss /= len(dl_val)
        print(
            f"Fine-tune Epoch [{epoch+1}/{args.epochs_finetune}], Train Loss: {avg_train_loss:.4f}, Val NLL: {val_loss:.4f}")
        if val_loss < best_val:
            print(f"  - 微调验证损失改善 ({best_val:.4f} -> {val_loss:.4f})。保存模型...")
            best_val = val_loss
            torch.save(model.state_dict(), temp_save_path)
            os.replace(temp_save_path, final_save_path)
            patience_counter = patience
        else:
            patience_counter -= 1
            if patience_counter == 0:
                print(f"验证损失连续 {patience} 轮未改善，触发早停。")
                break
    print(f"--- [阶段二] 微调完成，最佳模型已保存至 {final_save_path} ---")
    return best_val


def get_predictions(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            preds, _, _ = model(inputs)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)

# ========== 5. 主函数 ==========


def main():
    args = setup_args()
    DEVICE = torch.device(args.device)
    print(f"程序在：{DEVICE}  上运行")
    os.makedirs(args.save_path, exist_ok=True)

    pretrained_path = os.path.join(
        args.save_path, f'{args.opamp}_cmrr_pretrained.pth')
    finetuned_path = os.path.join(
        args.save_path, f'{args.opamp}_cmrr_finetuned.pth')

    try:
        hidden_dims_list = ast.literal_eval(args.hidden_dims)
        if not isinstance(hidden_dims_list, list):
            raise ValueError
    except (ValueError, SyntaxError):
        print(f"错误: --hidden_dims 参数格式不正确: {args.hidden_dims}")
        return

    data = get_data_and_scalers(opamp_type=args.opamp)
    try:
        y_cols = list(data["raw_target"][1].columns)
        cmrr_idx = y_cols.index('cmrr')
        print(f"检测到 'cmrr' 位于输出索引 {cmrr_idx}")
    except Exception as e:
        raise ValueError(f"无法在 Y 列中自动定位 'cmrr': {e}")
    output_dim = 1
    input_dim = data['source'][0].shape[1]
    # 示例：
    Xa_tr, Ya_tr_all = data['source_train']
    Ya_tr = Ya_tr_all[:, cmrr_idx:cmrr_idx+1]  # 保持 [N, 1] 维度

    Xa_va, Ya_va_all = data['source_val']
    Ya_va = Ya_va_all[:, cmrr_idx:cmrr_idx+1]

    Xs_s, Ys_s_all = data['source']
    Ys_s = Ys_s_all[:, cmrr_idx:cmrr_idx+1]

    Xb_tr, Yb_tr_all = data['target_train']
    Yb_tr = Yb_tr_all[:, cmrr_idx:cmrr_idx+1]

    Xb_va, Yb_va_all = data['target_val']
    Yb_va = Yb_va_all[:, cmrr_idx:cmrr_idx+1]

    best_finetune_nll = float('NaN')

    # --- 核心逻辑分支 ---

    # 1. 检查是否为“评估-A” (Evaluate-Only) 模式
    if args.evaluate and not args.restart and not args.finetune and not args.pretrain:
        print("--- [模式: 仅评估] ---")
    else:
        print("--- [模式: 训练] ---")

        if args.restart:
            print("--- --restart: 删除所有缓存的模型。---")
            if os.path.exists(pretrained_path):
                os.remove(pretrained_path)
            if os.path.exists(finetuned_path):
                os.remove(finetuned_path)
        elif args.finetune:
            print("--- --finetune: 删除缓存的微调模型。---")
            if os.path.exists(finetuned_path):
                os.remove(finetuned_path)

        # 2b. 阶段一：预训练
        if not os.path.exists(pretrained_path):
            print(f"\n{'='*30} 阶段一: 预训练元优化 {'='*30}")
            global_best_pretrain_val_loss = float('inf')

            pretrain_loader_A = make_loader(
                Xa_tr, Ya_tr, args.batch_a, shuffle=True)
            pretrain_loader_val = make_loader(
                Xa_va, Ya_va, args.batch_a, shuffle=False)

            num_experiments = config.CMRR_RESTART_PRETRAIN
            for i in range(num_experiments):
                print(f"\n--- [预训练实验 {i+1}/{num_experiments}] ---")

                model = AlignHeteroMLP(
                    input_dim=input_dim,
                    output_dim=output_dim,  # (output_dim=1)
                    hidden_dims=hidden_dims_list,
                    dropout_rate=args.dropout_rate
                ).to(DEVICE)

                scheduler_config = config.CMRR_PRETRAIN_SCHEDULER_CONFIGS[i % len(
                    config.CMRR_PRETRAIN_SCHEDULER_CONFIGS)]
                best_loss_this_run, best_state_dict_this_run = run_pretraining(
                    model, pretrain_loader_A, pretrain_loader_val, DEVICE, args, scheduler_config)

                if best_state_dict_this_run and best_loss_this_run < global_best_pretrain_val_loss:
                    global_best_pretrain_val_loss = best_loss_this_run
                    print(
                        f"  🏆 新的全局最佳预训练模型！源域验证损失: {global_best_pretrain_val_loss:.6f}。保存中...")
                    torch.save(best_state_dict_this_run, pretrained_path)
                del model, best_state_dict_this_run
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print(f"\n--- [阶段一完成] 所有预训练实验结束。---")
        else:
            print("--- [阶段一] 跳过预训练 (文件已存在) ---")

        if args.pretrain:
            print("--- [模式: 仅预训练] 完成。正在退出。 ---")
            return

        # 2c. 阶段二：微调
        if not os.path.exists(finetuned_path):
            if not os.path.exists(pretrained_path):
                print("\n[错误] 预训练阶段未能产生任何有效模型。无法进行微调。")
            else:
                print(f"\n{'='*30} 阶段二: 对最佳预训练模型进行微调 {'='*30}")

                model = AlignHeteroMLP(
                    input_dim=input_dim,
                    output_dim=output_dim,  # (output_dim=1)
                    hidden_dims=hidden_dims_list,
                    dropout_rate=args.dropout_rate
                ).to(DEVICE)

                print(f"加载最佳预训练权重从: {pretrained_path}")
                model.load_state_dict(torch.load(
                    pretrained_path, map_location=DEVICE))

                finetune_loaders = {
                    'source_full': make_loader(Xs_s, Ys_s, args.batch_a, shuffle=True, drop_last=True),
                    'target_train': make_loader(Xb_tr, Yb_tr, args.batch_b, shuffle=True),
                    'target_val': make_loader(Xb_va, Yb_va, args.batch_b, shuffle=False)
                }

                best_finetune_nll = run_finetuning(
                    model, finetune_loaders, DEVICE, finetuned_path, args)
        else:
            print("--- [阶段二] 跳过微调 (文件已存在) ---")

    # --- 3. (可选) 最终评估 ---
    if args.evaluate:
        print("\n--- [最终评估流程启动] ---")
        if not os.path.exists(finetuned_path):
            print(f"错误：未找到最终微调模型 {finetuned_path}。无法评估。")
            return

        model = AlignHeteroMLP(
            input_dim=input_dim,
            output_dim=output_dim,  # (output_dim=1)
            hidden_dims=hidden_dims_list,
            dropout_rate=args.dropout_rate
        ).to(DEVICE)

        print(f"加载最终模型 {finetuned_path} 进行评估...")
        model.load_state_dict(torch.load(finetuned_path, map_location=DEVICE))

        eval_loader = make_loader(
            Xb_va, Yb_va, args.batch_b, shuffle=False)

        pred_scaled, true_scaled = get_predictions(
            model, eval_loader, DEVICE)  # Shape [N, 1]

        print("为 'cmrr' (1D) 评估创建专用 scaler...")
        y_scaler_5d = data['y_scaler']

        # 1. 创建一个新的 1D scaler
        y_scaler_1d = StandardScaler()

        # 2. 从 5D scaler 复制 cmrr 的均值和方差
        y_scaler_1d.mean_ = np.array([y_scaler_5d.mean_[cmrr_idx]])
        y_scaler_1d.scale_ = np.array([y_scaler_5d.scale_[cmrr_idx]])
        y_scaler_1d.n_features_in_ = 1

        # 3. 调用评估函数，传入 1D scaler 和 1D 列名
        eval_metrics = calculate_and_print_metrics(
            pred_scaled,
            true_scaled,
            y_scaler_1d,
            output_cols=['cmrr']
        )

        final_results = {
            'opamp': args.opamp,
            'best_finetune_val_nll': best_finetune_nll,
            'evaluation_metrics': eval_metrics
        }
        if args.results_file:
            try:
                with open(args.results_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(final_results) + "\n")
                print(f"评估结果已成功追加至: {args.results_file}")
            except Exception as e:
                print(f"错误: 保存结果文件失败 - {e}")


if __name__ == "__main__":
    main()
