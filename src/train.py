# unified_train.py
import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import copy
import ast
import json

# --- 从项目模块中导入 ---
from data_loader import get_data_and_scalers
from models.align_hetero import AlignHeteroMLP
from loss_function import heteroscedastic_nll, batch_r2, coral_loss
from evaluate import calculate_and_print_metrics
import config

# ========== 1. 参数定义与解析 (重构) ==========


def setup_args():
    """设置和解析命令行参数，并将 config.py 中的设置作为默认值"""
    parser = argparse.ArgumentParser(description="统一的预训练与微调脚本")

    # --- 核心参数 ---
    parser.add_argument("--opamp", type=str,
                        default=config.OPAMP_TYPE, help="运放类型")
    parser.add_argument("--device", type=str,
                        default=config.DEVICE, help="设备 'cuda' or 'cpu'")
    parser.add_argument("--restart", action='store_true', help="强制重新执行预训练阶段")
    parser.add_argument("--save_path", type=str,
                        default="../results", help="预训练模型存放地址")
    parser.add_argument("--evaluate", action='store_true',
                        help="训练结束后，加载最佳模型并进行评估")
    parser.add_argument("--results_file", type=str, default=None,
                        help="如果提供，则将最终评估结果保存到此JSON文件")

    # --- 模型结构参数 ---
    parser.add_argument("--hidden_dims", type=str, default=str(config.HIDDEN_DIMS),
                        help="MLP隐藏层维度列表, e.g., '[256, 256, 256]'")
    parser.add_argument("--dropout_rate", type=float, default=config.DROPOUT_RATE,
                        help="Dropout比率")

    # --- 训练超参数 ---
    # <<< 将config中的参数全部移到这里，config的值作为默认值
    parser.add_argument("--lr_pretrain", type=float,
                        default=config.LEARNING_RATE_PRETRAIN, help="学习率")
    parser.add_argument("--lr_finetune", type=float,
                        default=config.LEARNING_RATE_FINETUNE, help="微调阶段 head 的学习率")
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


def make_loader(x, y, bs, shuffle=True, drop_last=False):
    """便捷函数，创建 DataLoader"""
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=shuffle, drop_last=drop_last)

# ========== 2. 阶段一：Backbone 预训练 ==========


def run_pretraining(model, train_loader, val_loader, device, args, scheduler_config):
    """在源域数据上仅预训练模型的backbone"""
    print(
        f"\n--- [子运行] 使用配置: T_0={scheduler_config['T_0']}, T_mult={scheduler_config['T_mult']} ---")

    optimizer = torch.optim.AdamW(
        model.backbone.parameters(), lr=args.lr_pretrain)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=scheduler_config['T_0'], T_mult=scheduler_config['T_mult'], eta_min=1e-6)
    criterion = torch.nn.HuberLoss(delta=1)
    best_val_loss_this_run = float('inf')

    patience = args.patience_pretrain
    patience_counter = patience  # 使用一个计数器

    best_state_dict_this_run = None  # <<< 在内存中保存最佳权重

    T_0 = scheduler_config['T_0']
    T_mult = scheduler_config['T_mult']
    current_T = T_0
    # 使用列表存储，因为我们需要最后一个元素来计算下一个重启点
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

    # 转换为集合以便在循环中快速查找
    restart_epochs = set(restart_epochs_list)
    print(f"优化器将在以下 epoch 结束后重置: {sorted(restart_epochs_list)}")

    for epoch in range(scheduler_config['epochs_pretrain']):
        # ... (循环内部的代码保持不变)
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


# ========== 3. 阶段二：整体模型微调 (重构) ==========

def run_finetuning(model, data_loaders, device, final_save_path, args):
    """使用复合损失对整个模型进行微调"""
    print("\n--- [阶段二] 开始整体模型微调 ---")

    # 为 backbone 和 head 设置不同的学习率
    optimizer_params = [
        {
            "params": model.backbone.parameters(),
            "lr": args.lr_finetune / 10  # 为 backbone 设置一个非常低的学习率
        },
        {
            "params": model.hetero_head.parameters(),
            "lr": args.lr_finetune  # 为新 head 设置一个相对较高的学习率
        }
    ]

    opt = torch.optim.AdamW(optimizer_params, weight_decay=1e-4)

    dl_A, dl_B, dl_val = data_loaders['source_full'], data_loaders['target_train'], data_loaders['target_val']
    dl_A_iter = iter(dl_A)

    best_val = float('inf')

    patience = args.patience_finetune
    patience_counter = patience  # 使用一个计数器

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
            patience_counter = patience  # 重置计数器
        else:
            patience_counter -= 1
            if patience_counter == 0:
                print(f"验证损失连续 {patience} 轮未改善，触发早停。")
                break

    print(f"--- [阶段二] 微调完成，最佳模型已保存至 {final_save_path} ---")


# ========== 4.（可选）获取最好模型的输出结果 ==========
def get_predictions(model, dataloader, device):
    """
    使用训练好的模型在数据集上进行预测，并返回Numpy数组。
    """
    model.eval()  # 确保模型处于评估模式
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            # 适配 AlignHeteroMLP 的输出，只取 mu
            preds, _, _ = model(inputs)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())  # labels 本身就在 CPU 上

    return np.concatenate(all_preds), np.concatenate(all_labels)

# ========== 5. 主函数 (已重构为“先选拔，后精调”模式) ==========


def main():
    args = setup_args()
    DEVICE = torch.device(args.device)
    print(f"程序在：{DEVICE}  上运行")
    os.makedirs(args.save_path, exist_ok=True)

    # --- 路径定义 ---
    pretrained_path = os.path.join(
        args.save_path, f'{args.opamp}_pretrained.pth')
    finetuned_path = os.path.join(
        args.save_path, f'{args.opamp}_finetuned.pth')

    try:
        hidden_dims_list = ast.literal_eval(args.hidden_dims)
        if not isinstance(hidden_dims_list, list):
            raise ValueError
    except (ValueError, SyntaxError):
        print(f"错误: --hidden_dims 参数格式不正确: {args.hidden_dims}")
        return

    # --- 数据准备 ---
    data = get_data_and_scalers(opamp_type=args.opamp)
    input_dim = data['source'][0].shape[1]
    output_dim = data['source'][1].shape[1]

    # --- 检查是否可以完全跳过 ---
    if os.path.exists(finetuned_path) and not args.restart:
        print(f"检测到最终模型 {finetuned_path} 且未指定 --restart。跳过所有训练。")
    else:
        # --- 阶段一：预训练元优化 ---
        global_best_pretrain_val_loss = float('inf')

        # 检查是否需要执行预训练
        if args.restart or not os.path.exists(pretrained_path):
            print(f"\n{'='*30} 阶段一: 预训练元优化 {'='*30}")

            pretrain_loader_A = make_loader(
                data['source_train'][0], data['source_train'][1], args.batch_a, shuffle=True)
            pretrain_loader_val = make_loader(
                data['source_val'][0], data['source_val'][1], args.batch_a, shuffle=False)

            num_experiments = config.RESTART_PRETRAIN
            for i in range(num_experiments):
                print(f"\n--- [预训练实验 {i+1}/{num_experiments}] ---")

                model = AlignHeteroMLP(
                    input_dim=input_dim, output_dim=output_dim,
                    hidden_dims=hidden_dims_list, dropout_rate=args.dropout_rate
                ).to(DEVICE)

                scheduler_config = config.PRETRAIN_SCHEDULER_CONFIGS[i % len(
                    config.PRETRAIN_SCHEDULER_CONFIGS)]

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

            print(
                f"\n--- [阶段一完成] 所有预训练实验结束。最佳源域验证损失: {global_best_pretrain_val_loss:.6f} ---")
        else:
            print(f"--- [阶段一] 检测到已存在的预训练模型 {pretrained_path}，跳过元优化。---")

        # --- 阶段二：对唯一的最佳预训练模型进行微调 ---
        if not os.path.exists(pretrained_path):
            print("\n[错误] 预训练阶段未能产生任何有效模型。无法进行微调。")
        else:
            print(f"\n{'='*30} 阶段二: 对最佳预训练模型进行微调 {'='*30}")
            model = AlignHeteroMLP(
                input_dim=input_dim, output_dim=output_dim,
                hidden_dims=hidden_dims_list, dropout_rate=args.dropout_rate
            ).to(DEVICE)

            print(f"加载最佳预训练权重从: {pretrained_path}")
            model.load_state_dict(torch.load(
                pretrained_path, map_location=DEVICE))

            finetune_loaders = {
                'source_full': make_loader(data['source'][0], data['source'][1], args.batch_a, shuffle=True, drop_last=True),
                'target_train': make_loader(data['target_train'][0], data['target_train'][1], args.batch_b, shuffle=True),
                'target_val': make_loader(data['target_val'][0], data['target_val'][1], args.batch_b, shuffle=False)
            }

            run_finetuning(model, finetune_loaders,
                           DEVICE, finetuned_path, args)

    # --- (可选) 最终评估 ---
    if args.evaluate:
        print("\n--- [最终评估流程启动] ---")
        if not os.path.exists(finetuned_path):
            print(f"错误：未找到最终微调模型 {finetuned_path}。无法评估。")
            return

        model = AlignHeteroMLP(
            input_dim=input_dim, output_dim=output_dim,
            hidden_dims=hidden_dims_list, dropout_rate=args.dropout_rate
        ).to(DEVICE)

        print(f"加载最终模型 {finetuned_path} 进行评估...")
        model.load_state_dict(torch.load(finetuned_path, map_location=DEVICE))

        # 准备评估所需的数据加载器
        eval_loader = make_loader(
            data['target_val'][0], data['target_val'][1], args.batch_b, shuffle=False)

        pred_scaled, true_scaled = get_predictions(model, eval_loader, DEVICE)
        eval_metrics = calculate_and_print_metrics(
            pred_scaled, true_scaled, data['y_scaler'])

        final_results = {
            'opamp': args.opamp,
            'evaluation_metrics_on_target_validation': eval_metrics
        }

        if args.results_file:
            try:
                # 使用 'a' 模式追加结果
                with open(args.results_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(final_results, indent=4) + "\n")
                print(f"评估结果已成功追加至: {args.results_file}")
            except Exception as e:
                print(f"错误: 保存结果文件失败 - {e}")


if __name__ == "__main__":
    main()
