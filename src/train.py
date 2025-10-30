# unified_train.py
import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import copy
import ast

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

# ========== 5. 主函数 ==========


def main():
    args = setup_args()
    DEVICE = torch.device(args.device)
    print(f"程序在：{DEVICE}  上运行")
    os.makedirs(args.save_path, exist_ok=True)

    finetuned_path = os.path.join(
        args.save_path, f'{args.opamp}_finetuned.pth')

    try:
        hidden_dims_list = ast.literal_eval(args.hidden_dims)
        if not isinstance(hidden_dims_list, list):
            raise ValueError
    except (ValueError, SyntaxError):
        print(f"错误: --hidden_dims 参数格式不正确: {args.hidden_dims}")
        return

    # --- 数据准备 (只需一次) ---
    data = get_data_and_scalers(opamp_type=args.opamp)
    input_dim = data['source'][0].shape[1]
    output_dim = data['source'][1].shape[1]

    pretrain_loader_A = make_loader(
        data['source_train'][0], data['source_train'][1], args.batch_a, shuffle=True)
    pretrain_loader_val = make_loader(
        data['source_val'][0], data['source_val'][1], args.batch_a, shuffle=False)

    finetune_loaders = {
        'source_full': make_loader(data['source'][0], data['source'][1], args.batch_a, shuffle=True, drop_last=True),
        'target_train': make_loader(data['target_train'][0], data['target_train'][1], args.batch_b, shuffle=True),
        'target_val': make_loader(data['target_val'][0], data['target_val'][1], args.batch_b, shuffle=False)
    }

    # --- 检查是否需要跳过 ---
    if os.path.exists(finetuned_path) and not args.restart:
        print(f"检测到最终模型 {finetuned_path} 已存在且未指定 --restart。跳过所有训练。")
        if args.evaluate:
            model = AlignHeteroMLP(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dims_list,
                dropout_rate=args.dropout_rate
            ).to(DEVICE)
            model.load_state_dict(torch.load(
                finetuned_path, map_location=DEVICE))
            pred_scaled, true_scaled = get_predictions(
                model, finetune_loaders['target_val'], DEVICE)
            calculate_and_print_metrics(
                pred_scaled, true_scaled, data['y_scaler'])
        return

    # --- 全局最优追踪器 ---
    global_best_finetune_val_nll = float('inf')
    global_best_finetune_state_dict = None

    # --- 主循环：遍历每个预训练配置，执行完整的 "预训练->微调" 流水线 ---
    num_pipelines = config.RESTART_PRETRAIN
    for i in range(num_pipelines):
        print(f"\n{'='*30} 完整流水线 {i+1}/{num_pipelines} {'='*30}")

        # 1. 每次都创建新模型，保证隔离
        model = AlignHeteroMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims_list,
            dropout_rate=args.dropout_rate
        ).to(DEVICE)

        # 2. 选择预训练配置并执行
        scheduler_config = config.PRETRAIN_SCHEDULER_CONFIGS[i % len(
            config.PRETRAIN_SCHEDULER_CONFIGS)]
        _, best_pretrained_state = run_pretraining(
            model, pretrain_loader_A, pretrain_loader_val, DEVICE, args, scheduler_config)

        if not best_pretrained_state:
            print("  [警告] 本次预训练未产生有效模型，跳过此流水线。")
            continue

        # 3. 加载最佳预训练权重，准备微调
        print("\n--- [加载预训练模型] ---")
        model.load_state_dict(best_pretrained_state)

        # 4. 执行微调，并保存到临时文件
        temp_finetune_path = os.path.join(
            args.save_path, f"{args.opamp}_finetune_temp_run_{i+1}.pth")
        run_finetuning(model, finetune_loaders, DEVICE,
                       temp_finetune_path, args)

        # 5. 评估本次微调结果，并与全局最优比较
        if os.path.exists(temp_finetune_path):
            model.load_state_dict(torch.load(
                temp_finetune_path, map_location=DEVICE))

            model.eval()
            current_pipeline_val_nll = 0.0
            with torch.no_grad():
                for xb, yb in finetune_loaders['target_val']:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    mu, logvar, _ = model(xb)
                    current_pipeline_val_nll += heteroscedastic_nll(
                        mu, logvar, yb).item()
            current_pipeline_val_nll /= len(finetune_loaders['target_val'])

            print(
                f"\n[流水线 {i+1} 总结] 最终微调验证集 NLL = {current_pipeline_val_nll:.6f}")

            # 6. 如果更优，则更新全局最佳模型
            if current_pipeline_val_nll < global_best_finetune_val_nll:
                global_best_finetune_val_nll = current_pipeline_val_nll
                global_best_finetune_state_dict = copy.deepcopy(
                    model.state_dict())
                print(
                    f"  🏆🏆🏆 新的全局最佳模型诞生！ Val NLL 更新为: {global_best_finetune_val_nll:.6f} 🏆🏆🏆")

            os.remove(temp_finetune_path)

        # 7. 清理内存
        del model, best_pretrained_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- 所有流水线结束后 ---
    if global_best_finetune_state_dict:
        print(f"\n{'='*30} 所有流水线执行完毕 {'='*30}")
        print(f"全局最优模型的微调验证 NLL 为: {global_best_finetune_val_nll:.6f}")
        print(f"正在保存最终模型至: {finetuned_path}")
        torch.save(global_best_finetune_state_dict, finetuned_path)
    else:
        print("\n[错误] 所有流水线均未成功生成模型，未保存任何最终模型。")
        return

    # --- (可选) 评估最终选出的模型 ---
    if args.evaluate:
        print("\n--- [最终评估流程启动] ---")
        model = AlignHeteroMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims_list,
            dropout_rate=args.dropout_rate
        ).to(DEVICE)
        model.load_state_dict(torch.load(finetuned_path, map_location=DEVICE))
        pred_scaled, true_scaled = get_predictions(
            model, finetune_loaders['target_val'], DEVICE)
        calculate_and_print_metrics(pred_scaled, true_scaled, data['y_scaler'])


if __name__ == "__main__":
    main()
