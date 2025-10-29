# unified_train.py (版本 4.0 - 配置驱动)
import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import copy

# --- 从项目模块中导入 ---
from data_loader import get_data_and_scalers
from models.align_hetero import AlignHeteroMLP
from loss_function import heteroscedastic_nll, batch_r2, coral_loss
from evaluate import calculate_and_print_metrics
from config import COMMON_CONFIG, TASK_CONFIGS  # <<< 导入新的 config 结构

# ========== 1. 参数定义与解析 (已简化) ==========


def setup_args():
    """
    一个更简洁的参数解析器，能动态地从 config 文件加载特定任务的默认值。
    """
    parser = argparse.ArgumentParser(description="统一的多任务训练脚本")
    parser.add_argument("--opamp", type=str, required=True,
                        choices=TASK_CONFIGS.keys(), help="必须指定的电路类型")

    # Step 1: 先解析出 opamp 类型，以便加载正确的默认值
    temp_args, other_args = parser.parse_known_args()
    opamp_type = temp_args.opamp

    # Step 2: 合并通用配置和特定任务的配置作为默认值
    defaults = {**COMMON_CONFIG, **TASK_CONFIGS.get(opamp_type, {})}

    # Step 3: 动态为所有简单类型的默认参数创建命令行开关
    for key, value in defaults.items():
        if isinstance(value, (list, dict)):
            continue  # 跳过复杂类型
        if isinstance(value, bool):
            # 为布尔值创建 --key / --no-key 形式的开关
            parser.add_argument(
                f"--{key}", action=argparse.BooleanOptionalAction, help=f"开关 '{key}'")
        else:
            parser.add_argument(
                f"--{key}", type=type(value), help=f"设置 '{key}'")

    # Step 4: 将合并后的配置设置为解析器的默认值并进行最终解析
    parser.set_defaults(**defaults)
    args = parser.parse_args()
    return args


# ========== 2. 数据加载器创建 ==========
def make_loader(x, y, bs, shuffle=True, drop_last=False):
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=shuffle, drop_last=drop_last)


# ========== 3. 预训练函数 ==========
def run_pretraining(model, train_loader, val_loader, device, args, scheduler_config):
    # 此函数逻辑已足够通用，无需修改
    print(f"\n--- [子运行] 使用配置: {scheduler_config} ---")
    optimizer = torch.optim.AdamW(
        model.backbone.parameters(), lr=args.lr_pretrain)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=scheduler_config['T_0'], T_mult=scheduler_config['T_mult'], eta_min=1e-6)
    criterion = torch.nn.HuberLoss(delta=1.0)
    best_val_loss_this_run = float('inf')
    best_state_dict_this_run = None
    epochs = scheduler_config['epochs_pretrain']
    # ... (内部逻辑保持不变)
    for epoch in range(epochs):
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
            f"Pre-train Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.2e}")

        if avg_val_loss < best_val_loss_this_run:
            best_val_loss_this_run = avg_val_loss
            best_state_dict_this_run = copy.deepcopy(model.state_dict())
            print(f"    - 本次运行最佳损失更新: {best_val_loss_this_run:.6f}")
    return best_val_loss_this_run, best_state_dict_this_run


# ========== 4. 微调函数 ==========
def run_finetuning(model, data_loaders, device, final_save_path, args):
    # 此函数逻辑已足够通用，无需修改
    print("\n--- [阶段二] 开始整体模型微调 ---")
    optimizer_params = [
        {"params": model.backbone.parameters(), "lr": args.lr_finetune / 10},
        {"params": model.hetero_head.parameters(), "lr": args.lr_finetune}
    ]
    opt = torch.optim.AdamW(optimizer_params, weight_decay=1e-4)
    dl_A, dl_B, dl_val = data_loaders['source'], data_loaders['target_train'], data_loaders['target_val']
    dl_A_iter = iter(dl_A)
    best_val = float('inf')
    patience_counter = args.patience_finetune
    temp_save_path = final_save_path + ".tmp"
    # ... (内部逻辑保持不变)
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
            loss = nll + args.alpha_r2 * r2_loss + args.lambda_coral * coral
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
            patience_counter = args.patience_finetune
        else:
            patience_counter -= 1
            if patience_counter == 0:
                print(f"验证损失连续 {args.patience_finetune} 轮未改善，触发早停。")
                break
    print(f"--- [阶段二] 微调完成，最佳模型已保存至 {final_save_path} ---")


# ========== 5. 预测函数 ==========
def get_predictions(model, dataloader, device):
    # 此函数逻辑已足够通用，无需修改
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            preds, _, _ = model(inputs)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


# ========== 6. 主函数 (全新逻辑) ==========
# 不再生成pretrain.pth，而是直接生成finetune.pth，这样可以元重启n次，只保留损失函数最小的最佳模型
def main():
    args = setup_args()
    DEVICE = torch.device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.save_path, exist_ok=True)

    print(f"--- 任务启动: {args.opamp} | 设备: {DEVICE} | 模式: 全流程搜索 ---")

    final_finetuned_path = os.path.join(
        args.save_path, f'{args.opamp}_finetuned.pth')

    # --- 数据准备 (只需一次) ---
    data = get_data_and_scalers(opamp_type=args.opamp)
    input_dim, output_dim = data['source'][0].shape[1], data['source'][1].shape[1]

    pretrain_loader_A = make_loader(
        data['source_train'][0], data['source_train'][1], args.batch_a, shuffle=True)
    pretrain_loader_val = make_loader(
        data['source_val'][0], data['source_val'][1], args.batch_a, shuffle=False)

    finetune_loaders = {
        'source': make_loader(data['source'][0], data['source'][1], args.batch_a, shuffle=True, drop_last=True),
        'target_train': make_loader(data['target_train'][0], data['target_train'][1], args.batch_b, shuffle=True),
        'target_val': make_loader(data['target_val'][0], data['target_val'][1], args.batch_b, shuffle=False)
    }

    # --- 全局最优追踪器 ---
    global_best_finetune_val_nll = float('inf')
    global_best_finetune_state_dict = None

    # --- 主循环：遍历每个预训练配置，执行完整的 "预训练->微调" 流水线 ---
    num_experiments = len(args.PRETRAIN_SCHEDULER_CONFIGS)

    # 检查是否需要跳过
    if os.path.exists(final_finetuned_path) and not args.restart:
        print(f"检测到最终模型 {final_finetuned_path} 已存在且未指定 --restart。跳过所有训练。")
        # 评估现有模型 (可选)
        if args.evaluate:
            model = AlignHeteroMLP(
                input_dim, output_dim, args.hidden_dim, args.num_layers, args.dropout_rate).to(DEVICE)
            model.load_state_dict(torch.load(
                final_finetuned_path, map_location=DEVICE))
            pred_scaled, true_scaled = get_predictions(
                model, finetune_loaders['target_val'], DEVICE)
            calculate_and_print_metrics(
                pred_scaled, true_scaled, data['y_scaler'])
        return

    for i, scheduler_config in enumerate(args.PRETRAIN_SCHEDULER_CONFIGS):
        print(f"\n{'='*30} 完整流水线 {i+1}/{num_experiments} {'='*30}")

        # 1. 每次都创建新模型，保证隔离
        model = AlignHeteroMLP(
            input_dim=input_dim, output_dim=output_dim,
            hidden_dim=args.hidden_dim, num_layers=args.num_layers,
            dropout_rate=args.dropout_rate
        ).to(DEVICE)

        # 2. 执行预训练
        _, best_pretrained_state = run_pretraining(
            model, pretrain_loader_A, pretrain_loader_val, DEVICE, args, scheduler_config)

        if not best_pretrained_state:
            print("  [警告] 本次预训练未产生有效模型，跳过此流水线。")
            continue

        # 3. 加载最佳预训练权重，准备微调
        print("\n--- [加载预训练模型] ---")
        model.load_state_dict(best_pretrained_state)

        # 4. 执行微调
        # 为本次微调创建一个临时的模型保存路径
        temp_finetune_path = os.path.join(
            args.save_path, f"{args.opamp}_finetune_temp_run_{i+1}.pth")

        run_finetuning(model, finetune_loaders, DEVICE,
                       temp_finetune_path, args)

        # 5. 评估本次微调结果，并与全局最优比较
        if os.path.exists(temp_finetune_path):
            # 加载本次微调的最佳状态
            model.load_state_dict(torch.load(
                temp_finetune_path, map_location=DEVICE))

            # 在验证集上计算最终NLL
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
                    f"  >>> 新的全局最佳模型诞生！ Val NLL 更新为: {global_best_finetune_val_nll:.6f} <<<")

            # 7. 清理临时文件
            os.remove(temp_finetune_path)

        # 8. 清理内存，准备下一次循环
        del model, best_pretrained_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- 所有流水线结束后 ---
    if global_best_finetune_state_dict:
        print(f"\n{'='*30} 所有流水线执行完毕 {'='*30}")
        print(f"全局最优模型的微调验证 NLL 为: {global_best_finetune_val_nll:.6f}")
        print(f"正在保存最终模型至: {final_finetuned_path}")
        torch.save(global_best_finetune_state_dict, final_finetuned_path)
    else:
        print("\n[错误] 所有流水线均未成功生成模型，未保存任何最终模型。")
        return

    # --- (可选) 评估最终选出的模型 ---
    if args.evaluate:
        print("\n--- [最终评估流程启动] ---")
        model = AlignHeteroMLP(input_dim, output_dim, args.hidden_dim,
                               args.num_layers, args.dropout_rate).to(DEVICE)
        model.load_state_dict(torch.load(
            final_finetuned_path, map_location=DEVICE))
        pred_scaled, true_scaled = get_predictions(
            model, finetune_loaders['target_val'], DEVICE)
        calculate_and_print_metrics(pred_scaled, true_scaled, data['y_scaler'])


if __name__ == "__main__":
    main()
