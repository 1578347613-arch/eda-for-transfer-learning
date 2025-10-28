# unified_train.py
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
# [修正] 导入新的 config 结构
from config import COMMON_CONFIG, TASK_CONFIGS

# ========== 1. 参数定义与解析 (重构) ==========
# "黄金标准" setup_args 函数
# 请在 unified_train.py 和 unified_inverse_train.py 中使用它


def setup_args():
    """
    一个健壮的参数解析器，能动态地从 config 文件加载默认值，并允许命令行覆盖。
    """
    parser = argparse.ArgumentParser(description="统一的多任务训练脚本")
    parser.add_argument("--opamp", type=str, required=True,
                        choices=TASK_CONFIGS.keys(), help="必须指定的电路类型")

    # --- Step 1: 先解析出 opamp 类型，以便加载正确的默认值 ---
    temp_args, _ = parser.parse_known_args()
    opamp_type = temp_args.opamp

    # --- Step 2: 合并通用配置和特定任务的配置作为默认值 ---
    defaults = {**COMMON_CONFIG, **TASK_CONFIGS.get(opamp_type, {})}

    # --- Step 3: 动态为所有简单类型的默认参数创建命令行开关 ---
    for key, value in defaults.items():
        # 跳过复杂类型，这些类型应在 config 文件中定义，不适合命令行修改
        if isinstance(value, (list, dict)):
            continue

        if isinstance(value, bool):
            if value is False:  # e.g., restart = False
                parser.add_argument(
                    f"--{key}", action="store_true", help=f"启用 '{key}' (默认: 关闭)")
            else:  # e.g., evaluate = True
                parser.add_argument(
                    f"--no-{key}", action="store_false", dest=key, help=f"禁用 '{key}' (默认: 开启)")
        else:
            parser.add_argument(
                f"--{key}", type=type(value), help=f"设置 '{key}' (默认: {value})")

    # --- Step 4: 将合并后的配置设置为解析器的默认值 ---
    parser.set_defaults(**defaults)

    # --- Step 5: 进行最终解析 ---
    # 命令行中提供的值将覆盖所有来自 config 文件的默认值
    args = parser.parse_args()

    return args


def make_loader(x, y, bs, shuffle=True, drop_last=False):
    """便捷函数，创建 DataLoader"""
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=shuffle, drop_last=drop_last)

# ========== 2. 阶段一：Backbone 预训练 ==========
# (此函数无需修改)


def run_pretraining(model, train_loader, val_loader, device, args, scheduler_config):
    """在源域数据上仅预训练模型的backbone"""
    print(
        f"\n--- [子运行] 使用配置: T_0={scheduler_config['T_0']}, T_mult={scheduler_config['T_mult']}, Epochs={scheduler_config['epochs_pretrain']} ---")

    optimizer = torch.optim.AdamW(
        model.backbone.parameters(), lr=args.lr_pretrain)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=scheduler_config['T_0'], T_mult=scheduler_config['T_mult'], eta_min=1e-6)
    criterion = torch.nn.HuberLoss(delta=1)
    best_val_loss_this_run = float('inf')

    best_state_dict_this_run = None

    T_0 = scheduler_config['T_0']
    T_mult = scheduler_config['T_mult']
    current_T = T_0

    restart_epochs_list = [current_T]
    max_epochs_for_calc = scheduler_config['epochs_pretrain']

    if T_mult > 1:
        while restart_epochs_list[-1] < max_epochs_for_calc:
            current_T *= T_mult
            next_restart = restart_epochs_list[-1] + current_T
            if next_restart <= max_epochs_for_calc:
                restart_epochs_list.append(next_restart)
            else:
                break
    else:  # T_mult == 1
        restart_epochs_list = list(
            range(current_T, max_epochs_for_calc + 1, current_T))

    restart_epochs = set(restart_epochs_list)
    print(f"优化器将在以下 epoch 结束后重置: {sorted(list(restart_epochs))}")

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


# ========== 3. 阶段二：整体模型微调 (重构) ==========
# (此函数无需修改)
def run_finetuning(model, data_loaders, device, final_save_path, args):
    """使用复合损失对整个模型进行微调"""
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
            torch.save(model.state_dict(), final_save_path)
            patience_counter = args.patience_finetune
        else:
            patience_counter -= 1
            if patience_counter == 0:
                print(f"验证损失连续 {args.patience_finetune} 轮未改善，触发早停。")
                break

    print(f"--- [阶段二] 微调完成，最佳模型已保存至 {final_save_path} ---")


# ========== 4.（可选）获取最好模型的输出结果 ==========
# (此函数无需修改)
def get_predictions(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
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
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.save_path, exist_ok=True)

    print(f"--- 任务启动: {args.opamp} | 设备: {DEVICE} ---")

    pretrained_path = os.path.join(
        args.save_path, f'{args.opamp}_pretrained.pth')
    finetuned_path = os.path.join(
        args.save_path, f'{args.opamp}_finetuned.pth')

    data = get_data_and_scalers(opamp_type=args.opamp)
    input_dim, output_dim = data['source'][0].shape[1], data['source'][1].shape[1]

    global_best_val_loss = float('inf')

    # === 阶段一：预训练 ===
    if args.restart or not os.path.exists(pretrained_path):
        pretrain_loader_A = make_loader(
            data['source_train'][0], data['source_train'][1], args.batch_a, shuffle=True)
        pretrain_loader_val = make_loader(
            data['source_val'][0], data['source_val'][1], args.batch_a, shuffle=False)

        # <<< [核心修正] 检查是否存在多阶段配置 >>>
        if hasattr(args, 'pretrain_scheduler_configs') and args.pretrain_scheduler_configs:
            # --- 逻辑 A：处理多阶段、独立重启的预训练 (for 5t_opamp) ---
            print(
                f"--- [阶段一] 启动多阶段预训练流程 ({len(args.pretrain_scheduler_configs)} 次独立实验) ---")

            for i, scheduler_config in enumerate(args.pretrain_scheduler_configs):
                print(
                    f"\n{'='*30} 独立实验 {i+1}/{len(args.pretrain_scheduler_configs)} {'='*30}")

                model = AlignHeteroMLP(
                    input_dim=input_dim, output_dim=output_dim,
                    hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                    dropout_rate=args.dropout_rate
                ).to(DEVICE)

                best_loss_this_run, best_state_dict_this_run = run_pretraining(
                    model, pretrain_loader_A, pretrain_loader_val, DEVICE, args, scheduler_config)
                print(
                    f"--- 独立实验 {i+1} 完成，本次最佳损失: {best_loss_this_run:.6f} ---")

                if best_state_dict_this_run and best_loss_this_run < global_best_val_loss:
                    global_best_val_loss = best_loss_this_run
                    print(
                        f"  🏆🏆🏆 新的全局最佳损失！ {global_best_val_loss:.6f}。正在覆盖 {pretrained_path}...")
                    torch.save(best_state_dict_this_run, pretrained_path)

            print(
                f"\n所有独立实验完成，全局最佳模型 (损失: {global_best_val_loss:.6f}) 已保存在 {pretrained_path}")

            del model
            del best_state_dict_this_run
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        else:
            # --- 逻辑 B：处理单次、标准的预训练 (for two_stage_opamp) ---
            print(f"--- [阶段一] 启动标准单次预训练流程 ---")

            model = AlignHeteroMLP(
                input_dim=input_dim, output_dim=output_dim,
                hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                dropout_rate=args.dropout_rate
            ).to(DEVICE)

            # 手动构建一个 scheduler_config 字典
            scheduler_config = {
                # 如果没有T_0，给一个默认值
                'T_0': getattr(args, 'T_0', args.epochs_pretrain // 5),
                'T_mult': getattr(args, 'T_mult', 1),
                'epochs_pretrain': args.epochs_pretrain
            }

            _, best_state_dict_this_run = run_pretraining(
                model, pretrain_loader_A, pretrain_loader_val, DEVICE, args, scheduler_config)

            if best_state_dict_this_run:
                print(f"标准预训练完成，保存模型到 {pretrained_path}")
                torch.save(best_state_dict_this_run, pretrained_path)
            else:
                print("错误：标准预训练未能产出有效模型。")

    # === 加载最佳预训练模型 ===
    print(f"--- [阶段一完成] 加载最终的最佳预训练模型: {pretrained_path} ---")
    model = AlignHeteroMLP(
        input_dim=input_dim, output_dim=output_dim,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        dropout_rate=args.dropout_rate
    ).to(DEVICE)
    # 确保文件存在才能加载
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE))
    else:
        print(f"警告：未找到预训练模型 {pretrained_path}。将使用随机初始化的模型进行微调。")

    finetune_loaders = {
        'source': make_loader(data['source'][0], data['source'][1], args.batch_a, shuffle=True, drop_last=True),
        'target_train': make_loader(data['target_train'][0], data['target_train'][1], args.batch_b, shuffle=True),
        'target_val': make_loader(data['target_val'][0], data['target_val'][1], args.batch_b, shuffle=False)
    }

    if os.path.exists(finetuned_path) and not args.restart:
        print(f"--- [阶段二] 检测到已有微调模型: {finetuned_path}，跳过微调并直接载入该权重 ---")
        model.load_state_dict(torch.load(finetuned_path, map_location=DEVICE))
    else:
        run_finetuning(model, finetune_loaders, DEVICE, finetuned_path, args)

    print("\n训练流程全部完成。")

    if args.evaluate:
        print("\n--- [评估流程启动] ---")
        if not os.path.exists(finetuned_path):
            print(f"错误：未找到已训练的模型文件 {finetuned_path}。跳过评估。")
            return

        print(f"为评估加载最佳模型权重: {finetuned_path}")
        # 模型结构已正确加载，只需加载权重
        model.load_state_dict(torch.load(finetuned_path, map_location=DEVICE))

        print("在验证集上生成预测...")
        pred_scaled, true_scaled = get_predictions(
            model, finetune_loaders['target_val'], DEVICE)
        calculate_and_print_metrics(pred_scaled, true_scaled, data['y_scaler'])


if __name__ == "__main__":
    main()

                