# unified_train.py (版本 4.0 - 配置驱动 - 已更新)
import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import copy
import ast  # <-- 导入 ast
import json  # <-- 导入 json

# --- 从项目模块中导入 ---
from data_loader import get_data_and_scalers
from models.align_hetero import AlignHeteroMLP
from loss_function import heteroscedastic_nll, batch_r2, coral_loss
from evaluate import calculate_and_print_metrics
# <<< 导入 config 时，现在导入 TASK_CONFIGS 和 COMMON_CONFIG
from config import COMMON_CONFIG, TASK_CONFIGS

# ========== 1. 参数定义与解析 (已更新) ==========


def setup_args():
    """
    一个更简洁的参数解析器，能动态地从 config 文件加载特定任务的默认值。
    """
    parser = argparse.ArgumentParser(description="统一的多任务训练脚本")
    parser.add_argument("--opamp", type=str, required=True,
                        choices=TASK_CONFIGS.keys(), help="必须指定的电路类型")

    # Step 1: 先解析出 opamp 类型
    temp_args, other_args = parser.parse_known_args()
    opamp_type = temp_args.opamp

    # Step 2: 合并通用配置和特定任务的配置作为默认值
    # 注意：config.py 必须在 src 目录或 python 路径下
    defaults = {**COMMON_CONFIG, **TASK_CONFIGS.get(opamp_type, {})}

    # Step 3: 动态为所有简单类型的默认参数创建命令行开关
    for key, value in defaults.items():
        if isinstance(value, (list, dict)):
            continue  # 跳过复杂类型 (如 hidden_dims 列表)
        if isinstance(value, bool):
            parser.add_argument(
                f"--{key}", action=argparse.BooleanOptionalAction, help=f"开关 '{key}'")
        else:
            parser.add_argument(
                f"--{key}", type=type(value), help=f"设置 '{key}'")

    # Step 4: 将合并后的配置设置为解析器的默认值并进行最终解析
    # 复杂类型 (如 hidden_dims) 会从这里被正确设置
    parser.set_defaults(**defaults)

    # --- 覆盖：手动添加在config中是列表，但希望在命令行中覆盖的参数 ---
    # 我们需要手动添加 hidden_dims，以便命令行可以覆盖 config
    parser.add_argument("--hidden_dims", type=str,
                        help="MLP隐藏层维度列表, e.g., '[256, 256]'")
    # 手动添加 --evaluate 和 --results_file (如果它们在 COMMON_CONFIG 中)
    if 'evaluate' not in defaults:
        parser.add_argument("--evaluate", action='store_true',
                            help="训练结束后，加载最佳模型并进行评估")
    if 'results_file' not in defaults:
        parser.add_argument("--results_file", type=str, default=None,
                            help="如果提供，则将最终评估结果保存到此JSON文件")

    args = parser.parse_args()
    return args


# ========== 2. 辅助函数 (DataLoader, 预训练, 微调, 预测) ==========
# (这些函数 run_pretraining, run_finetuning, get_predictions, make_loader 保持不变)
# ... (此处省略了这些函数的代码，它们和您文件中的保持一致) ...
def make_loader(x, y, bs, shuffle=True, drop_last=False):
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=shuffle, drop_last=drop_last)


def run_pretraining(model, train_loader, val_loader, device, args, scheduler_config):
    # ... (函数体与您文件中的完全一致)
    print(f"\n--- [子运行] 使用配置: {scheduler_config} ---")
    optimizer = torch.optim.AdamW(
        model.backbone.parameters(), lr=args.lr_pretrain)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=scheduler_config['T_0'], T_mult=scheduler_config['T_mult'], eta_min=1e-6)
    criterion = torch.nn.HuberLoss(delta=1.0)
    best_val_loss_this_run = float('inf')
    best_state_dict_this_run = None
    epochs = scheduler_config['epochs_pretrain']
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


def run_finetuning(model, data_loaders, device, final_save_path, args):
    # ... (函数体与您文件中的完全一致)
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
    return best_val  # <-- 返回最佳NLL


def get_predictions(model, dataloader, device):
    # ... (函数体与您文件中的完全一致)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            preds, _, _ = model(inputs)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)

# ========== 6. 主函数 (已更新) ==========


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

    # --- 解析 hidden_dims ---
    # 它可能来自 config (list) 或 命令行 (str)
    if isinstance(args.hidden_dims, str):
        try:
            hidden_dims_list = ast.literal_eval(args.hidden_dims)
        except (ValueError, SyntaxError):
            print(f"错误: --hidden_dims 参数格式不正确: {args.hidden_dims}")
            return
    else:
        hidden_dims_list = args.hidden_dims  # 它已经是列表了

    global_best_val_loss = float('inf')
    best_finetune_nll = float('NaN')  # 用于评估

    # === 阶段一：预训练 (统一逻辑) ===
    if args.restart or not os.path.exists(pretrained_path):
        pretrain_loader_A = make_loader(
            data['source_train'][0], data['source_train'][1], args.batch_a, shuffle=True)
        pretrain_loader_val = make_loader(
            data['source_val'][0], data['source_val'][1], args.batch_a, shuffle=False)

        # 统一的、由配置驱动的独立重启循环
        # 注意：args.PRETRAIN_SCHEDULER_CONFIGS 是从 config 加载的
        num_experiments = len(args.PRETRAIN_SCHEDULER_CONFIGS)
        print(f"--- [阶段一] 启动预训练流程 ({num_experiments} 次独立实验) ---")

        for i, scheduler_config in enumerate(args.PRETRAIN_SCHEDULER_CONFIGS):
            print(f"\n{'='*30} 独立实验 {i+1}/{num_experiments} {'='*30}")

            # <<< --- 核心修改 --- >>>
            model = AlignHeteroMLP(
                input_dim=input_dim, output_dim=output_dim,
                hidden_dims=hidden_dims_list,   # <-- 使用列表
                dropout_rate=args.dropout_rate
            ).to(DEVICE)
            # <<< --- 修改结束 --- >>>

            best_loss_this_run, best_state_dict_this_run = run_pretraining(
                model, pretrain_loader_A, pretrain_loader_val, DEVICE, args, scheduler_config)

            if best_state_dict_this_run and best_loss_this_run < global_best_val_loss:
                global_best_val_loss = best_loss_this_run
                print(
                    f"  [BEST] 新的全局最佳损失！ {global_best_val_loss:.6f}。正在覆盖 {pretrained_path}...")
                torch.save(best_state_dict_this_run, pretrained_path)

            del model, best_state_dict_this_run
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(
            f"\n所有独立实验完成，全局最佳模型 (损失: {global_best_val_loss:.6f}) 已保存在 {pretrained_path}")

    # === 加载最佳预训练模型 ===
    print(f"--- [阶段一完成] 加载最终的最佳预训练模型: {pretrained_path} ---")

    # <<< --- 核心修改 --- >>>
    model = AlignHeteroMLP(
        input_dim=input_dim, output_dim=output_dim,
        hidden_dims=hidden_dims_list,   # <-- 使用列表
        dropout_rate=args.dropout_rate
    ).to(DEVICE)
    # <<< --- 修改结束 --- >>>

    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE))
    else:
        print(f"警告：未找到预训练模型 {pretrained_path}。将使用随机初始化的模型进行微调。")

    # === 阶段二：微调 ===
    finetune_loaders = {
        'source': make_loader(data['source'][0], data['source'][1], args.batch_a, shuffle=True, drop_last=True),
        'target_train': make_loader(data['target_train'][0], data['target_train'][1], args.batch_b, shuffle=True),
        'target_val': make_loader(data['target_val'][0], data['target_val'][1], args.batch_b, shuffle=False)
    }

    if os.path.exists(finetuned_path) and not args.restart:
        model.load_state_dict(torch.load(finetuned_path, map_location=DEVICE))
    else:
        best_finetune_nll = run_finetuning(
            model, finetune_loaders, DEVICE, finetuned_path, args)

    # === (可选) 评估 ===
    if args.evaluate:
        print("\n--- [评估流程启动] ---")
        if not os.path.exists(finetuned_path):
            print(f"错误：未找到已训练的模型文件 {finetuned_path}。跳过评估。")
            return
        model.load_state_dict(torch.load(finetuned_path, map_location=DEVICE))

        # <<< --- 核心修改 (使用finetune_loaders) --- >>>
        pred_scaled, true_scaled = get_predictions(
            model, finetune_loaders['target_val'], DEVICE)

        eval_metrics = calculate_and_print_metrics(
            pred_scaled, true_scaled, data['y_scaler'])

        # --- 为 JSON 结果文件添加逻辑 ---
        final_results = {
            'opamp': args.opamp,
            'best_finetune_val_nll': best_finetune_nll,  # 捕获NLL
            'evaluation_metrics': eval_metrics
        }
        if args.results_file:
            try:
                with open(args.results_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(final_results, indent=4) + "\n")
                print(f"评估结果已成功追加至: {args.results_file}")
            except Exception as e:
                print(f"错误: 保存结果文件失败 - {e}")


if __name__ == "__main__":
    main()
