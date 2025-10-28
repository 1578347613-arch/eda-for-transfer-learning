# unified_train.py
import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# --- 从项目模块中导入 ---
from data_loader import get_data_and_scalers
from models.align_hetero import AlignHeteroMLP
from loss_function import heteroscedastic_nll, batch_r2, coral_loss
from evaluate import calculate_and_print_metrics
import config
from config import COMMON_CONFIG, TASK_CONFIGS

# ========== 1. 参数定义与解析 (重构) ==========
# "黄金标准" setup_args 函数
# 请在 unified_train.py 和 unified_inverse_train.py 中使用它


def setup_args():
    parser = argparse.ArgumentParser(description="统一训练脚本")
    parser.add_argument("--opamp", type=str, required=True,
                        choices=TASK_CONFIGS.keys(), help="电路类型")

    # --- 1. 自动添加所有可能的参数定义 ---
    # a. 从 COMMON_CONFIG 添加
    for key, value in COMMON_CONFIG.items():
        if isinstance(value, bool):
            if value is False:
                parser.add_argument(
                    f"--{key}", action="store_true", help=f"启用 '{key}' (开关)")
            else:
                parser.add_argument(
                    f"--no-{key}", action="store_false", dest=key, help=f"禁用 '{key}' (开关)")
        else:
            parser.add_argument(
                f"--{key}", type=type(value), help=f"设置 '{key}'")

    # b. 从 TASK_CONFIGS 添加专属参数
    all_task_keys = set().union(*(d.keys() for d in TASK_CONFIGS.values()))
    task_only_keys = all_task_keys - set(COMMON_CONFIG.keys())

    for key in sorted(list(task_only_keys)):
        # 简单的类型推断
        sample_val = TASK_CONFIGS[next(iter(TASK_CONFIGS))][key]
        parser.add_argument(
            f"--{key}", type=type(sample_val), help=f"任务参数: {key}")

    # --- 2. 应用默认值并最终解析 ---
    # a. 先应用通用默认值
    parser.set_defaults(**COMMON_CONFIG)

    # b. 解析一次，拿到 opamp 类型，再应用任务专属默认值
    # parse_known_args 不会因为不完整的命令行而报错
    temp_args, _ = parser.parse_known_args()
    if temp_args.opamp in TASK_CONFIGS:
        parser.set_defaults(**TASK_CONFIGS[temp_args.opamp])

    # c. 最后重新解析，命令行提供的值会覆盖所有默认值
    args = parser.parse_args()

    return args


def make_loader(x, y, bs, shuffle=True, drop_last=False):
    """便捷函数，创建 DataLoader"""
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=shuffle, drop_last=drop_last)

# ========== 2. 阶段一：Backbone 预训练 (已修改) ==========


def run_pretraining(model, train_loader, val_loader, device, args, scheduler_config):
    """
    在源域数据上仅预训练模型的backbone。
    此函数现在接受 scheduler_config 并返回最佳损失和模型状态，而不是直接保存。
    """
    print(f"\n--- [阶段一] 开始 Backbone 预训练 (配置: {scheduler_config}) ---")

    # 从配置中获取参数，如果未提供，则使用默认值
    epochs = scheduler_config.get("epochs_pretrain", args.epochs_pretrain)
    T_0 = scheduler_config.get("T_0", 200)  # 原始硬编码值为 200
    T_mult = scheduler_config.get("T_mult", 1)  # 原始硬编码值为 1

    optimizer = torch.optim.AdamW(
        model.backbone.parameters(), lr=args.lr_pretrain)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult, eta_min=1e-6)

    criterion = torch.nn.HuberLoss(delta=1)
    best_val_loss = float('inf')
    best_model_state_dict = None  # 在内存中保存最佳模型

    patience = args.patience_pretrain
    patience_counter = patience

    # --- 动态计算重启点 ---
    restart_epochs_list = []
    current_T = T_0
    current_restart_point = 0

    while True:
        current_restart_point += current_T
        if current_restart_point <= epochs:
            restart_epochs_list.append(current_restart_point)
            current_T *= T_mult
        else:
            break

    # 转换为集合以便在循环中快速查找
    restart_epochs = set(restart_epochs_list)
    if restart_epochs:
        print(f"优化器将在以下 epoch 结束后重置: {sorted(restart_epochs_list)}")
    # --- 重启点计算结束 ---

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

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state_dict = model.state_dict()  # 保存到内存
            patience_counter = patience  # 重置计数器
            print(f"  - (内存)预训练模型已更新，验证HuberLoss提升至: {best_val_loss:.6f}")
        else:
            patience_counter -= 1
            if patience_counter == 0:
                print(f"验证损失连续 {patience} 轮未改善，触发早停。")
                break

        if (epoch + 1) in restart_epochs and (epoch + 1) < epochs:
            print(f"--- Epoch {epoch+1} 是一个重启点。重置 AdamW 优化器状态！ ---")
            # 重置优化器状态
            optimizer = torch.optim.AdamW(
                model.backbone.parameters(), lr=args.lr_pretrain)
            # 将新优化器交给调度器
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=T_0, T_mult=T_mult, eta_min=1e-6)
            # 手动将 scheduler "快进" 到当前 epoch
            for _ in range(epoch + 1):
                scheduler.step()

    print(f"--- [阶段一] 本次运行完成，最佳损失: {best_val_loss:.6f} ---")
    return best_val_loss, best_model_state_dict


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

    dl_A, dl_B, dl_val = data_loaders['source'], data_loaders['target_train'], data_loaders['target_val']
    dl_A_iter = iter(dl_A)

    best_val = float('inf')

    patience = args.patience_finetune
    patience_counter = patience  # 使用一个计数器

    for epoch in range(args.epochs_finetune):
        model.train()
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

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                mu, logvar, _ = model(xb)
                val_loss += heteroscedastic_nll(mu, logvar, yb).item()
        val_loss /= len(dl_val)

        print(
            f"Fine-tune Epoch [{epoch+1}/{args.epochs_finetune}], Val NLL: {val_loss:.4f}")

        if val_loss < best_val:
            print(f"  - 微调验证损失改善 ({best_val:.4f} -> {val_loss:.4f})。保存模型...")
            best_val = val_loss
            torch.save(model.state_dict(), final_save_path)
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

# ========== 5. 主函数 (已修改) ==========


def main():
    # <<< 一次性解析所有参数
    args = setup_args()
    DEVICE = torch.device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.save_path, exist_ok=True)

    # 模型存储在./results下
    pretrained_path = os.path.join(
        args.save_path, f'{args.opamp}_pretrained.pth')
    finetuned_path = os.path.join(
        args.save_path, f'{args.opamp}_finetuned.pth')

    data = get_data_and_scalers(opamp_type=args.opamp)
    X_src, y_src = data['source']
    input_dim = X_src.shape[1]
    output_dim = y_src.shape[1]
    print(
        f"--- 动态检测到 {args.opamp} 的维度: Input={input_dim}, Output={output_dim} ---")
    X_src_train, y_src_train = data['source_train']
    X_src_val, y_src_val = data['source_val']
    X_trg_tr, y_trg_tr = data['target_train']
    X_trg_val, y_trg_val = data['target_val']

    # 预训练的训练集使用 source_train
    pretrain_loader_A = make_loader(
        X_src_train, y_src_train, args.batch_a, shuffle=True)
    # 预训练的验证集使用 source_val (不再是 target_val)
    pretrain_loader_val = make_loader(
        X_src_val, y_src_val, args.batch_a, shuffle=False)

    # 注意：input_dim 和 output_dim 不在命令行参数里，所以我们从字典里取
    # model_config = TASK_CONFIGS[args.opamp] # model_config 未被使用，注释掉

    # 仅初始化一个模型实例，用于加载权重或作为非5t opamp的基础
    model = AlignHeteroMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate
    ).to(DEVICE)

    # --- [阶段一] 预训练逻辑 (已重构) ---
    if args.restart or not os.path.exists(pretrained_path):

        # --- START: 5t Opamp 特殊元优化逻辑 ---
        if args.opamp == '5t':
            print("--- 检测到 5t opamp，启用“元优化”预训练策略 ---")

            # 在这里定义 5t 的专属配置
            RESTART_PRETRAIN = 9
            PRETRAIN_SCHEDULER_CONFIGS = [  # 重复执行三次元优化
                # --- 策略一：广泛探索 ---
                {"T_0": 50, "T_mult": 1, "epochs_pretrain": 100},  # 第1次重启
                {"T_0": 55, "T_mult": 1, "epochs_pretrain": 110},  # 第2次重启
                # --- 策略二：精细打磨 ---
                {"T_0": 125, "T_mult": 1, "epochs_pretrain": 125},  # 第3次重启
                # --- 策略一：广泛探索 ---
                {"T_0": 50, "T_mult": 1, "epochs_pretrain": 100},  # 第4次重启
                {"T_0": 55, "T_mult": 1, "epochs_pretrain": 110},  # 第5次重启
                # --- 策略二：精细打磨 ---
                {"T_0": 125, "T_mult": 1, "epochs_pretrain": 125},  # 第6次重启
                # --- 策略一：广泛探索 ---
                {"T_0": 50, "T_mult": 1, "epochs_pretrain": 100},  # 第7次重启
                {"T_0": 55, "T_mult": 1, "epochs_pretrain": 110},  # 第8次重启
                # --- 策略二：精细打磨 ---
                {"T_0": 125, "T_mult": 1, "epochs_pretrain": 125},  # 第9次重启
            ]

            global_best_val_loss = float('inf')

            print(f"--- [元优化流程启动] 将执行 {RESTART_PRETRAIN} 次独立预训练 ---")

            for i in range(RESTART_PRETRAIN):
                print(f"\n{'='*30} 人工重启 {i+1}/{RESTART_PRETRAIN} {'='*30}")

                # 每次都重新初始化一个新模型
                model_run = AlignHeteroMLP(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers,
                    dropout_rate=args.dropout_rate
                ).to(DEVICE)

                if i < len(PRETRAIN_SCHEDULER_CONFIGS):
                    current_scheduler_config = PRETRAIN_SCHEDULER_CONFIGS[i]
                else:
                    # 如果配置列表不够长，则复用最后一个
                    current_scheduler_config = PRETRAIN_SCHEDULER_CONFIGS[-1]

                best_loss_this_run, best_state_dict_this_run = run_pretraining(
                    model_run, pretrain_loader_A, pretrain_loader_val, DEVICE, args, current_scheduler_config
                )
                print(
                    f"--- 人工重启 {i+1} 完成，本次最佳损失: {best_loss_this_run:.6f} ---")

                if best_state_dict_this_run and best_loss_this_run < global_best_val_loss:
                    global_best_val_loss = best_loss_this_run
                    print(
                        f"  🏆🏆🏆 新的全局最佳损失！ {global_best_val_loss:.6f}。正在覆盖 {pretrained_path}...")
                    torch.save(best_state_dict_this_run, pretrained_path)
                elif not best_state_dict_this_run:
                    print(
                        f"  -- 本次运行未能产生有效模型 (损失: {best_loss_this_run:.6f})，不保存。")
                else:
                    print(
                        f"  -- 本次结果 ({best_loss_this_run:.6f}) 未超越全局最佳 ({global_best_val_loss:.6f})，不保存。")

            print(
                f"\n所有 {RESTART_PRETRAIN} 次重启完成。最终全局最佳损失: {global_best_val_loss:.6f}")
            print(f"--- [阶段一] 5t 元优化预训练完成 ---")

        # --- END: 5t Opamp 逻辑 ---

        # --- START: 其他 Opamp 的标准逻辑 ---
        else:
            print(f"--- [阶段一] {args.opamp} 启用标准预训练 ---")

            # 为标准逻辑创建一个默认的 scheduler_config
            # 注意：T_0=200, T_mult=1 是原文件中的硬编码值
            default_scheduler_config = {
                "epochs_pretrain": args.epochs_pretrain,
                "T_0": 200,
                "T_mult": 1
            }

            best_loss, best_state_dict = run_pretraining(
                model, pretrain_loader_A, pretrain_loader_val, DEVICE, args, default_scheduler_config
            )

            if best_state_dict:
                print(
                    f"--- [阶段一] 预训练完成，保存最佳模型 (损失: {best_loss:.6f}) 至 {pretrained_path} ---")
                torch.save(best_state_dict, pretrained_path)
            else:
                print(f"--- [阶段一] 预训练完成，但未能找到有效模型。跳过保存。 ---")

        # --- END: 其他 Opamp 逻辑 ---

        # --- 关键：无论哪种训练方式，训练后都从磁盘加载最佳模型 ---
        if os.path.exists(pretrained_path):
            print(f"--- [阶段一] 加载最佳预训练模型 {pretrained_path} 以进行微调 ---")
            model.load_state_dict(torch.load(
                pretrained_path, map_location=DEVICE))
        else:
            print(
                f"--- [阶段一] 警告：预训练未产生任何模型文件 ({pretrained_path})。微调将从随机权重开始！ ---")

    else:
        # 这部分逻辑 (加载现有模型) 保持不变
        print(f"--- [阶段一] 跳过预训练，加载已存在的模型: {pretrained_path} ---")
        model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE))

    # --- [阶段二] 微调逻辑 (保持不变) ---
    finetune_loaders = {
        'source': make_loader(X_src, y_src, args.batch_a, shuffle=True, drop_last=True),
        'target_train': make_loader(X_trg_tr, y_trg_tr, args.batch_b, shuffle=True),
        'target_val': make_loader(X_trg_val, y_trg_val, args.batch_b, shuffle=False)
    }
    if os.path.exists(finetuned_path) and not args.restart:
        print(f"--- [阶段二] 检测到已有微调模型: {finetuned_path}，跳过微调并直接载入该权重 ---")
        model.load_state_dict(torch.load(finetuned_path, map_location=DEVICE))
    else:
        # 只有在模型文件确实存在时才开始微调（防止预训练彻底失败）
        if os.path.exists(pretrained_path) or (args.restart and os.path.exists(pretrained_path)):
            run_finetuning(model, finetune_loaders,
                           DEVICE, finetuned_path, args)
        elif not os.path.exists(pretrained_path) and not args.restart:
            # 如果文件不存在，但我们没有重启，说明是加载模式，
            # 这种情况下 finetuned_path 应该存在 (如上一个 if 所示)，
            # 如果 finetuned_path 也不存在，我们就不应该运行 finetuning
            print(
                f"错误：未找到预训练模型 {pretrained_path} 且未找到微调模型 {finetuned_path}。无法开始微调。")
        else:
            print(f"警告：预训练失败，未生成 {pretrained_path}。跳过微调阶段。")

    print("\n训练流程全部完成。")

    # (可选)测试
    if args.evaluate:
        print("\n--- [评估流程启动] ---")

        # 1. 检查最佳模型文件是否存在
        if not os.path.exists(finetuned_path):
            print(f"错误：未找到已训练的模型文件 {finetuned_path}。跳过评估。")
            return

        # 2. 加载最佳模型权重
        print(f"为评估加载最佳模型权重: {finetuned_path}")
        model.load_state_dict(torch.load(finetuned_path, map_location=DEVICE))

        # 3. 获取预测值和真实值 (标准化空间)
        # 注意：finetune_loaders['target_val'] 是验证集的数据加载器
        print("在验证集上生成预测...")
        pred_scaled, true_scaled = get_predictions(
            model, finetune_loaders['target_val'], DEVICE)

        # 4. 调用外部评估函数
        calculate_and_print_metrics(pred_scaled, true_scaled, data['y_scaler'])


if __name__ == "__main__":
    main()
