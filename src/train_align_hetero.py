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

# ========== 2. 阶段一：Backbone 预训练 ==========


def run_pretraining(model, train_loader, val_loader, device, save_path, args):
    """在源域数据上仅预训练模型的backbone"""
    print("\n--- [阶段一] 开始 Backbone 预训练 ---")

    optimizer = torch.optim.AdamW(
        model.backbone.parameters(), lr=args.lr_pretrain)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=250, T_mult=1, eta_min=1e-6)
    criterion = torch.nn.HuberLoss(delta=1)
    best_val_loss = float('inf')

    patience = args.patience_pretrain
    patience_counter = patience  # 使用一个计数器

    for epoch in range(args.epochs_pretrain):
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
        print(
            f"Pre-train Epoch [{epoch+1}/{args.epochs_pretrain}], Val HuberLoss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            patience_counter = patience  # 重置计数器
            print(f"  - 预训练模型已保存，验证HuberLoss提升至: {best_val_loss:.6f}")
        else:
            patience_counter -= 1
            if patience_counter == 0:
                print(f"验证损失连续 {patience} 轮未改善，触发早停。")
                break

    print(f"--- [阶段一] 预训练完成，最佳模型已保存至 {save_path} ---")


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

# ========== 5. 主函数 ==========


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
    model_config = TASK_CONFIGS[args.opamp]
    model = AlignHeteroMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate
    ).to(DEVICE)

    if args.restart or not os.path.exists(pretrained_path):

        run_pretraining(model, pretrain_loader_A,
                        pretrain_loader_val, DEVICE, pretrained_path, args)
    else:
        print(f"--- [阶段一] 跳过预训练，加载已存在的模型: {pretrained_path} ---")
        model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE))

    finetune_loaders = {
        'source': make_loader(X_src, y_src, args.batch_a, shuffle=True, drop_last=True),
        'target_train': make_loader(X_trg_tr, y_trg_tr, args.batch_b, shuffle=True),
        'target_val': make_loader(X_trg_val, y_trg_val, args.batch_b, shuffle=False)
    }
    if os.path.exists(finetuned_path) and not args.restart:
        print(f"--- [阶段二] 检测到已有微调模型: {finetuned_path}，跳过微调并直接载入该权重 ---")
        model.load_state_dict(torch.load(finetuned_path, map_location=DEVICE))
    else:
        run_finetuning(model, finetune_loaders, DEVICE, finetuned_path, args)
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
