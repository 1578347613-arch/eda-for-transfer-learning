import os
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# --- 从您的项目模块中导入 ---
from data_loader import get_data_and_scalers
from models.align_hetero import AlignHeteroMLP
import config
from loss_function import heteroscedastic_nll


def setup_args():
    """设置和解析命令行参数"""
    parser = argparse.ArgumentParser(description="学习率范围测试脚本")
    parser.add_argument("--mode", type=str, default="pretrain", choices=["pretrain", "finetune"],
                        help="选择测试模式: 'pretrain' 或 'finetune'")
    parser.add_argument("--opamp", type=str,
                        default=config.OPAMP_TYPE, help="运放类型")
    parser.add_argument("--start_lr", type=float,
                        default=1e-7, help="起始的最小学习率")
    parser.add_argument("--end_lr", type=float, default=1.0, help="结束的最大学习率")
    parser.add_argument("--num_iter", type=int,
                        default=2000, help="测试的总步数（batch数量）")
    parser.add_argument("--batch_size", type=int,
                        default=128, help="测试时使用的Batch Size")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备 'cuda' or 'cpu'")
    parser.add_argument("--pretrained_path", type=str,
                        default="../results/{}_pretrained.pth".format(
                            config.OPAMP_TYPE),
                        help="预训练模型的路径 (仅在 finetune 模式下使用)")
    args = parser.parse_args()
    return args


def run_lr_range_test(model, optimizer, criterion, loader, device, start_lr, end_lr, num_iter):
    """执行学习率范围测试的核心逻辑"""
    lrs = []
    losses = []

    # 计算每一步学习率的乘法因子
    factor = (end_lr / start_lr) ** (1 / (num_iter - 1))

    # 获取数据加载器的迭代器
    data_iter = iter(loader)

    # 设置初始学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr

    model.train()  # 确保模型处于训练模式

    for i in range(num_iter):
        try:
            inputs, labels = next(data_iter)
        except StopIteration:
            # 如果数据用完，则重新创建迭代器
            data_iter = iter(loader)
            inputs, labels = next(data_iter)

        inputs, labels = inputs.to(device), labels.to(device)

        # 1. 前向传播
        optimizer.zero_grad()
        if isinstance(criterion, nn.HuberLoss):
            mu, _, _ = model(inputs)
            loss = criterion(mu, labels)
        else:  # 假设是 heteroscedastic_nll
            mu, logvar, _ = model(inputs)
            loss = criterion(mu, logvar, labels)

        # 2. 检查损失是否爆炸
        if torch.isnan(loss) or loss.item() > 4 * (losses[0] if losses else 1e9):
            print(f"损失在第 {i+1} 步爆炸，测试提前结束。")
            break

        # 3. 记录
        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())

        # 4. 反向传播和更新
        loss.backward()
        optimizer.step()

        # 5. 更新学习率以备下一步使用
        for param_group in optimizer.param_groups:
            param_group['lr'] *= factor

    return lrs, losses


def plot_results(lrs, losses, mode, save_path="../results"):
    """绘制并保存学习率与损失的关系图"""
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    # **关键**: X轴使用对数尺度，因为学习率的变化是指数级的
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title(f"Learning Rate Range Test ({mode} mode)")
    plt.grid(True)

    # 找到损失最低点，并用红点标记
    min_loss_idx = np.argmin(losses)
    min_loss_lr = lrs[min_loss_idx]
    plt.plot(min_loss_lr, losses[min_loss_idx], 'ro',
             label=f'Min Loss at LR={min_loss_lr:.2e}')
    plt.legend()

    # 保存图像
    plot_file = os.path.join(save_path, f"lr_range_test_{mode}.png")
    plt.savefig(plot_file)
    print(f"\n测试完成！结果图已保存至: {plot_file}")
    print(f"损失最低点对应的学习率约为: {min_loss_lr:.2e}")


def main():
    args = setup_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"--- 运行在设备: {device} ---")
    print(f"--- 当前测试模式: {args.mode.upper()} ---")

    # 1. 加载数据
    data = get_data_and_scalers(opamp_type=args.opamp)

    # 2. 初始化模型
    input_dim = data['source_train'][0].shape[1]
    output_dim = data['source_train'][1].shape[1]
    model = AlignHeteroMLP(input_dim=input_dim,
                           output_dim=output_dim).to(device)

    # --- [重要] 根据模式进行不同的设置 ---
    if args.mode == "pretrain":
        # --- 预训练模式设置 ---
        print("为预训练准备：使用源域数据，优化 backbone。")
        X_train, y_train = data['source_train']
        # 我们只测试 backbone
        optimizer = torch.optim.AdamW(
            model.backbone.parameters(), lr=args.start_lr)
        criterion = nn.HuberLoss()

    elif args.mode == "finetune":
        # --- 微调模式设置 ---
        print("为微调准备：加载预训练模型，使用目标域数据，优化 head。")

        # 检查并加载预训练权重
        if not os.path.exists(args.pretrained_path):
            raise FileNotFoundError(
                f"错误：在finetune模式下，未找到预训练模型文件: {args.pretrained_path}")
        print(f"加载预训练权重: {args.pretrained_path}")
        model.load_state_dict(torch.load(
            args.pretrained_path, map_location=device))

        X_train, y_train = data['target_train']
        # 我们只测试 hetero_head
        optimizer = torch.optim.AdamW(
            model.hetero_head.parameters(), lr=args.start_lr)
        criterion = heteroscedastic_nll

    # 3. 创建 DataLoader
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32)),
        batch_size=args.batch_size,
        shuffle=True
    )

    # 4. 运行测试
    print(
        f"开始学习率范围测试，从 {args.start_lr:.1e} 到 {args.end_lr:.1e}，共 {args.num_iter} 步...")
    lrs, losses = run_lr_range_test(
        model, optimizer, criterion, loader, device,
        args.start_lr, args.end_lr, args.num_iter
    )

    # 5. 绘制结果
    if lrs and losses:
        plot_results(lrs, losses, args.mode)
    else:
        print("未能收集到任何数据点，无法绘制图像。")


if __name__ == "__main__":
    main()
