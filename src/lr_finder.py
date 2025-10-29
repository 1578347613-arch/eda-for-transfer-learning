# src/lr_finder.py (已重构为可调用函数)
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path

# --- 从您的项目模块中导入 ---
# 假设此脚本仍在 src 目录中
from data_loader import get_data_and_scalers
from models.align_hetero import AlignHeteroMLP
from loss_function import heteroscedastic_nll
import config


def _run_lr_range_test(model, optimizer, criterion, loader, device, start_lr, end_lr, num_iter):
    """执行学习率范围测试的核心逻辑 (内部函数)"""
    lrs = []
    losses = []
    factor = (end_lr / start_lr) ** (1 / (num_iter - 1))
    data_iter = iter(loader)

    for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr

    model.train()
    for i in range(num_iter):
        try:
            inputs, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            inputs, labels = next(data_iter)

        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        if isinstance(criterion, nn.HuberLoss):
            mu, _, _ = model(inputs)
            loss = criterion(mu, labels)
        else:
            mu, logvar, _ = model(inputs)
            loss = criterion(mu, logvar, labels)

        if torch.isnan(loss) or loss.item() > 4 * (losses[0] if losses else 1e9):
            break

        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        for param_group in optimizer.param_groups:
            param_group['lr'] *= factor

    return lrs, losses


def find_optimal_lr(
    model_class,
    model_params,
    data,
    mode="pretrain",
    pretrained_weights_path=None,
    start_lr=1e-7,
    end_lr=1.0,
    num_iter=100,
    batch_size=128,
    device="cuda",
    smoothing_beta=0.98
):
    """
    一个可调用的函数，用于寻找给定模型和数据的最优学习率。

    返回:
        float: 推荐的最佳学习率。
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # 1. 实例化模型
    model = model_class(**model_params).to(device)

    # 2. 根据模式设置数据、优化器和损失函数
    if mode == "pretrain":
        X_train, y_train = data['source_train']
        optimizer = torch.optim.AdamW(model.backbone.parameters(), lr=start_lr)
        criterion = nn.HuberLoss()
    elif mode == "finetune":
        if not pretrained_weights_path or not os.path.exists(pretrained_weights_path):
            raise FileNotFoundError(
                f"微调模式下需要有效的预训练模型路径，但未找到: {pretrained_weights_path}")
        model.load_state_dict(torch.load(
            pretrained_weights_path, map_location=device))
        X_train, y_train = data['target_train']
        optimizer = torch.optim.AdamW(
            model.hetero_head.parameters(), lr=start_lr)
        criterion = heteroscedastic_nll
    else:
        raise ValueError("mode 必须是 'pretrain' 或 'finetune'")

    # 3. 创建数据加载器
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True
    )

    # 4. 运行学习率范围测试
    lrs, losses = _run_lr_range_test(
        model, optimizer, criterion, loader, device, start_lr, end_lr, num_iter)

    if not losses:
        print("警告: 学习率查找器未能收集到任何损失数据。返回默认值 1e-3。")
        return 1e-3

    # 5. 自动选择最佳学习率 (核心算法)
    # 使用指数移动平均平滑损失曲线以减少噪声
    smoothed_losses = []
    avg_loss = 0
    for i, loss in enumerate(losses):
        avg_loss = smoothing_beta * avg_loss + (1 - smoothing_beta) * loss
        smoothed_losses.append(avg_loss / (1 - smoothing_beta**(i+1)))

    # 找到损失下降最快的点 (梯度最小的点)
    # 我们在对数尺度的学习率上计算梯度
    log_lrs = np.log10(lrs)
    gradients = np.gradient(smoothed_losses, log_lrs)
    best_idx = np.argmin(gradients)

    # 最佳学习率通常是这个最速下降点，或者比最低损失点小一个数量级
    # 这里我们选择最速下降点作为推荐值
    optimal_lr = lrs[best_idx]

    # 清理内存
    del model, optimizer, loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return optimal_lr


# --- 保留原始的命令行执行功能，用于手动调试 ---
if __name__ == "__main__":
    # ... (这里可以保留或删除 setup_args 和 main 函数，因为主要功能已封装)
    print("此脚本现在主要作为工具模块使用。请从其他脚本中调用 find_optimal_lr 函数。")
