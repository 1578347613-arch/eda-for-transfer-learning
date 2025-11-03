# src/optimizer_utils.py (请替换整个文件)

import torch.nn as nn
from torch.optim import AdamW
from typing import List

# <<< --- 核心修改：v7 优化器，分离两个 Head --- >>>


def create_discriminative_optimizer(
    model: nn.Module,
    lr_backbone_head: float,  # Backbone Head (L5) 的学习率
    lr_hetero: float,        # Hetero Head 的学习率
    gap_ratio: float,
    internal_ratio: float,
    weight_decay: float = 1e-4
):
    """
    创建差分学习率优化器。(v7 - 双 Head 分离)
    - Group 1 (LR Finder 目标): Backbone Head (L5) @ lr_backbone_head
    - Group 2 (独立): Hetero Head @ lr_hetero
    - Group 3... (衰减): Backbone Trunk @ lr_backbone_head / gap_ratio ...
    """
    param_groups = []

    # 1. 抓取 backbone 中所有的线性层
    all_backbone_layers = [
        m for m in model.backbone.network
        if isinstance(m, nn.Linear)
    ]

    # 2. 拆分
    backbone_head_layer = all_backbone_layers[-1:]
    backbone_trunk = all_backbone_layers[:-1]

    if not backbone_trunk:
        print("警告: Backbone 只有一个线性层。")

    print(f"--- [Optimizer] 应用逐层差分学习率 (v7 - 双 Head 分离) ---")
    print(f"    - Hetero Head LR: {lr_hetero:.2e}")
    print(
        f"    - Backbone Head LR: {lr_backbone_head:.2e} (Gap={gap_ratio}, Internal={internal_ratio})")

    # <<< --- 组 1: Backbone Head (LR Finder 将跟踪此组) --- >>>
    param_groups.append({
        "params": backbone_head_layer[0].parameters(),
        "lr": lr_backbone_head
    })
    print(f"  - (Group 1) Backbone Head (L5) LR: {lr_backbone_head:.2e}")

    # <<< --- 组 2: Hetero Head (独立学习率) --- >>>
    param_groups.append({
        "params": model.hetero_head.parameters(),
        "lr": lr_hetero
    })
    print(f"  - (Group 2) Hetero Head LR: {lr_hetero:.2e}")

    # <<< --- 组 3+: Backbone Trunk (相对于 Backbone Head 衰减) --- >>>
    if backbone_trunk:
        top_trunk_layer = backbone_trunk[-1:]
        top_trunk_lr = lr_backbone_head / gap_ratio

        param_groups.append({
            "params": top_trunk_layer[0].parameters(),
            "lr": top_trunk_lr
        })
        print(
            f"  - (Group 3) Backbone Trunk (L4) LR: {top_trunk_lr:.2e} (Head / {gap_ratio})")

        remaining_trunk = backbone_trunk[:-1]
        num_remaining = len(remaining_trunk)

        for i, layer in enumerate(reversed(remaining_trunk)):
            exponent = i + 1
            lr = top_trunk_lr / (internal_ratio ** exponent)

            param_groups.append({
                "params": layer.parameters(),
                "lr": lr
            })
            print(
                f"  - (Group 4+) Backbone Trunk (L{num_remaining - i}) LR: {lr:.2e} (Top_Trunk / {internal_ratio**exponent})")

    # 5. 安全捕获 (不变)
    grouped_params = set()
    for group in param_groups:
        grouped_params.update(group['params'])
    other_params = [
        p for p in model.parameters()
        if p.requires_grad and p not in grouped_params
    ]
    if other_params:
        lowest_lr = lr_backbone_head / gap_ratio
        if backbone_trunk and remaining_trunk:
            lowest_lr = top_trunk_lr / (internal_ratio ** len(remaining_trunk))

        param_groups.append({
            "params": other_params,
            "lr": lowest_lr
        })
        print(f"  - (Group N) 其他参数 LR: {lowest_lr:.2e}")

    return AdamW(param_groups, weight_decay=weight_decay)
