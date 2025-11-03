# src/optimizer_utils.py (请替换整个文件)

import torch.nn as nn
from torch.optim import AdamW
from typing import List

# <<< --- 核心修改：重新引入 ratio 参数 --- >>>


def create_discriminative_optimizer(
    model: nn.Module,
    head_lr: float,
    gap_ratio: float,
    internal_ratio: float,
    weight_decay: float = 1e-4
):
    """
    创建差分学习率优化器。(v6 - 可配置比例)
    - Head: head_lr
    - Backbone 躯干顶层: head_lr / gap_ratio
    - Backbone 躯干其余层: 在 (head_lr / gap_ratio) 的基础上，每层再除以 internal_ratio
    """
    param_groups = []

    # 1. 抓取 backbone 中所有的线性层
    all_backbone_layers = [
        m for m in model.backbone.network
        if isinstance(m, nn.Linear)
    ]

    # 2. 拆分
    backbone_head_layer = all_backbone_layers[-1:]  # 最后一层
    backbone_trunk = all_backbone_layers[:-1]       # 其余层

    if not backbone_trunk:
        print("警告: Backbone 只有一个线性层。所有层都将被视为 Head。")

    print(
        f"--- [Optimizer] 应用逐层差分学习率 (v6 - Gap={gap_ratio}, Internal={internal_ratio}) ---")

    # 3. 添加 Head (学习率最高)
    head_params = list(backbone_head_layer[0].parameters()) + \
        list(model.hetero_head.parameters())
    param_groups.append({
        "params": head_params,
        "lr": head_lr
    })
    print(f"  - Model Heads (Backbone_last + Hetero) LR: {head_lr:.2e}")

    # 4. 添加 Backbone "躯干" 层 (从后向前)
    if backbone_trunk:
        top_trunk_layer = backbone_trunk[-1:]
        top_trunk_lr = head_lr / gap_ratio  # <<< --- 使用参数

        param_groups.append({
            "params": top_trunk_layer[0].parameters(),
            "lr": top_trunk_lr
        })
        print(
            f"  - Backbone Trunk (Top) LR: {top_trunk_lr:.2e} (Head / {gap_ratio})")

        # 躯干的 "其余层"
        remaining_trunk = backbone_trunk[:-1]
        num_remaining = len(remaining_trunk)

        for i, layer in enumerate(reversed(remaining_trunk)):
            exponent = i + 1
            lr = top_trunk_lr / (internal_ratio ** exponent)  # <<< --- 使用参数

            param_groups.append({
                "params": layer.parameters(),
                "lr": lr
            })
            print(
                f"  - Backbone Trunk (Layer {num_remaining - 1 - i}) LR: {lr:.2e} (Top_Trunk / {internal_ratio**exponent})")

    # 5. 安全捕获 (不变)
    grouped_params = set()
    for group in param_groups:
        grouped_params.update(group['params'])
    other_params = [
        p for p in model.parameters()
        if p.requires_grad and p not in grouped_params
    ]
    if other_params:
        lowest_lr = head_lr / gap_ratio
        if backbone_trunk and remaining_trunk:
            lowest_lr = top_trunk_lr / (internal_ratio ** len(remaining_trunk))

        param_groups.append({
            "params": other_params,
            "lr": lowest_lr
        })
        print(f"  - 其他参数 LR: {lowest_lr:.2e}")

    return AdamW(param_groups, weight_decay=weight_decay)
