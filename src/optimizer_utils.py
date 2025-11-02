# src/optimizer_utils.py
import torch.nn as nn
from torch.optim import AdamW
from typing import List


def create_discriminative_optimizer(model: nn.Module, head_lr: float, ratio: float, weight_decay: float = 1e-4):
    """
    创建差分学习率优化器。(v4 - Head First)
    - Head (hetero_head + backbone的最后一层) 学习率最高
    - Backbone 的其余层学习率从后向前按 `ratio` 呈指数衰减
    - Head Group 放在最前面，以便 LR Finder 跟踪。
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
    num_trunk_layers = len(backbone_trunk)

    print("--- [Optimizer] 应用逐层差分学习率 (Head First) ---")

    # 3. (核心修改) 添加所有的 "头" (学习率最高)
    #    这包括 backbone 的最后一层和 hetero_head
    head_params = list(backbone_head_layer[0].parameters()) + \
        list(model.hetero_head.parameters())

    param_groups.append({
        "params": head_params,
        "lr": head_lr
    })
    print(
        f"  - Model Heads (Backbone_last + Hetero) LR: {head_lr:.2e} (Ratio=1/1)")

    # 4. 从后向前添加 Backbone "躯干" 层 (学习率最低)
    for i, layer in enumerate(reversed(backbone_trunk)):
        # 衰减指数：躯干的最后一层是 1, ..., 躯干的第一层是 N
        exponent = i + 1
        lr = head_lr / (ratio ** exponent)

        param_groups.append({
            "params": layer.parameters(),
            "lr": lr
        })
        print(
            f"  - Backbone Trunk (Linear {num_trunk_layers - 1 - i}) LR: {lr:.2e} (Ratio=1/{ratio**exponent:.0f})")

    # 5. 安全捕获 (不变)
    grouped_params = set()
    for group in param_groups:
        grouped_params.update(group['params'])

    other_params = [
        p for p in model.parameters()
        if p.requires_grad and p not in grouped_params
    ]

    if other_params:
        lowest_lr = head_lr / (ratio ** num_trunk_layers)
        param_groups.append({
            "params": other_params,
            "lr": lowest_lr
        })
        print(f"  - 其他参数 (e.g., ReLU/Dropout) LR: {lowest_lr:.2e}")

    return AdamW(param_groups, weight_decay=weight_decay)
