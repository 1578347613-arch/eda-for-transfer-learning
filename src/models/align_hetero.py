# models/align_hetero.py
import torch.nn as nn
from .mlp import MLP
import config
from typing import List


class AlignHeteroMLP(nn.Module):
    """
    与基线 MLP 共享同构主干（不改变主干结构）：
    - backbone: 与 baseline MLP 结构一致
    - hetero_head: 产生 logvar（异方差）
    - forward: 返回 (mu, logvar, features)；features = mu（与现有训练脚本接口一致）
    """

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dims: List[int],
                 dropout_rate: float):
        super().__init__()
        self.backbone = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,      # 直接传递列表
            dropout_rate=dropout_rate
        )
        self.hetero_head = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        features = self.backbone(x)   # 主干输出
        mu = features
        logvar = self.hetero_head(features)
        return mu, logvar, features
