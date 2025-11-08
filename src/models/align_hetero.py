# models/align_hetero.py
import torch.nn as nn
from .mlp import MLP
from typing import List  # <-- 导入 List


class AlignHeteroMLP(nn.Module):
    """
    一个包装器，它使用 MLP 作为骨干，并添加一个异方差头 (hetero_head)
    来预测 logvar。
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims: List[int],    # <-- 改为列表
                 dropout_rate: float
                 ):
        super().__init__()
        self.backbone = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,  # <-- 传递列表
            dropout_rate=dropout_rate
        )
        # 异方差头
        self.hetero_head = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        features = self.backbone(x)
        mu = features
        logvar = self.hetero_head(features)
        return mu, logvar, features
