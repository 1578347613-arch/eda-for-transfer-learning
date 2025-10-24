# models/align_hetero.py

import torch.nn as nn
from .mlp import MLP

# --------------------------------------------------
# --- 删除以下所有对 config 的依赖 ---
# import config  <- 删除
# HIDDEN_DIM = config.HIDDEN_DIM  <- 删除
# NUM_LAYERS = config.NUM_LAYERS  <- 删除
# DROPOUT_RATE = config.DROPOUT_RATE  <- 删除
# --------------------------------------------------


class AlignHeteroMLP(nn.Module):
    """
    与基线 MLP 共享同构主干（不改变主干结构）：
    - backbone: 与 baseline MLP 结构一致
    - hetero_head: 产生 logvar（异方差）
    - forward: 返回 (mu, logvar, features)；features = mu（与现有训练脚本接口一致）
    """

    # --- 修改 __init__ 函数签名 ---
    # 删除依赖于旧config的默认值
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int,
                 hidden_dim: int,    # 不再有默认值，变为必需参数
                 num_layers: int,    # 不再有默认值，变为必需参数
                 dropout_rate: float # 不再有默认值，变为必需参数
                ):
        super().__init__()
        
        # 直接使用传入的参数来初始化子模块
        self.backbone = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        )
        self.hetero_head = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        features = self.backbone(x)
        mu = features
        logvar = self.hetero_head(features)
        return mu, logvar, features
