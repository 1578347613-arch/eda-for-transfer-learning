import torch
import torch.nn as nn
from .mlp import MLP
import config

class AlignHeteroMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AlignHeteroMLP, self).__init__()
        # 使用 config.py 中的超参数
        self.backbone = MLP(input_dim, output_dim, hidden_dim=config.HIDDEN_DIM, num_layers=config.NUM_LAYERS, dropout_rate=config.DROPOUT_RATE)
        self.hetero_head = nn.Linear(output_dim, output_dim)  # 用于异方差输出

    def forward(self, x):
        features = self.backbone(x)
        mu = features  # 均值
        logvar = self.hetero_head(features)  # 对应的方差
        return mu, logvar, features
