# models/mlp.py
import torch.nn as nn
import config
from typing import List


class MLP(nn.Module):

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dims: List[int],
                 dropout_rate: float):
        """
        构造函数，用于初始化 MLP 模型。

        参数:
        - input_dim (int): 输入特征的维度。
        - output_dim (int): 输出预测的维度。
        - hidden_dim (int): 每个隐藏层的神经元数量。
        - num_layers (int): 隐藏层的数量。注意这里的定义是指 hidden->hidden 层的数量，总的线性层会更多。
        - dropout_rate (float): 在 ReLU 激活后使用的 Dropout 比率，用于防止过拟合。
        """
        super().__init__()

        if not hidden_dims:
            raise ValueError("hidden_dims 列表不能为空。")

        layers = []

        # --- 动态构建网络层 ---
        # 1. 输入层: 将 input_dim 连接到第一个隐藏层
        current_dim = input_dim
        layers.append(nn.Linear(current_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        current_dim = hidden_dims[0]

        # 2. 隐藏层: 遍历 hidden_dims 列表的剩余部分来构建
        for h_dim in hidden_dims[1:]:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim

        # 3. 输出层: 将最后一个隐藏层连接到 output_dim
        layers.append(nn.Linear(current_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):

        return self.network(x)
