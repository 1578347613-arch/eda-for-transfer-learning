# models/mlp.py
import torch.nn as nn
from typing import List  # <-- 导入 List


class MLP(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: List[int],    # <-- 改为列表
                 dropout_rate: float
                 ):
        """
        构造函数，用于初始化可变隐藏层的 MLP 模型。

        参数:
        - input_dim (int): 输入特征的维度。
        - output_dim (int): 输出预测的维度。
        - hidden_dims (List[int]): 一个整数列表，定义每个隐藏层的维度。
        - dropout_rate (float): Dropout 比率。
        """
        super().__init__()

        if not hidden_dims:
            raise ValueError("hidden_dims 列表不能为空")

        layers = []
        current_dim = input_dim

        # --- 动态构建网络层 ---
        # 1. 循环构建所有隐藏层
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout_rate and dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim  # 更新当前维度为本层输出维度

        # 2. 输出层: 将最后一个隐藏层维度映射到最终的输出维度
        layers.append(nn.Linear(current_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
