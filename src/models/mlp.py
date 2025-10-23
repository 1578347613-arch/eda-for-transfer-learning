# models/mlp.py
import torch.nn as nn
import config
HIDDEN_DIM = config.HIDDEN_DIM
NUM_LAYERS = config.NUM_LAYERS
DROPOUT_RATE = config.DROPOUT_RATE


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim,
                 hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = NUM_LAYERS,
                 dropout_rate: float = DROPOUT_RATE):
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

        layers = []
        # --- 构建网络层 ---
        # 1. 输入层: 将输入维度映射到隐藏层维度
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())  # 使用 ReLU 作为激活函数
        if dropout_rate and dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        # 2. 隐藏层: (num_layers - 1) 个，保持维度不变 (hidden_dim -> hidden_dim)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate and dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

        # 3. 输出层: 将隐藏层维度映射到最终的输出维度
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):

        return self.network(x)
