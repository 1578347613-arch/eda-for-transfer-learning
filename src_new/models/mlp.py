# models/mlp.py
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=4, dropout_rate=0.1):
        """
        与旧 checkpoint 对齐的 MLP：
        - 顺序容器名：network（确保 state_dict 键名是 network.*）
        - 结构：Linear -> ReLU -> (Dropout) 重复；最后 Linear 输出
        - 不包含 LayerNorm / GELU（与 ckpt 保持一致）
        - num_layers 表示隐藏层数量（hidden->hidden 的次数）
        """
        super().__init__()

        layers = []
        # 输入层: input_dim -> hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout_rate and dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        # 隐藏层: (num_layers - 1) 个 hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate and dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

        # 输出层: hidden_dim -> output_dim
        layers.append(nn.Linear(hidden_dim, output_dim))

        # 必须叫 network，匹配 ckpt 键名
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
