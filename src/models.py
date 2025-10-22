# src/models.py

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=4, dropout_rate=0.1):
        """
        一个通用的多层感知机 (MLP) 模型。

        参数:
            input_dim (int): 输入特征的维度。
            output_dim (int): 输出目标的维度。
            hidden_dim (int): 每个隐藏层的神经元数量。
            num_layers (int): 隐藏层的数量。
            dropout_rate (float): Dropout的比率，用于正则化。
        """
        super(MLP, self).__init__()
        
        layers = []
        # 输入层
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU()) # GELU是比ReLU更平滑的现代激活函数
        layers.append(nn.Dropout(dropout_rate))

        # 隐藏层
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            
        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        # 将所有层打包成一个序列
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
class DualHeadMLP(nn.Module):
    """共享主干 + A/B 两个输出头；用于A上预训练，B上微调。"""
    def __init__(self, input_dim, output_dim, hidden_dim=512, num_layers=6, dropout_rate=0.1):
        super().__init__()
        trunk = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout_rate)]
        for _ in range(num_layers-1):
            trunk += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout_rate)]
        self.backbone = nn.Sequential(*trunk)
        self.head_A = nn.Linear(hidden_dim, output_dim)
        self.head_B = nn.Linear(hidden_dim, output_dim)

    def features(self, x):
        return self.backbone(x)

    def forward(self, x, domain='A'):
        h = self.features(x)
        if domain == 'A':
            return self.head_A(h)
        elif domain == 'B':
            return self.head_B(h)
        else:
            raise ValueError("domain must be 'A' or 'B'")
# 在文件末尾追加

import torch
import torch.nn as nn

class AlignHeteroMLP(nn.Module):
    """
    与基线MLP共享同构主干（不改变主干结构），
    仅在主干后增加两个B域输出头：mu 与 logvar（异方差）。
    """
    def __init__(self, input_dim, output_dim, hidden_dim=512, num_layers=6, dropout_rate=0.1):
        super().__init__()
        trunk = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout_rate)]
        for _ in range(num_layers - 1):
            trunk += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout_rate)]
        self.backbone = nn.Sequential(*trunk)
        # 两个并行的B域头：mu / logvar
        self.head_B_mu = nn.Linear(hidden_dim, output_dim)
        self.head_B_logvar = nn.Linear(hidden_dim, output_dim)

    def features(self, x):
        return self.backbone(x)  # 倒数第二层的特征表示（用于CORAL）

    def forward(self, x):
        h = self.features(x)
        mu = self.head_B_mu(h)
        logvar = self.head_B_logvar(h)
        return mu, logvar, h


def load_backbone_from_trained_mlp(trained_mlp: nn.Module, align_model: AlignHeteroMLP):
    """
    将已训练的单头MLP参数，拷到 AlignHeteroMLP 的 backbone；
    并用原MLP最后一层初始化 head_B_mu；head_B_logvar 用0初始化。
    """
    with torch.no_grad():
        # 拷贝 backbone（single.model 去掉最后一层Linear）
        for layer_align, layer_single in zip(align_model.backbone, trained_mlp.model[:-1]):
            if isinstance(layer_align, nn.Linear) and isinstance(layer_single, nn.Linear):
                layer_align.weight.copy_(layer_single.weight)
                layer_align.bias.copy_(layer_single.bias)
            elif isinstance(layer_align, nn.LayerNorm) and isinstance(layer_single, nn.LayerNorm):
                layer_align.weight.copy_(layer_single.weight)
                layer_align.bias.copy_(layer_single.bias)
            # GELU/Dropout 没有可拷参数

        # head_B_mu ← single.model[-1]
        last = trained_mlp.model[-1]
        align_model.head_B_mu.weight.copy_(last.weight)
        align_model.head_B_mu.bias.copy_(last.bias)

        # head_B_logvar = 0 初始化（学到不确定度）
        nn.init.zeros_(align_model.head_B_logvar.weight)
        nn.init.zeros_(align_model.head_B_logvar.bias)

if __name__ == '__main__':
    # 测试模型是否能正常工作
    # 7个输入特征, 5个输出目标
    model = MLP(input_dim=7, output_dim=5)
    test_input = torch.randn(64, 7) # 模拟一个batch的数据
    output = model(test_input)
    print("模型测试成功！")
    print("输入形状:", test_input.shape)
    print("输出形状:", output.shape)
