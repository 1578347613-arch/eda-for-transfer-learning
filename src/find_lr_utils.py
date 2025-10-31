# src/find_lr_utils.py (已修正)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch_lr_finder import LRFinder
import matplotlib.pyplot as plt

# 从您的项目模块中导入
from models.align_hetero import AlignHeteroMLP
from data_loader import get_data_and_scalers

# <<< --- 核心改动 1: 创建一个临时的模型包装器 --- >>>


class _LRFinderWrapper(nn.Module):
    """一个简单的包装器，确保模型的forward方法只返回mu，以适配HuberLoss。"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # 我们只关心预训练阶段用于HuberLoss的均值'mu'
        mu, _, _ = self.model(x)
        return mu


def find_pretrain_lr(
    model_class,
    model_params,
    data,
    end_lr=1,
    num_iter=100,
    batch_size=128,
    device="cuda"
):
    """
    使用 torch-lr-finder 为预训练阶段寻找一个建议的学习率。
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # 1. 实例化原始模型
    original_model = model_class(**model_params).to(device)

    # <<< --- 核心改动 2: 使用包装器 --- >>>
    # 将原始模型放入包装器中
    wrapped_model = _LRFinderWrapper(original_model)

    # 2. 准备预训练所需的数据、优化器和损失函数
    X_train, y_train = data['source_train']
    # 优化器仍然需要优化原始模型的参数
    optimizer = torch.optim.AdamW(
        original_model.backbone.parameters(), lr=1e-7)
    criterion = nn.HuberLoss()

    loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True
    )

    # 3. 初始化并运行 LRFinder，传入包装后的模型
    lr_finder = LRFinder(wrapped_model, optimizer, criterion, device=device)
    lr_finder.range_test(loader, end_lr=end_lr, num_iter=num_iter)

    # 4. 获取建议的学习率
    _, suggested_lr = lr_finder.plot(suggest_lr=True)
    plt.close()

    # 5. 清理内存并返回结果
    del original_model, wrapped_model, optimizer, loader, lr_finder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if suggested_lr is None:
        print("警告: LRFinder未能找到建议值，将使用默认值 3e-4。")
        return 3e-4

    return suggested_lr


# --- 手动调试入口 (不变) ---
if __name__ == '__main__':
    print("正在手动测试 find_pretrain_lr 函数...")

    test_data = get_data_and_scalers(opamp_type='5t_opamp')
    test_input_dim = test_data['source'][0].shape[1]
    test_output_dim = test_data['source'][1].shape[1]

    test_model_params = {
        'input_dim': test_input_dim,
        'output_dim': test_output_dim,
        'hidden_dims': [256, 256, 256, 256],
        'dropout_rate': 0.2
    }

    suggested_lr = find_pretrain_lr(
        AlignHeteroMLP, test_model_params, test_data)
    print(f"\n手动测试完成。找到的建议预训练学习率: {suggested_lr:.2e}")
