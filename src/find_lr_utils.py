# src/find_lr_utils.py (已更新为使用 "Min Loss" 启发式算法)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np  # <<< --- 需要 Numpy

# 导入 ignite 模块
from ignite.engine import create_supervised_trainer
from ignite.handlers import FastaiLRFinder

# 从您的项目模块中导入
from models.align_hetero import AlignHeteroMLP
from data_loader import get_data_and_scalers

# <<< --- 模型包装器 (不变) --- >>>


class _LRFinderWrapper(nn.Module):
    """一个简单的包装器，确保模型的forward方法只返回mu，以适配HuberLoss。"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        mu, _, _ = self.model(x)
        return mu


def find_pretrain_lr(
    model_class,
    model_params,
    data,
    end_lr=1.0,
    num_iter=1000,  # <<< --- 确保 num_iter 足够高以获得平滑曲线
    batch_size=128,
    device="cuda"
):
    """
    使用 pytorch-ignite 找到 "损失最低点" 对应的学习率。
    (根据实验验证，这是最佳策略)
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # 1. 实例化和准备
    original_model = model_class(**model_params).to(device)
    wrapped_model = _LRFinderWrapper(original_model)
    optimizer = torch.optim.AdamW(
        original_model.backbone.parameters(), lr=1e-7)
    criterion = nn.HuberLoss()

    X_train, y_train = data['source_train']

    loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True
    )

    # 2. 初始化 LRFinder 处理器
    lr_finder = FastaiLRFinder()

    # 3. 创建 ignite "trainer" 引擎
    trainer = create_supervised_trainer(
        wrapped_model, optimizer, criterion, device=device)

    # 4. 附加并运行 LRFinder
    to_save = {"model": original_model, "optimizer": optimizer}

    print(f"--- 正在运行 LR Finder (共 {num_iter} 步)... ---")
    with lr_finder.attach(
        trainer,
        to_save=to_save,
        end_lr=end_lr,
        num_iter=num_iter
    ) as trainer_with_lr_finder:
        trainer_with_lr_finder.run(loader, max_epochs=1)

    # <<< --- 核心改动: 实现 "Min Loss" 策略 --- >>>

    # 5. 获取平滑后的 LR 和 Loss 数据
    results = lr_finder.get_results()
    lrs = results["lr"]
    losses = results["loss"]

    # 6. 安全检查
    if not lrs or not losses or len(lrs) < 5:
        print("警告: LRFinder未能产生足够的有效数据。")
        return 3e-4  # 返回一个安全的默认值

    # 7. 找到最小损失的索引 (跳过前几个不稳定的点)
    skip_first = 5
    min_loss_idx = np.argmin(losses[skip_first:]) + skip_first
    suggested_lr = lrs[min_loss_idx]  # <-- 直接使用这个LR

    print(f"\n--- 自动化分析完成 ---")
    print(f"最小损失 {losses[min_loss_idx]:.4e} 出现在 LR={suggested_lr:.2e}")
    print(f"自动化建议 (Min Loss 规则): {suggested_lr:.2e}")

    # 9. 清理并返回建议值
    del original_model, wrapped_model, optimizer, loader, lr_finder, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return suggested_lr


# --- 手动调试入口 (不变) ---
if __name__ == '__main__':
    print("正在手动测试 find_pretrain_lr 函数...")

    test_data = get_data_and_scalers(opamp_type='5t_opamp')
    X_train, y_train = test_data['source_train']  # <-- 提前解包以获取数据

    test_input_dim = test_data['source'][0].shape[1]
    test_output_dim = test_data['source'][1].shape[1]

    test_model_params = {
        'input_dim': test_input_dim,
        'output_dim': test_output_dim,
        'hidden_dims': [256, 256, 256, 256],
        'dropout_rate': 0.2
    }

    suggested_lr = find_pretrain_lr(
        AlignHeteroMLP,
        test_model_params,
        test_data,
        num_iter=1000
    )
    print(f"\n手动测试完成。找到的自动化建议学习率: {suggested_lr:.2e}")
