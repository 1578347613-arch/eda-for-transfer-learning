# src/find_lr_utils.py (已修正 list.__format__ TypeError)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 导入 ignite 模块
from ignite.engine import create_supervised_trainer
from ignite.handlers import FastaiLRFinder

# 从您的项目模块中导入
from models.align_hetero import AlignHeteroMLP
from data_loader import get_data_and_scalers
from loss_function import heteroscedastic_nll

# <<< --- 模型包装器 (不变) --- >>>


class _LRFinderWrapper(nn.Module):
    # ... (代码不变) ...
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        mu, _, _ = self.model(x)
        return mu

# <<< --- 绘图辅助函数 (不变) --- >>>


def _save_plot(lrs, losses, suggested_lr, min_loss, title, save_path):
    # ... (代码不变) ...
    if not save_path:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.plot(suggested_lr, min_loss, 'ro', markersize=8,
             label=f'Min Loss at LR={suggested_lr:.2e}')
    plt.legend()
    try:
        plt.savefig(save_path)
        print(f"学习率曲线图已保存至: {save_path}")
    except Exception as e:
        print(f"警告: 保存图像失败 - {e}")
    plt.close()

# <<< --- Pretrain 函数 (不变) --- >>>


def find_pretrain_lr(
    model_class,
    model_params,
    data,
    end_lr=10.0,
    num_iter=1000,
    batch_size=128,
    device="cuda",
    save_plot_path: str = None
):
    # ... (此函数所有代码均不变) ...
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    original_model = model_class(**model_params).to(device)
    wrapped_model = _LRFinderWrapper(original_model)
    optimizer = torch.optim.AdamW(
        original_model.backbone.parameters(), lr=1e-7)
    criterion = nn.HuberLoss()
    X_train, y_train = data['source_train']
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True
    )
    lr_finder = FastaiLRFinder()
    trainer = create_supervised_trainer(
        wrapped_model, optimizer, criterion, device=device,
        output_transform=lambda x, y, y_pred, loss: loss.item()
    )
    to_save = {"model": original_model, "optimizer": optimizer}
    print(f"--- [Pretrain] 正在运行 LR Finder (共 {num_iter} 步)... ---")
    with lr_finder.attach(
        trainer, to_save=to_save, end_lr=end_lr, num_iter=num_iter
    ) as trainer_with_lr_finder:
        trainer_with_lr_finder.run(loader, max_epochs=1)
    results = lr_finder.get_results()
    lrs = results["lr"]
    losses = results["loss"]
    if not lrs or not losses or len(lrs) < 5:
        print("警告: LRFinder未能产生足够的有效数据。")
        return 3e-4
    skip_first = 5
    min_loss_idx = np.argmin(losses[skip_first:]) + skip_first
    min_loss = losses[min_loss_idx]
    suggested_lr = lrs[min_loss_idx]
    print(f"--- [Pretrain] 自动化分析完成 ---")
    print(f"最小损失 {min_loss:.4e} 出现在 LR={suggested_lr:.2e}")
    print(f"自动化建议 (Min Loss 规则): {suggested_lr:.2e}")
    _save_plot(lrs, losses, suggested_lr, min_loss,
               "LR Finder (Pretrain)", save_plot_path)
    del original_model, wrapped_model, optimizer, loader, lr_finder, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return suggested_lr

# <<< --- Finetune 函数 (已修正) --- >>>


def find_finetune_lr(
    model_class,
    model_params,
    data,
    pretrained_weights_path: str,
    end_lr=10.0,
    num_iter=1000,
    batch_size=128,
    device="cuda",
    save_plot_path: str = None
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    if not Path(pretrained_weights_path).exists():
        raise FileNotFoundError(f"错误: 未找到预训练模型路径: {pretrained_weights_path}")

    original_model = model_class(**model_params).to(device)
    original_model.load_state_dict(torch.load(
        pretrained_weights_path, map_location=device))

    # <<< --- 您的正确修改：使用10:1差分优化器 --- >>>
    start_lr = 1e-7
    optimizer_params = [
        {"params": original_model.backbone.parameters(), "lr": start_lr / 10},
        {"params": original_model.hetero_head.parameters(), "lr": start_lr}
    ]
    optimizer = torch.optim.AdamW(optimizer_params)
    # --- 修改结束 ---

    def criterion(model_output, y_true): return heteroscedastic_nll(
        model_output[0], model_output[1], y_true
    )
    X_train, y_train = data['target_train']
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True
    )
    lr_finder = FastaiLRFinder()

    # <<< --- 必要的Bug修复：修正 output_transform --- >>>
    trainer = create_supervised_trainer(
        original_model, optimizer, criterion, device=device,
        output_transform=lambda x, y, y_pred, loss: loss.item()
    )
    # --- 修复结束 ---

    to_save = {"model": original_model, "optimizer": optimizer}
    print(f"--- [Finetune] 正在运行 LR Finder (共 {num_iter} 步)... ---")
    with lr_finder.attach(
        trainer, to_save=to_save, end_lr=end_lr, num_iter=num_iter
    ) as trainer_with_lr_finder:
        trainer_with_lr_finder.run(loader, max_epochs=1)

    results = lr_finder.get_results()

    # <<< --- 核心修正：处理 LRs 列表 --- >>>
    # 这是 [[lr_b1, lr_h1], [lr_b2, lr_h2], ...]
    lrs_list_of_lists = results["lr"]
    # 这是 [loss1, loss2, ...] (已由 output_transform 修复)
    losses = results["loss"]

    # 我们只关心 head 的学习率，将其提取出来
    head_lrs = [lr_pair[1] for lr_pair in lrs_list_of_lists]
    # --- 修正结束 ---

    if not head_lrs or not losses or len(head_lrs) < 5:
        print("警告: LRFinder未能产生足够的有效数据。")
        return 1e-4

    skip_first = 5
    min_loss_idx = np.argmin(losses[skip_first:]) + skip_first
    min_loss = losses[min_loss_idx]
    suggested_lr = head_lrs[min_loss_idx]  # 从 head_lrs 列表中获取

    print(f"--- [Finetune] 自动化分析完成 ---")
    # 这里的 min_loss 和 suggested_lr 现在都是 float，不会再报错
    print(f"最小损失 {min_loss:.4e} 出现在 LR={suggested_lr:.2e}")
    print(f"自动化建议 (Min Loss 规则): {suggested_lr:.2e}")

    # <<< --- 核心修正：绘图时使用 head_lrs --- >>>
    _save_plot(head_lrs, losses, suggested_lr, min_loss,
               "LR Finder (Finetune)", save_plot_path)
    # --- 修正结束 ---

    del original_model, optimizer, loader, lr_finder, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return suggested_lr


# --- 手动调试入口 (不变) ---
if __name__ == '__main__':
    # ... (这部分代码无需修改) ...
    print("正在手动测试 find_lr_utils 函数...")

    OPAMP_TYPE = '5t_opamp'
    PRETRAINED_PATH = f"../results/{OPAMP_TYPE}_pretrained.pth"

    test_data = get_data_and_scalers(opamp_type=OPAMP_TYPE)
    test_input_dim = test_data['source'][0].shape[1]
    test_output_dim = test_data['source'][1].shape[1]

    # 确保这个结构与 PRETRAINED_PATH 匹配！
    test_model_params = {
        'input_dim': test_input_dim,
        'output_dim': test_output_dim,
        'hidden_dims': [128, 256, 512],
        'dropout_rate': 0.2
    }

    print("\n--- 测试 Pretrain Finder ---")
    suggested_lr_pretrain = find_pretrain_lr(
        AlignHeteroMLP,
        test_model_params,
        test_data,
        num_iter=1000,
        save_plot_path="lr_finder_pretrain_test.png"
    )
    print(f"手动测试 (Pretrain) 完成。建议学习率: {suggested_lr_pretrain:.2e}")

    print("\n--- 测试 Finetune Finder ---")
    if not Path(PRETRAINED_PATH).exists():
        print(f"警告: 未找到预训练模型 {PRETRAINED_PATH}，跳过 Finetune 测试。")
    else:
        print(
            f"确保 {PRETRAINED_PATH} 是用 {test_model_params['hidden_dims']} 结构训练的...")
        suggested_lr_finetune = find_finetune_lr(
            AlignHeteroMLP,
            test_model_params,
            test_data,
            pretrained_weights_path=PRETRAINED_PATH,
            num_iter=1000,
            save_plot_path="lr_finder_finetune_test.png"
        )
        print(f"手动测试 (Finetune) 完成。建议学习率: {suggested_lr_finetune:.2e}")
