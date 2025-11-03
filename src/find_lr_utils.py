# src/find_lr_utils.py (已更新：Finetune 使用 Min Loss / 2 规则)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from ignite.engine import create_supervised_trainer
from ignite.handlers import FastaiLRFinder

from models.align_hetero import AlignHeteroMLP
from data_loader import get_data_and_scalers
from loss_function import heteroscedastic_nll
from optimizer_utils import create_discriminative_optimizer
import config


class _LRFinderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        mu, _, _ = self.model(x)
        return mu


def _save_plot(lrs, losses, suggested_lr, min_loss, title, save_path):
    if not save_path:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    # 标记最低点
    plt.plot(lrs[np.argmin(losses)], min_loss, 'ro', markersize=8,
             label=f'Min Loss at LR={lrs[np.argmin(losses)]:.2e}')
    # 标记建议点
    plt.axvline(x=suggested_lr, color='g', linestyle='--',
                label=f'Suggested LR (Min/2)={suggested_lr:.2e}')
    plt.legend()
    try:
        plt.savefig(save_path)
        print(f"学习率曲线图已保存至: {save_path}")
    except Exception as e:
        print(f"警告: 保存图像失败 - {e}")
    plt.close()


def find_pretrain_lr(
    model_class, model_params, data, end_lr=10.0, num_iter=1000,
    batch_size=128, device="cuda", save_plot_path: str = None
):
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

    # Pretrain 规则：我们仍然使用"下降最快"的规则，这通常比"Min Loss"更稳健
    try:
        suggested_lr = lr_finder.suggested_lr()
        if not suggested_lr:
            raise Exception
        min_loss = losses[np.argmin(losses)]
        print(f"--- [Pretrain] 自动化分析完成 ---")
        print(f"自动化建议 (Steepest Descent 规则): {suggested_lr:.2e}")
    except Exception:
        # 如果 ignite 的 suggest_lr() 失败，回退到 Min Loss 规则
        skip_first = 5
        min_loss_idx = np.argmin(losses[skip_first:]) + skip_first
        min_loss = losses[min_loss_idx]
        suggested_lr = lrs[min_loss_idx]
        print(f"--- [Pretrain] 自动化分析完成 (回退到 Min Loss 规则) ---")
        print(f"最小损失 {min_loss:.4e} 出现在 LR={suggested_lr:.2e}")

    _save_plot(lrs, losses, suggested_lr, min_loss,
               "LR Finder (Pretrain)", save_plot_path)
    del original_model, wrapped_model, optimizer, loader, lr_finder, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return suggested_lr


# src/find_lr_utils.py (替换整个 find_finetune_lr 函数)

def find_finetune_lr(
    model_class, model_params, data, pretrained_weights_path: str,
    lr_hetero: float = config.LEARNING_RATE_HETERO,  # <-- 新增：固定的 Hetero LR
    gap_ratio: float = config.GAP_RATIO,
    internal_ratio: float = config.INTERNAL_RATIO,
    end_lr=10.0, num_iter=1000, batch_size=128, device="cuda", save_plot_path: str = None
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    if not Path(pretrained_weights_path).exists():
        raise FileNotFoundError(f"错误: 未找到预训练模型路径: {pretrained_weights_path}")

    original_model = model_class(**model_params).to(device)
    original_model.load_state_dict(torch.load(
        pretrained_weights_path, map_location=device))

    start_lr = 1e-7  # 这将是 lr_backbone_head 的起始值

    # <<< --- 核心修改：使用 v7 优化器 --- >>>
    print(f"--- [Finetune LR Finder] (v7) ---")
    print(f"    - 正在搜索: lr_backbone_head (从 {start_lr:.1e} 开始)")
    print(f"    - 固定: lr_hetero = {lr_hetero:.2e}")

    optimizer = create_discriminative_optimizer(
        model=original_model,
        lr_backbone_head=start_lr,       # <-- 搜索目标
        lr_hetero=lr_hetero,             # <-- 固定值
        gap_ratio=gap_ratio,
        internal_ratio=internal_ratio,
        weight_decay=1e-4
    )

    def criterion(model_output, y_true): return heteroscedastic_nll(
        model_output[0], model_output[1], y_true)

    X_train, y_train = data['target_train']
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True
    )
    lr_finder = FastaiLRFinder()
    trainer = create_supervised_trainer(
        original_model, optimizer, criterion, device=device,
        output_transform=lambda x, y, y_pred, loss: loss.item()
    )
    to_save = {"model": original_model, "optimizer": optimizer}
    print(f"--- [Finetune] 正在运行 LR Finder (共 {num_iter} 步)... ---")
    with lr_finder.attach(
        trainer, to_save=to_save, end_lr=end_lr, num_iter=num_iter
    ) as trainer_with_lr_finder:
        trainer_with_lr_finder.run(loader, max_epochs=1)

    results = lr_finder.get_results()
    lrs_list_of_lists = results["lr"]
    losses = results["loss"]

    # <<< --- 核心修改：LR 在第 0 组 (Backbone Head) --- >>>
    head_lrs = [lr_pair[0] for lr_pair in lrs_list_of_lists]

    if not head_lrs or not losses or len(head_lrs) < 5:
        print("警告: LRFinder未能产生足够的有效数据。")
        return 1e-4

    skip_first = 5
    min_loss_idx = np.argmin(losses[skip_first:]) + skip_first
    min_loss = losses[min_loss_idx]
    min_loss_lr = head_lrs[min_loss_idx]

    suggested_lr = min_loss_lr / 2.0

    print(f"--- [Finetune] 自动化分析完成 ---")
    print(f"最小损失 {min_loss:.4e} 出现在 LR={min_loss_lr:.2e}")
    print(
        f"自动化建议 (Min Loss / 2 规则): {suggested_lr:.2e} (将用于 lr_backbone_head)")

    _save_plot(head_lrs, losses, suggested_lr, min_loss,
               "LR Finder (Finetune - Backbone Head)", save_plot_path)

    del original_model, optimizer, loader, lr_finder, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 返回的是 lr_backbone_head
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
    test_model_params = {
        'input_dim': test_input_dim,
        'output_dim': test_output_dim,
        'hidden_dims': [128, 256, 256, 512],
        'dropout_rate': 0.2
    }
    print("\n--- 测试 Pretrain Finder ---")
    suggested_lr_pretrain = find_pretrain_lr(
        AlignHeteroMLP, test_model_params, test_data,
        num_iter=1000, save_plot_path="lr_finder_pretrain_test.png"
    )
    print(f"手动测试 (Pretrain) 完成。建议学习率: {suggested_lr_pretrain:.2e}")
    print("\n--- 测试 Finetune Finder ---")
    if not Path(PRETRAINED_PATH).exists():
        print(f"警告: 未找到预训练模型 {PRETRAINED_PATH}，跳过 Finetune 测试。")
    else:
        print(
            f"确保 {PRETRAINED_PATH} 是用 {test_model_params['hidden_dims']} 结构训练的...")
        suggested_lr_finetune = find_finetune_lr(
            AlignHeteroMLP, test_model_params, test_data,
            pretrained_weights_path=PRETRAINED_PATH,
            num_iter=1000, save_plot_path="lr_finder_finetune_test.png"
        )
        print(f"手动测试 (Finetune) 完成。建议学习率: {suggested_lr_finetune:.2e}")
