# src/unified_inverse_train.py

import os
import argparse
import json
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# --- 从项目模块中导入 ---
from data_loader import get_data_and_scalers
from config import COMMON_CONFIG, TASK_CONFIGS

# ==============================================================================
#  核心逻辑 (大部分从你的原始脚本复制而来，但移除了对旧config的依赖)
# ==============================================================================

class InverseMDN(nn.Module):
    def __init__(self, input_dim, output_dim, n_components, hidden_dim, num_layers):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        self.backbone = nn.Sequential(*layers)
        self.pi = nn.Linear(hidden_dim, n_components)
        self.mu = nn.Linear(hidden_dim, n_components * output_dim)
        self.sigma_raw = nn.Linear(hidden_dim, n_components * output_dim)
        self.n_components = n_components
        self.output_dim = output_dim
        self.softplus = nn.Softplus()

    def forward(self, y):
        h = self.backbone(y)
        pi = torch.softmax(self.pi(h), dim=-1)
        mu = self.mu(h).view(-1, self.n_components, self.output_dim)
        sigma = self.softplus(self.sigma_raw(h)).view(-1, self.n_components, self.output_dim) + 1e-6
        return pi, mu, sigma

def mdn_nll_loss(pi, mu, sigma, target_x):
    B, K, D = mu.shape
    target = target_x.unsqueeze(1).expand(B, K, D)
    log_prob = -0.5 * torch.sum(((target - mu) / sigma) ** 2 + 2 * torch.log(sigma) + np.log(2 * np.pi), dim=2)
    log_mix = torch.logsumexp(torch.log(pi + 1e-9) + log_prob, dim=1)
    return -torch.mean(log_mix)

def prepare_inverse_dataset(opamp_type, device):
    data = get_data_and_scalers(opamp_type=opamp_type)
    x_a, y_a = data["source"]
    x_b_tr, y_b_tr = data["target_train"]
    x_b_val, y_b_val = data["target_val"]
    
    x_all = np.vstack([x_a, x_b_tr, x_b_val]).astype(np.float32)
    y_all = np.vstack([y_a, y_b_tr, y_b_val]).astype(np.float32)

    return (
        torch.from_numpy(y_all).to(device),
        torch.from_numpy(x_all).to(device),
        data["x_scaler"],
        data["y_scaler"]
    )

def train_mdn(model, dataloader, optimizer, epochs, device):
    print(f"--- [反向模型] 开始训练 ---")
    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        for y_batch, x_batch in dataloader:
            optimizer.zero_grad(set_to_none=True)
            pi, mu, sigma = model(y_batch)
            loss = mdn_nll_loss(pi, mu, sigma, x_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * y_batch.size(0)
        
        avg_loss = total_loss / len(dataloader.dataset)
        if ep % 50 == 0 or ep == epochs:
            print(f"[MDN][Epoch {ep:04d}/{epochs}] NLL: {avg_loss:.4f}")

# ==============================================================================
#  流水线接口 (结构与 unified_train.py 类似)
# ==============================================================================

# src/unified_inverse_train.py (只修改 setup_args 函数)

# src/unified_inverse_train.py (只修改 setup_args 函数)

# "黄金标准" setup_args 函数
# 请在 unified_train.py 和 unified_inverse_train.py 中使用它

def setup_args():
    parser = argparse.ArgumentParser(description="统一训练脚本")
    parser.add_argument("--opamp", type=str, required=True, choices=TASK_CONFIGS.keys(), help="电路类型")

    # --- 1. 自动添加所有可能的参数定义 ---
    # a. 从 COMMON_CONFIG 添加
    for key, value in COMMON_CONFIG.items():
        if isinstance(value, bool):
            if value is False:
                parser.add_argument(f"--{key}", action="store_true", help=f"启用 '{key}' (开关)")
            else:
                parser.add_argument(f"--no-{key}", action="store_false", dest=key, help=f"禁用 '{key}' (开关)")
        else:
            parser.add_argument(f"--{key}", type=type(value), help=f"设置 '{key}'")

    # b. 从 TASK_CONFIGS 添加专属参数
    all_task_keys = set().union(*(d.keys() for d in TASK_CONFIGS.values()))
    task_only_keys = all_task_keys - set(COMMON_CONFIG.keys())
    
    for key in sorted(list(task_only_keys)):
        # 简单的类型推断
        sample_val = TASK_CONFIGS[next(iter(TASK_CONFIGS))][key]
        parser.add_argument(f"--{key}", type=type(sample_val), help=f"任务参数: {key}")

    # --- 2. 应用默认值并最终解析 ---
    # a. 先应用通用默认值
    parser.set_defaults(**COMMON_CONFIG)
    
    # b. 解析一次，拿到 opamp 类型，再应用任务专属默认值
    # parse_known_args 不会因为不完整的命令行而报错
    temp_args, _ = parser.parse_known_args()
    if temp_args.opamp in TASK_CONFIGS:
        parser.set_defaults(**TASK_CONFIGS[temp_args.opamp])
        
    # c. 最后重新解析，命令行提供的值会覆盖所有默认值
    args = parser.parse_args()
    
    return args


def main():
    args = setup_args()
    DEVICE = torch.device(args.device)
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- 1. 准备数据和路径 ---
    y_tensor, x_tensor, x_scaler, y_scaler = prepare_inverse_dataset(args.opamp, DEVICE)
    input_dim = y_tensor.shape[1]
    output_dim = x_tensor.shape[1]
    
    save_dir = Path(args.save_path)
    save_dir.mkdir(exist_ok=True)
    model_path = save_dir / f"mdn_{args.opamp}.pth"
    
    print(f"--- 任务: {args.opamp} | 设备: {DEVICE} ---")
    print(f"--- 动态检测到维度: Input(y)={input_dim}, Output(x)={output_dim} ---")

    # --- 2. 初始化模型和优化器 ---
    model = InverseMDN(
        input_dim=input_dim,
        output_dim=output_dim,
        n_components=args.mdn_components,
        hidden_dim=args.mdn_hidden_dim,
        num_layers=args.mdn_num_layers
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.mdn_lr, 
        weight_decay=args.mdn_weight_decay
    )
    
    dataset = TensorDataset(y_tensor, x_tensor)
    dataloader = DataLoader(dataset, batch_size=args.mdn_batch_size, shuffle=True)

    # --- 3. 训练 ---
    if args.restart or not model_path.exists():
        train_mdn(model, dataloader, optimizer, args.mdn_epochs, DEVICE)
        
        # --- 4. 保存模型和元数据 ---
        torch.save({
            "state_dict": model.state_dict(),
            "config": {
                "opamp_type": args.opamp,
                "input_dim": input_dim,
                "output_dim": output_dim,
                "n_components": args.mdn_components,
                "hidden_dim": args.mdn_hidden_dim,
                "num_layers": args.mdn_num_layers,
            }
        }, model_path)
        print(f"[MDN] 模型已保存到: {model_path}")

        # 保存scalers和元信息，供采样脚本使用
        x_scaler_path = save_dir / f"{args.opamp}_x_scaler.gz"
        y_scaler_path = save_dir / f"{args.opamp}_y_scaler.gz"
        joblib.dump(x_scaler, x_scaler_path)
        joblib.dump(y_scaler, y_scaler_path)
        
        meta = {
            "opamp": args.opamp,
            "model_path": str(model_path.resolve()),
            "x_scaler": str(x_scaler_path.resolve()),
            "y_scaler": str(y_scaler_path.resolve()),
        }
        meta_path = model_path.with_suffix(".json")
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"[MDN] Scalers和元信息已保存。")
        
    else:
        print(f"--- [反向模型] 跳过训练，模型已存在: {model_path} ---")

if __name__ == "__main__":
    main()
