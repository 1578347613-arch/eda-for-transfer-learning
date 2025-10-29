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
import config

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

def prepare_inverse_dataset(opamp_type, device, 
                            # 新增参数，允许控制验证集来源
                            val_set_source='target_val'): # 可以是 'target_val', 'source_val', or 'none'
    """准备反向训练所需的数据集，分离训练集和验证集"""
    print(f"--- [反向数据准备] opamp: {opamp_type}, val_source: {val_set_source} ---")
    data = get_data_and_scalers(opamp_type=opamp_type)
    
    # 获取所有分割好的数据块
    x_a_tr, y_a_tr = data["source_train"]
    x_a_val, y_a_val = data["source_val"]
    x_b_tr, y_b_tr = data["target_train"]
    x_b_val, y_b_val = data["target_val"]
        
    # 训练集 = source_train + target_train
    x_train = np.vstack([x_a_tr, x_b_tr]).astype(np.float32)
    y_train = np.vstack([y_a_tr, y_b_tr]).astype(np.float32)
    print(f"训练集维度: x={x_train.shape}, y={y_train.shape}")

    # 根据选择确定验证集
    x_val, y_val = None, None
    if val_set_source == 'target_val' and len(x_b_val) > 0:
        x_val, y_val = x_b_val.astype(np.float32), y_b_val.astype(np.float32)
        print(f"使用 target_val 作为验证集: x={x_val.shape}, y={y_val.shape}")
    elif val_set_source == 'source_val' and len(x_a_val) > 0:
        x_val, y_val = x_a_val.astype(np.float32), y_a_val.astype(np.float32)
        print(f"使用 source_val 作为验证集: x={x_val.shape}, y={y_val.shape}")
    else:
        print("[警告] 未指定或找不到有效的验证集，将不进行早停。")

    # 返回 Tensor 和 Scalers
    return (
        torch.from_numpy(y_train).to(device), # 注意：输入是 y
        torch.from_numpy(x_train).to(device), # 目标是 x
        torch.from_numpy(y_val).to(device) if y_val is not None else None, # 验证集 y (输入)
        torch.from_numpy(x_val).to(device) if x_val is not None else None, # 验证集 x (目标)
        data["x_scaler"],
        data["y_scaler"]
    )


def train_mdn(model, train_loader, val_loader, optimizer, epochs, patience, device, save_path,
              # --- 以下是新增的参数，用于保存 config ---
              input_dim, output_dim, args):
    """训练 MDN 模型，加入验证、早停，并正确保存 config"""
    print(f"--- [反向模型] 开始训练 (带早停, patience={patience}) ---")

    best_val_nll = float('inf')
    patience_counter = patience

    for ep in range(1, epochs + 1):
        # --- 训练部分 (保持不变) ---
        model.train()
        total_train_loss = 0.0
        for y_batch, x_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pi, mu, sigma = model(y_batch)
            loss = mdn_nll_loss(pi, mu, sigma, x_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_train_loss += loss.item() * y_batch.size(0)
        avg_train_loss = total_train_loss / len(train_loader.dataset)

        # --- 验证部分 (保持不变) ---
        val_nll = float('inf')
        if val_loader:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for y_batch_val, x_batch_val in val_loader:
                    pi_val, mu_val, sigma_val = model(y_batch_val)
                    val_loss_batch = mdn_nll_loss(pi_val, mu_val, sigma_val, x_batch_val)
                    total_val_loss += val_loss_batch.item() * y_batch_val.size(0)
            val_nll = total_val_loss / len(val_loader.dataset)

        # --- 打印日志 (保持不变) ---
        log_msg = f"[MDN][Epoch {ep:04d}/{epochs}] Train NLL: {avg_train_loss:.4f}"
        if val_loader:
            log_msg += f" | Val NLL: {val_nll:.4f}"
        print(log_msg)

        # --- 早停逻辑与模型保存 ---
        # ***** 核心修改：确保两个 torch.save 都包含 config *****
        save_content = {
            "state_dict": model.state_dict(),
            "config": {
                "opamp_type": args.opamp,
                "input_dim": input_dim,
                "output_dim": output_dim,
                "n_components": args.mdn_components,
                "hidden_dim": args.mdn_hidden_dim,
                "num_layers": args.mdn_num_layers,
            }
        }

        if val_loader: # 有验证集时，根据验证集表现保存
            if val_nll < best_val_nll:
                best_val_nll = val_nll
                torch.save(save_content, save_path) # <-- 保存完整内容
                patience_counter = patience
                print(f"  -> Val NLL improved to {best_val_nll:.4f}. Model saved to {save_path.name}")
            else:
                patience_counter -= 1
                if patience_counter == 0:
                    print(f"  -> Val NLL did not improve for {patience} epochs. Early stopping.")
                    break
        elif ep == epochs: # 没有验证集时，在最后一轮保存
             torch.save(save_content, save_path) # <-- 保存完整内容
             print(f"  -> Reached max epochs. Model saved to {save_path.name}")
        # ***** 结束修改 *****

    print(f"--- [反向模型] 训练结束 ---")

# ==============================================================================
#  流水线接口 (结构与 unified_train.py 类似)
# ==============================================================================

# src/unified_inverse_train.py (只修改 setup_args 函数)

# src/unified_inverse_train.py (只修改 setup_args 函数)

# "黄金标准" setup_args 函数
# 请在 unified_train.py 和 unified_inverse_train.py 中使用它

def setup_args():
    """设置和解析命令行参数 (反向 MDN 专属版)"""
    parser = argparse.ArgumentParser(description="统一的反向 MDN 训练脚本 (专属参数版)")

    # --- 核心参数 (读取 _INV 后缀的默认值) ---
    parser.add_argument("--opamp", type=str,
                        default=config.OPAMP_TYPE_INV, help="反向设计处理的运放类型")
    parser.add_argument("--device", type=str,
                        default=config.DEVICE_INV, help="反向设计使用的设备 'cuda' or 'cpu'")
    parser.add_argument("--restart", action='store_true',
                        default=config.RESTART_INV, help="强制重新执行训练 (即使模型已存在)")
    parser.add_argument("--save_path", type=str,
                        default=config.SAVE_PATH_INV, help="反向模型和 scaler 的存放地址")
    parser.add_argument("--seed", type=int,
                        default=config.SEED_INV, help="反向训练的随机种子")

    # --- MDN 模型与训练参数 (读取 MDN_ 前缀的默认值) ---
    parser.add_argument("--mdn_epochs", type=int,
                        default=config.MDN_EPOCHS, help="MDN 训练轮数")
    parser.add_argument("--mdn_patience", type=int,
                        default=config.MDN_PATIENCE, help="MDN 早停耐心轮数")
    parser.add_argument("--mdn_components", type=int,
                        default=config.MDN_COMPONENTS, help="MDN 高斯混合成分数量")
    parser.add_argument("--mdn_hidden_dim", type=int,
                        default=config.MDN_HIDDEN_DIM, help="MDN 隐藏层维度")
    parser.add_argument("--mdn_num_layers", type=int,
                        default=config.MDN_NUM_LAYERS, help="MDN 隐藏层数量")
    parser.add_argument("--mdn_batch_size", type=int,
                        default=config.MDN_BATCH_SIZE, help="MDN 训练批次大小")
    parser.add_argument("--mdn_lr", type=float,
                        default=config.MDN_LR, help="MDN 学习率")
    # 如果 config.py 里加了 MDN_WEIGHT_DECAY，这里也要加上对应的 argument
    parser.add_argument("--mdn_weight_decay", type=float, default=config.MDN_WEIGHT_DECAY, help="MDN 权重衰减")

    args = parser.parse_args()
    return args


def main():
    args = setup_args() # <-- 假设 setup_args 已经修复好了
    DEVICE = torch.device(args.device)
    
    # 设置随机种子 (保持不变)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- 1. 准备数据和路径 ---
    # ***** 修改：调用新的 prepare_inverse_dataset *****
    y_train_tensor, x_train_tensor, y_val_tensor, x_val_tensor, x_scaler, y_scaler = \
        prepare_inverse_dataset(args.opamp, DEVICE, val_set_source='target_val') # <-- 使用 target_val
    # ***** 结束修改 *****

    input_dim = y_train_tensor.shape[1]
    output_dim = x_train_tensor.shape[1]
    
    save_dir = Path(args.save_path)
    save_dir.mkdir(exist_ok=True)
    # ***** 修改：直接传递 save_dir / ... 给 train_mdn *****
    model_path = save_dir / f"mdn_{args.opamp}.pth" 
    # ***** 结束修改 *****
    
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
    
    # ***** 修改：从 args 读取 mdn_weight_decay (如果 setup_args 定义了它) *****
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.mdn_lr, 
        # weight_decay=args.mdn_weight_decay # 如果 setup_args 定义了就取消注释
    )
    # ***** 结束修改 *****
    
    # ***** 修改：创建 train 和 val loaders *****
    train_dataset = TensorDataset(y_train_tensor, x_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.mdn_batch_size, shuffle=True)
    
    val_loader = None
    if y_val_tensor is not None and x_val_tensor is not None:
        val_dataset = TensorDataset(y_val_tensor, x_val_tensor)
        # 验证时 batch_size 可以大一点，并且不需要 shuffle
        val_loader = DataLoader(val_dataset, batch_size=args.mdn_batch_size * 2, shuffle=False) 
    # ***** 结束修改 *****

    # --- 3. 训练 ---
    if args.restart or not model_path.exists():
        # ***** 修改：把 model_path 和 patience 传给 train_mdn *****
        train_mdn(model, train_loader, val_loader, optimizer, 
                  args.mdn_epochs, args.mdn_patience, DEVICE, model_path,
                  input_dim, output_dim, args) # <-- 传入 patience 和 save_path
        # ***** 结束修改 *****
        
        # --- 4. 保存模型和元数据 (这部分逻辑移到 train_mdn 内部保存最佳模型时执行更佳) ---
        # (我们暂时保留这里的 scaler 保存，但模型保存已在 train_mdn 完成)
        print(f"[MDN] 训练完成，最佳模型应已保存到: {model_path}")

        # 保存scalers和元信息 (保持不变)
        x_scaler_path = save_dir / f"{args.opamp}_x_scaler.gz"
        y_scaler_path = save_dir / f"{args.opamp}_y_scaler.gz"
        joblib.dump(x_scaler, x_scaler_path)
        joblib.dump(y_scaler, y_scaler_path)
        
        meta = { # (元信息可以保持不变)
            "opamp": args.opamp,
            "model_path": str(model_path.resolve()),
            "x_scaler": str(x_scaler_path.resolve()),
            "y_scaler": str(y_scaler_path.resolve()),
            # 可以考虑把训练结束时的 best_val_nll 也存进来
        }
        meta_path = model_path.with_suffix(".json")
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"[MDN] Scalers和元信息已保存。")
            
    else:
        print(f"--- [反向模型] 跳过训练，模型已存在: {model_path} ---")

if __name__ == "__main__":
    main()