# fused_project/src/unified_inverse_train.py (C3 融合版 v2 - 终极版！)
#
# 基础：来自 C2 (您发的 [2025-11-10_07:01:21] 版本)
# 手术：
#   1. 彻底替换 setup_args()，使其 100% 兼容 C3 "混合圣经"！
#   2. 替换 prepare_inverse_dataset() 为 C1 的“更优”逻辑 (训练所有数据)！
#   3. 修改 main() 来调用新函数！
#
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
# ⬇️ (C3 手术) 我们现在导入 C3 的“混合圣经”
import config 

# ==============================================================================
#  核心逻辑 (来自 C2，因为 C2 的 MDN 模型更强！)
# ==============================================================================

class InverseMDN(nn.Module):
    """(100% 保持 C2 的 5 层模型定义)"""
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
    """(100% 保持 C2 的 NLL 损失定义)"""
    B, K, D = mu.shape
    target = target_x.unsqueeze(1).expand(B, K, D)
    log_prob = -0.5 * torch.sum(((target - mu) / sigma) ** 2 + 2 * torch.log(sigma) + np.log(2 * np.pi), dim=2)
    log_mix = torch.logsumexp(torch.log(pi + 1e-9) + log_prob, dim=1)
    return -torch.mean(log_mix)

# ==========================================================
# ========== 👨‍⚕️ "C3 融合手术" 核心 1 👨‍⚕️ ==========
# ==========================================================
def prepare_inverse_dataset(opamp_type, device):
    """
    [C3 融合版] (逻辑来自 C1，因为它更优！)
    它会加载 A 域全集 + B 域 Tain/Val，让 MDN 在所有可用数据上训练！
    """
    data = get_data_and_scalers(opamp_type=opamp_type)
    
    # (我们假设 C3 会使用 C1 的 data_loader.py)
    x_a, y_a = data["source"]
    x_b_tr, y_b_tr = data["target_train"]
    x_b_val, y_b_val = data["target_val"]
    
    # 堆叠所有数据！(A-full + B-train + B-val)
    x_all = np.vstack([x_a, x_b_tr, x_b_val]).astype(np.float32)
    y_all = np.vstack([y_a, y_b_tr, y_b_val]).astype(np.float32)

    print(f"✅ [C3 MDN 数据] 成功堆叠所有数据 (Total: {len(x_all)} points)")

    return (
        torch.from_numpy(y_all).to(device),
        torch.from_numpy(x_all).to(device),
        data["x_scaler"],
        data["y_scaler"]
    )

def train_mdn(model, dataloader, optimizer, epochs, device):
    """(100% 保持 C2 的训练循环)"""
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

# ==========================================================
# ========== 👨‍⚕️ "C3 融合手术" 核心 2 👨‍⚕️ ==========
# ==========================================================

def setup_args():
    """
    [C3 融合版] 智能 setup_args (逻辑来自 C1 train_align_hetero.py)：
    它现在会读取 C3 "混合圣经" config.py 里的 TASK_CONFIGS 字典，
    并且只读取“黄金反向参数” ('mdn_...')！
    """
    parser = argparse.ArgumentParser(description="统一的反向 MDN 训练脚本 (C3 融合版)")
    parser.add_argument("--opamp", type=str, required=True,
                        choices=config.TASK_CONFIGS.keys(), help="必须指定的电路类型")

    # Step 1: 先解析出 opamp 类型
    temp_args, other_args = parser.parse_known_args()
    opamp_type = temp_args.opamp

    # Step 2: 合并 C3 "混合圣经" 的配置作为默认值
    # (COMMON_CONFIG + TASK_CONFIGS[opamp])
    defaults = {**config.COMMON_CONFIG, **config.TASK_CONFIGS.get(opamp_type, {})}

    # Step 3: 动态为所有简单类型的默认参数创建命令行开关
    for key, value in defaults.items():
        # (我们只关心 MDN 和 通用参数)
        if not key.startswith('mdn_') and key not in config.COMMON_CONFIG:
            continue # 跳过 C1 的正向参数 (hidden_dims, lr_pretrain...)
            
        if isinstance(value, (list, dict)):
            continue  # 跳过 C1 的 PRETRAIN_SCHEDULER_CONFIGS
        
        if isinstance(value, bool):
            parser.add_argument(
                f"--{key}", action=argparse.BooleanOptionalAction, help=f"开关 '{key}'")
        else:
            parser.add_argument(
                f"--{key}", type=type(value), help=f"设置 '{key}'")

    # Step 4: 将合并后的配置设置为解析器的默认值并进行最终解析
    parser.set_defaults(**defaults)
    args = parser.parse_args()
    
    return args

# ==========================================================
# ========== 👨‍⚕️ "C3 融合手术" 核心 3 👨‍⚕️ ==========
# ==========================================================

def main():
    args = setup_args() # <-- (C3 手术) 调用我们新的 C3 setup_args()
    DEVICE = torch.device(args.device)
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- 1. 准备数据和路径 ---
    # ⬇️ (C3 手术) 调用我们 C1 逻辑的 C3 prepare_inverse_dataset()
    y_tensor, x_tensor, x_scaler, y_scaler = prepare_inverse_dataset(args.opamp, DEVICE)
    input_dim = y_tensor.shape[1]
    output_dim = x_tensor.shape[1]
    
    # ⬇️ (C3 手术) 路径来自 C3 "混合圣经" 的 COMMON_CONFIG
    save_dir = Path(args.save_path) 
    save_dir.mkdir(exist_ok=True)
    model_path = save_dir / f"mdn_{args.opamp}.pth"
    
    print(f"--- [C3 黄金反向训练] 任务: {args.opamp} | 设备: {DEVICE} ---")
    print(f"--- 动态检测到维度: Input(y)={input_dim}, Output(x)={output_dim} ---")

    # --- 2. 初始化模型和优化器 ---
    # ⬇️ (C3 手术) 
    #    这里 100% 使用了 C3 "混合圣经" 里的“黄金反向参数”！
    #    (args.mdn_... 全部来自 C3 config.py 里的 TASK_CONFIGS！)
    print(f"✅ [C3] 正在构建 C2 黄金反向 MDN (L={args.mdn_num_layers}, H={args.mdn_hidden_dim}, K={args.mdn_components})...")
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
        # ⬇️ (C3 手术) 调用 C2 的训练循环
        train_mdn(model, dataloader, optimizer, args.mdn_epochs, DEVICE)
        
        # --- 4. 保存模型和元数据 ---
        print(f"[MDN] 正在保存模型和元数据...")
        torch.save({
            "state_dict": model.state_dict(),
            # ⬇️ (C3 手术) 
            #    我们将 C3 "混合圣经" 里的黄金参数保存到 .pth 里！
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

        # (C2 的 scaler 保存逻辑，100% 保留)
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