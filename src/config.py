# src/config.py (C3 融合版 - 终极圣经)
# 结合了 C1 的“黄金正向参数”和 C2 的“黄金反向参数”

import torch

# --- C1 的通用配置 (保留) ---
COMMON_CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'restart': False,
    'evaluate': True,
    'save_path': 'results', # 统一保存到 'src/results'
    'seed': 42
}

# --- C1 的 LOG 变换列 (保留，C1 C2 兼容) ---
LOG_TRANSFORMED_COLS = [
    "ugf",
    "cmrr",
    "dc_gain",
    "slewrate_pos",
]

# --- 核心：C1 和 C2 的“终极融合”字典！ ---
TASK_CONFIGS = {
    '5t_opamp': {
        
        # ==========================================================
        # ===== 1. C1 的“黄金正向参数” (100% 保留!) =====
        # ==========================================================
        
        # --- 训练设置 ---
        'RESTART_PRETRAIN': 6,
        'PRETRAIN_SCHEDULER_CONFIGS': [  # C1 的“Best-of-N”策略
            {"T_0": 90, "T_mult": 1, "epochs_pretrain": 90},
            {"T_0": 100, "T_mult": 1, "epochs_pretrain": 100},
            {"T_0": 110, "T_mult": 1, "epochs_pretrain": 110},
            {"T_0": 125, "T_mult": 1, "epochs_pretrain": 125},
            {"T_0": 135, "T_mult": 1, "epochs_pretrain": 135},
            {"T_0": 150, "T_mult": 1, "epochs_pretrain": 150},
        ],
        'lr_pretrain': 3e-3,
        'epochs_finetune': 100000,
        'patience_finetune': 200,
        'lr_finetune': 1e-3,
        'batch_a': 128,
        'batch_b': 64,
        'ensemble_alpha': [0.7, 0.7, 0.3, 0.7, 0.85], # C1 的集成权重
        
        # --- 模型设置 (C1 的黄金架构!) ---
        'hidden_dims': [128, 256, 256, 512], # <-- C1 复杂列表
        'num_layers': 4,                    # <-- C1/mlp.py 会忽略这个
        'dropout_rate': 0.2,

        # --- 损失函数权重 ---
        'lambda_coral': 0.1,
        'alpha_r2': 0,

        # ==========================================================
        # ===== 2. C2 的“黄金反向参数” (注入/覆盖!) =====
        # ==========================================================
        # (这些参数将由 C2 的 unified_inverse_train.py 读取)
        
        'mdn_components': 10,   # <-- C2 (原 C1: 20)
        'mdn_hidden_dim': 256,  # <-- C2 (原 C1: 128)
        'mdn_num_layers': 5,    # <-- C2 (原 C1: 3)
        'mdn_epochs': 1000,     # <-- C2 (原 C1: 500)
        'mdn_lr': 1e-3,
        'mdn_batch_size': 128,  # <-- C2 (原 C1: 256)
        'mdn_weight_decay': 1e-5,
    },

    'two_stage_opamp': {
        
        # ==========================================================
        # ===== 1. C1 的“黄金正向参数” (100% 保留!) =====
        # ==========================================================
        
        # --- 训练设置 ---
        'RESTART_PRETRAIN': 1,
        'PRETRAIN_SCHEDULER_CONFIGS': [ 
            {"T_0": 200, "T_mult": 1, "epochs_pretrain": 400},
        ],
        'lr_pretrain': 1e-3,
        'epochs_finetune': 10000,
        'patience_finetune': 500,
        'lr_finetune': 3.8e-3,
        'batch_a': 128,
        'batch_b': 64,
        'ensemble_alpha': [0.7, 0.7, 0.3, 0.7, 0.85],
        
        # --- 模型设置 (C1 的黄金架构!) ---
        'hidden_dims': [256, 256, 256, 256], # <-- C1 复杂列表
        'num_layers': 4,
        'dropout_rate': 0.2,

        # --- 损失函数权重 ---
        'lambda_coral': 0.025,
        'alpha_r2': 0,

        # ==========================================================
        # ===== 2. C2 的“黄金反向参数” (注入/覆盖!) =====
        # ==========================================================
        # (我们假设 C2 的黄金反向参数对所有 opamp 都适用)
        
        'mdn_components': 20,   # <-- C2 (原 C1: 20)
        'mdn_hidden_dim': 512,  # <-- C2 (原 C1: 128)
        'mdn_num_layers': 5,    # <-- C2 (原 C1: 3)
        'mdn_epochs': 2000,     # <-- C2 (原 C1: 500)
        'mdn_lr': 1e-3,
        'mdn_batch_size': 128,  # <-- C2 (原 C1: 256)
        'mdn_weight_decay': 1e-5,
    },
    
    # (如果 C1 还支持其他 opamp, 可以在这里添加...)
}