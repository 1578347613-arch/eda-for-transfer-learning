# src/config.py

import torch

# --- 通用配置 (所有模型共享) ---
COMMON_CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'restart': False,
    'evaluate': True,
    'save_path': 'results',
    'seed': 42
}

LOG_TRANSFORMED_COLS = [
    "ugf",
    "cmrr",
    "dc_gain",
    "slewrate_pos",
]

# --- 各模型专属配置 ---
TASK_CONFIGS = {
    '5t_opamp': {
        # --- [新] 分阶段预训练调度器设置 ---
        'restart_pretrain': 9,  # 控制预训练重启的次数
        'pretrain_scheduler_configs': [  # 这是一个包含多个预训练阶段的列表
            # --- 策略一：广泛探索 (重复三次) ---
            {"T_0": 50, "T_mult": 1, "epochs_pretrain": 100},
            {"T_0": 55, "T_mult": 1, "epochs_pretrain": 110},
            {"T_0": 125, "T_mult": 1, "epochs_pretrain": 125},
            {"T_0": 50, "T_mult": 1, "epochs_pretrain": 100},
            {"T_0": 55, "T_mult": 1, "epochs_pretrain": 110},
            {"T_0": 125, "T_mult": 1, "epochs_pretrain": 125},
            {"T_0": 50, "T_mult": 1, "epochs_pretrain": 100},
            {"T_0": 55, "T_mult": 1, "epochs_pretrain": 110},
            {"T_0": 125, "T_mult": 1, "epochs_pretrain": 125},
        ],
        # --- 训练设置 ---
        'patience_pretrain': 200,  # 更新后的预训练耐心值
        'lr_pretrain': 3e-3,
        'epochs_finetune': 100000,
        'patience_finetune': 750,  # 更新后的微调耐心值
        'lr_finetune': 3.8e-3,
        'batch_a': 128,
        'batch_b': 64,

        # --- 模型设置 ---
        'hidden_dim': 256,
        'num_layers': 4,
        'dropout_rate': 0.2,
        'ensemble_alpha': [0.7, 0.7, 0.3, 0.7, 0.85],

        # --- 损失函数权重 ---
        'lambda_nll': 1.0,
        'lambda_coral': 0.1,   # 更新后的 CORAL 权重
        'alpha_r2': 0,

        # --- 反向模型 ---
        'mdn_components': 20,
        'mdn_hidden_dim': 128,
        'mdn_num_layers': 3,
        'mdn_epochs': 500,
        'mdn_lr': 1e-3,
        'mdn_batch_size': 256,
        'mdn_weight_decay': 1e-5,
    },

    'two_stage_opamp': {
        # 训练设置
        'epochs_pretrain': 1000,
        'patience_pretrain': 1000,
        'lr_pretrain': 3e-3,
        'epochs_finetune': 100000,
        'patience_finetune': 1000,
        'lr_finetune': 1e-4,
        'batch_a': 128,
        'batch_b': 128,

        # 模型设置
        'hidden_dim': 512,
        'num_layers': 5,
        'dropout_rate': 0.15,

        # 损失函数权重
        'lambda_coral': 0.1,
        'alpha_r2': 1.0,
        # 反向模型
        'mdn_components': 25,
        'mdn_hidden_dim': 256,
        'mdn_num_layers': 4,
        'mdn_epochs': 600,
        'mdn_lr': 8e-4,
        'mdn_batch_size': 256,
        'mdn_weight_decay': 1e-5,
        'ensemble_alpha': [0.5] * 5,

    },

    # ... 其他任务 ...
}
