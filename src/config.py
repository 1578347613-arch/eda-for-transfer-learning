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
        # 训练设置
        'epochs_pretrain': 1000,
        'patience_pretrain': 200,
        'lr_pretrain': 3e-3,
        'epochs_finetune': 100000,
        'patience_finetune': 1000,
        'lr_finetune': 3.8e-3,
        'batch_a': 128,
        'batch_b': 64,
        'ensemble_alpha': [0.7, 0.7, 0.3, 0.7, 0.85],
        # 模型设置
        'hidden_dim': 256,
        'num_layers': 4,
        'dropout_rate': 0.2,

        # 损失函数权重
        'lambda_coral': 0.1,
        'alpha_r2': 0,
        # 反向模型
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
        'patience_pretrain': 200,
        'lr_pretrain': 3e-3,
        'epochs_finetune': 100000,
        'patience_finetune': 1000,
        'lr_finetune': 3.8e-3,
        'batch_a': 128,
        'batch_b': 64,
        'ensemble_alpha': [0.7, 0.7, 0.3, 0.7, 0.85],
        # 模型设置
        'hidden_dim': 256,
        'num_layers': 4,
        'dropout_rate': 0.2,

        # 损失函数权重
        'lambda_coral': 0.1,
        'alpha_r2': 0,
        # 反向模型
        'mdn_components': 20,
        'mdn_hidden_dim': 128,
        'mdn_num_layers': 3,
        'mdn_epochs': 500,
        'mdn_lr': 1e-3,
        'mdn_batch_size': 256,
        'mdn_weight_decay': 1e-5,

    },

    # ... 其他任务 ...
}
