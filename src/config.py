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
        'RESTART_PRETRAIN': 9,
        'PRETRAIN_SCHEDULER_CONFIGS': [  # 重复执行三次元优化
            # --- 策略一：广泛探索 ---
                {"T_0": 50, "T_mult": 1, "epochs_pretrain": 100},  # 第1次重启
                {"T_0": 55, "T_mult": 1, "epochs_pretrain": 110},  # 第2次重启
            # --- 策略二：精细打磨 ---
                {"T_0": 125, "T_mult": 1, "epochs_pretrain": 125},  # 第3次重启
            # --- 策略一：广泛探索 ---
                {"T_0": 50, "T_mult": 1, "epochs_pretrain": 100},  # 第4次重启
                {"T_0": 55, "T_mult": 1, "epochs_pretrain": 110},  # 第5次重启
            # --- 策略二：精细打磨 ---
                {"T_0": 125, "T_mult": 1, "epochs_pretrain": 125},  # 第6次重启
            # --- 策略一：广泛探索 ---
                {"T_0": 50, "T_mult": 1, "epochs_pretrain": 100},  # 第7次重启
                {"T_0": 55, "T_mult": 1, "epochs_pretrain": 110},  # 第8次重启
            # --- 策略二：精细打磨 ---
                {"T_0": 125, "T_mult": 1, "epochs_pretrain": 125},  # 第9次重启
        ],
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
        'epochs_pretrain': 1000,  # 正向已替换为scheduler,该参数不再需要
        'patience_pretrain': 200,  # 同上
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
        'epochs_pretrain': 4000,
        'patience_pretrain': 200,
        'lr_pretrain': 1e-3,
        'epochs_finetune': 10000,
        'patience_finetune': 500,
        'lr_finetune': 3.8e-3,
        'batch_a': 128,
        'batch_b': 64,
        'ensemble_alpha': [0.7, 0.7, 0.3, 0.7, 0.85],
        # 模型设置
        'hidden_dim': 256,
        'num_layers': 4,
        'dropout_rate': 0.2,

        # 损失函数权重
        'lambda_coral': 0.025,
        'alpha_r2': 0,


        # 反向模型
        'epochs_pretrain': 1000,  # 正向已替换为scheduler,该参数不再需要
        'patience_pretrain': 200,  # 同上
        'mdn_components': 20,
        'mdn_hidden_dim': 128,
        'mdn_num_layers': 3,
        'mdn_epochs': 500,
        'mdn_lr': 1e-3,
        'mdn_batch_size': 256,
        'mdn_weight_decay': 1e-5,

    },
}
