# src/config.py (修改后)

import torch

# --- 通用配置 (所有模型共享) ---
COMMON_CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'restart': False,
    'evaluate': True,
    'save_path': 'results',# 注意这里也改成了小写
    'seed': 42  # <--- 添加这一行！
}

# --- 各模型专属配置 ---
TASK_CONFIGS = {
    '5t_opamp': {
        # 训练设置
        'epochs_pretrain': 100,
        'patience_pretrain': 100,
        'epochs_finetune': 1000,
        'patience_finetune': 100,
        'lr': 1e-3, # learning_rate -> lr, 与 argparse 保持一致
        'batch_a': 128,
        'batch_b': 128,
        'ensemble_alpha': [0.7, 0.7, 0.3, 0.7, 0.85],
        # 模型设置
        'hidden_dim': 256,
        'num_layers': 4,
        'dropout_rate': 0.1,
        
        # 损失函数权重
        'lambda_coral': 0.05,
        'alpha_r2': 1.0,
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
        'epochs_pretrain': 120,
        'patience_pretrain': 100,
        'epochs_finetune': 1200,
        'patience_finetune': 120,
        'lr': 8e-4,
        'batch_a': 128,
        'batch_b': 128,
        
        # 模型设置
        'hidden_dim': 512,
        'num_layers': 5,
        'dropout_rate': 0.15,
        
        # 损失函数权重
        'lambda_coral': 0.1,
        'alpha_r2': 1.0,

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
