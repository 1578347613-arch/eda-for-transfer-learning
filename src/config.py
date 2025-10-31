import torch

# 基本设置
OPAMP_TYPE = '5t_opamp'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LOG_TRANSFORMED_COLS = [
    "ugf",
    "cmrr",
    "dc_gain",
    "slewrate_pos",
]

# 训练设置
RESTART_PRETRAIN = 1
PRETRAIN_SCHEDULER_CONFIGS = [  # 重复执行三次元优化
    # --- 策略一：广泛探索 ---
    # {"T_0": 50, "T_mult": 1, "epochs_pretrain": 100},  # 第1次重启：使用较长的周期进行稳定探索
    # {"T_0": 55, "T_mult": 1, "epochs_pretrain": 110},  # 第2次重启：使用更高的学习率进行激进探索

    # --- 策略二：精细打磨 ---
    # 第3次重启：使用较低的学习率进行一次长周期的精细搜索
    {"T_0": 125, "T_mult": 1, "epochs_pretrain": 125},


    # ... 您可以根据需要添加更多配置 ...
]

PATIENCE_PRETRAIN = 200  # 无早停
LEARNING_RATE_PRETRAIN = 3e-3


EPOCHS_FINETUNE = 100000  # 配合早停
PATIENCE_FINETUNE = 1000
LEARNING_RATE_FINETUNE = 3.8e-3
BATCH_A = 128
BATCH_B = 64  # 线性缩放规则

# 模型设置
HIDDEN_DIMS = [256, 256, 256, 256]
NUM_LAYERS = 4
DROPOUT_RATE = 0.2

# 权重 / 优化设置
LAMBDA_NLL = 1.0  # NLL 损失的权重 (主任务)
LAMBDA_CORAL = 0.1
ALPHA_R2 = 0
