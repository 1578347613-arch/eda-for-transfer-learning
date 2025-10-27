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
EPOCHS_PRETRAIN = 1000
PATIENCE_PRETRAIN = EPOCHS_PRETRAIN  # 无早停
LEARNING_RATE_PRETRAIN = 3e-3
T0_PRETRAIN = 125      # 第一个重启周期的长度 (epoch)
T_MULT_PRETRAIN = 1    # 每个重启周期后，周期长度的乘法因子


EPOCHS_FINETUNE = 100000  # 配合早停
PATIENCE_FINETUNE = 1000
LEARNING_RATE_FINETUNE = 3.8e-3
BATCH_A = 128
BATCH_B = 64  # 线性缩放规则

# 模型设置
HIDDEN_DIM = 256
NUM_LAYERS = 4
DROPOUT_RATE = 0.2

# 权重 / 优化设置
LAMBDA_NLL = 1.0  # NLL 损失的权重 (主任务)
LAMBDA_CORAL = 0
ALPHA_R2 = 0
