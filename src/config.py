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
EPOCHS_PRETRAIN = 250
PATIENCE_PRETRAIN = EPOCHS_PRETRAIN
LEARNING_RATE_PRETRAIN = 3e-3


EPOCHS_FINETUNE = 100000  # 配合早停
PATIENCE_FINETUNE = 1000
LEARNING_RATE_FINETUNE = 1e-4
BATCH_SIZE = 128

# 模型设置
HIDDEN_DIM = 256
NUM_LAYERS = 4
DROPOUT_RATE = 0.2
COMPONENTS = 10

# 反向设计（MDN）
MDN_EPOCHS = 60
MDN_COMPONENTS = 10
MDN_HIDDEN = 256
MDN_LAYERS = 4
MDN_BATCH_SIZE = 128
MDN_LR = 1e-3

# 权重 / 优化设置
LAMBDA_CORAL = 0.05
ALPHA_R2 = 1.0
L2SP_LAMBDA = 1e-4
LR_BIAS = 3e-4
LR_HEAD = 1e-4
LR_UNFREEZE = 5e-5
WEIGHT_DECAY = 1e-4

# A/B 域 DataLoader 的 batch size
BATCH_B = 128  # target data B
BATCH_A = 128  # source data A

# ====== 新增：微调阶段的 epoch 数 ======
# 第一阶段：只调 B 头的 bias
EPOCHS_BIAS = 10          # 你可以按需调大/调小
# 第二阶段：训练 B 头全部参数（配合 L2-SP）
EPOCHS_HEAD = 40
# 第三阶段：解冻主干最后一层 + B 头
EPOCHS_UNFREEZE = 20
