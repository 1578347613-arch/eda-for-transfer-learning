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
PATIENCE_PRETRAIN = 500
LEARNING_RATE_PRETRAIN = 3e-3
T0_PRETRAIN = 200      # 第一个重启周期的长度 (epoch)
T_MULT_PRETRAIN = 1    # 每个重启周期后，周期长度的乘法因子


EPOCHS_FINETUNE = 100000  # 配合早停
PATIENCE_FINETUNE = 500
LEARNING_RATE_FINETUNE = 3.8e-3
BATCH_A = 128
BATCH_B = 64  # 线性缩放规则

# 模型设置
HIDDEN_DIM = 256
NUM_LAYERS = 4
DROPOUT_RATE = 0.2

# 权重 / 优化设置
LAMBDA_NLL = 1.0  # NLL 损失的权重 (主任务)
LAMBDA_CORAL = 0.2
ALPHA_R2 = 1.0

# ==========================================================
# ========== 反向设计 (Inverse MDN) 专属参数 ==========
# ==========================================================
# (为了清晰，我们给它们加上 _INV 后缀，或者保留 MDN_ 前缀)

# --- 核心设置 (与正向分开！) ---
OPAMP_TYPE_INV = '5t_opamp'   # 反向设计默认处理的运放类型
DEVICE_INV = 'cuda' if torch.cuda.is_available() else 'cpu' # 反向设计使用的设备
SAVE_PATH_INV = "../results" # 反向模型 (.pth) 和 scaler (.gz) 的保存路径
SEED_INV = 42                # 反向训练的随机种子

# --- MDN 模型专属参数 ---
MDN_COMPONENTS = 10     # MDN 高斯混合成分数量
MDN_HIDDEN_DIM = 256    # MDN 隐藏层维度 (可以独立于正向模型调整！)
MDN_NUM_LAYERS = 4      # MDN 隐藏层数量 (可以独立于正向模型调整！)

# --- MDN 训练专属参数 ---
MDN_EPOCHS = 1000        # MDN 训练轮数 (可以先设大一点，手动停止)
MDN_PATIENCE = 1000      # <--- 新增！早停耐心设为 100 轮
MDN_BATCH_SIZE = 128    # MDN 训练批次大小
MDN_LR = 1e-3           # MDN 学习率
MDN_WEIGHT_DECAY = 1e-5 # 权重衰减，如果需要可以取消注释

# --- 反向训练是否强制重新开始 ---
RESTART_INV = False      # 如果为 True，即使 .pth 文件存在也会重新训练


# ==========================================================
# ========== 推理与提交 (Inference/Submission) 参数 ==========
# ==========================================================

# --- 正向集成 (Forward Ensemble) ---
ENSEMBLE_ALPHA = [0.5, 0.5, 0.5, 0.5, 0.5] # 平衡 "置信度" vs "历史MSE" 权重 (0=只看MSE, 1=只看置信度)