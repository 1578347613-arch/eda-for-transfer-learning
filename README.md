# EDA 运放设计自动化 - 迁移学习与混合反向预测

本项目旨在利用机器学习方法解决 EDA 领域的运放（Op-Amp）设计自动化问题。项目包含两个核心任务：
1.  **正向预测**：根据电路的设计参数（如晶体管尺寸），预测其性能指标（如增益、带宽等）。
2.  **反向设计**：根据给定的性能指标要求，反向推算出满足要求的设计参数。

为了提高模型在小样本目标工艺（Target Domain）上的表现，本项目采用了**迁移学习**策略。同时，通过**模型集成**和创新的**混合反向预测**方法，进一步提升了预测的准确性和鲁棒性。

## 项目结构

```
eda-for-transfer-learning-1/
├── data/
│   ├── 01_train_set/         # 训练数据
│   │   ├── 5t_opamp/
│   │   └── 2stage_opamp/
│   └── 02_public_test_set/   # 测试数据
│       └── features/
├── src/                      # 所有源代码
│   ├── models/               # 模型定义
│   ├── results/              # 训练好的模型和scaler存放处
│   ├── config.py             # 配置文件
│   ├── data_loader.py        # 数据加载与预处理
│   ├── train_align_hetero.py # 训练迁移学习正向模型
│   ├── train_target_only.py  # 训练基线正向模型
│   ├── unified_inverse_train.py # 训练反向MDN模型
│   ├── inverse_opt.py        # 反向优化模块
│   ├── run_training.py       # 自动化训练控制器 (核心)
│   └── generate_submission.py # 生成提交文件 (核心)
├── submission/               # 生成的最终结果
└── README.md                 # 本文档
```

## 环境设置

1.  **克隆项目**
    ```bash
    git clone <your-repo-url>
    cd eda-for-transfer-learning
    ```

2.  **创建并激活虚拟环境** (推荐)
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **安装依赖**
    (请根据你的项目创建一个 `requirements.txt` 文件)
    ```bash
    pip install torch pandas scikit-learn numpy tqdm joblib
    ```

## 核心流程 (快速开始)

**重要提示：** 本项目的所有脚本都设计为在 `src` 目录下执行。

#### 第1步：进入 `src` 目录

```bash
cd src
```

#### 第2步：训练所有模型

使用 `run_training.py` 自动化训练所有需要的模型。这会依次为 `5t_opamp` 和 `2stage_opamp` 训练三种模型，并将结果保存在 `src/results/` 目录下。

```bash
python run_training.py
```

#### 第3步：生成提交文件

训练完成后，使用 `generate_submission.py` 来加载训练好的模型，对测试集进行预测，并生成最终的提交文件。
A. 最干净的“单模型直出”+ 影子榜单
目的：先排除温度/集成对分数的影响，验证训练评测与提交分能否对齐。
```bash
python -m src.generate_submission \
  --forward_mode simple \
  --forward_model align \
  --offline_eval \
  --eval_split val
```
正向 A/B：用 *_finetuned.pth 单模型直出。
反向 C/D：默认 hybrid 会执行；如果暂时只想验证 A/B，可加 --inverse_strategy mdn_only（脚本会跳过反向，只打印提示）。
离线评测：在 target_val 上输出 A/B 的物理域指标。
想对比另一个头（只用 target-only）：

python -m src.generate_submission --forward_mode simple --forward_model target --offline_eval --eval_split val
B. 回归原来的集成策略（对照实验）
```bash
python -m src.generate_submission \
  --forward_mode ensemble \
  --offline_eval --eval_split val
```
C. 指定输入/输出目录（可选）
```bash
python -m src.generate_submission \
  --data_dir data/02_public_test_set \
  --output_dir submission_v2 \
  --forward_mode simple --forward_model align \
  --offline_eval --eval_split all
```
执行完毕后，你会在项目根目录的 `submission/` 文件夹下找到 `predA.csv`, `predB.csv`, `predC.csv`, `predD.csv` 四个结果文件。

---

## 详细用法说明

### 1. 模型训练 (`run_training.py`)

`run_training.py` 是一个自动化的训练流水线控制器，它通过调用其他具体的训练脚本来完成任务，极大地方便了模型管理和复现。

#### 基本用法

所有命令都在 `src` 目录下执行。

*   **训练所有任务的所有模型** (默认行为):
    ```bash
    python run_training.py
    ```

*   **仅训练特定电路类型**:
    ```bash
    # 仅训练 5t_opamp 的所有模型
    python run_training.py --opamp 5t_opamp

    # 从头开始训练（默认是所有模型，从pretrain到finetune到targetonly再到mdn）
    python run_training.py --opamp 5t_opamp --restart
    # 注意，如果你只是想要从头训练一个模型或者多个，那你只需要在results文件夹中把上一次留存的对应的ckpt删除，runtraining脚本检测不到ckpt，就会自动重新训练

    # 同时训练两种电路
    python run_training.py --opamp 5t_opamp 2stage_opamp
    ```

*   **仅训练特定类型的模型**:
    通过 `--mode` 参数选择训练模式。
    *   `align_hetero`: 训练带对齐损失的迁移学习正向模型 (生成 `_finetuned.pth`)。
    *   `target_only`: 训练仅使用目标域数据的正向基线模型 (生成 `_target_only.pth`)。
    *   `inverse`: 训练用于反向预测的混合密度网络 (MDN) 模型 (生成 `mdn_....pth`)。

    ```bash
    # 为所有电路训练迁移学习模型
    python run_training.py --mode align_hetero

    # 仅为 5t_opamp 训练反向模型
    python run_training.py --opamp 5t_opamp --mode inverse
    ```

*   **传递额外参数**:
    你可以向底层的训练脚本传递额外的参数，例如 `epochs`。
    ```bash
    # 训练迁移模型，并设置 epoch 为 50
    python run_training.py --mode align_hetero --epochs 50
    ```
    
### 2. 生成提交文件 (`generate_submission.py`)

该脚本负责利用训练好的模型进行推理，并生成符合比赛要求的四个提交文件。

#### 核心思路

##### 正向预测 (predA, predB) - **模型集成策略**

-   为了提高预测的泛化能力和鲁棒性，我们没有使用单一模型，而是采用了**动态加权集成**的策略。
-   脚本会同时加载两个正向模型：
    1.  **迁移学习模型 (`_finetuned.pth`)**: 在源域预训练，在目标域微调，擅长捕捉跨域的共性知识。
    2.  **目标域基线模型 (`_target_only.pth`)**: 仅在目标域数据上训练，更专注于目标域的特性。
-   对于每一个测试样本，脚本会根据两个模型预测结果的**置信度（方差）**和**历史表现（MSE）**动态计算一个权重，将两个模型的输出进行加权平均，得到最终的预测结果。

##### 反向预测 (predC, predD) - **混合策略 (MDN + 优化)**

-   反向设计是一个复杂的一对多问题（相同的性能可能对应多种设计方案）。直接用神经网络预测容易陷入局部最优。
-   本项目采用了一种创新的两阶段**混合策略**：
    1.  **生成初始解**: 首先，使用一个**混合密度网络 (MDN)** 模型 (`mdn_....pth`)。MDN 不直接输出一个解，而是输出一个概率混合分布。我们从这个分布中取期望，得到一个高质量的初始设计参数 (`x_init`)。
    2.  **优化精修**: 然后，将 `x_init` 作为起点，把训练好的**正向模型** (`_finetuned.pth`) 当作一个“虚拟的SPICE仿真器”。通过梯度下降法，微调设计参数，目标是让这个“虚拟仿真器”的输出与我们期望的性能指标尽可能接近。

-   这种“先猜后调”的混合策略，结合了深度学习的快速映射能力和传统优化的精确求解能力，效果远超单一方法。

#### 使用方法

脚本的使用非常简单，直接运行即可。

```bash
# 在 src 目录下运行
python generate_submission.py
```

*   **默认行为**:
    -   加载 `src/results/` 目录下的所有必需模型。
    -   读取 `data/02_public_test_set/features/` 中的测试数据。
    -   执行正向集成预测和反向混合预测。
    -   将结果保存在根目录的 `submission/` 文件夹中。

*   **参数选项**:
    -   `--inverse_strategy`: 默认为 `hybrid`，即我们设计的混合策略。
    -   `--data_dir`: 指定测试集数据的位置。
    -   `--output_dir`: 指定输出结果的目录。

---




#### config文件参数简介
config.py 结构与设计解析
这个配置文件将所有可调整的参数和设置从核心的算法代码中分离出来，实现了配置与逻辑的解耦。这样做带来了巨大的好处：

集中管理：所有超参数都在一个地方，查找和修改非常方便。
易于实验：你只需修改这个文件，就能快速尝试不同的模型结构、学习率或训练周期，而无需改动任何训练代码。
可复现性：固定的配置文件（特别是有了 seed）确保了实验结果可以被精确复现。
易于扩展：如果未来要添加一个新的电路类型（比如 cascode_opamp），你只需要在 TASK_CONFIGS 中增加一个新的字典条目即可。
整个文件分为两个核心部分：

1. COMMON_CONFIG：通用共享配置
COMMON_CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'restart': False,
    'evaluate': True,
    'save_path': 'results',
    'seed': 42
}
这部分定义了在整个项目中所有任务和模型都共享的全局设置。

'device': 自动检测是否有可用的 NVIDIA GPU。如果有，就使用 cuda 进行加速；否则，使用 cpu。这是所有 PyTorch 项目的标准做法。
'restart': 一个布尔开关，可能用于控制是否从头开始训练，而不是从检查点恢复。
'evaluate': 一个布尔开关，可能用于控制在训练结束后是否执行评估步骤。
'save_path': 定义了所有训练产物（模型权重 .pth、标准化器 _scaler.gz 等）的保存目录。将其统一在这里，可以确保所有脚本都将结果保存在 src/results 目录下，避免了混乱。
'seed': 这是一个至关重要的参数！ 通过设置随机种子，可以保证代码中所有的随机操作（如模型权重的初始化、数据的随机划分、dropout等）在每次运行时都是相同的。这使得你的实验结果完全可复现。
2. TASK_CONFIGS：任务专属配置
TASK_CONFIGS = {
    '5t_opamp': { ... },
    'two_stage_opamp': { ... },
}
这是配置文件的核心。它是一个字典，其中键（key）是任务的名称（例如 '5t_opamp'），值（value）是另一个包含该任务所有专属超参数的字典。

这种设计允许你为不同的电路类型（它们的数据分布和复杂性可能差异很大）量身定制最优的超参数。

让我们以 '5t_opamp' 为例，深入分析其内部的配置项：

a. 正向模型训练参数 (Forward Model Training)
这些参数控制正向模型（train_align_hetero.py 和 train_target_only.py）的训练过程。

'epochs_pretrain', 'patience_pretrain': 在源域数据上进行预训练时的最大轮数和早停耐心值。
'epochs_finetune', 'patience_finetune': 在目标域数据上进行微调时的最大轮数和早停耐心值。
'lr': 学习率 (Learning Rate)。
'batch_a', 'batch_b': 分别对应源域（工艺A）和目标域（工艺B）的数据批次大小。
b. 正向模型结构参数 (Forward Model Architecture)
这些参数定义了正向预测模型（AlignHeteroMLP）的神经网络结构。

'hidden_dim': 神经网络隐藏层的神经元数量。
'num_layers': 隐藏层的数量。
'dropout_rate': 在训练期间随机“关闭”一部分神经元的比例，用于防止过拟合。
c. 损失函数权重 (Loss Function Weights)
这些是迁移学习中非常关键的参数，用于平衡不同损失项。

'lambda_coral': CORAL 损失的权重。CORAL 是一种用于领域自适应 (Domain Adaptation) 的技术，它通过最小化源域和目标域数据在隐藏层特征的二阶统计量（协方差）差异，来“拉近”两个数据域的分布，从而帮助模型更好地将在源域学到的知识迁移到目标域。
'alpha_r2': 可能是 R² 损失项的权重，用于调整模型对预测性能的关注度。
d. 反向模型参数 (Inverse Model - MDN)
这些参数专门用于训练反向预测模型（unified_inverse_train.py）。

'mdn_components': 混合密度网络 (MDN) 中高斯分布的数量。这是 MDN 的核心参数，决定了模型能拟合多复杂的一对多映射关系。
'mdn_hidden_dim', 'mdn_num_layers': 反向模型的网络结构。
'mdn_epochs', 'mdn_lr', 'mdn_batch_size', 'mdn_weight_decay': 反向模型的训练超参数。
e. 集成预测参数 (Ensemble Prediction)
这个参数在最终推理阶段（generate_submission.py）使用。

'ensemble_alpha': 在进行模型集成时，用于加权平均“迁移学习模型”和“仅目标域模型”预测结果的权重。它是一个列表，意味着可以为每一个性能输出（如增益、带宽等）设置不同的权重，非常灵活。
