# eda-for-transfer-learning
for dataloader：
  好的，这些可视化图表非常直观地揭示了数据的核心特征。我们来逐一分析，并基于此设计一个健壮、合理的数据预处理方法。

            ### **数据特征分析**
            
            从你提供的图片中，我们可以总结出三个关键的、必须在预处理中解决的特性：
            
            1.  **输入特征分布对齐 (Input Feature Distributions are Aligned)**:
                *   **观察**: 查看 `w1`, `w2`, `w3`, `l1`, `l2`, `l3`, 和 `ibias` 的图。无论是均匀分布 (`w1`, `w2`, `w3`) 还是离散分布 (`l1`, `l2`, `l3`, `ibias`)，工艺A (蓝色) 和工艺B (红色) 的分布形状和范围**几乎完全一致**。
                *   **结论**: 这是一个非常好的消息。它意味着两个工艺的数据是在相同的设计空间内进行采样的。我们的模型不需要处理输入域的偏移，可以专注于学习从这个**共同的输入空间**到**不同的输出空间**的映射关系。
            
            2.  **输出性能存在显著的工艺漂移 (Significant Process Shift in Outputs)**:
                *   **观察**: `dc_gain` 的图是这个现象最典型的例子。工艺A的分布集中在100以下，而工艺B的分布则从150开始。**两个分布几乎没有任何重叠**。`slewrate_pos` 的分布也显示出形态上的差异。
                *   **结论**: 这正是本次竞赛的核心挑战。对于相同的输入参数，两个工艺产生的性能指标是系统性不同的。我们的模型**必须学习到这种“漂移”或“偏移”**。一个只在工艺A上训练的模型，在预测工艺B的 `dc_gain` 时会产生巨大的、系统性的偏差。
            
            3.  **部分输出性能数据极度偏斜/存在极端值 (Highly Skewed Outputs / Outliers)**:
                *   **观察**: `ugf` (单位增益带宽) 的图最为突出。绝大多数数据点都挤在接近0的一端，形成一个非常高的峰，同时有一条延伸到 `1e11` 的极长的尾巴。
                *   **结论**: 这种极度偏斜的分布对神经网络训练是**灾难性的**。如果使用均方误差（MSE）作为损失函数，那几个 `1e11` 的极端值所产生的巨大误差将完全主导模型的梯度更新，导致模型只关注如何拟合那几个点，而忽略了绝大多数正常数据，最终模型性能会非常差。
            
            ### **合理的DataLoader数据预处理方法**
            
            基于以上分析，一个合理的数据预处理流程应该包含以下三个步骤，并封装在一个 `DataLoader` 或数据准备函数中。
            
            ---
            
            #### **步骤 1: 对数变换 (Log Transform) 处理偏斜数据**
            
            **目的**: 解决第3个问题——数据极度偏斜。
            
            对数变换是处理这种长尾分布、使数据更接近正态分布的绝佳方法。它可以极大地压缩数据的范围，降低极端值的影响。
            
            *   **操作**: 对 `ugf` 和 `cmrr` (根据我们之前的统计分析，`cmrr` 也有类似问题) 这两列应用 `np.log1p()` 函数。`log1p(x)` 计算 `log(1+x)`，好处是当 `x=0` 时不会出错。
            
            #### **步骤 2: 数据标准化 (Standardization)**
            
            **目的**: 解决不同特征和目标之间巨大的数值范围差异，并为模型学习“工艺漂移”提供信号。
            
            神经网络对输入数据的尺度非常敏感。标准化（将数据转换为均值为0，方差为1）是必不可少的步骤。
            
            *   **操作**: 使用 `sklearn.preprocessing.StandardScaler`。
            *   **关键策略**:
                1.  **创建两个Scaler**: 一个用于输入特征 `X` (`x_scaler`)，一个用于（经过对数变换后的）输出目标 `Y` (`y_scaler`)。
                2.  **在工艺A (Source) 数据上 `fit` Scaler**: 我们将工艺A作为知识的来源。因此，我们**只在工艺A的数据上计算均值和标准差** (`x_scaler.fit(X_source)`, `y_scaler.fit(y_source)`)。
                3.  **用同一个Scaler去 `transform` 所有数据**: 使用在工艺A上 `fit` 好的 `scaler` 去转换工艺A和工艺B的数据。
            
            **💡 为什么这样做？**
            这至关重要。如果我们分别在A和B上 `fit`，那么两组数据的 `dc_gain` 都会被转换成均值为0的分布，模型就**看不到**那个巨大的工艺漂移了。而通过在A上 `fit` 并同时 `transform` A和B，工艺A的 `dc_gain` 转换后均值为0，而工艺B的 `dc_gain` 转换后会是一个**显著大于0的分布**。这个非零的均值，就是模型需要学习的“工艺漂移”信号！
            
            #### **步骤 3: 数据集划分 (Splitting)**
            
            **目的**: 为模型的预训练、微调和验证准备好数据集。
            
            *   **操作**:
                1.  **预训练集**: 处理后的工艺A数据 `(X_source_scaled, y_source_scaled)`。
                2.  **微调/验证集**: 将处理后的工艺B数据划分为训练集和验证集（例如，80%用于微调训练，20%用于验证）。





---

## 🚀 TL;DR 快速上手

```bash

python -m data_loader.cli --opamp 5t_opamp --val-split 0.2 --seed 42

# 3) 训练基线 MLP（保存 baseline 权重）
python -m training.train

# 4) 训练 AlignHeteroMLP + CORAL（保存 align_hetero 权重）
python -m training.train_align_coral

# 5) 微调 DualHeadMLP（分阶段训练，保存 dualhead 微调权重）
python -m fine_tune.fine_tune

# 6) 集成推理 & 指标评估（反标准化到物理单位）
python -m inference.infer_ensemble
```

---

## 📦 项目概览

本工程面向模拟电路（如运放）多目标回归，提供：

- **数据管道**：加载、预处理（含 `log1p`、标准化）以及保存/复用 scaler。  
- **模型库**：`MLP`、`AlignHeteroMLP`（异方差）、`DualHeadMLP`（双头微调）。  
- **训练脚本**：基线训练、带 CORAL 的跨域对齐训练、分阶段微调（L2-SP 正则）。  
- **推理脚本**：两模型不确定性感知加权 + MSE 权重融合，自动反标准化与评估。

---

## 🗂 目录结构

```
src_new/
├── config.py                 # 统一管理超参数&设备
│---inverse_mdn.py
----inverse_opt.py
├── data_loader/
│   ├── __init__.py           # 暴露 load/preprocess/split/scale 的统一API
│   ├── cli.py                # 命令行快速验证数据加载
│   ├── data_loader.py        # 加载+预处理+划分
│   └── scaler_utils.py       # 保存/加载 x,y 的 StandardScaler
│
├── models/
│   ├── mlp.py                # 基础 MLP
│   ├── align_hetero.py       # AlignHeteroMLP（backbone + hetero_head）
│   └── model_utils.py        # 保存/加载权重，主干迁移工具
│
├── training/
│   ├── train.py              # 训练基线 MLP
│   └── train_align_coral.py  # AlignHeteroMLP + CORAL 训练
│
├── fine_tune/
│   └── fine_tune.py          # DualHeadMLP 分阶段微调（bias→head→解冻最后层）
│
├── inference/
│   └── infer_ensemble.py     # 集成推理（温度标定+精度/MSE加权），评估MSE/MAE/R²
│
└── losses/
    └── loss_function.py      # heteroscedastic_nll / batch_r2 / coral_loss
# 注：若你使用 `from losses import x`，请确保 losses/__init__.py 已导出上述函数。
```

---

## 🔧 安装与依赖

**Python**：建议 3.10  
**PyTorch**：根据你的 CUDA 版本安装（https://pytorch.org/）  

若无 `requirements.txt`，可使用下列基础依赖（按需增减）：

```txt
numpy>=1.24
scikit-learn>=1.3
joblib>=1.3
torch>=2.1
tqdm>=4.66
```

---

## ⚙️ 配置（`config.py`）

统一管理所有超参数，随用随改，训练脚本自动读取：

```python
# 关键参数示例（实际以你的 config.py 为准）
OPAMP_TYPE   = '5t_opamp'
DEVICE       = 'cuda'  # 自动检测同样可：'cuda' if torch.cuda.is_available() else 'cpu'

# 训练
EPOCHS       = 50
PATIENCE     = 10
LEARNING_RATE= 1e-4
BATCH_SIZE   = 256

# 模型
HIDDEN_DIM   = 512
NUM_LAYERS   = 6
DROPOUT_RATE = 0.1

# 对齐/微调
LAMBDA_CORAL = 0.05
ALPHA_R2     = 1.0
L2SP_LAMBDA  = 1e-4
LR_BIAS      = 3e-4
LR_HEAD      = 1e-4
LR_UNFREEZE  = 5e-5
WEIGHT_DECAY = 1e-4
```

> **建议**：仅通过 `config.py` 改动超参数，避免在脚本内“硬编码”，保证全工程一致。

---

## 🧪 数据与预处理

- **数据入口**：`data_loader.get_data_and_scalers(opamp_type=OPAMP_TYPE)`  
  返回：
  ```python
  {
    "source": (X_source_scaled, y_source_scaled),
    "target_train": (X_target_train, y_target_train),
    "target_val": (X_target_val, y_target_val),
    "x_scaler": x_scaler,
    "y_scaler": y_scaler,
  }
  ```
- 预处理包含 `log1p`（对特定目标，如 `ugf`, `cmrr`）和标准化。反标准化与 `expm1` 在推理阶段自动完成。

---

## 🏗 模型与损失（API 速览）

### 模型

```python
from models import MLP, AlignHeteroMLP, DualHeadMLP

# MLP
m = MLP(input_dim, output_dim)                 # forward(x) -> y_hat

# AlignHeteroMLP（异方差）
m = AlignHeteroMLP(input_dim, output_dim)      # forward(x) -> (mu, logvar, features)

# DualHeadMLP（微调双头）
yB = model(x, domain='B')                      # 指定使用 B 头
```

### 模型工具

```python
from models.model_utils import (
  load_backbone_from_trained_mlp, save_model, load_model
)

load_backbone_from_trained_mlp(pretrained_mlp, align_model)
save_model(model, 'results/xxx.pth')
load_model(model, 'results/xxx.pth')
```

### 损失函数

```python
# 若无 __init__.py 导出，请改为：from losses.loss_function import ...
from losses import heteroscedastic_nll, batch_r2, coral_loss
```

- `heteroscedastic_nll(mu, logvar, y, reduction='mean')`
- `batch_r2(y_true, y_pred, eps=1e-8)`
- `coral_loss(feat_a, feat_b, unbiased=True, eps=1e-6)`

---
一。正向设计
## 🏃‍♂️ 训练/微调/推理流程

### 1) 训练基线 MLP

```bash
python -m training.train
```

- **输入**：`source` 作为训练集，`target_val` 作为验证集  
- **输出**：`results/{OPAMP_TYPE}_baseline_model.pth`  
- **日志**：打印每轮 Train/Val MSE

### 2) 训练 AlignHeteroMLP + CORAL

```bash
python -m training.train_align_coral
```

- 载入基线 MLP 作为 **backbone** 初始权重  
- 目标域 `B` 上训练异方差 NLL + `R²`（转化为损失）+ **CORAL**（跨域特征对齐）  
- **输出**：`results/{OPAMP_TYPE}_align_hetero_lambda{LAMBDA_CORAL:.3f}.pth`  
- **验证指标**：val NLL（越小越好）

### 3) 分阶段微调 DualHeadMLP

```bash
python -m fine_tune.fine_tune
```

分三阶段（均在目标域 `B`）：
1. **Bias 校准**：仅训练 `head_B.bias`
2. **训练 B 头权重**：启用 L2-SP 正则，约束偏离预训练主干
3. **部分解冻**：解冻主干最后一层 + B 头

**输出**：`results/{OPAMP_TYPE}_dualhead_finetuned.pth`

> 使用到的关键接口：
> - `run_epoch(model, loader, optimizer, loss_fn, phase, pretrained_state=None)`
> - `main()`：组织上述三阶段训练与早停

### 4) 集成推理与评估

```bash
python -m inference.infer_ensemble
```

做了以下工作：
- 载入两个异方差模型（示例：`align_hetero` 与 `target_only_hetero`）  
- **温度标定**（闭式解）校准方差  
- **样本级精度权重** + **指标级 MSE 权重** 融合  
- 反标准化 & `expm1`（对 `ugf`, `cmrr`）  
- 输出每个指标的 **MSE/MAE/R²**

**打印示例**：
```
=== Ensemble on B-VAL (物理单位) ===
slewrate_pos    MSE=...  MAE=...  R2=...
dc_gain         MSE=...  MAE=...  R2=...
ugf             MSE=...  MAE=...  R2=...
phase_margin    MSE=...  MAE=...  R2=...
cmrr            MSE=...  MAE=...  R2=...
```
二. 反向设计（inverse_mdn.py）
功能：通过训练混合密度网络（MDN）来学习从目标值 y 到输入值 x 的映射关系。支持两种模式：训练模式和采样模式。

1.1 训练模式
在训练模式下，使用一组标准化的目标数据 y_scaled 和输入数据 x_scaled 来训练一个 MDN 模型。模型将学习从目标输出到输入的映射。训练完成后，模型权重和标准化器（scaler）将被保存到指定路径。

用法：

python src/inverse_mdn.py --opamp 5t_opamp \
                          --save results/mdn_5t.pth \
                          --components 10 \
                          --hidden 256 \
                          --layers 4 \
                          --batch-size 128 \
                          --epochs 60 \
                          --lr 1e-3


1.2 采样模式
在采样模式下，用户提供一个目标 y_target，工具将基于已训练的 MDN 模型生成多个候选输入 x_scaled，这些输入能够使得模型的输出接近目标 y_target。

用法：

python src/inverse_mdn.py --sample \
                          --model results/mdn_5t.pth \
                          --y-target "2.5e8,200,1.5e6,65,20000" \
                          --n 64 \
                          --out results/inverse/init_64.npy


2. 反向优化（inverse_opt.py）
功能：使用反向优化算法，通过优化输入 x 使得模型的输出 y 满足用户指定的目标或约束条件。支持多种目标类型（最小化、最大化、目标值、范围等）和约束条件（如 ugf_band 和 pm_band）。

2.1 反向优化
在反向优化过程中，工具会使用多个初始点对输入 x 进行优化，最终得到一个最优的输入 x_scaled，使得其预测输出 y_scaled 达到给定目标。优化结果将保存在指定的目录中。

用法：

python src/inverse_opt.py --opamp 5t_opamp \
                          --ckpt results/5t_opamp_align_hetero_lambda0.050.pth \
                          --model-type align_hetero \
                          --y-target "2.5e8,200,1.5e6,65,20000" \
                          --goal "min,min,range,range,min" \
                          --ugf-band "8.0e5:2.0e6" \
                          --pm-band "60:75" \
                          --weights "0.05,0.40,0.90,0.10,0.65" \
                          --prior 1e-3 \
                          --init-npy results/inverse/init_1024.npy \
                          --n-init 1024 --steps 900 --lr 0.002 \
                          --finish-lbfgs 80 \
                          --save-dir results/inverse/try_hybrid_constrained_scaled_v2
---

## 🧭 常见问题（FAQ / Troubleshooting）

- **找不到 baseline 权重**  
  先运行：`python -m training.train`，会在 `results/` 下生成 `{OPAMP_TYPE}_baseline_model.pth`。

- **CUDA 不可用/显存不足**  
  在 `config.py` 将 `DEVICE='cpu'` 或减小 `BATCH_SIZE`/`HIDDEN_DIM`。

- **形状不匹配**  
  检查 `input_dim = X.shape[1]` 与模型初始化一致；`output_dim = y.shape[1]` 与任务指标数量一致。

- **CORAL 权重设置**  
  `LAMBDA_CORAL` 过大可能拖慢收敛；可先从 `0.01~0.05` 网格搜索。

- **R² 损失权重**  
  `ALPHA_R2` 控制 R² 目标；若 NLL 主导不足，可适当上调。

- **losses 导入失败**  
  若使用 `from losses import ...` 报错，请改用  
  `from losses.loss_function import heteroscedastic_nll, batch_r2, coral_loss`  
  或在 `losses/__init__.py` 中显式导出。

---

## 📌 重要输出与约定

- **模型权重**（默认保存在 `results/`）
  - 基线：`{OPAMP_TYPE}_baseline_model.pth`
  - 对齐：`{OPAMP_TYPE}_align_hetero_lambda{LAMBDA:.3f}.pth`
  - 微调：`{OPAMP_TYPE}_dualhead_finetuned.pth`
- **Scaler**：训练过程中会保存 `x/y` 的 scaler（路径见你的实现，一般在 `results/` 下）
- **列名约定**：`COLS = ['slewrate_pos', 'dc_gain', 'ugf', 'phase_margin', 'cmrr']`  
  其中 `ugf`, `cmrr` 在推理评估时会 `expm1` 反变换。

---

## 🧩 开发小贴士

- 已对梯度做 `clip_grad_norm_=1.0`，可提高训练稳定性。  
- 多卡/混合精度：当前代码未内置，可按需接入 `torch.nn.parallel` / AMP。  
- 想调参？只改 `config.py`，其余脚本无需改动。  
- 想自定义指标列：同步修改数据预处理及 `infer_ensemble.py` 中的 `COLS` 列表与反变换逻辑。



、
