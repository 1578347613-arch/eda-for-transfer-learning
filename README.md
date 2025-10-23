# eda-for-transfer-learning
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

python -m inverse_mdn   --opamp 5t_opamp   --components 10 --hidden 256 --layers 4   --batch-size 128 --epochs 60 --lr 1e-3


1.2 采样模式
在采样模式下，用户提供一个目标 y_target，工具将基于已训练的 MDN 模型生成多个候选输入 x_scaled，这些输入能够使得模型的输出接近目标 y_target。

用法：

python -m inverse_mdn --sample \
  --model ../results/mdn_5t_opamp.pth \
  --y-target "2.5e8,200,1.5e6,65,20000" \
  --n 64 \
  --out ../results/inverse/init_64.npy



2. 反向优化（inverse_opt.py）
功能：使用反向优化算法，通过优化输入 x 使得模型的输出 y 满足用户指定的目标或约束条件。支持多种目标类型（最小化、最大化、目标值、范围等）和约束条件（如 ugf_band 和 pm_band）。

2.1 反向优化
在反向优化过程中，工具会使用多个初始点对输入 x 进行优化，最终得到一个最优的输入 x_scaled，使得其预测输出 y_scaled 达到给定目标。优化结果将保存在指定的目录中。

用法：
align hetero直接优化：
python -m inverse_opt \
  --opamp 5t_opamp \
  --ckpt ../results/5t_opamp_align_hetero_lambda0.050.pth \
  --model-type align_hetero \
  --y-target "2.5e8,200,1.5e6,65,20000" \
  --goal "min,min,range,range,min" \
  --ugf-band "8.0e5:2.0e6" \
  --pm-band "60:75" \
  --weights "0.05,0.40,0.90,0.10,0.65" \
  --prior 1e-3 \
  --n-init 512 --steps 800 --lr 0.002 \
  --finish-lbfgs 80 \
  --save-dir ../results/inverse/run_align


  
搭配mdn初值（hybrid）
python -m inverse_opt \
  --opamp 5t_opamp \
  --ckpt ../results/5t_opamp_align_hetero_lambda0.050.pth \
  --model-type align_hetero \
  --y-target "2.5e8,200,1.5e6,65,20000" \
  --goal "min,min,range,range,min" \
  --ugf-band "8.0e5:2.0e6" \
  --pm-band "60:75" \
  --weights "0.05,0.40,0.90,0.10,0.65" \
  --prior 1e-3 \
  --n-init 512 --steps 800 --lr 0.002 \
  --finish-lbfgs 80 \
  --save-dir ../results/inverse/run_align

使用dualhead

python -m inverse_opt \
  --opamp 5t_opamp \
  --ckpt ../results/5t_opamp_dualhead_finetuned.pth \
  --model-type dualhead_b \
  --y-target "2.5e8,200,1.5e6,65,20000" \
  --goal "min,min,range,range,min"


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

---
## 一些参考公式
### 1. MSE (Mean Squared Error) - 均方误差

#### **公式 (Formula)**

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

* $n$: 样本数量
* $y_i$: 第 $i$ 个样本的真实值
* $\hat{y}_i$: 第 $i$ 个样本的模型预测值

#### **意义 (Meaning)**

MSE 计算的是预测值与真实值之差的平方的平均值。它衡量了模型的预测误差，值越小表示模型预测越精准。

#### **特点与应用 (Characteristics & Application)**

---

### 2. MAE (Mean Absolute Error) - 平均绝对误差

#### **公式 (Formula)**

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

#### **意义 (Meaning)**

MAE 计算的是预测值与真实值之差的绝对值的平均值。它直接衡量了预测值的平均误差大小。

---

### 3. R² (R-squared) - 决定系数

#### **公式 (Formula)**

$$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2} = 1 - \frac{\text{MSE}}{\text{Var}(y)}
$$

* $\bar{y}$: 所有真实值的平均值

#### **意义 (Meaning)**

R² 衡量的是模型能够解释的目标变量方差的比例。它的取值范围通常在 0 到 1 之间（但在某些情况下可能为负）。

* **$R^2 \approx 1$**: 模型几乎解释了所有的变异，拟合效果非常好。
* **$R^2 \approx 0$**: 模型的表现和直接用平均值进行预测差不多，拟合效果很差。
* **$R^2 < 0$**: 模型比直接用平均值预测还要糟糕。


---

### 4. CORAL (Correlation Alignment Loss) - 相关性对齐损失

#### **公式 (Formula)**

$$
L_{\text{CORAL}} = \frac{1}{4d^2} \| C_S - C_T \|_F^2
$$

* $d$: 特征的维度
* $C_S$: **源域**数据特征的协方差矩阵
* $C_T$: **目标域**数据特征的协方差矩阵
* $\| \cdot \|_F^2$: 矩阵的弗罗贝尼乌斯范数的平方（即矩阵所有元素的平方和）

#### **意义 (Meaning)**

CORAL 是一种用于**领域自适应 (Domain Adaptation)** 的损失函数。它的核心思想是：通过最小化源域和目标域特征分布的**二阶统计量（协方差）**的差异，来对齐两个域的特征分布。

简单来说，它迫使模型学习一种特征表示，在这种表示下，源域数据特征内部的相关性和目标域数据特征内部的相关性变得尽可能一致。

---

### 5. Heteroscedastic NLL - 异方差负对数似然损失

#### **公式 (Formula)**

假设预测的误差服从高斯分布 $N(\mu, \sigma^2)$，则损失函数为：

$$
L_{\text{NLL}} = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{(y_i - \mu_i)^2}{2\sigma_i^2} + \frac{1}{2}\log(\sigma_i^2) \right)
$$

在您的代码中，模型直接预测 $\mu_i$ 和对数方差 $\text{logvar}_i = \log(\sigma_i^2)$，所以公式变为：

$$
L_{\text{NLL}} = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{(y_i - \mu_i)^2}{2\exp(\text{logvar}_i)} + \frac{1}{2}\text{logvar}_i \right)
$$

#### **意义 (Meaning)**

这个损失函数让模型同时学习两件事：
1.  **预测均值 ($\mu_i$)**：即模型的直接预测值。
2.  **预测不确定性 ($\sigma_i^2$)**：模型为**每一个**预测给出一个方差，代表其置信度。

* **第一项 $\frac{(y_i - \mu_i)^2}{2\sigma_i^2}$**: 惩罚预测误差。这个惩罚会被模型预测的方差 $\sigma_i^2$ 所调节。如果模型对某个预测很不自信（给出了一个大的 $\sigma_i^2$），那么即使误差很大，这一项的惩罚也会变小。
* **第二项 $\frac{1}{2}\log(\sigma_i^2)$**: 正则化项。它惩罚模型无理由地给出过大的方差。如果没有这一项，模型会偷懒，对所有点都预测极大的方差来最小化第一项损失。
