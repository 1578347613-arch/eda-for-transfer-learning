# eda-for-transfer-learning（当前分支简要说明）

本分支提供基于 PyTorch 的多目标回归迁移/域自适应训练流程。请从 src 目录运行脚本。

快速开始

1) 切到源码目录

```bash
cd src
```

2) 预训练 + 微调 + 验证（示例）

```bash
python train.py --evaluate
```

数据与输出

- 数据路径（相对 src）：data\01_train_set\{opamp}\source|target\*.csv
- 预处理：目标列 ugf、cmrr 使用 log1p；仅在源域拟合 StandardScaler 并应用到 A/B；评估时自动 expm1 还原
- 输出权重：../results\{opamp}_pretrained.pth 与 ../results\{opamp}_finetuned.pth

命令行参数（train.py）

- --opamp <str>：运放类型（默认见 src\config.py）
- --device <str>：设备 'cuda' 或 'cpu'（默认见 src\config.py）
- --restart：强制重新进行预训练阶段
- --save_path <str>：模型保存目录，默认 ../results
- --evaluate：训练完成后在验证集上评估
- --lr <float>：学习率（默认见 src\config.py）
- --epochs_finetune <int>：微调阶段总轮数（默认见 src\config.py）
- --epochs_pretrain <int>：预训练阶段总轮数（默认见 src\config.py）
- --batch_a <int>：源域批大小（默认见 src\config.py）
- --batch_b <int>：目标域批大小（默认见 src\config.py）
- --patience <int>：早停耐心轮数（默认见 src\config.py）
- --lambda_coral <float>：CORAL 损失权重（默认见 src\config.py）
- --alpha_r2 <float>：R² 损失权重（默认见 src\config.py）

依赖与环境

- Python >= 3.10；安装依赖：

```bash
pip install -r requirements.txt
```

目录结构（关键文件）

- src\config.py：默认超参与设备
- src\data_loader.py：加载/预处理/划分与提供 scaler
- src\loss_function.py：heteroscedastic_nll / batch_r2 / coral_loss
- src\evaluate.py：反标准化与指标（MSE/MAE/R²）
- src\train.py：预训练 + 微调主入口
- src\models\align_hetero.py, mlp.py, dual_head_mlp.py, model_utils.py

# day1 主要修改：
- config.py:新增LOG_TRANSFORMED_COLS = ["ugf","cmrr","dc_gain","slewrate_pos"]。同时，dataloader.py:SKEWED_COLS_DEFAULT = config.LOG_TRANSFORMED_COLS；evaluate.py:LOG_TRANSFORMED_COLS = config.LOG_TRANSFORMED_COLS
- config.py:新增LEARNING_RATE_FINETUNE = 1e-4,分别设置预训练和微调的学习率，set_args函数同步增加可选命令
- train.py:新增模拟退火和热重启scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=250, T_mult=1, eta_min=1e-6)
- dataloader.py:新增"source_train": (X_source_train, y_source_train),"source_val": (X_source_val, y_source_val)用来预训练
- train.py:预训练阶段采用新的损失函数：criterion = torch.nn.HuberLoss(delta=1)
- train.py:微调阶段backbone和head设置不同的学习率
- train.py:删除了预训练阶段的早停机制

# day2 主要修改
- 新建lr_finder.py，通过学习率范围检测，找到最佳学习率。（大致拟合的曲线变化最大的点）
- 借助lr_finder，确定lr_pretrain=3e-3,lr_finetune=7.6e-3
- 尝试修改batch_B为64，根据学习率和batchsize的线性缩放规则，lr_finetune=3.8e-3,经过lr_finder检验这种规则的合理性，√
- 目前cmrr R2稳定在0.8以上
- day3计划调整batch_A,调整微调阶段的损失函数各部分权重