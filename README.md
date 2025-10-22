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
for src:
    <think>


      - 当前验证集（物理单位）指标（最终一版）：  
        - `slewrate_pos` R² **0.9839**  
        - `dc_gain` R² **0.9266**  
        - `ugf` R² **0.9460**  
        - `phase_margin` R² **0.9732**  
        - `cmrr` R² **0.6160**
      
      ---
      
      # 工程结构（含说明）
      ```
      eda-for-transfer-learning-1/
      ├── 1_EDA_5T_Opamp.ipynb        # EDA与分布可视化
      ├── README.md                   # 项目说明（可补充今日内容）
      ├── data/
      │   ├── 01_train_set/
      │   │   ├── 5t_opamp/           # 今天主要使用的数据集（A/B）
      │   │   │   ├── source/         # A工艺：pretrain_design_features.csv / pretrain_targets.csv
      │   │   │   └── target/         # B工艺：target_design_features.csv / target_targets.csv
      │   │   └── two_stage_opamp/    # 预留的两级运放同构目录
      │   └── 02_public_test_set/     # 公测特征（A/B/C/D），后续推理可用
      ├── notebooks/                  # 笔记本占位
      ├── results/                    # 训练产物
      │   ├── 5t_opamp_x_scaler.gz
      │   ├── 5t_opamp_y_scaler.gz
      │   ├── 5t_opamp_baseline_model.pth
      │   ├── 5t_opamp_dualhead_finetuned.pth
      │   ├── 5t_opamp_align_hetero_lambda0.050.pth
      │   └── 5t_opamp_target_only_hetero.pth
      └── src/
          ├── data_loader.py          # 数据读取/预处理/划分 & 保存scalers
          ├── models.py               # MLP/DualHead/AlignHeteroMLP 定义
          ├── losses.py               # hetero NLL / R² / CORAL
          ├── train.py                # A域基线预训练（MLP）
          ├── fine_tune.py            # DualHead三阶段B域微调
          ├── train_align_coral.py    # A↔B特征对齐 + B监督（异方差）
          ├── train_target_only.py    # 仅B域训练异方差模型（今日新增）
          ├── infer_ensemble.py       # 两模型动态加权集成（今日增强）
          └── evaluate.py             # 逆标准化评估（MSE/MAE/R²）
      ```
      
      ---
      
      # 各文件内容与接口（简要）
      
      ### `src/data_loader.py`
      - **作用**：加载 A/B 数据 → `log1p`(`ugf, cmrr`) → **仅在A** 上 `StandardScaler.fit`（X/Y 各一套）→ 统一 transform A/B → B 按 80/20 划分。保存 `x_scaler.gz / y_scaler.gz`（在 `results/`）。
      - **主接口**
        ```python
        data = get_data_and_scalers(opamp_type='5t_opamp')
        # 返回：
        # data['source']       = (X_A, y_A)
        # data['target_train'] = (X_B_tr, y_B_tr)
        # data['target_val']   = (X_B_va, y_B_va)
        ```
      
      ### `src/models.py`
      - **MLP**：`MLP(input_dim, output_dim, hidden_dim=512, num_layers=6, dropout_rate=0.1)`
      - **DualHeadMLP**：共享主干 + A/B 两输出头（B域微调用）
      - **AlignHeteroMLP**：共享主干 + B域 `mu`/`logvar` 双头（异方差）
      - **常用**：`model(x)` 返回 `(mu, logvar, feat)`；`feat` 为对齐用中间特征。
      
      ### `src/losses.py`
      - `heteroscedastic_nll(mu, logvar, y, reduction='mean')`
      - `batch_r2(y_true, y_pred, eps=1e-8)`
      - `coral_loss(feat_a, feat_b, unbiased=True, eps=1e-6)`
      
      ### `src/train.py`
      - **A域基线**（MLP），监控 B 验证，落盘：`../results/5t_opamp_baseline_model.pth`  
        运行：`python src/train.py`
      
      ### `src/fine_tune.py`
      - **三阶段微调**（DualHead）：Bias-only → Head-only(Huber+L2-SP) → 轻解冻末层。  
        落盘：`../results/5t_opamp_dualhead_finetuned.pth`  
        运行：`python src/fine_tune.py`
      
      ### `src/train_align_coral.py`
      - **A↔B特征对齐 + B监督**（AlignHeteroMLP；NLL + α·(1−R²) + λ·CORAL）。  
        今日已跑 λ=0.050，最佳 `valNLL ≈ -1.1843`。  
        落盘：`../results/5t_opamp_align_hetero_lambda0.050.pth`  
        运行：`python src/train_align_coral.py`
      
      ### `src/train_target_only.py`（今日新增）
      - **仅B域**训练异方差模型（AlignHeteroMLP）；早停保存最佳。  
        落盘：`../results/5t_opamp_target_only_hetero.pth`  
        运行：`python src/train_target_only.py`
      
      ### `src/infer_ensemble.py`（今日增强）
      - **做法**：  
        1) 两模型前向得 `mu, logvar`（标准化空间）；  
        2) **温度标定**（每模型、每指标）修正 `logvar`；  
        3) **precision** 裁剪（95%分位）避免单边爆权重；  
        4) **混合加权**：`w = α·precision(样本级) + (1−α)·MSE(维度级)`；  
        5) 加权融合均值 → 逆标准化（`ugf, cmrr` 再 `expm1`）→ 打印 MSE/MAE/R²。  
        运行：`python src/infer_ensemble.py`
      
      ### `src/evaluate.py`
      - **用途**：对任意模型输出做逆标准化评估（物理单位的 MSE/MAE/R²）。  
        运行：`python src/evaluate.py`
      
      ---
      
      # 典型运行顺序（命令速记）
      ```bash
      # 1) A域基线
      python src/train.py
      
      # 2) B域微调（可选）
      python src/fine_tune.py
      
      # 3) 对齐训练（A↔B + NLL + CORAL）
      python src/train_align_coral.py   # λ=0.05 已完成
      
      # 4) 目标域单模型（B-only）
      python src/train_target_only.py
      
      # 5) 集成评估（温度标定 + precision裁剪 + 混合加权）
      python src/infer_ensemble.py
      ```
      
      ---
      
      # 明天可选待办（轻量）
      - 对齐训练里**去掉 A 分支的 `no_grad`**、加 **λ 预热**（能稳住 `cmrr`）。  
      - 多跑两份对齐模型（λ=0.02/0.10），在集成里**按指标择优**。  
      - `losses.py` 增加 **稳定版 NLL（logvar裁剪+惩罚）**，减少“过度自信”。
      
      辛苦啦，今天的管线已经从数据→训练→对齐→目标模型→集成→评估全打通 ✅
、
