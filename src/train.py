# src/train.py

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# 从我们自己的模块中导入
from data_loader import get_data_and_scalers
from models import MLP

# --- 1. 设置超参数和设备 ---
EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
HIDDEN_DIM = 512
NUM_LAYERS = 6
OPAMP_TYPE = '5t_opamp'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# --- 2. 加载和准备数据 ---
data = get_data_and_scalers(opamp_type=OPAMP_TYPE)

# 预训练数据 (工艺A)
X_source, y_source = data['source']
# 验证数据 (工艺B的一部分)
X_val, y_val = data['target_val']

# 将NumPy数组转换为PyTorch Tensors
X_source_tensor = torch.tensor(X_source, dtype=torch.float32)
y_source_tensor = torch.tensor(y_source, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# 创建PyTorch DataLoader
train_dataset = TensorDataset(X_source_tensor, y_source_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# --- 3. 初始化模型、损失函数和优化器 ---
input_dim = X_source.shape[1]
output_dim = y_source.shape[1]

model = MLP(input_dim, output_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(device)
criterion = nn.MSELoss() # 均方误差损失，适用于回归问题
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE) # AdamW是Adam的改进版

print("模型已初始化:")
print(model)


# --- 4. 训练和验证循环 ---
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    # 训练模式
    model.train()
    total_train_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # 验证模式
    model.eval()
    total_val_loss = 0
    with torch.no_grad(): # 在验证时不需要计算梯度
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
            
    avg_val_loss = total_val_loss / len(val_loader)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}], 训练损失: {avg_train_loss:.6f}, 验证损失: {avg_val_loss:.6f}")

    # 保存表现最好的模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), f'results/{OPAMP_TYPE}_baseline_model.pth')
        print(f"模型已保存，验证损失提升至: {best_val_loss:.6f}")

print("\n--- 训练完成 ---")
print(f"最佳验证损失: {best_val_loss:.6f}")
