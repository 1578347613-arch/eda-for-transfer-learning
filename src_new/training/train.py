# training/train.py
import os
import torch
import copy
from torch.utils.data import DataLoader, TensorDataset
from data_loader import get_data_and_scalers
from models.mlp import MLP
import config

# --- 1. 使用配置文件中的超参数 ---
OPAMP_TYPE = config.OPAMP_TYPE
BATCH_SIZE = config.BATCH_SIZE
LEARNING_RATE = config.LEARNING_RATE
HIDDEN_DIM = config.HIDDEN_DIM
NUM_LAYERS = config.NUM_LAYERS
EPOCHS = config.EPOCHS
DEVICE = torch.device(config.DEVICE)

def make_loader(x, y, bs, shuffle=True, drop_last=False):
    """
    创建 DataLoader
    """
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=shuffle, drop_last=drop_last)

def main():
    # 加载数据
    data = get_data_and_scalers(opamp_type=OPAMP_TYPE)
    X_src, y_src = data['source']
    X_val, y_val = data['target_val']

    # 创建 DataLoader
    train_loader = make_loader(X_src, y_src, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val, y_val, BATCH_SIZE, shuffle=False)

    # 初始化模型
    model = MLP(input_dim=X_src.shape[1], output_dim=y_src.shape[1], hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(DEVICE)

    # 损失函数与优化器
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 训练和验证过程
    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # 验证过程
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{EPOCHS}], 训练损失: {avg_train_loss:.6f}, 验证损失: {avg_val_loss:.6f}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs('results', exist_ok=True)
            torch.save(model.state_dict(), f'results/{OPAMP_TYPE}_baseline_model.pth')
            print(f"模型已保存，验证损失提升至: {best_val_loss:.6f}")

    print(f"最佳验证损失: {best_val_loss:.6f}")

if __name__ == "__main__":
    main()
