# src/submit.py (参数化版本，用于自动化调用)
import os
import torch
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import sys
import ast

# --- 确保脚本可以找到您的自定义模块 ---
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# --- 从您的项目模块中导入 ---
try:
    from models.align_hetero import AlignHeteroMLP
    from data_loader import get_data_and_scalers, _add_physics_features
    import config
except ImportError as e:
    print(f"错误: 无法导入必要的模块。请确保此脚本位于 'src' 目录中。")
    print(f"详细信息: {e}")
    sys.exit(1)


def run_inference(opamp_type, model_path_str, output_file_str, test_file_str, hidden_dims, dropout_rate, device_str):
    """
    执行完整的推理流程：加载模型、加载数据、预处理、预测、后处理、保存。
    """
    print(f"--- [submit.py] 开始为 {opamp_type} 生成提交文件 ---")

    model_path = Path(model_path_str)
    test_file_path = Path(test_file_str)
    output_file_path = Path(output_file_str)
    device = torch.device(device_str)

    if not model_path.exists():
        print(f"❌ [submit.py] 错误: 找不到模型文件: {model_path}")
        return
    if not test_file_path.exists():
        print(f"❌ [submit.py] 错误: 找不到测试数据文件: {test_file_path}")
        return

    print("--- [submit.py] 正在加载 Scalers... ---")
    data_payload = get_data_and_scalers(opamp_type=opamp_type)
    x_scaler = data_payload['x_scaler']
    y_scaler = data_payload['y_scaler']
    train_x_cols = data_payload['raw_source'][0].columns.tolist()
    train_y_cols = data_payload['raw_source'][1].columns.tolist()

    print(f"--- [submit.py] 正在加载模型: {model_path.name} ---")
    model = AlignHeteroMLP(
        input_dim=x_scaler.n_features_in_,
        output_dim=y_scaler.n_features_in_,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"--- [submit.py] 正在读取和预处理测试数据... ---")
    X_test_df = pd.read_csv(test_file_path)
    print(f"--- [submit.py] 正在对测试集应用物理感知特征... ---")
    X_test_df_engineered = _add_physics_features(X_test_df, opamp_type)
    X_test_scaled = x_scaler.transform(X_test_df_engineered)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

    print(f"--- [submit.py] 正在执行模型推理... ---")
    with torch.no_grad():
        mu_scaled, _, _ = model(X_test_tensor)
        mu_scaled_np = mu_scaled.cpu().numpy()

    print("--- [submit.py] 正在反标准化和后处理... ---")
    y_pred_unscaled = y_scaler.inverse_transform(mu_scaled_np)
    y_pred_physical = y_pred_unscaled.copy()
    log_cols = config.LOG_TRANSFORMED_COLS

    for i, col_name in enumerate(train_y_cols):
        if col_name in log_cols:
            y_pred_physical[:, i] = np.expm1(y_pred_unscaled[:, i])

    print(f"--- [submit.py] 正在保存结果 (逗号分隔) 至: {output_file_path.name} ---")
    output_df = pd.DataFrame(y_pred_physical, columns=train_y_cols)
    output_df.to_csv(
        output_file_path,
        index=False,          # 不保存 pandas 的行索引
        sep=',',              # 使用逗号分隔
        float_format='%.10g'  # 保持与 np.savetxt 相同的浮点数精度
    )
    print(f"✅ [submit.py] 成功生成提交文件: {output_file_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="参数化的模型推理脚本")
    # 所有参数都是必需的，因为这是被自动化调用的
    parser.add_argument("--opamp", type=str, required=True,
                        help="运放类型 (e.g., 5t_opamp)")
    parser.add_argument("--model-path", type=str,
                        required=True, help="指向 .pth 模型文件的路径")
    parser.add_argument("--output-file", type=str,
                        required=True, help="输出的提交文件的路径")
    parser.add_argument("--test-file", type=str,
                        required=True, help="输入的测试数据 .csv 路径")
    parser.add_argument("--hidden-dims", type=str,
                        required=True, help="模型结构 e.g., '[128, 256]'")
    parser.add_argument("--dropout-rate", type=float,
                        required=True, help="模型的 dropout rate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备 'cuda' or 'cpu'")

    args = parser.parse_args()

    try:
        hidden_dims_list = ast.literal_eval(args.hidden_dims)
    except:
        print(f"错误: 无法解析 hidden-dims: {args.hidden_dims}")
        sys.exit(1)

    run_inference(
        args.opamp,
        args.model_path,
        args.output_file,
        args.test_file,
        hidden_dims_list,
        args.dropout_rate,
        args.device
    )
