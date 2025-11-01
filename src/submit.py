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
    from data_loader import get_data_and_scalers
    import config  # <-- 导入 config 文件
except ImportError as e:
    print(f"错误: 无法导入必要的模块。请确保此脚本位于 'src' 目录中。")
    print(f"详细信息: {e}")
    sys.exit(1)


def run_inference(opamp_type, model_path, test_file_path, output_file_path, hidden_dims, dropout_rate, device):
    """
    执行完整的推理流程：加载模型、加载数据、预处理、预测、后处理、保存。
    """
    print(f"--- [submit.py] 开始为 {opamp_type} 生成提交文件 ---")

    # --- 1. 检查路径 ---
    if not model_path.exists():
        print(f"❌ [submit.py] 错误: 找不到模型文件: {model_path}")
        print(f"   请先运行 train.py 生成 {model_path.name} 文件。")
        return
    if not test_file_path.exists():
        print(f"❌ [submit.py] 错误: 找不到测试数据文件: {test_file_path}")
        return

    # --- 2. 加载 Scalers ---
    print("--- [submit.py] 正在加载 Scalers... ---")
    data_payload = get_data_and_scalers(opamp_type=opamp_type)
    x_scaler = data_payload['x_scaler']
    y_scaler = data_payload['y_scaler']
    train_x_cols = data_payload['raw_source'][0].columns.tolist()
    train_y_cols = data_payload['raw_source'][1].columns.tolist()

    # --- 3. 加载模型 ---
    print(f"--- [submit.py] 正在加载模型: {model_path.name} ---")
    model = AlignHeteroMLP(
        input_dim=x_scaler.n_features_in_,
        output_dim=y_scaler.n_features_in_,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- 4. 加载并预处理测试数据 ---
    print(f"--- [submit.py] 正在读取和预处理测试数据... ---")
    X_test_df = pd.read_csv(test_file_path)
    X_test_df_reordered = X_test_df[train_x_cols]
    X_test_scaled = x_scaler.transform(X_test_df_reordered)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

    # --- 5. 执行模型推理 ---
    print(f"--- [submit.py] 正在执行模型推理... ---")
    with torch.no_grad():
        mu_scaled, _, _ = model(X_test_tensor)
        mu_scaled_np = mu_scaled.cpu().numpy()

    # --- 6. 后处理 (反标准化) ---
    print("--- [submit.py] 正在反标准化和后处理... ---")
    y_pred_unscaled = y_scaler.inverse_transform(mu_scaled_np)
    y_pred_physical = y_pred_unscaled.copy()
    log_cols = config.LOG_TRANSFORMED_COLS

    for i, col_name in enumerate(train_y_cols):
        if col_name in log_cols:
            y_pred_physical[:, i] = np.expm1(y_pred_unscaled[:, i])  # 还原log变换

    # --- 7. 保存为指定格式 ---
    print(f"--- [submit.py] 正在保存结果 (逗号分隔) 至: {output_file_path.name} ---")
    np.savetxt(
        output_file_path,
        y_pred_physical,
        fmt='%.10g',  # 使用通用浮点数格式
        delimiter=','   # 使用逗号分隔
    )
    print(f"✅ [submit.py] 成功生成提交文件: {output_file_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型推理脚本 (从config.py读取默认值)")

    # --- 所有参数现在都是可选的 ---
    parser.add_argument("--opamp", type=str, default=None,
                        help=f"运放类型 (默认: 从 config.py 读取 '{config.OPAMP_TYPE}')")
    parser.add_argument("--model-path", type=str, default=None,
                        help="指向 .pth 模型文件的路径 (默认: 自动推断)")
    parser.add_argument("--test-file", type=str, default=None,
                        help="输入的测试数据 .csv 路径 (默认: 自动推断)")
    parser.add_argument("--output-file", type=str, default=None,
                        help="输出的提交文件的路径 (默认: 自动推断)")
    parser.add_argument("--hidden-dims", type=str, default=None,
                        help=f"模型结构 (默认: 从 config.py 读取 '{config.HIDDEN_DIMS}')")
    parser.add_argument("--dropout-rate", type=float, default=None,
                        help=f"Dropout (默认: 从 config.py 读取 '{config.DROPOUT_RATE}')")
    parser.add_argument("--device", type=str, default=None,
                        help=f"设备 (默认: 从 config.py 读取 '{config.DEVICE}')")

    args = parser.parse_args()

    print("="*60)
    print("🚀 开始生成提交文件...")

    # --- 核心逻辑：使用 config.py 作为默认值 ---

    # 1. 设置基础参数
    opamp_type = args.opamp if args.opamp else config.OPAMP_TYPE
    device = torch.device(args.device if args.device else config.DEVICE)

    if args.hidden_dims:
        hidden_dims_list = ast.literal_eval(args.hidden_dims)
    else:
        hidden_dims_list = config.HIDDEN_DIMS

    dropout_rate = args.dropout_rate if args.dropout_rate is not None else config.DROPOUT_RATE

    # 2. 根据 opamp_type 自动推断文件路径
    if opamp_type == '5t_opamp':
        default_test_file = PROJECT_ROOT / \
            "data/02_public_test_set/features/features_A.csv"
        default_output_file = PROJECT_ROOT / "predA"
    elif opamp_type == 'two_stage_opamp':  # 增加了对B文件的支持
        default_test_file = PROJECT_ROOT / \
            "data/02_public_test_set/features/features_B.csv"
        default_output_file = PROJECT_ROOT / "predB"
    else:
        print(f"❌ 错误: 未知的 opamp_type '{opamp_type}'")
        sys.exit(1)

    # 默认模型路径 (假设 'results' 目录在项目根目录)
    default_model_path = PROJECT_ROOT / \
        "results" / f"{opamp_type}_finetuned.pth"

    # 3. 决定最终使用的路径 (命令行参数优先)
    model_path = Path(
        args.model_path) if args.model_path else default_model_path
    test_file_path = Path(
        args.test_file) if args.test_file else default_test_file
    output_file_path = Path(
        args.output_file) if args.output_file else default_output_file

    print(f"--- 最终配置 ---")
    print(f"  - Opamp 类型: {opamp_type}")
    print(f"  - 模型文件:   {model_path}")
    print(f"  - 测试文件:   {test_file_path}")
    print(f"  - 输出文件:   {output_file_path}")
    print(f"  - 设备:       {device}")
    print(f"  - 结构:       {hidden_dims_list}")
    print("="*60)

    # 4. 执行推理
    run_inference(
        opamp_type=opamp_type,
        model_path=model_path,
        test_file_path=test_file_path,
        output_file_path=output_file_path,
        hidden_dims=hidden_dims_list,
        dropout_rate=dropout_rate,
        device=device
    )
