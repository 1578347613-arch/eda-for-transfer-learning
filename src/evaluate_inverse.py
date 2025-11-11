# /src/evaluate_inverse.py (修正 v2 - 导入 Tuple)

import argparse
from pathlib import Path
import numpy as np
import torch
import joblib
import sys
from tqdm import tqdm
import math
from typing import Tuple # <--- 加上这一行！

# --- 路径和导入设置 (保持和 generate_submission 一致) ---
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "src"/"results" # <-- 确保指向正确的 results 目录！
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path: sys.path.insert(0, str(SRC_DIR))

# --- 导入我们需要的模块和函数 ---
import config # 导入我们修复好的扁平 config
from data_loader import get_data_and_scalers
# 从 generate_submission 导入模型加载函数 (它们已经被修复了！)
from generate_submission import load_forward_model, load_inverse_mdn_model, INVERSE_OUTPUT_COLS # <-- 从这里导入 INVERSE_OUTPUT_COLS
# 从 inverse_opt 导入核心优化函数和一些工具
from inverse_opt import optimize_x_multi_start, Y_LOG1P_INDEX, Y_NAMES # <-- 不再导入 INVERSE_OUTPUT_COLS

# --- 全局配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 核心评估函数 ---
def calculate_inverse_metrics(x_pred_phys: np.ndarray, x_true_phys: np.ndarray, eps=1e-9) -> Tuple[np.ndarray, np.ndarray]: # <-- 现在 Tuple 被认识了！
    """
    根据比赛 PDF 计算 MSE' 和 MAE'。
    输入 x_pred_phys 和 x_true_phys 必须是物理单位！
    """
    if x_pred_phys.shape != x_true_phys.shape:
        raise ValueError("预测值和真实值的形状不匹配！")
    
    # 保证分母不为零或极小值
    x_true_safe = np.where(np.abs(x_true_phys) < eps, eps, x_true_phys)
    
    # 计算相对误差 err = (y_k - y_k') / y_k  (PDF里的 y_k 是指 x_true, y_k' 是指 x_pred)
    relative_error = (x_true_phys - x_pred_phys) / x_true_safe
    
    # --- 计算每个维度的 MSE' 和 MAE' ---
    # MSE' = mean( ((x_true - x_pred) / x_true)^2 ) per dimension
    mse_prime_per_dim = np.mean(relative_error**2, axis=0)
    # MAE' = mean( |(x_true - x_pred) / x_true| ) per dimension
    mae_prime_per_dim = np.mean(np.abs(relative_error), axis=0)

    return mse_prime_per_dim, mae_prime_per_dim

def calculate_mock_fom(mse_prime_per_dim: np.ndarray, mae_prime_per_dim: np.ndarray) -> float:
    """
    根据验证集上的 MSE' 和 MAE' 计算一个“模拟 FoM”。
    这里我们用验证集上的最大值作为 _worst 的近似。
    """
    # 简单 FoM: 直接对 MSE' 和 MAE' 的平均值（在所有维度上）进行加权，越小越好。
    mock_fom = 0.5 * np.mean(mse_prime_per_dim) + 0.5 * np.mean(mae_prime_per_dim)
    print(f"[模拟 FoM 计算] 使用简化公式: 0.5*mean(MSE') + 0.5*mean(MAE') = {mock_fom:.6f} (越小越好)")
    
    return mock_fom

def main():
    parser = argparse.ArgumentParser(description="反向设计 C/D 题模拟评估脚本")
    parser.add_argument("--opamp", type=str, required=True, help="要评估的运放类型 (e.g., 5t_opamp)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    DEVICE = torch.device(args.device)

    print("="*50)
    print(f"开始反向设计模拟评估: {args.opamp}")
    print(f"评估将在验证集 (target_val) 上进行")
    print("="*50)

    # --- 1. 加载所有需要的东西 ---
    print("\n--- [步骤 1/4] 加载数据, Scalers 和模型 ---")
    try:
        # 加载数据和 Scalers
        all_data = get_data_and_scalers(opamp_type=args.opamp)
        x_val_scaled, y_val_scaled = all_data['target_val'] # 我们需要 x_val 作为标准答案！
        x_scaler = all_data['x_scaler']
        y_scaler = all_data['y_scaler']
        print("数据和 Scalers 加载成功。")

        # 自动查找并加载模型
        mdn_model_path = RESULTS_DIR / f"mdn_{args.opamp}.pth"
        fwd_model_path = RESULTS_DIR / f"{args.opamp}_finetuned.pth"
        
        mdn_model = load_inverse_mdn_model(args.opamp, mdn_model_path)
        # 注意：这里我们直接用 generate_submission 里的 load_forward_model，因为它已经被修复了！
        fwd_model = load_forward_model(args.opamp, fwd_model_path) 
        print("MDN 和正向模型加载成功。")

    except FileNotFoundError as e:
        print(f"[错误] 加载失败: {e}")
        print("请确保已成功运行 unified_inverse_train.py 和 train.py (用于正向模型)。")
        sys.exit(1)
    except Exception as e:
        print(f"[错误] 初始化失败: {e}")
        sys.exit(1)
        
    if len(x_val_scaled) == 0:
        print("[警告] 验证集为空，无法进行评估。")
        sys.exit(0)

    # --- 2. 模拟考试：遍历验证集，进行反向设计 ---
    print(f"\n--- [步骤 2/4] 在 {len(y_val_scaled)} 个验证集目标上运行反向设计 ---")
    # ***** 新增 *****
    x_init_scaled_list = [] # <--- 用于收集 MDN 的初始猜测
    # ***** 结束新增 *****
    x_pred_scaled_list = []
    
    # 获取反向输出的维度
    x_dim = x_val_scaled.shape[1] 


    for i in tqdm(range(len(y_val_scaled)), desc=f"Evaluating {args.opamp}"):
        y_target_scaled_row = y_val_scaled[i]
        
        # a. MDN 猜测初始解
        with torch.no_grad():
            y_tensor_row = torch.tensor(y_target_scaled_row, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            pi, mu, _ = mdn_model(y_tensor_row)
            # 使用期望值作为初始猜测
            x_init_scaled_row = torch.sum(pi.unsqueeze(-1) * mu, dim=1).cpu().numpy()
            
            # ***** 新增 *****
            x_init_scaled_list.append(x_init_scaled_row[0]) # <--- 收集初始猜测！
            # ***** 结束新增 *****

        # b. 调用 inverse_opt 进行优化
        best_x_scaled_row, _, _ = optimize_x_multi_start(
            model=fwd_model, 
            model_type="align_hetero", # 假设我们总是用 align_hetero
            x_dim=x_dim, 
            y_target_scaled=y_target_scaled_row,
            x_scaler=x_scaler, 
            y_scaler=y_scaler, 
            init_points_scaled=x_init_scaled_row, 
            device=str(DEVICE),
            opamp_type=args.opamp, 
        )
        # optimize_x_multi_start 返回的是 numpy array，直接取第一个元素
        if best_x_scaled_row is not None and len(best_x_scaled_row) > 0:
             x_pred_scaled_list.append(best_x_scaled_row[0]) 
        else:
             # 处理优化失败的情况，可以用初始猜测或 NaN 填充？
             print(f"[警告] 样本 {i} 的优化未能返回有效结果，使用初始猜测填充。")
             # ***** 修改 *****
             # 如果优化失败，最终结果也用初始猜测填充
             x_pred_scaled_list.append(x_init_scaled_row[0]) # <--- 改用初始猜测填充
             # ***** 结束修改 *****
    
    # ***** 新增 *****
    x_init_scaled = np.array(x_init_scaled_list) # <--- 把所有初始猜测变成 Numpy 数组
    # ***** 结束新增 *****
    x_pred_scaled = np.array(x_pred_scaled_list)

    # --- 3. 阅卷：计算分数 ---
    print("\n--- [步骤 3/4] 计算 MSE' 和 MAE' (物理单位) ---")
    
    # 先反标准化回物理单位
    # ***** 新增 *****
    x_init_phys = x_scaler.inverse_transform(x_init_scaled) # <--- 反标 MDN 猜测！
    # ***** 结束新增 *****
    x_pred_phys = x_scaler.inverse_transform(x_pred_scaled)
    x_true_phys = x_scaler.inverse_transform(x_val_scaled) # x_val 也要反标准化！
    
    # 获取正确的列名
    x_names = INVERSE_OUTPUT_COLS.get(args.opamp)
    if not x_names or len(x_names) != x_dim:
         x_names = [f"x{i}" for i in range(x_dim)] # Fallback

    try:
        # ***** 修改：计算两组成绩！ *****
        print("\n--- 正在计算 MDN 初始猜测的得分 ---")
        mse_prime_init, mae_prime_init = calculate_inverse_metrics(x_init_phys, x_true_phys)
        
        print("\n--- 正在计算最终优化结果的得分 ---")
        mse_prime_final, mae_prime_final = calculate_inverse_metrics(x_pred_phys, x_true_phys)
        # ***** 结束修改 *****
        
        
        # ***** 修改：打印两组成绩！ *****
        print("\n=== C/D 题模拟评估结果 (验证集) ===")
        print("--- MDN 初始猜测得分 ---")
        for i, name in enumerate(x_names):
             if i < len(mse_prime_init) and i < len(mae_prime_init):
                 print(f"{name:10s}  MSE'={mse_prime_init[i]:.6f}  MAE'={mae_prime_init[i]:.6f}")
             else: print(f"{name:10s}  指标计算出错")
        print("\nAvg (MDN Init)   MSE'={:.6f}  MAE'={:.6f}".format(
            np.mean(mse_prime_init), np.mean(mae_prime_init)
        ))
        
        print("\n--- 最终优化结果得分 ---")
        for i, name in enumerate(x_names):
             if i < len(mse_prime_final) and i < len(mae_prime_final):
                 print(f"{name:10s}  MSE'={mse_prime_final[i]:.6f}  MAE'={mae_prime_final[i]:.6f}")
             else: print(f"{name:10s}  指标计算出错")
        print("\nAvg (Final Opt)  MSE'={:.6f}  MAE'={:.6f}".format(
            np.mean(mse_prime_final), np.mean(mae_prime_final)
        ))
        # ***** 结束修改 *****

    except Exception as e:
        print(f"[错误] 计算分数时出错: {e}")
        # 即使算分出错，我们也要看预测值
        print("\n预测值 (物理单位) - 前5个:")
        print(x_pred_phys[:5]) 
        print("\n真实值 (物理单位) - 前5个:")
        print(x_true_phys[:5])

    # --- 4. (可选) 计算模拟 FoM ---
    print("\n--- [步骤 4/4] 计算模拟 FoM ---")
    try:
        # ***** 修改：确保用 final 的分数算 FoM *****
        if 'mse_prime_final' in locals() and 'mae_prime_final' in locals():
            mock_fom = calculate_mock_fom(mse_prime_final, mae_prime_final) # <-- 用 final 的！
        # ***** 结束修改 *****
        else:
            print("由于分数计算失败，无法计算模拟 FoM。")
    except NameError: 
        print("由于分数计算失败，无法计算模拟 FoM。")
    except Exception as e:
        print(f"[错误] 计算模拟 FoM 时出错: {e}")
        
    print("\n" + "="*50)
    print("模拟评估完成！")
    print("="*50)


if __name__ == "__main__":
    main()