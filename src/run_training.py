# /path/to/project/src/run_training.py (升级版)

import argparse
import subprocess
import os
import sys
from pathlib import Path

# --- 路径设置 ---
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import TASK_CONFIGS

def print_section(title: str):
    print("\n" + "="*50)
    print(f"=== {title.upper()} ===")
    print("="*50)

def run_command(command, cwd):
    print(f"\n[CMD] In directory '{cwd}', running: {' '.join(command)}")
    try:
        process = subprocess.Popen(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace'
        )
        for line in iter(process.stdout.readline, ''):
            print(line.strip())
        process.wait()
        retcode = process.returncode
        if retcode != 0:
            print(f"[ERROR] Command failed with return code {retcode}: {' '.join(command)}")
            return False
    except Exception as e:
        print(f"[ERROR] An exception occurred: {e}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(
        description="自动化训练流水线控制器",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--opamp", type=str, nargs='+', default=['all'],
        help="要训练的电路类型，'all' 或多个具体类型 (e.g., 5t_opamp 2stage_opamp)"
    )
    # --- 关键改动：更具体的 mode 选项 ---
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["all", "align_hetero", "target_only", "inverse"],
        help="选择训练模式:\n"
             "  - all: 依次运行所有训练 (align_hetero, target_only, inverse)\n"
             "  - align_hetero: 训练带对齐损失的正向模型 (原 forward)\n"
             "  - target_only: 训练仅使用目标域数据的正向模型\n"
             "  - inverse: 训练反向预测模型"
    )
    
    args, unknown_args = parser.parse_known_args()

    # --- 映射 mode 到脚本模块名 ---
    script_map = {
        "align_hetero": "train_align_hetero",  # 对应你重命名的 unified_train.py
        "target_only": "train_target_only",    # 对应新增的脚本
        "inverse": "unified_inverse_train"     # 保持不变
    }

    scripts_to_run = []
    if args.mode == 'all':
        scripts_to_run = list(script_map.values())
    elif args.mode in script_map:
        scripts_to_run = [script_map[args.mode]]
    else:
        raise ValueError(f"未知的 mode: {args.mode}")

    tasks_to_run = args.opamp
    if 'all' in tasks_to_run:
        tasks_to_run = list(TASK_CONFIGS.keys())

    print_section(f"AUTOMATED TRAINING (MODE: {args.mode.upper()})")
    print(f"Tasks: {', '.join(tasks_to_run)}")
    print(f"Scripts: {', '.join(scripts_to_run)}")

    for task in tasks_to_run:
        print_section(f"STARTING TASK: {task}")
        for script_module in scripts_to_run:
            print(f"\n--- Running: {script_module} for {task} ---")
            
            command = ["python", "-m", script_module, "--opamp", task]
            command.extend(unknown_args)
            
            if not run_command(command, cwd=str(SRC_DIR)):
                print(f"\n[FATAL] Script '{script_module}' failed for task '{task}'.")
                break # 失败则跳过该 opamp 的后续脚本
        else:
            continue # 循环正常结束，继续下一个 opamp
        break # 循环被 break，也跳出外层循环

    print_section("TRAINING PIPELINE FINISHED")

if __name__ == "__main__":
    main()
