# src/run_training.py (updated)

import argparse
import subprocess
import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import TASK_CONFIGS

def print_section(title: str):
    print("\n" + "="*60)
    print(f"=== {title.upper()} ===")
    print("="*60)

def run_command(command, cwd):
    print(f"\n[CMD] In '{cwd}': {' '.join(command)}")
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        for line in iter(process.stdout.readline, ''):
            print(line.strip())
        process.wait()
        rc = process.returncode
        if rc != 0:
            print(f"[ERROR] return code {rc}: {' '.join(command)}")
            return False
    except Exception as e:
        print(f"[ERROR] Exception while running {command}: {e}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(
        description="自动化训练流水线控制器 (A/B流程并存)",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--opamp", type=str, nargs='+', default=['all'],
        help="要训练的电路类型，'all' 或多个具体类型 (e.g., 5t_opamp two_stage_opamp)"
    )

    # 新增模式说明：
    #   align_hetero      -> 旧A版对齐训练脚本 (你的 unified_train / train_align_hetero)
    #   target_only       -> 旧A版 target-only baseline (如果你有)
    #   align_hetero_B    -> 新B版对齐训练 (train_align_hetero_B)
    #   target_only_B     -> 新B版 target-only baseline
    #   inverse           -> 反向模型
    #   all               -> A版: align_hetero, target_only, inverse
    #   all_B             -> B版: align_hetero_B, target_only_B, inverse

    parser.add_argument(
        "--mode", type=str, default="all",
        choices=[
            # 保留 A 版兼容（如果你还会用到）
            "all",
            "align_hetero",
            "target_only",

            # B 版主线
            "all_B",
            "align_hetero_B",
            "target_only_B",

            # 新增多目标 MoE
            "moe_multi",        # 只跑多目标 MoE 后处理
            "all_B_moe",        # B 主干 + 多目标 MoE + 反向

            "inverse",
        ],
        help=(
            "选择训练模式:\n"
            "  all          : A流程 align_hetero -> target_only -> inverse\n"
            "  all_B        : B流程 align_hetero_B -> target_only_B -> inverse\n"
            "  all_B_moe    : B流程 align_hetero_B -> moe_multi(多目标MoE) -> inverse\n"
            "  moe_multi    : 仅运行多目标MoE后处理（需已有 *_finetuned.pth）\n"
        )
    )


    args, unknown_args = parser.parse_known_args()

    # 这里把 mode 翻译成要跑的模块
    # 注意：这些模块名要和文件名对应（python -m <module>）
    script_map = {
        # A 流程（如需）
        "align_hetero":   "train_align_hetero",
        "target_only":    "train_target_only",

        # B 流程
        "align_hetero_B": "train_align_hetero_B",
        "target_only_B":  "train_target_only_B",

        # 多目标 MoE 后处理（新）
        "moe_multi":      "train_moe_residual_multi_B",

        # 反向
        "inverse":        "unified_inverse_train",
    }

    if args.mode == "all":
        scripts_to_run = ["train_align_hetero", "train_target_only", "unified_inverse_train"]
    elif args.mode == "all_B":
        scripts_to_run = ["train_align_hetero_B", "target_only_B", "unified_inverse_train"]
    elif args.mode == "all_B_moe":
        # 推荐的新整套：B 主干 + 多目标 MoE + 反向
        scripts_to_run = ["train_align_hetero_B", "train_moe_residual_multi_B", "unified_inverse_train"]
    else:
        scripts_to_run = [script_map[args.mode]]


    # 任务列表
    tasks_to_run = args.opamp
    if 'all' in tasks_to_run:
        tasks_to_run = list(TASK_CONFIGS.keys())

    print_section(f"PIPELINE START (mode={args.mode})")
    print(f"Tasks   : {', '.join(tasks_to_run)}")
    print(f"Scripts : {', '.join(scripts_to_run)}")
    print(f"Extra args forwarded to sub-scripts: {unknown_args}")

    for task in tasks_to_run:
        print_section(f"START TASK: {task}")
        for script_module in scripts_to_run:
            print(f"\n--- Running: {script_module}  (opamp={task}) ---")
            cmd = ["python", "-m", script_module, "--opamp", task]
            cmd.extend(unknown_args)

            ok = run_command(cmd, cwd=str(SRC_DIR))
            if not ok:
                print(f"\n[FATAL] Script '{script_module}' failed for opamp '{task}'. 停止该任务后续步骤。")
                break
        else:
            # 这个 else 是 for-else：只有内层循环没被break才会进入这里
            continue

        # 如果上面的内层循环 break 了（即某脚本失败），则也 break 外层
        break

    print_section("PIPELINE FINISHED")

if __name__ == "__main__":
    main()
