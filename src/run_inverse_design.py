# /path/to/project/run_inverse_design.py

import argparse
import subprocess
import os
from pathlib import Path

# --- 辅助函数，和 run_training.py 里的类似 ---
def run_command(command, cwd):
    """在指定目录下运行命令并处理输出"""
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
        print(f"[ERROR] An exception occurred while running the command: {e}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(
        description="反向设计优化控制器",
        formatter_class=argparse.RawTextHelpFormatter # 保持帮助信息格式
    )
    
    # --- 核心参数 ---
    parser.add_argument("--opamp", type=str, required=True, help="要进行反向设计的电路类型 (e.g., 5t_opamp)")
    parser.add_argument("--model-type", type=str, default="align_hetero", help="使用的正向模型类型")
    
    # --- 动态查找 checkpoint ---
    parser.add_argument("--ckpt", type=str, default=None, 
                        help="正向模型的权重文件路径。\n"
                             "如果留空，会自动在 'results/' 目录下查找 '{opamp}_finetuned.pth'")

    # --- 让用户可以覆盖保存目录 ---
    parser.add_argument("--save-dir", type=str, default=None, help="指定结果保存目录")

    # 解析已知和未知参数
    # 未知参数将全部传递给 inverse_opt.py
    args, unknown_args = parser.parse_known_args()

    # --- 准备命令 ---
    project_root = Path(__file__).resolve().parent
    src_dir = project_root 
    
    command = ["python", "-m", "inverse_opt", "--opamp", args.opamp]

    # 自动处理 checkpoint 路径
    ckpt_path = args.ckpt
    if not ckpt_path:
        # 自动查找默认的微调后模型
        potential_ckpt = project_root / "results" / f"{args.opamp}_finetuned.pth"
        if not potential_ckpt.exists():
            raise FileNotFoundError(
                f"自动查找模型失败，未找到 {potential_ckpt}。\n"
                "请使用 --ckpt 参数手动指定正向模型权重文件。"
            )
        ckpt_path = str(potential_ckpt)
        print(f"[INFO] 自动查找到模型权重: {ckpt_path}")
    
    command.extend(["--ckpt", ckpt_path])
    command.extend(["--model-type", args.model_type])

    # 处理保存目录
    if args.save_dir:
        command.extend(["--save-dir", args.save_dir])

    # 将所有其他参数原封不动地传递下去
    command.extend(unknown_args)

    print("=============================================")
    print(f"===   INVERSE DESIGN (OPTIMIZATION-BASED)  ===")
    print("=============================================")
    
    # 在 src 目录中执行命令
    run_command(command, cwd=str(src_dir))

    print("\n" + "="*45)
    print("===   INVERSE DESIGN FINISHED  ===")
    print("="*45)


if __name__ == "__main__":
    main()
