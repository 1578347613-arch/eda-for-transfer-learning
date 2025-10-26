# 该文件专为在kaggle等远程平台运行所设

import os
import subprocess
import datetime
import argparse

# ==============================================================================
# --- 1. 定义您的实验方案 (Experiment Suite) ---
# ==============================================================================
# 在这里，您可以丰富、修改或删除任何您想尝试的权重组合。
# 每一个字典代表一次完整的 `train.py` 运行。
EXPERIMENTS = [
    # --- 阶段一：建立基准 (关闭所有辅助损失) ---
    {
        "name": "baseline",
        "lambda_nll": 1.0,
        "lambda_coral": 0.0,
        "alpha_r2": 0.0,
    },

    # --- 阶段二：探索 CORAL 损失的权重 (对数尺度搜索) ---
    {
        "name": "coral_search_low",
        "lambda_nll": 1.0,
        "lambda_coral": 0.01,
        "alpha_r2": 0.0,
    },
    {
        "name": "coral_search_medium",
        "lambda_nll": 1.0,
        "lambda_coral": 0.1,
        "alpha_r2": 0.0,
    },
    {
        "name": "coral_search_high",
        "lambda_nll": 1.0,
        "lambda_coral": 1.0,
        "alpha_r2": 0.0,
    },
    {
        "name": "coral_search_high",
        "lambda_nll": 1.0,
        "lambda_coral": 10.0,
        "alpha_r2": 0.0,
    },
    {
        "name": "coral_search_high",
        "lambda_nll": 1.0,
        "lambda_coral": 100.0,
        "alpha_r2": 0.0,
    },

    # --- 阶段三：在最佳 CORAL 基础上，引入 R2 正则化 ---


    # --- 阶段四：(可选) 探索 NLL 和 CORAL 的平衡 ---
]


def setup_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="微调实验管理器脚本")
    parser.add_argument("--opamp", type=str,
                        default="5t_opamp", help="要运行的运放类型")
    parser.add_argument("--results_dir", type=str,
                        default="../results", help="预训练模型所在及日志保存的目录")
    args = parser.parse_args()
    return args


def main():
    """主执行函数"""
    cli_args = setup_args()

    # 1. 检查预训练模型是否存在，这是脚本运行的前提
    pretrained_model_path = os.path.join(
        cli_args.results_dir, f"{cli_args.opamp}_pretrained.pth")
    if not os.path.exists(pretrained_model_path):
        print(f"❌ 错误：预训练模型未找到！")
        print(f"   脚本期望在以下路径找到模型: {pretrained_model_path}")
        print("   请先完成预训练，或检查路径。")
        return

    # 2. 创建一个带时间戳的日志文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(
        cli_args.results_dir, f"finetune_log_{timestamp}.txt")
    print(f"预训练模型已找到: {pretrained_model_path}")
    print(f"所有微调实验的输出将被记录到: {log_filename}")

    # 3. 依次执行每一个实验
    for i, exp in enumerate(EXPERIMENTS):
        exp_name = exp['name']
        header = f"\n\n{'='*35} EXPERIMENT {i+1}/{len(EXPERIMENTS)}: {exp_name} {'='*35}\n"
        params_info = f"Parameters: lambda_nll={exp['lambda_nll']}, lambda_coral={exp['lambda_coral']}, alpha_r2={exp['alpha_r2']}\n"

        print(header + params_info)

        # 将实验标题写入日志文件
        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(header)
            f.write(params_info)

        # 4. 构建要执行的 train.py 命令
        # 它会自动找到并使用已有的预训练模型
        command = [
            "python", "train.py",
            "--opamp", cli_args.opamp,
            "--save_path", cli_args.results_dir,  # 让 train.py 在默认路径下工作
            "--lambda_nll", str(exp['lambda_nll']),
            "--lambda_coral", str(exp['lambda_coral']),
            "--alpha_r2", str(exp['alpha_r2']),
            "--evaluate"
        ]

        # 5. 执行命令并将所有输出追加到日志文件
        try:
            # 使用 subprocess.run，等待命令完成后一次性获取所有输出
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,  # 如果失败则抛出异常
                encoding='utf-8'
            )

            # 将完整的输出写入日志文件
            with open(log_filename, "a", encoding="utf-8") as f:
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n--- STDERR ---\n")
                    f.write(result.stderr)

            print(f"✅ 实验 {exp_name} 完成！日志已记录。")

        except subprocess.CalledProcessError as e:
            # 如果命令执行失败，将错误信息也记录下来
            failure_msg = f"❌ 实验 {exp_name} 执行失败！\n--- STDOUT ---\n{e.stdout}\n--- STDERR ---\n{e.stderr}"
            print(failure_msg)
            with open(log_filename, "a", encoding="utf-8") as f:
                f.write(failure_msg)
            continue  # 继续下一个实验

    print(f"\n🎉 所有微调实验已执行完毕！完整日志已保存在: {log_filename}")


if __name__ == "__main__":
    main()
