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
    {
        "name": "baseline",
        "lambda_nll": 1.0,
        "lambda_coral": 0.0,
        "alpha_r2": 0,
    },
    {
        "name": "baseline",
        "lambda_nll": 1.0,
        "lambda_coral": 0.01,
        "alpha_r2": 0,
    },
    {
        "name": "baseline",
        "lambda_nll": 1.0,
        "lambda_coral": 0.1,
        "alpha_r2": 0,
    },
    {
        "name": "baseline",
        "lambda_nll": 1.0,
        "lambda_coral": 1.0,
        "alpha_r2": 0,
    },



    {
        "name": "baseline",
        "lambda_nll": 1.0,
        "lambda_coral": 0.0,
        "alpha_r2": 0,
    },
    {
        "name": "baseline",
        "lambda_nll": 1.0,
        "lambda_coral": 0.01,
        "alpha_r2": 0,
    },
    {
        "name": "baseline",
        "lambda_nll": 1.0,
        "lambda_coral": 0.1,
        "alpha_r2": 0,
    },
    {
        "name": "baseline",
        "lambda_nll": 1.0,
        "lambda_coral": 1.0,
        "alpha_r2": 0,
    },


    {
        "name": "baseline",
        "lambda_nll": 1.0,
        "lambda_coral": 0.0,
        "alpha_r2": 0,
    },
    {
        "name": "baseline",
        "lambda_nll": 1.0,
        "lambda_coral": 0.01,
        "alpha_r2": 0,
    },
    {
        "name": "baseline",
        "lambda_nll": 1.0,
        "lambda_coral": 0.1,
        "alpha_r2": 0,
    },
    {
        "name": "baseline",
        "lambda_nll": 1.0,
        "lambda_coral": 1.0,
        "alpha_r2": 0,
    },






    {
        "name": "newbaseline",
        "lambda_nll": 1.0,
        "lambda_coral": 0.1,
        "alpha_r2": 0.1,
    },
    {
        "name": "r2_search_low",
        "lambda_nll": 1.0,
        "lambda_coral": 0.1,
        "alpha_r2": 0.5,
    },
    {
        "name": "r2_search_medium",
        "lambda_nll": 1.0,
        "lambda_coral": 0.1,
        "alpha_r2": 1,
    },
    {
        "name": "r2_search_high",
        "lambda_nll": 1.0,
        "lambda_coral": 0.1,
        "alpha_r2": 2,
    },
    {
        "name": "r2_search_high",
        "lambda_nll": 1.0,
        "lambda_coral": 0.1,
        "alpha_r2": 5,
    },
    {
        "name": "r2_search_high",
        "lambda_nll": 1.0,
        "lambda_coral": 0.1,
        "alpha_r2": 10,
    },



    {
        "name": "baseline",
        "lambda_nll": 1.0,
        "lambda_coral": 0.1,
        "alpha_r2": 0.1,
    },
    {
        "name": "r2_search_low",
        "lambda_nll": 1.0,
        "lambda_coral": 0.1,
        "alpha_r2": 0.5,
    },
    {
        "name": "r2_search_medium",
        "lambda_nll": 1.0,
        "lambda_coral": 0.1,
        "alpha_r2": 1,
    },
    {
        "name": "r2_search_high",
        "lambda_nll": 1.0,
        "lambda_coral": 0.1,
        "alpha_r2": 2,
    },
    {
        "name": "r2_search_high",
        "lambda_nll": 1.0,
        "lambda_coral": 0.1,
        "alpha_r2": 5,
    },
    {
        "name": "r2_search_high",
        "lambda_nll": 1.0,
        "lambda_coral": 0.1,
        "alpha_r2": 10,
    },




    {
        "name": "baseline",
        "lambda_nll": 1.0,
        "lambda_coral": 0.1,
        "alpha_r2": 0.1,
    },
    {
        "name": "r2_search_low",
        "lambda_nll": 1.0,
        "lambda_coral": 0.1,
        "alpha_r2": 0.5,
    },
    {
        "name": "r2_search_medium",
        "lambda_nll": 1.0,
        "lambda_coral": 0.1,
        "alpha_r2": 1,
    },
    {
        "name": "r2_search_high",
        "lambda_nll": 1.0,
        "lambda_coral": 0.1,
        "alpha_r2": 2,
    },
    {
        "name": "r2_search_high",
        "lambda_nll": 1.0,
        "lambda_coral": 0.1,
        "alpha_r2": 5,
    },
    {
        "name": "r2_search_high",
        "lambda_nll": 1.0,
        "lambda_coral": 0.1,
        "alpha_r2": 10,
    },


    {
        "name": "baseline",
        "lambda_nll": 1.0,
        "lambda_coral": 0.1,
        "alpha_r2": 0.1,
    },
    {
        "name": "r2_search_low",
        "lambda_nll": 1.0,
        "lambda_coral": 0.1,
        "alpha_r2": 0.5,
    },
    {
        "name": "r2_search_medium",
        "lambda_nll": 1.0,
        "lambda_coral": 0.1,
        "alpha_r2": 1,
    },
    {
        "name": "r2_search_high",
        "lambda_nll": 1.0,
        "lambda_coral": 0.1,
        "alpha_r2": 2,
    },
    {
        "name": "r2_search_high",
        "lambda_nll": 1.0,
        "lambda_coral": 0.1,
        "alpha_r2": 5,
    },
    {
        "name": "r2_search_high",
        "lambda_nll": 1.0,
        "lambda_coral": 0.1,
        "alpha_r2": 10,
    },



    {
        "name": "baseline",
        "lambda_nll": 1.0,
        "lambda_coral": 0.1,
        "alpha_r2": 0.1,
    },
    {
        "name": "r2_search_low",
        "lambda_nll": 1.0,
        "lambda_coral": 0.1,
        "alpha_r2": 0.5,
    },
    {
        "name": "r2_search_medium",
        "lambda_nll": 1.0,
        "lambda_coral": 0.1,
        "alpha_r2": 1,
    },
    {
        "name": "r2_search_high",
        "lambda_nll": 1.0,
        "lambda_coral": 0.1,
        "alpha_r2": 2,
    },
    {
        "name": "r2_search_high",
        "lambda_nll": 1.0,
        "lambda_coral": 0.1,
        "alpha_r2": 5,
    },
    {
        "name": "r2_search_high",
        "lambda_nll": 1.0,
        "lambda_coral": 0.1,
        "alpha_r2": 10,
    },



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

# <<< 核心修改：新增一个用于解析结果的函数 >>>


def parse_evaluation_results(output_string: str) -> str:
    """
    从 train.py 的完整标准输出中，只提取最后的评估指标部分。
    """
    # 定义评估结果块的起始标志
    start_marker = "=== 目标域验证集指标（物理单位）==="

    try:
        # 找到起始标志在输出字符串中的位置
        start_index = output_string.rfind(start_marker)

        if start_index == -1:
            # 如果没有找到标志，说明评估可能未执行或失败
            return "评估结果未在输出中找到。\n"

        # 从起始标志开始，提取所有剩余的文本
        return output_string[start_index:]

    except Exception as e:
        return f"解析输出时发生错误: {e}\n"


def main():
    """主执行函数"""
    cli_args = setup_args()

    # 1. 检查预训练模型是否存在
    pretrained_model_path = os.path.join(
        cli_args.results_dir, f"{cli_args.opamp}_pretrained.pth")
    if not os.path.exists(pretrained_model_path):
        print(f"❌ 错误：预训练模型未找到！路径: {pretrained_model_path}")
        return

    # 2. 创建一个带时间戳的日志文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(
        cli_args.results_dir, f"finetune_summary_{timestamp}.txt")
    print(f"预训练模型已找到: {pretrained_model_path}")
    print(f"所有实验的总结将被记录到: {log_filename}")

    # 3. 依次执行每一个实验
    for i, exp in enumerate(EXPERIMENTS):
        exp_name = exp['name']
        header = f"\n\n{'='*35} EXPERIMENT {i+1}/{len(EXPERIMENTS)}: {exp_name} {'='*35}\n"
        params_info = f"Parameters: lambda_nll={exp['lambda_nll']}, lambda_coral={exp['lambda_coral']}, alpha_r2={exp['alpha_r2']}\n"

        print(header + params_info.strip())
        print("正在运行，请稍候...")

        # 构建命令 (与之前相同)
        command = [
            "python", "train.py",
            "--opamp", cli_args.opamp,
            "--save_path", cli_args.results_dir,
            "--lambda_nll", str(exp['lambda_nll']),
            "--lambda_coral", str(exp['lambda_coral']),
            "--alpha_r2", str(exp['alpha_r2']),
            "--evaluate"
        ]

        # 执行命令并捕获输出
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                encoding='utf-8'
            )

            # <<< 核心修改：只解析和记录评估结果 >>>
            evaluation_summary = parse_evaluation_results(result.stdout)

            # 将实验标题和解析后的结果写入日志文件
            with open(log_filename, "a", encoding="utf-8") as f:
                f.write(header)
                f.write(params_info)
                f.write(evaluation_summary)

            print(f"✅ 实验 {exp_name} 完成！结果已记录。")

        except subprocess.CalledProcessError as e:
            failure_msg = f"❌ 实验 {exp_name} 执行失败！\n--- STDERR ---\n{e.stderr}"
            print(failure_msg)
            with open(log_filename, "a", encoding="utf-8") as f:
                f.write(header)
                f.write(params_info)
                f.write(failure_msg)
            continue

    print(f"\n🎉 所有微调实验已执行完毕！总结报告已保存在: {log_filename}")


if __name__ == "__main__":
    main()
