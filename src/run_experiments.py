# src/run_experiments.py (已更新：记录评估日志 + 自动提交)
import subprocess
import os
import time
import json
import shutil
from pathlib import Path
import logging  # <-- 导入日志模块
import sys
import re  # <-- 导入正则表达式模块

# --- 从项目模块中导入 ---
from find_lr_utils import find_pretrain_lr
from models.align_hetero import AlignHeteroMLP
from data_loader import get_data_and_scalers
from find_lr_utils import find_pretrain_lr, find_finetune_lr
import config  # <-- 导入 config 以获取默认值

# ==============================================================================
# --- 0. 路径和实验控制 ---
# ==============================================================================
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# --- 核心修改：不再自动清理，因为 submit.py 需要 .pth 文件 ---
CLEANUP_AFTER_RUN = False
SILENT_TRAINING = True

# ==============================================================================
# --- 1. 定义你的实验搜索空间 ---
# ==============================================================================
BASE_EXPERIMENT_GRID = [
    {"name": "0.2_ratio3", "hidden_dims": [
        128, 256, 256, 512], "dropout_rate": 0.2, "backbone_lr_ratio": 3.0},
    {"name": "0.2_ratio4", "hidden_dims": [
        128, 256, 256, 512], "dropout_rate": 0.2, "backbone_lr_ratio": 4.0},
    {"name": "0.2_ratio5", "hidden_dims": [
        128, 256, 256, 512], "dropout_rate": 0.2, "backbone_lr_ratio": 5.0},
    {"name": "0.2_ratio6", "hidden_dims": [
        128, 256, 256, 512], "dropout_rate": 0.2, "backbone_lr_ratio": 6.0},
    {"name": "0.2_ratio7", "hidden_dims": [
        128, 256, 256, 512], "dropout_rate": 0.2, "backbone_lr_ratio": 7.0},
    {"name": "0.2_ratio8", "hidden_dims": [
        128, 256, 256, 512], "dropout_rate": 0.2, "backbone_lr_ratio": 8.0},
    {"name": "0.2_ratio9", "hidden_dims": [
        128, 256, 256, 512], "dropout_rate": 0.2, "backbone_lr_ratio": 9.0},
    {"name": "0.2_ratio10", "hidden_dims": [
        128, 256, 256, 512], "dropout_rate": 0.2, "backbone_lr_ratio": 10.0},
    {"name": "0.2_ratio11", "hidden_dims": [
        128, 256, 256, 512], "dropout_rate": 0.2, "backbone_lr_ratio": 11.0},
    {"name": "0.2_ratio12", "hidden_dims": [
        128, 256, 256, 512], "dropout_rate": 0.2, "backbone_lr_ratio": 12.0},
    {"name": "0.2_ratio13", "hidden_dims": [
        128, 256, 256, 512], "dropout_rate": 0.2, "backbone_lr_ratio": 13.0},
    {"name": "0.2_ratio14", "hidden_dims": [
        128, 256, 256, 512], "dropout_rate": 0.2, "backbone_lr_ratio": 14.0},
    {"name": "0.2_ratio15", "hidden_dims": [
        128, 256, 256, 512], "dropout_rate": 0.2, "backbone_lr_ratio": 15.0},
]

# --- 实验控制设置 ---
NUM_REPETITIONS = 1
OPAMP_TYPE = '5t_opamp'
BASE_RESULTS_DIR = PROJECT_ROOT / "results_experiments_auto_lr"

# --- 提交文件设置 ---
TEST_FILE_PATH = PROJECT_ROOT / "data/02_public_test_set/features/features_A.csv"
SUBMISSION_FILE_PREFIX = "predA"  # 将生成 predA_1, predA_2 ...

# ==============================================================================
# --- 2. 设置日志系统 ---
# ==============================================================================
BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_LOG_FILE = BASE_RESULTS_DIR / "experiment_evaluation_log.txt"

# 创建一个只写入文件的 logger
file_logger = logging.getLogger('ExperimentLogger')
file_logger.setLevel(logging.INFO)
file_logger.propagate = False  # 防止日志向上传播
# 移除所有现有的 handlers (如果在 notebook 中重跑)
if file_logger.hasHandlers():
    file_logger.handlers.clear()
# 创建文件 handler
file_handler = logging.FileHandler(
    EVALUATION_LOG_FILE, mode='w', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(message)s'))
file_logger.addHandler(file_handler)

# ==============================================================================
# --- 3. 动态生成完整的实验列表 ---
# ==============================================================================
# ... (这部分逻辑不变) ...
EXPERIMENT_GRID = []
for exp_params in BASE_EXPERIMENT_GRID:
    for run_num in range(1, NUM_REPETITIONS + 1):
        new_params = exp_params.copy()
        new_params['name'] = f"{exp_params['name']}_run{run_num}"
        new_params['base_name'] = exp_params['name']
        EXPERIMENT_GRID.append(new_params)

# ==============================================================================
# --- 4. 辅助函数 ---
# ==============================================================================


def run_command(command, log_prefix=""):
    """
    辅助函数：执行子进程并捕获所有输出。
    返回: (bool: success, list: stdout_lines, str: stderr_output)
    """
    print(f"--- [CMD] {log_prefix} 正在执行: {' '.join(command)} ---")
    process = subprocess.Popen(
        command, cwd=SRC_DIR, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, encoding='utf-8'
    )
    stdout_lines = []

    # 实时打印到控制台 (如果非静默)
    if not SILENT_TRAINING:
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if line:
                print(f"[{log_prefix}] {line}")
            stdout_lines.append(line)
        process.wait()
        stderr_output = process.stderr.read()
    else:
        # 静默模式：只捕获，不打印
        stdout_data, stderr_output = process.communicate()
        stdout_lines = stdout_data.splitlines()

    if process.returncode != 0:
        print(f"⚠️ {log_prefix} 执行失败。")
        if SILENT_TRAINING:  # 仅在静默模式下失败时打印错误
            print("--- 错误日志开始 ---")
            print(stderr_output)
            print("--- 错误日志结束 ---")
        return False, stdout_lines, stderr_output

    print(f"--- [CMD] {log_prefix} 执行完毕 ---")
    return True, stdout_lines, stderr_output


def parse_evaluation_log(stdout_lines, exp_name, exp_num):
    """
    从 train.py 的 stdout 中提取评估日志块。
    """
    eval_block = []
    capturing = False

    # 匹配评估块的开始
    start_marker = re.compile(r"===\s*目标域验证集指标（物理单位）\s*===")

    for line in stdout_lines:
        if start_marker.search(line):
            capturing = True
            eval_block.append(
                f"=== 实验 {exp_num} ({exp_name})：目标域验证集指标（物理单位）===")
            continue

        if capturing and line.strip():  # 捕获所有非空行
            eval_block.append(line)

        if capturing and not line.strip():  # 遇到空行停止
            capturing = False
            break  # 评估块结束

    return "\n".join(eval_block)


# ==============================================================================
# --- 5. 实验执行与结果捕获 ---
# ==============================================================================
start_time = time.time()
print(f"--- 实验开始：共 {len(EXPERIMENT_GRID)} 次运行 ---")
print(f"--- 评估日志将保存到: {EVALUATION_LOG_FILE} ---")

print("正在预加载数据...")
data = get_data_and_scalers(opamp_type=OPAMP_TYPE)
input_dim = data['source'][0].shape[1]
output_dim = data['source'][1].shape[1]
print("数据加载完成。")

for i, params in enumerate(EXPERIMENT_GRID):
    exp_name = f"{i+1:02d}_{params['name']}"
    print(f"\n{'='*80}")
    print(f"🚀 开始实验 {i+1}/{len(EXPERIMENT_GRID)}: {exp_name}")

    exp_results_path = BASE_RESULTS_DIR / exp_name
    exp_results_path.mkdir(parents=True, exist_ok=True)

    # 定义模型参数和路径
    model_params = {
        'input_dim': input_dim, 'output_dim': output_dim,
        'hidden_dims': params['hidden_dims'], 'dropout_rate': params['dropout_rate']
    }
    pretrained_model_path = exp_results_path / f"{OPAMP_TYPE}_pretrained.pth"
    final_model_path = exp_results_path / f"{OPAMP_TYPE}_finetuned.pth"
    final_results_file = exp_results_path / "final_metrics.json"

    # --------------------------------------------------------------------------
    # --- 步骤 A: 寻找最优预训练学习率 ---
    # --------------------------------------------------------------------------
    print("\n--- 步骤 A: 正在寻找最优预训练学习率... ---")
    optimal_lr_pretrain = find_pretrain_lr(
        AlignHeteroMLP, model_params, data,
        save_plot_path=str(exp_results_path / "lr_finder_pretrain.png")
    )
    print(f"   - 找到的最优预训练学习率 (lr_pretrain): {optimal_lr_pretrain:.2e}")

    # --------------------------------------------------------------------------
    # --- 步骤 B: 运行 Pretrain-Only ---
    # --------------------------------------------------------------------------
    print("\n--- 步骤 B: 正在执行 Pretrain-Only... ---")
    pretrain_command = [
        "python", "train.py", "--opamp", OPAMP_TYPE,
        "--hidden_dims", str(params['hidden_dims']),
        "--dropout_rate", str(params['dropout_rate']),
        "--lr_pretrain", str(optimal_lr_pretrain),
        "--save_path", str(exp_results_path),
        "--restart",  # 确保重新运行预训练
        "--pretrain"  # <-- 关键：只运行预训练
    ]
    success, _, _ = run_command(pretrain_command, f"{exp_name}_Pretrain")

    if not success or not pretrained_model_path.exists():
        print(f"❌ 实验 {exp_name} 在预训练阶段失败。跳过此实验。")
        continue
    print(f"   - 预训练模型已保存至: {pretrained_model_path.name}")

    # --------------------------------------------------------------------------
    # --- 步骤 C: 寻找最优微调学习率 ---
    # --------------------------------------------------------------------------
    print("\n--- 步骤 C: 正在寻找最优微调学习率... ---")
    current_ratio = params['backbone_lr_ratio']
    optimal_lr_finetune = find_finetune_lr(
        AlignHeteroMLP, model_params, data,
        pretrained_weights_path=str(pretrained_model_path),
        backbone_lr_ratio=current_ratio,  # <-- 传入当前实验的 ratio
        save_plot_path=str(exp_results_path / "lr_finder_finetune.png")
    )
    print(f"   - 找到的最优微调学习率 (lr_finetune_head): {optimal_lr_finetune:.2e}")

    # --------------------------------------------------------------------------
    # --- 步骤 D: 运行 Finetune + Evaluate ---
    # --------------------------------------------------------------------------
    print(
        f"\n--- 步骤 D: 正在执行 Finetune + Evaluate (Ratio={current_ratio})... ---")
    finetune_command = [
        "python", "train.py", "--opamp", OPAMP_TYPE,
        "--hidden_dims", str(params['hidden_dims']),
        "--dropout_rate", str(params['dropout_rate']),

        # 传入自动找到的微调LR 和 网格中的Ratio
        "--lr_finetune", str(optimal_lr_finetune),
        "--backbone_lr_ratio", str(current_ratio),

        "--save_path", str(exp_results_path),
        "--results_file", str(final_results_file),

        "--finetune",   # <-- 关键：跳过预训练，只微调
        "--evaluate"    # <-- 关键：微调后立即评估
    ]

    success, stdout_lines, _ = run_command(
        finetune_command, f"{exp_name}_FinetuneEval")

    if not success:
        print(f"❌ 实验 {exp_name} 在微调/评估阶段失败。跳过此实验。")
        continue

    # --------------------------------------------------------------------------
    # --- 步骤 E: 提取日志并保存 ---
    # --------------------------------------------------------------------------
    print(f"\n--- 步骤 E: 提取日志并保存... ---")
    evaluation_text = parse_evaluation_log(stdout_lines, exp_name, i+1)
    if evaluation_text:
        file_logger.info(evaluation_text + "\n")
        print(f"✅ 评估日志已保存到 {EVALUATION_LOG_FILE.name}")
    else:
        print(f"⚠️ 警告: 未能从 {exp_name} 的训练输出中捕获到评估日志。")

    # --------------------------------------------------------------------------
    # --- 步骤 F: 生成提交文件 ---
    # --------------------------------------------------------------------------
    print(f"\n--- 步骤 F: 正在为实验 {i+1} 生成提交文件... ---")
    submission_path = BASE_RESULTS_DIR / f"{SUBMISSION_FILE_PREFIX}_{i+1}"

    if not final_model_path.exists():
        print(f"❌ 实验 {exp_name} 未能生成 {final_model_path.name}。无法提交。")
        continue

    submit_cmd = [
        "python", "submit.py",
        "--opamp", OPAMP_TYPE,
        "--model-path", str(final_model_path),
        "--output-file", str(submission_path),
        "--test-file", str(TEST_FILE_PATH),
        "--hidden-dims", str(model_params['hidden_dims']),
        "--dropout-rate", str(model_params['dropout_rate']),
        "--device", config.DEVICE
    ]
    success, _, _ = run_command(submit_cmd, f"{exp_name}_Submit")
    if success:
        print(f"✅ 成功生成提交文件: {submission_path.name}")
    else:
        print(f"❌ 生成提交文件失败: {submission_path.name}")

    # --------------------------------------------------------------------------
    # --- 步骤 G: 清理 (如果启用) ---
    # --------------------------------------------------------------------------
    if CLEANUP_AFTER_RUN:
        try:
            shutil.rmtree(exp_results_path)
            print(f"清理完毕: 已删除临时文件夹 {exp_results_path}")
        except Exception as e:
            print(f"⚠️ 清理失败: 删除文件夹 {exp_results_path} 时出错 - {e}")
# ==============================================================================
# --- 5. 汇总并展示最终结果 ---
# ==============================================================================
end_time = time.time()
total_duration = end_time - start_time
final_message = f"\n\n{'='*80}\n🎉 所有实验已完成！总耗时: {total_duration / 60:.2f} 分钟\n{'='*80}\n"
final_message += f"评估日志已全部保存到: {EVALUATION_LOG_FILE}\n"
final_message += f"提交文件已生成在: {BASE_RESULTS_DIR}\n"
print(final_message)
file_logger.info(final_message)  # 也在日志文件末尾写入总结
