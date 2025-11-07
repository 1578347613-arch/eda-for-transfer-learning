# src/run_experiments.py (已简化：跳过预训练和LR查找)
import subprocess
import os
import time
import json
import shutil
from pathlib import Path
import logging
import sys
import re

# --- 从项目模块中导入 ---
# (不再需要 find_lr_utils)
from data_loader import get_data_and_scalers
import config  # <-- 导入 config 以获取默认值和路径

# ==============================================================================
# --- 0. 路径和实验控制 ---
# ==============================================================================
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# <<< --- 核心修改：指向您已有的预训练模型 --- >>>
# (确保 config.OPAMP_TYPE 正确)
EXISTING_PRETRAIN_FILE = PROJECT_ROOT / "results" / \
    f"{config.OPAMP_TYPE}_pretrained.pth"

CLEANUP_AFTER_RUN = False
SILENT_TRAINING = False

# ==============================================================================
# --- 1. 定义你的实验搜索空间 ---
# ==============================================================================
# (此网格现在是您唯一要调整的)
BASE_EXPERIMENT_GRID = [
    {"name": "HBase_Scale_0.5", "backbone_lr_scale": 0.2},
    {"name": "HBase_Scale_0.1", "backbone_lr_scale": 0.1},
    {"name": "HBase_Scale_0.05", "backbone_lr_scale": 0.05},
    {"name": "HBase_Scale_0.01", "backbone_lr_scale": 0.01},
]

# --- 实验控制设置 ---
NUM_REPETITIONS = 3
OPAMP_TYPE = config.OPAMP_TYPE  # (从 config 加载)
BASE_RESULTS_DIR = PROJECT_ROOT / "results_experiments_finetune_only"

# --- 提交文件设置 ---
TEST_FILE_PATH = PROJECT_ROOT / "data/02_public_test_set/features/features_A.csv"
SUBMISSION_FILE_PREFIX = "predA"

# ==============================================================================
# --- 2. 设置日志系统 (保持不变) ---
# ==============================================================================
BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_LOG_FILE = BASE_RESULTS_DIR / "experiment_evaluation_log.txt"
file_logger = logging.getLogger('ExperimentLogger')
file_logger.setLevel(logging.INFO)
file_logger.propagate = False
if file_logger.hasHandlers():
    file_logger.handlers.clear()
file_handler = logging.FileHandler(
    EVALUATION_LOG_FILE, mode='w', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(message)s'))
file_logger.addHandler(file_handler)

# ==============================================================================
# --- 3. 动态生成完整的实验列表 (保持不变) ---
# ==============================================================================
EXPERIMENT_GRID = []
for exp_params in BASE_EXPERIMENT_GRID:
    for run_num in range(1, NUM_REPETITIONS + 1):
        new_params = exp_params.copy()
        new_params['name'] = f"{exp_params['name']}_run{run_num}"
        new_params['base_name'] = exp_params['name']
        EXPERIMENT_GRID.append(new_params)

# ==============================================================================
# --- 4. 辅助函数 (保持不变) ---
# ==============================================================================


def run_command(command, log_prefix=""):
    """
    辅助函数：执行子进程并捕获所有输出。
    """
    print(f"--- [CMD] {log_prefix} 正在执行: {' '.join(command)} ---")
    process = subprocess.Popen(
        command, cwd=SRC_DIR, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, encoding='utf-8'
    )
    stdout_lines = []
    if not SILENT_TRAINING:
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if line:
                print(f"[{log_prefix}] {line}")
            stdout_lines.append(line)
        process.wait()
        stderr_output = process.stderr.read()
    else:
        stdout_data, stderr_output = process.communicate()
        stdout_lines = stdout_data.splitlines()
    if process.returncode != 0:
        print(f"⚠️ {log_prefix} 执行失败。")
        if SILENT_TRAINING:
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
    start_marker = re.compile(r"===\s*目标域验证集指标（物理单位）\s*===")
    for line in stdout_lines:
        if start_marker.search(line):
            capturing = True
            eval_block.append(
                f"=== 实验 {exp_num} ({exp_name})：目标域验证集指标（物理单位）===")
            continue
        if capturing and line.strip():
            eval_block.append(line)
        if capturing and not line.strip():
            capturing = False
            break
    return "\n".join(eval_block)


# ==============================================================================
# --- 5. 实验执行与结果捕获 (已简化) ---
# ==============================================================================
start_time = time.time()
print(f"--- 实验开始：共 {len(EXPERIMENT_GRID)} 次运行 ---")
print(f"--- 评估日志将保存到: {EVALUATION_LOG_FILE} ---")
print(f"--- 将使用固定的预训练模型: {EXISTING_PRETRAIN_FILE.name} ---")
print(f"--- 将使用固定的基础学习率: {config.LEARNING_RATE_HETERO:.2e} ---")

for i, params in enumerate(EXPERIMENT_GRID):
    exp_name = f"{i+1:02d}_{params['name']}"
    print(f"\n{'='*80}")
    print(f"🚀 开始实验 {i+1}/{len(EXPERIMENT_GRID)}: {exp_name}")

    exp_results_path = BASE_RESULTS_DIR / exp_name
    exp_results_path.mkdir(parents=True, exist_ok=True)

    # 定义模型路径
    pretrained_model_path = exp_results_path / f"{OPAMP_TYPE}_pretrained.pth"
    final_model_path = exp_results_path / f"{OPAMP_TYPE}_finetuned.pth"
    final_results_file = exp_results_path / "final_metrics.json"

    # --------------------------------------------------------------------------
    # --- 步骤 A & B: (已禁用) 复制现有的 .pth 文件 ---
    # --------------------------------------------------------------------------
    print(f"\n--- 步骤 AB: 正在复制预训练模型... ---")
    if not EXISTING_PRETRAIN_FILE.exists():
        print(f"❌ 错误: 未找到您指定的预训练文件: {EXISTING_PRETRAIN_FILE}")
        continue
    try:
        shutil.copy(EXISTING_PRETRAIN_FILE, pretrained_model_path)
        print(
            f"   - 成功复制 {EXISTING_PRETRAIN_FILE.name} 到 {exp_results_path.name}")
    except Exception as e:
        print(f"❌ 复制文件失败: {e}")
        continue

    # --------------------------------------------------------------------------
    # --- 步骤 C: (已禁用) 寻找最优微调学习率 ---
    # --------------------------------------------------------------------------
    # (已跳过)

    # --------------------------------------------------------------------------
    # --- 步骤 D: 运行 Finetune + Evaluate (使用固定 LR) ---
    # --------------------------------------------------------------------------

    # <<< --- 核心修改：从 config 和 grid 读取 --- >>>
    current_lr_hetero = config.LEARNING_RATE_HETERO
    current_hidden_dims = config.HIDDEN_DIMS
    current_dropout_rate = config.DROPOUT_RATE
    current_backbone_scale = params['backbone_lr_scale']

    print(
        f"\n--- 步骤 D: 正在执行 Finetune + Evaluate (LR={current_lr_hetero:.2e}, Scale={current_backbone_scale})... ---")

    finetune_command = [
        "python", "train.py", "--opamp", OPAMP_TYPE,
        "--hidden_dims", str(current_hidden_dims),
        "--dropout_rate", str(current_dropout_rate),

        # <<< --- 使用固定的 LR 和 grid-searched scale --- >>>
        "--lr_hetero", str(current_lr_hetero),
        "--backbone_lr_scale", str(current_backbone_scale),

        "--save_path", str(exp_results_path),
        "--results_file", str(final_results_file),

        "--finetune",  # 强制重新微调
        "--evaluate"
    ]

    success, stdout_lines, _ = run_command(
        finetune_command, f"{exp_name}_FinetuneEval")

    if not success:
        print(f"❌ 实验 {exp_name} 在微调/评估阶段失败。跳过此实验。")
        continue

    # --------------------------------------------------------------------------
    # --- 步骤 E: 提取日志并保存 (保持不变) ---
    # --------------------------------------------------------------------------
    print(f"\n--- 步骤 E: 提取日志并保存... ---")
    evaluation_text = parse_evaluation_log(stdout_lines, exp_name, i+1)
    if evaluation_text:
        file_logger.info(evaluation_text + "\n")
        print(f"✅ 评估日志已保存到 {EVALUATION_LOG_FILE.name}")
    else:
        print(f"⚠️ 警告: 未能从 {exp_name} 的训练输出中捕获到评估日志。")

    # --------------------------------------------------------------------------
    # --- 步骤 F: 生成提交文件 (保持不变) ---
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
        "--hidden-dims", str(params['hidden_dims']),
        "--dropout-rate", str(params['dropout_rate']),
        "--device", config.DEVICE
    ]
    success, _, _ = run_command(submit_cmd, f"{exp_name}_Submit")
    if success:
        print(f"✅ 成功生成提交文件: {submission_path.name}")
    else:
        print(f"❌ 生成提交文件失败: {submission_path.name}")

    # --------------------------------------------------------------------------
    # --- 步骤 G: 清理 (保持不变) ---
    # --------------------------------------------------------------------------
    if CLEANUP_AFTER_RUN:
        try:
            shutil.rmtree(exp_results_path)
            print(f"清理完毕: 已删除临时文件夹 {exp_results_path}")
        except Exception as e:
            print(f"⚠️ 清理失败: 删除文件夹 {exp_results_path} 时出错 - {e}")

# ==============================================================================
# --- 5. 汇总并展示最终结果 (保持不变) ---
# ==============================================================================
end_time = time.time()
total_duration = end_time - start_time
final_message = f"\n\n{'='*80}\n🎉 所有实验已完成！总耗时: {total_duration / 60:.2f} 分钟\n{'='*80}\n"
final_message += f"评估日志已全部保存到: {EVALUATION_LOG_FILE}\n"
final_message += f"提交文件已生成在: {BASE_RESULTS_DIR}\n"
print(final_message)
file_logger.info(final_message)
