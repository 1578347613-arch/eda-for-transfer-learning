# src/run_experiments.py (最终形态 - 全自动流水线)
import subprocess
import os
import pandas as pd
import time
import json
import shutil
import logging
import sys
from pathlib import Path

# --- 从项目模块中导入 ---
from find_lr_utils import find_pretrain_lr, find_finetune_lr
from models.align_hetero import AlignHeteroMLP
from data_loader import get_data_and_scalers
import config

# ==============================================================================
# --- 0. 路径和实验控制 ---
# ==============================================================================
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# ==============================================================================
# --- 1. 定义你的实验搜索空间 ---
# ==============================================================================
# hidden_dims (必需): 模型结构
# dropout_rate (可选): 覆盖默认的 config.DROPOUT_RATE
# lr_finetune (可选): 覆盖自动查找的 lr_finetune (设为 "auto" 或省略则自动查找)
BASE_EXPERIMENT_GRID = [
    {
        "name": "1",
        "hidden_dims": [256, 256, 256, 256],
        "dropout_rate": 0.1
    },
    {
        "name": "2",
        "hidden_dims": [256, 256, 256, 256],
        "dropout_rate": 0.15
    },
    {
        "name": "3",
        "hidden_dims": [256, 256, 256, 256],
        "dropout_rate": 0.2
    },
    {
        "name": "4",
        "hidden_dims": [256, 256, 256, 256],
        "dropout_rate": 0.25
    },
    {
        "name": "5",
        "hidden_dims": [256, 256, 256, 256],
        "dropout_rate": 0.3
    },
    {
        "name": "6",
        "hidden_dims": [256, 256, 256, 256],
        "dropout_rate": 0.35
    },
    {
        "name": "7",
        "hidden_dims": [256, 256, 256, 256],
        "dropout_rate": 0.4
    },
    {
        "name": "8",
        "hidden_dims": [256, 256, 256, 256],
        "dropout_rate": 0.45
    },
    {
        "name": "1",
        "hidden_dims": [128, 256, 512],
        "dropout_rate": 0.05
    },
    {
        "name": "2",
        "hidden_dims": [128, 256, 512],
        "dropout_rate": 0.1
    },
    {
        "name": "3",
        "hidden_dims": [128, 256, 512],
        "dropout_rate": 0.15
    },
    {
        "name": "4",
        "hidden_dims": [128, 256, 512],
        "dropout_rate": 0.2
    },
    {
        "name": "5",
        "hidden_dims": [128, 256, 512],
        "dropout_rate": 0.25
    },
    {
        "name": "6",
        "hidden_dims": [128, 256, 512],
        "dropout_rate": 0.3
    },
    {
        "name": "7",
        "hidden_dims": [128, 256, 512],
        "dropout_rate": 0.35
    },
]

# --- 实验控制设置 ---
NUM_REPETITIONS = 1  # 建议先设为1，跑通后再改为3
OPAMP_TYPE = '5t_opamp'
BASE_RESULTS_DIR = PROJECT_ROOT / "results_experiments_full_auto"

# --- 提交文件设置 ---
TEST_FILE_PATH = PROJECT_ROOT / "data/02_public_test_set/features/features_A.csv"
SUBMISSION_FILE_PREFIX = "predA"  # 将生成 predA_1, predA_2 ...

# ==============================================================================
# --- 2. 设置日志系统 ---
# ==============================================================================
BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
log_file_path = BASE_RESULTS_DIR / "experiment_log.txt"

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # 只记录原始消息
    handlers=[
        logging.FileHandler(log_file_path, mode='w', encoding='utf-8'),  # 写入文件
        logging.StreamHandler(sys.stdout)  # 同时也打印到控制台
    ]
)
logger = logging.getLogger()

# ==============================================================================
# --- 3. 动态生成完整的实验列表 ---
# ==============================================================================
EXPERIMENT_GRID = []
for exp_params in BASE_EXPERIMENT_GRID:
    for run_num in range(1, NUM_REPETITIONS + 1):
        new_params = exp_params.copy()
        new_params['name'] = f"{exp_params['name']}_run{run_num}"
        new_params['base_name'] = exp_params['name']
        EXPERIMENT_GRID.append(new_params)

# ==============================================================================
# --- 4. 实验执行与结果捕获 ---
# ==============================================================================
RESULTS_DF = []
start_time = time.time()

logger.info(f"--- 实验开始：共 {len(EXPERIMENT_GRID)} 次运行 ---")
logger.info(f"--- 结果将保存在: {BASE_RESULTS_DIR} ---")

# --- 预先加载一次数据 ---
logger.info("正在预加载数据...")
data = get_data_and_scalers(opamp_type=OPAMP_TYPE)
input_dim = data['source'][0].shape[1]
output_dim = data['source'][1].shape[1]
logger.info("数据加载完成。")


def run_command(command, log_prefix=""):
    """辅助函数：执行子进程并实时打印输出"""
    logger.info(f"--- [CMD] {log_prefix} 正在执行: {' '.join(command)} ---")
    process = subprocess.Popen(
        command, cwd=SRC_DIR, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding='utf-8'
    )
    output_lines = []
    for line in iter(process.stdout.readline, ''):
        line = line.strip()
        if line:
            logger.info(f"[{log_prefix}] {line}")
            output_lines.append(line)
    process.wait()
    logger.info(f"--- [CMD] {log_prefix} 执行完毕 ---")
    return process.returncode == 0, output_lines


for i, params in enumerate(EXPERIMENT_GRID):
    exp_name = f"{i+1:02d}_{params['name']}"
    logger.info(f"\n{'='*80}")
    logger.info(f"🚀 开始实验 {i+1}/{len(EXPERIMENT_GRID)}: {exp_name}")
    logger.info(f"{'='*80}")

    exp_results_path = BASE_RESULTS_DIR / exp_name
    exp_results_path.mkdir(parents=True, exist_ok=True)

    # --- 准备模型参数 ---
    model_params = {
        'input_dim': input_dim, 'output_dim': output_dim,
        'hidden_dims': params['hidden_dims'],
        # 如果未在GRID中指定，则从config加载默认值
        'dropout_rate': params.get('dropout_rate', config.DROPOUT_RATE)
    }

    # --- 步骤 A: 寻找 lr_pretrain ---
    logger.info("--- 步骤 A: 正在寻找 lr_pretrain... ---")
    lr_plot_path_pre = exp_results_path / f"lr_finder_pretrain_{i+1}.png"
    optimal_lr_pretrain = find_pretrain_lr(
        AlignHeteroMLP, model_params, data,
        num_iter=1000, save_plot_path=str(lr_plot_path_pre)
    )
    logger.info(f"   -> 找到的最优 lr_pretrain: {optimal_lr_pretrain:.2e}")

    # --- 步骤 B: 运行 Pretrain-Only 来生成匹配的 .pth ---
    logger.info("--- 步骤 B: 正在生成匹配的预训练模型... ---")
    temp_pretrained_path = exp_results_path / f"{OPAMP_TYPE}_pretrained.pth"
    pretrain_cmd = [
        "python", "train.py", "--opamp", OPAMP_TYPE,
        "--hidden_dims", str(model_params['hidden_dims']),
        "--dropout_rate", str(model_params['dropout_rate']),
        "--lr_pretrain", str(optimal_lr_pretrain),
        "--save_path", str(exp_results_path),
        "--pretrain-only"  # <-- 使用新标志
    ]
    success, _ = run_command(pretrain_cmd, f"{exp_name}_Pretrain")
    if not success:
        logger.error(f"❌ 实验 {exp_name} 在步骤B（预训练）失败。跳过此实验。")
        continue

    # --- 步骤 C: 寻找 lr_finetune ---
    logger.info("--- 步骤 C: 正在寻找 lr_finetune... ---")
    if "lr_finetune" in params and params["lr_finetune"] != "auto":
        optimal_lr_finetune = params["lr_finetune"]
        logger.info(f"   -> 使用了实验网格中指定的 lr_finetune: {optimal_lr_finetune}")
    else:
        lr_plot_path_fine = exp_results_path / f"lr_finder_finetune_{i+1}.png"
        optimal_lr_finetune = find_finetune_lr(
            AlignHeteroMLP, model_params, data,
            pretrained_weights_path=str(temp_pretrained_path),
            num_iter=1000, save_plot_path=str(lr_plot_path_fine)
        )
        logger.info(
            f"   -> 找到的最优 lr_finetune (MinLoss/2): {optimal_lr_finetune:.2e}")

    # --- 步骤 D: 运行完整的训练和评估 ---
    logger.info("--- 步骤 D: 正在运行完整训练和评估... ---")
    final_metrics_file = exp_results_path / "final_metrics.json"
    full_train_cmd = [
        "python", "train.py", "--opamp", OPAMP_TYPE,
        "--hidden_dims", str(model_params['hidden_dims']),
        "--dropout_rate", str(model_params['dropout_rate']),
        "--lr_pretrain", str(optimal_lr_pretrain),
        "--lr_finetune", str(optimal_lr_finetune),
        "--save_path", str(exp_results_path),
        "--restart", "--evaluate",  # <-- 强制重新训练并评估
        "--results_file", str(final_metrics_file)
    ]

    # --- 关键：捕获评估日志 ---
    success, output_lines = run_command(
        full_train_cmd, f"{exp_name}_FullTrain")
    if not success:
        logger.error(f"❌ 实验 {exp_name} 在步骤D（完整训练）失败。跳过此实验。")
        continue

    # 将评估结果写入主日志
    eval_log_started = False
    logger.info(f"\n=== 实验 {i+1} ({exp_name})：目标域验证集指标（物理单位）===")
    for line in output_lines:
        if "=== 目标域验证集指标（物理单位） ===" in line:
            eval_log_started = True
            continue
        if eval_log_started and line.strip():
            logger.info(line)
        if eval_log_started and not line.strip():  # 遇到空行停止
            eval_log_started = False
    logger.info("========================================\n")

    # --- 步骤 E: 生成提交文件 ---
    logger.info(f"--- 步骤 E: 正在为实验 {i+1} 生成提交文件... ---")
    final_model_path = exp_results_path / f"{OPAMP_TYPE}_finetuned.pth"
    submission_path = BASE_RESULTS_DIR / f"{SUBMISSION_FILE_PREFIX}_{i+1}"

    if not final_model_path.exists():
        logger.error(f"❌ 实验 {exp_name} 未能生成 {final_model_path.name}。无法提交。")
        continue

    submit_cmd = [
        "python", "submit.py",
        "--opamp", OPAMP_TYPE,
        "--model-path", str(final_model_path),
        "--output-file", str(submission_path),
        "--test-file", str(TEST_FILE_PATH),
        "--hidden-dims", str(model_params['hidden_dims']),
        "--dropout-rate", str(model_params['dropout_rate'])
    ]
    success, _ = run_command(submit_cmd, f"{exp_name}_Submit")
    if success:
        logger.info(f"✅ 成功生成提交文件: {submission_path.name}")

    # --- 步骤 F: 记录最终结果 (从 JSON 文件) ---
    if final_metrics_file.exists():
        with open(final_metrics_file, 'r', encoding='utf-8') as f:
            # 读取文件内容，但注意我们是追加模式，可能包含多个JSON对象
            # 我们只取最后一个
            all_results = [json.loads(obj)
                           for obj in f.read().strip().split('\n') if obj]
            final_metrics = all_results[-1]

        final_nll = final_metrics.get('best_finetune_val_nll')
        avg_mse = final_metrics.get('evaluation_metrics', {}).get('avg_mse')

        RESULTS_DF.append({
            '实验名称': exp_name, '基础模型': params['base_name'],
            'hidden_dims': str(params['hidden_dims']),
            'dropout_rate': model_params['dropout_rate'],
            'lr_pretrain': f"{optimal_lr_pretrain:.2e}",
            'lr_finetune': f"{optimal_lr_finetune:.2e}",
            'final_val_nll': final_nll, 'avg_mse': avg_mse
        })
    else:
        logger.error(f"❌ 实验 {exp_name} 未能生成 {final_metrics_file.name}。")

# ==============================================================================
# --- 5. 汇总并展示最终结果 ---
# ==============================================================================
end_time = time.time()
total_duration = end_time - start_time
logger.info(f"\n\n{'='*80}")
logger.info(f"🎉 所有实验已完成！总耗时: {total_duration / 60:.2f} 分钟")
logger.info(f"主日志文件: {log_file_path}")
logger.info("="*80)

if RESULTS_DF:
    results_df = pd.DataFrame(RESULTS_DF)
    logger.info("\n📊 所有运行的详细结果 (从优到劣排序):")
    detailed_results = results_df.sort_values(
        by='final_val_nll', ascending=True)
    logger.info(detailed_results.to_string(index=False))
    summary_path = BASE_RESULTS_DIR / "experiment_summary_detailed.csv"
    detailed_results.to_csv(summary_path, index=False, encoding='utf-8-sig')
    logger.info(f"\n📄 详细结果已保存至: {summary_path}")

    logger.info("\n\n" + "="*80)
    logger.info("📈 按基础模型聚合的统计结果:")
    aggregated_df = results_df.groupby('基础模型')['final_val_nll'].agg(
        ['mean', 'std', 'min', 'max', 'count']).sort_values(by='mean', ascending=True)
    aggregated_df.rename(columns={'mean': '平均NLL', 'std': 'NLL标准差',
                         'min': '最佳NLL', 'max': '最差NLL', 'count': '运行次数'}, inplace=True)
    logger.info(aggregated_df)
    agg_summary_path = BASE_RESULTS_DIR / "experiment_summary_aggregated.csv"
    aggregated_df.to_csv(agg_summary_path, encoding='utf-8-sig')
    logger.info(f"\n📄 聚合统计结果已保存至: {agg_summary_path}")

    best_model_name = aggregated_df.index[0]
    best_model_stats = aggregated_df.iloc[0]
    logger.info("\n\n🏆 综合表现最佳的模型结构 (基于平均NLL):")
    logger.info(f"   - 名称: {best_model_name}")
    logger.info(f"   - 平均验证集NLL: {best_model_stats['平均NLL']:.6f}")
    logger.info(f"   - 稳定性 (标准差): {best_model_stats['NLL标准差']:.6f}")
