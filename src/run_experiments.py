# src/run_experiments.py (最终版 - 自动清理)
import subprocess
import os
import pandas as pd
import time
import json
import shutil  # <-- 导入 shutil 库用于删除文件夹
from pathlib import Path
from find_lr_utils import find_pretrain_lr
from models.align_hetero import AlignHeteroMLP  # 需要它来传递类
from data_loader import get_data_and_scalers  # 需要它来加载数据

# ==============================================================================
# --- 0. 路径和实验控制 ---
# ==============================================================================
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# <<< --- 核心改动：添加一个清理开关 --- >>>
# 设置为 True: 实验成功后自动删除模型和临时文件。
# 设置为 False: 保留所有文件。
CLEANUP_AFTER_RUN = True

# ==============================================================================
# --- 1. 定义你的实验搜索空间 ---
# ==============================================================================
BASE_EXPERIMENT_GRID = [
    {"name": "256, 128, 256]", "hidden_dims": [
        256, 128, 256], "dropout_rate": 0.2},
    {"name": "128, 256, 512]", "hidden_dims": [
        128, 256, 512], "dropout_rate": 0.2},
    {"name": "128, 256, 256]", "hidden_dims": [
        128, 256, 256], "dropout_rate": 0.2},
    {"name": "128, 256, 768]", "hidden_dims": [
        128, 256, 768], "dropout_rate": 0.2},
    {"name": "128, 128, 256]", "hidden_dims": [
        128, 128, 256], "dropout_rate": 0.2},
    {"name": "128, 128, 128]", "hidden_dims": [
        128, 128, 128], "dropout_rate": 0.2},
    {"name": "128, 128, 512]", "hidden_dims": [
        128, 128, 512], "dropout_rate": 0.2},
    {"name": "64, 128, 128]", "hidden_dims": [
        64, 128, 128], "dropout_rate": 0.2},
    {"name": "64, 128, 256]", "hidden_dims": [
        64, 128, 256], "dropout_rate": 0.2},
]

# --- 实验控制设置 ---
NUM_REPETITIONS = 1
OPAMP_TYPE = '5t_opamp'
BASE_RESULTS_DIR = PROJECT_ROOT / "results_experiments_fixed_lr"
FIXED_LR_FINETUNE = 1e-4

# ==============================================================================
# --- 2. 动态生成完整的实验列表 ---
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
# --- 3. 实验执行与结果捕获 ---
# ==============================================================================
RESULTS = []
start_time = time.time()
BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- 预先加载一次数据，避免在循环中重复IO ---
print("正在预加载数据...")
data = get_data_and_scalers(opamp_type=OPAMP_TYPE)
input_dim = data['source'][0].shape[1]
output_dim = data['source'][1].shape[1]
print("数据加载完成。")

for i, params in enumerate(EXPERIMENT_GRID):
    exp_name = f"{i+1:02d}_{params['name']}"
    # ... (打印实验信息的代码不变) ...
    print(f"\n{'='*80}")
    print(f"🚀 开始实验 {i+1}/{len(EXPERIMENT_GRID)}: {exp_name}")

    exp_results_path = BASE_RESULTS_DIR / exp_name
    exp_results_path.mkdir(parents=True, exist_ok=True)

    # --- 核心修改：自动寻找最优预训练学习率 ---
    print("\n--- 步骤 A: 正在为当前结构自动寻找最优预训练学习率... ---")
    model_params = {
        'input_dim': input_dim, 'output_dim': output_dim,
        'hidden_dims': params['hidden_dims'], 'dropout_rate': params['dropout_rate']
    }
    optimal_lr_pretrain = find_pretrain_lr(AlignHeteroMLP, model_params, data)
    print(f"   - 找到的最优预训练学习率 (lr_pretrain): {optimal_lr_pretrain:.2e}")

    final_results_file = exp_results_path / "final_metrics.json"

    command = [
        "python", "train.py", "--opamp", OPAMP_TYPE,
        "--hidden_dims", str(params['hidden_dims']),
        "--dropout_rate", str(params['dropout_rate']),
        "--lr_pretrain", str(optimal_lr_pretrain),  # <-- 使用自动找到的值
        "--lr_finetune", str(FIXED_LR_FINETUNE),
        "--save_path", str(exp_results_path),
        "--restart", "--evaluate",
        "--results_file", str(final_results_file)
    ]

    print(f"正在执行训练... 输出将直接打印到控制台。")
    process = subprocess.Popen(
        command, cwd=SRC_DIR, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding='utf-8'
    )

    for line in iter(process.stdout.readline, ''):
        print(line.strip())

    process.wait()

    # --- 读取结果文件 ---
    if final_results_file.exists():
        with open(final_results_file, 'r', encoding='utf-8') as f:
            final_metrics = json.load(f)

        final_nll = final_metrics.get('best_finetune_val_nll')
        avg_mse = final_metrics.get('evaluation_metrics', {}).get('avg_mse')

        print(
            f"✅ 实验 {exp_name} 完成。 最终 Val NLL: {final_nll:.6f}, Avg MSE: {avg_mse:.4g}")
        RESULTS.append({
            '完整实验名称': exp_name, '基础模型': params['base_name'],
            'hidden_dims': str(params['hidden_dims']), 'dropout_rate': params['dropout_rate'],
            'final_val_nll': final_nll, 'avg_mse': avg_mse
        })

        # <<< --- 核心改动：如果开关为True，则删除临时文件夹 --- >>>
        if CLEANUP_AFTER_RUN:
            try:
                shutil.rmtree(exp_results_path)
                print(f"清理完毕: 已删除临时文件夹 {exp_results_path}")
            except Exception as e:
                print(f"⚠️ 清理失败: 删除文件夹 {exp_results_path} 时出错 - {e}")

    else:
        # ... (处理失败情况的代码不变) ...
        print(f"⚠️ 实验 {exp_name} 完成，但未找到结果文件: {final_results_file}")
        RESULTS.append({
            '完整实验名称': exp_name, '基础模型': params['base_name'],
            'hidden_dims': str(params['hidden_dims']), 'dropout_rate': params['dropout_rate'],
            'final_val_nll': float('NaN'), 'avg_mse': float('NaN')
        })

# ==============================================================================
# --- 4. 汇总并展示最终结果 ---
# ==============================================================================
# ... (这部分代码无需任何修改) ...
end_time = time.time()
total_duration = end_time - start_time
print(f"\n\n{'='*80}\n🎉 所有实验已完成！总耗时: {total_duration / 60:.2f} 分钟\n{'='*80}")

if RESULTS:
    results_df = pd.DataFrame(RESULTS)
    print("\n📊 所有运行的详细结果 (从优到劣排序):")
    detailed_results = results_df.sort_values(
        by='final_val_nll', ascending=True)
    print(detailed_results.to_string(index=False))
    summary_path = BASE_RESULTS_DIR / "experiment_summary_detailed.csv"
    detailed_results.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"\n📄 详细结果已保存至: {summary_path}")

    print("\n\n" + "="*80)
    print("📈 按基础模型聚合的统计结果:")
    aggregated_df = results_df.groupby('基础模型')['final_val_nll'].agg(
        ['mean', 'std', 'min', 'max', 'count']).sort_values(by='mean', ascending=True)
    aggregated_df.rename(columns={'mean': '平均NLL', 'std': 'NLL标准差',
                         'min': '最佳NLL', 'max': '最差NLL', 'count': '运行次数'}, inplace=True)
    print(aggregated_df)
    agg_summary_path = BASE_RESULTS_DIR / "experiment_summary_aggregated.csv"
    aggregated_df.to_csv(agg_summary_path, encoding='utf-8-sig')
    print(f"\n📄 聚合统计结果已保存至: {agg_summary_path}")

    best_model_name = aggregated_df.index[0]
    best_model_stats = aggregated_df.iloc[0]
    print("\n\n🏆 综合表现最佳的模型结构 (基于平均NLL):")
    print(f"   - 名称: {best_model_name}")
    print(f"   - 平均验证集NLL: {best_model_stats['平均NLL']:.6f}")
    print(f"   - 稳定性 (标准差): {best_model_stats['NLL标准差']:.6f}")
