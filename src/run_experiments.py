# src/run_experiments.py (已更新：支持静默训练)
import subprocess
import os
import pandas as pd
import time
import json
import shutil
from pathlib import Path
from find_lr_utils import find_pretrain_lr
from models.align_hetero import AlignHeteroMLP
from data_loader import get_data_and_scalers

# ==============================================================================
# --- 0. 路径和实验控制 ---
# ==============================================================================
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

CLEANUP_AFTER_RUN = True
SILENT_TRAINING = True  # <-- 新增开关: 设置为 True 来禁用详细的训练日志

# ==============================================================================
# --- 1. 定义你的实验搜索空间 ---
# ==============================================================================
BASE_EXPERIMENT_GRID = [
    {"name": "128, 256, 512]", "hidden_dims": [
        128, 256, 512], "dropout_rate": 0.2},
    {"name": "128, 256, 512, 256]0.2", "hidden_dims": [
        128, 256, 512, 256], "dropout_rate": 0.2},
    {"name": "128, 256, 512, 256]0.3", "hidden_dims": [
        128, 256, 512, 256], "dropout_rate": 0.3},
    {"name": "128, 256, 512, 512]0.2", "hidden_dims": [
        128, 256, 512, 512], "dropout_rate": 0.2},
    {"name": "128, 256, 512, 512]0.3", "hidden_dims": [
        128, 256, 512, 512], "dropout_rate": 0.3},
    {"name": "128, 256, 512, 768]", "hidden_dims": [
        128, 256, 512, 768], "dropout_rate": 0.3},
    {"name": "128, 256, 512, 256, 128]", "hidden_dims": [
        128, 256, 512, 256, 128], "dropout_rate": 0.35},
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

# --- 预先加载一次数据 ---
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
        "--lr_pretrain", str(optimal_lr_pretrain),
        "--lr_finetune", str(FIXED_LR_FINETUNE),
        "--save_path", str(exp_results_path),
        "--restart", "--evaluate",
        "--results_file", str(final_results_file)
    ]

    # <<< --- 核心修改：控制训练日志输出 --- >>>
    if SILENT_TRAINING:
        print(f"正在静默执行训练... (详细日志已禁用)")
        process = subprocess.Popen(
            command, cwd=SRC_DIR, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding='utf-8'  # 捕获 stdout 和 stderr
        )
        # 等待进程结束并捕获所有输出，但不打印
        stdout_output, stderr_output = process.communicate()

        # 仅在发生错误时打印错误信息
        if process.returncode != 0:
            print(f"⚠️ 实验 {exp_name} 训练失败。以下是错误日志：")
            print(stderr_output)
    else:
        # 保持原来的详细日志行为
        print(f"正在执行训练... (详细日志已启用)")
        process = subprocess.Popen(
            command, cwd=SRC_DIR, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding='utf-8'
        )
        for line in iter(process.stdout.readline, ''):
            print(line.strip())
        process.wait()
    # --- 修改结束 ---

    # --- 读取结果文件 ---
    if final_results_file.exists():
        with open(final_results_file, 'r', encoding='utf-8') as f:
            # 读取文件内容，但注意我们是追加模式，可能包含多个JSON对象
            # 我们只取最后一个
            all_results = []
            file_content = f.read().strip()
            if file_content:
                json_objects = file_content.split('\n')
                for i, obj_str in enumerate(json_objects):
                    if not obj_str.strip():
                        continue
                    try:
                        all_results.append(json.loads(obj_str))
                    except json.JSONDecodeError as e:
                        print(
                            f"警告: 解析 final_metrics.json 的第 {i+1} 行失败 (内容: '{obj_str[:50]}...'): {e}")

            if not all_results:
                print(
                    f"⚠️ 实验 {exp_name} 完成，但 {final_results_file.name} 为空或无效。")
                final_nll = float('NaN')
                avg_mse = float('NaN')
            else:
                final_metrics = all_results[-1]  # 只取最后一个JSON对象
                final_nll = final_metrics.get('best_finetune_val_nll')
                avg_mse = final_metrics.get(
                    'evaluation_metrics', {}).get('avg_mse')

        print(
            f"✅ 实验 {exp_name} 完成。 最终 Val NLL: {final_nll:.6f}, Avg MSE: {avg_mse:.4g}")
        RESULTS.append({
            '完整实验名称': exp_name, '基础模型': params['base_name'],
            'hidden_dims': str(params['hidden_dims']), 'dropout_rate': params['dropout_rate'],
            'final_val_nll': final_nll, 'avg_mse': avg_mse
        })

        if CLEANUP_AFTER_RUN:
            try:
                shutil.rmtree(exp_results_path)
                print(f"清理完毕: 已删除临时文件夹 {exp_results_path}")
            except Exception as e:
                print(f"⚠️ 清理失败: 删除文件夹 {exp_results_path} 时出错 - {e}")
    else:
        print(f"⚠️ 实验 {exp_name} 完成，但未找到结果文件: {final_results_file}")
        RESULTS.append({
            '完整实验名称': exp_name, '基础模型': params['base_name'],
            'hidden_dims': str(params['hidden_dims']), 'dropout_rate': params['dropout_rate'],
            'final_val_nll': float('NaN'), 'avg_mse': float('NaN')
        })

# ==============================================================================
# --- 4. 汇总并展示最终结果 ---
# ==============================================================================
# ... (这部分代码无需修改) ...
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
