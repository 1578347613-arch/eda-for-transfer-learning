# src/run_experiments.py (简化版：使用固定的微调学习率)
import subprocess
import os
import pandas as pd
import time
from pathlib import Path

# ==============================================================================
# --- 0. 路径设置 ---
# ==============================================================================
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# ==============================================================================
# --- 1. 定义你的实验搜索空间 ---
# ==============================================================================
BASE_EXPERIMENT_GRID = [
    {"name": "基线模型_4x256", "hidden_dims": [
        256, 256, 256, 256], "dropout_rate": 0.2},
    {"name": "瓶颈结构_窄", "hidden_dims": [
        128, 256, 256, 128], "dropout_rate": 0.3},
    {"name": "瓶颈结构_宽", "hidden_dims": [
        256, 512, 512, 256], "dropout_rate": 0.4},
    {"name": "逐渐变窄_深", "hidden_dims": [512, 256, 128], "dropout_rate": 0.3},
    {"name": "逐渐变宽", "hidden_dims": [128, 256, 512], "dropout_rate": 0.2},
]

# --- 实验控制设置 ---
NUM_REPETITIONS = 3
OPAMP_TYPE = '5t_opamp'
BASE_RESULTS_DIR = PROJECT_ROOT / "results_experiments_fixed_lr"  # 使用新目录以区分

# <<< --- 核心改动：在这里设置一个固定的微调学习率 --- >>>
FIXED_LR_FINETUNE = 1e-4  # 您提议的、安全的小学习率

# ==============================================================================
# --- 2. 动态生成完整的实验列表 ---
# ==============================================================================
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

print(f"*** 所有微调将使用固定的学习率: {FIXED_LR_FINETUNE} ***")

for i, params in enumerate(EXPERIMENT_GRID):
    exp_name = f"{i+1:02d}_{params['name']}"
    print(f"\n{'='*80}")
    print(f"🚀 开始实验 {i+1}/{len(EXPERIMENT_GRID)}: {exp_name}")
    print(f"   - 结构 (hidden_dims): {params['hidden_dims']}")
    print(f"   - 丢弃率 (dropout_rate): {params['dropout_rate']}")
    print(f"{'='*80}")

    exp_results_path = BASE_RESULTS_DIR / exp_name
    exp_results_path.mkdir(parents=True, exist_ok=True)

    # 构建命令行指令 (不再需要自动查找LR)
    command = [
        "python", "train.py",
        "--opamp", OPAMP_TYPE,
        "--hidden_dims", str(params['hidden_dims']),
        "--dropout_rate", str(params['dropout_rate']),
        "--lr_finetune", str(FIXED_LR_FINETUNE),  # <-- 使用固定的学习率
        "--save_path", str(exp_results_path),
        "--restart"
    ]

    process = subprocess.Popen(
        command, cwd=SRC_DIR, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding='utf-8'
    )

    final_val_nll = None
    log_file_path = exp_results_path / "training.log"
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        for line in iter(process.stdout.readline, ''):
            print(line.strip())
            log_file.write(line)
            if "全局最优模型的微调验证 NLL 为:" in line:
                try:
                    final_val_nll = float(line.split(":")[1].strip())
                except (IndexError, ValueError):
                    pass
    process.wait()

    if final_val_nll is not None:
        RESULTS.append({
            '完整实验名称': exp_name, '基础模型': params['base_name'],
            'hidden_dims': str(params['hidden_dims']), 'dropout_rate': params['dropout_rate'],
            'lr_finetune': FIXED_LR_FINETUNE, 'final_val_nll': final_val_nll
        })
    else:
        # ... (记录 NaN 的逻辑不变) ...
        RESULTS.append({
            '完整实验名称': exp_name, '基础模型': params['base_name'],
            'hidden_dims': str(params['hidden_dims']), 'dropout_rate': params['dropout_rate'],
            'lr_finetune': FIXED_LR_FINETUNE, 'final_val_nll': float('NaN')
        })

# ==============================================================================
# --- 4. 汇总并展示最终结果 ---
# ==============================================================================
# ... (这部分代码无需任何修改，它会自动处理和展示结果) ...
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
