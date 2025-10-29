# src/run_experiments.py (已更新：适配在 src 目录内运行)
import subprocess
import os
import pandas as pd
import time
from pathlib import Path  # 导入 pathlib 库用于路径管理

# ==============================================================================
# --- 0. 路径设置 (新增) ---
# ==============================================================================
# 获取当前脚本文件所在的目录 (即 src 目录)
SRC_DIR = Path(__file__).resolve().parent
# 获取项目的根目录 (src 目录的上一级)
PROJECT_ROOT = SRC_DIR.parent

# ==============================================================================
# --- 1. 定义你的实验搜索空间 ---
# ==============================================================================
# 在这里定义所有你想测试的模型结构组合。
BASE_EXPERIMENT_GRID = [
    {
        "name": "基线模型_4x256",
        "hidden_dims": [256, 256, 256, 256],
        "dropout_rate": 0.2
    },
    {
        "name": "瓶颈结构_窄",
        "hidden_dims": [128, 256, 256, 128],
        "dropout_rate": 0.3
    },
    {
        "name": "瓶颈结构_宽",
        "hidden_dims": [256, 512, 512, 256],
        "dropout_rate": 0.4
    },
    {
        "name": "逐渐变窄_深",
        "hidden_dims": [512, 256, 128],
        "dropout_rate": 0.3
    },
    {
        "name": "逐渐变宽",
        "hidden_dims": [128, 256, 512],
        "dropout_rate": 0.2
    },
]

# --- 实验控制设置 ---
NUM_REPETITIONS = 3
OPAMP_TYPE = '5t_opamp'
# <-- 关键改动: 结果目录路径基于项目根目录构建
BASE_RESULTS_DIR = PROJECT_ROOT / "results_experiments"

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

# 确保主结果目录存在
BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

for i, params in enumerate(EXPERIMENT_GRID):
    exp_name = f"{i+1:02d}_{params['name']}"
    print(f"\n{'='*80}")
    print(f"🚀 开始实验 {i+1}/{len(EXPERIMENT_GRID)}: {exp_name}")
    print(f"   - 结构 (hidden_dims): {params['hidden_dims']}")
    print(f"   - 丢弃率 (dropout_rate): {params['dropout_rate']}")
    print(f"{'='*80}")

    # <-- 关键改动: 实验结果子目录也基于根目录构建
    exp_results_path = BASE_RESULTS_DIR / exp_name
    exp_results_path.mkdir(parents=True, exist_ok=True)

    # <-- 关键改动: 构建命令行指令
    command = [
        "python",
        "train.py",  # 直接调用同级文件
        "--opamp", OPAMP_TYPE,
        "--hidden_dims", str(params['hidden_dims']),
        "--dropout_rate", str(params['dropout_rate']),
        "--save_path", str(exp_results_path),  # 将Path对象转为字符串
        "--restart"
    ]

    # <-- 关键改动: 执行命令，并设置工作目录为 src 目录
    process = subprocess.Popen(
        command,
        cwd=SRC_DIR,  # 确保 train.py 在正确的环境下运行
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8'
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

    # 记录结果 (逻辑不变)
    if final_val_nll is not None:
        print(f"✅ 实验 {exp_name} 完成。 最终 Val NLL: {final_val_nll:.6f}")
        RESULTS.append({
            '完整实验名称': exp_name,
            '基础模型': params['base_name'],
            'hidden_dims': str(params['hidden_dims']),
            'dropout_rate': params['dropout_rate'],
            'final_val_nll': final_val_nll
        })
    else:
        print(f"⚠️ 实验 {exp_name} 完成，但未能捕获到最终 Val NLL。请检查日志: {log_file_path}")
        RESULTS.append({
            '完整实验名称': exp_name,
            '基础模型': params['base_name'],
            'hidden_dims': str(params['hidden_dims']),
            'dropout_rate': params['dropout_rate'],
            'final_val_nll': float('NaN')
        })


# ==============================================================================
# --- 4. 汇总并展示最终结果 (路径已更新) ---
# ==============================================================================
end_time = time.time()
total_duration = end_time - start_time

print("\n\n" + "="*80)
print("🎉 所有实验已完成！")
print(f"总耗时: {total_duration / 60:.2f} 分钟")
print("="*80)

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
    aggregated_df.rename(columns={
        'mean': '平均NLL', 'std': 'NLL标准差', 'min': '最佳NLL', 'max': '最差NLL', 'count': '运行次数'
    }, inplace=True)
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
else:
    print("未能成功记录任何实验结果。")
