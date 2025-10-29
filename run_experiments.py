# run_experiments.py (已更新：支持重复测试和聚合分析)
import subprocess
import os
import pandas as pd
import time

# ==============================================================================
# --- 1. 定义你的实验搜索空间 ---
# ==============================================================================
# 在这里定义所有你想测试的模型结构组合。
# 每个字典代表一种基础实验配置。
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
NUM_REPETITIONS = 3      # <-- 在这里设置每个实验重复的次数
OPAMP_TYPE = '5t_opamp'  # 你想测试的电路类型
BASE_RESULTS_DIR = "results_experiments"  # 所有实验结果的根目录

# ==============================================================================
# --- 2. 动态生成完整的实验列表 ---
# ==============================================================================
# 根据重复次数，自动生成一个扩展的实验列表
EXPERIMENT_GRID = []
for exp_params in BASE_EXPERIMENT_GRID:
    for run_num in range(1, NUM_REPETITIONS + 1):
        # 深度复制字典以避免互相影响
        new_params = exp_params.copy()
        # 为每次运行创建一个唯一的名称，例如 "基线模型_4x256_run1"
        new_params['name'] = f"{exp_params['name']}_run{run_num}"
        # 记录基础名称，用于后续聚合分析
        new_params['base_name'] = exp_params['name']
        EXPERIMENT_GRID.append(new_params)


# ==============================================================================
# --- 3. 实验执行与结果捕获 ---
# ==============================================================================

RESULTS = []
start_time = time.time()

# 确保主结果目录存在
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

for i, params in enumerate(EXPERIMENT_GRID):
    # 使用更新后的唯一名称
    exp_name = f"{i+1:02d}_{params['name']}"
    print(f"\n{'='*80}")
    print(f"🚀 开始实验 {i+1}/{len(EXPERIMENT_GRID)}: {exp_name}")
    print(f"   - 结构 (hidden_dims): {params['hidden_dims']}")
    print(f"   - 丢弃率 (dropout_rate): {params['dropout_rate']}")
    print(f"{'='*80}")

    # 为本次实验创建独立的输出目录
    exp_results_path = os.path.join(BASE_RESULTS_DIR, exp_name)
    os.makedirs(exp_results_path, exist_ok=True)

    # 构建命令行指令
    command = [
        "python",
        os.path.join("src", "train.py"),
        "--opamp", OPAMP_TYPE,
        "--hidden_dims", str(params['hidden_dims']),  # 将列表转换为字符串
        "--dropout_rate", str(params['dropout_rate']),
        "--save_path", exp_results_path,  # 使用独立路径
        "--restart"  # 确保每次都重新训练
    ]

    # 执行命令并实时捕获输出
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding='utf-8')

    final_val_nll = None
    log_file_path = os.path.join(exp_results_path, "training.log")

    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        for line in iter(process.stdout.readline, ''):
            print(line.strip())
            log_file.write(line)
            # 从日志中捕获最终的验证损失
            if "全局最优模型的微调验证 NLL 为:" in line:
                try:
                    final_val_nll = float(line.split(":")[1].strip())
                except (IndexError, ValueError):
                    pass

    process.wait()

    # 记录结果
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
# --- 4. 汇总并展示最终结果 ---
# ==============================================================================

end_time = time.time()
total_duration = end_time - start_time

print("\n\n" + "="*80)
print("🎉 所有实验已完成！")
print(f"总耗时: {total_duration / 60:.2f} 分钟")
print("="*80)

if RESULTS:
    results_df = pd.DataFrame(RESULTS)
    # --- 详细结果展示 ---
    print("\n📊 所有运行的详细结果 (从优到劣排序):")
    detailed_results = results_df.sort_values(
        by='final_val_nll', ascending=True)
    print(detailed_results.to_string(index=False))
    summary_path = os.path.join(
        BASE_RESULTS_DIR, "experiment_summary_detailed.csv")
    detailed_results.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"\n📄 详细结果已保存至: {summary_path}")

    # --- 聚合结果分析 ---
    print("\n\n" + "="*80)
    print("📈 按基础模型聚合的统计结果:")
    aggregated_df = results_df.groupby('基础模型')['final_val_nll'].agg(
        ['mean', 'std', 'min', 'max', 'count']).sort_values(by='mean', ascending=True)
    aggregated_df.rename(columns={
        'mean': '平均NLL', 'std': 'NLL标准差', 'min': '最佳NLL', 'max': '最差NLL', 'count': '运行次数'
    }, inplace=True)

    print(aggregated_df)
    agg_summary_path = os.path.join(
        BASE_RESULTS_DIR, "experiment_summary_aggregated.csv")
    aggregated_df.to_csv(agg_summary_path, encoding='utf-8-sig')
    print(f"\n📄 聚合统计结果已保存至: {agg_summary_path}")

    # --- 最终推荐 ---
    best_model_name = aggregated_df.index[0]
    best_model_stats = aggregated_df.iloc[0]
    print("\n\n🏆 综合表现最佳的模型结构 (基于平均NLL):")
    print(f"   - 名称: {best_model_name}")
    print(f"   - 平均验证集NLL: {best_model_stats['平均NLL']:.6f}")
    print(f"   - 稳定性 (标准差): {best_model_stats['NLL标准差']:.6f}")
else:
    print("未能成功记录任何实验结果。")
