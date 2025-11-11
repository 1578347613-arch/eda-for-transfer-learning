import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os

# --- 1. 命令行参数解析 (已更新) ---


def parse_arguments():
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description="用于运放（Op-Amp）迁移学习任务的数据可视化工具。",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--opamp',
        required=True,
        choices=['5t_opamp', 'two_stage_opamp'],
        help="指定要分析的运放类型。"
    )

    # [新] 使用 --domain 来指定要分析的域
    # 这比之前指定单个文件更清晰
    parser.add_argument(
        '--domain',
        required=True,
        choices=['source', 'target'],
        help="指定要分析的域 (source 或 target)。\n"
             "  - source: 加载 'pretrain' 文件\n"
             "  - target: 加载 'target' 文件"
    )

    args = parser.parse_args()
    return args


def get_file_paths(args: argparse.Namespace) -> tuple[str, str, str]:
    """
    [新] 根据解析的参数构建 *两个* 文件路径 (features 和 performance)。

    Returns:
        (path_features, path_performance, friendly_name)
    """
    base_dir = os.path.join("data", "01_train_set")
    opamp_type = args.opamp
    domain = args.domain

    path_features = ""
    path_performance = ""

    if domain == 'source':
        path_features = os.path.join(
            base_dir, opamp_type, domain, "pretrain_design_features.csv")
        path_performance = os.path.join(
            base_dir, opamp_type, domain, "pretrain_targets.csv")
        friendly_name = f"{opamp_type}_source"
    elif domain == 'target':
        path_features = os.path.join(
            base_dir, opamp_type, domain, "target_design_features.csv")
        path_performance = os.path.join(
            base_dir, opamp_type, domain, "target_targets.csv")
        friendly_name = f"{opamp_type}_target"

    return path_features, path_performance, friendly_name

# --- 2. 数据加载 (不变) ---


def load_data(filepath: str) -> pd.DataFrame | None:
    """
    从 CSV 文件加载数据。
    """
    try:
        data = pd.read_csv(filepath)
        data.columns = data.columns.str.strip()
        print(f"✔ 成功加载: {filepath}")
        print(f"  形状 (行, 列): {data.shape}")
        return data
    except FileNotFoundError:
        print(f"✘ 错误: 未找到文件 '{filepath}'。")
        return None
    except Exception as e:
        print(f"✘ 加载 {filepath} 时发生未知错误: {e}")
        return None

# --- 3. 核心可视化函数 ---


def plot_correlation_heatmap(data: pd.DataFrame, title_prefix: str, save_path: str):
    """
    [更新] 绘制标准(方阵)相关性热力图。
    用于 F-vs-F 或 P-vs-P。
    """
    print(f"\n[...正在生成 {title_prefix} 热力图...]")
    numerical_data = data.select_dtypes(include=[np.number])
    if numerical_data.empty:
        print("✘ 错误: 未找到可计算相关的数值列。")
        return

    plt.figure(figsize=(16, 12))
    corr_matrix = numerical_data.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(corr_matrix,
                mask=mask,
                annot=True,
                fmt=".2f",
                cmap='coolwarm',
                linewidths=0.5,
                annot_kws={"size": 8},  # 调整字体大小
                cbar_kws={"shrink": .8})

    plt.title(f'{title_prefix} 相关性热力图', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✔ 已保存热力图至: {save_path}")


def plot_distribution(data: pd.DataFrame, column_name: str, save_path_prefix: str):
    """
    绘制单个参数的分布图 (直方图 + 核密度估计)。
    """
    if column_name not in data.columns:
        return

    plt.figure(figsize=(10, 6))
    sns.histplot(data[column_name], kde=True, bins=50)
    plt.title(f'{column_name} 的数据分布', fontsize=15)
    plt.xlabel(column_name)
    plt.ylabel('频数 (Frequency)')
    plt.grid(True, linestyle='--', alpha=0.6)

    save_path = f"{save_path_prefix}_{column_name}.png"
    plt.savefig(save_path)
    plt.close()

# --- [!! 重点 !!] 你的新功能 ---


def plot_cross_correlation_heatmap(
    features_df: pd.DataFrame,
    performance_df: pd.DataFrame,
    title_prefix: str,
    save_path: str
):
    """
    [新] 绘制 *交叉相关性* 热力图 (Features vs. Performance)。
    这正是你想要的！
    """
    print(f"\n[...正在生成 {title_prefix} 交叉热力图...]")

    # 1. 将两个 DataFrame 合并
    # 假设它们的行是 1:1 对应的
    if len(features_df) != len(performance_df):
        print("✘ 错误: features 和 performance 的行数不匹配!")
        return

    try:
        combined_data = pd.concat([features_df, performance_df], axis=1)
    except pd.errors.InvalidIndexError:
        print("✘ 错误: Features 和 Performance 的索引冲突。正在重置索引...")
        features_df = features_df.reset_index(drop=True)
        performance_df = performance_df.reset_index(drop=True)
        combined_data = pd.concat([features_df, performance_df], axis=1)

    # 2. 计算完整的相关性矩阵
    corr_matrix = combined_data.corr()

    # 3. [关键] 只提取出 F vs. P 的部分
    # 行=features, 列=performance
    cross_corr = corr_matrix.loc[features_df.columns, performance_df.columns]

    if cross_corr.empty:
        print("✘ 错误: 无法计算交叉相关性。")
        return

    # 4. 绘制这个 *矩形* 热力图
    # 注意：这个图不是方的，所以我们不需要 mask
    plt.figure(figsize=(14, 10))  # 调整大小以适应矩形

    sns.heatmap(cross_corr,
                annot=True,         # 显示数值
                fmt=".2f",          # 两位小数
                cmap='coolwarm',    # 冷暖色
                linewidths=0.5,
                annot_kws={"size": 10})  # 字体大小

    plt.title(f'{title_prefix} 交叉相关性 (Features vs. Performance)', fontsize=16)
    plt.xlabel('Performance (Targets)')
    plt.ylabel('Features (Design)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✔ 已保存交叉热力图至: {save_path}")


# --- 4. 主执行函数 (已更新) ---

def main():
    """
    主函数：解析参数、加载数据并调用可视化。
    """
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. 解析参数
    args = parse_arguments()

    # 2. 构建文件路径
    path_feat, path_perf, friendly_name = get_file_paths(args)

    print(f"--- 任务开始 ---")
    print(f"  Op-Amp:   {args.opamp}")
    print(f"  Domain:   {args.domain}")
    print("--------------------")

    # 3. 加载 *两个* 数据文件
    data_feat = load_data(path_feat)
    data_perf = load_data(path_perf)

    if data_feat is None or data_perf is None:
        print("\n--- 数据加载不完整，程序退出 ---")
        return

    print("\n--- 可视化分析开始 ---")

    # --- 自动化分析 ---

    # 1. [新] 你要的交叉相关性热力图 (F vs. P)
    plot_cross_correlation_heatmap(
        data_feat,
        data_perf,
        title_prefix=friendly_name,
        save_path=f"reports/{friendly_name}_[CROSS]_Feat_vs_Perf.png"
    )

    # 2. [旧] 独立的热力图 (F vs. F)
    plot_correlation_heatmap(
        data_feat,
        title_prefix=f"{friendly_name}_Features",
        save_path=f"reports/{friendly_name}_[FEAT]_Heatmap.png"
    )

    # 3. [旧] 独立的热力图 (P vs. P)
    plot_correlation_heatmap(
        data_perf,
        title_prefix=f"{friendly_name}_Performance",
        save_path=f"reports/{friendly_name}_[PERF]_Heatmap.png"
    )

    # 4. [旧] 自动为所有列生成分布图
    print(f"\n[...正在为 {friendly_name} 生成所有参数的分布图...]")

    # 为 Features 绘制
    for col in data_feat.columns:
        plot_distribution(data_feat, col,
                          save_path_prefix=f"reports/dist/{friendly_name}_feat")
    # 为 Performance 绘制
    for col in data_perf.columns:
        plot_distribution(data_perf, col,
                          save_path_prefix=f"reports/dist/{friendly_name}_perf")

    print("\n--- 可视化分析完成 ---")
    print("所有报告已保存在 'reports/' 和 'reports/dist/' 文件夹中。")


if __name__ == "__main__":
    # 自动创建用于保存报告的文件夹
    if not os.path.exists("reports"):
        os.makedirs("reports")
    # 为分布图创建子文件夹
    if not os.path.exists("reports/dist"):
        os.makedirs("reports/dist")

    main()
