"""
命令行入口：用于快速跑一遍数据加载/预处理，并打印形状。
注意：不要把 main 放在 __init__.py。
"""

import argparse
from .data_loader import get_data_and_scalers

def main():
    ap = argparse.ArgumentParser("data_loader CLI")
    ap.add_argument("--opamp", type=str, default="5t_opamp", help="工艺类型名")
    ap.add_argument("--val-split", type=float, default=0.2, help="B 域验证集占比")
    ap.add_argument("--seed", type=int, default=42, help="划分随机种子")
    args = ap.parse_args()

    data = get_data_and_scalers(
        opamp_type=args.opamp,
        target_val_split=args.val_split,
        random_state=args.seed,
    )

    print("\n--- 数据集形状 ---")
    print("Source X:", data["source"][0].shape, "  Source y:", data["source"][1].shape)
    print("Target Train X:", data["target_train"][0].shape, "  Target Train y:", data["target_train"][1].shape)
    print("Target Val   X:", data["target_val"][0].shape, "  Target Val   y:", data["target_val"][1].shape)
    print("\n完成。scalers 已保存到 results/ 目录。")

if __name__ == "__main__":
    main()
