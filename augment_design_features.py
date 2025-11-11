#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强 two_stage_opamp 的 *_design_features.csv：

- 输入只用原始 13 列: w1..w5, l1..l5, cc, cr, ibias
- 输出时在同一个 CSV 里追加：
    * 明确的 W/L 特征: w1_over_l1..w5_over_l5, log_w1_over_l1..log_w5_over_l5
    * 通用几何/偏置特征: 归一化宽/长, 面积, mismatch, 各种 log/比值
    * CMRR 相关特征: cmrr_pair12_*, cmrr_pair34_*, cmrr_pair45_*
    * DC gain 相关特征: dcgain_stage1_proxy, dcgain_stage2_proxy, dcgain_total_proxy

用法（你指定的格式）：
    python augment_design_features.py --root ~/eda-for-transfer-learning-1 --inplace

不加 --inplace 时，会在同目录写 *_aug.csv，不覆盖原文件。
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

BASE_COLS = [
    "w1", "w2", "w3", "w4", "w5",
    "l1", "l2", "l3", "l4", "l5",
    "cc", "cr", "ibias",
]
EPS = 1e-12


def safe_log(x):
    return np.log(np.clip(x, EPS, None))


def augment_df(df: pd.DataFrame) -> pd.DataFrame:
    """在原始 df 的基础上追加派生特征，不删除任何原列。"""
    missing = [c for c in BASE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必需列: {missing}")

    out = df.copy()

    # ---------- 0) 基础聚合 ----------
    w_cols = [f"w{i}" for i in range(1, 6)]
    l_cols = [f"l{i}" for i in range(1, 6)]

    W = out[w_cols].to_numpy()
    L = out[l_cols].to_numpy()

    w_sum = np.clip(W.sum(axis=1), EPS, None)
    l_sum = np.clip(L.sum(axis=1), EPS, None)

    out["w_sum"] = w_sum
    out["l_sum"] = l_sum
    out["log_w_sum"] = safe_log(w_sum)
    out["log_l_sum"] = safe_log(l_sum)

    # ---------- 1) 偏置 / 补偿 的 log 与比值 ----------
    out["log_ibias"] = safe_log(out["ibias"].to_numpy())
    out["log_cc"] = safe_log(out["cc"].to_numpy())
    out["log_cr"] = safe_log(out["cr"].to_numpy())

    out["ibias_over_cc"] = out["ibias"] / np.clip(out["cc"], EPS, None)
    out["ibias_over_cr"] = out["ibias"] / np.clip(out["cr"], EPS, None)
    out["cr_over_cc"]    = out["cr"]    / np.clip(out["cc"], EPS, None)

    out["log_ibias_over_cc"] = out["log_ibias"] - out["log_cc"]
    out["log_ibias_over_cr"] = out["log_ibias"] - out["log_cr"]
    out["log_cr_over_cc"]    = out["log_cr"]    - out["log_cc"]

    # ---------- 2) 每支管子: W/L、归一化宽/长、面积/失配、log(W/L) ----------
    for i in range(1, 6):
        wi = out[f"w{i}"].to_numpy()
        li = out[f"l{i}"].to_numpy()

        # 明确的 W/L 列名
        wl = wi / np.clip(li, EPS, None)
        out[f"w{i}_over_l{i}"] = wl
        out[f"log_w{i}_over_l{i}"] = safe_log(wi) - safe_log(li)

        # 归一化宽/长（和电路的“资源分配”相关）
        out[f"w{i}_norm"] = wi / w_sum
        out[f"l{i}_norm"] = li / l_sum

        # 面积 & mismatch proxy（与 CMRR/offset 强相关）
        area_sqrt = np.sqrt(np.clip(wi * li, EPS, None))  # sqrt(WL)
        out[f"area{i}"] = area_sqrt
        out[f"mismatch{i}"] = 1.0 / area_sqrt

    # ---------- 3) 每支管子: gm/ro/Av/UGF 的 log 代理 ----------
    log_ibias = out["log_ibias"].to_numpy()
    log_cc    = out["log_cc"].to_numpy()

    for i in range(1, 6):
        wi = out[f"w{i}"].to_numpy()
        li = out[f"l{i}"].to_numpy()
        lw = safe_log(wi)
        ll = safe_log(li)

        # gm_hat ∝ sqrt((W/L) * Ibias)
        log_gm_hat = 0.5 * (lw - ll + log_ibias)
        # ro_hat ∝ L / Ibias
        log_ro_hat = ll - log_ibias
        # Av_hat = gm_hat * ro_hat
        log_av_hat = log_gm_hat + log_ro_hat
        # UGF_hat ∝ gm_hat / Cc
        log_ugf_hat = 0.5 * (lw - ll) + 0.5 * log_ibias - log_cc

        out[f"log_gm_hat_{i}"] = log_gm_hat
        out[f"log_ro_hat_{i}"] = log_ro_hat
        out[f"log_av_hat_{i}"] = log_av_hat
        out[f"log_ugf_hat_{i}"] = log_ugf_hat

    # ---------- 4) DC GAIN 相关特征（名字直接叫 dcgain_*） ----------
    # 这里假设 stage1 主要由管 1 决定，stage2 主要由管 3 决定（你后面可以根据拓扑再调整索引）
    out["dcgain_stage1_proxy"] = out["log_av_hat_1"]
    out["dcgain_stage2_proxy"] = out["log_av_hat_3"]
    out["dcgain_total_proxy"] = out["dcgain_stage1_proxy"] + out["dcgain_stage2_proxy"]

    # 也附带一个“整体增益等级”的粗 proxy（所有管子 Av_hat 之和）
    av_cols = [f"log_av_hat_{i}" for i in range(1, 6)]
    out["dcgain_all_devices_proxy"] = out[av_cols].sum(axis=1)

    # ---------- 5) CMRR 相关特征（配对匹配度） ----------
    # 假设一些常见配对：(1,2) 可能是输入对，(3,4) 某一级配对，(4,5) 镜像/负载配对（具体拓扑你最清楚）
    # 不管具体电路，配对面积比和归一化差都会强烈影响 CMRR。
    def add_cmrr_pair_features(i, j):
        ai = out[f"area{i}"]
        aj = out[f"area{j}"]
        # 面积比 & log 比
        out[f"cmrr_pair{i}{j}_area_ratio"] = ai / aj.replace(0, np.nan)
        out[f"cmrr_pair{i}{j}_area_ratio"] = out[f"cmrr_pair{i}{j}_area_ratio"].fillna(0.0)

        out[f"cmrr_pair{i}{j}_log_area_ratio"] = safe_log(ai.to_numpy()) - safe_log(aj.to_numpy())

        # 归一化面积差：|Ai - Aj| / ((Ai + Aj)/2)
        denom = (ai + aj) / 2.0
        num = (ai - aj).abs()
        norm_diff = num / denom.replace(0, np.nan)
        out[f"cmrr_pair{i}{j}_area_diff_norm"] = norm_diff.fillna(0.0)

        # mismatch proxy 方向上的差值
        mi = out[f"mismatch{i}"]
        mj = out[f"mismatch{j}"]
        out[f"cmrr_pair{i}{j}_mismatch_diff"] = (mi - mj).abs()

    # 三组典型配对（你之后可以根据实际拓扑再增删）
    add_cmrr_pair_features(1, 2)
    add_cmrr_pair_features(3, 4)
    add_cmrr_pair_features(4, 5)

    return out


def write_csv(df_aug: pd.DataFrame, src_path: Path, inplace: bool):
    if inplace:
        backup = src_path.with_suffix(src_path.suffix + ".bak")
        shutil.copy2(src_path, backup)
        df_aug.to_csv(src_path, index=False)
        print(f"[覆盖写入] {src_path}（已备份为 {backup.name}）")
    else:
        dst = src_path.with_name(src_path.stem + "_aug.csv")
        df_aug.to_csv(dst, index=False)
        print(f"[生成] {dst}")


def process_two_stage(root: Path, inplace: bool):
    base = root / "data" / "01_train_set" / "two_stage_opamp"
    targets = [
        base / "source" / "pretrain_design_features.csv",
        base / "target" / "target_design_features.csv",
    ]
    for p in targets:
        if not p.exists():
            print(f"[跳过] 未找到：{p}")
            continue
        print(f"\n[读取] {p}")
        df = pd.read_csv(p)
        before = len(df.columns)
        df_aug = augment_df(df)
        after = len(df_aug.columns)
        print(f"[列数] 原始 {before} → 增强 {after}（新增 {after - before} 列）")
        write_csv(df_aug, p, inplace)


def parse_args():
    ap = argparse.ArgumentParser(description="为 two_stage_opamp 的设计特征追加 W/L + CMRR + DCGAIN 相关派生特征")
    ap.add_argument("--root", type=str, required=True,
                    help="项目根目录，例如 ~/eda-for-transfer-learning-1")
    ap.add_argument("--inplace", action="store_true",
                    help="覆盖写回原 CSV（会先保存 .bak 备份）。不加则写 *_aug.csv")
    return ap.parse_args()


def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"根目录不存在：{root}")

    print(f"项目根目录：{root}")
    print(f"写回方式：{'覆盖写回(含 .bak 备份)' if args.inplace else '生成 *_aug.csv'}")
    process_two_stage(root, args.inplace)
    print("\n✅ 完成。已为 two_stage_opamp 的 design_features 加入 W/L 和 CMRR/DCGAIN 相关特征。")


if __name__ == "__main__":
    main()
