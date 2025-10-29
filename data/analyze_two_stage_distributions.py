#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_two_stage_distributions.py

用途：
- 对 two_stage_opamp 的 source/target 数据做分布体检（纯文本，不画图）
- 重点诊断 dc_gain / cmrr 预测差的潜在原因：分布漂移、覆盖度、极端偏斜、相关性跨域变化等

输入目录结构（--root 指向该目录）：
  <root>/
    source/
      pretrain_design_features.csv
      pretrain_targets.csv
    target/
      target_design_features.csv
      target_targets.csv

输出：
- 终端打印关键表
- 全量结果写入 <root>/analysis_reports/<时间戳>/*.csv
"""

import argparse
from pathlib import Path
import sys
import math
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# 可选：scipy做KS与Wasserstein；若不可用，自动降级
try:
    from scipy.stats import ks_2samp, wasserstein_distance, skew, kurtosis, pearsonr
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# ------- 配置：可能需要 log1p 的目标列（仅用于提示/检查；不修改数据） -------
DEFAULT_LOGY_CANDIDATES = ["slewrate_pos", "dc_gain", "ugf", "cmrr"]

# 显示选项
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 200)


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    df = pd.read_csv(path)
    return df


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def apply_alias(df: pd.DataFrame, alias_map: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """将 df 中的列名按 alias_map 的反向映射重命名：from_name -> to_name。
       alias_map: {'to_name': 'from_name'}（更符合人脑），内部会转换为 {from:to}
    """
    if not alias_map:
        return df, {}
    rename = {}
    for to_name, from_name in alias_map.items():
        if from_name in df.columns and to_name not in df.columns:
            rename[from_name] = to_name
    if rename:
        df = df.rename(columns=rename)
    return df, rename


def basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """计算基础统计：count / n_missing / mean / std / min / p5 / p25 / median / p75 / p95 / max / skew / kurt / n_nonpos"""
    stats = {}
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        arr = s.values
        arr = arr[~np.isnan(arr)]
        n = len(arr)
        if n == 0:
            stats[c] = {
                "count": 0, "n_missing": df.shape[0], "mean": np.nan, "std": np.nan,
                "min": np.nan, "p5": np.nan, "p25": np.nan, "median": np.nan, "p75": np.nan, "p95": np.nan, "max": np.nan,
                "skew": np.nan, "kurt": np.nan, "n_nonpos": np.nan
            }
            continue
        # 偏度/峰度
        if SCIPY_OK:
            sk = float(skew(arr, bias=False)) if n >= 3 else np.nan
            ku = float(kurtosis(arr, fisher=True, bias=False)) if n >= 4 else np.nan
        else:
            # 无 scipy 时，用近似或留空
            sk = np.nan
            ku = np.nan
        stats[c] = {
            "count": int(n),
            "n_missing": int(df.shape[0] - n),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if n > 1 else 0.0,
            "min": float(np.min(arr)),
            "p5": float(np.percentile(arr, 5)),
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(np.max(arr)),
            "skew": sk,
            "kurt": ku,
            "n_nonpos": int((arr <= 0).sum())
        }
    out = pd.DataFrame(stats).T
    out.index.name = "column"
    return out


def distribution_shift_table(src: pd.DataFrame, tgt: pd.DataFrame) -> pd.DataFrame:
    """对两个数据框相同列，计算分布漂移指标：
       - mean_src / mean_tgt / mean_diff_ratio
       - std_src / std_tgt / std_ratio
       - coverage_in_src: 目标样本落在 源[min,max] 区间内的比例（越低越容易外插）
       - KS_stat / KS_pvalue
       - WD (Wasserstein) 及 WD/std_src
    """
    common = [c for c in src.columns if c in tgt.columns]
    rows = []
    for c in common:
        a = pd.to_numeric(src[c], errors="coerce").values
        b = pd.to_numeric(tgt[c], errors="coerce").values
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
        if len(a) == 0 or len(b) == 0:
            continue
        mean_a = float(np.mean(a)); mean_b = float(np.mean(b))
        std_a = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
        std_b = float(np.std(b, ddof=1)) if len(b) > 1 else 0.0
        denom = max(abs(mean_a), 1e-12)
        mean_diff_ratio = (mean_b - mean_a) / denom
        std_ratio = (std_b / max(std_a, 1e-12)) if std_a > 0 else np.inf

        # 覆盖度：b 落在 [min(a), max(a)] 内的比例
        min_a = float(np.min(a)); max_a = float(np.max(a))
        coverage = float(np.mean((b >= min_a) & (b <= max_a)))

        # KS 与 Wasserstein
        if SCIPY_OK:
            ks = ks_2samp(a, b, alternative="two-sided", mode="auto")
            ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
            wd = float(wasserstein_distance(a, b))
        else:
            ks_stat, ks_p = np.nan, np.nan
            # 简单替代：均值差的绝对值作为粗略距离
            wd = float(abs(mean_b - mean_a))
        wd_std = wd / max(std_a, 1e-12) if std_a > 0 else np.inf

        rows.append({
            "column": c,
            "mean_src": mean_a, "mean_tgt": mean_b, "mean_diff_ratio": mean_diff_ratio,
            "std_src": std_a, "std_tgt": std_b, "std_ratio": std_ratio,
            "coverage_in_src": coverage,
            "KS_stat": ks_stat, "KS_pvalue": ks_p,
            "WD": wd, "WD_over_std_src": wd_std
        })
    out = pd.DataFrame(rows).sort_values("coverage_in_src")
    return out


def pearson_corr(X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """计算 X(特征) 与 y(目标) 的皮尔逊相关，返回长表：['feature','target','pearson_r','abs_r']"""
    feats = list(X.columns)
    tars = list(y.columns)
    rows = []
    for f in feats:
        xf = pd.to_numeric(X[f], errors="coerce").values
        xf = xf[~np.isnan(xf)]
        if len(xf) == 0:
            continue
        for t in tars:
            yt = pd.to_numeric(y[t], errors="coerce").values
            yt = yt[~np.isnan(yt)]
            # 对齐长度（随机截断较长的）——仅用于相关性粗评（避免复杂对齐逻辑）
            n = min(len(xf), len(yt))
            if n < 3:
                continue
            # 为稳定性，取前 n 个
            xv = xf[:n]
            yv = yt[:n]
            if SCIPY_OK:
                try:
                    r, _ = pearsonr(xv, yv)
                except Exception:
                    r = np.nan
            else:
                # 纯 numpy 计算皮尔逊
                xm = xv - np.mean(xv)
                ym = yv - np.mean(yv)
                denom = np.sqrt((xm**2).sum()) * np.sqrt((ym**2).sum())
                r = float((xm * ym).sum() / denom) if denom > 0 else np.nan
            rows.append({"feature": f, "target": t, "pearson_r": r, "abs_r": abs(r) if r==r else np.nan})
    out = pd.DataFrame(rows)
    out = out.sort_values(["target", "abs_r"], ascending=[True, False])
    return out


def print_section(title: str):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)


def main():
    ap = argparse.ArgumentParser(description="Analyze distributions for two_stage_opamp (source vs target)")
    ap.add_argument("--root", type=str, required=True, help="数据根目录（包含 source/ 与 target/ 子目录）")
    # 别名映射（例如 --alias rz cm --from cr cc 将把 'cr'->'rz'、'cc'->'cm'）
    ap.add_argument("--alias", nargs="*", default=[], help="期望列名（目标名）序列，如: rz cm")
    ap.add_argument("--from", dest="alias_from", nargs="*", default=[], help="测试CSV里实际列名序列，如: cr cc")
    # 指定哪些目标列强烈建议 log1p（仅用于报告）
    ap.add_argument("--logy", nargs="*", default=DEFAULT_LOGY_CANDIDATES, help="建议做 log1p 的目标列名，用于报告提示")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    src_feat_p = root / "source" / "pretrain_design_features.csv"
    src_tgt_p  = root / "source" / "pretrain_targets.csv"
    tgt_feat_p = root / "target" / "target_design_features.csv"
    tgt_tgt_p  = root / "target" / "target_targets.csv"

    # 读取
    print_section("加载数据")
    print(f"ROOT: {root}")
    src_X = read_csv_safe(src_feat_p)
    src_y = read_csv_safe(src_tgt_p)
    tgt_X = read_csv_safe(tgt_feat_p)
    tgt_y = read_csv_safe(tgt_tgt_p)
    print(f"source features: {src_X.shape}, source targets: {src_y.shape}")
    print(f"target features: {tgt_X.shape}, target targets: {tgt_y.shape}")

    # 别名映射
    alias_map = {}
    if args.alias and args.alias_from and len(args.alias) == len(args.alias_from):
        alias_map = {to: fr for to, fr in zip(args.alias, args.alias_from)}

    if alias_map:
        print_section("列名别名映射（features）")
        print(f"请求的重命名: {alias_map}  （含义：from -> to）")
        tgt_X, renamed_tgt = apply_alias(tgt_X, alias_map)
        src_X, renamed_src = apply_alias(src_X, alias_map)
        if renamed_src:
            print(f"[source] 已重命名: {renamed_src}")
        if renamed_tgt:
            print(f"[target] 已重命名: {renamed_tgt}")
    else:
        print_section("列名别名映射")
        print("未指定别名映射（--alias / --from），跳过。")

    # 转为数值
    src_X = coerce_numeric(src_X)
    src_y = coerce_numeric(src_y)
    tgt_X = coerce_numeric(tgt_X)
    tgt_y = coerce_numeric(tgt_y)

    # 输出目录
    out_dir = root / "analysis_reports" / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 基础统计
    print_section("基础统计：SOURCE - FEATURES")
    bs_src_X = basic_stats(src_X); print(bs_src_X.to_string())
    bs_src_X.to_csv(out_dir / "basic_stats_source_features.csv")

    print_section("基础统计：SOURCE - TARGETS")
    bs_src_y = basic_stats(src_y); print(bs_src_y.to_string())
    bs_src_y.to_csv(out_dir / "basic_stats_source_targets.csv")

    print_section("基础统计：TARGET - FEATURES")
    bs_tgt_X = basic_stats(tgt_X); print(bs_tgt_X.to_string())
    bs_tgt_X.to_csv(out_dir / "basic_stats_target_features.csv")

    print_section("基础统计：TARGET - TARGETS")
    bs_tgt_y = basic_stats(tgt_y); print(bs_tgt_y.to_string())
    bs_tgt_y.to_csv(out_dir / "basic_stats_target_targets.csv")

    # 分布漂移：features / targets
    print_section("分布漂移（FEATURES）：SOURCE vs TARGET")
    shift_X = distribution_shift_table(src_X, tgt_X)
    print(shift_X.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    shift_X.to_csv(out_dir / "shift_features_source_vs_target.csv", index=False)

    print_section("分布漂移（TARGETS）：SOURCE vs TARGET")
    # 仅对公共目标列
    common_y = [c for c in src_y.columns if c in tgt_y.columns]
    shift_y = distribution_shift_table(src_y[common_y], tgt_y[common_y])
    print(shift_y.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    shift_y.to_csv(out_dir / "shift_targets_source_vs_target.csv", index=False)

    # 针对 log1p 候选目标列的体检
    print_section("log1p 候选目标列体检（是否存在非正值 / 极端偏斜）")
    checks = []
    for col in args.logy:
        row = {"column": col, "in_source": col in src_y.columns, "in_target": col in tgt_y.columns}
        for dom, df in [("source", src_y), ("target", tgt_y)]:
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce").values
                s = s[~np.isnan(s)]
                n = len(s)
                n_nonpos = int((s <= 0).sum())
                skew_v = (skew(s, bias=False) if SCIPY_OK and n >= 3 else np.nan)
                row[f"{dom}_n"] = n
                row[f"{dom}_n_nonpos"] = n_nonpos
                row[f"{dom}_skew"] = float(skew_v) if skew_v==skew_v else np.nan
            else:
                row[f"{dom}_n"] = 0
                row[f"{dom}_n_nonpos"] = np.nan
                row[f"{dom}_skew"] = np.nan
        checks.append(row)
    df_checks = pd.DataFrame(checks)
    if not df_checks.empty:
        print(df_checks.to_string(index=False))
        df_checks.to_csv(out_dir / "logy_candidates_checks.csv", index=False)
    else:
        print("（无候选列或列不存在，跳过）")

    # 相关性（source / target）
    print_section("皮尔逊相关：FEATURES ↔ TARGETS（SOURCE）")
    corr_src = pearson_corr(src_X, src_y)
    print(corr_src.groupby("target").head(10).to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    corr_src.to_csv(out_dir / "corr_source.csv", index=False)

    print_section("皮尔逊相关：FEATURES ↔ TARGETS（TARGET）")
    corr_tgt = pearson_corr(tgt_X, tgt_y)
    print(corr_tgt.groupby("target").head(10).to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    corr_tgt.to_csv(out_dir / "corr_target.csv", index=False)

    # 针对 dc_gain / cmrr：输出 top-10 相关特征（源/目标），以及跨域差异最大的特征
    focus_cols = [c for c in ["dc_gain", "cmrr"] if (c in src_y.columns and c in tgt_y.columns)]
    for tar in focus_cols:
        print_section(f"[重点] {tar}: 源/目标 Top-10 相关特征 & 跨域相关性差异最大")
        top_src = corr_src[corr_src["target"] == tar].sort_values("abs_r", ascending=False).head(10)
        top_tgt = corr_tgt[corr_tgt["target"] == tar].sort_values("abs_r", ascending=False).head(10)
        print("-- SOURCE Top-10 --")
        print(top_src.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
        print("-- TARGET Top-10 --")
        print(top_tgt.to_string(index=False, float_format=lambda x: f"{x:.6g}"))

        # 跨域差异：以 feature 为键，r_target - r_source 的绝对值
        s_map = {r.feature: r.pearson_r for r in top_src.itertuples()}
        t_map = {r.feature: r.pearson_r for r in corr_tgt[corr_tgt["target"] == tar].itertuples()}
        # 用全体而不是仅 top_src
        s_all = {r.feature: r.pearson_r for r in corr_src[corr_src["target"] == tar].itertuples()}
        t_all = {r.feature: r.pearson_r for r in corr_tgt[corr_tgt["target"] == tar].itertuples()}
        feats = set(s_all) | set(t_all)
        diff_rows = []
        for f in feats:
            rs = s_all.get(f, np.nan)
            rt = t_all.get(f, np.nan)
            if rs==rs and rt==rt:
                diff = abs(rt - rs)
                diff_rows.append({"feature": f, "r_src": rs, "r_tgt": rt, "abs_diff": diff})
        diff_df = pd.DataFrame(diff_rows).sort_values("abs_diff", ascending=False).head(10)
        print("-- 相关性跨域变化 Top-10（abs(r_tgt - r_src)） --")
        print(diff_df.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
        diff_df.to_csv(out_dir / f"corr_shift_{tar}.csv", index=False)

    # 近似常量 / 高相似列（粗查）
    print_section("可疑列：近似常量（std 很小）与高度相似（|r|>0.995）的特征对（在 TARGET 上）")
    tiny_std = []
    for c in tgt_X.columns:
        s = pd.to_numeric(tgt_X[c], errors="coerce")
        arr = s.values[~np.isnan(s.values)]
        if len(arr) >= 2:
            std = float(np.std(arr, ddof=1))
            if std < 1e-12:
                tiny_std.append({"column": c, "std": std})
    if tiny_std:
        tiny_std_df = pd.DataFrame(tiny_std).sort_values("std")
        print("-- 近似常量列（TARGET） --")
        print(tiny_std_df.to_string(index=False))
        tiny_std_df.to_csv(out_dir / "near_constant_features_target.csv", index=False)
    else:
        print("TARGET 暂无近似常量列（阈值 1e-12）")

    # 高相似对（仅抽样简化）
    # 为避免 O(N^2) 大表，这里只在 200 列以内做全量两两；更多列则随机抽 200 列
    feat_cols = list(tgt_X.columns)
    if len(feat_cols) > 200:
        np.random.seed(42)
        feat_cols = list(np.random.choice(feat_cols, size=200, replace=False))
    corr_mat = tgt_X[feat_cols].corr(method="pearson")
    pairs = []
    for i, ci in enumerate(feat_cols):
        for j in range(i + 1, len(feat_cols)):
            cj = feat_cols[j]
            r = corr_mat.loc[ci, cj]
            if abs(r) > 0.995:
                pairs.append({"f1": ci, "f2": cj, "r": float(r)})
    if pairs:
        pair_df = pd.DataFrame(pairs).sort_values("r", ascending=False)
        print("-- 高度相似特征对（TARGET, |r|>0.995） --")
        print(pair_df.head(50).to_string(index=False, float_format=lambda x: f"{x:.6g}"))
        pair_df.to_csv(out_dir / "highly_correlated_feature_pairs_target.csv", index=False)
    else:
        print("TARGET 未发现 |r|>0.995 的特征对（或已抽样）。")

    # 聚焦 dc_gain / cmrr 的分布覆盖与漂移摘要
    focus_rows = []
    for tar in focus_cols:
        row = {"target": tar}
        if tar in shift_y["column"].values:
            r = shift_y[shift_y["column"] == tar].iloc[0].to_dict()
            row.update({
                "mean_src": r["mean_src"], "mean_tgt": r["mean_tgt"], "mean_diff_ratio": r["mean_diff_ratio"],
                "std_src": r["std_src"], "std_tgt": r["std_tgt"], "std_ratio": r["std_ratio"],
                "coverage_in_src": r["coverage_in_src"], "KS_stat": r["KS_stat"], "KS_pvalue": r["KS_pvalue"],
                "WD_over_std_src": r["WD_over_std_src"]
            })
        # 额外：是否存在非正值（影响 log1p）
        for dom, df in [("source", src_y), ("target", tgt_y)]:
            if tar in df.columns:
                arr = pd.to_numeric(df[tar], errors="coerce").values
                arr = arr[~np.isnan(arr)]
                row[f"{dom}_n_nonpos"] = int((arr <= 0).sum())
                row[f"{dom}_skew"] = float(skew(arr, bias=False)) if SCIPY_OK and len(arr)>=3 else np.nan
        focus_rows.append(row)
    if focus_rows:
        print_section("重点目标（dc_gain / cmrr）分布覆盖与漂移摘要")
        focus_df = pd.DataFrame(focus_rows)
        print(focus_df.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
        focus_df.to_csv(out_dir / "focus_targets_summary.csv", index=False)

    print_section("完成")
    print(f"完整报告已写入：{out_dir}")
    print("建议先看：shift_targets_source_vs_target.csv、focus_targets_summary.csv、corr_shift_dc_gain.csv / corr_shift_cmrr.csv")


if __name__ == "__main__":
    main()
