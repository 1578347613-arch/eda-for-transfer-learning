# dataright.py
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import shutil
import sys
import pandas as pd

# ---------- Feature bounds (A/B 域通用) ----------
FEATURE_BOUNDS = {
    "w1": (1.0, 50.0), "w2": (1.0, 50.0), "w3": (1.0, 50.0), "w4": (1.0, 50.0), "w5": (1.0, 50.0),
    "l1": (0.5, 2.0),  "l2": (0.5, 2.0),  "l3": (0.5, 2.0),  "l4": (0.5, 2.0),  "l5": (0.5, 2.0),
    "cm": (1e-13, 2e-12),  # aka cc
    "rz": (5.0e2, 1.0e5),  # aka cr
    "ibias": (1.0e-5, 1.0e-5),  # 若 two_stage 数据集 Ibias 固定为 10 µA；若文件里是浮动，可改为 (5e-6, 2e-5)
}

ALIASES = {"cc": "cm", "cr": "rz", "ib": "ibias"}

# ---------- Target bounds ----------
# 宽松（物理/定义域）：
TARGET_WIDE = {
    "slewrate_pos": (1e3, 1e9),
    "dc_gain": (1.0, 5e7),
    "ugf": (1.0, 1e11),
    "phase_margin": (0.0, 180.0),  # 下界严格用 (0,180]，实现里会用 >0 and <=180
    "cmrr": (1.0, 1e7),
}

# 严格（基于你的工况推导的“设计预期”范围）：
TARGET_STRICT = {
    "slewrate_pos": (1e6, 2.5e8),   # V/s
    "dc_gain": (10.0, 4e4),         # X
    "ugf": (1e6, 6.5e8),            # Hz
    "phase_margin": (0.0, 180.0),   # deg; 只做定义域过滤
    "cmrr": (1.0, 1e6),             # X (≈ 120 dB)
}

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c.strip() for c in df.columns]
    lower = [c.lower() for c in cols]
    mapping = {}
    for orig, low in zip(cols, lower):
        mapping[orig] = ALIASES.get(low, low)
    return df.rename(columns=mapping)

def _mask_feature(df_feat: pd.DataFrame) -> pd.Series:
    m = pd.Series(True, index=df_feat.index)
    for k, (lo, hi) in FEATURE_BOUNDS.items():
        if k in df_feat.columns:
            col = pd.to_numeric(df_feat[k], errors="coerce")
            ok = col.ge(lo) & col.le(hi)
            m &= ok.fillna(False)
    return m

def _mask_target(df_tgt: pd.DataFrame, bounds: dict) -> pd.Series:
    m = pd.Series(True, index=df_tgt.index)
    for k, (lo, hi) in bounds.items():
        if k in df_tgt.columns:
            col = pd.to_numeric(df_tgt[k], errors="coerce")
            if k == "phase_margin":
                ok = col.gt(lo) & col.le(hi)  # PM 用 (0, 180]
            else:
                ok = col.ge(lo) & col.le(hi)
            m &= ok.fillna(False)
    # 额外剔除 NaN/±inf
    m &= df_tgt.replace([float("inf"), float("-inf")], pd.NA).notna().all(axis=1)
    return m

def _backup_and_write(path: Path, df: pd.DataFrame):
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)
    df.to_csv(path, index=False)

def _clean_pair(feat_csv: Path, tgt_csv: Path, tgt_mode: str):
    df_feat = _norm_cols(pd.read_csv(feat_csv))
    df_tgt  = _norm_cols(pd.read_csv(tgt_csv))
    if len(df_feat) != len(df_tgt):
        n = min(len(df_feat), len(df_tgt))
        print(f"[WARN] Row mismatch {feat_csv.name}({len(df_feat)}) vs {tgt_csv.name}({len(df_tgt)}); trunc -> {n}")
        df_feat = df_feat.iloc[:n].copy()
        df_tgt  = df_tgt.iloc[:n].copy()

    n0 = len(df_feat)
    m_feat = _mask_feature(df_feat)

    if tgt_mode == "off":
        mask = m_feat
        tgt_bounds_used = None
    else:
        bounds = TARGET_STRICT if tgt_mode == "strict" else TARGET_WIDE
        m_tgt = _mask_target(df_tgt, bounds)
        mask = m_feat & m_tgt
        tgt_bounds_used = bounds

    kept = int(mask.sum())
    dropped = n0 - kept

    _backup_and_write(feat_csv, df_feat.loc[mask].reset_index(drop=True))
    _backup_and_write(tgt_csv,  df_tgt.loc[mask].reset_index(drop=True))

    msg = f"[OK] {feat_csv.parent.name}: kept {kept}/{n0}, dropped {dropped}"
    if tgt_bounds_used is not None:
        msg += " (targets=" + ("strict" if tgt_mode == "strict" else "wide") + ")"
    print(msg)
    return dropped

def main():
    ap = argparse.ArgumentParser(description="Clean two_stage_opamp by feature + target bounds.")
    ap.add_argument("--targets", choices=["strict", "wide", "off"], default="strict",
                    help="Target filtering mode. strict=设计预期范围（默认）；wide=物理合法；off=不检查 targets")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    base = root / "data"/"01_train_set" / "two_stage_opamp"
    pairs = [
        (base / "source" / "pretrain_design_features.csv", base / "source" / "pretrain_targets.csv"),
        (base / "target" / "target_design_features.csv",  base / "target" / "target_targets.csv"),
    ]

    total = 0
    for fcsv, tcsv in pairs:
        if fcsv.exists() and tcsv.exists():
            total += _clean_pair(fcsv, tcsv, args.targets)
        else:
            print(f"[SKIP] Missing: {fcsv} or {tcsv}")
    print(f"[DONE] Total dropped rows: {total}")

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
