"""
Precompute compact daily net-return series (`daily_navs.json`) and daily Rank IC series (`daily_ic.json`)
for exact Section IV time-range filtering (全部 / 2023至今 / 近3年 / 近1年 / 今年以来 / 自定义),
daily hover inspection, rolling 12-month excess returns, and dynamic period IC tests.

Stores net returns at 0.1 bps integer precision (`int(round(r * 1e5))`) so browser reconstructs
exact daily NAV, daily Sharpe, daily MDD, Calmar, and calendar-year splits with zero interpolation.

Usage:
    python tools/build_daily_payload.py --cache-dir /tmp/bonds_v2/cache --out-dir dist/bonds
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from run_full_validation import sanitize_json
from run_pro_benchmarks import add_derived_factor_cols, sim_flexible
from run_strategy_experiments import BUFFERS, FREQS, POOLS, WEIGHTS, precompute_daily_candidates, prepare_fast_engine, simulate_fast


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="/tmp/bonds_v2/cache")
    ap.add_argument("--out-dir", default="dist/bonds")
    args = ap.parse_args()

    t0 = time.time()
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(cache_dir / "panel_pro.pkl", "rb") as f:
        bundle = pickle.load(f)

    df = add_derived_factor_cols(bundle["df"])
    eng = prepare_fast_engine(df, bundle["exec_map"], bundle["cal_dates"])
    df = eng["df"]

    # 1. Daily NAV series (0.1 bps integers) for all 450 grid configs (0.10%) + key cost variants + benchmarks
    slices = {}
    for pkey, _, grades in POOLS:
        for rmode, excl in [("raw", False), ("filtered", True)]:
            daily, _ = precompute_daily_candidates(eng, grades, excl)
            slices[(pkey, rmode)] = daily

    dates_ref = eng["cal_dates"][2:]
    series_010 = {}
    eq_010 = {}
    key_cost_variants = {}  # id -> {"0.05%": [...], "0.20%": [...]}

    key_weights_cost = {"100/0", "75/25", "50/50", "25/75", "0/100"}
    for pkey, _, _ in POOLS:
        for rmode in ("raw", "filtered"):
            daily = slices[(pkey, rmode)]
            gr_eq, to_eq, _ = simulate_fast(eng, daily, 1.0, 5, 0.20, all_eq=True)
            eq_010[f"{pkey}_{rmode}"] = [int(round(x * 1e5)) for x in (gr_eq - to_eq * 0.0010)]
            key_cost_variants[f"eq|{pkey}_{rmode}"] = {
                "0.05%": [int(round(x * 1e5)) for x in (gr_eq - to_eq * 0.0005)],
                "0.20%": [int(round(x * 1e5)) for x in (gr_eq - to_eq * 0.0020)],
            }
            for w_lsm, w_lbl in WEIGHTS:
                for freq in FREQS:
                    for bkey, _, sell_frac in BUFFERS:
                        gr, to, _ = simulate_fast(eng, daily, w_lsm, freq, sell_frac)
                        cid = f"{pkey}|{rmode}|{w_lbl}|{freq}D|{bkey}"
                        series_010[cid] = [int(round(x * 1e5)) for x in (gr - to * 0.0010)]
                        if w_lbl in key_weights_cost and bkey in ("none", "buf_20_30"):
                            key_cost_variants[cid] = {
                                "0.05%": [int(round(x * 1e5)) for x in (gr - to * 0.0005)],
                                "0.20%": [int(round(x * 1e5)) for x in (gr - to * 0.0020)],
                            }

    # Include P1 Candidate & Core Peer Daily Series at 0.10% (plus 0.05/0.20 for candidates)
    df_aam_f = df[df["rating"].isin(["AA-", "AA", "AA+", "AAA"]) & (~df["is_risk_excluded"])]
    df_aa_f = df[df["rating"].isin(["AA", "AA+", "AAA"]) & (~df["is_risk_excluded"])]
    expr_75 = lambda g: 0.75 * g["pct_lsm"].values + 0.25 * g["pct_dl"].values
    expr_50 = lambda g: 0.50 * g["pct_lsm"].values + 0.50 * g["pct_dl"].values
    expr_def = lambda g: 0.40 * g["pct_lsm"].values + 0.40 * g["pct_dl"].values + 0.20 * g["pct_pb"].values

    extra_specs = [
        ("cand|enhanced_growth", df_aam_f, expr_75, 5, 20, 1.5, "equal", 0.10),
        ("cand|enhanced_balanced", df_aam_f, expr_50, 10, 20, 1.5, "equal", 0.10),
        ("cand|enhanced_defensive", df_aa_f, expr_50, 10, 20, 1.5, "inv_vol", 0.10),
        ("peer|aam_filt|f_bs_pure", df_aam_f, "f_bs_pure", 5, 0, 1.0, "equal", 1.0),
        ("peer|aam_filt|f_binom", df_aam_f, "f_binom", 5, 0, 1.0, "equal", 1.0),
        ("peer|aam_filt|tri_low", df_aam_f, "tri_low", 5, 0, 1.0, "equal", 1.0),
        ("peer|aam_filt|high_ytm", df_aam_f, "ytm_pct", 5, 0, 1.0, "equal", 1.0),
        ("peer|all_raw|f_bs_pure", df, "f_bs_pure", 5, 0, 1.0, "equal", 1.0),
        ("peer|all_raw|f_binom", df, "f_binom", 5, 0, 1.0, "equal", 1.0),
        ("peer|all_raw|tri_low", df, "tri_low", 5, 0, 1.0, "equal", 1.0),
    ]
    for eid, edf, eexpr, efreq, efn, esm, ewm, ecap in extra_specs:
        gr, to, _ = sim_flexible(eng, edf, eexpr, freq=efreq, fixed_n=efn, sell_mult=esm, weight_mode=ewm, max_single_w=ecap)
        series_010[eid] = [int(round(x * 1e5)) for x in (gr - to * 0.0010)]
        if eid.startswith("cand|"):
            key_cost_variants[eid] = {
                "0.05%": [int(round(x * 1e5)) for x in (gr - to * 0.0005)],
                "0.20%": [int(round(x * 1e5)) for x in (gr - to * 0.0020)],
            }

    navs_payload = sanitize_json({
        "dates": dates_ref,
        "scale": 1e-5,
        "benchmarks_010": eq_010,
        "series_010": series_010,
        "cost_variants": key_cost_variants,
    })
    nav_path = out_dir / "daily_navs.json"
    nav_path.write_text(json.dumps(navs_payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print(f"Saved {nav_path} ({nav_path.stat().st_size/1024:.1f} KB) in {time.time()-t0:.1f}s", flush=True)

    # 2. Daily Spearman Rank IC series for dynamic period recalculation across all 5 rating pools
    daily_ic = {}
    for pkey, _, grades in POOLS:
        sub = df if grades is None else df[df["rating"].isin(grades)]
        recs = []
        for gt, grp in sub.groupby("gt"):
            dt_str = str(grp["date"].iloc[0])
            row = [dt_str, int(len(grp))]
            for h in (5, 10, 20):
                s = grp[["f_four", f"ret_{h}"]].dropna()
                if len(s) >= 12 and s["f_four"].nunique() >= 4:
                    c = s["f_four"].rank().corr(s[f"ret_{h}"].rank())
                    row.append(round(float(c), 4) if pd.notna(c) else None)
                else:
                    row.append(None)
            recs.append(row)
        daily_ic[pkey] = recs

    ic_payload = sanitize_json({"cols": ["date", "n", "ic_5", "ic_10", "ic_20"], "pools": daily_ic})
    ic_path = out_dir / "daily_ic.json"
    ic_path.write_text(json.dumps(ic_payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print(f"Saved {ic_path} ({ic_path.stat().st_size/1024:.1f} KB)", flush=True)


if __name__ == "__main__":
    main()
