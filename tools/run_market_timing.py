"""
P2: Point-in-Time Market Valuation System, Conditional Forward Returns & Dynamic Position Sizing Study.

Strictly causal (expanding/rolling quantiles <= t, T+1 open execution, cash yield 2.0%/yr, 0.10% trade cost).
Evaluates 7 median price buckets + 5 causal valuation quantile buckets over h=20, 60, 120 trading days
with Newey-West CI adjusted for overlapping windows and effective independent sample count disclosure.
Tests 5 fixed/dynamic position-sizing rules without threshold grid-search overfitting.

Usage:
    python tools/run_market_timing.py --cache-dir /tmp/bonds_v2/cache --out dist/bonds/research_timing.json
"""

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from run_full_validation import newey_west_stats, sanitize_json
from run_pro_benchmarks import add_derived_factor_cols, sim_flexible, subperiod_stats
from run_strategy_experiments import compute_metrics, prepare_fast_engine

PRICE_BUCKETS = [
    ("lt_105", "105元以下", -1e9, 105.0),
    ("p105_110", "105–110元", 105.0, 110.0),
    ("p110_115", "110–115元", 110.0, 115.0),
    ("p115_120", "115–120元", 115.0, 120.0),
    ("p120_125", "120–125元", 120.0, 125.0),
    ("p125_130", "125–130元", 125.0, 130.0),
    ("ge_130", "130元以上", 130.0, 1e9),
]

POOL_DEFS = [
    ("all", "全市场", None),
    ("aa_minus", "AA-及以上", {"AA-", "AA", "AA+", "AAA"}),
    ("aa", "AA及以上", {"AA", "AA+", "AAA"}),
    ("aa_plus", "AA+及以上", {"AA+", "AAA"}),
    ("aaa", "仅AAA", {"AAA"}),
]


def fit_100par_prem(conv_vals: np.ndarray, prems: np.ndarray) -> float:
    """Standard domestic CB 100-par fitted conversion premium: prem ~ a + b*(100/conv) + c*(conv/100)."""
    mask = np.isfinite(conv_vals) & np.isfinite(prems) & (conv_vals >= 55.0) & (conv_vals <= 175.0) & (prems > -30.0) & (prems < 250.0)
    if mask.sum() < 12:
        return float(np.nanmedian(prems)) if np.any(np.isfinite(prems)) else np.nan
    cv = conv_vals[mask]
    y = prems[mask]
    X = np.column_stack([np.ones_like(cv), 100.0 / cv, cv / 100.0])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        pred_100 = float(beta[0] + beta[1] + beta[2])
        return max(-10.0, min(150.0, pred_100))
    except Exception:
        return float(np.nanmedian(prems))


def causal_pct(arr: np.ndarray, min_p: int = 60) -> np.ndarray:
    """Strictly causal expanding percentile up to day t (0..100)."""
    out = np.full(len(arr), np.nan, dtype=np.float64)
    for i in range(len(arr)):
        if not np.isfinite(arr[i]):
            continue
        hist = arr[:i + 1]
        hist = hist[np.isfinite(hist)]
        if len(hist) >= min_p:
            out[i] = float(np.mean(hist <= arr[i]) * 100.0)
    return out


def compute_pool_valuation_series(df_pool: pd.DataFrame, ew_daily_net: np.ndarray, cal_dates: list, is_all_pool: bool) -> pd.DataFrame:
    T = len(cal_dates)
    ew_nav = np.ones(T, dtype=np.float64)
    # ew_daily_net corresponds to transitions gt -> t2 (length T-2, aligned at dates[2:T])
    for i in range(len(ew_daily_net)):
        ew_nav[i + 2] = ew_nav[i + 1] * (1.0 + ew_daily_net[i])

    # Market trend (20d ret) & 20d realized vol on EW index
    s_nav = pd.Series(ew_nav)
    s_ret = s_nav.pct_change()
    trend_20 = (s_nav / s_nav.shift(20) - 1.0) * 100.0
    ma60_gap = (s_nav / s_nav.rolling(60, min_periods=20).mean() - 1.0) * 100.0
    vol_20 = s_ret.rolling(20, min_periods=10).std() * math.sqrt(252.0) * 100.0

    by_gt = {int(gt): grp for gt, grp in df_pool.groupby("gt")}
    rows = []
    for gt in range(T):
        grp = by_gt.get(gt)
        if grp is None or len(grp) < 5:
            rows.append({"gt": gt, "date": cal_dates[gt], "n": 0})
            continue
        cr = (
            float((grp["is_st"] | grp["has_def_warn"] | grp["has_spec_rat"]).mean() * 100.0)
            if is_all_pool
            else float((grp["close"] < 95.0).mean() * 100.0)
        )
        rows.append({
            "gt": gt,
            "date": cal_dates[gt],
            "n": int(len(grp)),
            "med_price": round(float(grp["close"].median()), 2),
            "med_prem": round(float(grp["prem"].median()), 2),
            "fitted_prem_100": round(fit_100par_prem(grp["conv_val"].values, grp["prem"].values), 2),
            "med_pb_prem": round(float(grp["pure_bond_prem"].median()), 2),
            "med_ytm": round(float(grp["ytm_pct"].median()), 2),
            "med_iv": round(float(grp["impl_vol"].median()), 2) if grp["impl_vol"].notna().sum() >= 5 else np.nan,
            "med_hv_iv": round(float(grp["hv_minus_iv"].median()), 2) if grp["hv_minus_iv"].notna().sum() >= 5 else np.nan,
            "lsm_underval_pct": round(float((grp["f_four"] > 0).mean() * 100.0), 1),
            "ew_nav": float(ew_nav[gt]),
            "trend_20": round(float(trend_20.iloc[gt]), 2) if pd.notna(trend_20.iloc[gt]) else np.nan,
            "ma60_gap": round(float(ma60_gap.iloc[gt]), 2) if pd.notna(ma60_gap.iloc[gt]) else np.nan,
            "mkt_vol20": round(float(vol_20.iloc[gt]), 2) if pd.notna(vol_20.iloc[gt]) else np.nan,
            "credit_risk_pct": round(cr, 2),
        })

    vdf = pd.DataFrame(rows)
    for col in ["med_price", "med_prem", "fitted_prem_100", "med_pb_prem", "med_ytm", "med_iv", "lsm_underval_pct"]:
        if col in vdf.columns:
            vdf[f"{col}_pct"] = causal_pct(vdf[col].values, min_p=60)
    # Composite causal valuation percentile: average of med_price_pct & fitted_prem_100_pct
    vdf["val_comp_pct"] = 0.5 * vdf["med_price_pct"] + 0.5 * vdf["fitted_prem_100_pct"]
    return vdf


def forward_regime_table(vdf: pd.DataFrame, nav_map: dict, horizons=(20, 60, 120), mode="price_buckets") -> list:
    """Evaluate forward h-day returns & in-horizon drawdown conditioned on signal-day gt valuation state.
    nav_map keys: 'ew', 'lsm', 'dl', 'blend' arrays of length T-2 aligned to gt -> t2.
    Notice signal at gt enters at t1=gt+1 Open and exits at t1+h=gt+1+h Open.
    Because nav[i] in ew/lsm arrays tracks daily compounding through dates[i]=cal_dates[i+2],
    entry equity index is i0 = gt (wait: daily[gt] goes t1 -> t2!). Let's align carefully:
      daily[gt] return is Open(t1=gt+1) -> Open(t2=gt+2).
      Cumulative NAV after k holding days [gt .. gt+h-1] has exact log/prod ratio:
      prod(1 + r[gt : gt+h]).
    """
    T = len(vdf)
    n_ret = len(nav_map["ew"])  # T - 2
    # Precompute daily gross-net NAV levels starting at index 0 before gt=0
    cum_nav = {}
    for k, ret_arr in nav_map.items():
        arr = np.ones(len(ret_arr) + 1, dtype=np.float64)
        arr[1:] = np.cumprod(1.0 + ret_arr)
        cum_nav[k] = arr

    def _stats_for_mask(gt_indices: np.ndarray, h: int) -> dict:
        valid_gts = [int(g) for g in gt_indices if 0 <= int(g) and int(g) + h <= n_ret]
        if len(valid_gts) < 5:
            return {"n_days": len(valid_gts), "n_eff": max(0, round(len(valid_gts) / h, 1)), "insufficient": True}
        ew_rets, mdds, lsm_ex, dl_ex, bl_ex = [], [], [], [], []
        ew_c = cum_nav["ew"]
        lsm_c = cum_nav["lsm"]
        dl_c = cum_nav["dl"]
        bl_c = cum_nav["blend"]
        for g in valid_gts:
            seg = ew_c[g:g + h + 1]
            r_ew = (seg[-1] / seg[0] - 1.0) * 100.0
            pk = np.maximum.accumulate(seg)
            mdd_h = float(np.min(seg / pk - 1.0) * 100.0)
            r_lsm = (lsm_c[g + h] / lsm_c[g] - 1.0) * 100.0
            r_dl = (dl_c[g + h] / dl_c[g] - 1.0) * 100.0
            r_bl = (bl_c[g + h] / bl_c[g] - 1.0) * 100.0
            ew_rets.append(r_ew)
            mdds.append(mdd_h)
            lsm_ex.append(r_lsm - r_ew)
            dl_ex.append(r_dl - r_ew)
            bl_ex.append(r_bl - r_ew)

        arr = np.array(ew_rets, dtype=np.float64)
        nw = newey_west_stats(arr, lags=h)
        n_eff = max(1.0, len(arr) / float(h))
        return {
            "n_days": len(arr),
            "n_eff": round(n_eff, 1),
            "low_sample_warn": n_eff < 4.0,
            "mean": round(nw["mean"], 2),
            "median": round(float(np.median(arr)), 2),
            "win_pct": round(float(np.mean(arr > 0) * 100.0), 1),
            "p10": round(float(np.percentile(arr, 10)), 2),
            "p90": round(float(np.percentile(arr, 90)), 2),
            "ci_low": round(nw["ci_low"], 2),
            "ci_high": round(nw["ci_high"], 2),
            "avg_mdd": round(float(np.mean(mdds)), 2),
            "worst_mdd": round(float(np.min(mdds)), 2),
            "lsm_excess": round(float(np.mean(lsm_ex)), 2),
            "dl_excess": round(float(np.mean(dl_ex)), 2),
            "blend_excess": round(float(np.mean(bl_ex)), 2),
            "insufficient": False,
        }

    buckets = []
    if mode == "price_buckets":
        mp = vdf["med_price"].values
        for bkey, blbl, lo, hi in PRICE_BUCKETS:
            idxs = np.where(np.isfinite(mp) & (mp >= lo) & (mp < hi))[0]
            h_map = {str(h): _stats_for_mask(idxs, h) for h in horizons}
            buckets.append({"key": bkey, "label": blbl, "horizons": h_map})
    else:
        vp = vdf["val_comp_pct"].values
        q_defs = [
            ("Q1", "Q1 极低估分位 (0–20%)", 0.0, 20.0),
            ("Q2", "Q2 偏低估分位 (20–40%)", 20.0, 40.0),
            ("Q3", "Q3 中性分位 (40–60%)", 40.0, 60.0),
            ("Q4", "Q4 偏高估分位 (60–80%)", 60.0, 80.0),
            ("Q5", "Q5 高估区间 (80–100%)", 80.0, 101.0),
        ]
        for qkey, qlbl, lo, hi in q_defs:
            idxs = np.where(np.isfinite(vp) & (vp >= lo) & (vp < hi))[0]
            h_map = {str(h): _stats_for_mask(idxs, h) for h in horizons}
            buckets.append({"key": qkey, "label": qlbl, "horizons": h_map})
    return buckets


def test_position_sizing(strat_net: np.ndarray, dates: list, vdf: pd.DataFrame, ew_net: np.ndarray) -> list:
    """Compare 5 causal position-sizing strategies with 2.0% annual cash yield, 0.10% cost & T+1 lag."""
    T_ret = len(strat_net)
    cash_daily = 0.02 / 252.0
    cost_rate = 0.0010

    # Align signal at day gt (index 0..T_ret-1): signal from vdf row gt controls exposure for return[gt] (t1 -> t2)!
    val_pct = vdf["val_comp_pct"].values[:T_ret]
    ma_gap = vdf["ma60_gap"].values[:T_ret]
    lsm_uv = vdf["lsm_underval_pct"].values[:T_ret]

    # Smooth signals over trailing 5 trading days to avoid noisy daily whipsaw
    s_vp = pd.Series(val_pct).rolling(5, min_periods=1).mean().values
    s_mg = pd.Series(ma_gap).rolling(5, min_periods=1).mean().values
    s_uv = pd.Series(lsm_uv).rolling(5, min_periods=1).mean().values

    # Build exposure series (each step only switches when held >= 5 days for dynamic strategies)
    def build_rule(kind: str) -> np.ndarray:
        exp = np.ones(T_ret, dtype=np.float64)
        if kind == "full_100":
            return exp
        if kind == "fixed_80":
            return np.full(T_ret, 0.80)
        if kind == "fixed_50":
            return np.full(T_ret, 0.50)

        cur_w = 0.80
        days_held = 0
        for i in range(T_ret):
            if kind == "val_tier":
                vp = s_vp[i]
                if not np.isfinite(vp):
                    tgt = 0.80
                elif vp < 20.0:
                    tgt = 1.00
                elif vp < 40.0:
                    tgt = 0.90
                elif vp < 60.0:
                    tgt = 0.80
                elif vp < 80.0:
                    tgt = 0.65
                else:
                    tgt = 0.50
            elif kind == "val_trend":
                vp = s_vp[i]
                mg = s_mg[i]
                if not np.isfinite(vp):
                    tgt = 0.80
                else:
                    base = 1.00 if vp < 20 else 0.90 if vp < 40 else 0.80 if vp < 60 else 0.65 if vp < 80 else 0.50
                    if np.isfinite(mg) and mg < -1.0:
                        tgt = max(0.30, base - 0.20)
                    else:
                        tgt = base
            else:  # lsm_breadth
                uv = s_uv[i]
                if not np.isfinite(uv):
                    tgt = 0.80
                elif uv >= 50.0:
                    tgt = 1.00
                elif uv >= 35.0:
                    tgt = 0.85
                elif uv >= 20.0:
                    tgt = 0.70
                else:
                    tgt = 0.50

            if days_held >= 5 or abs(tgt - cur_w) >= 0.25:
                if abs(tgt - cur_w) > 1e-6:
                    cur_w = tgt
                    days_held = 0
            days_held += 1
            exp[i] = cur_w
        return exp

    rules = [
        ("full_100", "1. 始终满仓 (100%)", "基准满仓控制：全周期不择时"),
        ("fixed_80", "2a. 固定 80% 仓位", "80%策略 + 20%货币现金(年化2%)"),
        ("fixed_50", "2b. 固定 50% 仓位", "50%策略 + 50%货币现金(年化2%)"),
        ("val_tier", "3. 滚动估值分位分档仓位", "因果历史估值分位: <20%满仓 → >80%半仓"),
        ("val_trend", "4. 估值 + 趋势联合仓位", "估值分档基础叠加跌破MA60下调20%"),
        ("lsm_breadth", "5. LSM 低估占比动态仓位", "低估占比>=50%满仓 → <20%半仓"),
    ]

    # Define bull-market benchmark mask: calendar years with ew_net > 10% (2020, 2021, 2025) for upside miss check
    dt_arr = np.array(dates)
    bull_mask = np.isin([d[:4] for d in dates], ["2020", "2021", "2025"])

    out = []
    full_net = strat_net
    for rkey, rname, rdesc in rules:
        w = build_rule(rkey)
        dw = np.abs(np.diff(np.insert(w, 0, w[0])))
        # Net daily portfolio return including cash yield and incremental position turnover friction
        r_dyn = w * strat_net + (1.0 - w) * cash_daily - dw * cost_rate
        nav = np.cumprod(1.0 + r_dyn)
        pk = np.maximum.accumulate(np.insert(nav, 0, 1.0))[1:]
        mdd = float(np.min(nav / pk - 1.0) * 100.0)
        tot = float((nav[-1] - 1.0) * 100.0)
        ann = float((nav[-1] ** (252.0 / len(r_dyn)) - 1.0) * 100.0)
        sharpe = float((np.mean(r_dyn) * 252.0 - 0.02) / (np.std(r_dyn) * math.sqrt(252.0) + 1e-9))
        calmar = float(ann / abs(mdd)) if mdd < -1e-6 else 0.0
        # Upside capture in bull years vs full_100
        bull_dyn = (np.prod(1.0 + r_dyn[bull_mask]) ** (252.0 / max(1, bull_mask.sum())) - 1.0) * 100.0
        bull_full = (np.prod(1.0 + full_net[bull_mask]) ** (252.0 / max(1, bull_mask.sum())) - 1.0) * 100.0
        miss_gap = round(bull_dyn - bull_full, 2)
        subs = subperiod_stats(r_dyn, np.zeros_like(r_dyn), dates, cost=0.0)

        si = list(range(0, len(nav), 10))
        if si[-1] != len(nav) - 1:
            si.append(len(nav) - 1)
        out.append({
            "key": rkey,
            "name": rname,
            "desc": rdesc,
            "avg_exposure_pct": round(float(np.mean(w) * 100.0), 1),
            "pos_switches": int(np.sum(dw > 1e-4)),
            "total_ret": round(tot, 2),
            "ann_ret": round(ann, 2),
            "mdd": round(mdd, 2),
            "sharpe": round(sharpe, 2),
            "calmar": round(calmar, 2),
            "bull_miss_ann_pct": miss_gap,
            "nav_010": [round(float(nav[i]), 4) for i in si],
            "subperiods": subs,
            "last_exposure_pct": round(float(w[-1] * 100.0), 1),
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="/tmp/bonds_v2/cache")
    ap.add_argument("--out", default="dist/bonds/research_timing.json")
    args = ap.parse_args()

    t0 = time.time()
    cache_dir = Path(args.cache_dir)
    with open(cache_dir / "panel_pro.pkl", "rb") as f:
        bundle = pickle.load(f)

    df = add_derived_factor_cols(bundle["df"])
    cal_dates = bundle["cal_dates"]
    eng = prepare_fast_engine(df, bundle["exec_map"], cal_dates)
    df = eng["df"]

    print("1. Computing historical point-in-time market valuation series across 5 rating pools...", flush=True)
    pool_series = {}
    latest_summary = {}
    regimes_by_pool = {}
    sizing_by_pool = {}

    for pkey, plabel, grades in POOL_DEFS:
        if grades is None:
            sub = df
            is_all = True
        else:
            sub = df[df["rating"].isin(grades) & (~df["is_risk_excluded"])]
            is_all = False

        gr_ew, to_ew, dt_ew = sim_flexible(eng, sub, None, freq=5, all_eq=True)
        net_ew = gr_ew - to_ew * 0.0010
        gr_lsm, to_lsm, _ = sim_flexible(eng, sub, "f_four", freq=5, top_frac=0.20)
        net_lsm = gr_lsm - to_lsm * 0.0010
        gr_dl, to_dl, _ = sim_flexible(eng, sub, "neg_dl", freq=5, top_frac=0.20)
        net_dl = gr_dl - to_dl * 0.0010
        expr_bl = lambda g: 0.50 * g["pct_lsm"].values + 0.50 * g["pct_dl"].values
        gr_bl, to_bl, _ = sim_flexible(eng, sub, expr_bl, freq=10, fixed_n=10 if pkey != "aaa" else 5, sell_mult=1.5)
        net_bl = gr_bl - to_bl * 0.0010

        vdf = compute_pool_valuation_series(sub, net_ew, cal_dates, is_all)
        last_row = vdf[vdf["n"] > 0].iloc[-1].to_dict()
        latest_summary[pkey] = {
            "pool_label": plabel,
            "date": last_row["date"],
            "n_bonds": int(last_row["n"]),
            "med_price": last_row.get("med_price"),
            "med_price_pct": round(float(last_row["med_price_pct"]), 1) if pd.notna(last_row.get("med_price_pct")) else None,
            "med_prem": last_row.get("med_prem"),
            "med_prem_pct": round(float(last_row["med_prem_pct"]), 1) if pd.notna(last_row.get("med_prem_pct")) else None,
            "fitted_prem_100": last_row.get("fitted_prem_100"),
            "fitted_prem_100_pct": round(float(last_row["fitted_prem_100_pct"]), 1) if pd.notna(last_row.get("fitted_prem_100_pct")) else None,
            "med_pb_prem": last_row.get("med_pb_prem"),
            "med_pb_prem_pct": round(float(last_row["med_pb_prem_pct"]), 1) if pd.notna(last_row.get("med_pb_prem_pct")) else None,
            "med_ytm": last_row.get("med_ytm"),
            "med_ytm_pct": round(float(last_row["med_ytm_pct"]), 1) if pd.notna(last_row.get("med_ytm_pct")) else None,
            "med_iv": last_row.get("med_iv"),
            "med_hv_iv": last_row.get("med_hv_iv"),
            "lsm_underval_pct": last_row.get("lsm_underval_pct"),
            "lsm_underval_pct_rank": round(float(last_row["lsm_underval_pct_pct"]), 1) if pd.notna(last_row.get("lsm_underval_pct_pct")) else None,
            "val_comp_pct": round(float(last_row["val_comp_pct"]), 1) if pd.notna(last_row.get("val_comp_pct")) else None,
            "trend_20": last_row.get("trend_20"),
            "mkt_vol20": last_row.get("mkt_vol20"),
            "credit_risk_pct": last_row.get("credit_risk_pct"),
        }

        # Downsample history series every 5 trading days (+ final day) for clean charting
        valid_v = vdf[vdf["n"] > 0].reset_index(drop=True)
        si = list(range(0, len(valid_v), 5))
        if si[-1] != len(valid_v) - 1:
            si.append(len(valid_v) - 1)
        sub_s = valid_v.iloc[si]
        pool_series[pkey] = {
            "dates": sub_s["date"].tolist(),
            "med_price": sub_s["med_price"].tolist(),
            "fitted_prem_100": sub_s["fitted_prem_100"].tolist(),
            "med_prem": sub_s["med_prem"].tolist(),
            "med_ytm": sub_s["med_ytm"].tolist(),
            "lsm_underval_pct": sub_s["lsm_underval_pct"].tolist(),
            "val_comp_pct": [round(float(x), 1) if pd.notna(x) else None for x in sub_s["val_comp_pct"]],
        }

        if pkey in ("all", "aa_minus", "aa"):
            nav_map = {"ew": net_ew, "lsm": net_lsm, "dl": net_dl, "blend": net_bl}
            regimes_by_pool[pkey] = {
                "price_buckets": forward_regime_table(vdf, nav_map, (20, 60, 120), mode="price_buckets"),
                "quantile_buckets": forward_regime_table(vdf, nav_map, (20, 60, 120), mode="quantile_buckets"),
            }
            sizing_by_pool[pkey] = {
                "on_balanced_strategy": test_position_sizing(net_bl, dt_ew, vdf, net_ew),
                "on_lsm_strategy": test_position_sizing(net_lsm, dt_ew, vdf, net_ew),
            }

    # NAV dates for position sizing curves
    n_ret = len(cal_dates) - 2
    dates_ret = cal_dates[2:]
    si_d = list(range(0, n_ret, 10))
    if si_d[-1] != n_ret - 1:
        si_d.append(n_ret - 1)

    payload = sanitize_json({
        "meta": {
            "period": f"{cal_dates[0]} -> {cal_dates[-1]}",
            "nav_dates": [dates_ret[i] for i in si_d],
            "cash_yield_ann": "2.00%",
            "trade_cost_oneway": "0.10%",
            "statistical_disclaimer": "历史条件收益受重叠窗口自相关与极端区间独立样本量(N_eff)限制；置信区间采用Newey-West调整(滞后=h)。任何价格阈值或分位数均代表历史统计特征，不构成未来收益预测或安全买点承诺。",
            "fitted_100par_formula": "每日截面拟合 Prem_i = a + b*(100/ConvVal_i) + c*(ConvVal_i/100)，取 ConvVal=100 处拟合溢价率",
        },
        "latest_by_pool": latest_summary,
        "history_series": pool_series,
        "regimes": regimes_by_pool,
        "position_sizing": sizing_by_pool,
    })

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved P2 research_timing.json ({out_p.stat().st_size/1024:.1f} KB) in {time.time()-t0:.1f}s", flush=True)
    for pk in ("all", "aa_minus"):
        lp = latest_summary[pk]
        print(f"  Latest [{pk}]: med_px={lp['med_price']} (pct={lp['med_price_pct']}%), fit100={lp['fitted_prem_100']}% (pct={lp['fitted_prem_100_pct']}%), underval={lp['lsm_underval_pct']}%", flush=True)


if __name__ == "__main__":
    main()
