"""
Full-Market Point-in-Time Validation Engine (2020–2026) for Frozen Four-Region LSM.

Independent Audit Guarantees:
1. Frozen paper parameters (p_reset=0.80, B=108.0, r=0.025, call_ratio=1.30, put_ratio=0.70).
2. Global SSE/SZSE trading calendar alignment: forward returns strictly require T+1+h <= T_end
   (zero right-edge truncation -> valid IC days strictly decrease as h grows).
3. Tradable T+1 Open execution: suspended bonds cannot be bought/sold and remain locked;
   genuinely delisted/redeemed/defaulted bonds realize terminal close + exit cost (plus 50% default stress test).
4. Strictly causal inputs: historical conversion price C_{i,t}, contractual maturity from issuance,
   trailing volatility, and NO ex-post credit ratings in Fama-MacBeth regressions.
5. Strict RFC-8259 JSON serialization (allow_nan=False; all NaN/Inf -> null).

Usage:
    python tools/run_full_validation.py [--cache-dir .cache_bonds] [--out dist/bonds/research_validation.json] [--pre-audit path/to/pre_audit.json] [--analyze-only]
"""

import argparse
import json
import math
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cb_pricing_model import price_ccb_four_region_lsm


def sanitize_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def newey_west_stats(series: np.ndarray, lags: int) -> dict:
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    T = len(x)
    if T < 5:
        return {"mean": 0.0, "std": 0.0, "se": 0.0, "t_stat": 0.0, "p_val": 1.0, "ci_low": 0.0, "ci_high": 0.0, "n": T}
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if T > 1 else 0.0
    e = x - mean
    L = max(1, min(lags, T - 2))
    nw_var = float(np.dot(e, e) / T)
    for lag in range(1, L + 1):
        w = 1.0 - lag / (L + 1.0)
        cov = float(np.dot(e[lag:], e[:-lag]) / T)
        nw_var += 2.0 * w * cov
    nw_var = max(1e-16, nw_var)
    se = math.sqrt(nw_var / T)
    t_stat = mean / se if se > 0 else 0.0
    p_val = math.erfc(abs(t_stat) / math.sqrt(2.0))
    return {
        "mean": mean,
        "std": std,
        "se": se,
        "t_stat": t_stat,
        "p_val": p_val,
        "ci_low": mean - 1.96 * se,
        "ci_high": mean + 1.96 * se,
        "n": T,
    }


def build_global_calendar(cb_files: list) -> list:
    date_counts = {}
    for p in cb_files:
        with open(p, "rb") as f:
            obj = pickle.load(f)
        b_hist = obj["b_hist"]
        for d in b_hist["date"].drop_duplicates():
            ds = pd.Timestamp(d).strftime("%Y-%m-%d")
            if "2020-01-02" <= ds <= "2026-12-31":
                date_counts[ds] = date_counts.get(ds, 0) + 1
    cal = sorted(d for d, cnt in date_counts.items() if cnt >= 20)
    return cal


def process_single_bond(args):
    cb_file_str, stk_dir_str, cal_dates, cal_map = args
    with open(cb_file_str, "rb") as f:
        obj = pickle.load(f)
    meta = obj["meta"]
    b_hist = obj["b_hist"].drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    va = obj["va"].drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    stk_path = Path(stk_dir_str) / f"{meta['stock_code']}.pkl"
    if not stk_path.exists():
        return None
    with open(stk_path, "rb") as f:
        s_hist = pickle.load(f).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    if len(b_hist) < 35 or len(s_hist) < 35:
        return None

    s_df = s_hist[["date", "close"]].rename(columns={"close": "s_close"})
    va_df = va[["date", "纯债价值", "转股价值"]].rename(columns={"纯债价值": "pure_bond", "转股价值": "em_conv"})

    # Keep exact bond trading rows first to compute causal indicators & conversion prices
    m = pd.merge(b_hist, s_df, on="date", how="left")
    m = pd.merge(m, va_df, on="date", how="left")
    m["s_close"] = pd.to_numeric(m["s_close"], errors="coerce").ffill(limit=5)
    m = m.dropna(subset=["open", "close", "s_close"]).reset_index(drop=True)
    if len(m) < 35:
        return None

    # Strict causal conversion price C_t
    init_c = float(meta["init_conv"]) if meta.get("init_conv") else 10.0
    em_conv = pd.to_numeric(m["em_conv"], errors="coerce")
    c_raw = np.where((em_conv > 1.0) & (m["s_close"] > 0), 100.0 * m["s_close"] / em_conv, np.nan)
    c_series = pd.Series(c_raw).where(lambda s: (s >= 0.15 * init_c) & (s <= 3.0 * init_c))
    c_arr = np.zeros(len(m))
    cur_c = init_c
    for k in range(len(m)):
        v = c_series.iloc[k]
        if pd.notna(v) and v > 0:
            cur_c = round(float(v), 2)
        c_arr[k] = cur_c

    # Strict issuance-known contractual maturity
    origin_dt = pd.to_datetime(meta["value_date"] or meta["listing_date"], errors="coerce")
    if pd.isna(origin_dt):
        origin_dt = m["date"].iloc[0]
    contract_yrs = int(round(float(meta.get("contract_years") or 6.0)))
    contract_mat_dt = origin_dt + pd.DateOffset(years=contract_yrs)

    s_closes = m["s_close"].astype(float)
    log_r = np.log(s_closes / s_closes.shift(1))
    log_r_clean = log_r.where(log_r.abs() <= 0.28)
    vols = (log_r_clean.rolling(60, min_periods=15).std() * np.sqrt(252.0)).fillna(0.28).clip(0.15, 0.70).values
    mom20 = ((s_closes / s_closes.shift(20) - 1.0) * 100.0).fillna(0.0).clip(-60.0, 150.0).values

    # Coupon schedule aligned to ex-dates
    coupons = meta.get("coupons") or [0.5] * contract_yrs
    coupon_vec = np.zeros(len(m))
    dates_pd = pd.to_datetime(m["date"])
    for yr_k in range(1, min(contract_yrs, len(coupons)) + 1):
        ex_target = origin_dt + pd.DateOffset(years=yr_k)
        hits = np.where(dates_pd >= ex_target)[0]
        if len(hits) > 0 and hits[0] < len(m):
            coupon_vec[hits[0]] += float(coupons[yr_k - 1])

    # Run frozen LSM + 2 ablation models cleanly on 2020+ dates
    n_rows = len(m)
    v_four = np.full(n_rows, np.nan)
    v_single = np.full(n_rows, np.nan)
    v_noreset = np.full(n_rows, np.nan)
    cutoff_dt = pd.Timestamp("2019-12-20")

    for k in range(25, n_rows):
        dt = dates_pd.iloc[k]
        if dt < cutoff_dt:
            continue
        if float(m["volume"].iloc[k] or 0) <= 0 or float(m["close"].iloc[k]) <= 1.0:
            continue
        T_rem = max(0.08, (contract_mat_dt - dt).days / 365.25)
        S0 = float(s_closes.iloc[k])
        C0 = float(c_arr[k])
        sig = float(vols[k])

        v_four[k] = price_ccb_four_region_lsm(
            S0=S0, C0=C0, FV=100.0, maturity_years=T_rem,
            r=0.025, q=0.0, sigma=sig, p_reset=0.80, B=108.0,
            n_steps=12, M=400, seed=42, segmented=True
        )
        v_single[k] = price_ccb_four_region_lsm(
            S0=S0, C0=C0, FV=100.0, maturity_years=T_rem,
            r=0.025, q=0.0, sigma=sig, p_reset=0.80, B=108.0,
            n_steps=12, M=400, seed=42, segmented=False
        )
        v_noreset[k] = price_ccb_four_region_lsm(
            S0=S0, C0=C0, FV=100.0, maturity_years=T_rem,
            r=0.025, q=0.0, sigma=sig, p_reset=0.0, B=108.0,
            n_steps=12, M=400, seed=42, segmented=True
        )

    # Map bond series onto EXACT Global Market Calendar (T_cal days)
    T_cal = len(cal_dates)
    g_open = np.full(T_cal, np.nan)
    g_close = np.full(T_cal, np.nan)
    g_vol = np.zeros(T_cal)
    g_coup = np.zeros(T_cal)
    g_vf = np.full(T_cal, np.nan)
    g_vs = np.full(T_cal, np.nan)
    g_vn = np.full(T_cal, np.nan)
    g_prem = np.full(T_cal, np.nan)
    g_dl = np.full(T_cal, np.nan)
    g_sig = np.full(T_cal, np.nan)
    g_mat = np.full(T_cal, np.nan)
    g_mom = np.full(T_cal, np.nan)

    conv_vals = (100.0 / c_arr) * s_closes.values
    prems = (m["close"].values - conv_vals) / np.maximum(0.01, conv_vals) * 100.0
    mats = np.maximum(0.08, (contract_mat_dt - dates_pd).dt.days.values / 365.25)

    date_strs = dates_pd.dt.strftime("%Y-%m-%d").values
    first_active_gt = None
    last_active_gt = None

    for k in range(n_rows):
        ds = date_strs[k]
        gt = cal_map.get(ds)
        if gt is None:
            continue
        o_v = float(m["open"].iloc[k])
        c_v = float(m["close"].iloc[k])
        vol_v = float(m["volume"].iloc[k] or 0)
        g_open[gt] = o_v
        g_close[gt] = c_v
        g_vol[gt] = vol_v
        g_coup[gt] = float(coupon_vec[k])
        g_vf[gt] = v_four[k]
        g_vs[gt] = v_single[k]
        g_vn[gt] = v_noreset[k]
        g_prem[gt] = prems[k]
        g_dl[gt] = c_v + prems[k]
        g_sig[gt] = vols[k] * 100.0
        g_mat[gt] = mats[k]
        g_mom[gt] = mom20[k]
        if c_v > 1.0 and vol_v > 0:
            if first_active_gt is None:
                first_active_gt = gt
            last_active_gt = gt

    if last_active_gt is None or first_active_gt is None:
        return None

    # Genuine delist / redemption / default occurs if trading ceased well before dataset end
    is_delisted = last_active_gt < (T_cal - 8)
    last_close_val = float(g_close[last_active_gt])
    is_distressed_delist = is_delisted and (last_close_val < 90.0)
    delist_payoff_base = last_close_val
    delist_payoff_stress = 0.50 * last_close_val if is_distressed_delist else last_close_val

    cum_coup = np.cumsum(g_coup)
    tradable = (g_open > 1.0) & (g_close > 1.0) & (g_vol > 0)

    # Compute STRICT calendar-aligned forward returns & tradability flags
    # Key audit fix: If gt + 1 + h >= T_cal -> MUST BE NaN (zero truncation!)
    rows_out = []
    for gt in range(T_cal - 1):
        if np.isnan(g_vf[gt]) or not tradable[gt]:
            continue
        # Must actually trade at exact gt + 1 on global calendar to enter at T+1 Open
        if not tradable[gt + 1]:
            continue
        p_in = float(g_open[gt + 1])

        row = {
            "gt": gt,
            "date": cal_dates[gt],
            "ticker": meta["ticker"],
            "open": float(g_open[gt]),
            "close": float(g_close[gt]),
            "open_next": p_in,
            "v_four": float(g_vf[gt]),
            "v_single": float(g_vs[gt]),
            "v_noreset": float(g_vn[gt]),
            "prem": float(g_prem[gt]),
            "double_low": float(g_dl[gt]),
            "sigma": float(g_sig[gt]),
            "maturity": float(g_mat[gt]),
            "mom20": float(g_mom[gt]),
            "is_delisted": is_delisted,
            "is_distressed_delist": is_distressed_delist,
            "last_gt": last_active_gt,
        }

        for h in (1, 2, 3, 5, 10, 15, 20):
            target_exit = gt + 1 + h
            if target_exit >= T_cal:
                # Beyond dataset boundary -> NaN, NEVER truncated!
                row[f"ret_{h}"] = np.nan
                if h == 20:
                    row["ret_20_stress"] = np.nan
                continue

            if is_delisted and target_exit > last_active_gt:
                c_earned = float(cum_coup[last_active_gt] - cum_coup[gt + 1]) if last_active_gt > gt + 1 else 0.0
                row[f"ret_{h}"] = (delist_payoff_base + c_earned - p_in) / p_in * 100.0
                if h == 20:
                    row["ret_20_stress"] = (delist_payoff_stress + c_earned - p_in) / p_in * 100.0
            else:
                # Find first tradable open >= target_exit (in case of brief suspension)
                exit_gt = None
                for probe in range(target_exit, min(last_active_gt + 1, target_exit + 10)):
                    if tradable[probe]:
                        exit_gt = probe
                        break
                if exit_gt is not None:
                    p_out = float(g_open[exit_gt])
                    c_earned = float(cum_coup[exit_gt] - cum_coup[gt + 1]) if exit_gt > gt + 1 else 0.0
                    row[f"ret_{h}"] = (p_out + c_earned - p_in) / p_in * 100.0
                    if h == 20:
                        row["ret_20_stress"] = row[f"ret_{h}"]
                elif is_delisted:
                    c_earned = float(cum_coup[last_active_gt] - cum_coup[gt + 1]) if last_active_gt > gt + 1 else 0.0
                    row[f"ret_{h}"] = (delist_payoff_base + c_earned - p_in) / p_in * 100.0
                    if h == 20:
                        row["ret_20_stress"] = (delist_payoff_stress + c_earned - p_in) / p_in * 100.0
                else:
                    row[f"ret_{h}"] = np.nan
                    if h == 20:
                        row["ret_20_stress"] = np.nan

        rows_out.append(row)

    # Also build bond execution arrays on global calendar for suspension-locked portfolio simulation
    exec_info = {
        "ticker": meta["ticker"],
        "tradable": tradable,
        "g_open": g_open,
        "g_close": g_close,
        "g_coup": g_coup,
        "last_gt": last_active_gt,
        "is_delisted": is_delisted,
        "delist_payoff_base": delist_payoff_base,
        "delist_payoff_stress": delist_payoff_stress,
    }
    return (pd.DataFrame(rows_out), exec_info) if rows_out else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=".cache_bonds")
    parser.add_argument("--out", default="dist/bonds/research_validation.json")
    parser.add_argument("--pre-audit", default=None)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    cache_base = Path(args.cache_dir)
    cb_dir = cache_base / "cbs"
    stk_dir = cache_base / "stocks"
    panel_path = cache_base / "panel_audit.pkl"

    if args.analyze_only and panel_path.exists():
        with open(panel_path, "rb") as f:
            bundle = pickle.load(f)
        analyze_panel(bundle, Path(args.out), args.pre_audit)
        return

    cb_files = sorted(str(p) for p in cb_dir.glob("*.pkl"))
    cal_dates = build_global_calendar(cb_files)
    cal_map = {d: i for i, d in enumerate(cal_dates)}
    print(f"Global market calendar: {len(cal_dates)} trading days ({cal_dates[0]} -> {cal_dates[-1]})", flush=True)

    t0 = time.time()
    frames = []
    exec_map = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(process_single_bond, (p, str(stk_dir), cal_dates, cal_map)) for p in cb_files]
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            if res is not None:
                df_b, einfo = res
                frames.append(df_b)
                exec_map[einfo["ticker"]] = einfo
            if i % 150 == 0 or i == len(futs):
                print(f"  Audited {i}/{len(cb_files)} CBs ({len(frames)} valid) in {time.time()-t0:.1f}s", flush=True)

    panel = pd.concat(frames, ignore_index=True)
    panel["f_four"] = (panel["v_four"] - panel["close"]) / panel["close"] * 100.0
    panel["f_single"] = (panel["v_single"] - panel["close"]) / panel["close"] * 100.0
    panel["f_noreset"] = (panel["v_noreset"] - panel["close"]) / panel["close"] * 100.0

    bundle = {"panel": panel, "exec_map": exec_map, "cal_dates": cal_dates}
    with open(panel_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Audit panel saved: {len(panel)} bond-days across {panel['ticker'].nunique()} CBs", flush=True)

    analyze_panel(bundle, Path(args.out), args.pre_audit)


def analyze_panel(bundle: dict, out_path: Path, pre_audit_path=None):
    panel = bundle["panel"]
    exec_map = bundle["exec_map"]
    cal_dates = bundle["cal_dates"]
    T_cal = len(cal_dates)
    by_gt = {gt: grp for gt, grp in panel.groupby("gt")}
    all_gts = sorted(by_gt.keys())

    # --- STAGE 2: Strict Calendar-Aligned IC & Significance ---
    horizons_ic = [1, 2, 3, 5, 10, 15, 20]
    daily_ic_records = {h: [] for h in horizons_ic}
    for gt in all_gts:
        g = by_gt[gt]
        ds = cal_dates[gt]
        for h in horizons_ic:
            if gt + 1 + h >= T_cal:
                continue
            col = f"ret_{h}"
            sub = g[["f_four", col]].dropna()
            if len(sub) >= 25 and sub["f_four"].nunique() > 5 and sub[col].nunique() > 5:
                ic = float(sub["f_four"].rank().corr(sub[col].rank()))
                if math.isfinite(ic):
                    daily_ic_records[h].append((ds, ic, len(sub)))

    def summarize_ic(recs, h):
        if not recs:
            return None
        arr = np.array([r[1] for r in recs])
        nw = newey_west_stats(arr, lags=h)
        return {
            "horizon": h,
            "days": len(arr),
            "avg_n": round(float(np.mean([r[2] for r in recs])), 1),
            "mean_ic": round(nw["mean"], 4),
            "std_ic": round(nw["std"], 4),
            "icir": round(nw["mean"] / nw["std"], 3) if nw["std"] > 0 else 0.0,
            "pos_pct": round(float(np.mean(arr > 0) * 100.0), 1),
            "t_nw": round(nw["t_stat"], 2),
            "p_val": round(nw["p_val"], 5),
            "ci95": [round(nw["ci_low"], 4), round(nw["ci_high"], 4)],
        }

    ic_summary = {str(h): summarize_ic(daily_ic_records[h], h) for h in (1, 5, 10, 20)}
    ic_decay = [{"h": h, **summarize_ic(daily_ic_records[h], h)} for h in horizons_ic]

    rolling_ic = {}
    for h in (5, 10, 20):
        recs = daily_ic_records[h]
        s = pd.Series([r[1] for r in recs], index=[r[0] for r in recs]).rolling(60).mean().dropna()
        rolling_ic[str(h)] = [[dt, round(float(val), 4)] for i, (dt, val) in enumerate(s.items()) if i % 5 == 0 or i == len(s) - 1]

    periods = [("2020–2022", "2020-01-01", "2022-12-31"), ("2023–2024", "2023-01-01", "2024-12-31"), ("2025–2026", "2025-01-01", "2026-12-31")]
    subperiod_ic = {}
    for pname, s_dt, e_dt in periods:
        subperiod_ic[pname] = {}
        for h in (1, 5, 10, 20):
            sub_recs = [r for r in daily_ic_records[h] if s_dt <= r[0] <= e_dt]
            subperiod_ic[pname][str(h)] = summarize_ic(sub_recs, h)

    # --- STAGE 3: Daily Quintiles (G1..G5) + 50% Default Stress Check ---
    quintiles = {}
    for h in (1, 5, 10, 20):
        col = f"ret_{h}"
        q_ex, q_raw, sp_list = {q: [] for q in range(5)}, {q: [] for q in range(5)}, []
        for gt in all_gts:
            if gt + 1 + h >= T_cal:
                continue
            g = by_gt[gt][["f_four", col]].dropna()
            if len(g) < 25:
                continue
            g = g.sort_values("f_four", ascending=True)
            mkt_mean = float(g[col].mean())
            bins = np.array_split(g[col].values, 5)
            for qi in range(5):
                qm = float(np.mean(bins[qi]))
                q_raw[qi].append(qm)
                q_ex[qi].append(qm - mkt_mean)
            sp_list.append(float(np.mean(bins[4]) - np.mean(bins[0])))
        nw_sp = newey_west_stats(np.array(sp_list), lags=h)
        quintiles[str(h)] = {
            "days": len(sp_list),
            "groups": [
                {"group": f"G{qi+1}", "label": ["G1 高估", "G2 偏高", "G3 中性", "G4 偏低", "G5 低估"][qi],
                 "excess_ret": round(float(np.mean(q_ex[qi])), 3), "raw_ret": round(float(np.mean(q_raw[qi])), 3)}
                for qi in range(5)
            ],
            "g5_minus_g1": round(nw_sp["mean"], 3),
            "t_nw": round(nw_sp["t_stat"], 2),
            "p_val": round(nw_sp["p_val"], 5),
        }

    # Audit-Grade Portfolio Simulator (True T+1 Open, Suspension-Locked, Real Delist Liquidation + Default Loss)
    def run_portfolio_sim(score_col: str, ascending: bool, freq: int = 5, start_gt: int = 0, end_gt: int = None,
                          top_frac: float = 0.20, all_eq: bool = False, q_slice=None, stress_default: bool = False, random_rng=None, rand_keep_frac=0.85):
        if end_gt is None:
            end_gt = T_cal - 2
        # Portfolio holds weights valued at mark-to-market; between t1=gt+1 open and t2=gt+2 open
        holdings = {}  # ticker -> {"val": float, "last_px": float}
        cash = 1.0
        daily_gross_ret = []
        daily_two_way_to = []
        used_dates = []
        rebal_step = 0

        for gt in range(start_gt, min(end_gt, T_cal - 2)):
            t1 = gt + 1  # execution open day
            t2 = gt + 2  # next day open

            # Step A: Classify current holdings at t1 open (tradable vs suspended-locked)
            tradable_keys = []
            locked_keys = []
            for tk, pos in list(holdings.items()):
                info = exec_map[tk]
                if info["tradable"][t1]:
                    tradable_keys.append(tk)
                else:
                    locked_keys.append(tk)

            nav_pre_rebal = cash + sum(holdings[k]["val"] for k in holdings)
            locked_val = sum(holdings[k]["val"] for k in locked_keys)
            free_capital = max(0.0, nav_pre_rebal - locked_val)

            # Step B: Rebalance ONLY tradable capital at t1 open (cannot buy bonds on their very last exchange day)
            two_way_to = 0.0
            should_rebal = (rebal_step % freq == 0) or (len(holdings) == 0 and free_capital > 0.5)
            rebal_step += 1

            g = by_gt.get(gt)
            if should_rebal and g is not None and free_capital > 1e-8:
                # Filter out bonds whose final exchange day is t1 or earlier (cannot enter at t1 into immediate delist)
                g_buy = g[~(g["is_delisted"] & (g["last_gt"] <= t1))]
                elig = g_buy if all_eq or random_rng is not None else g_buy.dropna(subset=[score_col])
                if len(elig) >= 15:
                    if q_slice is not None:
                        s_tks = elig.sort_values(score_col, ascending=True)["ticker"].values
                        picks = list(np.array_split(s_tks, 5)[q_slice])
                    elif random_rng is not None:
                        tks_arr = elig["ticker"].values
                        k_p = max(5, int(round(len(tks_arr) * top_frac)))
                        tks_set = set(tks_arr)
                        kept = [k for k in tradable_keys if k in tks_set]
                        n_keep = min(len(kept), int(round(k_p * rand_keep_frac)))
                        keep_sub = list(random_rng.choice(kept, size=n_keep, replace=False)) if n_keep > 0 else []
                        avail = [k for k in tks_arr if k not in set(keep_sub)]
                        need = max(0, k_p - len(keep_sub))
                        add_sub = list(random_rng.choice(avail, size=min(len(avail), need), replace=False)) if need > 0 else []
                        picks = keep_sub + add_sub
                    elif all_eq:
                        picks = list(elig["ticker"].values)
                    else:
                        k_p = max(5, int(round(len(elig) * top_frac)))
                        picks = list(elig.sort_values(score_col, ascending=ascending)["ticker"].iloc[:k_p])

                    target_w = {tk: (free_capital / nav_pre_rebal) / len(picks) for tk in picks} if picks else {}
                    old_w = {tk: holdings[tk]["val"] / nav_pre_rebal for tk in tradable_keys}
                    all_u = set(target_w.keys()) | set(old_w.keys())
                    two_way_to += sum(abs(target_w.get(k, 0.0) - old_w.get(k, 0.0)) for k in all_u)

                    new_holdings = {k: holdings[k] for k in locked_keys}
                    for tk, tw in target_w.items():
                        new_holdings[tk] = {"val": tw * nav_pre_rebal, "last_px": float(exec_map[tk]["g_open"][t1])}
                    holdings = new_holdings
                    cash = 0.0

            # Step C: Mark portfolio from t1 open -> t2 open; force-liquidate bonds whose last_gt == t1
            nav_after = cash
            for tk, pos in list(holdings.items()):
                info = exec_map[tk]
                if info["is_delisted"] and t1 >= info["last_gt"]:
                    # Bond reached final exchange trading day at t1 -> liquidate at delist close / stress recovery
                    payoff_px = info["delist_payoff_stress"] if stress_default else info["delist_payoff_base"]
                    c_day = float(info["g_coup"][info["last_gt"]])
                    exit_val = pos["val"] * ((payoff_px + c_day) / pos["last_px"])
                    two_way_to += exit_val / max(1e-9, nav_pre_rebal)
                    cash += exit_val
                    nav_after += exit_val
                    del holdings[tk]
                elif info["tradable"][t2]:
                    p_t2 = float(info["g_open"][t2])
                    c_day = float(info["g_coup"][t2])
                    pos["val"] = pos["val"] * ((p_t2 + c_day) / pos["last_px"])
                    pos["last_px"] = p_t2
                    nav_after += pos["val"]
                else:
                    # Suspended on t2 -> carry frozen value into next day
                    nav_after += pos["val"]

            r_gross = (nav_after / nav_pre_rebal - 1.0) if nav_pre_rebal > 0 else 0.0
            daily_gross_ret.append(r_gross)
            daily_two_way_to.append(two_way_to)
            used_dates.append(cal_dates[t2])

            # Normalize NAV back to 1.0 for clean multiplicative compounding
            if nav_after > 0:
                scale = 1.0 / nav_after
                cash *= scale
                for pos in holdings.values():
                    pos["val"] *= scale

        return np.array(daily_gross_ret), np.array(daily_two_way_to), used_dates

    # Five-Group weekly NAV curves (0.10% cost)
    q_navs, q_dates_raw = {}, None
    for qi in range(5):
        gr, to, u_dt = run_portfolio_sim("f_four", False, freq=5, q_slice=qi)
        nav = np.cumprod(1.0 + (gr - to * 0.001))
        q_navs[f"G{qi+1}"] = nav
        q_dates_raw = u_dt
    samp_q = list(range(0, len(q_dates_raw), 5))
    if samp_q[-1] != len(q_dates_raw) - 1:
        samp_q.append(len(q_dates_raw) - 1)
    q_curves = {k: [round(float(v[i]), 4) for i in samp_q] for k, v in q_navs.items()}
    q_curve_dates = [q_dates_raw[i] for i in samp_q]

    def metrics_from_arrays(g_arr, to_arr, u_dt):
        out = {
            "days": len(g_arr),
            "ann_to": round(float(np.sum(to_arr) * 0.5 * (252.0 / max(1, len(g_arr))) * 100.0), 1),
            "costs": {},
        }
        for c in (0.0005, 0.001, 0.002):
            net = g_arr - to_arr * c
            nav = np.cumprod(1.0 + net)
            peak = np.maximum.accumulate(np.insert(nav, 0, 1.0))[1:]
            mdd = float(np.min(nav / peak - 1.0) * 100.0)
            tot_net = float((nav[-1] - 1.0) * 100.0)
            ann_net = float((nav[-1] ** (252.0 / max(1, len(net))) - 1.0) * 100.0)
            sh = float((np.mean(net) * 252.0 - 0.02) / (np.std(net) * np.sqrt(252.0) + 1e-9))
            out["costs"][f"{c*100:.2f}%"] = {
                "total_ret": round(tot_net, 2),
                "ann_ret": round(ann_net, 2),
                "mdd": round(mdd, 2),
                "sharpe": round(sh, 2),
            }
        net_std = g_arr - to_arr * 0.001
        nav_std = np.cumprod(1.0 + net_std)
        si = list(range(0, len(nav_std), 5))
        if si[-1] != len(nav_std) - 1:
            si.append(len(nav_std) - 1)
        out["nav_dates"] = [u_dt[i] for i in si]
        out["nav_010"] = [round(float(nav_std[i]), 4) for i in si]
        return out

    strat_defs = [
        ("four_region_lsm", "Four-Region LSM (冻结原模型)", "f_four", False, False),
        ("mkt_ew", "全市场等权基准", "f_four", False, True),
        ("double_low", "传统双低 (价格+溢价)", "double_low", True, False),
        ("low_price", "单纯低价策略", "close", True, False),
        ("low_prem", "低转股溢价率策略", "prem", True, False),
        ("stock_mom", "正股 20 日动量", "mom20", False, False),
        ("single_region_lsm", "传统单一区域 LSM (去分段)", "f_single", False, False),
        ("no_reset_lsm", "无下修机制 LSM (p=0)", "f_noreset", False, False),
    ]

    strategies_res = {}
    for sk, sname, col, asc, is_all in strat_defs:
        gr, to, u_dt = run_portfolio_sim(col, asc, freq=5, all_eq=is_all)
        strategies_res[sk] = {"name": sname, **metrics_from_arrays(gr, to, u_dt)}

    # Stress check: 50% default recovery on distressed delisted bonds (last_close < 90)
    gr_st, to_st, _ = run_portfolio_sim("f_four", False, freq=5, stress_default=True)
    stress_four = metrics_from_arrays(gr_st, to_st, q_dates_raw)["costs"]["0.10%"]

    # Random MC control (60 independent runs, matched ~750% turnover)
    rng = np.random.default_rng(2026)
    rmc_tot, rmc_ann, rmc_sh = [], [], []
    for _ in range(60):
        gr, to, _ = run_portfolio_sim("f_four", False, freq=5, random_rng=rng, rand_keep_frac=0.85)
        net = gr - to * 0.001
        nav = np.cumprod(1.0 + net)
        rmc_tot.append((nav[-1] - 1.0) * 100.0)
        rmc_ann.append((nav[-1] ** (252.0 / len(net)) - 1.0) * 100.0)
        rmc_sh.append((np.mean(net) * 252.0 - 0.02) / (np.std(net) * np.sqrt(252.0) + 1e-9))

    random_mc = {
        "runs": 60,
        "total_ret_mean": round(float(np.mean(rmc_tot)), 2),
        "total_ret_ci95": [round(float(np.percentile(rmc_tot, 2.5)), 2), round(float(np.percentile(rmc_tot, 97.5)), 2)],
        "ann_ret_mean": round(float(np.mean(rmc_ann)), 2),
        "sharpe_mean": round(float(np.mean(rmc_sh)), 2),
    }

    # Sub-period strategy splits
    subperiod_strats = {}
    for pname, s_dt, e_dt in periods:
        s_gts = [gt for gt in all_gts if s_dt <= cal_dates[gt] <= e_dt]
        sg, eg = s_gts[0], s_gts[-1]
        subperiod_strats[pname] = {}
        for sk, col, asc, is_all in [("four_region_lsm", "f_four", False, False), ("double_low", "double_low", True, False), ("mkt_ew", "f_four", False, True)]:
            gr, to, u_dt = run_portfolio_sim(col, asc, freq=5, start_gt=sg, end_gt=eg, all_eq=is_all)
            subperiod_strats[pname][sk] = metrics_from_arrays(gr, to, u_dt)["costs"]["0.10%"]

    freq_robust = {}
    for f in (1, 5, 10, 20):
        gr, to, u_dt = run_portfolio_sim("f_four", False, freq=f)
        freq_robust[f"{f}D"] = metrics_from_arrays(gr, to, u_dt)["costs"]

    # --- STAGE 4B: Causal-Only Fama-MacBeth (Ex-post Rating REMOVED!) ---
    ctrl_cols = ["f_four", "close", "prem", "sigma", "maturity", "mom20"]
    ctrl_labels = {
        "f_four": "F_LSM (Four-Region 估值因子)",
        "close": "转债价格 (控制低价效应)",
        "prem": "转股溢价率 (控制溢价效应)",
        "sigma": "正股波动率",
        "maturity": "剩余期限",
        "mom20": "正股 20 日动量",
    }
    fmb = {}
    for h in (5, 10, 20):
        rcol = f"ret_{h}"
        slopes = {c: [] for c in ctrl_cols}
        r2_list = []
        for gt in all_gts:
            if gt + 1 + h >= T_cal:
                continue
            g = by_gt[gt][ctrl_cols + [rcol]].dropna()
            if len(g) < 35:
                continue
            X_raw = g[ctrl_cols].copy()
            for c in ctrl_cols:
                lo, hi = X_raw[c].quantile(0.01), X_raw[c].quantile(0.99)
                X_raw[c] = X_raw[c].clip(lo, hi)
                sd = float(X_raw[c].std())
                X_raw[c] = (X_raw[c] - X_raw[c].mean()) / sd if sd > 1e-8 else 0.0
            X = np.column_stack([np.ones(len(X_raw)), X_raw.values])
            y = g[rcol].values
            theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            yhat = X @ theta
            ss_res = float(np.sum((y - yhat) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2_list.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)
            for idx_c, c in enumerate(ctrl_cols):
                slopes[c].append(float(theta[idx_c + 1]))

        fmb[str(h)] = {
            "days": len(r2_list),
            "avg_r2": round(float(np.mean(r2_list)) * 100.0, 2),
            "note": "已剔除包含事后下调泄漏的 latest_rating，严格仅使用当日可得的 6 个因果变量。",
            "factors": [
                {
                    "key": c,
                    "label": ctrl_labels[c],
                    **{k: round(v, 4) if k in ("mean", "se", "ci_low", "ci_high", "p_val") else round(v, 2) if k == "t_stat" else v for k, v in newey_west_stats(np.array(slopes[c]), lags=h).items()}
                }
                for c in ctrl_cols
            ],
        }

    # Build Before-vs-After Audit Comparison Table
    audit_comparison = None
    if pre_audit_path and Path(pre_audit_path).exists():
        pre = json.loads(Path(pre_audit_path).read_text(encoding="utf-8"))
        audit_comparison = [
            {"metric": "1D 有效 Rank IC 天数", "before": str(pre["ic_summary"]["1"]["days"]), "after": str(ic_summary["1"]["days"]), "reason": "严格全局日历对齐并禁止尾部截断"},
            {"metric": "5D 有效 Rank IC 天数", "before": str(pre["ic_summary"]["5"]["days"]), "after": str(ic_summary["5"]["days"]), "reason": "同上，T+1+5 越界严格置 null"},
            {"metric": "10D 有效 Rank IC 天数", "before": str(pre["ic_summary"]["10"]["days"]), "after": str(ic_summary["10"]["days"]), "reason": "同上，T+1+10 越界严格置 null"},
            {"metric": "20D 有效 Rank IC 天数", "before": str(pre["ic_summary"]["20"]["days"]), "after": str(ic_summary["20"]["days"]), "reason": "同上，T+1+20 越界严格置 null"},
            {"metric": "20D Mean Rank IC (ICIR)", "before": f"{pre['ic_summary']['20']['mean_ic']} ({pre['ic_summary']['20']['icir']})", "after": f"{ic_summary['20']['mean_ic']} ({ic_summary['20']['icir']})", "reason": "全局日历与停牌/退市口径修复"},
            {"metric": "20D 五分组 G5-G1 多空价差", "before": f"+{pre['quintiles']['20']['g5_minus_g1']}% (t={pre['quintiles']['20']['t_nw']})", "after": f"+{quintiles['20']['g5_minus_g1']}% (t={quintiles['20']['t_nw']})", "reason": "去除右端截断与停牌滑行偏差"},
            {"metric": "Four-Region LSM 5D 净收益 (0.10%)", "before": f"+{pre['strategies']['four_region_lsm']['costs']['0.10%']['total_ret']}% (Sharpe {pre['strategies']['four_region_lsm']['costs']['0.10%']['sharpe']})", "after": f"+{strategies_res['four_region_lsm']['costs']['0.10%']['total_ret']}% (Sharpe {strategies_res['four_region_lsm']['costs']['0.10%']['sharpe']})", "reason": "停牌锁定不可成交 + 退市扣费强平"},
            {"metric": "违约退市 50% 折价压力测试", "before": "未单独压力测试", "after": f"+{stress_four['total_ret']}% (Sharpe {stress_four['sharpe']})", "reason": "退市末日收盘 <¥90 债券额外按 50% 清算"},
            {"metric": "Fama-MacBeth 20D β (F_LSM)", "before": f"+{pre['fama_macbeth']['20']['factors'][0]['mean']} (t={pre['fama_macbeth']['20']['factors'][0]['t_stat']}, 含事后评级)", "after": f"+{fmb['20']['factors'][0]['mean']} (t={fmb['20']['factors'][0]['t_stat']}, 无评级泄漏)", "reason": "剔除含未来下调信息的信用评级控制变量"},
        ]

    result = sanitize_json({
        "universe": {
            "total_bonds": int(panel["ticker"].nunique()),
            "total_bond_days": int(len(panel)),
            "trading_days": int(len(all_gts)),
            "start_date": cal_dates[all_gts[0]],
            "end_date": cal_dates[all_gts[-1]],
            "avg_daily_bonds": round(float(len(panel) / len(all_gts)), 1),
        },
        "ic_summary": ic_summary,
        "ic_decay": ic_decay,
        "rolling_ic": rolling_ic,
        "subperiod_ic": subperiod_ic,
        "quintiles": quintiles,
        "quintile_curves": {"dates": q_curve_dates, "navs": q_curves},
        "strategies": strategies_res,
        "stress_default_50pct": stress_four,
        "random_mc": random_mc,
        "subperiod_strats": subperiod_strats,
        "freq_robust": freq_robust,
        "fama_macbeth": fmb,
        "audit_comparison": audit_comparison,
    })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Saved audited JSON to {out_path} ({out_path.stat().st_size/1024:.1f} KB, allow_nan=False)", flush=True)
    if audit_comparison:
        print("\n=== AUDIT BEFORE vs AFTER COMPARISON ===")
        for row in audit_comparison:
            print(f"  {row['metric']:30s} | Before: {row['before']:26s} | After: {row['after']}")


if __name__ == "__main__":
    main()
