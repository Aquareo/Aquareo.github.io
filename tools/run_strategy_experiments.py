"""
Practical Enhancement Experiments on Frozen Four-Region LSM Model (2020–2026).

Evaluates strictly without changing any pricing model parameter (p_reset=0.80, B=108.0, r=0.025):
1. 5 point-in-time credit rating pools: All, AA-+, AA+, AA++, AAA-only (zero ex-post leakage).
2. Point-in-time risk & liquidity exclusions: historical ST/*ST underlying stock,
   announced CB default/termination risk or <= BBB+ rating, and trailing 20d median turnover < ¥5M.
3. Blended factor weights (LSM / Double-Low): 100/0, 75/25, 50/50, 25/75, 0/100 via cross-sectional percentile ranks.
4. Rebalance frequencies: 5D, 10D, 20D.
5. Rank hysteresis buffers: none (20/20), 20/30, 20/40.
6. Unified T+1 Open execution, suspension-lock, delist liquidation, 0.05%/0.10%/0.20% costs;
   reports total/annual return, MDD, Sharpe, Calmar, annual turnover, and yearly returns.
7. Stepwise independent component attribution + full 450-configuration matrix (all positive/negative preserved).
8. Walk-Forward Out-of-Sample validation (Expanding & Rolling 2Y) + Deflated Sharpe Ratio (DSR) + CSCV PBO.

Usage:
    python tools/run_strategy_experiments.py --panel /tmp/bonds_exp/cache/panel_audit.pkl \
        --cache-dir /tmp/bonds_exp/cache --out dist/bonds/research_experiments.json
"""

import argparse
from itertools import combinations
import json
import math
import pickle
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from run_full_validation import sanitize_json
from run_high_grade_validation import GRADES, attach_ratings, ticker as to_ts_code


POOLS = [
    ("all", "全市场", None),
    ("aa_minus", "AA-及以上", {"AA-", "AA", "AA+", "AAA"}),
    ("aa", "AA及以上", {"AA", "AA+", "AAA"}),
    ("aa_plus", "AA+及以上", {"AA+", "AAA"}),
    ("aaa", "仅AAA", {"AAA"}),
]
WEIGHTS = [(1.00, "100/0"), (0.75, "75/25"), (0.50, "50/50"), (0.25, "25/75"), (0.00, "0/100")]
FREQS = [5, 10, 20]
BUFFERS = [("none", "无缓冲", 0.20), ("buf_20_30", "20/30 缓冲", 0.30), ("buf_20_40", "20/40 缓冲", 0.40)]
COSTS = [(0.0005, "0.05%"), (0.0010, "0.10%"), (0.0020, "0.20%")]
LOW_LIQ_RMB = 5_000_000.0  # Trailing 20-day median daily RMB turnover < ¥500万元
SPECULATIVE_GRADES = {"BBB+", "BBB", "BBB-", "BB+", "BB", "BB-", "B+", "B", "B-", "CCC", "CC", "C"}


def build_risk_flags(panel: pd.DataFrame, cache_dir: Path, rating_events: pd.DataFrame) -> pd.DataFrame:
    """Attach strictly causal historical ST, default warning, speculative rating, and low-liquidity flags."""
    cb_dir = cache_dir / "cbs"
    st_path = cache_dir / "st_intervals.csv"
    def_path = cache_dir / "default_risk_events.csv"

    st_df = pd.read_csv(st_path, dtype=str) if st_path.exists() else pd.DataFrame(columns=["stock_code", "st_start", "st_end"])
    def_df = pd.read_csv(def_path, dtype=str) if def_path.exists() else pd.DataFrame(columns=["ts_code", "ann_date"])

    # Map stock_code & compute causal trailing 20d median daily turnover (close * volume, verified 1张=¥100par on SH & SZ)
    stk_map = {}
    liq_frames = []
    for p in cb_dir.glob("*.pkl"):
        with open(p, "rb") as f:
            obj = pickle.load(f)
        tk = str(obj["meta"]["ticker"])
        stk_map[tk] = str(obj["meta"]["stock_code"]).zfill(6)
        bh = obj["b_hist"][["date", "close", "volume"]].drop_duplicates("date").sort_values("date").copy()
        bh["date_str"] = pd.to_datetime(bh["date"]).dt.strftime("%Y-%m-%d")
        amt = pd.to_numeric(bh["close"], errors="coerce") * pd.to_numeric(bh["volume"], errors="coerce")
        bh["amt20_med"] = amt.rolling(20, min_periods=10).median()
        bh["ticker"] = tk
        liq_frames.append(bh[["ticker", "date_str", "amt20_med"]])

    liq_all = pd.concat(liq_frames, ignore_index=True)
    df = attach_ratings(panel, rating_events)
    df["stock_code"] = df["ticker"].map(stk_map).fillna("")
    df["ymd"] = df["date"].str.replace("-", "")
    df = df.merge(liq_all, left_on=["ticker", "date"], right_on=["ticker", "date_str"], how="left")
    df.drop(columns=["date_str"], inplace=True)

    # 1. Point-in-time ST flag: signal ymd strictly > st_start and <= st_end
    st_intervals_by_sc = {}
    for _, r in st_df.iterrows():
        sc = str(r["stock_code"]).zfill(6)
        st_intervals_by_sc.setdefault(sc, []).append((str(r["st_start"]), str(r["st_end"])))

    is_st = np.zeros(len(df), dtype=bool)
    sc_vals = df["stock_code"].values
    ymd_vals = df["ymd"].values
    for idx_sc, (sc, ym) in enumerate(zip(sc_vals, ymd_vals)):
        ivs = st_intervals_by_sc.get(sc)
        if ivs:
            for s0, s1 in ivs:
                if ym > s0 and ym <= s1:
                    is_st[idx_sc] = True
                    break

    # 2. Point-in-time announced default/delisting warning: ymd strictly > first warning ann_date
    if len(def_df):
        def_df["code6"] = def_df["ts_code"].astype(str).str.split(".").str[0]
        first_def = def_df.groupby("code6")["ann_date"].min().to_dict()
    else:
        first_def = {}
    code6_vals = df["ticker"].map(to_ts_code).values
    has_def_warn = np.array([ym > first_def.get(c6, "99999999") for c6, ym in zip(code6_vals, ymd_vals)], dtype=bool)

    # 3. Point-in-time speculative rating (<= BBB+)
    has_spec_rat = df["rating"].isin(SPECULATIVE_GRADES).values

    # 4. Causal trailing 20d median turnover < ¥5M
    amt20 = df["amt20_med"].values
    is_low_liq = np.where(np.isfinite(amt20), amt20 < LOW_LIQ_RMB, False)

    df["is_st"] = is_st
    df["has_def_warn"] = has_def_warn
    df["has_spec_rat"] = has_spec_rat
    df["is_low_liq"] = is_low_liq
    df["is_risk_excluded"] = is_st | has_def_warn | has_spec_rat | is_low_liq
    return df


def prepare_fast_engine(df: pd.DataFrame, exec_map: dict, cal_dates: list):
    """Pre-pack execution map & bond IDs into contiguous NumPy arrays for millisecond grid simulation."""
    all_tks = sorted(exec_map.keys())
    tk_to_id = {tk: i for i, tk in enumerate(all_tks)}
    n_cb = len(all_tks)
    T_cal = len(cal_dates)

    tradable = np.zeros((n_cb, T_cal), dtype=bool)
    g_open = np.zeros((n_cb, T_cal), dtype=np.float64)
    g_coup = np.zeros((n_cb, T_cal), dtype=np.float64)
    is_delisted = np.zeros(n_cb, dtype=bool)
    last_gt = np.zeros(n_cb, dtype=np.int32)
    delist_px = np.zeros(n_cb, dtype=np.float64)

    for tk, bid in tk_to_id.items():
        info = exec_map[tk]
        tradable[bid] = info["tradable"]
        g_open[bid] = np.nan_to_num(info["g_open"], nan=0.0)
        g_coup[bid] = info["g_coup"]
        is_delisted[bid] = bool(info["is_delisted"])
        last_gt[bid] = int(info["last_gt"])
        delist_px[bid] = float(info["delist_payoff_base"])

    df = df.copy()
    df["bid"] = df["ticker"].map(tk_to_id)
    return {
        "df": df,
        "tk_to_id": tk_to_id,
        "n_cb": n_cb,
        "T_cal": T_cal,
        "cal_dates": cal_dates,
        "tradable": tradable,
        "g_open": g_open,
        "g_coup": g_coup,
        "is_delisted": is_delisted,
        "last_gt": last_gt,
        "delist_px": delist_px,
    }


def precompute_daily_candidates(eng: dict, allowed_grades, exclude_risk: bool):
    """Precompute percentile ranks of f_four and double_low on every trading day gt for a (pool, risk_mode)."""
    df = eng["df"]
    mask = np.ones(len(df), dtype=bool)
    if allowed_grades is not None:
        mask &= df["rating"].isin(allowed_grades).values
    if exclude_risk:
        mask &= (~df["is_risk_excluded"].values)
    sub = df[mask]

    T_cal = eng["T_cal"]
    daily = [None] * T_cal
    for gt, grp in sub.groupby("gt"):
        t1 = int(gt) + 1
        g_buy = grp[~(grp["is_delisted"].values & (grp["last_gt"].values <= t1))].dropna(subset=["f_four", "double_low"])
        if len(g_buy) < 15:
            continue
        pct_lsm = g_buy["f_four"].rank(pct=True, ascending=True).values.astype(np.float64)
        pct_dl = g_buy["double_low"].rank(pct=True, ascending=False).values.astype(np.float64)
        bids = g_buy["bid"].values.astype(np.int32)
        daily[int(gt)] = (bids, pct_lsm, pct_dl)

    cov_info = {
        "bond_days": int(len(sub)),
        "unique_bonds": int(sub["ticker"].nunique()),
        "avg_daily": round(float(sub.groupby("gt").size().mean()), 1) if len(sub) else 0.0,
    }
    return daily, cov_info


def simulate_fast(eng: dict, daily: list, w_lsm: float, freq: int, sell_frac: float, all_eq: bool = False):
    """Exact replica of run_full_validation.run_portfolio_sim with percentile blend + optional hysteresis buffer."""
    T_cal = eng["T_cal"]
    tradable = eng["tradable"]
    g_open = eng["g_open"]
    g_coup = eng["g_coup"]
    is_delisted = eng["is_delisted"]
    last_gt = eng["last_gt"]
    delist_px = eng["delist_px"]

    holdings = {}  # bid -> [val, last_px]
    cash = 1.0
    daily_gross = np.zeros(T_cal - 2, dtype=np.float64)
    daily_to = np.zeros(T_cal - 2, dtype=np.float64)
    used_dates = eng["cal_dates"][2:T_cal]
    rebal_step = 0

    for gt in range(T_cal - 2):
        t1 = gt + 1
        t2 = gt + 2

        tradable_keys = []
        locked_keys = []
        locked_val = 0.0
        nav_pre = cash
        for bid, pos in holdings.items():
            nav_pre += pos[0]
            if tradable[bid, t1]:
                tradable_keys.append(bid)
            else:
                locked_keys.append(bid)
                locked_val += pos[0]

        free_cap = max(0.0, nav_pre - locked_val)
        two_way_to = 0.0
        should_rebal = (rebal_step % freq == 0) or (len(holdings) == 0 and free_cap > 0.5)
        rebal_step += 1

        cand = daily[gt]
        if should_rebal and cand is not None and free_cap > 1e-8:
            bids, pct_lsm, pct_dl = cand
            n_elig = len(bids)
            if all_eq:
                picks = bids.tolist()
            else:
                score = w_lsm * pct_lsm + (1.0 - w_lsm) * pct_dl
                # Deterministic tie-break using pct_lsm + tiny bid order
                order = np.argsort(-(score + 1e-9 * pct_lsm), kind="mergesort")
                ranked_bids = bids[order]
                k_target = min(n_elig, max(5, int(round(n_elig * 0.20))))

                if sell_frac <= 0.201:
                    picks = ranked_bids[:k_target].tolist()
                else:
                    k_sell = min(n_elig, max(k_target, int(round(n_elig * sell_frac))))
                    keep_zone = set(ranked_bids[:k_sell].tolist())
                    held_tradable_set = set(tradable_keys)
                    kept = [int(b) for b in ranked_bids[:k_sell] if b in held_tradable_set]
                    picks = list(kept)
                    kept_set = set(kept)
                    for b in ranked_bids:
                        if len(picks) >= k_target:
                            break
                        bi = int(b)
                        if bi not in kept_set:
                            picks.append(bi)
                            kept_set.add(bi)

            if picks:
                tw_each = (free_cap / nav_pre) / len(picks)
                target_w = {b: tw_each for b in picks}
                old_w = {b: holdings[b][0] / nav_pre for b in tradable_keys}
                for k in (set(target_w) | set(old_w)):
                    two_way_to += abs(target_w.get(k, 0.0) - old_w.get(k, 0.0))

                new_holdings = {b: holdings[b] for b in locked_keys}
                for b, tw in target_w.items():
                    new_holdings[b] = [tw * nav_pre, float(g_open[b, t1])]
                holdings = new_holdings
                cash = 0.0

        nav_after = cash
        to_del = []
        for bid, pos in holdings.items():
            if is_delisted[bid] and t1 >= last_gt[bid]:
                pay_px = delist_px[bid]
                c_day = float(g_coup[bid, last_gt[bid]])
                exit_val = pos[0] * ((pay_px + c_day) / pos[1])
                two_way_to += exit_val / max(1e-9, nav_pre)
                cash += exit_val
                nav_after += exit_val
                to_del.append(bid)
            elif tradable[bid, t2]:
                p_t2 = float(g_open[bid, t2])
                c_day = float(g_coup[bid, t2])
                pos[0] = pos[0] * ((p_t2 + c_day) / pos[1])
                pos[1] = p_t2
                nav_after += pos[0]
            else:
                nav_after += pos[0]
        for bid in to_del:
            del holdings[bid]

        daily_gross[gt] = (nav_after / nav_pre - 1.0) if nav_pre > 0 else 0.0
        daily_to[gt] = two_way_to
        if nav_after > 0:
            scale = 1.0 / nav_after
            cash *= scale
            for pos in holdings.values():
                pos[0] *= scale

    return daily_gross, daily_to, used_dates


def compute_metrics(gross: np.ndarray, to: np.ndarray, dates: list, include_nav: bool = False) -> dict:
    ann_to = round(float(np.sum(to) * 0.5 * (252.0 / max(1, len(gross))) * 100.0), 1)
    out = {"ann_to": ann_to, "costs": {}}
    years_arr = np.array([d[:4] for d in dates])
    uniq_yrs = sorted(set(years_arr))

    for c_val, c_key in COSTS:
        net = gross - to * c_val
        nav = np.cumprod(1.0 + net)
        peak = np.maximum.accumulate(np.insert(nav, 0, 1.0))[1:]
        mdd = float(np.min(nav / peak - 1.0) * 100.0)
        tot_ret = float((nav[-1] - 1.0) * 100.0)
        ann_ret = float((nav[-1] ** (252.0 / max(1, len(net))) - 1.0) * 100.0)
        sharpe = float((np.mean(net) * 252.0 - 0.02) / (np.std(net) * np.sqrt(252.0) + 1e-9))
        calmar = float(ann_ret / abs(mdd)) if mdd < -1e-6 else 0.0

        yr_map = {}
        for yr in uniq_yrs:
            ymask = years_arr == yr
            sub_net = net[ymask]
            ynav = np.cumprod(1.0 + sub_net)
            ypeak = np.maximum.accumulate(np.insert(ynav, 0, 1.0))[1:]
            yr_map[yr] = {
                "ret": round(float((ynav[-1] - 1.0) * 100.0), 2),
                "mdd": round(float(np.min(ynav / ypeak - 1.0) * 100.0), 2),
            }

        out["costs"][c_key] = {
            "total_ret": round(tot_ret, 2),
            "ann_ret": round(ann_ret, 2),
            "mdd": round(mdd, 2),
            "sharpe": round(sharpe, 2),
            "calmar": round(calmar, 2),
            "annual": yr_map,
        }
        if include_nav and c_key == "0.10%":
            si = list(range(0, len(nav), 10))
            if si[-1] != len(nav) - 1:
                si.append(len(nav) - 1)
            out["nav_010"] = [round(float(nav[i]), 4) for i in si]
    return out


def deflated_sharpe_ratio(net_returns: np.ndarray, all_sharpes: list) -> dict:
    """Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio accounting for multiple trials."""
    T = len(net_returns)
    sr_obs = float((np.mean(net_returns) * 252.0 - 0.02) / (np.std(net_returns) * np.sqrt(252.0) + 1e-9))
    sr_daily = sr_obs / math.sqrt(252.0)
    var_sr = float(np.var(np.array(all_sharpes) / math.sqrt(252.0), ddof=1))
    n_trials = max(2, len(all_sharpes))
    emc = 0.5772156649
    from scipy.stats import norm
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    sr_star = math.sqrt(max(1e-12, var_sr)) * ((1.0 - emc) * z1 + emc * z2)

    s = pd.Series(net_returns)
    skew = float(s.skew())
    kurt = float(s.kurt() + 3.0)  # Pearson kurtosis
    denom = math.sqrt(max(1e-9, (1.0 - skew * sr_daily + ((kurt - 1.0) / 4.0) * (sr_daily ** 2)) / max(1, T - 1)))
    z_dsr = (sr_daily - sr_star) / denom
    dsr_prob = float(norm.cdf(z_dsr))
    return {"sharpe_ann": round(sr_obs, 3), "sr_benchmark_max": round(sr_star * math.sqrt(252.0), 3), "dsr_prob": round(dsr_prob, 4), "n_trials": n_trials}


def compute_cscv_pbo(net_matrix: np.ndarray, n_blocks: int = 8) -> dict:
    """Combinatorially Symmetric Cross-Validation (S=8 blocks -> 70 IS/OOS splits) for PBO."""
    T, M = net_matrix.shape
    block_len = T // n_blocks
    blocks = [net_matrix[i * block_len:(i + 1) * block_len] for i in range(n_blocks)]

    def blk_sharpe(arr):
        mu = np.mean(arr, axis=0) * 252.0 - 0.02
        sd = np.std(arr, axis=0) * math.sqrt(252.0) + 1e-9
        return mu / sd

    half = n_blocks // 2
    logits = []
    oos_ranks = []
    oos_sharpes_of_is_best = []
    is_sharpes_of_is_best = []
    corrs = []
    for is_idx in combinations(range(n_blocks), half):
        oos_idx = [i for i in range(n_blocks) if i not in is_idx]
        is_ret = np.vstack([blocks[i] for i in is_idx])
        oos_ret = np.vstack([blocks[i] for i in oos_idx])
        sr_is = blk_sharpe(is_ret)
        sr_oos = blk_sharpe(oos_ret)
        best_j = int(np.argmax(sr_is))
        is_sharpes_of_is_best.append(float(sr_is[best_j]))
        oos_sharpes_of_is_best.append(float(sr_oos[best_j]))
        rk = float(np.sum(sr_oos <= sr_oos[best_j])) / float(M + 1)
        rk = min(0.999, max(0.001, rk))
        oos_ranks.append(rk)
        logits.append(math.log(rk / (1.0 - rk)))
        corrs.append(float(pd.Series(sr_is).rank().corr(pd.Series(sr_oos).rank())))

    pbo = float(np.mean(np.array(oos_ranks) < 0.5))
    return {
        "n_splits": len(logits),
        "n_configs": int(M),
        "pbo_pct": round(pbo * 100.0, 1),
        "median_is_best_sharpe": round(float(np.median(is_sharpes_of_is_best)), 2),
        "median_oos_sharpe": round(float(np.median(oos_sharpes_of_is_best)), 2),
        "mean_is_oos_rank_corr": round(float(np.mean(corrs)), 3),
        "sharpe_degradation_pct": round((1.0 - float(np.median(oos_sharpes_of_is_best)) / max(1e-6, float(np.median(is_sharpes_of_is_best)))) * 100.0, 1),
    }


def walk_forward_oos(configs_meta: list, net_matrix: np.ndarray, to_matrix: np.ndarray, dates: list, mode: str = "expanding") -> dict:
    """Strictly non-peeking walk-forward selection across 2021–2026 (train prior years -> test next year)."""
    years_arr = np.array([d[:4] for d in dates])
    test_years = ["2021", "2022", "2023", "2024", "2025", "2026"] if mode == "expanding" else ["2022", "2023", "2024", "2025", "2026"]

    oos_daily_rets = []
    oos_dates = []
    yearly_picks = []

    for ty in test_years:
        ty_int = int(ty)
        if mode == "expanding":
            train_mask = years_arr < ty
        else:
            train_mask = np.isin(years_arr, [str(ty_int - 2), str(ty_int - 1)])
        test_mask = years_arr == ty

        tr_ret = net_matrix[train_mask]
        mu = np.mean(tr_ret, axis=0) * 252.0 - 0.02
        sd = np.std(tr_ret, axis=0) * math.sqrt(252.0) + 1e-9
        sr_tr = mu / sd
        best_idx = int(np.argmax(sr_tr))
        chosen = configs_meta[best_idx]

        te_ret = net_matrix[test_mask, best_idx]
        te_dates = [d for d, m in zip(dates, test_mask) if m]
        oos_daily_rets.append(te_ret)
        oos_dates.extend(te_dates)

        ynav = np.cumprod(1.0 + te_ret)
        ypeak = np.maximum.accumulate(np.insert(ynav, 0, 1.0))[1:]
        yearly_picks.append({
            "test_year": ty,
            "train_window": f"2020–{ty_int-1}" if mode == "expanding" else f"{ty_int-2}–{ty_int-1}",
            "picked_id": chosen["id"],
            "picked_desc": chosen["desc"],
            "train_sharpe": round(float(sr_tr[best_idx]), 2),
            "oos_ret": round(float((ynav[-1] - 1.0) * 100.0), 2),
            "oos_mdd": round(float(np.min(ynav / ypeak - 1.0) * 100.0), 2),
            "oos_sharpe": round(float((np.mean(te_ret) * 252.0 - 0.02) / (np.std(te_ret) * math.sqrt(252.0) + 1e-9)), 2),
        })

    chained = np.concatenate(oos_daily_rets)
    nav = np.cumprod(1.0 + chained)
    peak = np.maximum.accumulate(np.insert(nav, 0, 1.0))[1:]
    mdd = float(np.min(nav / peak - 1.0) * 100.0)
    ann = float((nav[-1] ** (252.0 / len(chained)) - 1.0) * 100.0)
    sh = float((np.mean(chained) * 252.0 - 0.02) / (np.std(chained) * math.sqrt(252.0) + 1e-9))
    cal = float(ann / abs(mdd)) if mdd < -1e-6 else 0.0

    # Compare fixed priors over exact same OOS date window
    first_dt = oos_dates[0]
    oos_win_mask = np.array([d >= first_dt for d in dates])
    si = list(range(0, len(nav), 10))
    if si[-1] != len(nav) - 1:
        si.append(len(nav) - 1)

    return {
        "mode": mode,
        "period": f"{oos_dates[0]} → {oos_dates[-1]}",
        "total_ret": round(float((nav[-1] - 1.0) * 100.0), 2),
        "ann_ret": round(ann, 2),
        "mdd": round(mdd, 2),
        "sharpe": round(sh, 2),
        "calmar": round(cal, 2),
        "nav_dates": [oos_dates[i] for i in si],
        "nav_010": [round(float(nav[i]), 4) for i in si],
        "yearly_picks": yearly_picks,
        "oos_win_mask": oos_win_mask,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", default="/tmp/bonds_exp/cache/panel_audit.pkl")
    ap.add_argument("--cache-dir", default="/tmp/bonds_exp/cache")
    ap.add_argument("--out", default="dist/bonds/research_experiments.json")
    args = ap.parse_args()

    t0 = time.time()
    cache_dir = Path(args.cache_dir)
    with open(args.panel, "rb") as f:
        bundle = pickle.load(f)

    ev = pd.read_csv(cache_dir / "rating_events_all.csv", dtype=str)
    df = build_risk_flags(bundle["panel"], cache_dir, ev)
    eng = prepare_fast_engine(df, bundle["exec_map"], bundle["cal_dates"])

    # Audit summary of causal risk flags
    risk_stats = {
        "total_bond_days": int(len(df)),
        "st_bond_days": int(df["is_st"].sum()),
        "st_unique_bonds": int(df[df["is_st"]]["ticker"].nunique()),
        "def_warn_days": int(df["has_def_warn"].sum()),
        "def_warn_bonds": int(df[df["has_def_warn"]]["ticker"].nunique()),
        "spec_rating_days": int(df["has_spec_rat"].sum()),
        "low_liq_days": int(df["is_low_liq"].sum()),
        "any_risk_excluded_days": int(df["is_risk_excluded"].sum()),
        "any_risk_excluded_pct": round(float(df["is_risk_excluded"].mean() * 100.0), 2),
    }
    print("Risk exclusion summary:", risk_stats, flush=True)

    # Precompute daily candidates for 5 pools x 2 risk modes (10 slices)
    slices = {}
    pool_coverage = {}
    for pkey, plabel, grades in POOLS:
        for rmode, excl in [("raw", False), ("filtered", True)]:
            daily, cov = precompute_daily_candidates(eng, grades, excl)
            slices[(pkey, rmode)] = daily
            pool_coverage[f"{pkey}_{rmode}"] = cov

    # Verify numerical parity vs run_full_validation baseline (all, raw, 100/0, 5D, none)
    gr0, to0, dt0 = simulate_fast(eng, slices[("all", "raw")], 1.00, 5, 0.20)
    m0 = compute_metrics(gr0, to0, dt0)
    print("Baseline parity check (all_raw_100_5_none @ 0.10%):", m0["costs"]["0.10%"], "to:", m0["ann_to"], flush=True)

    # Equal-weight benchmarks for all 10 (pool, risk_mode) slices
    benchmarks = {}
    for pkey, plabel, _ in POOLS:
        for rmode in ("raw", "filtered"):
            gr_eq, to_eq, dt_eq = simulate_fast(eng, slices[(pkey, rmode)], 1.0, 5, 0.20, all_eq=True)
            benchmarks[f"{pkey}_{rmode}"] = compute_metrics(gr_eq, to_eq, dt_eq, include_nav=True)

    # Full 450-configuration combinatorial experiment grid
    grid_rows = []
    configs_meta = []
    net_list = []
    to_list = []
    all_sharpes_010 = []
    dates_ref = None

    for pkey, plabel, _ in POOLS:
        for rmode in ("raw", "filtered"):
            daily = slices[(pkey, rmode)]
            for w_lsm, w_lbl in WEIGHTS:
                for freq in FREQS:
                    for bkey, blbl, sell_frac in BUFFERS:
                        gr, to, dates = simulate_fast(eng, daily, w_lsm, freq, sell_frac)
                        dates_ref = dates
                        m = compute_metrics(gr, to, dates, include_nav=True)
                        cid = f"{pkey}|{rmode}|{w_lbl}|{freq}D|{bkey}"
                        row = {
                            "id": cid,
                            "pool": pkey,
                            "pool_label": plabel,
                            "risk_mode": rmode,
                            "w_lsm": w_lsm,
                            "weight_label": w_lbl,
                            "freq": freq,
                            "buffer": bkey,
                            "buffer_label": blbl,
                            **m,
                        }
                        grid_rows.append(row)
                        configs_meta.append({"id": cid, "desc": f"{plabel} · {'剔除风险' if rmode=='filtered' else '不剔除'} · LSM/双低 {w_lbl} · {freq}D · {blbl}"})
                        net_010 = gr - to * 0.0010
                        net_list.append(net_010)
                        to_list.append(to)
                        all_sharpes_010.append(m["costs"]["0.10%"]["sharpe"])

    net_matrix = np.column_stack(net_list)
    to_matrix = np.column_stack(to_list)
    print(f"Completed {len(grid_rows)} experiments in {time.time()-t0:.1f}s", flush=True)

    # Stepwise independent marginal attribution (#7) relative to Full Raw LSM 100/0 5D None
    by_id = {r["id"]: r for r in grid_rows}
    def row_summary(r, step_name):
        c = r["costs"]["0.10%"]
        return {
            "step": step_name, "id": r["id"], "ann_ret": c["ann_ret"], "total_ret": c["total_ret"],
            "mdd": c["mdd"], "sharpe": c["sharpe"], "calmar": c["calmar"], "ann_to": r["ann_to"],
        }

    stepwise = {
        "credit_filter": [row_summary(by_id[f"{pk}|raw|100/0|5D|none"], lbl) for pk, lbl, _ in POOLS],
        "risk_filter_by_pool": [
            {
                "pool": plbl,
                "raw": row_summary(by_id[f"{pk}|raw|100/0|5D|none"], "不剔除"),
                "filtered": row_summary(by_id[f"{pk}|filtered|100/0|5D|none"], "剔除高风险/低流"),
            }
            for pk, plbl, _ in POOLS
        ],
        "blend_weights": [row_summary(by_id[f"all|raw|{w_lbl}|5D|none"], f"LSM/双低 {w_lbl}") for _, w_lbl in WEIGHTS],
        "rebal_freq": [row_summary(by_id[f"all|raw|100/0|{f}D|none"], f"{f}D 调仓") for f in FREQS],
        "buffers": [row_summary(by_id[f"all|raw|100/0|5D|{bk}"], blbl) for bk, blbl, _ in BUFFERS],
    }

    # Walk-Forward OOS + Fixed Priors comparison + DSR + CSCV PBO (#8)
    wf_exp = walk_forward_oos(configs_meta, net_matrix, to_matrix, dates_ref, mode="expanding")
    wf_roll = walk_forward_oos(configs_meta, net_matrix, to_matrix, dates_ref, mode="rolling2y")
    pbo_res = compute_cscv_pbo(net_matrix, n_blocks=8)

    # Deflated Sharpe for key practical candidates + IS max Sharpe variant
    best_is_idx = int(np.argmax(all_sharpes_010))
    check_ids = [
        ("原始基准 LSM (全市场·不剔除·100/0·5D·无缓冲)", "all|raw|100/0|5D|none"),
        ("风险过滤 LSM (全市场·剔除·100/0·10D·20/30)", "all|filtered|100/0|10D|buf_20_30"),
        ("AA及以上稳健均衡 (AA+·剔除·50/50·10D·20/30)", "aa|filtered|50/50|10D|buf_20_30"),
        ("AA+高信用均衡 (AA+以上·剔除·75/25·10D·20/30)", "aa_plus|filtered|75/25|10D|buf_20_30"),
        ("样本内Sharpe最高配置 (警示过拟合对照)", configs_meta[best_is_idx]["id"]),
    ]
    id_to_col = {c["id"]: i for i, c in enumerate(configs_meta)}
    win_exp = wf_exp.pop("oos_win_mask")
    _ = wf_roll.pop("oos_win_mask")

    oos_priors = []
    for label, cid in check_ids:
        j = id_to_col[cid]
        sub_r = net_matrix[win_exp, j]
        nav = np.cumprod(1.0 + sub_r)
        pk = np.maximum.accumulate(np.insert(nav, 0, 1.0))[1:]
        mdd = float(np.min(nav / pk - 1.0) * 100.0)
        ann = float((nav[-1] ** (252.0 / len(sub_r)) - 1.0) * 100.0)
        sh = float((np.mean(sub_r) * 252.0 - 0.02) / (np.std(sub_r) * math.sqrt(252.0) + 1e-9))
        cal = float(ann / abs(mdd)) if mdd < -1e-6 else 0.0
        dsr = deflated_sharpe_ratio(net_matrix[:, j], all_sharpes_010)
        si = list(range(0, len(nav), 10))
        if si[-1] != len(nav) - 1:
            si.append(len(nav) - 1)
        oos_priors.append({
            "label": label,
            "id": cid,
            "oos_total_ret": round(float((nav[-1] - 1.0) * 100.0), 2),
            "oos_ann_ret": round(ann, 2),
            "oos_mdd": round(mdd, 2),
            "oos_sharpe": round(sh, 2),
            "oos_calmar": round(cal, 2),
            "full_sharpe": round(all_sharpes_010[j], 2),
            "dsr_prob": dsr["dsr_prob"],
            "sr_max_expected": dsr["sr_benchmark_max"],
            "nav_dates": [dates_ref[idx] for idx, keep in enumerate(win_exp) if keep][::10][:len(si)],
            "nav_010": [round(float(nav[i]), 4) for i in si],
        })

    si_ref = list(range(0, len(dates_ref), 10))
    if si_ref[-1] != len(dates_ref) - 1:
        si_ref.append(len(dates_ref) - 1)
    nav_dates_ref = [dates_ref[i] for i in si_ref]

    # Clean compact JSON payload
    out_obj = sanitize_json({
        "meta": {
            "total_configs": len(grid_rows),
            "low_liq_threshold_rmb": LOW_LIQ_RMB,
            "risk_stats": risk_stats,
            "pool_coverage": pool_coverage,
            "nav_dates": nav_dates_ref,
        },
        "benchmarks": benchmarks,
        "stepwise": stepwise,
        "walk_forward": {"expanding": wf_exp, "rolling2y": wf_roll, "priors_2021_2026": oos_priors},
        "overfitting": pbo_res,
        "grid": grid_rows,
    })

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out_obj, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Saved experiment results to {out_p} ({out_p.stat().st_size/1024:.1f} KB)", flush=True)


if __name__ == "__main__":
    main()
