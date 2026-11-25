"""
Enrich the audited full-market panel (`panel_audit.pkl`) with strictly causal peer pricing
& quantitative factors required for P1 (Broker Benchmarks) and P2 (Market Valuation & Timing):
  - Eastmoney point-in-time pure bond value (`pure_bond`) & pure bond premium (`pure_bond_prem`)
  - Causal initial issuance scale (`issue_scale` in RMB 100M, proxy disclosed honestly)
  - Causal contractual YTM (`ytm_pct`, using remaining issuance coupons + par + final coupon)
  - Black-Scholes valuations: `v_bs_pure` (EM pure bond + BS call) and `v_bs_b108` (B=108 floor + BS call)
  - 30-step Markov CRR Binomial Tree (`v_binom`, American conv + 1.30 call + 0.70 last-2y put, no reset/window)
  - BS Implied Volatility (`impl_vol`, %) & Vol Spread (`hv_minus_iv`, HV60 - IV, %)
  - Point-in-time credit ratings (`rating`) & causal risk flags (`is_st`, `has_def_warn`, `has_spec_rat`, `is_low_liq`, `is_risk_excluded`)

Usage:
    python tools/build_pro_panel.py --cache-dir /tmp/bonds_v2/cache
"""

import argparse
import pickle
import sys
import time
from pathlib import Path
import akshare as ak
import numpy as np
import pandas as pd
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from run_strategy_experiments import build_risk_flags


def bs_call_vec(S: np.ndarray, K: np.ndarray, T: np.ndarray, r: float, sigma: np.ndarray) -> np.ndarray:
    sqrtT = np.sqrt(np.maximum(1e-6, T))
    sigT = np.maximum(1e-6, sigma * sqrtT)
    d1 = (np.log(np.maximum(1e-6, S) / np.maximum(1e-6, K)) + (r + 0.5 * sigma * sigma) * T) / sigT
    d2 = d1 - sigT
    return S * ndtr(d1) - K * np.exp(-r * T) * ndtr(d2)


def solve_iv_vec(opt_px: np.ndarray, S: np.ndarray, K: np.ndarray, T: np.ndarray, r: float) -> np.ndarray:
    """28-step bisection for BS implied volatility (%) on valid positive-time-value rows."""
    ratio = 100.0 / np.maximum(1e-4, K)
    c_low = ratio * bs_call_vec(S, K, T, r, np.full_like(S, 0.05))
    c_high = ratio * bs_call_vec(S, K, T, r, np.full_like(S, 1.50))
    valid = np.isfinite(opt_px) & (opt_px > c_low + 0.05) & (opt_px < c_high - 0.05) & (S > 0) & (K > 0) & (T > 0.02)

    lo = np.full_like(S, 0.05)
    hi = np.full_like(S, 1.50)
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        val = ratio * bs_call_vec(S, K, T, r, mid)
        go_up = val < opt_px
        lo = np.where(go_up, mid, lo)
        hi = np.where(go_up, hi, mid)
    iv = 0.5 * (lo + hi) * 100.0
    return np.where(valid, iv, np.nan)


def binomial_crr_chunk(S0: np.ndarray, C0: np.ndarray, T: np.ndarray, sigma: np.ndarray, r: float = 0.025, B: float = 108.0, n_steps: int = 30) -> np.ndarray:
    """Vectorized CRR Binomial lattice (Markovian approximation: Am conv + 1.30 call + 0.70 last-2y put)."""
    m = len(S0)
    dt = np.maximum(1e-4, T / n_steps)
    disc = np.exp(-r * dt)
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = np.clip((np.exp(r * dt) - d) / np.maximum(1e-9, u - d), 0.01, 0.99)

    j = np.arange(n_steps + 1, dtype=np.float64)[None, :]
    S_T = S0[:, None] * (u[:, None] ** (n_steps - j)) * (d[:, None] ** j)
    conv_ratio = 100.0 / np.maximum(1e-4, C0[:, None])
    V = np.maximum(B, conv_ratio * S_T)

    for step in range(n_steps - 1, -1, -1):
        t_rem_after = T * (1.0 - step / n_steps)
        k_idx = np.arange(step + 1, dtype=np.float64)[None, :]
        S_t = S0[:, None] * (u[:, None] ** (step - k_idx)) * (d[:, None] ** k_idx)
        hold = disc[:, None] * (p[:, None] * V[:, :-1] + (1.0 - p[:, None]) * V[:, 1:])
        conv = conv_ratio * S_t
        val = np.maximum(hold, conv)

        # Conditional call cap above 1.30 * C0 (excluding t=0 issuance edge)
        if step > 0:
            call_mask = S_t >= (1.30 * C0[:, None])
            val = np.where(call_mask, np.maximum(conv, 103.0), val)
            # Conditional put floor in final 2 years below 0.70 * C0
            put_mask = (S_t <= (0.70 * C0[:, None])) & (t_rem_after[:, None] <= 2.0)
            val = np.where(put_mask, np.maximum(val, 101.0), val)
        V = val
    return V[:, 0]


def compute_ytm_fast(close_px: np.ndarray, T_rem: np.ndarray, coupons_padded: np.ndarray, n_coupons: np.ndarray) -> np.ndarray:
    """Vectorized 24-step bisection for pre-tax contractual YTM (%).
    coupons_padded has shape (N, 6) padded from the LEFT? No, stored as last_yr..first_yr so col 0 = final year (+100 par), col 1 = T-1, etc.
    """
    n = len(close_px)
    lo = np.full(n, -0.35, dtype=np.float64)
    hi = np.full(n, 0.80, dtype=np.float64)
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        pv = np.zeros(n, dtype=np.float64)
        for k in range(6):
            t_cf = T_rem - k
            active = (t_cf > 1e-4) & (k < n_coupons)
            cf = coupons_padded[:, k] + (100.0 if k == 0 else 0.0)
            disc = np.where(active, np.power(np.maximum(0.05, 1.0 + mid), -np.maximum(1e-4, t_cf)), 0.0)
            pv += np.where(active, cf * disc, 0.0)
        too_low_yield = pv > close_px
        lo = np.where(too_low_yield, mid, lo)
        hi = np.where(too_low_yield, hi, mid)
    return 0.5 * (lo + hi) * 100.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="/tmp/bonds_v2/cache")
    args = ap.parse_args()

    t0 = time.time()
    cache_dir = Path(args.cache_dir)
    with open(cache_dir / "panel_audit.pkl", "rb") as f:
        bundle = pickle.load(f)
    panel = bundle["panel"]
    ev = pd.read_csv(cache_dir / "rating_events_all.csv", dtype=str)

    print("1. Attaching point-in-time ratings and causal risk flags...", flush=True)
    df = build_risk_flags(panel, cache_dir, ev)

    print("2. Fetching issuance scale map from Eastmoney bond_cb_list_em...", flush=True)
    scale_map = {}
    try:
        df_em = ak.bond_cb_list_em()
        for _, r in df_em.iterrows():
            c = str(r.get("债券代码", "")).strip()
            sc = pd.to_numeric(r.get("发行规模"), errors="coerce")
            if c and pd.notna(sc) and sc > 0:
                scale_map[c] = float(sc)
    except Exception as e:
        print(f"Warning: bond_cb_list_em failed: {e}")
    df["issue_scale"] = df["ticker"].map(scale_map).astype(float)

    print("3. Merging Eastmoney pure_bond & issuance coupon schedules from cbs/*.pkl...", flush=True)
    va_frames = []
    coup_map = {}
    for p in (cache_dir / "cbs").glob("*.pkl"):
        with open(p, "rb") as f:
            obj = pickle.load(f)
        tk = str(obj["meta"]["ticker"])
        cps = [float(x) for x in (obj["meta"].get("coupons") or [0.3, 0.5, 1.0, 1.5, 2.0, 2.5])]
        # Reverse so col 0 = final year coupon, col 1 = penultimate, etc.
        rev_cps = list(reversed(cps))[:6]
        arr = np.zeros(6, dtype=np.float64)
        arr[:len(rev_cps)] = rev_cps
        coup_map[tk] = (arr, len(rev_cps))

        va = obj["va"][["date", "纯债价值"]].drop_duplicates("date").copy()
        va["date"] = pd.to_datetime(va["date"]).dt.strftime("%Y-%m-%d")
        va["pure_bond"] = pd.to_numeric(va["纯债价值"], errors="coerce")
        va["ticker"] = tk
        va_frames.append(va[["ticker", "date", "pure_bond"]])

    va_all = pd.concat(va_frames, ignore_index=True)
    df = df.merge(va_all, on=["ticker", "date"], how="left")

    # Fallback contractual PV when EM pure_bond missing (disclosed)
    em_pb_valid = df["pure_bond"].notna() & (df["pure_bond"] > 20.0)
    em_cov_pct = round(float(em_pb_valid.mean() * 100.0), 2)
    T_arr = df["maturity"].values.astype(np.float64)
    pb_fallback = 108.0 * np.exp(-0.025 * T_arr)
    df["pure_bond_eff"] = np.where(em_pb_valid, df["pure_bond"].values, pb_fallback)
    df["pure_bond_prem"] = (df["close"].values - df["pure_bond_eff"].values) / np.maximum(1.0, df["pure_bond_eff"].values) * 100.0

    # Derive exact causal conversion value & conversion price C_t
    conv_val = df["close"].values / np.maximum(0.05, 1.0 + df["prem"].values / 100.0)
    df["conv_val"] = conv_val
    # Note: conv_val = (100/C_t) * S_t -> in BS call with S_t normalized so C_t = 100: S_norm = conv_val, K = 100!
    # Mathematically identical: (100/C_t)*BS_Call(S_t, C_t) == BS_Call(conv_val, 100)!
    S_norm = np.maximum(0.05, conv_val)
    K_norm = np.full_like(S_norm, 100.0)
    sig_dec = np.clip(df["sigma"].values / 100.0, 0.15, 0.70)

    print("4. Computing Black-Scholes, Implied Vol, CRR Binomial Tree & YTM...", flush=True)
    bs_opt = bs_call_vec(S_norm, K_norm, T_arr, 0.025, sig_dec)
    df["v_bs_pure"] = df["pure_bond_eff"].values + bs_opt
    df["v_bs_b108"] = 108.0 * np.exp(-0.025 * T_arr) + bs_opt
    df["f_bs_pure"] = (df["v_bs_pure"].values - df["close"].values) / df["close"].values * 100.0
    df["f_bs_b108"] = (df["v_bs_b108"].values - df["close"].values) / df["close"].values * 100.0

    opt_mkt = df["close"].values - df["pure_bond_eff"].values
    iv = solve_iv_vec(opt_mkt, S_norm, K_norm, T_arr, 0.025)
    df["impl_vol"] = iv
    df["hv_minus_iv"] = df["sigma"].values - iv

    # CRR Binomial Tree in 50k chunks
    v_bin = np.zeros(len(df), dtype=np.float64)
    chunk = 50000
    for i0 in range(0, len(df), chunk):
        i1 = min(len(df), i0 + chunk)
        v_bin[i0:i1] = binomial_crr_chunk(S_norm[i0:i1], K_norm[i0:i1], T_arr[i0:i1], sig_dec[i0:i1], r=0.025, B=108.0, n_steps=30)
    df["v_binom"] = v_bin
    df["f_binom"] = (v_bin - df["close"].values) / df["close"].values * 100.0

    # YTM
    cps_pad = np.zeros((len(df), 6), dtype=np.float64)
    n_cps = np.zeros(len(df), dtype=np.int32)
    tks = df["ticker"].values
    for idx_r, tk in enumerate(tks):
        arr, nc = coup_map.get(tk, (np.array([2.5, 2.0, 1.5, 1.0, 0.5, 0.3]), 6))
        cps_pad[idx_r] = arr
        n_cps[idx_r] = nc
    df["ytm_pct"] = compute_ytm_fast(df["close"].values, T_arr, cps_pad, n_cps)

    iv_cov_pct = round(float(np.isfinite(iv).mean() * 100.0), 2)
    scale_cov_pct = round(float(df["issue_scale"].notna().mean() * 100.0), 2)
    out_meta = {
        "total_bond_days": int(len(df)),
        "em_pure_bond_cov_pct": em_cov_pct,
        "impl_vol_cov_pct": iv_cov_pct,
        "issue_scale_cov_pct": scale_cov_pct,
    }
    print(f"Enriched metadata: {out_meta} in {time.time()-t0:.1f}s", flush=True)

    out_bundle = {
        "df": df,
        "exec_map": bundle["exec_map"],
        "cal_dates": bundle["cal_dates"],
        "meta": out_meta,
    }
    with open(cache_dir / "panel_pro.pkl", "wb") as f:
        pickle.dump(out_bundle, f)
    print(f"Saved {cache_dir / 'panel_pro.pkl'}", flush=True)


if __name__ == "__main__":
    main()
