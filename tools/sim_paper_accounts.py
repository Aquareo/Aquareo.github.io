"""
P3: ¥100,000 Virtual Paper Trading Engine (Forward-Only, Idempotent, Strictly Causal T+1 Open Execution).

Features:
  - Fixed ¥100,000.00 initial capital across comparable credit-pool cohorts (AA-+, AA+, Full).
  - Discrete 10-bond lot execution (1手=10张), 0.05% commission (min ¥5/order) + 0.05% slippage,
    single-bond concentration caps, suspension locks, and delist/redemption cash settlement.
  - Strict SSE/SZSE trading calendar + Asia/Shanghai session phase separation:
      * Market Date vs Signal Date vs Execution Date vs System Update Time.
      * Idempotent: never double-books on midnight/weekend/holiday runs or repeated runs.
      * T+1 Open execution: signals generated after EOD close (signal_date = T) remain PENDING
        until the next trading day's real Open (exec_date = T+1).
      * Data anomaly guard: preserves prior valid ledger and flags STALE if feed fails.
      * Strategy versioning: never mutates historical track records of prior strategy versions.
"""

import json
import math
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pytz


DEFAULT_TIMING_THRESHOLDS = {
    # Frozen historical reference distributions from 2020-2026 audit (used only to map current valuation to causal tier)
    "med_price_p20_p80": [112.0, 117.0, 122.0, 127.0],
}


def resolve_market_session(now_cst: datetime, trade_dates: list = None) -> dict:
    """Resolve confirmed exchange market date & session phase in Asia/Shanghai timezone."""
    today_str = now_cst.strftime("%Y-%m-%d")
    hhmm = now_cst.hour * 100 + now_cst.minute

    if not trade_dates:
        # Try fetching SSE Composite daily dates as authoritative exchange calendar
        try:
            import akshare as ak
            cal = ak.stock_zh_a_hist_sina(symbol="sh000001")
            if cal is not None and not cal.empty:
                trade_dates = sorted(pd.to_datetime(cal["date"]).dt.strftime("%Y-%m-%d").unique().tolist())[-20:]
        except Exception:
            trade_dates = []

    if trade_dates:
        latest_cal = trade_dates[-1]
    else:
        latest_cal = today_str

    if latest_cal < today_str:
        # Pre-open or non-trading day: latest exchange bar is strictly closed EOD
        phase = "EOD_CONFIRMED"
        market_date = latest_cal
    else:
        # latest_cal == today_str
        if hhmm >= 1505:
            phase = "EOD_CONFIRMED"
            market_date = latest_cal
        elif hhmm >= 935:
            phase = "INTRADAY"
            market_date = latest_cal
        else:
            phase = "PREOPEN"
            market_date = trade_dates[-2] if len(trade_dates) >= 2 else latest_cal

    return {
        "market_date": market_date,
        "today_cst": today_str,
        "updated_at": now_cst.strftime("%Y-%m-%d %H:%M:%S CST"),
        "phase": phase,
        "trade_dates_tail": trade_dates[-5:] if trade_dates else [market_date],
    }


def _norm_rating(rt: str) -> str:
    return str(rt or "").strip().upper().replace("＋", "+").replace("－", "-")


def _filter_pool(evaluated: list, pool_key: str, risk_filter: bool) -> list:
    grade_sets = {
        "all": None,
        "aa_minus": {"AA-", "AA", "AA+", "AAA"},
        "aa": {"AA", "AA+", "AAA"},
        "aa_plus": {"AA+", "AAA"},
        "aaa": {"AAA"},
    }
    gs = grade_sets.get(pool_key)
    out = []
    for b in evaluated:
        if gs is not None and _norm_rating(b.get("rating")) not in gs:
            continue
        if risk_filter and b.get("is_risk_excluded"):
            continue
        out.append(b)
    return out


def _score_and_select(pool_items: list, w_lsm: float, fixed_n: int, sell_mult: float, current_held: set, weight_mode: str = "equal", max_single_w: float = 0.15) -> tuple:
    """Compute cross-sectional percentile rank scores & target portfolio weights strictly from EOD signals."""
    if not pool_items:
        return {}, []
    n = len(pool_items)
    df = pd.DataFrame(pool_items)
    p_lsm = df["undervaluation"].rank(pct=True, ascending=True).values
    p_dl = df["double_low"].rank(pct=True, ascending=False).values
    scores = w_lsm * p_lsm + (1.0 - w_lsm) * p_dl
    order = np.argsort(-(scores + 1e-9 * p_lsm), kind="mergesort")
    ranked = [pool_items[i] for i in order]

    if fixed_n == -1:  # Equal-weight entire pool (theoretical benchmark)
        k = len(ranked)
        return {b["ticker"]: round(1.0 / k, 6) for b in ranked}, ranked

    k_tgt = min(n, fixed_n) if fixed_n > 0 else min(n, max(5, int(round(n * 0.20))))
    if sell_mult <= 1.01:
        picks = ranked[:k_tgt]
    else:
        k_sell = min(n, max(k_tgt, int(round(k_tgt * sell_mult))))
        kept = [b for b in ranked[:k_sell] if b["ticker"] in current_held]
        picks = list(kept)
        kset = {b["ticker"] for b in kept}
        for b in ranked:
            if len(picks) >= k_tgt:
                break
            if b["ticker"] not in kset:
                picks.append(b)
                kset.add(b["ticker"])

    if not picks:
        return {}, []

    if weight_mode == "inv_vol":
        raw = np.array([1.0 / max(15.0, float(b.get("sigma") or 28.0)) for b in picks], dtype=np.float64)
    else:
        raw = np.ones(len(picks), dtype=np.float64)
    raw /= raw.sum()

    if max_single_w < 0.999 and len(picks) * max_single_w >= 1.0 - 1e-6:
        for _ in range(10):
            over = raw > max_single_w
            if not np.any(over):
                break
            excess = np.sum(raw[over] - max_single_w)
            raw[over] = max_single_w
            under = ~over
            if np.any(under) and raw[under].sum() > 0:
                raw[under] += excess * (raw[under] / raw[under].sum())

    weights = {b["ticker"]: round(float(raw[i]), 6) for i, b in enumerate(picks)}
    return weights, picks


def _compute_causal_valuation_exposure(evaluated: list, timing_ref: dict) -> tuple:
    """Compute strictly causal valuation tier & target exposure (50%..100%) using historical distribution."""
    aam = _filter_pool(evaluated, "aa_minus", True)
    if not aam:
        return 0.80, {"note": "insufficient sample"}
    med_px = float(np.median([b["bond_price"] for b in aam]))
    med_prem = float(np.median([b["premium_rate"] for b in aam]))
    under_pct = float(np.mean([1.0 if b["undervaluation"] > 0 else 0.0 for b in aam]) * 100.0)

    # Compare against pre-computed historical baseline series if available
    hist_series = ((timing_ref or {}).get("history_series") or {}).get("aa_minus") or {}
    px_hist = [x for x in (hist_series.get("med_price") or []) if isinstance(x, (int, float))]
    prem_hist = [x for x in (hist_series.get("med_prem") or []) if isinstance(x, (int, float))]
    px_pct = float(np.mean(np.array(px_hist) <= med_px) * 100.0) if len(px_hist) >= 30 else 50.0
    prem_pct = float(np.mean(np.array(prem_hist) <= med_prem) * 100.0) if len(prem_hist) >= 30 else 50.0
    comp_pct = round(0.5 * px_pct + 0.5 * prem_pct, 1)

    if comp_pct < 20.0:
        exp = 1.00
    elif comp_pct < 40.0:
        exp = 0.90
    elif comp_pct < 60.0:
        exp = 0.80
    elif comp_pct < 80.0:
        exp = 0.65
    else:
        exp = 0.50
    return exp, {"med_price": round(med_px, 2), "med_prem": round(med_prem, 2), "comp_pct": comp_pct, "under_pct": round(under_pct, 1)}


def _account_specs() -> list:
    return [
        # Primary Comparable Cohort: AA-及以上 · 因果剔除 ST/违约/低流
        {
            "account_id": "aam_lsm_v1", "cohort": "aa_minus", "cohort_label": "AA-及以上 (因果风控)",
            "name": "原始 Four-Region LSM (v1)", "version": "1.0",
            "pool": "aa_minus", "risk_filter": True, "w_lsm": 1.00, "freq": 5, "fixed_n": 0, "sell_mult": 1.0,
            "weight_mode": "equal", "max_single_w": 0.15, "dynamic_exposure": False, "execution_mode": "discrete_10lots",
            "desc": "冻结论文模型 100/0 · 5D · Top20% 等权 · 10张整数手实盘模拟",
        },
        {
            "account_id": "aam_balanced_v1", "cohort": "aa_minus", "cohort_label": "AA-及以上 (因果风控)",
            "name": "Enhanced LSM Balanced (v1)", "version": "1.0",
            "pool": "aa_minus", "risk_filter": True, "w_lsm": 0.50, "freq": 10, "fixed_n": 20, "sell_mult": 1.5,
            "weight_mode": "equal", "max_single_w": 0.10, "dynamic_exposure": False, "execution_mode": "discrete_10lots",
            "desc": "P1 验证主推：50/50 混合 · 10D · 20/30缓冲 · 固定20只等权 (单券<=10%, 适配¥10万)",
        },
        {
            "account_id": "aam_balanced_dyn_v1", "cohort": "aa_minus", "cohort_label": "AA-及以上 (因果风控)",
            "name": "Balanced + 动态估值仓位 (v1)", "version": "1.0",
            "pool": "aa_minus", "risk_filter": True, "w_lsm": 0.50, "freq": 10, "fixed_n": 20, "sell_mult": 1.5,
            "weight_mode": "equal", "max_single_w": 0.10, "dynamic_exposure": True, "execution_mode": "discrete_10lots",
            "desc": "P2 验证扩展：Balanced 策略 + 滚动估值分位动态仓位 (50%–100%, 现金享2%货币年化)",
        },
        {
            "account_id": "aam_dl_v1", "cohort": "aa_minus", "cohort_label": "AA-及以上 (因果风控)",
            "name": "传统双低 (v1)", "version": "1.0",
            "pool": "aa_minus", "risk_filter": True, "w_lsm": 0.00, "freq": 5, "fixed_n": 0, "sell_mult": 1.0,
            "weight_mode": "equal", "max_single_w": 0.15, "dynamic_exposure": False, "execution_mode": "discrete_10lots",
            "desc": "传统价格+溢价率双低 · 5D · Top20% 等权 · 同口径对照",
        },
        {
            "account_id": "aam_ew_v1", "cohort": "aa_minus", "cohort_label": "AA-及以上 (因果风控)",
            "name": "AA- 同池等权基准 (v1)", "version": "1.0",
            "pool": "aa_minus", "risk_filter": True, "w_lsm": 1.00, "freq": 5, "fixed_n": -1, "sell_mult": 1.0,
            "weight_mode": "equal", "max_single_w": 1.00, "dynamic_exposure": False, "execution_mode": "theoretical_fractional",
            "desc": "理论全池等权参照 (因¥10万无法购齐210+只×10张，按理论零整权数追踪)",
        },
        # Defensive Cohort: AA及以上 · 因果剔除
        {
            "account_id": "aa_defensive_v1", "cohort": "aa", "cohort_label": "AA及以上 (因果风控)",
            "name": "Enhanced LSM Defensive (v1)", "version": "1.0",
            "pool": "aa", "risk_filter": True, "w_lsm": 0.50, "freq": 10, "fixed_n": 20, "sell_mult": 1.5,
            "weight_mode": "inv_vol", "max_single_w": 0.10, "dynamic_exposure": False, "execution_mode": "discrete_10lots",
            "desc": "P1 验证防御版：AA以上 50/50 · 10D · 20/30缓冲 · 固定20只逆波加权 (单券<=10%)",
        },
        {
            "account_id": "aa_lsm_v1", "cohort": "aa", "cohort_label": "AA及以上 (因果风控)",
            "name": "AA 原始 LSM 对照 (v1)", "version": "1.0",
            "pool": "aa", "risk_filter": True, "w_lsm": 1.00, "freq": 5, "fixed_n": 0, "sell_mult": 1.0,
            "weight_mode": "equal", "max_single_w": 0.15, "dynamic_exposure": False, "execution_mode": "discrete_10lots",
            "desc": "AA及以上池原始冻结 LSM 对照组",
        },
        # Full Market Baseline Reference
        {
            "account_id": "all_lsm_v1", "cohort": "all", "cohort_label": "全市场 (原始基准)",
            "name": "全市场 原始 LSM 基准 (v1)", "version": "1.0",
            "pool": "all", "risk_filter": False, "w_lsm": 1.00, "freq": 5, "fixed_n": 0, "sell_mult": 1.0,
            "weight_mode": "equal", "max_single_w": 0.15, "dynamic_exposure": False, "execution_mode": "discrete_10lots",
            "desc": "全市场不剔除原始论文 LSM 连续对照基准",
        },
    ]


def _init_account(spec: dict, genesis_date: str) -> dict:
    return {
        "account_id": spec["account_id"],
        "cohort": spec["cohort"],
        "cohort_label": spec["cohort_label"],
        "name": spec["name"],
        "version": spec["version"],
        "execution_mode": spec["execution_mode"],
        "desc": spec["desc"],
        "frozen_spec": {k: spec[k] for k in ("pool", "risk_filter", "w_lsm", "freq", "fixed_n", "sell_mult", "weight_mode", "max_single_w", "dynamic_exposure")},
        "genesis_date": genesis_date,
        "initial_cash": 100_000.0,
        "cash": 100_000.0,
        "total_nav": 100_000.0,
        "peak_nav": 100_000.0,
        "max_dd_pct": 0.0,
        "total_fees": 0.0,
        "total_traded_rmb": 0.0,
        "last_market_date": None,
        "last_signal_date": None,
        "trading_days_since_signal": 0,
        "holdings": {},       # ticker -> {name, rating, qty, avg_cost, last_price, weight_pct, market_val, entry_date}
        "pending_signal": None,
        "daily_navs": [],     # [{date, nav, cash, pos_val, invested_pct, ret_pct, max_dd_pct, holdings_count, target_exposure}]
        "trades": [],         # latest 300 executions
        "signals_log": [],    # latest 120 signals
        "cash_log": [],       # latest 200 cash ledger entries
    }


def update_paper_accounts(dist_dir: Path, evaluated: list, session: dict, timing_ref: dict = None) -> dict:
    """Update all ¥100,000 paper trading accounts cleanly, idempotently, and with feed validation."""
    out_path = dist_dir / "paper_accounts.json"
    existing = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    market_date = session["market_date"]
    phase = session["phase"]
    updated_at = session["updated_at"]

    # Feed integrity guard
    valid_prices = [b["bond_price"] for b in evaluated if isinstance(b.get("bond_price"), (int, float)) and b["bond_price"] > 10]
    feed_ok = len(evaluated) >= 150 and len(valid_prices) >= 150 and float(np.median(valid_prices)) > 60.0

    if not feed_ok:
        if existing and isinstance(existing.get("accounts"), dict):
            existing["data_status"] = "STALE_PRESERVED"
            existing["stale_reason"] = f"数据源校验异常(有效转债={len(evaluated)})，已冻结保留上一有效账本，严禁虚构成交。"
            existing["last_check_at"] = updated_at
            out_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            return existing

    quote_map = {b["ticker"]: b for b in evaluated}
    specs = _account_specs()
    accounts = existing.get("accounts") if isinstance(existing.get("accounts"), dict) else {}

    val_exposure, val_diag = _compute_causal_valuation_exposure(evaluated, timing_ref)
    comm_rate = 0.0005
    slip_rate = 0.0005
    min_comm = 5.0

    for sp in specs:
        aid = sp["account_id"]
        if aid not in accounts:
            accounts[aid] = _init_account(sp, market_date)
        acc = accounts[aid]

        # Idempotent guard: if market_date already finalized EOD, only refresh preview valuations if needed without double-booking
        already_booked = acc.get("last_market_date") == market_date

        # Step 1: Execute PENDING signal strictly when current market_date > pending_signal.signal_date AND session has open quotes
        pend = acc.get("pending_signal")
        if pend and pend.get("status") == "PENDING" and pend.get("signal_date") and pend["signal_date"] < market_date and phase in ("INTRADAY", "EOD_CONFIRMED"):
            tgt_weights = pend.get("target_weights") or {}
            tgt_exp = float(pend.get("target_exposure", 1.0))

            if acc["execution_mode"] == "theoretical_fractional":
                # Rebalance theoretical fractional holdings at T+1 Open
                cur_nav = acc["cash"]
                for tk, h in acc["holdings"].items():
                    q = quote_map.get(tk)
                    px_open = float(q.get("open_price") or q["bond_price"]) if q else float(h["last_price"])
                    cur_nav += float(h["qty"]) * px_open
                old_w = {}
                for tk, h in acc["holdings"].items():
                    q = quote_map.get(tk)
                    px_open = float(q.get("open_price") or q["bond_price"]) if q else float(h["last_price"])
                    old_w[tk] = (float(h["qty"]) * px_open) / max(1.0, cur_nav)
                two_way_to = sum(abs(tgt_weights.get(k, 0.0) - old_w.get(k, 0.0)) for k in (set(tgt_weights) | set(old_w)))
                fee = cur_nav * two_way_to * (comm_rate + slip_rate)
                net_nav = max(0.0, cur_nav - fee)
                new_h = {}
                for tk, tw in tgt_weights.items():
                    q = quote_map.get(tk)
                    if not q:
                        continue
                    px_open = float(q.get("open_price") or q["bond_price"])
                    qty_f = (net_nav * tgt_exp * tw) / max(1.0, px_open)
                    new_h[tk] = {
                        "ticker": tk, "name": q["name"], "rating": q.get("rating", ""),
                        "qty": round(qty_f, 4), "avg_cost": round(px_open, 3), "last_price": round(q["bond_price"], 3),
                        "entry_date": market_date,
                    }
                acc["holdings"] = new_h
                acc["cash"] = round(net_nav * max(0.0, 1.0 - tgt_exp), 2)
                acc["total_fees"] = round(acc["total_fees"] + fee, 2)
                acc["total_traded_rmb"] = round(acc["total_traded_rmb"] + cur_nav * two_way_to, 2)
                acc["cash_log"] = ([{"date": market_date, "type": "REBAL_THEO", "delta": round(-fee, 2), "cash_after": acc["cash"], "note": f"等权再平衡理论摩擦(换手{two_way_to*100:.1f}%)"}] + acc["cash_log"])[:200]
            else:
                # Real 10-bond discrete lot execution at T+1 Open
                # 1A: Liquidate delisted / inactive positions first
                for tk in list(acc["holdings"].keys()):
                    if tk not in quote_map:
                        h = acc["holdings"].pop(tk)
                        px_exit = float(h["last_price"])
                        gross = h["qty"] * px_exit
                        fee = max(min_comm, gross * comm_rate) + gross * slip_rate
                        acc["cash"] = round(acc["cash"] + max(0.0, gross - fee), 2)
                        acc["total_fees"] = round(acc["total_fees"] + fee, 2)
                        acc["total_traded_rmb"] = round(acc["total_traded_rmb"] + gross, 2)
                        acc["trades"] = ([{"date": market_date, "signal_date": pend["signal_date"], "ticker": tk, "name": h["name"], "side": "SELL_DELIST", "lots": int(h["qty"] // 10), "qty": int(h["qty"]), "price": px_exit, "gross": round(gross, 2), "fee": round(fee, 2)}] + acc["trades"])[:300]
                        acc["cash_log"] = ([{"date": market_date, "type": "SELL_DELIST", "delta": round(gross - fee, 2), "cash_after": acc["cash"], "note": f"退市/强赎结算 {tk} {h['name']}"}] + acc["cash_log"])[:200]

                # Mark current tradable vs suspended at T+1 Open
                nav_open = acc["cash"]
                tradable_tks = []
                for tk, h in acc["holdings"].items():
                    q = quote_map[tk]
                    op = float(q.get("open_price") or q["bond_price"])
                    vol = float(q.get("volume") or 1.0)
                    nav_open += h["qty"] * op
                    if op > 1.0 and vol > 0:
                        tradable_tks.append(tk)

                invest_budget = nav_open * tgt_exp
                # 1B: Sells / Trims
                for tk in tradable_tks:
                    q = quote_map[tk]
                    op = float(q.get("open_price") or q["bond_price"])
                    cur_q = int(acc["holdings"][tk]["qty"])
                    tgt_rmb = invest_budget * float(tgt_weights.get(tk, 0.0))
                    tgt_lots = int(tgt_rmb // (op * 10.0))
                    tgt_q = tgt_lots * 10
                    if cur_q > tgt_q:
                        sell_q = cur_q - tgt_q
                        gross = sell_q * op
                        fee = max(min_comm, gross * comm_rate) + gross * slip_rate
                        acc["cash"] = round(acc["cash"] + gross - fee, 2)
                        acc["total_fees"] = round(acc["total_fees"] + fee, 2)
                        acc["total_traded_rmb"] = round(acc["total_traded_rmb"] + gross, 2)
                        acc["trades"] = ([{"date": market_date, "signal_date": pend["signal_date"], "ticker": tk, "name": q["name"], "side": "SELL", "lots": sell_q // 10, "qty": sell_q, "price": round(op, 3), "gross": round(gross, 2), "fee": round(fee, 2)}] + acc["trades"])[:300]
                        acc["cash_log"] = ([{"date": market_date, "type": "SELL", "delta": round(gross - fee, 2), "cash_after": acc["cash"], "note": f"卖出 {tk} {q['name']} {sell_q}张 @ ¥{op:.3f}"}] + acc["cash_log"])[:200]
                        if tgt_q > 0:
                            acc["holdings"][tk]["qty"] = tgt_q
                        else:
                            acc["holdings"].pop(tk, None)

                # 1C: Buys
                for tk, tw in tgt_weights.items():
                    q = quote_map.get(tk)
                    if not q:
                        continue
                    op = float(q.get("open_price") or q["bond_price"])
                    vol = float(q.get("volume") or 1.0)
                    if op <= 1.0 or vol <= 0:
                        continue  # Suspended/untradable at T+1 Open -> skip cleanly
                    cur_q = int(acc["holdings"].get(tk, {}).get("qty", 0))
                    tgt_rmb = min(invest_budget * float(tw), nav_open * sp["max_single_w"])
                    tgt_lots = int(tgt_rmb // (op * 10.0))
                    tgt_q = tgt_lots * 10
                    if tgt_q > cur_q:
                        need_lots = (tgt_q - cur_q) // 10
                        lot_px = op * 10.0
                        afford_lots = int(max(0.0, acc["cash"] - min_comm - 5.0) // (lot_px * (1.0 + comm_rate + slip_rate)))
                        buy_lots = min(need_lots, afford_lots)
                        if buy_lots > 0:
                            buy_q = buy_lots * 10
                            gross = buy_q * op
                            fee = max(min_comm, gross * comm_rate) + gross * slip_rate
                            if acc["cash"] >= gross + fee:
                                acc["cash"] = round(acc["cash"] - gross - fee, 2)
                                acc["total_fees"] = round(acc["total_fees"] + fee, 2)
                                acc["total_traded_rmb"] = round(acc["total_traded_rmb"] + gross, 2)
                                prev = acc["holdings"].get(tk)
                                if prev:
                                    new_cost = (prev["avg_cost"] * prev["qty"] + gross + fee) / (prev["qty"] + buy_q)
                                    prev["qty"] += buy_q
                                    prev["avg_cost"] = round(new_cost, 3)
                                else:
                                    acc["holdings"][tk] = {
                                        "ticker": tk, "name": q["name"], "rating": q.get("rating", ""),
                                        "qty": buy_q, "avg_cost": round((gross + fee) / buy_q, 3),
                                        "last_price": round(q["bond_price"], 3), "entry_date": market_date,
                                    }
                                acc["trades"] = ([{"date": market_date, "signal_date": pend["signal_date"], "ticker": tk, "name": q["name"], "side": "BUY", "lots": buy_lots, "qty": buy_q, "price": round(op, 3), "gross": round(gross, 2), "fee": round(fee, 2)}] + acc["trades"])[:300]
                                acc["cash_log"] = ([{"date": market_date, "type": "BUY", "delta": round(-(gross + fee), 2), "cash_after": acc["cash"], "note": f"买入 {tk} {q['name']} {buy_q}张 @ ¥{op:.3f}"}] + acc["cash_log"])[:200]

            pend["status"] = "EXECUTED"
            pend["exec_date"] = market_date
            for slog in acc.get("signals_log", []):
                if slog.get("signal_date") == pend.get("signal_date"):
                    slog["status"] = "EXECUTED"
                    slog["exec_date"] = market_date

        # Refresh live holding MTM values when INTRADAY (without locking EOD daily_navs)
        if phase == "INTRADAY":
            pv_live = 0.0
            for tk, h in acc["holdings"].items():
                q = quote_map.get(tk)
                c_px = float(q["bond_price"]) if q else float(h["last_price"])
                h["last_price"] = round(c_px, 3)
                mv = h["qty"] * c_px
                h["market_val"] = round(mv, 2)
                pv_live += mv
            acc["total_nav"] = round(acc["cash"] + pv_live, 2)
            for h in acc["holdings"].values():
                h["weight_pct"] = round(h["market_val"] / max(1.0, acc["total_nav"]) * 100.0, 2)

        # Step 2: EOD Valuation & Signal Generation (idempotent per market_date, ONLY on EOD_CONFIRMED)
        if phase == "EOD_CONFIRMED" and not already_booked:
            # Overnight 2.0% annual yield on cash (excluding Day-0 genesis)
            if acc["daily_navs"] and acc["cash"] > 10.0:
                interest = round(acc["cash"] * (0.02 / 252.0), 2)
                if interest > 0:
                    acc["cash"] = round(acc["cash"] + interest, 2)

            pos_val = 0.0
            for tk, h in acc["holdings"].items():
                q = quote_map.get(tk)
                c_px = float(q["bond_price"]) if q else float(h["last_price"])
                h["last_price"] = round(c_px, 3)
                if q:
                    h["rating"] = q.get("rating", h.get("rating", ""))
                mv = h["qty"] * c_px
                h["market_val"] = round(mv, 2)
                pos_val += mv

            tot_nav = round(acc["cash"] + pos_val, 2)
            for h in acc["holdings"].values():
                h["weight_pct"] = round(h["market_val"] / max(1.0, tot_nav) * 100.0, 2)

            acc["total_nav"] = tot_nav
            acc["peak_nav"] = round(max(acc["peak_nav"], tot_nav), 2)
            dd = ((tot_nav / max(1.0, acc["peak_nav"])) - 1.0) * 100.0
            acc["max_dd_pct"] = round(min(acc["max_dd_pct"], dd), 2)
            prev_nav = acc["daily_navs"][-1]["nav"] if acc["daily_navs"] else acc["initial_cash"]
            d_ret = ((tot_nav / max(1.0, prev_nav)) - 1.0) * 100.0 if acc["daily_navs"] else 0.0

            cur_exp = val_exposure if sp["dynamic_exposure"] else 1.0
            acc["daily_navs"].append({
                "date": market_date,
                "nav": tot_nav,
                "cash": round(acc["cash"], 2),
                "pos_val": round(pos_val, 2),
                "invested_pct": round(pos_val / max(1.0, tot_nav) * 100.0, 1),
                "ret_pct": round(tot_nav / acc["initial_cash"] * 100.0 - 100.0, 2),
                "daily_ret_pct": round(d_ret, 3),
                "max_dd_pct": acc["max_dd_pct"],
                "holdings_count": len(acc["holdings"]),
                "target_exposure_pct": round(cur_exp * 100.0, 0),
            })
            acc["daily_navs"] = acc["daily_navs"][-500:]
            acc["last_market_date"] = market_date

            # Determine if new EOD signal should be queued for next trading day T+1 Open
            since_sig = int(acc.get("trading_days_since_signal", 0)) + 1 if acc.get("last_signal_date") else sp["freq"]
            is_rebal_due = (since_sig >= sp["freq"]) or (len(acc["holdings"]) == 0 and acc.get("pending_signal") is None)
            if is_rebal_due:
                sub_pool = _filter_pool(evaluated, sp["pool"], sp["risk_filter"])
                w_map, picks = _score_and_select(
                    sub_pool, sp["w_lsm"], sp["fixed_n"], sp["sell_mult"],
                    set(acc["holdings"].keys()), sp["weight_mode"], sp["max_single_w"],
                )
                sig_rec = {
                    "signal_date": market_date,
                    "status": "PENDING",
                    "target_exec": "下一交易日开盘 (T+1 Open)",
                    "target_exposure": cur_exp,
                    "n_targets": len(w_map),
                    "target_weights": w_map,
                    "top_targets": [
                        {"ticker": b["ticker"], "name": b["name"], "rating": b.get("rating", ""), "close": b["bond_price"], "underval": b["undervaluation"], "dl": b["double_low"], "weight_pct": round(w_map[b["ticker"]] * 100.0, 2)}
                        for b in picks[:25]
                    ],
                }
                acc["pending_signal"] = sig_rec
                acc["last_signal_date"] = market_date
                acc["trading_days_since_signal"] = 0
                acc["signals_log"] = ([{"signal_date": market_date, "n_targets": len(w_map), "target_exposure_pct": round(cur_exp * 100.0, 0), "status": "PENDING", "preview": ", ".join(b["name"] for b in picks[:6])}] + acc["signals_log"])[:120]
            else:
                acc["trading_days_since_signal"] = since_sig

    payload = {
        "schema_version": "1.0",
        "data_status": "OK",
        "session": session,
        "valuation_signal_diag": val_diag,
        "execution_rules": {
            "initial_capital": 100_000.0,
            "min_trade_unit": "1手 = 10张 (实际整手截断)",
            "commission": "0.05% (单笔最低 ¥5)",
            "slippage": "0.05%",
            "cash_yield_ann": "2.00% (日计)",
            "execution_timing": "T日收盘生成信号 -> 下一交易日(T+1)开盘价执行，严禁使用未来数据或非交易日重复记账",
        },
        "accounts": accounts,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return payload
