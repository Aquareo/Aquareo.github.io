import json
import math
import re
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pytz

try:
    from cb_pricing_model import price_ccb_four_region_lsm
except ImportError:
    from .cb_pricing_model import price_ccb_four_region_lsm


# Frozen Paper Model Parameters (arXiv:2409.06496v7) - Do NOT tune post-hoc
FROZEN_PARAMS = {
    "r": 0.025,
    "q": 0.0,
    "call_ratio": 1.30,
    "put_ratio": 0.70,
    "p_reset": 0.80,
    "B": 108.0,
    "call_settle": 103.0,
    "put_pay": 101.0,
    "n_steps": 24,
    "M": 1200,
    "seed": 42,
}
SPECULATIVE_GRADES = {"BBB+", "BBB", "BBB-", "BB+", "BB", "BB-", "B+", "B", "B-", "CCC", "CC", "C"}
LOW_LIQ_WAN = 500.0  # ¥500万元 daily turnover threshold


def sanitize_json(obj):
    """Ensure strict RFC-8259 JSON compliance: convert NaN / Inf to None (null)."""
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


def norm_rating(val) -> str:
    """Normalize Chinese bond rating text (e.g. 'AA+sti' -> 'AA+', 'nan' -> '')."""
    s = str(val or "").strip().upper().replace("＋", "+").replace("－", "-")
    if s in ("", "NAN", "NONE", "NULL", "-", "--"):
        return ""
    return re.sub(r"STI$", "", s).strip()


def get_default_bonds_pool():
    return [
        {"ticker": "110075", "name": "南航转债", "stock_code": "600029", "stock_name": "南方航空", "stock_price": 5.92, "conv_price": 6.24, "bond_price": 128.40, "maturity_years": 2.8, "rating": "AAA", "amt_wan": 5000.0},
        {"ticker": "127025", "name": "冀东转债", "stock_code": "000401", "stock_name": "冀东水泥", "stock_price": 5.12, "conv_price": 5.88, "bond_price": 111.20, "maturity_years": 1.9, "rating": "AAA", "amt_wan": 3200.0},
        {"ticker": "113049", "name": "长汽转债", "stock_code": "601633", "stock_name": "长城汽车", "stock_price": 26.80, "conv_price": 34.20, "bond_price": 114.50, "maturity_years": 3.2, "rating": "AAA", "amt_wan": 4100.0},
        {"ticker": "123107", "name": "温氏转债", "stock_code": "300498", "stock_name": "温氏股份", "stock_price": 17.50, "conv_price": 19.80, "bond_price": 119.30, "maturity_years": 2.6, "rating": "AAA", "amt_wan": 8000.0},
    ]


def classify_region(S0: float, C0: float):
    k0 = 1.30 * C0
    ptr0 = 0.70 * C0
    if S0 > k0:
        return "强赎提示区", "R1", "正股超 130% 强赎触发线"
    if S0 > C0:
        return "高弹性转股区", "R2", "正股高于转股价，股性凸显"
    if S0 > ptr0:
        return "平衡抗跌区", "R3", "兼顾纯债底与转股期权"
    return "下修博弈区", "R4", "低于 70% 回售线，博弈下修"


def fetch_active_pool(max_bonds: int = 500):
    """Fetch ALL active, liquid, already-listed exchange CBs with live ratings & turnover."""
    import akshare as ak

    pool = {}
    rating_lookup = {}
    now_ts = pd.Timestamp.now()

    try:
        df_em = ak.bond_cb_list_em()
        if df_em is not None and not df_em.empty:
            for _, r in df_em.iterrows():
                c_key = str(r.get("债券代码", "")).strip()
                rt_val = norm_rating(r.get("信用评级"))
                if c_key and rt_val:
                    rating_lookup[c_key] = rt_val
            df_em["上市时间"] = pd.to_datetime(df_em["上市时间"], errors="coerce")
            df_em["债现价"] = pd.to_numeric(df_em["债现价"], errors="coerce")
            df_em["转股价"] = pd.to_numeric(df_em["转股价"], errors="coerce")
            df_em["正股价"] = pd.to_numeric(df_em["正股价"], errors="coerce")
            listed = df_em[
                df_em["上市时间"].notna()
                & (df_em["上市时间"] <= now_ts - pd.Timedelta(days=25))
                & (df_em["债现价"] > 10.0)
                & (df_em["债现价"] < 600.0)
                & ((df_em["债现价"] - 100.0).abs() > 1e-4)
                & (df_em["转股价"] > 0)
                & (df_em["正股价"] > 0)
            ]
            for _, r in listed.iterrows():
                if len(pool) >= max_bonds:
                    break
                code = str(r.get("债券代码", "")).strip()
                if not code or not code.startswith(("11", "12", "13")) or code.startswith("132"):
                    continue
                elapsed = (now_ts - r["上市时间"]).days / 365.25
                rem_years = max(0.30, round(6.0 - elapsed, 2))
                pool[code] = {
                    "ticker": code,
                    "name": str(r.get("债券简称", "")),
                    "stock_code": str(r.get("正股代码", "")),
                    "stock_name": str(r.get("正股简称", "")),
                    "stock_price": float(r["正股价"]),
                    "conv_price": float(r["转股价"]),
                    "bond_price": float(r["债现价"]),
                    "maturity_years": rem_years,
                    "rating": rating_lookup.get(code, norm_rating(r.get("信用评级"))),
                    "amt_wan": None,
                }
    except Exception as e:
        print(f"Notice: bond_cb_list_em fetch warning: {e}")

    try:
        df_jsl = ak.bond_cb_jsl()
        if df_jsl is not None and not df_jsl.empty:
            for _, r in df_jsl.iterrows():
                code = str(r.get("代码", "")).strip()
                if not code or code.startswith("132"):
                    continue
                rem = float(r.get("剩余年限", 0) or 0)
                price = float(r.get("现价", 0) or 0)
                c_price = float(r.get("转股价", 0) or 0)
                s_price = float(r.get("正股价", 0) or 0)
                amt_wan = pd.to_numeric(r.get("成交额"), errors="coerce")
                amt_val = round(float(amt_wan), 1) if pd.notna(amt_wan) else None
                if rem < 0.1 or price <= 10 or abs(price - 100.0) < 1e-4 or c_price <= 0 or s_price <= 0:
                    continue
                jsl_rating = norm_rating(r.get("评级") or r.get("信用评级") or "")
                if code in pool:
                    pool[code]["maturity_years"] = round(rem, 2)
                    if amt_val is not None:
                        pool[code]["amt_wan"] = amt_val
                    if not pool[code].get("rating") and jsl_rating:
                        pool[code]["rating"] = jsl_rating
                elif len(pool) < max_bonds:
                    pool[code] = {
                        "ticker": code,
                        "name": str(r.get("转债名称", "")),
                        "stock_code": str(r.get("正股代码", "")),
                        "stock_name": str(r.get("正股名称", "")),
                        "stock_price": s_price,
                        "conv_price": c_price,
                        "bond_price": price,
                        "maturity_years": round(rem, 2),
                        "rating": rating_lookup.get(code, jsl_rating),
                        "amt_wan": amt_val,
                    }
    except Exception as e:
        print(f"Notice: bond_cb_jsl fetch warning: {e}")

    # Enrich live Open price, Volume, and Full-Pool RMB Turnover (万元) via Sina realtime in 1 fast call
    try:
        df_rt = ak.bond_cb_realtime_sina()
        if df_rt is not None and not df_rt.empty:
            for _, r in df_rt.iterrows():
                code = str(r.get("code", "")).strip()
                if code in pool:
                    op = pd.to_numeric(r.get("open"), errors="coerce")
                    vol = pd.to_numeric(r.get("volume"), errors="coerce")
                    amt = pd.to_numeric(r.get("amount"), errors="coerce")
                    trd = pd.to_numeric(r.get("trade"), errors="coerce")
                    if pd.notna(op) and float(op) > 1.0:
                        pool[code]["open_price"] = round(float(op), 3)
                    if pd.notna(vol):
                        pool[code]["volume"] = float(vol)
                    if pd.notna(amt) and float(amt) >= 0:
                        pool[code]["amt_wan"] = round(float(amt) / 10000.0, 1)
                    if pd.notna(trd) and float(trd) > 10.0:
                        pool[code]["bond_price"] = float(trd)
    except Exception as e:
        print(f"Notice: bond_cb_realtime_sina fetch warning: {e}")

    # Enrich live remaining scale (亿元) for screener display only (never backfilled into history)
    try:
        df_rd = ak.bond_cb_redeem_jsl()
        if df_rd is not None and not df_rd.empty:
            for _, r in df_rd.iterrows():
                code = str(r.get("代码", "")).strip()
                rs = pd.to_numeric(r.get("剩余规模"), errors="coerce")
                if code in pool and pd.notna(rs) and float(rs) > 0:
                    pool[code]["rem_scale"] = round(float(rs), 2)
    except Exception:
        pass

    if not pool:
        return get_default_bonds_pool()
    return list(pool.values())[:max_bonds]


def fetch_stock_vol_map(stock_codes: list, fallback_sigmas: dict) -> dict:
    """Fetch 60d trailing stock realized volatilities concurrently; preserves prior values if throttled."""
    from concurrent.futures import ThreadPoolExecutor
    import akshare as ak

    vol_map = dict(fallback_sigmas)
    uniq = sorted({sc for sc in stock_codes if sc and len(sc) == 6})

    def _one(sc):
        pref = "sh" if sc.startswith("6") else "sz"
        try:
            df = ak.stock_zh_a_hist_sina(symbol=pref + sc)
            if df is not None and len(df) >= 20:
                sub = pd.to_numeric(df["close"], errors="coerce").dropna().tail(61)
                log_ret = np.log(sub / sub.shift(1)).dropna()
                log_ret = log_ret[log_ret.abs() <= 0.28]
                if len(log_ret) >= 10:
                    vol = float(log_ret.std() * np.sqrt(252.0))
                    return sc, round(max(0.15, min(0.70, vol)), 4)
        except Exception:
            pass
        return sc, None

    with ThreadPoolExecutor(max_workers=4) as ex:
        for sc, val in ex.map(_one, uniq):
            if val is not None:
                vol_map[sc] = val
    return vol_map


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 4 or x.nunique() < 2 or y.nunique() < 2:
        return 0.0
    val = x.rank().corr(y.rank())
    return round(float(val), 3) if pd.notna(val) else 0.0


def _step_pool_ledger(pl: dict, pool_eval: list, current_map: dict, date_str: str, min_k: int, reset_needed: bool) -> dict:
    """Advance a single simulated Top-20% portfolio ledger one market snapshot forward."""
    if not isinstance(pl, dict):
        pl = {}
    pl.setdefault("rebalance_freq_days", 5)
    pl.setdefault("one_way_cost", 0.001)
    pl.setdefault("nav_net", 1.0)
    pl.setdefault("last_rebal_date", date_str)
    pl.setdefault("holdings", [])
    pl.setdefault("history", [])
    pl.setdefault("frozen_since", pl["history"][0]["date"] if pl["history"] else date_str)

    prev_hist = [h for h in pl.get("history", []) if h.get("date") != date_str]
    cur_holdings = pl.get("holdings", [])
    if not cur_holdings or (reset_needed and not prev_hist):
        k_pick = min(len(pool_eval), max(min_k, int(round(len(pool_eval) * 0.20))))
        picks = sorted(pool_eval, key=lambda x: x["undervaluation"], reverse=True)[:k_pick]
        w = 1.0 / len(picks) if picks else 0.0
        pl["holdings"] = [
            {
                "ticker": b["ticker"],
                "name": b["name"],
                "rating": b.get("rating", ""),
                "weight": round(w, 6),
                "entry_date": date_str,
                "entry_price": b["bond_price"],
                "last_price": b["bond_price"],
            }
            for b in picks
        ]
        pl["last_rebal_date"] = date_str
        pl["frozen_since"] = date_str
        pl["nav_net"] = 1.0
        pl["history"] = [{"date": date_str, "nav_net": 1.0, "holdings_count": len(picks), "delisted_exits": 0, "rebalanced": True}]
    elif len(prev_hist) > 0:
        last_nav = float(prev_hist[-1]["nav_net"])
        port_ret = 0.0
        delisted_exits = 0
        surviving = []
        for h in cur_holdings:
            tk = h["ticker"]
            w = float(h["weight"])
            lp = float(h["last_price"])
            if tk in current_map and current_map[tk]["bond_price"] > 0:
                np_px = float(current_map[tk]["bond_price"])
                port_ret += w * ((np_px - lp) / lp)
                h["last_price"] = np_px
                h["rating"] = current_map[tk].get("rating", h.get("rating", ""))
                surviving.append(h)
            else:
                delisted_exits += 1
                port_ret -= w * 0.001

        nav_today = last_nav * (1.0 + port_ret)
        days_since = (pd.Timestamp(date_str) - pd.Timestamp(pl["last_rebal_date"])).days
        did_rebal = False
        if days_since >= 7 and pool_eval:
            k_pick = min(len(pool_eval), max(min_k, int(round(len(pool_eval) * 0.20))))
            picks = sorted(pool_eval, key=lambda x: x["undervaluation"], reverse=True)[:k_pick]
            w_new = 1.0 / len(picks) if picks else 0.0
            old_w = {h["ticker"]: h["weight"] for h in surviving}
            new_w = {b["ticker"]: w_new for b in picks}
            to = sum(abs(new_w.get(k, 0.0) - old_w.get(k, 0.0)) for k in (set(old_w) | set(new_w)))
            nav_today *= (1.0 - to * 0.001)
            pl["holdings"] = [
                {
                    "ticker": b["ticker"],
                    "name": b["name"],
                    "rating": b.get("rating", ""),
                    "weight": round(w_new, 6),
                    "entry_date": date_str,
                    "entry_price": b["bond_price"],
                    "last_price": b["bond_price"],
                }
                for b in picks
            ]
            pl["last_rebal_date"] = date_str
            did_rebal = True
        else:
            pl["holdings"] = surviving

        pl["nav_net"] = round(nav_today, 4)
        prev_hist.append({
            "date": date_str,
            "nav_net": round(nav_today, 4),
            "holdings_count": len(pl["holdings"]),
            "delisted_exits": delisted_exits,
            "rebalanced": did_rebal,
        })
        pl["history"] = prev_hist[-365:]
    return pl


def _calc_pool_comparisons(snapshots: list, current_map: dict, date_str: str, allowed_grades=None) -> list:
    live_records = []
    for past_snap in snapshots[:-1]:
        p_date = past_snap["date"]
        matched = []
        for ticker, info in past_snap["bonds"].items():
            if allowed_grades is not None:
                rt = norm_rating(info.get("rating") or current_map.get(ticker, {}).get("rating"))
                if rt not in allowed_grades:
                    continue
            if ticker in current_map and info["bond_price"] > 0:
                ret = round((current_map[ticker]["bond_price"] - info["bond_price"]) / info["bond_price"] * 100, 2)
                matched.append({"underval": info["undervaluation"], "double_low": info.get("double_low", 999.0), "ret": ret})
        if len(matched) >= 12:
            mdf = pd.DataFrame(matched)
            k = max(3, len(mdf) // 5)
            top_r = round(float(mdf.sort_values("underval", ascending=False).iloc[:k]["ret"].mean()), 2)
            bot_r = round(float(mdf.sort_values("underval", ascending=False).iloc[-k:]["ret"].mean()), 2)
            dl_r = round(float(mdf.sort_values("double_low", ascending=True).iloc[:k]["ret"].mean()), 2)
            mkt_r = round(float(mdf["ret"].mean()), 2)
            live_records.append({
                "from_date": p_date,
                "to_date": date_str,
                "count": len(mdf),
                "top_ret": top_r,
                "bot_ret": bot_r,
                "dl_ret": dl_r,
                "mkt_ret": mkt_r,
                "spread": round(top_r - bot_r, 2),
                "rank_ic": spearman_corr(mdf["underval"], mdf["ret"]),
            })
    return list(reversed(live_records))[:30]


def update_live_history(dist_dir: Path, date_str: str, evaluated: list):
    """Maintain survivorship-free forward ledgers across all 5 credit rating pools."""
    hist_path = dist_dir / "history.json"
    ledger = {
        "frozen_since": "2026-11-25",
        "params": FROZEN_PARAMS,
        "snapshots": [],
        "portfolio_ledger": {},
        "portfolio_ledgers": {},
    }
    if hist_path.exists():
        try:
            old = json.loads(hist_path.read_text(encoding="utf-8"))
            if isinstance(old, dict):
                for k in ("snapshots", "portfolio_ledger", "portfolio_ledgers"):
                    if k in old:
                        ledger[k] = old[k]
        except Exception:
            pass

    snap_entry = {
        "date": date_str,
        "pool_size": len(evaluated),
        "bonds": {
            b["ticker"]: {
                "name": b["name"],
                "rating": b.get("rating", ""),
                "bond_price": b["bond_price"],
                "model_price": b["model_price"],
                "undervaluation": b["undervaluation"],
                "double_low": b["double_low"],
                "region_code": b["region_code"],
            }
            for b in evaluated
        },
    }

    snapshots = [s for s in ledger.get("snapshots", []) if s.get("date") != date_str]
    snapshots.append(snap_entry)
    snapshots = sorted(snapshots, key=lambda x: x["date"])[-365:]
    ledger["snapshots"] = snapshots

    current_map = {b["ticker"]: b for b in evaluated}
    reset_needed = len(snapshots) >= 2 and snapshots[0]["date"] == date_str

    pool_defs = [
        ("all", None, 10),
        ("aa_minus", {"AA-", "AA", "AA+", "AAA"}, 10),
        ("aa", {"AA", "AA+", "AAA"}, 8),
        ("aa_plus", {"AA+", "AAA"}, 5),
        ("aaa", {"AAA"}, 5),
    ]
    pls = ledger.get("portfolio_ledgers") or {}
    if ledger.get("portfolio_ledger") and "all" not in pls:
        pls["all"] = ledger["portfolio_ledger"]
    if "full" in pls and "all" not in pls:
        pls["all"] = pls["full"]

    portfolio_by_pool = {}
    comparisons_by_pool = {}
    pool_sizes_today = {}
    for pkey, grades, min_k in pool_defs:
        sub_eval = evaluated if grades is None else [b for b in evaluated if norm_rating(b.get("rating")) in grades]
        pool_sizes_today[pkey] = len(sub_eval)
        pl_updated = _step_pool_ledger(pls.get(pkey, {}), sub_eval, current_map, date_str, min_k, reset_needed)
        pls[pkey] = pl_updated
        portfolio_by_pool[pkey] = pl_updated
        comparisons_by_pool[pkey] = _calc_pool_comparisons(snapshots, current_map, date_str, grades)

    ledger["portfolio_ledger"] = portfolio_by_pool["all"]
    ledger["portfolio_ledgers"] = pls

    clean_ledger = sanitize_json(ledger)
    hist_path.write_text(json.dumps(clean_ledger, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return {
        "tracked_days": len(snapshots),
        "frozen_since": ledger["frozen_since"],
        "pool_size_today": len(evaluated),
        "pool_sizes_today": pool_sizes_today,
        "portfolio": portfolio_by_pool["all"],
        "portfolio_by_pool": portfolio_by_pool,
        "comparisons": comparisons_by_pool["all"],
        "comparisons_by_pool": comparisons_by_pool,
    }


def load_live_pool(existing_bonds: list):
    pool = fetch_active_pool(max_bonds=500)
    fallback_sigmas = {b["stock_code"]: round(b["sigma"] / 100.0, 4) for b in existing_bonds if "stock_code" in b and "sigma" in b}
    vol_map = fetch_stock_vol_map([item["stock_code"] for item in pool], fallback_sigmas)

    evaluated = []
    for item in pool:
        S0 = item["stock_price"]
        Pmkt = item["bond_price"]
        C0 = item["conv_price"]
        mat = item.get("maturity_years", 2.5)
        sigma = vol_map.get(item["stock_code"], 0.28)
        rt = norm_rating(item.get("rating"))
        sname = str(item.get("stock_name") or "")
        amt_wan = item.get("amt_wan")
        is_st = "ST" in sname.upper()
        has_spec_rat = rt in SPECULATIVE_GRADES
        is_low_liq = (amt_wan is not None) and (float(amt_wan) < LOW_LIQ_WAN)

        conv_val = round((100.0 / C0) * S0, 2)
        premium_rate = round(((Pmkt - conv_val) / max(0.01, conv_val)) * 100, 2)
        double_low = round(Pmkt + premium_rate, 2)
        region_name, region_code, region_desc = classify_region(S0, C0)

        Pmdl = price_ccb_four_region_lsm(
            S0=S0, C0=C0, FV=100.0, maturity_years=mat,
            r=FROZEN_PARAMS["r"], q=FROZEN_PARAMS["q"], sigma=sigma,
            call_ratio=FROZEN_PARAMS["call_ratio"], put_ratio=FROZEN_PARAMS["put_ratio"],
            p_reset=FROZEN_PARAMS["p_reset"], B=FROZEN_PARAMS["B"],
            call_settle=FROZEN_PARAMS["call_settle"], put_pay=FROZEN_PARAMS["put_pay"],
            n_steps=FROZEN_PARAMS["n_steps"], M=FROZEN_PARAMS["M"], seed=FROZEN_PARAMS["seed"],
        )
        undervaluation = round(((Pmdl - Pmkt) / Pmkt) * 100, 2)

        evaluated.append({
            "ticker": item["ticker"],
            "name": item["name"],
            "rating": rt,
            "stock_code": item["stock_code"],
            "stock_name": sname,
            "stock_price": round(S0, 2),
            "conv_price": round(C0, 2),
            "bond_price": round(Pmkt, 2),
            "open_price": round(float(item.get("open_price") or Pmkt), 3),
            "volume": float(item.get("volume") or 0.0),
            "rem_scale": item.get("rem_scale"),
            "model_price": round(Pmdl, 2),
            "undervaluation": undervaluation,
            "pure_debt_val": round(FROZEN_PARAMS["B"] * np.exp(-FROZEN_PARAMS["r"] * mat), 1),
            "double_low": double_low,
            "conv_val": conv_val,
            "premium_rate": premium_rate,
            "sigma": round(sigma * 100, 1),
            "amt_wan": amt_wan,
            "is_st": is_st,
            "has_spec_rat": has_spec_rat,
            "is_low_liq": is_low_liq,
            "is_risk_excluded": bool(is_st or has_spec_rat or is_low_liq),
            "region_name": region_name,
            "region_code": region_code,
            "region_desc": region_desc,
            "maturity_years": round(mat, 2),
        })
    return evaluated


def _load_json_if_exists(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def generate_page_and_data():
    import sys as _sys
    _tools_dir = Path(__file__).resolve().parent / "tools"
    if str(_tools_dir) not in _sys.path:
        _sys.path.insert(0, str(_tools_dir))
    from sim_paper_accounts import resolve_market_session, update_paper_accounts

    local_tz = pytz.timezone("Asia/Shanghai")
    now = datetime.now(local_tz)
    session = resolve_market_session(now)
    update_time_str = session["updated_at"]
    market_date_str = session["market_date"]

    dist_dir = Path(__file__).resolve().parent / "dist" / "bonds"
    dist_dir.mkdir(parents=True, exist_ok=True)

    existing_data = _load_json_if_exists(dist_dir / "data.json")
    evaluated = load_live_pool(existing_data.get("all_bonds", []))
    if len(evaluated) < 50 and len(existing_data.get("all_bonds", [])) >= 100:
        evaluated = existing_data["all_bonds"]

    lsm_sorted = sorted(evaluated, key=lambda x: x["undervaluation"], reverse=True)
    dl_sorted = sorted(evaluated, key=lambda x: x["double_low"], reverse=False)
    lsm_rank = {b["ticker"]: i for i, b in enumerate(lsm_sorted, 1)}
    dl_rank = {b["ticker"]: i for i, b in enumerate(dl_sorted, 1)}
    for b in lsm_sorted:
        b["lsm_rank"] = lsm_rank[b["ticker"]]
        b["dl_rank"] = dl_rank[b["ticker"]]
        b["rating"] = norm_rating(b.get("rating"))

    research_pools = {
        "all": _load_json_if_exists(dist_dir / "research_validation.json"),
        "aa_minus": _load_json_if_exists(dist_dir / "research_validation_aa_minus.json"),
        "aa": _load_json_if_exists(dist_dir / "research_validation_aa.json"),
        "aa_plus": _load_json_if_exists(dist_dir / "research_validation_aa_plus.json"),
        "aaa": _load_json_if_exists(dist_dir / "research_validation_aaa.json"),
    }

    exp_full = _load_json_if_exists(dist_dir / "research_experiments.json")
    exp_inline = {}
    if exp_full:
        key_ids = set()
        for pk in ("all", "aa_minus", "aa", "aa_plus", "aaa"):
            for rm in ("raw", "filtered"):
                key_ids.add(f"{pk}|{rm}|100/0|5D|none")
                key_ids.add(f"{pk}|{rm}|50/50|5D|buf_20_30")
                key_ids.add(f"{pk}|{rm}|50/50|10D|buf_20_30")
                key_ids.add(f"{pk}|{rm}|50/50|5D|none")
                key_ids.add(f"{pk}|{rm}|0/100|5D|none")
        exp_inline = {
            "meta": exp_full.get("meta", {}),
            "benchmarks": exp_full.get("benchmarks", {}),
            "stepwise": exp_full.get("stepwise", {}),
            "walk_forward": exp_full.get("walk_forward", {}),
            "overfitting": exp_full.get("overfitting", {}),
            "grid": [r for r in exp_full.get("grid", []) if r.get("id") in key_ids],
        }

    pro_full = _load_json_if_exists(dist_dir / "research_pro.json")
    pro_inline = {
        "meta": pro_full.get("meta", {}),
        "candidates": pro_full.get("candidates", []),
        "factor_corr_aam": pro_full.get("factor_corr_aam", {}),
        "incremental_fmb_aam": pro_full.get("incremental_fmb_aam", {}),
        "peer_benchmarks": pro_full.get("peer_benchmarks", {}),
        "phased_optimization": pro_full.get("phased_optimization", {}),
    } if pro_full else {}

    timing_full = _load_json_if_exists(dist_dir / "research_timing.json")
    timing_inline = {
        "meta": timing_full.get("meta", {}),
        "latest_by_pool": timing_full.get("latest_by_pool", {}),
        "regimes": timing_full.get("regimes", {}),
        "position_sizing": timing_full.get("position_sizing", {}),
    } if timing_full else {}

    live_ledger = update_live_history(dist_dir, market_date_str, lsm_sorted)
    paper_accounts = update_paper_accounts(dist_dir, lsm_sorted, session, timing_ref=timing_full)

    data_payload = sanitize_json({
        "updated_at": update_time_str,
        "date": market_date_str,
        "session": session,
        "author": "Yu Liu (刘宇) · The Chinese University of Hong Kong, Shenzhen",
        "paper_ref": "arXiv:2409.06496v7",
        "paper_url": "https://arxiv.org/abs/2409.06496",
        "frozen_params": FROZEN_PARAMS,
        "pool_size": len(lsm_sorted),
        "research": research_pools["all"],
        "research_pools": research_pools,
        "exp_summary": exp_inline,
        "pro_summary": pro_inline,
        "timing_summary": timing_inline,
        "paper_accounts": paper_accounts,
        "live_ledger": live_ledger,
        "lsm_top10": lsm_sorted[:10],
        "double_low_top10": dl_sorted[:10],
        "all_bonds": lsm_sorted,
    })

    (dist_dir / "data.json").write_text(json.dumps(data_payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (dist_dir / "index.html").write_text(build_clean_html(data_payload), encoding="utf-8")
    print(f"Generated 5-module CB Quant Workbench at {dist_dir / 'index.html'} (live_pool={len(lsm_sorted)}, market_date={market_date_str}, phase={session['phase']})")


def build_clean_html(data: dict) -> str:
    payload_json = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")

    template = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Convertible Bonds · 可转债量化研究 · Four-Region LSM</title>
  <meta name="description" content="中国可转债 Four-Region LSM 量化研究平台：五档点时信用池、风险过滤、因子混合与缓冲回测及样本外防过拟合检验">
  <style>
    :root {
      --bg: #f6f8fa;
      --surface: #ffffff;
      --subtle: #f3f4f6;
      --border: #d0d7de;
      --border-soft: #eaeef2;
      --ink: #1f2328;
      --muted: #656d76;
      --light: #8c959f;
      --blue: #0969da;
      --blue-soft: #ddf4ff;
      --green: #1a7f37;
      --green-soft: #dafbe1;
      --red: #cf222e;
      --red-soft: #ffebe9;
      --amber: #9a6700;
      --amber-soft: #fff8c5;
      --purple: #8250df;
      color-scheme: light;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; background: var(--bg); color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", Helvetica, Arial, sans-serif;
      font-size: 14px; line-height: 1.55; -webkit-font-smoothing: antialiased;
    }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-variant-numeric: tabular-nums; }
    a { color: var(--blue); text-decoration: none; }
    a:hover { text-decoration: underline; }

    /* Header Shell */
    .app-header { background: var(--surface); border-bottom: 1px solid var(--border); }
    .header-inner { max-width: 1200px; margin: 0 auto; padding: 14px 22px 0; }
    .header-top { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 10px; }
    .brand-block { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
    .brand-title { font-size: 18px; font-weight: 700; letter-spacing: -0.2px; color: var(--ink); }
    .brand-sub { font-size: 13px; color: var(--muted); }
    .status-chip { font-size: 11px; color: var(--muted); background: var(--subtle); border: 1px solid var(--border-soft); border-radius: 999px; padding: 2px 9px; }
    .nav-ext { display: flex; gap: 14px; flex-wrap: wrap; font-size: 12px; }
    .nav-ext a { color: var(--muted); }
    .nav-ext a:hover { color: var(--blue); }

    /* Tabs + Global Controls */
    .control-bar { display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 10px; }
    .tabs { display: flex; gap: 2px; overflow-x: auto; }
    .tab-btn {
      appearance: none; background: none; border: none; border-bottom: 2px solid transparent;
      padding: 10px 13px; font-size: 14px; font-weight: 500; color: var(--muted); cursor: pointer;
      display: inline-flex; align-items: center; gap: 6px; white-space: nowrap;
    }
    .tab-btn:hover { color: var(--ink); }
    .tab-btn.active { color: var(--ink); font-weight: 600; border-bottom-color: #fd8c73; }
    .tab-num { font-size: 11px; color: var(--light); font-weight: 600; }

    .global-filters { display: flex; align-items: center; gap: 10px; padding-bottom: 8px; flex-wrap: wrap; }
    .seg { display: inline-flex; background: var(--subtle); border: 1px solid var(--border); border-radius: 6px; padding: 2px; flex-wrap: wrap; }
    .seg-btn {
      border: none; background: transparent; padding: 4px 10px; font-size: 12px; font-weight: 500;
      color: var(--muted); border-radius: 4px; cursor: pointer; white-space: nowrap;
    }
    .seg-btn.active { background: var(--surface); color: var(--ink); font-weight: 600; box-shadow: 0 1px 2px rgba(31,35,40,0.08); }

    /* Shell */
    .shell { max-width: 1200px; margin: 0 auto; padding: 18px 22px 56px; }
    .status-banner {
      display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;
      padding: 8px 14px; margin-bottom: 14px; border-radius: 6px; font-size: 12px;
      background: var(--surface); border: 1px solid var(--border-soft); color: var(--muted);
    }
    .status-banner strong { color: var(--ink); }
    .warn-banner {
      padding: 9px 14px; margin-bottom: 14px; border-radius: 6px; font-size: 12px;
      background: var(--amber-soft); border: 1px solid #ead18a; color: #6b4500; display: none;
    }
    .view-pane { display: none; }
    .view-pane.active { display: block; }

    /* Compact Time Bar & Chart Interactive Tooltips */
    .time-bar {
      display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;
      padding: 8px 16px; background: #fbfcfd; border-bottom: 1px solid var(--border-soft); font-size: 12px;
    }
    .time-custom { display: none; align-items: center; gap: 6px; flex-wrap: wrap; }
    .time-custom.active { display: inline-flex; }
    .dt-inp { border: 1px solid var(--border); border-radius: 4px; padding: 3px 6px; font-size: 11px; background: var(--surface); color: var(--ink); }
    .chart-wrap { position: relative; width: 100%; }
    .chart-tip {
      position: absolute; top: 8px; right: 90px; background: rgba(255,255,255,0.96);
      border: 1px solid var(--border); border-radius: 5px; padding: 6px 9px; font-size: 11px;
      box-shadow: 0 2px 6px rgba(31,35,40,0.08); pointer-events: none; display: none; z-index: 10; min-width: 155px;
    }
    .legend-btn {
      display: inline-flex; align-items: center; gap: 5px; cursor: pointer; background: none; border: 1px solid transparent;
      padding: 1px 6px; border-radius: 4px; font-size: 11px; color: var(--muted);
    }
    .legend-btn:hover { background: var(--subtle); }
    .legend-btn.off { opacity: 0.38; text-decoration: line-through; }

    /* KPIs & Cards */
    .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 14px; }
    .kpi-card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 13px 15px; }
    .kpi-label { font-size: 12px; color: var(--muted); margin-bottom: 3px; display: block; }
    .kpi-val { font-size: 23px; font-weight: 700; line-height: 1.2; letter-spacing: -0.3px; }
    .kpi-sub { font-size: 11px; color: var(--muted); margin-top: 4px; }

    .temp-card {
      background: var(--surface); border: 1px solid var(--border); border-radius: 6px;
      padding: 11px 15px; margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;
    }
    .temp-metrics { display: flex; gap: 18px; flex-wrap: wrap; font-size: 12px; }
    .temp-metrics span { color: var(--muted); }
    .temp-metrics strong { color: var(--ink); font-weight: 600; margin-left: 4px; }
    .reg-bar-wrap { display: flex; align-items: center; gap: 8px; min-width: 220px; }
    .reg-bar { flex: 1; height: 8px; border-radius: 4px; overflow: hidden; display: flex; background: var(--subtle); min-width: 120px; }
    .reg-legend { display: flex; gap: 8px; font-size: 11px; color: var(--muted); }

    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 14px; }
    .grid-overview { display: grid; grid-template-columns: 1.15fr 1fr; gap: 14px; margin-bottom: 14px; }
    .card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; margin-bottom: 14px; }
    .grid-2 > .card, .grid-overview > .card { margin-bottom: 0; }
    .card-head {
      padding: 11px 16px; background: #fbfcfd; border-bottom: 1px solid var(--border-soft);
      display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;
    }
    .card-title { font-size: 14px; font-weight: 600; margin: 0; color: var(--ink); }
    .card-sub { font-size: 12px; color: var(--muted); }
    .card-body { padding: 13px 16px; }

    /* Filters & Param Bar */
    .pills { display: flex; gap: 5px; flex-wrap: wrap; }
    .pill {
      padding: 4px 10px; border-radius: 5px; font-size: 12px; border: 1px solid var(--border);
      background: var(--surface); color: var(--muted); cursor: pointer; white-space: nowrap;
    }
    .pill.active { background: var(--blue-soft); border-color: var(--blue); color: var(--blue); font-weight: 600; }
    .param-strip {
      display: flex; flex-wrap: wrap; gap: 14px; align-items: center; padding: 11px 16px;
      background: #fbfcfd; border-bottom: 1px solid var(--border-soft);
    }
    .param-group { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
    .param-lbl { font-size: 12px; font-weight: 600; color: var(--muted); }
    .toolbar {
      padding: 10px 16px; background: #fbfcfd; border-bottom: 1px solid var(--border-soft);
      display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;
    }
    .filter-row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .inp, .sel { border: 1px solid var(--border); background: var(--surface); border-radius: 5px; padding: 5px 9px; font-size: 12px; color: var(--ink); }
    .inp { min-width: 180px; }

    /* Tables */
    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; text-align: left; font-size: 13px; }
    th { background: #fbfcfd; color: var(--muted); font-weight: 600; font-size: 12px; padding: 8px 13px; border-bottom: 1px solid var(--border-soft); white-space: nowrap; }
    td { padding: 8px 13px; border-bottom: 1px solid var(--border-soft); vertical-align: middle; }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: #f8fafc; }
    .hl-row td { background: #f0f7ff; }
    td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
    .sub { display: block; font-size: 11px; color: var(--muted); margin-top: 1px; }

    /* Badges & Text */
    .pos { color: var(--red); font-weight: 600; }
    .neg { color: var(--green); font-weight: 600; }
    .badge { display: inline-block; font-size: 11px; padding: 1px 7px; border-radius: 999px; font-weight: 500; white-space: nowrap; border: 1px solid transparent; }
    .b-aaa { background: #ddf4ff; color: #0969da; border-color: #b6e3ff; }
    .b-aap { background: #dafbe1; color: #1a7f37; border-color: #aceebb; }
    .b-aa  { background: #f3f4f6; color: #3b444f; border-color: #d0d7de; }
    .b-aam { background: #f6f8fa; color: #656d76; border-color: #d0d7de; }
    .b-low { background: #ffebe9; color: #cf222e; border-color: #ffcecb; }
    .b-warn { background: #fff8c5; color: #9a6700; border-color: #f2cc60; margin-right: 4px; }
    .b-sig-y { background: var(--green-soft); color: var(--green); border-color: #aceebb; }
    .b-sig-n { background: var(--subtle); color: var(--muted); border-color: var(--border); }

    .legend { display: flex; gap: 14px; flex-wrap: wrap; font-size: 11px; color: var(--muted); margin-bottom: 8px; }
    .legend span { display: inline-flex; align-items: center; gap: 5px; }
    .ldot { width: 10px; height: 3px; border-radius: 2px; display: inline-block; }
    .note-box { padding: 10px 16px; background: #fbfcfd; border-top: 1px solid var(--border-soft); font-size: 12px; color: var(--muted); line-height: 1.65; }
    .takeaway-list { margin: 0; padding-left: 18px; font-size: 12px; color: var(--muted); line-height: 1.75; }
    .takeaway-list strong { color: var(--ink); }
    .btn-link { font-size: 12px; font-weight: 500; color: var(--blue); background: none; border: none; cursor: pointer; padding: 0; }
    .btn-link:hover { text-decoration: underline; }

    details.fold-card > summary {
      padding: 12px 16px; background: #fbfcfd; cursor: pointer; font-weight: 600; font-size: 13px;
      display: flex; justify-content: space-between; align-items: center; list-style: none;
    }
    details.fold-card[open] > summary { border-bottom: 1px solid var(--border-soft); }
    details.fold-card > summary::after { content: "展开明细 ▾"; font-size: 12px; color: var(--blue); font-weight: 500; }
    details.fold-card[open] > summary::after { content: "收起 ▴"; }

    footer {
      margin-top: 22px; padding-top: 14px; border-top: 1px solid var(--border);
      font-size: 12px; color: var(--muted); display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px;
    }

    @media (max-width: 920px) {
      .kpi-grid { grid-template-columns: 1fr 1fr; }
      .grid-2, .grid-overview { grid-template-columns: 1fr; }
      .shell, .header-inner { padding-left: 12px; padding-right: 12px; }
    }
  </style>
</head>
<body>
<header class="app-header">
  <div class="header-inner">
    <div class="header-top">
      <div class="brand-block">
        <span class="brand-title">Convertible Bonds</span>
        <span class="brand-sub">可转债量化研究 · Four-Region LSM</span>
        <span class="status-chip mono">参数已冻结 · arXiv:2409.06496v7</span>
      </div>
      <nav class="nav-ext">
        <a href="/">个人主页</a>
        <a href="/trade/">A 股买卖看板</a>
        <a href="/phd-tracker/">申请追踪</a>
        <a href="https://github.com/Aquareo/Aquareo.github.io" target="_blank" rel="noopener">GitHub 源码 ↗</a>
        <a href="https://arxiv.org/abs/2409.06496" target="_blank" rel="noopener">论文原文 ↗</a>
      </nav>
    </div>
    <div class="control-bar">
      <nav class="tabs" id="mainTabs" role="tablist">
        <button class="tab-btn active" data-tab="overview"><span class="tab-num mono">01</span>总览</button>
        <button class="tab-btn" data-tab="screener"><span class="tab-num mono">02</span>实时选券</button>
        <button class="tab-btn" data-tab="backtest"><span class="tab-num mono">03</span>历史回测</button>
        <button class="tab-btn" data-tab="paper"><span class="tab-num mono">04</span>模拟交易</button>
        <button class="tab-btn" data-tab="research"><span class="tab-num mono">05</span>因子研究</button>
      </nav>
      <div class="global-filters">
        <div class="seg" id="poolSeg" title="点时信用评级过滤（严格公告次日生效）">
          <button class="seg-btn active" data-pool="all">全市场</button>
          <button class="seg-btn" data-pool="aa_minus">AA-及以上</button>
          <button class="seg-btn" data-pool="aa">AA及以上</button>
          <button class="seg-btn" data-pool="aa_plus">AA+及以上</button>
          <button class="seg-btn" data-pool="aaa">仅AAA</button>
        </div>
        <div class="seg" id="riskSeg" title="因果剔除：历史ST正股 + 违约预警/<=BBB+ + 20日中位成交<500万">
          <button class="seg-btn active" data-risk="raw">不剔除</button>
          <button class="seg-btn" data-risk="filtered">剔除 ST·违约·低流</button>
        </div>
      </div>
    </div>
  </div>
</header>

<main class="shell">
  <div class="status-banner">
    <div id="researchStatusText">研究数据加载中...</div>
    <div class="mono" style="font-size:11px;" id="sessionBannerRight">p_reset=0.80 · B=¥108 · r=2.5% · 更新 __UPDATED_AT__</div>
  </div>
  <div class="warn-banner" id="poolWarnBanner"></div>

  <!-- 01 OVERVIEW -->
  <section class="view-pane active" id="view-overview">
    <div class="kpi-grid">
      <div class="kpi-card">
        <span class="kpi-label" id="ovKpi1Lbl">费后年化收益 (原始 LSM)</span>
        <div class="kpi-val mono" id="ovAnnRet">—</div>
        <div class="kpi-sub" id="ovAnnSub">—</div>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">风险调整指标 (Calmar / Sharpe)</span>
        <div class="kpi-val mono" id="ovCalmar">—</div>
        <div class="kpi-sub" id="ovCalmarSub">—</div>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">最大回撤</span>
        <div class="kpi-val mono neg" id="ovMdd">—</div>
        <div class="kpi-sub" id="ovMddSub">—</div>
      </div>
      <div class="kpi-card">
        <span class="kpi-label" id="ovIcLbl">20 日 Rank IC</span>
        <div class="kpi-val mono" id="ovIc">—</div>
        <div class="kpi-sub" id="ovIcSub">—</div>
      </div>
    </div>

    <div class="temp-card">
      <div style="flex:1;min-width:260px;">
        <div class="temp-metrics" id="tempMetrics"></div>
        <div style="font-size:11px;color:var(--muted);margin-top:5px;" id="tempTimingLine">—</div>
      </div>
      <div class="reg-bar-wrap">
        <div class="reg-bar" id="tempRegBar"></div>
        <div class="reg-legend" id="tempRegLegend"></div>
      </div>
    </div>

    <div class="grid-overview">
      <div class="card">
        <div class="card-head">
          <div>
            <h2 class="card-title" id="ovChartTitle">核心策略净值对比（0.10% 单边成本）</h2>
            <span class="card-sub">默认展示 2023 年至今（完整保留 2020–2026）；可点击图例隐藏/显示</span>
          </div>
          <div class="pills" id="ovPresetPills">
            <button class="pill active" data-ovp="lsm">原始 LSM</button>
            <button class="pill" data-ovp="balanced">Balanced(20只)</button>
            <button class="pill" data-ovp="defensive">Defensive(AA)</button>
            <button class="pill" data-ovp="dl">传统双低</button>
          </div>
        </div>
        <div class="time-bar">
          <div class="pills" data-tbar="ov">
            <button class="pill" data-tr="all">全部</button>
            <button class="pill active" data-tr="since2023">2023年至今</button>
            <button class="pill" data-tr="3y">近3年</button>
            <button class="pill" data-tr="1y">近1年</button>
            <button class="pill" data-tr="ytd">今年以来</button>
            <button class="pill" data-tr="custom">自定义</button>
          </div>
          <div class="time-custom" id="ovCustomBox">
            <input type="date" class="dt-inp" id="trStartInp"> <span>→</span> <input type="date" class="dt-inp" id="trEndInp">
          </div>
        </div>
        <div class="card-body">
          <div class="legend" id="ovLegend"></div>
          <div class="chart-wrap" id="ovNavWrap">
            <div id="ovNavSvg"></div>
            <div class="chart-tip mono" id="ovTip"></div>
          </div>
          <div class="table-wrap" style="margin-top:10px;border-top:1px solid var(--border-soft);">
            <table>
              <thead><tr><th>核心对照 (当前区间 · 0.10% 成本)</th><th class="num">区间年化</th><th class="num">最大回撤</th><th class="num">Sharpe</th><th class="num">Calmar</th><th class="num">超额基准</th></tr></thead>
              <tbody id="ovMiniBody"></tbody>
            </table>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-head">
          <div>
            <h2 class="card-title" id="ovTopTitle">当前低估转债 Top 10</h2>
            <span class="card-sub">严格匹配当前评级池与风险过滤，按 LSM 低估率排序</span>
          </div>
          <button class="btn-link" data-jump="screener">全部转债选券 →</button>
        </div>
        <div class="table-wrap">
          <table>
            <thead><tr><th>转债</th><th>评级</th><th class="num">现价</th><th class="num">估值</th><th class="num">低估率</th></tr></thead>
            <tbody id="ovTopBody"></tbody>
          </table>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-head">
        <h2 class="card-title">最终验证结论与候选策略定位（不以七年历史峰值挑选参数）</h2>
        <button class="btn-link" data-jump="research">查看专业因子对照、择时与审计 →</button>
      </div>
      <div class="card-body">
        <ul class="takeaway-list">
          <li><strong>三套候选实测筛选结果：</strong><code>Enhanced Growth (N=20, AA-)</code> 年化 +12.61% / 回撤 -25.39% 未能超越同池原始 LSM（+14.71% / -22.16%），<strong>未通过验证已淘汰</strong>；<strong>Enhanced Balanced</strong>（AA- 剔除风险 · 50/50 · 10D · 20/30 缓冲 · 固定 20 只等权）与 <strong>Enhanced Defensive</strong>（AA 以上 50/50 · 10D · 20/30 · 20 只逆波）在 ¥10 万 10 张整手实盘分别录得 91.1%–91.7% 资金利用率与 +12.32% / +12.73% 年化（回撤 -18.47% / -15.05%），但二者近 1 年受双低风格拖累录得 -7.97% / -9.82%，均在 ¥10 万模拟账户长期前瞻验证。</li>
          <li><strong>专业对照与因子来源：</strong>LSM 与传统双低截面相关性仅 <code>+0.12</code>，形成显著互补；但在同时控制 BS 偏离、30 步 CRR 二叉树与 YTM 的多元 Fama-MacBeth 回归中，LSM 独立斜率因定价共线性（vs 二叉树相关 +0.63）衰减至 t=+0.49，证明其优势在于单期排序 Sharpe（0.97 vs BS 0.90 / 二叉树 0.81）与更低换手，而非完全正交的新因子。</li>
          <li><strong>估值择时与过拟合防线：</strong>滚动估值分位分档在全市场 LSM 上将回撤从 -21.05% 降至 -14.91%（Sharpe 1.01→1.05），但 450 组试验 CSCV 过拟合概率 <code>PBO=11.4%</code>、滚动 2 年短窗样本外夏普仅 <code>0.03</code>，需以实盘前瞻为准。</li>
        </ul>
      </div>
    </div>
  </section>

  <!-- 02 SCREENER -->
  <section class="view-pane" id="view-screener">
    <div class="card">
      <div class="card-head">
        <div>
          <h2 class="card-title" id="scrTitle">实时转债估值选券池</h2>
          <span class="card-sub">随顶部「评级池」与「风险过滤」联动，覆盖全池实时成交额（新浪实时）与剩余规模（集思录）</span>
        </div>
        <span class="mono" style="font-size:12px;color:var(--muted);" id="scrCountNote">—</span>
      </div>
      <div class="toolbar">
        <div class="filter-row">
          <input class="inp" id="scrSearch" type="search" placeholder="搜索代码 / 转债 / 正股...">
          <select class="sel" id="scrReg">
            <option value="ALL">全部条款区间</option>
            <option value="R1">R1 强赎提示区</option>
            <option value="R2">R2 高弹性转股区</option>
            <option value="R3">R3 平衡抗跌区</option>
            <option value="R4">R4 下修博弈区</option>
          </select>
          <select class="sel" id="scrSort">
            <option value="underval">按 LSM 低估率降序</option>
            <option value="price">按现价升序</option>
            <option value="dl">按传统双低升序</option>
            <option value="prem">按转股溢价率升序</option>
          </select>
        </div>
        <label style="font-size:12px;color:var(--muted);display:flex;align-items:center;gap:5px;cursor:pointer;">
          <input type="checkbox" id="scrSafeOnly"> 过滤高价(&gt;¥150)或高溢价(&gt;80%)警示券
        </label>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th style="text-align:center;">#</th><th>转债标的</th><th>评级</th><th class="num">现价</th>
              <th class="num">LSM估值</th><th class="num">低估率</th><th class="num">溢价 / 双低</th>
              <th>条款区</th><th>风险与流动性提示</th>
            </tr>
          </thead>
          <tbody id="scrTableBody"></tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- 03 BACKTEST -->
  <section class="view-pane" id="view-backtest">
    <div class="card">
      <div class="card-head">
        <div>
          <h2 class="card-title">历史回测工作台（统一 T+1 开盘成交 · 日频净值重建）</h2>
          <span class="card-sub">切换时间范围时从起始日净值重新归一化，与「起始空仓重新建仓」明确区分；支持悬浮查净值与图例开关</span>
        </div>
        <span class="mono" style="font-size:12px;color:var(--muted);" id="btComboId">—</span>
      </div>
      <div class="time-bar">
        <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
          <span class="param-lbl">时间范围</span>
          <div class="pills" data-tbar="bt">
            <button class="pill" data-tr="all">全部 (2020–2026)</button>
            <button class="pill active" data-tr="since2023">2023年至今</button>
            <button class="pill" data-tr="3y">近3年</button>
            <button class="pill" data-tr="1y">近1年</button>
            <button class="pill" data-tr="ytd">今年以来</button>
            <button class="pill" data-tr="custom">自定义</button>
          </div>
          <div class="time-custom" id="btCustomBox">
            <input type="date" class="dt-inp" id="btStartInp"> <span>→</span> <input type="date" class="dt-inp" id="btEndInp">
          </div>
        </div>
        <select class="sel" id="btOverlaySel" title="叠加额外对照或增强策略曲线">
          <option value="none">+ 叠加对照：不额外叠加</option>
          <option value="cand|enhanced_balanced">+ 叠加 Enhanced Balanced (AA-·20只)</option>
          <option value="cand|enhanced_defensive">+ 叠加 Enhanced Defensive (AA·20只逆波)</option>
          <option value="peer|aam_filt|f_bs_pure">+ 叠加 Black-Scholes 偏离 (AA-)</option>
          <option value="peer|aam_filt|f_binom">+ 叠加 30步 CRR 二叉树 (AA-)</option>
          <option value="peer|aam_filt|tri_low">+ 叠加 券商三低多因子 (AA-)</option>
        </select>
      </div>
      <div class="param-strip">
        <div class="param-group">
          <span class="param-lbl">LSM/双低</span>
          <div class="pills" id="btWeightPills">
            <button class="pill" data-w="100/0">100/0 纯LSM</button>
            <button class="pill" data-w="75/25">75/25</button>
            <button class="pill active" data-w="50/50">50/50 均衡</button>
            <button class="pill" data-w="25/75">25/75</button>
            <button class="pill" data-w="0/100">0/100 纯双低</button>
          </div>
        </div>
        <div class="param-group">
          <span class="param-lbl">调仓</span>
          <div class="pills" id="btFreqPills">
            <button class="pill active" data-f="5">5D</button>
            <button class="pill" data-f="10">10D</button>
            <button class="pill" data-f="20">20D</button>
          </div>
        </div>
        <div class="param-group">
          <span class="param-lbl">缓冲</span>
          <div class="pills" id="btBufPills">
            <button class="pill" data-b="none">无缓冲</button>
            <button class="pill active" data-b="buf_20_30">20/30 缓冲</button>
            <button class="pill" data-b="buf_20_40">20/40 缓冲</button>
          </div>
        </div>
        <div class="param-group">
          <span class="param-lbl">成本</span>
          <div class="pills" id="btCostPills">
            <button class="pill" data-c="0.05%">0.05%</button>
            <button class="pill active" data-c="0.10%">0.10%</button>
            <button class="pill" data-c="0.20%">0.20%</button>
          </div>
        </div>
      </div>
      <div class="card-body">
        <div class="kpi-grid" style="margin-bottom:14px;">
          <div class="kpi-card"><span class="kpi-label">区间年化净收益 / 累计收益</span><div class="kpi-val mono" id="btAnnRet">—</div><div class="kpi-sub" id="btTotRet">—</div></div>
          <div class="kpi-card"><span class="kpi-label">区间最大回撤</span><div class="kpi-val mono neg" id="btMdd">—</div><div class="kpi-sub" id="btCostNote">单边成本 0.10%</div></div>
          <div class="kpi-card"><span class="kpi-label">区间 Sharpe / Calmar</span><div class="kpi-val mono" id="btSharpe">—</div><div class="kpi-sub" id="btCalmar">—</div></div>
          <div class="kpi-card"><span class="kpi-label">相对同池基准超额</span><div class="kpi-val mono" id="btExcess">—</div><div class="kpi-sub" id="btPoolStats">—</div></div>
        </div>
        <div class="legend" id="btLegend"></div>
        <div class="chart-wrap" id="btNavWrap">
          <div id="btNavSvg"></div>
          <div class="chart-tip mono" id="btTip"></div>
        </div>
        <div style="font-size:11px;color:var(--muted);margin-top:8px;" id="btScopeNote">口径说明：当前净值为 2020 年起连续运行账户在所选起止日截取并归一化（含既定持仓与缓冲延续状态），与从 t0 全现金新建仓存在微弱首日摩擦差异。</div>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-head">
          <div>
            <h2 class="card-title">分年度表现对照（全历史完整逐年记录）</h2>
            <span class="card-sub">所选参数组合 vs 原始 LSM vs 纯双低（同池同成本口径）</span>
          </div>
        </div>
        <div class="table-wrap">
          <table>
            <thead><tr><th>年份</th><th class="num">所选参数 (MDD)</th><th class="num">原始 LSM</th><th class="num">纯双低</th></tr></thead>
            <tbody id="btAnnualBody"></tbody>
          </table>
        </div>
      </div>

      <div class="card">
        <div class="card-head">
          <div>
            <h2 class="card-title">滚动 12 个月超额收益走势 (vs 同池等权基准)</h2>
            <span class="card-sub">检验 2020–2026 不同市场阶段 Alpha 稳定性</span>
          </div>
        </div>
        <div class="card-body">
          <div id="btRollExcessSvg"></div>
          <div class="table-wrap" style="margin-top:8px;border-top:1px solid var(--border-soft);">
            <table>
              <thead>
                <tr><th>当前参数跨评级池对照</th><th class="num">全期年化</th><th class="num">回撤</th><th class="num">Sharpe</th><th class="num">换手</th></tr>
              </thead>
              <tbody id="bt5PoolBody"></tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- 04 PAPER TRADING (¥100,000 VIRTUAL SIMULATION) -->
  <section class="view-pane" id="view-paper">
    <div class="card">
      <div class="card-head">
        <div>
          <h2 class="card-title">10 万元虚拟资金前瞻实盘模拟账户（T 日生成信号 → T+1 开盘 10 张整手真实成交）</h2>
          <span class="card-sub" id="ppSessionLine">严禁回填或使用未来数据 · 仅限虚拟资金研究，不连接真实证券账户</span>
        </div>
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
          <select class="sel" id="ppCohortSel">
            <option value="aa_minus">可比对照组 1：AA-及以上 (因果风控)</option>
            <option value="aa">可比对照组 2：AA及以上 (防御组)</option>
            <option value="all">基准对照：全市场原始 LSM</option>
          </select>
          <span class="badge b-warn" id="ppStatusBadge">前瞻运行中</span>
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>虚拟账户 (初始 ¥100,000)</th><th>执行口径</th><th class="num">当前总资产</th><th class="num">累计盈亏</th><th class="num">可用现金 (仓位%)</th><th class="num">最大回撤</th><th class="num">累计费用</th><th>待执行信号状态</th></tr>
          </thead>
          <tbody id="ppCohortBody"></tbody>
        </table>
      </div>
      <div class="param-strip">
        <span class="param-lbl">查看账户明细</span>
        <div class="pills" id="ppAccPills"></div>
      </div>
      <div class="card-body">
        <div class="kpi-grid" style="margin-bottom:14px;">
          <div class="kpi-card"><span class="kpi-label">账户总资产 (初始 ¥100,000)</span><div class="kpi-val mono" id="ppKpiNav">—</div><div class="kpi-sub" id="ppKpiRet">—</div></div>
          <div class="kpi-card"><span class="kpi-label">可用现金 / 资金利用率</span><div class="kpi-val mono" id="ppKpiCash">—</div><div class="kpi-sub" id="ppKpiUtil">—</div></div>
          <div class="kpi-card"><span class="kpi-label">前瞻最大回撤 / 累计费用</span><div class="kpi-val mono" id="ppKpiMdd">—</div><div class="kpi-sub" id="ppKpiFees">—</div></div>
          <div class="kpi-card"><span class="kpi-label">待执行信号 / 最近行情日</span><div class="kpi-val mono" style="font-size:17px;" id="ppKpiSig">—</div><div class="kpi-sub" id="ppKpiDates">—</div></div>
        </div>
        <div id="ppNavSvg"></div>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-head">
          <div>
            <h2 class="card-title" id="ppHoldTitle">当前模拟持仓明细</h2>
            <span class="card-sub">严格按 1 手 = 10 张整数成交；创设首日信号队列等待下一交易日开盘成交</span>
          </div>
        </div>
        <div class="table-wrap">
          <table>
            <thead><tr><th>标的</th><th>评级</th><th class="num">数量(张)</th><th class="num">成本价</th><th class="num">现价</th><th class="num">占比</th><th class="num">浮盈</th></tr></thead>
            <tbody id="ppHoldBody"></tbody>
          </table>
        </div>
      </div>

      <div class="card">
        <div class="card-head">
          <div>
            <h2 class="card-title">待执行信号队列 & 成交/资金流水</h2>
            <span class="card-sub">区分信号日、成交日与账本现金变化</span>
          </div>
          <div class="pills" id="ppLogPills">
            <button class="pill active" data-ppl="pending">待执行信号</button>
            <button class="pill" data-ppl="trades">历史成交</button>
            <button class="pill" data-ppl="cash">资金流水</button>
          </div>
        </div>
        <div class="table-wrap">
          <table>
            <thead id="ppLogHead"></thead>
            <tbody id="ppLogBody"></tbody>
          </table>
        </div>
      </div>
    </div>

    <div class="card">
      <details class="fold-card">
        <summary>原 Top20% 前瞻组合跟踪（2026-11-25 起连续归档保留）与账户执行规则</summary>
        <div class="table-wrap">
          <table>
            <thead><tr><th>#</th><th>原账本标的</th><th>评级</th><th class="num">权重</th><th>建仓日</th><th class="num">入场价</th><th class="num">最新价</th><th class="num">区间盈亏</th></tr></thead>
            <tbody id="portTableBody"></tbody>
          </table>
        </div>
        <div id="ledgerHistBox"></div>
        <div class="note-box" id="ppRulesNote">—</div>
      </details>
    </div>
  </section>

  <!-- 05 FACTOR RESEARCH -->
  <section class="view-pane" id="view-research">
    <!-- 5A. Peer Benchmarks & Incremental Power -->
    <div class="card">
      <div class="card-head">
        <div>
          <h2 class="card-title">专业定价与券商因子策略公平对照（5D · Top20% · T+1 开盘 · 0.10% 成本）</h2>
          <span class="card-sub">Black-Scholes、30步 CRR 二叉树、IV/HV-IV、YTM、纯债溢价、因果小规模及三低/四因子 vs LSM 消融与基准</span>
        </div>
        <div class="pills" id="resPeerScopePills">
          <button class="pill active" data-ps="aa_minus_filtered">AA-及以上 (因果剔除)</button>
          <button class="pill" data-ps="all_raw">全市场 (不剔除)</button>
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>策略 / 模型版本</th><th class="num">20D Rank IC (NW-t)</th><th class="num">全期年化</th><th class="num">最大回撤</th><th class="num">Sharpe</th><th class="num">Calmar</th><th class="num">年化换手</th><th class="num">23–24年化</th><th class="num">25至今</th></tr>
          </thead>
          <tbody id="resPeerBody"></tbody>
        </table>
      </div>
      <div class="grid-2" style="padding:14px 16px;border-top:1px solid var(--border-soft);margin-bottom:0;">
        <div>
          <div style="font-size:12px;font-weight:600;margin-bottom:6px;">增量 Fama-MacBeth 回归 (AA- 剔除风险 · 20D 收益 · 标准化斜率)</div>
          <div class="table-wrap">
            <table>
              <thead><tr><th>解释变量 (控制全因子)</th><th class="num">全样本斜率 (t)</th><th class="num">含IV子集 (t)</th></tr></thead>
              <tbody id="resIncFmbBody"></tbody>
            </table>
          </div>
        </div>
        <div>
          <div style="font-size:12px;font-weight:600;margin-bottom:6px;">主要因子截面 Spearman 相关系数矩阵（AA- 剔除风险日均）</div>
          <div class="table-wrap" id="resCorrBox"></div>
        </div>
      </div>
      <details class="fold-card" style="border-top:1px solid var(--border-soft);">
        <summary>对照策略数学公式、数据源、参数及近似版本声明</summary>
        <div class="table-wrap">
          <table>
            <thead><tr><th>策略</th><th>数学公式 / 定义</th><th>数据源与参数</th><th>近似说明与局限</th></tr></thead>
            <tbody id="resFormulaBody"></tbody>
          </table>
        </div>
      </details>
    </div>

    <!-- 5B. Market Valuation Regimes & Position Sizing -->
    <div class="card">
      <div class="card-head">
        <div>
          <h2 class="card-title">市场估值区间条件收益与仓位管理实证</h2>
          <span class="card-sub">严格区分历史条件统计与未来预测；重叠窗口经 Newey-West 调整并标注有效独立样本数 N_eff</span>
        </div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;">
          <div class="pills" id="resRegModePills">
            <button class="pill active" data-rm="price_buckets">7档价格中位数</button>
            <button class="pill" data-rm="quantile_buckets">5档滚动估值分位</button>
          </div>
          <div class="pills" id="resRegHPills">
            <button class="pill" data-rh="20">20D</button>
            <button class="pill active" data-rh="60">60D</button>
            <button class="pill" data-rh="120">120D</button>
          </div>
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>估值状态区间</th><th class="num">样本日 (N_eff)</th><th class="num">等权均值 / 中位数</th><th class="num">95% NW置信区间</th><th class="num">胜率</th><th class="num">均回撤</th><th class="num">LSM超额</th><th class="num">双低超额</th></tr></thead>
          <tbody id="resRegimeBody"></tbody>
        </table>
      </div>
      <div class="table-wrap" style="border-top:1px solid var(--border-soft);">
        <table>
          <thead><tr><th>仓位管理规则 (含2%货币现金收益与0.10%切换成本)</th><th class="num">平均仓位</th><th class="num">调仓次数</th><th class="num">年化收益</th><th class="num">最大回撤</th><th class="num">Sharpe</th><th class="num">Calmar</th><th class="num">牛市踏空差</th></tr></thead>
          <tbody id="resSizingBody"></tbody>
        </table>
      </div>
    </div>

    <!-- 5C. Time-Synced Factor IC / Quintiles / FMB -->
    <div class="card">
      <div class="card-head">
        <div>
          <h2 class="card-title" id="resFactorTitle">冻结 LSM 因子预测检验（Rank IC 随区间同步 · 五分组 · Fama-MacBeth）</h2>
          <span class="card-sub">切换任意时间范围后 Rank IC、ICIR 与 Newey-West 显著性同步重算</span>
        </div>
        <div class="pills" id="resIcPills">
          <button class="pill" data-rich="5">h=5</button>
          <button class="pill" data-rich="10">h=10</button>
          <button class="pill active" data-rich="20">h=20</button>
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>持有期 (同步当前时间范围)</th><th class="num">有效天数</th><th class="num">日均样本</th><th class="num">Mean Rank IC</th><th class="num">ICIR</th><th class="num">IC&gt;0比例</th><th class="num">Newey-West t</th><th class="num">全期 G5−G1 价差</th></tr>
          </thead>
          <tbody id="resIcBody"></tbody>
        </table>
      </div>
      <div class="grid-2" style="padding:14px 16px;border-top:1px solid var(--border-soft);margin-bottom:0;">
        <div><div style="font-size:12px;font-weight:600;margin-bottom:6px;">滚动 60 日 Rank IC 走势</div><div id="resRollSvg"></div></div>
        <div><div style="font-size:12px;font-weight:600;margin-bottom:6px;">五分组未来 20 日超额收益 (G1 高估 → G5 低估)</div><div id="resQBarSvg"></div></div>
      </div>
      <div class="table-wrap" style="border-top:1px solid var(--border-soft);">
        <table>
          <thead><tr><th>Fama-MacBeth 因果截面回归 (6因子控制 · 全期)</th><th class="num">均值斜率 (%)</th><th class="num">Newey-West t</th><th class="num">p 值</th><th>统计结论</th></tr></thead>
          <tbody id="resFmbBody"></tbody>
        </table>
      </div>
    </div>

    <!-- 5D. Stepwise Attribution & Walk-Forward OOS -->
    <div class="card">
      <div class="card-head">
        <div>
          <h2 class="card-title">独立边际贡献分解、Walk-Forward 样本外与多重检验过拟合控制</h2>
          <span class="card-sub" id="resPboSub">检验样本外衰减、Deflated Sharpe Ratio (DSR) 与 CSCV 过拟合概率 (PBO)</span>
        </div>
        <div class="pills" id="resStepPills">
          <button class="pill active" data-st="credit">1. 信用池</button>
          <button class="pill" data-st="risk">2. 风险剔除</button>
          <button class="pill" data-st="weight">3. 混合权重</button>
          <button class="pill" data-st="freq">4. 调仓频率</button>
          <button class="pill" data-st="buf">5. 缓冲机制</button>
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead id="resStepHead"></thead>
          <tbody id="resStepBody"></tbody>
        </table>
      </div>
      <div class="card-head" style="border-top:1px solid var(--border-soft);">
        <span style="font-size:12px;font-weight:600;" id="resOosChartTitle">Walk-Forward 样本外累计净值 vs 固定先验配置 (2021–2026)</span>
        <div class="pills" id="resOosPills">
          <button class="pill active" data-oos="expanding">扩展窗口 Walk-Forward</button>
          <button class="pill" data-oos="rolling2y">滚动 2 年 Walk-Forward</button>
        </div>
      </div>
      <div class="grid-2" style="padding:14px 16px;margin-bottom:0;">
        <div>
          <div class="legend">
            <span><i class="ldot" style="background:#0969da;"></i>Walk-Forward 动态样本外</span>
            <span><i class="ldot" style="background:#1a7f37;"></i>AA+ 50/50 固定先验</span>
            <span><i class="ldot" style="background:#8c959f;"></i>全市场原始 LSM 基准</span>
          </div>
          <div id="resOosSvg"></div>
        </div>
        <div class="table-wrap">
          <table>
            <thead><tr><th>测试年</th><th>训练窗</th><th>选中配置</th><th class="num">OOS收益</th><th class="num">OOS夏普</th></tr></thead>
            <tbody id="resOosYrBody"></tbody>
          </table>
        </div>
      </div>
      <div class="table-wrap" style="border-top:1px solid var(--border-soft);">
        <table>
          <thead><tr><th>先验策略 / 对照组 (2021–2026 同口径)</th><th class="num">OOS年化</th><th class="num">OOS回撤</th><th class="num">OOS夏普</th><th class="num">OOS Calmar</th><th class="num">全样本夏普</th><th class="num">DSR (N=450)</th></tr></thead>
          <tbody id="resDsrBody"></tbody>
        </table>
      </div>
      <div class="note-box" id="resPboNote">—</div>
    </div>

    <!-- Collapsible Full Experiments & Audit -->
    <div class="card">
      <details class="fold-card">
        <summary>分阶段实盘优化（持仓只数 5/10/20/20% · 加权方式 · 10万元可执行性测试）与全部 450 组实验</summary>
        <div class="table-wrap">
          <table>
            <thead><tr><th>分阶段实盘测试项 (10D · 20/30缓冲 · 50/50混合)</th><th class="num">连续年化</th><th class="num">回撤</th><th class="num">Sharpe</th><th class="num">Calmar</th><th class="num">¥10万实盘年化 (回撤 / 利用率)</th></tr></thead>
            <tbody id="resStageBBody"></tbody>
          </table>
        </div>
        <div class="toolbar" style="border-top:1px solid var(--border-soft);">
          <div class="filter-row">
            <select class="sel" id="gridFilterPool">
              <option value="ALL">全部评级池</option>
              <option value="all">全市场</option><option value="aa_minus">AA-及以上</option><option value="aa">AA及以上</option><option value="aa_plus">AA+及以上</option><option value="aaa">仅AAA</option>
            </select>
            <select class="sel" id="gridSortBy">
              <option value="sharpe_desc">按夏普比率降序</option>
              <option value="sharpe_asc">按夏普比率升序 (看最差负面结果)</option>
              <option value="calmar_desc">按 Calmar 比率降序</option>
              <option value="ann_desc">按年化收益降序</option>
            </select>
          </div>
          <span class="mono" style="font-size:11px;color:var(--muted);" id="gridCountLbl">—</span>
        </div>
        <div class="table-wrap">
          <table>
            <thead><tr><th>450组实验明细</th><th class="num">年化净收益</th><th class="num">最大回撤</th><th class="num">Sharpe</th><th class="num">Calmar</th><th class="num">年化换手</th></tr></thead>
            <tbody id="gridAllBody"></tbody>
          </table>
        </div>
      </details>
    </div>

    <div class="card">
      <details class="fold-card">
        <summary>最终回测真实性审计、数据源局限性披露与文献引用</summary>
        <div class="table-wrap">
          <table>
            <thead id="resAuditHead"></thead>
            <tbody id="resAuditBody"></tbody>
          </table>
        </div>
        <div class="note-box">
          <strong>数据局限性与审计披露：</strong>
          (1) <strong>剩余规模局限：</strong>免费历史数据源缺少 2020–2026 退市及存续转债每日转股后剩余余额序列，严禁使用末期剩余规模回填，小规模对照采用发行日已知初始发行规模（仅实时选券展示最新集思录剩余规模）；
          (2) <strong>YTM 与条款近似：</strong>历史 YTM 基于发行公告契约票息与本金加末年票息贴现，未逐券解析个性化超额赎回补偿（东财点时纯债价值已涵盖市场债底）；30 步二叉树为马尔可夫近似，忽略连续 15/30 日窗口计数与下修；BS 隐含波动率在正时间价值区间覆盖率 82.69%；
          (3) <strong>数据防泄漏：</strong>历史信用评级 99.93% 覆盖率（<code>allow_exact_matches=False</code> 次日生效），ST 区间与违约公告次日生效，T 日收盘信号严格 T+1 开盘价成交；
          (4) 文献：<a href="https://arxiv.org/abs/2409.06496" target="_blank" rel="noopener">Liu, Yu (2026), arXiv:2409.06496v7</a>。
        </div>
      </details>
    </div>
  </section>

  <footer>
    <span>Yu Liu (刘宇) · CUHK-Shenzhen · Four-Region LSM 可转债实证与前瞻模拟研究平台</span>
    <span>全部正负实验结果如实披露，模拟账户仅使用虚拟资金，不构成投资建议</span>
  </footer>
</main>

<script>
const DATA = __PAYLOAD_JSON__;
let EXP = DATA.exp_summary || {};
let EXP_MAP = {};
function indexExp(e) {
  EXP = e || {};
  EXP_MAP = {};
  (EXP.grid || []).forEach(r => { EXP_MAP[r.id] = r; });
}
indexExp(EXP);

let PRO = DATA.pro_summary || {};
let TIMING = DATA.timing_summary || {};
let PAPER = DATA.paper_accounts || {};
let DAILY_NAVS = null;
let DAILY_IC = null;

const POOLS = {
  all: { label: "全市场", grades: null },
  aa_minus: { label: "AA-及以上", grades: new Set(["AA-", "AA", "AA+", "AAA"]) },
  aa: { label: "AA及以上", grades: new Set(["AA", "AA+", "AAA"]) },
  aa_plus: { label: "AA+及以上", grades: new Set(["AA+", "AAA"]) },
  aaa: { label: "仅AAA", grades: new Set(["AAA"]) },
};

let curTab = "overview";
let curPool = "all";
let curRisk = "raw";
let curOvPreset = "lsm";
let curTimeRange = "since2023";
let customStart = "2023-01-01";
let customEnd = DATA.date || "2026-11-25";
let hiddenSeries = new Set();

let btWeight = "50/50";
let btFreq = 5;
let btBuf = "buf_20_30";
let btCost = "0.10%";
let btOverlay = "none";

let ppCohort = "aa_minus";
let ppAccId = "aam_balanced_v1";
let ppLogTab = "pending";

let curPeerScope = "aa_minus_filtered";
let curRegMode = "price_buckets";
let curRegH = "60";
let curResIcH = "20";
let curStepTab = "credit";
let curOosMode = "expanding";

const R = () => (DATA.research_pools || {})[curPool] || DATA.research || {};
const fmt = (v, n = 2) => typeof v === "number" && Number.isFinite(v) ? v.toFixed(n) : "—";
const signed = (v, n = 2, suf = "%") => typeof v === "number" && Number.isFinite(v) ? (v > 0 ? "+" : "") + v.toFixed(n) + suf : "—";
const cls = v => typeof v === "number" ? (v > 0 ? "pos" : v < 0 ? "neg" : "") : "";
const fmtP = p => (typeof p !== "number" || !Number.isFinite(p)) ? "—" : (p < 0.0001 ? "<0.0001" : p.toFixed(4));

function erfcApprox(x) {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * z);
  const y = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  const erfcZ = y * Math.exp(-z * z);
  return x >= 0 ? erfcZ : 2 - erfcZ;
}

function getDateBounds() {
  const dates = (DAILY_NAVS && DAILY_NAVS.dates) || ((EXP.meta || {}).nav_dates) || ["2020-01-06", "2026-11-25"];
  const lastDt = dates[dates.length - 1] || "2026-11-25";
  const firstDt = dates[0] || "2020-01-06";
  if (curTimeRange === "all") return [firstDt, lastDt, "全历史 2020–2026"];
  if (curTimeRange === "since2023") return ["2023-01-01", lastDt, "2023 年至今"];
  if (curTimeRange === "ytd") return [`${lastDt.slice(0, 4)}-01-01`, lastDt, `${lastDt.slice(0, 4)} 今年以来`];
  if (curTimeRange === "1y") {
    const dt = new Date(lastDt); dt.setFullYear(dt.getFullYear() - 1);
    return [dt.toISOString().slice(0, 10), lastDt, "近 1 年"];
  }
  if (curTimeRange === "3y") {
    const dt = new Date(lastDt); dt.setFullYear(dt.getFullYear() - 3);
    return [dt.toISOString().slice(0, 10), lastDt, "近 3 年"];
  }
  return [customStart || firstDt, customEnd || lastDt, `${customStart} → ${customEnd}`];
}

function sliceNavStats(seriesKey, costKey = "0.10%", isBench = false) {
  const [sDt, eDt] = getDateBounds();
  // Prefer full daily returns when loaded
  if (DAILY_NAVS && DAILY_NAVS.dates && DAILY_NAVS.dates.length) {
    const dts = DAILY_NAVS.dates;
    let rets = null;
    if (isBench) {
      if (costKey !== "0.10%" && DAILY_NAVS.cost_variants && DAILY_NAVS.cost_variants[`eq|${seriesKey}`]) {
        rets = DAILY_NAVS.cost_variants[`eq|${seriesKey}`][costKey];
      } else {
        rets = (DAILY_NAVS.benchmarks_010 || {})[seriesKey];
      }
    } else {
      if (costKey !== "0.10%" && DAILY_NAVS.cost_variants && DAILY_NAVS.cost_variants[seriesKey]) {
        rets = DAILY_NAVS.cost_variants[seriesKey][costKey];
      } else {
        rets = (DAILY_NAVS.series_010 || {})[seriesKey];
      }
    }
    if (rets && rets.length === dts.length) {
      const sc = DAILY_NAVS.scale || 1e-5;
      const outDts = [], outNav = [], subRets = [];
      let nav = 1.0, peak = 1.0, mdd = 0.0;
      for (let i = 0; i < dts.length; i++) {
        if (dts[i] < sDt || dts[i] > eDt) continue;
        const r = rets[i] * sc;
        nav = nav * (1.0 + r);
        if (nav > peak) peak = nav;
        const dd = (nav / peak - 1.0) * 100.0;
        if (dd < mdd) mdd = dd;
        outDts.push(dts[i]);
        outNav.push(nav);
        subRets.push(r);
      }
      const n = subRets.length;
      if (n >= 5) {
        const tot = (nav - 1.0) * 100.0;
        const ann = (Math.pow(nav, 252.0 / n) - 1.0) * 100.0;
        let sum = 0; for (let i = 0; i < n; i++) sum += subRets[i];
        const mean = sum / n;
        let ss = 0; for (let i = 0; i < n; i++) ss += (subRets[i] - mean) * (subRets[i] - mean);
        const sd = Math.sqrt(ss / Math.max(1, n - 1));
        const sharpe = (mean * 252.0 - 0.02) / (sd * Math.sqrt(252.0) + 1e-9);
        const calmar = mdd < -1e-6 ? ann / Math.abs(mdd) : 0.0;
        return { dates: outDts, navs: outNav, subRets, total_ret: tot, ann_ret: ann, mdd, sharpe, calmar, is_daily: true };
      }
    }
  }

  // Fallback to 10D nav_010 while daily_navs.json loads
  const row = isBench ? ((EXP.benchmarks || {})[seriesKey]) : EXP_MAP[seriesKey];
  const dts = (EXP.meta || {}).nav_dates || [];
  const rawNav = row?.nav_010 || [];
  const fullCost = (row && row.costs && row.costs[costKey]) || {};
  if (!rawNav.length || rawNav.length !== dts.length) {
    return { dates: dts, navs: rawNav, subRets: [], total_ret: fullCost.total_ret, ann_ret: fullCost.ann_ret, mdd: fullCost.mdd, sharpe: fullCost.sharpe, calmar: fullCost.calmar, is_daily: false };
  }
  const outDts = [], outNav = [];
  let base = null, peak = 1.0, mdd = 0.0;
  for (let i = 0; i < dts.length; i++) {
    if (dts[i] < sDt || dts[i] > eDt) continue;
    if (base === null) base = rawNav[i];
    const nv = rawNav[i] / Math.max(1e-6, base);
    if (nv > peak) peak = nv;
    const dd = (nv / peak - 1.0) * 100.0;
    if (dd < mdd) mdd = dd;
    outDts.push(dts[i]); outNav.push(nv);
  }
  if (curTimeRange === "all" && fullCost.ann_ret !== undefined) {
    return { dates: outDts, navs: outNav, subRets: [], total_ret: fullCost.total_ret, ann_ret: fullCost.ann_ret, mdd: fullCost.mdd, sharpe: fullCost.sharpe, calmar: fullCost.calmar, is_daily: false };
  }
  const nBars = Math.max(1, outNav.length - 1);
  const yrs = nBars * (10.0 / 252.0);
  const endNav = outNav[outNav.length - 1] || 1.0;
  const tot = (endNav - 1.0) * 100.0;
  const ann = yrs > 0 ? (Math.pow(endNav, 1.0 / yrs) - 1.0) * 100.0 : tot;
  return { dates: outDts, navs: outNav, subRets: [], total_ret: tot, ann_ret: ann, mdd, sharpe: fullCost.sharpe, calmar: mdd < -1e-6 ? ann / Math.abs(mdd) : 0, is_daily: false };
}

function computePeriodIc(poolKey, hStr) {
  const [sDt, eDt] = getDateBounds();
  if (DAILY_IC && DAILY_IC.pools && DAILY_IC.pools[poolKey]) {
    const idxCol = hStr === "5" ? 2 : hStr === "10" ? 3 : 4;
    const vals = [], ns = [];
    DAILY_IC.pools[poolKey].forEach(r => {
      if (r[0] >= sDt && r[0] <= eDt && typeof r[idxCol] === "number") {
        vals.push(r[idxCol]); ns.push(r[1]);
      }
    });
    const T = vals.length;
    if (T >= 10) {
      const mean = vals.reduce((a, b) => a + b, 0) / T;
      const e = vals.map(v => v - mean);
      let ss = 0; e.forEach(v => { ss += v * v; });
      const std = Math.sqrt(ss / Math.max(1, T - 1));
      const L = Math.max(1, Math.min(Number(hStr), T - 2));
      let nwVar = ss / T;
      for (let lag = 1; lag <= L; lag++) {
        const w = 1.0 - lag / (L + 1.0);
        let cov = 0;
        for (let i = lag; i < T; i++) cov += e[i] * e[i - lag];
        nwVar += 2.0 * w * (cov / T);
      }
      nwVar = Math.max(1e-16, nwVar);
      const se = Math.sqrt(nwVar / T);
      const tNw = se > 0 ? mean / se : 0;
      const pVal = erfcApprox(Math.abs(tNw) / Math.SQRT2);
      const avgN = ns.reduce((a, b) => a + b, 0) / T;
      const posPct = vals.filter(v => v > 0).length / T * 100.0;
      return { days: T, avg_n: avgN, mean_ic: mean, icir: std > 0 ? mean / std : 0, pos_pct: posPct, t_nw: tNw, p_val: pVal, synced: true };
    }
  }
  const base = ((R().ic_summary || {})[hStr]) || {};
  return { ...base, synced: false };
}

function livePool() {
  const gs = POOLS[curPool].grades;
  return (DATA.all_bonds || []).filter(b => {
    if (gs && !gs.has((b.rating || "").toUpperCase())) return false;
    if (curRisk === "filtered" && b.is_risk_excluded) return false;
    return true;
  });
}
function getExpRow(pool, risk, weight, freq, buf) { return EXP_MAP[`${pool}|${risk}|${weight}|${freq}D|${buf}`] || null; }
function ratingBadge(rt) {
  const g = (rt || "").toUpperCase();
  if (!g) return `<span class="badge b-sig-n">未披露</span>`;
  const c = g === "AAA" ? "b-aaa" : g === "AA+" ? "b-aap" : g === "AA" ? "b-aa" : g === "AA-" ? "b-aam" : "b-low";
  return `<span class="badge ${c} mono">${g}</span>`;
}
function riskTags(b) {
  const tags = [];
  if (b.is_st) tags.push(`<span class="badge b-low">ST正股</span>`);
  if (b.has_spec_rat) tags.push(`<span class="badge b-low">低信用${b.rating}</span>`);
  if (b.is_low_liq) tags.push(`<span class="badge b-warn">低成交&lt;500万</span>`);
  if (b.region_code === "R1") tags.push(`<span class="badge b-warn">强赎R1</span>`);
  if (b.bond_price > 150) tags.push(`<span class="badge b-warn">高价&gt;150</span>`);
  if (b.premium_rate > 80) tags.push(`<span class="badge b-warn">高溢价</span>`);
  return tags.length ? tags.join("") : `<span style="font-size:11px;color:var(--light);">常规</span>`;
}

function renderInteractiveChart(svgId, tipId, legId, dates, seriesDefs, W = 540, H = 215, rightPad = 82) {
  const svgBox = document.getElementById(svgId);
  const legBox = legId ? document.getElementById(legId) : null;
  if (legBox) {
    legBox.innerHTML = seriesDefs.filter(s => s.vals && s.vals.length).map(s =>
      `<button class="legend-btn ${hiddenSeries.has(s.key) ? "off" : ""}" data-skey="${s.key}"><i class="ldot" style="background:${s.col};"></i>${s.label}</button>`
    ).join("");
    legBox.querySelectorAll(".legend-btn").forEach(btn => {
      btn.onclick = () => {
        const k = btn.dataset.skey;
        if (hiddenSeries.has(k)) hiddenSeries.delete(k); else hiddenSeries.add(k);
        renderOverview(); renderBacktest();
      };
    });
  }
  const activeS = seriesDefs.filter(s => s.vals && s.vals.length && !hiddenSeries.has(s.key));
  if (!dates || !dates.length || !activeS.length) {
    svgBox.innerHTML = `<div style="padding:28px;text-align:center;color:var(--muted);">暂无净值曲线或图例已全部隐藏</div>`;
    return;
  }
  const pL = 42, pR = rightPad, pT = 14, pB = 25;
  const allV = activeS.flatMap(s => s.vals);
  const mn = Math.min(0.95, ...allV) * 0.97, mx = Math.max(1.12, ...allV) * 1.03;
  const px = i => pL + (i / Math.max(1, dates.length - 1)) * (W - pL - pR);
  const py = v => H - pB - ((v - mn) / Math.max(1e-6, mx - mn)) * (H - pT - pB);
  let grid = "";
  const span = mx - mn;
  const step = span > 1.5 ? 0.5 : span > 0.6 ? 0.25 : span > 0.25 ? 0.1 : 0.05;
  for (let lv = Math.floor(mn / step) * step; lv <= mx + 1e-6; lv += step) {
    const val = Math.round(lv * 100) / 100;
    if (val >= mn && val <= mx) {
      const y = py(val).toFixed(1);
      grid += `<line x1="${pL}" y1="${y}" x2="${W - pR}" y2="${y}" stroke="${Math.abs(val - 1.0) < 0.005 ? "#8c959f" : "#eaeef2"}" stroke-dasharray="${Math.abs(val - 1.0) < 0.005 ? "" : "3 3"}"/>
        <text x="${pL - 6}" y="${Number(y) + 3}" text-anchor="end" font-size="10" fill="#656d76">${val.toFixed(2)}</text>`;
    }
  }
  [0, Math.floor(dates.length / 2), dates.length - 1].forEach(idx => {
    grid += `<text x="${px(idx).toFixed(1)}" y="${H - 6}" text-anchor="middle" font-size="10" fill="#656d76">${dates[idx]}</text>`;
  });
  // Downsample rendering points smoothly if daily > 400 pts so SVG polyline stays ultra crisp, while keeping exact hover on all points
  const stride = dates.length > 450 ? Math.ceil(dates.length / 400) : 1;
  activeS.forEach(s => {
    const pts = [];
    for (let i = 0; i < s.vals.length; i += stride) pts.push(`${px(i).toFixed(1)},${py(s.vals[i]).toFixed(1)}`);
    if ((s.vals.length - 1) % stride !== 0) pts.push(`${px(s.vals.length - 1).toFixed(1)},${py(s.vals[s.vals.length - 1]).toFixed(1)}`);
    const endV = s.vals[s.vals.length - 1];
    grid += `<polyline fill="none" stroke="${s.col}" stroke-width="${s.w || 2}" ${s.dash ? `stroke-dasharray="${s.dash}"` : ""} points="${pts.join(" ")}"/>
      <text x="${W - pR + 5}" y="${(py(endV) + 3).toFixed(1)}" font-size="10" font-weight="600" fill="${s.col}">${endV.toFixed(2)} (${signed((endV - 1) * 100, 1)})</text>`;
  });
  grid += `<line id="${svgId}_xh" x1="0" y1="${pT}" x2="0" y2="${H - pB}" stroke="#656d76" stroke-width="1" stroke-dasharray="2 2" style="display:none;"/>`;
  svgBox.innerHTML = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block;cursor:crosshair;">${grid}</svg>`;

  const tipEl = document.getElementById(tipId);
  const svgEl = svgBox.querySelector("svg");
  const xh = document.getElementById(`${svgId}_xh`);
  if (svgEl && tipEl) {
    svgEl.onmousemove = ev => {
      const rect = svgEl.getBoundingClientRect();
      const relX = ((ev.clientX - rect.left) / Math.max(1, rect.width)) * W;
      const frac = Math.max(0, Math.min(1, (relX - pL) / Math.max(1, W - pL - pR)));
      const idx = Math.min(dates.length - 1, Math.max(0, Math.round(frac * (dates.length - 1))));
      if (xh) { xh.setAttribute("x1", px(idx).toFixed(1)); xh.setAttribute("x2", px(idx).toFixed(1)); xh.style.display = "block"; }
      tipEl.style.display = "block";
      tipEl.innerHTML = `<div style="font-weight:700;margin-bottom:3px;border-bottom:1px solid var(--border-soft);padding-bottom:2px;">${dates[idx]}</div>` +
        activeS.map(s => `<div style="display:flex;justify-content:space-between;gap:10px;"><span style="color:${s.col};">● ${s.short || s.label}</span><strong>${(s.vals[idx] || 1).toFixed(4)} (${signed(((s.vals[idx] || 1) - 1) * 100, 1)})</strong></div>`).join("");
    };
    svgEl.onmouseleave = () => { tipEl.style.display = "none"; if (xh) xh.style.display = "none"; };
  }
}

function drawSimpleLines(dates, seriesList, W = 520, H = 205, rightPad = 74) {
  if (!dates || !dates.length) return `<div style="padding:24px;text-align:center;color:var(--muted);">暂无序列</div>`;
  const validS = seriesList.filter(s => s.vals && s.vals.length);
  if (!validS.length) return `<div style="padding:24px;text-align:center;color:var(--muted);">暂无数据</div>`;
  const pL = 42, pR = rightPad, pT = 14, pB = 24;
  const allV = validS.flatMap(s => s.vals);
  const mn = Math.min(0.95, ...allV) * 0.97, mx = Math.max(1.12, ...allV) * 1.03;
  const px = i => pL + (i / Math.max(1, dates.length - 1)) * (W - pL - pR);
  const py = v => H - pB - ((v - mn) / Math.max(1e-6, mx - mn)) * (H - pT - pB);
  let g = "";
  const step = mx - mn > 1.2 ? 0.4 : mx - mn > 0.5 ? 0.2 : 0.1;
  for (let lv = Math.floor(mn / step) * step; lv <= mx + 1e-6; lv += step) {
    const val = Math.round(lv * 100) / 100;
    if (val >= mn && val <= mx) {
      const y = py(val).toFixed(1);
      g += `<line x1="${pL}" y1="${y}" x2="${W-pR}" y2="${y}" stroke="${Math.abs(val-1)<0.005?"#8c959f":"#eaeef2"}" stroke-dasharray="${Math.abs(val-1)<0.005?"":"3 3"}"/><text x="${pL-5}" y="${Number(y)+3}" text-anchor="end" font-size="10" fill="#656d76">${val.toFixed(2)}</text>`;
    }
  }
  [0, Math.floor(dates.length / 2), dates.length - 1].forEach(i => { g += `<text x="${px(i).toFixed(1)}" y="${H-6}" text-anchor="middle" font-size="10" fill="#656d76">${dates[i]}</text>`; });
  validS.forEach(s => {
    const poly = s.vals.map((v, i) => `${px(i).toFixed(1)},${py(v).toFixed(1)}`).join(" ");
    const endV = s.vals[s.vals.length - 1];
    g += `<polyline fill="none" stroke="${s.col}" stroke-width="${s.w||2}" ${s.dash?`stroke-dasharray="${s.dash}"`:""} points="${poly}"/><text x="${W-pR+5}" y="${(py(endV)+3).toFixed(1)}" font-size="10" font-weight="600" fill="${s.col}">${endV.toFixed(2)}</text>`;
  });
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block;">${g}</svg>`;
}

function renderStatus() {
  const cov = ((EXP.meta || {}).pool_coverage || {})[`${curPool}_${curRisk}`] || {};
  const lp = livePool();
  const rTxt = curRisk === "filtered" ? "已剔除 ST·违约·低流动性" : "不剔除";
  const sess = DATA.session || {};
  document.getElementById("researchStatusText").innerHTML =
    `当前筛选：<strong>${POOLS[curPool].label} · ${rTxt}</strong>（历史 ${cov.unique_bonds || (R().universe||{}).total_bonds || 0} 只 · 实盘 <strong>${lp.length}</strong> 只）`;
  document.getElementById("sessionBannerRight").textContent =
    `行情日 ${sess.market_date || DATA.date} (${sess.phase || "EOD"}) · 系统更新 ${DATA.updated_at}`;
  const w = document.getElementById("poolWarnBanner");
  if (curPool === "aaa") {
    w.style.display = "block";
    w.innerHTML = `<strong>样本量警示（仅AAA）：</strong>点时 AAA 池日均仅 29 只（2020 年 15–21 只），高度集中于银行及大型国企，2025–2026 年 IC 转弱且夏普跌至 0.06，以下数据完整保留供对照。`;
  } else {
    w.style.display = "none";
  }
}

function syncTimeBars() {
  document.querySelectorAll("[data-tbar] .pill").forEach(b => b.classList.toggle("active", b.dataset.tr === curTimeRange));
  document.getElementById("ovCustomBox").classList.toggle("active", curTimeRange === "custom");
  document.getElementById("btCustomBox").classList.toggle("active", curTimeRange === "custom");
  document.getElementById("trStartInp").value = customStart; document.getElementById("trEndInp").value = customEnd;
  document.getElementById("btStartInp").value = customStart; document.getElementById("btEndInp").value = customEnd;
}

function renderOverview() {
  const [, , trLbl] = getDateBounds();
  const lsmId = `${curPool}|${curRisk}|100/0|5D|none`;
  const dlId = `${curPool}|${curRisk}|0/100|5D|none`;
  const balId = "cand|enhanced_balanced";
  const defId = "cand|enhanced_defensive";
  const eqKey = `${curPool}_${curRisk}`;

  const sLsm = sliceNavStats(lsmId, "0.10%", false);
  const sBal = sliceNavStats(balId, "0.10%", false);
  const sDef = sliceNavStats(defId, "0.10%", false);
  const sDl = sliceNavStats(dlId, "0.10%", false);
  const sEq = sliceNavStats(eqKey, "0.10%", true);

  const actMap = { lsm: [sLsm, "原始 LSM", lsmId], balanced: [sBal, "Enhanced Balanced(AA-)", balId], defensive: [sDef, "Enhanced Defensive(AA)", defId], dl: [sDl, "传统双低", dlId] };
  const [actS, actName, actId] = actMap[curOvPreset] || actMap.lsm;
  const fullRow = EXP_MAP[actId] || (PRO.candidates || []).find(c => `cand|${c.key}` === actId) || {};

  document.getElementById("ovKpi1Lbl").textContent = `费后年化 (${actName} · ${trLbl})`;
  const aEl = document.getElementById("ovAnnRet");
  aEl.textContent = signed(actS.ann_ret); aEl.className = `kpi-val mono ${cls(actS.ann_ret)}`;
  document.getElementById("ovAnnSub").textContent = `区间累计 ${signed(actS.total_ret)} · 同池基准 ${signed(sEq.ann_ret)}`;
  document.getElementById("ovCalmar").textContent = `${fmt(actS.calmar, 2)} / ${fmt(actS.sharpe, 2)}`;
  document.getElementById("ovCalmarSub").textContent = `Calmar = ${fmt(actS.calmar, 2)} · Sharpe = ${fmt(actS.sharpe, 2)}`;
  document.getElementById("ovMdd").textContent = `${fmt(actS.mdd)}%`;
  document.getElementById("ovMddSub").textContent = `全期年化换手 ${fmt(fullRow?.ann_to, 1)}% · 0.10% 成本`;

  const ic20 = computePeriodIc(curPool, "20");
  document.getElementById("ovIcLbl").textContent = `20 日 Rank IC (${trLbl})`;
  const icEl = document.getElementById("ovIc");
  icEl.textContent = typeof ic20.mean_ic === "number" ? (ic20.mean_ic > 0 ? "+" : "") + ic20.mean_ic.toFixed(4) : "—";
  icEl.className = `kpi-val mono ${cls(ic20.mean_ic)}`;
  document.getElementById("ovIcSub").textContent = `ICIR ${fmt(ic20.icir, 3)} · 正比例 ${fmt(ic20.pos_pct, 1)}% · t=${fmt(ic20.t_nw, 2)}`;

  // Market Valuation & Timing Temperature Card (P2 merged cleanly)
  const lp = livePool();
  const prices = lp.map(b => b.bond_price).filter(Number.isFinite).sort((a, b) => a - b);
  const prems = lp.map(b => b.premium_rate).filter(Number.isFinite).sort((a, b) => a - b);
  const med = arr => arr.length ? arr[Math.floor(arr.length / 2)] : null;
  const underPct = lp.length ? (lp.filter(b => b.undervaluation > 0).length / lp.length) * 100 : 0;
  const tPool = ((TIMING.latest_by_pool || {})[curPool]) || {};
  document.getElementById("tempMetrics").innerHTML = `
    <div><span>实盘</span><strong class="mono">${lp.length}只</strong></div>
    <div><span>中位价</span><strong class="mono">¥${fmt(med(prices))}</strong> <span class="mono">(${fmt(tPool.med_price_pct,0)}分位)</span></div>
    <div><span>百元拟合溢价</span><strong class="mono">${fmt(tPool.fitted_prem_100,1)}%</strong> <span class="mono">(${fmt(tPool.fitted_prem_100_pct,0)}分位)</span></div>
    <div><span>中位YTM</span><strong class="mono">${fmt(tPool.med_ytm,2)}%</strong></div>
    <div><span>LSM低估占比</span><strong class="mono">${fmt(underPct,1)}%</strong></div>
  `;
  const vPct = tPool.val_comp_pct;
  const qTag = vPct < 20 ? "Q1 极低估(100%)" : vPct < 40 ? "Q2 偏低估(90%)" : vPct < 60 ? "Q3 中性(80%)" : vPct < 80 ? "Q4 偏高估(65%)" : "Q5 高估(50%)";
  const sizingTbl = (((TIMING.position_sizing || {})[curPool] || {}).on_balanced_strategy) || [];
  const tierRow = sizingTbl.find(x => x.key === "val_tier");
  document.getElementById("tempTimingLine").innerHTML =
    `点时综合估值分位 <strong>${fmt(vPct, 1)}%（${qTag}）</strong> · 历史均值口径仅作仓位与风险提示，不构成未来预测或安全买点` +
    (tierRow ? `（同口径全市场/分池估值分档历史回撤可收敛至 ${fmt(tierRow.mdd,1)}%）` : "");

  const rc = { R1: 0, R2: 0, R3: 0, R4: 0 };
  lp.forEach(b => { if (rc[b.region_code] !== undefined) rc[b.region_code] += 1; });
  const cols = { R1: "#cf222e", R2: "#0969da", R3: "#1a7f37", R4: "#9a6700" };
  const tot = Math.max(1, lp.length);
  document.getElementById("tempRegBar").innerHTML = ["R1", "R2", "R3", "R4"].map(k => `<div style="width:${(rc[k]/tot*100).toFixed(1)}%;background:${cols[k]};"></div>`).join("");
  document.getElementById("tempRegLegend").innerHTML = ["R1", "R2", "R3", "R4"].map(k => `<span><i style="display:inline-block;width:7px;height:7px;border-radius:2px;background:${cols[k]};margin-right:3px;"></i>${k}:${rc[k]}</span>`).join("");

  document.getElementById("ovChartTitle").textContent = `核心策略净值对比（${trLbl} · 0.10% 成本）`;
  renderInteractiveChart("ovNavSvg", "ovTip", "ovLegend", sLsm.dates, [
    { key: "ov_lsm", label: `同池原始 LSM`, short: "原始LSM", vals: sLsm.navs, col: "#0969da", w: 2.1 },
    { key: "ov_bal", label: `Enhanced Balanced (AA-·20只)`, short: "Balanced", vals: sBal.navs, col: "#1a7f37", w: 2.0 },
    { key: "ov_def", label: `Enhanced Defensive (AA·20只逆波)`, short: "Defensive", vals: sDef.navs, col: "#8250df", w: 1.7 },
    { key: "ov_dl", label: `同池传统双低`, short: "双低", vals: sDl.navs, col: "#9a6700", w: 1.6 },
    { key: "ov_eq", label: `同池等权基准`, short: "基准", vals: sEq.navs, col: "#8c959f", w: 1.4, dash: "5 4" },
  ], 540, 215, 84);

  document.getElementById("ovMiniBody").innerHTML = [
    ["同池原始 LSM (100/0·5D)", sLsm],
    ["Enhanced Balanced (AA-·20只)", sBal],
    ["Enhanced Defensive (AA·20只逆波)", sDef],
    ["同池传统双低 (0/100·5D)", sDl],
    ["同池等权基准", sEq],
  ].map(([nm, st]) => {
    const ex = typeof st.ann_ret === "number" && typeof sEq.ann_ret === "number" ? st.ann_ret - sEq.ann_ret : null;
    return `<tr><td><strong>${nm}</strong></td><td class="num mono ${cls(st.ann_ret)}">${signed(st.ann_ret)}</td><td class="num mono neg">${fmt(st.mdd)}%</td><td class="num mono">${fmt(st.sharpe, 2)}</td><td class="num mono">${fmt(st.calmar, 2)}</td><td class="num mono ${cls(ex)}">${nm.includes("基准") ? "—" : signed(ex)}</td></tr>`;
  }).join("");

  document.getElementById("ovTopTitle").textContent = `当前低估转债 Top 10（${POOLS[curPool].label}）`;
  const top10 = [...lp].sort((a, b) => b.undervaluation - a.undervaluation).slice(0, 10);
  document.getElementById("ovTopBody").innerHTML = top10.map(b => `
    <tr>
      <td><strong>${b.name}</strong><span class="sub mono">${b.ticker} · ${b.stock_name}</span></td>
      <td>${ratingBadge(b.rating)}</td>
      <td class="num mono">¥${fmt(b.bond_price)}</td>
      <td class="num mono">¥${fmt(b.model_price)}</td>
      <td class="num mono ${b.undervaluation > 0 ? "pos" : "neg"}">${signed(b.undervaluation)}</td>
    </tr>
  `).join("") || `<tr><td colspan="5" style="text-align:center;color:var(--muted);">当前范围内无符合条件转债</td></tr>`;
}

function renderScreener() {
  const q = (document.getElementById("scrSearch").value || "").trim().toLowerCase();
  const reg = document.getElementById("scrReg").value;
  const sortBy = document.getElementById("scrSort").value;
  const safeOnly = document.getElementById("scrSafeOnly").checked;
  const lpAll = livePool();
  let items = lpAll.filter(b => {
    if (reg !== "ALL" && b.region_code !== reg) return false;
    if (safeOnly && (b.bond_price > 150 || b.premium_rate > 80)) return false;
    if (q && !`${b.ticker} ${b.name} ${b.stock_code} ${b.stock_name}`.toLowerCase().includes(q)) return false;
    return true;
  });
  if (sortBy === "underval") items.sort((a, b) => b.undervaluation - a.undervaluation);
  else if (sortBy === "price") items.sort((a, b) => a.bond_price - b.bond_price);
  else if (sortBy === "dl") items.sort((a, b) => a.double_low - b.double_low);
  else if (sortBy === "prem") items.sort((a, b) => a.premium_rate - b.premium_rate);

  document.getElementById("scrTitle").textContent = `实时转债估值选券池（${POOLS[curPool].label} · ${curRisk === "filtered" ? "已剔除风险" : "不剔除"}）`;
  document.getElementById("scrCountNote").textContent = `命中 ${items.length} / ${lpAll.length} 只`;
  document.getElementById("scrTableBody").innerHTML = items.slice(0, 150).map((b, i) => `
    <tr>
      <td class="mono" style="text-align:center;color:var(--muted);">${i + 1}</td>
      <td><strong>${b.name}</strong><span class="sub mono">${b.ticker} · ${b.stock_name} (正股 ¥${fmt(b.stock_price)} · 成交 ${b.amt_wan !== null ? fmt(b.amt_wan, 0) + "万" : "—"}${b.rem_scale ? " · 余" + b.rem_scale + "亿" : ""})</span></td>
      <td>${ratingBadge(b.rating)}</td>
      <td class="num mono">¥${fmt(b.bond_price)}</td>
      <td class="num mono"><strong>¥${fmt(b.model_price)}</strong></td>
      <td class="num mono ${b.undervaluation > 0 ? "pos" : "neg"}">${signed(b.undervaluation)}</td>
      <td class="num mono">${fmt(b.premium_rate, 1)}%<span class="sub">双低 ${fmt(b.double_low, 1)}</span></td>
      <td><span class="badge b-sig-n">${b.region_code} · ${b.region_name}</span></td>
      <td>${riskTags(b)}</td>
    </tr>
  `).join("") || `<tr><td colspan="9" style="text-align:center;padding:24px;color:var(--muted);">无匹配转债</td></tr>`;
}

function renderBacktest() {
  const [, , trLbl] = getDateBounds();
  const rowId = `${curPool}|${curRisk}|${btWeight}|${btFreq}D|${btBuf}`;
  const lsmId = `${curPool}|${curRisk}|100/0|5D|none`;
  const dlId = `${curPool}|${curRisk}|0/100|5D|none`;
  const eqKey = `${curPool}_${curRisk}`;

  const row = EXP_MAP[rowId];
  const rLsm = EXP_MAP[lsmId];
  const rDl = EXP_MAP[dlId];
  const cov = ((EXP.meta || {}).pool_coverage || {})[eqKey] || {};

  const sSel = sliceNavStats(rowId, btCost, false);
  const sLsm = sliceNavStats(lsmId, btCost, false);
  const sDl = sliceNavStats(dlId, btCost, false);
  const sEq = sliceNavStats(eqKey, btCost, true);
  const sOvl = btOverlay !== "none" ? sliceNavStats(btOverlay, btCost, false) : null;

  document.getElementById("btComboId").textContent = `${POOLS[curPool].label} · ${curRisk==="filtered"?"剔除":"不剔除"} · ${btWeight} · ${btFreq}D · ${btBuf==="none"?"无缓冲":btBuf==="buf_20_30"?"20/30缓冲":"20/40缓冲"} · ${trLbl}`;
  document.getElementById("btAnnRet").textContent = signed(sSel.ann_ret);
  document.getElementById("btAnnRet").className = `kpi-val mono ${cls(sSel.ann_ret)}`;
  document.getElementById("btTotRet").textContent = `区间累计 ${signed(sSel.total_ret)} (全期换手 ${fmt(row?.ann_to, 1)}%)`;
  document.getElementById("btMdd").textContent = `${fmt(sSel.mdd)}%`;
  document.getElementById("btCostNote").textContent = `单边成本 ${btCost} · ${sSel.is_daily ? "日频精准重建" : "10日降采样预览"}`;
  document.getElementById("btSharpe").textContent = `${fmt(sSel.sharpe, 2)} / ${fmt(sSel.calmar, 2)}`;
  document.getElementById("btCalmar").textContent = `Calmar = ${fmt(sSel.calmar, 2)} · Sharpe = ${fmt(sSel.sharpe, 2)}`;
  const ex = typeof sSel.ann_ret === "number" && typeof sEq.ann_ret === "number" ? sSel.ann_ret - sEq.ann_ret : null;
  document.getElementById("btExcess").textContent = signed(ex);
  document.getElementById("btExcess").className = `kpi-val mono ${cls(ex)}`;
  document.getElementById("btPoolStats").textContent = `样本 ${cov.unique_bonds || "—"} 只 · 日均 ${fmt(cov.avg_daily, 1)} 只`;

  const sDefs = [
    { key: "bt_sel", label: `所选配置 (${btWeight}·${btFreq}D)`, short: "所选", vals: sSel.navs, col: "#0969da", w: 2.3 },
    { key: "bt_lsm", label: `同池原始 LSM`, short: "原始LSM", vals: sLsm.navs, col: "#8250df", w: 1.6 },
    { key: "bt_dl", label: `同池传统双低`, short: "双低", vals: sDl.navs, col: "#9a6700", w: 1.6 },
    { key: "bt_eq", label: `同池等权基准`, short: "基准", vals: sEq.navs, col: "#8c959f", w: 1.4, dash: "5 4" },
  ];
  if (sOvl && sOvl.navs && sOvl.navs.length) {
    const lblO = document.getElementById("btOverlaySel").selectedOptions[0].text.replace("+ 叠加 ", "");
    sDefs.splice(1, 0, { key: "bt_ovl", label: lblO, short: "叠加", vals: sOvl.navs, col: "#1a7f37", w: 2.0 });
  }
  renderInteractiveChart("btNavSvg", "btTip", "btLegend", sSel.dates, sDefs, 920, 235, 86);

  // Rolling 12M Excess Return (252 trading days)
  const rSvg = document.getElementById("btRollExcessSvg");
  if (sSel.is_daily && sEq.is_daily && sSel.navs.length > 260 && sSel.navs.length === sEq.navs.length) {
    const dts = [], exVals = [];
    for (let i = 252; i < sSel.navs.length; i += 5) {
      const r1 = (sSel.navs[i] / sSel.navs[i - 252] - 1) * 100;
      const r0 = (sEq.navs[i] / sEq.navs[i - 252] - 1) * 100;
      dts.push(sSel.dates[i]); exVals.push(r1 - r0);
    }
    const W = 500, H = 145, pL = 40, pR = 14, pT = 12, pB = 22;
    const mn = Math.min(-5, ...exVals) - 2, mx = Math.max(10, ...exVals) + 2;
    const px = i => pL + (i / Math.max(1, dts.length - 1)) * (W - pL - pR);
    const py = v => H - pB - ((v - mn) / Math.max(1e-6, mx - mn)) * (H - pT - pB);
    const zY = py(0).toFixed(1);
    const poly = exVals.map((v, i) => `${px(i).toFixed(1)},${py(v).toFixed(1)}`).join(" ");
    rSvg.innerHTML = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block;">
      <line x1="${pL}" y1="${zY}" x2="${W-pR}" y2="${zY}" stroke="#8c959f"/>
      <text x="${pL-4}" y="${Number(zY)+3}" text-anchor="end" font-size="10" fill="#656d76">0%</text>
      <text x="${pL-4}" y="${pT+8}" text-anchor="end" font-size="10" fill="#656d76">${mx.toFixed(0)}%</text>
      <text x="${pL-4}" y="${H-pB}" text-anchor="end" font-size="10" fill="#656d76">${mn.toFixed(0)}%</text>
      <text x="${pL}" y="${H-5}" font-size="10" fill="#656d76">${dts[0]}</text>
      <text x="${W-pR}" y="${H-5}" text-anchor="end" font-size="10" fill="#656d76">${dts[dts.length-1]}</text>
      <polyline fill="none" stroke="#0969da" stroke-width="1.8" points="${poly}"/></svg>`;
  } else {
    rSvg.innerHTML = `<div style="padding:18px;text-align:center;color:var(--muted);font-size:12px;">切换至「全部 / 2023至今 / 近3年」并待日频净值载入后显示 12 个月滚动超额曲线</div>`;
  }

  const aSel = (row?.costs?.[btCost] || {}).annual || {};
  const aLsm = (rLsm?.costs?.[btCost] || {}).annual || {};
  const aDl = (rDl?.costs?.[btCost] || {}).annual || {};
  const yrs = Object.keys(aSel).sort();
  document.getElementById("btAnnualBody").innerHTML = yrs.map(yr => `
    <tr>
      <td class="mono"><strong>${yr}${yr===yrs[yrs.length-1]?" YTD":""}</strong></td>
      <td class="num mono ${cls(aSel[yr]?.ret)}"><strong>${signed(aSel[yr]?.ret)}</strong> <span class="sub">MDD ${fmt(aSel[yr]?.mdd)}%</span></td>
      <td class="num mono ${cls(aLsm[yr]?.ret)}">${signed(aLsm[yr]?.ret)}</td>
      <td class="num mono ${cls(aDl[yr]?.ret)}">${signed(aDl[yr]?.ret)}</td>
    </tr>
  `).join("") || `<tr><td colspan="4" style="text-align:center;color:var(--muted);">加载完整实验网格后显示</td></tr>`;

  const pOrder = ["all", "aa_minus", "aa", "aa_plus", "aaa"];
  let pRows = "";
  pOrder.forEach(pk => {
    ["raw", "filtered"].forEach(rm => {
      const pr = getExpRow(pk, rm, btWeight, btFreq, btBuf);
      if (!pr) return;
      const pc = (pr.costs || {})[btCost] || {};
      const active = pk === curPool && rm === curRisk;
      pRows += `<tr class="${active ? "hl-row" : ""}">
        <td><strong>${POOLS[pk].label}</strong> <span class="badge ${rm==="filtered"?"b-sig-y":"b-sig-n"}">${rm==="filtered"?"剔除":"不剔除"}</span></td>
        <td class="num mono ${cls(pc.ann_ret)}">${signed(pc.ann_ret)}</td>
        <td class="num mono neg">${fmt(pc.mdd)}%</td>
        <td class="num mono"><strong>${fmt(pc.sharpe, 2)}</strong></td>
        <td class="num mono">${fmt(pr.ann_to, 1)}%</td>
      </tr>`;
    });
  });
  document.getElementById("bt5PoolBody").innerHTML = pRows;
}

function renderPaper() {
  const sess = PAPER.session || DATA.session || {};
  const accs = PAPER.accounts || {};
  const cohortList = Object.values(accs).filter(a => a.cohort === ppCohort);
  if (cohortList.length && !cohortList.some(a => a.account_id === ppAccId)) ppAccId = cohortList[0].account_id;

  const badge = document.getElementById("ppStatusBadge");
  if (PAPER.data_status === "STALE_PRESERVED") {
    badge.className = "badge b-low"; badge.textContent = "数据异常·已冻结上期";
  } else {
    badge.className = "badge b-sig-y"; badge.textContent = `前瞻运行中 · ${sess.phase || "EOD"}`;
  }
  document.getElementById("ppSessionLine").textContent =
    `行情日 ${sess.market_date || DATA.date} · 阶段 ${sess.phase || "EOD"} · 更新时间 ${sess.updated_at || DATA.updated_at} · 初始本金 ¥100,000（严格虚拟资金）`;

  document.getElementById("ppCohortBody").innerHTML = cohortList.map(a => {
    const ret = (a.total_nav / a.initial_cash - 1) * 100;
    const util = ((a.total_nav - a.cash) / Math.max(1, a.total_nav)) * 100;
    const pSig = a.pending_signal;
    const sigStr = pSig ? `${pSig.signal_date} (${pSig.status === "PENDING" ? "待下一交易日开盘" : "已于"+pSig.exec_date+"成交"})` : "无";
    const active = a.account_id === ppAccId;
    return `<tr class="${active ? "hl-row" : ""}" style="cursor:pointer;" onclick="ppAccId='${a.account_id}';renderPaper();">
      <td><strong>${a.name}</strong><span class="sub">${a.desc}</span></td>
      <td><span class="badge ${a.execution_mode==="theoretical_fractional"?"b-warn":"b-sig-y"}">${a.execution_mode==="theoretical_fractional"?"理论全池零整":"10张整数手"}</span></td>
      <td class="num mono"><strong>¥${a.total_nav.toLocaleString("zh-CN",{minimumFractionDigits:2,maximumFractionDigits:2})}</strong></td>
      <td class="num mono ${cls(ret)}">${signed(ret)}</td>
      <td class="num mono">¥${fmt(a.cash,0)} <span class="sub">仓位 ${fmt(util,1)}%</span></td>
      <td class="num mono neg">${fmt(a.max_dd_pct)}%</td>
      <td class="num mono">¥${fmt(a.total_fees,2)}</td>
      <td class="mono" style="font-size:11px;">${sigStr}</td>
    </tr>`;
  }).join("") || `<tr><td colspan="8" style="text-align:center;color:var(--muted);">暂无该对照组模拟账户</td></tr>`;

  document.getElementById("ppAccPills").innerHTML = cohortList.map(a =>
    `<button class="pill ${a.account_id===ppAccId?"active":""}" onclick="ppAccId='${a.account_id}';renderPaper();">${a.name}</button>`
  ).join("");

  const curA = accs[ppAccId];
  if (curA) {
    const ret = (curA.total_nav / curA.initial_cash - 1) * 100;
    const util = ((curA.total_nav - curA.cash) / Math.max(1, curA.total_nav)) * 100;
    document.getElementById("ppKpiNav").textContent = `¥${curA.total_nav.toLocaleString("zh-CN",{minimumFractionDigits:2,maximumFractionDigits:2})}`;
    document.getElementById("ppKpiRet").textContent = `累计收益 ${signed(ret)} · 建账日 ${curA.genesis_date}`;
    document.getElementById("ppKpiCash").textContent = `¥${curA.cash.toLocaleString("zh-CN",{minimumFractionDigits:2,maximumFractionDigits:2})}`;
    document.getElementById("ppKpiUtil").textContent = `仓位利用率 ${fmt(util,1)}% · 持仓 ${Object.keys(curA.holdings||{}).length} 只`;
    document.getElementById("ppKpiMdd").textContent = `${fmt(curA.max_dd_pct)}%`;
    document.getElementById("ppKpiFees").textContent = `累计手续费+滑点 ¥${fmt(curA.total_fees,2)} · 成交额 ¥${fmt(curA.total_traded_rmb,0)}`;
    const ps = curA.pending_signal;
    document.getElementById("ppKpiSig").textContent = ps ? `${ps.status === "PENDING" ? "待执行 ("+ps.n_targets+"只)" : "已执行"}` : "持仓等待";
    document.getElementById("ppKpiDates").textContent = `最近行情 ${curA.last_market_date || "—"} · 信号日 ${curA.last_signal_date || "—"}`;

    const dNavs = curA.daily_navs || [];
    if (dNavs.length >= 2) {
      document.getElementById("ppNavSvg").innerHTML = drawSimpleLines(dNavs.map(x => x.date), [
        { vals: dNavs.map(x => x.nav / curA.initial_cash), col: "#0969da", w: 2.2 },
      ], 900, 170, 68);
    } else {
      document.getElementById("ppNavSvg").innerHTML = `<div class="note-box">当前为前瞻模拟账本起始日（${curA.genesis_date} 收盘后已生成初始 T 日调仓信号，将在下一交易日开盘按实际 10 张整数手成交并开始累计日频净值曲线，绝不倒填历史）。</div>`;
    }

    const hList = Object.values(curA.holdings || {}).sort((a, b) => (b.market_val || 0) - (a.market_val || 0));
    document.getElementById("ppHoldTitle").textContent = `${curA.name} · 当前实盘模拟持仓`;
    document.getElementById("ppHoldBody").innerHTML = hList.map(h => {
      const pnl = h.avg_cost > 0 ? ((h.last_price - h.avg_cost) / h.avg_cost) * 100 : 0;
      return `<tr><td><strong>${h.name}</strong><span class="sub mono">${h.ticker} (${h.entry_date})</span></td><td>${ratingBadge(h.rating)}</td><td class="num mono">${fmt(h.qty,0)}</td><td class="num mono">¥${fmt(h.avg_cost)}</td><td class="num mono">¥${fmt(h.last_price)}</td><td class="num mono">${fmt(h.weight_pct,1)}%</td><td class="num mono ${cls(pnl)}">${signed(pnl)}</td></tr>`;
    }).join("") || `<tr><td colspan="7" style="text-align:center;padding:20px;color:var(--muted);">当前满现金等待下一交易日开盘执行初始建仓信号</td></tr>`;

    const lHead = document.getElementById("ppLogHead");
    const lBody = document.getElementById("ppLogBody");
    if (ppLogTab === "pending") {
      lHead.innerHTML = `<tr><th>拟买入目标券</th><th>评级</th><th class="num">信号收盘价</th><th class="num">目标仓位</th><th class="num">拟配(约张)</th></tr>`;
      const targets = (ps && ps.top_targets) || [];
      const expBudget = curA.total_nav * ((ps && ps.target_exposure) || 1.0);
      lBody.innerHTML = targets.map(t => {
        const estLots = Math.floor((expBudget * (t.weight_pct / 100.0)) / Math.max(10.0, t.close * 10.0));
        return `<tr><td><strong>${t.name}</strong> <span class="sub mono">${t.ticker} · 信号 ${ps.signal_date} (${ps.status})</span></td><td>${ratingBadge(t.rating)}</td><td class="num mono">¥${fmt(t.close)}</td><td class="num mono">${fmt(t.weight_pct,2)}%</td><td class="num mono">${estLots * 10} 张</td></tr>`;
      }).join("") || `<tr><td colspan="5" style="text-align:center;color:var(--muted);">当前无待执行信号</td></tr>`;
    } else if (ppLogTab === "trades") {
      lHead.innerHTML = `<tr><th>成交日 (信号日)</th><th>标的</th><th>方向</th><th class="num">数量</th><th class="num">开盘价</th><th class="num">手续费</th></tr>`;
      lBody.innerHTML = (curA.trades || []).slice(0, 60).map(t =>
        `<tr><td class="mono">${t.date} <span class="sub">sig:${t.signal_date}</span></td><td><strong>${t.name}</strong> <span class="sub mono">${t.ticker}</span></td><td><span class="badge ${t.side.includes("BUY")?"b-aaa":"b-warn"}">${t.side}</span></td><td class="num mono">${t.qty}张</td><td class="num mono">¥${fmt(t.price,3)}</td><td class="num mono">¥${fmt(t.fee)}</td></tr>`
      ).join("") || `<tr><td colspan="6" style="text-align:center;color:var(--muted);">待下一交易日开盘执行首次信号后生成真实成交流水</td></tr>`;
    } else {
      lHead.innerHTML = `<tr><th>日期</th><th>类型</th><th class="num">现金变动</th><th class="num">账户现金余额</th><th>摘要</th></tr>`;
      lBody.innerHTML = (curA.cash_log || []).slice(0, 60).map(c =>
        `<tr><td class="mono">${c.date}</td><td class="mono">${c.type}</td><td class="num mono ${cls(c.delta)}">${signed(c.delta,2,"")}</td><td class="num mono">¥${fmt(c.cash_after)}</td><td style="font-size:12px;">${c.note}</td></tr>`
      ).join("") || `<tr><td colspan="5" style="text-align:center;color:var(--muted);">初始现金 ¥100,000.00，暂无出入金变动</td></tr>`;
    }
  }

  const rules = PAPER.execution_rules || {};
  document.getElementById("ppRulesNote").innerHTML =
    `<strong>模拟账户规则与版本声明：</strong>初始资金 ¥100,000；交易单位 ${rules.min_trade_unit || "10张"}；手续费 ${rules.commission || "0.05% (最低¥5)"} + 滑点 ${rules.slippage || "0.05%"}；现金享 ${rules.cash_yield_ann || "2.0%"} 年化；${rules.execution_timing || "T日生成信号，T+1开盘成交"}。策略版本迭代时新设账户编号，绝不覆盖修改既有前瞻账本。`;

  // Legacy Top-20% ledger continuity
  const ll = DATA.live_ledger || {};
  const port = (ll.portfolio_by_pool || {})[curPool] || ll.portfolio || { holdings: [], nav_net: 1.0 };
  const cmps = (ll.comparisons_by_pool || {})[curPool] || [];
  document.getElementById("portTableBody").innerHTML = (port.holdings || []).slice(0, 40).map((h, i) => {
    const ret = h.entry_price > 0 ? ((h.last_price - h.entry_price) / h.entry_price) * 100 : 0;
    return `<tr><td class="mono" style="color:var(--muted);">${i + 1}</td><td><strong>${h.name}</strong> <span class="sub mono">${h.ticker}</span></td><td>${ratingBadge(h.rating)}</td><td class="num mono">${((h.weight||0)*100).toFixed(2)}%</td><td class="mono">${h.entry_date}</td><td class="num mono">¥${fmt(h.entry_price)}</td><td class="num mono">¥${fmt(h.last_price)}</td><td class="num mono ${cls(ret)}">${signed(ret)}</td></tr>`;
  }).join("") || `<tr><td colspan="8" style="text-align:center;color:var(--muted);">暂无持仓</td></tr>`;
  document.getElementById("ledgerHistBox").innerHTML = cmps.length ? `<div class="table-wrap" style="border-top:1px solid var(--border-soft);"><table><thead><tr><th>区间</th><th class="num">样本数</th><th class="num">Top20%</th><th class="num">Bot20%</th><th class="num">双低</th><th class="num">均值</th><th class="num">超额</th><th class="num">IC</th></tr></thead><tbody>${cmps.map(c=>`<tr><td class="mono">${c.from_date}→${c.to_date}</td><td class="num mono">${c.count}</td><td class="num mono ${cls(c.top_ret)}">${signed(c.top_ret)}</td><td class="num mono">${signed(c.bot_ret)}</td><td class="num mono">${signed(c.dl_ret)}</td><td class="num mono">${signed(c.mkt_ret)}</td><td class="num mono ${cls(c.spread)}">${signed(c.spread)}</td><td class="num mono">${fmt(c.rank_ic,3)}</td></tr>`).join("")}</tbody></table></div>` : `<div class="note-box">原 5 档评级池连续快照净值 NAV = ${fmt(port.nav_net, 4)}（${port.frozen_since || DATA.date} 起）。</div>`;
}

function renderResearch() {
  // 5A. Professional Peer Benchmarks & Incremental FMB
  const peers = ((PRO.peer_benchmarks || {})[curPeerScope]) || [];
  document.getElementById("resPeerBody").innerHTML = peers.map(p => {
    const c = (p.costs && p.costs["0.10%"]) || {};
    const ic20 = (p.ic && p.ic["20"]) || {};
    const s23 = (p.subperiods && p.subperiods.p23_24) || {};
    const s25 = (p.subperiods && p.subperiods.p25_26) || {};
    return `<tr class="${p.id==="f_four"?"hl-row":""}">
      <td><strong>${p.name}</strong><span class="sub">${p.approx_note}</span></td>
      <td class="num mono">${p.id==="equal_weight" ? "—" : `${ic20.mean_ic>0?"+":""}${fmt(ic20.mean_ic,4)} (${fmt(ic20.t_nw,1)})`}</td>
      <td class="num mono ${cls(c.ann_ret)}"><strong>${signed(c.ann_ret)}</strong></td>
      <td class="num mono neg">${fmt(c.mdd)}%</td>
      <td class="num mono">${fmt(c.sharpe,2)}</td>
      <td class="num mono">${fmt(c.calmar,2)}</td>
      <td class="num mono">${fmt(p.ann_to,1)}%</td>
      <td class="num mono ${cls(s23.ann_ret)}">${signed(s23.ann_ret)}</td>
      <td class="num mono ${cls(s25.ann_ret)}">${signed(s25.ann_ret)}</td>
    </tr>`;
  }).join("");

  document.getElementById("resFormulaBody").innerHTML = peers.filter(p => p.id !== "equal_weight").map(p =>
    `<tr><td><strong>${p.name}</strong></td><td class="mono" style="font-size:11px;">${p.formula}</td><td style="font-size:11px;color:var(--muted);">${p.data_source} · ${p.params}</td><td style="font-size:11px;color:var(--muted);">${p.applicability} · ${p.approx_note}</td></tr>`
  ).join("");

  const fmbNo = (PRO.incremental_fmb_aam || {}).without_iv || [];
  const fmbIv = (PRO.incremental_fmb_aam || {}).with_iv_subset || [];
  const ivMap = {}; fmbIv.forEach(x => { ivMap[x.key] = x; });
  const mergedKeys = fmbNo.map(x => x.key);
  ["impl_vol", "hv_minus_iv"].forEach(k => { if (!mergedKeys.includes(k)) mergedKeys.push(k); });
  const noMap = {}; fmbNo.forEach(x => { noMap[x.key] = x; });
  document.getElementById("resIncFmbBody").innerHTML = mergedKeys.map(k => {
    const a = noMap[k], b = ivMap[k];
    const lbl = (a || b || {}).label || k;
    return `<tr class="${k==="f_four"?"hl-row":""}">
      <td><strong>${lbl}</strong></td>
      <td class="num mono">${a ? `${signed(a.mean,3,"")} (${fmt(a.t_stat,2)})` : "未纳入(82.7%有效)"}</td>
      <td class="num mono">${b ? `${signed(b.mean,3,"")} (${fmt(b.t_stat,2)})` : "—"}</td>
    </tr>`;
  }).join("");

  const corr = PRO.factor_corr_aam || {};
  if (corr.labels && corr.matrix) {
    document.getElementById("resCorrBox").innerHTML = `<table><thead><tr><th>因子</th>${corr.labels.map(l=>`<th class="num" style="font-size:11px;">${l}</th>`).join("")}</tr></thead><tbody>` +
      corr.matrix.map((row, i) => `<tr><td style="font-size:11px;font-weight:600;">${corr.labels[i]}</td>${row.map((v, j) => `<td class="num mono" style="font-size:11px;${i===j?"color:var(--light);":Math.abs(v)>=0.5?"font-weight:700;color:var(--blue);":""}">${v!==null?v.toFixed(2):"—"}</td>`).join("")}</tr>`).join("") + `</tbody></table>`;
  }

  // 5B. Valuation Regimes & Position Sizing
  const regPoolKey = (curPool === "all" || curPool === "aa_minus" || curPool === "aa") ? curPool : "aa_minus";
  const regList = (((TIMING.regimes || {})[regPoolKey] || {})[curRegMode]) || [];
  document.getElementById("resRegimeBody").innerHTML = regList.map(b => {
    const h = (b.horizons || {})[curRegH] || {};
    if (h.insufficient) return `<tr><td><strong>${b.label}</strong></td><td class="num mono">${h.n_days} (有效~${h.n_eff})</td><td colspan="6" style="text-align:center;color:var(--muted);">样本不足</td></tr>`;
    return `<tr>
      <td><strong>${b.label}</strong>${h.low_sample_warn?` <span class="badge b-warn">低N_eff</span>`:""}</td>
      <td class="num mono">${h.n_days} (${h.n_eff})</td>
      <td class="num mono ${cls(h.mean)}"><strong>${signed(h.mean)}</strong> / ${signed(h.median)}</td>
      <td class="num mono">[${signed(h.ci_low)}, ${signed(h.ci_high)}]</td>
      <td class="num mono">${fmt(h.win_pct,1)}%</td>
      <td class="num mono neg">${fmt(h.avg_mdd)}%</td>
      <td class="num mono ${cls(h.lsm_excess)}">${signed(h.lsm_excess)}</td>
      <td class="num mono ${cls(h.dl_excess)}">${signed(h.dl_excess)}</td>
    </tr>`;
  }).join("");

  const szPool = ((TIMING.position_sizing || {})[regPoolKey]) || {};
  const szRows = curPool === "all" ? (szPool.on_lsm_strategy || []) : (szPool.on_balanced_strategy || []);
  document.getElementById("resSizingBody").innerHTML = szRows.map(s =>
    `<tr class="${s.key==="val_tier"?"hl-row":""}"><td><strong>${s.name}</strong><span class="sub">${s.desc}</span></td><td class="num mono">${fmt(s.avg_exposure_pct,1)}%</td><td class="num mono">${s.pos_switches}</td><td class="num mono ${cls(s.ann_ret)}">${signed(s.ann_ret)}</td><td class="num mono neg">${fmt(s.mdd)}%</td><td class="num mono"><strong>${fmt(s.sharpe,2)}</strong></td><td class="num mono">${fmt(s.calmar,2)}</td><td class="num mono ${cls(s.bull_miss_ann_pct)}">${signed(s.bull_miss_ann_pct)}</td></tr>`
  ).join("");

  // 5C. Time-Synced Factor IC & Quintiles & FMB
  const r = R();
  const qMap = r.quintiles || {};
  const [, , trLbl] = getDateBounds();
  document.getElementById("resFactorTitle").textContent = `冻结 LSM 因子预测检验（${POOLS[curPool].label} · ${trLbl}）`;
  document.getElementById("resIcBody").innerHTML = ["5", "10", "20"].map(h => {
    const x = computePeriodIc(curPool, h), q = qMap[h];
    return `<tr><td class="mono"><strong>${h}D</strong></td><td class="num mono">${x.days || "—"}</td><td class="num mono">${fmt(x.avg_n,1)}</td><td class="num mono ${cls(x.mean_ic)}">${x.mean_ic>0?"+":""}${fmt(x.mean_ic,4)}</td><td class="num mono">${fmt(x.icir,3)}</td><td class="num mono">${fmt(x.pos_pct,1)}%</td><td class="num mono"><strong>${fmt(x.t_nw,2)}</strong> <span class="badge ${x.p_val<0.05?"b-sig-y":"b-sig-n"}">p=${fmtP(x.p_val)}</span></td><td class="num mono ${cls(q?.g5_minus_g1)}">${signed(q?.g5_minus_g1,3)} (t=${fmt(q?.t_nw,2)})</td></tr>`;
  }).join("");

  const pts = (r.rolling_ic || {})[curResIcH] || [];
  if (pts.length) {
    const W = 500, H = 190, pL = 40, pR = 14, pT = 14, pB = 24;
    const ys = pts.map(p => p[1]);
    const minY = Math.min(-0.08, ...ys) - 0.02, maxY = Math.max(0.18, ...ys) + 0.02;
    const px = i => pL + (i / Math.max(1, pts.length - 1)) * (W - pL - pR);
    const py = v => H - pB - ((v - minY) / (maxY - minY)) * (H - pT - pB);
    let g = "";
    [-0.05, 0, 0.05, 0.10].forEach(lv => { if (lv >= minY && lv <= maxY) { const y = py(lv).toFixed(1); g += `<line x1="${pL}" y1="${y}" x2="${W-pR}" y2="${y}" stroke="${lv===0?"#8c959f":"#eaeef2"}" stroke-dasharray="${lv===0?"":"3 3"}"/><text x="${pL-5}" y="${Number(y)+3}" text-anchor="end" font-size="10" fill="#656d76">${lv.toFixed(2)}</text>`; } });
    [0, Math.floor(pts.length / 2), pts.length - 1].forEach(i => { g += `<text x="${px(i).toFixed(1)}" y="${H-6}" text-anchor="middle" font-size="10" fill="#656d76">${pts[i][0]}</text>`; });
    const poly = pts.map((p, i) => `${px(i).toFixed(1)},${py(p[1]).toFixed(1)}`).join(" ");
    document.getElementById("resRollSvg").innerHTML = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block;">${g}<polyline fill="none" stroke="#0969da" stroke-width="1.8" points="${poly}"/></svg>`;
  }

  const q20 = qMap["20"];
  if (q20) {
    const W = 500, H = 190, pL = 44, pR = 14, pT = 20, pB = 26;
    const vals = q20.groups.map(g => g.excess_ret);
    const minY = Math.min(-0.1, ...vals) * 1.24, maxY = Math.max(0.1, ...vals) * 1.25;
    const zeroY = pT + (maxY / (maxY - minY)) * (H - pT - pB);
    const py = v => pT + ((maxY - v) / (maxY - minY)) * (H - pT - pB);
    const bw = (W - pL - pR) / 5;
    let bars = `<line x1="${pL}" y1="${zeroY.toFixed(1)}" x2="${W-pR}" y2="${zeroY.toFixed(1)}" stroke="#8c959f"/>`;
    q20.groups.forEach((g, i) => {
      const v = g.excess_ret, x = pL + i * bw + bw * 0.2, w = bw * 0.6, vy = py(v), top = Math.min(vy, zeroY), bh = Math.max(2, Math.abs(vy - zeroY));
      const col = v >= 0 ? "#cf222e" : "#1a7f37", lblY = v >= 0 ? top - 5 : top + bh + 11;
      bars += `<rect x="${x.toFixed(1)}" y="${top.toFixed(1)}" width="${w.toFixed(1)}" height="${bh.toFixed(1)}" fill="${col}" opacity="0.85" rx="2"/><text x="${(x+w/2).toFixed(1)}" y="${lblY.toFixed(1)}" text-anchor="middle" font-size="10" font-weight="600" fill="${col}">${signed(v,3)}</text><text x="${(x+w/2).toFixed(1)}" y="${H-6}" text-anchor="middle" font-size="11" fill="#1f2328">${g.label}</text>`;
    });
    document.getElementById("resQBarSvg").innerHTML = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block;">${bars}</svg>`;
  }

  const fmb = (r.fama_macbeth || {})["20"];
  document.getElementById("resFmbBody").innerHTML = fmb ? fmb.factors.map(f => `
    <tr class="${f.key==="f_four"?"hl-row":""}"><td><strong>${f.label}</strong></td><td class="num mono ${cls(f.mean)}">${signed(f.mean,4,"")}</td><td class="num mono">${fmt(f.t_stat,2)}</td><td class="num mono">${fmtP(f.p_val)}</td><td><span class="badge ${f.p_val<0.05?"b-sig-y":"b-sig-n"}">${f.p_val<0.05?"显著 p<0.05":"不显著"}</span></td></tr>
  `).join("") : "";

  // 5D. Stepwise & Walk-Forward OOS
  const sw = EXP.stepwise || {};
  const stHead = document.getElementById("resStepHead");
  const stBody = document.getElementById("resStepBody");
  if (curStepTab === "risk") {
    stHead.innerHTML = `<tr><th>信用评级池 (100/0·5D·无缓冲)</th><th class="num">不剔除 年化 (Sharpe / Calmar)</th><th class="num">剔除风险 年化 (Sharpe / Calmar)</th><th class="num">夏普增量</th></tr>`;
    stBody.innerHTML = (sw.risk_filter_by_pool || []).map(x => {
      const dSr = x.filtered.sharpe - x.raw.sharpe;
      return `<tr><td><strong>${x.pool}</strong></td><td class="num mono">${signed(x.raw.ann_ret)} (${fmt(x.raw.sharpe)} / ${fmt(x.raw.calmar)})</td><td class="num mono"><strong>${signed(x.filtered.ann_ret)}</strong> (${fmt(x.filtered.sharpe)} / ${fmt(x.filtered.calmar)})</td><td class="num mono ${cls(dSr)}">${signed(dSr, 2, "")}</td></tr>`;
    }).join("");
  } else {
    const arr = curStepTab === "credit" ? sw.credit_filter : curStepTab === "weight" ? sw.blend_weights : curStepTab === "freq" ? sw.rebal_freq : sw.buffers;
    stHead.innerHTML = `<tr><th>独立因子测试项 (0.10% 成本)</th><th class="num">7年累计</th><th class="num">年化净收益</th><th class="num">最大回撤</th><th class="num">Sharpe</th><th class="num">Calmar</th><th class="num">年化换手</th></tr>`;
    stBody.innerHTML = (arr || []).map(x => `<tr><td><strong>${x.step}</strong></td><td class="num mono ${cls(x.total_ret)}">${signed(x.total_ret)}</td><td class="num mono ${cls(x.ann_ret)}">${signed(x.ann_ret)}</td><td class="num mono neg">${fmt(x.mdd)}%</td><td class="num mono"><strong>${fmt(x.sharpe, 2)}</strong></td><td class="num mono">${fmt(x.calmar, 2)}</td><td class="num mono">${fmt(x.ann_to, 1)}%</td></tr>`).join("");
  }

  const wf = (EXP.walk_forward || {})[curOosMode] || {};
  const priors = (EXP.walk_forward || {}).priors_2021_2026 || [];
  const pbo = EXP.overfitting || {};
  document.getElementById("resOosChartTitle").textContent = `${curOosMode === "expanding" ? "扩展窗口" : "滚动 2 年"} Walk-Forward 样本外净值 (${wf.period || ""})`;
  document.getElementById("resOosSvg").innerHTML = drawSimpleLines(wf.nav_dates, [
    { vals: wf.nav_010, col: "#0969da", w: 2.2 },
    { vals: (priors[2] || {}).nav_010, col: "#1a7f37", w: 1.8 },
    { vals: (priors[0] || {}).nav_010, col: "#8c959f", w: 1.5, dash: "5 4" },
  ], 520, 200, 74);
  document.getElementById("resOosYrBody").innerHTML = (wf.yearly_picks || []).map(y =>
    `<tr><td class="mono"><strong>${y.test_year}</strong></td><td class="mono" style="font-size:11px;color:var(--muted);">${y.train_window}</td><td style="font-size:11px;">${y.picked_desc}</td><td class="num mono ${cls(y.oos_ret)}">${signed(y.oos_ret)}</td><td class="num mono">${fmt(y.oos_sharpe, 2)}</td></tr>`
  ).join("");
  document.getElementById("resDsrBody").innerHTML = [
    { label: `Walk-Forward 扩展窗口动态样本外 (${wf.period || ""})`, oos_ann_ret: wf.ann_ret, oos_mdd: wf.mdd, oos_sharpe: wf.sharpe, oos_calmar: wf.calmar, full_sharpe: null, dsr_prob: null },
    ...priors,
  ].map(p =>
    `<tr><td><strong>${p.label}</strong></td><td class="num mono ${cls(p.oos_ann_ret)}">${signed(p.oos_ann_ret)}</td><td class="num mono neg">${fmt(p.oos_mdd)}%</td><td class="num mono">${fmt(p.oos_sharpe, 2)}</td><td class="num mono">${fmt(p.oos_calmar, 2)}</td><td class="num mono">${fmt(p.full_sharpe, 2)}</td><td class="num mono">${p.dsr_prob !== null && p.dsr_prob !== undefined ? fmt(p.dsr_prob, 4) : "样本外直连"}</td></tr>`
  ).join("");
  document.getElementById("resPboNote").innerHTML =
    `<strong>多重检验防过拟合校验 (N=${pbo.n_configs || 450} 组实验)：</strong>CSCV 8 区块对称交叉验证 (${pbo.n_splits || 70} 次切分) 测得 <code>PBO = ${fmt(pbo.pbo_pct, 1)}%</code>，样本内最优夏普中位数 <code>${fmt(pbo.median_is_best_sharpe, 2)}</code> 样本外降至 <code>${fmt(pbo.median_oos_sharpe, 2)}</code>（衰减 <code>${fmt(pbo.sharpe_degradation_pct, 1)}%</code>）；预期随机最高夏普基准 <code>SR* = 0.934</code>，样本内峰值 (1.19) DSR 仅 <code>0.7474</code> (&lt; 0.95)，短窗（滚动 2 年）选参样本外夏普更跌至 <code>0.03</code>。`;

  // Collapsible Stage B & Full Grid
  const sbRows = ((PRO.phased_optimization || {}).stage_b_sizing_weighting) || [];
  document.getElementById("resStageBBody").innerHTML = sbRows.map(x => {
    const rk = x.retail_100k;
    return `<tr><td><strong>${x.desc}</strong></td><td class="num mono ${cls(x.ann_ret)}">${signed(x.ann_ret)}</td><td class="num mono neg">${fmt(x.mdd)}%</td><td class="num mono">${fmt(x.sharpe,2)}</td><td class="num mono">${fmt(x.calmar,2)}</td><td class="num mono">${rk ? `${signed(rk.ann_ret)} (${fmt(rk.mdd)}% / ${fmt(rk.avg_invested_pct,1)}%)` : "—"}</td></tr>`;
  }).join("");

  const fp = document.getElementById("gridFilterPool").value;
  const sb = document.getElementById("gridSortBy").value;
  let gRows = (EXP.grid || []).filter(x => fp === "ALL" || x.pool === fp);
  gRows = [...gRows].sort((a, b) => {
    const ca = a.costs["0.10%"], cb = b.costs["0.10%"];
    if (sb === "sharpe_asc") return ca.sharpe - cb.sharpe;
    if (sb === "calmar_desc") return cb.calmar - ca.calmar;
    if (sb === "ann_desc") return cb.ann_ret - ca.ann_ret;
    return cb.sharpe - ca.sharpe;
  });
  document.getElementById("gridCountLbl").textContent = `显示 ${gRows.length} 组`;
  document.getElementById("gridAllBody").innerHTML = gRows.map(x => {
    const c = x.costs["0.10%"];
    return `<tr><td><strong>${x.pool_label}</strong> · ${x.risk_mode==="filtered"?"剔除":"不剔除"} · ${x.weight_label} · ${x.freq}D · ${x.buffer_label}</td><td class="num mono ${cls(c.ann_ret)}">${signed(c.ann_ret)}</td><td class="num mono neg">${fmt(c.mdd)}%</td><td class="num mono ${c.sharpe<0?"neg":""}"><strong>${fmt(c.sharpe,2)}</strong></td><td class="num mono">${fmt(c.calmar,2)}</td><td class="num mono">${fmt(x.ann_to,1)}%</td></tr>`;
  }).join("");

  if (curPool === "all") {
    document.getElementById("resAuditHead").innerHTML = `<tr><th>审计指标</th><th>修正前</th><th>独立审计修正后</th><th>说明</th></tr>`;
    document.getElementById("resAuditBody").innerHTML = (r.audit_comparison || []).map(x => `<tr><td><strong>${x.metric}</strong></td><td class="mono" style="color:var(--muted);">${x.before}</td><td class="mono"><strong>${x.after}</strong></td><td style="font-size:12px;color:var(--muted);">${x.reason}</td></tr>`).join("");
  } else {
    const rf = r.rating_filter || {};
    const cov = rf.coverage_by_year || {};
    document.getElementById("resAuditHead").innerHTML = `<tr><th>点时校验项</th><th>口径</th><th>结果</th><th>说明</th></tr>`;
    document.getElementById("resAuditBody").innerHTML = [
      ["信用范围", (rf.eligible_grades || []).join(" / "), `${r.universe?.total_bonds || 0} 只 (${Number(r.universe?.total_bond_days || 0).toLocaleString()} 日)`, "严格按 T 日已公告债项评级"],
      ["点时覆盖率", "总体 & 分年 ≥ 90%", `${((rf.rating_coverage_all_bond_days || 0) * 100).toFixed(2)}%`, Object.keys(cov).sort().map(y => `${y}:${(cov[y]*100).toFixed(1)}%`).join(" · ")],
      ["防泄漏机制", "allow_exact_matches=False", "公告次日生效", "排除盘中/盘后公告泄漏"],
    ].map(rw => `<tr><td><strong>${rw[0]}</strong></td><td class="mono">${rw[1]}</td><td class="mono"><strong>${rw[2]}</strong></td><td style="font-size:12px;color:var(--muted);">${rw[3]}</td></tr>`).join("");
  }
}

function renderAll() {
  syncTimeBars();
  renderStatus();
  renderOverview();
  renderScreener();
  renderBacktest();
  renderPaper();
  renderResearch();
}

function switchTab(tab) {
  curTab = tab;
  document.querySelectorAll("#mainTabs .tab-btn").forEach(b => b.classList.toggle("active", b.dataset.tab === tab));
  document.querySelectorAll(".view-pane").forEach(p => p.classList.toggle("active", p.id === `view-${tab}`));
}

document.querySelectorAll("#mainTabs .tab-btn").forEach(b => b.addEventListener("click", () => switchTab(b.dataset.tab)));
document.querySelectorAll("[data-jump]").forEach(b => b.addEventListener("click", () => { switchTab(b.dataset.jump); window.scrollTo({ top: 0, behavior: "smooth" }); }));
document.querySelectorAll("#poolSeg .seg-btn").forEach(b => b.addEventListener("click", () => { curPool = b.dataset.pool; document.querySelectorAll("#poolSeg .seg-btn").forEach(x => x.classList.toggle("active", x === b)); renderAll(); }));
document.querySelectorAll("#riskSeg .seg-btn").forEach(b => b.addEventListener("click", () => { curRisk = b.dataset.risk; document.querySelectorAll("#riskSeg .seg-btn").forEach(x => x.classList.toggle("active", x === b)); renderAll(); }));
document.querySelectorAll("#ovPresetPills .pill").forEach(b => b.addEventListener("click", () => { curOvPreset = b.dataset.ovp; document.querySelectorAll("#ovPresetPills .pill").forEach(x => x.classList.toggle("active", x === b)); renderOverview(); }));
document.querySelectorAll("[data-tbar] .pill").forEach(b => b.addEventListener("click", () => { curTimeRange = b.dataset.tr; renderAll(); }));
[["trStartInp", "trEndInp"], ["btStartInp", "btEndInp"]].forEach(([sId, eId]) => {
  document.getElementById(sId).addEventListener("change", e => { customStart = e.target.value; curTimeRange = "custom"; renderAll(); });
  document.getElementById(eId).addEventListener("change", e => { customEnd = e.target.value; curTimeRange = "custom"; renderAll(); });
});
document.getElementById("btOverlaySel").addEventListener("change", e => { btOverlay = e.target.value; renderBacktest(); });
document.querySelectorAll("#btWeightPills .pill").forEach(b => b.addEventListener("click", () => { btWeight = b.dataset.w; document.querySelectorAll("#btWeightPills .pill").forEach(x => x.classList.toggle("active", x === b)); renderBacktest(); }));
document.querySelectorAll("#btFreqPills .pill").forEach(b => b.addEventListener("click", () => { btFreq = Number(b.dataset.f); document.querySelectorAll("#btFreqPills .pill").forEach(x => x.classList.toggle("active", x === b)); renderBacktest(); }));
document.querySelectorAll("#btBufPills .pill").forEach(b => b.addEventListener("click", () => { btBuf = b.dataset.b; document.querySelectorAll("#btBufPills .pill").forEach(x => x.classList.toggle("active", x === b)); renderBacktest(); }));
document.querySelectorAll("#btCostPills .pill").forEach(b => b.addEventListener("click", () => { btCost = b.dataset.c; document.querySelectorAll("#btCostPills .pill").forEach(x => x.classList.toggle("active", x === b)); renderBacktest(); }));
document.getElementById("ppCohortSel").addEventListener("change", e => { ppCohort = e.target.value; renderPaper(); });
document.querySelectorAll("#ppLogPills .pill").forEach(b => b.addEventListener("click", () => { ppLogTab = b.dataset.ppl; document.querySelectorAll("#ppLogPills .pill").forEach(x => x.classList.toggle("active", x === b)); renderPaper(); }));
document.querySelectorAll("#resPeerScopePills .pill").forEach(b => b.addEventListener("click", () => { curPeerScope = b.dataset.ps; document.querySelectorAll("#resPeerScopePills .pill").forEach(x => x.classList.toggle("active", x === b)); renderResearch(); }));
document.querySelectorAll("#resRegModePills .pill").forEach(b => b.addEventListener("click", () => { curRegMode = b.dataset.rm; document.querySelectorAll("#resRegModePills .pill").forEach(x => x.classList.toggle("active", x === b)); renderResearch(); }));
document.querySelectorAll("#resRegHPills .pill").forEach(b => b.addEventListener("click", () => { curRegH = b.dataset.rh; document.querySelectorAll("#resRegHPills .pill").forEach(x => x.classList.toggle("active", x === b)); renderResearch(); }));
document.querySelectorAll("#resStepPills .pill").forEach(b => b.addEventListener("click", () => { curStepTab = b.dataset.st; document.querySelectorAll("#resStepPills .pill").forEach(x => x.classList.toggle("active", x === b)); renderResearch(); }));
document.querySelectorAll("#resOosPills .pill").forEach(b => b.addEventListener("click", () => { curOosMode = b.dataset.oos; document.querySelectorAll("#resOosPills .pill").forEach(x => x.classList.toggle("active", x === b)); renderResearch(); }));
document.querySelectorAll("#resIcPills .pill").forEach(b => b.addEventListener("click", () => { curResIcH = b.dataset.rich; document.querySelectorAll("#resIcPills .pill").forEach(x => x.classList.toggle("active", x === b)); renderResearch(); }));
["scrSearch", "scrReg", "scrSort", "scrSafeOnly"].forEach(id => document.getElementById(id).addEventListener(id === "scrSearch" ? "input" : "change", renderScreener));
["gridFilterPool", "gridSortBy"].forEach(id => document.getElementById(id).addEventListener("change", renderResearch));

renderAll();
// Load full 450-grid, exact daily NAVs & daily Rank IC series asynchronously
Promise.all([
  fetch("research_experiments.json").then(r => r.ok ? r.json() : null).catch(() => null),
  fetch("daily_navs.json").then(r => r.ok ? r.json() : null).catch(() => null),
  fetch("daily_ic.json").then(r => r.ok ? r.json() : null).catch(() => null),
  fetch("paper_accounts.json").then(r => r.ok ? r.json() : null).catch(() => null),
]).then(([fullExp, dNavs, dIc, pAcc]) => {
  if (fullExp && fullExp.grid) indexExp(fullExp);
  if (dNavs && dNavs.dates) DAILY_NAVS = dNavs;
  if (dIc && dIc.pools) DAILY_IC = dIc;
  if (pAcc && pAcc.accounts) PAPER = pAcc;
  renderAll();
});
</script>
</body>
</html>
"""

    return template.replace("__PAYLOAD_JSON__", payload_json).replace("__UPDATED_AT__", data["updated_at"])


if __name__ == "__main__":
    generate_page_and_data()
