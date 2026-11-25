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

    if not pool:
        return get_default_bonds_pool()
    return list(pool.values())[:max_bonds]


def fetch_stock_vol_map(stock_codes: list, fallback_sigmas: dict) -> dict:
    """Sequentially fetch 60d trailing stock realized volatilities; preserves prior values if throttled."""
    import akshare as ak

    vol_map = dict(fallback_sigmas)
    uniq = sorted({sc for sc in stock_codes if sc and len(sc) == 6})
    fetched = 0
    for sc in uniq:
        pref = "sh" if sc.startswith("6") else "sz"
        try:
            df = ak.stock_zh_a_hist_sina(symbol=pref + sc)
            if df is not None and len(df) >= 20:
                sub = pd.to_numeric(df["close"], errors="coerce").dropna().tail(61)
                log_ret = np.log(sub / sub.shift(1)).dropna()
                log_ret = log_ret[log_ret.abs() <= 0.28]
                if len(log_ret) >= 10:
                    vol = float(log_ret.std() * np.sqrt(252.0))
                    vol_map[sc] = round(max(0.15, min(0.70, vol)), 4)
                    fetched += 1
        except Exception:
            if fetched == 0 and len(vol_map) > 10:
                break
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
    local_tz = pytz.timezone("Asia/Shanghai")
    now = datetime.now(local_tz)
    update_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    date_str = now.strftime("%Y-%m-%d")

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
        # Embed compact summary + key overview presets inline for instant render; full 450-grid loaded via fetch()
        key_ids = set()
        for pk in ("all", "aa_minus", "aa", "aa_plus", "aaa"):
            for rm in ("raw", "filtered"):
                key_ids.add(f"{pk}|{rm}|100/0|5D|none")
                key_ids.add(f"{pk}|{rm}|50/50|5D|buf_20_30")
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

    live_ledger = update_live_history(dist_dir, date_str, lsm_sorted)

    data_payload = sanitize_json({
        "updated_at": update_time_str,
        "date": date_str,
        "author": "Yu Liu (刘宇) · The Chinese University of Hong Kong, Shenzhen",
        "paper_ref": "arXiv:2409.06496v7",
        "paper_url": "https://arxiv.org/abs/2409.06496",
        "frozen_params": FROZEN_PARAMS,
        "pool_size": len(lsm_sorted),
        "research": research_pools["all"],
        "research_pools": research_pools,
        "exp_summary": exp_inline,
        "live_ledger": live_ledger,
        "lsm_top10": lsm_sorted[:10],
        "double_low_top10": dl_sorted[:10],
        "all_bonds": lsm_sorted,
    })

    (dist_dir / "data.json").write_text(json.dumps(data_payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (dist_dir / "index.html").write_text(build_clean_html(data_payload), encoding="utf-8")
    print(f"Generated 5-pool CB Quant Workbench at {dist_dir / 'index.html'} (live_pool={len(lsm_sorted)}, exp_grid={exp_full.get('meta',{}).get('total_configs',0)})")


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
        <button class="tab-btn" data-tab="research"><span class="tab-num mono">04</span>因子研究</button>
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
    <div class="mono" style="font-size:11px;">p_reset=0.80 · B=¥108 · r=2.5% · 实盘更新 __UPDATED_AT__</div>
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
        <span class="kpi-label">20 日 Rank IC</span>
        <div class="kpi-val mono" id="ovIc">—</div>
        <div class="kpi-sub" id="ovIcSub">—</div>
      </div>
    </div>

    <div class="temp-card">
      <div class="temp-metrics" id="tempMetrics"></div>
      <div class="reg-bar-wrap">
        <div class="reg-bar" id="tempRegBar"></div>
        <div class="reg-legend" id="tempRegLegend"></div>
      </div>
    </div>

    <div class="grid-overview">
      <div class="card">
        <div class="card-head">
          <div>
            <h2 class="card-title">主要策略净值对比（2020–2026 历史回测 · 0.10% 成本）</h2>
            <span class="card-sub">随顶部「评级池」与「风险过滤」联动，详细 450 组参数实验见历史回测/因子研究</span>
          </div>
          <div class="pills" id="ovPresetPills">
            <button class="pill active" data-ovp="lsm">原始 LSM</button>
            <button class="pill" data-ovp="blend">50/50 + 20/30缓冲</button>
            <button class="pill" data-ovp="dl">纯双低</button>
          </div>
        </div>
        <div class="card-body">
          <div class="legend">
            <span><i class="ldot" style="background:#0969da;"></i>原始 LSM (100/0·5D·无缓冲)</span>
            <span><i class="ldot" style="background:#1a7f37;"></i>均衡增强 (50/50·5D·20/30缓冲)</span>
            <span><i class="ldot" style="background:#9a6700;"></i>纯双低 (0/100·5D·无缓冲)</span>
            <span><i class="ldot" style="background:#8c959f;"></i>同池等权基准</span>
          </div>
          <div id="ovNavSvg"></div>
          <div class="table-wrap" style="margin-top:10px;border-top:1px solid var(--border-soft);">
            <table>
              <thead><tr><th>主要对照 (5D · 0.10% 成本)</th><th class="num">年化</th><th class="num">最大回撤</th><th class="num">Sharpe</th><th class="num">Calmar</th><th class="num">换手</th></tr></thead>
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
          <button class="btn-link" data-jump="screener">全部转债与模拟持仓 →</button>
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
        <h2 class="card-title">实战增强关键研究结论（禁止直接按七年最高收益挑选参数）</h2>
        <button class="btn-link" data-jump="research">查看完整消融、OOS 与过拟合检验 →</button>
      </div>
      <div class="card-body">
        <ul class="takeaway-list">
          <li><strong>独立归因有效但各司其职：</strong>在全市场内，因果剔除 ST/违约/低流动性（8.34% 样本日）使纯 LSM 年化由 +15.54% 提升至 +17.33%（Sharpe 1.01→1.10）；LSM 与双低 50/50 截面百分位混合将全市场回撤降至 -15.23%（Calmar 1.27）；<strong>20/30 缓冲机制</strong>将换手率砍半（759%→366%）并降低回撤 2.6%。</li>
          <li><strong>信用池权衡：</strong><code>AA-及以上</code>（+13.27% → 剔除后 +14.68%）与 <code>AA及以上</code>（+11.90% → +13.45%）兼顾高收益与控回撤；<code>AA+及以上</code> 绝对收益降至 +7.65% 但仍领先双低（+3.79%）；<strong>仅AAA</strong>（日均 29 只）样本过薄且 2025–2026 失效（Sharpe 0.06）。</li>
          <li><strong>过拟合防线提示：</strong>450 组参数实验的 CSCV 过拟合概率 <code>PBO = 11.4%</code>，样本外夏普平均衰减 <code>38.8%</code>，且短窗（滚动 2 年）追逐冠军参数的样本外夏普跌至 <code>0.03</code>。任何实盘选择必须基于机制稳健性而非 7 年样本内峰值。</li>
        </ul>
      </div>
    </div>
  </section>

  <!-- 02 SCREENER + FORWARD PORTFOLIO -->
  <section class="view-pane" id="view-screener">
    <div class="card">
      <div class="card-head">
        <div>
          <h2 class="card-title" id="scrTitle">实时转债估值选券池</h2>
          <span class="card-sub">随顶部「评级池」与「风险过滤」联动，实时标识正股 ST / 违约低评级 / 低成交警示</span>
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

    <div class="card">
      <div class="card-head">
        <div>
          <h2 class="card-title" id="portTitle">前瞻模拟持仓组合（Top 20% 等权）</h2>
          <span class="card-sub" id="portSub">独立于 2020–2026 历史回测的前瞻冻结账本 · 随评级池与风险过滤联动</span>
        </div>
        <span class="badge b-warn">实盘前瞻组合 ≠ 历史回测</span>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>#</th><th>持仓转债</th><th>评级</th><th class="num">目标权重</th><th>建仓日</th><th class="num">入场价</th><th class="num">最新价</th><th class="num">区间盈亏</th></tr>
          </thead>
          <tbody id="portTableBody"></tbody>
        </table>
      </div>
      <div id="ledgerHistBox"></div>
    </div>
  </section>

  <!-- 03 BACKTEST (450-COMBO PARAMETER WORKBENCH) -->
  <section class="view-pane" id="view-backtest">
    <div class="card">
      <div class="card-head">
        <div>
          <h2 class="card-title">实战增强参数实验台（统一 T+1 开盘成交口径）</h2>
          <span class="card-sub">结合顶部「评级池」与「风险过滤」，即时切换因子权重、调仓频率、缓冲机制与单边交易成本</span>
        </div>
        <span class="mono" style="font-size:12px;color:var(--muted);" id="btComboId">—</span>
      </div>
      <div class="param-strip">
        <div class="param-group">
          <span class="param-lbl">LSM/双低权重</span>
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
          <span class="param-lbl">单边成本</span>
          <div class="pills" id="btCostPills">
            <button class="pill" data-c="0.05%">0.05%</button>
            <button class="pill active" data-c="0.10%">0.10%</button>
            <button class="pill" data-c="0.20%">0.20%</button>
          </div>
        </div>
      </div>
      <div class="card-body">
        <div class="kpi-grid" style="margin-bottom:14px;">
          <div class="kpi-card"><span class="kpi-label">年化净收益 / 7年累计</span><div class="kpi-val mono" id="btAnnRet">—</div><div class="kpi-sub" id="btTotRet">—</div></div>
          <div class="kpi-card"><span class="kpi-label">最大回撤</span><div class="kpi-val mono neg" id="btMdd">—</div><div class="kpi-sub" id="btCostNote">单边成本 0.10%</div></div>
          <div class="kpi-card"><span class="kpi-label">夏普比率 / Calmar</span><div class="kpi-val mono" id="btSharpe">—</div><div class="kpi-sub" id="btCalmar">—</div></div>
          <div class="kpi-card"><span class="kpi-label">年化单边换手率</span><div class="kpi-val mono" id="btTo">—</div><div class="kpi-sub" id="btPoolStats">—</div></div>
        </div>
        <div class="legend">
          <span><i class="ldot" style="background:#0969da;"></i>当前所选参数组合</span>
          <span><i class="ldot" style="background:#8250df;"></i>同池原始 LSM (100/0·5D·无缓冲)</span>
          <span><i class="ldot" style="background:#9a6700;"></i>同池纯双低 (0/100·5D·无缓冲)</span>
          <span><i class="ldot" style="background:#8c959f;"></i>同池等权基准</span>
        </div>
        <div id="btNavSvg"></div>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-head">
          <div>
            <h2 class="card-title">分年度表现对照（2020–2026 YTD）</h2>
            <span class="card-sub">当前参数 vs 原始 LSM vs 纯双低（所选交易成本口径）</span>
          </div>
        </div>
        <div class="table-wrap">
          <table>
            <thead><tr><th>年份</th><th class="num">所选参数收益 (MDD)</th><th class="num">原始 LSM</th><th class="num">纯双低</th></tr></thead>
            <tbody id="btAnnualBody"></tbody>
          </table>
        </div>
      </div>

      <div class="card">
        <div class="card-head">
          <div>
            <h2 class="card-title">五档评级池 × 风险剔除核心对比（所选权重/频率/缓冲）</h2>
            <span class="card-sub">同一参数跨 5 档点时评级池横向对照，检验策略跨池稳定性</span>
          </div>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr><th>评级池 (不剔除 vs 剔除)</th><th class="num">年化</th><th class="num">最大回撤</th><th class="num">Sharpe</th><th class="num">Calmar</th><th class="num">换手</th></tr>
            </thead>
            <tbody id="bt5PoolBody"></tbody>
          </table>
        </div>
      </div>
    </div>
  </section>

  <!-- 04 RESEARCH (FACTORS, STEPWISE ATTRIBUTION, OOS & OVERFITTING CONTROL) -->
  <section class="view-pane" id="view-research">
    <!-- 4A. Stepwise Attribution -->
    <div class="card">
      <div class="card-head">
        <div>
          <h2 class="card-title">独立边际贡献分解（逐项消融检验 · 0.10% 单边成本）</h2>
          <span class="card-sub">先单独检验信用筛选、风险剔除、混合权重、调仓频率与缓冲机制各自贡献，再考察组合效果</span>
        </div>
        <div class="pills" id="resStepPills">
          <button class="pill active" data-st="credit">1. 信用评级池</button>
          <button class="pill" data-st="risk">2. 风险/流动性剔除</button>
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
    </div>

    <!-- 4B. Walk-Forward OOS & Overfitting Controls -->
    <div class="card">
      <div class="card-head">
        <div>
          <h2 class="card-title">滚动样本外检验（2021–2026 OOS）与多重检验过拟合控制</h2>
          <span class="card-sub" id="resPboSub">检验样本外衰减、Deflated Sharpe Ratio (DSR) 与 CSCV 过拟合概率 (PBO)</span>
        </div>
        <div class="pills" id="resOosPills">
          <button class="pill active" data-oos="expanding">扩展窗口 Walk-Forward</button>
          <button class="pill" data-oos="rolling2y">滚动 2 年 Walk-Forward</button>
        </div>
      </div>
      <div class="grid-2" style="padding:14px 16px;margin-bottom:0;">
        <div>
          <div style="font-size:12px;font-weight:600;margin-bottom:6px;" id="resOosChartTitle">Walk-Forward 样本外累计净值 vs 固定先验配置 (2021–2026)</div>
          <div class="legend">
            <span><i class="ldot" style="background:#0969da;"></i>Walk-Forward 动态样本外</span>
            <span><i class="ldot" style="background:#1a7f37;"></i>AA+ 50/50 均衡固定先验</span>
            <span><i class="ldot" style="background:#8c959f;"></i>全市场原始 LSM 基准</span>
          </div>
          <div id="resOosSvg"></div>
        </div>
        <div>
          <div style="font-size:12px;font-weight:600;margin-bottom:6px;">逐年样本外选参结果（严格使用该年之前历史最优 Sharpe）</div>
          <div class="table-wrap">
            <table>
              <thead><tr><th>测试年</th><th>训练窗</th><th>选中配置</th><th class="num">OOS收益</th><th class="num">OOS夏普</th></tr></thead>
              <tbody id="resOosYrBody"></tbody>
            </table>
          </div>
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

    <!-- 4C. Factor IC / Quintiles / Fama-MacBeth (Active Rating Pool) -->
    <div class="card">
      <div class="card-head">
        <div>
          <h2 class="card-title" id="resFactorTitle">冻结 LSM 因子统计预测检验（Rank IC · 五分组 · Fama-MacBeth）</h2>
          <span class="card-sub">随顶部 5 档「评级池」实时切换，原始 LSM 因子在纯因子截面上的预测能力检验</span>
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
            <tr><th>持有期</th><th class="num">有效天数</th><th class="num">日均样本</th><th class="num">Mean Rank IC</th><th class="num">ICIR</th><th class="num">IC&gt;0比例</th><th class="num">Newey-West t</th><th class="num">G5−G1 多空价差</th></tr>
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
          <thead><tr><th>Fama-MacBeth 因果截面回归 (6因子控制)</th><th class="num">均值斜率 (%)</th><th class="num">Newey-West t</th><th class="num">p 值</th><th>统计结论</th></tr></thead>
          <tbody id="resFmbBody"></tbody>
        </table>
      </div>
    </div>

    <!-- 4D. Collapsible Full 450-Config Grid & Audit -->
    <div class="card">
      <details class="fold-card" id="gridDetails">
        <summary>全部 450 组参数实验完整明细（保留所有正负结果，支持排序）</summary>
        <div class="toolbar">
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
            <thead><tr><th>实验组合</th><th class="num">年化净收益</th><th class="num">最大回撤</th><th class="num">Sharpe</th><th class="num">Calmar</th><th class="num">年化换手</th></tr></thead>
            <tbody id="gridAllBody"></tbody>
          </table>
        </div>
      </details>
    </div>

    <div class="card">
      <details class="fold-card">
        <summary>独立代码审计、因果防泄漏校验与文献引用</summary>
        <div class="table-wrap">
          <table>
            <thead id="resAuditHead"></thead>
            <tbody id="resAuditBody"></tbody>
          </table>
        </div>
        <div class="note-box">
          <strong>数据防泄漏与实验说明：</strong>
          (1) 信用评级采用东方财富首次发行债项评级 + 巨潮网 127 只变动转债历史跟踪评级 PDF 公告（<code>allow_exact_matches=False</code> 严格次日生效，99.93% 覆盖率）；
          (2) 风险过滤严格使用当时已知信息：深交所历史简称变更 / 巨潮网风险警示公告建构 ST 区间（次日生效）、<code>&lt;=BBB+</code> 降级及 11 只问题券首次公告违约风险提示、以及滚动 20 日中位成交额 <code>&lt; ¥500 万</code>（6.87% 样本日）；
          (3) 文献参考：<a href="https://arxiv.org/abs/2409.06496" target="_blank" rel="noopener">Liu, Yu (2026), arXiv:2409.06496v7</a>，脚本公开于 <code>tools/</code>。
        </div>
      </details>
    </div>
  </section>

  <footer>
    <span>Yu Liu (刘宇) · CUHK-Shenzhen · Four-Region LSM 冻结参数可转债实证工作台</span>
    <span>全部正负实验结果如实披露，仅供学术研究，不构成投资建议</span>
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
let btWeight = "50/50";
let btFreq = 5;
let btBuf = "buf_20_30";
let btCost = "0.10%";
let curResIcH = "20";
let curStepTab = "credit";
let curOosMode = "expanding";

const R = () => (DATA.research_pools || {})[curPool] || DATA.research || {};
const fmt = (v, n = 2) => typeof v === "number" && Number.isFinite(v) ? v.toFixed(n) : "—";
const signed = (v, n = 2, suf = "%") => typeof v === "number" && Number.isFinite(v) ? (v > 0 ? "+" : "") + v.toFixed(n) + suf : "—";
const cls = v => typeof v === "number" ? (v > 0 ? "pos" : v < 0 ? "neg" : "") : "";
const fmtP = p => (typeof p !== "number" || !Number.isFinite(p)) ? "—" : (p < 0.0001 ? "<0.0001" : p.toFixed(4));

function livePool() {
  const gs = POOLS[curPool].grades;
  return (DATA.all_bonds || []).filter(b => {
    if (gs && !gs.has((b.rating || "").toUpperCase())) return false;
    if (curRisk === "filtered" && b.is_risk_excluded) return false;
    return true;
  });
}

function getExpRow(pool, risk, weight, freq, buf) {
  return EXP_MAP[`${pool}|${risk}|${weight}|${freq}D|${buf}`] || null;
}

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

function drawLinesSvg(dates, seriesList, W = 540, H = 220, rightPad = 82) {
  if (!dates || !dates.length) return `<div style="padding:28px;text-align:center;color:var(--muted);">正在载入净值序列...</div>`;
  const pL = 42, pR = rightPad, pT = 16, pB = 26;
  const validS = seriesList.filter(s => s.vals && s.vals.length);
  if (!validS.length) return `<div style="padding:28px;text-align:center;color:var(--muted);">暂无数据</div>`;
  const allV = validS.flatMap(s => s.vals);
  const mn = Math.min(0.95, ...allV) * 0.96, mx = Math.max(1.25, ...allV) * 1.04;
  const px = i => pL + (i / Math.max(1, dates.length - 1)) * (W - pL - pR);
  const py = v => H - pB - ((v - mn) / Math.max(1e-6, mx - mn)) * (H - pT - pB);
  let grid = "";
  const step = mx - mn > 1.5 ? 0.5 : mx - mn > 0.6 ? 0.25 : 0.1;
  for (let lv = Math.floor(mn / step) * step; lv <= mx + 1e-6; lv += step) {
    const val = Math.round(lv * 100) / 100;
    if (val >= mn && val <= mx) {
      const y = py(val).toFixed(1);
      grid += `<line x1="${pL}" y1="${y}" x2="${W - pR}" y2="${y}" stroke="${Math.abs(val - 1.0) < 0.01 ? "#8c959f" : "#eaeef2"}" stroke-dasharray="${Math.abs(val - 1.0) < 0.01 ? "" : "3 3"}"/>
        <text x="${pL - 6}" y="${Number(y) + 3}" text-anchor="end" font-size="10" fill="#656d76">${val.toFixed(1)}</text>`;
    }
  }
  [0, Math.floor(dates.length / 2), dates.length - 1].forEach(idx => {
    grid += `<text x="${px(idx).toFixed(1)}" y="${H - 7}" text-anchor="middle" font-size="10" fill="#656d76">${dates[idx]}</text>`;
  });
  validS.forEach(s => {
    const poly = s.vals.map((v, i) => `${px(i).toFixed(1)},${py(v).toFixed(1)}`).join(" ");
    const endV = s.vals[s.vals.length - 1];
    grid += `<polyline fill="none" stroke="${s.col}" stroke-width="${s.w || 2}" ${s.dash ? `stroke-dasharray="${s.dash}"` : ""} points="${poly}"/>
      <text x="${W - pR + 5}" y="${(py(endV) + 3).toFixed(1)}" font-size="10" font-weight="600" fill="${s.col}">${endV.toFixed(2)} (${((endV - 1) * 100).toFixed(0)}%)</text>`;
  });
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block;">${grid}</svg>`;
}

function renderStatus() {
  const cov = ((EXP.meta || {}).pool_coverage || {})[`${curPool}_${curRisk}`] || {};
  const lp = livePool();
  const rTxt = curRisk === "filtered" ? "已剔除 ST·违约·低流动性" : "不剔除";
  document.getElementById("researchStatusText").innerHTML =
    `当前筛选：<strong>${POOLS[curPool].label} · ${rTxt}</strong>（历史 ${cov.unique_bonds || (R().universe||{}).total_bonds || 0} 只 · 日均 ${fmt(cov.avg_daily || (R().universe||{}).avg_daily_bonds, 1)} 只）· 实盘符合条件 <strong>${lp.length}</strong> 只`;
  const w = document.getElementById("poolWarnBanner");
  if (curPool === "aaa") {
    w.style.display = "block";
    w.innerHTML = `<strong>样本量警示（仅AAA）：</strong>点时 AAA 池日均仅 29 只（2020 年 15–21 只），高度集中于银行及大型国企，2025–2026 年 IC 转弱且夏普跌至 0.06，以下数据完整保留供对照。`;
  } else {
    w.style.display = "none";
  }
}

function renderOverview() {
  const rLsm = getExpRow(curPool, curRisk, "100/0", 5, "none");
  const rBlend = getExpRow(curPool, curRisk, "50/50", 5, "buf_20_30");
  const rDl = getExpRow(curPool, curRisk, "0/100", 5, "none");
  const rEq = ((EXP.benchmarks || {})[`${curPool}_${curRisk}`]) || {};
  const activeRow = curOvPreset === "blend" ? rBlend : curOvPreset === "dl" ? rDl : rLsm;
  const c = (activeRow && activeRow.costs ? activeRow.costs["0.10%"] : null) || {};
  const ic20 = (R().ic_summary || {})["20"] || {};

  document.getElementById("ovKpi1Lbl").textContent = `费后年化收益 (${curOvPreset === "blend" ? "50/50+缓冲" : curOvPreset === "dl" ? "纯双低" : "原始 LSM"})`;
  const aEl = document.getElementById("ovAnnRet");
  aEl.textContent = signed(c.ann_ret);
  aEl.className = `kpi-val mono ${cls(c.ann_ret)}`;
  document.getElementById("ovAnnSub").textContent = `7年累计 ${signed(c.total_ret)} · 同池基准 ${signed((rEq.costs?.["0.10%"] || {}).ann_ret)}`;

  document.getElementById("ovCalmar").textContent = `${fmt(c.calmar, 2)} / ${fmt(c.sharpe, 2)}`;
  document.getElementById("ovCalmarSub").textContent = `Calmar = ${fmt(c.calmar, 2)} · Sharpe = ${fmt(c.sharpe, 2)}`;

  document.getElementById("ovMdd").textContent = `${fmt(c.mdd)}%`;
  document.getElementById("ovMddSub").textContent = `年化单边换手 ${fmt(activeRow?.ann_to, 1)}% · 0.10% 成本`;

  const icEl = document.getElementById("ovIc");
  icEl.textContent = typeof ic20.mean_ic === "number" ? (ic20.mean_ic > 0 ? "+" : "") + ic20.mean_ic.toFixed(4) : "—";
  icEl.className = `kpi-val mono ${cls(ic20.mean_ic)}`;
  document.getElementById("ovIcSub").textContent = `ICIR ${fmt(ic20.icir, 3)} · 正比例 ${fmt(ic20.pos_pct, 1)}% · t=${fmt(ic20.t_nw, 2)}`;

  // Valuation temperature
  const lp = livePool();
  const prices = lp.map(b => b.bond_price).filter(Number.isFinite).sort((a, b) => a - b);
  const prems = lp.map(b => b.premium_rate).filter(Number.isFinite).sort((a, b) => a - b);
  const med = arr => arr.length ? arr[Math.floor(arr.length / 2)] : null;
  const underPct = lp.length ? (lp.filter(b => b.undervaluation > 0).length / lp.length) * 100 : 0;
  document.getElementById("tempMetrics").innerHTML = `
    <div><span>实盘在市</span><strong class="mono">${lp.length} 只</strong></div>
    <div><span>中位现价</span><strong class="mono">¥${fmt(med(prices))}</strong></div>
    <div><span>中位溢价率</span><strong class="mono">${fmt(med(prems), 1)}%</strong></div>
    <div><span>LSM 低估占比</span><strong class="mono">${fmt(underPct, 1)}%</strong></div>
  `;
  const rc = { R1: 0, R2: 0, R3: 0, R4: 0 };
  lp.forEach(b => { if (rc[b.region_code] !== undefined) rc[b.region_code] += 1; });
  const cols = { R1: "#cf222e", R2: "#0969da", R3: "#1a7f37", R4: "#9a6700" };
  const tot = Math.max(1, lp.length);
  document.getElementById("tempRegBar").innerHTML = ["R1", "R2", "R3", "R4"].map(k => `<div style="width:${(rc[k]/tot*100).toFixed(1)}%;background:${cols[k]};"></div>`).join("");
  document.getElementById("tempRegLegend").innerHTML = ["R1", "R2", "R3", "R4"].map(k => `<span><i style="display:inline-block;width:7px;height:7px;border-radius:2px;background:${cols[k]};margin-right:3px;"></i>${k}:${rc[k]}</span>`).join("");

  const dates = (EXP.meta || {}).nav_dates || [];
  document.getElementById("ovNavSvg").innerHTML = drawLinesSvg(dates, [
    { vals: rLsm?.nav_010, col: "#0969da", w: 2.1 },
    { vals: rBlend?.nav_010, col: "#1a7f37", w: 2.1 },
    { vals: rDl?.nav_010, col: "#9a6700", w: 1.7 },
    { vals: rEq?.nav_010, col: "#8c959f", w: 1.4, dash: "5 4" },
  ], 540, 215, 76);

  document.getElementById("ovMiniBody").innerHTML = [
    ["原始 LSM (100/0 · 无缓冲)", rLsm],
    ["均衡增强 (50/50 · 20/30 缓冲)", rBlend],
    ["纯双低 (0/100 · 无缓冲)", rDl],
    ["同池等权基准", rEq],
  ].map(([nm, obj]) => {
    const oc = (obj && obj.costs ? obj.costs["0.10%"] : null) || {};
    return `<tr><td><strong>${nm}</strong></td><td class="num mono ${cls(oc.ann_ret)}">${signed(oc.ann_ret)}</td><td class="num mono neg">${fmt(oc.mdd)}%</td><td class="num mono">${fmt(oc.sharpe, 2)}</td><td class="num mono">${fmt(oc.calmar, 2)}</td><td class="num mono">${fmt(obj?.ann_to, 1)}%</td></tr>`;
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
      <td><strong>${b.name}</strong><span class="sub mono">${b.ticker} · ${b.stock_name} (正股 ¥${fmt(b.stock_price)} / 成交 ${b.amt_wan !== null ? fmt(b.amt_wan, 0) + "万" : "—"})</span></td>
      <td>${ratingBadge(b.rating)}</td>
      <td class="num mono">¥${fmt(b.bond_price)}</td>
      <td class="num mono"><strong>¥${fmt(b.model_price)}</strong></td>
      <td class="num mono ${b.undervaluation > 0 ? "pos" : "neg"}">${signed(b.undervaluation)}</td>
      <td class="num mono">${fmt(b.premium_rate, 1)}%<span class="sub">双低 ${fmt(b.double_low, 1)}</span></td>
      <td><span class="badge b-sig-n">${b.region_code} · ${b.region_name}</span></td>
      <td>${riskTags(b)}</td>
    </tr>
  `).join("") || `<tr><td colspan="9" style="text-align:center;padding:24px;color:var(--muted);">无匹配转债</td></tr>`;

  const ll = DATA.live_ledger || {};
  const port = (ll.portfolio_by_pool || {})[curPool] || ll.portfolio || { holdings: [], nav_net: 1.0 };
  const cmps = (ll.comparisons_by_pool || {})[curPool] || [];
  document.getElementById("portTitle").textContent = `前瞻模拟持仓组合 · ${POOLS[curPool].label}（Top 20% 周频等权）`;
  document.getElementById("portSub").textContent = `冻结起点 ${port.frozen_since || DATA.date} · 当前净值 ${fmt(port.nav_net, 4)} · 共 ${(port.holdings||[]).length} 只持仓`;
  document.getElementById("portTableBody").innerHTML = (port.holdings || []).map((h, i) => {
    const ret = h.entry_price > 0 ? ((h.last_price - h.entry_price) / h.entry_price) * 100 : 0;
    return `<tr><td class="mono" style="color:var(--muted);">${i + 1}</td><td><strong>${h.name}</strong> <span class="sub mono">${h.ticker}</span></td><td>${ratingBadge(h.rating)}</td><td class="num mono">${((h.weight||0)*100).toFixed(2)}%</td><td class="mono">${h.entry_date}</td><td class="num mono">¥${fmt(h.entry_price)}</td><td class="num mono">¥${fmt(h.last_price)}</td><td class="num mono ${cls(ret)}">${signed(ret)}</td></tr>`;
  }).join("") || `<tr><td colspan="8" style="text-align:center;color:var(--muted);">暂无持仓</td></tr>`;
  document.getElementById("ledgerHistBox").innerHTML = cmps.length ? `<div class="table-wrap" style="border-top:1px solid var(--border-soft);"><table><thead><tr><th>区间</th><th class="num">样本数</th><th class="num">Top20%</th><th class="num">Bot20%</th><th class="num">双低</th><th class="num">均值</th><th class="num">超额</th><th class="num">IC</th></tr></thead><tbody>${cmps.map(c=>`<tr><td class="mono">${c.from_date}→${c.to_date}</td><td class="num mono">${c.count}</td><td class="num mono ${cls(c.top_ret)}">${signed(c.top_ret)}</td><td class="num mono">${signed(c.bot_ret)}</td><td class="num mono">${signed(c.dl_ret)}</td><td class="num mono">${signed(c.mkt_ret)}</td><td class="num mono ${cls(c.spread)}">${signed(c.spread)}</td><td class="num mono">${fmt(c.rank_ic,3)}</td></tr>`).join("")}</tbody></table></div>` : `<div class="note-box">已归档当前范围初始冻结信号与模拟持仓（net NAV = ${fmt(port.nav_net, 4)}），后续交易日收市后累计真实前瞻对比记录。</div>`;
}

function renderBacktest() {
  const row = getExpRow(curPool, curRisk, btWeight, btFreq, btBuf);
  const rLsm = getExpRow(curPool, curRisk, "100/0", 5, "none");
  const rDl = getExpRow(curPool, curRisk, "0/100", 5, "none");
  const rEq = ((EXP.benchmarks || {})[`${curPool}_${curRisk}`]) || {};
  const c = (row && row.costs ? row.costs[btCost] : null) || {};
  const cov = ((EXP.meta || {}).pool_coverage || {})[`${curPool}_${curRisk}`] || {};

  document.getElementById("btComboId").textContent = `${POOLS[curPool].label} · ${curRisk==="filtered"?"剔除风险":"不剔除"} · ${btWeight} · ${btFreq}D · ${btBuf==="none"?"无缓冲":btBuf==="buf_20_30"?"20/30缓冲":"20/40缓冲"}`;
  document.getElementById("btAnnRet").textContent = signed(c.ann_ret);
  document.getElementById("btAnnRet").className = `kpi-val mono ${cls(c.ann_ret)}`;
  document.getElementById("btTotRet").textContent = `7年累计 ${signed(c.total_ret)}`;
  document.getElementById("btMdd").textContent = `${fmt(c.mdd)}%`;
  document.getElementById("btCostNote").textContent = `单边交易成本 ${btCost}`;
  document.getElementById("btSharpe").textContent = `${fmt(c.sharpe, 2)} / ${fmt(c.calmar, 2)}`;
  document.getElementById("btCalmar").textContent = `Calmar = ${fmt(c.calmar, 2)} · Sharpe = ${fmt(c.sharpe, 2)}`;
  document.getElementById("btTo").textContent = `${fmt(row?.ann_to, 1)}%`;
  document.getElementById("btPoolStats").textContent = `样本 ${cov.unique_bonds || "—"} 只 · 日均 ${fmt(cov.avg_daily, 1)} 只`;

  const dates = (EXP.meta || {}).nav_dates || [];
  document.getElementById("btNavSvg").innerHTML = drawLinesSvg(dates, [
    { vals: row?.nav_010, col: "#0969da", w: 2.3 },
    { vals: rLsm?.nav_010, col: "#8250df", w: 1.6 },
    { vals: rDl?.nav_010, col: "#9a6700", w: 1.6 },
    { vals: rEq?.nav_010, col: "#8c959f", w: 1.4, dash: "5 4" },
  ], 920, 235, 86);

  // Annual Returns
  const aSel = c.annual || {};
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

  // 5 Pools x 2 Risk Modes matrix under selected weight/freq/buffer
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
        <td class="num mono">${fmt(pc.calmar, 2)}</td>
        <td class="num mono">${fmt(pr.ann_to, 1)}%</td>
      </tr>`;
    });
  });
  document.getElementById("bt5PoolBody").innerHTML = pRows;
}

function renderResearch() {
  // 4A. Stepwise Attribution
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
    stBody.innerHTML = (arr || []).map(x => `
      <tr><td><strong>${x.step}</strong></td><td class="num mono ${cls(x.total_ret)}">${signed(x.total_ret)}</td><td class="num mono ${cls(x.ann_ret)}">${signed(x.ann_ret)}</td><td class="num mono neg">${fmt(x.mdd)}%</td><td class="num mono"><strong>${fmt(x.sharpe, 2)}</strong></td><td class="num mono">${fmt(x.calmar, 2)}</td><td class="num mono">${fmt(x.ann_to, 1)}%</td></tr>
    `).join("");
  }

  // 4B. Walk-Forward OOS & Overfitting Controls
  const wf = (EXP.walk_forward || {})[curOosMode] || {};
  const priors = (EXP.walk_forward || {}).priors_2021_2026 || [];
  const pbo = EXP.overfitting || {};
  document.getElementById("resOosChartTitle").textContent = `${curOosMode === "expanding" ? "扩展窗口" : "滚动 2 年"} Walk-Forward 样本外净值 (${wf.period || ""})`;
  document.getElementById("resOosSvg").innerHTML = drawLinesSvg(wf.nav_dates, [
    { vals: wf.nav_010, col: "#0969da", w: 2.2 },
    { vals: (priors[2] || {}).nav_010, col: "#1a7f37", w: 1.8 },
    { vals: (priors[0] || {}).nav_010, col: "#8c959f", w: 1.5, dash: "5 4" },
  ], 520, 210, 74);
  document.getElementById("resOosYrBody").innerHTML = (wf.yearly_picks || []).map(y => `
    <tr><td class="mono"><strong>${y.test_year}</strong></td><td class="mono" style="font-size:11px;color:var(--muted);">${y.train_window}</td><td style="font-size:11px;">${y.picked_desc}</td><td class="num mono ${cls(y.oos_ret)}">${signed(y.oos_ret)}</td><td class="num mono">${fmt(y.oos_sharpe, 2)}</td></tr>
  `).join("");
  document.getElementById("resDsrBody").innerHTML = [
    { label: `Walk-Forward 扩展窗口动态样本外 (${wf.period || ""})`, oos_ann_ret: wf.ann_ret, oos_mdd: wf.mdd, oos_sharpe: wf.sharpe, oos_calmar: wf.calmar, full_sharpe: null, dsr_prob: null },
    ...priors,
  ].map(p => `
    <tr><td><strong>${p.label}</strong></td><td class="num mono ${cls(p.oos_ann_ret)}">${signed(p.oos_ann_ret)}</td><td class="num mono neg">${fmt(p.oos_mdd)}%</td><td class="num mono">${fmt(p.oos_sharpe, 2)}</td><td class="num mono">${fmt(p.oos_calmar, 2)}</td><td class="num mono">${fmt(p.full_sharpe, 2)}</td><td class="num mono">${p.dsr_prob !== null && p.dsr_prob !== undefined ? fmt(p.dsr_prob, 4) : "样本外直连"}</td></tr>
  `).join("");
  document.getElementById("resPboNote").innerHTML =
    `<strong>多重检验防过拟合校验 (N=${pbo.n_configs || 450} 组实验)：</strong>CSCV 8 区块对称交叉验证 (${pbo.n_splits || 70} 次切分) 测得 <code>PBO 过拟合概率 = ${fmt(pbo.pbo_pct, 1)}%</code>，样本内最优夏普中位数 <code>${fmt(pbo.median_is_best_sharpe, 2)}</code> 样本外降至 <code>${fmt(pbo.median_oos_sharpe, 2)}</code>（衰减 <code>${fmt(pbo.sharpe_degradation_pct, 1)}%</code>）；450 组相关试验预期随机最高夏普基准 <code>SR* = 0.934</code>，哪怕 7 年样本内最高夏普配置 (1.19) 的 DSR 置信概率也仅 <code>0.7474</code> (&lt; 0.95)，短窗（滚动 2 年）选参样本外夏普更跌至 <code>0.03</code>。`;

  // 4C. Factor IC / Quintiles / Fama-MacBeth
  const r = R();
  const ic = r.ic_summary || {};
  const qMap = r.quintiles || {};
  document.getElementById("resFactorTitle").textContent = `冻结 LSM 因子统计预测检验（${POOLS[curPool].label}）`;
  document.getElementById("resIcBody").innerHTML = ["1", "5", "10", "20"].map(h => {
    const x = ic[h], q = qMap[h];
    if (!x) return "";
    return `<tr><td class="mono"><strong>${h}D</strong></td><td class="num mono">${x.days}</td><td class="num mono">${fmt(x.avg_n,1)}</td><td class="num mono ${cls(x.mean_ic)}">${x.mean_ic>0?"+":""}${fmt(x.mean_ic,4)}</td><td class="num mono">${fmt(x.icir,3)}</td><td class="num mono">${fmt(x.pos_pct,1)}%</td><td class="num mono"><strong>${fmt(x.t_nw,2)}</strong> <span class="badge ${x.p_val<0.05?"b-sig-y":"b-sig-n"}">p=${fmtP(x.p_val)}</span></td><td class="num mono ${cls(q?.g5_minus_g1)}">${signed(q?.g5_minus_g1,3)} (t=${fmt(q?.t_nw,2)})</td></tr>`;
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

  // Collapsible Full Grid Table
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

  // Audit
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
  renderStatus();
  renderOverview();
  renderScreener();
  renderBacktest();
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
document.querySelectorAll("#btWeightPills .pill").forEach(b => b.addEventListener("click", () => { btWeight = b.dataset.w; document.querySelectorAll("#btWeightPills .pill").forEach(x => x.classList.toggle("active", x === b)); renderBacktest(); }));
document.querySelectorAll("#btFreqPills .pill").forEach(b => b.addEventListener("click", () => { btFreq = Number(b.dataset.f); document.querySelectorAll("#btFreqPills .pill").forEach(x => x.classList.toggle("active", x === b)); renderBacktest(); }));
document.querySelectorAll("#btBufPills .pill").forEach(b => b.addEventListener("click", () => { btBuf = b.dataset.b; document.querySelectorAll("#btBufPills .pill").forEach(x => x.classList.toggle("active", x === b)); renderBacktest(); }));
document.querySelectorAll("#btCostPills .pill").forEach(b => b.addEventListener("click", () => { btCost = b.dataset.c; document.querySelectorAll("#btCostPills .pill").forEach(x => x.classList.toggle("active", x === b)); renderBacktest(); }));
document.querySelectorAll("#resStepPills .pill").forEach(b => b.addEventListener("click", () => { curStepTab = b.dataset.st; document.querySelectorAll("#resStepPills .pill").forEach(x => x.classList.toggle("active", x === b)); renderResearch(); }));
document.querySelectorAll("#resOosPills .pill").forEach(b => b.addEventListener("click", () => { curOosMode = b.dataset.oos; document.querySelectorAll("#resOosPills .pill").forEach(x => x.classList.toggle("active", x === b)); renderResearch(); }));
document.querySelectorAll("#resIcPills .pill").forEach(b => b.addEventListener("click", () => { curResIcH = b.dataset.rich; document.querySelectorAll("#resIcPills .pill").forEach(x => x.classList.toggle("active", x === b)); renderResearch(); }));
["scrSearch", "scrReg", "scrSort", "scrSafeOnly"].forEach(id => document.getElementById(id).addEventListener(id === "scrSearch" ? "input" : "change", renderScreener));
["gridFilterPool", "gridSortBy"].forEach(id => document.getElementById(id).addEventListener("change", renderResearch));

renderAll();
// Load full 450-combo matrix asynchronously for Backtest & collapsible 450-grid
fetch("research_experiments.json").then(r => r.ok ? r.json() : null).then(fullExp => {
  if (fullExp && fullExp.grid) { indexExp(fullExp); renderAll(); }
}).catch(() => {});
</script>
</body>
</html>
"""

    return template.replace("__PAYLOAD_JSON__", payload_json).replace("__UPDATED_AT__", data["updated_at"])


if __name__ == "__main__":
    generate_page_and_data()
