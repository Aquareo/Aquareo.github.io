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
    s = re.sub(r"STI$", "", s).strip()
    return s


def get_default_bonds_pool():
    return [
        {"ticker": "110075", "name": "南航转债", "stock_code": "600029", "stock_name": "南方航空", "stock_price": 5.92, "conv_price": 6.24, "bond_price": 128.40, "maturity_years": 2.8, "rating": "AAA"},
        {"ticker": "127025", "name": "冀东转债", "stock_code": "000401", "stock_name": "冀东水泥", "stock_price": 5.12, "conv_price": 5.88, "bond_price": 111.20, "maturity_years": 1.9, "rating": "AAA"},
        {"ticker": "113049", "name": "长汽转债", "stock_code": "601633", "stock_name": "长城汽车", "stock_price": 26.80, "conv_price": 34.20, "bond_price": 114.50, "maturity_years": 3.2, "rating": "AAA"},
        {"ticker": "123107", "name": "温氏转债", "stock_code": "300498", "stock_name": "温氏股份", "stock_price": 17.50, "conv_price": 19.80, "bond_price": 119.30, "maturity_years": 2.6, "rating": "AAA"},
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
    """Fetch ALL active, liquid, already-listed exchange CBs across the whole market."""
    import akshare as ak

    pool = {}
    rating_lookup = {}
    now_ts = pd.Timestamp.now()

    # 1. Primary whole-market list from Eastmoney (covers ~300+ active listed CBs + bond credit rating)
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
                }
    except Exception as e:
        print(f"Notice: bond_cb_list_em fetch warning: {e}")

    # 2. Enrich / fill remaining maturity from Jisilu where available
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
                if rem < 0.1 or price <= 10 or abs(price - 100.0) < 1e-4 or c_price <= 0 or s_price <= 0:
                    continue
                jsl_rating = norm_rating(r.get("评级") or r.get("信用评级") or "")
                if code in pool:
                    pool[code]["maturity_years"] = round(rem, 2)
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
                # Fail fast on overseas CI runners where Sina times out
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
        pool_tks = {b["ticker"] for b in pool_eval}
        for h in cur_holdings:
            tk = h["ticker"]
            w = float(h["weight"])
            lp = float(h["last_price"])
            if tk in current_map and current_map[tk]["bond_price"] > 0:
                np_px = float(current_map[tk]["bond_price"])
                port_ret += w * ((np_px - lp) / lp)
                h["last_price"] = np_px
                h["rating"] = current_map[tk].get("rating", h.get("rating", ""))
                # Keep holding until scheduled rebalance if downgraded out, or liquidate if delisted
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
    """
    Maintain survivorship-free forward ledger (`history.json`) across all 3 universes
    ('full', 'aa_plus', 'aaa'), preserving backward compatibility for `portfolio_ledger`.
    """
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
                if "snapshots" in old:
                    ledger["snapshots"] = old["snapshots"]
                if "portfolio_ledger" in old:
                    ledger["portfolio_ledger"] = old["portfolio_ledger"]
                if "portfolio_ledgers" in old:
                    ledger["portfolio_ledgers"] = old["portfolio_ledgers"]
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
        ("full", None, 10),
        ("aa_plus", {"AA+", "AAA"}, 5),
        ("aaa", {"AAA"}, 5),
    ]
    pls = ledger.get("portfolio_ledgers") or {}
    if ledger.get("portfolio_ledger") and "full" not in pls:
        pls["full"] = ledger["portfolio_ledger"]

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

    ledger["portfolio_ledger"] = portfolio_by_pool["full"]
    ledger["portfolio_ledgers"] = pls

    clean_ledger = sanitize_json(ledger)
    hist_path.write_text(json.dumps(clean_ledger, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return {
        "tracked_days": len(snapshots),
        "frozen_since": ledger["frozen_since"],
        "pool_size_today": len(evaluated),
        "pool_sizes_today": pool_sizes_today,
        "portfolio": portfolio_by_pool["full"],
        "portfolio_by_pool": portfolio_by_pool,
        "comparisons": comparisons_by_pool["full"],
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
            "rating": item.get("rating", ""),
            "stock_code": item["stock_code"],
            "stock_name": item["stock_name"],
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

    research = _load_json_if_exists(dist_dir / "research_validation.json")
    research_aa = _load_json_if_exists(dist_dir / "research_validation_aa_plus.json")
    research_aaa = _load_json_if_exists(dist_dir / "research_validation_aaa.json")

    live_ledger = update_live_history(dist_dir, date_str, lsm_sorted)

    data_payload = sanitize_json({
        "updated_at": update_time_str,
        "date": date_str,
        "author": "Yu Liu (刘宇) · The Chinese University of Hong Kong, Shenzhen",
        "paper_ref": "arXiv:2409.06496v7",
        "paper_url": "https://arxiv.org/abs/2409.06496",
        "frozen_params": FROZEN_PARAMS,
        "pool_size": len(lsm_sorted),
        "research": research,
        "research_aa_plus": research_aa,
        "research_aaa": research_aaa,
        "live_ledger": live_ledger,
        "lsm_top10": lsm_sorted[:10],
        "double_low_top10": dl_sorted[:10],
        "all_bonds": lsm_sorted,
    })

    (dist_dir / "data.json").write_text(json.dumps(data_payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (dist_dir / "index.html").write_text(build_clean_html(data_payload), encoding="utf-8")
    print(f"Generated 4-tab CB Quant Workbench at {dist_dir / 'index.html'} (live_pool={len(lsm_sorted)}, full={research.get('universe',{}).get('total_bonds', 0)}, aa+={research_aa.get('universe',{}).get('total_bonds', 0)}, aaa={research_aaa.get('universe',{}).get('total_bonds', 0)})")


def build_clean_html(data: dict) -> str:
    payload_json = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")

    template = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Convertible Bonds · 可转债量化研究 · Four-Region LSM</title>
  <meta name="description" content="中国可转债 Four-Region LSM 量化研究平台：总览、实时选券、七年历史回测与因子统计检验（全市场 / AA+及以上 / 仅AAA）">
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

    /* Top App Header (GitHub / LeetCode Workbench Shell) */
    .app-header { background: var(--surface); border-bottom: 1px solid var(--border); }
    .header-inner { max-width: 1180px; margin: 0 auto; padding: 14px 24px 0; }
    .header-top { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 10px; }
    .brand-block { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
    .brand-title { font-size: 18px; font-weight: 700; letter-spacing: -0.2px; color: var(--ink); }
    .brand-sub { font-size: 13px; color: var(--muted); }
    .status-chip {
      font-size: 11px; color: var(--muted); background: var(--subtle);
      border: 1px solid var(--border-soft); border-radius: 999px; padding: 2px 9px;
    }
    .nav-ext { display: flex; gap: 14px; flex-wrap: wrap; font-size: 12px; }
    .nav-ext a { color: var(--muted); }
    .nav-ext a:hover { color: var(--blue); }

    /* Unified Tab Bar + Global Investment Scope */
    .control-bar {
      display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 12px;
    }
    .tabs { display: flex; gap: 4px; overflow-x: auto; }
    .tab-btn {
      appearance: none; background: none; border: none; border-bottom: 2px solid transparent;
      padding: 10px 14px; font-size: 14px; font-weight: 500; color: var(--muted); cursor: pointer;
      display: inline-flex; align-items: center; gap: 7px; white-space: nowrap;
    }
    .tab-btn:hover { color: var(--ink); }
    .tab-btn.active { color: var(--ink); font-weight: 600; border-bottom-color: #fd8c73; }
    .tab-num { font-size: 11px; color: var(--light); font-weight: 600; }

    .scope-wrap { display: flex; align-items: center; gap: 8px; padding-bottom: 8px; flex-wrap: wrap; }
    .scope-label { font-size: 12px; color: var(--muted); font-weight: 500; }
    .seg { display: inline-flex; background: var(--subtle); border: 1px solid var(--border); border-radius: 6px; padding: 2px; }
    .seg-btn {
      border: none; background: transparent; padding: 4px 11px; font-size: 12px; font-weight: 500;
      color: var(--muted); border-radius: 4px; cursor: pointer; white-space: nowrap;
    }
    .seg-btn.active { background: var(--surface); color: var(--ink); font-weight: 600; box-shadow: 0 1px 2px rgba(31,35,40,0.08); }

    /* Main Container */
    .shell { max-width: 1180px; margin: 0 auto; padding: 20px 24px 56px; }
    .status-banner {
      display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;
      padding: 9px 14px; margin-bottom: 16px; border-radius: 6px; font-size: 12px;
      background: var(--surface); border: 1px solid var(--border-soft); color: var(--muted);
    }
    .status-banner strong { color: var(--ink); }
    .warn-banner {
      padding: 9px 14px; margin-bottom: 16px; border-radius: 6px; font-size: 12px;
      background: var(--amber-soft); border: 1px solid #ead18a; color: #6b4500; display: none;
    }
    .view-pane { display: none; }
    .view-pane.active { display: block; }

    /* KPI Row */
    .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 16px; }
    .kpi-card {
      background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 14px 16px;
    }
    .kpi-label { font-size: 12px; color: var(--muted); margin-bottom: 4px; display: block; }
    .kpi-val { font-size: 24px; font-weight: 700; line-height: 1.2; letter-spacing: -0.3px; }
    .kpi-sub { font-size: 11px; color: var(--muted); margin-top: 5px; }

    /* Valuation Temperature Bar */
    .temp-card {
      background: var(--surface); border: 1px solid var(--border); border-radius: 6px;
      padding: 12px 16px; margin-bottom: 16px; display: flex; justify-content: space-between;
      align-items: center; flex-wrap: wrap; gap: 14px;
    }
    .temp-metrics { display: flex; gap: 22px; flex-wrap: wrap; font-size: 12px; }
    .temp-metrics span { color: var(--muted); }
    .temp-metrics strong { color: var(--ink); font-weight: 600; margin-left: 4px; }
    .reg-bar-wrap { display: flex; align-items: center; gap: 8px; min-width: 240px; }
    .reg-bar { flex: 1; height: 8px; border-radius: 4px; overflow: hidden; display: flex; background: var(--subtle); min-width: 140px; }
    .reg-legend { display: flex; gap: 8px; font-size: 11px; color: var(--muted); }

    /* Cards & Grids */
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
    .grid-overview { display: grid; grid-template-columns: 1.15fr 1fr; gap: 16px; margin-bottom: 16px; }
    .card {
      background: var(--surface); border: 1px solid var(--border); border-radius: 6px;
      overflow: hidden; margin-bottom: 16px;
    }
    .grid-2 > .card, .grid-overview > .card { margin-bottom: 0; }
    .card-head {
      padding: 12px 16px; background: #fbfcfd; border-bottom: 1px solid var(--border-soft);
      display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;
    }
    .card-title { font-size: 14px; font-weight: 600; margin: 0; color: var(--ink); }
    .card-sub { font-size: 12px; color: var(--muted); }
    .card-body { padding: 14px 16px; }

    /* Pills & Inputs */
    .pills { display: flex; gap: 5px; flex-wrap: wrap; }
    .pill {
      padding: 4px 10px; border-radius: 5px; font-size: 12px; border: 1px solid var(--border);
      background: var(--surface); color: var(--muted); cursor: pointer;
    }
    .pill.active { background: var(--blue-soft); border-color: var(--blue); color: var(--blue); font-weight: 600; }
    .toolbar {
      padding: 10px 16px; background: #fbfcfd; border-bottom: 1px solid var(--border-soft);
      display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;
    }
    .filter-row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .inp, .sel {
      border: 1px solid var(--border); background: var(--surface); border-radius: 5px;
      padding: 5px 10px; font-size: 12px; color: var(--ink);
    }
    .inp { min-width: 190px; }

    /* Tables */
    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; text-align: left; font-size: 13px; }
    th {
      background: #fbfcfd; color: var(--muted); font-weight: 600; font-size: 12px;
      padding: 9px 14px; border-bottom: 1px solid var(--border-soft); white-space: nowrap;
    }
    td { padding: 9px 14px; border-bottom: 1px solid var(--border-soft); vertical-align: middle; }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: #f8fafc; }
    .hl-row td { background: #f0f7ff; }
    td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
    .sub { display: block; font-size: 11px; color: var(--muted); margin-top: 1px; }

    /* Semantic Colors & Badges */
    .pos { color: var(--red); font-weight: 600; }
    .neg { color: var(--green); font-weight: 600; }
    .badge {
      display: inline-block; font-size: 11px; padding: 1px 7px; border-radius: 999px;
      font-weight: 500; white-space: nowrap; border: 1px solid transparent;
    }
    .b-aaa { background: #ddf4ff; color: #0969da; border-color: #b6e3ff; }
    .b-aap { background: #dafbe1; color: #1a7f37; border-color: #aceebb; }
    .b-aa  { background: #f3f4f6; color: #57606a; border-color: #d0d7de; }
    .b-low { background: #ffebe9; color: #cf222e; border-color: #ffcecb; }
    .b-warn { background: #fff8c5; color: #9a6700; border-color: #f2cc60; margin-right: 4px; }
    .b-sig-y { background: var(--green-soft); color: var(--green); border-color: #aceebb; }
    .b-sig-n { background: var(--subtle); color: var(--muted); border-color: var(--border); }

    .legend { display: flex; gap: 14px; flex-wrap: wrap; font-size: 11px; color: var(--muted); margin-bottom: 8px; }
    .legend span { display: inline-flex; align-items: center; gap: 5px; }
    .ldot { width: 10px; height: 3px; border-radius: 2px; display: inline-block; }
    .note-box {
      padding: 10px 16px; background: #fbfcfd; border-top: 1px solid var(--border-soft);
      font-size: 12px; color: var(--muted); line-height: 1.6;
    }
    .btn-link {
      font-size: 12px; font-weight: 500; color: var(--blue); background: none; border: none;
      cursor: pointer; padding: 0;
    }
    .btn-link:hover { text-decoration: underline; }

    footer {
      margin-top: 24px; padding-top: 14px; border-top: 1px solid var(--border);
      font-size: 12px; color: var(--muted); display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px;
    }

    @media (max-width: 900px) {
      .kpi-grid { grid-template-columns: 1fr 1fr; }
      .grid-2, .grid-overview { grid-template-columns: 1fr; }
      .shell, .header-inner { padding-left: 14px; padding-right: 14px; }
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
        <span class="status-chip mono">arXiv:2409.06496v7</span>
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
      <div class="scope-wrap">
        <span class="scope-label">投资范围</span>
        <div class="seg" id="poolSeg">
          <button class="seg-btn active" data-pool="full">全市场</button>
          <button class="seg-btn" data-pool="aa_plus">AA+及以上</button>
          <button class="seg-btn" data-pool="aaa">仅AAA</button>
        </div>
      </div>
    </div>
  </div>
</header>

<main class="shell">
  <div class="status-banner">
    <div id="researchStatusText">研究状态加载中...</div>
    <div class="mono" style="font-size:11px;">冻结参数 p_reset=0.80 · B=¥108 · r=2.5% · 实盘更新 __UPDATED_AT__</div>
  </div>
  <div class="warn-banner" id="poolWarnBanner"></div>

  <!-- 01 OVERVIEW -->
  <section class="view-pane active" id="view-overview">
    <div class="kpi-grid">
      <div class="kpi-card">
        <span class="kpi-label">费后年化收益 (周频 0.10%)</span>
        <div class="kpi-val mono" id="kpiAnnRet">—</div>
        <div class="kpi-sub" id="kpiAnnSub">—</div>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">年化超额收益 (vs 同池基准)</span>
        <div class="kpi-val mono" id="kpiExcess">—</div>
        <div class="kpi-sub" id="kpiExcessSub">—</div>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">最大回撤</span>
        <div class="kpi-val mono neg" id="kpiMdd">—</div>
        <div class="kpi-sub" id="kpiMddSub">—</div>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">20 日 Rank IC</span>
        <div class="kpi-val mono" id="kpiIc">—</div>
        <div class="kpi-sub" id="kpiIcSub">—</div>
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
            <h2 class="card-title">策略累计净值（2020–2026 历史回测）</h2>
            <span class="card-sub">T+1 开盘调仓 · 单边 0.10% 成本 · 历史回测不等同于实盘未来收益</span>
          </div>
          <button class="btn-link" data-jump="backtest">详细回测 →</button>
        </div>
        <div class="card-body">
          <div class="legend">
            <span><i class="ldot" style="background:#0969da;"></i>原始模型 Four-Region LSM</span>
            <span><i class="ldot" style="background:#9a6700;"></i>传统双低策略</span>
            <span><i class="ldot" style="background:#8c959f;"></i>同池等权基准</span>
          </div>
          <div id="ovNavSvg"></div>
        </div>
      </div>

      <div class="card">
        <div class="card-head">
          <div>
            <h2 class="card-title" id="ovTopTitle">当前低估转债 Top 10</h2>
            <span class="card-sub">按当前筛选投资范围内 LSM 理论低估率排序</span>
          </div>
          <button class="btn-link" data-jump="screener">全部转债 →</button>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr><th>转债</th><th>评级</th><th class="num">现价</th><th class="num">估值</th><th class="num">低估率</th></tr>
            </thead>
            <tbody id="ovTopBody"></tbody>
          </table>
        </div>
      </div>
    </div>
  </section>

  <!-- 02 SCREENER + FORWARD PORTFOLIO -->
  <section class="view-pane" id="view-screener">
    <div class="card">
      <div class="card-head">
        <div>
          <h2 class="card-title" id="scrTitle">实时转债估值选券池</h2>
          <span class="card-sub">基于 Python Four-Region LSM 冻结参数引擎实时计算，严格匹配当前投资范围</span>
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
              <th style="text-align:center;">#</th>
              <th>转债标的</th>
              <th>评级</th>
              <th class="num">现价</th>
              <th class="num">LSM估值</th>
              <th class="num">低估率</th>
              <th class="num">溢价 / 双低</th>
              <th>条款区</th>
              <th>风险提示</th>
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
          <span class="card-sub" id="portSub">独立于 2020–2026 历史回测的前瞻冻结账本 · 自动匹配当前投资范围</span>
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

  <!-- 03 BACKTEST -->
  <section class="view-pane" id="view-backtest">
    <div class="card">
      <div class="card-head">
        <div>
          <h2 class="card-title">三档投资范围核心对照（2020-01-02 → 2026-11-24）</h2>
          <span class="card-sub">统一周频 5D · T+1 开盘成交 · 单边 0.10% 成本 · 点时信用评级公告次日生效</span>
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>投资范围</th>
              <th>样本规模 (转债 / 日均)</th>
              <th class="num">原始模型年化</th>
              <th class="num">双低策略年化</th>
              <th class="num">同池基准年化</th>
              <th class="num">20日 Rank IC</th>
              <th class="num">最大回撤</th>
              <th class="num">夏普比率</th>
            </tr>
          </thead>
          <tbody id="bt3PoolBody"></tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <div class="card-head">
        <div>
          <h2 class="card-title">策略累计净值与多策略对照（当前投资范围）</h2>
          <span class="card-sub">净值曲线按 0.10% 标准单边成本绘制，下方明细表支持切换 0.05% / 0.10% / 0.20% 成本与调仓频率</span>
        </div>
        <div class="pills" id="btCostPills">
          <button class="pill" data-cost="0.05%">0.05% 低佣</button>
          <button class="pill active" data-cost="0.10%">0.10% 标准成本</button>
          <button class="pill" data-cost="0.20%">0.20% 高滑点</button>
        </div>
      </div>
      <div class="card-body">
        <div class="legend">
          <span><i class="ldot" style="background:#0969da;"></i>Four-Region LSM</span>
          <span><i class="ldot" style="background:#9a6700;"></i>传统双低</span>
          <span><i class="ldot" style="background:#1a7f37;"></i>无下修 LSM (p=0)</span>
          <span><i class="ldot" style="background:#8250df;"></i>单一区域 LSM</span>
          <span><i class="ldot" style="background:#8c959f;"></i>同池等权基准</span>
        </div>
        <div id="btNavSvg"></div>
      </div>
      <div class="table-wrap" style="border-top:1px solid var(--border-soft);">
        <table>
          <thead>
            <tr>
              <th>策略 / 对照组 (周频 5D)</th>
              <th>测试目的</th>
              <th class="num">累计净收益</th>
              <th class="num">年化净收益</th>
              <th class="num">最大回撤</th>
              <th class="num">夏普比率</th>
              <th class="num">年化单边换手</th>
            </tr>
          </thead>
          <tbody id="btStratBody"></tbody>
        </table>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-head">
          <div>
            <h2 class="card-title">分年度净收益对照（2020–2026 YTD）</h2>
            <span class="card-sub">按周频 0.10% 成本净值序列切分自然年度</span>
          </div>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr><th>年份</th><th class="num">Four-Region LSM</th><th class="num">传统双低</th><th class="num">同池基准</th><th class="num">LSM 年内最大回撤</th></tr>
            </thead>
            <tbody id="btAnnualBody"></tbody>
          </table>
        </div>
      </div>

      <div class="card">
        <div class="card-head">
          <div>
            <h2 class="card-title">调仓频率与市场阶段稳健性</h2>
            <span class="card-sub">检验 1D / 5D / 10D / 20D 频率与 2020–2026 阶段表现</span>
          </div>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr><th>调仓频率</th><th class="num">0.05% 成本</th><th class="num">0.10% 成本</th><th class="num">0.20% 成本</th></tr>
            </thead>
            <tbody id="btFreqBody"></tbody>
          </table>
        </div>
        <div class="table-wrap" style="border-top:1px solid var(--border-soft);">
          <table>
            <thead>
              <tr><th>市场时段</th><th class="num">LSM 收益 (Sharpe)</th><th class="num">双低收益</th><th class="num">基准收益</th></tr>
            </thead>
            <tbody id="btSubBody"></tbody>
          </table>
        </div>
      </div>
    </div>
  </section>

  <!-- 04 RESEARCH -->
  <section class="view-pane" id="view-research">
    <div class="card">
      <div class="card-head">
        <div>
          <h2 class="card-title">跨期预测能力检验（Rank IC · ICIR · 显著性与衰减）</h2>
          <span class="card-sub">严格对齐全局交易日历，越界收益窗口不截断、不填充</span>
        </div>
        <div class="pills" id="resIcPills">
          <button class="pill" data-rich="5">滚动60日 (h=5)</button>
          <button class="pill" data-rich="10">滚动60日 (h=10)</button>
          <button class="pill active" data-rich="20">滚动60日 (h=20)</button>
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>持有期</th><th class="num">有效天数</th><th class="num">日均样本</th><th class="num">Mean Rank IC</th>
              <th class="num">IC标准差</th><th class="num">ICIR</th><th class="num">IC&gt;0比例</th><th class="num">Newey-West t</th><th class="num">95% CI</th>
            </tr>
          </thead>
          <tbody id="resIcBody"></tbody>
        </table>
      </div>
      <div class="grid-2" style="padding:14px 16px;border-top:1px solid var(--border-soft);margin-bottom:0;">
        <div>
          <div style="font-size:12px;font-weight:600;margin-bottom:6px;">滚动 60 日 Rank IC 走势</div>
          <div id="resRollSvg"></div>
        </div>
        <div>
          <div style="font-size:12px;font-weight:600;margin-bottom:6px;">IC 持有期衰减曲线 (1D → 20D)</div>
          <div id="resDecaySvg"></div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-head">
        <div>
          <h2 class="card-title">五分组单调性检验（G1 高估 → G5 低估）</h2>
          <span class="card-sub" id="resQStat">—</span>
        </div>
        <div class="pills" id="resQPills">
          <button class="pill" data-rqh="1">未来 1 日</button>
          <button class="pill" data-rqh="5">未来 5 日</button>
          <button class="pill" data-rqh="10">未来 10 日</button>
          <button class="pill active" data-rqh="20">未来 20 日</button>
        </div>
      </div>
      <div class="grid-2" style="padding:14px 16px;margin-bottom:0;">
        <div>
          <div style="font-size:12px;font-weight:600;margin-bottom:6px;" id="resQBarTitle">五分组平均超额收益</div>
          <div id="resQBarSvg"></div>
        </div>
        <div>
          <div style="font-size:12px;font-weight:600;margin-bottom:6px;">2020–2026 五分组周频累计净值 (0.10% 成本)</div>
          <div class="legend">
            <span><i class="ldot" style="background:#cf222e;"></i>G5 最被低估</span>
            <span><i class="ldot" style="background:#0969da;"></i>G4</span>
            <span><i class="ldot" style="background:#656d76;"></i>G3</span>
            <span><i class="ldot" style="background:#9a6700;"></i>G2</span>
            <span><i class="ldot" style="background:#1a7f37;"></i>G1 最被高估</span>
          </div>
          <div id="resQNavSvg"></div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-head">
        <div>
          <h2 class="card-title">Fama-MacBeth 每日横截面回归（纯因果 6 因子对照）</h2>
          <span class="card-sub" id="resFmbNote">剔除含事后下调信息的评级变量，检验控制价格、溢价率、波动率、剩余期限与正股动量后的增量定价 β</span>
        </div>
        <div class="pills" id="resFmbPills">
          <button class="pill" data-rfh="5">h = 5 交易日</button>
          <button class="pill" data-rfh="10">h = 10 交易日</button>
          <button class="pill active" data-rfh="20">h = 20 交易日</button>
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>标准化因子 (1 SD)</th><th class="num">均值斜率 (%)</th><th class="num">NW 标准误</th>
              <th class="num">Newey-West t</th><th class="num">双侧 p 值</th><th class="num">95% CI</th><th>显著性</th>
            </tr>
          </thead>
          <tbody id="resFmbBody"></tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <div class="card-head">
        <div>
          <h2 class="card-title">工程审计记录、点时评级校验与文献说明</h2>
          <span class="card-sub" id="resAuditSub">代码审计修正对比与可复现说明（注：已移除原单券 BS 近似试算器，所有实盘与回测估值均由 Python LSM 蒙特卡洛引擎计算）</span>
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead id="resAuditHead"></thead>
          <tbody id="resAuditBody"></tbody>
        </table>
      </div>
      <div class="note-box">
        <strong>学术参考与复现脚本：</strong>
        Liu, Yu. (2026). <em>Valuation Model of Chinese Convertible Bonds Based on Monte Carlo Simulation: A Regression-Based Approximation and Error Analysis</em> (<a href="https://arxiv.org/abs/2409.06496" target="_blank" rel="noopener">arXiv:2409.06496v7</a>).
        全市场拉取 <code>tools/fetch_all_cbs.py</code> · 免费点时评级构建 <code>tools/fetch_free_ratings.py</code> · 回测引擎 <code>tools/run_full_validation.py</code> · 高评级过滤 <code>tools/run_high_grade_validation.py</code>。
      </div>
    </div>
  </section>

  <footer>
    <span>Yu Liu (刘宇) · CUHK-Shenzhen · Four-Region LSM 冻结参数实证工作台</span>
    <span>仅作学术研究验证，历史回测不构成投资建议</span>
  </footer>
</main>

<script>
const DATA = __PAYLOAD_JSON__;
const POOLS = {
  full: { label: "全市场", grades: null, r: () => DATA.research || {} },
  aa_plus: { label: "AA+及以上", grades: new Set(["AA+", "AAA"]), r: () => DATA.research_aa_plus || {} },
  aaa: { label: "仅AAA", grades: new Set(["AAA"]), r: () => DATA.research_aaa || {} },
};

let curTab = "overview";
let curPool = "full";
let curCost = "0.10%";
let curResIcH = "20";
let curResQH = "20";
let curResFmbH = "20";

const R = () => POOLS[curPool].r();
const fmt = (v, n = 2) => typeof v === "number" && Number.isFinite(v) ? v.toFixed(n) : "—";
const signed = (v, n = 2, suf = "%") => typeof v === "number" && Number.isFinite(v) ? (v > 0 ? "+" : "") + v.toFixed(n) + suf : "—";
const cls = v => typeof v === "number" ? (v > 0 ? "pos" : v < 0 ? "neg" : "") : "";
const fmtP = p => (typeof p !== "number" || !Number.isFinite(p)) ? "—" : (p < 0.0001 ? "<0.0001" : p.toFixed(4));

function livePool() {
  const gs = POOLS[curPool].grades;
  const all = DATA.all_bonds || [];
  return gs ? all.filter(b => gs.has((b.rating || "").toUpperCase())) : all;
}

function ratingBadge(rt) {
  const g = (rt || "").toUpperCase();
  if (!g) return `<span class="badge b-sig-n">未披露</span>`;
  const c = g === "AAA" ? "b-aaa" : g === "AA+" ? "b-aap" : g.startsWith("AA") ? "b-aa" : "b-low";
  return `<span class="badge ${c} mono">${g}</span>`;
}

function riskTags(b) {
  const tags = [];
  if (b.region_code === "R1") tags.push(`<span class="badge b-warn">强赎R1</span>`);
  if (b.bond_price > 150) tags.push(`<span class="badge b-low">高价&gt;150</span>`);
  if (b.premium_rate > 80) tags.push(`<span class="badge b-warn">高溢价</span>`);
  const g = (b.rating || "").toUpperCase();
  if (g && !g.startsWith("AA") && g !== "AAA") tags.push(`<span class="badge b-low">低信用 ${g}</span>`);
  return tags.length ? tags.join("") : `<span style="font-size:11px;color:var(--light);">常规</span>`;
}

function drawLinesSvg(dates, seriesList, W = 560, H = 220, rightPad = 78) {
  if (!dates || !dates.length) return `<div style="padding:32px;text-align:center;color:var(--muted);">暂无净值序列数据</div>`;
  const pL = 42, pR = rightPad, pT = 16, pB = 26;
  const allV = seriesList.flatMap(s => s.vals || []);
  const mn = Math.min(0.95, ...allV) * 0.96, mx = Math.max(1.25, ...allV) * 1.04;
  const px = i => pL + (i / Math.max(1, dates.length - 1)) * (W - pL - pR);
  const py = v => H - pB - ((v - mn) / Math.max(1e-6, mx - mn)) * (H - pT - pB);
  let grid = "";
  const ticks = [];
  const step = mx - mn > 1.2 ? 0.5 : 0.2;
  for (let lv = Math.floor(mn / step) * step; lv <= mx + 1e-6; lv += step) {
    const val = Math.round(lv * 100) / 100;
    if (val >= mn && val <= mx) ticks.push(val);
  }
  ticks.forEach(lv => {
    const y = py(lv).toFixed(1);
    grid += `<line x1="${pL}" y1="${y}" x2="${W - pR}" y2="${y}" stroke="${Math.abs(lv - 1.0) < 0.01 ? "#8c959f" : "#eaeef2"}" stroke-dasharray="${Math.abs(lv - 1.0) < 0.01 ? "" : "3 3"}"/>
      <text x="${pL - 6}" y="${Number(y) + 3}" text-anchor="end" font-size="10" fill="#656d76">${lv.toFixed(1)}</text>`;
  });
  [0, Math.floor(dates.length / 2), dates.length - 1].forEach(idx => {
    grid += `<text x="${px(idx).toFixed(1)}" y="${H - 7}" text-anchor="middle" font-size="10" fill="#656d76">${dates[idx]}</text>`;
  });
  seriesList.forEach(s => {
    if (!s.vals || !s.vals.length) return;
    const poly = s.vals.map((v, i) => `${px(i).toFixed(1)},${py(v).toFixed(1)}`).join(" ");
    const endV = s.vals[s.vals.length - 1];
    grid += `<polyline fill="none" stroke="${s.col}" stroke-width="${s.w || 2}" ${s.dash ? `stroke-dasharray="${s.dash}"` : ""} points="${poly}"/>
      <text x="${W - pR + 5}" y="${(py(endV) + 3).toFixed(1)}" font-size="10" font-weight="600" fill="${s.col}">${endV.toFixed(2)} (${((endV - 1) * 100).toFixed(0)}%)</text>`;
  });
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block;">${grid}</svg>`;
}

function renderStatus() {
  const r = R();
  const u = r.universe || {};
  const rf = r.rating_filter;
  const lp = livePool();
  const statusEl = document.getElementById("researchStatusText");
  const warnEl = document.getElementById("poolWarnBanner");
  if (rf) {
    statusEl.innerHTML = `当前范围：<strong>${POOLS[curPool].label}</strong>（历史 ${u.total_bonds || 0} 只 · 日均 ${fmt(u.avg_daily_bonds, 1)} 只 · 点时评级覆盖率 ${((rf.rating_coverage_all_bond_days || 0) * 100).toFixed(2)}%）· 实盘可筛 <strong>${lp.length}</strong> 只`;
  } else {
    statusEl.innerHTML = `当前范围：<strong>全市场</strong>（历史 ${u.total_bonds || 0} 只 · ${Number(u.total_bond_days || 0).toLocaleString()} 样本日 · 无幸存者偏差）· 实盘可筛 <strong>${lp.length}</strong> 只`;
  }
  if (curPool === "aaa") {
    warnEl.style.display = "block";
    warnEl.innerHTML = `<strong>样本量提示（仅AAA）：</strong>点时 AAA 池日均仅 ${fmt(u.avg_daily_bonds, 1)} 只转债（2020 年 15–21 只，2026 年 22–25 只），高度集中于银行与大型国企；周频组合每次仅持 5–8 只，2025–2026 年 20 日 IC 转弱，截面回归受限于每日至少 35 只门槛仅覆盖 ${((r.fama_macbeth || {})["20"] || {}).days || 0} 个交易日。`;
  } else {
    warnEl.style.display = "none";
  }
}

function renderOverview() {
  const r = R();
  const st = r.strategies || {};
  const lsm = (st.four_region_lsm?.costs || {})["0.10%"] || {};
  const mkt = (st.mkt_ew?.costs || {})["0.10%"] || {};
  const ic20 = (r.ic_summary || {})["20"] || {};
  const exAnn = (typeof lsm.ann_ret === "number" && typeof mkt.ann_ret === "number") ? (lsm.ann_ret - mkt.ann_ret) : null;

  const annEl = document.getElementById("kpiAnnRet");
  annEl.textContent = signed(lsm.ann_ret);
  annEl.className = `kpi-val mono ${cls(lsm.ann_ret)}`;
  document.getElementById("kpiAnnSub").textContent = `7年累计 ${signed(lsm.total_ret)} · Sharpe ${fmt(lsm.sharpe, 2)}`;

  const exEl = document.getElementById("kpiExcess");
  exEl.textContent = signed(exAnn);
  exEl.className = `kpi-val mono ${cls(exAnn)}`;
  document.getElementById("kpiExcessSub").textContent = `同池等权基准年化 ${signed(mkt.ann_ret)}`;

  document.getElementById("kpiMdd").textContent = `${fmt(lsm.mdd)}%`;
  document.getElementById("kpiMddSub").textContent = `日均 ${(r.universe || {}).avg_daily_bonds || "—"} 只 · 年化换手 ${fmt(st.four_region_lsm?.ann_to, 1)}%`;

  const icEl = document.getElementById("kpiIc");
  icEl.textContent = typeof ic20.mean_ic === "number" ? (ic20.mean_ic > 0 ? "+" : "") + ic20.mean_ic.toFixed(4) : "—";
  icEl.className = `kpi-val mono ${cls(ic20.mean_ic)}`;
  document.getElementById("kpiIcSub").textContent = `ICIR ${fmt(ic20.icir, 3)} · 正比例 ${fmt(ic20.pos_pct, 1)}% · t=${fmt(ic20.t_nw, 2)}`;

  // Market Valuation Temperature (filtered by active pool)
  const lp = livePool();
  const prices = lp.map(b => b.bond_price).filter(Number.isFinite).sort((a, b) => a - b);
  const prems = lp.map(b => b.premium_rate).filter(Number.isFinite).sort((a, b) => a - b);
  const med = arr => arr.length ? arr[Math.floor(arr.length / 2)] : null;
  const underPct = lp.length ? (lp.filter(b => b.undervaluation > 0).length / lp.length) * 100 : 0;
  document.getElementById("tempMetrics").innerHTML = `
    <div><span>实盘在市</span><strong class="mono">${lp.length} 只</strong></div>
    <div><span>中位现价</span><strong class="mono">¥${fmt(med(prices))}</strong></div>
    <div><span>中位溢价率</span><strong class="mono">${fmt(med(prems), 1)}%</strong></div>
    <div><span>模型低估占比</span><strong class="mono">${fmt(underPct, 1)}%</strong></div>
  `;
  const rc = { R1: 0, R2: 0, R3: 0, R4: 0 };
  lp.forEach(b => { if (rc[b.region_code] !== undefined) rc[b.region_code] += 1; });
  const cols = { R1: "#cf222e", R2: "#0969da", R3: "#1a7f37", R4: "#9a6700" };
  const tot = Math.max(1, lp.length);
  document.getElementById("tempRegBar").innerHTML = ["R1", "R2", "R3", "R4"].map(k =>
    `<div style="width:${(rc[k] / tot * 100).toFixed(1)}%;background:${cols[k]};" title="${k}: ${rc[k]}只"></div>`
  ).join("");
  document.getElementById("tempRegLegend").innerHTML = ["R1", "R2", "R3", "R4"].map(k =>
    `<span><i style="display:inline-block;width:7px;height:7px;border-radius:2px;background:${cols[k]};margin-right:3px;"></i>${k}:${rc[k]}</span>`
  ).join("");

  // Overview Equity Curve
  const first = st.four_region_lsm || {};
  document.getElementById("ovNavSvg").innerHTML = drawLinesSvg(first.nav_dates, [
    { vals: first.nav_010, col: "#0969da", w: 2.2 },
    { vals: (st.double_low || {}).nav_010, col: "#9a6700", w: 1.8 },
    { vals: (st.mkt_ew || {}).nav_010, col: "#8c959f", w: 1.5, dash: "5 4" },
  ], 540, 230, 78);

  // Overview Top 10 Undervalued Bonds (within active pool!)
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

  let items = livePool().filter(b => {
    if (reg !== "ALL" && b.region_code !== reg) return false;
    if (safeOnly && (b.bond_price > 150 || b.premium_rate > 80)) return false;
    if (q) {
      const hay = `${b.ticker} ${b.name} ${b.stock_code} ${b.stock_name}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });

  if (sortBy === "underval") items.sort((a, b) => b.undervaluation - a.undervaluation);
  else if (sortBy === "price") items.sort((a, b) => a.bond_price - b.bond_price);
  else if (sortBy === "dl") items.sort((a, b) => a.double_low - b.double_low);
  else if (sortBy === "prem") items.sort((a, b) => a.premium_rate - b.premium_rate);

  document.getElementById("scrTitle").textContent = `实时转债估值选券池（${POOLS[curPool].label}）`;
  document.getElementById("scrCountNote").textContent = `筛选命中 ${items.length} / ${livePool().length} 只`;
  document.getElementById("scrTableBody").innerHTML = items.slice(0, 150).map((b, i) => `
    <tr>
      <td class="mono" style="text-align:center;color:var(--muted);">${i + 1}</td>
      <td><strong>${b.name}</strong><span class="sub mono">${b.ticker} · ${b.stock_name} (正股 ¥${fmt(b.stock_price)} / 转股 ¥${fmt(b.conv_price)})</span></td>
      <td>${ratingBadge(b.rating)}</td>
      <td class="num mono">¥${fmt(b.bond_price)}</td>
      <td class="num mono"><strong>¥${fmt(b.model_price)}</strong></td>
      <td class="num mono ${b.undervaluation > 0 ? "pos" : "neg"}">${signed(b.undervaluation)}</td>
      <td class="num mono">${fmt(b.premium_rate, 1)}%<span class="sub">双低 ${fmt(b.double_low, 1)}</span></td>
      <td><span class="badge b-sig-n">${b.region_code} · ${b.region_name}</span></td>
      <td>${riskTags(b)}</td>
    </tr>
  `).join("") || `<tr><td colspan="9" style="text-align:center;padding:24px;color:var(--muted);">无匹配转债</td></tr>`;

  // Forward Simulated Portfolio for the ACTIVE pool!
  const ll = DATA.live_ledger || {};
  const portMap = ll.portfolio_by_pool || { full: ll.portfolio };
  const port = portMap[curPool] || { holdings: [], nav_net: 1.0, last_rebal_date: DATA.date, frozen_since: DATA.date };
  const cmpMap = ll.comparisons_by_pool || { full: ll.comparisons };
  const cmps = cmpMap[curPool] || [];

  document.getElementById("portTitle").textContent = `前瞻模拟持仓组合 · ${POOLS[curPool].label}（Top 20% 周频等权）`;
  document.getElementById("portSub").textContent = `启动冻结日 ${port.frozen_since || DATA.date} · 当前组合净值 ${fmt(port.nav_net, 4)} · 共 ${(port.holdings || []).length} 只持仓 · 上次调仓 ${port.last_rebal_date || DATA.date}`;
  document.getElementById("portTableBody").innerHTML = (port.holdings || []).map((h, i) => {
    const ret = h.entry_price > 0 ? ((h.last_price - h.entry_price) / h.entry_price) * 100 : 0;
    return `
      <tr>
        <td class="mono" style="color:var(--muted);">${i + 1}</td>
        <td><strong>${h.name}</strong> <span class="sub mono">${h.ticker}</span></td>
        <td>${ratingBadge(h.rating)}</td>
        <td class="num mono">${((h.weight || 0) * 100).toFixed(2)}%</td>
        <td class="mono">${h.entry_date}</td>
        <td class="num mono">¥${fmt(h.entry_price)}</td>
        <td class="num mono">¥${fmt(h.last_price)}</td>
        <td class="num mono ${cls(ret)}">${signed(ret)}</td>
      </tr>
    `;
  }).join("") || `<tr><td colspan="8" style="text-align:center;color:var(--muted);">暂无模拟持仓</td></tr>`;

  document.getElementById("ledgerHistBox").innerHTML = cmps.length ? `
    <div class="table-wrap" style="border-top:1px solid var(--border-soft);">
      <table>
        <thead><tr><th>前瞻区间</th><th class="num">匹配数</th><th class="num">低估 Top20%</th><th class="num">高估 Bot20%</th><th class="num">传统双低</th><th class="num">同池均值</th><th class="num">多空超额</th><th class="num">Rank IC</th></tr></thead>
        <tbody>${cmps.map(c => `<tr><td class="mono">${c.from_date} → ${c.to_date}</td><td class="num mono">${c.count}</td><td class="num mono ${cls(c.top_ret)}">${signed(c.top_ret)}</td><td class="num mono ${cls(c.bot_ret)}">${signed(c.bot_ret)}</td><td class="num mono ${cls(c.dl_ret)}">${signed(c.dl_ret)}</td><td class="num mono ${cls(c.mkt_ret)}">${signed(c.mkt_ret)}</td><td class="num mono ${cls(c.spread)}">${signed(c.spread)}</td><td class="num mono">${c.rank_ic > 0 ? "+" : ""}${fmt(c.rank_ic, 3)}</td></tr>`).join("")}</tbody>
      </table>
    </div>` : `<div class="note-box">当前范围已冻结记录今日信号与初始模拟持仓（net NAV = ${fmt(port.nav_net, 4)}），后续交易日收市后将在下方累积实盘前瞻兑现记录。</div>`;
}

function calcYearlyPerf(dates, navs) {
  const byYr = {};
  if (!dates || !navs) return byYr;
  dates.forEach((d, i) => {
    const yr = String(d).slice(0, 4);
    if (!byYr[yr]) byYr[yr] = { vals: [], startPrev: i > 0 ? navs[i - 1] : 1.0 };
    byYr[yr].vals.push(navs[i]);
  });
  const out = {};
  Object.keys(byYr).sort().forEach(yr => {
    const { vals, startPrev } = byYr[yr];
    const ret = ((vals[vals.length - 1] / Math.max(1e-9, startPrev)) - 1.0) * 100.0;
    let peak = startPrev, mdd = 0;
    vals.forEach(v => {
      if (v > peak) peak = v;
      const dd = (v / peak - 1.0) * 100.0;
      if (dd < mdd) mdd = dd;
    });
    out[yr] = { ret, mdd };
  });
  return out;
}

function renderBacktest() {
  // 3-Pool Comparison Table
  document.getElementById("bt3PoolBody").innerHTML = ["full", "aa_plus", "aaa"].map(pk => {
    const pr = POOLS[pk].r();
    const u = pr.universe || {};
    const st = pr.strategies || {};
    const lsm = (st.four_region_lsm?.costs || {})["0.10%"] || {};
    const dl = (st.double_low?.costs || {})["0.10%"] || {};
    const mk = (st.mkt_ew?.costs || {})["0.10%"] || {};
    const ic20 = (pr.ic_summary || {})["20"] || {};
    return `
      <tr class="${pk === curPool ? "hl-row" : ""}">
        <td><strong>${POOLS[pk].label}</strong></td>
        <td class="mono">${u.total_bonds || 0} 只 / 日均 ${fmt(u.avg_daily_bonds, 1)} 只</td>
        <td class="num mono ${cls(lsm.ann_ret)}">${signed(lsm.ann_ret)}</td>
        <td class="num mono ${cls(dl.ann_ret)}">${signed(dl.ann_ret)}</td>
        <td class="num mono ${cls(mk.ann_ret)}">${signed(mk.ann_ret)}</td>
        <td class="num mono">${typeof ic20.mean_ic === "number" ? (ic20.mean_ic > 0 ? "+" : "") + ic20.mean_ic.toFixed(4) : "—"}</td>
        <td class="num mono neg">${fmt(lsm.mdd)}%</td>
        <td class="num mono"><strong>${fmt(lsm.sharpe, 2)}</strong></td>
      </tr>
    `;
  }).join("");

  const r = R();
  const st = r.strategies || {};
  const first = st.four_region_lsm || {};
  document.getElementById("btNavSvg").innerHTML = drawLinesSvg(first.nav_dates, [
    { vals: first.nav_010, col: "#0969da", w: 2.2 },
    { vals: (st.double_low || {}).nav_010, col: "#9a6700", w: 1.8 },
    { vals: (st.no_reset_lsm || {}).nav_010, col: "#1a7f37", w: 1.5 },
    { vals: (st.single_region_lsm || {}).nav_010, col: "#8250df", w: 1.5 },
    { vals: (st.mkt_ew || {}).nav_010, col: "#8c959f", w: 1.4, dash: "5 4" },
  ], 920, 240, 86);

  const goals = {
    four_region_lsm: "主策略（冻结原论文参数）",
    mkt_ew: "同范围基准：等权持有",
    double_low: "经典双低对照组",
    low_price: "低价因子对照",
    low_prem: "低转股溢价率对照",
    stock_mom: "正股 20 日动量对照",
    single_region_lsm: "消融对照：去四区域分段",
    no_reset_lsm: "消融对照：关闭下修博弈 (p=0)",
  };
  let rows = Object.keys(goals).map(sk => {
    const o = st[sk];
    if (!o) return "";
    const c = (o.costs || {})[curCost] || {};
    return `
      <tr class="${sk === "four_region_lsm" ? "hl-row" : ""}">
        <td><strong>${o.name}</strong></td>
        <td style="font-size:12px;color:var(--muted);">${goals[sk]}</td>
        <td class="num mono ${cls(c.total_ret)}">${signed(c.total_ret)}</td>
        <td class="num mono ${cls(c.ann_ret)}">${signed(c.ann_ret)}</td>
        <td class="num mono neg">${fmt(c.mdd)}%</td>
        <td class="num mono"><strong>${fmt(c.sharpe, 2)}</strong></td>
        <td class="num mono">${fmt(o.ann_to, 1)}%</td>
      </tr>
    `;
  }).join("");

  if (r.stress_default_50pct) {
    const sd = r.stress_default_50pct;
    rows += `<tr><td><strong>违约退市 50% 折价压力测试</strong></td><td style="font-size:12px;color:var(--muted);">末日收盘 &lt;¥90 退市券按 50% 清算 (0.10% 成本)</td><td class="num mono ${cls(sd.total_ret)}">${signed(sd.total_ret)}</td><td class="num mono ${cls(sd.ann_ret)}">${signed(sd.ann_ret)}</td><td class="num mono neg">${fmt(sd.mdd)}%</td><td class="num mono">${fmt(sd.sharpe, 2)}</td><td class="num mono">—</td></tr>`;
  }
  if (r.random_mc) {
    const mc = r.random_mc;
    rows += `<tr><td><strong>随机选券蒙特卡洛 (${mc.runs} 次)</strong></td><td style="font-size:12px;color:var(--muted);">同换手随机抽样 95% 区间 [${fmt(mc.total_ret_ci95[0])}%, ${fmt(mc.total_ret_ci95[1])}%]</td><td class="num mono">${signed(mc.total_ret_mean)}</td><td class="num mono">${signed(mc.ann_ret_mean)}</td><td class="num mono">—</td><td class="num mono">${fmt(mc.sharpe_mean, 2)}</td><td class="num mono">~750%</td></tr>`;
  }
  document.getElementById("btStratBody").innerHTML = rows;

  // Annual Returns
  const yLsm = calcYearlyPerf(first.nav_dates, first.nav_010);
  const yDl = calcYearlyPerf((st.double_low || {}).nav_dates, (st.double_low || {}).nav_010);
  const yMk = calcYearlyPerf((st.mkt_ew || {}).nav_dates, (st.mkt_ew || {}).nav_010);
  const yrs = Object.keys(yLsm).sort();
  document.getElementById("btAnnualBody").innerHTML = yrs.map(yr => `
    <tr>
      <td class="mono"><strong>${yr}${yr === yrs[yrs.length - 1] ? " YTD" : ""}</strong></td>
      <td class="num mono ${cls(yLsm[yr]?.ret)}">${signed(yLsm[yr]?.ret)}</td>
      <td class="num mono ${cls(yDl[yr]?.ret)}">${signed(yDl[yr]?.ret)}</td>
      <td class="num mono ${cls(yMk[yr]?.ret)}">${signed(yMk[yr]?.ret)}</td>
      <td class="num mono neg">${fmt(yLsm[yr]?.mdd)}%</td>
    </tr>
  `).join("");

  // Frequency & Subperiods
  const fr = r.freq_robust || {};
  document.getElementById("btFreqBody").innerHTML = ["1D", "5D", "10D", "20D"].map(fk => {
    const f = fr[fk] || {};
    return `<tr><td class="mono"><strong>${fk}</strong></td>
      <td class="num mono">${signed((f["0.05%"] || {}).total_ret)} <span class="sub">SR ${fmt((f["0.05%"] || {}).sharpe)}</span></td>
      <td class="num mono">${signed((f["0.10%"] || {}).total_ret)} <span class="sub">SR ${fmt((f["0.10%"] || {}).sharpe)}</span></td>
      <td class="num mono">${signed((f["0.20%"] || {}).total_ret)} <span class="sub">SR ${fmt((f["0.20%"] || {}).sharpe)}</span></td></tr>`;
  }).join("");

  const sp = r.subperiod_strats || {};
  document.getElementById("btSubBody").innerHTML = Object.keys(sp).map(p => {
    const s = sp[p];
    return `<tr><td><strong>${p}</strong></td>
      <td class="num mono ${cls(s.four_region_lsm?.total_ret)}">${signed(s.four_region_lsm?.total_ret)} (${fmt(s.four_region_lsm?.sharpe)})</td>
      <td class="num mono ${cls(s.double_low?.total_ret)}">${signed(s.double_low?.total_ret)}</td>
      <td class="num mono ${cls(s.mkt_ew?.total_ret)}">${signed(s.mkt_ew?.total_ret)}</td></tr>`;
  }).join("");
}

function renderResearch() {
  const r = R();
  const ic = r.ic_summary || {};
  document.getElementById("resIcBody").innerHTML = ["1", "5", "10", "20"].map(h => {
    const x = ic[h];
    if (!x) return "";
    const sig = x.p_val < 0.05;
    return `<tr>
      <td class="mono"><strong>${h} 日</strong></td><td class="num mono">${x.days}</td><td class="num mono">${fmt(x.avg_n, 1)}</td>
      <td class="num mono ${cls(x.mean_ic)}">${x.mean_ic > 0 ? "+" : ""}${fmt(x.mean_ic, 4)}</td>
      <td class="num mono">${fmt(x.std_ic, 4)}</td><td class="num mono">${fmt(x.icir, 3)}</td>
      <td class="num mono">${fmt(x.pos_pct, 1)}%</td>
      <td class="num mono"><strong>${fmt(x.t_nw, 2)}</strong> <span class="badge ${sig ? "b-sig-y" : "b-sig-n"}">p=${fmtP(x.p_val)}</span></td>
      <td class="num mono">[${fmt(x.ci95[0], 4)}, ${fmt(x.ci95[1], 4)}]</td>
    </tr>`;
  }).join("");

  // Rolling 60D IC SVG
  const pts = (r.rolling_ic || {})[curResIcH] || [];
  if (pts.length) {
    const W = 500, H = 200, pL = 40, pR = 14, pT = 16, pB = 26;
    const ys = pts.map(p => p[1]);
    const minY = Math.min(-0.08, ...ys) - 0.02, maxY = Math.max(0.18, ...ys) + 0.02;
    const px = i => pL + (i / Math.max(1, pts.length - 1)) * (W - pL - pR);
    const py = v => H - pB - ((v - minY) / (maxY - minY)) * (H - pT - pB);
    let g = "";
    [-0.05, 0, 0.05, 0.10].forEach(lv => {
      if (lv >= minY && lv <= maxY) {
        const y = py(lv).toFixed(1);
        g += `<line x1="${pL}" y1="${y}" x2="${W - pR}" y2="${y}" stroke="${lv === 0 ? "#8c959f" : "#eaeef2"}" stroke-dasharray="${lv === 0 ? "" : "3 3"}"/>
          <text x="${pL - 5}" y="${Number(y) + 3}" text-anchor="end" font-size="10" fill="#656d76">${lv.toFixed(2)}</text>`;
      }
    });
    [0, Math.floor(pts.length / 2), pts.length - 1].forEach(i => { g += `<text x="${px(i).toFixed(1)}" y="${H - 7}" text-anchor="middle" font-size="10" fill="#656d76">${pts[i][0]}</text>`; });
    const poly = pts.map((p, i) => `${px(i).toFixed(1)},${py(p[1]).toFixed(1)}`).join(" ");
    document.getElementById("resRollSvg").innerHTML = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block;">${g}<polyline fill="none" stroke="#0969da" stroke-width="1.9" points="${poly}"/></svg>`;
  } else {
    document.getElementById("resRollSvg").innerHTML = `<div style="padding:28px;text-align:center;color:var(--muted);">暂无滚动 IC 数据</div>`;
  }

  // IC Decay SVG
  const dec = r.ic_decay || [];
  if (dec.length) {
    const W = 500, H = 200, pL = 40, pR = 16, pT = 20, pB = 26;
    const maxV = Math.max(0.10, ...dec.map(d => d.mean_ic)) * 1.18;
    const bw = (W - pL - pR) / dec.length;
    let bars = `<line x1="${pL}" y1="${H - pB}" x2="${W - pR}" y2="${H - pB}" stroke="#d0d7de"/>`;
    dec.forEach((d, i) => {
      const v = Math.max(0, d.mean_ic);
      const h = (v / maxV) * (H - pT - pB), x = pL + i * bw + bw * 0.22, w = bw * 0.56, y = H - pB - h;
      bars += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${w.toFixed(1)}" height="${Math.max(2, h).toFixed(1)}" fill="#0969da" opacity="0.86" rx="2"/>
        <text x="${(x + w / 2).toFixed(1)}" y="${(y - 5).toFixed(1)}" text-anchor="middle" font-size="10" font-weight="600" fill="#0969da">${d.mean_ic >= 0 ? "+" : ""}${d.mean_ic.toFixed(3)}</text>
        <text x="${(x + w / 2).toFixed(1)}" y="${H - 8}" text-anchor="middle" font-size="10" fill="#656d76">${d.h}D</text>`;
    });
    document.getElementById("resDecaySvg").innerHTML = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block;">${bars}</svg>`;
  }

  // Quintiles
  const q = (r.quintiles || {})[curResQH];
  if (q) {
    document.getElementById("resQBarTitle").textContent = `五分组未来 ${curResQH} 日平均超额收益 (${q.days} 天)`;
    document.getElementById("resQStat").textContent = `G5 − G1 = ${signed(q.g5_minus_g1, 3)} (t_NW = ${fmt(q.t_nw, 2)}, p = ${fmtP(q.p_val)})`;
    const W = 500, H = 210, pL = 44, pR = 14, pT = 22, pB = 30;
    const vals = q.groups.map(g => g.excess_ret);
    const minY = Math.min(-0.1, ...vals) * 1.24, maxY = Math.max(0.1, ...vals) * 1.25;
    const zeroY = pT + (maxY / (maxY - minY)) * (H - pT - pB);
    const py = v => pT + ((maxY - v) / (maxY - minY)) * (H - pT - pB);
    const bw = (W - pL - pR) / 5;
    let bars = `<line x1="${pL}" y1="${zeroY.toFixed(1)}" x2="${W - pR}" y2="${zeroY.toFixed(1)}" stroke="#8c959f"/>`;
    q.groups.forEach((g, i) => {
      const v = g.excess_ret, x = pL + i * bw + bw * 0.2, w = bw * 0.6, vy = py(v), top = Math.min(vy, zeroY), bh = Math.max(2, Math.abs(vy - zeroY));
      const col = v >= 0 ? "#cf222e" : "#1a7f37", lblY = v >= 0 ? top - 5 : top + bh + 12;
      bars += `<rect x="${x.toFixed(1)}" y="${top.toFixed(1)}" width="${w.toFixed(1)}" height="${bh.toFixed(1)}" fill="${col}" opacity="0.85" rx="2"/>
        <text x="${(x + w / 2).toFixed(1)}" y="${lblY.toFixed(1)}" text-anchor="middle" font-size="10" font-weight="600" fill="${col}">${signed(v, 3)}</text>
        <text x="${(x + w / 2).toFixed(1)}" y="${H - 7}" text-anchor="middle" font-size="11" fill="#1f2328">${g.label}</text>`;
    });
    document.getElementById("resQBarSvg").innerHTML = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block;">${bars}</svg>`;
  }

  const qc = r.quintile_curves;
  if (qc && qc.dates) {
    const qCols = { G5: "#cf222e", G4: "#0969da", G3: "#656d76", G2: "#9a6700", G1: "#1a7f37" };
    document.getElementById("resQNavSvg").innerHTML = drawLinesSvg(
      qc.dates,
      ["G1", "G2", "G3", "G4", "G5"].map(gk => ({ vals: qc.navs[gk], col: qCols[gk], w: (gk === "G5" || gk === "G1") ? 2.1 : 1.4 })),
      500, 210, 74
    );
  }

  // Fama-MacBeth
  const fmb = (r.fama_macbeth || {})[curResFmbH];
  if (fmb) {
    document.getElementById("resFmbNote").textContent = `有效回归日 ${fmb.days} 天 · 日均 R² ${fmt(fmb.avg_r2, 2)}% · 已剔除含事后下调信息的评级控制变量`;
    document.getElementById("resFmbBody").innerHTML = fmb.factors.map(f => {
      const sig = f.p_val < 0.05;
      return `<tr class="${f.key === "f_four" ? "hl-row" : ""}">
        <td><strong>${f.label}</strong></td>
        <td class="num mono ${cls(f.mean)}">${signed(f.mean, 4, "")}</td>
        <td class="num mono">${fmt(f.se, 4)}</td>
        <td class="num mono"><strong>${f.t_stat > 0 ? "+" : ""}${fmt(f.t_stat, 2)}</strong></td>
        <td class="num mono">${fmtP(f.p_val)}</td>
        <td class="num mono">[${fmt(f.ci_low, 4)}, ${fmt(f.ci_high, 4)}]</td>
        <td><span class="badge ${sig ? "b-sig-y" : "b-sig-n"}">${sig ? "显著 p<0.05" : "不显著"}</span></td>
      </tr>`;
    }).join("");
  }

  // Audit & Point-in-Time Rating Coverage
  if (curPool === "full") {
    document.getElementById("resAuditHead").innerHTML = `<tr><th>审计指标</th><th>审计修正前</th><th>审计修正后</th><th>说明</th></tr>`;
    document.getElementById("resAuditBody").innerHTML = (r.audit_comparison || []).map(x => `
      <tr><td><strong>${x.metric}</strong></td><td class="mono" style="color:var(--muted);">${x.before}</td><td class="mono"><strong>${x.after}</strong></td><td style="font-size:12px;color:var(--muted);">${x.reason}</td></tr>
    `).join("");
  } else {
    const rf = r.rating_filter || {};
    const cov = rf.coverage_by_year || {};
    document.getElementById("resAuditHead").innerHTML = `<tr><th>点时评级校验项目</th><th>设定 / 口径</th><th>实测结果</th><th>防泄漏机制说明</th></tr>`;
    document.getElementById("resAuditBody").innerHTML = [
      ["纳入评级范围", (rf.eligible_grades || []).join(" / "), `${r.universe?.total_bonds || 0} 只 (${Number(r.universe?.total_bond_days || 0).toLocaleString()} 样本日)`, "严格按 T 日已公告最新债项评级筛选"],
      ["全样本评级覆盖率", "总体 & 分年度 ≥ 90%", `${((rf.rating_coverage_all_bond_days || 0) * 100).toFixed(2)}%`, Object.keys(cov).sort().map(y => `${y}:${(cov[y] * 100).toFixed(1)}%`).join(" · ")],
      ["时点滞后约束", "allow_exact_matches=False", "严格公告次日生效", "公告当日不可使用新评级买入，防止盘中/盘后公告泄漏"],
      ["数据源与脚本", "东方财富发行评级 + 巨潮公告PDF", "全部公开可复现", "tools/fetch_free_ratings.py + tools/run_high_grade_validation.py"],
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
document.querySelectorAll("#poolSeg .seg-btn").forEach(b => b.addEventListener("click", () => {
  curPool = b.dataset.pool;
  document.querySelectorAll("#poolSeg .seg-btn").forEach(x => x.classList.toggle("active", x === b));
  renderAll();
}));
document.querySelectorAll("#btCostPills .pill").forEach(b => b.addEventListener("click", () => {
  curCost = b.dataset.cost;
  document.querySelectorAll("#btCostPills .pill").forEach(x => x.classList.toggle("active", x === b));
  renderBacktest();
}));
document.querySelectorAll("#resIcPills .pill").forEach(b => b.addEventListener("click", () => {
  curResIcH = b.dataset.rich;
  document.querySelectorAll("#resIcPills .pill").forEach(x => x.classList.toggle("active", x === b));
  renderResearch();
}));
document.querySelectorAll("#resQPills .pill").forEach(b => b.addEventListener("click", () => {
  curResQH = b.dataset.rqh;
  document.querySelectorAll("#resQPills .pill").forEach(x => x.classList.toggle("active", x === b));
  renderResearch();
}));
document.querySelectorAll("#resFmbPills .pill").forEach(b => b.addEventListener("click", () => {
  curResFmbH = b.dataset.rfh;
  document.querySelectorAll("#resFmbPills .pill").forEach(x => x.classList.toggle("active", x === b));
  renderResearch();
}));
["scrSearch", "scrReg", "scrSort", "scrSafeOnly"].forEach(id => {
  const el = document.getElementById(id);
  el.addEventListener(id === "scrSearch" ? "input" : "change", renderScreener);
});

renderAll();
</script>
</body>
</html>
"""

    return template.replace("__PAYLOAD_JSON__", payload_json).replace("__UPDATED_AT__", data["updated_at"])


if __name__ == "__main__":
    generate_page_and_data()
