import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pytz

try:
    from cb_pricing_model import price_ccb_four_region_lsm
except ImportError:
    from .cb_pricing_model import price_ccb_four_region_lsm


def get_default_bonds_pool():
    """Fallback representative universe if live network feeds are unreachable."""
    return [
        {"ticker": "110075", "name": "南航转债", "stock_code": "600029", "stock_name": "南方航空", "stock_price": 5.92, "conv_price": 6.24, "bond_price": 128.40, "maturity_years": 2.8},
        {"ticker": "113024", "name": "核建转债", "stock_code": "601611", "stock_name": "中国核建", "stock_price": 8.35, "conv_price": 9.15, "bond_price": 117.80, "maturity_years": 2.1},
        {"ticker": "127025", "name": "冀东转债", "stock_code": "000401", "stock_name": "冀东水泥", "stock_price": 5.12, "conv_price": 5.88, "bond_price": 111.20, "maturity_years": 1.9},
        {"ticker": "113049", "name": "长汽转债", "stock_code": "601633", "stock_name": "长城汽车", "stock_price": 26.80, "conv_price": 34.20, "bond_price": 114.50, "maturity_years": 3.2},
        {"ticker": "123107", "name": "温氏转债", "stock_code": "300498", "stock_name": "温氏股份", "stock_price": 17.50, "conv_price": 19.80, "bond_price": 119.30, "maturity_years": 2.6},
        {"ticker": "127056", "name": "中特转债", "stock_code": "000708", "stock_name": "中信特钢", "stock_price": 12.10, "conv_price": 14.80, "bond_price": 113.60, "maturity_years": 3.5},
        {"ticker": "110059", "name": "浦发转债", "stock_code": "600000", "stock_name": "浦发银行", "stock_price": 9.65, "conv_price": 12.80, "bond_price": 106.80, "maturity_years": 1.5},
        {"ticker": "113052", "name": "兴业转债", "stock_code": "601166", "stock_name": "兴业银行", "stock_price": 18.20, "conv_price": 24.50, "bond_price": 107.40, "maturity_years": 2.9},
        {"ticker": "110083", "name": "苏租转债", "stock_code": "600901", "stock_name": "江苏租赁", "stock_price": 4.88, "conv_price": 5.25, "bond_price": 126.10, "maturity_years": 3.1},
        {"ticker": "113043", "name": "财通转债", "stock_code": "601108", "stock_name": "财通证券", "stock_price": 7.42, "conv_price": 8.65, "bond_price": 112.50, "maturity_years": 2.3},
    ]


def classify_region(S0: float, C0: float):
    k0 = 1.30 * C0
    ptr0 = 0.70 * C0
    if S0 > k0:
        return "强赎提示区", "R1", "正股超过 130% 强赎线"
    if S0 > C0:
        return "高弹性转股区", "R2", "正股高于转股价，偏股弹性"
    if S0 > ptr0:
        return "平衡抗跌区", "R3", "纯债底与期权兼备"
    return "下修博弈区", "R4", "低于 70% 回售线，博弈下修"


def fetch_active_pool(max_bonds: int = 36):
    """Fetch active, liquid, already-listed convertible bonds (no unlisted 100.00 bonds)."""
    import akshare as ak

    pool = {}
    now_ts = pd.Timestamp.now()

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
                if rem < 0.3 or price < 85 or price > 260 or c_price <= 0 or s_price <= 0:
                    continue
                pool[code] = {
                    "ticker": code,
                    "name": str(r.get("转债名称", "")),
                    "stock_code": str(r.get("正股代码", "")),
                    "stock_name": str(r.get("正股名称", "")),
                    "stock_price": s_price,
                    "conv_price": c_price,
                    "bond_price": price,
                    "maturity_years": round(rem, 2),
                }
    except Exception as e:
        print(f"Notice: bond_cb_jsl fetch warning: {e}")

    try:
        df_em = ak.bond_cb_list_em()
        if df_em is not None and not df_em.empty:
            df_em["上市时间"] = pd.to_datetime(df_em["上市时间"], errors="coerce")
            df_em["债现价"] = pd.to_numeric(df_em["债现价"], errors="coerce")
            df_em["转股价"] = pd.to_numeric(df_em["转股价"], errors="coerce")
            df_em["正股价"] = pd.to_numeric(df_em["正股价"], errors="coerce")
            # Filter strictly to already listed bonds with valid trading prices
            listed = df_em[
                df_em["上市时间"].notna()
                & (df_em["上市时间"] <= now_ts - pd.Timedelta(days=120))
                & (df_em["上市时间"] >= now_ts - pd.Timedelta(days=1100))
                & (df_em["债现价"] >= 85)
                & (df_em["债现价"] <= 250)
                & (df_em["债现价"] != 100.0)
                & (df_em["转股价"] > 0)
                & (df_em["正股价"] > 0)
            ]
            for _, r in listed.iterrows():
                if len(pool) >= max_bonds:
                    break
                code = str(r.get("债券代码", "")).strip()
                if not code or code in pool or code.startswith("132"):
                    continue
                elapsed = (now_ts - r["上市时间"]).days / 365.25
                rem_years = max(0.5, round(6.0 - elapsed, 2))
                pool[code] = {
                    "ticker": code,
                    "name": str(r.get("债券简称", "")),
                    "stock_code": str(r.get("正股代码", "")),
                    "stock_name": str(r.get("正股简称", "")),
                    "stock_price": float(r["正股价"]),
                    "conv_price": float(r["转股价"]),
                    "bond_price": float(r["债现价"]),
                    "maturity_years": rem_years,
                }
    except Exception as e:
        print(f"Notice: bond_cb_list_em fetch warning: {e}")

    if not pool:
        return get_default_bonds_pool()
    return list(pool.values())[:max_bonds]


def fetch_pair_history(item):
    """Fetch aligned daily bond + stock history for a single convertible bond."""
    import akshare as ak

    b_code = item["ticker"]
    s_code = item["stock_code"]
    b_prefix = "sh" if b_code.startswith(("11", "13")) else "sz"
    s_prefix = "sh" if s_code.startswith("6") else "sz"

    try:
        b_hist = ak.bond_cb_daily_sina(symbol=b_prefix + b_code)
        s_hist = ak.stock_zh_a_hist_sina(symbol=s_prefix + s_code)
        if b_hist is None or b_hist.empty or s_hist is None or s_hist.empty:
            return item, None
        b_hist["date"] = pd.to_datetime(b_hist["date"])
        s_hist["date"] = pd.to_datetime(s_hist["date"])
        merged = (
            pd.merge(
                b_hist[["date", "close", "volume"]],
                s_hist[["date", "close"]],
                on="date",
                suffixes=("_bond", "_stock"),
            )
            .sort_values("date")
            .reset_index(drop=True)
        )
        if len(merged) < 40:
            return item, None
        # Ensure bond is actively trading recently (not delisted)
        last_date = merged["date"].iloc[-1]
        if (pd.Timestamp.now() - last_date).days > 14:
            return item, None
        return item, merged
    except Exception:
        return item, None


def estimate_sigma(stock_closes: pd.Series, end_idx: int, window: int = 60) -> float:
    start_idx = max(1, end_idx - window)
    sub = stock_closes.iloc[start_idx - 1 : end_idx + 1]
    if len(sub) < 15:
        return 0.28
    log_ret = np.log(sub / sub.shift(1)).dropna()
    if len(log_ret) < 10:
        return 0.28
    vol = float(log_ret.std() * np.sqrt(252))
    return round(max(0.15, min(0.65, vol)), 4)


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 4 or x.nunique() < 2 or y.nunique() < 2:
        return 0.0
    val = x.rank().corr(y.rank())
    return round(float(val), 3) if pd.notna(val) else 0.0


def evaluate_horizon(histories: list, horizon_days: int):
    """Run point-in-time valuation at T - horizon_days and validate against real market outcomes at T."""
    rows = []
    for item, merged in histories:
        if merged is None or len(merged) <= horizon_days + 25:
            continue
        idx_past = len(merged) - 1 - horizon_days
        sigma_past = estimate_sigma(merged["close_stock"], idx_past, window=60)
        S_past = float(merged.iloc[idx_past]["close_stock"])
        P_past = float(merged.iloc[idx_past]["close_bond"])
        P_now = float(merged.iloc[-1]["close_bond"])
        S_now = float(merged.iloc[-1]["close_stock"])
        d_past = merged.iloc[idx_past]["date"].strftime("%Y-%m-%d")
        d_now = merged.iloc[-1]["date"].strftime("%Y-%m-%d")
        C0 = item["conv_price"]
        mat_past = max(0.35, item["maturity_years"] + horizon_days / 252.0)

        P_mdl_past = price_ccb_four_region_lsm(
            S0=S_past, C0=C0, FV=100.0, maturity_years=mat_past,
            r=0.025, q=0.0, sigma=sigma_past,
            call_ratio=1.30, put_ratio=0.70, p_reset=0.80,
            B=108.0, n_steps=24, M=1200, seed=42
        )
        underval_past = round((P_mdl_past - P_past) / P_past * 100, 2)
        actual_ret = round((P_now - P_past) / P_past * 100, 2)
        stock_ret = round((S_now - S_past) / S_past * 100, 2)
        conv_val_past = (100.0 / C0) * S_past
        prem_past = (P_past - conv_val_past) / max(0.01, conv_val_past) * 100
        dl_past = round(P_past + prem_past, 2)
        gap_before = abs(P_past - P_mdl_past)
        gap_after = abs(P_now - P_mdl_past)
        converged = gap_after < gap_before

        reg_name, reg_code, _ = classify_region(S_past, C0)
        rows.append({
            "ticker": item["ticker"],
            "name": item["name"],
            "stock_name": item["stock_name"],
            "past_date": d_past,
            "now_date": d_now,
            "past_price": round(P_past, 2),
            "past_model": round(P_mdl_past, 2),
            "underval_past": underval_past,
            "now_price": round(P_now, 2),
            "actual_ret": actual_ret,
            "stock_ret": stock_ret,
            "double_low_past": dl_past,
            "region_code": reg_code,
            "region_name": reg_name,
            "converged": bool(converged),
        })

    if len(rows) < 6:
        return None

    df = pd.DataFrame(rows).sort_values("underval_past", ascending=False).reset_index(drop=True)
    n = len(df)
    k = max(2, n // 3)
    top_df = df.iloc[:k]
    mid_df = df.iloc[k : n - k] if n - 2 * k > 0 else df.iloc[k:k]
    bot_df = df.iloc[n - k :]
    dl_df = df.sort_values("double_low_past", ascending=True).iloc[:k]

    bench_avg = float(df["actual_ret"].mean())
    bench_med = float(df["actual_ret"].median())
    top_avg = float(top_df["actual_ret"].mean())
    mid_avg = float(mid_df["actual_ret"].mean()) if len(mid_df) > 0 else bench_avg
    bot_avg = float(bot_df["actual_ret"].mean())
    dl_avg = float(dl_df["actual_ret"].mean())
    spread = round(top_avg - bot_avg, 2)
    alpha_vs_mkt = round(top_avg - bench_avg, 2)
    alpha_vs_dl = round(top_avg - dl_avg, 2)
    rank_ic = spearman_corr(df["underval_past"], df["actual_ret"])
    conv_rate = round(float(df["converged"].mean()) * 100, 1)

    # Relative directional hit rate: top half should >= median, bottom half should <= median
    half = n // 2
    hits = []
    for idx, r in df.iterrows():
        rel_alpha = round(float(r["actual_ret"] - bench_med), 2)
        if idx < half:
            hit = float(r["actual_ret"]) >= bench_med
            expect = "看多 / 相对低估"
        else:
            hit = float(r["actual_ret"]) <= bench_med
            expect = "偏高估 / 相对谨慎"
        hits.append((hit, rel_alpha, expect))

    df["hit"] = [h[0] for h in hits]
    df["excess_vs_med"] = [h[1] for h in hits]
    df["expect_label"] = [h[2] for h in hits]
    hit_rate = round(float(df["hit"].mean()) * 100, 1)
    top_beat_rate = round(float((top_df["actual_ret"] >= bench_med).mean()) * 100, 1)

    return {
        "horizon": horizon_days,
        "sample_size": n,
        "past_date": str(df["past_date"].mode().iloc[0]),
        "now_date": str(df["now_date"].mode().iloc[0]),
        "summary": {
            "top_avg_ret": round(top_avg, 2),
            "mid_avg_ret": round(mid_avg, 2),
            "bot_avg_ret": round(bot_avg, 2),
            "mkt_avg_ret": round(bench_avg, 2),
            "mkt_med_ret": round(bench_med, 2),
            "dl_avg_ret": round(dl_avg, 2),
            "long_short_spread": spread,
            "alpha_vs_mkt": alpha_vs_mkt,
            "alpha_vs_dl": alpha_vs_dl,
            "hit_rate": hit_rate,
            "top_beat_rate": top_beat_rate,
            "rank_ic": rank_ic,
            "convergence_rate": conv_rate,
        },
        "rows": df.to_dict(orient="records"),
    }


def update_live_history(dist_dir: Path, date_str: str, evaluated: list):
    """Persist daily live snapshots to history.json and evaluate out-of-sample forward accuracy."""
    hist_path = dist_dir / "history.json"
    ledger = {"snapshots": []}
    if hist_path.exists():
        try:
            ledger = json.loads(hist_path.read_text(encoding="utf-8"))
        except Exception:
            ledger = {"snapshots": []}

    snap_entry = {
        "date": date_str,
        "bonds": {
            b["ticker"]: {
                "name": b["name"],
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
    snapshots = sorted(snapshots, key=lambda x: x["date"])[-180:]
    ledger["snapshots"] = snapshots
    hist_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")

    # Compute live forward comparison if we have prior live days
    current_prices = {b["ticker"]: b["bond_price"] for b in evaluated}
    live_records = []
    for past_snap in snapshots[:-1]:
        p_date = past_snap["date"]
        matched = []
        for ticker, info in past_snap["bonds"].items():
            if ticker in current_prices and info["bond_price"] > 0:
                ret = round((current_prices[ticker] - info["bond_price"]) / info["bond_price"] * 100, 2)
                matched.append({
                    "ticker": ticker,
                    "name": info["name"],
                    "underval": info["undervaluation"],
                    "past_price": info["bond_price"],
                    "past_model": info["model_price"],
                    "now_price": current_prices[ticker],
                    "ret": ret,
                })
        if len(matched) >= 6:
            mdf = pd.DataFrame(matched).sort_values("underval", ascending=False)
            k = max(2, len(mdf) // 3)
            top_r = round(float(mdf.iloc[:k]["ret"].mean()), 2)
            bot_r = round(float(mdf.iloc[-k:]["ret"].mean()), 2)
            mkt_r = round(float(mdf["ret"].mean()), 2)
            ic = spearman_corr(mdf["underval"], mdf["ret"])
            live_records.append({
                "from_date": p_date,
                "to_date": date_str,
                "count": len(mdf),
                "top_ret": top_r,
                "bot_ret": bot_r,
                "mkt_ret": mkt_r,
                "spread": round(top_r - bot_r, 2),
                "rank_ic": ic,
            })

    return {"tracked_days": len(snapshots), "comparisons": list(reversed(live_records))[:15]}


def load_and_evaluate():
    pool = fetch_active_pool(max_bonds=32)
    histories = []

    for item in pool:
        histories.append(fetch_pair_history(item))

    # Preserve deterministic order & drop delisted items if history check failed due to staleness
    valid_histories = [(item, df) for item, df in histories if df is not None]
    use_list = valid_histories if len(valid_histories) >= 10 else [(item, None) for item in pool]

    evaluated = []
    for item, merged in use_list:
        S0 = float(merged.iloc[-1]["close_stock"]) if merged is not None else item["stock_price"]
        Pmkt = float(merged.iloc[-1]["close_bond"]) if merged is not None else item["bond_price"]
        C0 = item["conv_price"]
        mat = item.get("maturity_years", 2.5)
        sigma = estimate_sigma(merged["close_stock"], len(merged) - 1, window=60) if merged is not None else 0.28

        conv_val = round((100.0 / C0) * S0, 2)
        premium_rate = round(((Pmkt - conv_val) / max(0.01, conv_val)) * 100, 2)
        double_low = round(Pmkt + premium_rate, 2)
        region_name, region_code, region_desc = classify_region(S0, C0)

        Pmdl = price_ccb_four_region_lsm(
            S0=S0, C0=C0, FV=100.0, maturity_years=mat,
            r=0.025, q=0.0, sigma=sigma,
            call_ratio=1.30, put_ratio=0.70, p_reset=0.80,
            B=108.0, n_steps=24, M=1600, seed=42
        )
        undervaluation = round(((Pmdl - Pmkt) / Pmkt) * 100, 2)

        evaluated.append({
            "ticker": item["ticker"],
            "name": item["name"],
            "stock_code": item["stock_code"],
            "stock_name": item["stock_name"],
            "stock_price": round(S0, 2),
            "conv_price": round(C0, 2),
            "bond_price": round(Pmkt, 2),
            "model_price": round(Pmdl, 2),
            "undervaluation": undervaluation,
            "double_low": double_low,
            "conv_val": conv_val,
            "premium_rate": premium_rate,
            "sigma": round(sigma * 100, 1),
            "region_name": region_name,
            "region_code": region_code,
            "region_desc": region_desc,
            "maturity_years": mat,
        })

    # Compute rolling historical validation across 20d, 10d, 5d
    validation = {}
    for h in (20, 10, 5):
        res = evaluate_horizon(valid_histories, h)
        if res:
            validation[str(h)] = res

    return evaluated, validation


def generate_page_and_data():
    local_tz = pytz.timezone("Asia/Shanghai")
    now = datetime.now(local_tz)
    update_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    date_str = now.strftime("%Y-%m-%d")

    evaluated, validation = load_and_evaluate()

    lsm_sorted = sorted(evaluated, key=lambda x: x["undervaluation"], reverse=True)
    dl_sorted = sorted(evaluated, key=lambda x: x["double_low"], reverse=False)
    for idx, item in enumerate(lsm_sorted, 1):
        item["lsm_rank"] = idx
    for idx, item in enumerate(dl_sorted, 1):
        item["dl_rank"] = idx

    dist_dir = Path(__file__).resolve().parent / "dist" / "bonds"
    dist_dir.mkdir(parents=True, exist_ok=True)
    live_ledger = update_live_history(dist_dir, date_str, lsm_sorted)

    data_payload = {
        "updated_at": update_time_str,
        "date": date_str,
        "author": "Yu Liu (刘宇) · The Chinese University of Hong Kong, Shenzhen",
        "paper_ref": "arXiv:2409.06496v7",
        "paper_url": "https://arxiv.org/abs/2409.06496",
        "pool_size": len(lsm_sorted),
        "parameters": {
            "M": 1600,
            "n_steps": 24,
            "r": 0.025,
            "p_reset": 0.80,
            "call_trigger": "130% / 15-30",
            "put_trigger": "70% / 30-30",
        },
        "validation": validation,
        "live_ledger": live_ledger,
        "all_bonds": lsm_sorted,
    }

    (dist_dir / "data.json").write_text(json.dumps(data_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (dist_dir / "index.html").write_text(build_clean_html(data_payload), encoding="utf-8")
    print(f"Generated accuracy-focused bonds dashboard at {dist_dir / 'index.html'} (pool={len(lsm_sorted)}, horizons={list(validation.keys())})")


def build_clean_html(data):
    payload_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>可转债定价检验 · Yu Liu</title>
  <meta name="description" content="基于 Four-Region LSM 模型的可转债定价准确性检验与实盘追踪">
  <style>
    :root {{
      --bg: #ffffff;
      --sub: #f8f9fb;
      --ink: #1f252e;
      --muted: #687382;
      --line: #e4e8ee;
      --blue: #2358a6;
      --blue-bg: #eef4fc;
      --up: #c43737;
      --up-bg: #fdf2f2;
      --down: #187a62;
      --down-bg: #f1faf7;
      color-scheme: light;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      font-size: 14px;
      line-height: 1.65;
      -webkit-font-smoothing: antialiased;
    }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-variant-numeric: tabular-nums; }}
    a {{ color: var(--blue); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}

    .shell {{ max-width: 1100px; margin: 0 auto; padding: 32px 24px 64px; }}

    .topbar {{
      display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 12px;
      padding-bottom: 16px; border-bottom: 1px solid var(--line); margin-bottom: 28px; font-size: 13px;
    }}
    .brand {{ font-weight: 650; color: var(--ink); font-size: 15px; }}
    .brand span {{ color: var(--muted); font-weight: 400; margin-left: 8px; font-size: 13px; }}
    .nav-links {{ display: flex; gap: 20px; }}
    .nav-links a {{ color: var(--muted); }}
    .nav-links a:hover {{ color: var(--blue); }}

    .head-block {{ display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 16px; margin-bottom: 24px; }}
    h1 {{ font-size: 23px; font-weight: 650; letter-spacing: -0.3px; margin: 0 0 6px; }}
    .lead {{ color: var(--muted); font-size: 13px; margin: 0; max-width: 720px; }}
    .meta-stamp {{ font-size: 12px; color: var(--muted); white-space: nowrap; }}

    .kpi-strip {{
      display: grid; grid-template-columns: repeat(4, 1fr);
      border: 1px solid var(--line); background: var(--sub); margin-bottom: 24px;
    }}
    .kpi {{ padding: 16px 18px; border-right: 1px solid var(--line); }}
    .kpi:last-child {{ border-right: 0; }}
    .kpi-label {{ display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }}
    .kpi-val {{ font-size: 22px; font-weight: 650; line-height: 1.25; }}
    .kpi-sub {{ font-size: 11px; color: var(--muted); margin-top: 4px; }}

    .tabs-bar {{
      display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;
      border-bottom: 1px solid var(--line); margin-bottom: 22px;
    }}
    .main-tabs {{ display: flex; gap: 24px; }}
    .main-tabs button {{
      border: 0; background: none; padding: 10px 2px; font-size: 14px; color: var(--muted);
      cursor: pointer; border-bottom: 2px solid transparent; margin-bottom: -1px; font-weight: 500;
    }}
    .main-tabs button.active {{ color: var(--ink); font-weight: 650; border-bottom-color: var(--blue); }}

    .sub-controls {{ display: flex; gap: 8px; padding-bottom: 6px; flex-wrap: wrap; }}
    .pill-btn {{
      border: 1px solid var(--line); background: #fff; color: var(--muted);
      padding: 4px 11px; font-size: 12px; border-radius: 3px; cursor: pointer;
    }}
    .pill-btn.active {{ background: var(--blue-bg); border-color: var(--blue); color: var(--blue); font-weight: 600; }}

    .charts-grid {{
      display: grid; grid-template-columns: 1fr 1.15fr; gap: 18px; margin-bottom: 24px;
    }}
    .panel {{ border: 1px solid var(--line); background: #fff; padding: 18px 20px; }}
    .panel-head {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 12px; }}
    .panel-head h3 {{ font-size: 14px; font-weight: 600; margin: 0; }}
    .panel-head span {{ font-size: 11px; color: var(--muted); }}

    .table-box {{ border: 1px solid var(--line); background: #fff; overflow-x: auto; }}
    .table-toolbar {{
      display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;
      padding: 12px 16px; border-bottom: 1px solid var(--line); background: var(--sub);
    }}
    .table-toolbar strong {{ font-size: 13px; }}
    .filter-inputs {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
    .filter-inputs input, .filter-inputs select {{
      border: 1px solid var(--line); background: #fff; padding: 5px 9px; font-size: 12px; border-radius: 3px; color: var(--ink);
    }}

    table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 13px; }}
    th {{
      padding: 10px 14px; font-size: 11px; font-weight: 600; color: var(--muted);
      border-bottom: 1px solid var(--line); background: #fcfcfd; white-space: nowrap;
    }}
    td {{ padding: 11px 14px; border-bottom: 1px solid var(--line); vertical-align: middle; }}
    tr:last-child td {{ border-bottom: 0; }}
    tr:hover td {{ background: #f9fafc; }}

    .up {{ color: var(--up); }}
    .down {{ color: var(--down); }}
    .muted {{ color: var(--muted); }}
    .tag {{
      display: inline-block; padding: 2px 6px; font-size: 11px; border-radius: 2px; font-weight: 500; white-space: nowrap;
    }}
    .tag-hit {{ background: var(--down-bg); color: var(--down); border: 1px solid #ccebe2; }}
    .tag-miss {{ background: #f3f4f6; color: var(--muted); border: 1px solid var(--line); }}
    .tag-up {{ background: var(--up-bg); color: var(--up); }}
    .tag-down {{ background: var(--down-bg); color: var(--down); }}
    .tag-reg {{ background: var(--sub); color: var(--ink); border: 1px solid var(--line); }}

    .calc-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; border: 1px solid var(--line); padding: 22px; background: var(--sub); }}
    .slider-row {{ margin-bottom: 14px; }}
    .slider-head {{ display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 5px; }}
    input[type=range] {{ width: 100%; accent-color: var(--blue); }}
    .calc-out {{ background: #fff; border: 1px solid var(--line); padding: 18px; }}
    .calc-nums {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding-bottom: 14px; border-bottom: 1px solid var(--line); margin-bottom: 12px; }}
    .calc-nums strong {{ display: block; font-size: 24px; margin-top: 2px; }}

    svg text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .scatter-tip {{
      margin-top: 8px; font-size: 12px; color: var(--muted); min-height: 18px;
    }}

    footer {{
      margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--line);
      font-size: 12px; color: var(--muted); line-height: 1.8;
    }}

    @media (max-width: 820px) {{
      .kpi-strip {{ grid-template-columns: 1fr 1fr; }}
      .kpi:nth-child(2) {{ border-right: 0; }}
      .kpi:nth-child(1), .kpi:nth-child(2) {{ border-bottom: 1px solid var(--line); }}
      .charts-grid, .calc-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
<div class="shell">
  <header class="topbar">
    <div class="brand">Yu Liu · 刘宇 <span>可转债定价检验</span></div>
    <nav class="nav-links">
      <a href="/">个人主页</a>
      <a href="/trade/">A 股买卖看板</a>
      <a href="/phd-tracker/">申请追踪</a>
      <a href="https://arxiv.org/abs/2409.06496" target="_blank" rel="noopener">arXiv:2409.06496 ↗</a>
    </nav>
  </header>

  <div class="head-block">
    <div>
      <h1>可转债模型定价有效性检验</h1>
      <p class="lead">基于 Four-Region LSM 定价模型（arXiv:2409.06496v7），使用上市转债与正股真实历史日线进行样本外验证，检验“低估判断”在随后市场走势中的兑现度与超额表现。</p>
    </div>
    <div class="meta-stamp mono" id="metaStamp">更新于 {data['updated_at']}</div>
  </div>

  <section class="kpi-strip" id="kpiStrip"></section>

  <div class="tabs-bar">
    <div class="main-tabs">
      <button class="active" data-tab="accuracy">定价准确性检验</button>
      <button data-tab="live">当期实时估值 ({data['pool_size']})</button>
      <button data-tab="calc">单券参数试算</button>
    </div>
    <div class="sub-controls" id="horizonControls">
      <button class="pill-btn active" data-horizon="20">近 20 交易日</button>
      <button class="pill-btn" data-horizon="10">近 10 交易日</button>
      <button class="pill-btn" data-horizon="5">近 5 交易日</button>
      <button class="pill-btn" data-horizon="ledger">实盘逐日追踪</button>
    </div>
  </div>

  <section id="tab-accuracy">
    <div id="accuracyBody"></div>
  </section>

  <section id="tab-live" hidden>
    <div class="table-box">
      <div class="table-toolbar">
        <strong>上市可转债实时模型定价表（剔除未上市新债）</strong>
        <div class="filter-inputs">
          <input type="search" id="liveSearch" placeholder="代码 / 转债 / 正股">
          <select id="liveRegion">
            <option value="all">全部条款区间</option>
            <option value="R3">R3 平衡抗跌区</option>
            <option value="R2">R2 高弹性转股区</option>
            <option value="R1">R1 强赎提示区</option>
            <option value="R4">R4 下修博弈区</option>
          </select>
          <select id="liveSort">
            <option value="underval">按低估空间排序</option>
            <option value="dl">按传统双低排序</option>
          </select>
        </div>
      </div>
      <table>
        <thead>
          <tr>
            <th>排名</th>
            <th>转债 / 正股</th>
            <th>市价</th>
            <th>模型理论价</th>
            <th>低估空间</th>
            <th>条款区间</th>
            <th>溢价率</th>
            <th>双低值</th>
            <th>波动率 / 年限</th>
          </tr>
        </thead>
        <tbody id="liveTableBody"></tbody>
      </table>
    </div>
  </section>

  <section id="tab-calc" hidden>
    <div class="calc-grid">
      <div>
        <div class="slider-row">
          <div class="slider-head"><span>正股现价 S₀</span><strong class="mono" id="txt-s0">¥10.00</strong></div>
          <input type="range" id="slider-s0" min="5" max="25" step="0.1" value="10.0">
        </div>
        <div class="slider-row">
          <div class="slider-head"><span>转股价 C₀</span><strong class="mono" id="txt-c0">¥10.00</strong></div>
          <input type="range" id="slider-c0" min="5" max="25" step="0.1" value="10.0">
        </div>
        <div class="slider-row">
          <div class="slider-head"><span>发行人下修意愿 p</span><strong class="mono" id="txt-p">80%</strong></div>
          <input type="range" id="slider-p" min="0" max="1" step="0.05" value="0.80">
        </div>
        <div class="slider-row">
          <div class="slider-head"><span>正股年化波动率 σ</span><strong class="mono" id="txt-sig">28%</strong></div>
          <input type="range" id="slider-sig" min="0.12" max="0.55" step="0.02" value="0.28">
        </div>
      </div>
      <div class="calc-out">
        <div class="calc-nums">
          <div><span class="muted">模型理论估值</span><strong class="mono up" id="res-mdl">¥120.45</strong></div>
          <div><span class="muted">转股平价</span><strong class="mono" id="res-conv">¥100.00</strong></div>
        </div>
        <div>
          <strong id="res-reg-title" style="display:block; margin-bottom:4px;">平衡抗跌区 (R3)</strong>
          <span id="res-reg-text" class="muted" style="font-size:13px;"></span>
        </div>
      </div>
    </div>
  </section>

  <footer>
    方法来源：Liu, Yu. (2026). <em>Valuation Model of Chinese Convertible Bonds Based on Monte Carlo Simulation: A Regression-Based Approximation and Error Analysis</em> (<a href="https://arxiv.org/abs/2409.06496" target="_blank" rel="noopener">arXiv:2409.06496v7</a>).<br>
    检验说明：历史窗口回测使用当日正股收盘价、前 60 日实际滚动波动率与当时剩余期限计算理论底价，不包含未来信息；不作为直接投资建议。
  </footer>
</div>

<script>
const DATA = {payload_json};
let currentTab = "accuracy";
let currentHorizon = "20";
if (!DATA.validation["20"]) {{
  currentHorizon = Object.keys(DATA.validation)[0] || "ledger";
}}

const fmt = (v, n = 2) => typeof v === "number" && Number.isFinite(v) ? v.toFixed(n) : "—";
const signed = (v, n = 2) => typeof v === "number" && Number.isFinite(v) ? (v > 0 ? "+" : "") + v.toFixed(n) + "%" : "—";
const cls = v => v > 0 ? "up" : v < 0 ? "down" : "muted";

function renderKPI() {{
  const v = DATA.validation[currentHorizon] || DATA.validation["20"] || Object.values(DATA.validation)[0];
  if (!v) {{
    document.getElementById("kpiStrip").innerHTML = `<div class="kpi"><span class="kpi-label">追踪状态</span><div class="kpi-val">等待更多历史数据</div></div>`;
    return;
  }}
  const s = v.summary;
  document.getElementById("kpiStrip").innerHTML = `
    <div class="kpi">
      <span class="kpi-label">多空超额 (${{v.horizon}} 日 Top vs 高估组)</span>
      <div class="kpi-val mono ${{cls(s.long_short_spread)}}">${{signed(s.long_short_spread)}}</div>
      <div class="kpi-sub">低估组 ${{signed(s.top_avg_ret)}} · 高估组 ${{signed(s.bot_avg_ret)}}</div>
    </div>
    <div class="kpi">
      <span class="kpi-label">相对方向命中率</span>
      <div class="kpi-val mono">${{fmt(s.hit_rate, 1)}}%</div>
      <div class="kpi-sub">低估组跑赢中位数比例 ${{fmt(s.top_beat_rate, 1)}}%</div>
    </div>
    <div class="kpi">
      <span class="kpi-label">排序相关性 (Rank IC)</span>
      <div class="kpi-val mono ${{cls(s.rank_ic)}}">${{s.rank_ic > 0 ? "+" : ""}}${{fmt(s.rank_ic, 3)}}</div>
      <div class="kpi-sub">价差收敛率 ${{fmt(s.convergence_rate, 1)}}%</div>
    </div>
    <div class="kpi">
      <span class="kpi-label">vs 传统双低组合超额</span>
      <div class="kpi-val mono ${{cls(s.alpha_vs_dl)}}">${{signed(s.alpha_vs_dl)}}</div>
      <div class="kpi-sub">双低组均值 ${{signed(s.dl_avg_ret)}} · 样本 ${{v.sample_size}} 只</div>
    </div>
  `;
}}

function renderBarsSVG(s) {{
  const items = [
    {{ label: "模型低估组 (Top 1/3)", val: s.top_avg_ret, hl: true }},
    {{ label: "中间组", val: s.mid_avg_ret }},
    {{ label: "高估警戒组 (Bottom 1/3)", val: s.bot_avg_ret }},
    {{ label: "传统双低组 (对照)", val: s.dl_avg_ret }},
    {{ label: "全市场样本均价", val: s.mkt_avg_ret }},
  ];
  const maxAbs = Math.max(2, ...items.map(x => Math.abs(x.val)));
  const W = 440, H = 200, midX = 250, barMaxW = 145;
  let rows = "";
  items.forEach((it, idx) => {{
    const y = 18 + idx * 36;
    const w = Math.abs(it.val) / maxAbs * barMaxW;
    const x = it.val >= 0 ? midX : midX - w;
    const color = it.val >= 0 ? "#c43737" : "#187a62";
    const opacity = it.hl ? "0.95" : "0.65";
    rows += `
      <text x="8" y="${{y + 15}}" font-size="12" fill="${{it.hl ? "#1f252e" : "#687382"}}" font-weight="${{it.hl ? "600" : "400"}}">${{it.label}}</text>
      <rect x="${{x}}" y="${{y + 2}}" width="${{Math.max(2, w)}}" height="18" fill="${{color}}" opacity="${{opacity}}" rx="2"></rect>
      <text x="${{it.val >= 0 ? x + w + 6 : x - 6}}" y="${{y + 15}}" font-size="11" font-weight="600" fill="${{color}}" text-anchor="${{it.val >= 0 ? "start" : "end"}}">${{signed(it.val)}}</text>
    `;
  }});
  return `<svg viewBox="0 0 ${{W}} ${{H}}" style="width:100%;height:auto;display:block;">
    <line x1="${{midX}}" y1="8" x2="${{midX}}" y2="${{H - 8}}" stroke="#cbd5e1" stroke-width="1"/>
    ${{rows}}
  </svg>`;
}}

function renderScatterSVG(rows, medRet) {{
  const W = 500, H = 215, padL = 44, padR = 20, padT = 18, padB = 30;
  const xs = rows.map(r => r.underval_past), ys = rows.map(r => r.actual_ret);
  const minX = Math.min(-15, ...xs) - 2, maxX = Math.max(10, ...xs) + 2;
  const minY = Math.min(-15, ...ys) - 2, maxY = Math.max(10, ...ys) + 2;
  const px = v => padL + (v - minX) / (maxX - minX) * (W - padL - padR);
  const py = v => H - padB - (v - minY) / (maxY - minY) * (H - padT - padB);
  const zeroX = px(0), medY = py(medRet);

  let dots = "";
  rows.forEach((r, idx) => {{
    const cx = px(r.underval_past), cy = py(r.actual_ret);
    const fill = r.hit ? "#2358a6" : "#94a3b8";
    dots += `<circle cx="${{cx.toFixed(1)}}" cy="${{cy.toFixed(1)}}" r="4.5" fill="${{fill}}" opacity="0.85" stroke="#fff" stroke-width="1" data-idx="${{idx}}" style="cursor:pointer;">
      <title>${{r.name}}：当时偏离 ${{signed(r.underval_past)}} → 实际涨跌 ${{signed(r.actual_ret)}}</title>
    </circle>`;
  }});

  return `<svg viewBox="0 0 ${{W}} ${{H}}" id="scatterSvg" style="width:100%;height:auto;display:block;">
    <rect x="${{zeroX}}" y="${{padT}}" width="${{W - padR - zeroX}}" height="${{medY - padT}}" fill="#fdf2f2" opacity="0.5"/>
    <rect x="${{padL}}" y="${{medY}}" width="${{zeroX - padL}}" height="${{H - padB - medY}}" fill="#f1faf7" opacity="0.6"/>
    <line x1="${{padL}}" y1="${{medY}}" x2="${{W - padR}}" y2="${{medY}}" stroke="#cbd5e1" stroke-dasharray="3 3"/>
    <line x1="${{zeroX}}" y1="${{padT}}" x2="${{zeroX}}" y2="${{H - padB}}" stroke="#cbd5e1" stroke-dasharray="3 3"/>
    <text x="${{W - padR - 4}}" y="${{padT + 12}}" text-anchor="end" font-size="10" fill="#c43737">低估且跑赢中位</text>
    <text x="${{padL + 4}}" y="${{H - padB - 6}}" text-anchor="start" font-size="10" fill="#187a62">高估且跑输中位</text>
    <text x="${{W / 2}}" y="${{H - 6}}" text-anchor="middle" font-size="10" fill="#687382">当时模型预测低估度 (%) →</text>
    <text x="12" y="${{H / 2}}" text-anchor="middle" font-size="10" fill="#687382" transform="rotate(-90 12 ${{H / 2}})">实际涨跌 (%)</text>
    ${{dots}}
  </svg>`;
}}

function renderAccuracy() {{
  const box = document.getElementById("accuracyBody");
  if (currentHorizon === "ledger") {{
    const l = DATA.live_ledger;
    const rows = l.comparisons.map(c => `
      <tr>
        <td class="mono">${{c.from_date}} → ${{c.to_date}}</td>
        <td class="mono">${{c.count}} 只</td>
        <td class="mono ${{cls(c.top_ret)}}">${{signed(c.top_ret)}}</td>
        <td class="mono ${{cls(c.bot_ret)}}">${{signed(c.bot_ret)}}</td>
        <td class="mono ${{cls(c.mkt_ret)}}">${{signed(c.mkt_ret)}}</td>
        <td class="mono ${{cls(c.spread)}}" style="font-weight:600">${{signed(c.spread)}}</td>
        <td class="mono ${{cls(c.rank_ic)}}">${{c.rank_ic > 0 ? "+" : ""}}${{fmt(c.rank_ic, 3)}}</td>
      </tr>
    `).join("");
    box.innerHTML = `
      <div class="table-box">
        <div class="table-toolbar">
          <strong>实盘逐日追踪账本（已累积 ${{l.tracked_days}} 个交易日快照）</strong>
          <span class="muted" style="font-size:12px;">每日自动记录当日定价，与未来真实市价核对，杜绝任何前瞻偏差</span>
        </div>
        ${{rows ? `<table><thead><tr><th>验证区间</th><th>标的数</th><th>低估组均涨跌</th><th>高估组均涨跌</th><th>全市场均涨跌</th><th>多空超额</th><th>Rank IC</th></tr></thead><tbody>${{rows}}</tbody></table>` : `<div style="padding:28px;color:var(--muted);text-align:center;">实盘账本刚起步，已存档今天（${{DATA.date}}）快照；明日起将在此展示每日实时前瞻跟踪结果。当前请切换上方“近 20 / 10 / 5 交易日”查看历史真实验证。</div>`}}
      </div>
    `;
    return;
  }}

  const v = DATA.validation[currentHorizon];
  if (!v) {{
    box.innerHTML = `<div class="panel muted">该窗口数据不足，请选择其他时间窗口。</div>`;
    return;
  }}
  const s = v.summary;
  const rowsHtml = v.rows.map(r => `
    <tr>
      <td><strong>${{r.name}}</strong><span style="display:block;font-size:11px;color:var(--muted);" class="mono">${{r.ticker}} · ${{r.stock_name}}</span></td>
      <td class="mono">¥${{fmt(r.past_price)}}</td>
      <td class="mono">¥${{fmt(r.past_model)}}</td>
      <td class="mono ${{cls(r.underval_past)}}" style="font-weight:600;">${{signed(r.underval_past)}}</td>
      <td class="mono">¥${{fmt(r.now_price)}}</td>
      <td class="mono ${{cls(r.actual_ret)}}" style="font-weight:600;">${{signed(r.actual_ret)}}</td>
      <td class="mono ${{cls(r.excess_vs_med)}}">${{signed(r.excess_vs_med)}}</td>
      <td><span class="tag ${{r.hit ? "tag-hit" : "tag-miss"}}">${{r.hit ? "符合预期" : "偏离"}}</span><span style="display:block;font-size:10px;color:var(--muted);margin-top:2px;">${{r.expect_label}}</span></td>
    </tr>
  `).join("");

  box.innerHTML = `
    <div class="charts-grid">
      <div class="panel">
        <div class="panel-head"><h3>分组真实收益对比 (${{v.past_date}} → ${{v.now_date}})</h3><span>低估组 vs 高估组 vs 基准</span></div>
        ${{renderBarsSVG(s)}}
      </div>
      <div class="panel">
        <div class="panel-head"><h3>预测偏离度 vs 实际涨跌幅分布</h3><span>Rank IC: ${{s.rank_ic > 0 ? "+" : ""}}${{fmt(s.rank_ic, 3)}} · 蓝点为命中</span></div>
        ${{renderScatterSVG(v.rows, s.mkt_med_ret)}}
        <div class="scatter-tip" id="scatterTip">鼠标悬停圆点查看单券定价与实际表现</div>
      </div>
    </div>
    <div class="table-box">
      <div class="table-toolbar">
        <strong>逐券定价回顾与结果核对 (${{v.past_date}} 定价 → ${{v.now_date}} 验证)</strong>
        <span class="muted" style="font-size:12px;">样本中位涨跌：${{signed(s.mkt_med_ret)}} · 方向命中率 ${{fmt(s.hit_rate, 1)}}%</span>
      </div>
      <table>
        <thead>
          <tr>
            <th>可转债标的</th>
            <th>当时市价</th>
            <th>当时模型价</th>
            <th>当时预测偏离</th>
            <th>当前市价</th>
            <th>区间真实涨跌</th>
            <th>相对中位超额</th>
            <th>验证结果</th>
          </tr>
        </thead>
        <tbody>${{rowsHtml}}</tbody>
      </table>
    </div>
  `;

  document.querySelectorAll("#scatterSvg circle").forEach(el => {{
    el.addEventListener("mouseenter", () => {{
      const r = v.rows[Number(el.dataset.idx)];
      if (r) document.getElementById("scatterTip").textContent = `${{r.name}} (${{r.ticker}})：当时市价 ¥${{fmt(r.past_price)}}，模型价 ¥${{fmt(r.past_model)}} (${{signed(r.underval_past)}}) → 现价 ¥${{fmt(r.now_price)}} (${{signed(r.actual_ret)}})`;
    }});
  }});
}}

function renderLiveTable() {{
  const q = document.getElementById("liveSearch").value.trim().toLowerCase();
  const reg = document.getElementById("liveRegion").value;
  const sort = document.getElementById("liveSort").value;
  let items = [...DATA.all_bonds];
  if (q) items = items.filter(b => `${{b.ticker}} ${{b.name}} ${{b.stock_name}}`.toLowerCase().includes(q));
  if (reg !== "all") items = items.filter(b => b.region_code === reg);
  items.sort((a, b) => sort === "dl" ? a.double_low - b.double_low : b.undervaluation - a.undervaluation);

  document.getElementById("liveTableBody").innerHTML = items.map((b, i) => `
    <tr>
      <td class="mono muted">#${{i + 1}}</td>
      <td><strong>${{b.name}}</strong><span style="display:block;font-size:11px;color:var(--muted);" class="mono">${{b.ticker}} · ${{b.stock_name}} (¥${{fmt(b.stock_price)}})</span></td>
      <td class="mono">¥${{fmt(b.bond_price)}}</td>
      <td class="mono" style="font-weight:600;">¥${{fmt(b.model_price)}}</td>
      <td class="mono ${{cls(b.undervaluation)}}" style="font-weight:600;">${{signed(b.undervaluation)}}</td>
      <td><span class="tag tag-reg">${{b.region_code}} ${{b.region_name}}</span><span style="display:block;font-size:11px;color:var(--muted);margin-top:2px;">${{b.region_desc}}</span></td>
      <td class="mono">${{fmt(b.premium_rate, 1)}}%</td>
      <td class="mono">${{fmt(b.double_low, 1)}}</td>
      <td class="mono muted">${{fmt(b.sigma, 1)}}% / ${{fmt(b.maturity_years, 1)}}年</td>
    </tr>
  `).join("") || `<tr><td colspan="9" class="muted" style="text-align:center;padding:24px;">无匹配转债</td></tr>`;
}}

function doCalc() {{
  const S0 = parseFloat(document.getElementById("slider-s0").value);
  const C0 = parseFloat(document.getElementById("slider-c0").value);
  const p = parseFloat(document.getElementById("slider-p").value);
  const sig = parseFloat(document.getElementById("slider-sig").value);
  document.getElementById("txt-s0").textContent = "¥" + S0.toFixed(2);
  document.getElementById("txt-c0").textContent = "¥" + C0.toFixed(2);
  document.getElementById("txt-p").textContent = (p * 100).toFixed(0) + "%";
  document.getElementById("txt-sig").textContent = (sig * 100).toFixed(0) + "%";

  const convVal = (100.0 / C0) * S0;
  document.getElementById("res-conv").textContent = "¥" + convVal.toFixed(2);
  const k0 = 1.30 * C0, ptr0 = 0.70 * C0;
  let title = "", desc = "";
  if (S0 > k0) {{ title = "强赎提示区 (R1)"; desc = "正股高于 130% 强赎触发价，股性主导，注意公告强赎风险。"; }}
  else if (S0 > C0) {{ title = "高弹性转股区 (R2)"; desc = "实值期权区，正股弹性直接传导至转债价格。"; }}
  else if (S0 > ptr0) {{ title = "平衡抗跌区 (R3)"; desc = "债底支撑与期权弹性兼备，防守性较强。"; }}
  else {{ title = "下修博弈区 (R4)"; desc = "跌破 70% 回售线，若公司下修意愿 p=" + (p * 100).toFixed(0) + "% 较高，理论底价明显抬升。"; }}
  document.getElementById("res-reg-title").textContent = title;
  document.getElementById("res-reg-text").textContent = desc;

  const T = 2.5, r = 0.025;
  const d1 = (Math.log(S0 / C0) + (r + 0.5 * sig * sig) * T) / (sig * Math.sqrt(T));
  const d2 = d1 - sig * Math.sqrt(T);
  const erf = x => {{ const s = x >= 0 ? 1 : -1; x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x); const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return s * y; }};
  const cdf = x => 0.5 * (1 + erf(x / Math.SQRT2));
  const bsCall = S0 * cdf(d1) - C0 * Math.exp(-r * T) * cdf(d2);
  const totalMdl = Math.max(convVal, Math.max(108 * Math.exp(-r * T) + (100 / C0) * bsCall + (S0 <= ptr0 ? p * 12.5 * (1 - S0 / C0) : p * 3.5), 103.68));
  document.getElementById("res-mdl").textContent = "¥" + totalMdl.toFixed(2);
}}

document.querySelectorAll("[data-tab]").forEach(btn => {{
  btn.addEventListener("click", () => {{
    currentTab = btn.dataset.tab;
    document.querySelectorAll("[data-tab]").forEach(b => b.classList.toggle("active", b === btn));
    ["accuracy", "live", "calc"].forEach(t => document.getElementById("tab-" + t).hidden = t !== currentTab);
    document.getElementById("horizonControls").style.display = currentTab === "accuracy" ? "flex" : "none";
  }});
}});

document.querySelectorAll("[data-horizon]").forEach(btn => {{
  btn.addEventListener("click", () => {{
    currentHorizon = btn.dataset.horizon;
    document.querySelectorAll("[data-horizon]").forEach(b => b.classList.toggle("active", b === btn));
    renderKPI();
    renderAccuracy();
  }});
}});

["liveSearch", "liveRegion", "liveSort"].forEach(id => document.getElementById(id).addEventListener("input", renderLiveTable));
["slider-s0", "slider-c0", "slider-p", "slider-sig"].forEach(id => document.getElementById(id).addEventListener("input", doCalc));

renderKPI();
renderAccuracy();
renderLiveTable();
doCalc();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    generate_page_and_data()
