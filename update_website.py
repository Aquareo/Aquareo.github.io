import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pytz

try:
    from cb_pricing_model import (
        price_ccb_four_region_lsm,
        price_ccb_calibrated_defensive,
    )
except ImportError:
    from .cb_pricing_model import (
        price_ccb_four_region_lsm,
        price_ccb_calibrated_defensive,
    )


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
        return "强赎提示区", "R1", "正股超 130% 强赎触发线"
    if S0 > C0:
        return "高弹性转股区", "R2", "正股高于转股价，股性凸显"
    if S0 > ptr0:
        return "平衡抗跌区", "R3", "兼顾纯债底与转股期权"
    return "下修博弈区", "R4", "低于 70% 回售线，博弈下修"


def fetch_active_pool(max_bonds: int = 36):
    """Fetch active, liquid, already-listed convertible bonds (strictly excluding unlisted 100.00 bonds)."""
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
                if rem < 0.3 or price < 85 or price > 260 or abs(price - 100.0) < 1e-4 or c_price <= 0 or s_price <= 0:
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
            listed = df_em[
                df_em["上市时间"].notna()
                & (df_em["上市时间"] <= now_ts - pd.Timedelta(days=120))
                & (df_em["上市时间"] >= now_ts - pd.Timedelta(days=1100))
                & (df_em["债现价"] >= 85)
                & (df_em["债现价"] <= 250)
                & ((df_em["债现价"] - 100.0).abs() > 1e-4)
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
    """Fetch aligned daily bond + stock history sequentially for a single CB."""
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
        if merged["close_bond"].nunique() < 8 or float(merged["close_bond"].std() or 0) < 0.05:
            return item, None
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


def summarize_model_metrics(df: pd.DataFrame, score_col: str, mdl_col: str, label: str) -> dict:
    """Compute portfolio & accuracy metrics for a given valuation score column."""
    sdf = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    n = len(sdf)
    k = max(2, n // 3)
    k10 = min(10, n)
    top_df = sdf.iloc[:k]
    mid_df = sdf.iloc[k : n - k] if n - 2 * k > 0 else sdf.iloc[k:k]
    bot_df = sdf.iloc[n - k :]
    dl_df = sdf.sort_values("double_low_past", ascending=True).iloc[:k]
    dl10_df = sdf.sort_values("double_low_past", ascending=True).iloc[:k10]

    bench_avg = float(sdf["actual_ret"].mean())
    bench_med = float(sdf["actual_ret"].median())
    top_avg = float(top_df["actual_ret"].mean())
    top10_avg = float(sdf.iloc[:k10]["actual_ret"].mean())
    mid_avg = float(mid_df["actual_ret"].mean()) if len(mid_df) > 0 else bench_avg
    bot_avg = float(bot_df["actual_ret"].mean())
    dl_avg = float(dl_df["actual_ret"].mean())
    dl10_avg = float(dl10_df["actual_ret"].mean())

    half = n // 2
    hits = [
        (float(sdf.iloc[i]["actual_ret"]) >= bench_med)
        if i < half
        else (float(sdf.iloc[i]["actual_ret"]) <= bench_med)
        for i in range(n)
    ]
    hit_rate = round(float(np.mean(hits)) * 100, 1)
    top_beat_rate = round(float((top_df["actual_ret"] >= bench_med).mean()) * 100, 1)
    conv_rate = round(
        float(((sdf["now_price"] - sdf[mdl_col]).abs() < (sdf["past_price"] - sdf[mdl_col]).abs()).mean()) * 100,
        1,
    )

    return {
        "model_label": label,
        "top_avg_ret": round(top_avg, 2),
        "top10_avg_ret": round(top10_avg, 2),
        "mid_avg_ret": round(mid_avg, 2),
        "bot_avg_ret": round(bot_avg, 2),
        "mkt_avg_ret": round(bench_avg, 2),
        "mkt_med_ret": round(bench_med, 2),
        "dl_avg_ret": round(dl_avg, 2),
        "dl10_avg_ret": round(dl10_avg, 2),
        "long_short_spread": round(top_avg - bot_avg, 2),
        "alpha_vs_mkt": round(top_avg - bench_avg, 2),
        "alpha_vs_dl": round(top_avg - dl_avg, 2),
        "top10_vs_dl10": round(top10_avg - dl10_avg, 2),
        "hit_rate": hit_rate,
        "top_beat_rate": top_beat_rate,
        "rank_ic": spearman_corr(sdf[score_col], sdf["actual_ret"]),
        "convergence_rate": conv_rate,
    }


def evaluate_horizon(histories: list, horizon_days: int):
    """Run point-in-time valuation for BOTH Original & Calibrated models at T - horizon_days."""
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

        # 1) Original Paper Model (p_reset=0.80, B=108.0)
        P_orig_past = price_ccb_four_region_lsm(
            S0=S_past, C0=C0, FV=100.0, maturity_years=mat_past,
            r=0.025, q=0.0, sigma=sigma_past,
            call_ratio=1.30, put_ratio=0.70, p_reset=0.80,
            B=108.0, n_steps=24, M=1200, seed=42
        )
        underval_orig_past = round((P_orig_past - P_past) / P_past * 100, 2)

        # 2) Calibrated Defensive Model (p_reset=0.40, B_mat, sig<=0.35, cushion/prem haircut)
        P_cal_past, _, b_floor_cal, haircut_pct = price_ccb_calibrated_defensive(
            S0=S_past, C0=C0, bond_price=P_past, maturity_years=mat_past,
            sigma=sigma_past, FV=100.0, n_steps=24, M=1200, seed=42
        )
        underval_cal_past = round((P_cal_past - P_past) / P_past * 100, 2)

        actual_ret = round((P_now - P_past) / P_past * 100, 2)
        stock_ret = round((S_now - S_past) / S_past * 100, 2)
        conv_val_past = (100.0 / C0) * S_past
        prem_past = round((P_past - conv_val_past) / max(0.01, conv_val_past) * 100, 2)
        dl_past = round(P_past + prem_past, 2)

        reg_name, reg_code, _ = classify_region(S_past, C0)
        rows.append({
            "ticker": item["ticker"],
            "name": item["name"],
            "stock_name": item["stock_name"],
            "past_date": d_past,
            "now_date": d_now,
            "past_price": round(P_past, 2),
            "now_price": round(P_now, 2),
            "past_model_orig": round(P_orig_past, 2),
            "underval_orig_past": underval_orig_past,
            "past_model_cal": round(P_cal_past, 2),
            "underval_cal_past": underval_cal_past,
            "prem_past": prem_past,
            "b_floor_cal": b_floor_cal,
            "haircut_pct": haircut_pct,
            "actual_ret": actual_ret,
            "stock_ret": stock_ret,
            "double_low_past": dl_past,
            "region_code": reg_code,
            "region_name": reg_name,
        })

    if len(rows) < 6:
        return None

    df = pd.DataFrame(rows)
    bench_med = float(df["actual_ret"].median())
    n = len(df)
    half = n // 2

    # Tag hit verdicts for both models relative to sample median
    df_orig_sorted = df.sort_values("underval_orig_past", ascending=False).reset_index(drop=True)
    orig_top_tickers = set(df_orig_sorted.iloc[:half]["ticker"])

    df_cal_sorted = df.sort_values("underval_cal_past", ascending=False).reset_index(drop=True)
    cal_top_tickers = set(df_cal_sorted.iloc[:half]["ticker"])

    for r in rows:
        is_orig_bull = r["ticker"] in orig_top_tickers
        is_cal_bull = r["ticker"] in cal_top_tickers
        r["orig_expect"] = "看多(低估)" if is_orig_bull else "谨慎(偏高)"
        r["cal_expect"] = "看多(低估)" if is_cal_bull else "谨慎(偏高)"
        r["hit_orig"] = bool((r["actual_ret"] >= bench_med) if is_orig_bull else (r["actual_ret"] <= bench_med))
        r["hit_cal"] = bool((r["actual_ret"] >= bench_med) if is_cal_bull else (r["actual_ret"] <= bench_med))
        r["excess_vs_med"] = round(r["actual_ret"] - bench_med, 2)

    orig_summary = summarize_model_metrics(df, "underval_orig_past", "past_model_orig", "原始论文模型")
    cal_summary = summarize_model_metrics(df, "underval_cal_past", "past_model_cal", "修正防守模型")

    rows_sorted = sorted(rows, key=lambda x: x["underval_cal_past"], reverse=True)
    return {
        "horizon": horizon_days,
        "sample_size": n,
        "past_date": str(df["past_date"].mode().iloc[0]),
        "now_date": str(df["now_date"].mode().iloc[0]),
        "orig": orig_summary,
        "cal": cal_summary,
        "rows": rows_sorted,
    }


def update_live_history(dist_dir: Path, date_str: str, evaluated: list):
    """Persist daily live snapshots with both models to history.json and track out-of-sample results."""
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
                "model_price_cal": b["model_price_cal"],
                "undervaluation_cal": b["undervaluation_cal"],
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

    current_prices = {b["ticker"]: b["bond_price"] for b in evaluated}
    live_records = []
    for past_snap in snapshots[:-1]:
        p_date = past_snap["date"]
        matched = []
        for ticker, info in past_snap["bonds"].items():
            if ticker in current_prices and info["bond_price"] > 0:
                ret = round((current_prices[ticker] - info["bond_price"]) / info["bond_price"] * 100, 2)
                matched.append({
                    "underval_orig": info["undervaluation"],
                    "underval_cal": info.get("undervaluation_cal", info["undervaluation"]),
                    "double_low": info.get("double_low", 999.0),
                    "ret": ret,
                })
        if len(matched) >= 6:
            mdf = pd.DataFrame(matched)
            k = max(2, len(mdf) // 3)
            orig_top = round(float(mdf.sort_values("underval_orig", ascending=False).iloc[:k]["ret"].mean()), 2)
            orig_bot = round(float(mdf.sort_values("underval_orig", ascending=False).iloc[-k:]["ret"].mean()), 2)
            cal_top = round(float(mdf.sort_values("underval_cal", ascending=False).iloc[:k]["ret"].mean()), 2)
            cal_bot = round(float(mdf.sort_values("underval_cal", ascending=False).iloc[-k:]["ret"].mean()), 2)
            dl_top = round(float(mdf.sort_values("double_low", ascending=True).iloc[:k]["ret"].mean()), 2)
            mkt_r = round(float(mdf["ret"].mean()), 2)
            live_records.append({
                "from_date": p_date,
                "to_date": date_str,
                "count": len(mdf),
                "orig_top": orig_top,
                "orig_spread": round(orig_top - orig_bot, 2),
                "orig_ic": spearman_corr(mdf["underval_orig"], mdf["ret"]),
                "cal_top": cal_top,
                "cal_spread": round(cal_top - cal_bot, 2),
                "cal_ic": spearman_corr(mdf["underval_cal"], mdf["ret"]),
                "dl_top": dl_top,
                "mkt_ret": mkt_r,
            })

    return {"tracked_days": len(snapshots), "comparisons": list(reversed(live_records))[:15]}


def load_and_evaluate():
    pool = fetch_active_pool(max_bonds=36)
    histories = [fetch_pair_history(item) for item in pool]

    valid_histories = [(item, df) for item, df in histories if df is not None]
    use_list = valid_histories if len(valid_histories) >= 10 else [(item, None) for item in pool]

    evaluated = []
    for item, merged in use_list:
        S0 = float(merged.iloc[-1]["close_stock"]) if merged is not None else item["stock_price"]
        Pmkt = float(merged.iloc[-1]["close_bond"]) if merged is not None else item["bond_price"]
        if abs(Pmkt - 100.0) < 1e-4:
            continue
        C0 = item["conv_price"]
        mat = item.get("maturity_years", 2.5)
        sigma = estimate_sigma(merged["close_stock"], len(merged) - 1, window=60) if merged is not None else 0.28

        conv_val = round((100.0 / C0) * S0, 2)
        premium_rate = round(((Pmkt - conv_val) / max(0.01, conv_val)) * 100, 2)
        double_low = round(Pmkt + premium_rate, 2)
        region_name, region_code, region_desc = classify_region(S0, C0)

        # Original Paper Model
        Pmdl_orig = price_ccb_four_region_lsm(
            S0=S0, C0=C0, FV=100.0, maturity_years=mat,
            r=0.025, q=0.0, sigma=sigma,
            call_ratio=1.30, put_ratio=0.70, p_reset=0.80,
            B=108.0, n_steps=24, M=1600, seed=42
        )
        underval_orig = round(((Pmdl_orig - Pmkt) / Pmkt) * 100, 2)

        # Calibrated Defensive Model
        Pmdl_cal, Pmdl_cal_raw, B_cal, haircut_pct = price_ccb_calibrated_defensive(
            S0=S0, C0=C0, bond_price=Pmkt, maturity_years=mat,
            sigma=sigma, FV=100.0, n_steps=24, M=1600, seed=42
        )
        underval_cal = round(((Pmdl_cal - Pmkt) / Pmkt) * 100, 2)

        evaluated.append({
            "ticker": item["ticker"],
            "name": item["name"],
            "stock_code": item["stock_code"],
            "stock_name": item["stock_name"],
            "stock_price": round(S0, 2),
            "conv_price": round(C0, 2),
            "bond_price": round(Pmkt, 2),
            "model_price": round(Pmdl_orig, 2),
            "undervaluation": underval_orig,
            "model_price_cal": round(Pmdl_cal, 2),
            "model_price_cal_raw": round(Pmdl_cal_raw, 2),
            "undervaluation_cal": underval_cal,
            "bond_floor_cal": B_cal,
            "haircut_pct": haircut_pct,
            "pure_debt_val": round(108.0 * np.exp(-0.025 * mat), 1),
            "double_low": double_low,
            "conv_val": conv_val,
            "premium_rate": premium_rate,
            "sigma": round(sigma * 100, 1),
            "region_name": region_name,
            "region_code": region_code,
            "region_desc": region_desc,
            "maturity_years": round(mat, 2),
        })

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

    cal_sorted = sorted(evaluated, key=lambda x: x["undervaluation_cal"], reverse=True)
    orig_sorted = sorted(evaluated, key=lambda x: x["undervaluation"], reverse=True)
    dl_sorted = sorted(evaluated, key=lambda x: x["double_low"], reverse=False)

    cal_rank_map = {b["ticker"]: i for i, b in enumerate(cal_sorted, 1)}
    orig_rank_map = {b["ticker"]: i for i, b in enumerate(orig_sorted, 1)}
    dl_rank_map = {b["ticker"]: i for i, b in enumerate(dl_sorted, 1)}
    for b in evaluated:
        b["cal_rank"] = cal_rank_map[b["ticker"]]
        b["lsm_rank"] = orig_rank_map[b["ticker"]]
        b["dl_rank"] = dl_rank_map[b["ticker"]]

    dist_dir = Path(__file__).resolve().parent / "dist" / "bonds"
    dist_dir.mkdir(parents=True, exist_ok=True)

    existing_data = {}
    data_file = dist_dir / "data.json"
    if data_file.exists():
        try:
            existing_data = json.loads(data_file.read_text(encoding="utf-8"))
        except Exception:
            existing_data = {}

    # Preserve verified historical validation & pool breadth if remote CI runner (e.g. overseas IP) fails history fetches
    if existing_data.get("validation"):
        for hk, hv in existing_data["validation"].items():
            if hk not in validation:
                validation[hk] = hv

    if len(evaluated) < 25 and existing_data.get("all_bonds"):
        seen = {b["ticker"] for b in evaluated}
        for old_b in existing_data["all_bonds"]:
            if old_b.get("ticker") not in seen and len(evaluated) < 36:
                evaluated.append(old_b)
                seen.add(old_b.get("ticker"))
        cal_sorted = sorted(evaluated, key=lambda x: x["undervaluation_cal"], reverse=True)
        orig_sorted = sorted(evaluated, key=lambda x: x["undervaluation"], reverse=True)
        dl_sorted = sorted(evaluated, key=lambda x: x["double_low"], reverse=False)
        for i, b in enumerate(cal_sorted, 1):
            b["cal_rank"] = i
        for i, b in enumerate(orig_sorted, 1):
            b["lsm_rank"] = i
        for i, b in enumerate(dl_sorted, 1):
            b["dl_rank"] = i

    live_ledger = update_live_history(dist_dir, date_str, cal_sorted)

    data_payload = {
        "updated_at": update_time_str,
        "date": date_str,
        "author": "Yu Liu (刘宇) · The Chinese University of Hong Kong, Shenzhen",
        "paper_ref": "arXiv:2409.06496v7",
        "paper_url": "https://arxiv.org/abs/2409.06496",
        "pool_size": len(cal_sorted),
        "models": {
            "orig": {"name": "原始论文模型", "p_reset": 0.80, "B": 108.0, "desc": "标准 Four-Region LSM"},
            "cal": {"name": "修正防守模型", "p_reset": 0.40, "B": "min(106, 99+1.8T)", "desc": "保守下修 + 波动率封顶 + 高价/高溢价滞后折价"},
        },
        "validation": validation,
        "live_ledger": live_ledger,
        "cal_top10": cal_sorted[:10],
        "orig_top10": orig_sorted[:10],
        "double_low_top10": dl_sorted[:10],
        "all_bonds": cal_sorted,
    }

    (dist_dir / "data.json").write_text(json.dumps(data_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (dist_dir / "index.html").write_text(build_clean_html(data_payload), encoding="utf-8")
    print(f"Generated dual-model bonds dashboard at {dist_dir / 'index.html'} (pool={len(cal_sorted)}, horizons={list(validation.keys())})")


def build_clean_html(data):
    payload_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    v20 = data["validation"].get("20") or (list(data["validation"].values())[0] if data["validation"] else None)

    cal_top10_underv = round(sum(b["undervaluation_cal"] for b in data["cal_top10"]) / max(1, len(data["cal_top10"])), 1)
    orig_top10_underv = round(sum(b["undervaluation"] for b in data["orig_top10"]) / max(1, len(data["orig_top10"])), 1)

    if v20:
        c20, o20 = v20["cal"], v20["orig"]
        card1_val = f"{c20['top_avg_ret']:+.2f}%"
        card1_desc = f"Rank IC {c20['rank_ic']:+.3f} · 多空超额 {c20['long_short_spread']:+.2f}% · 当期低估 {cal_top10_underv:+.1f}%"
        card2_val = f"{o20['top_avg_ret']:+.2f}%"
        card2_desc = f"Rank IC {o20['rank_ic']:+.3f} · 多空超额 {o20['long_short_spread']:+.2f}% · 当期低估 {orig_top10_underv:+.1f}%"
        card3_val = f"{c20['dl_avg_ret']:+.2f}%"
        card3_desc = f"修正组跑赢双低 {c20['alpha_vs_dl']:+.2f}% · 全市场均值 {c20['mkt_avg_ret']:+.2f}%"
        h_label = f"{v20['horizon']}日"
    else:
        card1_val = f"{cal_top10_underv:+.1f}%"
        card1_desc = "当前修正防守模型 Top 10 平均低估空间"
        card2_val = f"{orig_top10_underv:+.1f}%"
        card2_desc = "当前原始论文模型 Top 10 平均低估空间"
        card3_val = f"{data['pool_size']} 只"
        card3_desc = "双低基准对照样本"
        h_label = "近期"

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>可转债量化估值看板 · 双模型对照与准确率检验 · Yu Liu</title>
  <meta name="description" content="基于 Four-Region LSM 原始论文模型与修正防守模型的可转债估值选券及历史准确率验证看板">
  <style>
    :root {{
      --ink: #232a33;
      --muted: #697583;
      --light-muted: #8b96a5;
      --blue: #245ca9;
      --blue-light: #f0f5fc;
      --line: #e6eaf0;
      --soft: #f8f9fc;
      --green: #0f766e;
      --green-light: #f0fdfa;
      --up: #c43737;
      --up-light: #fdf2f2;
      color-scheme: light;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #ffffff;
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      font-size: 14px;
      line-height: 1.68;
      -webkit-font-smoothing: antialiased;
    }}
    .font-mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-variant-numeric: tabular-nums; }}
    a {{ color: var(--blue); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}

    .shell {{ max-width: 1100px; margin: 0 auto; padding: 36px 24px 64px; }}

    .topbar {{
      display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 12px;
      padding-bottom: 18px; border-bottom: 1px solid var(--line); margin-bottom: 28px; font-size: 13px;
    }}
    .topbar-brand {{ font-weight: 600; color: var(--ink); font-size: 15px; }}
    .topbar-brand span {{ color: var(--muted); font-weight: 400; font-size: 13px; margin-left: 8px; }}
    .topbar-links {{ display: flex; gap: 18px; flex-wrap: wrap; }}
    .topbar-links a {{ color: var(--muted); }}
    .topbar-links a:hover {{ color: var(--blue); }}

    .header-area {{ margin-bottom: 24px; }}
    h1 {{ font-size: 25px; font-weight: 650; letter-spacing: -0.3px; line-height: 1.38; margin: 0 0 8px; }}
    .header-lead {{ color: var(--muted); font-size: 14px; margin: 0; max-width: 880px; }}

    .stats-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }}
    .stat-card {{ background: var(--soft); border: 1px solid var(--line); border-radius: 8px; padding: 16px 18px; }}
    .stat-card .label {{ font-size: 12px; color: var(--muted); display: block; margin-bottom: 4px; }}
    .stat-card .val {{ font-size: 22px; font-weight: 650; line-height: 1.2; color: var(--ink); }}
    .stat-card .desc {{ font-size: 11px; color: var(--light-muted); margin-top: 6px; line-height: 1.45; }}

    .explainer-box {{
      background: #f5f8fd; border-left: 3px solid #3671c2; padding: 15px 20px;
      border-radius: 0 8px 8px 0; margin-bottom: 28px; font-size: 13px; color: #3b4859;
    }}
    .explainer-box strong {{ color: var(--ink); }}
    .explainer-box p {{ margin: 6px 0 0; }}

    .card {{
      background: #fff; border: 1px solid var(--line); border-radius: 8px;
      overflow: hidden; margin-bottom: 28px;
    }}
    .card-header {{
      padding: 15px 20px; background: #fafbfe; border-bottom: 1px solid var(--line);
      display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;
    }}
    .card-title {{ font-size: 15px; font-weight: 600; margin: 0; }}
    .card-subtitle {{ font-size: 12px; color: var(--muted); margin-top: 2px; }}
    .tab-pills {{ display: flex; gap: 6px; flex-wrap: wrap; }}
    .tab-btn {{
      padding: 5px 12px; border-radius: 6px; font-size: 12px; border: 1px solid #dce2ea;
      background: #fff; color: #4b586a; cursor: pointer; transition: all 0.15s;
    }}
    .tab-btn.active {{ background: #eaf0fa; border-color: #3671c2; color: var(--blue); font-weight: 600; }}

    .subtoolbar {{
      display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;
      padding: 10px 18px; background: #fcfdfe; border-bottom: 1px solid var(--line);
    }}
    .subtoolbar input, .subtoolbar select {{
      border: 1px solid #dce2ea; background: #fff; border-radius: 5px;
      padding: 5px 10px; font-size: 12px; color: var(--ink);
    }}

    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 13px; }}
    th {{
      background: #fcfdfe; color: var(--muted); font-weight: 500;
      padding: 11px 14px; border-bottom: 1px solid var(--line); font-size: 12px; white-space: nowrap;
    }}
    td {{ padding: 12px 14px; border-bottom: 1px solid var(--line); vertical-align: top; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #fafbfe; }}
    td.num {{ font-variant-numeric: tabular-nums; white-space: nowrap; }}
    .subtext {{ display: block; font-size: 11px; color: var(--muted); margin-top: 2px; line-height: 1.4; }}
    .badge {{
      display: inline-block; font-size: 11px; padding: 2px 7px; border-radius: 4px;
      font-weight: 500; white-space: nowrap;
    }}
    .tag-hit {{ color: #0f766e; background: #f0fdfa; border: 1px solid #99f6e4; }}
    .tag-miss {{ color: #64748b; background: #f8fafc; border: 1px solid #e2e8f0; }}

    .pos {{ color: var(--up); font-weight: 600; }}
    .neg {{ color: var(--green); font-weight: 600; }}
    .underv {{ color: var(--green); font-weight: 600; }}
    .overv {{ color: var(--muted); }}

    .cmp-grid {{
      display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
      padding: 16px 20px; border-bottom: 1px solid var(--line); background: #fcfdfe;
    }}
    .cmp-box {{ border: 1px solid var(--line); background: #fff; border-radius: 6px; padding: 12px 14px; }}
    .cmp-box .title {{ font-size: 12px; color: var(--muted); margin-bottom: 4px; display: block; }}
    .cmp-box .main {{ font-size: 18px; font-weight: 650; line-height: 1.3; }}
    .cmp-box .detail {{ font-size: 11px; color: var(--muted); margin-top: 4px; line-height: 1.5; }}

    .charts-grid {{
      display: grid; grid-template-columns: 1fr 1.12fr; gap: 18px;
      padding: 18px 20px; border-bottom: 1px solid var(--line);
    }}
    .chart-panel {{ border: 1px solid var(--line); border-radius: 6px; padding: 14px 16px; background: #fff; }}
    .chart-head {{ display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }}
    .chart-head h3 {{ margin: 0; font-size: 13px; font-weight: 600; }}
    .chart-head span {{ font-size: 11px; color: var(--muted); }}
    .scatter-tip {{ margin-top: 6px; font-size: 12px; color: var(--muted); min-height: 18px; }}

    .calc-section {{
      background: var(--soft); border: 1px solid var(--line); border-radius: 8px;
      padding: 22px 24px 20px; margin-bottom: 28px;
    }}
    .calc-title {{ font-size: 15px; font-weight: 600; margin: 0 0 4px; }}
    .calc-subtitle {{ color: var(--muted); font-size: 13px; margin: 0 0 18px; }}
    .calc-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; align-items: start; }}
    .input-group {{ margin-bottom: 12px; }}
    .input-header {{ display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 4px; }}
    .input-header span:last-child {{ font-weight: 600; color: var(--blue); }}
    input[type=range] {{
      width: 100%; height: 5px; background: #d8e0ea; border-radius: 3px; outline: none; -webkit-appearance: none;
    }}
    input[type=range]::-webkit-slider-thumb {{
      -webkit-appearance: none; width: 15px; height: 15px; border-radius: 50%;
      background: var(--blue); cursor: pointer; border: 2px solid #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }}
    .calc-result-box {{ background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 18px 20px; }}
    .res-metric {{
      display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;
      margin-bottom: 14px; padding-bottom: 14px; border-bottom: 1px solid var(--line);
    }}
    .res-metric .val {{ font-size: 21px; font-weight: 700; margin-top: 2px; }}
    .res-status {{ font-size: 13px; line-height: 1.6; color: var(--muted); }}
    .res-status strong {{ color: var(--ink); display: block; margin-bottom: 2px; }}

    footer {{
      margin-top: 32px; padding-top: 18px; border-top: 1px solid var(--line);
      font-size: 12px; color: var(--muted); line-height: 1.8;
    }}

    @media (max-width: 820px) {{
      .stats-row, .cmp-grid {{ grid-template-columns: 1fr 1fr; }}
      .charts-grid, .calc-grid, .res-metric {{ grid-template-columns: 1fr; }}
      .shell {{ padding: 22px 14px; }}
    }}
  </style>
</head>
<body>

<div class="shell">
  <header class="topbar">
    <div class="topbar-brand">
      Yu Liu · 刘宇 <span>可转债量化估值看板</span>
    </div>
    <nav class="topbar-links">
      <a href="/">个人主页</a>
      <a href="/trade/">A 股买卖看板</a>
      <a href="/phd-tracker/">申请追踪</a>
      <a href="https://arxiv.org/abs/2409.06496" target="_blank" rel="noopener">arXiv:2409.06496 ↗</a>
    </nav>
  </header>

  <div class="header-area">
    <h1>中国可转债双模型定价与历史准确率检验看板</h1>
    <p class="header-lead">
      保留<strong>原始论文 Four-Region LSM 模型</strong>（arXiv:2409.06496v7），同时引入针对快速杀跌场景的<strong>修正防守模型</strong>。
      直观对照两套模型的当前选券结果，并通过近 20 / 10 / 5 交易日真实走势检验它们到底准不准、相对传统双低表现如何。
    </p>
  </div>

  <div class="stats-row">
    <div class="stat-card">
      <span class="label">修正防守模型 · {h_label} 低估组 (Top 1/3)</span>
      <div class="val font-mono">{card1_val}</div>
      <div class="desc">{card1_desc}</div>
    </div>
    <div class="stat-card">
      <span class="label">原始论文模型 · {h_label} 低估组 (Top 1/3)</span>
      <div class="val font-mono">{card2_val}</div>
      <div class="desc">{card2_desc}</div>
    </div>
    <div class="stat-card">
      <span class="label">传统双低基准 · {h_label} 组合收益</span>
      <div class="val font-mono">{card3_val}</div>
      <div class="desc">{card3_desc}</div>
    </div>
    <div class="stat-card">
      <span class="label">追踪样本与更新状态</span>
      <div class="val" style="font-size:15px; font-weight:600; padding-top:4px;">{data['pool_size']} 只上市转债 · 每日自动更新</div>
      <div class="desc">更新于 {data['updated_at']} · 实盘账本已录 {data['live_ledger']['tracked_days']} 天</div>
    </div>
  </div>

  <div class="explainer-box">
    <strong>💡 为什么要同时看“原始论文模型”和“修正防守模型”？</strong>
    <p>
      <strong>1. 原始模型的问题：</strong>原论文设定下修概率 <code>p=0.80</code>、统一债底 <code>¥108</code>，会把部分现价 ¥125+、转股溢价率高于 40% 的高弹性券算出很高“理论下修底价”。但在 20 日快速杀跌里，上市公司来不及公告下修，高溢价券缺乏纯债保护垫，会直接跟着正股下挫。<br>
      <strong>2. 修正模型的改进：</strong>保持 Four-Region LSM 核心算法，将下修概率调保守为 <code>p=0.40</code>、年化波动率封顶 <code>35%</code>、纯债底随剩余年限锚定 <code>min(106, 99+1.8T)</code>，并对市价超 ¥115 与溢价率超 25% 的标的叠加“下修滞后防守折价”。两者并列对照，方便观察市场风格切换时哪个模型更稳。
    </p>
  </div>

  <!-- Section 1: Classic Top 10 Portfolio Card (always visible) -->
  <div class="card">
    <div class="card-header">
      <div>
        <h2 class="card-title">可转债精选持仓组合（当期实时双模型对比）</h2>
        <div class="card-subtitle">直接在主视图查看修正模型、原论文模型、传统双低 Top 10 及全市场跟踪明细</div>
      </div>
      <div class="tab-pills" id="portfolioPills">
        <button class="tab-btn active" data-port="cal">🛡️ 修正防守模型 Top 10</button>
        <button class="tab-btn" data-port="orig">📐 原始论文模型 Top 10</button>
        <button class="tab-btn" data-port="dl">⚖️ 传统双低组合 Top 10</button>
        <button class="tab-btn" data-port="all">📋 全部追踪 ({data['pool_size']}只)</button>
      </div>
    </div>

    <div class="subtoolbar" id="allFilterBar" style="display:none;">
      <span class="subtext" style="margin:0;">支持按代码/名称模糊搜索与条款区间筛选</span>
      <div style="display:flex; gap:8px; flex-wrap:wrap;">
        <input type="search" id="liveSearch" placeholder="搜索代码 / 转债 / 正股">
        <select id="liveRegion">
          <option value="all">全部条款区间</option>
          <option value="R3">R3 平衡抗跌区</option>
          <option value="R2">R2 高弹性转股区</option>
          <option value="R1">R1 强赎提示区</option>
          <option value="R4">R4 下修博弈区</option>
        </select>
        <select id="liveSort">
          <option value="cal">按修正模型低估排序</option>
          <option value="orig">按原始模型低估排序</option>
          <option value="dl">按传统双低排序</option>
        </select>
      </div>
    </div>

    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th style="width:54px; text-align:center;">排名</th>
            <th>可转债标的</th>
            <th>市场现价</th>
            <th>修正模型估值 (空间)</th>
            <th>原始模型估值 (空间)</th>
            <th>所处状态与条款特征</th>
            <th>溢价率 / 双低值</th>
            <th>保守债底 / 折价调整</th>
          </tr>
        </thead>
        <tbody id="portfolioBody"></tbody>
      </table>
    </div>
  </div>

  <!-- Section 2: Historical Accuracy Verification Card (both models vs benchmark) -->
  <div class="card">
    <div class="card-header">
      <div>
        <h2 class="card-title">双模型历史准确性对照检验（样本外实证：到底准不准？）</h2>
        <div class="card-subtitle">严格使用历史当日正股收盘价、前 60 日滚动波动率与剩余期限计算当日估值，对照随后真实涨跌</div>
      </div>
      <div class="tab-pills" id="horizonPills">
        <button class="tab-btn active" data-horizon="20">近 20 交易日</button>
        <button class="tab-btn" data-horizon="10">近 10 交易日</button>
        <button class="tab-btn" data-horizon="5">近 5 交易日</button>
        <button class="tab-btn" data-horizon="ledger">实盘逐日账本</button>
      </div>
    </div>

    <div id="validationContent"></div>
  </div>

  <!-- Section 3: Interactive Single-Bond Calculator (always visible, both models) -->
  <div class="calc-section">
    <h3 class="calc-title">⚡ 单券双模型定价敏感性试算工具</h3>
    <p class="calc-subtitle">同步对比同等条件下“原始论文模型”与“修正防守模型”的估值差异，直观感受市价安全垫与下修意愿的作用</p>

    <div class="calc-grid">
      <div>
        <div class="input-group">
          <div class="input-header"><span>正股当前价格 S₀</span><span id="txt-s0" class="font-mono">¥10.00</span></div>
          <input type="range" id="slider-s0" min="5" max="25" step="0.1" value="10.0">
        </div>
        <div class="input-group">
          <div class="input-header"><span>当前转股价格 C₀</span><span id="txt-c0" class="font-mono">¥10.00</span></div>
          <input type="range" id="slider-c0" min="5" max="25" step="0.1" value="10.0">
        </div>
        <div class="input-group">
          <div class="input-header"><span>假设转债当前市场价 P₀ (用于算折价)</span><span id="txt-pmkt" class="font-mono">¥115.00</span></div>
          <input type="range" id="slider-pmkt" min="95" max="170" step="0.5" value="115.0">
        </div>
        <div class="input-group">
          <div class="input-header"><span>原模型下修意愿 p (修正模型恒用 40% 保守值)</span><span id="txt-p" class="font-mono">80%</span></div>
          <input type="range" id="slider-p" min="0" max="1" step="0.05" value="0.80">
        </div>
        <div class="input-group">
          <div class="input-header"><span>正股年化波动率 σ</span><span id="txt-sig" class="font-mono">28%</span></div>
          <input type="range" id="slider-sig" min="0.12" max="0.55" step="0.02" value="0.28">
        </div>
      </div>

      <div class="calc-result-box">
        <div class="res-metric">
          <div>
            <span style="font-size:12px; color:var(--muted);">修正防守模型</span>
            <div id="res-cal" class="val font-mono" style="color:var(--blue);">¥115.20</div>
            <span id="res-cal-sub" class="subtext">防守折价 0.0%</span>
          </div>
          <div>
            <span style="font-size:12px; color:var(--muted);">原始论文模型</span>
            <div id="res-orig" class="val font-mono" style="color:var(--green);">¥120.45</div>
            <span id="res-orig-sub" class="subtext">含下修博弈 p=80%</span>
          </div>
          <div>
            <span style="font-size:12px; color:var(--muted);">转股平价 / 溢价率</span>
            <div id="res-conv" class="val font-mono" style="color:var(--ink);">¥100.00</div>
            <span id="res-prem-sub" class="subtext">按市价溢价 +15.0%</span>
          </div>
        </div>
        <div class="res-status">
          <strong id="res-reg-title">平衡抗跌区 (R3)</strong>
          <span id="res-reg-text"></span>
        </div>
      </div>
    </div>
  </div>

  <footer>
    <strong>学术引用与检验说明：</strong><br>
    Liu, Yu. (2026). <em>Valuation Model of Chinese Convertible Bonds Based on Monte Carlo Simulation: A Regression-Based Approximation and Error Analysis</em> (<a href="https://arxiv.org/abs/2409.06496" target="_blank" rel="noopener">arXiv:2409.06496v7</a>).<br>
    检验均基于真实日线前向滚动窗口（不含未来信息），仅作学术与模型诊断展示，不构成投资建议。
  </footer>
</div>

<script>
const DATA = {payload_json};
let currentPort = "cal";
let currentHorizon = DATA.validation["20"] ? "20" : (Object.keys(DATA.validation)[0] || "ledger");
let scatterModel = "cal";

const fmt = (v, n = 2) => typeof v === "number" && Number.isFinite(v) ? v.toFixed(n) : "—";
const signed = (v, n = 2) => typeof v === "number" && Number.isFinite(v) ? (v > 0 ? "+" : "") + v.toFixed(n) + "%" : "—";
const retCls = v => v > 0 ? "pos" : v < 0 ? "neg" : "";

function regionBadgeStyle(code) {{
  return {{
    "R1": "color:#92400e; background:#fef3c7; border:1px solid #fde68a;",
    "R2": "color:#166534; background:#dcfce7; border:1px solid #bbf7d0;",
    "R3": "color:#1e40af; background:#dbeafe; border:1px solid #bfdbfe;",
    "R4": "color:#6b21a8; background:#f3e8ff; border:1px solid #e9d5ff;"
  }}[code] || "color:#475569; background:#f1f5f9; border:1px solid #e2e8f0;";
}}

function renderPortfolio() {{
  const filterBar = document.getElementById("allFilterBar");
  filterBar.style.display = currentPort === "all" ? "flex" : "none";

  let items = [];
  if (currentPort === "cal") items = DATA.cal_top10;
  else if (currentPort === "orig") items = DATA.orig_top10;
  else if (currentPort === "dl") items = DATA.double_low_top10;
  else {{
    const q = document.getElementById("liveSearch").value.trim().toLowerCase();
    const reg = document.getElementById("liveRegion").value;
    const sortKey = document.getElementById("liveSort").value;
    items = [...DATA.all_bonds];
    if (q) items = items.filter(b => `${{b.ticker}} ${{b.name}} ${{b.stock_name}}`.toLowerCase().includes(q));
    if (reg !== "all") items = items.filter(b => b.region_code === reg);
    items.sort((a, b) => sortKey === "dl" ? a.double_low - b.double_low : sortKey === "orig" ? b.undervaluation - a.undervaluation : b.undervaluation_cal - a.undervaluation_cal);
  }}

  const rowsHtml = items.map((b, idx) => {{
    const rank = currentPort === "cal" ? b.cal_rank : currentPort === "orig" ? b.lsm_rank : currentPort === "dl" ? b.dl_rank : idx + 1;
    const calCls = b.undervaluation_cal > 0 ? "underv" : "overv";
    const origCls = b.undervaluation > 0 ? "underv" : "overv";
    return `
      <tr>
        <td class="num font-mono" style="text-align:center;">#${{rank}}</td>
        <td>
          <strong>${{b.name}}</strong>
          <span class="subtext font-mono">${{b.ticker}} · ${{b.stock_name}} (正股 ¥${{fmt(b.stock_price)}} / 转股 ¥${{fmt(b.conv_price)}})</span>
        </td>
        <td class="num font-mono" style="font-weight:600;">¥${{fmt(b.bond_price)}}</td>
        <td class="num font-mono">
          <strong class="${{calCls}}">¥${{fmt(b.model_price_cal)}}</strong>
          <span class="subtext ${{calCls}}">${{signed(b.undervaluation_cal)}}</span>
        </td>
        <td class="num font-mono">
          <span class="${{origCls}}">¥${{fmt(b.model_price)}}</span>
          <span class="subtext ${{origCls}}">${{signed(b.undervaluation)}}</span>
        </td>
        <td>
          <span class="badge" style="${{regionBadgeStyle(b.region_code)}}">${{b.region_code}} · ${{b.region_name}}</span>
          <span class="subtext">${{b.region_desc}}</span>
        </td>
        <td class="num font-mono">
          ${{fmt(b.premium_rate, 1)}}%
          <span class="subtext">双低 ${{fmt(b.double_low, 1)}}</span>
        </td>
        <td class="num font-mono">
          ¥${{fmt(b.bond_floor_cal, 1)}}
          <span class="subtext">折价 -${{fmt(b.haircut_pct, 1)}}% · ${{fmt(b.maturity_years, 1)}}年</span>
        </td>
      </tr>
    `;
  }}).join("") || `<tr><td colspan="8" style="text-align:center;color:var(--muted);padding:24px;">无匹配转债记录</td></tr>`;

  document.getElementById("portfolioBody").innerHTML = rowsHtml;
}}

function renderBarsSVG(v) {{
  const c = v.cal, o = v.orig;
  const items = [
    {{ label: "修正模型低估组 (Top 1/3)", val: c.top_avg_ret, hl: true, colorMode: "cal" }},
    {{ label: "原始模型低估组 (Top 1/3)", val: o.top_avg_ret, hl: true, colorMode: "orig" }},
    {{ label: "传统双低组合 (对照)", val: c.dl_avg_ret }},
    {{ label: "全市场样本均价", val: c.mkt_avg_ret }},
    {{ label: "修正模型高估组 (Bot 1/3)", val: c.bot_avg_ret }},
    {{ label: "原始模型高估组 (Bot 1/3)", val: o.bot_avg_ret }},
  ];
  const minVal = Math.min(0, ...items.map(x => x.val));
  const maxVal = Math.max(0, ...items.map(x => x.val));
  const span = Math.max(2, maxVal - minVal);
  const W = 490, H = 226, plotLeft = 225, plotRight = 435;
  const zeroX = plotLeft + (0 - minVal) / span * (plotRight - plotLeft);
  const toX = x => plotLeft + (x - minVal) / span * (plotRight - plotLeft);

  let rows = "";
  items.forEach((it, idx) => {{
    const y = 14 + idx * 34;
    const vx = toX(it.val);
    const rx = Math.min(zeroX, vx);
    const rw = Math.max(2, Math.abs(vx - zeroX));
    const fill = it.colorMode === "cal" ? "#245ca9" : it.colorMode === "orig" ? "#0f766e" : (it.val >= 0 ? "#c43737" : "#7c8b9b");
    const op = it.hl ? "0.92" : "0.62";
    const textX = it.val >= 0 ? rx + rw + 6 : rx - 6;
    const anchor = it.val >= 0 ? "start" : "end";
    rows += `
      <text x="6" y="${{y + 14}}" font-size="12" fill="${{it.hl ? "#1f252e" : "#697583"}}" font-weight="${{it.hl ? "600" : "400"}}">${{it.label}}</text>
      <rect x="${{rx.toFixed(1)}}" y="${{y + 2}}" width="${{rw.toFixed(1)}}" height="17" fill="${{fill}}" opacity="${{op}}" rx="2"></rect>
      <text x="${{textX.toFixed(1)}}" y="${{y + 14}}" font-size="11" font-weight="600" fill="${{fill}}" text-anchor="${{anchor}}">${{signed(it.val)}}</text>
    `;
  }});

  return `<svg viewBox="0 0 ${{W}} ${{H}}" style="width:100%;height:auto;display:block;">
    <line x1="${{zeroX.toFixed(1)}}" y1="8" x2="${{zeroX.toFixed(1)}}" y2="${{H - 8}}" stroke="#cbd5e1" stroke-width="1"/>
    ${{rows}}
  </svg>`;
}}

function renderScatterSVG(rows, medRet) {{
  const W = 500, H = 226, padL = 46, padR = 22, padT = 24, padB = 34;
  const xKey = scatterModel === "cal" ? "underval_cal_past" : "underval_orig_past";
  const hitKey = scatterModel === "cal" ? "hit_cal" : "hit_orig";
  const xs = rows.map(r => r[xKey]), ys = rows.map(r => r.actual_ret);
  const minX = Math.min(-12, ...xs) - 2, maxX = Math.max(8, ...xs) + 2;
  const minY = Math.min(-12, ...ys) - 2, maxY = Math.max(4, ...ys) + 3;
  const px = v => padL + (v - minX) / (maxX - minX) * (W - padL - padR);
  const py = v => H - padB - (v - minY) / (maxY - minY) * (H - padT - padB);
  const zeroX = px(0), medY = py(medRet);

  let dots = "";
  rows.forEach((r, idx) => {{
    const cx = px(r[xKey]), cy = py(r.actual_ret);
    const fill = r[hitKey] ? "#245ca9" : "#94a3b8";
    dots += `<circle cx="${{cx.toFixed(1)}}" cy="${{cy.toFixed(1)}}" r="4.5" fill="${{fill}}" opacity="0.88" stroke="#fff" stroke-width="1" data-idx="${{idx}}" style="cursor:pointer;">
      <title>${{r.name}}：当时偏离 ${{signed(r[xKey])}} → 实际 ${{signed(r.actual_ret)}}</title>
    </circle>`;
  }});

  return `<svg viewBox="0 0 ${{W}} ${{H}}" id="scatterSvg" style="width:100%;height:auto;display:block;">
    <rect x="${{zeroX}}" y="${{padT}}" width="${{Math.max(0, W - padR - zeroX)}}" height="${{Math.max(0, medY - padT)}}" fill="#fdf2f2" opacity="0.55"/>
    <rect x="${{padL}}" y="${{medY}}" width="${{Math.max(0, zeroX - padL)}}" height="${{Math.max(0, H - padB - medY)}}" fill="#f0fdfa" opacity="0.65"/>
    <line x1="${{padL}}" y1="${{medY}}" x2="${{W - padR}}" y2="${{medY}}" stroke="#cbd5e1" stroke-dasharray="3 3"/>
    <line x1="${{zeroX}}" y1="${{padT}}" x2="${{zeroX}}" y2="${{H - padB}}" stroke="#cbd5e1" stroke-dasharray="3 3"/>
    <text x="${{W - padR - 4}}" y="${{padT - 7}}" text-anchor="end" font-size="10" fill="#c43737">右上: 看低估且跑赢中位</text>
    <text x="${{padL + 4}}" y="${{padT - 7}}" text-anchor="start" font-size="10" fill="#0f766e">左下: 看高估且走弱</text>
    <text x="${{W / 2}}" y="${{H - 6}}" text-anchor="middle" font-size="10" fill="#697583">${{scatterModel === "cal" ? "修正防守模型" : "原始论文模型"}} 当时低估度 (%) →</text>
    <text x="12" y="${{H / 2}}" text-anchor="middle" font-size="10" fill="#697583" transform="rotate(-90 12 ${{H / 2}})">实际涨跌 (%)</text>
    ${{dots}}
  </svg>`;
}}

function renderValidation() {{
  const box = document.getElementById("validationContent");
  if (currentHorizon === "ledger") {{
    const l = DATA.live_ledger;
    const rows = l.comparisons.map(c => `
      <tr>
        <td class="font-mono">${{c.from_date}} → ${{c.to_date}}</td>
        <td class="num font-mono">${{c.count}} 只</td>
        <td class="num font-mono ${{retCls(c.cal_top)}}">${{signed(c.cal_top)}} <span class="subtext">IC ${{c.cal_ic > 0 ? "+" : ""}}${{fmt(c.cal_ic, 3)}}</span></td>
        <td class="num font-mono ${{retCls(c.orig_top)}}">${{signed(c.orig_top)}} <span class="subtext">IC ${{c.orig_ic > 0 ? "+" : ""}}${{fmt(c.orig_ic, 3)}}</span></td>
        <td class="num font-mono ${{retCls(c.dl_top)}}">${{signed(c.dl_top)}}</td>
        <td class="num font-mono ${{retCls(c.mkt_ret)}}">${{signed(c.mkt_ret)}}</td>
        <td class="num font-mono ${{retCls(c.cal_spread)}}">${{signed(c.cal_spread)}}</td>
      </tr>
    `).join("");
    box.innerHTML = `
      <div class="subtoolbar">
        <strong>实盘逐日追踪账本（已记录 ${{l.tracked_days}} 个交易日快照）</strong>
        <span class="subtext" style="margin:0;">每日收市自动写入当日双模型预测，用于后续真实验证</span>
      </div>
      <div class="table-wrap">
        ${{rows ? `<table><thead><tr><th>区间</th><th>样本数</th><th>修正模型 Top 1/3</th><th>原始模型 Top 1/3</th><th>双低基准</th><th>全市场均价</th><th>修正多空超额</th></tr></thead><tbody>${{rows}}</tbody></table>` : `<div style="padding:28px;color:var(--muted);text-align:center;">实盘账本已升级保存今天（${{DATA.date}}）双模型快照；明日起将在此累积逐日实时检验记录。上方可切换“近 20 / 10 / 5 交易日”查看历史样本外对照数据。</div>`}}
      </div>
    `;
    return;
  }}

  const v = DATA.validation[currentHorizon];
  if (!v) {{
    box.innerHTML = `<div style="padding:24px;color:var(--muted);">该窗口暂无足够历史样本。</div>`;
    return;
  }}
  const c = v.cal, o = v.orig;
  const spreadDiff = (c.long_short_spread - o.long_short_spread).toFixed(2);
  const topDiff = (c.top_avg_ret - o.top_avg_ret).toFixed(2);
  const icDiff = (c.rank_ic - o.rank_ic).toFixed(3);

  const rowsHtml = v.rows.map(r => `
    <tr>
      <td>
        <strong>${{r.name}}</strong>
        <span class="subtext font-mono">${{r.ticker}} · ${{r.stock_name}} · 溢价 ${{fmt(r.prem_past, 1)}}%</span>
      </td>
      <td class="num font-mono">
        ¥${{fmt(r.past_price)}} → ¥${{fmt(r.now_price)}}
        <span class="subtext ${{retCls(r.actual_ret)}}">区间 ${{signed(r.actual_ret)}}</span>
      </td>
      <td class="num font-mono">
        ¥${{fmt(r.past_model_cal)}} <span class="${{r.underval_cal_past > 0 ? "underv" : "overv"}}">(${{signed(r.underval_cal_past)}})</span>
        <span class="subtext">${{r.cal_expect}}</span>
      </td>
      <td class="num font-mono">
        ¥${{fmt(r.past_model_orig)}} <span class="${{r.underval_orig_past > 0 ? "underv" : "overv"}}">(${{signed(r.underval_orig_past)}})</span>
        <span class="subtext">${{r.orig_expect}}</span>
      </td>
      <td>
        <span class="badge ${{r.hit_cal ? "tag-hit" : "tag-miss"}}">修正 ${{r.hit_cal ? "命中" : "偏离"}}</span>
        <span class="badge ${{r.hit_orig ? "tag-hit" : "tag-miss"}}" style="margin-left:4px;">原版 ${{r.hit_orig ? "命中" : "偏离"}}</span>
        <span class="subtext">相对中位 ${{signed(r.excess_vs_med)}}</span>
      </td>
    </tr>
  `).join("");

  box.innerHTML = `
    <div class="cmp-grid">
      <div class="cmp-box">
        <span class="title">🛡️ 修正防守模型 (${{v.horizon}} 日)</span>
        <div class="main font-mono ${{retCls(c.top_avg_ret)}}">Top 1/3: ${{signed(c.top_avg_ret)}}</div>
        <div class="detail font-mono">Top 10: ${{signed(c.top10_avg_ret)}} · Rank IC: ${{c.rank_ic > 0 ? "+" : ""}}${{fmt(c.rank_ic, 3)}}<br>多空超额 ${{signed(c.long_short_spread)}} · 命中率 ${{fmt(c.hit_rate, 1)}}%</div>
      </div>
      <div class="cmp-box">
        <span class="title">📐 原始论文模型 (${{v.horizon}} 日)</span>
        <div class="main font-mono ${{retCls(o.top_avg_ret)}}">Top 1/3: ${{signed(o.top_avg_ret)}}</div>
        <div class="detail font-mono">Top 10: ${{signed(o.top10_avg_ret)}} · Rank IC: ${{o.rank_ic > 0 ? "+" : ""}}${{fmt(o.rank_ic, 3)}}<br>多空超额 ${{signed(o.long_short_spread)}} · 命中率 ${{fmt(o.hit_rate, 1)}}%</div>
      </div>
      <div class="cmp-box">
        <span class="title">⚖️ 传统双低与市场基准</span>
        <div class="main font-mono ${{retCls(c.dl_avg_ret)}}">双低 1/3: ${{signed(c.dl_avg_ret)}}</div>
        <div class="detail font-mono">双低 Top 10: ${{signed(c.dl10_avg_ret)}} · 全市场: ${{signed(c.mkt_avg_ret)}}<br>修正组 vs 双低: ${{signed(c.alpha_vs_dl)}}</div>
      </div>
      <div class="cmp-box" style="background:#f5f8fd;">
        <span class="title">📈 修正后实证改善幅度</span>
        <div class="main font-mono" style="color:var(--blue);">IC ${{Number(icDiff) >= 0 ? "+" : ""}}${{icDiff}}</div>
        <div class="detail font-mono">低估组收益差: ${{Number(topDiff) >= 0 ? "+" : ""}}${{topDiff}}%<br>多空超额差: ${{Number(spreadDiff) >= 0 ? "+" : ""}}${{spreadDiff}}%</div>
      </div>
    </div>

    <div class="charts-grid">
      <div class="chart-panel">
        <div class="chart-head">
          <h3>分组真实收益对比 (${{v.past_date}} → ${{v.now_date}})</h3>
          <span>样本 ${{v.sample_size}} 只</span>
        </div>
        ${{renderBarsSVG(v)}}
      </div>
      <div class="chart-panel">
        <div class="chart-head">
          <h3>当时低估度 vs 后续真实涨跌散点</h3>
          <div class="tab-pills">
            <button class="tab-btn ${{scatterModel === "cal" ? "active" : ""}}" onclick="setScatter('cal')" style="padding:2px 8px;font-size:11px;">修正模型 (IC ${{c.rank_ic > 0 ? "+" : ""}}${{fmt(c.rank_ic, 3)}})</button>
            <button class="tab-btn ${{scatterModel === "orig" ? "active" : ""}}" onclick="setScatter('orig')" style="padding:2px 8px;font-size:11px;">原始模型 (IC ${{o.rank_ic > 0 ? "+" : ""}}${{fmt(o.rank_ic, 3)}})</button>
          </div>
        </div>
        ${{renderScatterSVG(v.rows, c.mkt_med_ret)}}
        <div class="scatter-tip" id="scatterTip">鼠标悬停圆点查看单券两套模型当时估价与真实表现</div>
      </div>
    </div>

    <div class="subtoolbar">
      <strong>逐券历史定价对照明细 (${{v.past_date}} 当时预测 → ${{v.now_date}} 实际验证)</strong>
      <span class="subtext" style="margin:0;">样本真实中位涨跌：${{signed(c.mkt_med_ret)}}（按修正模型当时低估度降序排列）</span>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>可转债标的 (含当时溢价率)</th>
            <th>区间价格变化与实际涨跌</th>
            <th>修正模型当时估值 (判断)</th>
            <th>原始模型当时估值 (判断)</th>
            <th>两模型命中对照</th>
          </tr>
        </thead>
        <tbody>${{rowsHtml}}</tbody>
      </table>
    </div>
  `;

  document.querySelectorAll("#scatterSvg circle").forEach(el => {{
    el.addEventListener("mouseenter", () => {{
      const r = v.rows[Number(el.dataset.idx)];
      if (r) {{
        document.getElementById("scatterTip").textContent =
          `${{r.name}} (${{r.ticker}})：当时市价 ¥${{fmt(r.past_price)}} | 修正估 ¥${{fmt(r.past_model_cal)}} (${{signed(r.underval_cal_past)}}) | 原版估 ¥${{fmt(r.past_model_orig)}} (${{signed(r.underval_orig_past)}}) → 现价 ¥${{fmt(r.now_price)}} (${{signed(r.actual_ret)}})`;
      }}
    }});
  }});
}}

function setScatter(m) {{
  scatterModel = m;
  renderValidation();
}}

function approxErf(x) {{
  const s = x >= 0 ? 1 : -1;
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}}

function doCalc() {{
  const S0 = parseFloat(document.getElementById("slider-s0").value);
  const C0 = parseFloat(document.getElementById("slider-c0").value);
  const Pmkt = parseFloat(document.getElementById("slider-pmkt").value);
  const p = parseFloat(document.getElementById("slider-p").value);
  const sig = parseFloat(document.getElementById("slider-sig").value);

  document.getElementById("txt-s0").textContent = "¥" + S0.toFixed(2);
  document.getElementById("txt-c0").textContent = "¥" + C0.toFixed(2);
  document.getElementById("txt-pmkt").textContent = "¥" + Pmkt.toFixed(2);
  document.getElementById("txt-p").textContent = (p * 100).toFixed(0) + "%";
  document.getElementById("txt-sig").textContent = (sig * 100).toFixed(0) + "%";

  const convVal = (100.0 / C0) * S0;
  const prem = ((Pmkt - convVal) / Math.max(0.01, convVal)) * 100;
  document.getElementById("res-conv").textContent = "¥" + convVal.toFixed(2);
  document.getElementById("res-prem-sub").textContent = "按市价溢价 " + (prem > 0 ? "+" : "") + prem.toFixed(1) + "%";

  const k0 = 1.30 * C0, ptr0 = 0.70 * C0;
  let title = "", desc = "";
  if (S0 > k0) {{
    title = "强赎提示区 (R1)";
    desc = "正股高于 130% 强赎线，转债强股性主导，修正模型在高价端会扣减滞后折价以防止追高。";
  }} else if (S0 > C0) {{
    title = "高弹性转股区 (R2)";
    desc = "正股高于转股价，转股期权处于实值区间，若市价与溢价率同时偏高，修正模型将适度提示安全垫不足。";
  }} else if (S0 > ptr0) {{
    title = "平衡抗跌区 (R3)";
    desc = "兼备纯债防守与向上期权；修正模型以 ¥103.5~¥106 真实债底为锚，避免高估虚值弹性。";
  }} else {{
    title = "下修博弈区 (R4)";
    desc = "跌破 70% 回售线。原模型按 p=" + (p * 100).toFixed(0) + "% 计算乐观下修弹性；修正模型按保守 p=40% 与滞后折价评估，防备公告慢导致的短线续跌。";
  }}
  document.getElementById("res-reg-title").textContent = title;
  document.getElementById("res-reg-text").textContent = desc;

  const T = 2.5, r = 0.025;
  const cdf = x => 0.5 * (1 + approxErf(x / Math.SQRT2));

  // Original approximation
  const d1 = (Math.log(S0 / C0) + (r + 0.5 * sig * sig) * T) / (sig * Math.sqrt(T));
  const d2 = d1 - sig * Math.sqrt(T);
  const bsOrig = (100.0 / C0) * (S0 * cdf(d1) - C0 * Math.exp(-r * T) * cdf(d2));
  const resetOrig = (S0 <= ptr0) ? (p * 12.5 * (1.0 - S0 / C0)) : (p * 3.5);
  const totalOrig = Math.max(convVal, Math.max(108.0 * Math.exp(-r * T) + bsOrig + resetOrig, 108.0 * 0.96));

  // Calibrated conservative approximation
  const sigCal = Math.min(sig, 0.35);
  const bCal = Math.min(106.0, 99.0 + 1.8 * T);
  const d1c = (Math.log(S0 / C0) + (r + 0.5 * sigCal * sigCal) * T) / (sigCal * Math.sqrt(T));
  const d2c = d1c - sigCal * Math.sqrt(T);
  const bsCal = (100.0 / C0) * (S0 * cdf(d1c) - C0 * Math.exp(-r * T) * cdf(d2c));
  const resetCal = (S0 <= ptr0) ? (0.40 * 12.5 * (1.0 - S0 / C0)) : (0.40 * 3.5);
  const rawCal = Math.max(convVal, Math.max(bCal * Math.exp(-r * T) + bsCal + resetCal, bCal * 0.96));
  const haircut = Math.min(0.22, 0.25 * Math.max(0.0, (Pmkt - 115.0) / 100.0) + 0.12 * Math.max(0.0, (prem - 25.0) / 100.0));
  const totalCal = rawCal * (1.0 - haircut);

  document.getElementById("res-orig").textContent = "¥" + totalOrig.toFixed(2);
  document.getElementById("res-orig-sub").textContent = "vs 市价 " + (((totalOrig - Pmkt) / Pmkt) * 100 >= 0 ? "+" : "") + (((totalOrig - Pmkt) / Pmkt) * 100).toFixed(1) + "%";
  document.getElementById("res-cal").textContent = "¥" + totalCal.toFixed(2);
  document.getElementById("res-cal-sub").textContent = "折价 -" + (haircut * 100).toFixed(1) + "% · vs 市价 " + (((totalCal - Pmkt) / Pmkt) * 100 >= 0 ? "+" : "") + (((totalCal - Pmkt) / Pmkt) * 100).toFixed(1) + "%";
}}

document.querySelectorAll("#portfolioPills [data-port]").forEach(btn => {{
  btn.addEventListener("click", () => {{
    currentPort = btn.dataset.port;
    document.querySelectorAll("#portfolioPills .tab-btn").forEach(b => b.classList.toggle("active", b === btn));
    renderPortfolio();
  }});
}});

document.querySelectorAll("#horizonPills [data-horizon]").forEach(btn => {{
  btn.addEventListener("click", () => {{
    currentHorizon = btn.dataset.horizon;
    document.querySelectorAll("#horizonPills .tab-btn").forEach(b => b.classList.toggle("active", b === btn));
    renderValidation();
  }});
}});

["liveSearch", "liveRegion", "liveSort"].forEach(id => document.getElementById(id).addEventListener("input", renderPortfolio));
["slider-s0", "slider-c0", "slider-pmkt", "slider-p", "slider-sig"].forEach(id => document.getElementById(id).addEventListener("input", doCalc));

renderPortfolio();
renderValidation();
doCalc();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    generate_page_and_data()
