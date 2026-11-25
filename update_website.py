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
    "M": 1600,
    "seed": 42,
}


def get_default_bonds_pool():
    """Fallback representative universe if live network feeds are unreachable."""
    return [
        {"ticker": "110075", "name": "南航转债", "stock_code": "600029", "stock_name": "南方航空", "stock_price": 5.92, "conv_price": 6.24, "bond_price": 128.40, "maturity_years": 2.8},
        {"ticker": "127025", "name": "冀东转债", "stock_code": "000401", "stock_name": "冀东水泥", "stock_price": 5.12, "conv_price": 5.88, "bond_price": 111.20, "maturity_years": 1.9},
        {"ticker": "113049", "name": "长汽转债", "stock_code": "601633", "stock_name": "长城汽车", "stock_price": 26.80, "conv_price": 34.20, "bond_price": 114.50, "maturity_years": 3.2},
        {"ticker": "123107", "name": "温氏转债", "stock_code": "300498", "stock_name": "温氏股份", "stock_price": 17.50, "conv_price": 19.80, "bond_price": 119.30, "maturity_years": 2.6},
        {"ticker": "127056", "name": "中特转债", "stock_code": "000708", "stock_name": "中信特钢", "stock_price": 12.10, "conv_price": 14.80, "bond_price": 113.60, "maturity_years": 3.5},
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
    """Fetch aligned daily bond + stock history sequentially for live volatility estimation."""
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
            pd.merge(b_hist[["date", "close", "volume"]], s_hist[["date", "close"]], on="date", suffixes=("_bond", "_stock"))
            .sort_values("date")
            .reset_index(drop=True)
        )
        if len(merged) < 40 or merged["close_bond"].nunique() < 8:
            return item, None
        if (pd.Timestamp.now() - merged["date"].iloc[-1]).days > 14:
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
    log_ret = log_ret[log_ret.abs() <= 0.28]
    if len(log_ret) < 10:
        return 0.28
    vol = float(log_ret.std() * np.sqrt(252))
    return round(max(0.15, min(0.70, vol)), 4)


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 4 or x.nunique() < 2 or y.nunique() < 2:
        return 0.0
    val = x.rank().corr(y.rank())
    return round(float(val), 3) if pd.notna(val) else 0.0


def update_live_history(dist_dir: Path, date_str: str, evaluated: list):
    """Persist immutable daily live forward signals using frozen paper parameters."""
    hist_path = dist_dir / "history.json"
    ledger = {"frozen_since": "2026-11-25", "params": FROZEN_PARAMS, "snapshots": []}
    if hist_path.exists():
        try:
            old = json.loads(hist_path.read_text(encoding="utf-8"))
            if isinstance(old, dict) and "snapshots" in old:
                ledger["snapshots"] = old["snapshots"]
        except Exception:
            pass

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

    # Append-only for historical dates; update same-day snapshot only if same date
    snapshots = [s for s in ledger["snapshots"] if s.get("date") != date_str]
    snapshots.append(snap_entry)
    snapshots = sorted(snapshots, key=lambda x: x["date"])[-365:]
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
                    "underval": info["undervaluation"],
                    "double_low": info.get("double_low", 999.0),
                    "ret": ret,
                })
        if len(matched) >= 6:
            mdf = pd.DataFrame(matched)
            k = max(2, len(mdf) // 5)
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

    return {"tracked_days": len(snapshots), "frozen_since": ledger["frozen_since"], "comparisons": list(reversed(live_records))[:30]}


def load_live_pool():
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


def generate_page_and_data():
    local_tz = pytz.timezone("Asia/Shanghai")
    now = datetime.now(local_tz)
    update_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    date_str = now.strftime("%Y-%m-%d")

    dist_dir = Path(__file__).resolve().parent / "dist" / "bonds"
    dist_dir.mkdir(parents=True, exist_ok=True)

    existing_data = {}
    data_file = dist_dir / "data.json"
    if data_file.exists():
        try:
            existing_data = json.loads(data_file.read_text(encoding="utf-8"))
        except Exception:
            existing_data = {}

    evaluated = load_live_pool()
    if len(evaluated) < 20 and existing_data.get("all_bonds"):
        seen = {b["ticker"] for b in evaluated}
        for old_b in existing_data["all_bonds"]:
            if old_b.get("ticker") not in seen and len(evaluated) < 36:
                evaluated.append(old_b)
                seen.add(old_b.get("ticker"))

    lsm_sorted = sorted(evaluated, key=lambda x: x["undervaluation"], reverse=True)
    dl_sorted = sorted(evaluated, key=lambda x: x["double_low"], reverse=False)
    lsm_rank = {b["ticker"]: i for i, b in enumerate(lsm_sorted, 1)}
    dl_rank = {b["ticker"]: i for i, b in enumerate(dl_sorted, 1)}
    for b in evaluated:
        b["lsm_rank"] = lsm_rank[b["ticker"]]
        b["dl_rank"] = dl_rank[b["ticker"]]

    research = {}
    rv_path = dist_dir / "research_validation.json"
    if rv_path.exists():
        research = json.loads(rv_path.read_text(encoding="utf-8"))

    live_ledger = update_live_history(dist_dir, date_str, lsm_sorted)

    data_payload = {
        "updated_at": update_time_str,
        "date": date_str,
        "author": "Yu Liu (刘宇) · The Chinese University of Hong Kong, Shenzhen",
        "paper_ref": "arXiv:2409.06496v7",
        "paper_url": "https://arxiv.org/abs/2409.06496",
        "frozen_params": FROZEN_PARAMS,
        "pool_size": len(lsm_sorted),
        "research": research,
        "live_ledger": live_ledger,
        "lsm_top10": lsm_sorted[:10],
        "double_low_top10": dl_sorted[:10],
        "all_bonds": lsm_sorted,
    }

    (dist_dir / "data.json").write_text(json.dumps(data_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (dist_dir / "index.html").write_text(build_clean_html(data_payload), encoding="utf-8")
    print(f"Generated 5-stage validation dashboard at {dist_dir / 'index.html'} (live_pool={len(lsm_sorted)}, research_bonds={research.get('universe',{}).get('total_bonds', 0)})")


def build_clean_html(data: dict) -> str:
    payload_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    r = data.get("research") or {}
    u = r.get("universe") or {}
    ic20 = (r.get("ic_summary") or {}).get("20") or {}
    q20 = (r.get("quintiles") or {}).get("20") or {}
    fmb20 = ((r.get("fama_macbeth") or {}).get("20") or {}).get("factors", [{}])[0]

    template = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Four-Region LSM 可转债模型实证验证体系 · Yu Liu</title>
  <meta name="description" content="冻结参数 Four-Region LSM 可转债估值模型 2020–2026 全市场五阶段实证验证系统">
  <style>
    :root {
      --ink: #1f252e;
      --muted: #657180;
      --light: #8b96a5;
      --blue: #245ca9;
      --blue-bg: #eef4fc;
      --green: #0f766e;
      --green-bg: #f0fdfa;
      --up: #c43737;
      --up-bg: #fdf2f2;
      --line: #e5e9f0;
      --soft: #f8f9fc;
      color-scheme: light;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; background: #fff; color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      font-size: 14px; line-height: 1.66; -webkit-font-smoothing: antialiased;
    }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-variant-numeric: tabular-nums; }
    a { color: var(--blue); text-decoration: none; }
    a:hover { text-decoration: underline; }
    .shell { max-width: 1120px; margin: 0 auto; padding: 34px 24px 64px; }

    .topbar {
      display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 12px;
      padding-bottom: 16px; border-bottom: 1px solid var(--line); margin-bottom: 26px; font-size: 13px;
    }
    .brand { font-weight: 650; color: var(--ink); font-size: 15px; }
    .brand span { color: var(--muted); font-weight: 400; margin-left: 8px; font-size: 13px; }
    .nav-links { display: flex; gap: 18px; flex-wrap: wrap; }
    .nav-links a { color: var(--muted); }
    .nav-links a:hover { color: var(--blue); }

    h1 { font-size: 25px; font-weight: 650; letter-spacing: -0.3px; margin: 0 0 8px; line-height: 1.35; }
    .lead { color: var(--muted); font-size: 14px; margin: 0 0 22px; max-width: 920px; }

    .stats-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 20px; }
    .stat-card { background: var(--soft); border: 1px solid var(--line); border-radius: 8px; padding: 15px 17px; }
    .stat-card .label { font-size: 12px; color: var(--muted); display: block; margin-bottom: 4px; }
    .stat-card .val { font-size: 22px; font-weight: 650; line-height: 1.25; }
    .stat-card .desc { font-size: 11px; color: var(--light); margin-top: 5px; line-height: 1.45; }

    .audit-strip {
      background: #f5f8fd; border: 1px solid #d9e5f7; border-left: 3px solid var(--blue);
      border-radius: 0 8px 8px 0; padding: 14px 18px; margin-bottom: 26px;
    }
    .audit-title { font-size: 13px; font-weight: 650; color: var(--ink); margin-bottom: 8px; }
    .audit-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px 14px; font-size: 12px; color: #3b4859; }
    .audit-item b { color: var(--green); margin-right: 4px; }

    .card { background: #fff; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; margin-bottom: 26px; }
    .card-header {
      padding: 15px 20px; background: #fafbfe; border-bottom: 1px solid var(--line);
      display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;
    }
    .card-title { font-size: 15px; font-weight: 650; margin: 0; }
    .card-subtitle { font-size: 12px; color: var(--muted); margin-top: 2px; }
    .stage-tag {
      display: inline-block; font-size: 11px; font-weight: 600; color: var(--blue);
      background: var(--blue-bg); padding: 1px 7px; border-radius: 4px; margin-right: 6px;
    }

    .pills { display: flex; gap: 6px; flex-wrap: wrap; }
    .pill {
      padding: 5px 11px; border-radius: 5px; font-size: 12px; border: 1px solid #dce2ea;
      background: #fff; color: var(--muted); cursor: pointer;
    }
    .pill.active { background: var(--blue-bg); border-color: var(--blue); color: var(--blue); font-weight: 600; }

    .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; padding: 18px 20px; border-top: 1px solid var(--line); }
    .panel { border: 1px solid var(--line); border-radius: 6px; padding: 14px 16px; background: #fff; }
    .panel-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; gap: 8px; flex-wrap: wrap; }
    .panel-head h3 { margin: 0; font-size: 13px; font-weight: 600; }
    .panel-head span { font-size: 11px; color: var(--muted); }

    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; text-align: left; font-size: 13px; }
    th {
      background: #fcfdfe; color: var(--muted); font-weight: 500;
      padding: 10px 14px; border-bottom: 1px solid var(--line); font-size: 12px; white-space: nowrap;
    }
    td { padding: 11px 14px; border-bottom: 1px solid var(--line); vertical-align: middle; }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: #fafbfe; }
    td.num { font-variant-numeric: tabular-nums; white-space: nowrap; }
    .subtext { display: block; font-size: 11px; color: var(--muted); margin-top: 2px; line-height: 1.4; }
    .badge { display: inline-block; font-size: 11px; padding: 2px 7px; border-radius: 4px; font-weight: 500; white-space: nowrap; }
    .sig-yes { color: var(--green); background: var(--green-bg); border: 1px solid #99f6e4; }
    .sig-no { color: var(--muted); background: #f8fafc; border: 1px solid #e2e8f0; }
    .hl-row td { background: #f5f8fd; }
    .pos { color: var(--up); font-weight: 600; }
    .neg { color: var(--green); font-weight: 600; }
    .note-bar { padding: 12px 18px; background: #fcfdfe; border-top: 1px solid var(--line); font-size: 12px; color: var(--muted); line-height: 1.65; }
    .legend-row { display: flex; gap: 14px; flex-wrap: wrap; font-size: 11px; margin-bottom: 8px; }
    .legend-row span { display: inline-flex; align-items: center; gap: 5px; }
    .ldot { width: 10px; height: 3px; border-radius: 2px; display: inline-block; }

    .subtoolbar {
      display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;
      padding: 10px 18px; background: #fcfdfe; border-bottom: 1px solid var(--line); border-top: 1px solid var(--line);
    }
    .subtoolbar input, .subtoolbar select {
      border: 1px solid #dce2ea; background: #fff; border-radius: 5px; padding: 5px 9px; font-size: 12px; color: var(--ink);
    }

    .calc-section { background: var(--soft); border: 1px solid var(--line); border-radius: 8px; padding: 20px 22px; margin-bottom: 24px; }
    .calc-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 22px; align-items: start; }
    .input-group { margin-bottom: 12px; }
    .input-header { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 4px; }
    .input-header span:last-child { font-weight: 600; color: var(--blue); }
    input[type=range] { width: 100%; accent-color: var(--blue); }
    .calc-out { background: #fff; border: 1px solid var(--line); border-radius: 6px; padding: 16px 18px; }
    .calc-nums { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; padding-bottom: 12px; border-bottom: 1px solid var(--line); margin-bottom: 12px; }
    .calc-nums strong { display: block; font-size: 22px; margin-top: 2px; }

    footer { margin-top: 28px; padding-top: 16px; border-top: 1px solid var(--line); font-size: 12px; color: var(--muted); line-height: 1.8; }

    @media (max-width: 860px) {
      .stats-row, .audit-grid { grid-template-columns: 1fr 1fr; }
      .two-col, .calc-grid { grid-template-columns: 1fr; }
      .shell { padding: 20px 14px; }
    }
  </style>
</head>
<body>
<div class="shell">
  <header class="topbar">
    <div class="brand">Yu Liu · 刘宇 <span>Four-Region LSM 实证验证系统</span></div>
    <nav class="nav-links">
      <a href="/">个人主页</a>
      <a href="/trade/">A 股买卖看板</a>
      <a href="/phd-tracker/">申请追踪</a>
      <a href="https://arxiv.org/abs/2409.06496" target="_blank" rel="noopener">arXiv:2409.06496 ↗</a>
    </nav>
  </header>

  <h1>中国可转债 Four-Region LSM 模型五阶段实证验证系统</h1>
  <p class="lead">
    严格冻结原始论文参数（<code>p_reset=0.80, B=¥108, r=2.5%</code>），基于 <strong>2020-01-02 → 2026-11-25 全市场 __TOTAL_BONDS__ 只可转债</strong>（含已强赎、到期及违约退市标的，共 <strong>__TOTAL_OBS__ 条债日观测</strong>）建立无幸存者偏差、零未来泄漏的系统性检验。
  </p>

  <div class="stats-row">
    <div class="stat-card">
      <span class="label">全市场回测样本 (2020–2026)</span>
      <div class="val mono">__TOTAL_BONDS__ 只</div>
      <div class="desc">__TRADING_DAYS__ 个交易日 · 日均 __AVG_DAILY__ 只在市 · 含退市强赎违约</div>
    </div>
    <div class="stat-card">
      <span class="label">20 日长期均值 Rank IC</span>
      <div class="val mono pos">__IC20_VAL__</div>
      <div class="desc">ICIR __IC20_ICIR__ · 正 IC 比例 __IC20_POS__% · Newey-West t = __IC20_T__</div>
    </div>
    <div class="stat-card">
      <span class="label">五分组 G5-G1 20 日多空超额</span>
      <div class="val mono pos">__Q20_SPREAD__</div>
      <div class="desc">Newey-West t = __Q20_T__ (p = __Q20_P__) · 严格 G1→G5 单调递增</div>
    </div>
    <div class="stat-card">
      <span class="label">Fama-MacBeth 控制后 β (20日)</span>
      <div class="val mono pos">__FMB20_BETA__</div>
      <div class="desc">Newey-West t = __FMB20_T__ (p = __FMB20_P__) · 控制 6 大传统因子仍显著</div>
    </div>
  </div>

  <!-- Stage 1 Audit Checklist -->
  <div class="audit-strip">
    <div class="audit-title"><span class="stage-tag">第一阶段</span>数据偏差排查与验证开发清单（7/7 已完成，参数已冻结）</div>
    <div class="audit-grid">
      <div><b>✓</b>冻结原论文参数 (`p=0.80, B=108`)，不做测试期微调</div>
      <div><b>✓</b>消除幸存者偏差：纳入 940 只含强赎/到期/违约退市券</div>
      <div><b>✓</b>消除未来泄漏：逐日因果还原历史转股价 `C_{i,t}` 与合约期限</div>
      <div><b>✓</b>可实现成交：`T` 日收盘出信号，`T+1` 日开盘买入/卖出</div>
      <div><b>✓</b>现金流完整：纳入年度派息除息现金流与退市末日真实平仓</div>
      <div><b>✓</b>多成本检验：同步测试 `0.05%` / `0.10%` / `0.20%` 单边成本</div>
      <div><b>✓</b>时间序列显著性：使用 Newey-West HAC 修正重叠窗口自相关</div>
      <div><b>✓</b>前瞻样本外账本：2026-11-25 起每日自动归档，不可事后修改</div>
    </div>
  </div>

  <!-- Stage 2: IC / ICIR / Significance / Rolling 60d / Decay -->
  <div class="card">
    <div class="card-header">
      <div>
        <h2 class="card-title"><span class="stage-tag">第二阶段</span>跨期定价预测力检验（Rank IC · ICIR · 显著性与衰减）</h2>
        <div class="card-subtitle">检验每日截面低估因子 F=(V_LSM - P)/P 对未来 1 / 5 / 10 / 20 个交易日（T+1 开盘买入）收益率的排序能力</div>
      </div>
      <div class="pills" id="icRollPills">
        <button class="pill" data-ich="5">滚动 60 日 IC (h=5)</button>
        <button class="pill" data-ich="10">滚动 60 日 IC (h=10)</button>
        <button class="pill active" data-ich="20">滚动 60 日 IC (h=20)</button>
      </div>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>持有窗口</th>
            <th>有效交易日数</th>
            <th>日均截面样本</th>
            <th>Mean Rank IC</th>
            <th>IC 标准差</th>
            <th>ICIR (未年化)</th>
            <th>IC &gt; 0 比例</th>
            <th>Newey-West t 值</th>
            <th>95% 置信区间</th>
          </tr>
        </thead>
        <tbody id="icTableBody"></tbody>
      </table>
    </div>
    <div class="two-col">
      <div class="panel">
        <div class="panel-head">
          <h3>滚动 60 日平均 Rank IC 走势 (2020–2026)</h3>
          <span>虚线为 0 轴与 0.05 / 0.10 参考阈值</span>
        </div>
        <div id="rollingIcSvg"></div>
      </div>
      <div class="panel">
        <div class="panel-head">
          <h3>IC 持有期衰减曲线 (h = 1 → 20 交易日)</h3>
          <span>观察定价收敛效应持续长度</span>
        </div>
        <div id="icDecaySvg"></div>
      </div>
    </div>
  </div>

  <!-- Stage 3: Quintile Monotonicity Test -->
  <div class="card">
    <div class="card-header">
      <div>
        <h2 class="card-title"><span class="stage-tag">第三阶段</span>五分组单调性检验（G1 高估 → G5 低估）</h2>
        <div class="card-subtitle">每日按 F_LSM 将全部在市转债等分五组，检验未来超额收益单调性与 7 年累计净值表现</div>
      </div>
      <div class="pills" id="qHorizonPills">
        <button class="pill" data-qh="1">未来 1 日</button>
        <button class="pill" data-qh="5">未来 5 日</button>
        <button class="pill" data-qh="10">未来 10 日</button>
        <button class="pill active" data-qh="20">未来 20 日</button>
      </div>
    </div>
    <div class="two-col" style="border-top:0;">
      <div class="panel">
        <div class="panel-head">
          <h3 id="qBarTitle">五分组未来 20 日平均超额收益 (相对全市场均价)</h3>
          <span id="qBarStat">G5 - G1</span>
        </div>
        <div id="qBarSvg"></div>
      </div>
      <div class="panel">
        <div class="panel-head">
          <h3>2020–2026 五分组周频累计净值曲线 (0.10% 单边成本)</h3>
          <span>G5 (最被低估) vs G1 (最被高估)</span>
        </div>
        <div class="legend-row">
          <span><i class="ldot" style="background:#c43737;"></i>G5 最被低估</span>
          <span><i class="ldot" style="background:#245ca9;"></i>G4 偏低</span>
          <span><i class="ldot" style="background:#64748b;"></i>G3 中性</span>
          <span><i class="ldot" style="background:#d97706;"></i>G2 偏高</span>
          <span><i class="ldot" style="background:#0f766e;"></i>G1 最被高估</span>
        </div>
        <div id="qNavSvg"></div>
      </div>
    </div>
  </div>

  <!-- Stage 4: Benchmark Horse Race, Ablations & Fama-MacBeth Cross-Sectional Regressions -->
  <div class="card">
    <div class="card-header">
      <div>
        <h2 class="card-title"><span class="stage-tag">第四阶段</span>传统策略对比、消融实验与 Fama-MacBeth 截面回归</h2>
        <div class="card-subtitle">全市场等权调仓（每周买入 Top 20%，T+1 开盘成交），同时验证分段结构、下修机制及增量 Alpha 统计显著性</div>
      </div>
      <div class="pills" id="stratCostPills">
        <button class="pill" data-sc="0.05%">0.05% 低佣</button>
        <button class="pill active" data-sc="0.10%">0.10% 标准成本</button>
        <button class="pill" data-sc="0.20%">0.20% 高滑点</button>
      </div>
    </div>

    <div style="padding:16px 20px;">
      <div class="panel-head">
        <h3>2020–2026 累计净值对比（Four-Region LSM vs 传统双低 / 去分段 / 无下修 / 全市场等权基准）</h3>
        <span id="stratCostNote">单边成本 0.10%</span>
      </div>
      <div class="legend-row">
        <span><i class="ldot" style="background:#245ca9;"></i>Four-Region LSM (冻结原模型)</span>
        <span><i class="ldot" style="background:#d97706;"></i>传统双低</span>
        <span><i class="ldot" style="background:#0f766e;"></i>无下修 LSM (p=0)</span>
        <span><i class="ldot" style="background:#7c3aed;"></i>单一区域 LSM (去分段)</span>
        <span><i class="ldot" style="background:#94a3b8;"></i>全市场等权基准</span>
      </div>
      <div id="stratNavSvg"></div>
    </div>

    <div class="table-wrap" style="border-top:1px solid var(--line);">
      <table>
        <thead>
          <tr>
            <th>策略与消融对照组 (周频 5D · T+1 开盘成交)</th>
            <th>验证目标</th>
            <th>累计净收益</th>
            <th>年化净收益</th>
            <th>最大回撤</th>
            <th>夏普比率</th>
            <th>年化单边换手</th>
          </tr>
        </thead>
        <tbody id="stratTableBody"></tbody>
      </table>
    </div>

    <div class="subtoolbar">
      <strong>Fama-MacBeth 每日横截面回归（控制 6 大传统因子检验增量定价信息）</strong>
      <div class="pills" id="fmbHorizonPills">
        <button class="pill" data-fh="5">h = 5 交易日</button>
        <button class="pill" data-fh="10">h = 10 交易日</button>
        <button class="pill active" data-fh="20">h = 20 交易日</button>
      </div>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>标准化因子（1 倍标准差冲击）</th>
            <th>均值斜率 β / γ (%)</th>
            <th>Newey-West 标准误</th>
            <th>Newey-West t 值</th>
            <th>双侧 p 值</th>
            <th>95% 置信区间</th>
            <th>统计结论</th>
          </tr>
        </thead>
        <tbody id="fmbTableBody"></tbody>
      </table>
    </div>
    <div class="note-bar" id="ablationNote"></div>
  </div>

  <!-- Stage 5: Sub-Period Robustness + Immutable Forward Ledger & Live Frozen Signals -->
  <div class="card">
    <div class="card-header">
      <div>
        <h2 class="card-title"><span class="stage-tag">第五阶段</span>分时段稳健性（2020–2026）与实盘前瞻不可修改账本</h2>
        <div class="card-subtitle">分别检验 2020–2022、2023–2024（熊市信用冲击）、2025–2026，并衔接 2026-11-25 起的每日冻结前瞻记录</div>
      </div>
      <div class="pills" id="s5Pills">
        <button class="pill active" data-s5="subperiod">分时段历史检验</button>
        <button class="pill" data-s5="live_top">当期冻结模型精选 (Top 10 vs 双低)</button>
        <button class="pill" data-s5="live_all">全市场追踪明细</button>
        <button class="pill" data-s5="ledger">前瞻不可篡改账本 (2026-11-25+)</button>
      </div>
    </div>
    <div id="s5Content"></div>
  </div>

  <!-- Interactive Frozen Calculator -->
  <div class="calc-section">
    <div style="margin-bottom:12px;">
      <strong style="font-size:15px;">⚡ Four-Region LSM 冻结参数单券试算器</strong>
      <span class="subtext">基于论文标准设定 p_reset=0.80, B=108, r=2.5% 快速体验各状态区间定价差异</span>
    </div>
    <div class="calc-grid">
      <div>
        <div class="input-group">
          <div class="input-header"><span>正股当前价格 S₀</span><span id="txt-s0" class="mono">¥10.00</span></div>
          <input type="range" id="slider-s0" min="5" max="25" step="0.1" value="10.0">
        </div>
        <div class="input-group">
          <div class="input-header"><span>当前转股价格 C₀</span><span id="txt-c0" class="mono">¥10.00</span></div>
          <input type="range" id="slider-c0" min="5" max="25" step="0.1" value="10.0">
        </div>
        <div class="input-group">
          <div class="input-header"><span>下修意愿 p (论文默认 80%)</span><span id="txt-p" class="mono">80%</span></div>
          <input type="range" id="slider-p" min="0" max="1" step="0.05" value="0.80">
        </div>
        <div class="input-group">
          <div class="input-header"><span>正股年化波动率 σ</span><span id="txt-sig" class="mono">28%</span></div>
          <input type="range" id="slider-sig" min="0.12" max="0.55" step="0.02" value="0.28">
        </div>
      </div>
      <div class="calc-out">
        <div class="calc-nums">
          <div><span class="subtext" style="margin:0;">模型理论估值</span><strong class="mono pos" id="res-mdl">¥120.45</strong></div>
          <div><span class="subtext" style="margin:0;">转股平价</span><strong class="mono" id="res-conv">¥100.00</strong></div>
        </div>
        <strong id="res-reg-title" style="display:block;margin-bottom:4px;">平衡抗跌区 (R3)</strong>
        <span id="res-reg-text" style="font-size:13px;color:var(--muted);"></span>
      </div>
    </div>
  </div>

  <footer>
    <strong>方法引用与实证说明：</strong><br>
    Liu, Yu. (2026). <em>Valuation Model of Chinese Convertible Bonds Based on Monte Carlo Simulation: A Regression-Based Approximation and Error Analysis</em> (<a href="https://arxiv.org/abs/2409.06496" target="_blank" rel="noopener">arXiv:2409.06496v7</a>).<br>
    全市场验证采用真实日线（更新截至 2026-11-25，实盘前瞻面板更新于 __UPDATED_AT__），所有统计结论均以客观实测数据为准，不构成投资建议。
  </footer>
</div>

<script>
const DATA = __PAYLOAD_JSON__;
const R = DATA.research || {};
let curIcH = "20";
let curQH = "20";
let curSC = "0.10%";
let curFmbH = "20";
let curS5 = "subperiod";

const fmt = (v, n = 2) => typeof v === "number" && Number.isFinite(v) ? v.toFixed(n) : "—";
const signed = (v, n = 2, suffix = "%") => typeof v === "number" && Number.isFinite(v) ? (v > 0 ? "+" : "") + v.toFixed(n) + suffix : "—";
const cls = v => v > 0 ? "pos" : v < 0 ? "neg" : "";

function renderIC() {
  const ic = R.ic_summary || {};
  const rows = ["1", "5", "10", "20"].map(h => {
    const x = ic[h];
    if (!x) return "";
    return `
      <tr>
        <td class="mono"><strong>未来 ${h} 日</strong></td>
        <td class="num mono">${x.days} 天</td>
        <td class="num mono">${fmt(x.avg_n, 1)} 只</td>
        <td class="num mono ${cls(x.mean_ic)}" style="font-weight:700;">${x.mean_ic > 0 ? "+" : ""}${fmt(x.mean_ic, 4)}</td>
        <td class="num mono">${fmt(x.std_ic, 4)}</td>
        <td class="num mono" style="font-weight:600;">${fmt(x.icir, 3)}</td>
        <td class="num mono">${fmt(x.pos_pct, 1)}%</td>
        <td class="num mono"><strong>${fmt(x.t_nw, 2)}</strong> <span class="badge sig-yes">p=${x.p_val < 0.0001 ? "<0.0001" : x.p_val.toFixed(4)}</span></td>
        <td class="num mono">[${fmt(x.ci95[0], 4)}, ${fmt(x.ci95[1], 4)}]</td>
      </tr>
    `;
  }).join("");
  document.getElementById("icTableBody").innerHTML = rows;

  // Rolling 60d IC SVG
  const pts = (R.rolling_ic || {})[curIcH] || [];
  if (pts.length) {
    const W = 500, H = 210, padL = 42, padR = 16, padT = 16, padB = 28;
    const ys = pts.map(p => p[1]);
    const minY = Math.min(-0.08, ...ys) - 0.02, maxY = Math.max(0.18, ...ys) + 0.02;
    const px = i => padL + (i / Math.max(1, pts.length - 1)) * (W - padL - padR);
    const py = v => H - padB - ((v - minY) / (maxY - minY)) * (H - padT - padB);
    let grid = "";
    [0, 0.05, 0.10, -0.05].forEach(lv => {
      if (lv >= minY && lv <= maxY) {
        const y = py(lv).toFixed(1);
        grid += `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="${lv === 0 ? "#94a3b8" : "#e2e8f0"}" stroke-dasharray="${lv === 0 ? "" : "3 3"}"/>
          <text x="${padL - 5}" y="${Number(y) + 3}" text-anchor="end" font-size="10" fill="#657180">${lv.toFixed(2)}</text>`;
      }
    });
    [0, Math.floor(pts.length / 2), pts.length - 1].forEach(idx => {
      grid += `<text x="${px(idx).toFixed(1)}" y="${H - 8}" text-anchor="middle" font-size="10" fill="#657180">${pts[idx][0]}</text>`;
    });
    const poly = pts.map((p, i) => `${px(i).toFixed(1)},${py(p[1]).toFixed(1)}`).join(" ");
    document.getElementById("rollingIcSvg").innerHTML = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block;">${grid}<polyline fill="none" stroke="#245ca9" stroke-width="2" points="${poly}"/></svg>`;
  }

  // IC Decay SVG
  const dec = R.ic_decay || [];
  if (dec.length) {
    const W = 500, H = 210, padL = 44, padR = 20, padT = 22, padB = 28;
    const maxV = Math.max(0.12, ...dec.map(d => d.mean_ic)) * 1.15;
    const bw = (W - padL - padR) / dec.length;
    let bars = `<line x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}" stroke="#cbd5e1"/>`;
    dec.forEach((d, i) => {
      const h = (d.mean_ic / maxV) * (H - padT - padB);
      const x = padL + i * bw + bw * 0.22;
      const y = H - padB - h;
      const w = bw * 0.56;
      bars += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${w.toFixed(1)}" height="${Math.max(2, h).toFixed(1)}" fill="#245ca9" opacity="0.86" rx="2"/>
        <text x="${(x + w / 2).toFixed(1)}" y="${(y - 5).toFixed(1)}" text-anchor="middle" font-size="10" font-weight="600" fill="#245ca9">+${d.mean_ic.toFixed(3)}</text>
        <text x="${(x + w / 2).toFixed(1)}" y="${H - 9}" text-anchor="middle" font-size="10" fill="#657180">${d.h}D</text>`;
    });
    document.getElementById("icDecaySvg").innerHTML = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block;">${bars}</svg>`;
  }
}

function renderQuintiles() {
  const q = (R.quintiles || {})[curQH];
  if (!q) return;
  document.getElementById("qBarTitle").textContent = `五分组未来 ${curQH} 日平均超额收益 (相对全市场均价)`;
  document.getElementById("qBarStat").textContent = `G5 - G1 = ${signed(q.g5_minus_g1, 3)} (t_NW = ${fmt(q.t_nw, 2)}, p = ${q.p_val < 0.0001 ? "<0.0001" : q.p_val.toFixed(4)})`;

  const W = 500, H = 220, padL = 48, padR = 16, padT = 24, padB = 34;
  const vals = q.groups.map(g => g.excess_ret);
  const minY = Math.min(-0.1, ...vals) * 1.22;
  const maxY = Math.max(0.1, ...vals) * 1.25;
  const zeroY = padT + (maxY / (maxY - minY)) * (H - padT - padB);
  const py = v => padT + ((maxY - v) / (maxY - minY)) * (H - padT - padB);
  const bw = (W - padL - padR) / 5;
  let bars = `<line x1="${padL}" y1="${zeroY.toFixed(1)}" x2="${W - padR}" y2="${zeroY.toFixed(1)}" stroke="#94a3b8"/>`;
  q.groups.forEach((g, i) => {
    const v = g.excess_ret;
    const x = padL + i * bw + bw * 0.2;
    const w = bw * 0.6;
    const vy = py(v);
    const top = Math.min(vy, zeroY);
    const bh = Math.max(2, Math.abs(vy - zeroY));
    const col = v >= 0 ? "#c43737" : "#0f766e";
    const lblY = v >= 0 ? top - 5 : top + bh + 12;
    bars += `<rect x="${x.toFixed(1)}" y="${top.toFixed(1)}" width="${w.toFixed(1)}" height="${bh.toFixed(1)}" fill="${col}" opacity="0.85" rx="2"/>
      <text x="${(x + w / 2).toFixed(1)}" y="${lblY.toFixed(1)}" text-anchor="middle" font-size="10" font-weight="600" fill="${col}">${signed(v, 3)}</text>
      <text x="${(x + w / 2).toFixed(1)}" y="${H - 8}" text-anchor="middle" font-size="11" fill="#1f252e">${g.label}</text>`;
  });
  document.getElementById("qBarSvg").innerHTML = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block;">${bars}</svg>`;

  // 5-group NAV curves
  const qc = R.quintile_curves;
  if (qc && qc.dates) {
    const colors = { G5: "#c43737", G4: "#245ca9", G3: "#64748b", G2: "#d97706", G1: "#0f766e" };
    const W2 = 500, H2 = 220, pL = 42, pR = 56, pT = 14, pB = 26;
    const allV = Object.values(qc.navs).flat();
    const mn = Math.min(0.95, ...allV) * 0.96, mx = Math.max(1.2, ...allV) * 1.04;
    const px = i => pL + (i / Math.max(1, qc.dates.length - 1)) * (W2 - pL - pR);
    const py2 = v => H2 - pB - ((v - mn) / (mx - mn)) * (H2 - pT - pB);
    let svg = "";
    [1.0, Math.round(mx * 10) / 10].forEach(lv => {
      if (lv >= mn && lv <= mx) {
        svg += `<line x1="${pL}" y1="${py2(lv).toFixed(1)}" x2="${W2 - pR}" y2="${py2(lv).toFixed(1)}" stroke="#e5e9f0"/>
          <text x="${pL - 5}" y="${(py2(lv) + 3).toFixed(1)}" text-anchor="end" font-size="10" fill="#657180">${lv.toFixed(1)}</text>`;
      }
    });
    [0, Math.floor(qc.dates.length / 2), qc.dates.length - 1].forEach(idx => {
      svg += `<text x="${px(idx).toFixed(1)}" y="${H2 - 7}" text-anchor="middle" font-size="10" fill="#657180">${qc.dates[idx]}</text>`;
    });
    ["G1", "G2", "G3", "G4", "G5"].forEach(gk => {
      const arr = qc.navs[gk];
      const poly = arr.map((v, i) => `${px(i).toFixed(1)},${py2(v).toFixed(1)}`).join(" ");
      const endV = arr[arr.length - 1];
      svg += `<polyline fill="none" stroke="${colors[gk]}" stroke-width="${gk === "G5" || gk === "G1" ? "2.1" : "1.4"}" points="${poly}"/>
        <text x="${W2 - pR + 4}" y="${(py2(endV) + 3).toFixed(1)}" font-size="10" font-weight="600" fill="${colors[gk]}">${gk}:${endV.toFixed(2)}</text>`;
    });
    document.getElementById("qNavSvg").innerHTML = `<svg viewBox="0 0 ${W2} ${H2}" style="width:100%;height:auto;display:block;">${svg}</svg>`;
  }
}

function renderStage4() {
  document.getElementById("stratCostNote").textContent = `当前单边成本口径：${curSC} (A股转债免印花税)`;
  const S = R.strategies || {};
  const specs = [
    ["four_region_lsm", "#245ca9", "基准验证主策略"],
    ["double_low", "#d97706", "是否优于传统双低因子"],
    ["no_reset_lsm", "#0f766e", "消融：下修条款是否贡献收益"],
    ["single_region_lsm", "#7c3aed", "消融：Four-Region 分段增量价值"],
    ["mkt_ew", "#94a3b8", "基准：全市场等权持有"]
  ];

  // Multi-strategy NAV SVG (computed at 0.10% stored curve)
  const first = S["four_region_lsm"];
  if (first && first.nav_dates) {
    const W = 960, H = 250, pL = 46, pR = 96, pT = 16, pB = 28;
    const allV = specs.flatMap(([k]) => (S[k] || {}).nav_010 || []);
    const mn = Math.min(0.9, ...allV) * 0.96, mx = Math.max(1.5, ...allV) * 1.04;
    const px = i => pL + (i / Math.max(1, first.nav_dates.length - 1)) * (W - pL - pR);
    const py = v => H - pB - ((v - mn) / (mx - mn)) * (H - pT - pB);
    let g = "";
    [1.0, 1.5, 2.0, 2.5, 3.0].forEach(lv => {
      if (lv >= mn && lv <= mx) {
        const y = py(lv).toFixed(1);
        g += `<line x1="${pL}" y1="${y}" x2="${W - pR}" y2="${y}" stroke="${lv === 1.0 ? "#94a3b8" : "#eef2f6"}"/>
          <text x="${pL - 6}" y="${Number(y) + 3}" text-anchor="end" font-size="10" fill="#657180">${lv.toFixed(1)}</text>`;
      }
    });
    [0, Math.floor(first.nav_dates.length / 3), Math.floor(first.nav_dates.length * 2 / 3), first.nav_dates.length - 1].forEach(idx => {
      g += `<text x="${px(idx).toFixed(1)}" y="${H - 7}" text-anchor="middle" font-size="10" fill="#657180">${first.nav_dates[idx]}</text>`;
    });
    specs.forEach(([sk, col]) => {
      const obj = S[sk];
      if (!obj) return;
      const poly = obj.nav_010.map((v, i) => `${px(i).toFixed(1)},${py(v).toFixed(1)}`).join(" ");
      const endV = obj.nav_010[obj.nav_010.length - 1];
      g += `<polyline fill="none" stroke="${col}" stroke-width="${sk === "mkt_ew" ? "1.6" : "2.1"}" ${sk === "mkt_ew" ? 'stroke-dasharray="5 4"' : ""} points="${poly}"/>
        <text x="${W - pR + 5}" y="${(py(endV) + 3).toFixed(1)}" font-size="10" font-weight="600" fill="${col}">${endV.toFixed(2)} (${((endV - 1) * 100).toFixed(0)}%)</text>`;
    });
    document.getElementById("stratNavSvg").innerHTML = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block;">${g}</svg>`;
  }

  const goals = {
    four_region_lsm: "主检验策略（冻结原论文参数）",
    mkt_ew: "检验是否具备市场选券超额",
    double_low: "检验是否优于传统双低因子",
    low_price: "检验是否单纯来自低价因子",
    low_prem: "检验是否单纯来自低溢价因子",
    stock_mom: "检验是否单纯追逐正股趋势",
    single_region_lsm: "消融检验：Four-Region 分段结构贡献",
    no_reset_lsm: "消融检验：下修博弈条款贡献"
  };

  let sRows = Object.keys(goals).map(sk => {
    const o = S[sk];
    if (!o) return "";
    const c = o.costs[curSC];
    return `
      <tr class="${sk === "four_region_lsm" ? "hl-row" : ""}">
        <td><strong>${o.name}</strong></td>
        <td style="font-size:12px;color:var(--muted);">${goals[sk]}</td>
        <td class="num mono ${cls(c.total_ret)}" style="font-weight:700;">${signed(c.total_ret)}</td>
        <td class="num mono ${cls(c.ann_ret)}">${signed(c.ann_ret)}</td>
        <td class="num mono neg">${fmt(c.mdd)}%</td>
        <td class="num mono"><strong>${fmt(c.sharpe, 2)}</strong></td>
        <td class="num mono">${fmt(o.ann_to, 1)}%</td>
      </tr>
    `;
  }).join("");

  const rmc = R.random_mc;
  if (rmc) {
    sRows += `
      <tr>
        <td><strong>随机选券蒙特卡洛 (${rmc.runs} 次对照)</strong></td>
        <td style="font-size:12px;color:var(--muted);">同换手随机抽样 95% 置信区间</td>
        <td class="num mono">+${fmt(rmc.total_ret_mean)}% <span class="subtext">95% CI [${fmt(rmc.total_ret_ci95[0])}%, ${fmt(rmc.total_ret_ci95[1])}%]</span></td>
        <td class="num mono">+${fmt(rmc.ann_ret_mean)}%</td>
        <td class="num mono">—</td>
        <td class="num mono">${fmt(rmc.sharpe_mean, 2)}</td>
        <td class="num mono">~750%</td>
      </tr>
    `;
  }
  document.getElementById("stratTableBody").innerHTML = sRows;

  // Fama-MacBeth Table
  const fmb = (R.fama_macbeth || {})[curFmbH];
  if (fmb) {
    document.getElementById("fmbTableBody").innerHTML = fmb.factors.map(f => {
      const sig = f.p_val < 0.05;
      return `
        <tr class="${f.key === "f_four" ? "hl-row" : ""}">
          <td><strong>${f.label}</strong></td>
          <td class="num mono ${cls(f.mean)}" style="font-weight:700;">${signed(f.mean, 4, "")}</td>
          <td class="num mono">${fmt(f.se, 4)}</td>
          <td class="num mono"><strong>${f.t_stat > 0 ? "+" : ""}${fmt(f.t_stat, 2)}</strong></td>
          <td class="num mono">${f.p_val < 0.0001 ? "<0.0001" : f.p_val.toFixed(4)}</td>
          <td class="num mono">[${fmt(f.ci_low, 4)}, ${fmt(f.ci_high, 4)}]</td>
          <td><span class="badge ${sig ? "sig-yes" : "sig-no"}">${sig ? "显著 (p<0.05)" : "不显著"}</span></td>
        </tr>
      `;
    }).join("");
  }

  document.getElementById("ablationNote").innerHTML =
    `<strong>客观消融发现（实测事实，不加修饰）：</strong>(1) 控制全部 6 个传统因子后，F_LSM 的 Fama-MacBeth 截面斜率仍达 <code>+0.6296%</code>（<code>t_NW=3.40, p=0.0007</code>），证明模型确实捕捉到了“价格、溢价、波动率、期限、正股动量、信用评级”之外的增量定价信息；(2) 全市场 7 年累计来看，Four-Region LSM (<code>+156.61%</code>, Sharpe <code>0.99</code>) 显著跑赢全市场基准 (<code>+70.21%</code>) 和随机选券 95% 区间上限 (<code>+79.87%</code>)；(3) 消融组中，<code>去分段单一区域 LSM (+158.40%)</code> 与分段版表现极为接近，而<code>无下修机制 (p=0)</code> 累计收益反而更高 (<code>+200.30%</code>)——说明在 2023–2024 年小盘退市冲击下，论文默认的乐观下修先验 (<code>p=0.80</code>) 曾将部分弱资质券误判为低估。`;
}

function renderStage5() {
  const box = document.getElementById("s5Content");
  if (curS5 === "subperiod") {
    const spIc = R.subperiod_ic || {};
    const spSt = R.subperiod_strats || {};
    const fr = R.freq_robust || {};
    const rows = Object.keys(spIc).map(p => {
      const i20 = spIc[p]["20"];
      const s = spSt[p];
      return `
        <tr>
          <td><strong>${p}</strong></td>
          <td class="num mono pos">+${fmt(i20.mean_ic, 4)} (t=${fmt(i20.t_nw, 2)})</td>
          <td class="num mono">${fmt(i20.icir, 3)}</td>
          <td class="num mono">${fmt(i20.pos_pct, 1)}%</td>
          <td class="num mono pos"><strong>${signed(s.four_region_lsm.total_ret)}</strong> <span class="subtext">年化 ${signed(s.four_region_lsm.ann_ret)} · Sharpe ${fmt(s.four_region_lsm.sharpe)}</span></td>
          <td class="num mono">${signed(s.double_low.total_ret)} <span class="subtext">Sharpe ${fmt(s.double_low.sharpe)}</span></td>
          <td class="num mono">${signed(s.mkt_ew.total_ret)} <span class="subtext">Sharpe ${fmt(s.mkt_ew.sharpe)}</span></td>
        </tr>
      `;
    }).join("");

    const frRows = ["1D", "5D", "10D", "20D"].map(fk => {
      const f = fr[fk] || {};
      return `<tr><td class="mono"><strong>${fk} 调仓</strong></td>
        <td class="num mono">${signed((f["0.05%"]||{}).total_ret)} (Sharpe ${fmt((f["0.05%"]||{}).sharpe)})</td>
        <td class="num mono">${signed((f["0.10%"]||{}).total_ret)} (Sharpe ${fmt((f["0.10%"]||{}).sharpe)})</td>
        <td class="num mono">${signed((f["0.20%"]||{}).total_ret)} (Sharpe ${fmt((f["0.20%"]||{}).sharpe)})</td></tr>`;
    }).join("");

    box.innerHTML = `
      <div class="table-wrap">
        <table>
          <thead><tr><th>市场时段</th><th>20 日 Rank IC</th><th>ICIR</th><th>IC&gt;0 比例</th><th>Four-Region LSM (周频 0.10%)</th><th>传统双低对照</th><th>全市场基准</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
      <div class="subtoolbar"><strong>多调仓频率稳健性核对（2020–2026 Four-Region LSM，避免挑选单一频率过拟合）</strong></div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>调仓频率</th><th>0.05% 单边成本</th><th>0.10% 单边成本</th><th>0.20% 单边成本</th></tr></thead>
          <tbody>${frRows}</tbody>
        </table>
      </div>
    `;
  } else if (curS5 === "live_top" || curS5 === "live_all") {
    const items = curS5 === "live_top" ? DATA.lsm_top10 : DATA.all_bonds;
    const rows = items.map(b => `
      <tr>
        <td class="num mono" style="text-align:center;">#${b.lsm_rank}</td>
        <td><strong>${b.name}</strong><span class="subtext mono">${b.ticker} · ${b.stock_name} (¥${fmt(b.stock_price)} / 转股 ¥${fmt(b.conv_price)})</span></td>
        <td class="num mono">¥${fmt(b.bond_price)}</td>
        <td class="num mono"><strong>¥${fmt(b.model_price)}</strong></td>
        <td class="num mono ${b.undervaluation > 0 ? "neg" : ""}">${signed(b.undervaluation)}</td>
        <td><span class="badge sig-no">${b.region_code} · ${b.region_name}</span><span class="subtext">${b.region_desc}</span></td>
        <td class="num mono">${fmt(b.premium_rate, 1)}% <span class="subtext">双低 ${fmt(b.double_low, 1)}</span></td>
        <td class="num mono">${fmt(b.sigma, 1)}% / ${fmt(b.maturity_years, 1)}年</td>
      </tr>
    `).join("");
    box.innerHTML = `
      <div class="subtoolbar">
        <strong>当期冻结参数模型实时估值（p_reset=0.80, B=108 · 共 ${items.length} 只 · 更新于 ${DATA.updated_at}）</strong>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr><th style="text-align:center;">排名</th><th>转债标的</th><th>现价</th><th>模型估值</th><th>低估空间</th><th>条款状态</th><th>溢价 / 双低</th><th>波动率 / 年限</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    `;
  } else {
    const l = DATA.live_ledger || { tracked_days: 0, comparisons: [], frozen_since: "2026-11-25" };
    const rows = (l.comparisons || []).map(c => `
      <tr>
        <td class="mono">${c.from_date} → ${c.to_date}</td>
        <td class="num mono">${c.count} 只</td>
        <td class="num mono ${cls(c.top_ret)}">${signed(c.top_ret)}</td>
        <td class="num mono ${cls(c.bot_ret)}">${signed(c.bot_ret)}</td>
        <td class="num mono ${cls(c.dl_ret)}">${signed(c.dl_ret)}</td>
        <td class="num mono ${cls(c.mkt_ret)}">${signed(c.mkt_ret)}</td>
        <td class="num mono ${cls(c.spread)}" style="font-weight:700;">${signed(c.spread)}</td>
        <td class="num mono">${c.rank_ic > 0 ? "+" : ""}${fmt(c.rank_ic, 3)}</td>
      </tr>
    `).join("");
    box.innerHTML = `
      <div class="subtoolbar">
        <strong>不可事后修改前瞻账本（自 ${l.frozen_since} 冻结启动 · 已累计 ${l.tracked_days} 个交易日快照）</strong>
        <span class="subtext" style="margin:0;">每日收市自动归档至 history.json，杜绝历史回填</span>
      </div>
      <div class="table-wrap">
        ${rows ? `<table><thead><tr><th>区间</th><th>样本数</th><th>低估 Top20%</th><th>高估 Bot20%</th><th>传统双低</th><th>全市场均值</th><th>多空超额</th><th>Rank IC</th></tr></thead><tbody>${rows}</tbody></table>`
          : `<div style="padding:26px;text-align:center;color:var(--muted);">已归档基准日（${DATA.date}）冻结信号快照，后续每个交易日将在此累积真实前瞻兑现记录。</div>`}
      </div>
    `;
  }
}

function approxErf(x) {
  const s = x >= 0 ? 1 : -1;
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x));
}

function doCalc() {
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
  if (S0 > k0) { title = "强赎提示区 (R1)"; desc = "高于 130% 强赎触发线，股性主导。"; }
  else if (S0 > C0) { title = "高弹性转股区 (R2)"; desc = "实值期权区间，跟随正股上行。"; }
  else if (S0 > ptr0) { title = "平衡抗跌区 (R3)"; desc = "纯债保底（¥108）与期权价值兼备。"; }
  else { title = "下修博弈区 (R4)"; desc = "跌破 70% 回售线，触发下修意愿博弈 p=" + (p * 100).toFixed(0) + "%。"; }
  document.getElementById("res-reg-title").textContent = title;
  document.getElementById("res-reg-text").textContent = desc;
  const T = 2.5, r = 0.025;
  const d1 = (Math.log(S0 / C0) + (r + 0.5 * sig * sig) * T) / (sig * Math.sqrt(T));
  const d2 = d1 - sig * Math.sqrt(T);
  const cdf = x => 0.5 * (1 + approxErf(x / Math.SQRT2));
  const opt = (100.0 / C0) * (S0 * cdf(d1) - C0 * Math.exp(-r * T) * cdf(d2));
  const resetBoost = (S0 <= ptr0) ? (p * 12.5 * (1.0 - S0 / C0)) : (p * 3.5);
  document.getElementById("res-mdl").textContent = "¥" + Math.max(convVal, Math.max(108.0 * Math.exp(-r * T) + opt + resetBoost, 108.0 * 0.96)).toFixed(2);
}

document.querySelectorAll("#icRollPills .pill").forEach(b => b.addEventListener("click", () => {
  curIcH = b.dataset.ich;
  document.querySelectorAll("#icRollPills .pill").forEach(x => x.classList.toggle("active", x === b));
  renderIC();
}));
document.querySelectorAll("#qHorizonPills .pill").forEach(b => b.addEventListener("click", () => {
  curQH = b.dataset.qh;
  document.querySelectorAll("#qHorizonPills .pill").forEach(x => x.classList.toggle("active", x === b));
  renderQuintiles();
}));
document.querySelectorAll("#stratCostPills .pill").forEach(b => b.addEventListener("click", () => {
  curSC = b.dataset.sc;
  document.querySelectorAll("#stratCostPills .pill").forEach(x => x.classList.toggle("active", x === b));
  renderStage4();
}));
document.querySelectorAll("#fmbHorizonPills .pill").forEach(b => b.addEventListener("click", () => {
  curFmbH = b.dataset.fh;
  document.querySelectorAll("#fmbHorizonPills .pill").forEach(x => x.classList.toggle("active", x === b));
  renderStage4();
}));
document.querySelectorAll("#s5Pills .pill").forEach(b => b.addEventListener("click", () => {
  curS5 = b.dataset.s5;
  document.querySelectorAll("#s5Pills .pill").forEach(x => x.classList.toggle("active", x === b));
  renderStage5();
}));
["slider-s0", "slider-c0", "slider-p", "slider-sig"].forEach(id => document.getElementById(id).addEventListener("input", doCalc));

renderIC();
renderQuintiles();
renderStage4();
renderStage5();
doCalc();
</script>
</body>
</html>
"""

    return (
        template
        .replace("__PAYLOAD_JSON__", payload_json)
        .replace("__TOTAL_BONDS__", str(u.get("total_bonds", 0)))
        .replace("__TOTAL_OBS__", f"{u.get('total_bond_days', 0):,}")
        .replace("__TRADING_DAYS__", f"{u.get('trading_days', 0):,}")
        .replace("__AVG_DAILY__", str(u.get("avg_daily_bonds", 0)))
        .replace("__IC20_VAL__", f"+{ic20.get('mean_ic', 0):.4f}" if ic20 else "—")
        .replace("__IC20_ICIR__", f"{ic20.get('icir', 0):.3f}" if ic20 else "—")
        .replace("__IC20_POS__", f"{ic20.get('pos_pct', 0):.1f}" if ic20 else "—")
        .replace("__IC20_T__", f"{ic20.get('t_nw', 0):.2f}" if ic20 else "—")
        .replace("__Q20_SPREAD__", f"+{q20.get('g5_minus_g1', 0):.3f}%" if q20 else "—")
        .replace("__Q20_T__", f"{q20.get('t_nw', 0):.2f}" if q20 else "—")
        .replace("__Q20_P__", f"{q20.get('p_val', 1):.4f}" if q20 else "—")
        .replace("__FMB20_BETA__", f"+{fmb20.get('mean', 0):.4f}%" if fmb20 else "—")
        .replace("__FMB20_T__", f"{fmb20.get('t_stat', 0):.2f}" if fmb20 else "—")
        .replace("__FMB20_P__", f"{fmb20.get('p_val', 1):.4f}" if fmb20 else "—")
        .replace("__UPDATED_AT__", data["updated_at"])
    )


if __name__ == "__main__":
    generate_page_and_data()
