"""
Fetch and locally cache the full Chinese exchange-listed convertible bond universe (2017–2026),
including active, forced-redeemed, matured, and defaulted/delisted bonds and their underlying stocks.

Sources (via AkShare):
- ak.bond_cb_list_em(): full historical CB registry
- ak.bond_cb_info_em(symbol): issuance terms, listing/cease/delist dates, coupon schedule
- ak.bond_cb_value_analysis_em(symbol): daily historical conversion value & pure bond value
- ak.bond_cb_daily_sina(symbol): daily unadjusted CB OHLCV
- ak.stock_zh_a_hist_sina(symbol): daily unadjusted underlying A-share stock OHLCV

Usage:
    python tools/fetch_all_cbs.py [--cache-dir /path/to/cache] [--workers 3]
"""

import argparse
import pickle
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import akshare as ak
import pandas as pd


def parse_coupon_years(explain: str, raw_expire) -> float:
    """Infer original contractual maturity (years) strictly from coupon schedule, avoiding ex-post early call leak."""
    if isinstance(explain, str) and explain:
        yrs = re.findall(r"第[一二三四五六七八九十\d]+年", explain)
        if len(yrs) >= 3:
            return float(len(yrs))
    try:
        v = float(raw_expire)
        if v >= 4.0:
            return float(round(v))
    except Exception:
        pass
    return 6.0


def parse_coupons(explain: str) -> list:
    if not isinstance(explain, str) or not explain:
        return [0.3, 0.5, 1.0, 1.5, 2.0, 2.5]
    vals = re.findall(r"(\d+(?:\.\d+)?)%", explain)
    return [float(x) for x in vals] if vals else [0.3, 0.5, 1.0, 1.5, 2.0, 2.5]


def fetch_stock_one(args):
    scode, stk_dir = args
    p = Path(stk_dir) / f"{scode}.pkl"
    if p.exists():
        return scode, True
    s_pref = "sh" if scode.startswith("6") else "sz"
    for attempt in range(4):
        try:
            time.sleep(0.10)
            df = ak.stock_zh_a_hist_sina(symbol=s_pref + scode)
            if df is not None and not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df = df[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)
                with open(p, "wb") as f:
                    pickle.dump(df, f)
                return scode, True
        except Exception:
            time.sleep(0.6 * (attempt + 1))
    return scode, False


def fetch_cb_one(args):
    row, cb_dir = args
    code = str(row["债券代码"]).strip()
    p = Path(cb_dir) / f"{code}.pkl"
    if p.exists():
        return code, True

    b_pref = "sh" if code.startswith(("11", "13")) else "sz"
    for attempt in range(4):
        try:
            time.sleep(0.12)
            b_hist = ak.bond_cb_daily_sina(symbol=b_pref + code)
            va = ak.bond_cb_value_analysis_em(symbol=code)
            info_df = ak.bond_cb_info_em(symbol=code)
            if b_hist is None or b_hist.empty or va is None or va.empty or info_df is None or info_df.empty:
                return code, False

            info = info_df.iloc[0].to_dict()
            b_hist["date"] = pd.to_datetime(b_hist["date"])
            b_hist = b_hist[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)
            va["日期"] = pd.to_datetime(va["日期"])
            va = va.rename(columns={"日期": "date"})
            va = va[["date", "纯债价值", "转股价值", "转股溢价率"]].sort_values("date").reset_index(drop=True)

            meta = {
                "ticker": code,
                "name": str(row.get("债券简称", "")),
                "stock_code": str(row.get("正股代码", "")).strip(),
                "stock_name": str(row.get("正股简称", "")),
                "latest_rating_expost": str(info.get("RATING") or row.get("信用评级") or ""),
                "listing_date": str(info.get("LISTING_DATE") or row.get("上市时间") or ""),
                "value_date": str(info.get("VALUE_DATE") or info.get("LISTING_DATE") or row.get("上市时间") or ""),
                "delist_date": str(info.get("DELIST_DATE") or ""),
                "cease_date": str(info.get("CEASE_DATE") or ""),
                "expire_date": str(info.get("EXPIRE_DATE") or ""),
                "pay_day": str(info.get("PAY_INTEREST_DAY") or ""),
                "contract_years": parse_coupon_years(str(info.get("INTEREST_RATE_EXPLAIN") or ""), info.get("BOND_EXPIRE")),
                "coupons": parse_coupons(str(info.get("INTEREST_RATE_EXPLAIN") or "")),
                "init_conv": float(info.get("INITIAL_TRANSFER_PRICE") or row.get("转股价") or 10.0) if pd.notna(info.get("INITIAL_TRANSFER_PRICE") or row.get("转股价")) else 10.0,
                "latest_conv": float(row.get("转股价")) if pd.notna(row.get("转股价")) else None,
            }
            with open(p, "wb") as f:
                pickle.dump({"meta": meta, "b_hist": b_hist, "va": va}, f)
            return code, True
        except Exception:
            time.sleep(0.8 * (attempt + 1))
    return code, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=".cache_bonds")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    base = Path(args.cache_dir)
    cb_dir = base / "cbs"
    stk_dir = base / "stocks"
    cb_dir.mkdir(parents=True, exist_ok=True)
    stk_dir.mkdir(parents=True, exist_ok=True)

    df_em = ak.bond_cb_list_em()
    df_em["上市时间_dt"] = pd.to_datetime(df_em["上市时间"], errors="coerce")
    cand = df_em[
        df_em["上市时间_dt"].between("2017-01-01", "2026-10-15")
        & df_em["债券代码"].astype(str).str.startswith(("11", "12", "13"))
        & df_em["正股代码"].astype(str).str.match(r"^(0|3|6)\d{5}$")
    ].copy()
    print(f"Eligible exchange CBs 2017-2026: {len(cand)}", flush=True)

    stock_codes = sorted(cand["正股代码"].astype(str).unique().tolist())
    print(f"Fetching {len(stock_codes)} unique underlying stocks...", flush=True)
    ok_stk = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fetch_stock_one, (sc, str(stk_dir))) for sc in stock_codes]
        for i, fut in enumerate(as_completed(futs), 1):
            _, ok = fut.result()
            ok_stk += int(ok)
            if i % 150 == 0 or i == len(futs):
                print(f"  Stocks {i}/{len(futs)} (ok={ok_stk})", flush=True)

    rows = cand.to_dict("records")
    print(f"Fetching {len(rows)} CB histories & clauses...", flush=True)
    ok_cb = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fetch_cb_one, (r, str(cb_dir))) for r in rows]
        for i, fut in enumerate(as_completed(futs), 1):
            _, ok = fut.result()
            ok_cb += int(ok)
            if i % 100 == 0 or i == len(futs):
                print(f"  CBs {i}/{len(futs)} (ok={ok_cb})", flush=True)

    print(f"DONE fetching: stocks={ok_stk}/{len(stock_codes)}, cbs={ok_cb}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
