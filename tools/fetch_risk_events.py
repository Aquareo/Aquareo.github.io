"""
Build strictly causal historical risk-event tables for Chinese CBs (2017–2026):
1. `st_intervals.csv`: exact historical ST / *ST date intervals per underlying stock (`stock_code, st_start, st_end, source`)
   - SZSE dated short-name history (`ak.stock_info_sz_change_name("简称变更")`)
   - Eastmoney F10 `CompanySurvey` `FORMERNAME` scan across all CB underlying stocks + CNInfo `风险警示` notices for any SH/SZ stock ever carrying ST/*ST
2. `default_risk_events.csv`: exact first public announcement date of bond default / inability-to-pay / delisting warning
   (`ts_code, ann_date, risk_type, title`) from CNInfo for distressed/downgraded/ST CBs.

Usage:
    python tools/fetch_risk_events.py --out-dir /path/to/cache --ratings-csv /path/to/rating_events.csv
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import time
import akshare as ak
import pandas as pd
import requests


def scan_former_st(stock_code: str) -> tuple:
    mkt = "SH" if stock_code.startswith("6") else "SZ"
    for attempt in range(3):
        try:
            r = requests.get(
                f"https://emweb.securities.eastmoney.com/PC_HSF10/CompanySurvey/PageAjax?code={mkt}{stock_code}",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=8,
            ).json()
            jb = r.get("jbzl") or []
            if jb:
                cur_nm = str(jb[0].get("SECURITY_NAME_ABBR") or "")
                former = str(jb[0].get("FORMERNAME") or "")
                if "ST" in cur_nm.upper() or "ST" in former.upper() or cur_nm.startswith("R") or "退" in cur_nm:
                    return stock_code, True, f"{cur_nm} ({former})"
            return stock_code, False, ""
        except Exception:
            time.sleep(0.3 * (attempt + 1))
    return stock_code, False, ""


def fetch_cninfo_st_intervals(stock_code: str) -> list:
    intervals = []
    frames = []
    for kw in ("风险警示", "退市整理"):
        for attempt in range(3):
            try:
                part = ak.stock_zh_a_disclosure_report_cninfo(
                    symbol=stock_code, market="沪深京", keyword=kw, category="",
                    start_date="20170101", end_date="20261201",
                )
                if part is not None and not part.empty:
                    frames.append(part)
                break
            except Exception:
                time.sleep(0.8 * (attempt + 1))
    if not frames:
        return intervals
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["公告链接"])
    df["title"] = df["公告标题"].astype(str).str.replace(r"<[^>]+>", "", regex=True)
    df["ymd"] = df["公告时间"].astype(str).str[:10].str.replace("-", "")
    df = df.sort_values("ymd")
    cur_start = None
    for _, r in df.iterrows():
        t = r["title"]
        d = r["ymd"]
        if any(k in t for k in ("被实施退市风险警示", "被实施其他风险警示", "实施退市风险警示暨", "实施其他风险警示暨", "被叠加实施", "进入退市整理期", "股票将被实施")) and not any(x in t for x in ("可能被实施", "预计", "进展", "核查")):
            if cur_start is None:
                cur_start = d
        elif any(k in t for k in ("撤销退市风险警示", "撤销其他风险警示", "撤销公司股票风险警示")) and "申请" not in t and "可能" not in t:
            if cur_start is not None:
                intervals.append({"stock_code": stock_code, "st_start": cur_start, "st_end": d, "source": "CNInfo风险警示"})
                cur_start = None
    if cur_start is not None:
        intervals.append({"stock_code": stock_code, "st_start": cur_start, "st_end": "20991231", "source": "CNInfo风险警示"})
    return intervals


def fmt_ts(code: str) -> str:
    return f"{code}.SH" if code.startswith(("11", "13")) else f"{code}.SZ"


def fetch_cninfo_default_warnings(code: str, name: str, stock_code: str) -> list:
    out = []
    frames = []
    for kw in ("兑付", "违约", "终止上市"):
        for attempt in range(3):
            try:
                part = ak.stock_zh_a_disclosure_report_cninfo(
                    symbol=stock_code, market="沪深京", keyword=kw, category="",
                    start_date="20180101", end_date="20261201",
                )
                if part is not None and not part.empty:
                    frames.append(part)
                break
            except Exception:
                time.sleep(0.8 * (attempt + 1))
    if not frames:
        return out
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["公告链接"])
    df["title"] = df["公告标题"].astype(str).str.replace(r"<[^>]+>", "", regex=True)
    df["ymd"] = df["公告时间"].astype(str).str[:10].str.replace("-", "")
    df = df.sort_values("ymd")
    for _, r in df.iterrows():
        t = r["title"]
        if any(b in t for b in ("转债", "可转换公司债券", name)) and any(
            k in t for k in ("无法按期", "未能按期", "本息兑付存在不确定", "存在违约", "可能发生违约", "违约风险", "终止上市", "进入退市整理")
        ):
            # Exclude routine forced-redemption/maturity payment notices!
            if any(ok in t for ok in ("到期兑付暨摘牌", "提前赎回", "赎回结果", "付息公告")):
                continue
            out.append({"ts_code": fmt_ts(code), "ann_date": r["ymd"], "risk_type": "违约/退市风险公告", "title": t[:80]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="/tmp/bonds_exp/cache")
    ap.add_argument("--ratings-csv", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_em = ak.bond_cb_list_em()
    df_em["上市时间_dt"] = pd.to_datetime(df_em["上市时间"], errors="coerce")
    cand = df_em[
        df_em["上市时间_dt"].between("2017-01-01", "2026-10-15")
        & df_em["债券代码"].astype(str).str.startswith(("11", "12", "13"))
        & df_em["正股代码"].astype(str).str.match(r"^(0|3|6)\d{5}$")
    ].copy()
    scodes = sorted(cand["正股代码"].astype(str).unique().tolist())

    print(f"Scanning F10 name history across {len(scodes)} underlying stocks...", flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        scan_res = list(ex.map(scan_former_st, scodes))
    st_cand_stocks = sorted({sc for sc, hit, _ in scan_res if hit})
    print(f"Stocks ever carrying ST/*ST/退 in name history: {len(st_cand_stocks)}", flush=True)

    # 1A. Exact SZSE dated name-change table
    st_intervals = []
    sz_handled = set()
    try:
        sz_chg = ak.stock_info_sz_change_name(symbol="简称变更")
        sz_chg["证券代码"] = sz_chg["证券代码"].astype(str).str.zfill(6)
        sz_chg["ymd"] = pd.to_datetime(sz_chg["变更日期"], errors="coerce").dt.strftime("%Y%m%d")
        sz_sub = sz_chg[sz_chg["证券代码"].isin(set(scodes))].sort_values(["证券代码", "ymd"])
        for sc, grp in sz_sub.groupby("证券代码"):
            cur_start = None
            has_st = False
            for _, r in grp.iterrows():
                post = str(r.get("变更后简称") or "").upper()
                d = str(r["ymd"])
                if "ST" in post or post.startswith("R") or post.endswith("退"):
                    has_st = True
                    if cur_start is None:
                        cur_start = d
                else:
                    if cur_start is not None:
                        st_intervals.append({"stock_code": sc, "st_start": cur_start, "st_end": d, "source": "SZSE简称变更"})
                        cur_start = None
            if cur_start is not None:
                st_intervals.append({"stock_code": sc, "st_start": cur_start, "st_end": "20991231", "source": "SZSE简称变更"})
            if has_st:
                sz_handled.add(sc)
    except Exception as e:
        print("Warning: SZSE name change fetch failed:", e, flush=True)

    # 1B. CNInfo risk-warning announcements for all SH/unhandled ST candidate stocks
    need_cn_st = [sc for sc in st_cand_stocks if sc not in sz_handled]
    print(f"Fetching CNInfo 风险警示 notices for {len(need_cn_st)} remaining ST stocks...", flush=True)
    for sc in need_cn_st:
        ivs = fetch_cninfo_st_intervals(sc)
        if ivs:
            st_intervals.extend(ivs)
        time.sleep(0.15)

    st_df = pd.DataFrame(st_intervals).drop_duplicates().sort_values(["stock_code", "st_start"]) if st_intervals else pd.DataFrame(columns=["stock_code", "st_start", "st_end", "source"])
    st_df.to_csv(out_dir / "st_intervals.csv", index=False, encoding="utf-8")
    print(f"Saved {len(st_df)} historical ST intervals across {st_df['stock_code'].nunique() if len(st_df) else 0} stocks.", flush=True)

    # 2. CB announced default / termination warnings for downgraded or ST-linked CBs
    rat_df = pd.read_csv(args.ratings_csv, dtype=str)
    low_rat_ts = set(rat_df[rat_df["rating"].isin(["BBB+", "BBB", "BBB-", "BB+", "BB", "BB-", "B+", "B", "B-", "CCC", "CC", "C"])]["ts_code"].unique())
    changed_ts = set(rat_df.groupby("ts_code")["rating"].nunique().loc[lambda s: s > 1].index)
    st_cbs = set(cand[cand["正股代码"].astype(str).isin(set(st_cand_stocks))]["债券代码"].astype(str))
    watch_codes = sorted({c.split(".")[0] for c in (low_rat_ts | changed_ts)} | st_cbs)
    watch_cand = cand[cand["债券代码"].astype(str).isin(watch_codes)]
    print(f"Scanning CNInfo default/delisting risk announcements for {len(watch_cand)} distressed/ST/changed CBs...", flush=True)
    def_rows = []
    for _, r in watch_cand.iterrows():
        def_rows.extend(fetch_cninfo_default_warnings(str(r["债券代码"]), str(r["债券简称"]), str(r["正股代码"])))
        time.sleep(0.1)
    def_df = pd.DataFrame(def_rows).drop_duplicates(subset=["ts_code", "ann_date"]) if def_rows else pd.DataFrame(columns=["ts_code", "ann_date", "risk_type", "title"])
    def_df.to_csv(out_dir / "default_risk_events.csv", index=False, encoding="utf-8")
    print(f"Saved {len(def_df)} default/delisting warning events across {def_df['ts_code'].nunique() if len(def_df) else 0} CBs.", flush=True)


if __name__ == "__main__":
    main()
