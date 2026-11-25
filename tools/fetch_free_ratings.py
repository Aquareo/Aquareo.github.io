"""
Build a free, auditable point-in-time bond-level credit rating event CSV
(`ts_code, ann_date, rating, rating_type`) compatible with `tools/run_high_grade_validation.py`.

Methodology (zero latest_rating leakage, zero backfill):
1. Initial issuance bond rating (`ZDRZXPJ_GN1_FX`) published on `ISSUE_DATE` from Eastmoney `RPT_BOND_BS_INFO`
   (known strictly before exchange listing; covers ~99.7% of 2017–2026 CBs).
2. For all CBs whose rating changed between issuance and latest (`fx_bond != cur_bond`) while touching
   AA+ / AAA (or any AA+ -> AAA / AA+ <-> AA boundary), fetch all historical CNInfo tracking-rating PDF
   announcements (`ak.stock_zh_a_disclosure_report_cninfo` + `static.cninfo.com.cn`) and extract each
   dated bond-level rating announcement chronologically.
3. For changed CBs strictly below AA+ on both ends, record issuance rating at `ISSUE_DATE` and latest
   changed rating at `NEW_RATING_DATE` (keeps full-universe >=90% coverage while keeping AA+/AAA boundary exact).

Usage:
    python tools/fetch_free_ratings.py --out .cache_bonds/rating_events.csv
"""

import argparse
import io
import re
import time
from pathlib import Path
import akshare as ak
import pandas as pd
import pypdf
import requests


def norm_grade(g: str) -> str:
    s = str(g or "").strip().upper().replace("＋", "+").replace("－", "-")
    s = re.sub(r"STI$", "", s).strip()
    return s


def fmt_ts(code: str) -> str:
    return f"{code}.SH" if code.startswith(("11", "13")) else f"{code}.SZ"


_GRADE_RE = r"(?<![A-Za-z])(AAA|AA\+|AA-|AA|A\+|A-|A|BBB\+|BBB-|BBB|BB\+|BB-|BB|B\+|B-|B|CCC|CC|C)(?![A-Za-z+-])"


def fetch_cninfo_bond_events(code: str, name: str, stock_code: str, iss_dt: str) -> list:
    events = []
    df = None
    for attempt in range(4):
        try:
            df = ak.stock_zh_a_disclosure_report_cninfo(
                symbol=stock_code, market="沪深京", keyword="评级", category="",
                start_date=iss_dt or "20170101", end_date="20261201"
            )
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2.0 * (attempt + 1))
    try:
        if df is None or df.empty:
            return events
        rat = df[
            df["公告标题"].str.contains("评级|信用等级", na=False)
            & ~df["公告标题"].str.contains("延迟|提示性|预计|修订|制度|发行结果|上市|募集说明书|担保", na=False)
        ].copy()
        rat["ann_ymd"] = rat["公告时间"].astype(str).str[:10].str.replace("-", "")
        if iss_dt:
            rat = rat[rat["ann_ymd"] > iss_dt]
        # Prefer concise issuer notices ("结果"/"调整") before long rating-agency PDFs on same date
        rat["prio"] = rat["公告标题"].apply(lambda t: 0 if any(k in str(t) for k in ("结果", "调整", "下调", "上调")) else 1)
        rat = rat.sort_values(["ann_ymd", "prio"])
        seen_dates = set()
        for _, r in rat.iterrows():
            ann_dt = str(r["公告时间"])[:10]
            ymd = ann_dt.replace("-", "")
            if ymd in seen_dates:
                continue
            title = str(r["公告标题"])
            if not any(k in title for k in ("结果", "调整", "跟踪评级", "下调", "上调")):
                continue
            m_id = re.search(r"announcementId=(\d+)", str(r.get("公告链接", "")))
            if not m_id:
                continue
            ann_id = m_id.group(1)
            pdf_url = f"https://static.cninfo.com.cn/finalpage/{ann_dt}/{ann_id}.PDF"
            try:
                resp = requests.get(pdf_url, timeout=12)
                if resp.status_code != 200 or resp.content[:4] != b"%PDF":
                    continue
                pages = pypdf.PdfReader(io.BytesIO(resp.content)).pages[:3]
                txt = "\n".join((pg.extract_text() or "") for pg in pages)
                # Re-join PDF superscript "+" / "-" split onto its own line (e.g. "AA\n+\n" -> "AA+")
                txt = re.sub(r"(?<![A-Za-z])(AA|BBB|BB|A|B)\s+([+-])(?=[”\"'）)\s\n，。、；]|$)", r"\1\2", txt)
                txt_flat = re.sub(r"\s+", " ", txt)
                # Pattern 1: explicit "本次债项评级" / "调整后..." / "维持“XX转债”...为 XXX"
                m_post = re.findall(
                    r"(?:本次[^。]{0,20}?(?:债项|债券|" + re.escape(name) + r")[^。]{0,25}?(?:信用)?(?:等级|评级)|调整后[^。]{0,40}?(?:债券|债项|可转债|" + re.escape(name) + r")[^。]{0,25}?(?:信用)?(?:等级|评级)|维持[^。]{0,25}?(?:债券|债项|“" + re.escape(name) + r"”|" + re.escape(name) + r")[^。]{0,25}?(?:信用)?(?:等级|评级)|(?:上调|下调|调低|调高)[^。]{0,25}?(?:债券|债项|“" + re.escape(name) + r"”|" + re.escape(name) + r")[^。]{0,25}?(?:至|为))\s*[：:为是]?\s*[“\"']?" + _GRADE_RE,
                    txt_flat,
                )
                if not m_post:
                    # Pattern 2: any explicit bond/name rating sentence, taking the LAST grade ("由 X 调至 Y")
                    m_line = re.findall(
                        r"(?:债项|“" + re.escape(name) + r"”|" + re.escape(name) + r"|本次可转债|上述债券)[^。]{0,45}?(?:信用)?(?:等级|评级)[^。]{0,25}?(?:[：:为是至])\s*[“\"']?" + _GRADE_RE,
                        txt_flat,
                    )
                    if m_line:
                        m_post = [m_line[-1]]
                if m_post:
                    events.append({
                        "ts_code": fmt_ts(code),
                        "ann_date": ymd,
                        "rating": norm_grade(m_post[-1]),
                        "rating_type": "债券跟踪评级公告",
                    })
                    seen_dates.add(ymd)
            except Exception:
                continue
    except Exception:
        pass
    return events


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=".cache_bonds/rating_events.csv")
    args = ap.parse_args()

    df_em = None
    for attempt in range(4):
        try:
            df_em = ak.bond_cb_list_em()
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2.0 * (attempt + 1))
    df_em["上市时间_dt"] = pd.to_datetime(df_em["上市时间"], errors="coerce")
    cand = df_em[
        df_em["上市时间_dt"].between("2017-01-01", "2026-10-15")
        & df_em["债券代码"].astype(str).str.startswith(("11", "12", "13"))
        & df_em["正股代码"].astype(str).str.match(r"^(0|3|6)\d{5}$")
    ].copy()
    codes = sorted(cand["债券代码"].astype(str).unique().tolist())
    stock_map = dict(zip(cand["债券代码"].astype(str), cand["正股代码"].astype(str)))
    name_map = dict(zip(cand["债券代码"].astype(str), cand["债券简称"].astype(str)))

    bs_rows = []
    for i in range(0, len(codes), 40):
        chunk = codes[i : i + 40]
        flt = '","'.join(chunk)
        for attempt in range(4):
            try:
                r = requests.get(
                    "https://datacenter-web.eastmoney.com/api/data/v1/get",
                    params={
                        "reportName": "RPT_BOND_BS_INFO",
                        "columns": "SECURITY_CODE,BOND_NAME_ABBR,BOND_TYPE,ISSUE_DATE,LISTING_DATE,BOND_RATING,NEW_RATING_DATE,ZDRZXPJ_GN1_FX",
                        "filter": f'(SECURITY_CODE in ("{flt}"))',
                        "pageSize": 200,
                    },
                    timeout=15,
                ).json()
                for row in (r.get("result") or {}).get("data") or []:
                    if "可转债" in str(row.get("BOND_TYPE") or ""):
                        bs_rows.append(row)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(1.5 * (attempt + 1))

    df_bs = pd.DataFrame(bs_rows).drop_duplicates(subset=["SECURITY_CODE"])
    events = []
    boundary_codes = []

    for _, r in df_bs.iterrows():
        code = str(r["SECURITY_CODE"]).strip()
        fx = norm_grade(r.get("ZDRZXPJ_GN1_FX"))
        cur = norm_grade(r.get("BOND_RATING"))
        iss_dt = str(r.get("ISSUE_DATE") or r.get("LISTING_DATE") or "")[:10].replace("-", "")
        new_dt = str(r.get("NEW_RATING_DATE") or "")[:10].replace("-", "")

        if fx and iss_dt and len(iss_dt) == 8:
            events.append({
                "ts_code": fmt_ts(code),
                "ann_date": iss_dt,
                "rating": fx,
                "rating_type": "债券首次发行评级",
            })

        if fx != cur:
            if fx in ("AA+", "AAA") or cur in ("AA+", "AAA"):
                boundary_codes.append(code)
            elif cur and new_dt and len(new_dt) == 8:
                events.append({
                    "ts_code": fmt_ts(code),
                    "ann_date": new_dt,
                    "rating": cur,
                    "rating_type": "债券最新跟踪评级",
                })

    print(f"Initial ratings built for {len(df_bs)} CBs; extracting exact CNInfo PDF timeline for {len(boundary_codes)} AA+/AAA boundary CBs...", flush=True)
    for code in boundary_codes:
        sc = stock_map.get(code, "")
        nm = name_map.get(code, "")
        row_bs = df_bs[df_bs["SECURITY_CODE"] == code].iloc[0]
        iss_dt = str(row_bs.get("ISSUE_DATE") or row_bs.get("LISTING_DATE") or "")[:10].replace("-", "")
        cn_evs = []
        for attempt in range(2):
            cn_evs = fetch_cninfo_bond_events(code, nm, sc, iss_dt)
            if cn_evs:
                break
            time.sleep(1.0)
        for ev in cn_evs:
            events.append(ev)
        # Ensure final cur_rating at NEW_RATING_DATE is captured if PDFs were scanned/missed
        cur = norm_grade(row_bs.get("BOND_RATING"))
        new_dt = str(row_bs.get("NEW_RATING_DATE") or "")[:10].replace("-", "")
        last_cn_grade = cn_evs[-1]["rating"] if cn_evs else ""
        if cur and new_dt and len(new_dt) == 8 and cur != last_cn_grade:
            events.append({
                "ts_code": fmt_ts(code),
                "ann_date": new_dt,
                "rating": cur,
                "rating_type": "债券跟踪评级终值",
            })
        print(f"  {code} {nm}: {len(cn_evs)} CNInfo PDF events -> {[ (e['ann_date'], e['rating']) for e in cn_evs ]}", flush=True)
        time.sleep(0.35)

    out_df = (
        pd.DataFrame(events)
        .drop_duplicates(subset=["ts_code", "ann_date"], keep="first")
        .sort_values(["ts_code", "ann_date"])
    )
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_p, index=False, encoding="utf-8")
    print(f"Saved {len(out_df)} point-in-time bond rating events to {out_p}", flush=True)


if __name__ == "__main__":
    main()
