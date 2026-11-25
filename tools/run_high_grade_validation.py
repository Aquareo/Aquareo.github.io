"""AA+/AAA point-in-time validation; reuses the audited full-market panel.
Run from repo root:
  python tools/run_high_grade_validation.py --ratings-csv rating_events.csv
or (Tushare cb_rating permission required; set TUSHARE_TOKEN locally):
  python tools/run_high_grade_validation.py --fetch-tushare
CSV columns: ts_code, ann_date, rating, rating_type; rating_date optional.
Never use latest_rating or backfill. No change to pricing parameters.
"""
import argparse
import os
import pickle
from pathlib import Path
import json
import numpy as np
import pandas as pd
from run_full_validation import analyze_panel

GRADES = ["D", "C", "CC", "CCC", "B-", "B", "B+", "BB-", "BB", "BB+",
          "BBB-", "BBB", "BBB+", "A-", "A", "A+", "AA-", "AA", "AA+", "AAA"]
GRADE_ORDER = {v: i + 1 for i, v in enumerate(GRADES)}

def ticker(s):
    return str(s).split(".")[0].strip().zfill(6)

def parse_date(s):
    return pd.to_datetime(s.astype(str).str.replace("-", "", regex=False)
                          .str.replace(r"\.0$", "", regex=True), format="%Y%m%d", errors="coerce")

def fetch_ratings(codes):
    import tushare as ts
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise SystemExit("Set TUSHARE_TOKEN in your local environment; never commit credentials.")
    api = ts.pro_api(token)
    # Shanghai convertible bond codes begin with 11; Shenzhen with 12.
    symbols = [c + (".SH" if c.startswith("11") else ".SZ") for c in codes]
    frames = []
    for i in range(0, len(symbols), 20):
        part = api.cb_rating(ts_code=",".join(symbols[i:i+20]),
                             fields="ts_code,ann_date,rating_date,rating_type,rating")
        if part is None or len(part) >= 3000:
            raise RuntimeError("Unverified or potentially truncated rating response")
        frames.append(part)
    return pd.concat(frames, ignore_index=True)

def attach_ratings(panel, events):
    required = {"ts_code", "ann_date", "rating", "rating_type"}
    if not required.issubset(events.columns):
        raise ValueError("Missing rating event fields: " + str(required - set(events.columns)))
    ev = events.copy()
    ev["ticker"] = ev["ts_code"].map(ticker)
    ev["published"] = parse_date(ev["ann_date"])
    # Require explicit bond-level type; issuer-level or undocumented type is not a substitute.
    typ = ev["rating_type"].fillna("").astype(str)
    print("Available rating types:", typ.value_counts().to_dict(), flush=True)
    ev = ev[typ.str.contains("债券|债项", regex=True) & ~typ.str.contains("主体|发行人", regex=True)]
    ev = ev.dropna(subset=["published"])
    ev["rating"] = ev["rating"].fillna("").astype(str).str.strip().str.upper().str.replace("＋", "+").str.replace("－", "-")
    # Conflicting same-day announcements: use the lower grade; unknown event -> ineligible.
    ev["rank"] = ev["rating"].map(GRADE_ORDER).fillna(0).astype(int)
    ev = ev.sort_values(["ticker", "published", "rank"]).drop_duplicates(["ticker", "published"])
    p = panel.copy()
    p["ticker"] = p["ticker"].map(ticker)
    p["signal_date"] = pd.to_datetime(p["date"])
    # Date-only publication timestamp: conservative next-day availability.
    joined = pd.merge_asof(p.sort_values("signal_date"),
                           ev[["ticker", "published", "rating", "rank"]].sort_values("published"),
                           left_on="signal_date", right_on="published", by="ticker",
                           direction="backward", allow_exact_matches=False)
    return joined

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", default=".cache_bonds/panel_audit.pkl")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--ratings-csv")
    source.add_argument("--fetch-tushare", action="store_true")
    ap.add_argument("--rating-cache", default=".cache_bonds/rating_events.csv")
    ap.add_argument("--out", default="dist/bonds/research_validation_aa_plus.json")
    ap.add_argument("--grades", default="AA+,AAA", help="Comma-separated bond grades to keep, e.g. AA+,AAA or AAA")
    ap.add_argument("--min-coverage", type=float, default=0.90)
    args = ap.parse_args()
    target_grades = [g.strip().upper() for g in args.grades.split(",") if g.strip()]
    # Only open a trusted locally generated pickle.
    with open(args.panel, "rb") as f:
        bundle = pickle.load(f)
    panel = bundle["panel"]
    codes = sorted(set(panel["ticker"].map(ticker)))
    if args.fetch_tushare:
        ev = fetch_ratings(codes)
        cache = Path(args.rating_cache)
        cache.parent.mkdir(parents=True, exist_ok=True)
        ev.to_csv(cache, index=False)
    else:
        ev = pd.read_csv(args.ratings_csv, dtype=str)
    joined = attach_ratings(panel, ev)
    known = joined["rank"].fillna(0) > 0
    high = joined[joined["rating"].isin(target_grades)].copy()
    daily = high.groupby("gt").size()
    coverage = float(known.mean())
    coverage_by_year = known.groupby(joined["signal_date"].dt.year).mean()
    print("Rating coverage overall:", coverage, "by year:", coverage_by_year.to_dict(), flush=True)
    print("Eligible grade counts:", high["rating"].value_counts().to_dict(), flush=True)
    if coverage < args.min_coverage or (coverage_by_year < args.min_coverage).any():
        raise SystemExit("Insufficient historical rating coverage; no result published.")
    if daily.empty or (daily >= 35).sum() < 100:
        raise SystemExit("Too few high-grade observations for the existing validation engine.")
    # Keep the full execution map so downgraded/suspended positions can still be liquidated
    # according to the same simulator; restrict new picks and the benchmark to eligible rows.
    filtered = dict(bundle, panel=high)
    out = Path(args.out)
    analyze_panel(filtered, out)
    result = json.loads(out.read_text(encoding="utf-8"))
    result["rating_filter"] = {
        "eligible_grades": target_grades,
        "availability": "strictly after announcement date",
        "rating_coverage_all_bond_days": round(coverage, 6),
        "coverage_by_year": {str(k): round(float(v), 6) for k, v in coverage_by_year.items()},
        "eligible_bond_days": int(len(high)),
        "eligible_unique_bonds": int(high["ticker"].nunique()),
        "median_daily_eligible": float(daily.median()),
        "source": "Tushare cb_rating" if args.fetch_tushare else "user-supplied rating event CSV",
        "limitations": "Inherited execution, delisting and numerical approximations from full-market engine. Rating eligibility checked at signal date; exits on scheduled rebalance, subject to tradability."
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print("Saved", out)

if __name__ == "__main__":
    main()
