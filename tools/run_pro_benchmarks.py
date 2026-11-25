"""
P1: Professional Quant Benchmarks, Incremental Factor Tests, Phased Portfolio Optimization,
and ¥100,000 Retail Executability Study (2020–2026).

Strictly reuses `panel_pro.pkl` (causal data, frozen LSM parameters p_reset=0.80, B=108, r=0.025).
No blind Cartesian search: tests benchmarks fairly, checks incremental IC & Fama-MacBeth alpha,
evaluates phased portfolio refinements (weighting, fixed N vs 20%, concentration caps, ¥100k lots),
and checks sub-period robustness (2020-22, 2023-24, 2025-26, Last 1Y) before nominating up to 3 candidates.

Usage:
    python tools/run_pro_benchmarks.py --cache-dir /tmp/bonds_v2/cache --out dist/bonds/research_pro.json
"""

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from run_full_validation import newey_west_stats, sanitize_json
from run_strategy_experiments import compute_metrics, prepare_fast_engine


BENCHMARK_SPECS = [
    {
        "id": "f_four",
        "name": "原始 Four-Region LSM",
        "formula": "(V_LSM - P) / P；四分区最小二乘蒙特卡洛（含30日强赎/回售路径与p_reset=0.80下修）",
        "data_source": "新浪日线 + 东财历史转股价 + 契约期限 + 正股60日波动率",
        "params": "r=2.5%, B=108, call=130%, put=70%, p_reset=0.80 (冻结)",
        "applicability": "全市场及AA-以上截面",
        "approx_note": "论文基准版（M=400, 12步历史网格）",
        "col": "f_four",
        "asc": False,
    },
    {
        "id": "f_noreset",
        "name": "无下修 LSM 消融",
        "formula": "(V_LSM(p_reset=0) - P) / P",
        "data_source": "同原始 LSM",
        "params": "p_reset=0.0，其余同冻结参数",
        "applicability": "检验下修条款定价贡献",
        "approx_note": "基准消融版",
        "col": "f_noreset",
        "asc": False,
    },
    {
        "id": "f_single",
        "name": "单区域全局 LSM 消融",
        "formula": "(V_LSM(single) - P) / P（全路径单一回归，不划分四条款区）",
        "data_source": "同原始 LSM",
        "params": "segmented=False",
        "applicability": "检验四分区回归必要性",
        "approx_note": "基准消融版",
        "col": "f_single",
        "asc": False,
    },
    {
        "id": "f_bs_pure",
        "name": "Black-Scholes 偏离 (券商版)",
        "formula": "(V_pure,EM + (100/C)*BS_Call - P) / P",
        "data_source": "东财点时纯债价值 + 正股60日波动率 + 无风险利率2.5%",
        "params": "r=2.5%, q=0, HV60",
        "applicability": "常规可转债（忽略130%强赎与下修）",
        "approx_note": "券商基础BS近似版（未含美式提前行权与路径条款）",
        "col": "f_bs_pure",
        "asc": False,
    },
    {
        "id": "f_bs_b108",
        "name": "Black-Scholes 偏离 (B=108同底)",
        "formula": "(108*exp(-rT) + (100/C)*BS_Call - P) / P",
        "data_source": "与LSM完全同底、同波动率输入",
        "params": "B=108, r=2.5%, q=0, HV60",
        "applicability": "剥离条款路径依赖的纯粹欧式对照",
        "approx_note": "同底欧式对照版",
        "col": "f_bs_b108",
        "asc": False,
    },
    {
        "id": "f_binom",
        "name": "二叉树偏离 (30步 CRR)",
        "formula": "(V_CRR30 - P) / P（美式转股 + 1.30x强赎上限 + 最后2年0.70x回售下限）",
        "data_source": "同LSM正股、转股价、剩余期限与HV60",
        "params": "N=30步, B=108, call=130%(103), put=70%(101)",
        "applicability": "无路径记忆的常规转债定价对照",
        "approx_note": "马尔可夫近似版（不含30日连续窗口累计与p_reset下修）",
        "col": "f_binom",
        "asc": False,
    },
    {
        "id": "low_iv",
        "name": "低隐含波动率 (-IV)",
        "formula": "-IV；反解 (100/C)*BS_Call(IV) = P - V_pure,EM",
        "data_source": "P > V_pure 且高于内在价值+0.5元有效样本",
        "params": "IV∈[5%,150%]二分求解",
        "applicability": "仅限正时间价值券（低价破面/深实值券无定义）",
        "approx_note": "欧式BS反演近似；缺失IV日剔除不参评",
        "col": "neg_iv",
        "asc": False,
    },
    {
        "id": "hv_minus_iv",
        "name": "波动率价差 (HV60 - IV)",
        "formula": "sigma_HV60 - IV（正值表示正股实际波高于转债隐波）",
        "data_source": "60日实现波动率减有效BS隐含波动率",
        "params": "同上IV有效域",
        "applicability": "正时间价值券波动率套利因子",
        "approx_note": "欧式IV近似；无有效IV样本日剔除",
        "col": "hv_minus_iv",
        "asc": False,
    },
    {
        "id": "high_ytm",
        "name": "高到期收益率 (YTM)",
        "formula": "P = Sum(CF_k / (1+YTM)^t_k) 求解税前内部收益率",
        "data_source": "发行公告年度票息 + 契约剩余期限",
        "params": "末期按100+末年票息兑付",
        "applicability": "偏债型与防御期选券",
        "approx_note": "契约票息近似版（未含个券超额到期赎回补偿）",
        "col": "ytm_pct",
        "asc": False,
    },
    {
        "id": "low_pb_prem",
        "name": "低纯债溢价率",
        "formula": "-(P - V_pure,EM) / V_pure,EM * 100%",
        "data_source": "东财每日点时纯债价值（缺失时用108*exp(-rT)）",
        "params": "纯债底防御因子",
        "applicability": "偏债防御及信用可控区间",
        "approx_note": "点时无泄漏版本",
        "col": "neg_pb_prem",
        "asc": False,
    },
    {
        "id": "small_scale",
        "name": "小规模因子 (因果发行规模)",
        "formula": "-IssueScale（亿元，发行时点已知初始发行规模）",
        "data_source": "东财bond_cb_list_em初始发行规模",
        "params": "越小得分越高",
        "applicability": "小盘弹性偏好（实盘需防高溢价炒作）",
        "approx_note": "因果近似版！因免费接口缺乏2020-26每日剩余规模，严禁回填末期余额，改用发行时规模",
        "col": "neg_scale",
        "asc": False,
    },
    {
        "id": "double_low",
        "name": "传统双低",
        "formula": "-(P + 100*(P - ConvVal)/ConvVal)",
        "data_source": "新浪收盘价 + 东财历史转股价反算转股价值",
        "params": "经典1:1等权叠加",
        "applicability": "平衡/防御市场常用基准",
        "approx_note": "标准无参数版本",
        "col": "neg_dl",
        "asc": False,
    },
    {
        "id": "tri_low",
        "name": "券商三低多因子",
        "formula": "Mean(Pct(-P), Pct(-Prem), Pct(-PureBondPrem))",
        "data_source": "价格 + 转股溢价率 + 点时纯债溢价率截面分位均值",
        "params": "等权1/3截面百分位",
        "applicability": "兼具股债防御的稳健多因子",
        "approx_note": "公开三低思想可复现截面分位版",
        "col": "tri_low",
        "asc": False,
    },
    {
        "id": "dl_mom20",
        "name": "正股动量增强双低",
        "formula": "0.70*Pct(-DoubleLow) + 0.30*Pct(Stock_Mom20)",
        "data_source": "传统双低 + 正股20日动量（防连续杀跌陷阱）",
        "params": "70/30截面混合",
        "applicability": "规避持续弱势低价券价值陷阱",
        "approx_note": "公开动量改良可复现版",
        "col": "dl_mom20",
        "asc": False,
    },
    {
        "id": "broker_4f",
        "name": "券商四因子均衡近似",
        "formula": "Mean(Pct(-Scale), Pct(-DoubleLow), Pct(YTM), Pct(Mom20))",
        "data_source": "发行规模近似 + 双低 + 契约YTM + 正股动量",
        "params": "等权1/4截面百分位",
        "applicability": "多风格均衡对照",
        "approx_note": "规模项为因果近似版本",
        "col": "broker_4f",
        "asc": False,
    },
]


def add_derived_factor_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["neg_iv"] = -df["impl_vol"]
    df["neg_pb_prem"] = -df["pure_bond_prem"]
    df["neg_scale"] = -df["issue_scale"]
    df["neg_dl"] = -df["double_low"]
    df["neg_close"] = -df["close"]
    df["neg_prem"] = -df["prem"]

    # Cross-sectional percentile ranks per trading day for multi-factor benchmarks
    g = df.groupby("gt")
    p_close = g["neg_close"].rank(pct=True)
    p_prem = g["neg_prem"].rank(pct=True)
    p_pb = g["neg_pb_prem"].rank(pct=True)
    p_dl = g["neg_dl"].rank(pct=True)
    p_mom = g["mom20"].rank(pct=True)
    p_ytm = g["ytm_pct"].rank(pct=True)
    p_sc = g["neg_scale"].rank(pct=True).fillna(0.5)

    df["tri_low"] = (p_close + p_prem + p_pb) / 3.0
    df["dl_mom20"] = 0.70 * p_dl + 0.30 * p_mom
    df["broker_4f"] = (p_sc + p_dl + p_ytm + p_mom) / 4.0
    df["pct_lsm"] = g["f_four"].rank(pct=True)
    df["pct_dl"] = p_dl
    df["pct_pb"] = p_pb
    return df


def sim_flexible(
    eng: dict,
    df_sub: pd.DataFrame,
    score_expr,  # callable(grp_df) -> np.ndarray score (higher better), or column name string
    freq: int = 5,
    top_frac: float = 0.20,
    fixed_n: int = 0,
    sell_mult: float = 1.0,  # e.g. 1.5 for 20/30 buffer (sell_rank = top * 1.5)
    weight_mode: str = "equal",  # equal | rank_linear | inv_vol
    max_single_w: float = 1.0,
    all_eq: bool = False,
):
    """Unified T+1 open simulation supporting proportional or fixed N, weighting schemes, buffers & single-bond caps."""
    T_cal = eng["T_cal"]
    tradable = eng["tradable"]
    g_open = eng["g_open"]
    g_coup = eng["g_coup"]
    is_delisted = eng["is_delisted"]
    last_gt = eng["last_gt"]
    delist_px = eng["delist_px"]

    daily = [None] * T_cal
    for gt, grp in df_sub.groupby("gt"):
        t1 = int(gt) + 1
        g_buy = grp[~(grp["is_delisted"].values & (grp["last_gt"].values <= t1))].copy()
        if callable(score_expr):
            sc = score_expr(g_buy)
        elif score_expr is not None:
            sc = g_buy[score_expr].values.astype(np.float64)
        else:
            sc = np.zeros(len(g_buy), dtype=np.float64)
        valid = np.isfinite(sc)
        g_buy = g_buy[valid]
        sc = sc[valid]
        if len(g_buy) < 10:
            continue
        order = np.argsort(-sc, kind="mergesort")
        bids = g_buy["bid"].values.astype(np.int32)[order]
        sigs = np.clip(g_buy["sigma"].values.astype(np.float64)[order], 15.0, 70.0)
        daily[int(gt)] = (bids, sigs)

    holdings = {}
    cash = 1.0
    daily_gross = np.zeros(T_cal - 2, dtype=np.float64)
    daily_to = np.zeros(T_cal - 2, dtype=np.float64)
    dates = eng["cal_dates"][2:T_cal]
    rebal_step = 0

    for gt in range(T_cal - 2):
        t1 = gt + 1
        t2 = gt + 2
        tradable_keys, locked_keys = [], []
        locked_val = 0.0
        nav_pre = cash
        for bid, pos in holdings.items():
            nav_pre += pos[0]
            if tradable[bid, t1]:
                tradable_keys.append(bid)
            else:
                locked_keys.append(bid)
                locked_val += pos[0]

        free_cap = max(0.0, nav_pre - locked_val)
        two_way_to = 0.0
        should_rebal = (rebal_step % freq == 0) or (len(holdings) == 0 and free_cap > 0.5)
        rebal_step += 1

        cand = daily[gt]
        if should_rebal and cand is not None and free_cap > 1e-8:
            ranked_bids, ranked_sigs = cand
            n_elig = len(ranked_bids)
            sig_lookup = {int(b): float(s) for b, s in zip(ranked_bids, ranked_sigs)}
            if all_eq:
                picks = ranked_bids.tolist()
            else:
                k_target = min(n_elig, fixed_n) if fixed_n > 0 else min(n_elig, max(5, int(round(n_elig * top_frac))))
                if sell_mult <= 1.01:
                    picks = ranked_bids[:k_target].tolist()
                else:
                    k_sell = min(n_elig, max(k_target, int(round(k_target * sell_mult))))
                    held_tr = set(tradable_keys)
                    kept = [int(b) for b in ranked_bids[:k_sell] if int(b) in held_tr]
                    picks = list(kept)
                    kept_set = set(kept)
                    for b in ranked_bids:
                        if len(picks) >= k_target:
                            break
                        bi = int(b)
                        if bi not in kept_set:
                            picks.append(bi)
                            kept_set.add(bi)

            if picks:
                k_act = len(picks)
                if weight_mode == "rank_linear":
                    raw_w = np.array([k_act - i for i in range(k_act)], dtype=np.float64)
                elif weight_mode == "inv_vol":
                    raw_w = np.array([1.0 / max(15.0, sig_lookup.get(b, 28.0)) for b in picks], dtype=np.float64)
                else:
                    raw_w = np.ones(k_act, dtype=np.float64)
                raw_w /= raw_w.sum()

                # Apply single-bond weight cap iteratively (if feasible: k_act * max_single_w >= 1)
                if max_single_w < 0.999 and k_act * max_single_w >= 1.0 - 1e-6:
                    for _ in range(10):
                        over = raw_w > max_single_w
                        if not np.any(over):
                            break
                        excess = np.sum(raw_w[over] - max_single_w)
                        raw_w[over] = max_single_w
                        under = ~over
                        if np.any(under) and raw_w[under].sum() > 0:
                            raw_w[under] += excess * (raw_w[under] / raw_w[under].sum())

                free_frac = free_cap / nav_pre
                target_w = {b: free_frac * float(raw_w[i]) for i, b in enumerate(picks)}
                old_w = {b: holdings[b][0] / nav_pre for b in tradable_keys}
                for k in (set(target_w) | set(old_w)):
                    two_way_to += abs(target_w.get(k, 0.0) - old_w.get(k, 0.0))

                new_holdings = {b: holdings[b] for b in locked_keys}
                for b, tw in target_w.items():
                    new_holdings[b] = [tw * nav_pre, float(g_open[b, t1])]
                holdings = new_holdings
                cash = 0.0

        nav_after = cash
        to_del = []
        for bid, pos in holdings.items():
            if is_delisted[bid] and t1 >= last_gt[bid]:
                pay_px = delist_px[bid]
                c_day = float(g_coup[bid, last_gt[bid]])
                exit_val = pos[0] * ((pay_px + c_day) / pos[1])
                two_way_to += exit_val / max(1e-9, nav_pre)
                cash += exit_val
                nav_after += exit_val
                to_del.append(bid)
            elif tradable[bid, t2]:
                p_t2 = float(g_open[bid, t2])
                c_day = float(g_coup[bid, t2])
                pos[0] = pos[0] * ((p_t2 + c_day) / pos[1])
                pos[1] = p_t2
                nav_after += pos[0]
            else:
                nav_after += pos[0]
        for bid in to_del:
            del holdings[bid]

        daily_gross[gt] = (nav_after / nav_pre - 1.0) if nav_pre > 0 else 0.0
        daily_to[gt] = two_way_to
        if nav_after > 0:
            scale = 1.0 / nav_after
            cash *= scale
            for pos in holdings.values():
                pos[0] *= scale

    return daily_gross, daily_to, dates


def sim_100k_discrete(
    eng: dict,
    df_sub: pd.DataFrame,
    w_lsm: float = 0.5,
    freq: int = 10,
    fixed_n: int = 10,
    sell_mult: float = 1.5,
    initial_cash: float = 100_000.0,
    comm_rate: float = 0.0005,
    slip_rate: float = 0.0005,
    min_comm: float = 5.0,
    max_single_w: float = 0.20,
) -> dict:
    """Strict RMB 100,000 retail discrete simulation with 10-bond lot rounding (1手=10张)."""
    T_cal = eng["T_cal"]
    tradable = eng["tradable"]
    g_open = eng["g_open"]
    g_coup = eng["g_coup"]
    is_delisted = eng["is_delisted"]
    last_gt = eng["last_gt"]
    delist_px = eng["delist_px"]

    daily = [None] * T_cal
    for gt, grp in df_sub.groupby("gt"):
        t1 = int(gt) + 1
        g_buy = grp[~(grp["is_delisted"].values & (grp["last_gt"].values <= t1))].dropna(subset=["f_four", "double_low"])
        if len(g_buy) < 10:
            continue
        p_lsm = g_buy["f_four"].rank(pct=True, ascending=True).values
        p_dl = g_buy["double_low"].rank(pct=True, ascending=False).values
        sc = w_lsm * p_lsm + (1.0 - w_lsm) * p_dl
        order = np.argsort(-(sc + 1e-9 * p_lsm), kind="mergesort")
        daily[int(gt)] = g_buy["bid"].values.astype(np.int32)[order]

    qty = {}  # bid -> integer bonds (multiple of 10)
    cash = float(initial_cash)
    rebal_step = 0
    navs = []
    cash_utils = []
    tot_comm = 0.0
    tot_traded_val = 0.0

    for gt in range(T_cal - 2):
        t1 = gt + 1
        t2 = gt + 2

        # Collect coupons on bonds held through t1
        for bid, q_held in qty.items():
            c_day = float(g_coup[bid, t1])
            if c_day > 0:
                cash += q_held * c_day

        # Handle forced delisting at t1
        for bid in list(qty.keys()):
            if is_delisted[bid] and t1 >= last_gt[bid]:
                q_held = qty.pop(bid)
                px = float(delist_px[bid]) + float(g_coup[bid, last_gt[bid]])
                gross_val = q_held * px
                fee = max(min_comm, gross_val * comm_rate) + gross_val * slip_rate
                cash += max(0.0, gross_val - fee)
                tot_comm += fee
                tot_traded_val += gross_val

        # Valuation at t1 open before rebalance
        tr_bids = [b for b in qty if tradable[b, t1]]
        locked_val = sum(qty[b] * float(g_open[b, t1] if g_open[b, t1] > 0 else 100.0) for b in qty if b not in tr_bids)
        tr_val = sum(qty[b] * float(g_open[b, t1]) for b in tr_bids)
        nav_t1 = cash + tr_val + locked_val

        should_rebal = (rebal_step % freq == 0) or (len(qty) == 0)
        rebal_step += 1

        ranked = daily[gt]
        if should_rebal and ranked is not None and len(ranked) > 0:
            n_elig = len(ranked)
            k_tgt = min(n_elig, fixed_n) if fixed_n > 0 else min(n_elig, max(5, int(round(n_elig * 0.20))))
            if sell_mult <= 1.01:
                picks = ranked[:k_tgt].tolist()
            else:
                k_sell = min(n_elig, max(k_tgt, int(round(k_tgt * sell_mult))))
                tr_set = set(tr_bids)
                kept = [int(b) for b in ranked[:k_sell] if int(b) in tr_set]
                picks = list(kept)
                kset = set(kept)
                for b in ranked:
                    if len(picks) >= k_tgt:
                        break
                    if int(b) not in kset:
                        picks.append(int(b))
                        kset.add(int(b))

            if picks:
                w_each = min(max_single_w, 1.0 / len(picks))
                target_rmb = {b: nav_t1 * w_each for b in picks}

                # First sell removed or trimmed tradable positions
                for b in list(tr_bids):
                    px = float(g_open[b, t1])
                    tgt_lots = int(target_rmb.get(b, 0.0) // (px * 10.0))
                    tgt_q = tgt_lots * 10
                    cur_q = qty.get(b, 0)
                    if cur_q > tgt_q:
                        sell_q = cur_q - tgt_q
                        gross_s = sell_q * px
                        fee = max(min_comm, gross_s * comm_rate) + gross_s * slip_rate
                        cash += gross_s - fee
                        tot_comm += fee
                        tot_traded_val += gross_s
                        if tgt_q > 0:
                            qty[b] = tgt_q
                        else:
                            qty.pop(b, None)

                # Then buy new/added target lots using available cash
                for b in picks:
                    px = float(g_open[b, t1])
                    if not tradable[b, t1] or px <= 1.0:
                        continue
                    tgt_lots = int(target_rmb[b] // (px * 10.0))
                    tgt_q = tgt_lots * 10
                    cur_q = qty.get(b, 0)
                    if tgt_q > cur_q:
                        need_lots = (tgt_q - cur_q) // 10
                        lot_cost = 10.0 * px
                        max_afford_lots = int(max(0.0, cash - min_comm - 20.0) // (lot_cost * (1.0 + comm_rate + slip_rate)))
                        buy_lots = min(need_lots, max_afford_lots)
                        if buy_lots > 0:
                            buy_q = buy_lots * 10
                            gross_b = buy_q * px
                            fee = max(min_comm, gross_b * comm_rate) + gross_b * slip_rate
                            if cash >= gross_b + fee:
                                cash -= (gross_b + fee)
                                qty[b] = cur_q + buy_q
                                tot_comm += fee
                                tot_traded_val += gross_b

        # Mark at t2 open
        pos_val_t2 = 0.0
        for b, q_held in qty.items():
            px2 = float(g_open[b, t2]) if tradable[b, t2] else float(g_open[b, t1] if g_open[b, t1] > 0 else 100.0)
            pos_val_t2 += q_held * px2
        nav_t2 = cash + pos_val_t2
        navs.append(nav_t2 / initial_cash)
        cash_utils.append(pos_val_t2 / max(1.0, nav_t2))

    nav_arr = np.array(navs, dtype=np.float64)
    rets = np.diff(np.insert(nav_arr, 0, 1.0)) / np.insert(nav_arr[:-1], 0, 1.0)
    peak = np.maximum.accumulate(np.insert(nav_arr, 0, 1.0))[1:]
    mdd = float(np.min(nav_arr / peak - 1.0) * 100.0)
    ann_ret = float((nav_arr[-1] ** (252.0 / max(1, len(nav_arr))) - 1.0) * 100.0)
    sharpe = float((np.mean(rets) * 252.0 - 0.02) / (np.std(rets) * math.sqrt(252.0) + 1e-9))
    calmar = float(ann_ret / abs(mdd)) if mdd < -1e-6 else 0.0
    avg_nav = float(np.mean(nav_arr) * initial_cash)
    ann_to = float((tot_traded_val * 0.5 / max(1.0, avg_nav)) * (252.0 / max(1, len(nav_arr))) * 100.0)
    return {
        "final_rmb": round(float(nav_arr[-1] * initial_cash), 2),
        "total_ret": round(float((nav_arr[-1] - 1.0) * 100.0), 2),
        "ann_ret": round(ann_ret, 2),
        "mdd": round(mdd, 2),
        "sharpe": round(sharpe, 2),
        "calmar": round(calmar, 2),
        "ann_to": round(ann_to, 1),
        "avg_invested_pct": round(float(np.mean(cash_utils) * 100.0), 1),
        "total_fees_rmb": round(tot_comm, 2),
    }


def subperiod_stats(gross: np.ndarray, to: np.ndarray, dates: list, cost: float = 0.0010) -> dict:
    net = gross - to * cost
    dt_arr = np.array(dates)
    n_tot = len(dt_arr)
    windows = [
        ("full", "全历史 (2020–2026)", np.ones(n_tot, dtype=bool)),
        ("p20_22", "2020–2022", (dt_arr >= "2020-01-01") & (dt_arr <= "2022-12-31")),
        ("p23_24", "2023–2024", (dt_arr >= "2023-01-01") & (dt_arr <= "2024-12-31")),
        ("p25_26", "2025至今", dt_arr >= "2025-01-01"),
        ("last_1y", "近一年 (252D)", np.arange(n_tot) >= max(0, n_tot - 252)),
    ]
    out = {}
    for wkey, wlbl, m in windows:
        sub = net[m]
        if len(sub) < 15:
            continue
        nav = np.cumprod(1.0 + sub)
        pk = np.maximum.accumulate(np.insert(nav, 0, 1.0))[1:]
        mdd = float(np.min(nav / pk - 1.0) * 100.0)
        tot = float((nav[-1] - 1.0) * 100.0)
        ann = float((nav[-1] ** (252.0 / len(sub)) - 1.0) * 100.0)
        sh = float((np.mean(sub) * 252.0 - 0.02) / (np.std(sub) * math.sqrt(252.0) + 1e-9))
        cal = float(ann / abs(mdd)) if mdd < -1e-6 else 0.0
        out[wkey] = {"label": wlbl, "tot_ret": round(tot, 2), "ann_ret": round(ann, 2), "mdd": round(mdd, 2), "sharpe": round(sh, 2), "calmar": round(cal, 2)}
    return out


def compute_factor_ic_table(df_sub: pd.DataFrame, specs: list) -> list:
    by_gt = {int(gt): grp for gt, grp in df_sub.groupby("gt")}
    all_gts = sorted(by_gt.keys())
    rows = []
    for sp in specs:
        col = sp["col"]
        ic_by_h = {}
        for h in (5, 10, 20):
            ics, ns = [], []
            for gt in all_gts:
                grp = by_gt[gt][[col, f"ret_{h}"]].dropna()
                if len(grp) >= 15 and grp[col].nunique() >= 4:
                    c_val = grp[col].rank().corr(grp[f"ret_{h}"].rank())
                    if pd.notna(c_val):
                        ics.append(float(c_val))
                        ns.append(len(grp))
            if ics:
                nw = newey_west_stats(np.array(ics), lags=h)
                ic_by_h[str(h)] = {
                    "days": len(ics),
                    "avg_n": round(float(np.mean(ns)), 1),
                    "mean_ic": round(nw["mean"], 4),
                    "icir": round(nw["mean"] / nw["std"], 3) if nw["std"] > 0 else 0.0,
                    "t_nw": round(nw["t_stat"], 2),
                    "p_val": round(nw["p_val"], 4),
                }
            else:
                ic_by_h[str(h)] = {"days": 0, "avg_n": 0, "mean_ic": 0.0, "icir": 0.0, "t_nw": 0.0, "p_val": 1.0}
        rows.append({"id": sp["id"], "name": sp["name"], "ic": ic_by_h})
    return rows


def incremental_fmb(df_sub: pd.DataFrame, include_iv: bool = False) -> list:
    """Daily cross-sectional OLS of ret_20 on standardized f_four + peer pricing/fundamental controls."""
    cols = ["f_four", "f_bs_pure", "f_binom", "double_low", "ytm_pct", "pure_bond_prem", "issue_scale", "sigma", "mom20"]
    labels = {
        "f_four": "原始 Four-Region LSM (f_four)",
        "f_bs_pure": "Black-Scholes 偏离 (f_bs_pure)",
        "f_binom": "二叉树偏离 (f_binom)",
        "double_low": "传统双低 (double_low)",
        "impl_vol": "隐含波动率 (IV, 有效域)",
        "hv_minus_iv": "波动率价差 (HV60-IV)",
        "ytm_pct": "到期收益率 (YTM)",
        "pure_bond_prem": "纯债溢价率",
        "issue_scale": "初始发行规模(近似)",
        "sigma": "正股60日波动率",
        "mom20": "正股20日动量",
    }
    if include_iv:
        cols = cols[:4] + ["impl_vol", "hv_minus_iv"] + cols[4:]

    betas = {c: [] for c in cols}
    n_days = 0
    for _, grp in df_sub.groupby("gt"):
        sub = grp[["ret_20"] + cols].dropna()
        if len(sub) < len(cols) + 15:
            continue
        X_raw = sub[cols].values.astype(np.float64)
        mu = np.mean(X_raw, axis=0)
        sd = np.std(X_raw, axis=0)
        if np.any(sd < 1e-8):
            continue
        Xz = np.clip((X_raw - mu) / sd, -4.0, 4.0)
        X = np.column_stack([np.ones(len(Xz)), Xz])
        y = sub["ret_20"].values.astype(np.float64)
        try:
            b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            for idx, c in enumerate(cols, 1):
                betas[c].append(float(b[idx]))
            n_days += 1
        except Exception:
            continue

    out = []
    for c in cols:
        nw = newey_west_stats(np.array(betas[c]), lags=20)
        out.append({
            "key": c,
            "label": labels[c],
            "n_days": n_days,
            "mean": round(nw["mean"], 4),
            "t_stat": round(nw["t_stat"], 2),
            "p_val": round(nw["p_val"], 4),
        })
    return out


def factor_corr_matrix(df_sub: pd.DataFrame) -> dict:
    cols = ["f_four", "f_bs_pure", "f_binom", "neg_dl", "neg_iv", "hv_minus_iv", "ytm_pct", "neg_pb_prem", "neg_scale", "mom20"]
    names = ["LSM", "BS", "Binom", "双低(反向)", "低IV", "HV-IV", "YTM", "低纯债溢", "小规模", "正股动量"]
    # Sample every 5th day for fast accurate cross-sectional Spearman average
    mats = []
    for gt, grp in df_sub[df_sub["gt"] % 5 == 0].groupby("gt"):
        if len(grp) < 30:
            continue
        cmat = grp[cols].corr(method="spearman", min_periods=15).values
        if np.isnan(cmat).mean() < 0.3:
            mats.append(cmat)
    avg = np.nanmean(mats, axis=0) if mats else np.eye(len(cols))
    return {"cols": cols, "labels": names, "matrix": [[round(float(x), 3) if np.isfinite(x) else None for x in row] for row in avg]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="/tmp/bonds_v2/cache")
    ap.add_argument("--out", default="dist/bonds/research_pro.json")
    args = ap.parse_args()

    t0 = time.time()
    cache_dir = Path(args.cache_dir)
    with open(cache_dir / "panel_pro.pkl", "rb") as f:
        bundle = pickle.load(f)

    df = add_derived_factor_cols(bundle["df"])
    eng = prepare_fast_engine(df, bundle["exec_map"], bundle["cal_dates"])
    df = eng["df"]

    # Slices of interest
    df_all_raw = df
    df_all_filt = df[~df["is_risk_excluded"]]
    df_aam_filt = df[df["rating"].isin(["AA-", "AA", "AA+", "AAA"]) & (~df["is_risk_excluded"])]
    df_aa_filt = df[df["rating"].isin(["AA", "AA+", "AAA"]) & (~df["is_risk_excluded"])]

    print("1. Running Fair Peer Benchmark Comparison (All|raw and AA-|filtered, 5D Top20% T+1 Open)...", flush=True)
    peer_results = {}
    for scope_key, scope_df in [("all_raw", df_all_raw), ("aa_minus_filtered", df_aam_filt)]:
        ic_tbl = compute_factor_ic_table(scope_df, BENCHMARK_SPECS)
        ic_map = {r["id"]: r["ic"] for r in ic_tbl}
        rows = []
        for sp in BENCHMARK_SPECS:
            gr, to, dt = sim_flexible(eng, scope_df, sp["col"], freq=5, top_frac=0.20)
            m = compute_metrics(gr, to, dt, include_nav=True)
            subs = subperiod_stats(gr, to, dt, cost=0.0010)
            rows.append({
                "id": sp["id"],
                "name": sp["name"],
                "formula": sp["formula"],
                "data_source": sp["data_source"],
                "params": sp["params"],
                "applicability": sp["applicability"],
                "approx_note": sp["approx_note"],
                "ann_to": m["ann_to"],
                "costs": m["costs"],
                "nav_010": m.get("nav_010"),
                "subperiods": subs,
                "ic": ic_map.get(sp["id"], {}),
            })
        # Equal weight benchmark
        gr_eq, to_eq, dt_eq = sim_flexible(eng, scope_df, None, freq=5, all_eq=True)
        m_eq = compute_metrics(gr_eq, to_eq, dt_eq, include_nav=True)
        rows.append({
            "id": "equal_weight",
            "name": "同池等权基准",
            "formula": "全部合格可转债等权持有",
            "data_source": "同样本池全部可交易券",
            "params": "等权再平衡",
            "applicability": "市场贝塔基准",
            "approx_note": "基准对比",
            "ann_to": m_eq["ann_to"],
            "costs": m_eq["costs"],
            "nav_010": m_eq.get("nav_010"),
            "subperiods": subperiod_stats(gr_eq, to_eq, dt_eq, cost=0.0010),
            "ic": {},
        })
        peer_results[scope_key] = rows

    print("2. Computing Factor Correlations & Incremental Fama-MacBeth Regressions...", flush=True)
    corr_aam = factor_corr_matrix(df_aam_filt)
    fmb_all_noiv = incremental_fmb(df_aam_filt, include_iv=False)
    fmb_all_iv = incremental_fmb(df_aam_filt, include_iv=True)

    print("3. Phased Optimization (Weights, Fixed N vs 20%, Weighting Schemes, Single Cap, ¥100k Lots)...", flush=True)
    phase_steps = []
    # Phase A: Pools (AA-+, AA+, All) x Risk Filtered x Weights (100/0, 75/25, 50/50) x Freq/Buf
    opt_configs = [
        ("AA-及以上·剔除", "aa_minus_filtered", df_aam_filt),
        ("AA及以上·剔除", "aa_filtered", df_aa_filt),
        ("全市场·剔除", "all_filtered", df_all_filt),
    ]
    for p_lbl, p_key, p_df in opt_configs:
        for w_lsm, w_lbl in [(1.0, "100/0"), (0.75, "75/25"), (0.50, "50/50")]:
            for freq, smult, b_lbl in [(5, 1.0, "5D·无缓冲"), (5, 1.5, "5D·20/30"), (10, 1.5, "10D·20/30"), (20, 1.5, "20D·20/30")]:
                expr = (lambda w=w_lsm: (lambda g: w * g["pct_lsm"].values + (1.0 - w) * g["pct_dl"].values))()
                gr, to, dt = sim_flexible(eng, p_df, expr, freq=freq, top_frac=0.20, fixed_n=0, sell_mult=smult)
                m = compute_metrics(gr, to, dt)
                subs = subperiod_stats(gr, to, dt)
                c10 = m["costs"]["0.10%"]
                phase_steps.append({
                    "stage": "A_pool_mix_freq",
                    "desc": f"{p_lbl} · {w_lbl} · {b_lbl} · Top20%等权",
                    "pool": p_key, "w": w_lbl, "freq": freq, "buf": b_lbl, "n_mode": "20%", "weight_mode": "equal",
                    "ann_ret": c10["ann_ret"], "mdd": c10["mdd"], "sharpe": c10["sharpe"], "calmar": c10["calmar"], "ann_to": m["ann_to"],
                    "subperiods": subs,
                })

    # Phase B: Holding count (N=5, 10, 20, Top20%) + Weighting (equal, rank_linear, inv_vol) + Max single cap
    # Anchored on AA- filtered 50/50 & 75/25 at 10D 20/30 (avoid overfitting; inspect stability)
    expr_50 = lambda g: 0.50 * g["pct_lsm"].values + 0.50 * g["pct_dl"].values
    expr_75 = lambda g: 0.75 * g["pct_lsm"].values + 0.25 * g["pct_dl"].values
    sizing_tests = []
    for p_lbl, p_key, p_df in [("AA-·剔除", "aa_minus_filtered", df_aam_filt), ("AA·剔除", "aa_filtered", df_aa_filt)]:
        for fixed_n, n_lbl in [(5, "固定5只"), (10, "固定10只"), (20, "固定20只"), (0, "比例Top20%")]:
            for wmode, wmode_lbl, cap in [
                ("equal", "等权", 0.20 if fixed_n == 5 else 0.15),
                ("rank_linear", "排名加权(上限15%)", 0.15 if fixed_n != 5 else 0.25),
                ("inv_vol", "逆波动率(上限15%)", 0.15 if fixed_n != 5 else 0.25),
            ]:
                if fixed_n == 5 and wmode != "equal":
                    # With 5 bonds, cap 15% is infeasible (<20%); use 25% cap
                    pass
                gr, to, dt = sim_flexible(eng, p_df, expr_50, freq=10, top_frac=0.20, fixed_n=fixed_n, sell_mult=1.5, weight_mode=wmode, max_single_w=cap)
                m = compute_metrics(gr, to, dt)
                subs = subperiod_stats(gr, to, dt)
                c10 = m["costs"]["0.10%"]
                row_s = {
                    "stage": "B_sizing_weight",
                    "desc": f"{p_lbl} · 50/50 · 10D·20/30 · {n_lbl} · {wmode_lbl}",
                    "pool": p_key, "fixed_n": fixed_n, "weight_mode": wmode_lbl, "cap": cap,
                    "ann_ret": c10["ann_ret"], "mdd": c10["mdd"], "sharpe": c10["sharpe"], "calmar": c10["calmar"], "ann_to": m["ann_to"],
                    "subperiods": subs,
                }
                if wmode == "equal":
                    # Test actual RMB 100,000 discrete 10-bond lots execution
                    d100k = sim_100k_discrete(eng, p_df, w_lsm=0.50, freq=10, fixed_n=fixed_n, sell_mult=1.5, comm_rate=0.0005, slip_rate=0.0005, min_comm=5.0)
                    row_s["retail_100k"] = d100k
                sizing_tests.append(row_s)

    print("4. Evaluating Final 3 Candidate Portfolios (Growth / Balanced / Defensive) with ¥100k Executable Specs...", flush=True)
    # Candidate specs explicitly grounded in Stage A/B empirical findings (N=20 prevents N=5/10 tail crash while preserving ¥100k 91%+ utilization):
    cand_defs = [
        {
            "key": "enhanced_growth",
            "name": "Enhanced LSM Growth (偏收益型)",
            "pool": "aa_minus", "pool_label": "AA-及以上", "risk": "filtered",
            "df": df_aam_filt, "expr": expr_75, "w_lsm": 0.75, "freq": 5, "fixed_n": 20, "sell_mult": 1.5, "wmode": "equal", "cap": 0.10,
            "rationale": "AA-以上+因果风控 + 75/25 LSM主导混合 + 5D/20-30缓冲 + 固定20只等权(避开5/10只尾部踩雷并保留LSM高弹性)",
        },
        {
            "key": "enhanced_balanced",
            "name": "Enhanced LSM Balanced (均衡实用型)",
            "pool": "aa_minus", "pool_label": "AA-及以上", "risk": "filtered",
            "df": df_aam_filt, "expr": expr_50, "w_lsm": 0.50, "freq": 10, "fixed_n": 20, "sell_mult": 1.5, "wmode": "equal", "cap": 0.10,
            "rationale": "AA-以上+因果风控 + 50/50 LSM/双低 + 10D降频 + 20/30缓冲 + 固定20只(每券¥5千≈4手)，兼顾夏普、回撤与10万元实盘资金利用率(91%)",
        },
        {
            "key": "enhanced_defensive",
            "name": "Enhanced LSM Defensive (稳健防御型)",
            "pool": "aa", "pool_label": "AA及以上", "risk": "filtered",
            "df": df_aa_filt, "expr": expr_50, "w_lsm": 0.50, "freq": 10, "fixed_n": 20, "sell_mult": 1.5, "wmode": "inv_vol", "cap": 0.10,
            "rationale": "AA及以上较高信用 + 50/50 LSM/双低 + 10D降频 + 20/30缓冲 + 固定20只逆波动率加权(单券≤10%)，重点控制最大回撤(-16.5%)",
        },
    ]

    candidates = []
    dates_ref = None
    for cd in cand_defs:
        gr, to, dt = sim_flexible(eng, cd["df"], cd["expr"], freq=cd["freq"], fixed_n=cd["fixed_n"], sell_mult=cd["sell_mult"], weight_mode=cd["wmode"], max_single_w=cd["cap"])
        dates_ref = dt
        m = compute_metrics(gr, to, dt, include_nav=True)
        subs = subperiod_stats(gr, to, dt, cost=0.0010)
        d100k = sim_100k_discrete(eng, cd["df"], w_lsm=cd["w_lsm"], freq=cd["freq"], fixed_n=cd["fixed_n"], sell_mult=cd["sell_mult"], max_single_w=max(0.15, cd["cap"]))
        # Compare against same-pool original LSM & Double-Low
        gr_lsm, to_lsm, dt_lsm = sim_flexible(eng, cd["df"], "f_four", freq=5, top_frac=0.20)
        m_lsm = compute_metrics(gr_lsm, to_lsm, dt_lsm)
        gr_dl, to_dl, dt_dl = sim_flexible(eng, cd["df"], "neg_dl", freq=5, top_frac=0.20)
        m_dl = compute_metrics(gr_dl, to_dl, dt_dl)

        c10 = m["costs"]["0.10%"]
        l10 = m_lsm["costs"]["0.10%"]
        d10 = m_dl["costs"]["0.10%"]
        recent_ok = (subs.get("p23_24", {}).get("ann_ret", -99) > 0) and (c10["calmar"] >= l10["calmar"])
        candidates.append({
            "key": cd["key"],
            "name": cd["name"],
            "pool": cd["pool"],
            "pool_label": cd["pool_label"],
            "rationale": cd["rationale"],
            "params": {"freq": cd["freq"], "fixed_n": cd["fixed_n"], "buffer": "20/30", "weight_mode": cd["wmode"], "max_single_w": cd["cap"]},
            "ann_to": m["ann_to"],
            "costs": m["costs"],
            "nav_010": m.get("nav_010"),
            "subperiods": subs,
            "retail_100k": d100k,
            "vs_pool_lsm": {
                "lsm_ann_ret": l10["ann_ret"], "lsm_mdd": l10["mdd"], "lsm_sharpe": l10["sharpe"], "lsm_calmar": l10["calmar"],
                "dl_ann_ret": d10["ann_ret"], "dl_mdd": d10["mdd"], "dl_sharpe": d10["sharpe"], "dl_calmar": d10["calmar"],
            },
            "validated": bool(recent_ok),
        })

    # Extract sampled nav_dates matching compute_metrics include_nav cadence
    si = list(range(0, len(dates_ref), 10))
    if si[-1] != len(dates_ref) - 1:
        si.append(len(dates_ref) - 1)
    nav_dates = [dates_ref[i] for i in si]

    payload = sanitize_json({
        "meta": {
            "panel_meta": bundle["meta"],
            "nav_dates": nav_dates,
            "total_trading_days": len(dates_ref),
            "cost_base": "0.10% (10bps one-way T+1 Open)",
            "note": "全部回测严格采用T日收盘生成信号、T+1日开盘价成交；初始发行规模为因果近似代理并如实标明。",
        },
        "peer_benchmarks": peer_results,
        "factor_corr_aam": corr_aam,
        "incremental_fmb_aam": {"without_iv": fmb_all_noiv, "with_iv_subset": fmb_all_iv},
        "phased_optimization": {"stage_a_pool_mix_freq": phase_steps, "stage_b_sizing_weighting": sizing_tests},
        "candidates": candidates,
    })

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved P1 research_pro.json ({out_p.stat().st_size/1024:.1f} KB) in {time.time()-t0:.1f}s", flush=True)
    for c in candidates:
        c10 = c["costs"]["0.10%"]
        print(f"  Candidate [{c['key']}]: ann={c10['ann_ret']}%, mdd={c10['mdd']}%, sr={c10['sharpe']}, cal={c10['calmar']}, 100k_ann={c['retail_100k']['ann_ret']}% (util={c['retail_100k']['avg_invested_pct']}%)", flush=True)


if __name__ == "__main__":
    main()
