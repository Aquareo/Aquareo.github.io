# Four-Region LSM 可转债模型独立代码审计与复现指南

本文档记录 `2020–2026` 全市场可转债（`940+` 只含退市、强赎与违约券）五阶段实证验证的代码审计发现、修复逻辑、数据口径与完整本地复现命令。

## 1. 环境准备与复现步骤

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas akshare numpy pytz

# 第一步：全量抓取 2017–2026 年在市、强赎、到期及违约退市转债与正股历史（支持断点缓存）
python tools/fetch_all_cbs.py --cache-dir .cache_bonds --workers 3

# 第二步：构建全局市场日历对齐面板，运行五阶段检验并输出严格合法 JSON（allow_nan=False）
python tools/run_full_validation.py --cache-dir .cache_bonds --out dist/bonds/research_validation.json

# 第三步：更新全市场实时冻结模型前瞻快照与实盘模拟持仓账本，生成静态看板
python update_website.py
```

## 2. 独立代码审计发现与修复明细

| 审计项 | 修复前问题 | 修复方案 |
| :--- | :--- | :--- |
| **未来收益标签右端截断** | 原代码用单券自身交易日索引起跳 `k+1+h`，遇到数据集终点 `2026-11-25` 时落入 `else` 分支按末日收盘价强行截断计算，导致 `1D/5D/10D/20D` 的有效 IC 天数异常同为 `1670`。 | 统一映射至全局沪深交易日历 `M_dates`；凡 `gt + 1 + h >= len(M_dates)` 严格置 `NaN`（JSON 为 `null`），仅真实退市/强赎于 `last_gt < T_cal - 8` 的标的计入提前清盘收益。修复后各窗口有效天数严格递减。 |
| **停牌期间日历错位** | 单券若中途停牌，`k+1` 跳过停牌日，导致按“事件日后第 `h` 个成交日”而非真实市场第 `h` 日计算收益，且 `T+1` 停牌日被误算成交。 | 强制要求信号日 `gt` 与成交日 `gt+1` 在全局市场日历均放量可交易（`open>1, vol>0`）；回测持有期内若遇停牌则锁定权重顺延至复牌开盘日，禁止停牌日虚构成交与零成本调出。 |
| **Fama-MacBeth 事后评级泄漏** | `bond_cb_info_em` 仅提供最新下调后信用评级（如搜特/正邦/鸿达退市前已降为 `CC/C`），将其代入 2020–2022 截面回归构成未来信息泄漏。 | 从 Fama-MacBeth 截面回归中剔除含事后下调泄漏的评级字段，仅保留当日因果可得的 `P`、溢价率、波动率、合约剩余期限、正股 20 日动量 6 个因子。 |
| **退市违约清盘与成本** | 到期/退市末日未单独计提退出成本，亦未测试低价违约退市后流动性折价损失。 | 真实退市日末强制按收盘价平仓转入现金并计提单边退市交易费用；额外提供末日收盘 `< ¥90` 违约退市券按 `50%` 回收清盘的极端压力测试结果。 |
| **前瞻账本幸存者偏差与池容量** | 实时前瞻面板曾限流 36 只，且缺少完整组合持仓变动与退市处理记录。 | 扩大至全市场在市转债（300+ 只），在 `dist/bonds/history.json` 中每日保存全市场冻结参数信号及 `Top 20%` 周频模拟组合持仓状态与费后净值。 |
| **JSON 标准合规性** | Python 默认 `json.dumps(..., allow_nan=True)` 可能写出非标准 `NaN`。 | 全链路新增 `sanitize_json()` 并强制 `allow_nan=False`，所有缺失浮点值严格序列化为 `null`。 |

## 3. 已公开仓库脚本
* [`tools/fetch_all_cbs.py`](../tools/fetch_all_cbs.py)：全市场历史转债与正股缓存抓取器
* [`tools/run_full_validation.py`](../tools/run_full_validation.py)：无前瞻偏差全市场回测与五阶段统计验证脚本
* [`cb_pricing_model.py`](../cb_pricing_model.py)：冻结参数的 `Four-Region LSM` 及去分段消融模型
* [`update_website.py`](../update_website.py)：全市场前瞻信号追踪、模拟持仓账本更新及网页构建脚本
