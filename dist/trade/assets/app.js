"use strict";
const $ = id => document.getElementById(id), C = TradingCore, L = SignalLab;
const e = v => String(v ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, n = 2) => typeof v === "number" && Number.isFinite(v) ? v.toLocaleString("zh-CN", { minimumFractionDigits: n, maximumFractionDigits: n }) : "—";
const pct = (v, signed = false) => v == null || !Number.isFinite(v) ? "—" : `${signed && v > 0 ? "+" : ""}${fmt(v)}%`;
const color = v => v == null || v === 0 ? "muted" : v > 0 ? "up" : "down";
const safeURL = v => { try { const u = new URL(v); return ["https:", "http:"].includes(u.protocol) ? u.href : "#"; } catch { return "#"; } };
const shortTime = v => v ? v.replace("T", " ").slice(5, 16) : "—";
const sourceTime = v => v ? v.replace("T", " ").slice(11, 19) : "—";
const storage = {
  get: (key, fallback) => { try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; } },
  set: (key, value) => { try { localStorage.setItem(key, JSON.stringify(value)); } catch {} }
};

let chartMinutes = 5;
const params = new URLSearchParams(location.search);
const validStrats = new Set(L.STRATEGY_IDS);
const state = {
  manifest: null, stocks: [], quotes: {},
  code: params.get("stock") || "300750",
  strategy: validStrats.has(params.get("strategy")) ? params.get("strategy") : "donchian",
  quality: params.get("quality") === "original" ? "original" : "filtered",
  calendar: [], fundFlows: new Map(), rankCache: new Map(), focusSignal: null,
  mode: "m5", view: "market", detail: "signals", query: "", filter: "all", sort: "name", descending: true,
  limit: 110, offset: 0, paused: false, researchRefreshing: false, scanning: false, scanDone: 0,
  quoteBusy: false, quoteError: null, quoteMissing: [], quoteFailures: 0, quoteNextAt: 0, startedAt: Date.now(),
  tTrades: storage.get("trading.t-trades.v1", []), datasets: new Map(), barRequests: new Set(), details: new Map(),
  baselines: new Map(), journal: storage.get("trading.journal.v3", []), positions: storage.get("trading.positions.v2", {}),
  points: [], hover: null, quoteCycles: 0, quoteSuccess: null, beepAt: 0
};

const statusNames = {
  buy_pending: "买入信号", sell_pending: "卖出信号", sell_locked: "风险预警 · T+1",
  holding: "模拟持有", watching: "观望中", insufficient: "数据不足"
};

const stockBy = code => state.stocks.find(s => s.code === code);
const selected = () => stockBy(state.code);
const dataset = (code = state.code, mode = state.mode) => state.datasets.get(`${code}_${mode}`);
const result = (code = state.code, mode = state.mode, strategy = state.strategy) => dataset(code, mode)?.results?.[strategy];

async function getJSON(url) {
  const r = await fetch(url, { cache: "no-cache" });
  if (!r.ok) throw Error(`读取失败 ${r.status}`);
  return r.json();
}
function costOptions(code) {
  const s = state.manifest?.settings || {};
  return { code, commission: s.commission_rate ?? 0.0003, tax: s.sell_tax_rate ?? 0.0005, slippage: s.slippage_rate ?? 0.001, calendar: state.calendar };
}
function evaluationOptions(code, mode = "m5") {
  return { ...costOptions(code), mode, calendar: state.calendar, threshold: Number($("accuracyThreshold")?.value) || 1 };
}
function compute(data, code, mode) {
  data.bars = data.bars.filter(b => (!b.synthetic && !b.provisional) || C.asEpoch(b.time) > Date.now() - 2000);
  data.results = {};
  const daily = mode === "day" ? data.bars : dataset(code, "day")?.bars || [];
  const now = Date.now();
  for (const strategy of L.STRATEGY_IDS) {
    data.results[strategy] = L.analyze(data.bars, { now, ...costOptions(code), strategy, mode, daily, quality: state.quality });
  }
  data.rankReports = null;
  data.summaryCompare = null;
  return data;
}

function refreshWindow() {
  const m = C.market(), day = new Date(m.day + "T12:00:00+08:00").getUTCDay();
  return m.open || (day > 0 && day < 6 && ((m.time > "11:30:00" && m.time < "11:35:00") || (m.time > "15:00:00" && m.time < "15:05:00")));
}
function syncURL() {
  const u = new URL(location.href);
  u.searchParams.set("stock", state.code);
  u.searchParams.set("strategy", state.strategy);
  u.searchParams.set("quality", state.quality);
  history.replaceState(null, "", u);
}
function announce(text) { if ($("bottomStatus")) $("bottomStatus").textContent = text; }

function mergeForming(data, quote) {
  if (!data || !quote || !C.market().open || C.quoteState(quote).kind !== "ok") return;
  const time = quote.as_of, minute = Number(time.slice(14, 16)), hour = Number(time.slice(11, 13));
  if (!((hour === 9 && minute >= 30) || hour === 10 || (hour === 11 && minute < 30) || hour === 13 || hour === 14)) return;
  const bucketEpoch = Math.ceil((C.asEpoch(time) + 1) / 300000) * 300000, bucket = C.stamp(bucketEpoch);
  const last = data.bars.at(-1);
  if (!last || last.time > bucket || bucket.slice(0, 10) !== time.slice(0, 10)) return;
  if (last.time === bucket) {
    last.close = quote.price; last.high = Math.max(last.high, quote.price); last.low = Math.min(last.low, quote.price);
  } else {
    const total = data.bars.filter(b => b.time.slice(0, 10) === time.slice(0, 10)).reduce((a, b) => a + b.volume, 0);
    data.bars.push({ time: bucket, open: quote.price, close: quote.price, high: quote.price, low: quote.price, volume: Math.max(0, (quote.volume || 0) - total), synthetic: true });
  }
  compute(data, quote.code, "m5");
}

function freshness(code = state.code, mode = state.mode) {
  const d = dataset(code, mode), q = state.quotes[code];
  if (!d) return { ok: false, text: "等待 K 线" };
  if (d.error || d.snapshot) return { ok: false, text: d.snapshot ? "离线快照" : "K 线异常" };
  if (state.quoteMissing.includes(code)) return { ok: false, text: "行情待恢复" };
  const qs = C.quoteState(q);
  if (qs.kind !== "ok") return { ok: false, text: qs.text };
  if (C.market().open && mode === "m5") {
    const now = Date.now(), time = C.market().time, last = d.bars.at(-1)?.time;
    if (last && C.asEpoch(last) < now - 420000 && time > "09:36:00" && !(time < "13:06:00" && time > "13:00:00")) {
      return { ok: false, text: "分钟线滞后" };
    }
  }
  return { ok: true, text: C.market().open ? "盘中实时" : "最新收盘" };
}

function getLatestLevels(r, q) {
  if (!r?.bars?.length) return null;
  const idx = r.bars.findIndex(b => b.time === r.last_confirmed);
  const i = idx >= 0 ? idx : r.bars.length - 1;
  const ind = r.indicators?.[i], bar = r.bars[i];
  if (!ind || !bar) return null;
  const px = q?.price > 0 ? q.price : bar.close;
  if (state.strategy === "donchian") {
    const buyPct = ind.upper ? (ind.upper / px - 1) * 100 : null;
    const sellPct = ind.lower ? (ind.lower / px - 1) * 100 : null;
    return { type: "price", buyLevel: ind.upper, sellLevel: ind.lower, buyPct, sellPct, px };
  }
  return { type: "rsi", rsi: ind.rsi14, prevRsi: ind.prevRsi, px };
}

function sideLabel(code, mode = "m5") {
  const r = result(code, mode);
  if (!r) return { text: "—", className: "none", priority: 0 };
  const q = state.quotes[code];
  if (r.preview) return { text: r.preview.side === "buy" ? "形成买点" : "形成卖点", className: "forming", priority: 3 };
  if (r.pending) {
    if (r.pending.side === "buy") return { text: "买入", className: "buy", priority: 5 };
    return { text: r.status === "sell_locked" ? "T+1卖" : "卖出", className: "sell", priority: 4 };
  }
  const lv = getLatestLevels(r, q);
  if (lv?.type === "price" && lv.buyPct != null && lv.buyPct > 0 && lv.buyPct <= 1.8) {
    return { text: `差${lv.buyPct.toFixed(1)}%买`, className: "forming", priority: 2 };
  }
  if (lv?.type === "rsi" && lv.rsi != null && lv.rsi <= 35) {
    return { text: `RSI ${lv.rsi.toFixed(0)}`, className: "forming", priority: 2 };
  }
  if (r.status === "holding") return { text: "持有", className: "holding", priority: 1 };
  return { text: "观望", className: "none", priority: 0 };
}

function renderRadar() {
  const box = $("signalRadar");
  if (!box || !state.stocks.length) return;
  const buys = [], sells = [], near = [];
  for (const s of state.stocks) {
    const sig = sideLabel(s.code, "m5");
    const q = state.quotes[s.code];
    const px = q?.price ? `¥${fmt(q.price)}` : "";
    if (sig.className === "buy") buys.push({ s, sig, px });
    else if (sig.className === "sell") sells.push({ s, sig, px });
    else if (sig.className === "forming") near.push({ s, sig, px });
  }
  if (!buys.length && !sells.length && !near.length) {
    box.hidden = false;
    box.innerHTML = `<span class="radar-empty">当前 38 只自选暂无新确认买/卖点 · 切换上方「全池信号」或下方策略查看历史买/卖位置</span>`;
    return;
  }
  box.hidden = false;
  const chips = (arr, cls) => arr.slice(0, 6).map(({ s, sig, px }) =>
    `<button class="radar-chip ${cls}" data-open-stock="${s.code}">${e(s.name)} <small>${px}</small> <em>${e(sig.text)}</em></button>`
  ).join("");
  box.innerHTML = `
    ${buys.length ? `<div class="radar-group"><span class="radar-title up">买点触发 (${buys.length})</span>${chips(buys, "buy")}</div>` : ""}
    ${sells.length ? `<div class="radar-group"><span class="radar-title down">卖点触发 (${sells.length})</span>${chips(sells, "sell")}</div>` : ""}
    ${near.length ? `<div class="radar-group"><span class="radar-title warn">临近 / 形成中 (${near.length})</span>${chips(near, "forming")}</div>` : ""}
  `;
}

async function refreshQuotes({ force = false } = {}) {
  if (state.quoteBusy || state.paused || (!force && Date.now() < state.quoteNextAt)) return;
  state.quoteBusy = true;
  try {
    const indices = [{ code: "000001", exchange: "SH" }, { code: "399001", exchange: "SZ" }, { code: "000300", exchange: "SH" }];
    const quotes = await LiveFeed.quotes([...state.stocks, ...indices]);
    for (const [code, q] of Object.entries(quotes)) if (!state.quotes[code] || q.as_of >= state.quotes[code].as_of) state.quotes[code] = q;
    state.quoteMissing = state.stocks.filter(s => !quotes[s.code]).map(s => s.code);
    state.quoteSuccess = Date.now();
    if (state.quoteMissing.length) { state.quoteFailures++; state.quoteError = `${state.stocks.length - state.quoteMissing.length}/${state.stocks.length} 更新成功`; }
    else { state.quoteFailures = 0; state.quoteError = null; }
    state.quoteNextAt = state.quoteError ? Date.now() + (refreshWindow() ? Math.min(120000, 15000 * 2 ** Math.min(state.quoteFailures, 3)) : 300000) : 0;
    for (const s of state.stocks) if (!state.quoteMissing.includes(s.code)) mergeForming(dataset(s.code, "m5"), state.quotes[s.code]);
    renderQuoteStrip(); renderRadar(); renderWatchlist(); renderInstrument(); renderActionGuide();
    if (state.view === "market") { renderChart(); renderDetails(); }
    if (state.view === "grid") renderGrid();
    updateConnection();
  } catch (error) {
    state.quoteFailures++; state.quoteMissing = state.stocks.map(s => s.code); state.quoteError = error.message;
    state.quoteNextAt = Date.now() + (refreshWindow() ? Math.min(120000, 15000 * 2 ** Math.min(state.quoteFailures, 3)) : 300000);
    updateConnection(); renderWatchlist(); renderActionGuide(); renderDetails();
  } finally { state.quoteBusy = false; }
}

function updateConnection() {
  const q = state.quotes[state.code], health = C.quoteState(q), paused = state.paused || document.hidden;
  $("connection").textContent = paused ? "已暂停" : state.quoteError ? "连接异常" : health.text;
  $("connection").className = `connection ${paused ? "" : state.quoteError ? "error" : health.kind}`;
  const notice = [];
  if (state.quoteError) notice.push(`行情连接待恢复（${state.quoteError}），保留最近数据。`);
  $("notice").hidden = !notice.length; $("notice").textContent = notice.join(" ");
  $("quoteTime").textContent = q ? sourceTime(q.as_of) : "未连接";
  $("bottomStatus").textContent = paused ? "自动刷新已暂停" : `行情 ${state.quoteSuccess ? C.stamp(state.quoteSuccess).slice(11) : "—"} · ${state.stocks.length} 只自选`;
}

async function loadBars(code, mode = "m5", { force = false } = {}) {
  const key = `${code}_${mode}`, old = dataset(code, mode);
  if (state.barRequests.has(key)) return;
  if (old && !force && Date.now() - old.loadedAt < 25000) return;
  state.barRequests.add(key);
  if (old) old.loading = true;
  try {
    const data = await LiveFeed.bars(stockBy(code), mode);
    if (data.quote && (!state.quotes[code] || data.quote.as_of >= state.quotes[code].as_of)) {
      state.quotes[code] = data.quote;
      state.quoteMissing = state.quoteMissing.filter(x => x !== code);
      if (!state.quoteMissing.length && state.quoteError) { state.quoteError = null; state.quoteFailures = 0; state.quoteNextAt = 0; state.quoteSuccess = Date.now(); updateConnection(); }
    }
    Object.assign(data, { loadedAt: Date.now(), snapshot: false, error: null, loading: false });
    if (mode === "m5") mergeForming(data, state.quotes[code]);
    compute(data, code, mode); state.datasets.set(key, data);
    if (mode === "m5") captureSignals(code, data);
    else if (dataset(code, "m5")) { compute(dataset(code, "m5"), code, "m5"); updateJournal(code); }
  } catch (error) {
    if (old) { old.error = error.message; old.loading = false; }
    else if (mode === "day") {
      try { const saved = await getJSON(`data/stocks/${code}.json`); state.datasets.set(key, compute({ bars: saved.bars.map(b => ({ ...b, time: b.date })), loadedAt: Date.now(), snapshot: true, error: error.message, source: saved.history_source || "日线快照", basis: "前复权日线" }, code, mode)); } catch {}
    } else if (mode === "m5") {
      try { const saved = await getJSON(`data/intraday/${code}.json`); state.datasets.set(key, compute({ ...saved, loadedAt: Date.now(), snapshot: true, error: error.message }, code, mode)); } catch { state.datasets.set(key, { bars: [], results: null, error: error.message, loadedAt: Date.now(), snapshot: true }); }
    }
  }
  state.barRequests.delete(key);
  if (code === state.code && mode === state.mode) renderSelected();
  renderRadar(); renderWatchlist();
  if (state.view === "grid") renderGrid();
  if (state.view === "research" && !state.scanning && !state.researchRefreshing) renderResearch();
}

async function scanPool(force = false) {
  if (state.scanning || state.paused) return;
  state.scanning = true; state.scanDone = 0;
  const ordered = [selected(), ...state.stocks.filter(s => s.code !== state.code)].filter(Boolean);
  let cursor = 0;
  async function worker() {
    while (cursor < ordered.length) {
      const s = ordered[cursor++];
      if (!dataset(s.code, "day") || Date.now() - dataset(s.code, "day").loadedAt > 300000) await loadBars(s.code, "day", { force: true });
      await loadBars(s.code, "m5", { force });
      state.scanDone++;
      $("scanStatus").textContent = `扫描 ${state.scanDone}/${ordered.length}`;
      $("gridScanStatus").textContent = `已扫描 ${state.scanDone}/${ordered.length}`;
    }
  }
  try { await Promise.all([worker(), worker(), worker(), worker()]); }
  finally {
    state.scanning = false;
    const count = state.stocks.filter(s => dataset(s.code, "m5")?.results && !dataset(s.code, "m5").error).length;
    $("scanStatus").textContent = `${count}/${state.stocks.length} 在线`;
    $("gridScanStatus").textContent = `${count}/${state.stocks.length} 已连接`;
    renderRadar();
    if (state.view === "research") renderResearch();
  }
}

function captureSignals(code, data) {
  const observed = C.stamp(Date.now()), quote = state.quotes[code], fresh = freshness(code, "m5").ok;
  const reports = {};
  for (const quality of ["original", "filtered"]) {
    reports[quality] = {};
    for (const strategy of L.STRATEGY_IDS) {
      const r = quality === state.quality ? data.results[strategy] : L.analyze(data.bars, { ...costOptions(code), now: data.results[strategy].compute_epoch, strategy, quality, mode: "m5", daily: dataset(code, "day")?.bars || [] });
      reports[quality][strategy] = r;
      const key = `${code}_${strategy}_${quality}_${L.VERSION}`, previous = state.baselines.get(key);
      if (previous && fresh) {
        for (const order of r.signals.filter(o => o.time > previous)) {
          const id = `${key}_${order.time}_${order.side}`;
          if (!state.journal.some(x => x.id === id)) {
            state.journal.push({ id, version: L.VERSION, quality, code, name: stockBy(code).name, strategy, time: order.time, side: order.side, price: r.bars[order.index].close, reason: order.reason, checks: structuredClone(order.checks), regime: order.regime, observed_at: observed, alert_outcomes: {}, observed_price: quote?.as_of >= order.time && quote.as_of <= observed && quote.price > 0 ? quote.price : null, quote_as_of: quote?.as_of, t1_locked: order.t1_locked, costs: costOptions(code), threshold: 1, outcomes: {} });
            if (quality === state.quality && strategy === state.strategy) beep();
          }
        }
      }
      if (r.last_confirmed) state.baselines.set(key, r.last_confirmed);
    }
  }
  updateJournal(code, reports);
}

function updateJournal(code, reports = {}) {
  const data = dataset(code, "m5");
  if (!data?.bars?.length) return;
  for (const event of state.journal.filter(x => x.code === code)) {
    const report = reports[event.quality]?.[event.strategy] || (event.quality === state.quality ? data.results?.[event.strategy] : null);
    if (!event.execution && report) {
      const order = report.signals.find(x => x.time === event.time && x.side === event.side);
      if (order?.status === "filled") { const fill = report.bars[order.fill_index]; event.execution = { status: "filled", entry_open: fill.open, fill_bar_time: order.fill_bar_time, fill_time: order.fill_time, fill_price: order.fill_price }; }
      else if (order?.status === "cancelled") event.execution = { status: "cancelled", cancel_time: order.cancel_time };
    }
    if (event.execution) {
      const ev = L.evaluate({ ...event, ...event.execution }, data.bars, { ...event.costs, mode: "m5", calendar: state.calendar, threshold: event.threshold });
      for (const h of L.HORIZONS) if (!event.outcomes[h] && ev.outcomes[h]) event.outcomes[h] = ev.outcomes[h];
    }
  }
  storage.set("trading.journal.v3", state.journal);
}

function beep() {
  if (!$("soundToggle")?.checked || Date.now() - state.beepAt < 10000) return;
  state.beepAt = Date.now();
  try {
    const ctx = state.audio || new AudioContext(); state.audio = ctx;
    const osc = ctx.createOscillator(), gain = ctx.createGain();
    osc.connect(gain); gain.connect(ctx.destination); osc.frequency.value = 740; gain.gain.value = 0.05;
    osc.start(); osc.stop(ctx.currentTime + 0.14);
  } catch {}
}

async function loadDetail(code) {
  if (state.details.has(code)) { refreshNews(code); return; }
  try {
    const detail = await getJSON(`data/stocks/${code}.json`);
    state.details.set(code, detail);
    if (!dataset(code, "day")) {
      state.datasets.set(`${code}_day`, compute({ bars: detail.bars.map(b => ({ time: b.date, ...b })), snapshot: true, loadedAt: 0, source: detail.history_source || "备用日线", basis: "前复权日线", fetched_at: detail.history_fetched_at }, code, "day"));
    }
    if (code === state.code) { renderActionGuide(); renderDetails(); if (state.mode === "day") renderSelected(); }
    refreshNews(code);
  } catch (error) { announce(`读取股票详情失败：${error.message}`); }
}

async function refreshNews(code = state.code) {
  const detail = state.details.get(code);
  if (!detail || detail.news_loading) return;
  detail.news_loading = true;
  try { detail.news = await LiveFeed.news(code); detail.news_fetched_at = C.stamp(Date.now()); detail.news_status = "ok"; detail.news_live_error = null; }
  catch (error) { detail.news_live_error = error.message; }
  finally { detail.news_loading = false; if (code === state.code && state.detail === "flow_news") renderDetails(); }
}

async function refreshDailyResearch() {
  if (state.researchRefreshing) return;
  state.researchRefreshing = true;
  let cursor = 0;
  async function worker() { while (cursor < state.stocks.length) await loadBars(state.stocks[cursor++].code, "day", { force: true }); }
  try { await Promise.all([worker(), worker(), worker(), worker()]); }
  finally { state.researchRefreshing = false; if (state.view === "research") renderResearch(); }
}

function selectStock(code) {
  if (!stockBy(code)) return;
  state.code = code; state.offset = 0; syncURL();
  renderWatchlist(); renderSelected(); state.focusSignal = null;
  loadDetail(code); loadBars(code, state.mode);
  if (state.detail === "flow_news") refreshFundFlow(code);
}

function renderQuoteStrip() {
  $("indexQuotes").innerHTML = [["000001", "上证指数"], ["399001", "深证成指"], ["000300", "沪深 300"]].map(([code, name]) => {
    const q = state.quotes[code];
    return `<span>${name}<b class="${color(q?.change_pct)}">${fmt(q?.price)}</b><em class="${color(q?.change_pct)}">${pct(q?.change_pct, true)}</em></span>`;
  }).join("");
}

function renderWatchlist() {
  let stocks = state.stocks.filter(s => (!state.query || `${s.code} ${s.name} ${s.original_code}`.includes(state.query)));
  if (state.filter !== "all") {
    stocks = stocks.filter(s => {
      const sig = sideLabel(s.code);
      return sig.className === state.filter;
    });
  }
  if (state.sort !== "name") {
    stocks.sort((a, b) => {
      const field = state.sort === "price" ? "price" : "change_pct";
      return ((state.quotes[a.code]?.[field] ?? -Infinity) - (state.quotes[b.code]?.[field] ?? -Infinity)) * (state.descending ? -1 : 1);
    });
  } else {
    // Bubble active signals near the top when default sort is used
    stocks = [...stocks].sort((a, b) => sideLabel(b.code).priority - sideLabel(a.code).priority);
  }
  $("watchCount").textContent = stocks.length;
  const scroll = $("watchRows").scrollTop;
  $("watchRows").innerHTML = stocks.length ? stocks.map(s => {
    const q = state.quotes[s.code], sig = sideLabel(s.code);
    return `<button class="watch-row ${s.code === state.code ? "selected" : ""}" role="listitem" data-stock="${s.code}" aria-pressed="${s.code === state.code}">
      <span class="symbol"><b>${e(s.name)}</b><small>${s.code}</small></span>
      <span class="price ${color(q?.change_pct)}">${fmt(q?.price)}</span>
      <span class="change ${color(q?.change_pct)}">${pct(q?.change_pct, true)}</span>
      <span class="signal-mini ${sig.className}">${sig.text}</span>
    </button>`;
  }).join("") : '<div class="empty">无符合条件的股票</div>';
  $("watchRows").scrollTop = scroll;
}

function renderInstrument() {
  const s = selected(); if (!s) return;
  const q = state.quotes[s.code] || {};
  $("stockName").textContent = s.name;
  $("stockCode").textContent = `${s.exchange} ${s.code}`;
  $("stockBoard").textContent = s.board;
  $("lastPrice").textContent = fmt(q.price);
  $("lastPrice").className = color(q.change_pct);
  $("priceChange").textContent = `${q.change > 0 ? "+" : ""}${fmt(q.change)}`;
  $("priceChangePct").textContent = pct(q.change_pct, true);
  $("priceChange").className = color(q.change_pct);
  $("priceChangePct").className = color(q.change_pct);
  $("instrumentTime").textContent = q.as_of ? `${sourceTime(q.as_of)} 行情` : "等待行情";
  $("quoteDetails").innerHTML = [
    ["今开", fmt(q.open)], ["昨收", fmt(q.previous_close)],
    ["最高", fmt(q.high)], ["最低", fmt(q.low)],
    ["换手", pct(q.turnover_pct)], ["成交量", q.volume == null ? "—" : `${fmt(q.volume / 10000)}万手`]
  ].map(([label, value]) => `<div><span>${label}</span><b>${value}</b></div>`).join("");
}

function displaySignals(code = state.code, mode = state.mode, strategy = state.strategy) {
  const report = result(code, mode, strategy);
  if (!report) return [];
  const signals = new Map(report.signals.map(x => [`${x.time}_${x.side}`, x]));
  if (mode === "m5") {
    for (const event of state.journal.filter(x => x.code === code && x.strategy === strategy && x.quality === state.quality)) {
      const index = report.bars.findIndex(b => b.time === event.time); if (index < 0) continue;
      const execution = event.execution || { status: "pending" }, fillIndex = report.bars.findIndex(b => b.time === execution.fill_bar_time);
      signals.set(`${event.time}_${event.side}`, { ...event, ...execution, index, fill_index: fillIndex, journalId: event.id, liveRecord: true });
    }
  }
  return [...signals.values()].sort((a, b) => a.time.localeCompare(b.time));
}

function renderSelected() {
  renderInstrument();
  renderActionGuide();
  renderChart();
  renderDetails();
  renderSignalInspector();
}

function renderActionGuide() {
  const box = $("decision");
  if (!box) return;
  const s = selected(); if (!s) return;
  const q = state.quotes[s.code], r = result(s.code, state.mode, state.strategy);
  const lv = getLatestLevels(r, q);
  const events = r ? L.observations(r, { ...evaluationOptions(s.code, state.mode), threshold: 1 }) : [];
  const buyStats = L.stats(events, 3, { side: "buy" });
  const m = r?.metrics;

  let badgeText = "观望等待中", badgeCls = "wait", plainExplain = "";
  if (r?.pending?.side === "buy") {
    badgeText = "触发买点 · 可关注买入";
    badgeCls = "buy";
    plainExplain = `已确认买入信号：${r.pending.reason}。`;
  } else if (r?.pending?.side === "sell") {
    badgeText = r.status === "sell_locked" ? "触发卖点 (今日买入 T+1 锁定)" : "触发卖点 · 注意止盈/止损";
    badgeCls = "sell";
    plainExplain = `已确认卖出信号：${r.pending.reason}。`;
  } else if (r?.preview) {
    badgeText = r.preview.side === "buy" ? "盘中形成买点中" : "盘中形成卖点中";
    badgeCls = "forming";
    plainExplain = `当前 K 线尚未收盘，若维持当前价格到收线，将触发${r.preview.side === "buy" ? "买入" : "卖出"}信号。`;
  } else if (lv?.type === "price") {
    if (lv.buyPct != null && lv.buyPct > 0 && lv.buyPct <= 1.8) {
      badgeText = "逼近买点 · 等突破";
      badgeCls = "forming";
    } else if (r?.status === "holding") {
      badgeText = "模拟持仓中 · 未到卖点";
      badgeCls = "holding";
    }
    plainExplain = `现价 ¥${fmt(lv.px)}。${lv.buyPct > 0 ? `只要向上突破 ¥${fmt(lv.buyLevel)}（还差 +${fmt(lv.buyPct, 1)}%），就会发出买入信号；` : "价格已站上突破线，等待量价过滤或下一根 K 线确认；"}若向下跌破 ¥${fmt(lv.sellLevel)}（${pct(lv.sellPct, true)}）则发出卖出/止损信号。`;
  } else if (lv?.type === "rsi") {
    if (lv.rsi != null && lv.rsi <= 35) {
      badgeText = "接近超卖低吸区";
      badgeCls = "forming";
    } else if (r?.status === "holding") {
      badgeText = "模拟持仓中 · 未到止盈";
      badgeCls = "holding";
    }
    plainExplain = `当前 RSI14 为 ${fmt(lv.rsi, 1)}。当跌破 30 进入超卖并拐头向上突破 30 时触发“低吸买点”；反弹至 ≥55 时触发“止盈卖点”。`;
  } else {
    plainExplain = "正在等待 K 线数据加载完成后计算具体买卖价位。";
  }

  let levelsHtml = "";
  if (lv?.type === "price") {
    levelsHtml = `
      <div class="trigger-box">
        <div class="trigger-item">
          <span class="trigger-label up">买入条件（突破即买）</span>
          <strong class="mono">¥${fmt(lv.buyLevel)}</strong>
          <small class="mono muted">${lv.buyPct != null ? (lv.buyPct > 0 ? `再涨 +${fmt(lv.buyPct, 1)}% 触发买入` : "已站上突破位") : "—"}</small>
        </div>
        <div class="trigger-item">
          <span class="trigger-label down">卖出条件（跌破即卖）</span>
          <strong class="mono">¥${fmt(lv.sellLevel)}</strong>
          <small class="mono muted">${lv.sellPct != null ? `跌 ${pct(lv.sellPct, true)} 触发止损/卖出` : "—"}</small>
        </div>
      </div>
    `;
  } else if (lv?.type === "rsi") {
    const rsiVal = lv.rsi;
    levelsHtml = `
      <div class="trigger-box">
        <div class="trigger-item">
          <span class="trigger-label">当前强弱指标 RSI14</span>
          <strong class="mono ${rsiVal <= 30 ? "up" : rsiVal >= 55 ? "down" : ""}">${fmt(rsiVal, 1)}</strong>
          <small class="muted">${rsiVal <= 30 ? "处于超卖低吸区" : rsiVal >= 55 ? "处于超买止盈区" : "处于正常震荡区间"}</small>
        </div>
        <div class="trigger-item">
          <span class="trigger-label">买卖规则</span>
          <strong style="font-size:13px;">买: ≤30 拐头向上</strong>
          <small class="muted">卖: 反弹到 ≥55 止盈</small>
        </div>
      </div>
    `;
  } else {
    levelsHtml = `<div class="trigger-box"><div class="trigger-item"><span class="muted">加载中…</span></div></div>`;
  }

  const stratButtons = L.STRATEGY_IDS.map(id =>
    `<button class="seg-btn ${state.strategy === id ? "active" : ""}" data-strategy-switch="${id}">${e(C.STRATEGIES[id].name)}</button>`
  ).join("");

  box.className = `signal-hero ${badgeCls}`;
  box.innerHTML = `
    <div class="hero-topbar">
      <div class="seg-group">${stratButtons}</div>
      <button class="filter-toggle ${state.quality === "filtered" ? "on" : ""}" data-detail-action="${state.quality === "filtered" ? "original" : "filtered"}">
        趋势量价过滤：${state.quality === "filtered" ? "开启" : "关闭"}
      </button>
      <span class="hero-win mono">闭环胜率 ${m?.closed_trades ? pct(m.win_rate, false) : "—"} (${m?.winning_trades ?? 0}/${m?.closed_trades ?? 0}) · 3日买点胜率 ${buyStats.n ? pct(buyStats.accuracy, false) : "—"}</span>
    </div>
    <div class="hero-main-row">
      <div class="hero-verdict">
        <div class="verdict-top">
          <span class="verdict-pill ${badgeCls}">${e(badgeText)}</span>
          <span class="verdict-time mono">${state.mode === "day" ? "日线" : "5分钟"} · ${e(freshness(s.code, state.mode).text)}</span>
        </div>
        <p class="verdict-reason">${e(plainExplain)}</p>
      </div>
      ${levelsHtml}
    </div>
  `;
}

function svgPath(rows, field, x, y) {
  let path = "", continuous = false;
  rows.forEach((row, i) => {
    const value = row?.[field];
    if (value == null || !Number.isFinite(value)) { continuous = false; return; }
    path += `${continuous ? "L" : "M"}${x(i).toFixed(2)},${y(value).toFixed(2)} `;
    continuous = true;
  });
  return path;
}

function renderChart() {
  if (state.view !== "market") return;
  const data = dataset(), r = result(), svg = $("mainChart");
  state.points = []; $("chartTooltip").hidden = true;
  if (!data?.bars.length || !r) {
    $("chartMessage").hidden = false;
    $("chartMessage").textContent = data?.error ? `K 线暂不可用 · ${data.error}` : "正在读取 K 线…";
    svg.innerHTML = ""; return;
  }
  $("chartMessage").hidden = true;
  const grouped = state.mode === "m5" && chartMinutes > 5;
  const source = grouped ? C.chartBars(r.bars.filter(b => !b.synthetic && C.asEpoch(b.time) <= Date.now() - 2000), chartMinutes) : r.bars;
  if (!source.length) { svg.innerHTML = ""; $("chartMessage").hidden = false; $("chartMessage").textContent = "等待完整周期 K 线"; return; }

  const total = source.length, end = Math.max(1, total - state.offset), start = Math.max(0, end - state.limit), bars = source.slice(start, end);
  const indicators = grouped ? C.indicators(source) : r.indicators;
  const inds = indicators.slice(start, end);
  const visible = new Map();
  bars.forEach((bar, i) => {
    for (const t of grouped ? r.bars.filter(b => b.time >= bar.first && b.time <= bar.last).map(b => b.time) : [bar.time]) visible.set(t, i);
  });

  const width = Math.max(340, $("chartWrap").clientWidth), height = $("chartWrap").clientHeight, left = 14, right = width - 66, top = 20, bottom = height - 88;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  const lastInd = inds.at(-1);
  const extraLevels = state.strategy === "donchian" && lastInd ? [lastInd.upper, lastInd.lower] : [];
  const values = bars.flatMap((b, i) => [b.high, b.low, inds[i]?.ma20, inds[i]?.ma60, ...extraLevels]).filter(Number.isFinite);
  const low = Math.min(...values), high = Math.max(...values), padding = (high - low) * 0.12 || 1, min = low - padding, max = high + padding;
  const x = i => left + (i + 0.5) / bars.length * (right - left);
  const y = v => bottom - (v - min) / (max - min) * (bottom - top);
  const candle = Math.max(1.5, Math.min(10, (right - left) / bars.length * 0.65));

  const theme = getComputedStyle(document.body);
  const grid = theme.getPropertyValue("--line").trim(), muted = theme.getPropertyValue("--muted").trim();
  const red = theme.getPropertyValue("--red").trim(), green = theme.getPropertyValue("--green").trim(), bg = theme.getPropertyValue("--bg").trim();

  let out = "";
  for (let j = 0; j < 5; j++) {
    const value = min + (max - min) * j / 4, yy = y(value);
    out += `<line x1="0" y1="${yy}" x2="${right}" y2="${yy}" stroke="${grid}" stroke-width="0.7"/><text x="${right + 8}" y="${yy + 3}" fill="${muted}" font-size="11">${fmt(value)}</text>`;
  }
  const count = width < 500 ? 3 : 5;
  for (let j = 0; j < count; j++) {
    const i = Math.round((bars.length - 1) * j / (count - 1));
    const text = state.mode === "m5" ? bars[i].time.slice(5, 16).replace("T", " ") : bars[i].time.slice(2);
    out += `<text x="${x(i)}" y="${height - 10}" fill="${muted}" text-anchor="${j === 0 ? "start" : j === count - 1 ? "end" : "middle"}" font-size="10">${text}</text>`;
  }

  // Subtle MA20 & MA60 lines
  out += `<path d="${svgPath(inds, "ma20", x, y)}" fill="none" stroke="#c59b49" stroke-width="1.15" opacity="0.85"/>`;
  out += `<path d="${svgPath(inds, "ma60", x, y)}" fill="none" stroke="#6f87ad" stroke-width="1.15" opacity="0.85"/>`;

  // Donchian Channel + Horizontal Trigger Reference Lines
  if (state.strategy === "donchian") {
    out += `<path d="${svgPath(inds, "upper", x, y)}" fill="none" stroke="${red}" stroke-width="0.9" stroke-dasharray="3 3" opacity="0.35"/>`;
    out += `<path d="${svgPath(inds, "lower", x, y)}" fill="none" stroke="${green}" stroke-width="0.9" stroke-dasharray="3 3" opacity="0.35"/>`;
    if (lastInd?.upper && lastInd.upper >= min && lastInd.upper <= max) {
      const uy = y(lastInd.upper);
      out += `<line x1="0" x2="${right}" y1="${uy}" y2="${uy}" stroke="${red}" stroke-width="1" stroke-dasharray="4 3" opacity="0.75"/>`;
      out += `<rect x="${right + 2}" y="${uy - 8}" width="62" height="16" rx="2" fill="${red}" opacity="0.15"/><text x="${right + 33}" y="${uy + 3}" text-anchor="middle" fill="${red}" font-size="10" font-weight="600">买 ${fmt(lastInd.upper)}</text>`;
    }
    if (lastInd?.lower && lastInd.lower >= min && lastInd.lower <= max) {
      const ly = y(lastInd.lower);
      out += `<line x1="0" x2="${right}" y1="${ly}" y2="${ly}" stroke="${green}" stroke-width="1" stroke-dasharray="4 3" opacity="0.75"/>`;
      out += `<rect x="${right + 2}" y="${ly - 8}" width="62" height="16" rx="2" fill="${green}" opacity="0.15"/><text x="${right + 33}" y="${ly + 3}" text-anchor="middle" fill="${green}" font-size="10" font-weight="600">卖 ${fmt(lastInd.lower)}</text>`;
    }
  }

  // Candlesticks + Volume
  const maxVolume = Math.max(1, ...bars.map(b => b.volume));
  bars.forEach((b, i) => {
    const xx = x(i), cl = b.close >= b.open ? red : green;
    const forming = b.provisional || C.asEpoch(b.time) > Date.now() - 2000;
    const h = b.volume / maxVolume * 40;
    out += `<g opacity="${forming ? 0.55 : 1}">
      <line x1="${xx}" x2="${xx}" y1="${y(b.high)}" y2="${y(b.low)}" stroke="${cl}" stroke-width="1.1"/>
      <rect x="${xx - candle / 2}" y="${Math.min(y(b.open), y(b.close))}" width="${candle}" height="${Math.max(1.2, Math.abs(y(b.open) - y(b.close)))}" fill="${cl}"/>
      <rect x="${xx - candle / 2}" y="${height - 29 - h}" width="${candle}" height="${h}" fill="${cl}" opacity="0.35"/>
    </g>`;
  });

  // Prominent Buy & Sell Signal Markers
  const markerSlots = new Map();
  for (const order of displaySignals()) {
    if (!visible.has(order.time) || order.status === "cancelled") continue;
    const i = visible.get(order.time), b = bars[i], isBuy = order.side === "buy";
    const xx = x(i), slotKey = i + order.side, slot = markerSlots.get(slotKey) || 0;
    markerSlots.set(slotKey, slot + 1);
    const baseY = isBuy ? y(b.low) + 14 + slot * 22 : y(b.high) - 14 - slot * 22;
    const cl = isBuy ? red : green;
    const label = `${isBuy ? "B" : "S"} ${fmt(b.close)}`;
    const boxW = label.length * 6.4 + 10;
    const rectY = isBuy ? baseY + 2 : baseY - 18;
    out += `<g data-signal-time="${e(order.time)}" ${order.journalId ? `data-journal-id="${e(order.journalId)}"` : ""} tabindex="0" role="button" style="cursor:pointer;">
      <title>点击查看 ${isBuy ? "买入" : "卖出"} 详情 (${e(order.reason)})</title>
      <path d="${isBuy ? `M${xx} ${y(b.low) + 3}l-4 7h8Z` : `M${xx} ${y(b.high) - 3}l-4 -7h8Z`}" fill="${cl}"/>
      <rect x="${xx - boxW / 2}" y="${rectY}" width="${boxW}" height="16" rx="3" fill="${cl}"/>
      <text x="${xx}" y="${rectY + 11.5}" text-anchor="middle" font-size="10" fill="#ffffff" font-weight="650">${label}</text>
    </g>`;
  }

  // Provisional forming signal diamond
  if (!grouped && r.preview) {
    const i = bars.findIndex(b => b.time === r.preview.time);
    if (i >= 0) {
      const xx = x(i), yy = y(bars[i].close);
      out += `<path d="M${xx} ${yy - 7}l7 7-7 7-7-7Z" fill="${bg}" stroke="#c28127" stroke-width="2" data-signal-time="${e(r.preview.time)}" style="cursor:pointer;"/>`;
    }
  }

  const last = bars.at(-1), lastY = y(last.close), lastColor = last.close >= last.open ? red : green;
  out += `<line x1="0" x2="${right}" y1="${lastY}" y2="${lastY}" stroke="${lastColor}" stroke-width="0.8" stroke-dasharray="2 3" opacity="0.7"/>
    <rect x="${right + 2}" y="${lastY - 8}" width="62" height="16" rx="2" fill="${lastColor}"/>
    <text x="${right + 33}" y="${lastY + 3}" text-anchor="middle" fill="#fff" font-size="11" font-weight="600">${fmt(last.close)}</text>
    <g id="crosshair" visibility="hidden"><line id="crosshairLine" x1="0" x2="0" y1="8" y2="${height - 25}" stroke="${muted}" stroke-dasharray="3 4" stroke-width="0.7"/><circle id="crosshairDot" r="3" fill="${bg}" stroke="${muted}"/></g>`;

  svg.innerHTML = out;
  state.points = bars.map((b, i) => ({ bar: b, index: start + i, x: x(i), y: y(b.close) }));
  state.chartLayout = { width, left, right };
  $("chartOHLC").textContent = `${shortTime(last.time)}  开 ${fmt(last.open)}  高 ${fmt(last.high)}  低 ${fmt(last.low)}  收 ${fmt(last.close)}`;
  $("chartBasis").textContent = state.mode === "m5" ? `${chartMinutes} 分钟图 · 信号按 5 分钟确认` : "日线图 · 前复权";
  $("chartCoverage").textContent = `${shortTime(bars[0].time)} — ${shortTime(last.time)} (${bars.length} 根)`;
  $("feedStatus").textContent = freshness().text;
}

function hover(index) {
  const point = state.points[index]; if (!point) return;
  state.hover = index;
  const b = point.bar, line = $("crosshairLine"), dot = $("crosshairDot");
  if (!line || !dot) return;
  $("crosshair").setAttribute("visibility", "visible");
  line.setAttribute("x1", point.x); line.setAttribute("x2", point.x);
  dot.setAttribute("cx", point.x); dot.setAttribute("cy", point.y);
  const orders = displaySignals().filter(o => b.first ? o.time >= b.first && o.time <= b.last : o.time === b.time || o.fill_bar_time === b.time);
  $("chartTooltip").innerHTML = `<strong>${e(shortTime(b.time))}</strong><br>开 ${fmt(b.open)} · 高 ${fmt(b.high)}<br>低 ${fmt(b.low)} · 收 ${fmt(b.close)}<br>量 ${fmt(b.volume, 0)} 手${orders.map(o => `<br><b class="${o.side === "buy" ? "up" : "down"}">${o.side === "buy" ? "▲ 买入点" : "▼ 卖出点"}</b> ${e(o.reason)}`).join("")}`;
  $("chartTooltip").hidden = false;
  $("chartTooltip").style.left = index > state.points.length / 2 ? "14px" : "auto";
  $("chartTooltip").style.right = index > state.points.length / 2 ? "auto" : "72px";
}

function table(headers, rows) {
  return `<div class="table-scroll"><table class="data-table"><thead><tr>${headers.map(h => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody></table></div>`;
}

function renderDetails() {
  const r = result(), orders = displaySignals().reverse(), trades = [...(r?.trades || [])].reverse(), detail = state.details.get(state.code);
  $("signalCount").textContent = orders.length;
  $("tradeCount").textContent = trades.length;
  $("newsCount").textContent = detail?.news?.length || 0;
  document.querySelectorAll("[data-detail]").forEach(b => b.classList.toggle("active", b.dataset.detail === state.detail));

  let output = "";
  if (state.detail === "signals") {
    const buys = orders.filter(o => o.side === "buy" && o.status !== "cancelled").length;
    const sells = orders.filter(o => o.side === "sell" && o.status !== "cancelled").length;
    output = orders.length
      ? `<div class="detail-summary">共确认买点 <b class="up">${buys}</b> 个 · 卖点 <b class="down">${sells}</b> 个（点击时间查看触发条件与 3 日验证）<button class="text-action" data-detail-action="widen">展开全部历史 K 线</button></div>
         ${table(["触发时间", "信号方向", "信号价", "触发原因", "3日方向验证", "模拟执行"], orders.map(o => {
           const bar = r.bars[o.index];
           const ev = o.liveRecord ? o.outcomes?.[3] : L.evaluate({ ...o, entry_open: r.bars[o.fill_index]?.open }, r.bars, { ...evaluationOptions(state.code, state.mode), threshold: 1 }).outcomes?.[3];
           return `<tr>
             <td><button class="link-button mono" data-signal-time="${e(o.time)}">${e(shortTime(o.time))}</button></td>
             <td><span class="pill-side ${o.side === "buy" ? "buy" : "sell"}">${o.side === "buy" ? "买入 B" : "卖出 S"}</span></td>
             <td class="mono">${bar ? fmt(bar.close) : "—"}</td>
             <td>${e(o.reason)}</td>
             <td>${outcomeHTML(ev, 1)}</td>
             <td>${o.status === "filled" ? `${e(shortTime(o.fill_time))} @ ${fmt(o.fill_price)}` : o.status === "cancelled" ? "未成交取消" : "待开盘确认"}</td>
           </tr>`;
         }))}`
      : `<div class="detail-empty"><h3>近期窗口暂无已确认的买卖点</h3><p>您可以切换上方「超卖低吸 · RSI」或点击展开全部历史 K 线查看更早信号。</p><button class="text-action" data-detail-action="widen">展开全部历史 K 线</button></div>`;
  } else if (state.detail === "trades") {
    output = trades.length
      ? `<div class="detail-summary">已平仓模拟交易 ${trades.length} 笔（已扣模型佣金 0.03%、滑点 0.1% 及卖出印花税 0.05%）</div>
         ${table(["买入时间", "买入价", "卖出时间", "卖出价", "扣费净收益"], trades.map(t => `<tr><td class="mono">${e(shortTime(t.entry_time))}</td><td class="mono">${fmt(t.entry_price)}</td><td class="mono">${e(shortTime(t.exit_time))}</td><td class="mono">${fmt(t.exit_price)}</td><td class="mono ${color(t.return_pct)}" style="font-weight:600">${pct(t.return_pct, true)}</td></tr>`))}`
      : `<div class="detail-empty"><h3>当前区间暂无完整平仓记录</h3><p>只有同时完成买入与卖出信号才计为一笔闭环交易。</p></div>`;
  } else if (state.detail === "position") {
    const p = state.positions[state.code], q = state.quotes[state.code], t = tAnalysis();
    const rows = state.tTrades.filter(x => x.code === state.code && x.day === t.day);
    output = `<div class="pos-panel-wrap">
      <div class="pos-card">
        <div>
          <strong>实际持仓记录（仅本地保存）</strong>
          <p class="muted">${p ? `${e(p.buy_date)} 买入 · 可卖底仓 ${p.available_qty || 0} 股 · ${p.buy_date < C.market().day ? '<span class="down">今日可卖</span>' : '<span class="warn">今日买入 T+1 锁定</span>'}${p.cost && q?.price ? ` · 浮动盈亏 <b class="${color(q.price / p.cost - 1)}">${pct((q.price / p.cost - 1) * 100, true)}</b>` : ""}` : "尚未录入该股持仓，录入后可提示 T+1 可卖状态与浮盈亏。"}</p>
        </div>
        <button class="seg-btn active" data-open-position>设置 / 修改持仓</button>
      </div>
      <div class="pos-card">
        <div>
          <strong>盘中做 T 参考 · ${e(t.status === "sell_watch" ? "减仓观察" : t.status === "buyback" ? "买回观察" : "暂无做 T 触发")}</strong>
          <p class="muted">${e(t.reason)} · 含成本买回参考价: <b class="mono">${fmt(t.target)}</b> · 日内均价估计: <b class="mono">${fmt(t.vwap)}</b></p>
        </div>
        <div style="display:flex;gap:8px;">
          <button class="seg-btn" data-t-action="sell" ${t.account.available ? "" : "disabled"}>登记卖出</button>
          <button class="seg-btn" data-t-action="buy" ${t.account.outstanding ? "" : "disabled"}>登记买回</button>
        </div>
      </div>
      ${rows.length ? table(["登记时间", "方向", "数量", "成交均价"], rows.toReversed().map(x => `<tr><td>${e(shortTime(x.time))}</td><td>${x.side === "sell" ? "卖出" : "买回"}</td><td>${x.qty}</td><td>${fmt(x.price)}</td></tr>`)) : ""}
    </div>`;
  } else {
    output = `${fundFlowHTML()}<div class="news-block-head"><strong>公司近期资讯与公告</strong><a href="https://www.cninfo.com.cn/new/disclosure/stock?stockCode=${state.code}" target="_blank" rel="noopener noreferrer">巨潮公告 ↗</a></div>${(detail?.news || []).slice(0, 5).map(n => `<a class="news-detail" href="${e(safeURL(n.url))}" target="_blank" rel="noopener noreferrer"><h3>${e(n.title)}</h3><small>${e(n.source)} · ${e(shortTime(n.published_at))}</small></a>`).join("") || '<div class="detail-empty"><p>暂无近期资讯</p></div>'}`;
  }
  $("detailContent").innerHTML = output;
}

function miniSVG(stock) {
  const data = dataset(stock.code, "m5"), bars = data?.bars.slice(-70) || [];
  if (bars.length < 2) return '<svg viewBox="0 0 260 95"><text x="130" y="49" text-anchor="middle" fill="#969da5" font-size="10">加载 K 线…</text></svg>';
  const min = Math.min(...bars.map(b => b.low)), max = Math.max(...bars.map(b => b.high)), span = max - min || 1;
  const x = i => 5 + i / (bars.length - 1) * 250, y = v => 75 - (v - min) / span * 56;
  let marks = "";
  for (const order of displaySignals(stock.code, "m5")) {
    const i = bars.findIndex(b => b.time === order.time);
    if (i >= 0 && order.status !== "cancelled") {
      const cl = order.side === "buy" ? "#c43737" : "#187a62";
      marks += `<circle cx="${x(i)}" cy="${y(bars[i].close)}" r="3.5" fill="${cl}"/><text x="${x(i)}" y="${y(bars[i].close) + (order.side === "buy" ? 13 : -7)}" fill="${cl}" font-size="8" font-weight="700" text-anchor="middle">${order.side === "buy" ? "B" : "S"}</text>`;
    }
  }
  return `<svg viewBox="0 0 260 95" preserveAspectRatio="none" aria-hidden="true"><path d="${svgPath(bars, "close", x, y)}" stroke="#8d9fa5" stroke-width="1.2" fill="none"/>${marks}</svg>`;
}

function renderGrid() {
  const sorted = [...state.stocks].sort((a, b) => sideLabel(b.code).priority - sideLabel(a.code).priority);
  $("multiGrid").innerHTML = sorted.map(s => {
    const q = state.quotes[s.code], sig = sideLabel(s.code), r = result(s.code, "m5");
    return `<button class="mini-chart ${sig.priority >= 4 ? "has-signal" : ""}" data-open-stock="${s.code}">
      <div class="mini-top"><span><b>${e(s.name)}</b><small>${s.code}</small></span><span class="signal-mini ${sig.className}">${sig.text}</span></div>
      <div class="mini-price"><span>${fmt(q?.price)}</span><span class="${color(q?.change_pct)}">${pct(q?.change_pct, true)}</span></div>
      ${miniSVG(s)}
      <div class="mini-foot"><span>${r?.last_confirmed ? sourceTime(r.last_confirmed).slice(0, 5) : "—"}</span><span>${r ? `闭环 ${r.metrics.closed_trades} 笔 · 胜率 ${pct(r.metrics.win_rate)}` : "—"}</span></div>
    </button>`;
  }).join("");
}

function rankingReports(stock, mode) {
  const data = dataset(stock.code, mode); if (!data?.results) return [];
  if (data.rankReports) return data.rankReports;
  const daily = dataset(stock.code, "day")?.bars || [];
  data.rankReports = L.VARIANTS.flatMap(variant => ["filtered", "original"].map(quality => {
    const full = L.analyze(data.bars, { ...costOptions(stock.code), mode, strategy: variant.strategy, params: variant.params, quality, daily });
    return { id: `${variant.id}_${quality}`, name: `${variant.name} · ${quality === "filtered" ? "过滤版" : "原始版"}`, quality, variant, report: { bars: full.bars, signals: full.signals, metrics: full.metrics } };
  }));
  return data.rankReports;
}

function pooledStats(items, horizon, options) {
  const events = items.flatMap(x => x.events), s = L.stats(events, horizon, options);
  let baseSum = 0, baseWeight = 0;
  for (const item of items) {
    const local = L.stats(item.events, horizon, options), base = L.baseline(item.report, local.rows, { ...item.options, horizon });
    if (base.accuracy != null && base.n === local.n) { baseSum += base.accuracy * local.n; baseWeight += local.n; }
  }
  return { ...s, baseline: baseWeight ? baseSum / baseWeight : null, baselineN: baseWeight };
}

let researchPreparing = false;
async function renderResearch() {
  if (researchPreparing) return;
  const mode = $("researchPeriod").value;
  const pending = state.stocks.filter(stock => dataset(stock.code, mode)?.results && !dataset(stock.code, mode).rankReports);
  if (!pending.length) { renderResearchReady(); return; }
  researchPreparing = true;
  try {
    for (let i = 0; i < pending.length; i++) {
      if (state.view !== "research" || $("researchPeriod").value !== mode) return;
      $("accuracyContent").innerHTML = `<p class="research-note">正在汇总 ${i + 1} / ${pending.length} 只股票的双策略检验结果…</p>`;
      rankingReports(pending[i], mode);
      if (i % 4 === 0) await new Promise(r => setTimeout(r, 0));
    }
    if (state.view === "research") renderResearchReady();
  } finally { researchPreparing = false; }
}

function renderResearchReady() {
  const mode = $("researchPeriod").value, horizon = Number($("accuracyHorizon").value), threshold = Number($("accuracyThreshold").value);
  const options = { side: "all", regime: "all", partition: "all", cutoff: "" }, groups = new Map();

  for (const stock of state.stocks) {
    const data = dataset(stock.code, mode); if (!data?.results) continue;
    for (const variant of rankingReports(stock, mode)) {
      const evalOptions = { ...evaluationOptions(stock.code, mode), threshold };
      const events = L.observations(variant.report, evalOptions);
      if (!groups.has(variant.id)) groups.set(variant.id, { ...variant, items: [] });
      groups.get(variant.id).items.push({ stock, report: variant.report, events, options: evalOptions });
    }
  }

  const rows = [...groups.values()].map(g => {
    const allEv = g.items.flatMap(x => x.events);
    const trades = g.items.reduce((a, x) => a + x.report.metrics.closed_trades, 0);
    const wins = g.items.reduce((a, x) => a + x.report.metrics.winning_trades, 0);
    const avgRet = g.items.length ? g.items.reduce((a, x) => a + x.report.metrics.total_return_pct, 0) / g.items.length : 0;
    return {
      ...g, stats: pooledStats(g.items, horizon, options),
      buy: L.stats(allEv, horizon, { ...options, side: "buy" }),
      sell: L.stats(allEv, horizon, { ...options, side: "sell" }),
      trades, wins, tradeWinRate: trades ? wins / trades * 100 : null, avgRet
    };
  });

  $("accuracyContent").innerHTML = table(
    ["核心策略", "买入信号正确率", "卖出避跌正确率", "闭环交易胜率", "单股均收益", "同日基准对比"],
    rows.map(r => {
      const base = r.stats.baselineN === r.stats.n ? r.stats.baseline : null;
      const diff = base != null && r.stats.accuracy != null ? r.stats.accuracy - base : null;
      return `<tr>
        <td><strong>${e(r.name)}</strong></td>
        <td class="mono">${pct(r.buy.accuracy)} <small>${r.buy.wins}/${r.buy.n}</small></td>
        <td class="mono">${pct(r.sell.accuracy)} <small>${r.sell.wins}/${r.sell.n}</small></td>
        <td class="mono">${pct(r.tradeWinRate)} <small>${r.wins}/${r.trades} 笔</small></td>
        <td class="mono ${color(r.avgRet)}">${pct(r.avgRet, true)}</td>
        <td class="mono">${base != null ? `${fmt(diff > 0 ? "+" : "")}${fmt(diff)}%` : "—"}</td>
      </tr>`;
    })
  );

  const chosen = $("rankVariant").value || "donchian-standard_filtered";
  const current = rows.find(r => r.id === chosen);
  $("rankStrategyTitle").textContent = current ? current.name : "";
  $("stockAccuracyContent").innerHTML = current ? table(
    ["股票", "买点正确率", "卖点正确率", "闭环胜率", "区间收益", "买入持有基准"],
    current.items.map(item => {
      const buy = L.stats(item.events, horizon, { ...options, side: "buy" });
      const sell = L.stats(item.events, horizon, { ...options, side: "sell" });
      const m = item.report.metrics;
      return `<tr>
        <td><button class="link-button" data-open-stock="${item.stock.code}">${e(item.stock.name)}</button> <small class="muted">${item.stock.code}</small></td>
        <td class="mono">${pct(buy.accuracy)} (${buy.wins}/${buy.n})</td>
        <td class="mono">${pct(sell.accuracy)} (${sell.wins}/${sell.n})</td>
        <td class="mono">${pct(m.win_rate)} (${m.closed_trades} 笔)</td>
        <td class="mono ${color(m.total_return_pct)}">${pct(m.total_return_pct, true)}</td>
        <td class="mono muted">${pct(m.baseline_return_pct, true)}</td>
      </tr>`;
    })
  ) : '<div class="empty">加载中…</div>';
  $("equityChart").innerHTML = equityChartHTML(result(state.code, mode));
}

function equityChartHTML(report) {
  if (!report?.curve.length) return "";
  const rows = report.curve, width = 720, height = 175, values = rows.flatMap(r => [r.equity, r.baseline]);
  const min = Math.min(1, ...values), max = Math.max(1, ...values), span = max - min || 0.01;
  const x = i => 12 + i / Math.max(1, rows.length - 1) * 696, y = v => 140 - (v - min) / span * 120;
  return `<div class="equity-title">${e(selected().name)} · ${e(C.STRATEGIES[state.strategy].name)} 资金净值曲线 <small>蓝线：策略净值 · 虚线：买入持有基准</small></div>
    <svg class="equity-chart" viewBox="0 0 ${width} ${height}"><line x1="12" x2="708" y1="${y(1)}" y2="${y(1)}" stroke="var(--line)"/>
    <path d="${svgPath(rows, "baseline", x, y)}" fill="none" stroke="var(--muted)" stroke-dasharray="4 3" stroke-width="1.2"/>
    <path d="${svgPath(rows, "equity", x, y)}" fill="none" stroke="var(--blue)" stroke-width="1.8"/>
    <text x="12" y="168" fill="var(--muted)" font-size="11">${e(shortTime(rows[0].time))}</text>
    <text x="708" y="168" text-anchor="end" fill="var(--muted)" font-size="11">${e(shortTime(rows.at(-1).time))}</text></svg>`;
}

function outcomeHTML(outcome, threshold = 1) {
  if (!outcome) return '<span class="muted">待验证</span>';
  const correct = L.exceeds(outcome.direction_return_pct, threshold);
  return `<span class="${correct ? "verified" : "muted"}">${correct ? "命中" : "未达"} (${pct(outcome.direction_return_pct, true)})</span>`;
}

function renderSignalInspector() {
  const focus = state.focusSignal, container = $("signalInspector");
  if (!focus || focus.code !== state.code || focus.mode !== state.mode || focus.strategy !== state.strategy || focus.quality !== state.quality) {
    container.hidden = true; return;
  }
  const r = result(); if (!r) { container.hidden = true; return; }
  const order = displaySignals().find(x => x.time === focus.time) || (r.preview?.time === focus.time ? r.preview : null);
  if (!order) { container.hidden = true; return; }
  const assessed = L.evaluate({ ...order, entry_open: r.bars[order.fill_index]?.open }, r.bars, { ...evaluationOptions(state.code, state.mode), threshold: 1 });
  container.hidden = false;
  container.innerHTML = `<div class="inspector-head">
    <div><span class="pill-side ${order.side === "buy" ? "buy" : "sell"}">${order.side === "buy" ? "买入信号" : "卖出信号"}</span>
    <strong>${e(order.reason)}</strong></div>
    <button id="closeSignalInspector" aria-label="关闭">×</button>
  </div>
  <div class="signal-facts">
    <div>确认时间 <strong>${e(order.time.replace("T", " "))}</strong></div>
    <div>模拟执行 <strong>${order.fill_time ? `${e(order.fill_time.replace("T", " "))} @ ${fmt(order.fill_price)}` : "等待后续开盘"}</strong></div>
    ${L.HORIZONS.map(h => `<div>${h}日后验证 <strong>${outcomeHTML(assessed.outcomes[h], 1)}</strong></div>`).join("")}
  </div>`;
}

function tAnalysis() {
  return IntradayT.analyze(dataset(state.code, "m5")?.bars || [], { position: state.positions[state.code], trades: state.tTrades.filter(t => t.code === state.code), costs: costOptions(state.code), healthy: freshness(state.code, "m5").ok && !state.paused && !document.hidden });
}

const money = v => v == null ? "—" : `${v > 0 ? "+" : ""}${fmt(v / (Math.abs(v) >= 1e8 ? 1e8 : 1e4))}${Math.abs(v) >= 1e8 ? " 亿" : " 万"}`;
async function refreshFundFlow(code = state.code, { force = false } = {}) {
  const prev = state.fundFlows.get(code);
  if (prev?.loading || (!force && prev?.attemptedAt && Date.now() - prev.attemptedAt < 60000)) return;
  const current = { ...prev, loading: true, attemptedAt: Date.now() }; state.fundFlows.set(code, current);
  try { const data = await FundFlow.load(stockBy(code)); state.fundFlows.set(code, { ...data, loading: false, snapshot: false, error: null, attemptedAt: current.attemptedAt }); }
  catch (error) {
    if (prev?.rows) state.fundFlows.set(code, { ...prev, loading: false, error: error.message, attemptedAt: current.attemptedAt });
    else { try { const saved = await getJSON(`data/fund-flow/${code}.json`); const data = FundFlow.parse(saved.payload, stockBy(code), { fetchedAt: saved.fetched_at }); state.fundFlows.set(code, { ...data, snapshot: true, loading: false, error: error.message, attemptedAt: current.attemptedAt }); } catch { state.fundFlows.set(code, { loading: false, error: error.message, attemptedAt: current.attemptedAt }); } }
  }
  if (code === state.code && state.detail === "flow_news") renderDetails();
}

function fundFlowHTML() {
  const data = state.fundFlows.get(state.code);
  if (!data?.rows) return `<div class="detail-summary">资金流向加载中或暂不可用 <button class="text-action" data-detail-action="flow">刷新资金 ↻</button></div>`;
  const latest = data.latest, totals = FundFlow.totals(data);
  return `<div class="flow-summary">
    <div><small>最新日主力净额 (${e(latest.date)})</small><strong class="${color(latest.main)}">${money(latest.main)}</strong></div>
    <div><small>5 日主力累计</small><strong class="${color(totals[5])}">${money(totals[5])}</strong></div>
    <div><small>20 日主力累计</small><strong class="${color(totals[20])}">${money(totals[20])}</strong></div>
  </div>`;
}

function setQuality(quality) {
  state.quality = quality; state.focusSignal = null;
  for (const [key, data] of state.datasets) if (data.bars?.length) compute(data, key.split("_")[0], key.endsWith("_day") ? "day" : "m5");
  syncURL(); renderRadar(); renderSelected(); renderWatchlist();
  if (state.view === "grid") renderGrid();
  if (state.view === "research") renderResearch();
}

function changeStrategy(strategy) {
  if (!validStrats.has(strategy)) return;
  state.strategy = strategy;
  if ($("gridStratSelect")) $("gridStratSelect").value = strategy;
  syncURL(); renderRadar(); renderSelected(); renderWatchlist();
  if (state.view === "grid") renderGrid();
  if (state.view === "research") renderResearch();
}

function applyView(view) {
  state.view = ["market", "grid", "research"].includes(view) ? view : "market";
  document.querySelectorAll(".view").forEach(v => v.hidden = v.id !== `${state.view}View`);
  document.querySelectorAll("[data-view]").forEach(a => a.classList.toggle("active", a.dataset.view === state.view));
  if (state.view === "market") renderSelected();
  if (state.view === "grid") renderGrid();
  if (state.view === "research") renderResearch();
}

function exportTrades() {
  const r = result(); if (!r) return;
  const headers = ["买入时间", "买入价", "卖出时间", "卖出价", "扣费净收益%", "持有K线数"];
  const rows = r.trades.map(t => [t.entry_time, t.entry_price, t.exit_time, t.exit_price, t.return_pct, t.holding_bars]);
  const cell = v => '"' + String(v ?? "").replaceAll('"', '""') + '"';
  const content = "\uFEFF" + [headers, ...rows].map(row => row.map(cell).join(",")).join("\r\n");
  const url = URL.createObjectURL(new Blob([content], { type: "text/csv;charset=utf-8" })), a = document.createElement("a");
  a.href = url; a.download = `${state.code}_${state.mode}_${state.strategy}_trades.csv`; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// Event Listeners
document.addEventListener("click", ev => {
  const stock = ev.target.closest("[data-stock]"); if (stock) selectStock(stock.dataset.stock);
  const open = ev.target.closest("[data-open-stock]"); if (open) { selectStock(open.dataset.openStock); location.hash = "market"; }
  const stratSwitch = ev.target.closest("[data-strategy-switch]"); if (stratSwitch) changeStrategy(stratSwitch.dataset.strategySwitch);
  const filterBtn = ev.target.closest("[data-filter]");
  if (filterBtn) {
    state.filter = filterBtn.dataset.filter;
    document.querySelectorAll("[data-filter]").forEach(b => b.classList.toggle("active", b === filterBtn));
    renderWatchlist();
  }
  const timeframe = ev.target.closest("[data-timeframe]");
  if (timeframe) {
    state.mode = timeframe.dataset.timeframe === "day" ? "day" : "m5";
    chartMinutes = Number(timeframe.dataset.timeframe.slice(1)) || 5;
    state.offset = 0; state.limit = 110;
    document.querySelectorAll("[data-timeframe]").forEach(b => b.classList.toggle("active", b === timeframe));
    renderSelected(); loadBars(state.code, state.mode);
  }
  const detail = ev.target.closest("[data-detail]");
  if (detail) { state.detail = detail.dataset.detail; renderDetails(); if (state.detail === "flow_news") refreshFundFlow(); }
  const sort = ev.target.closest("[data-sort]");
  if (sort) { if (state.sort === sort.dataset.sort) state.descending = !state.descending; state.sort = sort.dataset.sort; renderWatchlist(); }
  const sigTarget = ev.target.closest("[data-signal-time]");
  if (sigTarget) { state.focusSignal = { time: sigTarget.dataset.signalTime, code: state.code, mode: state.mode, strategy: state.strategy, quality: state.quality }; renderSignalInspector(); }
  if (ev.target.closest("#closeSignalInspector")) { $("signalInspector").hidden = true; state.focusSignal = null; }
  const action = ev.target.closest("[data-detail-action]")?.dataset.detailAction;
  if (action === "original" || action === "filtered") setQuality(action);
  if (action === "widen") { state.limit = Math.min(640, dataset()?.bars.length || 640); state.offset = 0; renderChart(); }
  if (action === "flow") refreshFundFlow(state.code, { force: true });
  if (ev.target.closest("[data-open-position]")) {
    const p = state.positions[state.code];
    $("positionTitle").textContent = `${selected().name} · 持仓`;
    $("buyDate").value = p?.buy_date || C.market().day; $("buyDate").max = C.market().day;
    $("costPrice").value = p?.cost || ""; $("availableQty").value = p?.available_qty || "";
    $("positionDialog").showModal();
  }
  const tAction = ev.target.closest("[data-t-action]")?.dataset.tAction;
  if (tAction) {
    const t = tAnalysis(), qty = tAction === "sell" ? t.account.available : t.account.outstanding;
    pendingTTrade = { side: tAction, code: state.code, day: t.day };
    $("tTradeTitle").textContent = `${selected().name} · 登记${tAction === "sell" ? "卖出" : "买回"}`;
    $("tTradePrice").value = state.quotes[state.code]?.price || ""; $("tTradeQty").value = Math.min(100, qty); $("tTradeQty").max = qty;
    $("tTradeError").textContent = ""; $("tTradeHint").textContent = `当前${tAction === "sell" ? "可卖底仓" : "待买回"} ${qty} 股`;
    $("tTradeDialog").showModal();
  }
});

let pendingTTrade = null;
$("closePosition").addEventListener("click", () => $("positionDialog").close());
$("clearPosition").addEventListener("click", () => { delete state.positions[state.code]; storage.set("trading.positions.v2", state.positions); $("positionDialog").close(); renderActionGuide(); renderDetails(); });
$("positionForm").addEventListener("submit", ev => {
  ev.preventDefault();
  const date = $("buyDate").value, cost = Number($("costPrice").value) || null;
  if (!date || date > C.market().day) return;
  state.positions[state.code] = { buy_date: date, cost, available_qty: Number($("availableQty").value) || 0, available_date: C.market().day };
  storage.set("trading.positions.v2", state.positions); $("positionDialog").close(); renderActionGuide(); renderDetails();
});
$("tTradeClose").addEventListener("click", () => $("tTradeDialog").close());
$("tTradeForm").addEventListener("submit", ev => {
  ev.preventDefault(); if (!pendingTTrade) return;
  const trade = { ...pendingTTrade, price: Number($("tTradePrice").value), qty: Number($("tTradeQty").value), time: C.stamp(Date.now()) };
  try {
    IntradayT.validateTrade(state.positions[trade.code], state.tTrades.filter(x => x.code === trade.code), trade, { costs: costOptions(trade.code) });
    const next = [...state.tTrades, trade]; localStorage.setItem("trading.t-trades.v1", JSON.stringify(next)); state.tTrades = next;
    $("tTradeDialog").close(); renderActionGuide(); renderDetails();
  } catch (err) { $("tTradeError").textContent = err.message; }
});

$("searchInput").addEventListener("input", ev => { state.query = ev.target.value.trim(); renderWatchlist(); });
$("gridStratSelect").addEventListener("change", ev => changeStrategy(ev.target.value));
$("researchPeriod").addEventListener("change", () => { renderResearch(); if ($("researchPeriod").value === "day") refreshDailyResearch(); });
for (const id of ["accuracyHorizon", "accuracyThreshold", "rankVariant"]) $(id).addEventListener("change", renderResearch);
$("refreshBtn").addEventListener("click", () => { if (state.paused) { state.paused = false; $("pauseBtn").textContent = "Ⅱ"; } refreshQuotes({ force: true }); loadBars(state.code, state.mode, { force: true }); scanPool(true); });
$("pauseBtn").addEventListener("click", () => { state.paused = !state.paused; $("pauseBtn").textContent = state.paused ? "▷" : "Ⅱ"; if (!state.paused) { refreshQuotes(); scanPool(); } updateConnection(); });
$("themeBtn").addEventListener("click", () => { document.body.classList.toggle("dark"); storage.set("trading.theme", document.body.classList.contains("dark") ? "dark" : "light"); renderChart(); });
$("zoomIn").addEventListener("click", () => { state.limit = Math.max(25, state.limit - 20); renderChart(); });
$("zoomOut").addEventListener("click", () => { state.limit = Math.min(640, state.limit + 20); renderChart(); });
$("resetZoom").addEventListener("click", () => { state.offset = 0; state.limit = 110; renderChart(); });
$("exportBtn").addEventListener("click", exportTrades);
$("chartWrap").addEventListener("wheel", ev => { ev.preventDefault(); state.limit = Math.max(25, Math.min(640, state.limit + (ev.deltaY > 0 ? 12 : -12))); renderChart(); }, { passive: false });
$("chartWrap").addEventListener("pointermove", ev => {
  if (!state.points.length) return;
  const rect = $("mainChart").getBoundingClientRect(), layout = state.chartLayout;
  const xx = (ev.clientX - rect.left) / rect.width * layout.width;
  hover(Math.max(0, Math.min(state.points.length - 1, Math.floor((xx - layout.left) / (layout.right - layout.left) * state.points.length))));
});
$("chartWrap").addEventListener("pointerleave", () => { $("chartTooltip").hidden = true; $("crosshair")?.setAttribute("visibility", "hidden"); });
window.addEventListener("hashchange", () => applyView(location.hash.slice(1)));
let resizeTimer; window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(renderChart, 100); });

async function initialize() {
  try {
    state.manifest = await getJSON("data/manifest.json");
    try { const benchmark = await getJSON("data/benchmark.json"); state.calendar = benchmark.bars.map(b => b.date || b.time); } catch {}
    state.stocks = state.manifest.stocks;
    if (!stockBy(state.code)) state.code = state.stocks[0].code;
    for (const s of state.stocks) if (s.quote) state.quotes[s.code] = { ...s.quote, code: s.code, as_of: s.quote.as_of.slice(0, 19) };
    if (storage.get("trading.theme", "light") === "dark") document.body.classList.add("dark");
    $("workspace").hidden = false; $("loading").hidden = true;
    applyView(location.hash.slice(1));
    renderQuoteStrip(); renderWatchlist(); loadDetail(state.code); refreshQuotes(); scanPool(); syncURL();
    setInterval(() => { const m = C.market(); $("marketSession").textContent = m.label; $("liveClock").textContent = m.time; }, 1000);
    setInterval(() => { if (!state.paused && !document.hidden && refreshWindow()) refreshQuotes(); }, 15000);
    setInterval(() => { if (!state.paused && !document.hidden && refreshWindow()) loadBars(state.code, state.mode, { force: true }); }, 30000);
    setInterval(() => { if (!state.paused && !document.hidden && refreshWindow()) scanPool(true); }, 60000);
  } catch (error) {
    $("loading").hidden = true; $("fatal").hidden = false;
    $("fatal").textContent = `无法加载行情数据：${error.message}`;
  }
}

initialize();
