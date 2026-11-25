"use strict";
const $=id=>document.getElementById(id),C=TradingCore,L=SignalLab;
const e=value=>String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt=(v,n=2)=>typeof v==="number"&&Number.isFinite(v)?v.toLocaleString("zh-CN",{minimumFractionDigits:n,maximumFractionDigits:n}):"—";
const pct=(v,signed=false)=>v==null?"—":`${signed&&v>0?"+":""}${fmt(v)}%`;
const color=v=>v==null||v===0?"muted":v>0?"up":"down";
const safeURL=v=>{try{const u=new URL(v);return ["https:","http:"].includes(u.protocol)?u.href:"#";}catch{return "#";}};
const shortTime=v=>v?v.replace("T"," ").slice(5,16):"—";
const sourceTime=v=>v?v.replace("T"," ").slice(11,19):"—";
const storage={get:(key,fallback)=>{try{return JSON.parse(localStorage.getItem(key))??fallback;}catch{return fallback;}},set:(key,value)=>{try{localStorage.setItem(key,JSON.stringify(value));}catch{}}};
let chartMinutes=5;
const params=new URLSearchParams(location.search);
const state={manifest:null,stocks:[],quotes:{},code:params.get("stock")||"300750",strategy:Object.hasOwn(C.STRATEGIES,params.get("strategy"))?params.get("strategy"):"donchian",quality:params.get("quality")==="original"?"original":"filtered",calendar:[],fundFlows:new Map(),rankCache:new Map(),focusSignal:null,mode:"m5",view:"market",detail:"signals",query:"",filter:"all",sort:"name",descending:true,limit:110,offset:0,paused:false,researchRefreshing:false,scanning:false,scanDone:0,quoteBusy:false,quoteError:null,quoteMissing:[],quoteFailures:0,quoteNextAt:0,startedAt:Date.now(),tTrades:storage.get("trading.t-trades.v1",[]),datasets:new Map(),barRequests:new Set(),details:new Map(),baselines:new Map(),journal:storage.get("trading.journal.v3",[]),positions:storage.get("trading.positions.v2",{}),points:[],hover:null,quoteCycles:0,quoteSuccess:null,beepAt:0};
const statusNames={buy_pending:"买入信号",sell_pending:"卖出信号",sell_locked:"风险预警 · T+1",holding:"模拟持有",watching:"观察",insufficient:"数据不足"};
const stockBy=code=>state.stocks.find(s=>s.code===code),selected=()=>stockBy(state.code),dataset=(code=state.code,mode=state.mode)=>state.datasets.get(`${code}_${mode}`),result=(code=state.code,mode=state.mode,strategy=state.strategy)=>dataset(code,mode)?.results?.[strategy];

async function getJSON(url){const r=await fetch(url,{cache:"no-cache"});if(!r.ok)throw Error(`读取数据失败 ${r.status}`);return r.json();}
function costOptions(code){const settings=state.manifest?.settings||{};return {code,commission:settings.commission_rate??.0003,tax:settings.sell_tax_rate??.0005,slippage:settings.slippage_rate??.001,calendar:state.calendar};}
function evaluationOptions(code,mode="m5"){return {...costOptions(code),mode,calendar:state.calendar,threshold:Number($("accuracyThreshold").value)||1};}
function compute(data,code,mode){data.bars=data.bars.filter(b=>(!b.synthetic&&!b.provisional)||C.asEpoch(b.time)>Date.now()-2000);data.results={};const daily=mode==="day"?data.bars:dataset(code,"day")?.bars||[];const now=Date.now();for(const strategy of L.STRATEGY_IDS)data.results[strategy]=L.analyze(data.bars,{now,...costOptions(code),strategy,mode,daily,quality:state.quality});data.rankReports=null;data.summaryCompare=null;return data;}

function refreshWindow(){const m=C.market(),day=new Date(m.day+"T12:00:00+08:00").getUTCDay();return m.open||(day>0&&day<6&&((m.time>"11:30:00"&&m.time<"11:35:00")||(m.time>"15:00:00"&&m.time<"15:05:00")));}
function syncURL(){const u=new URL(location.href);u.searchParams.set("stock",state.code);u.searchParams.set("strategy",state.strategy);u.searchParams.set("quality",state.quality);history.replaceState(null,"",u);}
function announce(text){$("bottomStatus").textContent=text;}

function mergeForming(data,quote){
  if(!data||!quote||!C.market().open||C.quoteState(quote).kind!=="ok")return;
  const time=quote.as_of,minute=Number(time.slice(14,16)),hour=Number(time.slice(11,13));
  if(!((hour===9&&minute>=30)||hour===10||(hour===11&&minute<30)||hour===13||hour===14))return;
  const bucketEpoch=Math.ceil((C.asEpoch(time)+1)/300000)*300000,bucket=C.stamp(bucketEpoch);
  const last=data.bars.at(-1);if(!last||last.time>bucket||bucket.slice(0,10)!==time.slice(0,10))return;
  if(last.time===bucket){last.close=quote.price;last.high=Math.max(last.high,quote.price);last.low=Math.min(last.low,quote.price);}
  else{const total=data.bars.filter(b=>b.time.slice(0,10)===time.slice(0,10)).reduce((a,b)=>a+b.volume,0);data.bars.push({time:bucket,open:quote.price,close:quote.price,high:quote.price,low:quote.price,volume:Math.max(0,(quote.volume||0)-total),synthetic:true});}
  compute(data,quote.code,"m5");
}

function freshness(code=state.code,mode=state.mode){
  const d=dataset(code,mode),q=state.quotes[code];
  if(!d)return {ok:false,text:"等待 K 线"};
  if(d.error||d.snapshot)return {ok:false,text:d.snapshot?"离线快照":"K 线刷新失败"};
  if(state.quoteMissing.includes(code))return {ok:false,text:"该股行情尚未恢复"};
  const qs=C.quoteState(q);if(qs.kind!=="ok")return {ok:false,text:qs.text};
  if(C.market().open&&mode==="m5"){
    const now=Date.now(),time=C.market().time,last=d.bars.at(-1)?.time;
    if(last&&C.asEpoch(last)<now-420000&&time>"09:36:00"&&!(time<"13:06:00"&&time>"13:00:00"))return {ok:false,text:"分钟 K 线滞后"};
  }
  return {ok:true,text:C.market().open?"盘中连接":"最近行情"};
}

function sideLabel(code,mode="m5"){
  const r=result(code,mode);if(!r)return {text:"—",className:"none"};
  if(!freshness(code,mode).ok)return {text:"快照",className:"none"};
  if(r.preview)return {text:r.preview.side==="buy"?"◇ 买":"◇ 卖",className:"forming"};
  if(r.pending)return {text:r.pending.side==="buy"?"买入":r.status==="sell_locked"?"T+1":"卖出",className:r.pending.side};
  return {text:r.status==="holding"?"持有":"—",className:"none"};
}

async function refreshQuotes({force=false}={}){
  if(state.quoteBusy||state.paused||(!force&&Date.now()<state.quoteNextAt))return;state.quoteBusy=true;
  try{
    const indices=[{code:"000001",exchange:"SH"},{code:"399001",exchange:"SZ"},{code:"000300",exchange:"SH"}];
    const quotes=await LiveFeed.quotes([...state.stocks,...indices]);
    for(const [code,q] of Object.entries(quotes))if(!state.quotes[code]||q.as_of>=state.quotes[code].as_of)state.quotes[code]=q;
    state.quoteMissing=state.stocks.filter(s=>!quotes[s.code]).map(s=>s.code);
    state.quoteSuccess=Date.now();
    if(state.quoteMissing.length){state.quoteFailures++;state.quoteError=`${state.stocks.length-state.quoteMissing.length} / ${state.stocks.length} 只更新成功，${state.quoteMissing.length} 只暂未返回`;}
    else{state.quoteFailures=0;state.quoteError=null;}
    state.quoteNextAt=state.quoteError?Date.now()+(refreshWindow()?Math.min(120000,15000*2**Math.min(state.quoteFailures,3)):300000):0;
    for(const s of state.stocks)if(!state.quoteMissing.includes(s.code))mergeForming(dataset(s.code,"m5"),state.quotes[s.code]);
    renderQuoteStrip();renderWatchlist();renderInstrument();renderSide();
    if(state.view==="market"){renderChart();renderDetails();}if(state.view==="grid")renderGrid();
    updateConnection();
  }catch(error){
    state.quoteFailures++;state.quoteMissing=state.stocks.map(s=>s.code);state.quoteError=error.message;
    state.quoteNextAt=Date.now()+(refreshWindow()?Math.min(120000,15000*2**Math.min(state.quoteFailures,3)):300000);
    updateConnection();renderWatchlist();renderSide();renderDetails();
  }finally{state.quoteBusy=false;}
}

function updateConnection(){
  const q=state.quotes[state.code],health=C.quoteState(q),paused=state.paused||document.hidden;
  $("connection").textContent=paused?"刷新已暂停":state.quoteError?"行情连接异常":health.text;
  $("connection").className=`connection ${paused?"":state.quoteError?"error":health.kind}`;
  const notice=[];
  if(state.quoteError)notice.push(`行情连接待恢复：${state.quoteError}。已保留原数据和源时间；约 ${Math.max(0,Math.ceil((state.quoteNextAt-Date.now())/1000))} 秒后自动重试，也可点击右上角刷新。`);
  if(health.kind==="stale")notice.push("上游行情滞后，暂不视为当前可执行信号。");
  $("notice").hidden=!notice.length;$("notice").textContent=notice.join(" ");
  $("quoteTime").textContent=q?sourceTime(q.as_of):"未连接";
  $("bottomStatus").textContent=paused?"自动刷新暂停":state.quoteError?`连接待恢复 · ${Math.max(0,Math.ceil((state.quoteNextAt-Date.now())/1000))} 秒后重试`:`行情 ${state.quoteSuccess?C.stamp(state.quoteSuccess).slice(11):"—"} 更新 · ${state.stocks.length} 只自选 · 信号按 5 分钟收线确认`;
}

async function loadBars(code,mode="m5",{force=false}={}){
  const key=`${code}_${mode}`,old=dataset(code,mode);if(state.barRequests.has(key))return;
  if(old&&!force&&Date.now()-old.loadedAt<25000)return;
  state.barRequests.add(key);
  if(old)old.loading=true;
  try{
    const data=await LiveFeed.bars(stockBy(code),mode);
    if(data.quote&&(!state.quotes[code]||data.quote.as_of>=state.quotes[code].as_of)){state.quotes[code]=data.quote;state.quoteMissing=state.quoteMissing.filter(x=>x!==code);if(!state.quoteMissing.length&&state.quoteError){state.quoteError=null;state.quoteFailures=0;state.quoteNextAt=0;state.quoteSuccess=Date.now();updateConnection();}}
    Object.assign(data,{loadedAt:Date.now(),snapshot:false,error:null,loading:false});
    if(mode==="m5")mergeForming(data,state.quotes[code]);compute(data,code,mode);state.datasets.set(key,data);
    if(mode==="m5")captureSignals(code,data);
    else if(dataset(code,"m5")){compute(dataset(code,"m5"),code,"m5");updateJournal(code);}
  }catch(error){
    if(old){old.error=error.message;old.loading=false;}
    else if(mode==="day"){
      try{const saved=await getJSON(`data/stocks/${code}.json`);state.datasets.set(key,compute({bars:saved.bars.map(b=>({...b,time:b.date})),loadedAt:Date.now(),snapshot:true,error:error.message,source:saved.history_source||"备用日线",basis:"前复权日线"},code,mode));}catch{}
    }
    else if(mode==="m5"){
      try{const saved=await getJSON(`data/intraday/${code}.json`);state.datasets.set(key,compute({...saved,loadedAt:Date.now(),snapshot:true,error:error.message},code,mode));}catch{state.datasets.set(key,{bars:[],results:null,error:error.message,loadedAt:Date.now(),snapshot:true});}
    }
  }
  state.barRequests.delete(key);
  if(code===state.code&&mode===state.mode)renderSelected();
  renderWatchlist();if(state.view==="grid")renderGrid();if(state.view==="research"&&!state.scanning&&!state.researchRefreshing)renderResearch();
}

async function scanPool(force=false){
  if(state.scanning||state.paused)return;state.scanning=true;state.scanDone=0;
  const ordered=[selected(),...state.stocks.filter(s=>s.code!==state.code)].filter(Boolean);let cursor=0;
  async function worker(){while(cursor<ordered.length){const s=ordered[cursor++];if(!dataset(s.code,"day")||Date.now()-dataset(s.code,"day").loadedAt>300000)await loadBars(s.code,"day",{force:true});await loadBars(s.code,"m5",{force});state.scanDone++;$("scanStatus").textContent=`5 分钟扫描 ${state.scanDone} / ${ordered.length}`;$("gridScanStatus").textContent=`已扫描 ${state.scanDone} / ${ordered.length}`;}}
  try{await Promise.all([worker(),worker(),worker(),worker()]);}
  finally{state.scanning=false;const count=state.stocks.filter(s=>dataset(s.code,"m5")?.results&&!dataset(s.code,"m5").error).length;$("scanStatus").textContent=`5 分钟 ${count} / ${state.stocks.length}`;$("gridScanStatus").textContent=`${count} / ${state.stocks.length} 已连接`;if(state.view==="research")renderResearch();}
}

function captureSignals(code,data){
  const observed=C.stamp(Date.now()),quote=state.quotes[code],fresh=freshness(code,"m5").ok;
  const reports={};
  for(const quality of ["original","filtered"]){
    reports[quality]={};
    for(const strategy of L.STRATEGY_IDS){
      const r=quality===state.quality?data.results[strategy]:L.analyze(data.bars,{
        ...costOptions(code),now:data.results[strategy].compute_epoch,strategy,quality,mode:"m5",daily:dataset(code,"day")?.bars||[]
      });
      reports[quality][strategy]=r;
      const key=`${code}_${strategy}_${quality}_${L.VERSION}`,previous=state.baselines.get(key);
      if(previous&&fresh){
        for(const order of r.signals.filter(o=>o.time>previous)){
          const id=`${key}_${order.time}_${order.side}`;
          if(!state.journal.some(x=>x.id===id)){
            state.journal.push({id,version:L.VERSION,quality,code,name:stockBy(code).name,strategy,time:order.time,side:order.side,price:r.bars[order.index].close,reason:order.reason,checks:structuredClone(order.checks),regime:order.regime,observed_at:observed,alert_outcomes:{},observed_price:quote?.as_of>=order.time&&quote.as_of<=observed&&quote.price>0?quote.price:null,quote_as_of:quote?.as_of,t1_locked:order.t1_locked,costs:{commission:costOptions(code).commission,tax:costOptions(code).tax,slippage:costOptions(code).slippage},threshold:1,outcomes:{}});
            if(quality===state.quality&&strategy===state.strategy)beep();
          }
        }
      }
      if(r.last_confirmed)state.baselines.set(key,r.last_confirmed);
    }
  }
  updateJournal(code,reports);
  renderLatestAlerts();
}
function updateJournal(code,reports={}){
  const data=dataset(code,"m5");if(!data?.bars?.length)return;
  for(const event of state.journal.filter(x=>x.code===code)){
    const report=reports[event.quality]?.[event.strategy]||(event.quality===state.quality?data.results?.[event.strategy]:null);
    if(!event.execution&&report){
      const order=report.signals.find(x=>x.time===event.time&&x.side===event.side);
      if(order?.status==="filled"){const fill=report.bars[order.fill_index];event.execution={status:"filled",entry_open:fill.open,fill_bar_time:order.fill_bar_time,fill_time:order.fill_time,fill_price:order.fill_price};}
      else if(order?.status==="cancelled")event.execution={status:"cancelled",cancel_time:order.cancel_time};
    }
    if(event.alert_outcomes&&event.execution?.status==="filled"&&!data.snapshot&&!data.error){
      event.alert_execution??=L.alertFill(event,data.bars);
      if(event.alert_execution){
        const scored=L.evaluate({...event,...event.alert_execution},data.bars,{...event.costs,calendar:state.calendar});
        for(const h of L.HORIZONS)if(scored.outcomes[h])event.alert_outcomes[h]??=scored.outcomes[h];
      }
    }
    if(event.execution){const ev=L.evaluate({...event,...event.execution},data.bars,{...event.costs,mode:"m5",calendar:state.calendar,threshold:event.threshold});for(const h of L.HORIZONS)if(!event.outcomes[h]&&ev.outcomes[h])event.outcomes[h]=ev.outcomes[h];}
  }
  storage.set("trading.journal.v3",state.journal);$("observedCount").textContent=state.journal.filter(x=>x.code===state.code&&x.strategy===state.strategy&&x.quality===state.quality).length;
}

function beep(){if(!$("soundToggle").checked||Date.now()-state.beepAt<10000)return;state.beepAt=Date.now();try{const ctx=state.audio||new AudioContext();state.audio=ctx;const oscillator=ctx.createOscillator(),gain=ctx.createGain();oscillator.connect(gain);gain.connect(ctx.destination);oscillator.frequency.value=740;gain.gain.value=.05;oscillator.start();oscillator.stop(ctx.currentTime+.14);}catch{}}

async function loadDetail(code){
  if(state.details.has(code)){refreshNews(code);return;}
  try{const detail=await getJSON(`data/stocks/${code}.json`);state.details.set(code,detail);
    if(!dataset(code,"day"))state.datasets.set(`${code}_day`,compute({bars:detail.bars.map(b=>({time:b.date,...b})),snapshot:true,loadedAt:0,source:detail.history_source||"备用日线",basis:"前复权日线",fetched_at:detail.history_fetched_at},code,"day"));
    if(code===state.code){renderSide();renderDetails();if(state.mode==="day")renderSelected();}
    refreshNews(code);
  }catch(error){announce(`新闻快照读取失败：${error.message}`);}
}

async function refreshNews(code=state.code){
  const detail=state.details.get(code);if(!detail||detail.news_loading)return;
  detail.news_loading=true;if(code===state.code&&state.detail==="news")renderDetails();
  try{detail.news=await LiveFeed.news(code);detail.news_fetched_at=C.stamp(Date.now());detail.news_status="ok";detail.news_live_error=null;}
  catch(error){detail.news_live_error=error.message;}
  finally{detail.news_loading=false;if(code===state.code){renderSide();if(state.detail==="news")renderDetails();}}
}

async function refreshDailyResearch(){
  if(state.researchRefreshing)return;state.researchRefreshing=true;
  let cursor=0;async function worker(){while(cursor<state.stocks.length){await loadBars(state.stocks[cursor++].code,"day",{force:true});}}
  try{await Promise.all([worker(),worker(),worker(),worker()]);}finally{state.researchRefreshing=false;if(state.view==="research")renderResearch();}
}

function selectStock(code){if(!stockBy(code))return;state.code=code;state.offset=0;syncURL();renderWatchlist();renderSelected();state.focusSignal=null;loadDetail(code);loadBars(code,state.mode);if(state.detail==="flow")refreshFundFlow(code);}

function renderQuoteStrip(){
  $("indexQuotes").innerHTML=[["000001","上证指数"],["399001","深证成指"],["000300","沪深 300"]].map(([code,name])=>{const q=state.quotes[code];return `<span>${name}<b class="${color(q?.change_pct)}">${fmt(q?.price)}</b><em class="${color(q?.change_pct)}">${pct(q?.change_pct,true)}</em></span>`;}).join("");
}

function renderWatchlist(){
  let stocks=state.stocks.filter(s=>(!state.query||`${s.code} ${s.name} ${s.original_code}`.includes(state.query)));
  if(state.filter!=="all")stocks=stocks.filter(s=>{const signal=sideLabel(s.code);return state.filter==="errors"?Boolean(dataset(s.code,"m5")?.error):signal.className===state.filter;});
  if(state.sort!=="name")stocks.sort((a,b)=>{const field=state.sort==="price"?"price":"change_pct";return ((state.quotes[a.code]?.[field]??-Infinity)-(state.quotes[b.code]?.[field]??-Infinity))*(state.descending?-1:1);});
  $("watchCount").textContent=stocks.length;
  const scroll=$("watchRows").scrollTop,horizontal=$("watchRows").scrollLeft;
  $("watchRows").innerHTML=stocks.length?stocks.map(s=>{const q=state.quotes[s.code],signal=sideLabel(s.code);return `<button class="watch-row ${s.code===state.code?"selected":""}" role="listitem" data-stock="${s.code}" aria-pressed="${s.code===state.code}"><span class="symbol"><b>${e(s.name)}</b><small>${s.code}</small></span><span class="price ${color(q?.change_pct)}">${fmt(q?.price)}</span><span class="change ${color(q?.change_pct)}">${pct(q?.change_pct,true)}</span><span class="signal-mini ${signal.className}">${signal.text}</span></button>`;}).join(""):'<div class="empty">没有匹配的股票</div>';
  $("watchRows").scrollTop=scroll;$("watchRows").scrollLeft=horizontal;
}

function renderInstrument(){
  const s=selected();if(!s)return;const q=state.quotes[s.code]||{};
  $("stockName").textContent=s.name;$("stockCode").textContent=`${s.exchange} ${s.code}`;$("stockBoard").textContent=s.board;
  $("lastPrice").textContent=fmt(q.price);$("lastPrice").className=color(q.change_pct);$("priceChange").textContent=`${q.change>0?"+":""}${fmt(q.change)}`;$("priceChangePct").textContent=pct(q.change_pct,true);$("priceChange").className=color(q.change_pct);$("priceChangePct").className=color(q.change_pct);
  $("instrumentTime").textContent=q.as_of?`${sourceTime(q.as_of)} 行情${q.source?.includes("K线")?" · K线接口报价":""}` :"尚未连接";
  $("quoteDetails").innerHTML=[["今开",fmt(q.open)],["昨收",fmt(q.previous_close)],["最高",fmt(q.high)],["最低",fmt(q.low)],["换手",pct(q.turnover_pct)],["成交量",q.volume==null?"—":`${fmt(q.volume/10000)} 万手`]].map(([label,value])=>`<div><span>${label}</span><b>${value}</b></div>`).join("");
}

function displaySignals(code=state.code,mode=state.mode,strategy=state.strategy){
  const report=result(code,mode,strategy);if(!report)return [];
  const signals=new Map(report.signals.map(x=>[`${x.time}_${x.side}`,x]));
  if(mode==="m5")for(const event of state.journal.filter(x=>x.code===code&&x.strategy===strategy&&x.quality===state.quality)){
    const index=report.bars.findIndex(b=>b.time===event.time);if(index<0)continue;
    const execution=event.execution||{status:"pending"},fillIndex=report.bars.findIndex(b=>b.time===execution.fill_bar_time);
    signals.set(`${event.time}_${event.side}`,{...event,...execution,index,fill_index:fillIndex,journalId:event.id,liveRecord:true});
  }
  return [...signals.values()].sort((a,b)=>a.time.localeCompare(b.time));
}
function renderLatestAlerts(){
  const groups=new Map();
  for(const event of state.journal){
    if(event.strategy!==state.strategy)continue;
    const key=[event.code,event.time,event.side,event.version].join("_");
    const pair=groups.get(key)||{};
    pair[event.quality]=event;
    groups.set(key,pair);
  }
  const entries=[...groups.values()].map(pair=>{
    const event=pair.filtered||pair.original;
    const label=pair.filtered?"过滤版确认":event.alert_outcomes?"原始版":"原始版 · 旧记录";
    return {event,label};
  }).sort((a,b)=>b.event.observed_at.localeCompare(a.event.observed_at)||b.event.time.localeCompare(a.event.time)).slice(0,8);
  const rows=entries.map(({event:x,label})=>{
    const change=x.observed_price>0&&x.price>0?(x.observed_price/x.price-1)*100:null;
    const position=state.positions[x.code];
    const holding=x.side==="sell"&&position? (position.buy_date===x.observed_at.slice(0,10)?"捕获当日 T+1 锁定":position.buy_date<x.observed_at.slice(0,10)?"捕获当日可卖":""):"";
    return `<tr><td><button class="link-button" data-open-stock="${e(x.code)}">${e(x.name)}</button></td><td>${x.side==="buy"?"B 买入":"S 卖出"}<small>${e(label)} ${e(holding)}</small></td><td>${e(shortTime(x.time))}<small>${e(shortTime(x.observed_at))}</small></td><td>${pct(change,true)}<small>${change===null?"暂无有效报价":e(sourceTime(x.quote_as_of))}</small></td><td>${e(x.reason)}</td><td>${pct(x.outcomes?.[3]?.direction_return_pct,true)} / ${pct(x.alert_outcomes?.[3]?.direction_return_pct,true)}</td></tr>`;
  });
  $("latestAlerts").innerHTML=rows.length?table(["股票","信号","确认 / 首次看到","捕获价偏离","触发原因","3 日净方向幅度 · 规则 / 提醒"],rows):'<p class="muted">当前策略暂无本机捕获记录。保持页面打开，收线后更新。</p>';
}
function renderSelected(){renderLatestAlerts();renderInstrument();renderChart();renderSide();renderDetails();renderSignalInspector();}

function renderSide(){
  renderTWatch();
  const s=selected();if(!s)return;const q=state.quotes[s.code]||{},r=result(),data=dataset(),health=freshness(),position=state.positions[s.code],last=r?.last_confirmed,bar=r?.bars.find(b=>b.time===last),index=bar?r.bars.indexOf(bar):-1,indicator=index>=0?r.indicators[index]:null;
  $("signalTime").textContent=r?.pending?shortTime(r.pending.time):last?shortTime(last):"—";$("signalValue").className="signal-value";
  let label=r?statusNames[r.status]:"等待数据",reason=r?.pending?.reason||"当前无新的买卖条件，继续观察。";
  if(r?.status==="holding")reason="策略模拟仓位持有中；不等同于你的实际持仓。";
  if(r?.status==="sell_locked")reason="退出条件已出现，但模拟仓位为当日新买入，须等待下一交易日。";
  if(r?.pending)$("signalValue").classList.add(r.pending.side==="buy"?"up":"down");
  if(position&&bar&&indicator){const [,sell]=C.conditions(bar,indicator,state.strategy);if(sell){const locked=position.buy_date>=bar.time.slice(0,10);label=locked?"风险预警 · T+1":"已有持仓退出信号";reason=locked?"已满足退出规则，但录入的持仓为当日买入，当前不能据此卖出。":"已满足退出规则，录入持仓的买入日期早于信号日。";$("signalValue").className="signal-value down";}}
  if(!health.ok){label=r?`历史 · ${label}`:"等待数据";reason=`${health.text}。${reason}`;$("signalValue").className="signal-value muted";}
  $("signalValue").textContent=label;$("signalReason").textContent=reason;
  renderActionGuide();
  $("signalChecks").innerHTML=state.quality==="filtered"&&r?.audit?r.audit.checks.map(c=>`<div class="check-row"><span class="${c.pass?"verified":"muted"}">${c.pass?"✓":"·"} ${e(c.label)}</span><small>${e(c.value)}</small></div>`).join(""):`<p class="small-muted">原始策略对照：不启用额外趋势与量价过滤。</p>`;
  $("signalLevels").innerHTML=indicator?strategyLevels(indicator).map(([title,value])=>`<div><small>${e(title)}</small><strong>${fmt(value)}</strong></div>`).join(""):"";

  const preview=r?.preview;$("provisionalSignal").hidden=!preview;$("provisionalSignal").textContent=preview?`◇ ${preview.side==="buy"?"买入":"卖出"}条件正在形成 · ${sourceTime(preview.time).slice(0,5)} 收线前可变化${preview.t1_locked?" · T+1 限制":""}`:"";
  $("positionStatus").innerHTML=position?`${e(position.buy_date)} 买入 · <span class="${position.buy_date<C.market().day?"verified":"warn"}">${position.buy_date<C.market().day?"可卖状态":"T+1 锁定"}</span>${position.cost&&q.price?`<br><span class="${color(q.price/position.cost-1)}">浮动 ${pct((q.price/position.cost-1)*100,true)}</span>`:""}`:"未录入持仓";
  const assessed=r?L.observations(r,{...evaluationOptions(s.code,state.mode),threshold:1}):[],buyStats=L.stats(assessed,3,{side:"buy"}),sellStats=L.stats(assessed,3,{side:"sell"});
  $("statsPeriod").textContent="3 交易日 · 有效幅度 > 1%";
  $("quickStats").innerHTML=[["买入正确率",pct(buyStats.accuracy)],["卖出正确率",pct(sellStats.accuracy)],["买入 正确 / 已验证",`${buyStats.wins} / ${buyStats.n}`],["卖出 正确 / 已验证",`${sellStats.wins} / ${sellStats.n}`]].map(([title,value])=>`<div><small>${title}</small><strong>${value}</strong></div>`).join("");
  $("sampleNote").textContent=`另有 ${buyStats.pending+sellStats.pending} 个待验证；${buyStats.cancelled+sellStats.cancelled} 个未成交取消。扣除模型成本；卖出统计为退出避跌效果。${buyStats.n+sellStats.n<30?"样本偏少，不能据此推断优势。":""}`;
  $("companyDetails").innerHTML=[["市盈率 TTM",q.pe_ttm>0?fmt(q.pe_ttm)+" 倍":q.pe_ttm<0?"亏损 / 不适用":"—"],["市净率",fmt(q.pb)+" 倍"],["总市值",fmt(q.market_cap_yi,1)+" 亿"],["涨停 / 跌停",`${fmt(q.upper_limit)} / ${fmt(q.lower_limit)}`]].map(([title,value])=>`<div class="company-row"><span>${title}</span><b>${value}</b></div>`).join("");
  $("companyLink").href=`https://www.cninfo.com.cn/new/disclosure/stock?stockCode=${s.code}`;
  const detail=state.details.get(s.code),news=detail?.news||[];$("newsTime").textContent=detail?.news_fetched_at?shortTime(detail.news_fetched_at)+(detail.news_live_error?" · 快照":""):"快照";
  $("newsBrief").innerHTML=news.slice(0,3).map(n=>`<a class="brief-item" href="${e(safeURL(n.url))}" target="_blank" rel="noopener noreferrer"><h3>${e(n.title)}</h3><small><span>${e(n.source)}</span><span>${n.published_at.slice(5,10)}</span></small></a>`).join("")||'<p class="small-muted">暂无相关资讯。</p>';
}

function svgPath(rows,field,x,y){let path="",continuous=false;rows.forEach((row,i)=>{const value=row?.[field];if(value==null||!Number.isFinite(value)){continuous=false;return;}path+=`${continuous?"L":"M"}${x(i).toFixed(2)},${y(value).toFixed(2)} `;continuous=true;});return path;}

function contextSignals(report){
  const selectedSignals=displaySignals();
  if(state.quality!=="filtered")return selectedSignals;
  const data=dataset();
  if(!data.originalChart||data.originalChart.base!==report){
    data.originalChart={base:report,report:L.analyze(report.bars,{...costOptions(state.code),mode:state.mode,
      strategy:state.strategy,quality:"original",daily:dataset(state.code,"day")?.bars||[]})};
  }
  const keys=new Set(selectedSignals.map(s=>s.time+s.side));
  const candidates=data.originalChart.report.signals.filter(s=>!keys.has(s.time+s.side)).map(s=>({...s,candidate:true}));
  return [...selectedSignals,...candidates].sort((a,b)=>a.time.localeCompare(b.time));
}
function renderChart(){
  if(state.view!=="market")return;
  const data=dataset(),r=result(),svg=$("mainChart");state.points=[];$("chartTooltip").hidden=true;
  if(!data?.bars.length||!r){$("chartMessage").hidden=false;$("chartMessage").textContent=data?.error?`K 线暂不可用 · ${data.error}`:"正在读取 K 线…";svg.innerHTML="";return;}
  $("chartMessage").hidden=true;
  const grouped=state.mode==="m5"&&chartMinutes>5;
  const source=grouped?C.chartBars(r.bars.filter(b=>!b.synthetic&&C.asEpoch(b.time)<=Date.now()-2000),chartMinutes):r.bars;
  if(!source.length){svg.innerHTML="";$("chartMessage").hidden=false;$("chartMessage").textContent="等待完整周期 K 线";$("chartCoverage").textContent="0 根";return;}
  const total=source.length,end=Math.max(1,total-state.offset),start=Math.max(0,end-state.limit),bars=source.slice(start,end);
  const indicators=grouped?C.indicators(source):r.indicators;
  const shorts=C.shortAverages(source);
  const inds=indicators.slice(start,end).map((item,i)=>({...item,...shorts[start+i]}));
  const visible=new Map();
  bars.forEach((bar,i)=>{for(const t of grouped?r.bars.filter(b=>b.time>=bar.first&&b.time<=bar.last).map(b=>b.time):[bar.time])visible.set(t,i);});
  const width=Math.max(340,$("chartWrap").clientWidth),height=$("chartWrap").clientHeight,left=14,right=width-60,top=18,bottom=height-92;
  svg.setAttribute("viewBox",`0 0 ${width} ${height}`);
  const values=bars.flatMap((b,i)=>[b.high,b.low,inds[i]?.ma5,inds[i]?.ma10,inds[i]?.ma20,inds[i]?.ma60,...(state.strategy==="bollinger"?[inds[i]?.bbLower,inds[i]?.bbUpper]:[])]).filter(Number.isFinite),low=Math.min(...values),high=Math.max(...values),padding=(high-low)*.1||1,min=low-padding,max=high+padding;
  const x=i=>left+(i+.5)/bars.length*(right-left),y=v=>bottom-(v-min)/(max-min)*(bottom-top),candle=Math.max(1,Math.min(9,(right-left)/bars.length*.65));
  const theme=getComputedStyle(document.body),grid=theme.getPropertyValue("--line").trim(),muted=theme.getPropertyValue("--muted").trim(),red=theme.getPropertyValue("--red").trim(),green=theme.getPropertyValue("--green").trim(),bg=theme.getPropertyValue("--bg").trim();
  let out="";
  for(let j=0;j<5;j++){const value=min+(max-min)*j/4,yy=y(value);out+=`<line x1="0" y1="${yy}" x2="${right}" y2="${yy}" stroke="${grid}" stroke-width=".7"/><text x="${right+8}" y="${yy+3}" fill="${muted}" font-size="11">${fmt(value)}</text>`;}
  const count=width<500?3:5;for(let j=0;j<count;j++){const i=Math.round((bars.length-1)*j/(count-1)),text=state.mode==="m5"?bars[i].time.slice(5,16).replace("T"," "):bars[i].time.slice(2);out+=`<text x="${x(i)}" y="${height-12}" fill="${muted}" text-anchor="${j===0?"start":j===count-1?"end":"middle"}" font-size="10">${text}</text>`;}
  if(state.strategy==="donchian")for(const field of ["upper","lower"])out+=`<path d="${svgPath(inds,field,x,y)}" fill="none" stroke="#aebbb2" stroke-width=".8" stroke-dasharray="3 3" opacity=".8"/>`;
  if(state.strategy==="bollinger")for(const field of ["bbLower","bbUpper"])out+=`<path d="${svgPath(inds,field,x,y)}" fill="none" stroke="#8996b1" stroke-dasharray="4 3" stroke-width="1"/>`;
  out+=`<path d="${svgPath(inds,"ma20",x,y)}" fill="none" stroke="#bd995c" stroke-width="1.1"/><path d="${svgPath(inds,"ma60",x,y)}" fill="none" stroke="#8b9dc0" stroke-width="1.1"/>`;
  for(const [field,color] of [["ma5","#b86c97"],["ma10","#439a9a"]])out+=`<path d="${svgPath(inds,field,x,y)}" fill="none" stroke="${color}" stroke-width="1.1"/>`;
  const markerSlots=new Map();
  const maxVolume=Math.max(1,...bars.map(b=>b.volume));
  bars.forEach((b,i)=>{const xx=x(i),cl=b.close>=b.open?red:green,forming=b.provisional||C.asEpoch(b.time)>Date.now()-2000,h=b.volume/maxVolume*42;out+=`<g opacity="${forming?.55:1}"><line x1="${xx}" x2="${xx}" y1="${y(b.high)}" y2="${y(b.low)}" stroke="${cl}" stroke-width="1"/><rect x="${xx-candle/2}" y="${Math.min(y(b.open),y(b.close))}" width="${candle}" height="${Math.max(1,Math.abs(y(b.open)-y(b.close)))}" fill="${cl}"/><rect x="${xx-candle/2}" y="${height-31-h}" width="${candle}" height="${h}" fill="${cl}" opacity=".4"/></g>`;});
  out+=`<text x="14" y="${height-76}" fill="${muted}" font-size="10">VOL</text>`;
  for(const order of contextSignals(r)){if(!visible.has(order.time))continue;const i=visible.get(order.time),b=bars[i],isBuy=order.side==="buy",xx=x(i),slotKey=i+order.side,slot=markerSlots.get(slotKey)||0,yy=y(isBuy?b.low:b.high)+(isBuy?1:-1)*(14+slot*15),cl=isBuy?red:green;markerSlots.set(slotKey,slot+1);
    if(order.candidate){
      out+=`<g data-candidate="${e(order.time)}" role="button" tabindex="0" aria-label="原始策略候选 ${e(order.time)}"><title>原始策略触发 · 点击查看</title><circle cx="${xx}" cy="${yy}" r="8" fill="${bg}" stroke="${cl}"/><text x="${xx}" y="${yy+3}" text-anchor="middle" font-size="11" fill="${cl}">${isBuy?"B":"S"}</text></g>`;
      continue;
    }
    const ev=order.liveRecord?{state:order.outcomes[3]?"evaluated":"confirmed"}:L.evaluate({...order,entry_open:r.bars[order.fill_index]?.open},r.bars,{...evaluationOptions(state.code,state.mode),threshold:1});out+=`<g data-signal-time="${e(order.time)}" ${order.journalId?`data-journal-id="${e(order.journalId)}"`:""} opacity="${order.status==="cancelled"?.45:1}" tabindex="0" role="button" aria-label="${isBuy?"买入":"卖出"} ${e(order.time)} ${ev.state==="evaluated"?"已验证":"已确认"}"><title>点击查看触发条件与 1/3/5 日结果</title><circle cx="${xx}" cy="${yy}" r="13" fill="transparent"/><path d="${isBuy?`M${xx} ${yy-4}l-4 6h8Z`:`M${xx} ${yy+4}l-4 -6h8Z`}" fill="${cl}"/><text x="${xx}" y="${yy+(isBuy?13:-7)}" text-anchor="middle" font-size="11" fill="${cl}" font-weight="600">${isBuy?"B":"S"}${order.status==="cancelled"?"×":order.t1_locked?"*":""}${ev.state==="evaluated"?"·":""}</text></g>`;
    if(visible.has(order.fill_bar_time))out+=`<circle cx="${x(visible.get(order.fill_bar_time))}" cy="${y(order.fill_price)}" r="2.6" fill="${bg}" stroke="${cl}" stroke-width="1.2"/>`;
  }
  const t=tAnalysis();if(["sell_watch","buyback"].includes(t.status)&&visible.has(t.last)){const i=visible.get(t.last),buy=t.status==="buyback",xx=x(i),yy=y(buy?bars[i].low:bars[i].high)+(buy?40:-40);out+=`<g aria-label="做 T ${buy?"买回":"减仓"}观察"><title>${e(t.reason)} · ${e(t.last)}</title><text x="${xx}" y="${yy}" fill="${buy?red:green}" text-anchor="middle" font-size="12" font-weight="650">${buy?"T回":"T卖"}</text></g>`;}
  if(!grouped&&r.preview){const i=bars.findIndex(b=>b.time===r.preview.time);if(i>=0){const xx=x(i),yy=y(bars[i].close);out+=`<path d="M${xx} ${yy-6}l6 6-6 6-6-6Z" fill="${bg}" stroke="#b4853e" stroke-width="1.4" data-signal-time="${e(r.preview.time)}" role="button" tabindex="0" aria-label="形成中信号"/>`;}}
  const last=bars.at(-1),lastY=y(last.close),lastColor=last.close>=last.open?red:green;
  out+=`<line x1="0" x2="${right}" y1="${lastY}" y2="${lastY}" stroke="${lastColor}" stroke-width=".7" stroke-dasharray="2 4" opacity=".7"/><rect x="${right+2}" y="${lastY-8}" width="55" height="16" fill="${lastColor}"/><text x="${right+29}" y="${lastY+3}" text-anchor="middle" fill="${bg}" font-size="11">${fmt(last.close)}</text><g id="crosshair" visibility="hidden"><line id="crosshairLine" x1="0" x2="0" y1="8" y2="${height-25}" stroke="${muted}" stroke-dasharray="3 4" stroke-width=".7"/><circle id="crosshairDot" r="3" fill="${bg}" stroke="${muted}"/></g>`;
  svg.innerHTML=out;svg.setAttribute("aria-label",`${selected().name} ${state.mode==="m5"?chartMinutes+" 分钟":"日线"} K线及买卖信号`);
  state.points=bars.map((b,i)=>({bar:b,index:start+i,x:x(i),y:y(b.close)}));state.chartLayout={width,left,right};
  $("chartOHLC").textContent=`${shortTime(last.time)}  开 ${fmt(last.open)}  高 ${fmt(last.high)}  低 ${fmt(last.low)}  收 ${fmt(last.close)}`;
  $("chartBasis").textContent=state.mode==="m5"?`${chartMinutes} 分钟图 · 5 分钟信号 · 不复权`:"日线图 · 日线信号 · 前复权";
  $("chartCoverage").textContent=`${shortTime(bars[0].time)} — ${shortTime(last.time)} · ${bars.length} 根`;
  $("feedStatus").textContent=freshness().text;
}

function hover(index){const point=state.points[index];if(!point)return;state.hover=index;const b=point.bar,line=$("crosshairLine"),dot=$("crosshairDot");if(!line||!dot)return;$("crosshair").setAttribute("visibility","visible");line.setAttribute("x1",point.x);line.setAttribute("x2",point.x);dot.setAttribute("cx",point.x);dot.setAttribute("cy",point.y);const orders=contextSignals(result()).filter(o=>b.first?o.time>=b.first&&o.time<=b.last:o.time===b.time||o.fill_bar_time===b.time);$("chartTooltip").innerHTML=`<strong>${e(shortTime(b.time))}</strong><br>开 ${fmt(b.open)}　高 ${fmt(b.high)}<br>低 ${fmt(b.low)}　收 ${fmt(b.close)}<br>成交量 ${fmt(b.volume,0)} 手${orders.map(o=>`<br>${o.time===b.time?(o.side==="buy"?"B 买入信号":"S 卖出信号"):(o.side==="buy"?"买入模拟成交":"卖出模拟成交")}${o.t1_locked?" · T+1 延迟":""}`).join("")}`;$("chartTooltip").hidden=false;$("chartTooltip").style.left=index>state.points.length/2?"14px":"auto";$("chartTooltip").style.right=index>state.points.length/2?"auto":"67px";}

function table(headers,rows){return `<div class="table-scroll"><table class="data-table"><thead><tr>${headers.map(h=>`<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody></table></div>`;}
function detailContext(r){const data=dataset();return `<div class="detail-context"><span>${e(selected()?.name)} · ${state.mode==="m5"?"5 分钟":"日线"} · ${state.quality==="filtered"?"趋势量价过滤":"原始对照"}</span><span>${r?.bars.length??0} 根已加载 · 图中 ${state.points.length} 根 · ${e(freshness().text)}</span></div>`;}
function currentExit(r){
  if(!r?.last_confirmed)return {text:"退出条件等待数据",active:false};
  const i=r.bars.findIndex(b=>b.time===r.last_confirmed),[,sell]=C.conditions(r.bars[i],r.indicators[i],state.strategy),p=state.positions[state.code];
  if(!sell)return {text:"当前未触发退出条件",active:false};
  return {text:p?(p.buy_date>=r.last_confirmed.slice(0,10)?"风险预警：当日新买持仓受 T+1 限制":"已有持仓出现退出条件"):"退出条件已出现：供已有可卖持仓核对",active:true};
}
function signalDiagnostic(r){
  if(!r?.last_confirmed)return `<div class="detail-empty"><h3>${dataset()?.loading?"正在读取 K 线":"等待足够的完整 K 线"}</h3><p>至少需要 61 根 K 线；当前有 ${r?.bars.length??0} 根。数据尚未就绪时不会生成买卖提示。</p><button class="text-action" data-detail-action="refresh">重新获取数据 ↻</button></div>`;
  const index=r.bars.findIndex(b=>b.time===r.last_confirmed),bar=r.bars[index],[buy]=C.conditions(bar,r.indicators[index],state.strategy),health=freshness(),checks=r.audit?.checks||[],blocked=checks.filter(c=>!c.pass),exit=currentExit(r);
  const title=!health.ok?"数据暂未就绪，先查看历史条件":r.preview?"本根 K 线条件正在形成":state.quality==="filtered"&&buy&&blocked.length?"买入触发已出现，过滤尚未通过":r.status==="holding"?"模拟仓位持有中，避免重复提示买入":"当前没有新的确认买点";
  return `<div class="signal-diagnostic"><div class="diagnostic-title"><h3>${e(title)}</h3><small>检查至 ${e(shortTime(r.last_confirmed))}</small></div><div class="diagnostic-line"><span class="${buy?"verified":"muted"}">${buy?"✓":"·"} ${e(C.STRATEGIES[state.strategy].buy)}</span><small>${strategyLevels(r.indicators[index]).map(([name,value])=>`${e(name)} ${fmt(value)}`).join(" / ")}</small></div>${state.quality==="filtered"?checks.filter(c=>c.key!=="trigger").map(c=>`<div class="diagnostic-line"><span class="${c.pass?"verified":"muted"}">${c.pass?"✓":"·"} ${e(c.label)}</span><small>${e(c.value)}</small></div>`).join(""):""}<p class="exit-assessment ${exit.active?"warn":"muted"}">${e(exit.text)}${!health.ok?" · 历史检查，当前不视为可执行提示":""}。退出条件独立检查，不要求此前出现模拟买点。</p><div class="detail-actions">${state.quality==="filtered"?'<button class="text-action" data-detail-action="original">查看原始策略对照</button>':'<button class="text-action" data-detail-action="filtered">恢复趋势量价过滤</button>'}<button class="text-action" data-detail-action="widen">展开已加载历史</button>${!state.positions[state.code]?'<button class="text-action" data-detail-action="position">录入我的持仓</button>':""}${!health.ok?'<button class="text-action" data-detail-action="refresh">重新获取数据 ↻</button>':""}</div></div>`;
}
function renderDetails(){
  const r=result(),data=dataset(),orders=displaySignals().reverse(),trades=[...(r?.trades||[])].reverse(),detail=state.details.get(state.code),events=state.journal.filter(o=>o.code===state.code&&o.strategy===state.strategy&&o.quality===state.quality).toReversed();
  $("observedCount").textContent=events.length;$("signalCount").textContent=orders.length;$("tradeCount").textContent=trades.length;$("newsCount").textContent=detail?.news?.length||0;
  document.querySelectorAll("[data-detail]").forEach(b=>b.classList.toggle("active",b.dataset.detail===state.detail));
  let output=detailContext(r);
  if(state.detail==="signals"){
    output+=orders.length?`<div class="detail-summary">已确认买点 ${orders.filter(o=>o.side==="buy").length} 个 · 卖点 ${orders.filter(o=>o.side==="sell").length} 个 · 含执行前取消的历史记录。点击时间查看原因与验证结果。<button class="text-action" data-detail-action="widen">展开历史</button></div>${table(["确认时间 · 点击查看","方向","规则","执行状态"],orders.map(o=>`<tr><td><button class="link-button" data-signal-time="${e(o.time)}" ${o.journalId?`data-journal-id="${e(o.journalId)}"`:""}>${e(shortTime(o.time))}</button></td><td class="${o.side==="buy"?"up":"down"}">${o.side==="buy"?"买入":"卖出"}</td><td>${e(o.reason)}</td><td>${o.status==="filled"?`已模拟成交<small>${e(shortTime(o.fill_time))} · ${fmt(o.fill_price)}</small>`:o.status==="cancelled"?"确认后取消 · 未成交":o.t1_locked?"T+1 / 受限等待":"等待后续 K 线执行"}</td></tr>`))}<div class="exit-strip">${e(currentExit(r).text)}${!freshness().ok?" · 历史数据":""}。</div>`:signalDiagnostic(r);
  }else if(state.detail==="trades"){
    output+=trades.length?`<div class="detail-summary">完整平仓 ${trades.length} 笔 · 已计模型佣金、滑点与卖出税费。未平仓不计胜率。</div>${table(["买入时间","买入价","卖出时间","卖出价","扣费收益"],trades.map(t=>`<tr><td>${e(shortTime(t.entry_time))}</td><td>${fmt(t.entry_price)}</td><td>${e(shortTime(t.exit_time))}</td><td>${fmt(t.exit_price)}</td><td class="${color(t.return_pct)}">${pct(t.return_pct,true)}</td></tr>`))}`:`<div class="detail-empty"><h3>${r?.entry?"模拟仓位尚未平仓":"尚无完整的模拟买卖交易"}</h3><p>${r?.entry?`模拟买入 ${e(shortTime(r.entry.time))} · ${fmt(r.entry.price)}。${r.status==="sell_locked"?"退出条件已出现，正在等待 T+1 可卖日期。":"等待退出规则及可执行的后续 K 线。"}`:"只有买入和卖出都模拟执行后，才会出现一笔完整交易。没有交易也可能是过滤未通过或历史窗口不足。"}</p><p>这里是策略模拟账户，实际持仓可在右侧录入；两者分别显示。</p><button class="text-action" data-detail-action="signals">查看信号与条件</button></div>`;
  }else if(state.detail==="flow"){
    output+=fundFlowHTML();
  }else if(state.detail==="news"){
    output+=`<div class="detail-summary">${detail?.news_live_error?`资讯更新暂不可用：${e(detail.news_live_error)}。以下保留已有资讯。`:detail?.news_loading?"正在获取相关资讯…":"相关标题按发布时间排列，公告以公司原文为准。"}${detail?.news_fetched_at?`<small>上次获取 ${e(shortTime(detail.news_fetched_at))}</small>`:""}<button class="text-action" data-detail-action="news">重新获取资讯 ↻</button></div>`;
    output+=(detail?.news||[]).map(n=>`<a class="news-detail" href="${e(safeURL(n.url))}" target="_blank" rel="noopener noreferrer"><h3>${e(n.title)}</h3><small>${e(n.source)} · ${e(shortTime(n.published_at))}</small></a>`).join("")||`<div class="detail-empty"><h3>${detail?.news_loading?"正在读取资讯":"暂未取得相关报道"}</h3><p>这不代表公司没有新闻或公告。可以重试，或打开公司披露页面。</p><a class="text-action" href="https://www.cninfo.com.cn/new/disclosure/stock?stockCode=${state.code}" target="_blank" rel="noopener noreferrer">查看公司公告 ↗</a></div>`;
  }else{
    output+=events.length?`<div class="detail-summary">当前股票与规则捕获 ${events.length} 个 · 此浏览器全部 ${state.journal.length} 个。原始原因保留，后续结果追加。<button class="text-action" data-detail-action="export-observed">导出观察记录 ↓</button></div>${table(["确认 / 捕获时间","方向","1 交易日","3 交易日","5 交易日"],events.map(o=>`<tr><td><button class="link-button" data-journal-id="${e(o.id)}">${e(shortTime(o.time))}</button><small>捕获 ${e(shortTime(o.observed_at))}</small></td><td>${o.side==="buy"?"买入":"卖出"}</td>${L.HORIZONS.map(h=>`<td>${outcomeHTML(o.outcomes[h],o.threshold)}</td>`).join("")}</tr>`))}`:`<div class="detail-empty"><h3>正在等待新的确认信号</h3><p>本次打开时间 ${e(shortTime(C.stamp(state.startedAt)))}。${r?.last_confirmed?`当前观察基线 ${e(shortTime(r.last_confirmed))}。`:"正在建立观察基线。"}</p><p>首次加载的 ${orders.length} 个历史信号不会记作实时捕获；开始观察后新确认的信号才会保存。只统计当前股票、策略和过滤方式。${state.journal.length?`此浏览器其他筛选下有 ${state.journal.length} 个记录。`:""}</p><p>保持网页在前台并连接行情。页面关闭、隐藏或暂停后，不持续后台扫描。</p><button class="text-action" data-detail-action="signals">查看历史信号与条件</button></div>`;
  }
  $("detailContent").innerHTML=output;
}

function miniSVG(stock){
  const data=dataset(stock.code,"m5"),r=result(stock.code,"m5"),bars=data?.bars.slice(-70)||[];if(bars.length<2)return '<svg viewBox="0 0 260 95"><text x="130" y="49" text-anchor="middle" fill="#969da5" font-size="10">等待分钟行情</text></svg>';
  const min=Math.min(...bars.map(b=>b.low)),max=Math.max(...bars.map(b=>b.high)),span=max-min||1,x=i=>5+i/(bars.length-1)*250,y=v=>75-(v-min)/span*56;
  let marks="";for(const order of displaySignals(stock.code,"m5")){const i=bars.findIndex(b=>b.time===order.time);if(i>=0&&order.status!=="cancelled"){const cl=order.side==="buy"?"#cc7270":"#489580";marks+=`<circle cx="${x(i)}" cy="${y(bars[i].close)}" r="3" fill="${cl}"/><text x="${x(i)}" y="${y(bars[i].close)+(order.side==="buy"?12:-8)}" fill="${cl}" font-size="7" text-anchor="middle">${order.side==="buy"?"B":"S"}</text>`;}}
  return `<svg viewBox="0 0 260 95" preserveAspectRatio="none" aria-hidden="true"><path d="${svgPath(bars,"close",x,y)}" stroke="#8d9fa5" stroke-width="1.1" fill="none"/>${marks}</svg>`;
}

function renderGrid(){
  $("multiGrid").innerHTML=state.stocks.map(s=>{const q=state.quotes[s.code],signal=sideLabel(s.code),r=result(s.code,"m5");return `<button class="mini-chart" data-open-stock="${s.code}"><div class="mini-top"><span><b>${e(s.name)}</b><small>${s.code}</small></span><span class="signal-mini ${signal.className}">${signal.text}</span></div><div class="mini-price"><span>${fmt(q?.price)}</span><span class="${color(q?.change_pct)}">${pct(q?.change_pct,true)}</span></div>${miniSVG(s)}<div class="mini-foot"><span>5 分钟 · ${r?.last_confirmed?sourceTime(r.last_confirmed).slice(0,5):"等待数据"}</span><span>${r?`${r.metrics.closed_trades} 笔 / 胜率 ${pct(r.metrics.win_rate)}`:"—"}</span></div></button>`;}).join("");
}

function renderTradeResearch(){
  const mode=$("researchPeriod").value;
  const reports=state.stocks.map(s=>({stock:s,metrics:mode==="m5"?result(s.code,"m5")?.metrics:result(s.code,"day")?.metrics}));
  const included=reports.filter(x=>x.metrics&&(mode==="day"||freshness(x.stock.code,"m5").ok)),n=included.reduce((sum,x)=>sum+x.metrics.closed_trades,0),wins=included.reduce((sum,x)=>sum+x.metrics.winning_trades,0);
  const ci=C.wilson(wins,n);
  $("equityChart").innerHTML=equityChartHTML(result(state.code,mode));
  $("tradeResearchContent").innerHTML=`<div class="research-strategies">${L.STRATEGY_IDS.map(id=>`<button data-research-strategy="${id}" class="${state.strategy===id?"active":""}">${e(C.STRATEGIES[id].name)}</button>`).join("")}</div><div class="research-summary"><div>有效股票<strong>${included.length} / ${state.stocks.length}</strong></div><div>已平仓交易<strong>${n} 笔</strong></div><div>已平仓胜率<strong>${pct(n?wins/n*100:null)}</strong></div><div>95% 描述区间<strong>${ci?`${fmt(ci[0],1)}–${fmt(ci[1],1)}%`:"—"}</strong></div></div><p class="research-note">${mode==="m5"?"5 分钟近期历史，按 A 股 T+1 限制模拟；分钟数据窗口较短，样本量有限。":"日线较长历史，打开视图后重新获取，以表中实际区间为准。"}已扣单边佣金 0.03%、滑点 0.1%、卖出税费 0.05%。每股独立账户，未平仓不计胜率；汇总不是组合收益。不同策略与周期需分别评价。新增三类策略为实验规则，不保证提高胜率。</p>${table(["股票","区间收益","买入持有","最大回撤","胜率 / 笔数","数据区间"],reports.map(({stock:s,metrics:m})=>`<tr><td><button class="link-button" data-open-stock="${s.code}">${e(s.name)}</button><small>${s.code}</small></td><td class="${color(m?.total_return_pct)}">${pct(m?.total_return_pct,true)}</td><td>${pct(m?.baseline_return_pct,true)}</td><td>${pct(m?.max_drawdown_pct)}</td><td>${pct(m?.win_rate)} / ${m?.closed_trades??0} 笔</td><td>${m?.start?`${e(shortTime(m.start))}<small>至 ${e(shortTime(m.end))}</small>`:"等待数据"}</td></tr>`))}`;
}

function renderAbout(){
  const corrections=state.stocks.filter(s=>s.original_code!==s.code);
  $("aboutContent").innerHTML=`<h2>信号怎样产生，怎样验证</h2><h3>现在有几种策略？状态是什么意思？</h3><p>当前有五类固定规则：唐奇安突破、均线趋势、MACD 动量、RSI 超卖修复、布林带回归。MACD/RSI/布林带为新增实验策略，需积累样本。“原始/过滤”是筛选模式，是同一策略是否附加趋势量价门槛。排行榜的不同参数也是对照版本，不是更多独立预测模型。</p><p>“5 分钟趋势信号状态”按已收线 5 分钟判断：观察=没有新的买入或退出条件；模拟持有=策略回测账户有仓位，不是你的真实持仓；买入/卖出信号=条件确认、等待后续可执行价格；风险预警 T+1=出现退出条件但当日新买不能按模型卖出。切换 15/30/60/120 分钟图只改变图表显示，趋势信号仍为 5 分钟；日线图另用日线规则。</p><p>做 T 是另一个独立观察模块：仅按已有可卖底仓，检查上冲后转弱、登记卖出后的价差与 MA5 止跌条件。先在持仓里确认今日初始可卖数量，真实成交后手动登记。不自动下单，不承诺获利，不混入趋势胜率。</p><h3>策略目录</h3>${strategyCatalogHTML()}<h3>三层买入规则</h3><p>默认保留 5 分钟触发，以前一交易日已完成的日线判断趋势：MA20 高于 MA60 且收盘高于 MA20。买入触发使用唐奇安前 20 根最高价突破或 MA20/MA60 趋势，同时要求当前周期快均线高于慢均线、量比至少 1.2、波动不极端。分钟量比使用前 3–5 个交易日相同时刻成交量的均值；同刻历史不足不确认。RSI/布林带修复规则的过滤模式不强制快均线高于慢均线，改为避开明确日线下跌，并要求量比至少 0.8；这样不会把趋势突破的门槛机械套用到均值回归。波动为前 20 根真实振幅均值除以现价，分钟需在 0.08%–3%、日线在 0.3%–12%，当前振幅不超过 3 倍前期 ATR。完整阈值在信号详情公开。这里是固定规则，没有训练预测模型。</p><h3>确认与卖出</h3><p>◇ 为未收线形成中，可能消失；B / S 为收线后确认，× 表示执行前取消；实时捕获的标记会叠加保留，不因滚动窗口重新计算而覆盖。小点后缀表示 3 个交易日标签已到期，不代表预测正确，点击可查看是否达到阈值。唐奇安跌破前 10 根低点或均线趋势转弱提示退出；卖出不受买入放量和日线过滤阻挡。确认至少等待收线后 2 秒。回测使用后续完整 K 线的开盘价模拟成交，无法保证实际能按该价执行。</p><h3>1 / 3 / 5 交易日评价</h3><p>从模拟执行日之后第 1、3、5 个交易日的收盘评价，默认有效幅度严格大于 1%。买入按目标收盘卖出的往返成本后收益判定；卖出按模拟卖出净得价款与后续收盘价之差评价避跌效果，计退出费用，不计重新买入。两者分别统计，卖出方向正确不等同于改善整个持仓账户的收益。单边滑点 0.1%、佣金 0.03%、卖出税费 0.05% 为模型假设。未成交、取消、未到期和缺少目标收盘不当成成功或失败。分钟评价只用不复权分钟线，目标日需要 15:00 的完整收盘线，绝不拼接前复权日线；共同交易日历来自基准和已加载行情。</p><h3>准确率排行榜与历史留出</h3><p>比较唐奇安 20/10、40/20，均线 20/60、10/30，以及 MACD 12/26/9、RSI14、布林带 20/2，各有原始及过滤版本，共 14 组。按固定规则计算，没有自动挑选最优参数。买卖分别展示正确数、成熟样本数、平均净方向幅度与 Wilson 95% 描述区间；重叠信号存在相关性，区间不是收益保证。30 个成熟样本以下不编号。默认分钟历史留出自 2026-11-04、日线自 2026-06-01 起，可自行改日期。按执行日切分，并剔除早期样本中跨越边界的标签。这是回顾性留出，不宣称独立前瞻验证；反复看结果调参会损害留出意义。可按上涨、震荡、下跌环境查看，环境在信号时由过去日线决定。</p><h3>基准</h3><p>信号基准取同股票、同执行日、同方向的可用开盘时点，不依赖买卖触发规则；逐信号匹配、等权汇总，同样扣模型成本和近似涨跌停限制。它是固定时点对照，不是随机抽样，也不是可直接交易的组合。完整交易回测另列区间收益、最大回撤与买入持有基准。准确率高仍可能输给市场普涨。</p><h3>T+1 与实际持仓</h3><p>模拟当日买入的股票不能当日卖出，退出执行延迟到后续交易日。实际持仓可录入买入日期和成本，当日新买遇到卖出条件只显示风险预警；早于信号日买入才显示已有持仓退出提示。不连接券商，不自动下单；未录入时信号只供观察。</p><h3>实时捕获与数据</h3><p>首次加载只建立基线。随后新信号保存当时的规则版本、过滤模式、原因、条件数值、确认时间与捕获时间；原始字段不会被后续行情覆盖，执行和各周期结果首次得到后追加并保留。历史重算与实时捕获明确分开。记录仅在此浏览器，可被本机清除或修改，不是服务端不可篡改账本；跨设备不共享。旧版 5/20 根记录保留在旧存储中，不混入新口径。</p><h3>更新与局限</h3><p>前台打开时即时行情每 15 秒、当前 K 线每 30 秒、38 股每 60 秒检查，日线趋势和新闻约每 5 分钟刷新。隐藏或关闭后不后台扫描。公开行情可能延迟或失效，备用快照会标记来源。分钟历史约 640 根，通常只有约 13 个交易日；日线更长，但不能替代分钟长期检验。未精确重建历史 ST、除权除息、整手、最低佣金和盘口排队。增加过滤不保证提高准确率，不保证信号可成交或盈利。</p><h3>方法依据</h3><p><a href="https://www.sse.com.cn/lawandrules/sselawsrules2025/trade/universal/c/c_20260424_10816492.shtml" target="_blank" rel="noopener noreferrer">上交所交易规则</a> · <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html" target="_blank" rel="noopener noreferrer">时间顺序验证与间隔</a></p><h3>股票代码核对</h3><p>原始 TXT 保留，以股票名称为准修正 ${corrections.length} 处代码。</p>${table(["名称","原代码","采用代码","依据"],corrections.map(s=>`<tr><td>${e(s.name)}</td><td>${s.original_code}</td><td>${s.code}</td><td><a href="${e(safeURL(s.correction_source))}" target="_blank" rel="noopener noreferrer">公司披露 ↗</a></td></tr>`))}`;
}

function applyView(view){state.view=["market","grid","research","about"].includes(view)?view:"market";document.querySelectorAll(".view").forEach(v=>v.hidden=v.id!==`${state.view}View`);document.querySelectorAll("[data-view]").forEach(a=>{a.classList.toggle("active",a.dataset.view===state.view);if(a.dataset.view===state.view)a.setAttribute("aria-current","page");else a.removeAttribute("aria-current");});if(state.view==="market")renderSelected();if(state.view==="grid")renderGrid();if(state.view==="research")renderResearch();if(state.view==="about")renderAbout();}

function exportTrades(){const r=result();if(!r)return;const headers=["买入时间","买入价含滑点","卖出时间","卖出价含滑点","扣费后收益%","持有K线数"],rows=r.trades.map(t=>[t.entry_time,t.entry_price,t.exit_time,t.exit_price,t.return_pct,t.holding_bars]),cell=v=>'"'+String(v??"").replaceAll('"','""')+'"',content="\uFEFF"+[headers,...rows].map(row=>row.map(cell).join(",")).join("\r\n"),url=URL.createObjectURL(new Blob([content],{type:"text/csv;charset=utf-8"})),a=document.createElement("a");a.href=url;a.download=`${state.code}_${state.mode}_${state.strategy}_trades.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}

function zoom(delta){state.limit=Math.max(25,Math.min(640,state.limit+delta));renderChart();}
function changeStrategy(strategy){if(!Object.hasOwn(C.STRATEGIES,strategy))return;state.strategy=strategy;$("strategySelect").value=strategy;syncURL();renderSelected();renderWatchlist();if(state.view==="grid")renderGrid();if(state.view==="research")renderResearch();}

document.addEventListener("change",event=>{if(event.target.matches("[data-strategy-picker]"))changeStrategy(event.target.value);});
document.addEventListener("click",event=>{
  const stock=event.target.closest("[data-stock]");if(stock)selectStock(stock.dataset.stock);
  const open=event.target.closest("[data-open-stock]");if(open){selectStock(open.dataset.openStock);location.hash="market";}
  const timeframe=event.target.closest("[data-timeframe]");if(timeframe){state.mode=timeframe.dataset.timeframe==="day"?"day":"m5";chartMinutes=Number(timeframe.dataset.timeframe.slice(1))||5;state.offset=0;state.limit=110;document.querySelectorAll("[data-timeframe]").forEach(b=>b.classList.toggle("active",b===timeframe));renderSelected();loadBars(state.code,state.mode);}
  const detail=event.target.closest("[data-detail]");if(detail){state.detail=detail.dataset.detail;renderDetails();if(state.detail==="flow")refreshFundFlow();}
  const sort=event.target.closest("[data-sort]");if(sort){if(state.sort===sort.dataset.sort)state.descending=!state.descending;state.sort=sort.dataset.sort;renderWatchlist();}
  const strategy=event.target.closest("[data-research-strategy]");if(strategy)changeStrategy(strategy.dataset.researchStrategy);
});
$("closeStrategySettings").addEventListener("click",()=>{document.querySelector(".strategy-settings").open=false;});
document.addEventListener("keydown",ev=>{if(ev.key==="Escape")document.querySelector(".strategy-settings").open=false;});
document.addEventListener("click",ev=>{if(!ev.target.closest(".strategy-settings"))document.querySelector(".strategy-settings").open=false;});
$("strategyHelpBtn").addEventListener("click",()=>{location.hash="about";});
$("strategySelect").addEventListener("change",ev=>changeStrategy(ev.target.value));
$("researchPeriod").addEventListener("change",()=>{renderResearch();if($("researchPeriod").value==="day")refreshDailyResearch();});
$("searchInput").addEventListener("input",ev=>{state.query=ev.target.value.trim();renderWatchlist();});
$("watchFilter").addEventListener("change",ev=>{state.filter=ev.target.value;renderWatchlist();});
$("refreshBtn").addEventListener("click",()=>{if(state.paused){state.paused=false;$("pauseBtn").textContent="Ⅱ";}refreshQuotes({force:true});loadBars(state.code,state.mode,{force:true});scanPool(true);});
$("pauseBtn").addEventListener("click",()=>{state.paused=!state.paused;$("pauseBtn").textContent=state.paused?"▷":"Ⅱ";$("pauseBtn").setAttribute("aria-label",state.paused?"恢复刷新":"暂停自动刷新");if(!state.paused){refreshQuotes();scanPool();}updateConnection();renderSide();});
$("themeBtn").addEventListener("click",()=>{document.body.classList.toggle("dark");storage.set("trading.theme",document.body.classList.contains("dark")?"dark":"light");renderChart();});
$("zoomIn").addEventListener("click",()=>zoom(-20));$("zoomOut").addEventListener("click",()=>zoom(20));$("resetZoom").addEventListener("click",()=>{state.offset=0;state.limit=110;renderChart();});$("exportBtn").addEventListener("click",exportTrades);
$("chartWrap").addEventListener("wheel",ev=>{ev.preventDefault();zoom(ev.deltaY>0?10:-10);},{passive:false});
$("chartWrap").addEventListener("pointermove",ev=>{if(!state.points.length)return;const rect=$("mainChart").getBoundingClientRect(),layout=state.chartLayout,xx=(ev.clientX-rect.left)/rect.width*layout.width,i=Math.max(0,Math.min(state.points.length-1,Math.floor((xx-layout.left)/(layout.right-layout.left)*state.points.length)));hover(i);});
$("chartWrap").addEventListener("pointerleave",()=>{$("chartTooltip").hidden=true;$("crosshair")?.setAttribute("visibility","hidden");});
$("chartWrap").addEventListener("keydown",ev=>{if(!["ArrowLeft","ArrowRight","Home","End","Escape"].includes(ev.key))return;ev.preventDefault();if(ev.key==="Escape"){$("chartTooltip").hidden=true;return;}const last=state.points.length-1,index=state.hover??last;hover(ev.key==="Home"?0:ev.key==="End"?last:Math.max(0,Math.min(last,index+(ev.key==="ArrowLeft"?-1:1))));});
$("soundToggle").addEventListener("change",()=>{if($("soundToggle").checked){try{state.audio=state.audio||new AudioContext();state.audio.resume();}catch{}}});
$("positionBtn").addEventListener("click",()=>{const p=state.positions[state.code];$("positionTitle").textContent=`${selected().name} · 持仓`;$("buyDate").value=p?.buy_date||C.market().day;$("buyDate").max=C.market().day;$("costPrice").value=p?.cost||"";$("availableQty").value=p?.available_qty||"";$("positionDialog").showModal();});
$("closePosition").addEventListener("click",()=>$("positionDialog").close());
$("clearPosition").addEventListener("click",()=>{delete state.positions[state.code];storage.set("trading.positions.v2",state.positions);$("positionDialog").close();renderSide();});
$("positionForm").addEventListener("submit",ev=>{ev.preventDefault();const date=$("buyDate").value,cost=Number($("costPrice").value)||null;if(!date||date>C.market().day)return;state.positions[state.code]={buy_date:date,cost,available_qty:Number($("availableQty").value)||0,available_date:C.market().day};storage.set("trading.positions.v2",state.positions);$("positionDialog").close();renderSide();});
window.addEventListener("hashchange",()=>{applyView(location.hash.slice(1));if(innerWidth<600)window.scrollTo(0,0);});
let resizeTimer;window.addEventListener("resize",()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(renderChart,100);});
document.addEventListener("visibilitychange",()=>{if(!document.hidden&&!state.paused){refreshQuotes();loadBars(state.code,state.mode,{force:true});scanPool();}updateConnection();renderSide();});

async function initialize(){
  try{
    state.manifest=await getJSON("data/manifest.json");
    try{const benchmark=await getJSON("data/benchmark.json");state.calendar=benchmark.bars.map(b=>b.date||b.time);}catch{}state.stocks=state.manifest.stocks;if(!stockBy(state.code))state.code=state.stocks[0].code;
    for(const s of state.stocks)if(s.quote)state.quotes[s.code]={...s.quote,code:s.code,as_of:s.quote.as_of.slice(0,19)};
    if(storage.get("trading.theme","light")==="dark")document.body.classList.add("dark");
    $("workspace").hidden=false;$("loading").hidden=true;$("strategySelect").value=state.strategy;$("qualitySelect").value=state.quality;$("githubLink").href=`https://github.com/${state.manifest.settings.github_owner}/${state.manifest.settings.repository}`;
    applyView(location.hash.slice(1));renderWatchlist();renderQuoteStrip();loadDetail(state.code);refreshQuotes();scanPool();syncURL();
    setInterval(()=>{const m=C.market();$("marketSession").textContent=m.label;$("liveClock").textContent=m.time;if(state.quoteError)updateConnection();},1000);
    setInterval(()=>{if(state.paused||document.hidden)return;state.quoteCycles++;if(refreshWindow()||state.quoteCycles%20===0||(state.quoteError&&Date.now()>=state.quoteNextAt))refreshQuotes();if(!refreshWindow()&&state.quoteCycles%20===0){scanPool(true);if(state.mode==="day")loadBars(state.code,state.mode,{force:true});}},15000);
    setInterval(()=>{if(!state.paused&&!document.hidden&&refreshWindow())loadBars(state.code,state.mode,{force:true});},30000);
    setInterval(()=>{if(!state.paused&&!document.hidden&&refreshWindow())scanPool(true);},60000);
    setInterval(()=>{if(!state.paused&&!document.hidden)refreshNews();},300000);
    setInterval(()=>{if(!state.paused&&!document.hidden&&state.detail==="flow"&&(refreshWindow()||Date.now()-(state.fundFlows.get(state.code)?.attemptedAt||0)>=300000))refreshFundFlow();},60000);
  }catch(error){$("loading").hidden=true;$("fatal").hidden=false;$("fatal").textContent=`无法读取自选股：${error.message}\n请通过 python3 scripts/preview.py 启动本地预览。`;}
}
function outcomeHTML(outcome,threshold=1){
  if(!outcome)return '<span class="muted">待验证</span>';
  const correct=L.exceeds(outcome.direction_return_pct,threshold);
  return `<span class="${correct?"verified":"muted"}">${correct?"正确":"未达阈值"}</span><small>${pct(outcome.direction_return_pct,true)} · ${e(outcome.target_day)}</small>`;
}
function renderSignalInspector(){
  const focus=state.focusSignal,container=$("signalInspector");if(!focus||focus.code!==state.code||focus.mode!==state.mode||focus.strategy!==state.strategy||focus.quality!==state.quality){container.hidden=true;return;}
  const r=result();if(!r){container.hidden=true;return;}
  const recorded=focus.journalId?state.journal.find(x=>x.id===focus.journalId):null;
  let order=recorded?{...recorded,...recorded.execution}:displaySignals().find(x=>x.time===focus.time);
  if(!order&&r.preview?.time===focus.time)order=r.preview;
  if(!order){container.hidden=true;return;}
  const assessed=recorded?{state:recorded.outcomes[3]?"evaluated":"confirmed",outcomes:recorded.outcomes}:L.evaluate({...order,entry_open:r.bars[order.fill_index]?.open},r.bars,{...evaluationOptions(state.code,state.mode),threshold:1});
  const forming=order.status==="forming",phase=forming?"形成中":order.status==="cancelled"?"已确认 · 未成交取消":assessed.state==="evaluated"?"已验证 · 查看各周期结果":"已确认 · 等待执行或到期";
  const held=state.positions[state.code],locked=order.side==="sell"&&held&&held.buy_date>=order.time.slice(0,10);
  const checks=order.checks||[];
  container.hidden=false;
  container.innerHTML=`<div class="inspector-head"><div><span class="badge">${e(phase)}</span><h3>${order.side==="buy"?"买入信号":locked?"风险预警 · 当日持仓不可卖":"卖出信号 · 供已有可卖持仓"}</h3></div><button id="closeSignalInspector" aria-label="关闭信号详情">×</button></div><p>${e(order.reason)}${state.quality==="original"?"。原始对照：下列过滤检查仅供观察，不参与信号确认。":""}${order.side==="sell"?"。退出提示不受买入量价过滤阻挡。":""}</p><div class="signal-facts"><div>确认时间<strong>${forming?"尚未收线":e(order.time.replace("T"," "))}</strong></div><div>观察来源<strong>${recorded?`实时捕获 ${e(recorded.observed_at.replace("T"," "))}`:"历史计算 · 非实时捕获"}</strong></div><div>模拟执行<strong>${order.fill_time?`${e(order.fill_time.replace("T"," "))} · ${fmt(order.fill_price)}`:order.status==="cancelled"?"执行前条件失效":"等待后续完整 K 线"}</strong></div></div>${checks.length?`<div class="inspector-checks">${checks.map(c=>`<div><span class="${c.pass?"verified":"muted"}">${c.pass?"✓":"·"} ${e(c.label)}</span><small>${e(c.value)}</small></div>`).join("")}</div>`:""}<div class="outcome-grid">${L.HORIZONS.map(h=>`<div><span>${h} 个交易日后</span><strong>${forming?"待确认":outcomeHTML(assessed.outcomes[h],recorded?.threshold??1)}</strong></div>`).join("")}</div><p class="small-muted">有效幅度严格大于 ${recorded?.threshold??1}%。从后续可执行 K 线开盘模拟；买入计往返费用，卖出计退出费用后的避跌幅度，卖出正确率不等于账户盈利。${state.mode==="m5"?"目标日须有 15:00 分钟收盘线；缺失则等待，不拼接复权日线。":"日线在同一复权序列评价。"}</p>`;
}
function rankingReports(stock,mode){
  const data=dataset(stock.code,mode);if(!data?.results)return [];
  if(data.rankReports)return data.rankReports;
  const daily=dataset(stock.code,"day")?.bars||[];
  data.rankReports=L.VARIANTS.flatMap(variant=>["filtered","original"].map(quality=>{
    const full=L.analyze(data.bars,{...costOptions(stock.code),mode,strategy:variant.strategy,params:variant.params,quality,daily});
    // Rankings only need bars, signals and metrics; release per-variant curves and indicators.
    const report={bars:full.bars,signals:full.signals,metrics:full.metrics};
    return {id:`${variant.id}_${quality}`,name:`${variant.name} · ${quality==="filtered"?"过滤":"原始"}`,quality,variant,report};
  }));
  return data.rankReports;
}
function pooledStats(items,horizon,options){
  const events=items.flatMap(x=>x.events),s=L.stats(events,horizon,options);
  let baseSum=0,baseAvg=0,baseWeight=0;
  for(const item of items){
    const local=L.stats(item.events,horizon,options),base=L.baseline(item.report,local.rows,{...item.options,horizon});
    if(base.accuracy!=null&&base.n===local.n){baseSum+=base.accuracy*local.n;baseAvg+=base.average*local.n;baseWeight+=local.n;}
  }
  return {...s,baseline:baseWeight?baseSum/baseWeight:null,baselineAverage:baseWeight?baseAvg/baseWeight:null,baselineN:baseWeight};
}
let researchPreparing=false;
async function renderResearch(){
  if(researchPreparing)return;
  const mode=$("researchPeriod").value,pending=state.stocks.filter(stock=>dataset(stock.code,mode)?.results&&!dataset(stock.code,mode).rankReports);
  if(!pending.length){renderResearchReady();return;}
  researchPreparing=true;renderTradeResearch();
  try{
    for(let i=0;i<pending.length;i++){
      if(state.view!=="research"||$("researchPeriod").value!==mode)return;
      $("accuracyContent").innerHTML=`<p class="research-note" role="status">正在计算策略比较 · ${i+1} / ${pending.length} 只股票 · ${L.VARIANTS.length*2} 组固定规则</p>`;
      rankingReports(pending[i],mode);
      if(i%2===0)await new Promise(resolve=>setTimeout(resolve,0));
    }
    if(state.view==="research")renderResearchReady();
  }finally{researchPreparing=false;if(state.view==="research"&&$("researchPeriod").value!==mode)setTimeout(renderResearch,0);}
}
function renderResearchReady(){
  renderTradeResearch();
  const mode=$("researchPeriod").value,horizon=Number($("accuracyHorizon").value),threshold=Number($("accuracyThreshold").value),side=$("accuracySide").value,regime=$("accuracyRegime").value,partition=$("accuracyPartition").value,cutoff=$("accuracyCutoff").value,live=partition==="live";
  const options={side,regime,partition:live?"all":partition,cutoff},groups=new Map();
  let available=0,snapshots=0;
  for(const stock of state.stocks){const data=dataset(stock.code,mode);if(!data?.results)continue;available++;if(data.snapshot||data.error)snapshots++;
    for(const variant of rankingReports(stock,mode)){
      const evalOptions={...evaluationOptions(stock.code,mode),threshold};
      let events=L.observations(variant.report,evalOptions);
      if(live){events=mode!=="m5"||!variant.variant.id.endsWith("standard")?[]:state.journal.filter(x=>x.code===stock.code&&x.strategy===variant.variant.strategy&&x.quality===variant.quality).map(x=>({...x,...x.execution,outcomes:Object.fromEntries(Object.entries(x.outcomes).map(([h,o])=>[h,{...o,correct:L.exceeds(o.direction_return_pct,threshold)}]))}));}
      const item={stock,report:variant.report,events,options:evalOptions};
      if(!groups.has(variant.id))groups.set(variant.id,{...variant,items:[]});groups.get(variant.id).items.push(item);
    }
  }
  const rows=[...groups.values()].map(g=>({...g,stats:pooledStats(g.items,horizon,options),buy:L.stats(g.items.flatMap(x=>x.events),horizon,{...options,side:"buy"}),sell:L.stats(g.items.flatMap(x=>x.events),horizon,{...options,side:"sell"})}));
  rows.sort((a,b)=>Number(b.stats.n>=30)-Number(a.stats.n>=30)||(b.stats.ci?.[0]??-1)-(a.stats.ci?.[0]??-1));
  const chosen=$("rankVariant").value||"donchian-standard_filtered",current=rows.find(r=>r.id===chosen),summary=current?.stats;
  const matched=summary?.baselineN===summary?.n,improvement=matched&&summary?.accuracy!=null&&summary.baseline!=null?summary.accuracy-summary.baseline:null;
  $("accuracyContent").innerHTML=`<div class="research-summary"><div>已验证 / 全部<strong>${summary?.n??0} / ${summary?.total??0}</strong></div><div>预测正确<strong>${summary?.wins??0} 个</strong></div><div>净方向幅度均值<strong>${pct(summary?.average,true)}</strong></div><div>相对同时点基准<strong>${improvement==null?"—":`${fmt(improvement)} 个百分点`}</strong></div></div><p class="research-note">${available} / ${state.stocks.length} 只已加载${snapshots?` · ${snapshots} 只含备用或刷新失败数据`:""}。${mode==="m5"?"分钟历史约 13 个交易日，3/5 日评价可能样本很少。":"日线是另一周期的对照，不代表分钟策略的长期表现。"}${live?"仅展示本机真实捕获记录；非标准参数未进行实时捕获。":`历史留出起点 ${e(cutoff)}：按模拟执行日分组，早期样本跨越留出边界的标签会剔除。参数预先固定，未根据留出结果调参；这是回顾性留出，并非已完成前瞻验证。`}<br>正确标准：成本后方向幅度严格大于 ${threshold}%。未到期 ${summary?.pending??0} 个，取消 ${summary?.cancelled??0} 个，跨界剔除 ${summary?.purged??0} 个。少于 30 个成熟信号不编号排名。基准为同股、同执行日、同方向的可用时点等权结果，再按信号等权汇总；不宣称显著性。</p>${table(["序号 / 固定规则","买入正确 / 已验证","卖出正确 / 已验证","净方向均值","基准正确率 / 提升","95% 描述区间"],rows.map((row,i)=>{const s=row.stats,base=s.baselineN===s.n?s.baseline:null;return `<tr><td>${s.n>=30?i+1:"—"} · ${e(row.name)}<small>${s.n<30?"样本不足":"重叠信号非独立样本"}</small></td><td>${row.buy.wins} / ${row.buy.n}<small>${pct(row.buy.accuracy)}</small></td><td>${row.sell.wins} / ${row.sell.n}<small>${pct(row.sell.accuracy)}</small></td><td>${pct(s.average,true)}</td><td>${pct(base)}<small>${base!=null&&s.accuracy!=null?`${fmt(s.accuracy-base)} 个百分点`:"待匹配基准"}</small></td><td>${s.ci?`${fmt(s.ci[0],1)}–${fmt(s.ci[1],1)}%`:"—"}</td></tr>`;}))}`;
  $("stockAccuracyContent").innerHTML=current?table(["股票","买入正确 / 已验证","卖出正确 / 已验证","净方向均值","同日基准","待验证"],current.items.map(item=>{
    const s=L.stats(item.events,horizon,options),buy=L.stats(item.events,horizon,{...options,side:"buy"}),sell=L.stats(item.events,horizon,{...options,side:"sell"}),base=L.baseline(item.report,s.rows,{...item.options,horizon});
    return `<tr><td><button class="link-button" data-open-stock="${item.stock.code}">${e(item.stock.name)}</button><small>${item.stock.code}</small></td><td>${buy.wins} / ${buy.n}<small>${pct(buy.accuracy)}</small></td><td>${sell.wins} / ${sell.n}<small>${pct(sell.accuracy)}</small></td><td>${pct(s.average,true)}</td><td>${base.n===s.n?pct(base.accuracy):"—"}</td><td>${s.pending}</td></tr>`;
  })):"<div class=\"empty\">等待扫描和历史数据。</div>";
}
function setQuality(quality){
  state.quality=quality;state.focusSignal=null;$("qualitySelect").value=quality;
  for(const [key,data] of state.datasets)if(data.bars?.length)compute(data,key.split("_")[0],key.endsWith("_day")?"day":"m5");
  syncURL();renderSelected();renderWatchlist();if(state.view==="grid")renderGrid();if(state.view==="research")renderResearch();
}
$("strategySelect").innerHTML=L.STRATEGY_IDS.map(id=>`<option value="${id}">${e(C.STRATEGIES[id].name)}</option>`).join("");
$("rankVariant").innerHTML=L.VARIANTS.flatMap(v=>["filtered","original"].map(q=>`<option value="${v.id}_${q}">${e(v.name)} · ${q==="filtered"?"趋势量价过滤":"原始对照"}</option>`)).join("");
$("qualitySelect").addEventListener("change",ev=>setQuality(ev.target.value));
for(const id of ["accuracyHorizon","accuracyThreshold","accuracySide","accuracyRegime","accuracyPartition","accuracyCutoff","rankVariant"])$(id).addEventListener("change",renderResearch);
$("researchPeriod").addEventListener("change",()=>{$("accuracyCutoff").value=$("researchPeriod").value==="day"?"2026-06-01":"2026-11-04";renderResearch();});
function openCandidate(time){
  setQuality("original");
  inspectSignal(time);
}
$("mainChart").addEventListener("click",ev=>{const node=ev.target.closest("[data-candidate]");if(node)openCandidate(node.dataset.candidate);});
$("mainChart").addEventListener("keydown",ev=>{const node=ev.target.closest("[data-candidate]");if(node&&["Enter"," "].includes(ev.key)){ev.preventDefault();openCandidate(node.dataset.candidate);}});
function inspectSignal(time,journalId=null){state.focusSignal={time,journalId,code:state.code,mode:state.mode,strategy:state.strategy,quality:state.quality};renderSignalInspector();}
document.addEventListener("click",ev=>{const target=ev.target.closest("[data-signal-time]");if(target)inspectSignal(target.dataset.signalTime);const record=ev.target.closest("[data-journal-id]");if(record){const item=state.journal.find(x=>x.id===record.dataset.journalId);if(item)inspectSignal(item.time,item.id);}if(ev.target.closest("#closeSignalInspector")){$("signalInspector").hidden=true;state.focusSignal=null;}});
$("mainChart").addEventListener("keydown",ev=>{const target=ev.target.closest("[data-signal-time]");if(target&&["Enter"," "].includes(ev.key)){ev.preventDefault();ev.stopPropagation();inspectSignal(target.dataset.signalTime);}});

document.addEventListener("click",ev=>{
  const action=ev.target.closest("[data-detail-action]")?.dataset.detailAction;if(!action)return;
  if(action==="original"||action==="filtered")setQuality(action);
  if(action==="widen"){state.limit=Math.min(640,dataset()?.bars.length||640);state.offset=0;renderChart();renderDetails();}
  if(action==="signals"){state.detail="signals";renderDetails();}
  if(action==="position")$("positionBtn").click();
  if(action==="refresh")$("refreshBtn").click();
  if(action==="news")refreshNews();
  if(action==="flow")refreshFundFlow(state.code,{force:true});
  if(action==="export-observed"){
    const events=state.journal.filter(x=>x.code===state.code&&x.strategy===state.strategy&&x.quality===state.quality);
    const url=URL.createObjectURL(new Blob([JSON.stringify({version:L.VERSION,exported_at:C.stamp(Date.now()),events},null,2)],{type:"application/json"})),a=document.createElement("a");
    a.href=url;a.download=`${state.code}_${state.strategy}_${state.quality}_observations.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
});

const money=value=>value==null?"—":`${value>0?"+":""}${fmt(value/(Math.abs(value)>=1e8?1e8:1e4))}${Math.abs(value)>=1e8?" 亿":" 万"}`;
async function refreshFundFlow(code=state.code,{force=false}={}){
  const previous=state.fundFlows.get(code);if(previous?.loading||(!force&&previous?.attemptedAt&&Date.now()-previous.attemptedAt<60000))return;
  const current={...previous,loading:true,attemptedAt:Date.now()};state.fundFlows.set(code,current);if(code===state.code&&state.detail==="flow")renderDetails();
  try{const data=await FundFlow.load(stockBy(code));state.fundFlows.set(code,{...data,loading:false,snapshot:false,error:null,attemptedAt:current.attemptedAt});}
  catch(error){
    if(previous?.rows)state.fundFlows.set(code,{...previous,loading:false,error:error.message,attemptedAt:current.attemptedAt});
    else{try{const saved=await getJSON(`data/fund-flow/${code}.json`);const data=FundFlow.parse(saved.payload,stockBy(code),{fetchedAt:saved.fetched_at});state.fundFlows.set(code,{...data,snapshot:true,loading:false,error:error.message,attemptedAt:current.attemptedAt});}catch{state.fundFlows.set(code,{loading:false,error:error.message,attemptedAt:current.attemptedAt});}}
  }
  if(code===state.code&&state.detail==="flow")renderDetails();
}
function fundFlowHTML(){
  const data=state.fundFlows.get(state.code),url=`https://data.eastmoney.com/zjlx/${state.code}.html`;
  if(!data?.rows)return `<div class="detail-empty"><h3>${data?.loading?"正在获取资金流向…":"暂未取得资金流向"}</h3><p>${e(data?.error||"按订单大小分类的净流入数据将在此显示。接口缺失时不估算、不补零。")}</p><button class="text-action" data-detail-action="flow">重新获取 ↻</button> · <a class="text-action" href="${url}" target="_blank" rel="noopener noreferrer">东方财富原始页面 ↗</a></div>`;
  const latest=data.latest,totals=FundFlow.totals(data),partial=C.asEpoch(latest.date)>Date.now()-2000,stale=C.market().open&&latest.date<C.market().day;
  const rows=data.rows.slice(-20),max=Math.max(1,...rows.map(r=>Math.abs(r.main))),width=680,step=640/rows.length;
  const chart=`<svg class="flow-chart" viewBox="0 0 ${width} 160" role="img" aria-label="最近 ${rows.length} 个交易日主力净流入柱形图，红色为净流入，绿色为净流出"><line x1="18" x2="662" y1="72" y2="72" stroke="var(--line)"/>${rows.map((row,i)=>{const h=Math.abs(row.main)/max*55,x=20+i*step,y=row.main>=0?72-h:72;return `<rect x="${x}" y="${y}" width="${Math.max(2,step*.58)}" height="${Math.max(1,h)}" fill="${row.main>=0?"var(--red)":"var(--green)"}"><title>${e(row.date)} · 主力 ${money(row.main)}</title></rect>${i===0||i===rows.length-1||i===Math.floor(rows.length/2)?`<text x="${x}" y="153" font-size="11" fill="var(--muted)" text-anchor="${i===rows.length-1?"end":"start"}">${row.date.slice(5)}</text>`:""}`;}).join("")}</svg>`;
  return `<div class="detail-summary">东方财富 · 最新资金日期 ${e(latest.date)}${partial?" · 当日累计，尚未收盘":" · 最近可用交易日"}${stale?" · 尚未取得当日数据":""}${data.snapshot?" · 备用快照":""}${data.loading?" · 更新中":""}<small>获取时间 ${e(C.stamp(Date.parse(data.fetched_at)).replace("T"," "))}；接口未提供精确的逐笔更新时间。${data.error?`更新暂不可用：${e(data.error)}，保留已有数据。`:""}</small><button class="text-action" data-detail-action="flow">刷新资金 ↻</button> · <a class="text-action" href="${url}" target="_blank" rel="noopener noreferrer">来源原文 ↗</a></div><div class="flow-summary"><div><small>最近一日主力净额</small><strong class="${color(latest.main)}">${money(latest.main)}</strong></div><div><small>已收盘 5 日主力累计</small><strong class="${color(totals[5])}">${money(totals[5])}</strong></div><div><small>已收盘 20 日主力累计</small><strong class="${color(totals[20])}">${money(totals[20])}</strong></div></div>${table(["订单分类","净流入 / 净流出","净额占成交额"],[["主力（超大＋大单）","main"],["超大单","super"],["大单","large"],["中单","medium"],["小单","small"]].map(([label,key])=>`<tr><td>${label}</td><td class="${color(latest[key])}">${money(latest[key])}</td><td>${pct(latest[key+"_pct"],true)}</td></tr>`))}<div class="flow-chart-title">近 ${rows.length} 个交易日 · 主力净流入 / 流出</div>${chart}<p class="flow-note">正值为净流入，负值为净流出；单位为人民币。展示的是按订单大小划分的净额，不是分别列出的流入、流出总额，也不代表机构真实身份或公司账户资金。主力已包含超大单与大单，勿重复相加。${partial?"当日柱形为盘中累计；5/20 日累计仅含已收盘日期。":""}不足指定天数时显示 —。资金数据仅供观察，不参与现有 B/S 信号及历史胜率计算。</p>${table(["日期","主力净额","超大单","大单","中单","小单"],rows.toReversed().map(row=>`<tr><td>${row.date}</td>${["main","super","large","medium","small"].map(key=>`<td class="${color(row[key])}">${money(row[key])}</td>`).join("")}</tr>`))}`;
}

function tAnalysis(){return IntradayT.analyze(dataset(state.code,"m5")?.bars||[],{position:state.positions[state.code],trades:state.tTrades.filter(t=>t.code===state.code),costs:costOptions(state.code),healthy:freshness(state.code,"m5").ok&&!state.paused&&!document.hidden});}
function renderActionGuide(){
  const report=result(state.code,"m5");
  const quote=state.quotes[state.code],t=tAnalysis(),guide=ActionGuide.resolve({report,quote,position:state.positions[state.code],t,strategy:state.strategy,healthy:freshness(state.code,"m5").ok,paused:state.paused||document.hidden});
  const index=report?.bars.findIndex(b=>b.time===report.last_confirmed),bar=index>=0?report.bars[index]:null;
  const strategyName=C.STRATEGIES[state.strategy].name,history=report?report.signals.at(-1):null;
  const events=report?L.observations(report,{...evaluationOptions(state.code,"m5"),threshold:1}):[],buy=L.stats(events,3,{side:"buy"}),sell=L.stats(events,3,{side:"sell"});
  renderStrategySummary(report,buy,sell);
  $("decision").className=`decision action-guide ${guide.kind}`;
  const expanded=$("decision").querySelector(".action-explanation")?.open;
  $("decision").innerHTML=`<div class="action-main"><small>当前判断 · ${strategyName}</small><strong>${e(guide.title)}</strong><p>${e(guide.reason)}</p></div><div class="action-brief"><span>确认 ${e(guide.time?.replace("T"," ")||"等待收线")} · ${e(freshness(state.code,"m5").text)}</span><button class="text-action" data-open-position>设置持仓</button><a href="#research">策略表现 ↗</a></div><details class="action-explanation" ${expanded?"open":""}><summary>查看判断依据与历史验证</summary><div class="action-facts"><span>报价 ${fmt(quote?.price)} · ${e(quote?.as_of?.replace("T"," ")||"未连接")}</span><span>确认收盘 ${fmt(bar?.close)} · ${state.quality==="filtered"?"趋势量价过滤":"原始规则"}</span></div><p>当前规则历史 3 日验证：买入 ${buy.wins}/${buy.n}（${pct(buy.accuracy)}），卖出 ${sell.wins}/${sell.n}（${pct(sell.accuracy)}）。${buy.n<30||sell.n<30?"样本不足，尚不能据此选择最优策略。":"需同时看收益、回撤与基准。"}历史正确率不是本次交易的概率。</p><p>${history?`最近历史触发 ${e(shortTime(history.time))} ${history.side==="buy"?"B":"S"}：${e(history.reason)}。`:"暂无历史触发。"}当前判断固定按 5 分钟确认；图表周期仅改变显示。做 T 尚无独立盈利效果验证。</p></details>`;

}
function renderTWatch(){
  const target=$("tWatch");if(!target||!state.stocks.length)return;
  const t=tAnalysis(),labels={unavailable:"做 T 观察 · 当前不发提示",warming:"做 T 观察 · 等待收线",setup:"做 T 观察 · 待录入底仓",locked:"做 T 观察 · 可卖数量受限",sell_watch:"底仓减仓观察",buyback:"卖后买回观察",wait_buyback:"已有卖出 · 等待买回",watch:"做 T 观察 · 暂无触发"};
  const rows=state.tTrades.filter(x=>x.code===state.code&&x.day===t.day),weekday=new Date(t.day+"T12:00:00+08:00").getUTCDay(),canRecord=weekday>0&&weekday<6;
  target.innerHTML=`<div class="t-watch-head"><strong class="${t.status==="sell_watch"?"down":t.status==="buyback"?"up":""}">${e(labels[t.status])}</strong><span>${t.last?`确认 ${e(sourceTime(t.last))}`:"固定按 5 分钟收线检查"}</span></div><p>${e(t.reason)}</p><div class="t-watch-values"><span>剩余可卖底仓 <b>${t.account.available} 股</b></span><span>已卖未买回 <b>${t.account.outstanding} 股</b></span><span>含成本买回参考 <b>${fmt(t.target)}</b></span><span>当日均价估计 <b>${fmt(t.vwap)}</b></span></div><div class="t-actions"><button data-t-action="position">设置底仓</button><button data-t-action="sell" ${t.account.available&&canRecord?"":"disabled"}>登记已成交卖出</button><button data-t-action="buy" ${t.account.outstanding&&canRecord?"":"disabled"}>登记已成交买回</button></div><details><summary>做 T 规则与本日登记（${rows.length}）</summary><p>仅支持“先卖已有可卖底仓，再买回”的观察；新买回股份当天不增加可卖数量。上冲偏离当日均价 ≥ 0.4% 后跌回 MA5，显示减仓观察；已登记卖出后，价格低于买回参考并重新上穿 MA5，显示买回观察。不是高胜率承诺，可能继续上涨而无法低价买回。</p><p>买回参考计入单笔最低佣金 5 元、配置佣金和卖出税费、未来买入滑点 0.1%，并预留原卖出金额 0.3% 的价差目标。请按实际券商费用核对。只登记你已在券商完成的成交，不自动下单，不参与趋势策略胜率。</p>${rows.length?table(["登记时间","成交方向","数量","实际均价"],rows.toReversed().map(x=>`<tr><td>${e(shortTime(x.time))}</td><td>${x.side==="sell"?"底仓卖出":"买回"}</td><td>${x.qty}</td><td>${fmt(x.price)}</td></tr>`)):"<p>本日尚无成交登记；数量仅来自本机录入，不与券商同步。跨日请重新核对初始可卖数量，历史登记不自动滚入新一天。</p>"}</details>`;
}
let pendingTTrade=null;
document.addEventListener("click",ev=>{
  if(ev.target.closest("[data-open-position]")){$("positionBtn").click();return;}
  const action=ev.target.closest("[data-t-action]")?.dataset.tAction;if(!action)return;
  if(action==="position"){$("positionBtn").click();return;}
  const t=tAnalysis(),qty=action==="sell"?t.account.available:t.account.outstanding;pendingTTrade={side:action,code:state.code,day:t.day};
  $("tTradeTitle").textContent=`${selected().name} · 登记已成交${action==="sell"?"卖出":"买回"}`;
  $("tTradePrice").value=state.quotes[state.code]?.price||"";$("tTradeQty").value=Math.min(100,qty);$("tTradeQty").max=qty;$("tTradeError").textContent="";$("tTradeHint").textContent=`请填写券商实际成交均价和股数。当前${action==="sell"?"剩余可卖底仓":"待买回数量"} ${qty} 股；不自动下单。`;
  $("tTradeDialog").showModal();
});
$("tTradeClose").addEventListener("click",()=>$("tTradeDialog").close());
$("tTradeForm").addEventListener("submit",ev=>{
  ev.preventDefault();if(!pendingTTrade)return;
  if(pendingTTrade.day!==C.market().day){$("tTradeError").textContent="交易日已变化，请关闭后重新登记";return;}
  const trade={...pendingTTrade,price:Number($("tTradePrice").value),qty:Number($("tTradeQty").value),time:C.stamp(Date.now())};
  try{IntradayT.validateTrade(state.positions[trade.code],state.tTrades.filter(x=>x.code===trade.code),trade,{costs:costOptions(trade.code)});const next=[...state.tTrades,trade];localStorage.setItem("trading.t-trades.v1",JSON.stringify(next));state.tTrades=next;$("tTradeDialog").close();renderTWatch();renderActionGuide();}
  catch(error){$("tTradeError").textContent=error.message;}
});

function equityChartHTML(report){
  if(!report?.curve.length)return '<p class="research-note">等待完整历史后展示策略净值与买入持有曲线。</p>';
  const rows=report.curve,width=720,height=180,values=rows.flatMap(r=>[r.equity,r.baseline]),min=Math.min(1,...values),max=Math.max(1,...values),span=max-min||.01,x=i=>12+i/Math.max(1,rows.length-1)*696,y=v=>145-(v-min)/span*125;
  return `<div class="equity-title">${e(selected().name)} · ${e(C.STRATEGIES[state.strategy].name)}<small>蓝线：策略净值 · 虚线：买入持有 · 归一化初值 1</small></div><svg class="equity-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="当前股票策略净值与买入持有对照"><line x1="12" x2="708" y1="${y(1)}" y2="${y(1)}" stroke="var(--line)"/><path d="${svgPath(rows,"baseline",x,y)}" fill="none" stroke="var(--muted)" stroke-dasharray="4 3" stroke-width="1.2"/><path d="${svgPath(rows,"equity",x,y)}" fill="none" stroke="var(--blue)" stroke-width="1.8"/><text x="12" y="173" fill="var(--muted)" font-size="11">${e(shortTime(rows[0].time))}</text><text x="708" y="173" text-anchor="end" fill="var(--muted)" font-size="11">${e(shortTime(rows.at(-1).time))}</text></svg><p class="research-note">单股独立账户净值，以当前数据窗口为准；数据区间 ${e(rows[0].time)} 至 ${e(rows.at(-1).time)}，未平仓按收盘市值计入净值，未计入胜率。不是 38 股组合收益。</p>`;
}
function strategyLevels(ind){
  if(state.strategy==="donchian")return [["突破参考",ind.upper],["退出参考",ind.lower]];
  if(state.strategy==="macd")return [["DIF",ind.macd],["DEA",ind.macdSignal]];
  if(state.strategy==="rsi")return [["RSI14",ind.rsi14],["修复退出阈值",55]];
  if(state.strategy==="bollinger")return [["布林下轨",ind.bbLower],["布林中轨",ind.bbMid]];
  return [["MA20",ind.ma20],["MA60",ind.ma60]];
}
function strategyCatalogHTML(){return L.STRATEGY_IDS.map(id=>{const meta=C.STRATEGIES[id];return `<article class="catalog-item"><h3>${e(meta.name)} <span class="badge">${["donchian","sma"].includes(id)?"固定规则":"新增实验"}</span></h3><p>${e(meta.family)} · 买入：${e(meta.buy)}。退出：${e(meta.sell)}。</p><p>适用思路：${e(meta.family)}。主要局限：${e(meta.risk)}。版本 v1；统一使用已收线、后续开盘模拟、费用与 T+1。</p><button class="text-action" data-research-strategy="${id}">查看此策略</button> · <a class="text-action" href="${e(safeURL(meta.source))}" target="_blank" rel="noopener noreferrer">指标依据 ↗</a></article>`;}).join("");}
function renderStrategySummary(report,buy,sell){
  const box=$("strategySummary"),m=report?.metrics,meta=C.STRATEGIES[state.strategy],experimental=!["donchian","sma"].includes(state.strategy);
  const data=dataset(state.code,"m5"),otherQuality=state.quality==="filtered"?"original":"filtered";
  let other=null;
  if(report){
    if(data.summaryCompare?.base===report)other=data.summaryCompare.report;
    else{other=otherQuality==="original"&&data.originalChart?.base===report?data.originalChart.report:L.analyze(data.bars,{...costOptions(state.code),now:report.compute_epoch,strategy:state.strategy,quality:otherQuality,mode:"m5",daily:dataset(state.code,"day")?.bars||[]});data.summaryCompare={base:report,report:other};}
  }
  const original=state.quality==="original"?report:other,filtered=state.quality==="filtered"?report:other;
  const factor=m?.profit_factor,qualityNote=state.quality==="filtered"?"过滤版":"原始版";
  box.innerHTML=`<div class="strategy-summary-head"><div><label class="quick-strategy-label">当前策略 <select data-strategy-picker aria-label="切换当前策略">${L.STRATEGY_IDS.map(id=>`<option value="${id}" ${id===state.strategy?"selected":""}>${e(C.STRATEGIES[id].name)}</option>`).join("")}</select></label> <span class="badge">${qualityNote}${experimental?" · 实验策略":""}</span><small>${e(meta.family)} · ${e(meta.risk)}</small></div><div class="strategy-summary-links"><a href="#research">完整策略比较 ↗</a><button class="text-action" data-export-research>导出报告 ↓</button></div></div><div class="performance-strip">${[["已平仓胜率",m?.closed_trades?pct(m.win_rate):"暂无平仓",m?`${m.winning_trades} / ${m.closed_trades} 笔盈利`:"等待数据",""],["区间收益 · 扣费",pct(m?.total_return_pct,true),"含未平仓市值",color(m?.total_return_pct)],["最大回撤",pct(m?.max_drawdown_pct),"按区间净值估算",""],["盈利因子",factor==null?"—":fmt(factor),m?.closed_trades?m.gross_loss>0?"盈利总额 / 亏损总额":"尚无亏损分母":"尚无完整交易",""]].map(([label,value,note,cl])=>`<div><span>${label}</span><strong class="${cl}">${value}</strong><small>${e(note)}</small></div>`).join("")}</div><div class="quality-compare">${[["original","原始",original],["filtered","过滤",filtered]].map(([quality,label,r])=>`<button class="${quality===state.quality?"active":""}" data-detail-action="${quality}">${label}胜率 ${r?.metrics.closed_trades?pct(r.metrics.win_rate):"暂无平仓"}<small>盈利 ${r?.metrics.winning_trades??0} / 平仓 ${r?.metrics.closed_trades??0}</small></button>`).join("")}</div><div class="strategy-evidence"><span>3 日方向正确率：买入 ${pct(buy.accuracy)}（${buy.wins}/${buy.n}），卖出 ${pct(sell.accuracy)}（${sell.wins}/${sell.n}）</span><span>${m?.start?`${e(shortTime(m.start))} — ${e(shortTime(m.end))}`:"等待完整历史"}</span></div><p>${(m?.closed_trades||0)<30?"平仓样本不足 30 笔，仅供观察。":"历史回测不能保证后续表现。"}5 分钟近期历史；未平仓不计胜率，方向正确率不等于交易胜率。${!freshness(state.code,"m5").ok?`数据状态：${e(freshness(state.code,"m5").text)}。`:""}</p>`;
}
document.addEventListener("click",event=>{
  if(!event.target.closest("[data-export-research]"))return;
  const data=dataset(state.code,"m5"),report=result(state.code,"m5");if(!report)return;
  const payload={schema:"trading-research-v1",exported_at:C.stamp(Date.now()),computed_at:report.computed_at,compute_epoch:report.compute_epoch,code:state.code,name:selected().name,strategy:state.strategy,definition:C.STRATEGIES[state.strategy],quality:state.quality,mode:"m5",rule_version:L.VERSION,source:data.source,basis:data.basis,snapshot:Boolean(data.snapshot),fetched_at:data.fetched_at,costs:costOptions(state.code),metrics:report.metrics,bars:report.bars,daily_reference:dataset(state.code,"day")?.bars||[],signals:report.signals,trades:report.trades};
  const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:"application/json"})),a=document.createElement("a");a.href=url;a.download=`${state.code}_${state.strategy}_${state.quality}_research.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
$("strategyCatalog").innerHTML=strategyCatalogHTML();

initialize();
