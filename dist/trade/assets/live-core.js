/* Pure deterministic signal engine, shared by the terminal and browser tests. */
"use strict";
(function(root){
  const TZ="Asia/Shanghai";
  const stamp=date=>new Date(date).toLocaleString("sv-SE",{timeZone:TZ,hour12:false}).replace(" ","T");
  const asEpoch=value=>Date.parse(value.slice(0,19)+(value.length===10?"T15:00:00":"")+"+08:00");
  function market(now=Date.now()){
    const text=stamp(now),day=text.slice(0,10),time=text.slice(11,19),weekday=new Date(day+"T12:00:00+08:00").getUTCDay();
    const open=weekday>0&&weekday<6&&((time>="09:30:00"&&time<="11:30:05")||(time>="13:00:00"&&time<="15:00:05"));
    return {day,time,open,label:open?"交易时段":time>"11:30:05"&&time<"13:00:00"&&weekday>0&&weekday<6?"午间休市":"非交易时段"};
  }
  function validate(rows){
    let previous="";
    for(const b of rows){
      if(!/^\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2})?$/.test(b.time)||b.time<=previous)throw Error("K 线时间重复或无序");
      if(![b.open,b.close,b.high,b.low,b.volume].every(Number.isFinite)||Math.min(b.open,b.close,b.high,b.low)<=0||b.volume<0||b.low>Math.min(b.open,b.close)+.011||b.high<Math.max(b.open,b.close)-.011)throw Error("K 线数值异常");
      previous=b.time;
    }
    return rows;
  }
  const STRATEGIES={
    donchian:{name:"唐奇安突破 20/10",family:"趋势突破",buy:"收盘突破前 20 根最高价",sell:"收盘跌破前 10 根最低价",risk:"震荡行情容易反复止损"},
    sma:{name:"均线趋势 20/60",family:"趋势跟随",buy:"MA20 高于 MA60",sell:"MA20 低于 MA60",risk:"反应较慢，横盘时容易来回交易"},
    macd:{name:"MACD 动量 12/26/9",family:"动量转强",buy:"DIF 从下向上穿越 DEA",sell:"DIF 低于 DEA",risk:"震荡区间可能频繁假交叉"},
    rsi:{name:"RSI 超卖修复 14",family:"均值回归",buy:"RSI14 从不高于 30 回升到 30 以上",sell:"RSI14 达到 55 或再次跌破 20",risk:"强下跌中超卖可能延续"},
    bollinger:{name:"布林带回归 20/2",family:"区间修复",buy:"前收盘在下轨外，当前回到下轨内且低于中轨",sell:"收盘回到中轨，或跌至下轨的 98% 以下",risk:"单边下跌可能沿下轨继续走弱"}
  };
  const strategySources={donchian:"https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/price-channels",sma:"https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/sma",macd:"https://www.tradingview.com/support/solutions/43000644943-macd-strategy/",rsi:"https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/RSI",bollinger:"https://www.tradingview.com/support/solutions/43000501840-bollinger-bands-bb/"};
  for(const [id,meta] of Object.entries(STRATEGIES))Object.assign(meta,{version:"v1",source:strategySources[id]});
  function indicators(bars,{fastPeriod=20,slowPeriod=60,entryPeriod=20,exitPeriod=10}={}){
    const out=[];let fast=0,slow=0,gain=0,loss=0,ema12=null,ema26=null,dea=null;const difSeeds=[];
    bars.forEach((b,i)=>{
      fast+=b.close;slow+=b.close;if(i>=fastPeriod)fast-=bars[i-fastPeriod].close;if(i>=slowPeriod)slow-=bars[i-slowPeriod].close;
      const delta=i?b.close-bars[i-1].close:0;
      if(i>0&&i<=14){gain+=Math.max(0,delta)/14;loss+=Math.max(0,-delta)/14;}
      else if(i>14){gain=(gain*13+Math.max(0,delta))/14;loss=(loss*13+Math.max(0,-delta))/14;}
      const rsi14=i<14?null:loss===0?(gain===0?50:100):100-100/(1+gain/loss);
      if(i===11)ema12=bars.slice(0,12).reduce((sum,x)=>sum+x.close,0)/12;else if(i>11)ema12+=2/13*(b.close-ema12);
      if(i===25)ema26=bars.slice(0,26).reduce((sum,x)=>sum+x.close,0)/26;else if(i>25)ema26+=2/27*(b.close-ema26);
      const macd=ema26===null?null:ema12-ema26;
      if(macd!==null){if(difSeeds.length<9){difSeeds.push(macd);if(difSeeds.length===9)dea=difSeeds.reduce((a,x)=>a+x,0)/9;}else dea+=.2*(macd-dea);}
      const sample=bars.slice(Math.max(0,i-19),i+1),bbMid=i>=19?sample.reduce((sum,x)=>sum+x.close,0)/20:null;
      const std=bbMid===null?null:Math.sqrt(sample.reduce((sum,x)=>sum+(x.close-bbMid)**2,0)/20),previous=out.at(-1);
      out.push({rsi14,macd,macdSignal:dea,prevRsi:previous?.rsi14??null,prevMacd:previous?.macd??null,prevMacdSignal:previous?.macdSignal??null,prevClose:bars[i-1]?.close??null,prevBBLower:previous?.bbLower??null,bbMid,bbLower:std===null?null:bbMid-2*std,bbUpper:std===null?null:bbMid+2*std,ma20:i>=fastPeriod-1?fast/fastPeriod:null,ma60:i>=slowPeriod-1?slow/slowPeriod:null,upper:i>=entryPeriod?Math.max(...bars.slice(i-entryPeriod,i).map(x=>x.high)):null,lower:i>=exitPeriod?Math.min(...bars.slice(i-exitPeriod,i).map(x=>x.low)):null});
    });return out;
  }
  function conditions(b,ind,strategy){
    if(!Object.hasOwn(STRATEGIES,strategy))throw Error("未知策略："+strategy);
    if(strategy==="donchian")return [ind.upper!=null&&b.close>ind.upper,ind.lower!=null&&b.close<ind.lower];
    if(strategy==="sma")return [ind.ma60!=null&&ind.ma20>ind.ma60,ind.ma60!=null&&ind.ma20<ind.ma60];
    if(strategy==="macd")return [ind.prevMacdSignal!=null&&ind.prevMacd<=ind.prevMacdSignal&&ind.macd>ind.macdSignal,ind.macdSignal!=null&&ind.macd<ind.macdSignal];
    if(strategy==="rsi")return [ind.prevRsi!=null&&ind.prevRsi<=30&&ind.rsi14>30,ind.rsi14!=null&&(ind.rsi14>=55||ind.rsi14<20)];
    return [ind.prevBBLower!=null&&ind.prevClose<ind.prevBBLower&&b.close>ind.bbLower&&b.close<ind.bbMid,ind.bbMid!=null&&(b.close>=ind.bbMid||b.close<ind.bbLower*.98)];
  }

  function wilson(wins,n){if(!n)return null;const z=1.96,p=wins/n,d=1+z*z/n,c=(p+z*z/(2*n))/d,delta=z*Math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [(c-delta)*100,(c+delta)*100];}
  function reason(strategy,side){return STRATEGIES[strategy][side==="buy"?"buy":"sell"];}

  function barStart(time,mode){if(mode==="day")return time+"T09:30:00";return stamp(asEpoch(time)-5*60000);}
  function backtest(bars,{strategy="donchian",mode="m5",commission=.0003,tax=.0005,slippage=.001,code="",upperLimit=null,lowerLimit=null,now=Date.now(),params={},qualify=null}={}){
    // Quote-only candles may preview the current interval. An expired synthetic
    // candle must wait for the minute feed, never become a confirmed signal.
    bars=bars.filter(b=>(!b.synthetic&&!b.provisional)||asEpoch(b.time)>now-2000);
    validate(bars);
    const ind=indicators(bars,params),confirmed=bars.map((b,i)=>({b,i})).filter(({b})=>mode==="day"?asEpoch(b.time)<=now-2000:asEpoch(b.time)<=now-2000);
    let cash=1,units=0,entry=null,pending=null,peak=1,maxDD=0,exposure=0,baselineUnits=null;
    const signals=[],trades=[],curve=[];
    const buyFactor=(1+slippage)*(1+commission),sellFactor=(1-slippage)*(1-commission-tax);
    const active=confirmed.filter(({i})=>i>=60);
    // Day reference closes for conservative historical opening limit checks.
    const prevDayClose=new Map();let priorDay=null,lastClose=null;
    for(const {b} of confirmed){const d=b.time.slice(0,10);if(d!==priorDay&&priorDay!==null)prevDayClose.set(d,lastClose);priorDay=d;lastClose=b.close;}
    for(const {b,i} of active){
      const day=b.time.slice(0,10),reference=mode==="day"?bars[i-1]?.close:prevDayClose.get(day);
      const limit=code.startsWith("30") ? 0.195 : 0.095,gap=reference?b.open/reference-1:0;
      const canSell=!entry||day>entry.day;
      if(baselineUnits===null&&i>active[0].i&&b.volume>0)baselineUnits=1/(b.open*buyFactor);
      if(pending){
        const locked=pending.side==="sell"&&!canSell;
        const constrained=b.volume<=0||(pending.side==="buy"&&gap>=limit)||(pending.side==="sell"&&gap<=-limit);
        if(locked||constrained){pending.blocked_bars++;if(locked)pending.t1_locked=true;}
        else{
          const fill=b.open*(pending.side==="buy"?1+slippage:1-slippage);
          Object.assign(pending,{status:"filled",fill_time:barStart(b.time,mode),fill_bar_time:b.time,fill_index:i,fill_price:fill});
          if(pending.side==="buy"){entry={day,index:i,time:barStart(b.time,mode),price:fill,capital:cash,signal_time:pending.time};units=cash/(b.open*buyFactor);cash=0;}
          else{cash=units*b.open*sellFactor;units=0;trades.push({entry_time:entry.time,entry_signal_time:entry.signal_time,entry_price:entry.price,exit_time:barStart(b.time,mode),exit_signal_time:pending.time,exit_price:fill,pnl:cash-entry.capital,return_pct:(cash/entry.capital-1)*100,holding_bars:i-entry.index});entry=null;}
          pending=null;
        }
      }
      const equity=cash+units*b.close;peak=Math.max(peak,equity);maxDD=Math.min(maxDD,equity/peak-1);exposure+=units>0?1:0;
      curve.push({time:b.time,equity,baseline:baselineUnits===null?1:baselineUnits*b.close});
      const [rawBuy,sell]=conditions(b,ind[i],strategy),gate=qualify?qualify(b,i,ind[i],bars):{pass:true,checks:[]},buy=rawBuy&&gate.pass;
      if(pending?.side==="buy"&&!buy){pending.status="cancelled";pending.cancel_time=b.time;pending=null;}
      if(!pending){const side=!units&&buy&&b.volume>0?"buy":units&&sell?"sell":null;if(side){pending={time:b.time,index:i,side,reason:reason(strategy,side)+(side==="buy"&&qualify?"；趋势、量价和波动过滤通过":""),checks:gate.checks||[],regime:gate.regime||"unknown",status:"pending",blocked_bars:0,t1_locked:side==="sell"&&entry?.day===day};signals.push(pending);}}
    }
    const last=active.at(-1),lastEquity=curve.at(-1)?.equity??1,wins=trades.filter(t=>t.return_pct>0).length;
    const grossProfit=trades.reduce((sum,t)=>sum+Math.max(0,t.pnl),0),grossLoss=trades.reduce((sum,t)=>sum+Math.max(0,-t.pnl),0);
    let status=pending?(pending.side==="buy"?"buy_pending":pending.t1_locked&&entry?.day===last?.b.time.slice(0,10)?"sell_locked":"sell_pending"):units?"holding":"watching";
    if(!active.length)status="insufficient";
    const forming=bars.filter(b=>mode==="day"?asEpoch(b.time)>now-2000:asEpoch(b.time)>now-2000).at(-1);
    let preview=null;
    if(forming&&bars.indexOf(forming)>=60&&!pending){const index=bars.indexOf(forming),[rawBuy,sell]=conditions(forming,ind[index],strategy),gate=qualify?qualify(forming,index,ind[index],bars):{pass:true},buy=rawBuy&&gate.pass,side=!units&&buy?"buy":units&&sell?"sell":null;if(side)preview={time:forming.time,side,reason:reason(strategy,side),status:"forming",t1_locked:side==="sell"&&entry?.day===forming.time.slice(0,10)};}
    return {bars,indicators:ind,signals,trades,curve,status,pending,preview,entry,metrics:{profit_factor:grossLoss>0?grossProfit/grossLoss:null,gross_profit:grossProfit,gross_loss:grossLoss,closed_trades:trades.length,winning_trades:wins,win_rate:trades.length?wins/trades.length*100:null,win_rate_ci:wilson(wins,trades.length),total_return_pct:(lastEquity-1)*100,baseline_return_pct:((curve.at(-1)?.baseline??1)-1)*100,max_drawdown_pct:maxDD*100,exposure_pct:curve.length?exposure/curve.length*100:0,start:curve[0]?.time,end:curve.at(-1)?.time},last_confirmed:last?.b.time||null};
  }
  function quoteState(quote,now=Date.now()){
    if(!quote?.as_of)return {kind:"missing",text:"未连接"};
    const m=market(now),day=quote.as_of.slice(0,10),age=(now-asEpoch(quote.as_of))/1000;
    if(day<m.day&&m.open)return {kind:"stale",text:"行情日期落后"};
    if(m.open&&age>90)return {kind:"stale",text:`延迟 ${Math.floor(age/60)} 分钟`};
    return {kind:"ok",text:m.open?"行情已连接":"最近行情"};
  }
  // Source candles are labelled by their closing time. Reset each half-day.
  function chartBars(bars,minutes){
    if(minutes===5)return bars;
    const result=[];
    let group=null;
    for(const b of bars){
      const minute=Number(b.time.slice(11,13))*60+Number(b.time.slice(14,16));
      const start=minute<=690?570:780,offset=minute-start;
      if(offset<=0||offset>120||offset%5!==0)continue;
      const end=start+Math.ceil(offset/minutes)*minutes;
      const time=b.time.slice(0,11)+String(Math.floor(end/60)).padStart(2,"0")+":"+String(end%60).padStart(2,"0")+":00";
      if(group?.time!==time){
        group={...b,time,first:b.time,count:1};result.push(group);
      }else{
        group.high=Math.max(group.high,b.high);group.low=Math.min(group.low,b.low);
        group.close=b.close;group.volume+=b.volume;group.count++;
      }
      group.last=b.time;
    }
    return result.filter(b=>b.count===minutes/5);
  }
  function shortAverages(bars){
    return bars.map((_,i)=>{
      const at=n=>i+1<n?null:bars.slice(i+1-n,i+1).reduce((sum,b)=>sum+b.close,0)/n;
      return {ma5:at(5),ma10:at(10)};
    });
  }
  root.TradingCore={STRATEGIES,reason,chartBars,shortAverages,stamp,asEpoch,market,validate,indicators,conditions,wilson,backtest,quoteState,barStart};
})(globalThis);
