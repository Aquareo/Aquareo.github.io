/* Causal filters and fixed-horizon research. No fitting or future-bar filters. */
"use strict";
(function(root){
  const C=root.TradingCore,VERSION="signal-lab-2",HORIZONS=[1,3,5];
  const STRATEGY_IDS=Object.keys(C.STRATEGIES);
  const VARIANTS=[
    {id:"donchian-standard",strategy:"donchian",name:"唐奇安 20/10",params:{entryPeriod:20,exitPeriod:10}},
    {id:"donchian-slow",strategy:"donchian",name:"唐奇安 40/20",params:{entryPeriod:40,exitPeriod:20}},
    {id:"sma-standard",strategy:"sma",name:"均线 20/60",params:{fastPeriod:20,slowPeriod:60}},
    {id:"sma-fast",strategy:"sma",name:"均线 10/30",params:{fastPeriod:10,slowPeriod:30}},
    ...["macd","rsi","bollinger"].map(strategy=>({id:strategy+"-standard",strategy,name:C.STRATEGIES[strategy].name,params:{}}))
  ];
  const exceeds=(value,threshold)=>value-threshold>1e-9;
  const mean=a=>a.length?a.reduce((s,x)=>s+x,0)/a.length:null;
  const date=b=>String(b.time||b.date).slice(0,10);
  const closed=(bars,now)=>bars.filter(b=>!b.synthetic&&!b.provisional&&C.asEpoch(b.time)<=now-2000);
  function dailyContext(daily,now=Date.now()){
    const bars=closed(daily,now),ind=C.indicators(bars);
    return {bars,ind,days:bars.map(date)};
  }
  function trendAt(context,time){
    const day=time.slice(0,10);let lo=0,hi=context.days.length;
    while(lo<hi){const mid=(lo+hi)>>1;if(context.days[mid]<day)lo=mid+1;else hi=mid;}
    const i=lo-1,b=context.bars[i],ind=context.ind[i];
    if(!b||!ind?.ma60)return {pass:false,regime:"unknown",day:b?date(b):null,ma20:null,ma60:null,close:b?.close};
    const regime=ind.ma20>ind.ma60&&b.close>ind.ma20?"up":ind.ma20<ind.ma60&&b.close<ind.ma20?"down":"range";
    return {pass:regime==="up",regime,day:date(b),ma20:ind.ma20,ma60:ind.ma60,close:b.close};
  }
  function auditor(daily,mode="m5",now=Date.now(),params={},calendar=[],strategy="donchian"){
    const context=dailyContext(daily,now);let preparedBars=null,prepared=[];
    const trends=new Map();
    function prepare(bars){
      const volumes=[0],ranges=[0],slots=new Map();prepared=[];preparedBars=bars;
      for(let i=0;i<bars.length;i++){
        const b=bars[i],previous=bars[i-1]?.close,slot=b.time.slice(11),peers=slots.get(slot)||[];
        volumes.push(volumes.at(-1)+b.volume);
        ranges.push(ranges.at(-1)+Math.max(b.high-b.low,previous?Math.abs(b.high-previous):0,previous?Math.abs(b.low-previous):0));
        const volumeBase=mode==="m5"?(peers.length>=3?mean(peers.map(x=>x.volume)):null):(i>=20?(volumes[i]-volumes[i-20])/20:null);
        const atr=i>=20?(ranges[i]-ranges[i-20])/20:null;
        prepared.push({volumeBase,atr});
        slots.set(slot,[...peers,{day:date(b),volume:b.volume}].slice(-5));
      }
    }
    return function(b,i,ind,bars){
      if(preparedBars!==bars)prepare(bars);
      const day=b.time.slice(0,10);
      if(!trends.has(day)){
        const trend=trendAt(context,b.time);let lo=0,hi=calendar.length;
        while(lo<hi){const mid=(lo+hi)>>1;if(calendar[mid]<day)lo=mid+1;else hi=mid;}
        const expected=calendar[lo-1];
        if(expected&&trend.day!==expected)trend.pass=false;
        trends.set(day,{trend,expected});
      }
      const stored=trends.get(day),trend={...stored.trend},expected=stored.expected,reversion=["rsi","bollinger"].includes(strategy);
      if(reversion)trend.pass=trend.regime!=="down"&&trend.regime!=="unknown"&&(!expected||trend.day===expected);
      const {volumeBase,atr}=prepared[i],ratio=volumeBase>0?b.volume/volumeBase:null,atrPct=atr?atr/b.close*100:null,rangeATR=atr?(b.high-b.low)/atr:null;
      const thresholds=mode==="m5"?[.08,3]:[.3,12];
      const checks=[
        {key:"trend",label:reversion?"日线避开明确下跌环境":"上一交易日日线趋势",pass:trend.pass,value:trend.day?`${trend.day} · MA20 ${trend.ma20?.toFixed(2)??"—"} / MA60 ${trend.ma60?.toFixed(2)??"—"}`:"缺少日线",...trend},
        {key:"trigger",label:reversion?"修复触发独立判断":"当前周期趋势",pass:reversion||(ind.ma20!=null&&ind.ma60!=null&&ind.ma20>ind.ma60),value:reversion?"使用超卖/下轨重返规则":`MA${params.fastPeriod||20} > MA${params.slowPeriod||60}`},
        {key:"volume",label:`${mode==="m5"?"历史同刻":"前 20 日均量"}比 ≥ ${reversion ? .8 : 1.2}`,pass:ratio!=null&&ratio>=(reversion ? .8 : 1.2),value:ratio==null?"同刻历史不足":`${ratio.toFixed(2)} 倍`,ratio},
        {key:"volatility",label:`波动 ${thresholds[0]}–${thresholds[1]}% · 振幅 ≤ 3 ATR`,pass:atrPct!=null&&atrPct>=thresholds[0]&&atrPct<=thresholds[1]&&rangeATR<=3,value:atrPct==null?"历史不足":`${atrPct.toFixed(2)}% · ${rangeATR.toFixed(2)} ATR`,atrPct,rangeATR}
      ];
      return {pass:checks.every(c=>c.pass),checks,regime:trend.regime};
    };
  }
  function analyze(bars,{daily=[],mode="m5",strategy="donchian",quality="filtered",params={},now=Date.now(),...options}={}){
    const audit=auditor(daily,mode,now,params,options.calendar||[],strategy),report=C.backtest(bars,{...options,strategy,mode,params,now,qualify:quality==="filtered"?audit:null});
    const index=report.bars.findIndex(b=>b.time===report.last_confirmed);
    report.audit=index>=0?audit(report.bars[index],index,report.indicators[index],report.bars):null;
    for(const order of report.signals){
      if(quality!=="filtered"){const gate=audit(report.bars[order.index],order.index,report.indicators[order.index],report.bars);order.regime=gate.regime;order.checks=gate.checks;}
      order.reason=!["donchian","sma"].includes(strategy)?C.reason(strategy,order.side):strategy==="donchian"?(order.side==="buy"?`收盘突破前 ${params.entryPeriod||20} 根最高价`:`收盘跌破前 ${params.exitPeriod||10} 根最低价`):`MA${params.fastPeriod||20} ${order.side==="buy"?"高于":"低于"} MA${params.slowPeriod||60}`;
      if(quality==="filtered"&&order.side==="buy")order.reason+="；趋势、量价与波动过滤通过";
      const at=report.indicators[order.index],bar=report.bars[order.index];
      const inputs=strategy==="donchian"?{close:bar.close,upper:at.upper,lower:at.lower}:strategy==="sma"?{maFast:at.ma20,maSlow:at.ma60}:strategy==="macd"?{previousDIF:at.prevMacd,previousDEA:at.prevMacdSignal,DIF:at.macd,DEA:at.macdSignal}:strategy==="rsi"?{previousRSI:at.prevRsi,RSI:at.rsi14}:{previousClose:at.prevClose,previousLower:at.prevBBLower,close:bar.close,lower:at.bbLower,middle:at.bbMid};
      order.checks=[{key:"rule",label:order.reason,pass:true,inputs,value:Object.entries(inputs).map(([key,value])=>`${key} ${value==null?"—":Number(value.toFixed(4))}`).join(" / ")},...order.checks];

    }
    report.quality=quality;report.version=VERSION;report.computed_at=C.stamp(now);report.compute_epoch=now;
    return report;
  }
  function sessionDays(bars,calendar=[]){return [...new Set([...calendar.map(x=>typeof x==="string"?x:date(x)),...bars.map(date)])].sort();}
  function returnAt(entry,target,side,{commission=.0003,tax=.0005,slippage=.001}={}){
    if(side==="buy")return (target*(1-slippage)*(1-commission-tax)/(entry*(1+slippage)*(1+commission))-1)*100;
    // Benefit of selling versus continuing to hold, after sell-side costs.
    return ((entry*(1-slippage)*(1-commission-tax)-target)/entry)*100;
  }
  function evaluator(bars,{mode="m5",calendar=[],now=Date.now(),threshold=1,...costs}={}){
    const confirmed=closed(bars,now),days=sessionDays(confirmed,calendar),lastByDay=new Map();
    for(const b of confirmed)if(mode==="day"||b.time.slice(11)==="15:00:00")lastByDay.set(date(b),b);
    return function(order){const out={};
    if(order.status!=="filled"||!Number.isFinite(order.entry_open)||!order.fill_bar_time)return {state:order.status==="cancelled"?"cancelled":"awaiting_fill",outcomes:out};
    const fillDay=order.fill_bar_time.slice(0,10),start=days.indexOf(fillDay);
    for(const h of HORIZONS){
      const targetDay=days[start+h],target=lastByDay.get(targetDay);
      if(start<0||!target)continue;
      // Days are counted on the common trading calendar, never elapsed wall time.
      const value=returnAt(order.entry_open,target.close,order.side,costs);
      out[h]={target_day:targetDay,time:target.time,raw_return_pct:(target.close/order.entry_open-1)*100,direction_return_pct:value,correct:exceeds(value,threshold),threshold,evaluated_at:C.stamp(now)};
    }
    return {state:out[3]?"evaluated":"confirmed",outcomes:out};
    };
  }
  // Use completed real candles only. The rule fill supplies the earliest
  // execution allowed by the strategy position (including its T+1 lock).
  function alertFill(event,bars,now=Date.now()){
    const rule=event.execution;
    if(rule?.status!=="filled"||!event.observed_at||!bars.length)return null;
    const earliest=event.observed_at>rule.fill_time?event.observed_at:rule.fill_time;
    // If the observation has rolled out of history, the first eligible open is unknown.
    if(bars[0].time>earliest)return null;
    const limit=event.code.startsWith("30")?.195:.095;
    let day="",previousClose=null,lastClose=null;
    for(const b of bars){
      if(b.synthetic||b.provisional||C.asEpoch(b.time)>now-2000)continue;
      const current=date(b);
      if(current!==day){previousClose=lastClose;day=current;}
      lastClose=b.close;
      const start=C.barStart(b.time,"m5");
      if(start<earliest||b.volume<=0)continue;
      const gap=previousClose?b.open/previousClose-1:0;
      if(event.side==="buy"?gap>=limit:gap<=-limit)continue;
      return {status:"filled",fill_time:start,fill_bar_time:b.time,entry_open:b.open};
    }
    return null;
  }
  function evaluate(order,bars,options={}){return evaluator(bars,options)(order);}
  function observations(report,options={}){
    const score=evaluator(report.bars,options);
    return report.signals.map(order=>{
      const fill=report.bars[order.fill_index];
      const event={...order,entry_open:fill?.open};
      return {...event,...score(event)};
    });
  }
  function stats(events,horizon=3,{side="all",regime="all",partition="all",cutoff=""}={}){
    const relevant=events.filter(x=>(side==="all"||x.side===side)&&(regime==="all"||x.regime===regime));
    const inPartition=event=>{const day=(event.fill_bar_time||event.time).slice(0,10);return partition==="all"||(partition==="test"?day>=cutoff:day<cutoff);};
    const eligible=relevant.filter(inPartition),purged=eligible.filter(x=>partition==="train"&&x.outcomes?.[horizon]?.target_day>=cutoff);
    const rows=eligible.filter(x=>x.outcomes?.[horizon]&&!purged.includes(x));
    const wins=rows.filter(x=>x.outcomes[horizon].correct).length;
    return {total:eligible.length,n:rows.length,wins,pending:eligible.filter(x=>!x.outcomes?.[horizon]&&x.status!=="cancelled").length,cancelled:eligible.filter(x=>x.status==="cancelled").length,purged:purged.length,accuracy:rows.length?wins/rows.length*100:null,average:mean(rows.map(x=>x.outcomes[horizon].direction_return_pct)),ci:C.wilson(wins,rows.length),rows};
  }
  function baseline(report,events,{mode="m5",calendar=[],horizon=3,threshold=1,now=Date.now(),...costs}={}){
    if(!events.length)return {accuracy:null,average:null,n:0};
    const wanted=new Set(events.map(event=>`${event.fill_bar_time.slice(0,10)}_${event.side}`)),groups=new Map(),score=evaluator(report.bars,{mode,calendar,threshold,now,...costs});
    const previousClose=new Map();let priorDay=null,lastClose=null;
    for(const b of report.bars){const day=date(b);if(priorDay!==day&&priorDay!==null)previousClose.set(day,lastClose);priorDay=day;lastClose=b.close;}
    for(let i=61;i<report.bars.length;i++){
      const b=report.bars[i],day=date(b);if(b.synthetic||b.provisional||b.volume<=0||C.asEpoch(b.time)>now-2000)continue;
      for(const side of ["buy","sell"]){const key=`${day}_${side}`;if(!wanted.has(key))continue;
        const prior=previousClose.get(day),limit=String(costs.code||"").startsWith("30")?.195:.095;
        if(prior&&((side==="buy"&&b.open/prior-1>=limit)||(side==="sell"&&b.open/prior-1<=-limit)))continue;
        const value=score({side,status:"filled",entry_open:b.open,fill_bar_time:b.time}).outcomes[horizon];
        if(value){if(!groups.has(key))groups.set(key,[]);groups.get(key).push(value);}
      }
    }
    const means=events.map(event=>groups.get(`${event.fill_bar_time.slice(0,10)}_${event.side}`)).filter(rows=>rows?.length).map(rows=>({accuracy:mean(rows.map(x=>x.correct?100:0)),average:mean(rows.map(x=>x.direction_return_pct))}));
    return {accuracy:mean(means.map(x=>x.accuracy)),average:mean(means.map(x=>x.average)),n:means.length};
  }

  root.SignalLab={STRATEGY_IDS,VERSION,HORIZONS,VARIANTS,alertFill,exceeds,dailyContext,trendAt,auditor,analyze,evaluate,observations,stats,baseline,returnAt,sessionDays};
})(globalThis);
