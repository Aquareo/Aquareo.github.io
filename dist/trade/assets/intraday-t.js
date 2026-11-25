/* Research-only sell-first / buy-back watch for manually recorded settled shares. */
"use strict";
(function(root){
  const C=root.TradingCore;
  function fee(notional,{commission=.0003,tax=.0005}={},sell=false){return Math.max(5,notional*commission)+(sell?notional*tax:0);}
  function ledger(position,trades,{now=Date.now(),costs={}}={}){
    const day=C.market(now).day,rows=trades.filter(t=>t.day===day),available=position?.buy_date<day&&position?.available_date===day?Math.max(0,Number(position.available_qty)||0):0;
    let sold=0,bought=0;const lots=[];
    for(const t of rows){if(t.side==="sell"){sold+=t.qty;lots.push({qty:t.qty,price:t.price,net:t.price*t.qty-fee(t.price*t.qty,costs,true)});}else{bought+=t.qty;let left=t.qty;for(const lot of lots){const used=Math.min(left,lot.qty),fraction=lot.qty?used/lot.qty:0;lot.net*=1-fraction;lot.qty-=used;left-=used;if(!left)break;}}}
    const outstanding=lots.reduce((s,l)=>s+l.qty,0),cash=lots.reduce((s,l)=>s+l.net,0),gross=lots.reduce((s,l)=>s+l.qty*l.price,0);
    return {available:Math.max(0,available-sold),sold,bought,outstanding,cash,average:outstanding?gross/outstanding:null};
  }
  function validateTrade(position,trades,trade,options={}){
    if(!["sell","buy"].includes(trade.side)||!Number.isFinite(trade.price)||trade.price<=0||!Number.isInteger(trade.qty)||trade.qty<=0||trade.qty%100!==0)throw Error("请输入有效成交价与 100 股整数倍数量");
    const account=ledger(position,trades,options);
    if(trade.side==="sell"&&trade.qty>account.available)throw Error("超过录入的剩余可卖底仓；当日买回不会增加当日可卖数量");
    if(trade.side==="buy"&&trade.qty>account.outstanding)throw Error("买回数量不能超过本日已登记卖出且尚未买回的数量");
    return true;
  }
  function buyTarget(account,{commission=.0003,slippage=.001}={},edge=.003){
    if(!account.outstanding)return null;
    const budget=account.cash-account.average*account.outstanding*edge;
    if(budget<=5)return null;
    let lo=0,hi=account.average;
    for(let i=0;i<45;i++){const mid=(lo+hi)/2,notional=mid*(1+slippage)*account.outstanding,total=notional+Math.max(5,notional*commission);if(total<=budget)lo=mid;else hi=mid;}
    return Math.floor(lo*100)/100;
  }
  function analyze(bars,{position=null,trades=[],now=Date.now(),costs={},healthy=false}={}){
    const day=C.market(now).day,session=bars.filter(b=>b.time.slice(0,10)===day&&!b.synthetic&&!b.provisional&&C.asEpoch(b.time)<=now-2000),account=ledger(position,trades,{now,costs}),target=buyTarget(account,costs);
    const base={account,target,day,status:"watch",reason:"等待盘中条件",last:null,vwap:null};
    const market=C.market(now),sessionOpen=market.open&&market.time<"15:00:00"&&!(market.time>="11:30:00"&&market.time<"13:00:00");
    if(!healthy||!sessionOpen)return {...base,status:"unavailable",reason:sessionOpen?"实时检测暂停或行情未就绪，不发当前提示":"非交易时段，仅显示登记状态与买回参考"};
    if(session.length<10)return {...base,status:"warming",reason:"等待至少 10 根当日已收线的 5 分钟 K 线"};
    if(now-C.asEpoch(session.at(-1).time)>420000||session.at(-1).time<C.stamp(Math.floor((now-2000)/300000)*300000))return {...base,status:"unavailable",reason:"当日分钟线滞后，不发当前提示"};
    const ind=C.shortAverages(session),i=session.length-1,b=session[i],prev=session[i-1],volume=session.reduce((s,x)=>s+x.volume,0),vwap=volume>0?session.reduce((s,x)=>s+(x.high+x.low+x.close)/3*x.volume,0)/volume:null;
    if(b.volume<=0)return {...base,status:"unavailable",reason:"当前 K 线无成交量，暂停做 T 提示"};
    const above=session.slice(-4,-1).some(x=>vwap&&x.high>=vwap*1.004),turnDown=prev.close>=ind[i-1].ma5&&b.close<ind[i].ma5,turnUp=prev.close<=ind[i-1].ma5&&b.close>ind[i].ma5;
    Object.assign(base,{last:b.time,vwap,price:b.close});
    if(account.outstanding){return {...base,status:target&&b.close<=target&&turnUp?"buyback":"wait_buyback",reason:target&&b.close<=target&&turnUp?"价格低于含成本买回参考，且收盘重新上穿 MA5":"已卖未买回；等待目标价与止跌条件，同时关注踏空风险"};}
    if(!position||!Number(position.available_qty)||position.available_date!==day)return {...base,status:"setup",reason:"需先核对并保存今日可卖底仓；不自动复用昨日数量"};
    if(!account.available)return {...base,status:"locked",reason:position.buy_date>=day?"当日新买持仓受 T+1 限制，不提示可执行卖出":"本日录入的可卖底仓已用完；当日买回不可再次卖出"};
    return {...base,status:above&&turnDown?"sell_watch":"watch",reason:above&&turnDown?"近期高点偏离当日均价至少 0.4%，随后收盘跌回 MA5 下方":"等待价格上冲后转弱；不把所有趋势 S 点当作做 T 卖点"};
  }
  root.IntradayT={fee,ledger,validateTrade,buyTarget,analyze};
})(globalThis);
