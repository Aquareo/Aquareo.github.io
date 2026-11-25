/* Eastmoney daily order-size net flows, in CNY. Separate from trading signals. */
"use strict";
(function(root){
  const C=root.TradingCore;
  function parse(payload,stock,{now=Date.now(),fetchedAt=new Date(now).toISOString()}={}){
    const data=payload?.data;if(payload?.rc!==0||data?.code!==stock.code||!Array.isArray(data.klines)||!data.klines.length)throw Error("未返回该股票的资金流向");
    const numeric=value=>value!==""&&value!=="-"&&Number.isFinite(Number(value))?Number(value):null;
    const today=C.market(now).day,seen=new Set();
    const rows=data.klines.map(line=>{
      const f=String(line).split(","),day=f[0];if(!/^\d{4}-\d{2}-\d{2}$/.test(day)||!Number.isFinite(C.asEpoch(day))||f.length<13||seen.has(day))throw Error("资金日期或字段格式异常");seen.add(day);
      const row={date:day,main:numeric(f[1]),small:numeric(f[2]),medium:numeric(f[3]),large:numeric(f[4]),super:numeric(f[5]),main_pct:numeric(f[6]),small_pct:numeric(f[7]),medium_pct:numeric(f[8]),large_pct:numeric(f[9]),super_pct:numeric(f[10]),close:numeric(f[11]),change_pct:numeric(f[12])};
      if([row.main,row.small,row.medium,row.large,row.super].some(x=>x===null))throw Error("资金金额缺失，不能按零处理");
      if(Math.abs(row.main-row.large-row.super)>Math.max(100,Math.abs(row.main)*.00001))throw Error("主力与大小单统计口径不一致");
      return row;
    }).filter(row=>row.date<=today).sort((a,b)=>a.date.localeCompare(b.date)).slice(-25);
    if(!rows.length)throw Error("没有可用的资金流向日期");
    return {code:stock.code,name:data.name||stock.name,source:"东方财富",fetched_at:fetchedAt,rows,latest:rows.at(-1)};
  }
  async function load(stock){
    const secid=(stock.exchange==="SH"?"1":"0")+"."+stock.code;
    const route=`/api/qt/stock/fflow/daykline/get?lmt=25&klt=101&secid=${secid}&fields1=f1,f2,f3,f7&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63`;
    let failure;
    for(const host of ["https://push2his.eastmoney.com","https://push2.eastmoney.com"]){
      try{const result=parse(JSON.parse(await root.LiveFeed.fetchText(host+route+`&r=${Math.floor(Date.now()/60000)}`,false,{timeoutMs:8000})),stock);if(result.rows.length<2)throw Error("资金历史暂不足，保留原数据");return result;}catch(error){failure=error;}
    }
    throw failure||Error("资金接口暂不可用");
  }
  function totals(flow,now=Date.now()){
    const completed=flow.rows.filter(row=>C.asEpoch(row.date)<=now-2000);
    return Object.fromEntries([5,20].map(n=>[n,completed.length>=n?completed.slice(-n).reduce((sum,row)=>sum+row.main,0):null]));
  }
  root.FundFlow={parse,load,totals};
})(globalThis);
