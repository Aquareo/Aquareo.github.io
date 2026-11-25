"use strict";
(function(root){
  function feedError(code,message){return Object.assign(new Error(message),{code});}
  async function fetchText(url,gbk=false,{timeoutMs=10000}={}){
    const controller=new AbortController();let timedOut=false;
    const timeout=setTimeout(()=>{timedOut=true;controller.abort();},timeoutMs);
    try{
      const r=await fetch(url,{signal:controller.signal,cache:"no-store",credentials:"omit",referrerPolicy:"no-referrer"});
      if(!r.ok)throw feedError(r.status===429?"rate_limit":"http",r.status===429?"数据源暂时限流":`数据源返回 HTTP ${r.status}`);
      return gbk?new TextDecoder("gb18030").decode(await r.arrayBuffer()):await r.text();
    }catch(error){
      if(timedOut)throw feedError("timeout",`数据源响应超过 ${Math.ceil(timeoutMs/1000)} 秒`);
      if(typeof error.code==="string")throw error;
      if(error.name==="AbortError")throw feedError("aborted","数据请求已中断");
      throw feedError("network","无法连接数据源，可能是网络或跨域限制");
    }finally{clearTimeout(timeout);}
  }

  function timestamp(raw){return `${raw.slice(0,4)}-${raw.slice(4,6)}-${raw.slice(6,8)}T${raw.slice(8,10)}:${raw.slice(10,12)}:${raw.slice(12,14)||"00"}`;}
  const number=v=>v!==""&&Number.isFinite(Number(v))?Number(v):null;
  const quoteCache=new Map();
  function quoteFields(f){
    if(!Array.isArray(f)||f.length<50||!/^\d{14}$/.test(f[30]))return null;
    const q={name:f[1],code:f[2],price:number(f[3]),previous_close:number(f[4]),open:number(f[5]),volume:number(f[6]),as_of:timestamp(f[30]),change:number(f[31]),change_pct:number(f[32]),high:number(f[33]),low:number(f[34]),turnover_pct:number(f[38]),pe_ttm:number(f[39]),market_cap_yi:number(f[45]),pb:number(f[46]),upper_limit:number(f[47]),lower_limit:number(f[48]),source:"腾讯财经"};
    return q.price>0&&Number.isFinite(TradingCore.asEpoch(q.as_of))?q:null;
  }
  function parseQuotes(raw){const result={};for(const match of raw.matchAll(/v_([a-z]{2}\d{6})="([^"]*)";/g)){const q=quoteFields(match[2].split("~"));if(q)result[q.code]=q;}return result;}
  function cachedQuote(code){const saved=quoteCache.get(code),maxAge=TradingCore.market().open?45000:300000;return saved&&Date.now()-saved.receivedAt<maxAge?saved.quote:null;}

  async function quotes(stocks){
    const groups=[];for(let i=0;i<stocks.length;i+=14)groups.push(stocks.slice(i,i+14));
    const result={},errors=[];let cursor=0;
    async function batch(group){
      let remaining=group,lastError;
      for(const host of ["https://qt.gtimg.cn/"]){
        if(!remaining.length)break;
        try{
          const symbols=remaining.map(s=>s.exchange.toLowerCase()+s.code).join(",");
          const parsed=parseQuotes(await fetchText(`${host}q=${symbols}&r=${Date.now()}`,true,{timeoutMs:8000}));
          for(const stock of remaining){const q=parsed[stock.code];if(q&&Number.isFinite(q.price)&&q.price>0&&Number.isFinite(TradingCore.asEpoch(q.as_of))){result[stock.code]=q;}}
          remaining=remaining.filter(s=>!result[s.code]);
          if(remaining.length)lastError=feedError("incomplete","数据源返回部分股票行情");
        }catch(error){lastError=error;}
      }
      // The K-line feeds have their own working CORS hosts and include full
      // quotes. Reuse only recently received quotes with their original source time.
      for(const stock of remaining){const q=cachedQuote(stock.code);if(q)result[stock.code]=q;}
      remaining=remaining.filter(s=>!result[s.code]);
      if(remaining.length)errors.push({codes:remaining.map(s=>s.code),code:lastError?.code||"missing",message:lastError?.message||"行情缺失"});
    }
    async function worker(){while(cursor<groups.length)await batch(groups[cursor++]);}
    await Promise.all([worker(),worker()]);
    if(!Object.keys(result).length)throw feedError(errors[0]?.code||"missing",errors[0]?.message||"未返回可用行情");
    Object.defineProperty(result,"health",{value:{errors,received:Object.keys(result).length,requested:stocks.length},enumerable:false});
    return result;
  }

  async function bars(stock,mode="m5"){
    const symbol=stock.exchange.toLowerCase()+stock.code;
    const route=mode==="m5"?`appstock/app/kline/mkline?param=${symbol},m5,,640`:`appstock/app/fqkline/get?param=${symbol},day,,,640,qfq`;
    let failure;
    for(const host of ["https://proxy.finance.qq.com/ifzqgtimg/","https://web.ifzq.gtimg.cn/"]){
      try{
        const payload=JSON.parse(await fetchText(host+route+`&r=${Math.floor(Date.now()/15000)}`)),data=payload.data?.[symbol];
        const rows=mode==="m5"?data?.m5:data?.qfqday||data?.day;
        if(!rows?.length)throw Error("未返回 K 线");
        const quote=quoteFields(data?.qt?.[symbol]);
        if(quote)quoteCache.set(stock.code,{quote:{...quote,source:"腾讯财经 · K线接口报价"},receivedAt:Date.now()});
        const result=rows.map(r=>({time:mode==="m5"?timestamp(r[0]):r[0],open:Number(r[1]),close:Number(r[2]),high:Number(r[3]),low:Number(r[4]),volume:Number(r[5])}));
        TradingCore.validate(result);
        for(const bar of result)bar.provisional=TradingCore.asEpoch(bar.time)>Date.now()-2000;
        return {bars:result,quote:quote?{...quote,source:"腾讯财经 · K线接口报价"}:null,source:"腾讯财经",fetched_at:new Date().toISOString(),basis:mode==="m5"?"不复权分钟线":"前复权日线"};
      }catch(error){failure=error;}
    }
    if(mode==="day"){
      try{
        const secid=(stock.exchange==="SH"?"1":"0")+"."+stock.code;
        const url=`https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=${secid}&klt=101&fqt=1&beg=20200101&end=20500101&lmt=640&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56`;
        const payload=JSON.parse(await fetchText(url)),rows=payload.data?.klines;if(!rows?.length)throw Error("未返回日线");
        const result=rows.slice(-640).map(line=>{const r=line.split(",");return {time:r[0],open:Number(r[1]),close:Number(r[2]),high:Number(r[3]),low:Number(r[4]),volume:Number(r[5])};});
        TradingCore.validate(result);for(const bar of result)bar.provisional=TradingCore.asEpoch(bar.time)>Date.now()-2000;return {bars:result,source:"东方财富",fetched_at:new Date().toISOString(),basis:"前复权日线"};
      }catch(error){failure=error;}
    }
    throw failure||Error("行情源不可用");
  }
  async function news(code){
    const params={uid:"",keyword:code,type:["cmsArticleWebOld"],client:"web",clientType:"web",clientVersion:"curr",param:{cmsArticleWebOld:{searchScope:"default",sort:"latest",pageIndex:1,pageSize:30,preTag:"",postTag:""}}};
    const text=await fetchText("https://search-api-web.eastmoney.com/search/jsonp?cb=cb&param="+encodeURIComponent(JSON.stringify(params)));
    const payload=JSON.parse(text.slice(text.indexOf("(")+1,text.lastIndexOf(")"))),items=payload.result?.cmsArticleWebOld;
    if(payload.code!==0||!Array.isArray(items))throw Error("新闻接口结构异常");
    const seen=new Set(),clean=value=>String(value||"").replace(/<[^>]*>/g,"").trim();
    return items.map(item=>({title:clean(item.title),url:String(item.url||""),published_at:String(item.date||"").replace(" ","T")+"+08:00",source:clean(item.mediaName)||"东方财富",tag:"相关报道"})).filter(item=>{if(!item.title||!/^https?:\/\//.test(item.url)||seen.has(item.url)||!Number.isFinite(Date.parse(item.published_at))||Date.parse(item.published_at)>Date.now())return false;seen.add(item.url);return true;}).sort((a,b)=>b.published_at.localeCompare(a.published_at)).slice(0,10);
  }
  root.LiveFeed={quotes,bars,news,parseQuotes,timestamp,fetchText};
})(globalThis);
