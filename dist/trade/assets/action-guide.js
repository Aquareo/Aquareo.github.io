/* Current conditional reference; historical markers are never current orders. */
"use strict";
(function(root){
  const C=root.TradingCore;
  function resolve({report,quote,position=null,t=null,strategy="donchian",healthy=false,paused=false,now=Date.now()}={}){
    const make=(kind,title,reason,side=null)=>({kind,title,reason,side,time:report?.last_confirmed||null});
    if(paused)return make("data","检测已暂停","恢复前台刷新后再检查当前买卖条件。");
    if(!report)return make("data","数据未就绪","等待分钟线加载后检查条件。");
    const market=C.market(now);
    if(!market.open||market.time>="15:00:00"||(market.time>="11:30:00"&&market.time<"13:00:00"))return make("wait","休市 · 等待开盘","历史 B/S 只供复盘，当前不作为买卖提示。");
    if(!healthy||!quote?.price||!quote?.as_of)return make("data","数据未就绪","行情或分钟线异常时保留历史，不给当前方向提示。");
    const index=report.bars.findIndex(b=>b.time===report.last_confirmed),bar=report.bars[index];
    if(!bar||bar.time.slice(0,10)!==market.day||now-C.asEpoch(bar.time)>420000||bar.time<C.stamp(Math.floor((now-2000)/300000)*300000)||quote.as_of<bar.time)return make("data","等待最新确认 K 线","报价、交易日期与完整 5 分钟 K 线需要对齐。");
    if(bar.volume===0)return make("data","等待有效成交量","当前 K 线无成交量，先核对交易状态与上游数据。");
    const [buy,sell]=C.conditions(bar,report.indicators[index],strategy),held=position&&position.buy_date<=market.day,locked=held&&position.buy_date===market.day;
    if(locked&&sell)return make("risk","风险预警 · 当日仓不可卖","退出条件出现，但录入的是当日新买持仓；按 T+1 区分。");
    if(t?.account?.outstanding){
      if(t.status==="buyback"&&sell&&t.account.available>0)return make("wait","信号分歧 · 等待核对","做 T 满足买回观察，但趋势退出条件也在，先核对剩余持仓计划。");
      if(t.status==="buyback")return make("buy","买回观察 · 已卖底仓",t.reason,"buy");
      return make("wait","等待买回条件",t.reason||"本日已有登记卖出，尚未确认买回条件；注意踏空风险。");
    }
    if(held&&sell){
      if(t?.account?.available>0)return make("sell","卖出观察 · 已有底仓","已收线触发当前趋势退出规则，且录入的今日可卖底仓仍有余量。","sell");
      return make("risk","退出条件出现 · 核对底仓","先核对并保存今日可卖股数，不将未知数量当作可执行卖出。");
    }
    if(t?.status==="sell_watch")return make("sell","做 T 减仓观察",t.reason,"sell");
    const freshBuy=report.pending?.side==="buy"&&report.pending.status==="pending"&&report.pending.time===bar.time&&buy;
    if(freshBuy){
      if(held)return make("wait","已有持仓 · 暂不加仓","趋势买入条件出现；当前模块不自动建议对已有仓位加仓。");
      if(Math.abs(quote.price/bar.close-1)>.01+1e-9)return make("wait","现价偏离 · 等待新确认","现价偏离信号收盘价超过 1%，保留信号但不提示追价买入。");
      return make("buy","买入观察 · 条件已确认",report.pending.reason,"buy");
    }
    if(buy&&report.quality==="filtered"&&!report.audit?.pass){const failed=report.audit?.checks.filter(x=>!x.pass).map(x=>x.label)||[];return make("wait","等待 · 买入过滤未通过",failed.length?failed.join("；"):"候选条件尚未满足全部确认门槛。");}
    if(sell&&!held)return make("wait","等待 · 未录入可卖持仓","退出条件存在；如有底仓请先录入，不将卖出提示当作卖空。");
    if(report.status==="holding")return make("wait",held?"持仓观察 · 退出条件未触发":"等待 · 已进入模拟持有阶段",held?"已有实际持仓，当前规则尚未触发退出条件。":"策略已进入模拟持有阶段，当前没有新的首次入场确认，不复用历史买点。");
    return make("wait","等待 · 暂无新买卖条件",report.preview?"当前 K 线条件仍在形成，等收线后确认。":"没有新的当前触发，历史 B/S 不作为此刻操作指令。");
  }
  root.ActionGuide={resolve};
})(globalThis);
