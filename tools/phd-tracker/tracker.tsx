'use client';
import PlanBoard from './plan-board';
import { useRef, useState, useEffect } from 'react';
import { type TrackerState, type Application, type TimelineEvent, type Task, stateSchema, statusNames, eventNames, includedSchoolPlan } from './schema';

type View = 'overview'|'applications'|'timeline'|'rounds';
type Modal = {kind:'application'|'event'|'task';id?:string};
const viewNames:Record<View,string>={overview:'2027 申请总览',applications:'学校与进展',timeline:'申请时间线',rounds:'轮次与 DDL'};
function today(){return new Intl.DateTimeFormat('en-CA',{timeZone:'Asia/Shanghai',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date());}
function dateLabel(date:string){return date?date.replaceAll('-','.'): '未定日期';}

function schoolText(app?:Application){return app?[app.schoolEn||app.school,app.schoolZh!==(app.schoolEn||app.school)?app.schoolZh:''].filter(Boolean).join(' · '):'';}
function schoolSearchText(app:Application){return [app.school,app.schoolEn,app.schoolZh].filter(Boolean).join(' ');}
export function SchoolName({app,onOpen}:{app:Application;onOpen?:()=>void}){
 const english=app.schoolEn||app.school,zh=app.schoolZh;
 const url='https://cn.bing.com/search?q='+encodeURIComponent([zh,english,'百科'].filter(Boolean).join(' '));
 const chinese=zh?<a className="school-zh" href={url} target="_blank" rel="noopener noreferrer" title={'在必应搜索'+zh+'的百科介绍'} aria-label={zh+'：必应百科搜索（新窗口）'}>{zh} <small aria-hidden="true">↗</small></a>:null;
 return <span className="school-label">{english===zh?chinese:onOpen?<button className="text-button school-name" onClick={onOpen}>{english}</button>:<span className="school-name">{english}</span>}{zh&&english!==zh&&<span className="school-translation">（{chinese}）</span>}</span>;
}
function MailLink({source}:{source:string}){return source?<a className="mail-link" href={source} target="_blank" rel="noopener noreferrer">查看原邮件</a>:null;}
function Badge({status}:{status:Application['status']}){return <span className={`badge ${status}`}>{statusNames[status]}</span>}
function Icon({name}:{name:View}){const paths={overview:'M3 3h7v7H3zM14 3h7v7h-7zM3 14h7v7H3zM14 14h7v7h-7z',applications:'M4 5h16v15H4zM8 5V3h8v2M8 10h8M8 14h8',timeline:'M6 3v18M6 6h14M6 12h10M6 18h14',rounds:'M4 5h16v16H4zM8 2v6M16 2v6M4 10h16M8 14h2M14 14h2',tasks:'m3 6 2 2 4-4M12 6h9m-18 7 2 2 4-4m3 2h9m-18 7 2 2 4-4m3 2h9'};return <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name]}/></svg>}
export default function Tracker({initial,initialRevision,initialSaved,displayName,onSave,onExport,onImport,onLock}:{initial:TrackerState;initialRevision:number;initialSaved:string;displayName:string;onSave:(state:TrackerState)=>Promise<string>;onExport:(state:TrackerState)=>Promise<void>;onImport:(file:File)=>Promise<TrackerState>;onLock:()=>void}) {
 const [state,setState]=useState(initial),[view,setView]=useState<View>('overview'),[query,setQuery]=useState(''),[status,setStatus]=useState('all'),[scope,setScope]=useState('all'),[busy,setBusy]=useState(false),[error,setError]=useState(''),[saved,setSaved]=useState(initialSaved),[modal,setModal]=useState<Modal|null>(null),[showDone,setShowDone]=useState(false),[notice,setNotice]=useState('');
 const revision=useRef(initialRevision),importRef=useRef<HTMLInputElement>(null),dialogRef=useRef<HTMLDialogElement>(null);
 const [date,setDate]=useState(initial.importedAt);useEffect(()=>{setDate(today());},[]);
 useEffect(()=>{if(modal&&!dialogRef.current?.open)dialogRef.current?.showModal();},[modal]);
 const apps=state.applications;
 const appById=(id:string)=>apps.find(a=>a.id===id);
 const active=apps.filter(a=>a.category==='application'&&a.status!=='draft');
 const drafts=apps.filter(a=>a.status==='draft');
 const advice=apps.filter(a=>a.category==='advice');
 const matches=(value:string)=>value.toLowerCase().includes(query.trim().toLowerCase());
 const filteredApps=active.filter(a=>matches([schoolSearchText(a),a.person,a.topic,a.notes].join(' '))&&(status==='all'||a.status===status)).sort((a,b)=>b.latest.localeCompare(a.latest));
 const events=state.events.slice().reverse().filter(e=>e.origin!=='sync'&&(scope==='all'?appById(e.appId)?.category!=='advice':e.appId===scope)&&matches([e.title,e.detail,appById(e.appId)?.person,schoolText(appById(e.appId))].join(' '))).sort((a,b)=>b.date.localeCompare(a.date));
 function showTimeline(id:string){setScope(id);setQuery('');setView('timeline');}
 function latestEvent(id:string){return state.events.slice().reverse().filter(e=>e.appId===id&&e.origin!=='sync').sort((a,b)=>b.date.localeCompare(a.date))[0];}
 async function persist(next:TrackerState){
  setBusy(true);setError('');setNotice('');
  try{
   const parsed=stateSchema.safeParse(next);if(!parsed.success)throw new Error('记录格式有误：请检查日期、字段长度和邮件链接。');
   const updatedAt=await onSave(parsed.data);setState(parsed.data);setSaved(updatedAt);return true;
  }catch(e){setError(e instanceof Error?e.message:'保存失败，请重试。');return false;}finally{setBusy(false);}
 }
 function navigate(next:View){setView(next);setQuery('');setScope('all');setStatus('all');}
 async function submit(event:React.FormEvent<HTMLFormElement>){
  event.preventDefault();if(!modal)return;
  const data=new FormData(event.currentTarget);const get=(key:string)=>String(data.get(key)||'').trim();const id=modal.id||crypto.randomUUID();
  let next:TrackerState;
  if(modal.kind==='application'){
   const old=apps.find(a=>a.id===id);
   const item:Application={...old,id,school:get('school'),schoolZh:get('schoolZh'),person:get('person'),topic:get('topic'),status:get('status') as Application['status'],category:get('category') as Application['category'],latest:get('latest'),deadline:get('deadline'),next:get('next'),notes:get('notes'),source:get('source')};
   next={...state,applications:old?apps.map(a=>a.id===id?item:a):[...apps,item]};
  }else if(modal.kind==='event'){
   const item:TimelineEvent={id,appId:get('appId'),date:get('date'),title:get('title'),detail:get('detail'),type:get('type') as TimelineEvent['type'],source:get('source'),origin:'manual'};
   next={...state,events:modal.id?state.events.map(e=>e.id===id?item:e):[...state.events,item]};
  }else{
   const item:Task={id,appId:get('appId'),title:get('title'),due:get('due'),detail:get('detail'),done:data.get('done')==='on',source:get('source')};
   next={...state,tasks:modal.id?state.tasks.map(t=>t.id===id?item:t):[...state.tasks,item]};
  }
  if(await persist(next)){setModal(null);setNotice('已加密保存到此浏览器。');}
 }
 async function remove(){
  if(!modal?.id||!confirm('确定删除这条记录？你可以先导出加密备份。'))return;
  const id=modal.id;
  const next=modal.kind==='application'?{...state,applications:apps.filter(a=>a.id!==id),events:state.events.map(e=>e.appId===id?{...e,appId:''}:e),tasks:state.tasks.map(t=>t.appId===id?{...t,appId:''}:t),schoolPlans:state.schoolPlans.map(p=>({...p,contactIds:p.contactIds.filter(x=>x!==id)}))}:modal.kind==='event'?{...state,events:state.events.filter(e=>e.id!==id)}:{...state,tasks:state.tasks.filter(t=>t.id!==id)};
  if(await persist(next))setModal(null);
 }
 async function importFile(file:File){
  try{
   if(file.size>5_000_000)throw new Error('文件过大，请使用此页面导出的 加密备份。');
   const parsed=stateSchema.safeParse(await onImport(file));if(!parsed.success)throw new Error('不是有效的申请记录备份。');
   if(confirm('导入会覆盖此浏览器的个人修改，不会改动每日邮件汇总。建议先导出加密备份。确定继续？')){if(await persist(parsed.data))setNotice('加密备份已导入此浏览器。');}
  }catch(e){setError(e instanceof Error?e.message:'无法读取备份。');}
 }
 function AppTable({items}:{items:Application[]}){return items.length?<div className="table-scroll school-table"><table><thead><tr><th>学校</th><th>导师</th><th>当前状态</th><th>最近进展</th><th>申请轮次 / 时间</th><th aria-label="操作"></th></tr></thead><tbody>{items.map(a=>{const latest=latestEvent(a.id);return <tr key={a.id}><td><SchoolName app={a} onOpen={()=>showTimeline(a.id)}/></td><td data-label="导师">{a.person}</td><td data-label="状态"><Badge status={a.status}/></td><td className="school-progress" data-label="最近进展"><time>{dateLabel(latest?.date||a.latest)}</time><p>{latest?.title||(a.status==='draft'?'联系邮件尚未发送':'暂无进展记录')}</p></td><td className="round-preview" data-label="申请时间">{a.rounds?.length?<><strong>{(a.rounds.find(r=>r.primary)||a.rounds[0]).name}</strong><span>开放：{(a.rounds.find(r=>r.primary)||a.rounds[0]).opens||'官方未注明'}</span><span>DDL：{(a.rounds.find(r=>r.primary)||a.rounds[0]).closes||'待具体公告'}</span>{a.rounds.filter(r=>r.kind==='funding'&&!r.primary).map(r=><span className="funding-date" key={r.name}>资助 DDL：{r.closes||'待确认'}</span>)}<button className="text-button" onClick={()=>{setScope(state.schoolPlans.find(p=>p.contactIds.includes(a.id))?.id||a.id);setQuery('');setView('rounds');}}>全部轮次与来源</button></>:<span className="muted">{a.status==='draft'?'未核对（草稿）':'待核对'}</span>}</td><td className="school-actions"><button className="text-button" onClick={()=>showTimeline(a.id)}>时间线</button><button className="text-button muted" onClick={()=>setModal({kind:'application',id:a.id})}>详情</button></td></tr>})}</tbody></table></div>:<Empty text="没有符合条件的学校。"/>}
 function Timeline({items}:{items:TimelineEvent[]}){return items.length?<ol className="timeline">{items.map((e,i)=><li key={e.id}><div className="timeline-date">{i===0||items[i-1].date!==e.date?dateLabel(e.date):''}</div><div className="timeline-body"><div className="eyebrow">{appById(e.appId)?<><SchoolName app={appById(e.appId)!}/> · {appById(e.appId)!.person}</>:'个人记录'} <span>· {eventNames[e.type]}</span></div><h3>{e.title}</h3><p>{e.detail}</p><div className="event-bottom"><span className="muted">{e.origin==='email'?'来自邮件':e.origin==='sync'?'每日检查':'手动记录'}</span><MailLink source={e.source}/><button className="text-button muted" onClick={()=>setModal({kind:'event',id:e.id})}>编辑</button></div></div></li>)}</ol>:<Empty text="还没有这部分记录，可以添加一条进展。"/>}
 function Rounds(){
  const planned=state.schoolPlans.filter(p=>includedSchoolPlan(p,apps)).map(p=>({...p,person:'',rounds:p.rounds.length?p.rounds:p.contactIds.flatMap(id=>appById(id)?.rounds||[])}));
  const covered=new Set(state.schoolPlans.flatMap(p=>p.contactIds));
  const selected=[...planned,...active.filter(a=>!covered.has(a.id))].filter(a=>(scope==='all'||a.id===scope)&&matches(schoolSearchText(a as Application)));
  const verified=selected.flatMap(a=>a.rounds||[]).map(r=>r.verifiedAt).sort().at(-1);
  function phase(r:NonNullable<Application['rounds']>[number]){if(r.closes&&r.closes<date)return '已截止';if(r.opens&&r.opens>date)return '未开放';if(r.opens&&r.closes||r.knownOpen)return '开放中';return r.closes?'日期已公布':'待具体公告';}
  return <section className="rounds-section"><p className="result-count">包括未联系的计划学校 · 最近核对：{verified?dateLabel(verified):'未核对'} · 入学申请与资助申请分别列出</p><div className="table-scroll rounds-table"><table><thead><tr><th>学校 / 轮次</th><th>开放时间</th><th>截止日期 DDL</th><th>说明与官方来源</th></tr></thead><tbody>{selected.flatMap(a=>(a.rounds||[]).map(r=><tr key={a.id+r.name}><td><SchoolName app={a as Application}/><p>{r.name}</p><span className="badge">{phase(r)}</span></td><td data-label="开放时间"><time>{r.opens?dateLabel(r.opens):'官方未注明'}</time>{!r.opens&&r.knownOpen&&<small>已确认开放，起始日未核实</small>}</td><td data-label="截止日期"><strong>{r.closes?dateLabel(r.closes):'依具体岗位／资助通知'}</strong>{r.deadlineTime&&<small>{r.deadlineTime}</small>}<small>{r.timezone}</small></td><td className="round-note"><p>{r.note}</p><a className="mail-link" href={r.source} target="_blank" rel="noopener noreferrer">官方来源</a><small>核对：{dateLabel(r.verifiedAt)}</small></td></tr>))}</tbody></table></div>{selected.some(a=>!a.rounds?.length)&&<details className="plan-dates"><summary>待核实轮次的学校（{selected.filter(a=>!a.rounds?.length).length}）</summary>{selected.filter(a=>!a.rounds?.length).map(a=><p key={a.id}><SchoolName app={a as Application}/> · 开放时间与 DDL 待核实</p>)}</details>}</section>;
 }
 const editedApp=modal?.kind==='application'?apps.find(a=>a.id===modal.id):undefined;
 const editedEvent=modal?.kind==='event'?state.events.find(e=>e.id===modal.id):undefined;
 const editedTask=modal?.kind==='task'?state.tasks.find(t=>t.id===modal.id):undefined;
 return <div className="shell">
  <a className="skip-link" href="#main">跳转到正文</a>
  <aside className="sidebar"><div className="identity"><span className="monogram">YL</span><div><strong>刘宇</strong><small>博士申请记录</small></div></div><nav aria-label="主导航">{(Object.keys(viewNames) as View[]).map(v=><button key={v} aria-current={view===v?'page':undefined} onClick={()=>navigate(v)}><Icon name={v}/>{viewNames[v]}</button>)}</nav><div className="sidebar-caption"><span>2027 入学季</span><p>数值分析 · 随机分析<br/>科学计算</p></div><div className="sidebar-bottom"><span className="private-label"><svg width="14" height="14" viewBox="0 0 20 20" fill="none" stroke="currentColor" aria-hidden="true"><rect x="4" y="9" width="12" height="9" rx="2"/><path d="M6 9V6a4 4 0 0 1 8 0v3"/></svg>密码保护</span><button className="text-button" onClick={onLock}>锁定</button></div></aside>
  <main id="main" className="content"><header className="page-header"><div><p className="kicker">FALL 2027</p><h1>{viewNames[view]}</h1><p className="lead">{view==='overview'?'学校规划、导师联系与正式网申分别记录。':view==='applications'?'按学校查看联系与申请进度。':view==='rounds'?'分开查看入学申请与奖学金轮次，所有日期附官方来源。':'按日期查看每一次联系、回复和交流安排。'}</p></div><button className="primary" hidden={view==='overview'} disabled={busy} onClick={()=>setModal({kind:view==='timeline'?'event':'application'})}>＋ {view==='timeline'?'记录进展':'添加学校'}</button></header>
  <p className="storage-notice">邮件汇总自动更新；手动修改仅加密保存在当前浏览器，尚未上传云端。换设备前请导出加密备份。</p><div className="save-bar"><span>{busy?'正在保存…':saved?'本机已保存 · '+new Intl.DateTimeFormat('zh-CN',{timeZone:'Asia/Shanghai',month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit'}).format(new Date(saved)):'邮件整理于 '+dateLabel(state.importedAt)}</span><div><button className="text-button" onClick={()=>onExport(state)}>导出加密备份</button><button className="text-button" disabled={busy} onClick={()=>importRef.current?.click()}>导入加密备份</button><input ref={importRef} type="file" accept=".json,application/json" hidden onChange={e=>{const f=e.target.files?.[0];if(f)importFile(f);e.target.value='';}}/></div></div>
  {error&&<div role="alert" className="error-banner">{error}<button className="text-button" onClick={()=>onExport(state)}>导出当前记录</button></div>}{notice&&<p role="status" className="notice">{notice}</p>}
  {state.sync&&<details className="sync-line"><summary><span>邮件检查：{new Intl.DateTimeFormat('zh-CN',{timeZone:'Asia/Shanghai',month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit'}).format(new Date(state.sync.checkedAt))}</span><span>{state.sync.status==='ok'?`${state.sync.newMessages} 封新邮件`:'本次检查未完成'}</span><span className="muted">每天 08:00 更新 · 查看摘要</span></summary><ul>{state.sync.summary.map((item,i)=><li key={i}>{item}</li>)}</ul></details>}
  <div className="filters"><label className="search"><span className="sr-only">搜索学校或记录</span><input value={query} onChange={e=>setQuery(e.target.value)} placeholder={view==='timeline'?'搜索时间线':'搜索学校或导师'}/></label>{view==='overview'?null:view==='applications'?<label><span className="sr-only">按状态筛选</span><select value={status} onChange={e=>setStatus(e.target.value)}><option value="all">全部状态</option>{Object.entries(statusNames).filter(([key])=>key!=='draft').map(([key,name])=><option key={key} value={key}>{name}</option>)}</select></label>:<label><span className="sr-only">按学校筛选</span><select value={scope} onChange={e=>setScope(e.target.value)}><option value="all">全部学校</option>{(view==='rounds'?[...state.schoolPlans.filter(p=>includedSchoolPlan(p,apps)).map(p=>({...p,person:''})),...apps.filter(a=>a.status!=='draft'&&!state.schoolPlans.some(p=>p.contactIds.includes(a.id)))]:apps.filter(a=>a.status!=='draft')).map(a=><option key={a.id} value={a.id}>{schoolText(a as Application)} {a.person&&' · '+a.person}</option>)}</select></label>}</div>
  {view==='overview'?<PlanBoard state={state} query={query} busy={busy} onSave={persist} onTimeline={showTimeline}/>:view==='applications'?<>
   <p className="result-count">{active.length} 位已联系导师 · {active.filter(a=>['applied','offer'].includes(a.status)).length} 条记录已确认正式提交 <span>（联系导师不等同于提交申请）</span></p>
   <AppTable items={filteredApps}/>
   <details className="secondary-records"><summary>未发送草稿 <span>{drafts.length}</span></summary><p className="muted">以下尚未发送，不计入已联系学校。</p><AppTable items={drafts.filter(a=>matches(schoolSearchText(a)+' '+a.person))}/></details>
   {advice.length>0&&<details className="secondary-records"><summary>学业咨询 <span>{advice.length}</span></summary><AppTable items={advice}/></details>}
  </>:view==='rounds'?<Rounds/>:<><div className="timeline-heading"><h2>{scope==='all'?'全部学校':appById(scope)&&<SchoolName app={appById(scope)!}/>}</h2><span className="muted">{events.length} 条进展 · 最新在前</span></div><Timeline items={events}/></>}
  <footer><p>邮件整理截至 {dateLabel(state.importedAt)} · 通信时间为北京时间；招生时间以所列时区为准</p><details><summary>数据与同步说明</summary><p>{state.importNote}</p><p>邮件汇总由云端任务每天更新；页面不直接登录 Gmail，不会代发邮件。个人笔记加密保存在此浏览器，换设备请导出、导入加密备份。每日任务可能因连接或服务状态延迟，请核对上方实际检查时间。</p><p>已用页面密码解锁。申请状态与邮件措辞分开记录；鼓励申请不代表录取，交流提议不代表正式面试通知。</p></details></footer>
 </main>
 {modal&&<dialog ref={dialogRef} className="edit-dialog" onCancel={e=>{e.preventDefault();if(!busy)setModal(null);}} aria-labelledby="dialog-title"><form key={modal.kind+(modal.id||'new')} onSubmit={submit}><div className="dialog-heading"><h2 id="dialog-title">{modal.id?'编辑':'添加'}{modal.kind==='application'?'申请记录':'进展记录'}</h2><button type="button" className="close" aria-label="关闭" disabled={busy} onClick={()=>setModal(null)}>×</button></div><fieldset disabled={busy}>
 {modal.kind==='application'?<><div className="form-row"><label>导师 / 联系人（可留空）<input name="person" defaultValue={editedApp?.person} maxLength={300}/></label><label>学校<input name="school" defaultValue={editedApp?.school} required maxLength={300}/></label></div><label>学校中文名（可选，用于必应百科搜索）<input name="schoolZh" defaultValue={editedApp?.schoolZh} maxLength={300}/></label><label>研究方向<input name="topic" defaultValue={editedApp?.topic} maxLength={300}/></label><div className="form-row"><label>当前状态<select name="status" defaultValue={editedApp?.status||'draft'}>{Object.entries(statusNames).map(([key,name])=><option key={key} value={key}>{name}</option>)}</select></label><label>记录类别<select name="category" defaultValue={editedApp?.category||'application'}><option value="application">博士申请</option><option value="advice">学业咨询</option></select></label></div><div className="form-row"><label>最近记录日期<input name="latest" type="date" defaultValue={editedApp?.latest||date}/></label><label>已确认的截止日期<input name="deadline" type="date" defaultValue={editedApp?.deadline}/></label></div><input type="hidden" name="next" value={editedApp?.next||''}/><label>备注<textarea name="notes" defaultValue={editedApp?.notes} rows={4} maxLength={6000}/></label><label>来源邮件链接（可选）<input name="source" type="url" defaultValue={editedApp?.source} placeholder="https://mail.google.com/…"/></label></>:<><label>{modal.kind==='event'?'进展标题':'待办事项'}<input name="title" defaultValue={editedEvent?.title||editedTask?.title} required maxLength={300}/></label><label>关联申请<select name="appId" defaultValue={editedEvent?.appId||editedTask?.appId||''}><option value="">个人记录 / 申请准备</option>{apps.map(a=><option key={a.id} value={a.id}>{a.person} · {schoolText(a)}</option>)}</select></label>{modal.kind==='event'?<div className="form-row"><label>日期<input type="date" name="date" defaultValue={editedEvent?.date||date} required/></label><label>类型<select name="type" defaultValue={editedEvent?.type||'note'}>{Object.entries(eventNames).map(([key,name])=><option key={key} value={key}>{name}</option>)}</select></label></div>:<label>计划日期（可留空）<input type="date" name="due" defaultValue={editedTask?.due}/></label>}<label>内容<textarea name="detail" defaultValue={editedEvent?.detail||editedTask?.detail} rows={5} maxLength={modal.kind==='event'?6000:2000}/></label><label>来源邮件链接（可选）<input name="source" type="url" defaultValue={editedEvent?.source||editedTask?.source} placeholder="https://mail.google.com/…"/></label>{modal.kind==='task'&&<label className="checkbox-label"><input type="checkbox" name="done" defaultChecked={editedTask?.done}/>已完成</label>}{editedEvent?.origin==='email'&&<p className="muted">修改后将标为手动记录，原邮件链接会保留。</p>}</>}
 </fieldset><div className="dialog-actions">{modal.id&&<button type="button" className="danger text-button" disabled={busy} onClick={remove}>删除</button>}<button type="button" disabled={busy} onClick={()=>setModal(null)}>取消</button><button type="submit" className="primary" disabled={busy}>{busy?'保存中…':'保存记录'}</button></div>{error&&<p role="alert" className="form-error">{error}</p>}</form></dialog>}
 </div>;
}
function Stat({value,label}:{value:number;label:string}){return <div className="stat"><strong>{value}</strong><span>{label}</span></div>}
function Empty({text}:{text:string}){return <div className="empty">{text}</div>}
