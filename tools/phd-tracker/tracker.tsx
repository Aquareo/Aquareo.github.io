'use client';
import { useRef, useState, useEffect } from 'react';
import { type TrackerState, type Application, type TimelineEvent, type Task, stateSchema, statusNames, eventNames } from './schema';

type View = 'overview'|'applications'|'timeline'|'tasks';
type Modal = {kind:'application'|'event'|'task';id?:string};
const viewNames:Record<View,string>={overview:'总览',applications:'申请记录',timeline:'时间线',tasks:'待办事项'};
function today(){return new Intl.DateTimeFormat('en-CA',{timeZone:'Asia/Shanghai',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date());}
function dateLabel(date:string){return date?date.replaceAll('-','.'): '未定日期';}

function MailLink({source}:{source:string}){return source?<a className="mail-link" href={source} target="_blank" rel="noopener noreferrer">查看原邮件</a>:null;}
function Badge({status}:{status:Application['status']}){return <span className={`badge ${status}`}>{statusNames[status]}</span>}
function Icon({name}:{name:View}){const paths={overview:'M3 3h7v7H3zM14 3h7v7h-7zM3 14h7v7H3zM14 14h7v7h-7z',applications:'M4 5h16v15H4zM8 5V3h8v2M8 10h8M8 14h8',timeline:'M6 3v18M6 6h14M6 12h10M6 18h14',tasks:'m3 6 2 2 4-4M12 6h9m-18 7 2 2 4-4m3 2h9m-18 7 2 2 4-4m3 2h9'};return <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name]}/></svg>}
export default function Tracker({initial,initialRevision,initialSaved,displayName,onSave,onExport,onImport,onLock}:{initial:TrackerState;initialRevision:number;initialSaved:string;displayName:string;onSave:(state:TrackerState)=>Promise<string>;onExport:(state:TrackerState)=>Promise<void>;onImport:(file:File)=>Promise<TrackerState>;onLock:()=>void}) {
 const [state,setState]=useState(initial),[view,setView]=useState<View>('overview'),[query,setQuery]=useState(''),[status,setStatus]=useState('all'),[scope,setScope]=useState('all'),[busy,setBusy]=useState(false),[error,setError]=useState(''),[saved,setSaved]=useState(initialSaved),[modal,setModal]=useState<Modal|null>(null),[showDone,setShowDone]=useState(false),[notice,setNotice]=useState('');
 const revision=useRef(initialRevision),importRef=useRef<HTMLInputElement>(null),dialogRef=useRef<HTMLDialogElement>(null);
 const [date,setDate]=useState(initial.importedAt);useEffect(()=>{setDate(today());},[]);
 useEffect(()=>{if(modal&&!dialogRef.current?.open)dialogRef.current?.showModal();},[modal]);
 const apps=state.applications;
 const appById=(id:string)=>apps.find(a=>a.id===id);
 const active=apps.filter(a=>a.category==='application'&&a.status!=='draft');
 const drafts=apps.filter(a=>a.status==='draft');
 const openTasks=state.tasks.filter(t=>!t.done).sort((a,b)=>(a.due||'9999').localeCompare(b.due||'9999'));
 const matches=(value:string)=>value.toLowerCase().includes(query.trim().toLowerCase());
 const filteredApps=apps.filter(a=>matches([a.school,a.person,a.topic,a.next,a.notes].join(' '))&&(status==='all'?a.status!=='draft':status==='everything'||a.status===status));
 const events=state.events.filter(e=>(scope==='all'||e.appId===scope)&&matches([e.title,e.detail,appById(e.appId)?.person,appById(e.appId)?.school].join(' '))).sort((a,b)=>b.date.localeCompare(a.date));
 const tasks=state.tasks.filter(t=>(showDone||!t.done)&&(scope==='all'||t.appId===scope)&&matches([t.title,t.detail,appById(t.appId)?.person].join(' '))).sort((a,b)=>Number(a.done)-Number(b.done)||(a.due||'9999').localeCompare(b.due||'9999'));
 async function persist(next:TrackerState){
  setBusy(true);setError('');setNotice('');
  try{
   const parsed=stateSchema.safeParse(next);if(!parsed.success)throw new Error('记录格式有误：请检查日期、字段长度和邮件链接。');
   const updatedAt=await onSave(parsed.data);setState(parsed.data);setSaved(updatedAt);return true;
  }catch(e){setError(e instanceof Error?e.message:'保存失败，请重试。');return false;}finally{setBusy(false);}
 }
 function navigate(next:View){setView(next);setQuery('');setScope('all');setStatus('all');}
 async function toggleTask(task:Task){await persist({...state,tasks:state.tasks.map(t=>t.id===task.id?{...t,done:!t.done}:t)});}
 async function submit(event:React.FormEvent<HTMLFormElement>){
  event.preventDefault();if(!modal)return;
  const data=new FormData(event.currentTarget);const get=(key:string)=>String(data.get(key)||'').trim();const id=modal.id||crypto.randomUUID();
  let next:TrackerState;
  if(modal.kind==='application'){
   const old=apps.find(a=>a.id===id);
   const item:Application={id,school:get('school'),person:get('person'),topic:get('topic'),status:get('status') as Application['status'],category:get('category') as Application['category'],latest:get('latest'),deadline:get('deadline'),next:get('next'),notes:get('notes'),source:get('source')};
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
  const next=modal.kind==='application'?{...state,applications:apps.filter(a=>a.id!==id),events:state.events.map(e=>e.appId===id?{...e,appId:''}:e),tasks:state.tasks.map(t=>t.appId===id?{...t,appId:''}:t)}:modal.kind==='event'?{...state,events:state.events.filter(e=>e.id!==id)}:{...state,tasks:state.tasks.filter(t=>t.id!==id)};
  if(await persist(next))setModal(null);
 }
 async function importFile(file:File){
  try{
   if(file.size>5_000_000)throw new Error('文件过大，请使用此页面导出的 加密备份。');
   const parsed=stateSchema.safeParse(await onImport(file));if(!parsed.success)throw new Error('不是有效的申请记录备份。');
   if(confirm('导入会覆盖此浏览器的个人修改，不会改动每日邮件汇总。建议先导出加密备份。确定继续？')){if(await persist(parsed.data))setNotice('加密备份已导入此浏览器。');}
  }catch(e){setError(e instanceof Error?e.message:'无法读取备份。');}
 }
 function AppTable({items}:{items:Application[]}){return items.length?<div className="table-scroll"><table><thead><tr><th>学校 / 导师</th><th>当前进展</th><th>下一步</th><th>最近记录</th><th aria-label="操作"></th></tr></thead><tbody>{items.map(a=><tr key={a.id}><td><button className="text-button person" onClick={()=>setModal({kind:'application',id:a.id})}>{a.person}</button><span className="school">{a.school}</span>{a.category==='advice'&&<small className="muted">学业咨询</small>}</td><td><Badge status={a.status}/></td><td className="next-cell">{a.next||'暂无待办'}{a.deadline&&<small className="deadline">截止：{dateLabel(a.deadline)}</small>}</td><td className="date-cell">{dateLabel(a.latest)}<MailLink source={a.source}/></td><td><button className="text-button" onClick={()=>setModal({kind:'application',id:a.id})}>编辑</button></td></tr>)}</tbody></table></div>:<Empty text="没有符合条件的记录。"/>}
 function Timeline({items}:{items:TimelineEvent[]}){return items.length?<ol className="timeline">{items.map((e,i)=><li key={e.id}><div className="timeline-date">{i===0||items[i-1].date!==e.date?dateLabel(e.date):''}</div><div className="timeline-body"><div className="eyebrow">{appById(e.appId)?.person||'个人记录'} <span>· {eventNames[e.type]}</span></div><h3>{e.title}</h3><p>{e.detail}</p><div className="event-bottom"><span className="muted">{e.origin==='email'?'来自邮件':e.origin==='sync'?'每日检查':'手动记录'}</span><MailLink source={e.source}/><button className="text-button muted" onClick={()=>setModal({kind:'event',id:e.id})}>编辑</button></div></div></li>)}</ol>:<Empty text="还没有这部分记录，可以添加一条进展。"/>}
 function TaskList({items}:{items:Task[]}){return items.length?<ul className="task-list">{items.map(t=><li className={t.done?'done':''} key={t.id}><input aria-label={'完成：'+t.title} type="checkbox" checked={t.done} disabled={busy} onChange={()=>toggleTask(t)}/><div><button className="text-button task-title" onClick={()=>setModal({kind:'task',id:t.id})}>{t.title}</button><p className="task-meta">{appById(t.appId)?.person||'申请准备'}{t.due&&<> · <span className={t.due<date&&!t.done?'overdue':''}>{dateLabel(t.due)}{t.due<date&&!t.done?' · 日期已过':''}</span></>}</p><p className="task-detail">{t.detail}</p><MailLink source={t.source}/></div></li>)}</ul>:<Empty text="没有未完成事项。你可以添加新的待办。"/>}
 const editedApp=modal?.kind==='application'?apps.find(a=>a.id===modal.id):undefined;
 const editedEvent=modal?.kind==='event'?state.events.find(e=>e.id===modal.id):undefined;
 const editedTask=modal?.kind==='task'?state.tasks.find(t=>t.id===modal.id):undefined;
 return <div className="shell">
  <a className="skip-link" href="#main">跳转到正文</a>
  <aside className="sidebar"><div className="identity"><span className="monogram">YL</span><div><strong>刘宇</strong><small>博士申请记录</small></div></div><nav aria-label="主导航">{(Object.keys(viewNames) as View[]).map(v=><button key={v} aria-current={view===v?'page':undefined} onClick={()=>navigate(v)}><Icon name={v}/>{viewNames[v]}{v==='tasks'&&openTasks.length>0&&<span className="nav-count">{openTasks.length}</span>}</button>)}</nav><div className="sidebar-caption"><span>2027 入学季</span><p>数值分析 · 随机分析<br/>科学计算</p></div><div className="sidebar-bottom"><span className="private-label"><svg width="14" height="14" viewBox="0 0 20 20" fill="none" stroke="currentColor" aria-hidden="true"><rect x="4" y="9" width="12" height="9" rx="2"/><path d="M6 9V6a4 4 0 0 1 8 0v3"/></svg>密码保护</span><button className="text-button" onClick={onLock}>锁定</button></div></aside>
  <main id="main" className="content"><header className="page-header"><div><p className="kicker">FALL 2027 / APPLICATION JOURNAL</p><h1>{view==='overview'?'博士申请记录':viewNames[view]}</h1><p className="lead">{view==='overview'?'把每一次联系、交流和决定，按时间留在这里。':view==='applications'?'记录真实进展，区分已联系、草稿与正式申请。':view==='timeline'?'来自邮件的进展，和你自己的申请笔记。':'下一步要做的事，按日期排列。'}</p></div><button className="primary" disabled={busy} onClick={()=>setModal({kind:view==='applications'?'application':view==='tasks'?'task':'event'})}>＋ {view==='applications'?'添加申请':view==='tasks'?'添加待办':'记录进展'}</button></header>
  <div className="save-bar"><span>{busy?'正在保存…':saved?'本机已保存 · '+new Intl.DateTimeFormat('zh-CN',{timeZone:'Asia/Shanghai',month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit'}).format(new Date(saved)):'邮件整理于 '+dateLabel(state.importedAt)}</span><div><button className="text-button" onClick={()=>onExport(state)}>导出加密备份</button><button className="text-button" disabled={busy} onClick={()=>importRef.current?.click()}>导入加密备份</button><input ref={importRef} type="file" accept=".json,application/json" hidden onChange={e=>{const f=e.target.files?.[0];if(f)importFile(f);e.target.value='';}}/></div></div>
  {error&&<div role="alert" className="error-banner">{error}<button className="text-button" onClick={()=>onExport(state)}>导出当前记录</button></div>}{notice&&<p role="status" className="notice">{notice}</p>}
  {state.sync&&<section className="daily-summary" aria-label="每日邮件更新"><div><h2>每日更新</h2><span className="muted">检查时间：{new Intl.DateTimeFormat('zh-CN',{timeZone:'Asia/Shanghai',month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit'}).format(new Date(state.sync.checkedAt))} · 每天 08:00</span></div><p>{state.sync.status==='ok'?`本次发现 ${state.sync.newMessages} 封新的申请相关邮件`:'本次检查未完成，显示上次记录'}</p><ul>{state.sync.summary.map((s,i)=><li key={i}>{s}</li>)}</ul></section>}
  {view==='overview'?<>
   <section className="stats" aria-label="申请概况"><Stat value={active.length} label="已联系导师"/><Stat value={active.filter(a=>a.status==='meeting').length} label="交流安排中"/><Stat value={active.filter(a=>a.status==='waiting').length} label="等待回复"/><button className="stat" onClick={()=>{navigate('applications');setStatus('draft');}}><strong>{drafts.length}</strong><span>草稿 · 未发送</span></button></section>
   {openTasks.some(t=>t.due)&&<section className="up-next"><span className="section-label">最近安排</span><div><h2>{openTasks.find(t=>t.due)?.title}</h2><p>{openTasks.find(t=>t.due)?.detail}</p><button className="text-button" onClick={()=>navigate('tasks')}>查看待办</button></div><time>{dateLabel(openTasks.find(t=>t.due)!.due)}</time></section>}
   <div className="overview-grid"><section><div className="section-heading"><h2>最近进展</h2><button className="text-button" onClick={()=>navigate('timeline')}>全部时间线</button></div><Timeline items={state.events.slice().sort((a,b)=>b.date.localeCompare(a.date)).slice(0,5)}/></section><section className="todo-panel"><div className="section-heading"><h2>待办事项 <span className="count">{openTasks.length}</span></h2><button className="text-button" disabled={busy} onClick={()=>setModal({kind:'task'})}>添加</button></div><TaskList items={openTasks.slice(0,4)}/><button className="text-button" onClick={()=>navigate('tasks')}>查看全部待办</button></section></div>
   <section className="overview-applications"><div className="section-heading"><h2>申请进展</h2><button className="text-button" onClick={()=>navigate('applications')}>管理全部记录</button></div><AppTable items={active}/></section>
  </>:<><div className="filters"><label className="search"><span className="sr-only">搜索记录</span><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true"><circle cx="10" cy="10" r="6"/><path d="m15 15 5 5"/></svg><input value={query} onChange={e=>setQuery(e.target.value)} placeholder="搜索学校、导师或记录"/></label>{view==='applications'?<label><span className="sr-only">按状态筛选</span><select value={status} onChange={e=>setStatus(e.target.value)}><option value="all">已联系与学业咨询</option><option value="everything">全部记录（含草稿）</option>{Object.entries(statusNames).map(([key,name])=><option key={key} value={key}>{name}</option>)}</select></label>:<label><span className="sr-only">按导师筛选</span><select value={scope} onChange={e=>setScope(e.target.value)}><option value="all">所有导师与记录</option>{apps.map(a=><option key={a.id} value={a.id}>{a.person}</option>)}</select></label>}{view==='tasks'&&<label className="checkbox-label"><input type="checkbox" checked={showDone} onChange={e=>setShowDone(e.target.checked)}/>显示已完成</label>}</div>
   {view==='applications'?<><p className="result-count">{filteredApps.length} 条记录 · 草稿不计为已联系；截止日期仅在确认后填写。</p><AppTable items={filteredApps}/></>:view==='timeline'?<Timeline items={events}/>:<TaskList items={tasks}/>}</>}
  <footer><p>邮件整理截至 {dateLabel(state.importedAt)} · 时间显示为北京时间</p><details><summary>数据与同步说明</summary><p>{state.importNote}</p><p>邮件汇总由云端任务每天更新；页面不直接登录 Gmail，不会代发邮件。个人笔记与勾选状态加密保存在此浏览器，换设备请导出、导入加密备份。每日任务可能因连接或服务状态延迟，请核对上方实际检查时间。</p><p>已用页面密码解锁。申请状态与邮件措辞分开记录；鼓励申请不代表录取，交流提议不代表正式面试通知。</p></details></footer>
 </main>
 {modal&&<dialog ref={dialogRef} className="edit-dialog" onCancel={e=>{e.preventDefault();if(!busy)setModal(null);}} aria-labelledby="dialog-title"><form key={modal.kind+(modal.id||'new')} onSubmit={submit}><div className="dialog-heading"><h2 id="dialog-title">{modal.id?'编辑':'添加'}{modal.kind==='application'?'申请记录':modal.kind==='event'?'进展记录':'待办事项'}</h2><button type="button" className="close" aria-label="关闭" disabled={busy} onClick={()=>setModal(null)}>×</button></div><fieldset disabled={busy}>
 {modal.kind==='application'?<><div className="form-row"><label>导师 / 联系人<input name="person" defaultValue={editedApp?.person} required maxLength={300}/></label><label>学校<input name="school" defaultValue={editedApp?.school} required maxLength={300}/></label></div><label>研究方向<input name="topic" defaultValue={editedApp?.topic} maxLength={300}/></label><div className="form-row"><label>当前状态<select name="status" defaultValue={editedApp?.status||'draft'}>{Object.entries(statusNames).map(([key,name])=><option key={key} value={key}>{name}</option>)}</select></label><label>记录类别<select name="category" defaultValue={editedApp?.category||'application'}><option value="application">博士申请</option><option value="advice">学业咨询</option></select></label></div><div className="form-row"><label>最近记录日期<input name="latest" type="date" defaultValue={editedApp?.latest||date}/></label><label>已确认的截止日期<input name="deadline" type="date" defaultValue={editedApp?.deadline}/></label></div><label>下一步<textarea name="next" defaultValue={editedApp?.next} rows={2} maxLength={2000}/></label><label>备注<textarea name="notes" defaultValue={editedApp?.notes} rows={4} maxLength={6000}/></label><label>来源邮件链接（可选）<input name="source" type="url" defaultValue={editedApp?.source} placeholder="https://mail.google.com/…"/></label></>:<><label>{modal.kind==='event'?'进展标题':'待办事项'}<input name="title" defaultValue={editedEvent?.title||editedTask?.title} required maxLength={300}/></label><label>关联申请<select name="appId" defaultValue={editedEvent?.appId||editedTask?.appId||''}><option value="">个人记录 / 申请准备</option>{apps.map(a=><option key={a.id} value={a.id}>{a.person} · {a.school}</option>)}</select></label>{modal.kind==='event'?<div className="form-row"><label>日期<input type="date" name="date" defaultValue={editedEvent?.date||date} required/></label><label>类型<select name="type" defaultValue={editedEvent?.type||'note'}>{Object.entries(eventNames).map(([key,name])=><option key={key} value={key}>{name}</option>)}</select></label></div>:<label>计划日期（可留空）<input type="date" name="due" defaultValue={editedTask?.due}/></label>}<label>内容<textarea name="detail" defaultValue={editedEvent?.detail||editedTask?.detail} rows={5} maxLength={modal.kind==='event'?6000:2000}/></label><label>来源邮件链接（可选）<input name="source" type="url" defaultValue={editedEvent?.source||editedTask?.source} placeholder="https://mail.google.com/…"/></label>{modal.kind==='task'&&<label className="checkbox-label"><input type="checkbox" name="done" defaultChecked={editedTask?.done}/>已完成</label>}{editedEvent?.origin==='email'&&<p className="muted">修改后将标为手动记录，原邮件链接会保留。</p>}</>}
 </fieldset><div className="dialog-actions">{modal.id&&<button type="button" className="danger text-button" disabled={busy} onClick={remove}>删除</button>}<button type="button" disabled={busy} onClick={()=>setModal(null)}>取消</button><button type="submit" className="primary" disabled={busy}>{busy?'保存中…':'保存记录'}</button></div>{error&&<p role="alert" className="form-error">{error}</p>}</form></dialog>}
 </div>;
}
function Stat({value,label}:{value:number;label:string}){return <div className="stat"><strong>{value}</strong><span>{label}</span></div>}
function Empty({text}:{text:string}){return <div className="empty">{text}</div>}
