import React,{useState,useEffect,useRef} from 'react';
import {createRoot} from 'react-dom/client';
import Tracker from './tracker';
import {stateSchema,type TrackerState} from './schema';
import {unlockVault,openEnvelope,sealEnvelope,createOverlay,applyOverlay} from './crypto.mjs';
import './style.css';

type Session={vault:any;key:CryptoKey;base:TrackerState;initial:TrackerState;saved:string};
function saveDownload(data:unknown,name:string){const url=URL.createObjectURL(new Blob([JSON.stringify(data)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function App(){
 const [session,setSession]=useState<Session|null>(null),[password,setPassword]=useState(''),[visible,setVisible]=useState(false),[busy,setBusy]=useState(false),[error,setError]=useState('');
 const localCorrupt=useRef(false);
 const localRaw=useRef<string|null>(null),timer=useRef<ReturnType<typeof setTimeout>|null>(null);
 const lock=()=>{setSession(null);setPassword('');setError('');if(timer.current)clearTimeout(timer.current);};
 useEffect(()=>{if(!session)return;const reset=()=>{if(timer.current)clearTimeout(timer.current);timer.current=setTimeout(lock,15*60*1000);};reset();window.addEventListener('pointerdown',reset);window.addEventListener('keydown',reset);return()=>{if(timer.current)clearTimeout(timer.current);window.removeEventListener('pointerdown',reset);window.removeEventListener('keydown',reset);};},[session]);
 async function readJson(path:string){const r=await fetch(path+'?v='+Date.now(),{cache:'no-store',credentials:'omit',referrerPolicy:'no-referrer'});if(!r.ok)throw Error('读取记录失败，请检查网络并刷新重试。');return r.json();}
 async function login(event:React.FormEvent){
  event.preventDefault();setBusy(true);setError('');
  try{
   if(!window.crypto?.subtle)throw Error('请使用支持加密功能的现代浏览器，并通过 HTTPS 打开页面。');
   const vault=await readJson('./vault.json');
   let key:CryptoKey;
   try{key=await unlockVault(vault,password);}catch{throw Error('密码不正确，或解锁配置已损坏。请重新输入。');}
   const envelope=await readJson('./payload.enc.json');
   const base=stateSchema.parse(await openEnvelope(envelope,vault,key,'snapshot'));
   let initial=base,saved='';localRaw.current=null;localCorrupt.current=false;
   try{const raw=localStorage.getItem('yu-phd-local:'+vault.keyId);localRaw.current=raw;if(raw){const overlay=await openEnvelope(JSON.parse(raw),vault,key,'local');initial=stateSchema.parse(applyOverlay(base,overlay));saved=overlay.updatedAt;}}
   catch{localCorrupt.current=true;if(!confirm('无法读取此浏览器的加密笔记。是否暂时只查看邮件汇总？原笔记不会被删除。'))throw Error('本机笔记未能读取；请先保留备份再重试。');}
   setPassword('');setSession({vault,key,base,initial,saved});
  }catch(e){setError(e instanceof Error?e.message:'无法解锁记录，请刷新重试。');}finally{setBusy(false);}
 }
 async function persist(state:TrackerState){
  if(!session)throw Error('请重新解锁。');if(localCorrupt.current)throw Error('本机原笔记无法读取，已保留原数据并停止写入。请先备份原记录。');const storageKey='yu-phd-local:'+session.vault.keyId;
  if(localStorage.getItem(storageKey)!==localRaw.current)throw Error('另一个标签页已修改了本机笔记。请先导出当前备份，再刷新页面。');
  const overlay=createOverlay(session.base,state),envelope=await sealEnvelope(overlay,session.vault,'local'),raw=JSON.stringify(envelope);
  try{localStorage.setItem(storageKey,raw);}catch{throw Error('浏览器存储不可用或已满。请先导出加密备份。');}
  localRaw.current=raw;return overlay.updatedAt;
 }
 async function exportBackup(state:TrackerState){if(!session)return;const envelope=await sealEnvelope(state,session.vault,'backup');saveDownload(envelope,'phd-tracker-backup-'+new Date().toISOString().slice(0,10)+'.json');}
 async function importBackup(file:File){if(!session)throw Error('请先解锁。');return stateSchema.parse(await openEnvelope(JSON.parse(await file.text()),session.vault,session.key,'backup'));}
 if(session)return <Tracker initial={session.initial} initialRevision={0} initialSaved={session.saved} displayName="刘宇" onSave={persist} onExport={exportBackup} onImport={importBackup} onLock={lock}/>;
 return <main className="vault-screen"><div className="vault-box"><span className="monogram">YL</span><h1>博士申请记录</h1><p>输入专用密码，查看申请进展与每日更新。</p><form onSubmit={login}><label htmlFor="password">访问密码</label><input id="password" name="password" type={visible?'text':'password'} autoComplete="current-password" value={password} onChange={e=>setPassword(e.target.value)} required autoFocus disabled={busy}/><label className="show-password"><input type="checkbox" checked={visible} onChange={e=>setVisible(e.target.checked)}/>显示密码</label><button className="primary" type="submit" disabled={busy}>{busy?'正在解锁…':'解锁申请记录'}</button>{error&&<div className="vault-error" role="alert">{error}</div>}</form><p className="vault-footnote">密码仅在本机解锁时使用，不会发送到服务器。<br/>闲置 15 分钟后自动锁定。手动笔记加密保存在此浏览器。</p><a className="mail-link" href="../">返回个人主页</a></div></main>;
}
createRoot(document.getElementById('root')!).render(<App/>);
