const encoder=new TextEncoder();
export const bytesToB64=bytes=>{let s='';for(const b of new Uint8Array(bytes))s+=String.fromCharCode(b);return btoa(s);};
const fromB64=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));
const aad=(kind,id)=>encoder.encode(`yu-phd-tracker/${kind}/v1:${id}`);
export async function unlockVault(vault,password){
 if(vault.format!=='phd-tracker-vault'||vault.version!==1||vault.kdf.iterations<600000||vault.kdf.iterations>2000000)throw Error('不支持的加密配置。');
 const material=await crypto.subtle.importKey('raw',encoder.encode(password),'PBKDF2',false,['deriveKey']);
 const key=await crypto.subtle.deriveKey({name:'PBKDF2',hash:'SHA-256',salt:fromB64(vault.kdf.salt),iterations:vault.kdf.iterations},material,{name:'AES-GCM',length:256},false,['decrypt']);
 const raw=await crypto.subtle.decrypt({name:'AES-GCM',iv:fromB64(vault.privateKey.iv),additionalData:encoder.encode('yu-phd-tracker/private-key/v1')},key,fromB64(vault.privateKey.ciphertext));
 return crypto.subtle.importKey('pkcs8',raw,{name:'RSA-OAEP',hash:'SHA-256'},false,['decrypt']);
}
export async function openEnvelope(envelope,vault,privateKey,expectedKind='snapshot'){
 if(envelope.format!=='phd-tracker-envelope'||envelope.version!==1||envelope.keyId!==vault.keyId||envelope.kind!==expectedKind)throw Error('备份或数据与当前页面不匹配。');
 const raw=await crypto.subtle.decrypt('RSA-OAEP',privateKey,fromB64(envelope.encryptedKey));
 const key=await crypto.subtle.importKey('raw',raw,'AES-GCM',false,['decrypt']);
 const clear=await crypto.subtle.decrypt({name:'AES-GCM',iv:fromB64(envelope.iv),additionalData:aad(expectedKind,vault.keyId)},key,fromB64(envelope.ciphertext));
 return JSON.parse(new TextDecoder().decode(clear));
}
export async function sealEnvelope(data,vault,kind='local'){
 const publicKey=await crypto.subtle.importKey('spki',fromB64(vault.publicKey),{name:'RSA-OAEP',hash:'SHA-256'},false,['encrypt']);
 const key=await crypto.subtle.generateKey({name:'AES-GCM',length:256},true,['encrypt']);
 const raw=await crypto.subtle.exportKey('raw',key),iv=crypto.getRandomValues(new Uint8Array(12));
 const ciphertext=await crypto.subtle.encrypt({name:'AES-GCM',iv,additionalData:aad(kind,vault.keyId)},key,encoder.encode(JSON.stringify(data)));
 const encryptedKey=await crypto.subtle.encrypt('RSA-OAEP',publicKey,raw);
 return {format:'phd-tracker-envelope',version:1,kind,keyId:vault.keyId,encryptedKey:bytesToB64(encryptedKey),iv:bytesToB64(iv),ciphertext:bytesToB64(ciphertext)};
}
const collections=['applications','events','tasks','schoolPlans'];
export function createOverlay(base,next){
 const result={version:1,updatedAt:new Date().toISOString(),collections:{}};
 for(const name of collections){const before=new Map((base[name]||[]).map(item=>[item.id,item])),after=new Map((next[name]||[]).map(item=>[item.id,item])),changes=[];
  for(const [id,item] of after){const old=before.get(id),patch={};for(const [key,value] of Object.entries(item))if(!old||JSON.stringify(value)!==JSON.stringify(old[key]))Object.defineProperty(patch,key,{value,enumerable:true});if(Object.keys(patch).length)changes.push({id,patch});}
  for(const id of before.keys())if(!after.has(id))changes.push({id,deleted:true});result.collections[name]=changes;
 }return result;
}
export function applyOverlay(base,overlay){
 if(overlay.version!==1||!overlay.collections)throw Error('本机记录格式有误，请先导出备份。');
 const result=structuredClone(base);
 for(const name of collections){const items=new Map((result[name]||[]).map(item=>[item.id,item]));
  if(name==='schoolPlans'&&!overlay.collections[name])continue;
  if(!Array.isArray(overlay.collections[name]))throw Error('本机记录格式有误。');
  for(const change of overlay.collections[name]){if(typeof change.id!=='string')throw Error('记录 ID 有误。');if(change.deleted)items.delete(change.id);else items.set(change.id,{...items.get(change.id),...change.patch,id:change.id});}
  result[name]=[...items.values()];
 }
 const ids=new Set(result.applications.map(a=>a.id));for(const name of ['events','tasks'])result[name]=(result[name]||[]).map(item=>item.appId&&!ids.has(item.appId)?{...item,appId:''}:item);
 return result;
}
