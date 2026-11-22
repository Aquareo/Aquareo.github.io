import * as esbuild from 'esbuild';
import {dirname,resolve,relative,basename} from 'node:path';
import {fileURLToPath} from 'node:url';
import {writeFile,mkdir} from 'node:fs/promises';
const here=dirname(fileURLToPath(import.meta.url)),out=resolve(here,'../../dist/phd-tracker');
await mkdir(out,{recursive:true});
const result=await esbuild.build({absWorkingDir:here,entryPoints:['main.tsx'],bundle:true,jsx:'automatic',format:'esm',minify:true,target:['es2022'],outdir:resolve(out,'assets'),entryNames:'app.[hash]',metafile:true,define:{'process.env.NODE_ENV':'"production"'},legalComments:'none'});
const outputs=Object.keys(result.metafile.outputs),js=outputs.find(x=>x.endsWith('.js')),css=outputs.find(x=>x.endsWith('.css'));
const html=`<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow"><meta name="referrer" content="no-referrer"><meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'self'; form-action 'none'"><title>博士申请记录 · Yu Liu</title><link rel="stylesheet" href="assets/${basename(css)}"><script type="module" src="assets/${basename(js)}"></script></head><body><div id="root"></div><noscript>此页面使用浏览器加密功能，需要启用 JavaScript 才能输入密码查看记录。</noscript></body></html>\n`;
await writeFile(resolve(out,'index.html'),html);console.log('Built password-protected tracker assets. No journal data is bundled.');
