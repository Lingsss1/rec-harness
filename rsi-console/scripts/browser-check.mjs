import {chromium} from '@playwright/test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
const page=await browser.newPage({viewport:{width:1440,height:1100}});const errors=[];page.on('pageerror',e=>errors.push(e.message));
try{
 await page.goto('http://127.0.0.1:4317');await page.locator('.runItem').filter({hasText:'验收 · SASRec CPU 真实闭环'}).click();await page.getByText('验收 · SASRec CPU 真实闭环',{exact:true}).last().waitFor();
 await page.getByRole('button',{name:'第 1 轮',exact:true}).click();
 await page.getByRole('heading',{name:'模型结论'}).waitFor();
 assert.ok(await page.getByText('未同时满足验证 NDCG').count()||await page.getByText('验证 NDCG 严格提升').count());
 await page.screenshot({path:'rsi-console/var/console-desktop.png',fullPage:true});
 await page.getByRole('button',{name:'Agent 交接',exact:true}).click();assert.equal(await page.locator('.agentCard').count(),5);
 await page.getByRole('button',{name:'代码与产物',exact:true}).click();await page.getByRole('button',{name:/round-1\/model.patch/}).click();await page.getByRole('dialog').waitFor();assert.match(await page.getByRole('dialog').innerText(),/GELU/);await page.getByRole('button',{name:'关闭',exact:true}).click();
 await page.getByRole('button',{name:'冻结协议',exact:true}).click();assert.match(await page.locator('.protocol').innerText(),/0.0503/);
 await page.getByRole('button',{name:'◷ 回放',exact:true}).click();assert.equal(await page.getByRole('slider').count(),1);
 await page.setViewportSize({width:390,height:844});await page.screenshot({path:'rsi-console/var/console-mobile.png',fullPage:true});assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth));
 await page.getByRole('button',{name:'＋ 新建实验',exact:true}).click();await page.getByRole('dialog').waitFor();await page.locator('select[name=mode]').selectOption('demo');await page.locator('input[name=name]').fill('浏览器验收 · 演示');await page.getByRole('button',{name:'冻结协议并创建',exact:true}).click();await page.getByRole('button',{name:'▶ 启动',exact:true}).click();await page.getByText('已完成',{exact:true}).waitFor();assert.match(await page.locator('.mode').innerText(),/演示数据/);
 // SSE cursor must omit preceding events and reconnect to persisted sequence.
 const runs=await (await page.request.get('http://127.0.0.1:4317/api/runs')).json();const demo=runs.find(r=>r.name==='浏览器验收 · 演示');const events=await (await page.request.get(`http://127.0.0.1:4317/api/runs/${demo.id}/events`)).json();const cursor=events[2].seq;const replay=await (await page.request.get(`http://127.0.0.1:4317/api/runs/${demo.id}/events?after=${cursor}`)).json();assert.ok(replay.every(e=>e.seq>cursor));
 const replayed=await page.evaluate(async({id,cursor})=>await new Promise((resolve,reject)=>{const es=new EventSource(`/api/runs/${id}/stream?after=${cursor}`);const t=setTimeout(()=>{es.close();reject(Error('SSE timeout'))},5000);es.onmessage=e=>{clearTimeout(t);es.close();resolve(JSON.parse(e.data).seq)}}),{id:demo.id,cursor});assert.equal(replayed,replay[0].seq);
 assert.equal((await page.request.post('http://127.0.0.1:4317/api/runs',{headers:{Origin:'https://evil.example'},data:{mode:'demo'}})).status(),403);
 assert.equal(errors.length,0,errors.join('\n'));fs.writeFileSync('rsi-console/var/browser-check.json',JSON.stringify({passed:true,checks:['desktop','mobile overflow','real metrics','five agent roles','patch viewer','final protocol','history replay','demo create/start/complete','SSE cursor','cross-origin rejection','no page errors']},null,2));console.log('Browser checks passed');
}finally{await browser.close();}
