import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { Store,write,redact } from './store.mjs';
import { ROOT,config,protocol,preflight } from './config.mjs';
import { Scheduler } from './scheduler.mjs';
const port=Number(process.env.PORT||4317);const store=new Store(process.env.RSI_DATA||path.join(ROOT,'var'));const lock=path.join(store.root,'service.lock');
const birth=pid=>{try{return fs.readFileSync(`/proc/${pid}/stat`,'utf8').split(') ')[1].split(' ')[19];}catch{return null;}};
if(fs.existsSync(lock)){let old;try{old=JSON.parse(fs.readFileSync(lock,'utf8'));}catch{throw Error('服务锁损坏，请人工核对');}if(birth(old.pid)===old.birth){if(!process.argv.includes('--restart'))throw Error('已有控制台后端持有调度锁');if(store.all().some(r=>['running','pausing'].includes(r.status)))throw Error('有任务运行，不能自动重启');process.kill(old.pid,'SIGTERM');for(let i=0;i<20&&fs.existsSync(lock);i++)await new Promise(resolve=>setTimeout(resolve,100));if(fs.existsSync(lock))throw Error('旧服务尚未退出');}else fs.unlinkSync(lock);}
fs.writeFileSync(lock,JSON.stringify({pid:process.pid,birth:birth(process.pid)}),{flag:'wx'});
process.on('exit',()=>{try{const current=JSON.parse(fs.readFileSync(lock,'utf8'));if(current.pid===process.pid)fs.unlinkSync(lock);}catch{}});
const scheduler=new Scheduler(store);scheduler.recover();
const send=(res,status,obj)=>{res.writeHead(status,{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-store'});res.end(JSON.stringify(obj));};
const body=async req=>{let text='';for await(const chunk of req){text+=chunk;if(text.length>65536)throw Error('请求过大');}return text?JSON.parse(text):{};};
const clients=new Set();
const server=http.createServer(async(req,res)=>{
 res.setHeader('X-Content-Type-Options','nosniff');res.setHeader('Referrer-Policy','no-referrer');res.setHeader('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'");
 try{
  if(![`127.0.0.1:${port}`,`localhost:${port}`].includes(req.headers.host))return send(res,403,{error:'仅允许本机 Host'});
  if(req.headers.origin&&!['http://127.0.0.1:'+port,'http://localhost:'+port,'http://127.0.0.1:5173','http://localhost:5173'].includes(req.headers.origin))return send(res,403,{error:'跨站请求被拒绝'});
  if(req.method==='POST'&&req.headers['content-type']!=='application/json')return send(res,415,{error:'需要 application/json'});
  const u=new URL(req.url,'http://localhost');const parts=u.pathname.split('/').filter(Boolean);
  if(u.pathname==='/api/health')return send(res,200,{ok:true,active:scheduler.active});
  if(u.pathname==='/api/preflight')return send(res,200,preflight(config({device:u.searchParams.get('device')||'cpu'})));
  if(u.pathname==='/api/runs'&&req.method==='GET')return send(res,200,store.all());
  if(u.pathname==='/api/runs'&&req.method==='POST'){const c=config(await body(req));const r=store.create(c,protocol(c));write(path.join(store.root,r.id,'protocol.json'),r.protocol);fs.mkdirSync(path.join(store.root,r.id,'harness'),{recursive:true});fs.copyFileSync(path.join(ROOT,'scripts/train.py'),path.join(store.root,r.id,'harness/train.py'));fs.cpSync(path.join(ROOT,'skills'),path.join(store.root,r.id,'skills'),{recursive:true});return send(res,201,r);}
  if(parts[0]==='api'&&parts[1]==='runs'&&parts[2]){const id=parts[2];const r=store.get(id);const operation=parts[3];
   if(!operation&&req.method==='GET')return send(res,200,r);
   if(operation==='control'&&req.method==='POST'){const b=await body(req);return send(res,200,scheduler.control(id,b.action));}
   if(operation==='approvals'&&req.method==='POST'){const b=await body(req);scheduler.respond(id,b.requestId,b.decision);return send(res,200,{ok:true});}
   if(operation==='events'&&req.method==='GET')return send(res,200,store.events(id,Number(u.searchParams.get('after'))||0));
   if(operation==='stream'&&req.method==='GET'){
    let after=Number(req.headers['last-event-id']||u.searchParams.get('after'))||0;res.writeHead(200,{'Content-Type':'text/event-stream','Cache-Control':'no-cache','Connection':'keep-alive','X-Accel-Buffering':'no'});const emit=e=>{if(e.runId!==id||e.seq<=after)return;after=e.seq;res.write(`id: ${e.seq}\ndata: ${JSON.stringify(e)}\n\n`);};
    store.on('event',emit);for(const e of store.events(id,after))emit(e);const timer=setInterval(()=>res.write(': heartbeat\n\n'),15000);clients.add(res);req.on('close',()=>{clearInterval(timer);store.off('event',emit);clients.delete(res);});return;
   }
   if(operation==='artifacts'&&req.method==='GET'){const rel=u.searchParams.get('path');if(!rel)return send(res,200,r.artifacts);if(!r.artifacts.some(a=>a.path===rel))throw Error('未登记的产物');const p=store.safe(id,rel);if(fs.statSync(p).size>2000000)return send(res,413,{error:'大产物请在本机实验目录查看'});res.writeHead(200,{'Content-Type':'text/plain; charset=utf-8'});res.end(redact(fs.readFileSync(p,'utf8')));return;}
   if(['models','rsi','agents','metrics'].includes(operation)&&req.method==='GET')return send(res,200,operation==='metrics'?r.models.map(x=>({id:x.id,protocolId:r.protocol.id,metrics:x.metrics??null,evidence:x.evidence??[]})):r[operation]);
  }
  if(u.pathname.startsWith('/api/'))return send(res,404,{error:'接口不存在'});
  const root=path.join(ROOT,'dist');let file=path.resolve(root,'.'+decodeURIComponent(u.pathname));if(!file.startsWith(root+path.sep))file=path.join(root,'index.html');if(!fs.existsSync(file)||!fs.statSync(file).isFile())file=path.join(root,'index.html');if(!fs.existsSync(file))return send(res,503,{error:'请先 npm run build'});const ext=path.extname(file);res.setHeader('Content-Type',({'.html':'text/html; charset=utf-8','.js':'text/javascript','.css':'text/css','.svg':'image/svg+xml'})[ext]||'application/octet-stream');fs.createReadStream(file).pipe(res);
 }catch(e){send(res,400,{error:redact(e.message)});}
});
server.listen(port,'127.0.0.1',()=>console.log(`RSI Console http://127.0.0.1:${port}`));
function shutdown(){if(scheduler.active)scheduler.control(scheduler.active,'stop');for(const client of clients)client.end();server.close(()=>{store.close();process.exit(0);});setTimeout(()=>process.exit(0),2000).unref();}process.on('SIGINT',shutdown);process.on('SIGTERM',shutdown);
