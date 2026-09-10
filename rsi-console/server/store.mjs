import { DatabaseSync } from 'node:sqlite';
import { EventEmitter } from 'node:events';
import fs from 'node:fs';
import path from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
export const hash=x=>createHash('sha256').update(x).digest('hex');
export const json=p=>JSON.parse(fs.readFileSync(p,'utf8'));
export const write=(p,v)=>{fs.mkdirSync(path.dirname(p),{recursive:true});fs.writeFileSync(p,JSON.stringify(v,null,2));};
export const redact=x=>String(x).replace(/\b(?:sk-[\w-]{8,}|Bearer\s+[\w.\-]+)\b/gi,'[REDACTED]').replace(/((?:api[_-]?key|access[_-]?token|refresh[_-]?token)\s*[=:]\s*)[^\s,]+/gi,'$1[REDACTED]');
export class Store extends EventEmitter{
 constructor(root){super();this.root=root;fs.mkdirSync(root,{recursive:true});this.db=new DatabaseSync(path.join(root,'console.sqlite'));this.db.exec(`PRAGMA journal_mode=WAL; CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, body TEXT NOT NULL); CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, dedupe TEXT UNIQUE, body TEXT NOT NULL);`);}
 all(){return this.db.prepare('SELECT body FROM runs ORDER BY rowid DESC').all().map(x=>JSON.parse(x.body));}
 get(id){const x=this.db.prepare('SELECT body FROM runs WHERE id=?').get(id);if(!x)throw Error('任务不存在');return JSON.parse(x.body);}
 save(run){this.db.prepare('INSERT INTO runs VALUES(?,?) ON CONFLICT(id) DO UPDATE SET body=excluded.body').run(run.id,JSON.stringify(run));return run;}
 create(config,protocol){const r={id:randomUUID(),name:config.name,mode:config.mode,status:'created',created:new Date().toISOString(),config,protocol,step:0,round:0,calls:0,elapsedMs:0,models:[],rsi:[{id:'tool-v0',parent:null,status:'active',kind:'tool',round:0,reason:'原始诊断工具 · 项目内副本'},{id:'skill-v0',parent:null,status:'active',kind:'skill',round:0,reason:'三个原始 Skill · 项目内副本'},{id:'rsi-v0',parent:null,status:'active',kind:'strategy',reason:'原始研究 Skill 与冻结策略',round:0}],activeRsi:'rsi-v0',agents:[],artifacts:[],approvals:[]};this.save(r);this.event(r,'created',{message:'任务已创建，协议已冻结'});return r;}
 event(r,type,data={},dedupe=null){const e={runId:r.id,round:r.round,agent:null,modelVersion:r.currentModel??r.models.at(-1)?.id??null,rsiVersion:r.activeRsi,protocolId:r.protocol.id,evidence:[],at:new Date().toISOString(),type,...data};const result=this.db.prepare('INSERT OR IGNORE INTO events(run_id,dedupe,body) VALUES(?,?,?)').run(r.id,dedupe,JSON.stringify(e));if(!result.changes)return null; e.seq=Number(result.lastInsertRowid);this.emit('event',e);return e;}
 events(id,after=0){return this.db.prepare('SELECT seq,body FROM events WHERE run_id=? AND seq>? ORDER BY seq').all(id,after).map(x=>({...JSON.parse(x.body),seq:x.seq}));}
 dir(id){this.get(id);return path.join(this.root,id);}
 artifact(r,relative){const p=this.safe(r.id,relative);const a={path:relative,sha256:hash(fs.readFileSync(p)),round:r.round};const previous=r.artifacts.find(x=>x.path===relative);if(!previous)r.artifacts.push(a);else if(previous.sha256!==a.sha256){previous.revisions=[...(previous.revisions||[]),{sha256:previous.sha256,at:new Date().toISOString()}];previous.sha256=a.sha256;}this.save(r);return a;}
 safe(id,relative){const base=fs.realpathSync(this.dir(id));const p=fs.realpathSync(path.resolve(base,relative));if(!p.startsWith(base+path.sep)||!fs.statSync(p).isFile())throw Error('禁止访问实验目录外的文件');return p;}
 close(){this.db.close();}
}
