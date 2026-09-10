import path from 'node:path';
import {Store} from '../server/store.mjs';
import {Scheduler} from '../server/scheduler.mjs';
import {ROOT} from '../server/config.mjs';
import {materializePairedCandidate} from '../server/rsi-candidates.mjs';
const store=new Store(path.join(ROOT,'var'));
try{if(store.all().some(r=>['running','pausing'].includes(r.status)))throw Error('另有任务运行，暂不补充验收产物');const r=store.all().find(r=>r.mode==='real'&&r.status==='completed');if(!r)throw Error('缺少完成的真实验收');if(!r.rsi.some(v=>v.id==='tool-v0')){r.rsi.unshift({id:'tool-v0',parent:null,status:'active',kind:'tool',round:0,reason:'原始诊断工具 · 项目内副本'},{id:'skill-v0',parent:null,status:'active',kind:'skill',round:0,reason:'三个原始 Skill · 项目内副本'});store.save(r);}await materializePairedCandidate(new Scheduler(store),r);console.log('Tool and Skill candidate materialized:',r.id);}finally{store.close();}
