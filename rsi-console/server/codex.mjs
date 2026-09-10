import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { EventEmitter } from 'node:events';
import { redact } from './store.mjs';
export class Codex extends EventEmitter{
 constructor({cwd,onEvent=()=>{},onApproval=()=>{},onProcess=()=>{},spawnProcess=spawn}){super();this.seq=0;this.pending=new Map();this.notes=[];this.cwd=cwd;this.onEvent=onEvent;this.onApproval=onApproval;
  this.child=spawnProcess(process.env.RSI_CODEX||'codex',['app-server','--stdio'],{cwd,stdio:['pipe','pipe','pipe'],detached:true});onProcess(this.child);
  this.child.on('error',e=>this.fail(e));this.child.on('exit',()=>this.fail(Error('Codex App Server 已退出')));
  this.child.stderr.on('data',x=>this.emit('diagnostic',redact(x.toString())));
  createInterface({input:this.child.stdout}).on('line',line=>{try{this.message(JSON.parse(line));}catch(e){this.emit('diagnostic',redact(e.message));}});
 }
 send(m){this.child.stdin.write(JSON.stringify(m)+'\n');}
 rpc(method,params){const id=++this.seq;return new Promise((resolve,reject)=>{const timer=setTimeout(()=>{this.pending.delete(id);reject(Error('Codex RPC 超时: '+method));},45000);this.pending.set(id,{resolve,reject,timer});this.send({id,method,params});});}
 message(m){if(m.id!==undefined&&!m.method){const q=this.pending.get(m.id);if(q){clearTimeout(q.timer);this.pending.delete(m.id);m.error?q.reject(Error(m.error.message)):q.resolve(m.result);}return;}
  if(m.id!==undefined&&m.method){this.onApproval({requestId:m.id,method:m.method,params:m.params});return;}
  this.notes.push(m);this.emit('notification',m);this.onEvent(m);
 }
 fail(e){for(const p of this.pending.values()){clearTimeout(p.timer);p.reject(e);}this.pending.clear();this.emit('closed',e);}
 async init(){await this.rpc('initialize',{clientInfo:{name:'rsi_console',version:'0.1.0'},capabilities:{experimentalApi:false}});this.send({method:'initialized',params:{}});}
 async account(){return this.rpc('account/read',{refreshToken:false});}
 async run(text,skill,timeoutMs=180000){const {thread}=await this.rpc('thread/start',{cwd:this.cwd,sandbox:'read-only',approvalPolicy:'on-request',ephemeral:true,developerInstructions:'You are an isolated research agent. Return the requested JSON only. Do not execute commands, access credentials, edit files, invoke other agents, or change protocols/budgets. All necessary evidence is included in the user message. Skills are research guidance; the deterministic host performs writes and validation.'});this.threadId=thread.id;
  const input=[{type:'text',text}];if(skill)input.push({type:'skill',name:skill.name,path:skill.path});
  let timer,listener,closed;
  const done=new Promise((resolve,reject)=>{listener=m=>{if(m.method==='turn/completed'&&m.params?.threadId===thread.id){const turn=m.params.turn;turn.status==='completed'?resolve(turn):reject(Error(turn.error?.message||'Codex turn '+turn.status));}};closed=reject;this.on('notification',listener);this.on('closed',closed);timer=setTimeout(()=>reject(Error('Agent 时间预算耗尽')),timeoutMs);});
  done.catch(()=>{});
  // Install listeners before turn/start so fast completion cannot be lost.
  try{const {turn}=await this.rpc('turn/start',{threadId:thread.id,input,outputSchema:{type:'object',properties:{summary:{type:'string'},edit:{type:'string',enum:['gelu','one_layer','three_layers','none']},strategy:{type:'string'},decisions:{type:'object',properties:Object.fromEntries(['test-leak','stale','missing','valid','null','cost'].map(k=>[k,{type:'string'}])),required:['test-leak','stale','missing','valid','null','cost'],additionalProperties:false}},required:['summary','edit','strategy','decisions'],additionalProperties:false}});this.turnId=turn.id;await done;const messages=this.notes.filter(x=>x.method==='item/completed'&&x.params?.threadId===thread.id&&x.params.item?.type==='agentMessage');const raw=messages.at(-1)?.params.item.text;if(!raw)throw Error('Agent 未产生结构化结果');return {threadId:thread.id,turnId:turn.id,result:JSON.parse(raw)};}
  finally{clearTimeout(timer);this.off('notification',listener);this.off('closed',closed);}
 }
 respond(id,result){this.send({id,result});}
 stop(){try{process.kill(-this.child.pid,'SIGTERM');}catch{}setTimeout(()=>{try{process.kill(-this.child.pid,'SIGKILL');}catch{}},1500).unref();}
}
