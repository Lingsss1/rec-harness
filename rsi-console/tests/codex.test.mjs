import test from 'node:test';
import assert from 'node:assert/strict';
import {EventEmitter} from 'node:events';
import {PassThrough,Writable} from 'node:stream';
import {Codex} from '../server/codex.mjs';
function fakeProcess(handle) {
 const child=new EventEmitter();
 child.stdout=new PassThrough();
 child.stderr=new PassThrough();
 const emit=m=>child.stdout.write(JSON.stringify(m)+'\n');
 child.stdin=new Writable({write(buf,encoding,done){
  const m=JSON.parse(buf.toString());queueMicrotask(()=>handle(m,emit));done();
 }});
 return child;
}
test('completion-before-RPC-response and duplicate events preserve real result',async()=>{
 let seenSkill=false;
 const c=new Codex({cwd:'/tmp',spawnProcess:()=>fakeProcess((m,emit)=>{
  if(m.method==='initialize')emit({id:m.id,result:{}});
  if(m.method==='thread/start')emit({id:m.id,result:{thread:{id:'t1'}}});
  if(m.method==='turn/start'){
   seenSkill=m.params.input.some(i=>i.type==='skill');
   const msg={method:'item/completed',params:{threadId:'t1',item:{type:'agentMessage',text:JSON.stringify({summary:'verified',edit:'none',strategy:'s',decisions:{}})}}};
   emit(msg);emit(msg);
   emit({method:'turn/completed',params:{threadId:'t1',turn:{id:'turn1',status:'completed'}}});
   emit({id:m.id,result:{turn:{id:'turn1'}}});
  }
 })});
 await c.init();
 const result=await c.run('test',{name:'skill',path:'/tmp/SKILL.md'},1000);
 assert.equal(result.result.summary,'verified');assert.equal(result.threadId,'t1');assert.equal(seenSkill,true);
});
test('failed turn returns error without fabricated result',async()=>{
 const c=new Codex({cwd:'/tmp',spawnProcess:()=>fakeProcess((m,emit)=>{
  if(m.method==='thread/start')emit({id:m.id,result:{thread:{id:'x'}}});
  if(m.method==='turn/start'){
   emit({id:m.id,result:{turn:{id:'y'}}});
   emit({method:'turn/completed',params:{threadId:'x',turn:{status:'failed',error:{message:'controlled failure'}}}});
  }
 })});
 await assert.rejects(c.run('test',null,1000),/controlled failure/);
});
