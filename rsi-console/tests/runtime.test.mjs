import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {Store,write,hash} from '../server/store.mjs';
import {Scheduler} from '../server/scheduler.mjs';
import {config,protocol,ROOT,UPSTREAM,PYTHON} from '../server/config.mjs';
test('frozen real runner emits rank-backed metrics, resumes idempotently and rejects modified harness',{skip:!fs.existsSync(PYTHON)},async()=>{
 const store=new Store(fs.mkdtempSync(path.join(os.tmpdir(),'rsi-real-test-')));const c=config({epochs:1,trainUsers:32,evalUsers:32});const p=protocol(c);const r=store.create(c,p);const dir=store.dir(r.id);write(path.join(dir,'protocol.json'),p);fs.mkdirSync(path.join(dir,'harness'),{recursive:true});fs.copyFileSync(path.join(ROOT,'scripts/train.py'),path.join(dir,'harness/train.py'));fs.mkdirSync(path.join(dir,'models/m0'),{recursive:true});const source=path.join(UPSTREAM,'examples/sasrec_cds/initial_code/models.py');fs.copyFileSync(source,path.join(dir,'models/m0/models.py'));r.models=[{id:'m0',parent:null,round:0,codeIdentity:hash(fs.readFileSync(source))}];store.save(r);const scheduler=new Scheduler(store);await scheduler.train(r,r.models[0]);const model=store.get(r.id).models[0];assert.ok(Number.isFinite(model.metrics.hr));const before=fs.statSync(path.join(dir,'models/m0/checkpoint.pt')).mtimeMs;await scheduler.train(store.get(r.id),model);assert.equal(fs.statSync(path.join(dir,'models/m0/checkpoint.pt')).mtimeMs,before);fs.appendFileSync(path.join(dir,'harness/train.py'),'\n# mutation');await assert.rejects(scheduler.train(store.get(r.id),model),/冻结评估代码/);store.close();
});
