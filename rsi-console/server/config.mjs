import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { hash,json } from './store.mjs';
export const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
export const UPSTREAM=path.join(ROOT,'vendor/self_evolverec');
export const PYTHON=process.env.RSI_PYTHON||path.join(ROOT,'.venv/bin/python');
export function config(input){
 const integer=(name,min,max,def)=>{const v=input[name]??def;if(!Number.isInteger(v)||v<min||v>max)throw Error(name+' 超出范围');return v;};
 if(!['real','demo'].includes(input.mode??'real'))throw Error('运行模式无效');
 if(input.device&&!['cpu','cuda'].includes(input.device))throw Error('设备无效');
 return {name:String(input.name||'SASRec · Amazon CDs').slice(0,100),mode:input.mode??'real',device:input.device??'cpu',minutes:integer('minutes',1,1440,30),callBudget:integer('callBudget',1,100,10),rounds:integer('rounds',1,5,1),epochs:integer('epochs',1,50,2),trainUsers:integer('trainUsers',32,20000,512),evalUsers:integer('evalUsers',32,20000,256),seed:integer('seed',0,2147483647,42)};
}
export function protocol(c){const files=['train.json','valid.json','validation_candidate.json','item2id.json','user2id.json','test.json','test_candidate.json','meta.json','train_review.json'];const dataHashes={};for(const f of files){const p=path.join(UPSTREAM,'data_cache/CDs_and_Vinyl',f);dataHashes[f]=fs.existsSync(p)?hash(fs.readFileSync(p)):null;}const p={version:2,runnerHash:hash(fs.readFileSync(path.join(ROOT,'scripts/train.py'))),utilityHash:hash(fs.readFileSync(path.join(UPSTREAM,'simulator/util_functions.py'))),upstream:UPSTREAM,commit:json(path.join(ROOT,'upstream.lock.json')).self_evolverec.commit,dataHashes,seed:c.seed,epochs:c.epochs,trainUsers:c.trainUsers,evalUsers:c.evalUsers,device:c.device,hidden:32,maxlen:50,batchSize:32,metrics:['NDCG@5','HR@5'],primary:'ndcg',keep:'ndcg strictly improves AND hr does not decrease',testPolicy:'final incumbent only',sampling:'deterministic seeded user subset, upstream fixed candidate lists',profile:'CPU/GPU bounded validation; not paper reproduction'};p.id=hash(JSON.stringify(p));return p;}
export function preflight(c){const blockers=[];const p=protocol(c);if(Object.values(p.dataHashes).some(x=>!x))blockers.push('缺少 Self-EvolveRec 数据，请运行安装脚本');const revision=spawnSync('git',['-C',UPSTREAM,'rev-parse','HEAD'],{encoding:'utf8'});if(revision.stdout?.trim()!==p.commit)blockers.push('上游提交与锁定版本不一致');const py=spawnSync(PYTHON,['-c',`import torch,numpy,tqdm; print(torch.__version__); ${c.device==='cuda'?'assert torch.cuda.is_available(), "CUDA unavailable"':''}`],{encoding:'utf8',timeout:15000});if(py.status!==0)blockers.push(c.device==='cuda'?'Python 依赖或 CUDA 不可用':'Python 缺少 torch/numpy/tqdm');const login=spawnSync(process.env.RSI_CODEX||'codex',['login','status'],{encoding:'utf8',timeout:10000});if(login.status!==0)blockers.push('Codex 不可用或尚未登录');return {blockers,device:c.device,protocol:p.id,codex:login.status===0,python:py.status===0,gpuSlots:1,maxAgents:2};}
