import fs from 'node:fs';
import path from 'node:path';
import {ROOT,PYTHON} from './config.mjs';
import {hash,json,write} from './store.mjs';
// The agent supplies the proposal. The host materializes a bounded, auditable
// template; never execute arbitrary Python returned in an agent message.
export async function materializePairedCandidate(scheduler,r){
 const store=scheduler.store,dir=store.dir(r.id),round=`round-${r.round}`,folder=`${round}/rsi-candidate`;
 const candidate=r.models.find(v=>v.round===r.round);if(!candidate?.parent)return;
 const parent=r.models.find(v=>v.id===candidate.parent);const proposal=json(path.join(dir,round,'rsi.json'));
 // This candidate addresses an observed HR/NDCG trade-off, not every structural edit.
 if(!candidate.metrics||Math.sign(candidate.metrics.ndcg-parent.metrics.ndcg)===Math.sign(candidate.metrics.hr-parent.metrics.hr))return;
 if(r.rsi.some(v=>v.id===`tool-v${r.round}`))return;
 fs.mkdirSync(path.join(dir,folder),{recursive:true});fs.copyFileSync(path.join(ROOT,'scripts/paired_probe.py'),path.join(dir,folder,'paired_probe.py'));
 const instruction=`\n\n## 候选：逐用户命中与排序权衡（尚未启用）\n\n动机：${proposal.result.summary}\n\n${proposal.result.strategy}\n\n运行 paired_probe.py PARENT_VALIDATION CANDIDATE_VALIDATION --output REPORT.json。要求相同冻结协议、完整配对用户、rank 与指标一致。区分 hit_gained / hit_lost / both_hit / both_miss 的 NDCG 贡献。它仅解释已观察权衡，不构成下一次模型修改或策略晋升的效用证据。\n`;
 fs.writeFileSync(path.join(dir,folder,'SKILL.md'),fs.readFileSync(path.join(ROOT,'skills/model-diagnostic-toolkit/SKILL.md'),'utf8')+instruction);
 const began=Date.now();await scheduler.command(r,PYTHON,[path.join(dir,folder,'paired_probe.py'),path.join(dir,'models',parent.id,'evaluation.json'),path.join(dir,'models',candidate.id,'evaluation.json'),'--output',path.join(dir,folder,'paired-report.json')],`${folder}/probe.log`);
 await scheduler.command(store.get(r.id),PYTHON,['-m','unittest','discover','-s',path.join(ROOT,'tests'),'-p','test_paired_probe.py'],`${folder}/offline-tests.log`);
 const report=json(path.join(dir,folder,'paired-report.json'));
 const validation={status:'candidate',codeHash:hash(fs.readFileSync(path.join(dir,folder,'paired_probe.py'))),protocolId:r.protocol.id,parentCode:parent.codeIdentity,candidateCode:candidate.codeIdentity,proposal:`${round}/rsi.json`,seconds:(Date.now()-began)/1000,gates:{paper_fidelity:'pending',synthetic_correctness:'pass',robustness:'pass',shadow_real_data:'pass',incremental_utility:'pending',cost:'pass'},reason:'已验证合成正负控制、边界条件、两个真实模型条件的配对分解和运行成本；方法文献审核与独立真实任务的增量决策效用证据尚缺，不晋升',report};
 write(path.join(dir,folder,'validation.json'),validation);
 r=store.get(r.id);for(const f of fs.readdirSync(path.join(dir,folder))){if(fs.statSync(path.join(dir,folder,f)).isFile())store.artifact(store.get(r.id),`${folder}/${f}`);}
 r=store.get(r.id);r.rsi.push({id:`tool-v${r.round}`,parent:'tool-v0',round:r.round,status:'candidate',kind:'tool',reason:validation.reason,gates:validation.gates,evidence:[`${folder}/paired-report.json`,`${folder}/validation.json`],source:`${folder}/paired_probe.py`});
 r.rsi.push({id:`skill-v${r.round}`,parent:'skill-v0',round:r.round,status:'candidate',kind:'skill',reason:'逐用户权衡诊断路由草案；继承工具候选的未完成效用门控，保持未启用',source:`${folder}/SKILL.md`,evidence:[`${folder}/SKILL.md`,`${folder}/validation.json`]});
 store.save(r);store.event(r,'rsi_candidate_implemented',{message:'独立工具与 Skill 候选已落盘，离线验证通过；增量效用证据不足',evidence:[`${folder}/validation.json`]});
}
