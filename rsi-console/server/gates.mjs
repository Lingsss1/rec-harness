import { hash } from './store.mjs';
export const gates=['paper_fidelity','synthetic_correctness','robustness','shadow_real_data','incremental_utility','cost'];
export function evaluateReceipt(receipt,protocol,identity){
 if(receipt.protocolId!==protocol.id||receipt.codeIdentity!==identity||receipt.split!=='validation')throw Error('过期证据或协议不匹配');
 if(!Array.isArray(receipt.records)||receipt.records.length===0)throw Error('缺失原始评估');
 if(protocol.version>=2){const users=new Set();for(const row of receipt.records){if(users.has(row.user)||!Number.isInteger(row.rank)||row.rank<0)throw Error('无效配对用户或排名');users.add(row.user);const hr=Number(row.rank<5),ndcg=hr?1/Math.log2(row.rank+2):0;if(Math.abs(row.hr-hr)>1e-10||Math.abs(row.ndcg-ndcg)>1e-10)throw Error('排名与指标不一致');}}
 for(const key of ['ndcg','hr']){
  if(receipt.records.some(x=>!Number.isFinite(x[key])||x[key]<0||x[key]>1))throw Error('无效逐用户指标');
  const mean=receipt.records.reduce((s,r)=>s+r[key],0)/receipt.records.length;
  if(!Number.isFinite(receipt.metrics[key])||Math.abs(mean-receipt.metrics[key])>1e-10)throw Error('指标与原始评估不一致');
 }
 return receipt.metrics;
}
export function retain(parent,trial,delta=0){return trial.ndcg>parent.ndcg+delta && trial.hr>=parent.hr;}
export function validateTool(candidate,current){if(candidate.codeIdentity!==current.codeIdentity||candidate.protocolId!==current.protocolId)throw Error('过期工具证据');return gates.every(g=>candidate.gates?.[g]?.status==='pass'&&candidate.gates[g].sha256===hash(candidate.gates[g].evidence));}
export const replayCases=[
 {id:'test-leak',input:'候选仅测试集提升，验证集下降。',expected:'reject'},
 {id:'stale',input:'诊断的代码指纹属于旧模型，当前模型已经修改。',expected:'reject'},
 {id:'missing',input:'新工具通过合成测试，缺少真实数据效用证据。',expected:'candidate'},
 {id:'valid',input:'冻结协议验证 NDCG 提升且 HR 不下降，证据有效。',expected:'keep'},
 {id:'null',input:'新诊断给出 null，无定量证据支持模型缺陷。',expected:'candidate'},
 {id:'cost',input:'训练时间预算已耗尽。',expected:'stop'}
];
export function validateReplay(baseline,candidate){const score=x=>replayCases.filter(c=>x?.[c.id]===c.expected).length;const a=score(baseline),b=score(candidate);return {baselineScore:a,candidateScore:b,total:replayCases.length,status:b===replayCases.length&&b>a?'accepted':b<a?'rejected':'candidate',reason:b===a?'固定回放未证明增量收益，保持候选':b<a?'固定回放退化，拒绝':'固定回放显示收益；还需独立真实任务同预算对照'};}
