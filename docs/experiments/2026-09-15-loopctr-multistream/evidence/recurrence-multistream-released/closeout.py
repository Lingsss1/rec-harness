from pathlib import Path
import json,hashlib,datetime,csv,io,fcntl
R=Path('/home/liuzhishan/lhr/qformer-loop/taobao-transfer/recurrence-multistream-released');M=R.parent/'memory'
load=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def main():
 out=R/'reports/closeout';assert not out.exists();out.mkdir()
 pc=load(R/'reports/paired-analysis/completion.json');assert pc['state']=='completed' and sha(R/'reports/paired-analysis/result.json')==pc['result_sha256']
 st=load(R/'reports/paired-analysis/result.json');lat=load(R/'reports/latency/completion.json');assert lat['state']=='completed'
 names=load(R/'authorization.json')['experiments'];res={n:load(R/'runs'/n/'result.json') for n in names};base=res['cross8'];effects=st['effects']
 rows=['# 原始 Cross8 双流信息流实验','', '固定原始发布数据、17 个动作 ID、宽度 256、8 个查询槽、seed 20262027、PSGDKron 每步更新、clip1；不用 mature。八组各训练一轮 2193 步，无额外训练 seed。','', '| 模型 | Macro AUC | 相对 Cross8 | 95% 配对用户区间 | 推理 ms/1024 |','|---|---:|---:|---|---:|']
 for n in names:
  ci=effects[n+'__minus__cross8']['metrics']['macro_auc']['ci95'] if n!='cross8' else None
  rows.append(f"| {n} | {res[n]['macro_AUC']:.9f} | {res[n]['macro_AUC']-base['macro_AUC']:+.9f} | {str(ci) if ci else '基线'} | {lat['measurements'][n]['mean_ms']:.3f} |")
 rows+=['','500 次配对用户 bootstrap，268201 用户、2871767 行；区间仅衡量该训练 seed 下验证用户抽样不确定性，不包含训练随机性；28 组比较未做多重校正。','', '动态写入是本轮最高点估计，四项 AUC 均高于 Cross8，四项 logloss 均下降；不自动提升为全局最优。静态、动态读取、单独混合和全部动态是独立训练，不能仅凭分数将差异完全归因于推理时的信息增量。','', '下一步优先对动态写入做第一轮/第二轮分别关闭与共享写入对照，并在保存模型上检查两流差异和写入门控的任务、历史长度分布；关闭干预只能提供依赖证据，仍需训练消融验证。保持原始数据协议、固定 seed。','', '机制限制与下一轮具体消融见 MECHANISM_NOTES.md：动态写入仍包含静态读取/旁路参数；收益也可能是残差更新幅度调节，不能直接证明双流记忆有效。', '', '完整 pairwise 结果：reports/paired-analysis/result.json；延迟：reports/latency/completion.json。父调度进程提前消失原因未确认；八个模型均有 exit0、清理完成和完整产物校验，恢复仅补汇总并首次执行统计，没有重跑训练。']
 (R/'RESULTS.md').write_text('\n'.join(rows)+'\n')
 lock=(M/'.multistream-phase.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 prior=load(R/'reports/memory-formal-start.json');before={n:sha(M/n) for n in prior['after_sha256']};assert before==prior['after_sha256'],'Concurrent memory update'
 backup=out/'memory-before';backup.mkdir()
 for n in before:(backup/n).write_bytes((M/n).read_bytes())
 strategy=load(M/'strategy.json');existing=[json.loads(s) for s in (M/'episodes.jsonl').read_text().splitlines()];ids={e['experiment_id'] for e in existing};episodes=[];ledger=[]
 fields=next(csv.reader(io.StringIO((M/'results.tsv').read_text()),delimiter='\t'))
 for n in names:
  ident=f'released-multistream--{n}--s20262027';assert ident not in ids
  v=res[n];delta=v['macro_AUC']-base['macro_AUC'];effect=effects.get(n+'__minus__cross8');ci=effect['metrics']['macro_auc']['ci95'] if effect else None
  decision='discard' if ci and ci[1]<0 else 'provisional'
  e=dict(experiment_id=ident,parent_experiment_id=None if n=='cross8' else 'released-multistream--cross8--s20262027',timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),dataset_id='taobao-unirank-release-v1',experiment_type='formal_training',family='released_cross8_multistream',code_id=sha(R/'multistream_model.py'),hypothesis='Test separate work streams and input-dependent read/write/bypass routing under the original Cross8 protocol',change=n,input_protocol_id='original-released-cross8-action17-width256-q8',budget=dict(formal_training_runs=1,seeds=[20262027],new_training_seeds=[],optimizer_steps=2193),metrics=dict(primary=dict(name='macro_AUC',value=v['macro_AUC'],delta=delta),per_task=v['validation'],matched_parent_effect=effect,forward_latency=lat['measurements'][n]),decision=decision,decision_reason='Baseline reference' if n=='cross8' else 'Exploratory matched comparison; no global incumbent promotion',validity=dict(valid=True,test_evaluated=False,source_hashes_verified=True),confidence='single_seed_exploratory_paired_users_no_multiplicity_correction',source_identity=dict(result=str(R/'runs'/n/'result.json'),result_sha256=sha(R/'runs'/n/'result.json'),paired=str(R/'reports/paired-analysis/result.json')),global_best_changed=False)
  episodes.append(e);cost=load(R/'reports'/n/'formal/exit.json')['cost'];ledger.append(dict(experiment_id=ident,parent_id=e['parent_experiment_id'] or '',code_id=e['code_id'],primary_metric=v['macro_AUC'],primary_delta=delta,guardrails=json.dumps(v['validation']),peak_vram_mb=cost['gpu_peak_bytes']/2**20,runtime_sec=cost['wall_seconds'],status=decision,hypothesis=e['hypothesis'],change=n))
 buf=io.StringIO();writer=csv.DictWriter(buf,fieldnames=fields,delimiter='\t',lineterminator='\n');writer.writerows(ledger)
 prepared={n:(M/n).read_bytes() for n in before}
 prepared['episodes.jsonl']+=(''.join(json.dumps(e,ensure_ascii=False)+'\n' for e in episodes)).encode();prepared['results.tsv']+=buf.getvalue().encode()
 strategy['completed_training_runs']+=8;strategy['last_consolidated_episode']=episodes[-1]['experiment_id'];strategy['phase']='explore';strategy['latest_multistream_outcome']=dict(directory=str(R),completed=8,best='dynamic_write',macro_auc=res['dynamic_write']['macro_AUC'],delta_to_cross8=res['dynamic_write']['macro_AUC']-base['macro_AUC'],effect=effects['dynamic_write__minus__cross8']['metrics']['macro_auc'],global_best_changed=False,next='Separate first-round and second-round writing; retain original released protocol')
 for q in strategy['next_queue']:
  if q.get('family')=='released_cross8_multistream':q.update(state='completed',report=str(R/'RESULTS.md'))
 prepared['strategy.json']=(json.dumps(strategy,ensure_ascii=False,indent=2)+'\n').encode()
 failures=[dict(failure_id='released-multistream-handoff-preclaim',family='released_cross8_multistream',phase='orchestration_before_claim',error='Unexpected queued smoke already started',evidence=str(R/'reports/handoff-preclaim-incident.json'),resolved=True,training_restarts=0),dict(failure_id='released-multistream-parent-disappeared',family='released_cross8_multistream',phase='orchestration_after_dispatch',error='Scheduler and postprocess waiter absent; underlying cause unknown',evidence=str(R/'reports/orchestration-recovery.json'),resolved=True,training_restarts=0)]
 prepared['failures.jsonl']+=(''.join(json.dumps(f)+'\n' for f in failures)).encode()
 # Keep lessons unchanged: a single exploratory seed is not independent mechanism confirmation.
 for n,data in prepared.items():
  assert sha(M/n)==before[n];p=M/(n+'.tmp');p.write_bytes(data);p.replace(M/n)
 save(out/'completion.json',dict(state='completed',before_sha256=before,after_sha256={n:sha(M/n) for n in before},formal_training_completed=strategy['completed_training_runs'],episodes=len(existing)+8,lessons_changed=False,global_best_changed=False,report_sha256=sha(R/'RESULTS.md')))
 print(json.dumps(strategy['latest_multistream_outcome']))
if __name__=='__main__':main()
