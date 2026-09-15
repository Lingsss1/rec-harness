"""Append eight actual outcomes once; refresh lineage after memory commit."""
from pathlib import Path
import json,hashlib,datetime,csv,io,fcntl,subprocess,sys
R=Path(__file__).resolve().parent;M=R.parent/'memory'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def main():
 out=R/'reports/closeout';assert not out.exists();out.mkdir()
 pc=load(R/'reports/paired-analysis/completion.json');assert pc['state']=='completed' and sha(R/'reports/paired-analysis/result.json')==pc['result_sha256']
 stats=load(R/'reports/paired-analysis/result.json');lat=load(R/'reports/latency/completion.json');assert lat['state']=='completed'
 paths={n:Path(p) for n,p in load(R/'analysis-runs.json').items()};res={n:load(p/'result.json') for n,p in paths.items()};names=list(load(R/'configs.json'));parent=res['dynamic_write'];effects=stats['effects']
 lines=['# 动态写入轮次定位结果','', '原始Cross8发布数据、17动作、width256、q8、两轮、PSGDKron每步、clip1、固定seed20262027。八个新训练；Cross8与dynamic_write为上轮固定保存参照，不计新增训练。','', '| 模型 | Macro AUC | Δ dynamic_write | 95% 配对用户区间 | 推理 ms/1024 |','|---|---:|---:|---|---:|']
 for n,v in res.items():
  ci=effects.get(n+'__minus__dynamic_write',{}).get('metrics',{}).get('macro_auc',{}).get('ci95')
  lines.append(f"| {n} | {v['macro_AUC']:.9f} | {v['macro_AUC']-parent['macro_AUC']:+.9f} | {ci if ci else '参照'} | {lat['measurements'][n]['mean_ms']:.3f} |")
 lines+=['','完整45个对照、945个区间见reports/paired-analysis/result.json。500次配对用户重采样；单训练seed，未做多重校正，不表示训练随机性范围。','', '只将本轮干预下的实际差值作为证据；first_only/second_only控制动态部分，静态偏置仍保留；同参数布局不等于有效容量相同。不自动提升全局最优，不跨mature协议比较。']
 (R/'RESULTS.md').write_text('\n'.join(lines)+'\n')
 lock=(M/'.write-localization-memory.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 prior=load(R/'reports/memory-start.json');before={n:sha(M/n) for n in prior['after_sha256']};assert before==prior['after_sha256'],'Concurrent memory edit; results preserved, merge needed'
 backup=out/'memory-before';backup.mkdir()
 for n in before:(backup/n).write_bytes((M/n).read_bytes())
 existing=[json.loads(x) for x in (M/'episodes.jsonl').read_text().splitlines()];ids={e['experiment_id'] for e in existing};episodes=[];rows=[];fields=next(csv.reader(io.StringIO((M/'results.tsv').read_text()),delimiter='\t'))
 for n in names:
  ident='write-localization--'+n+'--s20262027';assert ident not in ids
  v=res[n];e=load(R/'reports'/n/'formal/exit.json');assert e['exit_code']==0 and e['cleanup']['completed']
  effect=effects[n+'__minus__dynamic_write'];ci=effect['metrics']['macro_auc']['ci95'];delta=v['macro_AUC']-parent['macro_AUC'];status='discard' if ci[1]<0 else 'provisional';cost=e['cost']
  episode=dict(experiment_id=ident,parent_experiment_id='released-multistream--dynamic_write--s20262027',comparison_parent_ids=['released-multistream--dynamic_write--s20262027','released-multistream--cross8--s20262027'],timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),dataset_id='taobao-unirank-release-v1',experiment_type='formal_training',family='write_localization',code_id=sha(R/'multistream_model.py'),hypothesis=n,change=n,input_protocol_id='original-released-cross8-action17-width256-q8',budget=dict(formal_training_runs=1,seeds=[20262027],new_training_seeds=[],optimizer_steps=2193),metrics=dict(primary=dict(name='macro_AUC',value=v['macro_AUC'],delta=delta),per_task=v['validation'],matched_parent_effect=effect,forward_latency=lat['measurements'][n]),decision=status,decision_reason='Matched exploratory user-cluster interval; no automatic global promotion',validity=dict(valid=True,test_evaluated=False),confidence='single_seed_exploratory_paired_users_no_multiplicity_correction',source_identity=dict(result=str(paths[n]/'result.json'),result_sha256=sha(paths[n]/'result.json'),paired=str(R/'reports/paired-analysis/result.json')),global_best_changed=False,diagnostic_task_id=None)
  episodes.append(episode);rows.append(dict(experiment_id=ident,parent_id=episode['parent_experiment_id'],code_id=episode['code_id'],primary_metric=v['macro_AUC'],primary_delta=delta,guardrails=json.dumps(v['validation']),peak_vram_mb=cost['gpu_peak_bytes']/2**20,runtime_sec=cost['wall_seconds'],status=status,hypothesis=n,change=n))
 buf=io.StringIO();csv.DictWriter(buf,fieldnames=fields,delimiter='\t',lineterminator='\n').writerows(rows)
 prepared={n:(M/n).read_bytes() for n in before};prepared['episodes.jsonl']+=(''.join(json.dumps(e,ensure_ascii=False)+'\n' for e in episodes)).encode();prepared['results.tsv']+=buf.getvalue().encode()
 strategy=load(M/'strategy.json');strategy['completed_training_runs']+=8;strategy['last_consolidated_episode']=episodes[-1]['experiment_id'];strategy['phase']='results_pending_interpretation';strategy['latest_write_localization_outcome']=dict(report=str(R/'RESULTS.md'),completed=8,best_new_by_point=max(names,key=lambda n:res[n]['macro_AUC']),global_best_changed=False)
 for q in strategy['next_queue']:
  if q.get('family')=='write_localization':q.update(state='completed',report=str(R/'RESULTS.md'))
 prepared['strategy.json']=(json.dumps(strategy,ensure_ascii=False,indent=2)+'\n').encode()
 for n,b in prepared.items():
  assert sha(M/n)==before[n];p=M/(n+'.tmp');p.write_bytes(b);p.replace(M/n)
 save(out/'memory-completion.json',dict(state='completed',before_sha256=before,after_sha256={n:sha(M/n) for n in before},formal_training_completed=strategy['completed_training_runs'],episodes=len(existing)+8,global_best_changed=False,report_sha256=sha(R/'RESULTS.md')))
 subprocess.run([sys.executable,'-B',str(R.parents[1]/'lineage/update.py')],check=True)
 save(out/'completion.json',dict(state='completed_pending_interpretation',lineage_sha256=sha(R.parents[1]/'lineage/experiments.json')))
if __name__=='__main__':main()
