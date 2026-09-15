"""All trained contrasts from saved arrays, accepted user-cluster equations."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='4'
import json,importlib.util,time,resource,signal
from pathlib import Path
import numpy as np
from common import require,digest
import paired_core as core
R=Path(__file__).resolve().parent

def load(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
 out=R/'reports/paired-analysis';require(not out.exists(),'No statistics replay');out.mkdir()
 start=time.monotonic();signal.signal(signal.SIGALRM,lambda s,f:(_ for _ in ()).throw(TimeoutError('Statistics wall limit')));signal.alarm(1800)
 helperpath=R.parent/'recurrence-task-effects/run_bootstrap.py';comparepath=R.parents[1]/'compare_runs.py'
 require(digest(helperpath)=='f1bf123b463d24000789290c488c67ede29e48749871b4fb4cc13b787c9433c9' and digest(comparepath)=='640eb5ba7c346013ea26af93303b384596bf6bd9569905fcdf7b1b3992c26df0','Accepted helper changed')
 spec=importlib.util.spec_from_file_location('user_helper',helperpath);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 protocol=load(R/'analysis-protocol.json');names,coef=core.comparison_spec(protocol);coef=np.asarray(coef)
 files={str(p):digest(p) for p in [Path(__file__).resolve(),R/'paired_core.py',R/'analysis-protocol.json',helperpath,comparepath]}
 arrays=[];results={};identity=[]
 try:
  for name in core.MODEL_ORDER:
   run=R/'runs'/name;e=load(R/'reports'/name/'formal/exit.json');c=load(run/'completion.json');v=load(run/'result.json');m=load(run/'manifest.json')
   require(e['exit_code']==0 and e['cleanup']['completed'] and c['state']=='completed' and v['optimizer_steps']==2193 and v['action_vocab']==17,'Formal admission failed')
   for p in [run/'result.json',run/'validation_predictions.npz',run/'manifest.json']:
    h=digest(p);require(c['artifacts'][str(p)]==h,'Result identity changed');files[str(p)]=h
   arrays.append(helper.load_npz(run/'validation_predictions.npz'));results[name]=v;identity.append(v['original_parameters_sha256'])
   require(m['inputs']==load(R/'authorization.json')['inputs'],'Original input files differ')
  require(len(set(identity))==1,'Original parameter initialization differs')
  users,groups,ids=helper.paired_arrays(arrays[0],arrays[0])
  for a in arrays[1:]:helper.paired_arrays(arrays[0],a)
  require(len(users)==268201 and len(groups)==2871767,'Full validation population changed')
  save(out/'claim.json',dict(state='started',sources=files,models=core.MODEL_ORDER,effects=28,draws=500,seed=20260906))
  def progress(v):
   if v['stage']=='observed':
    for i,n in enumerate(core.MODEL_ORDER):
     for j,t in enumerate(core.LABELS):require(abs(v['observed_auc'][i,j]-results[n]['validation'][t]['AUC'])<1e-12,'Source AUC differs')
    print(json.dumps(dict(stage='source_auc_verified')),flush=True)
   else:print(json.dumps(v),flush=True)
  stats=core.shared_bootstrap(arrays,users,groups,*helper.original_auc_functions(comparepath),helper.bce_rows,progress=progress)
  observed=core.effect_metrics(stats['point_auc'],stats['point_bce'],coef);draws=core.effect_metrics(stats['draw_auc'],stats['draw_bce'],coef);ci=np.quantile(draws,[.025,.975],axis=0)
  effects={n:dict(coefficients=dict(zip(core.MODEL_ORDER,coef[i].tolist())),metrics={m:dict(estimate=float(observed[i,j]),ci95=ci[:,i,j].tolist()) for j,m in enumerate(core.METRIC_ORDER)}) for i,n in enumerate(names)}
  np.savez_compressed(out/'draws.npz',**{k:v for k,v in stats.items() if isinstance(v,np.ndarray)},effects=draws)
  save(out/'result.json',dict(effects=effects,intervals=588,rows=len(groups),users=len(users),draws=500,draw_sha256=stats['resample_indices_sha256'],model_results=results,uncertainty=protocol['scope']))
  require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<16*2**30,'Statistics RSS limit exceeded')
  for p,h in files.items():require(digest(p)==h,'Analysis input changed')
  save(out/'completion.json',dict(state='completed',files=files,result_sha256=digest(out/'result.json'),draws_sha256=digest(out/'draws.npz'),wall_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,model_forwards=0,optimizer_updates=0))
 except BaseException as exc:save(out/'failure.json',dict(state='failed',error=repr(exc),automatic_retry=False));raise
 finally:signal.alarm(0)
if __name__=='__main__':main()
