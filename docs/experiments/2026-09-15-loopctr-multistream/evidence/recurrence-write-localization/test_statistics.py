import json,importlib.util,time
from pathlib import Path
import numpy as np
import paired_core as core
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('accepted',R.parent/'recurrence-task-effects/run_bootstrap.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
start=time.monotonic();users=np.arange(4);groups=np.repeat(users,4);labels=np.tile([0,1,0,1],4);arrays=[]
for m in range(10):
 a={}
 for j,t in enumerate(core.LABELS):a['prediction_'+t]=(.1+.8*((np.arange(16)*(m+1)+j)%17)/17).astype(np.float32);a['label_'+t]=labels
 arrays.append(a)
s=core.shared_bootstrap(arrays,users,groups,*h.original_auc_functions(R.parents[1]/'compare_runs.py'),h.bce_rows,repetitions=7,seed=20260906)
rng=np.random.default_rng(20260906)
for draw in range(7):
 sampled=rng.integers(4,size=4);rows=np.concatenate([np.flatnonzero(groups==u) for u in sampled])
 for m,a in enumerate(arrays):
  for j,t in enumerate(core.LABELS):
   p=a['prediction_'+t][rows];y=labels[rows];positive=p[y==1];negative=p[y==0]
   auc=np.mean((positive[:,None]>negative).astype(float)+.5*(positive[:,None]==negative));np.testing.assert_allclose(s['draw_auc'][draw,m,j],auc,rtol=0,atol=1e-12)
   p=np.clip(p,np.finfo(p.dtype).eps,1-np.finfo(p.dtype).eps).astype(np.float64);bce=-(y*np.log(p)+(1-y)*np.log1p(-p)).mean();np.testing.assert_allclose(s['draw_bce'][draw,m,j],bce,rtol=0,atol=1e-12)
names,c=core.comparison_spec(json.loads((R/'analysis-protocol.json').read_text()));c=np.asarray(c);assert c.shape==(45,10)
point=core.effect_metrics(s['point_auc'],s['point_bce'],c);assert point.shape==(45,21)
for k,(_,coef) in enumerate(core.EXPECTED_EFFECTS):
 pos=core.MODEL_ORDER.index(next(n for n,v in coef.items() if v==1));neg=core.MODEL_ORDER.index(next(n for n,v in coef.items() if v==-1))
 np.testing.assert_allclose(point[k,0],(s['point_auc'][pos]-s['point_auc'][neg]).mean(),rtol=0,atol=1e-12)
print(json.dumps(dict(state='passed',expanded_user_oracle_draws=7,models=10,effects=45,metrics_per_effect=21,wall_seconds=time.monotonic()-start)))
