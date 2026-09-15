"""Native released Taobao runner; bounded worker owns GPU and process cleanup."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='4'
import argparse,hashlib,json,sys,time,types,importlib,resource
from pathlib import Path
from functools import partial
from common import require,digest,stat_identity,object_sha,optimizer_layout,normalize_layout
R=Path(__file__).resolve().parent;T=R.parent;ROOT=T.parent
LABELS=('is_click','cart','fav','buy')
def save(p,v):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def module_setup():
 sys.path[:0]=[str(T/'optimizer-psgd/deps'),str(ROOT/'runtime/deps'),str(ROOT/'runtime/UniRank')]
 pkg=types.ModuleType('released_stream_models');pkg.__path__=[str(R),str(R/'source-snapshot')];sys.modules[pkg.__name__]=pkg
 return importlib.import_module(pkg.__name__+'.multistream_model')
def create(params,gpu,out):
 M=module_setup()
 import torch,heavyball,yaml
 from unirank.features import FeatureMap
 from unirank.pytorch.torch_utils import seed_everything
 from unirank.utils import set_logger
 data=ROOT/'data/Taobao_Action';dataset=yaml.safe_load((ROOT/'upstream/config/dataset_config.yaml').read_text())['Taobao_Action']
 for key in list(dataset):
  if key.startswith('test_'):dataset[key]=None
 for split in ('train','valid'):
  for kind in ('data','user_info','item_info'):dataset[split+'_'+kind]=str(data/split/kind)
 p=dict(params);p.update(dataset);p.update(gpu=gpu,distributed=False,seed=20262027,model_id=params['name'],model_root=str(out),drop_last=False)
 fm=FeatureMap('Taobao_Action',str(data));fm.load(str(data/'feature_map.json'),p)
 require(fm.labels==list(LABELS) and fm.features['action']['vocab_size']==17,'Original 17-action released map required')
 set_logger(p);seed_everything(20262027)
 cls=getattr(importlib.import_module('released_stream_models.'+('QFormerCross8' if p['model']=='QFormerCross8' else 'QFormerCross8NoFeedback' if p['model']=='QFormerCross8NoFeedback' else 'multistream_model')),p['model'])
 p['dense_optimizer']=partial(heavyball.PSGDKron,**params['dense_optimizer_options'])
 model=cls(fm,**p)
 return model,fm,p

def main():
 a=argparse.ArgumentParser();a.add_argument('--name',required=True);a.add_argument('--gpu',type=int,required=True);a.add_argument('--phase',choices=('smoke','formal'),required=True);args=a.parse_args()
 auth=json.loads((R/'authorization.json').read_text());params=json.loads((R/'configs.json').read_text())[args.name];params['name']=args.name
 require(auth['state']=='authorized' and args.name in auth['experiments'],'Unauthorized cell')
 for p,h in auth['sources'].items():require(digest(p)==h,'Source changed: '+p)
 for p,s in auth['input_stats'].items():require(stat_identity(p)==s,'Input stat changed')
 for p,h in auth['inputs'].items():
  if Path(p).name in ('feature_map.json','meta_data.json'):require(digest(p)==h,'Raw metadata changed')
 cache=auth['caches'][args.name]
 for kind,env in [('inductor','TORCHINDUCTOR_CACHE_DIR'),('triton','TRITON_CACHE_DIR')]:os.environ[env]=cache[kind]
 out=R/('smoke-runs' if args.phase=='smoke' else 'runs')/args.name;require(not out.exists(),'No replay');out.mkdir(parents=True)
 started=time.monotonic();steps=0;forwards=0
 save(out/'claim.json',dict(state='started',phase=args.phase,name=args.name,authorization_sha256=digest(R/'authorization.json'),automatic_retry=False))
 try:
  module_setup()
  import torch,numpy as np,heavyball
  from unirank.pytorch.dataloaders import RankDataLoader
  from unirank.metrics import evaluate_metrics
  torch.set_num_threads(4);torch.cuda.set_device(args.gpu)
  require(not torch.backends.cuda.matmul.allow_tf32,'Native TF32 policy changed')
  model,fm,p=create(params,args.gpu,out)
  named=dict(model.named_parameters());layout=optimizer_layout(model)
  require(isinstance(model.dense_optimizer,heavyball.PSGDKron) and model.dense_optimizer.precond_schedule==1.0,'PSGD every step required')
  require(isinstance(model.sparse_optimizer,torch.optim.Adagrad),'Sparse Adagrad required')
  require(all(g['lr']==1e-4 and g['weight_decay']==0 for g in model.dense_optimizer.param_groups),'Dense settings changed')
  require(all(g['lr']==.05 for g in model.sparse_optimizer.param_groups),'Sparse LR changed')
  gate='unified_qformer.router.logits'
  def tsha(t):return hashlib.sha256(t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
  init={n:dict(shape=list(v.shape),sha256=tsha(v)) for n,v in named.items()}
  original={n:v for n,v in init.items() if n!=gate};initsha=object_sha(init)
  if args.phase=='formal':
   review=json.loads((R/'reports/smoke-review.json').read_text());require(review['state']=='passed' and review['initializers'][args.name]==initsha,'Own smoke initializer differs')
  save(out/'initial-identity.json',dict(parameters=init,sha256=initsha,original_parameters_sha256=object_sha(original),layout=normalize_layout(layout)))
  save(out/'manifest.json',dict(params={**params,'gpu':args.gpu},sources=auth['sources'],inputs=auth['inputs'],input_stat_snapshot=auth['input_stats'],loader='unirank.pytorch.dataloaders.UniRankDataloader',action_vocab=17,history_transform='none_added',test_evaluated=False,initial_sha256=initsha,parameters=sum(v.numel() for v in named.values()),precision='BF16;TF32off',input_hash_scope=auth['input_hash_scope']))
  # RankDataLoader uses its original release behavior; no maturity adapter imported.
  loaderparams=dict(p)
  if args.phase=='smoke':loaderparams['valid_data']=None
  train,valid=RankDataLoader(fm,stage='train',**loaderparams).make_iterator()
  require(train.num_samples==17822296 and len(train)==2193,'Native training budget changed')
  counts=dict(dense=0,sparse=0);hooks=[]
  for role in counts:
   def count(o,a,k,role=role):counts[role]+=1
   hooks.append(getattr(model,role+'_optimizer').register_step_post_hook(count))
  native_forward=model.forward
  def counted(*a,**kw):
   nonlocal forwards
   forwards+=1;return native_forward(*a,**kw)
  model.forward=counted
  def gradcheck():
   for n,v in named.items():
    g=v.grad
    if g is not None:require(bool(torch.isfinite(g.coalesce().values() if g.is_sparse else g).all()),'Nonfinite gradient: '+n)
  nativeclip=model._clip_dense_gradients
  def clip():gradcheck();v=nativeclip();gradcheck();return v
  if args.phase=='smoke':model._clip_dense_gradients=clip
  torch.cuda.reset_peak_memory_stats(args.gpu)
  if args.phase=='smoke':
   train.set_epoch(0);iterator=iter(train);events=[];parity=None
   try:
    for i in range(2):
     batch=next(iterator)
     if i==0 and args.name!='cross8':
      model.eval();stage=model.unified_qformer
      with torch.no_grad(),model._get_amp_context():
       native=model(batch)
       from released_stream_models.QFormerCross8NoFeedback import NoFeedbackStage
       old=stage.__dict__.get('forward_components');had='forward_components' in stage.__dict__
       stage.forward_components=types.MethodType(NoFeedbackStage.forward_components,stage)
       try:reference=model(batch)
       finally:
        if had:stage.forward_components=old
        else:del stage.forward_components
      require(all(torch.equal(native[k],reference[k]) for k in native),'BF16 native no-feedback parity failed');parity=dict(rows=len(batch[0]['is_click']),exact=True)
     model.train();model._batch_index=i;model._max_gradient_norm=1;model._optimizer_zero_grad()
     watched=[n for n in named if n==gate or n.endswith('cross_attn.w_q.weight') or n=='embedding_layer.embedding_layer.embedding_layers.action.weight']
     before={n:named[n].detach().clone() for n in watched};heads=[list(x.parameters())[-1].detach().clone() for x in model.tower]
     loss=model.train_step(batch);steps+=1;gradcheck()
     changes={n:int(torch.count_nonzero(before[n]!=named[n])) for n in watched}
     require(bool(torch.isfinite(loss)) and all(torch.any(list(t.parameters())[-1]!=b) for t,b in zip(model.tower,heads)),'Nonfinite loss or inactive task head')
     require(gate not in named or changes[gate]>0,'Router did not update')
     require(any(v>0 for k,v in changes.items() if 'cross_attn' in k),'Native attention did not update')
     events.append(dict(step=steps,loss=float(loss.detach()),changes=changes,all_heads_updated=True,finite_gradients=True))
   finally:
    if hasattr(iterator,'close'):iterator.close()
   require(counts==dict(dense=2,sparse=2),'Smoke optimizer count mismatch')
   result=dict(smoke_only=True,steps=events,optimizer_steps=steps,forwards=forwards,native_parity=parity)
  else:
   require(valid.num_samples==2871767,'Validation rows changed')
   native_step=model.train_step
   def step(batch):
    nonlocal steps
    require(steps<2193,'Extra update prohibited');loss=native_step(batch);steps+=1
    require(bool(torch.isfinite(loss)),'Nonfinite training loss')
    if steps==1 or steps%25==0:save(out/'progress.json',dict(steps=steps,total_steps=2193,loss=float(loss.detach()),elapsed_seconds=time.monotonic()-started))
    return loss
   model.train_step=step;fitstart=time.monotonic();model.fit(train,validation_data=valid,**p);fitwall=time.monotonic()-fitstart
   require(steps==2193 and counts==dict(dense=2193,sparse=2193),'Native update count mismatch')
   pred={t:[] for t in LABELS};truth={t:[] for t in LABELS};ids={k:[] for k in ('user_id','item_id','seq_len')};model.eval()
   with torch.no_grad(),model._get_amp_context():
    for batch in valid:
     v=model(batch)
     for t in LABELS:pred[t].append(v[t+'_pred'].float().cpu().numpy().ravel());truth[t].append(batch[0][t].cpu().numpy().ravel())
     for k,src in [('user_id','user_id'),('item_id','item_index'),('seq_len','seq_len')]:ids[k].append(batch[0][src].cpu().numpy().ravel())
   arrays={k:np.concatenate(v) for k,v in ids.items()};metrics={}
   for t in LABELS:
    x=np.concatenate(pred[t]);y=np.concatenate(truth[t]);require(len(x)==2871767 and np.isfinite(x).all(),'Prediction export invalid')
    arrays['prediction_'+t]=x;arrays['label_'+t]=y;metrics[t]=evaluate_metrics(y,x,['AUC','logloss'],arrays['user_id'])
   np.savez_compressed(out/'validation_predictions.npz',**arrays)
   require(forwards==2193+2*len(valid),'Unexpected forward count')
   result=dict(smoke_only=False,validation=metrics,macro_AUC=float(np.mean([x['AUC'] for x in metrics.values()])),macro_logloss=float(np.mean([x['logloss'] for x in metrics.values()])),optimizer_steps=steps,validation_rows=2871767,train_rows=17822296,forwards=forwards,fit_wall_seconds_including_validation=fitwall,checkpoint=model.checkpoint,predictions=str(out/'validation_predictions.npz'))
  for hk in hooks:hk.remove()
  require(optimizer_layout(model)==layout,'Optimizer layout changed')
  for path,h in auth['sources'].items():require(digest(path)==h,'Source changed during run')
  for path,s in auth['input_stats'].items():require(stat_identity(path)==s,'Input changed during run')
  result.update(parameter_count=sum(x.numel() for x in named.values()),initial_sha256=initsha,original_parameters_sha256=object_sha(original),test_evaluated=False,action_vocab=17,wall_seconds=time.monotonic()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated(args.gpu),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
  save(out/('smoke.json' if args.phase=='smoke' else 'result.json'),result)
  artifacts=[f for f in out.iterdir() if f.is_file()]
  if args.phase=='formal':artifacts.append(Path(model.checkpoint))
  save(out/'completion.json',dict(state='completed',name=args.name,phase=args.phase,artifacts={str(p):digest(p) for p in artifacts},optimizer_steps=steps,automatic_retry=False))
  print(json.dumps(dict(state='completed',name=args.name,phase=args.phase,steps=steps,macro_AUC=result.get('macro_AUC')),allow_nan=False),flush=True)
 except BaseException as exc:
  save(out/'failure.json',dict(state='failed',error=repr(exc),steps=steps,forwards=forwards,wall_seconds=time.monotonic()-started,automatic_retry=False));raise
if __name__=='__main__':main()
