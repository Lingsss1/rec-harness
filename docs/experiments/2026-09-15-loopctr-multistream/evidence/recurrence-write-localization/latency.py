"""Same original validation batch, same physical GPU; saved checkpoints only."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='4'
import argparse,fcntl,json,time,gc,signal,resource
from pathlib import Path
from common import require,digest,stat_identity
from process_utils import gpu_snapshot
from runner import create,save,module_setup
R=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=4);a=p.parse_args();out=R/'reports/latency';require(not out.exists(),'No latency replay');out.mkdir()
 auth=json.loads((R/'authorization.json').read_text());configs=json.loads((R/'latency-configs.json').read_text());start=time.monotonic()
 lock=open('/tmp/qformer-latency-gpu-'+str(a.gpu)+'.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 snap=gpu_snapshot(a.gpu,10,idle=True);signal.signal(signal.SIGALRM,lambda s,f:(_ for _ in ()).throw(TimeoutError('Latency1800s limit')));signal.alarm(1800)
 measurements={};batch=None;calls=0
 try:
  module_setup()
  import torch,numpy as np
  from unirank.pytorch.dataloaders import UniRankDataloader
  torch.set_num_threads(4);torch.cuda.set_device(a.gpu);require(not torch.backends.cuda.matmul.allow_tf32,'TF32 changed')
  for name,params in configs.items():
   run=Path(json.loads((R/'analysis-runs.json').read_text())[name]);v=json.loads((run/'result.json').read_text());cp=Path(v['checkpoint']);c=json.loads((run/'completion.json').read_text())
   require(c['state']=='completed' and c['artifacts'][str(cp)]==digest(cp),'Checkpoint identity differs')
   cache=auth['measurement_caches'][name]
   for k,env in [('inductor','TORCHINDUCTOR_CACHE_DIR'),('triton','TRITON_CACHE_DIR')]:os.environ[env]=cache[k]
   local=out/name;local.mkdir();model,fm,full=create(dict(params,name=name),a.gpu,local)
   model.load_state_dict(torch.load(cp,map_location=model.device,weights_only=True));model.eval()
   if batch is None:
    data=R.parents[1]/'data/Taobao_Action';loader=UniRankDataloader(fm,str(data/'valid/data'),user_info=str(data/'valid/user_info'),item_info=str(data/'valid/item_info'),split='valid',batch_size=1024,shuffle=False,num_workers=0,max_len=100,padding='pre',distributed=False,drop_last=False)
    iterator=iter(loader)
    try:batch=next(iterator)
    finally:
     if hasattr(iterator,'close'):iterator.close()
    require(len(batch[0]['is_click'])==1024,'Latency batch length differs')
    import hashlib
    h=hashlib.sha256()
    def move(x):
     if isinstance(x,torch.Tensor):h.update(str((x.dtype,list(x.shape))).encode());h.update(x.contiguous().numpy().tobytes());return x.to(model.device)
     if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
     if isinstance(x,tuple):return tuple(move(v) for v in x)
     if isinstance(x,list):return [move(v) for v in x]
     raise TypeError(type(x))
    batch=move(batch);batch_hash=h.hexdigest();del loader
   torch.cuda.reset_peak_memory_stats(a.gpu);times=[]
   with torch.no_grad(),model._get_amp_context():
    for _ in range(10):model(batch);calls+=1
    torch.cuda.synchronize(a.gpu)
    for _ in range(40):
     left=torch.cuda.Event(enable_timing=True);right=torch.cuda.Event(enable_timing=True);left.record();model(batch);right.record();right.synchronize();times.append(left.elapsed_time(right));calls+=1
   measurements[name]=dict(mean_ms=float(np.mean(times)),p50_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)),timing_ms=times,batch_size=1024,warmups=10,iterations=40,parameter_count=sum(p.numel() for p in model.parameters()),peak_cuda_bytes=torch.cuda.max_memory_allocated(a.gpu),checkpoint_sha256=digest(cp),batch_sha256=batch_hash)
   save(out/'progress.json',dict(measurements=measurements,forwards=calls));del model;gc.collect();torch.cuda.empty_cache();gpu_snapshot(a.gpu,10,expected_uuid=snap['uuid'])
  require(calls==500,'Latency call budget differs')
  save(out/'completion.json',dict(state='completed',measurements=measurements,gpu=snap,model_forwards=calls,training_updates=0,wall_seconds=time.monotonic()-start,scope='Original released first1024 validation rows, same GPU, model-only CUDA timing excludes constructors/checkpoint/IO/transfers',peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
 except BaseException as exc:save(out/'failure.json',dict(state='failed',error=repr(exc),forwards=calls,automatic_retry=False));raise
 finally:signal.alarm(0);lock.close()
if __name__=='__main__':main()
