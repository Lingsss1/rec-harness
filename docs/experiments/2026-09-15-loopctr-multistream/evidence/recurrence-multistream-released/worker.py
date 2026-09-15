"""One cell, owned process-tree cleanup, no retry or unrelated process control."""
import argparse,fcntl,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
from common import require,digest
from process_utils import gpu_snapshot,enable_subreaper,process_tree_snapshot,cleanup_owned_processes
R=Path(__file__).resolve().parent

def save(p,v):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--phase',choices=('smoke','formal'),required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
 auth=json.loads((R/'authorization.json').read_text());require(a.name in auth['experiments'] and a.gpu in auth['gpu_pool'],'Cell/GPU authorization differs')
 out=R/'reports'/a.name/a.phase;require(not out.exists(),'No worker replay');out.mkdir(parents=True)
 start=time.monotonic();child=None;code=126;error=None;limits=auth['limits'][a.phase];cleanup=None;cost=dict(wall_seconds=0,cpu_seconds=0,peak_rss_bytes=0,gpu_peak_bytes=0);base=resource.getrusage(resource.RUSAGE_CHILDREN)
 save(out/'claim.json',dict(pid=os.getpid(),name=a.name,phase=a.phase,gpu=a.gpu,authorization_sha256=digest(R/'authorization.json')))
 def interrupted(s,f):raise InterruptedError('Worker interrupted '+str(s))
 for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,interrupted)
 enable_subreaper()
 lock=open('/tmp/qformer-latency-gpu-'+str(a.gpu)+'.lock','a')
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);snap=gpu_snapshot(a.gpu,10,idle=True)
  for path,h in auth['sources'].items():require(digest(path)==h,'Frozen source changed')
  if a.phase=='formal':require(json.loads((R/'reports/smoke-review.json').read_text())['state']=='passed','All smokes must pass first')
  command=[sys.executable,'-B',str(R/'runner.py'),'--name',a.name,'--phase',a.phase,'--gpu',str(a.gpu)]
  with (out/'runner.log').open('x') as log:
   child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   while True:
    ret=child.poll();tree=process_tree_snapshot();usage=resource.getrusage(resource.RUSAGE_CHILDREN);g=gpu_snapshot(a.gpu,10,expected_uuid=snap['uuid'])
    cost=dict(wall_seconds=time.monotonic()-start,cpu_seconds=tree['cpu_seconds']+usage.ru_utime+usage.ru_stime-base.ru_utime-base.ru_stime,peak_rss_bytes=max(cost['peak_rss_bytes'],tree['rss_bytes'],usage.ru_maxrss*1024),gpu_peak_bytes=max(cost['gpu_peak_bytes'],g['memory_used_bytes']))
    save(out/'status.json',dict(state='running' if ret is None else 'runner_exited',name=a.name,gpu=a.gpu,runner_pid=child.pid,returncode=ret,cost=cost))
    require(all(cost[k]<=limits[k] for k in cost),'Sampled resource budget exceeded')
    if ret is not None:code=ret;break
    time.sleep(2)
  require(code==0,'Runner failed')
  run=R/('smoke-runs' if a.phase=='smoke' else 'runs')/a.name;completion=json.loads((run/'completion.json').read_text())
  require(completion['state']=='completed' and completion['optimizer_steps']==(2 if a.phase=='smoke' else 2193),'Incomplete native run')
  for path,h in completion['artifacts'].items():require(digest(path)==h,'Saved run artifact changed')
  for path,h in auth['sources'].items():require(digest(path)==h,'Source changed during run')
 except BaseException as exc:error=repr(exc);code=code if code else 126
 finally:
  for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,signal.SIG_IGN)
  cleanup=cleanup_owned_processes([] if child is None else [child],10)
  if not cleanup['completed']:code=126;error=(error or '')+' incomplete cleanup'
  cost['wall_seconds']=time.monotonic()-start
  save(out/'exit.json',dict(state='completed' if code==0 else 'failed',name=a.name,phase=a.phase,exit_code=code,runner_return_code=None if child is None else child.returncode,gpu=a.gpu,error=error,cleanup=cleanup,cost=cost,automatic_retry=False))
  lock.close()
 print(json.dumps(dict(name=a.name,phase=a.phase,exit_code=code,error=error)),flush=True);return code
if __name__=='__main__':sys.exit(main())
