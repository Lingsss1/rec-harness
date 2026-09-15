"""Four free GPUs, eight cells, all smokes precede any formal fit; preserve failures."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from common import require,digest
R=Path(__file__).resolve().parent

def save(p,v):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def phase_run(phase,auth):
 folder=R/'reports'/('launch-'+phase);require(not folder.exists(),'No phase replay');folder.mkdir()
 pending=list(auth['experiments']);running={};done={};logs=[];start=time.monotonic();failed=False
 try:
  while pending or running:
   for gpu in auth['gpu_pool']:
    if gpu not in running and pending and not failed:
     name=pending.pop(0);f=(folder/(name+'.log')).open('x');logs.append(f)
     p=subprocess.Popen([sys.executable,'-B',str(R/'worker.py'),'--name',name,'--phase',phase,'--gpu',str(gpu)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
     running[gpu]=(name,p);print(json.dumps(dict(event='started',name=name,phase=phase,gpu=gpu,pid=p.pid)),flush=True)
   for gpu,(name,p) in list(running.items()):
    code=p.poll()
    if code is not None:
     done[name]=code;del running[gpu]
     if code:failed=True
     print(json.dumps(dict(event='exited',name=name,phase=phase,exit_code=code)),flush=True)
   save(folder/'status.json',dict(state='running',phase=phase,pending=pending,running={str(g):dict(name=n,pid=p.pid) for g,(n,p) in running.items()},completed=done))
   if failed and not running:break
   time.sleep(2)
  terminal=dict(state='completed' if not failed and not pending else 'failed',phase=phase,exits=done,not_started=pending,wall_seconds=time.monotonic()-start,automatic_retry=False)
  save(folder/'completion.json',terminal);require(terminal['state']=='completed','Phase failure; no automatic retry')
 finally:
  for f in logs:f.close()

def main():
 lock=(R/'.launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 auth=json.loads((R/'authorization.json').read_text())
 for p,h in auth['sources'].items():require(digest(p)==h,'Frozen source changed')
 save(R/'reports/launcher.json',dict(pid=os.getpid(),state='started',authorization_sha256=digest(R/'authorization.json')))
 phase_run('smoke',auth)
 init={};original=[]
 for name in auth['experiments']:
  run=R/'smoke-runs'/name;s=json.loads((run/'smoke.json').read_text());e=json.loads((R/'reports'/name/'smoke/exit.json').read_text())
  require(e['exit_code']==0 and e['cleanup']['completed'] and s['optimizer_steps']==2,'Smoke not cleanly completed')
  require(all(x['finite_gradients'] and x['all_heads_updated'] for x in s['steps']),'Missing smoke gate')
  if name!='cross8':require(s['native_parity']['exact'],'No-feedback parity failed')
  init[name]=s['initial_sha256'];original.append(s['original_parameters_sha256'])
 require(len(set(original))==1,'Original Cross8 parameters do not match across all cells')
 require(original[0]==auth['reference_original_parameters_sha256'],'Reused reference initialization differs')
 save(R/'reports/smoke-review.json',dict(state='passed',initializers=init,original_parameters_sha256=original[0],steps=4,additional_training_seeds=[],automatic_retry=False))
 phase_run('formal',auth)
 save(R/'reports/launcher-completion.json',dict(state='completed_training_pending_statistics',experiments=list(auth['experiments'])))
if __name__=='__main__':main()
