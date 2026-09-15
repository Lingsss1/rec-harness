"""Explicit phase-boundary GPU expansion. Reuses all eight already-started smokes; no model replay."""
import fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
from common import require,digest
from process_utils import gpu_snapshot
from launch import save,phase_run
R=Path(__file__).resolve().parent

def main():
 hand=json.loads((R/'reports/scheduler-handoff.json').read_text());pid=hand['launcher_pid'];auth=json.loads((R/'authorization.json').read_text());first=auth['experiments']
 require(all((R/'reports'/n/'smoke/claim.json').exists() for n in first),'All eight existing smokes must have claims')
 require(not (R/'runs').exists(),'Formal phase must not have started')
 start=time.monotonic()
 while not all((R/'reports'/n/'smoke/exit.json').exists() for n in first):
  require(time.monotonic()-start<1200,'First smoke handoff deadline expired');time.sleep(5)
 for n in first:
  e=json.loads((R/'reports'/n/'smoke/exit.json').read_text());require(e['exit_code']==0 and e['cleanup']['completed'],'Existing smoke failed; preserve without retry')
 # Scheduler is stopped, and all of its model workers are terminal and cleaned.
 require(b'launch.py' in Path('/proc',str(pid),'cmdline').read_bytes(),'Scheduler identity changed')
 os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
 for _ in range(100):
  p=Path('/proc',str(pid),'stat')
  if not p.exists() or p.read_text().split(') ',1)[1][0]=='Z':break
  time.sleep(.1)
 lock=(R/'.launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for gpu in range(8):gpu_snapshot(gpu,10,idle=True)
 original=R/'authorization-original-four-gpu.json';require(not original.exists(),'No authorization replay');original.write_bytes((R/'authorization.json').read_bytes())
 auth['gpu_pool']=list(range(8));auth['supersedes_at_clean_smoke_boundary']=dict(path=str(original),sha256=digest(original),completed_smokes=first,reason='Other user GPU jobs completed; expand user-authorized eight-model scheduling')
 auth['sources'][str(Path(__file__).resolve())]=digest(Path(__file__))
 auth['sources'][str(R/'reports/config-clarification.json')]=digest(R/'reports/config-clarification.json')
 save(R/'authorization.json',auth)
 folder=R/'reports/launch-smoke-expansion';folder.mkdir();jobs=[];logs=[]
 for gpu,name in enumerate([]):
  f=(folder/(name+'.log')).open('x');logs.append(f);p=subprocess.Popen([sys.executable,'-B',str(R/'worker.py'),'--name',name,'--phase','smoke','--gpu',str(gpu)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);jobs.append((name,p));print(json.dumps(dict(event='started',name=name,phase='smoke',gpu=gpu,pid=p.pid)),flush=True)
 save(folder/'claim.json',dict(pid=os.getpid(),workers={n:p.pid for n,p in jobs},original_four_smokes_reused=True))
 codes={n:p.wait() for n,p in jobs}
 for f in logs:f.close()
 save(folder/'completion.json',dict(state='completed' if all(c==0 for c in codes.values()) else 'failed',exits=codes,automatic_retry=False))
 require(all(c==0 for c in codes.values()),'Expansion smoke failed; no formal run')
 init={};originals=[]
 for n in auth['experiments']:
  e=json.loads((R/'reports'/n/'smoke/exit.json').read_text());s=json.loads((R/'smoke-runs'/n/'smoke.json').read_text())
  require(e['exit_code']==0 and e['cleanup']['completed'] and s['optimizer_steps']==2,'Invalid smoke')
  require(all(x['all_heads_updated'] and x['finite_gradients'] for x in s['steps']),'Missing smoke gate')
  if n!='cross8':require(s['native_parity']['exact'],'Native parity failed')
  init[n]=s['initial_sha256'];originals.append(s['original_parameters_sha256'])
 require(len(set(originals))==1,'Common original initialization differs')
 save(R/'reports/smoke-review.json',dict(state='passed',initializers=init,original_parameters_sha256=originals[0],steps=16,original_first_wave_reused=True,additional_training_seeds=[],automatic_retry=False))
 save(R/'reports/launch-smoke/completion.json',dict(state='completed',exits={n:0 for n in auth['experiments']},phase_boundary_gpu_expansion=True,training_restarts=0))
 save(R/'reports/scheduler-handoff-completion.json',dict(state='completed',old_scheduler_pid=pid,new_scheduler_pid=os.getpid(),gpu_pool=auth['gpu_pool'],model_restarts=0,smoke_replays=0))
 phase_run('formal',auth)
 save(R/'reports/launcher-completion.json',dict(state='completed_training_pending_statistics',experiments=auth['experiments']))
if __name__=='__main__':main()
