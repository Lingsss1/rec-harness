"""Detached sequential phase owner; logs/terminal records survive the interactive turn."""
from pathlib import Path
import subprocess,sys,json,os,time
R=Path(__file__).resolve().parent

def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def main():
 out=R/'reports/pipeline';out.mkdir();save(out/'claim.json',dict(pid=os.getpid(),automatic_retry=False));jobs={};logs=[]
 try:
  with (out/'training.log').open('x') as f:
   code=subprocess.call([sys.executable,'-B',str(R/'launch.py')],stdout=f,stderr=subprocess.STDOUT)
  save(out/'training-exit.json',dict(exit_code=code));assert code==0,'Training phase failed; no replay'
  for name,script in [('paired','analyze.py'),('latency','latency.py')]:
   f=(out/(name+'.log')).open('x');logs.append(f);p=subprocess.Popen([sys.executable,'-B',str(R/script)],stdout=f,stderr=subprocess.STDOUT);jobs[name]=p;save(out/(name+'-claim.json'),dict(pid=p.pid))
  codes={n:p.wait() for n,p in jobs.items()};save(out/'measurement-exits.json',codes);assert all(v==0 for v in codes.values()),'Measurement failure; preserve completed siblings'
  with (out/'closeout.log').open('x') as f:code=subprocess.call([sys.executable,'-B',str(R/'closeout.py')],stdout=f,stderr=subprocess.STDOUT)
  assert code==0,'Closeout failed; preserve measurements'
  save(out/'completion.json',dict(state='completed',training_restarts=0))
 except BaseException as exc:save(out/'failure.json',dict(state='failed',error=repr(exc),children={n:dict(pid=p.pid,return_code=p.poll()) for n,p in jobs.items()},automatic_retry=False));raise
 finally:
  for f in logs:f.close()
if __name__=='__main__':main()
