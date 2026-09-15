"""Wait for actual eight-fit completion, then analyze and time without retraining."""
import json,subprocess,sys,time,os
from pathlib import Path
from common import require,digest
R=Path(__file__).resolve().parent

def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def main():
 out=R/'reports/postprocess';require(not out.exists(),'No pipeline replay');out.mkdir()
 files={str(R/n):digest(R/n) for n in ('analyze.py','paired_core.py','analysis-protocol.json','latency.py','runner.py','common.py','multistream_model.py')}
 save(out/'claim.json',dict(pid=os.getpid(),files=files,automatic_retry=False));start=time.monotonic();jobs={};logs=[]
 try:
  while not (R/'reports/launcher-completion.json').exists():
   require(time.monotonic()-start<18000,'Training completion deadline expired')
   for phase in ('smoke','formal'):
    p=R/'reports'/('launch-'+phase)/'completion.json'
    if p.exists():require(json.loads(p.read_text())['state']=='completed','Training phase failed; no postprocess')
   time.sleep(5)
  for p,h in files.items():require(digest(p)==h,'Frozen measurement source changed')
  for name,script in [('paired','analyze.py'),('latency','latency.py')]:
   f=(out/(name+'.log')).open('x');logs.append(f);p=subprocess.Popen([sys.executable,'-B',str(R/script)],stdout=f,stderr=subprocess.STDOUT);jobs[name]=p;save(out/(name+'-child.json'),dict(pid=p.pid))
  codes={n:p.wait() for n,p in jobs.items()};require(all(c==0 for c in codes.values()),'Measurement failure; completed siblings preserved')
  save(out/'completion.json',dict(state='completed_pending_interpretation',exit_codes=codes,training_restarts=0))
 except BaseException as exc:save(out/'failure.json',dict(state='failed',error=repr(exc),children={n:dict(pid=p.pid,return_code=p.poll()) for n,p in jobs.items()},automatic_retry=False));raise
 finally:
  for f in logs:f.close()
if __name__=='__main__':main()
