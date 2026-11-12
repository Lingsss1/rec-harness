"""Resume only the failed memory index path; preserve all scientific history."""
import copy
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).absolute().parents[1]
PROJECT=ROOT.parents[1]
SOURCE=ROOT/'source-at-memory-index-repair-v3'
sys.path.insert(0,str(SOURCE))
from common import read,write,sha
from knowledge import scoped_cards
from knowledge_boundary import proc,live_children

arm=ROOT/'enhanced';status=read(arm/'status.json');before=read(arm/'state.json')
assert status['phase']=='failed_before_assessment' and not before['research_closed']
assert before['last_runtime_error']['error']=="unhashable type: 'slice'"
old=read(arm/'worker-process.json');oldproc=proc(old)
assert not oldproc or oldproc[2]=='Z'
assert not live_children(old['pid'])
assert before['trials']==read(ROOT/'trajectory-audit-20261105/enhanced-state.json')['trials']
assert before['memory']==read(ROOT/'trajectory-audit-20261105/enhanced-state.json')['memory']
manifest=read(ROOT/'memory-index-source-manifest.json')
assert all(sha(SOURCE/relative)==digest for relative,digest in manifest['files'].items())
protocol=read(ROOT/'protocol.json');budget=read(ROOT/'budget.json');budget_digest=sha(ROOT/'budget.json')
cards=scoped_cards([],{'sources':[]},before['memory'],protocol['data_identity'],protocol['identity'])
assert len(cards)==33 and len(next(c for c in cards if c['id']=='agent-r0034')['proposed_updates'])==5
monitor=read(ROOT/'suite-process.json');info=proc(monitor)
assert info and info[2] not in ['Z','T']
archive=arm/'maintenance/memory-index-repair-v3';archive.mkdir(parents=True,exist_ok=False)
write(archive/'state-before.json',before);write(archive/'status-before.json',status)
write(archive/'worker-process-before.json',old);write(archive/'previous-handoff.json',read(arm/'knowledge-handoff.json'))
state=copy.deepcopy(before);old_error=state.pop('last_runtime_error')
state.setdefault('resolved_runtime_errors',[]).append({'error':old_error,'resolved_unix':time.time(),
    'resolution':'Host index now accepts list and records-wrapper memory updates; scientific records unchanged',
    'revision':'knowledge-access-v3','evidence':'memory-index-source-manifest.json'})
next_round=max([int(p.name[1:]) for p in (arm/'rounds').glob('r*')])+1
os.kill(monitor['pid'],signal.SIGSTOP)
try:
    write(arm/'state.json',state)
    write(arm/'knowledge-handoff.json',{'held_unix':time.time(),'original_process':old,'next_round':next_round,
        'state_sha256':sha(arm/'state.json'),'protocol_hash':protocol['identity'],
        'committed_trials':len(state['trials']),'committed_memory':len(state['memory']),'active_children':[],
        'reason':'resume after verified host memory-wrapper type fix; no scientific policy change'})
    write(arm/'status.json',{'phase':'knowledge_maintenance','updated_unix':time.time(),'next_round':next_round})
    with (arm/'memory-index-resume.log').open('a') as stream:
        process=subprocess.Popen([str(PROJECT/'benchmark_res/.venv/bin/python'),'-u',str(SOURCE/'controller.py'),
            'worker','--root',str(ROOT),'--mode','enhanced','--gpus','4,5,6,7','--resume'],cwd=PROJECT,
            env={**os.environ,'REC_CTR_PROJECT':str(PROJECT)},stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    record={'pid':process.pid,'start_ticks':Path(f'/proc/{process.pid}/stat').read_text().split()[21],
            'launched_unix':time.time(),'runtime_revision':'knowledge-access-v3','checkpoint_resume':True}
    write(arm/'worker-process.json',record)
    for _ in range(100):
        if process.poll() is not None:raise RuntimeError('Worker failed; inspect memory-index-resume.log')
        current=read(arm/'status.json')
        if current.get('pid')==process.pid and current.get('phase')=='research':break
        time.sleep(0.05)
    else:raise RuntimeError('Worker did not reach research')
    after=read(arm/'state.json');assert after['trials']==before['trials'] and after['memory']==before['memory']
    assert sha(ROOT/'budget.json')==budget_digest
    recovery={'at_unix':time.time(),'old_status':status,'old_process':old,'new_process':record,'start_round':next_round,
        'preserved_trials':len(after['trials']),'preserved_memory':len(after['memory']),
        'deadline_unix':budget['deadline_unix'],'budget_reset':False,'baselines_retrained':False,
        'scientific_policy_changed':False,'scientific_files_unchanged':True,'checks_passed':14,
        'downtime_seconds':time.time()-status['updated_unix'],'snapshot':'trajectory-audit-20261105',
        'cause':'Host assumed memory_updates was a list; agent-r0034 contains a valid records wrapper',
        'source_manifest':'memory-index-source-manifest.json','warning':'Include downtime and runtime repair in any workflow comparison'}
    write(ROOT/'memory-index-recovery.json',recovery)
    repair=read(ROOT/'knowledge-repair.json');repair.setdefault('followup_repairs',[]).append({'revision':'knowledge-access-v3',
        'mode':'enhanced','at_unix':time.time(),'start_round':next_round,'details':'memory-index-recovery.json'})
    write(ROOT/'knowledge-repair.json',repair)
    print({'resumed_pid':process.pid,'round':next_round,'trials':len(after['trials']),'memory':len(after['memory'])},flush=True)
finally:
    current=proc(monitor)
    if current and current[2]=='T':os.kill(monitor['pid'],signal.SIGCONT)
