"""Apply an authorized runtime revision only at a checked idle checkpoint."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROJECT=Path(__file__).absolute().parents[2]
ROOT=Path(__file__).absolute().parent
SOURCE=ROOT/'source-at-knowledge-repair-v2'
sys.path.insert(0,str(SOURCE))
from common import read,write,sha
from knowledge_boundary import live_children,proc
from controller import resume_checkpoint


def launch(script,args,logfile):
    python=PROJECT/'benchmark_res/.venv/bin/python'
    with logfile.open('a') as stream:
        process=subprocess.Popen([str(python),'-u',str(SOURCE/script),*args],cwd=PROJECT,
            env={**os.environ,'REC_CTR_PROJECT':str(PROJECT)},stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    return process,{'pid':process.pid,'start_ticks':Path(f'/proc/{process.pid}/stat').read_text().split()[21],
                    'launched_unix':time.time(),'runtime_revision':'knowledge-access-v2','checkpoint_resume':True}


def apply(mode):
    arm=ROOT/mode;protocol=read(ROOT/'protocol.json')
    state,marker=resume_checkpoint(arm,protocol)
    old=marker['original_process'];info=proc(old)
    if not info or info[2]!='T' or live_children(old['pid']):raise ValueError('Old controller is not held idle')
    manifest=read(ROOT/'knowledge-source-manifest.json')
    for relative,digest in manifest['files'].items():
        if sha(SOURCE/relative)!=digest:raise ValueError('Source snapshot changed: '+relative)
    for name,digest in manifest['scientific_files_unchanged'].items():
        if sha(ROOT/'source-at-launch'/name)!=digest:raise ValueError('Scientific file identity changed')
    if not read(ROOT/'knowledge-preflight-verification.json')['result']['status']=='ready':raise ValueError('Preflight failed')
    budget=read(ROOT/'budget.json');budget_sha=sha(ROOT/'budget.json')
    repair_path=ROOT/'knowledge-repair.json'
    repair=read(repair_path) if repair_path.exists() else {
        'revision':'knowledge-access-v2','requested_action':'repair knowledge access and retrieval',
        'started_unix':time.time(),'source_snapshot':str(SOURCE),'source_manifest':'knowledge-source-manifest.json',
        'original_deadline_unix':budget['deadline_unix'],'protocol_hash':protocol['identity'],
        'budget_reset':False,'baselines_retrained':False,'personal_knowledge_enabled':False,
        'fixed_instructions_sha256':read(ROOT/'knowledge-preflight/agent-receipt.json')['fixed_instructions_sha256'],
        'preflight_ref':'knowledge-preflight-verification.json','checks_passed':12,'arms':{},
        'attribution_limit':'Mid-run runtime repair; report pre/post rounds separately. Source access and declared adoption do not establish a causal knowledge gain.',
        'test_policy':'auto remains validation-only under auto-validation-only.json; no closed assessments supplied to research'}
    monitor=read(ROOT/'suite-process.json');monitor_info=proc(monitor)
    if not monitor_info or monitor_info[2]=='Z':raise ValueError('Budget monitor is not alive')
    os.kill(monitor['pid'],signal.SIGSTOP)
    new_process=None;replacement_monitor=None
    try:
        write(arm/'maintenance'/'worker-process-before-knowledge-repair.json',read(arm/'worker-process.json'))
        write(arm/'maintenance'/'handoff-before-knowledge-repair.json',marker)
        os.kill(old['pid'],signal.SIGKILL)
        for _ in range(100):
            current=proc(old)
            if not current or current[2]=='Z':break
            time.sleep(0.02)
        else:raise RuntimeError('Old worker did not exit')
        new_process,new_record=launch('controller.py',['worker','--root',str(ROOT),'--mode',mode,
            '--gpus',','.join(map(str,budget['gpus'][mode])),'--resume'],arm/'knowledge-resume.log')
        write(arm/'worker-process.json',new_record)
        for _ in range(100):
            if new_process.poll() is not None:raise RuntimeError('Resumed worker exited; inspect knowledge-resume.log')
            current=read(arm/'status.json')
            if current.get('pid')==new_process.pid and current.get('phase')=='research':break
            time.sleep(0.05)
        else:raise RuntimeError('Resumed worker did not reach research')
        after=read(arm/'state.json')
        if [row['id'] for row in after['trials']]!=[row['id'] for row in state['trials']]:raise ValueError('Committed trial identities changed')
        if after['memory']!=state['memory']:raise ValueError('Committed memory changed')
        if sha(ROOT/'budget.json')!=budget_sha:raise ValueError('Budget was changed')
        repair['arms'][mode]={'applied_unix':time.time(),'from_process':old,'to_process':new_record,
            'start_round':marker['next_round'],'preserved_trials':len(state['trials']),
            'preserved_memory':len(state['memory']),'checkpoint_state_sha256':marker['state_sha256'],
            'deadline_unix':budget['deadline_unix'],'status':'running'}
        write(repair_path,repair)
        if not monitor.get('runtime_revision')=='knowledge-access-v2':
            os.kill(monitor['pid'],signal.SIGKILL)
            replacement_monitor,new_monitor_record=launch('recovery_monitor.py',['--root',str(ROOT)],ROOT/'knowledge-monitor.log')
            write(ROOT/'suite-process.json',new_monitor_record)
            print('Budget monitor upgraded:',replacement_monitor.pid,flush=True)
        else:os.kill(monitor['pid'],signal.SIGCONT)
        print(mode,'resumed',new_process.pid,'round',marker['next_round'],'preserved',len(state['trials']),'trials',flush=True)
    finally:
        current=proc(monitor)
        if current and current[2]=='T':os.kill(monitor['pid'],signal.SIGCONT)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['auto','enhanced'])
    apply(parser.parse_args().mode)
