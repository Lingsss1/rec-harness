"""Hold controllers between completed rounds; never interrupt model/LLM children."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

from common import read,write,sha


def proc(record):
    path=Path('/proc')/str(record['pid'])/'stat'
    if not path.exists():return None
    fields=path.read_text().split()
    if record.get('start_ticks') and record['start_ticks']!=fields[21]:return None
    return fields


def live_children(pid):
    result=set()
    for path in (Path('/proc')/str(pid)/'task').glob('*/children'):
        try:result.update(int(value) for value in path.read_text().split())
        except OSError:pass
    live=[]
    for child in result:
        info=proc({'pid':child})
        if info and info[2]!='Z':live.append(child)
    return live


def hold(root):
    pending={}
    for mode in ['auto','enhanced']:
        arm=root/mode;status=read(arm/'status.json');record=read(arm/'worker-process.json')
        pending[mode]={'process':record,'after_round':status.get('round',-1)}
    write(root/'knowledge-boundary-plan.json',{'at':time.time(),'pending':pending,'clock_reset':False})
    deadline=read(root/'budget.json')['deadline_unix']
    while pending:
        if time.time()>=deadline:raise TimeoutError('Original deadline reached before handoff')
        for mode,plan in list(pending.items()):
            arm=root/mode;status=read(arm/'status.json');info=proc(plan['process'])
            if not info or info[2]=='Z':raise RuntimeError(mode+' controller exited before handoff')
            if status.get('phase')!='research' or status.get('round',-1)<=plan['after_round']:continue
            pid=plan['process']['pid'];os.kill(pid,signal.SIGSTOP)
            try:
                for _ in range(100):
                    frozen=proc(plan['process'])
                    if frozen and frozen[2]=='T':break
                    time.sleep(0.01)
                else:raise RuntimeError('Controller did not enter maintenance stop')
                status=read(arm/'status.json');round_id=status['round'];directory=arm/'rounds'/f'r{round_id:04d}'
                started=any(path.stat().st_size>0 for path in directory.glob('*/codex.jsonl'))
                completed=any(directory.glob('*/agent-receipt.json')) or (directory/'propose/proposal.json').exists()
                children=live_children(pid)
                if started or completed or children:
                    plan['after_round']=round_id;os.kill(pid,signal.SIGCONT)
                    print(json.dumps({'mode':mode,'skip_busy_boundary':round_id,'children':children}),flush=True)
                    continue
                state=read(arm/'state.json')
                if state.get('research_closed'):raise RuntimeError('Cannot resume a closed research study')
                marker={'held_unix':time.time(),'original_process':plan['process'],'next_round':round_id,
                        'state_sha256':sha(arm/'state.json'),'protocol_hash':read(root/'protocol.json')['identity'],
                        'committed_trials':len(state['trials']),'committed_memory':len(state.get('memory',[])),
                        'unstarted_round_dir':str(directory),'active_children':[],'reason':'authorized industry knowledge access repair'}
                write(arm/'knowledge-handoff.json',marker)
                write(arm/'status.json',{**status,'phase':'knowledge_maintenance','updated_unix':time.time(),
                      'next_round':round_id,'message':'completed round preserved; no model or LLM child interrupted'})
                print(json.dumps({'mode':mode,'held':True,'next_round':round_id,'trials':len(state['trials'])}),flush=True)
                del pending[mode]
            except BaseException:
                if not (arm/'knowledge-handoff.json').exists():os.kill(pid,signal.SIGCONT)
                raise
        time.sleep(0.05)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args();hold(args.root.resolve())
