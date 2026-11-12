"""Sample once for all panes; render read-only dashboard views inside tmux."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import traceback

from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from collect import Collector,tail_records
from render import render


def atomic(path,value):
    path=Path(path);temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
    temporary.replace(path)


def sample(root,interval):
    output=root/'dashboard';output.mkdir(exist_ok=True)
    with (output/'sampler.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic(output/'sampler.json',{'pid':os.getpid(),'start_ticks':Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
                                      'started_unix':time.time(),'read_only_experiment_observer':True})
        collector=Collector(root);last_history=0
        while True:
            started=time.monotonic()
            try:
                snapshot=collector.snapshot()
                atomic(output/'snapshot.json',snapshot)
                if time.time()-last_history>=5:
                    row={'at':snapshot['updated_unix'],'arms':{mode:{'best_auc':arm['best']['auc'] if arm['best'] else None,
                         'trials':arm['counts']['total'],'phase':arm['phase'],'tokens':arm['usage']['input_tokens']} for mode,arm in snapshot['arms'].items()},
                         'gpu_utilization':{str(item['index']):item['utilization'] for item in snapshot['resources']['gpus']},
                         'cpu_percent':snapshot['resources'].get('cpu_percent')}
                    with (output/'telemetry.jsonl').open('a') as stream:stream.write(json.dumps(row,separators=(',',':'))+'\n')
                    last_history=time.time()
            except Exception as error:
                atomic(output/'sampler-error.json',{'at':time.time(),'error':repr(error)})
                traceback.print_exc()
            time.sleep(max(0.1,interval-(time.monotonic()-started)))


def view(root,name,once=False,width=None,height=None):
    console=Console(width=width,height=height,force_terminal=None if not once else False,
                    color_system='auto' if not once else None)
    path=root/'dashboard/snapshot.json'
    def content():
        if not path.exists():return Panel('等待只读采样器生成第一份快照…')
        snapshot=json.loads(path.read_text())
        return render(name,snapshot,console.width,console.height,tail_records(root/'dashboard/telemetry.jsonl',80))
    if once:
        console.print(content());return
    with Live(content(),console=console,screen=True,auto_refresh=False,vertical_overflow='crop') as live:
        signature=None
        while True:
            try:
                current=(path.stat().st_mtime_ns,console.width,console.height) if path.exists() else None
                if current!=signature:
                    live.update(content(),refresh=True);signature=current
            except (OSError,ValueError):pass
            time.sleep(0.5)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['sample','view'])
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--interval',type=float,default=2)
    parser.add_argument('--view',default='overview')
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--width',type=int);parser.add_argument('--height',type=int)
    args=parser.parse_args()
    try:
        if args.action=='sample':sample(args.root.resolve(),args.interval)
        else:view(args.root.resolve(),args.view,args.once,args.width,args.height)
    except KeyboardInterrupt:pass
