"""Finish policy-independent checks; never choose a policy or launch training."""
import argparse
import os
from pathlib import Path
import time

from common import read,write


def run(root):
    write(root/'preparer-process.json',{'pid':os.getpid(),'start_ticks':Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]})
    while True:
        state=read(root/'acquisition-status.json')
        if state['phase']=='failed':raise ValueError('Dataset acquisition failed')
        if state['phase']=='downloaded_and_verified':break
        if (root/'STOP').exists():return
        time.sleep(5)
    from campaign import audit
    from budget import refresh
    refresh(root)
    audit(root)
    from train import prepare
    prepare(root)
    write(root/'preparation-ready.json',{'phase':'data_and_din_adapter_ready','at_unix':time.time(),
        'history_policy':'awaiting explicit choice','formal_training_started':False})
    print('Pinned data verified; train/valid audit and DIN feature map ready. No training launched.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    try:run(args.root.resolve())
    except BaseException as error:
        write(args.root/'preparation-failure.json',{'error':repr(error),'at_unix':time.time()});raise
