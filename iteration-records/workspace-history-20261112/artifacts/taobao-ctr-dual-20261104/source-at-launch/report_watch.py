"""Refresh the host-only report without exposing cross-arm feedback to researchers."""
import argparse
from pathlib import Path
import time
import traceback

from common import read
from controller import comparison


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args();root=args.root.resolve()
    while True:
        try:
            comparison(root)
            phase=read(root/'status.json')['phase'] if (root/'status.json').exists() else None
            if phase in {'completed','completed_with_failures','failed','budget_reached','stopped_during_acquisition'}:
                time.sleep(2);comparison(root);break
            if time.time()>read(root/'budget.json')['deadline_unix']+120:break
        except Exception:
            traceback.print_exc()
        time.sleep(20)
