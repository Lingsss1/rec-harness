"""Detached launch; existing nonempty runs are never silently restarted."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from mlevolve_codex.run import ROOT, WORKSPACE
from mle_star_codex.run import save


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / 'launcher.json').exists():
        raise RuntimeError('Existing launch; inspect retained process and outcomes first')
    argv = [sys.executable, '-u', '-m', 'mlevolve_codex.run']
    with (ROOT / 'controller.log').open('a') as out:
        proc = subprocess.Popen(argv, cwd=WORKSPACE, stdin=subprocess.DEVNULL,
            stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
    save(ROOT / 'launcher.json', {'pid': proc.pid, 'command': argv, 'time': time.time()})
    time.sleep(2)
    if proc.poll() is not None:
        raise RuntimeError('Controller exited; inspect controller.log')
    print(json.dumps({'pid': proc.pid, 'log': str(ROOT / 'controller.log')}))


if __name__ == '__main__':
    main()
