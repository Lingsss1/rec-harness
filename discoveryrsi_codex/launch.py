"""Detach a new comparison; preserve any earlier run instead of restarting it."""
import json
import os
import subprocess
import sys
import time

from discoveryrsi_codex.run import ROOT, WORKSPACE
from mle_star_codex.run import save


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if any((ROOT / name).exists() for name in ['launcher.json', 'frozen.json', 'progress.json']):
        raise RuntimeError('Existing run; inspect before recovery')
    command = [sys.executable, '-u', '-m', 'discoveryrsi_codex.run']
    with (ROOT / 'controller.log').open('a') as log:
        proc = subprocess.Popen(command, cwd=WORKSPACE, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
    save(ROOT / 'launcher.json', {'pid': proc.pid, 'command': command, 'time': time.time()})
    time.sleep(2)
    if proc.poll() is not None:
        raise RuntimeError('Controller exited; inspect controller.log')
    print(json.dumps({'pid': proc.pid, 'log': str(ROOT / 'controller.log')}))


if __name__ == '__main__':
    main()
