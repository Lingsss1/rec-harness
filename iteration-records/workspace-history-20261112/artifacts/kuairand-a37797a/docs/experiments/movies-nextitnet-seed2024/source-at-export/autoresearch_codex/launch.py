"""Detach the validated comparison controller and retain its log and PID."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    a = p.parse_args()
    config = a.config.resolve()
    c = json.loads(config.read_text())
    root = Path(c['run_dir'])
    root.mkdir(parents=True, exist_ok=True)
    prior = root / 'launcher.json'
    if prior.exists():
        pid = json.loads(prior.read_text())['pid']
        if Path(f'/proc/{pid}').exists():
            raise RuntimeError(f'controller pid {pid} still exists; inspect before relaunch')
    argv = [sys.executable, '-u', '-m', 'autoresearch_codex.run', '--config', str(config)]
    with (root / 'controller.log').open('a') as log:
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, cwd=Path(__file__).resolve().parents[1], start_new_session=True)
    receipt = {'pid': process.pid, 'command': argv, 'started_at': time.time()}
    prior.write_text(json.dumps(receipt, indent=2))
    time.sleep(2)
    if process.poll() is not None:
        raise RuntimeError('controller stopped immediately; inspect controller.log')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
