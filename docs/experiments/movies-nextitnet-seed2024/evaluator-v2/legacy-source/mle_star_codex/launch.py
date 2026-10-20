"""Start a detached controller with an explicit log and PID receipt."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    a = p.parse_args()
    config_path = a.config.resolve()
    c = json.loads(config_path.read_text())
    run = Path(c['run_dir'])
    if (run / 'launcher.json').exists():
        prior = json.loads((run / 'launcher.json').read_text())
        if Path(f'/proc/{prior["pid"]}').exists():
            raise RuntimeError(f'controller PID {prior["pid"]} already exists; inspect before relaunch')
    command = [sys.executable, '-u', '-m', 'mle_star_codex.run', '--config', str(config_path)]
    with (run / 'controller.log').open('a') as log:
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, cwd=Path(__file__).resolve().parents[1], start_new_session=True)
    receipt = {'pid': proc.pid, 'command': command, 'started_at': time.time()}
    (run / 'launcher.json').write_text(json.dumps(receipt, indent=2))
    time.sleep(2)
    if proc.poll() is not None:
        raise RuntimeError('controller exited immediately; inspect controller.log')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
