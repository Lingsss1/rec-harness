"""Launch one isolated repetition with the unchanged MLEvolve rerun worker."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
sys.path.insert(0, str(WORKSPACE))
from amazon_aligned import reruns
from amazon_aligned.core import read, save
from amazon_aligned.prepare import sha

reruns.DESTINATION = ROOT


def inventory():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                      '--format=csv,noheader,nounits'], text=True)
    return {int(row.split(',')[0]): [int(x.strip()) for x in row.split(',')[1:]]
            for row in output.strip().splitlines()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.launch == args.worker:
        parser.error('Specify exactly one of --launch or --worker')
    plan = read(ROOT/'plan.json')
    config = read(ROOT/'config.json')
    if sha(Path(__file__)) != read(ROOT/'registration.json')['runner_sha256']:
        raise RuntimeError('Registered runner changed')
    reruns.check_files(plan)
    lock = (ROOT/('launch.lock' if args.launch else 'worker.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    method = ROOT/'mlevolve'
    method.mkdir(exist_ok=True)
    if args.launch:
        if (ROOT/'launcher.json').exists() or (method/'events.jsonl').exists():
            raise RuntimeError('Already launched; do not duplicate or reset the search')
        command = [config['controller_python'], '-u', str(Path(__file__)), '--worker']
        with (method/'controller.log').open('a') as log:
            child = subprocess.Popen(command, cwd=WORKSPACE, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES':'', 'OMP_NUM_THREADS':'4',
                     'MKL_NUM_THREADS':'4', 'PYTHONHASHSEED':'42'})
        record = {'pid':child.pid, 'command':command, 'started_at':time.time()}
        save(ROOT/'launcher.json', record)
        save(method/'launcher.json', record)
        print(json.dumps(record))
        return
    os.sched_setaffinity(0, set(config['cpu_cores']))
    try:
        while True:
            devices = inventory()
            if all(devices[g][0] < 512 and devices[g][1] < 5 for g in config['gpus']):
                break
            save(ROOT/'status.json', {'status':'waiting_for_idle_gpus', 'devices':devices,
                                     'updated_at':time.time()})
            time.sleep(15)
        reruns.check_files(plan)
        save(ROOT/'status.json', {'status':'running', 'pid':os.getpid(),
                                 'gpus':config['gpus'], 'started_at':time.time()})
        reruns.run_worker(plan, ROOT/'config.json')
        save(ROOT/'status.json', read(method/'status.json'))
    except BaseException:
        save(ROOT/'status.json', {'status':'failed', 'error':traceback.format_exc(), 'time':time.time()})
        raise


if __name__ == '__main__':
    main()
