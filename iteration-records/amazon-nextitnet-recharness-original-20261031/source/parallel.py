"""Resource-only rescheduling of the four authorized original-RecHarness runs."""
import argparse
import fcntl
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('original_runner', ROOT / 'runner.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
read, save = legacy.read, legacy.save


def live(pid):
    try:
        os.kill(pid, 0)
        stat = (Path('/proc') / str(pid) / 'stat').read_text()
        return stat[stat.rfind(')') + 2:].split()[0] != 'Z'
    except (ProcessLookupError, FileNotFoundError):
        return False


def worker(family, queue):
    legacy.check()
    config = read(ROOT / 'runs' / family / 'config.json')
    gpus = [0, 1, 2, 3] if queue == 0 else [4, 5, 6, 7]
    config['gpus'] = gpus
    config['cpu_cores'] = list(range(gpus[0] * 16, (gpus[-1] + 1) * 16))
    folder = Path(config['root']) / 'recharness'
    folder.mkdir(exist_ok=True)
    lock = (folder / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder / 'events.jsonl').exists():
        raise RuntimeError('Formal job already started; refusing duplicate execution')
    save(Path(config['root']) / 'effective-config.json', config)
    os.sched_setaffinity(0, set(config['cpu_cores']))
    try:
        legacy.OriginalCorrectedController(config).run()
    except BaseException:
        save(folder / 'failure.json', {'error': traceback.format_exc(), 'time': time.time()})
        raise


def run_queue(index):
    directory = ROOT / 'parallel' / str(index)
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpus = list(range(index * 4, index * 4 + 4))
    leases = []
    for gpu in gpus:
        handle = (legacy.WORKSPACE / 'artifacts/gpu-leases' / f'gpu-{gpu}.lock').open('a')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        leases.append(handle)
    completed = []
    try:
        if index == 0:
            pid = read(ROOT / 'runs/gru4rec/launcher.json')['pid']
            save(directory / 'status.json', {'status': 'running', 'family': 'gru4rec', 'adopted_pid': pid,
                                            'gpus': gpus, 'completed': [], 'time': time.time()})
            while live(pid):
                time.sleep(5)
            if not (ROOT / 'runs/gru4rec/recharness/summary.json').exists():
                raise RuntimeError('Existing GRU4Rec run exited without completion; inspect logs')
            completed.append('gru4rec')
        families = ['bert4rec'] if index == 0 else ['nextitnet', 'hstu']
        for family in families:
            legacy.check()
            # A lease alone is insufficient for older unrelated launchers.
            while True:
                output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                    '--format=csv,noheader,nounits'], text=True)
                devices = {int(line.split(',')[0]): [int(v.strip()) for v in line.split(',')[1:]] for line in output.splitlines()}
                if all(devices[g][0] < 512 and devices[g][1] < 5 for g in gpus):
                    break
                save(directory / 'status.json', {'status': 'waiting_for_idle_gpus', 'family': family,
                    'gpus': gpus, 'completed': completed, 'time': time.time()})
                time.sleep(10)
            save(directory / 'status.json', {'status': 'running', 'family': family,
                'gpus': gpus, 'completed': completed, 'time': time.time()})
            folder = ROOT / 'runs' / family
            with (folder / 'controller.log').open('w') as log:
                process = subprocess.Popen([str(legacy.PYTHON), '-u', str(Path(__file__).resolve()),
                    'worker', '--family', family, '--queue', str(index)], cwd=legacy.WORKSPACE,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                save(folder / 'launcher.json', {'pid': process.pid, 'gpus': gpus, 'time': time.time()})
                code = process.wait()
            if code:
                raise RuntimeError('Formal job failed; queue stopped: ' + family)
            completed.append(family)
        save(directory / 'status.json', {'status': 'complete', 'completed': completed, 'gpus': gpus, 'time': time.time()})
    except BaseException:
        save(directory / 'status.json', {'status': 'failed', 'completed': completed,
            'gpus': gpus, 'error': traceback.format_exc(), 'time': time.time()})
        raise


def launch():
    legacy.check()
    if (ROOT / 'parallel/launcher.json').exists():
        raise RuntimeError('Parallel scheduling already launched')
    current = read(ROOT / 'status.json')
    if current['status'] != 'running' or current['family'] != 'gru4rec':
        raise RuntimeError('Serial pipeline advanced; re-audit before rescheduling')
    for family in ['nextitnet', 'bert4rec', 'hstu']:
        if (ROOT / 'runs' / family / 'recharness/events.jsonl').exists():
            raise RuntimeError('An intended parallel job already started: ' + family)
    pid = read(ROOT / 'launcher.json')['pid']
    command = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
    if str(ROOT / 'runner.py').encode() not in command or b'pipeline' not in command:
        raise RuntimeError('Recorded PID is not the original serial scheduler')
    os.kill(pid, signal.SIGSTOP)
    # Freeze scheduler first, so it cannot dispatch a duplicate job during takeover.
    for family in ['nextitnet', 'bert4rec', 'hstu']:
        if (ROOT / 'runs' / family / 'recharness/events.jsonl').exists():
            os.kill(pid, signal.SIGCONT)
            raise RuntimeError('Serial queue advanced during takeover')
    save(ROOT / 'parallel/original-serial-status.json', current)
    os.kill(pid, signal.SIGTERM)
    os.kill(pid, signal.SIGCONT)
    for _ in range(30):
        if not live(pid):
            break
        time.sleep(.1)
    if live(pid):
        raise RuntimeError('Serial scheduler did not exit; no parallel jobs started')
    processes = []
    for index in [0, 1]:
        directory = ROOT / 'parallel' / str(index)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / 'supervisor.log').open('w') as log:
            process = subprocess.Popen([str(legacy.PYTHON), '-u', str(Path(__file__).resolve()),
                'queue', '--queue', str(index)], cwd=legacy.WORKSPACE,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append({'queue': index, 'pid': process.pid, 'gpus': list(range(index * 4, index * 4 + 4))})
    save(ROOT / 'parallel/launcher.json', {'processes': processes, 'time': time.time(),
        'old_scheduler_pid': pid, 'change': 'resource scheduling only; same four formal jobs and budgets',
        'script_sha256': legacy.sha(__file__), 'order': [['gru4rec (adopt existing)', 'bert4rec'], ['nextitnet', 'hstu']]})
    save(ROOT / 'status.json', {'status': 'running', 'scheduling': 'two parallel four-GPU queues',
        'queue_status_paths': [str(ROOT / 'parallel' / str(i) / 'status.json') for i in [0, 1]],
        'time': time.time()})
    print('Parallel queues launched: GPUs 0-3 GRU4Rec -> BERT4Rec; GPUs 4-7 NextItNet -> HSTU')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['launch', 'queue', 'worker'])
    parser.add_argument('--queue', type=int, choices=[0, 1], default=0)
    parser.add_argument('--family', choices=legacy.FAMILIES)
    args = parser.parse_args()
    {'launch': launch, 'queue': lambda: run_queue(args.queue),
     'worker': lambda: worker(args.family, args.queue)}[args.action]()
