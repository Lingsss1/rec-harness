"""Persistent eight-GPU fixed experiment supervisor, with durable states and costs."""
import argparse
import fcntl
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

from common import read, write

DATASETS = ['QK-Video', 'KuaiRand', 'TAAC-25', 'MerRec']


def spawn(root, dataset, kind, gpu, calibration):
    folder = root / 'runs' / dataset / kind
    output = folder / ('calibration-01' if calibration else 'fit-01')
    output.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, '-u', str(root / 'source/train.py'), '--root', str(root),
           '--dataset', dataset, '--kind', kind, '--output', str(output)]
    if calibration:
        cmd += ['--calibration-steps', '20']
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '4',
                   'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'PYTHONHASHSEED': '20262027'}
    log = (output / 'train.log').open('ab')
    child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                             env=environment, start_new_session=True)
    log.close()
    record = {'pid': child.pid, 'started_unix': time.time(), 'gpu': gpu, 'command': cmd,
              'start_ticks': Path('/proc/' + str(child.pid) + '/stat').read_text().split()[21]}
    write(output / 'process.json', record)
    return child, output, record


def build_report(root, jobs):
    rows = []
    for dataset in DATASETS:
        candidate_path = root / 'runs' / dataset / 'candidate/fit-01/result.json'
        baseline_path = root / 'runs' / dataset / 'din/fit-01/result.json'
        row = {'dataset': dataset, 'status': 'pending'}
        if candidate_path.exists() and baseline_path.exists():
            candidate = read(candidate_path)
            baseline = read(baseline_path)
            row = {'dataset': dataset, 'status': 'complete',
                   'candidate_auc': candidate['test']['auc'], 'din_auc': baseline['test']['auc'],
                   'delta_macro_auc': candidate['test']['auc'] - baseline['test']['auc'],
                   'candidate_mean_logloss': candidate['test']['logloss'],
                   'din_mean_logloss': baseline['test']['logloss'],
                   'candidate_parameters': candidate['parameters'], 'din_parameters': baseline['parameters'],
                   'candidate_train_seconds': candidate['train_seconds'], 'din_train_seconds': baseline['train_seconds'],
                   'per_task': {label: {'candidate_auc': metric['auc'],
                       'din_auc': baseline['test']['per_task'][label]['auc'],
                       'delta_auc': metric['auc'] - baseline['test']['per_task'][label]['auc'],
                       'candidate_logloss': metric['logloss'],
                       'din_logloss': baseline['test']['per_task'][label]['logloss']}
                       for label, metric in candidate['test']['per_task'].items()}}
        rows.append(row)
    state = {'at_unix': time.time(), 'mode': 'fixed architecture, scratch retraining, paired DIN; no search',
             'jobs': jobs, 'datasets': rows,
             'conclusion': 'No aggregate generalization claim until completed paired task-level results are reviewed.',
             'limitations': ['one seed', 'architecture transfer, not zero-shot', 'official offline feedback semantics']}
    write(root / 'report.json', state)
    lines = ['UniRank fixed architecture transfer / 全原生任务平均 AUC', '']
    for row in rows:
        if row['status'] == 'complete':
            lines.append(f"{row['dataset']}: candidate {row['candidate_auc']:.8f}, DIN {row['din_auc']:.8f}, delta {row['delta_macro_auc']:+.8f}")
            for task, values in row['per_task'].items():
                lines.append(f"  {task}: {values['candidate_auc']:.8f} vs {values['din_auc']:.8f}, delta {values['delta_auc']:+.8f}")
        else:
            lines.append(row['dataset'] + ': pending')
    (root / 'report.txt').write_text('\n'.join(lines) + '\n')


def supervise(root):
    root = root.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    lock = (root / 'supervisor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write(root / 'supervisor-process.json', {'pid': os.getpid(), 'started_unix': time.time()})
    checks = read(root / 'interface-checks.json')
    if checks.get('status') != 'passed':
        raise ValueError('Interface parity/schema checks have not passed')
    jobs = {}
    children = {}
    for index, dataset in enumerate(DATASETS):
        for offset, kind in enumerate(['candidate', 'din']):
            name = dataset + '/' + kind
            jobs[name] = {'dataset': dataset, 'kind': kind, 'gpu': 2 * index + offset, 'phase': 'waiting_data'}
    while True:
        if (root / 'STOP').exists():
            for child, output, record in children.values():
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
            write(root / 'supervisor-status.json', {'phase': 'stopping', 'at_unix': time.time(), 'jobs': jobs})
            for child, output, record in children.values():
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
            return
        for name, job in jobs.items():
            if job['phase'] in ['failed', 'complete', 'timeout']:
                continue
            if name in children:
                child, output, record = children[name]
                code = child.poll()
                if code is None:
                    progress = output / 'progress.json'
                    if progress.exists():
                        job['progress'] = read(progress)
                    elapsed = time.time() - record['started_unix']
                    cap = read(root / 'datasets' / job['dataset'] / 'protocol.json')['run_wall_cap_seconds']
                    if elapsed > (1800 if job['phase'] == 'calibrating' else cap):
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                        job['phase'] = 'timeout'
                        write(output / 'failure.json', {'status': 'timeout', 'wall_seconds': elapsed})
                    continue
                result_path = output / 'result.json'
                if code != 0 or not result_path.exists():
                    job.update(phase='failed', exit_code=code, output=str(output))
                    if (output / 'failure.json').exists():
                        job['failure'] = read(output / 'failure.json')
                    continue
                result = read(result_path)
                if job['phase'] == 'calibrating':
                    job['calibration'] = result
                    job['phase'] = 'ready_full_run'
                    del children[name]
                else:
                    job.update(phase='complete', result=str(result_path))
                    continue
            if job['phase'] == 'waiting_data':
                status = read(root / 'datasets' / job['dataset'] / 'acquisition-status.json')
                if status['phase'] == 'failed':
                    job.update(phase='failed', failure={'reason': 'data acquisition failed'})
                    continue
                if status['phase'] != 'downloaded_and_verified':
                    continue
                job['phase'] = 'calibrating'
                calibration = True
            elif job['phase'] == 'ready_full_run':
                job['phase'] = 'training'
                calibration = False
            else:
                continue
            child, output, record = spawn(root, job['dataset'], job['kind'], job['gpu'], calibration)
            children[name] = (child, output, record)
            job.update(pid=child.pid, output=str(output), started_unix=record['started_unix'])
            job.pop('progress', None)
        build_report(root, jobs)
        finished = all(job['phase'] in ['complete', 'failed', 'timeout'] for job in jobs.values())
        write(root / 'supervisor-status.json', {'phase': 'finished' if finished else 'running',
              'at_unix': time.time(), 'jobs': jobs})
        if finished:
            return
        time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    supervise(parser.parse_args().root)
