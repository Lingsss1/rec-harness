"""Four new fixed fits on GPUs4..7; compare with previously frozen controls."""
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


def spawn(root, dataset, gpu, calibration):
    output = root / 'runs' / dataset / 'candidate' / ('calibration-01' if calibration else 'fit-01')
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', str(root / 'source/train.py'), '--root', str(root),
               '--dataset', dataset, '--kind', 'candidate', '--output', str(output)]
    if calibration:
        command += ['--calibration-steps', '20']
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '4',
                   'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'PYTHONHASHSEED': '20262027'}
    with (output / 'train.log').open('ab') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 start_new_session=True, env=environment)
    record = {'pid': child.pid, 'start_ticks': Path('/proc/' + str(child.pid) + '/stat').read_text().split()[21],
              'started_unix': time.time(), 'gpu': gpu, 'command': command}
    write(output / 'process.json', record)
    return child, output, record


def report(root, jobs):
    rows = []
    lines = ['最终第一名 g3-r0018 的跨数据集结构迁移（从头训练）', '']
    for dataset in DATASETS:
        protocol = read(root / 'datasets' / dataset / 'protocol.json')
        primary = protocol['architecture_primary_task']
        din = read(root / 'reference-runs' / dataset / 'din/result.json')
        previous = read(root / 'reference-runs' / dataset / 'previous_macro_candidate/result.json')
        row = {'dataset': dataset, 'primary_task': primary, 'primary_is_click': primary in ['click', 'is_click'],
               'din': {'macro_auc': din['test']['auc'], 'primary_auc': din['test']['per_task'][primary]['auc']},
               'previous_macro_candidate': {'macro_auc': previous['test']['auc'],
                                           'primary_auc': previous['test']['per_task'][primary]['auc']},
               'status': jobs[dataset]['phase'], 'references_reused_without_new_training': True,
               'previous_model_id': read(root / 'setup.json')['previous_model']}
        path = root / 'runs' / dataset / 'candidate/fit-01/result.json'
        if path.exists():
            result = read(path)
            if result['training_loss_hook_calls'] != (result['optimizer_steps'] if read(root / 'setup.json').get('training_loss_hook_expected', True) else 0):
                raise ValueError('The inherited training-loss hook was not used in every step')
            row.update(status='complete', current_best={
                'macro_auc': result['test']['auc'], 'primary_auc': result['test']['per_task'][primary]['auc'],
                'mean_logloss': result['test']['logloss'], 'parameters': result['parameters'],
                'train_seconds': result['train_seconds'], 'peak_gpu_bytes': result['peak_gpu_bytes'],
                'inference_seconds': result['test']['inference_seconds'],
                'batch_latency_ms_p50': result['test']['batch_latency_ms_p50']},
                delta_macro_vs_din=result['test']['auc'] - din['test']['auc'],
                delta_macro_vs_previous=result['test']['auc'] - previous['test']['auc'],
                per_task={task: {'auc': value['auc'], 'logloss': value['logloss'],
                    'din_auc': din['test']['per_task'][task]['auc'],
                    'previous_auc': previous['test']['per_task'][task]['auc'],
                    'delta_vs_din': value['auc'] - din['test']['per_task'][task]['auc'],
                    'delta_vs_previous': value['auc'] - previous['test']['per_task'][task]['auc']}
                    for task, value in result['test']['per_task'].items()})
            lines.append(f"{dataset}: {primary} AUC {row['current_best']['primary_auc']:.8f}; native-task mean {result['test']['auc']:.8f}; delta DIN {row['delta_macro_vs_din']:+.8f}; delta prior model {row['delta_macro_vs_previous']:+.8f}")
            for task, value in row['per_task'].items():
                lines.append(f"  {task}: {value['auc']:.8f} | DIN delta {value['delta_vs_din']:+.8f} | previous model delta {value['delta_vs_previous']:+.8f}")
        else:
            lines.append(dataset + ': ' + row['status'])
        rows.append(row)
    write(root / 'report.json', {'at_unix': time.time(), 'source_model': read(root / 'setup.json')['source_model'],
          'source_selection': read(root / 'source-selection.json'), 'datasets': rows,
          'limitations': ['source is the final validation-ranked top topic representative; its source-domain test was observed before transfer',
                         'target test sets have been evaluated for a different model; exploratory follow-up',
                         'single seed; scratch architecture transfer, not zero-shot',
                         'MerRec has no click label: Like is its native primary task'],
          'no_search': True})
    (root / 'report.txt').write_text('\n'.join(lines) + '\n')


def supervise(root):
    root = root.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    with (root / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if read(root / 'interface-checks.json')['status'] != 'passed':
            raise ValueError('Interface checks must pass first')
        if read(root / 'cached-release-checks.json')['status'] != 'passed':
            raise ValueError('Dataset revalidation must pass first')
        if read(root / 'taobao-validation-replay/result.json')['status'] != 'passed':
            raise ValueError('Adapted Taobao checkpoint replay must pass before training')
        write(root / 'supervisor-process.json', {'pid': os.getpid(), 'started_unix': time.time()})
        jobs = {name: {'dataset': name, 'gpu': 4 + index, 'phase': 'ready'} for index, name in enumerate(DATASETS)}
        children = {}
        while True:
            if (root / 'STOP').exists():
                for child, output, record in children.values():
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                for child, output, record in children.values():
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                write(root / 'supervisor-status.json', {'phase': 'stopped', 'at_unix': time.time(), 'jobs': jobs})
                return
            for dataset, job in jobs.items():
                if job['phase'] in ['complete', 'failed', 'timeout']:
                    continue
                if dataset in children:
                    child, output, record = children[dataset]
                    code = child.poll()
                    if code is None:
                        if (output / 'progress.json').exists():
                            job['progress'] = read(output / 'progress.json')
                        cap = 1800 if job['phase'] == 'calibrating' else read(root / 'datasets' / dataset / 'protocol.json')['run_wall_cap_seconds']
                        if time.time() - record['started_unix'] > cap:
                            os.killpg(child.pid, signal.SIGTERM)
                            try:
                                child.wait(timeout=20)
                            except subprocess.TimeoutExpired:
                                os.killpg(child.pid, signal.SIGKILL)
                            job['phase'] = 'timeout'
                            write(output / 'failure.json', {'status': 'timeout', 'wall_seconds': time.time() - record['started_unix']})
                        continue
                    if code or not (output / 'result.json').exists():
                        job.update(phase='failed', exit_code=code)
                        if (output / 'failure.json').exists():
                            job['failure'] = read(output / 'failure.json')
                        continue
                    result = read(output / 'result.json')
                    if job['phase'] == 'calibrating':
                        if result['training_loss_hook_calls'] != (result['steps'] if read(root / 'setup.json').get('training_loss_hook_expected', True) else 0):
                            raise ValueError('Calibration did not execute the original loss hook')
                        job['calibration'] = result
                        job['phase'] = 'ready_fit'
                    else:
                        job.update(phase='complete', result=str(output / 'result.json'))
                        continue
                if job['phase'] not in ['ready', 'ready_fit']:
                    continue
                calibration = job['phase'] == 'ready'
                job['phase'] = 'calibrating' if calibration else 'training'
                child, output, record = spawn(root, dataset, job['gpu'], calibration)
                children[dataset] = (child, output, record)
                job.update(pid=child.pid, output=str(output), started_unix=record['started_unix'])
                job.pop('progress', None)
            report(root, jobs)
            finished = all(x['phase'] in ['complete', 'failed', 'timeout'] for x in jobs.values())
            write(root / 'supervisor-status.json', {'phase': 'finished' if finished else 'running',
                  'at_unix': time.time(), 'jobs': jobs})
            if finished:
                return
            time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    supervise(parser.parse_args().root)
