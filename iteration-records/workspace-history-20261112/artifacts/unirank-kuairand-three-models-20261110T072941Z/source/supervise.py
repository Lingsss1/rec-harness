"""Two new from-scratch fits on GPUs1/2; observe the existing HeMix GPU0 fit."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from common import read, write, sha

VARIANTS = [('joint1', 1), ('joint3', 2)]


def report(root, jobs):
    setup = read(root / 'setup.json')
    hemix = Path(setup['hemix_campaign'])
    audit = read(root / 'parameter-audit.json')
    locations = [('hemix80', hemix / 'runs/hemix_token80/fit-01/result.json', 0, 80, 3)]
    locations += [(kind, root / 'runs' / kind / 'fit-01/result.json', gpu, 256, 1 if kind == 'joint1' else 3)
                  for kind, gpu in VARIANTS]
    rows = []
    reference = read(root / 'reference/ours32-result.json')
    for name, path, gpu, width, depth in locations:
        status = read(hemix / 'status.json') if name == 'hemix80' else jobs[name]
        item = {'model': name, 'gpu': gpu, 'embedding_dim': 16, 'token_dim': width,
                'joint_layers': depth, 'status': status.get('phase', 'pending'),
                **audit[name], 'output': str(path.parent)}
        if path.exists():
            result = read(path)
            for task, metrics in result['test']['per_task'].items():
                b = reference['test']['per_task'][task]
                assert metrics['samples'] == b['samples'] and metrics['positives'] == b['positives']
            item.update(status='complete', validation_macro_auc=result['validation']['auc'],
                test_macro_auc=result['test']['auc'], test_macro_logloss=result['test']['logloss'],
                per_task=result['test']['per_task'], train_seconds=result['train_seconds'],
                inference_seconds=result['test']['inference_seconds'], peak_gpu_bytes=result['peak_gpu_bytes'],
                delta_macro_vs_ours32=result['test']['auc'] - reference['test']['auc'],
                result_path=str(path), checkpoint_sha256=result['checkpoint_sha256'])
        rows.append(item)
    write(root / 'report.json', {'at_unix': time.time(), 'dataset': 'KuaiRand', 'rows': rows,
        'ours32_reference': {'test_macro_auc': reference['test']['auc'],
                             'source': setup['reference_campaign'], 'new_fit_performed': False},
        'only_two_new_fits': True, 'original_hemix_not_trained': True,
        'limitations': ['single seed', 'same dimensions do not imply equal dense parameter count',
                       'all target tests are supplementary exploratory comparisons; no test-based reselection']})
    lines = ['KuaiRand：HeMix80 / 我们256一层 / 我们256三层',
             f"历史参照 ours32 测试平均AUC {reference['test']['auc']:.8f}", '']
    for item in rows:
        text = (f"{item['model']}: {item['status']}; embedding16, token{item['token_dim']}, depth{item['joint_layers']}; "
                f"total {item['parameters']:,}, dense {item['dense_parameters']:,}")
        if item['status'] == 'complete':
            text += f"; test mean AUC {item['test_macro_auc']:.8f}; mean Logloss {item['test_macro_logloss']:.8f}; train {item['train_seconds']/3600:.3f}h"
        lines.append(text)
        for task, values in item.get('per_task', {}).items():
            lines.append(f"  {task}: AUC {values['auc']:.8f}; Logloss {values['logloss']:.8f}")
    (root / 'report.txt').write_text('\n'.join(lines) + '\n')
    return rows


def spawn(root, kind, gpu, calibration):
    output = root / 'runs' / kind / ('calibration-01' if calibration else 'fit-01')
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', str(root / 'source/train.py'), '--root', str(root),
               '--dataset', 'KuaiRand', '--kind', kind, '--output', str(output)]
    if calibration:
        command += ['--calibration-steps', '20']
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '4',
           'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'PYTHONHASHSEED': '20262027'}
    with (output / 'train.log').open('ab') as log:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, env=env)
    receipt = {'pid': child.pid, 'started_unix': time.time(), 'command': command, 'gpu': gpu,
               'start_ticks': Path('/proc/' + str(child.pid) + '/stat').read_text().split()[21]}
    write(output / 'process.json', receipt)
    return child, output, receipt


def stop(child):
    if child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)


def supervise(root):
    root = root.resolve()
    with (root / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert read(root / 'preflight.json')['status'] == 'passed'
        write(root / 'supervisor-process.json', {'pid': os.getpid(), 'started_unix': time.time()})
        jobs = {kind: {'phase': 'ready', 'gpu': gpu} for kind, gpu in VARIANTS}
        children = {}
        while True:
            if (root / 'STOP').exists():
                for child, output, receipt in children.values():
                    stop(child)
                write(root / 'status.json', {'phase': 'stopped', 'at_unix': time.time(), 'jobs': jobs,
                                             'hemix_external_fit_untouched': True})
                return
            for kind, job in jobs.items():
                if job['phase'] in ['complete', 'failed', 'timeout']:
                    continue
                if kind in children:
                    child, output, receipt = children[kind]
                    code = child.poll()
                    if code is None:
                        cap = 1800 if job['phase'] == 'calibrating' else 172800
                        if time.time() - receipt['started_unix'] > cap:
                            stop(child)
                            job['phase'] = 'timeout'
                        continue
                    if code or not (output / 'result.json').exists():
                        job.update(phase='failed', exit_code=code)
                        continue
                    result = read(output / 'result.json')
                    expected = read(root / 'parameter-audit.json')[kind]
                    for key in ['parameters', 'embedding_parameters', 'dense_parameters']:
                        assert result[key] == expected[key], kind + '/' + key
                    job['phase'] = 'ready_fit' if job['phase'] == 'calibrating' else 'complete'
                    if job['phase'] == 'complete':
                        continue
                calibration = job['phase'] == 'ready'
                child, output, receipt = spawn(root, kind, job['gpu'], calibration)
                children[kind] = (child, output, receipt)
                job.update(phase='calibrating' if calibration else 'training', pid=child.pid,
                           output=str(output), started_unix=receipt['started_unix'])
            rows = report(root, jobs)
            finished = all(j['phase'] in ['complete', 'failed', 'timeout'] for j in jobs.values())
            phase = 'running' if not finished else 'complete' if all(r['status'] == 'complete' for r in rows) else 'finished_with_incomplete_results'
            write(root / 'status.json', {'phase': phase, 'at_unix': time.time(), 'jobs': jobs,
                                         'hemix_external_fit_untouched': True})
            if finished:
                return
            time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    supervise(parser.parse_args().root)
