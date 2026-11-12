"""Run only width-80 HeMix: calibration, scratch fit, frozen-checkpoint test."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from common import read, write, sha


def report(root):
    ours_root = Path(read(root / 'setup.json')['reference_campaign'])
    ours_file = ours_root / 'runs/KuaiRand/candidate/fit-01/result.json'
    ours_path = root / 'reference/ours-result.json'
    if ours_file.exists() and not ours_path.exists():
        r = read(ours_file)
        protocol = read(ours_root / 'datasets/KuaiRand/protocol.json')
        assert r['status'] == 'success' and r['protocol_hash'] == protocol['identity']
        assert r['source_sha256'] == sha(ours_root / 'source/model.py')
        write(ours_path, r)
    def row(label, path):
        if not path.exists():
            return {'model': label, 'status': 'pending'}
        result = read(path)
        return {'model': label, 'status': 'complete', 'macro_auc': result['test']['auc'],
                'macro_logloss': result['test']['logloss'], 'parameters': result['parameters'],
                'embedding_parameters': result['embedding_parameters'], 'dense_parameters': result['dense_parameters'],
                'train_seconds': result['train_seconds'], 'inference_seconds': result['test']['inference_seconds'],
                'peak_gpu_bytes': result['peak_gpu_bytes'], 'per_task': result['test']['per_task']}
    rows = [row('ours_g3-r0018', ours_path),
            row('hemix_token80', root / 'runs/hemix_token80/fit-01/result.json')]
    if all(r['status'] == 'complete' for r in rows):
        for task, value in rows[0]['per_task'].items():
            other = rows[1]['per_task'][task]
            assert value['samples'] == other['samples'] and value['positives'] == other['positives']
    result = {'at_unix': time.time(), 'dataset': 'KuaiRand', 'rows': rows,
              'selection': 'all six native task AUCs, equal mean on validation',
              'original_hemix_not_trained': True, 'only_width_chosen_by_parameter_count_not_auc': True,
              'limitations': ['single seed', 'fixed architecture transfer with scratch retraining',
                             'target test split was observed for prior architectures']}
    write(root / 'report.json', result)
    lines = ['KuaiRand：稠密参数接近的 HeMix 与我们的方法（原版 HeMix 不训练）', '']
    for r in rows:
        if r['status'] != 'complete':
            lines.append(r['model'] + ': pending')
            continue
        lines.append(f"{r['model']}: mean AUC {r['macro_auc']:.8f}; Logloss {r['macro_logloss']:.8f}; total {r['parameters']:,}; dense {r['dense_parameters']:,}; train {r['train_seconds']/3600:.3f}h")
        for task, value in r['per_task'].items():
            lines.append(f"  {task}: AUC {value['auc']:.8f}; Logloss {value['logloss']:.8f}")
    (root / 'report.txt').write_text('\n'.join(lines) + '\n')


def supervise(root):
    root = root.resolve()
    with (root / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert read(root / 'preflight.json')['status'] == 'passed'
        write(root / 'supervisor-process.json', {'pid': os.getpid(), 'started_unix': time.time()})
        for calibration, phase, cap in [(True, 'calibrating', 1800), (False, 'training', 172800)]:
            output = root / 'runs/hemix_token80' / ('calibration-01' if calibration else 'fit-01')
            output.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-u', str(root / 'source/train.py'), '--root', str(root),
                       '--dataset', 'KuaiRand', '--kind', 'candidate', '--output', str(output)]
            if calibration:
                command += ['--calibration-steps', '20']
            environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '4',
                           'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'PYTHONHASHSEED': '20262027'}
            with (output / 'train.log').open('ab') as log:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, env=environment)
            start = time.time()
            write(output / 'process.json', {'pid': child.pid, 'started_unix': start, 'gpu': 0, 'command': command,
                  'start_ticks': Path('/proc/' + str(child.pid) + '/stat').read_text().split()[21]})
            while child.poll() is None:
                if (root / 'STOP').exists() or time.time() - start > cap:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                    write(root / 'status.json', {'phase': 'stopped' if (root / 'STOP').exists() else 'timeout',
                          'at_unix': time.time(), 'output': str(output)})
                    return
                write(root / 'status.json', {'phase': phase, 'at_unix': time.time(), 'output': str(output), 'gpu': 0})
                report(root)
                time.sleep(5)
            if child.returncode or not (output / 'result.json').exists():
                write(root / 'status.json', {'phase': 'failed', 'at_unix': time.time(),
                      'output': str(output), 'exit_code': child.returncode})
                return
            result = read(output / 'result.json')
            expected = read(root / 'parameter-audit.json')['hemix_token80']
            for key in ['parameters', 'embedding_parameters', 'dense_parameters']:
                assert result[key] == expected[key], key
        report(root)
        write(root / 'status.json', {'phase': 'complete', 'at_unix': time.time(),
              'output': str(root / 'runs/hemix_token80/fit-01'), 'original_hemix_not_trained': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    supervise(parser.parse_args().root)
