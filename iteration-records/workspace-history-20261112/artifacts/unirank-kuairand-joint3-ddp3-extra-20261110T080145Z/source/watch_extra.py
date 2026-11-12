"""Supervise only the additional three-layer DDP fit; never signal old jobs."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from common import read, write, sha


def supervise(root):
    with (root / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        setup = read(root / 'execution-freeze.json')
        base = Path(setup['base_campaign'])
        assert read(base / 'datasets/KuaiRand/protocol.json')['identity'] == setup['base_protocol_hash']
        assert sha(root / 'source/train_ddp.py') == setup['executor_sha256']
        assert sha(base / 'datasets/KuaiRand/joint3-config.json') == setup['model_config_sha256']
        output = root / 'fit-01'
        output.mkdir(exist_ok=True)
        assert not (output / 'process.json').exists(), 'Use a new attempt directory'
        command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                   '--nproc-per-node=3', str(root / 'source/train_ddp.py'), '--root', str(base),
                   '--kind', 'joint3', '--output', str(output)]
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': ','.join(map(str, setup['gpus'])),
                       'NCCL_IB_DISABLE': '1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                       'OPENBLAS_NUM_THREADS': '4', 'PYTHONHASHSEED': '20262027'}
        with (output / 'train.log').open('ab') as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, env=environment)
        started = time.time()
        write(output / 'process.json', {'pid': child.pid, 'started_unix': started, 'command': command,
              'gpus': setup['gpus'], 'start_ticks': Path('/proc/' + str(child.pid) + '/stat').read_text().split()[21]})
        write(root / 'supervisor-process.json', {'pid': os.getpid(), 'started_unix': started})
        while child.poll() is None:
            if (root / 'STOP').exists() or time.time() - started > 172800:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                write(root / 'status.json', {'phase': 'stopped' if (root / 'STOP').exists() else 'timeout',
                      'at_unix': time.time(), 'output': str(output), 'existing_jobs_untouched': True})
                return
            write(root / 'status.json', {'phase': 'training', 'at_unix': time.time(), 'output': str(output),
                  'gpus': setup['gpus'], 'world_size': 3, 'global_batch_size': 8192})
            time.sleep(5)
        if child.returncode or not (output / 'result.json').exists():
            write(root / 'status.json', {'phase': 'failed', 'at_unix': time.time(),
                  'exit_code': child.returncode, 'output': str(output)})
            return
        result = read(output / 'result.json')
        assert result['status'] == 'success' and result['world_size'] == 3 and result['global_batch_size'] == 8192
        assert result['effective_config']['token_dim'] == 256 and result['effective_config']['joint_layers'] == 3
        assert result['executor_sha256'] == setup['executor_sha256']
        for key, expected in setup['parameter_counts'].items():
            assert result[key] == expected, key
        assert result['optimizer_steps'] == setup['global_optimizer_steps']
        protocol = read(base / 'datasets/KuaiRand/protocol.json')
        assert result['test']['samples'] == protocol['sample_size']['test']
        assert result['validation']['samples'] == protocol['sample_size']['valid']
        write(root / 'report.json', {'status': 'complete', 'completed_unix': time.time(),
              'execution_freeze_sha256': sha(root / 'execution-freeze.json'), 'result': result,
              'base_protocol_hash': setup['base_protocol_hash'], 'existing_jobs_restarted': False})
        write(root / 'status.json', {'phase': 'complete', 'at_unix': time.time(), 'output': str(output)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    supervise(parser.parse_args().root.resolve())
