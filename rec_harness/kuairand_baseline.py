"""Pinned four-model baseline coordinator: two two-GPU slots, durable receipts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

from .core import ContractError, canonical, digest, read_json, write_json

COMMIT = 'a37797a3767e622641ef48a52a6276307a4126b3'
MODELS = ('hstu', 'ultra_hstu', 'dare', 'ccformer')


@contextmanager
def lock(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def command(argv, output, *, cwd, env, gpus=()):
    """A failed attempt is retained, never silently converted into success."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    attempts = sorted(output.glob('attempt-*.json'))
    attempt = len(attempts) + 1
    record = output / ('attempt-%04d.json' % attempt)
    state = {'argv': list(map(str, argv)), 'cwd': str(cwd), 'gpus': list(gpus),
             'started_at': time.time(), 'status': 'running', 'attempt': attempt}
    write_json(record, state)
    process = None
    try:
        with (output/('attempt-%04d.log' % attempt)).open('w') as log:
            process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state['pid'] = process.pid
            from .process_identity import identity
            state['worker'] = identity(process.pid)
            write_json(record, state)
            code = process.wait()
        state.update(returncode=code, status='completed' if code == 0 else 'failed')
    finally:
        if process is not None:
            # Preserve identity/cost, but never leave an old DDP group beside its resumed job.
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait()
        if state['status'] == 'running': state['status'] = 'interrupted'
        state['wall_seconds'] = time.time() - state['started_at']
        state['allocated_gpu_seconds'] = state['wall_seconds'] * len(gpus)
        write_json(record, state)
    if state['status'] != 'completed':
        raise ContractError('execution blocked; inspect ' + str(record))
    return state


def environment(benchmark, workspace, gpus):
    return {**os.environ, 'CUDA_VISIBLE_DEVICES': ','.join(map(str, gpus)), 'OMP_NUM_THREADS': '4',
            'PYTHONPATH': str(workspace) + os.pathsep + str(benchmark)}


def run(config):
    benchmark, output = Path(config['benchmark']).resolve(), Path(config['output']).resolve()
    workspace = Path(__file__).resolve().parents[1]
    python = str(Path(config['python']).absolute())
    if config['gpus'] != [0, 1, 2, 3]: raise ContractError('baseline GPUs must be 0–3')
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=benchmark, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', '.'], cwd=benchmark, text=True)
    if actual != COMMIT or dirty: raise ContractError('baseline must be the unmodified pinned checkout')
    output.mkdir(parents=True, exist_ok=True)
    state = {'status': 'running', 'stage': 'preflight', 'commit': COMMIT, 'config_hash': digest(config)}
    with lock(output/'.controller.lock'):
        if (output/'controller.json').exists() and read_json(output/'controller.json')['config_hash'] != digest(config):
            raise ContractError('baseline configuration drift')
        write_json(output/'controller.json', state)
        def one(model, gpus, preflight):
            env = environment(benchmark, workspace, gpus)
            prefix = [python, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2', '-m']
            if preflight:
                for check in ('smoke', 'full-size'):
                    target = output/'preflight'/model/check
                    if (target/'passed.json').exists(): continue
                    if check == 'smoke':
                        job = prefix + ['longseq_bench.train', '--model', model, '--smoke', '--precision', 'fp32',
                                        '--output', str(target/'run')]
                        if (target/'run/last.pt').exists(): job += ['--resume', str(target/'run/last.pt')]
                        elif (target/'run').exists():
                            (target/'run').rename(target/('interrupted-%d' % time.time_ns()))
                    else:
                        job = prefix + ['rec_harness.kuairand_worker', 'full-size', '--benchmark', str(benchmark),
                                        '--model', model, '--output', str(target/'run')]
                    command(job, target, cwd=workspace, env=env, gpus=gpus)
                    write_json(target/'passed.json', {'kind': check, 'synthetic': True, 'model': model})
                return
            target = output/model
            if (target/'final.json').exists(): return
            job = prefix + ['longseq_bench.train', '--model', model, '--cache', str(Path(config['data_root']).resolve()/'cache-l1024'),
                '--output', str(target), '--precision', 'fp32', '--allow-data-variant']
            if (target/'last.pt').exists(): job += ['--resume', str(target/'last.pt')]
            elif target.exists(): target.rename(output/('interrupted-'+model+'-'+str(time.time_ns())))
            command(job, output/'jobs'/model, cwd=workspace, env=env, gpus=gpus)

        def slot(gpus, models, preflight):
            for model in models:
                if (output/'STOP').exists(): return
                one(model, gpus, preflight)

        try:
            # Both streams must finish all checks before ANY formal training.
            with ThreadPoolExecutor(2) as pool:
                futures = [pool.submit(slot, [0, 1], MODELS[::2], True),
                           pool.submit(slot, [2, 3], MODELS[1::2], True)]
                for future in futures: future.result()
            if (output/'STOP').exists(): return
            state['stage'] = 'prepare'
            write_json(output/'controller.json', state)
            command([python, '-m', 'longseq_bench.pipeline', '--prepare-only', '--data-root',
                     str(Path(config['data_root']).resolve()), '--output', str(output/'preparation')],
                    output/'jobs/preparation', cwd=workspace, env=environment(benchmark, workspace, []))
            state['stage'] = 'formal-training'
            write_json(output/'controller.json', state)
            with ThreadPoolExecutor(2) as pool:
                futures = [pool.submit(slot, [0, 1], MODELS[::2], False),
                           pool.submit(slot, [2, 3], MODELS[1::2], False)]
                for future in futures: future.result()
            if (output/'STOP').exists(): return
            command([python, str(benchmark/'scripts/summarize.py'), str(output), '--require-complete'],
                    output/'jobs/summary', cwd=benchmark, env=environment(benchmark, workspace, []))
            state.update(status='completed', stage='complete')
        except BaseException as error:
            state.update(status='blocked', error=str(error))
            raise
        finally: write_json(output/'controller.json', state)
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('config')
    a = p.parse_args(); print(canonical(run(read_json(a.config))))


if __name__ == '__main__': main()
