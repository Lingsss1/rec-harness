"""Start the two queued pinned baselines beside the existing jobs, then join safely.

This sidecar leaves the live coordinator and its frozen Python files untouched.
An owned STOP marker retires its queued jobs after the current jobs complete;
the sidecar publishes completion only after acquiring the original output lock.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from .core import ContractError, canonical, digest, read_json, write_json
from .kuairand_baseline import COMMIT, MODELS, command, environment, lock
from .process_identity import identity, session_live

ADDED = {'dare': [0, 1], 'ccformer': [2, 3]}


def latest_job(output, model):
    paths = sorted((Path(output)/'jobs'/model).glob('attempt-*.json'))
    return read_json(paths[-1]) if paths else None


def owns_marker(output, token):
    try: return read_json(Path(output)/'STOP').get('collocation_owner') == token
    except (FileNotFoundError, ValueError, AttributeError): return False


def arm(output, token):
    """Retire only queued work, while both original jobs are far from finishing."""
    output = Path(output)
    state = read_json(output/'controller.json')
    if state['status'] != 'running' or state['stage'] != 'formal-training':
        raise ContractError('live baseline must be in formal training for this handoff')
    if (output.parent/'STOP').exists(): raise ContractError('suite stop is already requested')
    for model in ('hstu', 'ultra_hstu'):
        if (output/model/'final.json').exists() or read_json(output/model/'latest.json')['epoch'] >= 98:
            raise ContractError('primary job is too close to queue handoff; reconcile its worker first')
    for model in ADDED:
        job = latest_job(output, model)
        if job and job['status'] == 'running' and ('worker' not in job or session_live(job['worker'])):
            raise ContractError('queued model is already running: '+model)
        if (output/model).exists() and not (output/model/'last.pt').exists() and not (output/model/'final.json').exists():
            raise ContractError('preserve and audit unfinished output before launching '+model)
    # Exclusive creation cannot replace an operator's stop request.
    with (output/'STOP').open('x') as handle:
        handle.write(canonical({'collocation_owner': token,
            'purpose': 'Retire the original queued launches; current HSTU/ULTRA jobs continue.'})+'\n')
        handle.flush(); os.fsync(handle.fileno())
    return state


def train_argv(config, model):
    output, data = Path(config['output']).resolve(), Path(config['data_root']).resolve()
    argv = [str(Path(config['python']).absolute()), '-m', 'torch.distributed.run', '--standalone',
        '--nproc-per-node=2', '-m', 'longseq_bench.train', '--model', model,
        '--cache', str(data/'cache-l1024'), '--output', str(output/model),
        '--precision', 'fp32', '--allow-data-variant']
    if (output/model/'last.pt').exists(): argv += ['--resume', str(output/model/'last.pt')]
    return argv


def publish(config, token, scheduling):
    """Called only while holding the old coordinator's output lock."""
    output, benchmark = Path(config['output']).resolve(), Path(config['benchmark']).resolve()
    if not all((output/name/'final.json').exists() for name in MODELS):
        raise ContractError('all four final results are required before publication')
    workspace = Path(__file__).resolve().parents[1]
    command([str(Path(config['python']).absolute()), str(benchmark/'scripts/summarize.py'),
        str(output), '--require-complete'], output/'jobs/summary', cwd=benchmark,
        env=environment(benchmark, workspace, []))
    state = read_json(output/'controller.json')
    write_json(output/'summary-scheduling.json', {**scheduling,
        'cost_qualification': 'Per-run rank times include co-resident contention. Summed reservations overlap on physical GPUs; not exclusive model costs.'})
    write_json(output/'controller.json', {**state, 'status': 'completed', 'stage': 'complete',
        'scheduling_receipt': 'summary-scheduling.json', 'completed_at': time.time()})
    if owns_marker(output, token): (output/'STOP').unlink()


def terminate_owned(output, snapshots):
    """Cancel only workers created by this sidecar, never the original jobs."""
    for model in ADDED:
        row = latest_job(output, model)
        if not row or row.get('status') != 'running' or row == snapshots.get(model): continue
        worker = row.get('worker')
        if not worker: continue
        try:
            current = identity(worker['pid'])
            if current['start_ticks'] == worker['start_ticks'] and current['boot_id'] == worker['boot_id']:
                os.killpg(worker['pid'], signal.SIGTERM)
        except (FileNotFoundError, ProcessLookupError): pass


def run(config):
    output, benchmark = Path(config['output']).resolve(), Path(config['benchmark']).resolve()
    workspace = Path(__file__).resolve().parents[1]
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=benchmark, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', '.'], cwd=benchmark, text=True)
    if actual != COMMIT or dirty: raise ContractError('pinned benchmark checkout differs')
    if config['gpus'] != [0, 1, 2, 3]: raise ContractError('baseline GPU allocation differs')
    with lock(output/'.collocation.lock'):
        path = output/'co-location.json'
        if path.exists(): raise ContractError('collocation already has a receipt; preserve it and inspect before retry')
        token = uuid.uuid4().hex
        previous = {m: latest_job(output, m) for m in ADDED}
        initial = arm(output, token)
        if initial['config_hash'] != digest(config):
            if owns_marker(output, token): (output/'STOP').unlink()
            raise ContractError('baseline config drift')
        state = {'status': 'starting', 'owner': token, 'started_at': time.time(),
            'user_instruction': 'Start the other two models concurrently on the same GPUs.',
            'original_jobs_untouched': ['hstu', 'ultra_hstu'], 'added_jobs': ADDED,
            'config_hash': digest(config), 'queue_handoff': 'owned STOP, then original controller lock',
            'shared_gpu_costs': True}
        write_json(path, state)
        def one(model, gpus):
            if (output/model/'final.json').exists(): return
            if (output.parent/'STOP').exists(): raise ContractError('suite stop requested before added launch')
            command(train_argv(config, model), output/'jobs'/model, cwd=workspace,
                env=environment(benchmark, workspace, gpus), gpus=gpus)
        failure = None
        pool = ThreadPoolExecutor(2)
        try:
            futures = [pool.submit(one, model, gpus) for model, gpus in ADDED.items()]
            state['status'] = 'running'; write_json(path, state)
            for future in as_completed(futures): future.result()
        except BaseException as error:
            failure = str(error); terminate_owned(output, previous)
            state.update(status='fallback', error=failure); write_json(path, state)
            if owns_marker(output, token): (output/'STOP').unlink()
        finally: pool.shutdown(wait=True)
        if failure:
            # The old coordinator can still finish its surviving queue. Once it
            # releases the lock, resume any skipped job from its original checkpoint.
            while True:
                try:
                    with lock(output/'.controller.lock'): break
                except BlockingIOError: time.sleep(10)
            if (output.parent/'STOP').exists():
                state['status'] = 'stopped'; write_json(path, state); return
            from .kuairand_baseline import run as original_run
            original_run(config)
            state['status'] = 'completed_via_original_resume'; write_json(path, state); return
        state['status'] = 'waiting_primary_jobs'; write_json(path, state)
        while True:
            current = read_json(output/'controller.json')
            if current['status'] == 'blocked':
                state.update(status='blocked', error=current.get('error', 'primary coordinator blocked'))
                write_json(path, state); raise ContractError(state['error'])
            if all((output/name/'final.json').exists() for name in MODELS):
                try:
                    with lock(output/'.controller.lock'):
                        publish(config, token, state)
                    break
                except BlockingIOError: pass
            time.sleep(10)
        state.update(status='completed', completed_at=time.time()); write_json(path, state)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['launch', 'run', 'status'])
    p.add_argument('--config', default='examples/kuairand/baseline.json')
    a = p.parse_args(); config = read_json(a.config); output = Path(config['output']).resolve()
    if a.action == 'launch':
        argv = [sys.executable, '-m', 'rec_harness.kuairand_collocate', 'run', '--config', str(Path(a.config).resolve())]
        with (output/'co-location.log').open('a') as log:
            process = subprocess.Popen(argv, cwd=Path(__file__).resolve().parents[1], stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(output/'co-location-launch.json', {'argv': argv, 'worker': identity(process.pid)})
        print(canonical({'pid': process.pid, 'added_jobs': ADDED}))
    elif a.action == 'run': run(config)
    else: print(canonical(read_json(output/'co-location.json')))


if __name__ == '__main__': main()
