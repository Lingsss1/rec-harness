"""Resumable bounded GPU comparisons; selection consumes validation only."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from rec_harness.agents import run_process
from rec_harness.core import Artifacts, ContractError, digest, file_hash, read_json, write_json
from rec_harness.store import Store
from .amazon_identity import record_version
from .contract_gate import run_gate
from rec_harness.process_identity import assert_stopped

DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']


def init_study(root, project, *, budget=14400, predecessor=None, continuation_required=False):
    root, project = Path(root).resolve(), Path(project).resolve()
    root.mkdir(parents=True, exist_ok=False)
    sources = {
        'fixed_baseline': project / 'results/nextitnet-fixedbase-project-12h-20260921/fixed_baseline/best.py',
        'legacy_incumbent': project / 'workspace/nextitnet-fixedbase-project-12h-20260921/nextitnet_perdataset/best.py',
    }
    frozen = root / 'frozen'; frozen.mkdir()
    for model_version, path in sources.items():
        shutil.copyfile(path, frozen / f'{model_version}.py')
    shutil.copyfile(Path(__file__).with_name('amazon_worker.py'), frozen / 'worker.py')
    data = project / 'input/amazon-project-defaults/trainval'
    data_hashes = {f'{ds}_{split}.txt': file_hash(data / f'{ds}_{split}.txt') for ds in DATASETS for split in ['train', 'valid']}
    config = {
        'study_id': root.name, 'datasets': DATASETS, 'development_seeds': [42], 'confirmation_seeds': [17, 83],
        'model_versions': {model_version: {'path': str(frozen / f'{model_version}.py'), 'sha256': file_hash(frozen / f'{model_version}.py')} for model_version in sources},
        'worker': str(frozen / 'worker.py'), 'worker_sha256': file_hash(frozen / 'worker.py'),
        'python': str(project / '.venv-run/bin/python'), 'data': str(data), 'data_hashes': data_hashes,
        'protocol': {'epochs': 200, 'batches': 200, 'eval_users': 1000, 'eval_seed': 20260923, 'patience': 20, 'eval_every': 5,
                     'metric': 'HR@10', 'aggregation': 'unweighted_mean_four_categories', 'minimum_effect': 0.005,
                     'checkpoint_selection': 'public_validation_only', 'ranking': 'positive_plus_99_unique_negatives',
                     'loss': 'numerically_stable_pairwise_BPR', 'cudnn_deterministic': False, 'initialization': 'independent_per_category_and_seed'},
        'budget_gpu_seconds': budget, 'job_timeout_seconds': 1800, 'gpu_ids': [0, 1, 2, 3],
        'trusted_baselines_only': True, 'hidden_evaluation': 'not_available_to_worker',
    }
    if predecessor:
        prior = Store(Path(predecessor) / 'memory')
        config['predecessor'] = {'path': str(predecessor), 'charged_gpu_seconds': prior.budget()['charged_or_reserved'], 'event_head': prior.verify()['head']}
        prior.close()
    if continuation_required:
        config['continuation_required']=True
    config['protocol_hash'] = digest(config['protocol'])
    write_json(root / 'study.json', config)
    write_json(root / 'study-lock.json', {'config_hash': digest(config)})
    store = Store(root / 'memory'); store.configure_budget(budget)
    if predecessor:
        cost = config['predecessor']['charged_gpu_seconds']
        store.reserve('predecessor-consumed', cost); store.settle('predecessor-consumed', cost, 'retained')
    store.event('study_registered', {'config_hash': digest(config), 'protocol_hash': config['protocol_hash'], 'model_versions': config['model_versions'], 'data_hashes': data_hashes})
    store.close()
    return config


def recover_study(root, store):
    # A released controller lock does not imply its GPU descendants have stopped.
    assert_stopped(root / 'jobs')
    pending = [r[0] for r in store.db.execute("SELECT id FROM budget WHERE status='running'")]
    receipts = {r.get('task_id', r['job_id']): r for r in
                (read_json(p) for p in (root / 'jobs').glob('*/completed.json'))}
    for task_id in pending:
        result = receipts.get(task_id)
        if result:
            store.settle(task_id, result['allocated_gpu_seconds'], result['status'])
            store.event('amazon_job_recovered', result)
        else:
            # Unknown work stays charged at the reservation, never refunded.
            store.settle(task_id, None, 'interrupted')


def reserve_attempt(root, store, job_id, timeout):
    task_id, attempt = job_id, 1
    while store.db.execute('SELECT 1 FROM budget WHERE id=?', (task_id,)).fetchone():
        attempt += 1
        task_id = f'{job_id}-retry-{attempt}'
    directory = root / 'jobs' / job_id
    if directory.exists():
        assert_stopped(directory)
        history = root / 'attempt-history' / job_id
        history.mkdir(parents=True, exist_ok=True)
        # Preserve every failed/interrupted file before any output is replaced.
        destination = history / f'before-{task_id}'
        directory.rename(destination)
    store.reserve(task_id, timeout)
    store.event('amazon_attempt_reserved', {'job_id': job_id, 'task_id': task_id, 'attempt': attempt})
    return task_id


def execute_job(config, root, model_version, dataset, seed, gpu, task_id=None):
    job_id = f'{model_version}-{dataset}-{seed}'
    directory = root / 'jobs' / job_id
    directory.mkdir(parents=True, exist_ok=True)
    source = config['model_versions'][model_version]
    if file_hash(source['path']) != source['sha256'] or file_hash(config['worker']) != config['worker_sha256']:
        raise ContractError('frozen source or worker changed')
    for split in ('train', 'valid'):
        name = f'{dataset}_{split}.txt'
        if file_hash(Path(config['data']) / name) != config['data_hashes'][name]:
            raise ContractError('frozen data changed')
    argv = [config['python'], '-u', config['worker'], '--source', source['path'], '--data', config['data'],
            '--dataset', dataset, '--output', str(directory), '--seed', str(seed)]
    for key in ('epochs', 'batches', 'eval_users', 'eval_seed', 'patience', 'eval_every'):
        argv.extend(['--' + key.replace('_', '-'), str(config['protocol'][key])])
    # Do not inherit model hyperparameter overrides or tracking/API credentials.
    env = {k: v for k, v in os.environ.items() if k in {'PATH', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH', 'CUDA_HOME'}}
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONHASHSEED=str(seed))
    write_json(directory / 'request.json', {'argv': argv, 'gpu': gpu, 'task_id': task_id or job_id, 'protocol_hash': config['protocol_hash'], 'source_hash': source['sha256'], 'contract_gate': 'pre-and-post-v1', 'execution_id': config.get('execution', {}).get('execution_id', 'direct-reviewed-v1')})
    started = time.monotonic()
    def remaining():
        return max(0, config['job_timeout_seconds'] - (time.monotonic() - started))
    pre = run_gate(config['python'], source, job_id, directory / 'pre-contract', env, remaining())
    post = None
    if pre['status'] != 'success':
        process = {'status': 'not_started', 'reason': 'pre-training contract failed'}
        result = {'status': 'contract_failure', 'metrics': None}
    elif remaining() <= 0:
        process = {'status': 'not_started', 'reason': 'shared deadline exhausted'}
        result = {'status': 'timeout', 'metrics': None}
    else:
        if config.get('execution'):
            from rec_harness.gpu_job import run_reviewed_job
            execution = config['execution']
            files = {'worker.py': {'path': config['worker'], 'sha256': config['worker_sha256']}, 'model.py': source}
            for split in ('train', 'valid'):
                name = f'{dataset}_{split}.txt'
                files['data/' + name] = {'path': str(Path(config['data']) / name), 'sha256': config['data_hashes'][name]}
            arguments = ['--source', '/inputs/model.py', '--data', '/inputs/data', '--dataset', dataset,
                         '--output', '/output', '--seed', str(seed), '--threads', '2']
            for key in ('epochs', 'batches', 'eval_users', 'eval_seed', 'patience', 'eval_every'):
                arguments += ['--' + key.replace('_', '-'), str(config['protocol'][key])]
            isolated = run_reviewed_job(directory / 'isolated', execution['recipe'], files, 'worker.py', arguments,
                        python=config['python'], device=gpu, uid_registry=root.parent / 'gpu-job-uids',
                        timeout=remaining(), artifacts=Artifacts(root / 'artifacts/objects'),
                        runtime_archive=execution.get('runtime_archive'))
            process = isolated['process']
            Artifacts(root / 'artifacts/objects').restore(isolated['collection']['artifacts'], directory)
        else:
            process = run_process(argv, root, directory / 'process', timeout=remaining(), env=env)
        result = read_json(directory / 'result.json') if process['status'] == 'success' and (directory / 'result.json').exists() else {'status': process['status'] if process['status'] != 'success' else 'invalid', 'metrics': None}
        if result['status'] == 'success':
            if not result.get('checkpoint_sha256') or not (directory / 'checkpoint.pt').is_file():
                result.update(status='contract_failure', metrics=None)
                post = {'status': 'invalid', 'reason': 'training checkpoint missing'}
            else:
                post = run_gate(config['python'], source, job_id, directory / 'post-contract', env, remaining(),
                                checkpoint=directory / 'checkpoint.pt', checkpoint_hash=result['checkpoint_sha256'])
                if post['status'] != 'success':
                    result.update(status='contract_failure', metrics=None)
    elapsed = time.monotonic() - started
    if process['status'] == 'not_started':
        process.update(completed_at=time.time(), started_at=None, wall_seconds=0)
    result.update(job_id=job_id, task_id=task_id or job_id, model_version=model_version, dataset=dataset, seed=seed,
                  source_hash=source['sha256'], worker_hash=config['worker_sha256'],
                  execution_id=config.get('execution', {}).get('execution_id', 'direct-reviewed-v1'),
                  allocated_gpu_seconds=elapsed, process=process, contract_gates={'pre': pre, 'post': post})
    write_json(directory / 'completed.json', result)
    return result


def run_study(root, model_versions=None, seeds=None, *, arms=None, execution_id=None):
    # Compatibility for old launch scripts; new API names model versions explicitly.
    if arms is not None:
        if model_versions is not None: raise ContractError("use model_versions only")
        model_versions = arms
    root = Path(root).resolve()
    config = read_json(root / 'study.json')
    if digest(config) != read_json(root / 'study-lock.json')['config_hash']:
        raise ContractError('study configuration changed')
    with (root / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        store = Store(root / 'memory')
        try:
            from .amazon_controls import load_config
            config = load_config(root, store)
            from .amazon_campaign import campaign_guard
            campaign_guard(config,store)
            if execution_id:
                from .amazon_execution import resolve_execution
                config['execution'] = resolve_execution(store, execution_id)
            if any(v not in config['model_versions'] for v in (model_versions or [])):
                raise ContractError('unregistered model version')
            allowed_seeds = config['development_seeds'] + config['confirmation_seeds']
            if any(seed not in allowed_seeds for seed in (seeds or [])):
                raise ContractError('seed outside frozen study')
            from .amazon_prefix_candidate import plan_guard, plan_remaining
            active_plan = plan_guard(store, list(model_versions or config['model_versions']),
                                     seeds or config['development_seeds'], config['confirmation_seeds'])
            recover_study(root, store)
            jobs = [(v, ds, seed) for v in (model_versions or config['model_versions']) for seed in (seeds or config['development_seeds']) for ds in config['datasets']]
            requested = list(jobs)
            def successful(job):
                path = root / 'jobs' / '-'.join(map(str, job)) / 'completed.json'
                if not path.exists() or read_json(path)['status'] != 'success':
                    return False
                if read_json(path).get('execution_id', 'direct-reviewed-v1') != (execution_id or 'direct-reviewed-v1'):
                    raise ContractError('completed job uses another execution version; use a new study')
                return True
            jobs = [job for job in jobs if not successful(job)]
            if jobs and config.get('execution',{}).get('runtime_archive'):
                from .amazon_execution import preflight_execution
                preflight_execution(store,config['execution'],config['python'])
            write_json(root / 'status.json', {'status': 'running', 'requested_jobs': requested, 'pending_jobs': jobs})
            gpu_ids = config['gpu_ids']
            with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
                from rec_harness.gpu_scheduler import dispatch
                task_ids = {}
                def launch(job, gpu):
                    model_version, dataset, seed = job
                    job_id = f'{model_version}-{dataset}-{seed}'
                    task_id = reserve_attempt(root, store, job_id, config['job_timeout_seconds'])
                    future = pool.submit(execute_job, config, root, model_version, dataset, seed, gpu, task_id)
                    task_ids[future] = task_id
                    return future
                def complete(job, gpu, future):
                    job_id = task_ids.pop(future)
                    try:
                        result = future.result()
                        store.settle(job_id, result['allocated_gpu_seconds'], result['status'])
                        store.event('amazon_job_completed', result)
                        print(json.dumps({'job': job_id, 'status': result['status'], 'metrics': result['metrics'], 'budget': store.budget()}), flush=True)
                    except Exception as error:
                        store.settle(job_id, None, 'invalid')
                        store.event('amazon_job_failed', {'job_id': job_id, 'error': str(error)})
                        print(json.dumps({'job': job_id, 'error': str(error)}), flush=True)
                jobs = dispatch(jobs, gpu_ids,
                    lambda: plan_remaining(store, active_plan) >= config['job_timeout_seconds'], launch, complete)
                if jobs: store.event('study_budget_stop', store.budget())
            results = [read_json(p) for p in (root / 'jobs').glob('*/completed.json')]
            results = [r for r in results if r.get('execution_id', 'direct-reviewed-v1') == (execution_id or 'direct-reviewed-v1')]
            report = {'jobs': results, 'budget': store.budget(), 'protocol_hash': config['protocol_hash'], 'aggregate_validation': {}}
            report['completed_seeds'] = {}
            for model_version in config['model_versions']:
                selected = [r for r in results if record_version(r) == model_version and r['status'] == 'success']
                complete_seeds = [seed for seed in sorted({r['seed'] for r in selected})
                                  if {r['dataset'] for r in selected if r['seed'] == seed} == set(config['datasets'])]
                selected = [r for r in selected if r['seed'] in complete_seeds]
                report['completed_seeds'][model_version] = complete_seeds
                if selected:
                    report['aggregate_validation'][model_version] = {metric: sum(r['metrics'][metric] for r in selected) / len(selected) for metric in selected[0]['metrics']}
            shared = set.intersection(*(set(v) for v in report['completed_seeds'].values()))
            report['paired_seed_ids'] = sorted(shared)
            report['paired_aggregate_validation'] = {}
            if shared:
                for model_version in config['model_versions']:
                    selected = [r for r in results if record_version(r) == model_version and r['seed'] in shared and r['status'] == 'success']
                    report['paired_aggregate_validation'][model_version] = {metric: sum(r['metrics'][metric] for r in selected) / len(selected) for metric in selected[0]['metrics']}
            write_json(root / 'validation-report.json', report)
            unfinished = [job for job in requested if not successful(job)]
            write_json(root / 'status.json', {'status': 'completed' if not unfinished else ('budget_stopped' if jobs else 'incomplete'), 'remaining_jobs': unfinished, 'completed_jobs': len(results)})
            return report
        finally:
            store.close()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('init'); p.add_argument('directory'); p.add_argument('--project', required=True)
    p = sub.add_parser('run'); p.add_argument('directory'); p.add_argument('--model-version', action='append'); p.add_argument('--seed', type=int, action='append'); p.add_argument('--execution')
    p = sub.add_parser('register-execution'); p.add_argument('directory'); p.add_argument('recipe'); p.add_argument('--runtime-template')
    args = parser.parse_args()
    if args.action == 'register-execution':
        from .amazon_execution import register_execution
        store = Store(Path(args.directory) / 'memory')
        try: result = register_execution(store, read_json(args.recipe), runtime_template=args.runtime_template)
        finally: store.close()
    else:
        result = init_study(args.directory, args.project) if args.action == 'init' else run_study(args.directory, model_versions=args.model_version, seeds=args.seed, execution_id=args.execution)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
