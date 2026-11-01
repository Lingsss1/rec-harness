"""Original 74a927f RecHarness, four corrected cold starts, isolated matched execution."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import fcntl
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
sys.path.insert(0, str(WORKSPACE))
from amazon_family_compare.bootstrap import bootstrap
bootstrap()
from amazon_family_compare.core import Controller as BaseController, read, save
from amazon_family_compare.controller import FamilyController
from amazon_family_compare.recharness import RecHarnessController, native, digest
from amazon_family_compare.prepare import REPO, sha, DATASETS

FAMILIES = ['gru4rec', 'nextitnet', 'bert4rec', 'hstu']
COMMIT = '74a927fefaf7d120c52b6e3a0809958bd33984f0'
PYTHON = WORKSPACE / 'artifacts/MLEvolve/.venv/bin/python'


class OriginalCorrectedController(RecHarnessController, FamilyController):
    """Original native routing; family-aware transport and matched host grading."""
    def setup(self):
        super().setup()
        family = self.c['model_family']
        os.environ['GAGC_COLD_START'] = family + '_perdataset'
        (self.workspace / 'predict.py').write_text(
            (REPO / 'gagc/templates' / (family + '_perdataset') / 'predict.py').read_text())
        self.event('original_native_setup', reference_commit=COMMIT, family=family,
                   cold_start_sha256=self.c['baseline']['sha256'], ensemble=False)

    def trial(self, key, source, gpu, requested=None, files=None, smoke=False):
        # Native round/slot/attempt identities must remain unique. A repeated
        # proposal is a fresh trial, never a replay with a duplicate GPU charge.
        return BaseController.trial(self, key, source, gpu, requested, files, smoke)

    def finish(self):
        # RecHarnessController.run has already frozen the native incumbent after
        # promotion/rollback. Do not replace it with the global-best candidate.
        trial = Path(self.best['output'])
        audit = read(trial / 'single-model-audit.json')
        if audit.get('eligible') is not True or audit.get('registered_family') != self.c['model_family']:
            raise RuntimeError('Native incumbent failed single-model/family eligibility')
        if sha(trial / 'working/predict.py') != audit['predictor_sha256']:
            raise RuntimeError('Selected predictor changed after eligibility audit')
        for path, expected in audit['checkpoint_sha256'].items():
            if sha(path) != expected:
                raise RuntimeError('Selected checkpoint changed after eligibility audit')
        save(self.root / 'single-model-selection.json', {'policy': 'original native incumbent after promotion/rollback',
             'selected': self.brief(self.best), 'family': self.c['model_family'], 'audit': audit})
        return BaseController.finish(self)


def files():
    paths = [Path(__file__).resolve(), ROOT / 'plan.json', ROOT / 'baseline-provenance.json']
    paths += list((ROOT / 'baselines').glob('*/train.py'))
    paths += list((ROOT / 'runs').glob('*/config.json'))
    paths += list((WORKSPACE / 'amazon_family_compare').glob('*.py'))
    paths += [p for p in (REPO / 'gagc').rglob('*.py') if '__pycache__' not in p.parts]
    paths += [WORKSPACE / 'staged-autoresearch/staged_research/codex.py']
    config = read(ROOT / 'runs/gru4rec/config.json')
    data = Path(config['data_dir'])
    paths += [data / 'prepared.json', data / 'trainval/dataset_stats.json']
    for name in DATASETS:
        paths += [data / 'trainval' / (name + suffix) for suffix in ['_train.txt', '_valid.txt', '_text_name_dict.json.gz']]
        paths += [data / 'test' / (name + '_test.txt')]
    return {str(p): sha(p) for p in sorted(set(paths))}


def check():
    if files() != read(ROOT / 'frozen.json')['files']:
        raise RuntimeError('Frozen original/corrected experiment source, config or data changed')


def prepare():
    if (ROOT / 'plan.json').exists():
        raise RuntimeError('Existing experiment must not be overwritten')
    head = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    tracked = subprocess.check_output(['git', '-C', str(REPO), 'diff', 'HEAD', '--', 'gagc'], text=True)
    if head != COMMIT or tracked.strip():
        raise RuntimeError('Original native RecHarness is not clean at the requested commit')
    provenance = read(WORKSPACE / 'artifacts/model-family-comparison-20261029/audit/baseline-provenance.json')
    jobs = []
    registered = {}
    for family in FAMILIES:
        info = provenance[family]
        if sha(info['corrected_source']) != info['corrected_sha256']:
            raise RuntimeError('Previously verified corrected baseline changed: ' + family)
        destination = ROOT / 'baselines' / family / 'train.py'
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(info['corrected_source'], destination)
        registered[family] = {**info, 'selected_source': str(destination), 'selected_sha256': sha(destination)}
        config = deepcopy(read(WORKSPACE / 'artifacts/model-family-comparison-20261029/runs' / (family + '-mlestar') / 'config.json'))
        folder = ROOT / 'runs' / family
        config.update(root=str(folder), methods=['recharness'], method='recharness', model_family=family,
            gpus=[0, 1, 2, 3], cpu_cores=list(range(64)), search_budget_gpu_seconds=43200,
            worker_python=str(PYTHON), controller_python=str(PYTHON), single_model_search=True,
            llm_model='gpt-6.1-sol', llm_effort='medium', llm_slot_directory=str(ROOT / 'llm-slots'),
            global_llm_concurrency=4, baseline={'source': str(destination), 'sha256': sha(destination)})
        config['alignment'].update(reference_commit=COMMIT, cold_start=family + '_perdataset_corrected',
            method_adaptations={'recharness': {
                'native': 'unmodified 74a927f propose/execute-group/promote/update state, Thompson routing and memory',
                'host': 'existing matched Amazon worker, family audit, single-model audit and final grader',
                'transport': 'existing resilient Codex gpt-6.1-sol/medium, same-request retry',
                'selection': 'original native incumbent after promotion and rollback; held-out test once',
                'correction': 'verified corrected cold start only; no jump-v2/v3/v4 policy changes'}})
        config['replication'] = {'repetition': 1, 'fresh_baseline_and_llm_calls': True,
             'cross_run_response_or_memory_reuse': False, 'search_gpu_seconds': 43200}
        save(folder / 'config.json', config)
        jobs.append({'family': family, 'config': str(folder / 'config.json'), 'root': str(folder)})
    save(ROOT / 'baseline-provenance.json', registered)
    save(ROOT / 'plan.json', {'reference_commit': COMMIT, 'jobs': jobs, 'gpus': [0, 1, 2, 3],
        'search_gpu_seconds_per_job': 43200, 'total_search_gpu_hours': 48,
        'repetitions_per_family': 1, 'ensemble': False, 'llm_model': 'gpt-6.1-sol', 'llm_effort': 'medium',
        'queue_order': FAMILIES, 'data': 'same four Amazon categories and seed-0 splits',
        'training_seed': 42, 'evaluation_seed': 42, 'prepared_at': time.time()})
    save(ROOT / 'frozen.json', {'files': files(), 'time': time.time()})
    print('Prepared four original-RecHarness/corrected-model runs', flush=True)


def preflight():
    check()
    directory = ROOT / 'preflight'
    directory.mkdir(exist_ok=False)
    with (directory / 'cpu-tests.log').open('w') as log:
        subprocess.run([str(PYTHON), '-m', 'pytest', 'amazon_family_compare/test_families.py',
            'amazon_family_compare/test_accounting.py', 'amazon_family_compare/test_protocol_fixes.py',
            'amazon_family_compare/test_resilient_self.py', '-q'], cwd=WORKSPACE,
            stdout=log, stderr=subprocess.STDOUT, check=True)
    jobs = read(ROOT / 'plan.json')['jobs']
    def baseline(pair):
        job, gpu = pair
        config = read(job['config'])
        config['root'] = str(directory / job['family'])
        controller = OriginalCorrectedController(config, smoke=True)
        result = controller.trial('baseline', controller.baseline(), gpu, requested=900, smoke=True)
        if result['status'] != 'success':
            raise RuntimeError('Corrected model GPU preflight failed: ' + json.dumps(result))
        audit = read(Path(result['output']) / 'single-model-audit.json')
        if audit.get('registered_family') != job['family']:
            raise RuntimeError('Corrected model family audit failed')
        return {'family': job['family'], 'receipt': result}
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(baseline, zip(jobs, range(4))))
    save(directory / 'four-models.json', {'passed': True, 'models': receipts})
    config = read(jobs[0]['config'])
    config['root'] = str(directory / 'native-workflow')
    controller = OriginalCorrectedController(config, smoke=True)
    controller.setup()
    specs = json.loads(native.propose_action_group(controller.store.state().to_json()))
    baseline_results = controller.group(specs)
    if len(baseline_results) != 1 or not native._is_valid_trial(baseline_results[0]):
        raise RuntimeError('Original native baseline workflow failed')
    specs = json.loads(native.propose_action_group(controller.store.state().to_json(), json.dumps(baseline_results[0])))
    if len(specs) != 4:
        raise RuntimeError('Original native Thompson workflow did not select four trials')
    results = controller.group(specs, baseline_results[0])
    if not all(native._is_valid_trial(r) for r in results):
        raise RuntimeError('Original native mutation/promotion preflight failed: ' + json.dumps(results))
    state = controller.store.state()
    elapsed = sum(float(r['wall_time_secs']) for r in baseline_results + results)
    if abs(state.global_budget - (43200 - elapsed)) > 1e-5 or state.round_idx != 2 or not state.baseline_done:
        raise RuntimeError('Original native accounting/state preflight failed')
    save(directory / 'acceptance.json', {'passed': True, 'time': time.time(), 'corrected_baselines': receipts,
        'native_trials': results, 'native_round': state.round_idx, 'llm_model': config['llm_model'],
        'native_smoke_gpu_seconds': elapsed, 'budget': 'outside formal search'})
    print('Four corrected models and original native workflow passed preflight', flush=True)


def worker(family):
    check()
    config = read(ROOT / 'runs' / family / 'config.json')
    folder = Path(config['root']) / 'recharness'
    folder.mkdir(exist_ok=True)
    lock = (folder / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder / 'events.jsonl').exists():
        raise RuntimeError('Formal job already started; no silent replay')
    os.sched_setaffinity(0, set(config['cpu_cores']))
    try:
        OriginalCorrectedController(config).run()
    except BaseException:
        save(folder / 'failure.json', {'error': traceback.format_exc(), 'time': time.time()})
        raise


def pipeline():
    lock = (ROOT / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    leases = []
    for gpu in range(4):
        handle = (WORKSPACE / 'artifacts/gpu-leases' / f'gpu-{gpu}.lock').open('a')
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                save(ROOT / 'status.json', {'status': 'waiting_for_gpu_lease', 'gpu': gpu, 'time': time.time()})
                time.sleep(15)
        leases.append(handle)
    completed = []
    try:
        while True:
            output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'], text=True)
            inventory = {int(line.split(',')[0]): [int(v.strip()) for v in line.split(',')[1:]] for line in output.splitlines()}
            if all(inventory[g][0] < 512 and inventory[g][1] < 5 for g in range(4)):
                break
            save(ROOT / 'status.json', {'status': 'waiting_for_idle_gpus', 'inventory': inventory, 'time': time.time()})
            time.sleep(15)
        save(ROOT / 'status.json', {'status': 'preflight', 'gpus': [0, 1, 2, 3], 'time': time.time()})
        preflight()
        for family in FAMILIES:
            check()
            save(ROOT / 'status.json', {'status': 'running', 'family': family, 'completed': completed,
                                       'gpus': [0, 1, 2, 3], 'time': time.time()})
            folder = ROOT / 'runs' / family
            with (folder / 'controller.log').open('w') as log:
                process = subprocess.Popen([str(PYTHON), '-u', str(Path(__file__).resolve()), 'worker', '--family', family],
                    cwd=WORKSPACE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                save(folder / 'launcher.json', {'pid': process.pid, 'time': time.time()})
                code = process.wait()
            if code:
                raise RuntimeError('Formal job failed; queue stopped: ' + family)
            completed.append(family)
        save(ROOT / 'status.json', {'status': 'complete', 'completed': completed, 'time': time.time()})
    except BaseException:
        save(ROOT / 'status.json', {'status': 'failed', 'completed': completed,
            'error': traceback.format_exc(), 'time': time.time()})
        raise


def launch():
    check()
    if (ROOT / 'launcher.json').exists():
        raise RuntimeError('Pipeline already launched')
    with (ROOT / 'pipeline.log').open('w') as log:
        process = subprocess.Popen([str(PYTHON), '-u', str(Path(__file__).resolve()), 'pipeline'],
            cwd=WORKSPACE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(ROOT / 'launcher.json', {'pid': process.pid, 'time': time.time(), 'gpus': [0, 1, 2, 3]})
    print(json.dumps({'launched': process.pid, 'root': str(ROOT), 'formal_jobs': 4, 'search_gpu_hours': 48}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'launch', 'pipeline', 'worker'])
    parser.add_argument('--family', choices=FAMILIES)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    {'prepare': prepare, 'launch': launch, 'pipeline': pipeline, 'worker': lambda: worker(args.family)}[args.action]()
