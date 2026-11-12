"""Post-search frozen top-k checkpoint assessment; no training or test-based reselection."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

from common import read, write, sha, identity
from selection import validate_policy, validation, checkpoint_key

CLOSED_LANES = {'research_stopped', 'cancelled', 'failed_before_assessment'}
CLOSED_CAMPAIGNS = {'topic_research_stopped', 'topic_research_with_failures'}


def ranked_unique(rows):
    eligible = [r for r in rows if r.get('status') == 'success' and r.get('metrics', {}).get('validation')]
    if len({r['id'] for r in eligible}) != len(eligible):
        raise ValueError('Duplicate trial identifiers')
    for row in eligible:
        validation(row)
    eligible.sort(key=lambda r: checkpoint_key(r['metrics']['validation']) +
                  (r['metrics'].get('parameters', math.inf), r['id']))
    ranked, seen, duplicates = [], {}, []
    for row in eligible:
        key = identity({'source': row['source_sha256'], 'checkpoint': row['metrics']['checkpoint_sha256'],
                        'effective_config': row['metrics']['effective_config']})
        if key in seen:
            duplicates.append({'id': row['id'], 'same_model_as': seen[key]})
        else:
            seen[key] = row['id']
            ranked.append(row)
    return ranked, duplicates


def topic_representatives(states, references):
    """Reconstruct the retained best after each completed round, one final model per topic."""
    catalog = {row['id']: row for row in references + [r for s in states for r in s['trials']]}
    representatives, summaries, traces = [], [], []
    for state in states:
        topics = {t['id']: t for t in state.get('topics', [])}
        cutoff = state.get('round', max((r.get('round', -1) for r in state['trials']), default=-1) + 1)
        grouped = {}
        for row in state['trials']:
            if row.get('round', -1) >= cutoff:
                continue  # A successful variant inside an unfinished round is not yet a retained round winner.
            topic_id = row.get('topic_id')
            if not topic_id:
                raise ValueError('A completed research trial has no topic identifier')
            grouped.setdefault(topic_id, []).append(row)
        for topic_id, rows in sorted(grouped.items()):
            topic = topics.get(topic_id, {})
            starting = list(references)
            for trial_id in [topic.get('entry_trial_id')] + topic.get('import_trial_refs', []):
                if trial_id in catalog:
                    starting.append(catalog[trial_id])
            eligible = list({r['id']: r for r in starting}.values())
            incumbent = None
            for index in sorted({r['round'] for r in rows}):
                eligible.extend(r for r in rows if r['round'] == index)
                ranked, _ = ranked_unique(eligible)
                incumbent = ranked[0] if ranked else None
                traces.append({'topic_id': topic_id, 'round': index,
                    'best_trial_id': incumbent['id'] if incumbent else None,
                    'validation_macro_auc': validation(incumbent)['macro_auc'] if incumbent else None})
            if incumbent:
                representative = copy.deepcopy(incumbent)
                representative['represented_topic_id'] = topic_id
                representative['represented_topic_ids'] = [topic_id]
                representatives.append(representative)
                entry = catalog.get(topic.get('entry_trial_id'))
                summaries.append({'topic_id': topic_id, 'title': topic.get('title', topic_id),
                    'best_trial_id': incumbent['id'], 'validation_macro_auc': validation(incumbent)['macro_auc'],
                    'completed_rounds': len({r['round'] for r in rows}),
                    'entry_trial_id': topic.get('entry_trial_id'), 'source_topic_ids': topic.get('source_topic_ids', []),
                    'gain_vs_entry': validation(incumbent)['macro_auc'] - validation(entry)['macro_auc'] if entry else None})
    return representatives, summaries, traces


def selection(root, top_k=3, include_baseline=True):
    if type(top_k) is not int or top_k < 1:
        raise ValueError('top_k must be a positive integer')
    status = read(root / 'status.json')
    if status['phase'] not in CLOSED_CAMPAIGNS or (root / 'STOP').exists():
        raise ValueError('Research must finish normally before automatic final selection')
    protocol = read(root / 'protocol.json')
    validate_policy(protocol['selection_policy'])
    states = [read(root / 'topics' / f'gpu-{g}' / 'state.json') for g in read(root / 'budget.json')['gpus']]
    if any(s['phase'] not in CLOSED_LANES for s in states):
        raise ValueError('Every research lane must close before assessment')
    references = copy.deepcopy(read(root / 'references.json')['trials'])
    rows = list(references)
    for state in states:
        rows.extend(state['trials'])
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate committed trial identifiers')
    representatives, topic_summaries, topic_trace = topic_representatives(states, references)
    # Several topics may retain the exact same imported starting checkpoint.
    by_trial = {}
    for row in representatives:
        if row['id'] in by_trial:
            by_trial[row['id']]['represented_topic_ids'] += row['represented_topic_ids']
        else:
            by_trial[row['id']] = row
    ranked, duplicates = ranked_unique(list(by_trial.values()))
    for duplicate in duplicates:
        by_trial[duplicate['same_model_as']]['represented_topic_ids'] += by_trial[duplicate['id']]['represented_topic_ids']
    picked = [(f'rank_{i + 1}', row, i + 1) for i, row in enumerate(ranked[:top_k])]
    baseline = next((r for r in rows if r.get('baseline_kind') == 'din' and r['status'] == 'success'), None)
    baseline_key = None
    if include_baseline and baseline:
        baseline_identity = (baseline['source_sha256'], baseline['metrics']['checkpoint_sha256'],
                             identity(baseline['metrics']['effective_config']))
        baseline_key = next((key for key, row, rank in picked if
            (row['source_sha256'], row['metrics']['checkpoint_sha256'], identity(row['metrics']['effective_config'])) == baseline_identity), None)
        if baseline_key is None:
            baseline_key = 'din'
            picked.append(('din', baseline, None))
    if not picked:
        raise ValueError('No completed topic checkpoint or baseline is available')
    for name, expected in protocol['source_sha256'].items():
        if sha(root / 'source' / name) != expected:
            raise ValueError('Frozen research host changed: ' + name)
    suite = []
    for key, row, rank in picked:
        source = Path(row['model_path'])
        checkpoint = Path(row['output']) / 'checkpoint.pt'
        metrics = row['metrics']
        if metrics['data_identity'] != protocol['data_identity'] or metrics['protocol_hash'] != protocol['identity']:
            raise ValueError('Checkpoint belongs to a different data/protocol identity')
        if sha(source) != row['source_sha256'] or sha(checkpoint) != metrics['checkpoint_sha256']:
            raise ValueError('Selected source/checkpoint changed: ' + row['id'])
        suite.append({'key': key, 'rank_by_validation': rank, 'id': row['id'], 'role': row['role'],
            'source_path': str(source), 'source_sha256': row['source_sha256'],
            'checkpoint_path': str(checkpoint), 'checkpoint_sha256': metrics['checkpoint_sha256'],
            'config': row['config'], 'effective_config': metrics['effective_config'], 'seed': row['seed'],
            'validation': metrics['validation'], 'parameters': metrics['parameters'],
            'train_seconds': metrics['train_seconds'], 'training_job_wall_seconds': row['wall_seconds'],
            'original_trial_output': row['output'], 'study_ref': row.get('study_ref'),
            'topic_id': row.get('represented_topic_id'), 'topic_ids': row.get('represented_topic_ids', [])})
    freeze = {'frozen_unix': time.time(), 'requested_top_k': top_k,
        'available_unique_models': len(ranked), 'selected_top_count': min(top_k, len(ranked)),
        'selection_metric': protocol['primary'], 'selection_policy': protocol['selection_policy'],
        'protocol_hash': protocol['identity'], 'data_identity': protocol['data_identity'],
        'suite': suite, 'din_reference_key': baseline_key, 'exact_duplicate_models': duplicates,
        'selection_scope': 'one final representative per actual topic ID, reconstructed from retained best checkpoints after completed rounds; top-k topic representatives by validation macro AUC',
        'topic_representatives': topic_summaries, 'topic_incumbent_trace': topic_trace,
        'duplicate_definition': 'same source SHA256, effective configuration and checkpoint SHA256',
        'test_used_for_selection': False, 'no_training': True,
        'assessor_sha256': sha(Path(__file__)),
        'test_policy': 'freeze the complete suite before evaluation; replay validation first; one test execution per unique model; preserve validation ranks',
        'evaluation_type': 'supplementary reuse of the official test split assessed in earlier campaigns; test inputs/results remain closed to this search',
        'search_budget': read(root / 'budget.json'),
        'budget_scope': 'User-requested final inference after research closes; costs recorded separately from the12h search; no extra training, tuning or research calls',
        'research_agent_information_boundary': 'User-requested side assessments are not supplied to research agents or used to rank topic checkpoints',
        'prior_user_requested_assessment': read(root / 'latest-immediate-assessment.json') if (root / 'latest-immediate-assessment.json').exists() else None,
        'search_summary': {'rounds': sum(s['completed_research_rounds'] for s in states),
            'training_attempts': sum(len(s['trials']) for s in states),
            'successful_attempts': sum(r['status'] == 'success' for s in states for r in s['trials'])}}
    freeze['identity'] = identity(freeze)
    return freeze


def live_research_children(root):
    alive = []
    for pattern in ['topics/**/process.json', 'topics/**/agent-process.json']:
        for path in root.glob(pattern):
            record = read(path)
            pid = record.get('pid')
            proc = Path('/proc/' + str(pid) + '/stat')
            try:
                fields = proc.read_text().split()
                if fields[2] != 'Z' and fields[21] == record.get('start_ticks'):
                    alive.append(pid)
            except FileNotFoundError:
                pass
    return alive


def wait_for_gpus(gpus, assessment, root=None):
    while True:
        if root is not None and (root / 'STOP').exists():
            raise ValueError('Assessment cancelled while waiting for GPUs')
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=10, check=True)
        usage = {int(row.split(',')[0]): int(row.split(',')[1]) for row in result.stdout.splitlines()}
        busy = {g: usage[g] for g in gpus if usage[g] > 512}
        if not busy:
            return
        write(assessment / 'status.json', {'phase': 'waiting_for_free_research_gpus', 'at_unix': time.time(), 'busy_memory_mib': busy})
        time.sleep(10)


def worker(root, assessment, key, split):
    if (root / 'STOP').exists():
        raise ValueError('Assessment cancelled by STOP file')
    freeze = read(assessment / 'selection-freeze.json')
    row = next(row for row in freeze['suite'] if row['key'] == key)
    output = assessment / key / split
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'worker-start.json').exists() or (output / 'result.json').exists():
        raise ValueError('This frozen checkpoint/split already started; no automatic duplicate evaluation')
    if split == 'test':
        gate = read(assessment / 'validation-replay-gate.json')
        if gate['status'] != 'passed' or gate['freeze_identity'] != freeze['identity']:
            raise ValueError('The entire frozen suite must pass validation replay before test access')
    write(output / 'worker-start.json', {'pid': os.getpid(), 'started_unix': time.time(), 'split': split,
          'freeze_identity': freeze['identity'], 'checkpoint_sha256': row['checkpoint_sha256']})
    sys.path.insert(0, str(root / 'source'))
    import train as host
    import numpy as np
    import torch
    started = time.monotonic()
    random.seed(row['seed'])
    np.random.seed(row['seed'])
    torch.manual_seed(row['seed'])
    torch.cuda.manual_seed_all(row['seed'])
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    if sha(row['source_path']) != row['source_sha256'] or sha(row['checkpoint_path']) != row['checkpoint_sha256']:
        raise ValueError('Frozen model artifacts changed')
    protocol = read(root / 'protocol.json')
    if protocol['identity'] != freeze['protocol_hash']:
        raise ValueError('Frozen research protocol changed')
    # Preserve the original constructor model_id, without saving to the original trial.
    model, fmap, params, effective = host.make_model(root, Path(row['source_path']), row['config'],
                                                    Path(row['original_trial_output']), row['seed'])
    if identity(effective) != identity(row['effective_config']):
        raise ValueError('Restored effective model configuration differs')
    model.load_state_dict(torch.load(row['checkpoint_path'], map_location='cpu', weights_only=True), strict=True)
    if split == 'test':
        params['test_data'] = str(root / 'data/Taobao_Action/test/data')
    metrics = host.score(model, host.loader(root, fmap, params, split), list(fmap.labels), torch.device('cuda:0'), output)
    expected = read(root / 'data/Taobao_Action/meta_data.json')['sample_size'][split]
    if metrics['samples'] != expected:
        raise ValueError('Incomplete sample coverage')
    if split == 'valid':
        for task, value in metrics['per_task'].items():
            for metric in ['auc', 'logloss']:
                if abs(value[metric] - row['validation']['per_task'][task][metric]) > 1e-8:
                    raise ValueError('Checkpoint validation replay mismatch: ' + task + '/' + metric)
    write(output / 'result.json', {'status': 'success', 'key': key, 'id': row['id'], 'split': split,
          'rank_by_validation': row['rank_by_validation'], 'metrics': metrics, 'seed': row['seed'],
          'parameters': sum(p.numel() for p in model.parameters()), 'no_training': True,
          'source_sha256': row['source_sha256'], 'checkpoint_sha256': row['checkpoint_sha256'],
          'freeze_identity': freeze['identity'], 'wall_seconds': time.monotonic() - started,
          'hardware': torch.cuda.get_device_name(), 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES')})


def assess(root, assessment, top_k, include_baseline):
    if (assessment / 'selection-freeze.json').exists():
        raise ValueError('This final assessment is already frozen; do not automatically reopen')
    started = time.monotonic()
    freeze = selection(root, top_k, include_baseline)
    # The freeze is durable BEFORE any loader (including test) is opened.
    write(assessment / 'selection-freeze.json', freeze)
    status = read(root / 'status.json')
    status.update(assessment_launched=True, automatic_assessment='final-assessment-top3',
                  assessment_requested_unix=time.time())
    write(root / 'status.json', status)
    for row in freeze['suite']:
        package = assessment / 'suite' / row['key']
        package.mkdir(parents=True)
        shutil.copy2(row['source_path'], package / 'model.py')
        shutil.copy2(row['checkpoint_path'], package / 'checkpoint.pt')
        write(package / 'model-config.json', row['config'])
        write(package / 'manifest.json', row)
    for record in read(root / 'dataset-source-manifest.json')['files']:
        if sha(root / 'data/Taobao_Action' / record['path']) != record['local_sha256']:
            raise ValueError('Released data file changed: ' + record['path'])
    gpus = read(root / 'budget.json')['gpus']

    def execute(item):
        gpu, row, split = item
        output = assessment / row['key'] / split
        output.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--root', str(root),
                   '--assessment-dir', str(assessment), '--worker', row['key'], '--split', split]
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '4',
               'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}
        with (output / 'process.log').open('ab') as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
            write(output / 'process.json', {'pid': process.pid, 'gpu': gpu, 'started_unix': time.time(), 'argv': command,
                  'start_ticks': Path('/proc/' + str(process.pid) + '/stat').read_text().split()[21]})
            expires = time.monotonic() + 600
            while process.poll() is None:
                if (root / 'STOP').exists() or time.monotonic() >= expires:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise ValueError('Final inference cancelled or exceeded600s: ' + row['id'])
                time.sleep(.5)
            code = process.returncode
        if code:
            raise ValueError('Final inference failed; inspect ' + str(output / 'process.log'))
        return read(output / 'result.json')

    all_results = {}
    for split in ['valid', 'test']:
        if (root / 'STOP').exists():
            raise ValueError('Assessment cancelled by STOP file')
        wait_for_gpus(gpus[:min(len(gpus), len(freeze['suite']))], assessment, root)
        if (root / 'STOP').exists():
            raise ValueError('Assessment cancelled by STOP file')
        write(assessment / 'status.json', {'phase': 'validation_replay' if split == 'valid' else 'test',
              'at_unix': time.time(), 'freeze_identity': freeze['identity'], 'models': [r['id'] for r in freeze['suite']]})
        results = []
        # Batch scheduling never assigns simultaneous models to the same GPU.
        for start in range(0, len(freeze['suite']), len(gpus)):
            batch = freeze['suite'][start:start + len(gpus)]
            with ThreadPoolExecutor(len(batch)) as executor:
                results += list(executor.map(execute, [(gpu, row, split) for gpu, row in zip(gpus, batch)]))
        all_results[split] = {r['key']: r for r in results}
        if split == 'valid':
            write(assessment / 'validation-replay-gate.json', {'status': 'passed', 'at_unix': time.time(),
                  'freeze_identity': freeze['identity'], 'tolerance': 1e-8, 'models': [r['id'] for r in results]})
    baseline_key = freeze['din_reference_key']
    baseline = all_results['test'][baseline_key]['metrics'] if baseline_key else None
    ranked = []
    for row in freeze['suite']:
        if row['rank_by_validation'] is None:
            continue
        metrics = all_results['test'][row['key']]['metrics']
        ranked.append({'rank_by_validation': row['rank_by_validation'], 'id': row['id'], 'key': row['key'],
            'topic_id': row.get('topic_id'),
            'validation_macro_auc': validation({'metrics': {'validation': row['validation']}})['macro_auc'],
            'test': metrics, 'parameters': row['parameters'], 'training_seconds': row['train_seconds'],
            'delta_macro_auc_vs_din': metrics['macro_auc'] - baseline['macro_auc'] if baseline else None,
            'per_task_delta_vs_din': {t: {m: metrics['per_task'][t][m] - baseline['per_task'][t][m]
                                         for m in ['auc', 'logloss']} for t in metrics['per_task']} if baseline else {}})
    result = {'status': 'complete', 'completed_unix': time.time(), 'selection_freeze': freeze,
              'ranked_by_validation': ranked, 'test': all_results['test'],
              'assessment_wall_seconds': time.monotonic() - started,
              'assessment_job_wall_seconds': sum(r['wall_seconds'] for stage in all_results.values() for r in stage.values()),
              'no_retraining': True, 'no_test_based_reselection': True,
              'budget_scope': freeze['budget_scope'], 'limitations': [freeze['evaluation_type'], 'single training seed']}
    write(assessment / 'report.json', result)
    lines = ['Top3 checkpoint 测试结果（排名始终依据验证四任务平均AUC）', '']
    for row in ranked:
        m = row['test']
        lines.append(f"#{row['rank_by_validation']} {row['id']} | validation {row['validation_macro_auc']:.8f} | test mean AUC {m['macro_auc']:.8f} | test mean Logloss {m['macro_logloss']:.8f}")
        for task, value in m['per_task'].items():
            lines.append(f"  {task}: AUC {value['auc']:.8f}, Logloss {value['logloss']:.8f}")
    (assessment / 'report.txt').write_text('\n'.join(lines) + '\n')
    write(assessment / 'status.json', {'phase': 'complete', 'at_unix': time.time(), 'freeze_identity': freeze['identity']})
    status = read(root / 'status.json')
    status.update(assessment_complete=True, final_assessment_report=str(assessment / 'report.json'))
    write(root / 'status.json', status)


def watch(root, assessment, top_k, include_baseline):
    assessment.mkdir(parents=True, exist_ok=True)
    with (assessment / 'assessment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(assessment / 'watcher-process.json', {'pid': os.getpid(), 'started_unix': time.time(),
              'start_ticks': Path('/proc/' + str(os.getpid()) + '/stat').read_text().split()[21]})
        while True:
            status = read(root / 'status.json')
            if (root / 'STOP').exists() or status['phase'] == 'terminated_by_user':
                write(assessment / 'status.json', {'phase': 'cancelled_with_research', 'at_unix': time.time()})
                return
            if status['phase'] == 'setup_failed':
                raise ValueError('Research setup failed')
            if status['phase'] in CLOSED_CAMPAIGNS and not live_research_children(root):
                assess(root, assessment, top_k, include_baseline)
                return
            write(assessment / 'status.json', {'phase': 'waiting_for_research_completion', 'at_unix': time.time(),
                  'requested_top_k': top_k, 'research_deadline_unix': read(root / 'budget.json')['deadline_unix'],
                  'test_accessed': False})
            state_paths = [root / 'topics' / f'gpu-{g}' / 'state.json' for g in read(root / 'budget.json')['gpus']]
            if (root / 'references.json').exists() and all(p.exists() for p in state_paths):
                _, summaries, trace = topic_representatives([read(p) for p in state_paths], read(root / 'references.json')['trials'])
                write(assessment / 'topic-incumbents-preview.json', {'updated_unix': time.time(),
                      'preview_only': True, 'selection_not_frozen': True, 'topics': summaries, 'round_trace': trace})
            time.sleep(10)


def view(assessment):
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    console = Console()
    def render():
        status = read(assessment / 'status.json') if (assessment / 'status.json').exists() else {}
        table = Table(title='收尾前三 checkpoint 评估 / 验证平均AUC固定排名', expand=True)
        for name in ['验证排名', '课题', 'Checkpoint', '验证平均AUC', '测试平均AUC', '点击', '加购', '收藏', '购买']:
            table.add_column(name)
        if (assessment / 'selection-freeze.json').exists():
            for row in read(assessment / 'selection-freeze.json')['suite']:
                path = assessment / row['key'] / 'test/result.json'
                test = read(path)['metrics'] if path.exists() else None
                table.add_row(str(row['rank_by_validation'] or 'DIN'), ','.join(row.get('topic_ids', [])) or 'DIN', row['id'],
                    f"{validation({'metrics': {'validation': row['validation']}})['macro_auc']:.6f}",
                    f"{test['macro_auc']:.6f}" if test else '-',
                    *[f"{test['per_task'][task]['auc']:.6f}" if test else '-' for task in ['is_click', 'cart', 'fav', 'buy']])
        elif (assessment / 'topic-incumbents-preview.json').exists():
            preview = read(assessment / 'topic-incumbents-preview.json')
            for row in sorted(preview['topics'], key=lambda x: (-x['validation_macro_auc'], x['topic_id'])):
                table.add_row('预览', row['topic_id'], row['best_trial_id'], f"{row['validation_macro_auc']:.6f}",
                              '-', '-', '-', '-', '-')
        return Panel(table, title=status.get('phase', 'scheduled'), subtitle='搜索结束后自动触发；不重训；先验证重放再测试；评估成本单列')
    with Live(render(), console=console, screen=True, auto_refresh=False) as live:
        while True:
            time.sleep(5)
            live.update(render(), refresh=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--assessment-dir', type=Path)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--without-baseline', action='store_true')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--view', action='store_true')
    parser.add_argument('--worker')
    parser.add_argument('--split', choices=['valid', 'test'])
    args = parser.parse_args()
    root = args.root.resolve()
    assessment = args.assessment_dir.resolve() if args.assessment_dir else root / 'final-assessment-top3'
    try:
        if args.worker:
            worker(root, assessment, args.worker, args.split)
        elif args.view:
            view(assessment)
        elif args.watch:
            watch(root, assessment, args.top_k, not args.without_baseline)
        else:
            assessment.mkdir(parents=True, exist_ok=True)
            with (assessment / 'assessment.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                assess(root, assessment, args.top_k, not args.without_baseline)
    except BaseException as error:
        if not args.view:
            output = assessment / args.worker / str(args.split) if args.worker else assessment
            write(output / 'failure.json', {'at_unix': time.time(), 'error': repr(error)})
            if not args.worker:
                write(assessment / 'status.json', {'phase': 'cancelled' if (root / 'STOP').exists() else 'failed',
                      'at_unix': time.time(), 'error': repr(error)})
        raise
