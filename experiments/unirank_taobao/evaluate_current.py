"""Explicit user-requested mid-search checkpoint test, isolated from research trajectories."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from common import read, write, sha, identity
from selection import best_trial, aggregate, validate_policy


def run(root, output, gpu):
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise ValueError('Use a fresh evaluation directory')
    output.mkdir(parents=True)
    protocol = read(root / 'protocol.json')
    validate_policy(protocol['selection_policy'])
    rows = list(read(root / 'references.json')['trials'])
    for g in read(root / 'budget.json')['gpus']:
        rows.extend(read(root / 'topics' / f'gpu-{g}' / 'state.json')['trials'])
    chosen = best_trial({'trials': rows})
    if chosen is None:
        raise ValueError('No completed successful checkpoint')
    metrics = chosen['metrics']
    if metrics['data_identity'] != protocol['data_identity'] or metrics['protocol_hash'] != protocol['identity']:
        raise ValueError('Selected checkpoint data/protocol mismatch')
    for name, expected in protocol['source_sha256'].items():
        if sha(root / 'source' / name) != expected:
            raise ValueError('Frozen host changed: ' + name)
    original_source = Path(chosen['model_path'])
    original_checkpoint = Path(chosen['output']) / 'checkpoint.pt'
    if sha(original_source) != chosen['source_sha256'] or sha(original_checkpoint) != metrics['checkpoint_sha256']:
        raise ValueError('Selected source/checkpoint changed')
    package = output / 'model'
    package.mkdir()
    shutil.copy2(original_source, package / 'model.py')
    shutil.copy2(original_checkpoint, package / 'checkpoint.pt')
    write(package / 'model-config.json', chosen['config'])
    row = {'key': 'current_best', 'id': chosen['id'], 'role': chosen['role'], 'rank_by_validation': 1,
           'source_path': str(package / 'model.py'), 'source_sha256': chosen['source_sha256'],
           'checkpoint_path': str(package / 'checkpoint.pt'), 'checkpoint_sha256': metrics['checkpoint_sha256'],
           'config': chosen['config'], 'effective_config': metrics['effective_config'], 'seed': chosen['seed'],
           'validation': metrics['validation'], 'parameters': metrics['parameters'],
           'train_seconds': metrics['train_seconds'], 'original_trial_output': chosen['output'],
           'original_source_path': str(original_source), 'original_checkpoint_path': str(original_checkpoint)}
    freeze = {'frozen_unix': time.time(), 'suite': [row], 'protocol_hash': protocol['identity'],
              'data_identity': protocol['data_identity'], 'selection_policy': protocol['selection_policy'],
              'selection': 'highest four-task validation macro AUC among currently completed successful trials',
              'test_used_for_selection': False, 'no_training': True,
              'authority': 'user: immediately evaluate the currently best checkpoint',
              'research_continues': True, 'results_shared_with_research_agent': False,
              'successful_trials_seen': sum(r['status'] == 'success' for r in rows),
              'evaluation_type': 'user-requested supplementary evaluation during ongoing research; official test split was assessed previously',
              'budget_scope': 'separately recorded inference; the research clock continues unchanged; no extra training or research time'}
    freeze['identity'] = identity(freeze)
    write(output / 'selection-freeze.json', freeze)
    started = time.monotonic()
    print('FROZEN', row['id'], 'validation_macro_auc', aggregate(row['validation']['per_task'])['macro_auc'], flush=True)
    for record in read(root / 'dataset-source-manifest.json')['files']:
        if sha(root / 'data/Taobao_Action' / record['path']) != record['local_sha256']:
            raise ValueError('Released data file changed: ' + record['path'])
    for split in ['valid', 'test']:
        write(output / 'status.json', {'phase': 'validation_replay' if split == 'valid' else 'test',
              'at_unix': time.time(), 'id': row['id'], 'gpu': gpu})
        stage = output / row['key'] / split
        stage.mkdir(parents=True)
        command = [sys.executable, '-u', str(root / 'postprocess/assess_topk.py'), '--root', str(root),
                   '--assessment-dir', str(output), '--worker', row['key'], '--split', split]
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '4',
               'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}
        with (stage / 'process.log').open('w') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     env=env, start_new_session=True)
            write(stage / 'process.json', {'pid': child.pid, 'gpu': gpu, 'started_unix': time.time(), 'argv': command,
                  'start_ticks': Path('/proc/' + str(child.pid) + '/stat').read_text().split()[21]})
            try:
                code = child.wait(timeout=600)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                raise ValueError('Checkpoint inference exceeded600s')
        if code:
            raise ValueError('Inference failed; inspect ' + str(stage / 'process.log'))
        if split == 'valid':
            write(output / 'validation-replay-gate.json', {'status': 'passed', 'at_unix': time.time(),
                  'freeze_identity': freeze['identity'], 'tolerance': 1e-8, 'models': [row['id']]})
    valid = read(output / 'current_best/valid/result.json')
    test = read(output / 'current_best/test/result.json')
    result = {'status': 'complete', 'completed_unix': time.time(), 'selection_freeze': freeze,
              'validation_replay': valid, 'test': test, 'evaluation_wall_seconds': time.monotonic() - started,
              'evaluation_job_wall_seconds': valid['wall_seconds'] + test['wall_seconds'],
              'no_retraining': True, 'research_continues': True, 'results_shared_with_research_agent': False,
              'after_search_top3_assessment_still_enabled': True}
    # Reuse a previously measured DIN result only if the exact model and weight identities match.
    previous = Path(read(root / 'setup.json')['previous_campaign'])
    old_path = previous / 'final-assessment/din/test/result.json'
    if old_path.exists():
        current_din = next(r for r in rows if r.get('baseline_kind') == 'din')
        old = read(old_path)
        old_freeze = read(previous / 'final-assessment/selection-freeze.json')
        if (current_din['source_sha256'] == old['source_sha256'] and
                current_din['metrics']['checkpoint_sha256'] == old['checkpoint_sha256'] and
                protocol['data_identity'] == old_freeze['data_identity']):
            baseline = {**old['metrics'], **aggregate(old['metrics']['per_task'])}
            result['din_reference'] = {'metrics': baseline, 'reused_existing_measurement': True,
                'exact_source_checkpoint_data_identity_match': True, 'source': str(old_path), 'source_sha256': sha(old_path)}
            result['gain_vs_din'] = {'macro_auc': test['metrics']['macro_auc'] - baseline['macro_auc'],
                'macro_logloss_reduction': baseline['macro_logloss'] - test['metrics']['macro_logloss'],
                'per_task': {t: {m: test['metrics']['per_task'][t][m] - baseline['per_task'][t][m]
                                for m in ['auc', 'logloss']} for t in protocol['tasks']}}
    write(output / 'report.json', result)
    lines = [row['id'], f"Validation mean AUC: {aggregate(row['validation']['per_task'])['macro_auc']:.9f}",
             f"Test mean AUC: {test['metrics']['macro_auc']:.9f}",
             f"Test mean Logloss: {test['metrics']['macro_logloss']:.9f}"]
    lines += [f"{t}: AUC {v['auc']:.9f}, Logloss {v['logloss']:.9f}" for t, v in test['metrics']['per_task'].items()]
    (output / 'report.txt').write_text('\n'.join(lines) + '\n')
    write(output / 'status.json', {'phase': 'complete', 'at_unix': time.time(), 'id': row['id']})
    write(root / 'latest-immediate-assessment.json', {'directory': str(output), 'report': str(output / 'report.json'),
          'id': row['id'], 'source_sha256': row['source_sha256'], 'checkpoint_sha256': row['checkpoint_sha256'],
          'completed_unix': result['completed_unix'], 'results_shared_with_research_agent': False})
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    try:
        run(args.root, args.output, args.gpu)
    except BaseException as error:
        write(args.output / 'failure.json', {'at_unix': time.time(), 'error': repr(error)})
        write(args.output / 'status.json', {'phase': 'failed', 'at_unix': time.time()})
        raise
