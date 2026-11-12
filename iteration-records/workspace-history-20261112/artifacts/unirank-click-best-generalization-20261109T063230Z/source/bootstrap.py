"""Freeze a four-dataset transfer run for the maximum observed Taobao test-click model."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT = Path(__file__).absolute().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / 'experiments/unirank_generalization'))
from common import read, write, sha, identity

DATASETS = ['QK-Video', 'KuaiRand', 'TAAC-25', 'MerRec']
RECIPE_KEYS = ['tasks', 'sample_size', 'epochs', 'batch_size', 'history_limit', 'seed', 'num_workers',
               'dense_optimizer', 'dense_learning_rate', 'dense_weight_decay', 'embedding_optimizer',
               'embedding_learning_rate', 'gradient_clip', 'gradient_clip_scope', 'precision', 'drop_last',
               'history_policy', 'history_semantics']


def bootstrap(previous, root):
    previous, root = previous.resolve(), root.resolve()
    if root.exists():
        raise ValueError('Use a new campaign directory')
    prior_setup = read(previous / 'setup.json')
    taobao = Path(prior_setup['source_campaign'])
    assessed = read(taobao / 'final-assessment/report.json')
    suite = assessed['selection_freeze']['suite']
    tested = [{'id': row['id'], 'key': row['key'],
               'test_click_auc': assessed['test'][row['key']]['metrics']['per_task']['is_click']['auc']}
              for row in suite]
    supplementary = read(taobao / 'additional-assessments/multitask-audit-best-v1/report.json')
    tested.append({'id': supplementary['selection_freeze']['id'], 'key': 'macro_audit',
                   'test_click_auc': supplementary['test']['per_task']['is_click']['auc']})
    winner = max(tested, key=lambda row: row['test_click_auc'])
    if winner['id'] != 'g1-r0088-membership_full_first_mean_only':
        raise ValueError('The inspected source is no longer the best observed tested click model')
    original = next(row for row in suite if row['id'] == winner['id'])
    if sha(Path(original['source_path'])) != original['source_sha256']:
        raise ValueError('Original source identity mismatch')
    root.mkdir(parents=True)
    setup = {'created_unix': time.time(), 'source_model': winner['id'], 'source_campaign': str(taobao),
             'previous_cross_dataset_campaign': str(previous), 'upstream_path': prior_setup['upstream_path'],
             'upstream_commit': prior_setup['upstream_commit'], 'requested_datasets': DATASETS,
             'mode': 'fixed architecture transfer with native task heads; scratch retraining; reused validated controls',
             'no_search': True, 'gpus': [4, 5, 6, 7], 'source_selection': winner,
             'selection_warning': 'Source selected by observed Taobao test click AUC at explicit user request; post-test exploratory selection. Target test splits were evaluated for a different architecture previously.'}
    write(root / 'setup.json', setup)
    write(root / 'source-selection.json', {'candidates': tested, 'selected': original,
          'selection_basis': 'maximum observed Taobao test click AUC, explicitly requested by user',
          'source_test_used_for_selection': True, 'new_unseen_holdout_claim': False})
    source = root / 'source'
    source.mkdir()
    for name in ['model.py', 'adapt.py', 'test_contract.py', 'bootstrap.py', 'supervise.py', 'dashboard.py']:
        shutil.copy2(HERE / name, source / name)
    for name in ['train.py', 'common.py']:
        shutil.copy2(PROJECT / 'experiments/unirank_generalization' / name, source / name)
    shutil.copy2(previous / 'source/baseline.py', source / 'baseline.py')
    shutil.copy2(original['source_path'], source / 'taobao-original.py')
    patch = difflib.unified_diff((source / 'taobao-original.py').read_text().splitlines(True),
                                (source / 'model.py').read_text().splitlines(True),
                                fromfile='tested-taobao-original.py', tofile='native-task-interface.py')
    (root / 'interface-adaptation.patch').write_text(''.join(patch))
    source_hashes = {p.name: sha(p) for p in source.glob('*.py')}
    for dataset in DATASETS:
        old = previous / 'datasets' / dataset
        folder = root / 'datasets' / dataset
        folder.mkdir(parents=True)
        for name in ['dataset-source-manifest.json', 'meta_data.json']:
            shutil.copy2(old / name, folder / name)
        shutil.copytree(old / 'prepared', folder / 'prepared')
        (folder / 'data').symlink_to(old / 'data', target_is_directory=True)
        write(folder / 'acquisition-status.json', {'phase': 'rechecking_cached_release', 'at_unix': time.time()})
        params = read(old / 'parameters.json')
        for split in ['train', 'valid', 'test']:
            params[split + '_data'] = str(folder / 'data' / split / 'data')
            for kind in ['user_info', 'item_info']:
                params[split + '_' + kind] = str(folder / 'data' / split / kind)
        params['data_root'] = str(folder / 'data')
        write(folder / 'parameters.json', params)
        old_protocol = read(old / 'protocol.json')
        config = {**original['config'], 'primary_task': old_protocol['architecture_primary_task'],
                  'match_fields': ['item_id'] + old_protocol['architecture_match_fields']}
        write(folder / 'candidate-config.json', config)
        protocol = copy.deepcopy(old_protocol)
        protocol.pop('identity')
        protocol.update(frozen_unix=time.time(), candidate=winner['id'], source_sha256=source_hashes,
                        source_selection='observed source-domain test click AUC; explicitly user requested',
                        architecture_match_fields=config['match_fields'],
                        match_slot_rule='three fixed slots: item identity plus two metadata matches; absent metadata contributes exact zeros',
                        match_semantics='item identity plus ' + old_protocol['match_semantics'],
                        training_loss='original model.training_loss hook: joint equal sum of native true-label BCE',
                        gradient_transfer='one primary context edge plus one edge per native auxiliary; ordered FP64 left-associated sum; original occupied-cycle-source-bit transfer retained',
                        runs=['candidate'], reused_references=['din', 'previous_macro_candidate'],
                        target_reporting='click/native primary task AUC and all-native-task macro AUC, per-task Logloss and cost',
                        source_taobao_test_click_auc=winner['test_click_auc'])
        protocol['limitations'] += ['source architecture selected using already observed Taobao test click results',
                                     'target test splits were previously evaluated for a different architecture; this is an exploratory transfer follow-up',
                                     'MerRec has no click label: its native primary task is Like, not CTR']
        assert all(protocol[key] == old_protocol[key] for key in RECIPE_KEYS)
        protocol['identity'] = identity(protocol)
        write(folder / 'protocol.json', protocol)
        imported = []
        for kind, label in [('din', 'din'), ('candidate', 'previous_macro_candidate')]:
            for name in ['result.json', 'selection.json']:
                src = previous / 'runs' / dataset / kind / 'fit-01' / name
                dst = root / 'reference-runs' / dataset / label / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            result = read(previous / 'runs' / dataset / kind / 'fit-01/result.json')
            if result['status'] != 'success' or result['dataset_revision'] != protocol['revision']:
                raise ValueError('Reference data/result mismatch')
            imported.append({'kind': label, 'source_campaign': str(previous),
                             'result_sha256': sha(root / 'reference-runs' / dataset / label / 'result.json'),
                             'original_protocol_hash': result['protocol_hash'], 'new_fit_performed': False})
        write(folder / 'reference-reuse.json', {'shared_recipe': {k: protocol[k] for k in RECIPE_KEYS},
              'recipe_identity': identity({k: protocol[k] for k in RECIPE_KEYS}),
              'feature_map_sha256': sha(folder / 'prepared/feature_map.json'), 'references': imported})

    def verify_dataset(dataset):
        folder = root / 'datasets' / dataset
        manifest = read(folder / 'dataset-source-manifest.json')
        for record in manifest['files']:
            path = folder / 'data' / record['path']
            if path.stat().st_size != record['bytes'] or sha(path) != record['local_sha256']:
                raise ValueError('Cached release changed: ' + dataset + '/' + record['path'])
        write(folder / 'acquisition-status.json', {'phase': 'downloaded_and_verified', 'at_unix': time.time(),
              'bytes': manifest['total_bytes'], 'total_bytes': manifest['total_bytes'],
              'completed_files': len(manifest['files']), 'total_files': len(manifest['files']),
              'original_verified_unix': manifest['verified_unix'], 'rechecked_cached_files': True})
        return {'dataset': dataset, 'files': len(manifest['files']), 'bytes': manifest['total_bytes']}

    write(root / 'supervisor-status.json', {'phase': 'preflight', 'at_unix': time.time(), 'jobs': {}})
    with ThreadPoolExecutor(4) as executor:
        checks = list(executor.map(verify_dataset, DATASETS))
    write(root / 'cached-release-checks.json', {'status': 'passed', 'at_unix': time.time(), 'datasets': checks})
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '4', 'PYTHONDONTWRITEBYTECODE': '1'}
    checked = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(HERE),
                              '-p', 'test_contract.py', '-v'], env=env, capture_output=True, text=True)
    (root / 'interface-checks.log').write_text(checked.stdout + checked.stderr)
    if checked.returncode:
        raise ValueError('Interface parity/schema checks failed')
    write(root / 'interface-checks.json', {'status': 'passed', 'tests': 4, 'at_unix': time.time(),
          'cpu_and_bf16_cuda_exact_parity': ['initialization', 'forward predictions', 'training loss', 'all parameter gradients', 'running buffers', 'eval predictions'],
          'task_counts': [2, 4, 5, 6], 'original_sha256': original['source_sha256'],
          'source_sha256': sha(source / 'model.py'), 'original_joint_loss_preserved': True})
    return root


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(bootstrap(args.previous, args.root))
