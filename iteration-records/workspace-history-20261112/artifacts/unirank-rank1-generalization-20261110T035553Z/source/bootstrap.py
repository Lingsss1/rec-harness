"""Freeze rank-1 architecture and identical-recipe controls for four datasets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import difflib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / 'experiments/unirank_generalization'))
from common import read, write, sha, identity

DATASETS = ['QK-Video', 'KuaiRand', 'TAAC-25', 'MerRec']
RECIPE_KEYS = ['tasks', 'sample_size', 'epochs', 'batch_size', 'history_limit', 'seed', 'num_workers',
               'dense_optimizer', 'dense_learning_rate', 'dense_weight_decay', 'embedding_optimizer',
               'embedding_learning_rate', 'gradient_clip', 'gradient_clip_scope', 'precision',
               'drop_last', 'history_policy', 'history_semantics', 'selection', 'checkpoint_key']
MODEL_ID = 'g3-r0018-pair_occupancy_task_private_linear'


def bootstrap(previous, assessment, root):
    previous, assessment, root = previous.resolve(), assessment.resolve(), root.resolve()
    if root.exists():
        raise ValueError('Use a fresh output directory')
    frozen = read(assessment / 'selection-freeze.json')
    observed = read(assessment / 'report.json')
    row = next(r for r in frozen['suite'] if r['key'] == 'rank_1')
    if observed['status'] != 'complete' or row['id'] != MODEL_ID or row['rank_by_validation'] != 1:
        raise ValueError('Expected the completed validation-ranked top3 assessment')
    for field in ['source', 'checkpoint']:
        if sha(row[field + '_path']) != row[field + '_sha256']:
            raise ValueError('Frozen source/checkpoint changed')
    old_setup = read(previous / 'setup.json')
    taobao = assessment.parent
    root.mkdir(parents=True)
    setup = {'created_unix': time.time(), 'source_model': MODEL_ID, 'source_campaign': str(taobao),
             'source_assessment': str(assessment), 'previous_cross_dataset_campaign': str(previous),
             'previous_model': old_setup['source_model'],
             'upstream_path': old_setup['upstream_path'], 'upstream_commit': old_setup['upstream_commit'],
             'requested_datasets': DATASETS, 'gpus': [4, 5, 6, 7], 'training_loss_hook_expected': False,
             'mode': 'fixed architecture transfer; native schemas; scratch retraining', 'no_search': True,
             'source_validation_macro_auc': row['validation']['macro_auc'],
             'source_test_macro_auc': observed['test']['rank_1']['metrics']['macro_auc'],
             'source_test_click_auc': observed['test']['rank_1']['metrics']['per_task']['is_click']['auc'],
             'selection_warning': 'Source selected by validation macro AUC per topic; source and target tests were observed for prior models; exploratory transfer follow-up.'}
    write(root / 'setup.json', setup)
    write(root / 'source-selection.json', {'selected': row,
          'selection_basis': frozen['selection_scope'], 'source_test_used_for_selection': False,
          'source_assessment_report_sha256': sha(assessment / 'report.json'),
          'new_unseen_holdout_claim': False})
    source = root / 'source'
    source.mkdir()
    for name in ['model.py', 'adapt.py', 'test_contract.py', 'bootstrap.py', 'supervise.py', 'dashboard.py', 'replay.py']:
        shutil.copy2(HERE / name, source / name)
    for name in ['train.py', 'common.py', 'baseline.py']:
        shutil.copy2(previous / 'source' / name, source / name)
    shutil.copy2(row['source_path'], source / 'taobao-original.py')
    shutil.copy2(row['checkpoint_path'], source / 'taobao-checkpoint.pt')
    write(source / 'taobao-config.json', row['config'])
    (root / 'interface-adaptation.patch').write_text(''.join(difflib.unified_diff(
        (source / 'taobao-original.py').read_text().splitlines(True),
        (source / 'model.py').read_text().splitlines(True),
        fromfile='rank1-taobao-original.py', tofile='native-schema-model.py')))
    hashes = {p.name: sha(p) for p in source.glob('*.py')}
    for dataset in DATASETS:
        old = previous / 'datasets' / dataset
        folder = root / 'datasets' / dataset
        folder.mkdir(parents=True)
        for name in ['dataset-source-manifest.json', 'meta_data.json']:
            shutil.copy2(old / name, folder / name)
        shutil.copytree(old / 'prepared', folder / 'prepared')
        (folder / 'data').symlink_to((old / 'data').resolve(), target_is_directory=True)
        params = read(old / 'parameters.json')
        for split in ['train', 'valid', 'test']:
            params[split + '_data'] = str(folder / 'data' / split / 'data')
            for kind in ['user_info', 'item_info']:
                params[split + '_' + kind] = str(folder / 'data' / split / kind)
        params['data_root'] = str(folder / 'data')
        write(folder / 'parameters.json', params)
        write(folder / 'candidate-config.json', row['config'])
        old_protocol = read(old / 'protocol.json')
        protocol = copy.deepcopy(old_protocol)
        protocol.pop('identity')
        protocol.update(frozen_unix=time.time(), candidate=MODEL_ID, source_sha256=hashes,
            source_selection='final top topic representative ranked by validation four-task macro AUC',
            source_taobao_test_click_auc=setup['source_test_click_auc'],
            primary_role_rule='primary task is a reporting label only; all task scorer-input gradients remain open',
            match_slot_rule='global plus each native item-field equality; all unordered native item-field pairs; five fixed temporal bins',
            training_loss='equal sum of native true-label BCE; no custom training loss hook',
            gradient_transfer='original shared scorer trunk and task-private heads; all scorer-input gradients open; no custom gradient transport',
            reused_references=['din', 'previous_macro_candidate'],
            previous_macro_candidate_id=old_setup['source_model'],
            limitations=['single seed; fixed one-epoch architecture transfer with retraining, not zero-shot',
                         'source selected by validation macro AUC; its Taobao test has already been observed',
                         'target test sets have already been evaluated for prior architectures; exploratory follow-up',
                         'official_offline feedback semantics; delayed outcomes follow the pinned release',
                         'MerRec has Like as its primary task and no click label',
                         'native task count, item-field count and pair count differ across datasets'])
        if any(protocol[k] != old_protocol[k] for k in RECIPE_KEYS):
            raise ValueError('Shared training/scoring recipe changed')
        protocol['identity'] = identity(protocol)
        write(folder / 'protocol.json', protocol)
        imported = []
        for label, origin in [
            ('din', previous / 'reference-runs' / dataset / 'din'),
            ('previous_macro_candidate', previous / 'runs' / dataset / 'candidate/fit-01'),
        ]:
            result = read(origin / 'result.json')
            if (result['status'] != 'success' or result['dataset_revision'] != protocol['revision']
                    or result['test']['samples'] != protocol['sample_size']['test']
                    or set(result['test']['per_task']) != set(protocol['tasks'])):
                raise ValueError('Historical reference release/task/sample mismatch')
            dst = root / 'reference-runs' / dataset / label
            dst.mkdir(parents=True)
            for name in ['result.json', 'selection.json']:
                shutil.copy2(origin / name, dst / name)
            imported.append({'kind': label, 'source_path': str(origin),
                             'result_sha256': sha(dst / 'result.json'),
                             'original_protocol_hash': result['protocol_hash'], 'new_fit_performed': False})
        prior_reuse = read(old / 'reference-reuse.json')
        shared_recipe = {k: protocol[k] for k in RECIPE_KEYS}
        if any(prior_reuse['shared_recipe'][k] != protocol[k]
               for k in prior_reuse['shared_recipe']):
            raise ValueError('DIN reference recipe differs')
        if sha(folder / 'prepared/feature_map.json') != prior_reuse['feature_map_sha256']:
            raise ValueError('Feature map differs from reused references')
        write(folder / 'reference-reuse.json', {'shared_recipe': shared_recipe,
              'recipe_identity': identity(shared_recipe), 'references': imported,
              'feature_map_sha256': sha(folder / 'prepared/feature_map.json')})

    def verify_dataset(dataset):
        folder = root / 'datasets' / dataset
        manifest = read(folder / 'dataset-source-manifest.json')
        for record in manifest['files']:
            path = folder / 'data' / record['path']
            if path.stat().st_size != record['bytes'] or sha(path) != record['local_sha256']:
                raise ValueError('Cached dataset changed: ' + dataset + '/' + record['path'])
        write(folder / 'acquisition-status.json', {'phase': 'downloaded_and_verified', 'at_unix': time.time(),
              'bytes': manifest['total_bytes'], 'total_bytes': manifest['total_bytes'],
              'completed_files': len(manifest['files']), 'total_files': len(manifest['files']),
              'original_verified_unix': manifest['verified_unix'], 'rechecked_cached_files': True})
        return {'dataset': dataset, 'files': len(manifest['files']), 'bytes': manifest['total_bytes']}

    write(root / 'supervisor-status.json', {'phase': 'preflight', 'at_unix': time.time(), 'jobs': {}})
    with ThreadPoolExecutor(4) as pool:
        checks = list(pool.map(verify_dataset, DATASETS))
    write(root / 'cached-release-checks.json', {'status': 'passed', 'at_unix': time.time(), 'datasets': checks})
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '4', 'PYTHONDONTWRITEBYTECODE': '1',
           'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}
    checked = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(source),
                              '-p', 'test_contract.py', '-v'], env=env, capture_output=True, text=True)
    (root / 'interface-checks.log').write_text(checked.stdout + checked.stderr)
    if checked.returncode:
        raise ValueError('Schema/CPU/BF16 parity checks failed; inspect interface-checks.log')
    write(root / 'interface-checks.json', {'status': 'passed', 'at_unix': time.time(),
          'exact_cpu_and_bf16_cuda_parity': ['initialization and RNG', 'predictions', 'loss', 'gradients', 'buffers'],
          'task_counts': [2, 4, 5, 6], 'native_pairs_and_feedback_shapes_checked': True,
          'original_sha256': row['source_sha256'], 'adapted_sha256': sha(source / 'model.py')})
    return root


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--assessment', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(bootstrap(args.previous, args.assessment, args.root))
