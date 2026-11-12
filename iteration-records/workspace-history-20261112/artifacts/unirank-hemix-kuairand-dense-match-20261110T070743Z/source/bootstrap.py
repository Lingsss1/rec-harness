"""Stage only the requested width-80 KuaiRand fit; no original HeMix training."""
import argparse
import copy
import difflib
import importlib.util
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

RECIPE_KEYS = ['tasks', 'sample_size', 'epochs', 'batch_size', 'history_limit', 'seed', 'num_workers',
              'dense_optimizer', 'dense_learning_rate', 'dense_weight_decay', 'embedding_optimizer',
              'embedding_learning_rate', 'gradient_clip', 'gradient_clip_scope', 'precision',
              'drop_last', 'history_policy', 'history_semantics', 'selection', 'checkpoint_key']


def bootstrap(reference, root):
    reference, root = reference.resolve(), root.resolve()
    if root.exists():
        raise ValueError('Use a new output directory')
    prior = read(reference / 'setup.json')
    if prior['source_model'] != 'g3-r0018-pair_occupancy_task_private_linear':
        raise ValueError('Expected final rank-1 reference')
    old = reference / 'datasets/KuaiRand'
    old_protocol = read(old / 'protocol.json')
    root.mkdir(parents=True)
    write(root / 'setup.json', {'created_unix': time.time(), 'dataset': 'KuaiRand', 'gpu': 0,
          'reference_campaign': str(reference), 'reference_model': prior['source_model'],
          'candidate': 'HeMix-token80', 'upstream_path': prior['upstream_path'],
          'upstream_commit': prior['upstream_commit'], 'no_search': True,
          'original_hemix_training_authorized': False,
          'change': 'token_dim 256 -> 80; all other released HeMix architecture settings retained',
          'selection_basis_for_width': 'parameter counts only, before any target candidate AUC',
          'comparison_scope': 'single-seed fixed recipe; final rank-1 reference reused from its same-dataset fit'})
    source = root / 'source'
    source.mkdir()
    for name in ['model.py', 'bootstrap.py', 'supervise.py', 'dashboard.py', 'test_contract.py']:
        shutil.copy2(HERE / name, source / name)
    shutil.copy2(PROJECT / 'UniRank/model_zoo/HeMix.py', source / 'HeMix.py')
    shutil.copy2(reference / 'source/common.py', source / 'common.py')
    trainer = (reference / 'source/train.py').read_text()
    # Metadata only: expose actual dense/embedding counts in calibration too.
    before = "'parameters': sum(p.numel() for p in model.parameters()),\n                      'peak_gpu_bytes':"
    after = "'parameters': sum(p.numel() for p in model.parameters()),\n                      'embedding_parameters': sum(p.numel() for p in sparse),\n                      'dense_parameters': sum(p.numel() for p in dense),\n                      'peak_gpu_bytes':"
    if trainer.count(before) != 1:
        raise ValueError('Unexpected trainer source')
    (source / 'train.py').write_text(trainer.replace(before, after))
    (root / 'host-metadata-only.patch').write_text(''.join(difflib.unified_diff(
        trainer.splitlines(True), (source / 'train.py').read_text().splitlines(True),
        fromfile='reference-host.py', tofile='candidate-host.py')))
    dependency_files = {p.relative_to(Path(prior['upstream_path'])).as_posix(): sha(p)
                        for p in Path(prior['upstream_path']).glob('unirank/**/*.py')}
    for name, expected in dependency_files.items():
        if sha(PROJECT / 'UniRank' / name) != expected:
            raise ValueError('Pinned framework differs from clean source: ' + name)
    if sha(source / 'HeMix.py') != sha(Path(prior['upstream_path']) / 'model_zoo/HeMix.py'):
        raise ValueError('Pinned HeMix differs from reference framework')
    write(root / 'upstream-source-identity.json', {'upstream_commit': prior['upstream_commit'],
          'files_sha256': dependency_files, 'hemix_sha256': sha(source / 'HeMix.py')})
    folder = root / 'datasets/KuaiRand'
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
    sys.path.insert(0, str(source))
    sys.path.insert(0, prior['upstream_path'])
    import torch
    from unittest.mock import patch
    from unirank.features import FeatureMap
    from model import CONFIG, Model
    write(folder / 'candidate-config.json', CONFIG)
    fmap = FeatureMap(params['dataset_id'], str(folder / 'prepared'))
    fmap.load(str(folder / 'prepared/feature_map.json'), {'embedding_dim': 16})
    with torch.device('meta'), patch.object(Model, 'model_to_device', lambda self: None):
        model = Model(fmap, gpu=-1, model_root=str(root / 'audit'), **params, **CONFIG)
    ids = {id(p) for m in model.modules() if isinstance(m, torch.nn.Embedding)
           for p in m.parameters(recurse=False)}
    total = sum(p.numel() for p in model.parameters())
    embeddings = sum(p.numel() for p in model.parameters() if id(p) in ids)
    reference_calibration = read(reference / 'runs/KuaiRand/candidate/calibration-01/result.json')
    ours = {'parameters': reference_calibration['parameters'], 'embedding_parameters': embeddings,
            'dense_parameters': reference_calibration['parameters'] - embeddings}
    small = {'parameters': total, 'embedding_parameters': embeddings, 'dense_parameters': total - embeddings}
    if abs(small['dense_parameters'] / ours['dense_parameters'] - 1) > .03:
        raise ValueError('Dense budget is not within 3% of the reference')
    write(root / 'parameter-audit.json', {'ours': ours, 'hemix_token80': small,
          'dense_relative_difference': small['dense_parameters'] / ours['dense_parameters'] - 1,
          'total_relative_difference': total / ours['parameters'] - 1,
          'native_embedding_specs_identical': True, 'chosen_without_auc_observations': True,
          'original_hemix_training_requested': False})
    protocol = copy.deepcopy(old_protocol)
    protocol.pop('identity')
    for key in ['architecture_match_fields', 'primary_role_rule', 'match_slot_rule', 'match_semantics', 'gradient_transfer']:
        protocol.pop(key, None)
    protocol.update(frozen_unix=time.time(), candidate='HeMix-token80', source_selection='parameter-count matching only',
        architecture='released HeMix, token width 80, three layers, four NS tokens and eight retrieved tokens',
        training_loss='equal sum of all six native true-label BCE; no training_loss hook',
        runs=['candidate'], original_hemix_not_trained=True,
        reused_reference=str(reference / 'runs/KuaiRand/candidate/fit-01'),
        source_sha256={p.name: sha(p) for p in source.glob('*.py')},
        limitations=['single seed and fixed architecture; no AUC-based width search',
                     'same official_offline released feedback semantics as the reference',
                     'target test previously observed for other architectures; exploratory comparison',
                     'reference selected on Taobao validation macro AUC, then retrained on KuaiRand'])
    for key in RECIPE_KEYS:
        assert protocol[key] == old_protocol[key], key
    protocol['identity'] = identity(protocol)
    write(folder / 'protocol.json', protocol)
    (root / 'reference').mkdir()
    for name in ['protocol.json', 'candidate-config.json', 'prepared/feature_map.json']:
        destination = root / 'reference' / Path(name).name
        shutil.copy2(old / name, destination)
    shutil.copy2(reference / 'source/model.py', root / 'reference/model.py')
    write(root / 'reference/provenance.json', {'campaign': str(reference), 'source_sha256': sha(reference / 'source/model.py'),
          'protocol_hash': old_protocol['identity'], 'feature_map_sha256': sha(old / 'prepared/feature_map.json'),
          'recipe': {key: protocol[key] for key in RECIPE_KEYS}, 'new_reference_fit_performed': False})
    manifest = read(folder / 'dataset-source-manifest.json')
    for row in manifest['files']:
        file = folder / 'data' / row['path']
        if file.stat().st_size != row['bytes'] or sha(file) != row['local_sha256']:
            raise ValueError('Pinned cached data changed: ' + row['path'])
    write(folder / 'acquisition-status.json', {'phase': 'downloaded_and_verified', 'at_unix': time.time(),
          'bytes': manifest['total_bytes'], 'total_bytes': manifest['total_bytes'],
          'rechecked_cached_files': True, 'completed_files': len(manifest['files']), 'total_files': len(manifest['files'])})
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}
    tests = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(source),
                            '-p', 'test_contract.py', '-v'], env=environment, capture_output=True, text=True)
    (root / 'interface-checks.log').write_text(tests.stdout + tests.stderr)
    if tests.returncode:
        raise ValueError('Adapter parity checks failed; see interface-checks.log')
    write(root / 'preflight.json', {'status': 'passed', 'at_unix': time.time(),
          'native_data_rehashed': True, 'same_training_recipe': True, 'same_native_feature_map': True,
          'cpu_and_cuda_bf16_width80_exact_initialization_predictions_loss_gradients': True,
          'actual_meta_parameter_count': small, 'gpu': 0, 'original_hemix_not_trained': True})
    write(root / 'status.json', {'phase': 'ready', 'at_unix': time.time()})
    return root


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(bootstrap(args.reference, args.root))
