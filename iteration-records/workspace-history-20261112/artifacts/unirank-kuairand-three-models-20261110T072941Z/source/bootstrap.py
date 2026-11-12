"""Freeze the two requested embedding16/token256 KuaiRand depth controls."""
import argparse
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

RECIPE_KEYS = ['tasks', 'sample_size', 'epochs', 'batch_size', 'history_limit', 'seed', 'num_workers',
              'dense_optimizer', 'dense_learning_rate', 'dense_weight_decay', 'embedding_optimizer',
              'embedding_learning_rate', 'gradient_clip', 'gradient_clip_scope', 'precision',
              'drop_last', 'history_policy', 'history_semantics', 'selection', 'checkpoint_key']


def bootstrap(reference, hemix, root):
    reference, hemix, root = [p.resolve() for p in [reference, hemix, root]]
    if root.exists():
        raise ValueError('Use a fresh output directory')
    prior = read(reference / 'setup.json')
    old = reference / 'datasets/KuaiRand'
    original_protocol = read(old / 'protocol.json')
    hemix_protocol = read(hemix / 'datasets/KuaiRand/protocol.json')
    for key in RECIPE_KEYS:
        assert original_protocol[key] == hemix_protocol[key], key
    reference_result = read(reference / 'runs/KuaiRand/candidate/fit-01/result.json')
    assert reference_result['status'] == 'success'
    assert reference_result['source_sha256'] == sha(reference / 'source/model.py')
    assert read(hemix / 'datasets/KuaiRand/candidate-config.json')['token_dim'] == 80
    root.mkdir(parents=True)
    write(root / 'setup.json', {'created_unix': time.time(), 'dataset': 'KuaiRand',
        'upstream_path': prior['upstream_path'], 'upstream_commit': prior['upstream_commit'],
        'reference_campaign': str(reference), 'hemix_campaign': str(hemix),
        'reference_model': prior['source_model'], 'new_fits': ['joint1', 'joint3'],
        'gpus': {'hemix_existing': 0, 'joint1': 1, 'joint3': 2},
        'embedding_dim': 16, 'token_dim': 256, 'joint_layers': [1, 3],
        'no_search': True, 'selection_basis': 'explicit user-specified width and depth',
        'existing_hemix_training_untouched': True, 'original_hemix_not_trained': True})
    source = root / 'source'
    source.mkdir()
    for name in ['model.py', 'bootstrap.py', 'preflight.py', 'supervise.py', 'dashboard.py']:
        shutil.copy2(HERE / name, source / name)
    shutil.copy2(reference / 'source/model.py', source / 'base_model.py')
    shutil.copy2(reference / 'source/common.py', source / 'common.py')
    trainer = (hemix / 'source/train.py').read_text()
    new_trainer = trainer.replace(
        "source = root / 'source' / ('model.py' if kind == 'candidate' else 'baseline.py')",
        "source = root / 'source/model.py'")
    new_trainer = new_trainer.replace("choices=['candidate', 'din']", "choices=['joint1', 'joint3']")
    anchor = "    acquisition = read(folder / 'acquisition-status.json')"
    assert new_trainer.count(anchor) == 1
    new_trainer = new_trainer.replace(anchor,
        "    if sha(folder / (args.kind + '-config.json')) != protocol['variant_config_sha256'][args.kind]:\n"
        "        raise ValueError('Frozen variant configuration changed')\n" + anchor)
    (source / 'train.py').write_text(new_trainer)
    (root / 'host-adaptation.patch').write_text(''.join(difflib.unified_diff(trainer.splitlines(True),
        new_trainer.splitlines(True), fromfile='same-recipe-hemix-host.py', tofile='width-depth-host.py')))
    shutil.copy2(hemix / 'upstream-source-identity.json', root / 'upstream-source-identity.json')
    for name, expected in read(root / 'upstream-source-identity.json')['files_sha256'].items():
        assert sha(Path(prior['upstream_path']) / name) == expected, name
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
    configurations = {kind: {**CONFIG, 'embedding_dim': 16, 'token_dim': 256, 'joint_layers': depth}
                      for kind, depth in [('joint1', 1), ('joint3', 3)]}
    audit = {'hemix80': read(hemix / 'parameter-audit.json')['hemix_token80']}
    fmap = FeatureMap(params['dataset_id'], str(folder / 'prepared'))
    fmap.load(str(folder / 'prepared/feature_map.json'), {'embedding_dim': 16})
    for kind, config in configurations.items():
        write(folder / (kind + '-config.json'), config)
        with torch.device('meta'), patch.object(Model, 'model_to_device', lambda self: None):
            model = Model(fmap, gpu=-1, model_root=str(root / 'audit'), **params, **config)
        ids = {id(p) for m in model.modules() if isinstance(m, torch.nn.Embedding)
               for p in m.parameters(recurse=False)}
        total = sum(p.numel() for p in model.parameters())
        embeddings = sum(p.numel() for p in model.parameters() if id(p) in ids)
        assert embeddings == audit['hemix80']['embedding_parameters']
        audit[kind] = {'parameters': total, 'embedding_parameters': embeddings, 'dense_parameters': total - embeddings}
    audit['ours32_historical'] = {k: reference_result[k] for k in ['parameters', 'embedding_parameters', 'dense_parameters']}
    assert audit['joint3']['parameters'] - audit['joint1']['parameters'] == 1054208
    write(root / 'parameter-audit.json', audit)
    protocol = copy.deepcopy(original_protocol)
    protocol.pop('identity')
    protocol.update(frozen_unix=time.time(), candidate='winner-embedding16-token256-depth1-or3',
        source_selection='explicit user request, no width/depth search based on metrics',
        runs=['joint1', 'joint3'], architecture='same frozen winner; common token256; one or three independent JointBlocks',
        source_sha256={p.name: sha(p) for p in source.glob('*.py')},
        variant_config_sha256={kind: sha(folder / (kind + '-config.json')) for kind in configurations},
        training_loss='equal sum of all six native true-label BCE, no custom loss hook',
        initialization_note='full scratch training; depth controls preserve every common initial parameter and host sampling RNG',
        limitations=['single seed; supplementary target tests, previously observed for earlier models',
                     'matching width and depth does not match dense parameter count or operation types',
                     'HeMix remains token80 as requested; no original HeMix training'])
    for key in RECIPE_KEYS:
        assert protocol[key] == original_protocol[key], key
    protocol['identity'] = identity(protocol)
    write(folder / 'protocol.json', protocol)
    (root / 'reference').mkdir()
    write(root / 'reference/ours32-result.json', reference_result)
    shutil.copy2(reference / 'runs/KuaiRand/candidate/fit-01/selection.json', root / 'reference/ours32-selection.json')
    write(root / 'reference/provenance.json', {'ours32_source_sha256': sha(reference / 'source/model.py'),
        'ours32_protocol_hash': original_protocol['identity'],
        'ours32_feature_map_sha256': sha(old / 'prepared/feature_map.json'),
        'hemix_feature_map_sha256': sha(hemix / 'datasets/KuaiRand/prepared/feature_map.json'),
        'shared_recipe': {key: protocol[key] for key in RECIPE_KEYS}})
    assert sha(old / 'prepared/feature_map.json') == sha(hemix / 'datasets/KuaiRand/prepared/feature_map.json')
    manifest = read(folder / 'dataset-source-manifest.json')
    for row in manifest['files']:
        path = folder / 'data' / row['path']
        assert path.stat().st_size == row['bytes'] and sha(path) == row['local_sha256'], row['path']
    write(folder / 'acquisition-status.json', {'phase': 'downloaded_and_verified', 'at_unix': time.time(),
        'bytes': manifest['total_bytes'], 'total_bytes': manifest['total_bytes'],
        'completed_files': len(manifest['files']), 'total_files': len(manifest['files']), 'rechecked_cached_files': True})
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '1', 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}
    checked = subprocess.run([sys.executable, '-B', str(source / 'preflight.py'), '--root', str(root)],
                             env=env, capture_output=True, text=True)
    (root / 'interface-checks.log').write_text(checked.stdout + checked.stderr)
    if checked.returncode:
        raise ValueError('Numerical width/depth checks failed; inspect interface-checks.log')
    write(root / 'preflight.json', {'status': 'passed', 'at_unix': time.time(),
        'data_rehashed': True, 'same_native_features_and_recipe': True,
        'embedding_dim': 16, 'token_dim': 256, 'independent_joint_depths': [1, 3],
        'all_common_initial_parameters_and_host_rng_equal_across_depths': True,
        'three_blocks_receive_gradients_cpu_and_bf16_cuda': True,
        'existing_training_processes_untouched': True})
    write(root / 'status.json', {'phase': 'ready', 'at_unix': time.time(), 'jobs': {}})
    return root


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--hemix', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(bootstrap(args.reference, args.hemix, args.root))
