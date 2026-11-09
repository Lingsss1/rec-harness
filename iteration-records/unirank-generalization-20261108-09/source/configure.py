"""Freeze dataset-native schemas and a paired, no-search transfer protocol."""
import argparse
import copy
from pathlib import Path
import shutil
import time
import yaml

from common import identity, read, sha, write

PROJECT = Path(__file__).absolute().parents[2]
DATASETS = {
    'QK-Video': ('QK_Video_Action', 'click', ['video_category', None]),
    'KuaiRand': ('KuaiRand_Video_Action', 'is_click', ['video_type', 'primary_tag']),
    'TAAC-25': ('TencentGR_10M_Action', 'is_click', ['100', '101']),
    'MerRec': ('MerRec_Action', 'Like', ['c2_id', 'brand_id']),
}


def configure(root):
    root = root.resolve()
    setup = read(root / 'setup.json')
    upstream = Path(read(Path(setup['source_campaign']) / 'setup.json')['upstream_path'])
    setup['upstream_path'] = str(upstream)
    setup['upstream_commit'] = '90bb99a60724e975ec565836266fd86dcad0039d'
    write(root / 'setup.json', setup)
    source = root / 'source'
    source.mkdir(exist_ok=True)
    for name in ['model.py', 'train.py', 'common.py', 'configure.py', 'test_contract.py', 'supervise.py', 'dashboard.py']:
        shutil.copy2(Path(__file__).with_name(name), source / name)
    shutil.copy2(PROJECT / 'experiments/unirank_taobao/baseline.py', source / 'baseline.py')
    shutil.copy2(Path(setup['source_campaign']) / 'best-model-multitask-audit/model.py', source / 'taobao-original.py')
    columns = yaml.safe_load((upstream / 'config/dataset_config.yaml').read_text())
    for name, (dataset_id, primary, matches) in DATASETS.items():
        folder = root / 'datasets' / name
        meta = read(folder / 'meta_data.json')
        template = copy.deepcopy(columns[dataset_id])
        features = []
        for col in template['feature_cols']:
            spec = {k: v for k, v in col.items() if k not in ['name', 'active', 'dtype']}
            spec.setdefault('source', '')
            if spec['type'] != 'meta':
                spec.update(vocab_size=meta['vocab_size'][col['name']], padding_idx=0)
            if spec['type'] == 'sequence':
                # This is FeatureProcessor.fit_sequence_col's published default.
                spec['feature_encoder'] = 'layers.MaskedAveragePooling()'
            features.append({col['name']: spec})
        fmap = {'dataset_id': dataset_id, 'labels': list(meta['label']), 'features': features,
                'total_features': sum(meta['vocab_size'].values()),
                'num_fields': sum(next(iter(x.values()))['type'] != 'meta' for x in features),
                'input_length': sum(next(iter(x.values())).get('max_len', 1) for x in features) + len(meta['label'])}
        write(folder / 'prepared/feature_map.json', fmap)
        paths = {}
        for split in ['train', 'valid', 'test']:
            paths[split + '_data'] = str(folder / 'data' / split / 'data')
            for kind in ['user_info', 'item_info']:
                paths[split + '_' + kind] = str(folder / 'data' / split / kind)
        params = {**paths, 'dataset_id': dataset_id, 'data_root': str(folder / 'data'),
                  'verbose': 0, 'metrics': ['AUC', 'logloss'], 'task': 'binary_classification',
                  'num_tasks': len(meta['label']), 'enable_torch_compile': False, 'enable_bf16': True,
                  'distributed': False, 'block_cache_size': 2, 'group_id': None}
        write(folder / 'parameters.json', params)
        config = {'primary_task': primary, 'match_fields': matches}
        write(folder / 'candidate-config.json', config)
        write(folder / 'din-config.json', {})
        manifest = read(folder / 'dataset-source-manifest.json')
        protocol = {
            'frozen_unix': time.time(), 'dataset': name, 'dataset_id': dataset_id,
            'revision': manifest['revision'], 'tasks': list(meta['label']),
            'sample_size': meta['sample_size'], 'epochs': 1, 'batch_size': 8192,
            'history_limit': 100, 'seed': 20262027, 'num_workers': 0,
            'selection': 'all-native-task-macro-auc-v1', 'checkpoint_key': ['-macro_auc', 'mean_logloss'],
            'training_loss': 'equal sum of native task mean BCE',
            'dense_optimizer': 'AdamW', 'dense_learning_rate': 1e-4, 'dense_weight_decay': 0,
            'embedding_optimizer': 'Adagrad', 'embedding_learning_rate': 0.05,
            'gradient_clip': 10.0, 'gradient_clip_scope': 'dense parameters',
            'precision': 'BF16 autocast; TF32 off; no torch.compile',
            'drop_last': {'train': True, 'valid': False, 'test': False},
            'history_policy': 'official_offline', 'history_semantics': 'exact released prefix/action feedback',
            'initialization': 'from scratch, dataset-native vocabularies',
            'candidate': 'g3-r0029-context_shared_primary; fixed interface adaptation only',
            'architecture_primary_task': primary, 'architecture_match_fields': matches,
            'primary_role_rule': 'first native task in pinned publisher metadata; inherited gradient asymmetry retained',
            'match_slot_rule': 'fixed two raw categorical metadata equality slots; absent slot and intersection are zeros',
            'match_semantics': ('anonymous categorical IDs, no category/brand claim' if name == 'TAAC-25' else
                                'category only; no brand field' if name == 'QK-Video' else
                                'video type and primary tag; no brand claim' if name == 'KuaiRand' else
                                'category level2 and brand'),
            'no_search': True, 'runs': ['candidate', 'din'],
            'test_policy': 'freeze validation-selected checkpoint and hash before one final test per run',
            'calibration': '20 training steps; no validation/test; reset to original seed for full run; cost retained',
            'run_wall_cap_seconds': 48 * 3600,
            'limitations': ['single seed initially; architectural transfer with retraining, not zero-shot transfer',
                           'per-dataset feature vocabularies and number of native tasks differ',
                           'offline released feedback may include delayed outcomes not mature at exposure time'],
            'benchmark_deviations': ['single GPU per model, batch8192; no distributed global batch',
                                     'validation and test include all remainders; in-process loading due 512MiB shared memory',
                                     'DIN is a field/task adaptation of local activation pooling, not exact paper reproduction'],
            'source_sha256': {p.name: sha(p) for p in source.glob('*.py')},
            'data_files': [{k: row[k] for k in ['path', 'bytes', 'lfs_sha256']} for row in manifest['files']],
        }
        protocol['identity'] = identity(protocol)
        write(folder / 'protocol.json', protocol)
    return root


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    configure(parser.parse_args().root)
