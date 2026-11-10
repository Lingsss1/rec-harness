"""Replay the original rank-1 Taobao checkpoint with the adapted model, validation only."""
import argparse
from pathlib import Path
import random
import sys
import time

from common import read, write, sha


def replay(root):
    import numpy as np
    import torch
    setup = read(root / 'setup.json')
    taobao = Path(setup['source_campaign'])
    sys.path.insert(0, str(taobao / 'source'))
    import train as host
    row = read(root / 'source-selection.json')['selected']
    output = root / 'taobao-validation-replay'
    output.mkdir(exist_ok=True)
    random.seed(row['seed'])
    np.random.seed(row['seed'])
    torch.manual_seed(row['seed'])
    torch.cuda.manual_seed_all(row['seed'])
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    model, fmap, params, effective = host.make_model(taobao, root / 'source/model.py', row['config'],
                                                    Path(row['original_trial_output']), row['seed'])
    if sha(root / 'source/taobao-checkpoint.pt') != row['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed')
    model.load_state_dict(torch.load(root / 'source/taobao-checkpoint.pt', map_location='cpu', weights_only=True), strict=True)
    metrics = host.score(model, host.loader(taobao, fmap, params, 'valid'), list(fmap.labels),
                         torch.device('cuda:0'), output)
    for task, values in metrics['per_task'].items():
        for key in ['auc', 'logloss']:
            if abs(values[key] - row['validation']['per_task'][task][key]) > 1e-8:
                raise ValueError('Adapted checkpoint replay differs: ' + task + '/' + key)
    if metrics['samples'] != read(taobao / 'data/Taobao_Action/meta_data.json')['sample_size']['valid']:
        raise ValueError('Validation sample coverage mismatch')
    write(output / 'result.json', {'status': 'passed', 'no_retraining': True, 'test_accessed': False,
          'checkpoint_sha256': row['checkpoint_sha256'], 'adapted_source_sha256': sha(root / 'source/model.py'),
          'metrics': metrics, 'wall_seconds': time.monotonic() - started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    replay(parser.parse_args().root.resolve())
