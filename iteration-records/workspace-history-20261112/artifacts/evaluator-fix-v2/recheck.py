"""Recheck historical sources/checkpoints on CPU. No fitting or result rewriting."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from mle_star_codex import worker as fixed

torch.set_num_threads(4)
base = ROOT / 'artifacts/evaluator-fix-v2'
old_path = base / 'legacy-source/mle_star_codex/worker.py'
spec = importlib.util.spec_from_file_location('historical_worker', old_path)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
results = []
staged = ROOT / 'artifacts/staged-movies-2024-aligned44/study/trajectories/movies_nextitnet'
for arm in 'ABCD':
    p = staged / arm / '1'
    state = json.loads((p / 'state.json').read_text())
    selected = {r['trial'] for r in state['records'] if not r['valid']} | {state['best_trial']}
    for trial in sorted(selected):
        source = p / 'artifacts' / ('t%03d' % trial) / 'model.py'
        checkpoint = p / 'validation' / ('t%03d' % trial) / 'adapter_scratch/training/checkpoint.pt'
        torch.manual_seed(2024)
        module, model = fixed.load(source, 17566, 'cpu')
        if checkpoint.exists():
            model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu'))
        legacy_error = None
        try:
            legacy.check(model, module.MAXLEN, 'cpu')
        except (AssertionError, RuntimeError) as exc:
            legacy_error = str(exc)
        current = fixed.check(model, module.MAXLEN, 'cpu')
        results.append({'arm': arm, 'trial': trial, 'trained_checkpoint': checkpoint.exists(),
            'source_sha256': fixed.sha(source), 'checkpoint_sha256': fixed.sha(checkpoint) if checkpoint.exists() else None,
            'legacy_error': legacy_error, 'fixed_checks': current})
        print('rechecked', arm, trial, 'checkpoint' if checkpoint.exists() else 'initialization', flush=True)

p = ROOT / 'artifacts/mle-star-movies-2024'
source = p / 'models/baseline.py'
torch.manual_seed(2024)
module, model = fixed.load(source, 17566, 'cpu')
model.load_state_dict(torch.load(p / 'trials/baseline/checkpoint.pt', weights_only=True, map_location='cpu'))
train = fixed.read(p / 'data/trainval/Movies_and_TV_train.txt')
valid = fixed.read(p / 'data/trainval/Movies_and_TV_valid.txt')
rows = fixed.rows_for(train, valid, 17566, 200)
before = legacy.predict(model, rows, module.MAXLEN, 'cpu')
audit = fixed.check(model, module.MAXLEN, 'cpu')
after = fixed.predict(model, rows, module.MAXLEN, 'cpu')
assert np.array_equal(before, after)
assert fixed.metrics(after)['HR@10'] == 0.395
summary = {'check_version': fixed.CHECK_VERSION, 'training_calls_added': 0, 'test_calls_added': 0,
    'legacy_worker_sha256': fixed.sha(old_path), 'fixed_worker_sha256': fixed.sha(fixed.__file__),
    'historical_experiment_files_modified': False, 'rechecked': results,
    'baseline': {'predictions_identical_before_after_check': True, 'validation_users': len(rows),
        'metrics': fixed.metrics(after), 'checks': audit},
    'limitations': ['Initialization-only checks do not imply successful training.',
        'Passing corrected checks does not retroactively change the original search path or final selection.']}
(base / 'regression.json').write_text(json.dumps(summary, indent=2) + '\n')
print('RECHECK_PASSED', len(results), 'baseline HR@10', summary['baseline']['metrics']['HR@10'])
