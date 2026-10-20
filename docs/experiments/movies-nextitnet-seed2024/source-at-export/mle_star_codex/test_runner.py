"""Tests for scope confinement, budget recovery and upstream ranking parity."""
import importlib.util
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest

from mle_star_codex.run import Controller, baseline_source, blend, source_parts, validate_source

REPO = Path(__file__).resolve().parents[1] / 'artifacts/RecHarness-6lyc'


@pytest.fixture
def source():
    return baseline_source(REPO / 'gagc/templates/nextitnet_perdataset/train.py')


def test_selected_block_is_enforced(source):
    source_parts(source)
    changed = source.replace('return residual + y', 'return residual + 0.5 * y')
    assert validate_source(changed, source, 'NextItNetBlock') == ['NextItNetBlock']
    with pytest.raises(ValueError, match='escaped'):
        validate_source(changed, source, 'NextItNetModel')
    with pytest.raises(ValueError, match='escaped'):
        validate_source(changed.replace('LR = 0.001', 'LR = 0.002'), source, 'NextItNetBlock')


def test_reject_io_and_top_level_side_effects(source):
    with pytest.raises(ValueError):
        source_parts(source + '\nopen("test.txt").read()\n')
    with pytest.raises(ValueError):
        source_parts(source.replace('return residual + y', 'import os\n        return residual + y'))
    with pytest.raises(ValueError):
        source_parts(source.replace('return residual + y', 'torch.load("test.txt")\n        return residual + y'))


def test_ensemble_weights_and_ties():
    scores = np.ones((3, 100))
    for transform in ('raw', 'standardize', 'softmax', 'rank'):
        result = blend([scores, scores * 2], {'weights': [1., 1.], 'transform': transform})
        assert np.all(result == result[:, :1])
    with pytest.raises(ValueError):
        blend([scores], {'weights': [-1], 'transform': 'raw'})
    with pytest.raises(ValueError):
        blend([scores], {'weights': [float('nan')], 'transform': 'raw'})


def test_interrupted_budget_is_never_free(tmp_path):
    controller = Controller({'run_dir': str(tmp_path)})
    completed = tmp_path / 'trials/completed'
    completed.mkdir(parents=True)
    (completed / 'receipt.json').write_text(json.dumps({'allocated_gpu_seconds': 7.25}))
    interrupted = tmp_path / 'trials/interrupted'
    interrupted.mkdir()
    (interrupted / 'request.json').write_text(json.dumps({'timeout': 600}))
    assert controller.spent() == 607.25


def test_candidate_rows_and_ranking_match_upstream(tmp_path, monkeypatch):
    from mle_star_codex.worker import rows_for, metrics
    monkeypatch.syspath_prepend(str(REPO))
    from gagc.benchmarks.amazon_reviews.harness import _evaluate_one_dataset
    train = {u: [u, u + 1, u + 2] for u in range(1, 21)}
    valid = {u: [u + 30] for u in train}
    for name, values in [('train', train), ('valid', valid)]:
        (tmp_path / f'Movies_and_TV_{name}.txt').write_text(''.join(f'{u} {i}\n' for u, seq in values.items() for i in seq))
    (tmp_path / 'dataset_stats.json').write_text(json.dumps({'Movies_and_TV': {'itemnum': 250}}))
    captured = []
    def prediction(user, history, candidates):
        captured.append({'user': user, 'history': history, 'candidates': candidates})
        return [float(i % 17) for i in candidates]
    upstream = _evaluate_one_dataset(predict_fn=prediction, dataset_name='Movies_and_TV',
        data_dir=str(tmp_path), test_dir=None, mode='val', max_eval_users=10, seed=42, num_candidates=99)
    rows = rows_for(train, valid, 250, 10)
    assert rows == captured
    values = np.asarray([[float(i % 17) for i in row['candidates']] for row in rows])
    assert {k: round(v, 4) for k, v in metrics(values).items()} == upstream.metrics
