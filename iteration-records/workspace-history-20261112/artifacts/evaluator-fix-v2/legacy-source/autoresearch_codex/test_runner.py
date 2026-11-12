"""Verify greedy rollback, experiment count, and the paired freeze contract."""
from pathlib import Path

import pytest

from autoresearch_codex.run import (
    AutoresearchController, choose_single, freeze, keep_or_discard, run_search,
)
from mle_star_codex.run import read, sha

ROOT = Path(__file__).resolve().parents[1]


def trial(name, score, **extra):
    return {'id': name, 'status': 'success', 'metrics': {'HR@10': score}, **extra}


def test_greedy_revert_and_single_validation_selection():
    parent = trial('parent', .6)
    assert keep_or_discard(parent, trial('worse', .5)) == (parent, 'discard')
    assert keep_or_discard(parent, trial('equal', .6)) == (parent, 'discard')
    assert keep_or_discard(parent, {'id': 'bad', 'status': 'crash'}) == (parent, 'crash')
    better = trial('better', .61)
    assert keep_or_discard(parent, better) == (better, 'keep')
    assert choose_single([parent, better]) is better
    assert choose_single([parent, trial('tie', .6)]) is parent


def test_rounds_consume_slots_and_failed_candidates_do_not_become_parents(tmp_path):
    from mle_star_codex.run import baseline_source
    template = ROOT / 'artifacts/RecHarness-6lyc/gagc/templates/nextitnet_perdataset/train.py'
    c = {'run_dir': str(tmp_path), 'repo': str(template.parents[3]), 'gpus': [0, 1, 2, 3],
         'iterations_per_trajectory': 3}

    class Fake(AutoresearchController):
        def jobs(self, specs, final=False):
            assert len(specs) == 1 and specs[0]['key'] == 'baseline'
            return [trial('baseline', .5, source=str(specs[0]['model']), source_sha256=sha(specs[0]['model']))]

        def propose(self, key, instruction, items, debug_remaining):
            assert debug_remaining == 0
            i = int(key.rsplit('-', 1)[1])
            if i >= 2:
                assert items[0]['parent_id'] == 'iteration-01-t0'
                assert all(x['parent_id'] == 'baseline' for x in items[1:])
            results = []
            for t, item in enumerate(items):
                p = self.code / (item['id'] + '.py')
                p.write_text(item['source'])
                r = trial(item['id'], .6 if t == 0 else .4, source=str(p), source_sha256=sha(p), hypothesis='test')
                if t == 2:
                    r.update(status='crash', error='synthetic OOM')
                results.append(r)
            return results

    result = run_search(Fake(c))
    assert result['candidate_attempt_slots'] == 12
    assert result['completed_iterations_per_trajectory'] == 3
    assert result['selected']['id'] == 'iteration-01-t0'
    assert len((tmp_path / 'results.tsv').read_text().splitlines()) == 14  # header + baseline + 12
    assert len(list((tmp_path / 'rounds').glob('*.json'))) == 3


def paired_config(tmp_path):
    c = read(ROOT / 'artifacts/mle-star-movies-2024/config.json')
    for k in ('initial_candidates', 'outer_rounds', 'inner_rounds', 'ablations_per_round', 'ensemble_rounds'):
        c.pop(k)
    c.update(run_dir=str(tmp_path), iterations_per_trajectory=11,
        reference_run=str(ROOT / 'artifacts/mle-star-movies-2024'),
        upstream_program=str(ROOT / 'artifacts/karpathy-autoresearch/program.md'))
    return c


def test_freeze_accepts_exact_paired_protocol_and_rejects_metric_change(tmp_path):
    c = paired_config(tmp_path)
    freeze(AutoresearchController(c))
    frozen = read(tmp_path / 'frozen.json')
    assert frozen['candidate_training_call_allowance'] == 44
    assert frozen['shared_worker_sha256'] == read(Path(c['reference_run']) / 'frozen.json')['code_hashes']['worker.py']
    c['eval_users'] = 1000
    with pytest.raises(ValueError, match='settings differ'):
        freeze(AutoresearchController(c))


def test_freeze_rejects_unmatched_iteration_count(tmp_path):
    c = paired_config(tmp_path)
    c['iterations_per_trajectory'] = 10
    with pytest.raises(ValueError, match='not matched'):
        freeze(AutoresearchController(c))
