"""Exercise real upstream selection/replay with a fake model and GPU transport."""
import json
import re
from pathlib import Path

import pytest

from discoveryrsi_codex.run import (
    CodexController, MoviesHarness, WORKSPACE, choose_best, config, source_parts,
)

BASELINE = (WORKSPACE / 'artifacts/mle-star-movies-2024/models/baseline.py').read_text()


class FakeController(CodexController):
    def __init__(self, root, *, episodes=11, invalid=False):
        c = config(root)
        c['episodes'] = episodes
        super().__init__(c)
        self.prompts = []
        self.waves = []
        self.invalid = invalid
        self.c['baseline_test'] = 'SECRET_TEST_RESULT'

    def query(self, key, prompt, schema):
        self.prompts.append((key, prompt))
        if key.startswith('propose-'):
            return {'candidates': [dict(claim=f'Claim {i}', plan='One-model comparator and falsification.',
                                       alternatives=['Noise'], evidence=[]) for i in range(1, 5)]}
        if key.startswith('compile-'):
            n = int(key.split('-')[-1])
            return {'models': [{'id': f'episode-{n}-{i}',
                'source': 'not valid python!!!' if self.invalid and i == 4 else BASELINE,
                'implementation_note': 'Mock compilation'} for i in range(1, 5)]}
        if key.startswith('critic-'):
            # Native retention prefers candidate 2, while validation prefers 1.
            i = int(key.rsplit('-', 1)[-1])
            return {'score': {1: 6, 2: 9, 3: 5, 4: 0}[i], 'critique': 'Mock scientific critique'}
        raise AssertionError(key)

    def jobs(self, specs, final=False):
        self.waves.append([s['key'] for s in specs])
        receipts = []
        for spec in specs:
            _, episode, i = spec['key'].split('-')
            value = 0.62 + int(episode) * .001 if i == '1' else 0.59
            receipts.append({'id': spec['key'], 'status': 'success',
                'source': str(spec['model']), 'metrics': {'HR@10': value},
                'allocated_gpu_seconds': 1})
        return receipts

    def event(self, *args, **kwargs):
        pass


def harness_at(root, **kwargs):
    controller = FakeController(root, **kwargs)
    (controller.code / 'baseline.py').write_text(BASELINE)
    baseline = {'id': 'baseline', 'status': 'success', 'metrics': {'HR@10': .395},
                'source': str(controller.code / 'baseline.py')}
    return MoviesHarness(controller, baseline)


def test_native_retention_budget_lineage_and_final_selection(tmp_path):
    harness = harness_at(tmp_path)
    episodes = harness.run()
    assert len(episodes) == 11
    assert len(harness.receipts) == 45  # baseline + 44 slots
    assert len(harness.controller.waves) == 11
    assert all(len(wave) == 4 for wave in harness.controller.waves)
    assert [e.selected_id for e in episodes] == [f'episode-{n}-2' for n in range(1, 12)]
    assert choose_best(list(harness.receipts.values()))['id'] == 'episode-11-1'
    for n, episode in enumerate(episodes, 1):
        assert all(c.parents == ([] if n == 1 else [f'episode-{n-1}-2']) for c in episode.candidates)
        assert all(any('HOST VALIDATION RECEIPT:' in e for e in c.evidence) for c in episode.candidates)
    replay = [json.loads(s) for s in (tmp_path / 'episodes.jsonl').read_text().splitlines()]
    assert len(replay) == 11
    assert replay[-1]['selected_id'] == 'episode-11-2'
    assert len(harness.controller.prompts) == 66


def test_invalid_source_consumes_slot_and_is_not_silently_retrained(tmp_path):
    harness = harness_at(tmp_path, episodes=2, invalid=True)
    episodes = harness.run()
    assert len(episodes) == 2
    assert len(harness.receipts) == 9
    assert sum(len(wave) for wave in harness.controller.waves) == 6
    assert harness.receipts['episode-1-4']['status'] == 'invalid_source'
    assert harness.receipts['episode-2-4']['status'] == 'invalid_source'
    assert len(list((tmp_path / 'invalid').glob('*.json'))) == 2
    progress = json.loads((tmp_path / 'progress.json').read_text())
    assert progress['completed_candidates'] == 8
    assert progress['successful_candidates'] == 6


def test_agent_context_carries_actual_results_but_no_test_values(tmp_path):
    harness = harness_at(tmp_path, episodes=2)
    harness.run()
    for key, prompt in harness.controller.prompts:
        assert 'SECRET_TEST_RESULT' not in prompt
        assert 'baseline_test' not in prompt
        assert 'selected_test' not in prompt
    next_prompt = next(p for k, p in harness.controller.prompts if k == 'propose-02')
    assert 'episode-1-2' in next_prompt
    assert 'Mock scientific critique' in next_prompt
    assert 'host_result' in next_prompt
    assert '0.59' in next_prompt


def test_final_selection_keeps_first_valid_candidate_on_tie():
    receipts = [{'id': 'bad', 'status': 'invalid_source'},
                {'id': 'a', 'status': 'success', 'metrics': {'HR@10': .7}},
                {'id': 'b', 'status': 'success', 'metrics': {'HR@10': .7}}]
    assert choose_best(receipts)['id'] == 'a'
    with pytest.raises(RuntimeError):
        choose_best(receipts[:1])


def test_reference_baseline_obeys_source_boundary():
    assert len(source_parts(BASELINE)) == 10
