"""Checkpoint reevaluation must select on validation before any test request."""
import json

from mle_star_codex import reevaluate as reeval


def result(key, score, status='success'):
    return {'id': key, 'status': status, 'metrics': {'HR@10': score}}


def test_recovery_selection_ignores_failures_auxiliaries_and_prefers_original_on_ties():
    entries = [
        {'id': 'recovered', 'family': 'C', 'kind': 'recovered'},
        {'id': 'original', 'family': 'C', 'kind': 'original'},
        {'id': 'auxiliary', 'family': 'C', 'kind': 'ensemble_auxiliary'},
        {'id': 'failed', 'family': 'C', 'kind': 'recovered'},
    ]
    values = [result('original', .665), result('recovered', .665),
              result('auxiliary', .9), result('failed', 1, 'failed')]
    assert reeval.select_validation(entries, values) == {'C': 'original'}
    values[1]['metrics']['HR@10'] = .67
    assert reeval.select_validation(entries, values) == {'C': 'recovered'}


def test_selection_is_written_before_test_and_no_training_interface_is_called(tmp_path, monkeypatch):
    entries = [{'id': 'original', 'family': 'C', 'kind': 'original'},
               {'id': 'recovered', 'family': 'C', 'kind': 'recovered'}]
    ensemble = {'transform': 'raw', 'members': []}
    monkeypatch.setattr(reeval, 'collect', lambda: (entries, [], ensemble))
    phases = []
    def evaluate(root, selected, phase, devices):
        phases.append(phase)
        if phase == 'validation':
            assert not (root / 'selection.json').exists()
            return [result('original', .6), result('recovered', .67)]
        assert json.loads((root / 'selection.json').read_text())['selected'] == {'C': 'recovered'}
        assert [entry['id'] for entry in selected] == ['recovered']
        # Stop before reporting: only the ordering and test request are under test.
        raise StopIteration('selection-before-test verified')
    monkeypatch.setattr(reeval, 'evaluate_phase', evaluate)
    import pytest
    with pytest.raises(StopIteration, match='selection-before-test verified'):
        reeval.run(tmp_path / 'new-run', [0, 1, 2, 3])
    assert phases == ['validation', 'test']
