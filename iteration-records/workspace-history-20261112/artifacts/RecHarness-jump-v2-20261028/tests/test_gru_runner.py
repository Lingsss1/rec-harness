"""GRU indexing and experiment boundaries that matter for the new runs."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
import gru_runner as runner


def config(tmp_path):
    return {**runner.read(runner.DESTINATION / 'run-01/config.json'), 'root': str(tmp_path)}


def test_gru_indexing_matches_last_nonpadding_state(tmp_path):
    torch = pytest.importorskip('torch')
    c = runner.GRUJumpController(config(tmp_path))
    source = tmp_path / 'train.py'
    source.write_text(c.baseline())
    spec = importlib.util.spec_from_file_location('gru_indexing_test', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.GRU4RecModel(20, hidden_units=8, dropout_rate=0.).eval()
    seq = torch.tensor([[0, 0, 1, 2], [0, 3, 4, 5], [0, 0, 0, 0]])
    with torch.no_grad():
        recurrent, _ = model.gru(model.item_emb(seq))
        torch.testing.assert_close(model.encode(seq)[:2], recurrent[:2, -1])
        assert torch.isfinite(model.encode(seq)).all()
        # The old count-1 expression would choose a padded position here.
        assert not torch.allclose(recurrent[0, 1], model.encode(seq)[0])


def test_formal_and_smoke_gru_environment_are_distinct(tmp_path):
    formal = runner.GRUJumpController(config(tmp_path / 'formal'))
    smoke = runner.GRUJumpController(config(tmp_path / 'smoke'), smoke=True)
    assert formal.environment(4)['GRU4REC_EPOCHS'] == '200'
    assert smoke.environment(4)['GRU4REC_EPOCHS'] == '5'
    assert smoke.environment(4)['GRU4REC_BATCHES'] == '2'
    assert formal.c['single_model_search'] is True
    assert formal.c['llm_model'] == 'gpt-6.1-sol'
    assert formal.c['llm_effort'] == 'medium'


def test_formal_three_runs_have_independent_budget_and_cache():
    plan = runner.read(runner.DESTINATION / 'plan.json')
    assert len(plan['jobs']) == 3
    configs = [runner.read(j['config']) for j in plan['jobs']]
    assert {c['search_budget_gpu_seconds'] for c in configs} == {43200}
    assert len({c['root'] for c in configs}) == 3
    assert len({c['gru_baseline']['sha256'] for c in configs}) == 1
    assert all(c['replication']['fresh_baseline_weights'] for c in configs)
    assert all(not c['replication']['cross_run_response_or_memory_reuse'] for c in configs)
    assert all(c['alignment']['cold_start'] == 'gru4rec_perdataset' for c in configs)
    assert all(len(c['gpus']) == 4 and len(c['cpu_cores']) == 64 for c in configs)
    assert plan['total_gpu_seconds'] == 129600


def test_transport_prompt_and_retry_preserve_gru_request(tmp_path, monkeypatch):
    controller = runner.GRUJumpController(config(tmp_path))
    calls = []
    def request(self, instruction, output_schema, search, key):
        calls.append((instruction, output_schema, search, key, self.scope.get()))
        if len(calls) == 1:
            raise runner.transport.TransportUnavailable('fault injection')
        return {'edits': []}
    monkeypatch.setattr(runner.transport.ResilientSelfController, 'llm', request)
    monkeypatch.setattr(runner.time, 'sleep', lambda _: None)
    assert controller.llm('a mutation', {}, False, 'round-1') == {'edits': []}
    assert calls[0] == calls[1]
    assert calls[0][-1] == 'round-1'
    assert 'GRU4Rec' in calls[0][0]
    assert 'Start from RecHarness sasrec_perdataset.' not in runner.transport.TASK
    assert 'Start from RecHarness gru4rec_perdataset.' in runner.transport.TASK


def test_finish_blocks_changed_checkpoint_before_test(tmp_path):
    controller = runner.GRUJumpController(config(tmp_path))
    trial = tmp_path / 'selected'
    (trial / 'working').mkdir(parents=True)
    (trial / 'working/predict.py').write_text('pass\n')
    checkpoint = trial / 'working/a.pt'
    checkpoint.write_text('changed')
    runner.save(trial / 'single-model-audit.json', {'eligible': True,
        'predictor_sha256': runner.sha(trial / 'working/predict.py'),
        'checkpoint_sha256': {str(checkpoint): 'oldhash'}})
    controller.best = {'output': str(trial)}
    with pytest.raises(RuntimeError, match='checkpoint changed'):
        controller.finish()
