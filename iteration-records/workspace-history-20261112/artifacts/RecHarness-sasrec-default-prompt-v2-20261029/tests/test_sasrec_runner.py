"""Default-cold-start SASRec configuration and experiment boundaries that matter for the new runs."""
import importlib.util
import ast
import json
from pathlib import Path
import sys

import pytest

FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
import sasrec_runner as runner


def config(tmp_path):
    return {**runner.read(runner.DESTINATION / 'run-01/config.json'), 'root': str(tmp_path)}


def test_original_default_source_and_parameters_are_preserved(tmp_path, monkeypatch):
    c = runner.SASRecJumpController(config(tmp_path))
    original = Path(c.c['base_model']['original_source'])
    assert c.baseline() == original.read_text()
    assert '_hpo_os' not in c.baseline()
    assert 'candidate_id' not in c.c['base_model']
    source = tmp_path / 'train.py'
    source.write_text(c.baseline())
    for key in list(runner.os.environ):
        if key.startswith('SASREC_'):
            monkeypatch.delenv(key)
    monkeypatch.setenv('SASREC_EPOCHS', '200')
    monkeypatch.setenv('SASREC_BATCHES', '200')
    spec = importlib.util.spec_from_file_location('sasrec_binding_test', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.HIDDEN_UNITS == 64 and module.NUM_BLOCKS == 2 and module.NUM_HEADS == 1
    assert module.MAXLEN == 128 and module.NUM_EPOCHS == 200 and module.NUM_BATCHES == 200
    assert module.DROPOUT_RATE == pytest.approx(0.5)
    assert module.LR == pytest.approx(0.001)


def test_formal_and_smoke_sasrec_environment_are_distinct(tmp_path):
    formal = runner.SASRecJumpController(config(tmp_path / 'formal'))
    smoke = runner.SASRecJumpController(config(tmp_path / 'smoke'), smoke=True)
    assert formal.environment(4)['SASREC_EPOCHS'] == '200'
    assert smoke.environment(4)['SASREC_EPOCHS'] == '5'
    assert smoke.environment(4)['SASREC_BATCHES'] == '2'
    assert formal.c['single_model_search'] is True
    assert formal.c['llm_model'] == 'gpt-6.1-sol'
    assert formal.c['llm_effort'] == 'medium'


def test_two_fresh_searches_each_have_full_43200_budget():
    plan = runner.read(runner.DESTINATION / 'plan.json')
    assert len(plan['jobs']) == 2
    configs = [runner.read(j['config']) for j in plan['jobs']]
    assert {c['search_budget_gpu_seconds'] for c in configs} == {43200}
    assert len({c['root'] for c in configs}) == 2
    assert len({c['base_model']['sha256'] for c in configs}) == 1
    assert all(c['replication']['fresh_baseline_weights'] for c in configs)
    assert all(not c['replication']['cross_run_response_or_memory_reuse'] for c in configs)
    assert all(c['alignment']['cold_start'] == 'sasrec_perdataset' for c in configs)
    assert all(len(c['gpus']) == 4 and len(c['cpu_cores']) == 64 for c in configs)
    assert plan['total_gpu_seconds'] == 86400
    assert all(c['prior_protocol_charge']['gpu_seconds'] == 0 for c in configs)
    assert [c['replication']['repetition'] for c in configs] == [1, 2]
    assert configs[0]['gpus'] == [0, 1, 2, 3] and configs[1]['gpus'] == [4, 5, 6, 7]
    assert not set(configs[0]['cpu_cores']) & set(configs[1]['cpu_cores'])


def test_transport_prompt_and_retry_preserve_sasrec_request(tmp_path, monkeypatch):
    controller = runner.SASRecJumpController(config(tmp_path))
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
    assert 'SASRec' in calls[0][0]
    assert 'GRU' not in runner.transport.TASK
    assert 'original RecHarness SASRec default' in runner.transport.TASK


def test_finish_blocks_changed_checkpoint_before_test(tmp_path):
    controller = runner.SASRecJumpController(config(tmp_path))
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


def test_search_controls_match_live_gru_jump():
    original = runner.read(runner.WORKSPACE / 'artifacts/recharness-gru-jump-129600-20261028/run-01/config.json')
    current = runner.read(runner.DESTINATION / 'run-01/config.json')
    for key in ['data_dir', 'worker_python', 'controller_python', 'trial_floor_reference_seconds',
                'trial_timeout_seconds', 'minimum_trial_seconds', 'final_timeout_seconds',
                'llm_model', 'llm_effort', 'llm_timeout', 'llm_attempts', 'retry_backoff_seconds',
                'global_llm_concurrency', 'single_model_search', 'llm_slot_directory']:
        assert current[key] == original[key], key
    for key in ['datasets', 'preprocessing_seed', 'training_seed', 'evaluation_seed',
                'validation_users_per_dataset', 'test_users_per_dataset', 'negative_candidates',
                'training_environment_defaults', 'accounting']:
        assert current['alignment'][key] == original['alignment'][key], key
    for name in ['tools.py', 'grpo.py', 'state.py', 'jump_policy.py']:
        # Prompt literals intentionally changed; routing and accounting code did not.
        class NormalizeStrings(ast.NodeTransformer):
            def visit_Constant(self, node):
                if isinstance(node.value, str): node.value = '<string>'
                return node
        current_tree = NormalizeStrings().visit(ast.parse((runner.FORK/'gagc'/name).read_text()))
        reference_tree = NormalizeStrings().visit(ast.parse((runner.WORKSPACE/'artifacts/RecHarness-jump-v3-20261028/gagc'/name).read_text()))
        assert ast.dump(current_tree) == ast.dump(reference_tree), name
    assert current['search_budget_gpu_seconds'] == 43200
    assert current['gpus'] == [0, 1, 2, 3] and current['cpu_cores'] == list(range(64))
