from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gagc import tools, semantic_arms as policy, jump_policy
from gagc.state import ArmState, ThompsonState
import sasrec_runner as runner


class Store:
    def __init__(self, state): self.value = {'content': state.to_json()}
    def get(self, namespace, key): return SimpleNamespace(value=deepcopy(self.value))
    def put(self, namespace, key, value): self.value = deepcopy(value)
    def state(self): return ThompsonState.from_json(self.value['content'])


@pytest.fixture
def h(tmp_path, monkeypatch):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    for name in ['best.py', 'train.py', 'predict.py']:
        (workspace / name).write_text("marker = 'original'\n")
    state = ThompsonState(arms={a: ArmState(a) for a in policy.ALL}, global_budget=43200,
        initial_budget_secs=43200, score_history=[.595], baseline_done=True, round_idx=3,
        current_basin_score=.595, basin_score_history={'default': [.595]}, recent_trial_costs=[1000.]*8)
    store = Store(state)
    for name, value in {'_STORE': store, '_LOGS_ROOT': str(tmp_path/'logs'),
        '_WORKSPACE_ROOT': str(workspace), '_BENCHMARK_MODE': 'amazon_reviews',
        '_ROUTING_POLICY': 'thompson', '_LAST_PROMOTION_RESULT': {}, '_LAST_TRIAL_RESULTS': [],
        '_SERVER_CONFIG': tools.ServerConfig(num_gpus=4)}.items(): monkeypatch.setattr(tools, name, value)
    monkeypatch.setenv('GAGC_ARM_DESIGN', 'semantic_v3')
    monkeypatch.setenv('GAGC_JUMP_ENABLED', 'true')
    return SimpleNamespace(store=store, workspace=workspace)


def step(h, scores, number, arms=None, jump=False):
    arms = arms or [policy.JUMP if jump else 'training_strategy'] * len(scores)
    plan = jump_policy.budget_plan(h.store.state(), 10800, 4, 4)
    results = []
    for i, (arm, score) in enumerate(zip(arms, scores)):
        results.append({'spec': {'arm': arm, 'dimension': arm, 'delta': 1., 'is_jumping': jump,
            'jump_budget_plan': plan if jump else {}}, 'val_score': score, 'val_metrics': {},
            'wall_time_secs': 100., 'timed_out': False, 'oom': False, 'error_message': None,
            'trial_group_id': f'semantic-{number}', 'trial_id': i,
            'mutated_code_content': f"marker = 'candidate-{number}-{i}'\n",
            'mutated_predict_content': f"marker = 'predict-{number}-{i}'\n"})
    promoted = json.loads(tools.promote_winner(json.dumps(results), script_path=str(h.workspace/'best.py')))
    tools.update_thompson_state(json.dumps(results), trial_group_id=f'semantic-{number}')
    return promoted, h.store.state()


def test_eight_directions_in_four_parent_categories(h):
    assert len(policy.REGULAR) == 8
    assert len(set(v['parent'] for v in policy.REGULAR.values())) == 4
    assert len(tools._active_exploiting_arms()) == 8
    assert tools._active_jumping_dims() == {policy.JUMP}
    assert len(tools._active_arms()) == 9  # eight normal arms and one outcome counter


def test_one_ranking_eight_draws_top_four_unique(h, monkeypatch):
    draws = Mock(side_effect=[.1, .8, .2, .7, .3, .6, .4, .5])
    monkeypatch.setattr(policy.random, 'betavariate', draws)
    selected, trace = policy.sample(h.store.state(), 4, with_trace=True)
    assert selected == ['embedding_representation', 'network_architecture', 'interest_aggregation', 'training_strategy']
    assert len(set(selected)) == 4 and draws.call_count == 8
    assert all(len(t['sampled_values']) == 8 for t in trace)


def test_later_round_can_repeat_the_same_four(h, monkeypatch):
    monkeypatch.setattr(policy.random, 'betavariate', Mock(side_effect=list(range(8))*2))
    assert policy.sample(h.store.state(), 4) == policy.sample(h.store.state(), 4)


def test_jump_uses_direction_posterior_and_skips_same_parent(h, monkeypatch):
    monkeypatch.setattr(policy.random, 'betavariate', Mock(side_effect=[.9, .8, .7, .6, .5, .4, .3, .2]))
    selected, trace = policy.sample(h.store.state(), 4, jump=True, with_trace=True)
    assert selected == [policy.JUMP]
    assert trace[0]['selected_directions'] == ['feature_construction', 'interaction_mechanism']


def test_proposal_caps_and_scopes_are_logged(h):
    specs = json.loads(tools.propose_action_group(h.store.state().to_json()))
    assert len(specs) == 4 and len({s['arm'] for s in specs}) == 4
    assert all(s['selected_categories'] == [s['arm']] for s in specs)
    assert len({s['hypothesis_key'] for s in specs}) == 4
    assert sum(s['max_trial_gpu_seconds'] for s in specs) <= 43200 - .35*43200


def test_budget_gate_returns_one_two_direction_jump(h):
    state = h.store.state()
    state.global_budget = 16000
    state.score_history = [.50, .55, .595]
    state.basin_score_history = {'default': list(state.score_history)}
    h.store.value = {'content': state.to_json()}
    specs = json.loads(tools.propose_action_group(state.to_json()))
    assert len(specs) == 1 and specs[0]['arm'] == policy.JUMP
    dirs = specs[0]['selected_categories']
    assert len(dirs) == 2 and policy.REGULAR[dirs[0]]['parent'] != policy.REGULAR[dirs[1]]['parent']
    assert specs[0]['jump_trigger'] == 'reserved_exploration_phase'


def test_single_trial_retune_gets_credit_for_real_improvement(h):
    _, state = step(h, [.60], 0)
    assert state.arms['training_strategy'].alpha == 2
    assert set(state.arms) == set(policy.ALL)


def test_tie_gets_no_success_credit(h):
    _, state = step(h, [.595], 0)
    assert state.arms['training_strategy'].alpha == 1
    assert state.arms['training_strategy'].beta == 2


@pytest.mark.parametrize('success', [True, False])
def test_combination_delayed_credit_and_complete_window(h, success):
    step(h, [.50], 0, jump=True)
    source = h.store.state().basin_arms['default']
    assert source[policy.JUMP].alpha == source[policy.JUMP].beta == 1
    for i, score in enumerate([.53, .56, .60, .61] if success else [.51, .52, .53, .54], 1):
        _, state = step(h, [score], i)
    assert not state.pending_queue and not state.provisional_incumbent
    source = state.basin_arms['default']
    assert source[policy.JUMP].alpha == (2 if success else 1)
    assert source[policy.JUMP].beta == (1 if success else 2)
    # Retune observations belong to the new basin, never the source components.
    assert all(source[name].alpha == source[name].beta == 1 for name in policy.REGULAR)
    assert state.current_basin_score == (.61 if success else .595)
    assert state.global_budget == 43200 - 500


def controller(tmp_path, run=1):
    config = runner.read(runner.DESTINATION/f'run-{run:02d}/config.json')
    return runner.SASRecJumpController({**config, 'root': str(tmp_path)})


def test_fresh_budget_is_43200_and_existing_spending_is_not_reset(tmp_path):
    c = controller(tmp_path)
    assert c.c['prior_protocol_charge']['gpu_seconds'] == 0
    assert c.ledger.available() == 43200
    c.ledger.reserve('settled-test', 600)
    c.ledger.settle('settled-test', 5.)
    c2 = controller(tmp_path)
    assert c2.ledger.available() == 43195


@pytest.mark.parametrize('invalid', ['missing_direction', 'wrong_direction', 'comment_only'])
def test_structured_jump_edits_reject_incomplete_scope(tmp_path, monkeypatch, invalid):
    c = controller(tmp_path)
    selected = ['embedding_representation', 'training_strategy']
    entries = [{'category': selected[0], 'rationale': 'change dimension', 'edits': [{'find': 'x = 1', 'replace': 'x = 2'}]},
        {'category': selected[1], 'rationale': 'change training', 'edits': [{'find': 'y = 1', 'replace': 'y = 2'}]}]
    if invalid == 'missing_direction': entries.pop()
    if invalid == 'wrong_direction': entries[1]['category'] = 'sequence_modeling'
    if invalid == 'comment_only': entries[1]['edits'][0]['replace'] = 'y = 1  # only a comment'
    monkeypatch.setattr(c, 'llm', lambda *a, **k: {'category_edits': entries, 'change_summary': 'test'})
    with pytest.raises(ValueError):
        c.implement('x = 1\ny = 1\n', {'arm': policy.JUMP, 'selected_categories': selected}, '', 1, 'test')


def test_slot_hypotheses_never_overwrite_each_other(h):
    specs = [{'arm': 'training_strategy', 'dimension': 'training_strategy', 'hypothesis_key': f'slot-{i}'} for i in range(2)]
    prepared, error = tools._prepare_specs_for_execution(json.dumps(specs),
        json.dumps({'slot-0': 'first idea', 'slot-1': 'second idea'}), 'group', str(h.workspace/'best.py'))
    assert error is None
    assert [s['agent_hypothesis'] for s in prepared] == ['first idea', 'second idea']
