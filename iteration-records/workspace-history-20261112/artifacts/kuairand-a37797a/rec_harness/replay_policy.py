"""Prefix-only protocol between a replay simulator and isolated policy processes."""
from pathlib import Path

from .core import ContractError, canonical, digest, public_only, write_json
from .replay import Replay


def validate_action(action, view):
    fields = {'parents', 'action_kind', 'select', 'memory', 'rationale'}
    if not isinstance(action, dict) or set(action) != fields:
        raise ContractError('policy action fields must be parents/action_kind/select/memory/rationale')
    if not isinstance(action['parents'], list) or not all(isinstance(p, str) for p in action['parents']):
        raise ContractError('parents must be string array')
    if not isinstance(action['action_kind'], str) or not isinstance(action['rationale'], str):
        raise ContractError('action kind and rationale must be strings')
    if action['select'] is not None and action['select'] not in view['observations']:
        raise ContractError('selection outside visible prefix')
    public_only(action)
    if len(canonical(action).encode()) > 65536:
        raise ContractError('action and policy memory exceed 64 KiB')


def run_policy(world, policy, sandbox, destination, budget, *, seed=0, max_steps=100, direction='maximize'):
    if type(seed) is not int or type(max_steps) is not int or max_steps < 1:
        raise ContractError('integer seed and positive max_steps required')
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    replay = Replay(world, budget, direction)
    memory, previous, trace = None, None, []
    write_json(destination / 'manifest.json', {'world_hash': world['hash'], 'seed': seed,
               'budget': budget, 'max_steps': max_steps, 'direction': direction,
               'world_qualification': world['qualification'], 'online_utility_evidence': False})
    try:
        for step in range(max_steps):
            view = replay.view()
            visible = {'view': view, 'previous_outcome': previous, 'memory': memory,
                       'seed': seed, 'step': step, 'direction': direction, 'worker_limit': world['workers']}
            action = sandbox.decide(policy, visible, destination / f'decision-{step:04d}')
            validate_action(action, view)
            if action['select'] is not None:
                replay.select(action['select'])
            previous = replay.step(action['parents'], action_kind=action['action_kind'])
            memory = action['memory']
            trace.append({'visible_state': visible, 'visible_hash': digest(visible), 'action': action,
                          'outcome': previous, 'report': replay.report()})
            write_json(destination / 'trace.json', trace)
            if replay.stopped:
                break
        if not replay.stopped:
            replay.stopped = 'policy_step_limit'
    except Exception as error:
        write_json(destination / 'failure.json', {'error': str(error), 'completed_decisions': len(trace)})
        raise
    report = {'report': replay.report(), 'visible_state': replay.view(),
              'trace_hash': digest(trace), 'online_utility_evidence': False}
    write_json(destination / 'report.json', report)
    return report
