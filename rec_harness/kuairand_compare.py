"""Read-only graph-D/no-graph-D progress and cost comparison; excludes final scores."""
import argparse
from pathlib import Path

from .core import ContractError, canonical, read_json
from .store import Store


def report(root):
    root = Path(root)
    result = {'qualification': 'Exploratory single-seed trajectories; different requested experiment counts and costs are retained. '
        'This is not independent replication or proof of graph utility.', 'final_results_included': False, 'arms': {}}
    protocols = []
    for name in ('graph', 'no-graph'):
        campaign = root/name/'campaign'; cfg_path = campaign/'kuairand-auto.json'
        if not cfg_path.exists():
            result['arms'][name] = {'status': 'not_registered', 'frontier': []}; continue
        cfg = read_json(cfg_path); protocols.append(cfg['protocol_hash']); state = read_json(campaign/'state.json')
        store = Store(campaign/'memory', read_only=True)
        try:
            integrity = store.verify(); nodes = {n['id']: n for n in store.snapshot()['nodes']}
            for r in state['records']:
                if r['id'] not in nodes or nodes[r['id']]['attrs']['result'] != r['result']:
                    raise ContractError('published comparison record differs from immutable graph fact')
        finally: store.close()
        token_usage = {'input_tokens': 0, 'output_tokens': 0, 'cached_input_tokens': 0}
        role_calls = 0; usage_receipts = 0
        # Include failed/repaired calls as well as accepted calls; never visit host-only-final.
        for top in ('stages', 'failed-calls'):
            for path in (campaign/top).rglob('request.json'): role_calls += 1
            for path in (campaign/top).rglob('usage.json'):
                usage_receipts += 1
                for usage in read_json(path).get('usage', []):
                    for key in token_usage:
                        value = (usage or {}).get(key)
                        if value is not None: token_usage[key] += value
        attempts = [read_json(p) for p in (campaign/'models').rglob('attempt-*.json')]
        rows = []
        for r in state['records']:
            v = r['result']
            rows.append({'id': r['id'], 'valid': r['valid'], 'stage': r['stage'], 'slot': r['slot'],
                'validation_metrics': v.get('metrics'), 'training_gpu_seconds': v.get('training_gpu_seconds'),
                'scoring_seconds': v.get('scoring_seconds'), 'failure_kind': None if r['valid'] else v['kind'],
                'on_frontier': r['id'] in state['frontier']})
        result['arms'][name] = {'status': state['status'], 'stage': state['stage'], 'records': rows,
            'frontier': state['frontier'], 'integrity': integrity, 'protocol_hash': cfg['protocol_hash'],
            'costs': {'training_gpu_seconds_successful_epochs': sum(r['result'].get('training_gpu_seconds', 0) for r in state['records']),
                'allocated_gpu_seconds_terminal_attempts': sum(a.get('allocated_gpu_seconds', 0) for a in attempts if a['status'] != 'running'),
                'pending_attempts': sum(a['status'] == 'running' for a in attempts),
                'role_calls': role_calls, 'usage_receipts': usage_receipts, 'reported_tokens': token_usage,
                'currency_cost': None, 'qualification': 'Allocated attempt time and successful-epoch time overlap; do not add them. Missing currency cost is unknown.'}}
    if len(set(protocols)) > 1: raise ContractError('comparison arms have different frozen protocols')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('root', nargs='?', default='artifacts/kuairand-research')
    a = p.parse_args(); print(canonical(report(a.root)))


if __name__ == '__main__': main()
