"""Frozen real-training audit archive; does not invent research transitions."""
import math
from pathlib import Path

from rec_harness.core import Artifacts, ContractError, digest, read_json, write_json
from rec_harness.store import Store
from .amazon_controls import load_config
from .amazon_identity import record_version
from .amazon_report import audit_job


def account(events, receipts):
    """Reconcile every reservation, retaining failures and unknown costs."""
    ledger = {}
    for event in events:
        payload = event['payload']
        if event['kind'] == 'budget_reserved':
            key = payload['task_id']
            if key in ledger:
                raise ContractError('duplicate budget reservation')
            ledger[key] = {'task_id': key, 'reserved': payload['amount'], 'actual': None,
                           'status': 'running', 'reservation_seq': event['seq']}
        elif event['kind'] == 'budget_settled':
            key = payload['task_id']
            if key not in ledger or ledger[key]['status'] != 'running':
                raise ContractError('settlement without active reservation')
            ledger[key].update(actual=payload['actual'], status=payload['status'])
    seen = set()
    for receipt in receipts:
        key = receipt.get('task_id', receipt['job_id'])
        if key in seen or key not in ledger:
            raise ContractError('duplicate or unreserved receipt')
        seen.add(key)
        entry = ledger[key]
        if (entry['status'] != receipt['status'] or entry['actual'] is None
                or not math.isclose(entry['actual'], receipt['allocated_gpu_seconds'], rel_tol=0, abs_tol=1e-9)):
            raise ContractError('receipt and settled cost disagree')
    for key, entry in ledger.items():
        entry['receipt_in_archive'] = key in seen
        entry['charged_or_reserved'] = entry['reserved'] if entry['actual'] is None else entry['actual']
    return list(ledger.values())


def archive_study(root, output):
    root = Path(root)
    config = load_config(root)
    store = Store(root / 'memory')
    artifacts = Artifacts(root / 'artifacts/objects')
    try:
        integrity = store.verify()
        events = store.events()
        receipts, records = [], []
        missing = []
        for version in config['model_versions']:
            for dataset in config['datasets']:
                for seed in config['development_seeds'] + config['confirmation_seeds']:
                    job = f'{version}-{dataset}-{seed}'
                    directory = root / 'jobs' / job
                    if not (directory / 'completed.json').exists():
                        missing.append(job)
                        continue
                    receipt = read_json(directory / 'completed.json')
                    task = receipt.get('task_id', receipt['job_id'])
                    if receipt['status'] == 'success':
                        audit_job(root, config, version, dataset, seed)
                    elif (record_version(receipt), receipt['dataset'], receipt['seed']) != (version, dataset, seed):
                        raise ContractError('failure receipt identity mismatch')
                    receipts.append(receipt)
                    records.append({'task_id': task, 'job_id': job, 'model_version': version,
                        'dataset': dataset, 'seed': seed, 'status': receipt['status'],
                        'partition': 'development' if seed in config['development_seeds'] else 'confirmation',
                        'metrics': receipt['metrics'] if receipt['status'] == 'success' else None,
                        'process': receipt['process'], 'source_hash': receipt['source_hash'],
                        'request_ref': artifacts.put(directory / 'request.json'),
                        'receipt_ref': artifacts.put(directory / 'completed.json'),
                        'primary_parent_id': None, 'research_context_dependencies': None})
        ledger = account(events, receipts)
        order = {row['task_id']: row['reservation_seq'] for row in ledger}
        records.sort(key=lambda row: order[row['task_id']])
        total = sum(row['charged_or_reserved'] for row in ledger)
        if not math.isclose(total, store.budget()['charged_or_reserved'], rel_tol=0, abs_tol=1e-8):
            raise ContractError('event ledger disagrees with budget projection')
        # A live writer could otherwise mix a new receipt with an old event prefix.
        if store.verify() != integrity:
            raise ContractError('study changed during archive; retry from stable state')
        body = {'schema': 'amazon-training-audit-v1', 'qualification': 'audit_only',
            'reasons': ['fixed_training_schedule_has_no_research_policy_requests',
                        'model_design_parent_and_visible_context_not_recorded',
                        'asynchronous_gpu_completion_is_not_synchronous_research_transition'],
            'protocol_hash': config['protocol_hash'], 'source_event_head': integrity['head'],
            'source_event_count': integrity['events'], 'records': records, 'missing_jobs': missing,
            'resource_ledger': ledger, 'charged_or_reserved_gpu_seconds': total,
            'scope': 'validation only; scores across datasets must not be ranked as competing models',
            'online_utility_evidence': False}
        result = {**body, 'hash': digest(body)}
        write_json(output, result)
        return result
    finally:
        store.close()
