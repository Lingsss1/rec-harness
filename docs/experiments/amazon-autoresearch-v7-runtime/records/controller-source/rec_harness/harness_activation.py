"""Import an independently validated policy only when creating a new campaign."""
from .core import ContractError, digest, read_json
from .evolution import Evolution, validate_policy


def load_active_policy(root, identity):
    evolution = Evolution(root)
    try:
        integrity = evolution.store.verify()
        active = evolution._active('harness')
        if active is None:
            reverted=any(e['kind']=='candidate_rollback' and e['payload'].get('kind')=='harness' and e['payload']['to'] is None for e in evolution.store.events())
            if reverted:
                if read_json(evolution.root/'active-harness.json') is not None:raise ContractError('baseline pointer differs from committed events')
                return {},{'candidate_id':None,'source_event_head':integrity['head'],'policy_hash':digest({}),'status':'builtin_after_rollback'}
            raise ContractError('no committed active harness')
        if read_json(evolution.root / 'active-harness.json') != active:
            raise ContractError('active harness pointer differs from committed events')
        candidate = evolution.verified_candidate(active['candidate_id'])
        if candidate['kind'] != 'harness' or candidate['implementation'] != active['implementation'] or candidate['plan_hash'] != active['plan_hash']:
            raise ContractError('active harness candidate identity mismatch')
        plan = read_json(evolution.root / 'candidates' / candidate['candidate_id'] / 'plan.json')
        if identity != plan['identity']:
            raise ContractError('campaign harness identity differs from validated conditions')
        if set(candidate['implementation']) != {'policy.json'}:
            raise ContractError('harness implementation must contain policy.json only')
        policy = read_json(evolution.artifacts.verify(candidate['implementation']['policy.json']))
        validate_policy(policy)
        if evolution.store.verify() != integrity:
            raise ContractError('harness registry changed during activation')
        return policy, {'candidate_id': candidate['candidate_id'], 'plan_hash': candidate['plan_hash'],
                        'source_event_head': integrity['head'], 'policy_hash': digest(policy),
                        'validated_identity': identity}
    finally:
        evolution.store.close()
