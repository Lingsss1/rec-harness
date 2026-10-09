"""Freeze research handoffs before outcomes; registration does not admit code."""
import fcntl
from pathlib import Path

from rec_harness.agents import schema_for, validate_schema
from rec_harness.core import Artifacts, ContractError, digest, file_hash, identifier, read_json, validate_card
from rec_harness.store import Store
from .amazon_controls import load_config


def compile_plan(config, cards, versions, remaining):
    if set(cards) != {'gap', 'method', 'proposal'}:
        raise ContractError('gap, method and proposal handoffs required')
    gap, method, proposal = (cards[k] for k in ('gap', 'method', 'proposal'))
    validate_schema(gap, schema_for('evidence'))
    validate_schema({'method': method, 'proposal': proposal}, schema_for('hypothesis'))
    for kind, card in cards.items(): validate_card(kind, card)
    baseline = proposal['base_model_id']
    if baseline not in config['model_versions'] or gap['model_version'] != baseline:
        raise ContractError('research handoff base model differs from registered study')
    protocol = config['protocol_hash']
    seeds = config['development_seeds'] + config['confirmation_seeds']
    if gap['protocol_hash'] != protocol or proposal['train_protocol_hash'] != protocol or proposal['seeds'] != seeds:
        raise ContractError('research handoff changed frozen protocol or seeds')
    if proposal['gap_ids'] != [gap['gap_id']] or proposal['method_ids'] != [method['method_id']]:
        raise ContractError('research handoff contains unresolved gap or method references')
    for value in (gap['gap_id'], method['method_id'], proposal['proposal_id']): identifier(value)
    if not 0 < proposal['budget'] <= remaining:
        raise ContractError('research proposal exceeds available budget or has no budget')
    if not proposal['falsification_condition'].strip() or not proposal['insertion_contract'].strip():
        raise ContractError('explicit falsification and insertion contracts required')
    if not gap['alternative_explanations'] or not gap['predicted_observations']:
        raise ContractError('gap requires competing explanations and predictions')
    if set(versions) != {'candidate', 'controls'} or not versions['controls']:
        raise ContractError('candidate and explicit control source versions required')
    candidate = versions['candidate']
    controls = versions['controls']
    all_versions = [candidate, *controls]
    names = [v['model_version'] for v in all_versions]
    for name in names: identifier(name)
    if len(set(names)) != len(names) or candidate['model_version'] in config['model_versions']:
        raise ContractError('candidate must be new and planned model versions distinct')
    reused_controls = []
    for control in controls:
        name = control['model_version']
        if name in config['model_versions']:
            if control['source']['sha256'] != config['model_versions'][name]['sha256']:
                raise ContractError('reused control differs from registered source')
            reused_controls.append(name)
    if proposal['controls'] != [v['model_version'] for v in controls]:
        raise ContractError('proposal controls must match explicit executable control versions')
    if any(set(v) != {'model_version', 'source'} for v in all_versions):
        raise ContractError('unexpected model version fields')
    if len({v['source']['sha256'] for v in all_versions}) != len(all_versions):
        raise ContractError('candidate and controls must have distinct source identities')
    if candidate['source']['sha256'] == config['model_versions'][baseline]['sha256']:
        raise ContractError('candidate source is identical to baseline')
    from .insertion_contract import parse_contract
    insertion_scope=parse_contract(proposal['insertion_contract'])
    phases = []
    for partition in ('development', 'confirmation'):
        jobs = [{'model_version': name, 'dataset': dataset, 'seed': seed}
                for name in dict.fromkeys([baseline, *names]) for dataset in config['datasets']
                for seed in config[partition + '_seeds']]
        phases.append({'partition': partition, 'jobs': jobs,
                       'release_condition': 'code_admission' if partition == 'development' else 'frozen_selection_after_development'})
    return {'proposal_id': proposal['proposal_id'], 'baseline': baseline, 'protocol_hash': protocol,
            'cards': cards, 'versions': versions, 'phases': phases,
            **({'insertion_scope':insertion_scope} if insertion_scope is not None else {}),
            'reused_control_versions': reused_controls,
            'control_reuse_scope': 'existing outcomes remain previously exposed evidence; reuse requires matched protocol/execution; historical cost stays charged to study ledger', 'requested_budget_gpu_seconds': proposal['budget'],
            'budget_policy': 'shared study ledger; declaration is not a reservation',
            'baseline_policy': 'reuse only independently audited matching protocol and execution; otherwise rerun and charge',
            'qualification': 'preregistered_design_only', 'candidate_admitted': False,
            'confirmation_scope': 'new initializations on shared validation users; not final test'}


def register_plan(root, cards, versions, *, selection_policy=None):
    root = Path(root)
    with (root / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        store = Store(root / 'memory')
        try:
            config = load_config(root, store)
            artifacts = Artifacts(root / 'artifacts/objects')
            frozen = {'candidate': dict(versions['candidate']), 'controls': [dict(v) for v in versions['controls']]}
            for version in [frozen['candidate'], *frozen['controls']]:
                source = version['source']
                path = Path(source['path'])
                if path.is_symlink() or not path.is_file() or file_hash(path) != source['sha256']:
                    raise ContractError('planned source changed or is not a regular file')
                ref = artifacts.put(path)
                if ref['sha256'] != source['sha256']:
                    raise ContractError('planned source changed during archival')
                version['source'] = ref
            identity_body = {'cards': cards, 'versions': frozen}
            if selection_policy is not None:
                from .amazon_promotion import validate_policy
                validate_policy(selection_policy)
                if selection_policy=='baseline-single-seed-v1':
                    from .amazon_promotion import validate_single_seed
                    validate_single_seed(config)
                identity_body['selection_policy'] = selection_policy
            identity = digest(identity_body)
            for event in store.events():
                if event['kind'] != 'amazon_research_plan_registered': continue
                prior = event['payload']
                if prior['proposal_id'] == cards['proposal']['proposal_id']:
                    if prior['handoff_hash'] != identity:
                        raise ContractError('registered proposal cannot be rewritten')
                    return prior
                previous_names = {prior['versions']['candidate']['model_version'],
                                  *(v['model_version'] for v in prior['versions']['controls'])}
                current_names = {frozen['candidate']['model_version'],
                                 *(v['model_version'] for v in frozen['controls'])}
                reused = {v['model_version'] for v in frozen['controls'] if v['model_version'] in config['model_versions']}
                if (previous_names & current_names) - reused:
                    raise ContractError('model version already belongs to another registered plan')
            plan = compile_plan(config, cards, frozen, store.budget()['remaining'])
            if selection_policy is not None:plan['selection_policy']=selection_policy
            # Changing a candidate name cannot erase prior outcome exposure.
            hashes = {v['source']['sha256'] for v in [frozen['candidate'], *frozen['controls']]
                      if v['model_version'] not in plan['reused_control_versions']}
            for parent in ('jobs', 'attempt-history'):
                for receipt in (root / parent).rglob('completed.json'):
                    if read_json(receipt).get('source_hash') in hashes:
                        raise ContractError('planned candidate or control outcomes precede preregistration')
            state = store.verify()
            record = {**plan, 'handoff_hash': identity, 'before_event_head': state['head'],
                      'budget_at_registration': store.budget()}
            store.event('amazon_research_plan_registered', record)
            return record
        finally:
            store.close()
