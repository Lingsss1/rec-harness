"""Admission for exact, reviewed aggregation edits; no arbitrary code admission."""
import fcntl
import uuid
from pathlib import Path

from rec_harness.core import Artifacts, ContractError, digest, file_hash, read_json
from rec_harness.store import Store
from .contract_gate import run_gate

OLD = '        blended = 0.85 * x + 0.15 * context'
EDITS = {
    'nextitnet_prefix020_v1': '        blended = 0.80 * x + 0.20 * context',
    'nextitnet_incumbent_scale16over17_v1': '        blended = (16.0 / 17.0) * (0.85 * x + 0.15 * context)',
}


RECENCY_VERSION = 'nextitnet_linear_recency_v1'
RECENCY_OLD = '''        preceding_sum = nn.functional.pad((x * valid).cumsum(dim=1)[:, :-1], (0, 0, 1, 0))
        preceding_count = nn.functional.pad(valid.cumsum(dim=1)[:, :-1], (0, 0, 1, 0))'''
RECENCY_NEW = '''        weights = valid * valid.cumsum(dim=1)
        preceding_sum = nn.functional.pad((x * weights).cumsum(dim=1)[:, :-1], (0, 0, 1, 0))
        preceding_count = nn.functional.pad(weights.cumsum(dim=1)[:, :-1], (0, 0, 1, 0))'''

COSINE_VERSION = 'nextitnet_cosine_context_v1'
COSINE_OLD = RECENCY_OLD + '\n        context = preceding_sum / preceding_count.clamp(min=1.0)'
COSINE_NEW = '''        preceding_count = nn.functional.pad(valid.cumsum(dim=1)[:, :-1], (0, 0, 1, 0))
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        unit = nn.functional.normalize(work, p=2, dim=-1, eps=1e-6)
        logits = torch.matmul(unit, unit.transpose(-1, -2))
        allowed = torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool).tril(-1)
        allowed = allowed.unsqueeze(0) & valid.squeeze(-1).bool().unsqueeze(1) & valid.squeeze(-1).bool().unsqueeze(2)
        logits = logits.masked_fill(~allowed, float('-inf'))
        logits = torch.where(allowed.any(dim=-1, keepdim=True), logits, torch.zeros_like(logits))
        weights = torch.softmax(logits, dim=-1).masked_fill(~allowed, 0.0)
        context = torch.matmul(weights, work).to(x.dtype)'''


def expected_sources(config, family='prefix_weight'):
    source = config['model_versions']['legacy_incumbent']
    if file_hash(source['path']) != source['sha256']:
        raise ContractError('reviewed incumbent source changed')
    text = Path(source['path']).read_text()
    if text.count(OLD) != 1:
        raise ContractError('reviewed prefix assignment not unique')
    if family == 'linear_recency':
        if text.count(RECENCY_OLD) != 1:
            raise ContractError('reviewed uniform prefix block not unique')
        return {RECENCY_VERSION: text.replace(RECENCY_OLD, RECENCY_NEW)}
    if family == 'cosine_context':
        if text.count(COSINE_OLD) != 1:
            raise ContractError('reviewed uniform context block not unique')
        return {COSINE_VERSION: text.replace(COSINE_OLD, COSINE_NEW)}
    if family != 'prefix_weight':
        raise ContractError('unknown reviewed prefix family')
    return {name: text.replace(OLD, replacement) for name, replacement in EDITS.items()}


def apply_registration(config, item, artifacts):
    expected = expected_sources(config, item.get('family', 'prefix_weight'))
    if item['protocol_hash'] != config['protocol_hash'] or set(item['versions']) != set(expected):
        raise ContractError('prefix candidate registration identity mismatch')
    for name, version in item['versions'].items():
        if name in config['model_versions']:
            raise ContractError('duplicate prefix candidate registration')
        source = version['source']
        if Path(source['path']).read_text() != expected[name] or file_hash(source['path']) != source['sha256']:
            raise ContractError('admitted prefix candidate source changed')
        artifacts.verify(version['source_ref']); artifacts.verify(version['probe_ref'])
        if version['source_ref']['sha256'] != source['sha256']:
            raise ContractError('prefix candidate archive identity mismatch')
        config['model_versions'][name] = source


def admit_plan(root, proposal_id):
    from .amazon_controls import load_config
    root = Path(root).resolve()
    with (root / 'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        store = Store(root / 'memory')
        try:
            config = load_config(root,store)
            prior = [e['payload'] for e in store.events() if e['kind']=='amazon_prefix_candidate_admitted' and e['payload']['proposal_id']==proposal_id]
            if prior: return prior[0]
            plans = [e['payload'] for e in store.events() if e['kind']=='amazon_research_plan_registered' and e['payload']['proposal_id']==proposal_id]
            if len(plans)!=1: raise ContractError('unique preregistered research plan required')
            plan=plans[0]
            variants=[plan['versions']['candidate'],*plan['versions']['controls']]
            family='prefix_weight'
            candidate_name=plan['versions']['candidate']['model_version']
            if candidate_name in (RECENCY_VERSION,COSINE_VERSION):
                family='linear_recency' if candidate_name==RECENCY_VERSION else 'cosine_context'
                if [v['model_version'] for v in plan['versions']['controls']]!=['legacy_incumbent'] or plan.get('reused_control_versions')!=['legacy_incumbent']:
                    raise ContractError('aggregation candidate requires unchanged incumbent control')
                control=plan['versions']['controls'][0]
                if control['source']['sha256']!=config['model_versions']['legacy_incumbent']['sha256']:
                    raise ContractError('reused incumbent control identity changed')
                variants=[plan['versions']['candidate']]
            if (family=='prefix_weight' and [v['model_version'] for v in variants] != list(EDITS)) or plan['baseline']!='legacy_incumbent':
                raise ContractError('plan is outside reviewed prefix transformation')
            artifacts=Artifacts(root/'artifacts/objects'); expected=expected_sources(config,family)
            # Verify exact byte edits before executing any candidate source.
            for version in variants:
                ref=version['source']; artifacts.verify(ref)
                if (artifacts.root/ref['sha256']).read_bytes()!=expected[version['model_version']].encode():
                    raise ContractError('candidate exceeds reviewed aggregation edit')
            directory=root/'prefix-admissions'/proposal_id
            if directory.exists():
                # No admission event exists: retain the uncommitted attempt before retrying.
                previous=root/'prefix-admission-attempts'/proposal_id/uuid.uuid4().hex
                previous.parent.mkdir(parents=True,exist_ok=True)
                directory.rename(previous)
                store.event('amazon_prefix_admission_retried',{'proposal_id':proposal_id,
                    'previous_attempt':str(previous.relative_to(root)),
                    'scope':'uncommitted CPU admission attempt; no training was scheduled'})
            directory.mkdir(parents=True,exist_ok=False)
            admitted={}
            for version in variants:
                name=version['model_version']; source_path=directory/(name+'.py')
                source_path.write_text(expected[name])
                source={'path':str(source_path.resolve()),'sha256':file_hash(source_path)}
                gate=run_gate(config['python'],source,name,directory/name,{},60)
                if gate['status']!='success': raise ContractError('reviewed prefix source failed CPU contract')
                probe=read_json(directory/name/'report.json')
                gradients=probe['checks']['parameter_gradients']
                if not gradients or not all(g.get('finite') is True for g in gradients.values()):
                    raise ContractError('reviewed prefix parameter gradients invalid')
                admitted[name]={'source':source,'source_ref':artifacts.put(source_path),
                                'probe_ref':artifacts.put(directory/name/'report.json'),'gate':gate}
            record={'proposal_id':proposal_id,'plan_handoff_hash':plan['handoff_hash'],
                    'protocol_hash':config['protocol_hash'],'versions':admitted,'family':family,
                    'scope':'exact reviewed prefix transformation; no arbitrary generated code',
                    'confirmation_released':False}
            store.event('amazon_prefix_candidate_admitted',record)
            return record
        finally: store.close()


def plan_guard(store, versions, seeds, confirmation_seeds):
    """Return a plan when running admitted versions; confirmation needs a release."""
    events=store.events()
    admissions=[e['payload'] for e in events if e['kind'] in {'amazon_prefix_candidate_admitted','amazon_model_candidate_admitted'}
                and set(versions)&set(e['payload']['versions'])]
    if not admissions: return None
    if len(admissions)!=1 or not set(versions).issubset(admissions[0]['versions']):
        raise ContractError('run one admitted candidate/control plan at a time')
    admission=admissions[0]
    plans=[e['payload'] for e in events if e['kind']=='amazon_research_plan_registered'
           and e['payload']['proposal_id']==admission['proposal_id']]
    if len(plans)!=1 or plans[0]['handoff_hash']!=admission['plan_handoff_hash']:
        raise ContractError('admission preregistration identity mismatch')
    plan=plans[0]
    if any(e['kind']=='amazon_research_plan_blocked' and
           e['payload'].get('plan_handoff_hash')==plan['handoff_hash'] for e in events):
        raise ContractError('research plan blocked; inspect archived feasibility decision')
    if set(seeds)&set(confirmation_seeds) and not any(
            e['kind']=='amazon_confirmation_released' and e['payload'].get('plan_handoff_hash')==plan['handoff_hash'] for e in events):
        raise ContractError('confirmation requires frozen selection after development')
    return plan


def plan_remaining(store, plan):
    if plan is None: return store.budget()['remaining']
    versions=[plan['versions']['candidate']['model_version'],*(v['model_version'] for v in plan['versions']['controls'])]
    versions=[v for v in versions if v not in plan.get('reused_control_versions',[])]
    spent=0
    for row in store.db.execute('SELECT id,reserved,actual FROM budget'):
        if any(row['id'].startswith(name+'-') for name in versions):
            spent += row['reserved'] if row['actual'] is None else row['actual']
    return min(store.budget()['remaining'],plan['requested_budget_gpu_seconds']-spent)
