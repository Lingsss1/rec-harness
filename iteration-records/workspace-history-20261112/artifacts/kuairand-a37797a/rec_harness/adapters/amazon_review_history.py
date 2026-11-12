"""Retain an interrupted review's costs and accepted roles in a successor."""
from pathlib import Path
import shutil

from rec_harness.core import ContractError,digest,file_hash,read_json,write_json
from rec_harness.cost_audit import audit_cost


def retain_predecessor(controller,binding,folder):
    source=controller.config.get('review_predecessor')
    if not source:return None
    from .amazon_autonomous import AmazonAutoresearch
    from .amazon_auto_audit import verify_reused_role
    source=Path(source).resolve()
    seen={controller.root.resolve()};cursor=source
    while cursor:
        if cursor in seen:raise ContractError('review predecessor cycle')
        seen.add(cursor)
        config=read_json(cursor/'amazon-auto.json')
        cursor=Path(config['review_predecessor']).resolve() if config.get('review_predecessor') else None
    previous=AmazonAutoresearch(source,load_agent=False,read_only=True)
    try:
        with previous.lock():
            if not previous.config.get('review_only'):raise ContractError('predecessor must be review-only')
            if (source/'review-result.json').exists():raise ContractError('completed review needs no successor')
            if any(previous.config.get(k)!=controller.config.get(k) for k in ['study','study_config_hash','execution_id']):
                raise ContractError('review predecessor execution differs')
            if read_json(source/'review-recovery.json')!=binding:raise ContractError('review predecessor binding differs')
            if previous.config.get('review_predecessor'):verify_predecessor(previous)
            cost=audit_cost(previous.store,previous.config['role_call_limit'])
            if cost['pending']:raise ContractError('review predecessor has pending calls')
            original=previous.store.db.execute("SELECT * FROM budget WHERE id='recovered-design-prior-calls'").fetchone()
            amount=binding['prior_role_cost']
            if original is None or original['actual']!=amount or original['reserved']!=amount or original['status']!='retained':
                raise ContractError('predecessor original design cost differs')
            extra=cost['charged_or_reserved']-amount
            if extra<0:raise ContractError('negative predecessor review cost')
            marker=controller.root/'review-predecessor.json'
            if marker.exists():
                record=read_json(marker)
                if record['source']!=str(source) or record['cost']!=cost or record['binding_hash']!=digest(binding):
                    raise ContractError('review predecessor changed')
                if record['config_hash']!=digest(previous.config):raise ContractError('review predecessor configuration changed')
            else:
                files=[p for base in [source/'rounds',source/'review-retries'] for p in base.rglob('*') if p.is_file()]
                refs=controller.artifacts.snapshot(source,[str(p.relative_to(source)) for p in files])
                record={'source':str(source),'config_hash':digest(previous.config),'binding_hash':digest(binding),
                        'cost':cost,'additional_cost':extra,'archive':refs}
                write_json(marker,record)
            for name,ref in record['archive'].items():
                controller.artifacts.verify(ref)
                if file_hash(source/name)!=ref['sha256']:raise ContractError('predecessor archived artifact changed')
            task='recovered-review-prior-calls'
            retained=controller.store.db.execute('SELECT * FROM budget WHERE id=?',(task,)).fetchone()
            if retained is None:controller.store.reserve(task,extra)
            elif retained['reserved']!=extra:raise ContractError('retained review cost changed')
            if retained is None or retained['status']=='running':controller.store.settle(task,extra,'retained')
            elif retained['actual']!=extra or retained['status']!='retained':raise ContractError('retained review settlement changed')
            for role in ['evaluation','lead','knowledge']:
                src=source/'rounds/round-1'/role
                if not (src/'accepted.json').exists():continue
                accepted=read_json(src/'accepted.json')
                if digest(read_json(src/'context.json'))!=accepted['context_hash'] or digest(read_json(src/'response.json'))!=accepted['response_hash']:
                    raise ContractError('predecessor accepted role changed')
                if (src/'reuse.json').exists():reuse=verify_reused_role(src,accepted)
                else:
                    matches=[e for e in previous.store.events() if e['kind']=='role_completed'
                             and e['payload']['role']==role and e['payload']['task'].startswith('round-1-'+role+'-')
                             and all(e['payload'].get(k)==v for k,v in accepted.items())]
                    if len(matches)!=1:raise ContractError('predecessor role needs unique completion event')
                    reuse={'source_role_directory':str(src),'source_response_hash':accepted['response_hash'],
                           'source_event_hash':matches[0]['hash'],'source_controller_head':cost['integrity']['head']}
                target=folder/role
                for path in src.rglob('*'):
                    if not path.is_file() or path.name=='reuse.json':continue
                    destination=target/path.relative_to(src);destination.parent.mkdir(parents=True,exist_ok=True)
                    if destination.exists() and file_hash(destination)!=file_hash(path):raise ContractError('reused review role changed')
                    if not destination.exists():shutil.copyfile(path,destination)
                if (target/'reuse.json').exists() and read_json(target/'reuse.json')!=reuse:raise ContractError('review reuse provenance changed')
                write_json(target/'reuse.json',reuse);verify_reused_role(target,accepted)
            event={'record_hash':digest(record),'archive':controller.artifacts.put(marker)}
            existing=[e['payload'] for e in controller.store.events() if e['kind']=='amazon_review_predecessor_retained']
            if existing and existing!=[event]:raise ContractError('committed review predecessor changed')
            if not existing:controller.store.event('amazon_review_predecessor_retained',event)
            return record
    finally:previous.close()


def verify_predecessor(controller,seen=None):
    """Recheck the immutable history chain, including all unsuccessful attempts."""
    if not controller.config.get('review_predecessor'):return None
    from .amazon_autonomous import AmazonAutoresearch
    seen=set() if seen is None else seen
    if controller.root in seen:raise ContractError('review predecessor cycle')
    seen.add(controller.root)
    record=read_json(controller.root/'review-predecessor.json')
    events=[e['payload'] for e in controller.store.events() if e['kind']=='amazon_review_predecessor_retained']
    if len(events)!=1 or events[0]['record_hash']!=digest(record):raise ContractError('committed review predecessor changed')
    if read_json(controller.artifacts.verify(events[0]['archive']))!=record:raise ContractError('review predecessor receipt differs')
    if str(Path(controller.config['review_predecessor']).resolve())!=record['source']:
        raise ContractError('review predecessor source changed')
    previous=AmazonAutoresearch(record['source'],load_agent=False,read_only=True)
    try:
        if digest(previous.config)!=record['config_hash']:raise ContractError('review predecessor configuration changed')
        if audit_cost(previous.store,previous.config['role_call_limit'])!=record['cost']:
            raise ContractError('review predecessor ledger changed')
        binding=read_json(controller.root/'review-recovery.json')
        if digest(binding)!=record['binding_hash'] or read_json(previous.root/'review-recovery.json')!=binding:
            raise ContractError('review predecessor binding changed')
        amount=record['cost']['charged_or_reserved']-binding['prior_role_cost']
        retained=controller.store.db.execute("SELECT * FROM budget WHERE id='recovered-review-prior-calls'").fetchone()
        if (amount<0 or record['additional_cost']!=amount or retained is None or retained['actual']!=amount
            or retained['reserved']!=amount or retained['status']!='retained'):
            raise ContractError('review predecessor retained cost changed')
        for name,ref in record['archive'].items():
            controller.artifacts.verify(ref)
            if file_hash(previous.root/name)!=ref['sha256']:raise ContractError('predecessor archived artifact changed')
        verify_predecessor(previous,seen)
        return record
    finally:previous.close()
