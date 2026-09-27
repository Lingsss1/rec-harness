"""Local controller/cycle integrity audit; never send confirmation data to roles."""
from pathlib import Path
from rec_harness.agents import schema_for,validate_schema
from rec_harness.core import Artifacts,ContractError,digest,file_hash,read_json
from rec_harness.store import Store
from rec_harness.cost_audit import audit_cost,verify_cost_prefix


def verify_reused_role(folder,accepted):
    reuse=read_json(folder/'reuse.json');source=Path(reuse['source_role_directory'])
    if reuse['source_response_hash']!=accepted['response_hash']:raise ContractError('reused response identity changed')
    old=Store(source.parent.parent.parent/'memory',read_only=True)
    try:
        integrity=old.verify()
        if integrity['head']!=reuse['source_controller_head']:raise ContractError('reused source controller changed')
        events=[e for e in old.events() if e['hash']==reuse['source_event_hash']]
        if len(events)!=1 or events[0]['kind']!='role_completed':raise ContractError('reused completion event missing')
        value=events[0]['payload']
        if value['role']!=folder.name or any(value.get(k)!=v for k,v in accepted.items()):
            raise ContractError('reused role event binding changed')
        artifacts=Artifacts(old.root.parent/'artifacts/objects')
        for name,ref in value.get('archive',{}).items():
            artifacts.verify(ref)
            if file_hash(source/name)!=ref['sha256'] or file_hash(folder/name)!=ref['sha256']:
                raise ContractError('reused committed artifact changed')
    finally:old.close()
    return reuse


def verify_reopen_history(controller):
    events=controller.store.events();known={};count=0;latest_transition=None
    for event in events:
        if event['kind']=='auto_round_reopened':
            value=event['payload'];failed=known.get(value['failed_event_hash'])
            if not failed or failed!=latest_transition or failed['kind']!='auto_round_completed' or failed['payload']['result'].get('status')!='failed':
                raise ContractError('reopened round lacks prior failure')
            expected=failed['payload']['state']
            if value['state']!={'incumbent':expected['incumbent'],'round':expected['round']-1,'status':'ready'}:
                raise ContractError('reopened state differs from failed round')
            verify_cost_prefix(controller.store,{'schema':'controller-cost-prefix-v1',
                'memory':str(controller.store.root.resolve()),'cutoff_seq':event['seq']-1,
                'event_head':event['previous_hash'],'limit':controller.config['role_call_limit'],
                'charged_or_reserved':value['retained_role_cost'],'pending':0})
            refs=value['prior_artifacts']
            if any(name in refs for name in ['handoff.json','preregistration.json','admission.json','private-confirmation-started.json']):
                raise ContractError('reopened archive is not pre-admission')
            if not value['controller_sources'] or 'failure.json' not in refs:
                raise ContractError('reopened round archive incomplete')
            for ref in [*refs.values(),*value['controller_sources'].values()]:controller.artifacts.verify(ref)
            if read_json(controller.artifacts.verify(refs['failure.json']))!=failed['payload']['result']:
                raise ContractError('reopened failure archive differs')
            # A cached role completion must match the pre-repair bytes. A newly
            # submitted role is separately bound by its new input/output receipt.
            for later in events[event['seq']:]:
                if later['kind']=='auto_round_reopened':break
                if later['kind']!='role_completed':continue
                result=later['payload'];task=result['task']
                if not task.startswith('round-'+str(expected['round'])+'-'):continue
                prior_submission=any(e['kind']=='agent_submitted' and e['payload']['task_id']==task for e in events[:event['seq']-1])
                if not prior_submission:continue
                for name,key in [('context.json','context_hash'),('response.json','response_hash'),('model/model.py','source_hash')]:
                    if key not in result:continue
                    ref=refs.get(result['role']+'/'+name)
                    if ref is None:raise ContractError('reused role absent from failure archive')
                    path=controller.artifacts.verify(ref)
                    actual=file_hash(path) if key=='source_hash' else digest(read_json(path))
                    if actual!=result[key]:raise ContractError('reused role differs from pre-repair archive')
            count+=1
        known[event['hash']]=event
        if event['kind'] in {'auto_round_completed','auto_round_reopened'}:latest_transition=event
    return count


def verify_campaign(controller):
    root=controller.root;config=read_json(root/'amazon-auto.json')
    if digest(config)!=read_json(root/'auto-lock.json')['config_hash']:raise ContractError('autonomous configuration changed')
    if digest(read_json(controller.study/'study.json'))!=config['study_config_hash']:raise ContractError('bound study changed')
    integrity=controller.store.verify();events=controller.store.events()
    ledger=Store(controller.study/'memory');artifacts=Artifacts(controller.study/'artifacts/objects')
    role_count=0;local_only=0;partial=0;checked=set();cycles=0;measurement_bindings=0;legacy_measurements=0;reused_roles=0
    try:
        study_integrity=ledger.verify()
        cost=audit_cost(controller.store,config['role_call_limit'])
        reopened=verify_reopen_history(controller)
        records={e['payload']['identity']:e['payload'] for e in ledger.events() if e['kind']=='amazon_research_cycle_completed'}
        for marker in (root/'rounds').glob('round-*/*/accepted.json'):
            role=marker.parent.name
            if role not in {'evidence','measurement','hypothesis','experiment','evaluation','lead','knowledge'}:continue
            accepted=read_json(marker);response=read_json(marker.parent/'response.json');validate_schema(response,schema_for(role))
            if role=='measurement':
                from rec_harness.measurement_discovery import validate_review
                validate_review(read_json(marker.parent/'context.json'),response)
            if digest(response)!=accepted['response_hash'] or digest(read_json(marker.parent/'context.json'))!=accepted['context_hash']:
                raise ContractError('accepted role content changed')
            if 'source_hash' in accepted and file_hash(marker.parent/'model/model.py')!=accepted['source_hash']:
                raise ContractError('accepted role source changed')
            completed=[e['payload'] for e in events if e['kind']=='role_completed' and e['payload']['role']==role
                and e['payload']['task'].startswith(marker.parent.parent.name+'-'+role+'-')]
            if completed:
                matches=[value for value in completed if value['response_hash']==accepted['response_hash']]
                if not matches:raise ContractError('accepted response differs from committed role')
                if not any(all(value.get(key,accepted[key])==accepted[key] for key in accepted) for value in matches):
                    raise ContractError('accepted binding differs from committed role')
                if not any(set(accepted).issubset(value) for value in matches):partial+=1
            elif (marker.parent/'reuse.json').exists():
                verify_reused_role(marker.parent,accepted);reused_roles+=1
            else:local_only+=1
            role_count+=1
        references=[(event['payload']['result'],event['payload']['state']['round'])
                    for event in events if event['kind']=='auto_round_completed']
        if config.get('review_only') and (root/'review-result.json').exists():
            recovered=read_json(root/'review-result.json')
            if recovered.get('status')!='completed' or recovered['role_budget']!=controller.store.budget():
                raise ContractError('recovered review terminal receipt changed')
            binding=read_json(root/'review-recovery.json')
            original=Store(Path(binding['source_round']).parent.parent/'memory',read_only=True)
            try:original_cost=audit_cost(original,original.budget()['limit'])
            finally:original.close()
            retained=controller.store.db.execute("SELECT * FROM budget WHERE id='recovered-design-prior-calls'").fetchone()
            amount=original_cost['charged_or_reserved']
            if (original_cost['integrity']['head']!=binding['prior_role_head'] or amount!=binding['prior_role_cost']
                or amount!=recovered['prior_role_cost'] or retained is None or retained['actual']!=amount
                or retained['reserved']!=amount or retained['status']!='retained'):
                raise ContractError('recovered review inherited calls differ from source ledger')
            references.append((recovered,1))
        for result,round_number in references:
            if 'cycle_identity' not in result:continue
            cycle=records.get(result['cycle_identity'])
            if not cycle or cycle['readout_hash']!=result['development_hash']:raise ContractError('round lacks its committed development cycle')
            folder=root/'rounds'/('round-'+str(round_number))
            if config.get('review_only') and result.get('selection')!=cycle['selection']:
                raise ContractError('recovered review selection differs from committed cycle')
            if read_json(folder/'cycle.json')!=cycle:raise ContractError('cycle projection differs from committed study event')
            if (folder/'measurement/accepted.json').exists():
                if 'measurement' in cycle['roles']:measurement_bindings+=1
                else:legacy_measurements+=1
            for role,receipt in cycle['roles'].items():
                for name,ref in receipt['artifacts'].items():
                    artifacts.verify(ref);checked.add(ref['sha256'])
                    if file_hash(folder/role/name)!=ref['sha256']:raise ContractError('role archive differs from committed cycle')
            cycles+=1
        # Local objects include independent CPU rescore outputs. Verify bytes
        # without reading checkpoint contents or exposing confirmation metrics.
        for path in controller.artifacts.root.iterdir():
            if not path.is_file() or len(path.name)!=64:raise ContractError('unexpected controller artifact entry')
            controller.artifacts.verify({'sha256':path.name,'size':path.stat().st_size});checked.add(path.name)
        if ledger.verify()!=study_integrity:raise ContractError('study changed during campaign verification')
        if controller.store.verify()!=integrity:raise ContractError('controller changed during verification')
        return {**integrity,'study_integrity':study_integrity,'verified_roles':role_count,
            'verified_source_reopenings':reopened,'role_costs':cost,
            'roles_without_completion_event':local_only,'verified_cycles':cycles,'verified_artifacts':len(checked),
            'roles_with_legacy_partial_event_binding':partial,
            'roles_verified_via_original_completion_events':reused_roles,
            'measurement_cycle_bindings':measurement_bindings,'measurement_roles_without_cycle_binding':legacy_measurements,
            'qualification':'controller, role bindings and completed development cycle integrity; not a GPU job or private-confirmation re-audit'}
    finally:ledger.close()
