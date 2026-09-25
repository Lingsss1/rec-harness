"""Local controller/cycle integrity audit; never send confirmation data to roles."""
from rec_harness.agents import schema_for,validate_schema
from rec_harness.core import Artifacts,ContractError,digest,file_hash,read_json
from rec_harness.store import Store


def verify_campaign(controller):
    root=controller.root;config=read_json(root/'amazon-auto.json')
    if digest(config)!=read_json(root/'auto-lock.json')['config_hash']:raise ContractError('autonomous configuration changed')
    if digest(read_json(controller.study/'study.json'))!=config['study_config_hash']:raise ContractError('bound study changed')
    integrity=controller.store.verify();events=controller.store.events()
    ledger=Store(controller.study/'memory');artifacts=Artifacts(controller.study/'artifacts/objects')
    role_count=0;local_only=0;partial=0;checked=set();cycles=0
    try:
        study_integrity=ledger.verify()
        records={e['payload']['identity']:e['payload'] for e in ledger.events() if e['kind']=='amazon_research_cycle_completed'}
        for marker in (root/'rounds').glob('round-*/*/accepted.json'):
            role=marker.parent.name
            if role not in {'evidence','measurement','hypothesis','experiment','evaluation','lead','knowledge'}:continue
            accepted=read_json(marker);response=read_json(marker.parent/'response.json');validate_schema(response,schema_for(role))
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
            else:local_only+=1
            role_count+=1
        for event in events:
            if event['kind']!='auto_round_completed':continue
            result=event['payload']['result']
            if 'cycle_identity' not in result:continue
            cycle=records.get(result['cycle_identity'])
            if not cycle or cycle['readout_hash']!=result['development_hash']:raise ContractError('round lacks its committed development cycle')
            folder=root/'rounds'/('round-'+str(event['payload']['state']['round']))
            if read_json(folder/'cycle.json')!=cycle:raise ContractError('cycle projection differs from committed study event')
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
            'roles_without_completion_event':local_only,'verified_cycles':cycles,'verified_artifacts':len(checked),
            'roles_with_legacy_partial_event_binding':partial,
            'qualification':'controller, role bindings and completed development cycle integrity; not a GPU job or private-confirmation re-audit'}
    finally:ledger.close()
