"""Finalize one audited development cycle; never promote from agent prose alone."""
import fcntl
import json
from pathlib import Path

from rec_harness.agents import validate_schema,schema_for
from rec_harness.core import Artifacts,ContractError,digest,file_hash,read_json,write_json
from rec_harness.runtime import node
from rec_harness.store import Store
from .amazon_selection import development_readout


from rec_harness.knowledge import validate_knowledge_patch


def failed_role_archives(directory):
    """Include explicit host-repair archives as well as contract retries."""
    directory=Path(directory)
    return sorted({folder.name for pattern in ('*-rejected-*','*-transport-archive-*')
                   for folder in directory.glob(pattern) if folder.is_dir()})


def finalize_cycle(root, proposal_id, directory):
    root,directory=Path(root),Path(directory)
    compact=(directory/'researcher/accepted.json').exists()
    decision_path=directory/('analyst/response.json' if compact else 'lead/response.json')
    decision=read_json(decision_path)
    knowledge=decision['knowledge'] if compact else read_json(directory/'knowledge/response.json')
    decision=decision['decision'] if compact else decision
    validate_schema(decision,schema_for('lead'));validate_schema(knowledge,schema_for('knowledge'))
    if compact:
        reviewer=read_json(directory/'reviewer/response.json')
        from rec_harness.code_review import validate_code_review
        validate_code_review(reviewer,allow_legacy=True,require_approval=True)
    graph=read_json(root/'research-readouts'/(proposal_id+'-graph.json'))
    patch=json.loads(knowledge['patch_json']);validate_knowledge_patch(patch,graph)
    if (directory/'measurement').exists():
        from rec_harness.measurement_discovery import validate_review
        validate_review(read_json(directory/'measurement/context.json'),read_json(directory/'measurement/response.json'))
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        store=Store(root/'memory')
        try:
            report=development_readout(root,proposal_id,store)
            if read_json(directory/'development-readout.json')['hash']!=report['hash'] or report['hash'] not in decision['evidence_refs']:
                raise ContractError('decision/readout evidence identity changed')
            if decision['conclusion_status'] not in {'unknown','provisional'}:
                raise ContractError('development cannot finalize a scientific confirmation')
            selection=decision['selection']
            if not report['summary']['eligible_for_confirmation']: selection='discard'
            if report['summary'].get('selection_policy') in {'baseline-positive-v1','baseline-single-seed-v1'}:
                selection='keep' if report['summary']['eligible_for_confirmation'] else 'discard'
            identity=digest([proposal_id,report['hash'],decision,knowledge])
            prior=[e['payload'] for e in store.events() if e['kind']=='amazon_research_cycle_completed' and e['payload']['proposal_id']==proposal_id]
            if prior:
                if prior[0]['identity']!=identity:raise ContractError('completed research cycle cannot be rewritten')
                for role,record in prior[0]['roles'].items():
                    for name,ref in record.get('artifacts',{}).items():
                        path=directory/role/name
                        if not path.is_file() or file_hash(path)!=ref['sha256']:
                            raise ContractError('committed '+role+' archive changed')
                write_json(directory/'cycle.json',prior[0]);return prior[0]
            current=store.snapshot()
            if {n['id'] for n in patch['nodes']} & {n['id'] for n in current['nodes']} or {e['id'] for e in patch['edges']} & {e['id'] for e in current['edges']}:
                raise ContractError('knowledge may not overwrite existing global graph identities')
            if store.version!=patch['base_version']:raise ContractError('knowledge graph became stale')
            artifacts=Artifacts(root/'artifacts/objects');roles={}
            role_names=['researcher','experiment','reviewer','analyst'] if compact else ['evidence','hypothesis','experiment','evaluation','lead','knowledge']
            if (directory/'measurement').exists():role_names.insert(1,'measurement')
            if (directory/'evidence-corrected').exists(): role_names.insert(1,'evidence-corrected')
            rejected=failed_role_archives(directory)
            role_names.extend(rejected)
            for role in role_names:
                folder=directory/role
                if role in rejected:
                    files=[name for name in ['context.json','request.json','response.json','resources.json','schema.json','rejection.json','literature-search.json','stdout.jsonl','process.json','launch.json'] if (folder/name).is_file()]
                    roles[role]={'artifacts':artifacts.snapshot(folder,files),'response_validation':{'status':'rejected'},'scope':'preserved failed attempt; not accepted evidence'}
                    continue
                resources=read_json(folder/'resources.json')
                if resources['status']!='success':raise ContractError('role has no successful terminal receipt')
                files=['context.json','request.json','response.json','resources.json','schema.json']
                if (folder/'accepted.json').exists():
                    files.extend('model/'+name for name in read_json(folder/'accepted.json').get('model_source_hashes',{}))
                files.extend(name for name in ['literature-search.json','stdout.jsonl','process.json','launch.json'] if (folder/name).is_file())
                rejection=read_json(folder/'rejection.json') if (folder/'rejection.json').exists() else None
                if rejection:files.append('rejection.json')
                reuse=None
                if (folder/'reuse.json').exists():
                    reuse=read_json(folder/'reuse.json')
                    if reuse['source_response_hash']!=digest(read_json(folder/'response.json')):
                        raise ContractError('reused role response identity changed')
                    original=Path(reuse['source_role_directory'])
                    if any((folder/name).read_bytes()!=(original/name).read_bytes() for name in files):
                        raise ContractError('reused role receipts differ from original call')
                    files.append('reuse.json')
                refs=artifacts.snapshot(folder,files)
                roles[role]={'artifacts':refs,'resources':resources}
                if reuse:roles[role]['reused_call']=reuse
                if rejection:roles[role]['response_validation']=rejection
            fact_id='amazon-cycle-'+identity[:24]
            patch['nodes'].append(node(fact_id,'DecisionTrace',{'selection':selection,
                'scientific_conclusion':decision['conclusion_status'],'readout_hash':report['hash'],
                'eligible_for_confirmation':report['summary']['eligible_for_confirmation'],
                'decision_ref':artifacts.put(decision_path),
                **({'decision_field':'decision'} if compact else {}),
                'scope':'development selection; incumbent unchanged unless separately confirmed'},graph['task_id'],report['protocol_hash']))
            record={'identity':identity,'proposal_id':proposal_id,'selection':selection,
                    'conclusion_status':decision['conclusion_status'],'readout_hash':report['hash'],
                    'roles':roles,'budget':store.budget(),'graph_version':store.version+1,
                    'scope':'completed development research cycle, not complete optimization goal',
                    'confirmation_released':False}
            store.patch(patch, accompanying_events=[('amazon_research_cycle_completed',record)])
            write_json(directory/'cycle.json',record)
            return record
        finally:store.close()
