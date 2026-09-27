"""Link archived development contracts without expanding tensor graphs into prompts."""
from pathlib import Path
from rec_harness.core import Artifacts,ContractError,digest,read_json,write_json
from rec_harness.runtime import node,edge
from .amazon_training_evidence import training_contract_evidence


def attach_contracts(root,store,config,report,task,protocol,fact_id):
    root=Path(root);destination=store.root.parent
    if report.get('hash')!=digest({k:v for k,v in report.items() if k!='hash'}):
        raise ContractError('contract graph requires intact development readout')
    if report.get('protocol_hash')!=protocol or config.get('protocol_hash')!=protocol:
        raise ContractError('contract graph protocol differs')
    snapshot=store.snapshot();known={n['id']:n for n in snapshot['nodes']}
    if fact_id not in known:raise ContractError('contract graph development fact missing')
    artifacts=Artifacts(destination/'artifacts/objects');nodes={};relations=[];observations=[]
    for row in report['rows']:
        version,dataset,seed=(row[k] for k in ('model_version','dataset','seed'))
        job=f'{version}-{dataset}-{seed}'
        if job not in known or known[job]['type']!='Experiment':raise ContractError('contract graph source experiment missing')
        result=read_json(root/'jobs'/job/'completed.json')
        if digest(result)!=report['source_receipts'].get(job,{}).get('receipt_hash'):
            raise ContractError('contract graph receipt differs from development readout')
        evidence=training_contract_evidence(root,config,version,dataset,seed,result)
        source=report['source_hashes'][version]
        if result.get('source_hash')!=source:raise ContractError('contract graph model source differs')
        model_id='contract-model-'+digest([task,version,source])[:24]
        nodes[model_id]=node(model_id,'ModelVersion',{'design_id':version,'code_hash':source,
            'scope':'Development model identity; synthetic contracts are not quality evidence.'},task,protocol)
        identity='training-contract-'+digest([task,job,evidence])[:24]
        path=destination/'research-readouts'/(identity+'.json')
        if path.exists():
            if digest(read_json(path))!=digest(evidence):raise ContractError('contract graph cached summary changed')
        else:write_json(path,evidence)
        summary_ref=artifacts.put(path);reports={};stages={}
        for stage,value in evidence.get('stages',{}).items():
            ref=artifacts.put(root/'jobs'/job/(stage+'-contract')/'report.json')
            archived=read_json(artifacts.verify(ref))
            if archived.get('hash')!=value['report_hash'] or digest({k:v for k,v in archived.items() if k!='hash'})!=value['report_hash']:
                raise ContractError('contract report changed during archival')
            reports[stage]=ref
            graph=value['tensor_graph']
            stages[stage]={'weight_state':value['weight_state'],'report_hash':value['report_hash'],
                'checks':value['checks'],'sequence_semantics_status':value['sequence_semantics']['status'],
                'tensor_graph_status':graph['status'],
                **({'tensor_graph_hash':graph['graph_hash'],'operation_count':graph['operation_count']} if graph['status']=='verified' else {})}
        attrs={'kind':'synthetic_training_contract','partition':'development','model_version':version,
            'dataset':dataset,'seed':seed,'status':evidence['status'],'valid':evidence['status']=='verified',
            'independent_unit':False,'scientific_conclusion':'unknown','evidence_dependency_ids':[job],
            'summary_ref':summary_ref,'report_refs':reports,'stages':stages,
            'scope':'Archived implementation checks; not another independent run or mechanism/quality evidence.'}
        nodes[identity]=node(identity,'Observation',attrs,task,protocol);observations.append(identity)
        for suffix,left,right,relation in [('model',identity,model_id,'ABOUT'),('source',identity,job,'DERIVED_FROM'),('used',fact_id,identity,'USES')]:
            relations.append(edge(identity+'-'+suffix,left,right,relation,[job],task,store.version+1))
    missing=[]
    for identity,value in nodes.items():
        if identity in known and known[identity]!=value:raise ContractError('contract graph node changed')
        if identity not in known:missing.append(value)
    existing_edges={e['id']:e for e in snapshot['edges']};new_edges=[]
    for relation in relations:
        old=existing_edges.get(relation['id'])
        if old is not None:
            if (any(old[key]!=relation[key] for key in ('source','target','relation')) or
                any(old['attrs'].get(key)!=relation['attrs'][key] for key in ('scope','evidence_refs','origin','status'))):
                raise ContractError('contract graph relation changed')
        else:new_edges.append(relation)
    if missing or new_edges:store.patch({'base_version':store.version,'nodes':missing,'edges':new_edges})
    return {'observations':observations,'added_nodes':len(missing),'added_edges':len(new_edges)}
