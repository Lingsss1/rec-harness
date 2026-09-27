"""Link verified pre-design history aggregates into the development knowledge view."""
import fcntl
from rec_harness.core import Artifacts,ContractError,digest,read_json,write_json
from rec_harness.runtime import node,edge
from rec_harness.store import Store
from .amazon_baseline_history import verify_visible_history


def attach_history(controller,folder,graph,projection):
    verify_visible_history(controller,folder,projection)
    report=read_json(folder/'baseline-history.json')
    implementation=digest(sorted(report['binding']['code_hashes'].values()))
    task=graph['task_id'];protocol=report['binding']['protocol_hash']
    prefix='history-'+digest([task,projection['report_hash']])[:24]
    tool_id=prefix+'-tool';observation_id=prefix+'-observation'
    with (controller.study/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        store=Store(controller.study/'memory')
        try:
            artifacts=Artifacts(controller.study/'artifacts/objects')
            path=controller.study/'research-readouts'/(prefix+'.json');write_json(path,projection)
            ref=artifacts.put(path)
            nodes=[node(tool_id,'ToolVersion',{'implementation_hash':implementation,
                'version':'baseline-history-v1','scope':'Exploratory builtin aggregate; not utility-promoted.'},task,protocol),
                node(observation_id,'Observation',{**projection,'partition':'development','valid':True,
                    'independent_unit':False,'summary_ref':ref,
                    'source_receipt_hashes':sorted(v for k,v in report['binding']['input_hashes'].items() if k.endswith('/completed.json')),
                    'scientific_conclusion':'unknown'},task,protocol)]
            snapshot=store.snapshot();known={n['id']:n for n in snapshot['nodes']}
            for value in nodes:
                if value['id'] in known and known[value['id']]!=value:raise ContractError('history graph identity changed')
            missing=[n for n in nodes if n['id'] not in known]
            if missing:
                if len(missing)!=2:raise ContractError('incomplete history graph binding')
                store.patch({'base_version':store.version,'nodes':nodes,'edges':[
                    edge(prefix+'-measured',observation_id,tool_id,'MEASURED_BY',[observation_id],task,store.version+1)]})
            elif not any(e['id']==prefix+'-measured' and e['source']==observation_id and e['target']==tool_id
                         and e['relation']=='MEASURED_BY' for e in snapshot['edges']):
                raise ContractError('history graph measurement edge missing')
            snapshot=store.snapshot();selected=[n for n in snapshot['nodes'] if n['attrs']['task_id']==task]
            ids={n['id'] for n in selected}
            result={**graph,'graph_version':store.version,'nodes':selected,
                    'edges':[e for e in snapshot['edges'] if e['source'] in ids and e['target'] in ids],
                    'integrity':store.verify()}
            return result
        finally:store.close()
