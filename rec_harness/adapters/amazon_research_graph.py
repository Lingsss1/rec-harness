"""Connect preregistered research cards to audited development facts."""
import fcntl
from pathlib import Path

from rec_harness.core import Artifacts, ContractError, digest, write_json
from rec_harness.runtime import node,edge
from rec_harness.store import Store
from .amazon_evidence import ingest
from .amazon_selection import development_readout


def ingest_development(root, proposal_id):
    root=Path(root)
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        report=development_readout(root,proposal_id)
        ingest(root)
        store=Store(root/'memory')
        try:
            plan=next(e['payload'] for e in store.events() if e['kind']=='amazon_research_plan_registered' and e['payload']['proposal_id']==proposal_id)
            task='amazon-research-'+digest(proposal_id)[:16]; protocol=report['protocol_hash']
            prefix=task+'-'+report['hash'][:16]; fact_id=prefix+'-development'
            known={n['id'] for n in store.snapshot()['nodes']}
            if fact_id not in known:
                sources=sorted(report['source_receipts'])
                if not set(sources).issubset(known):raise ContractError('development source experiments missing')
                path=root/'research-readouts'/(report['hash']+'.json');write_json(path,report)
                ref=Artifacts(root/'artifacts/objects').put(path)
                ids={kind:prefix+'-'+kind for kind in ('gap','method','proposal')}
                nodes=[node(ids[kind],typ,plan['cards'][kind],task,protocol,'agent_hypothesis')
                       for kind,typ in [('gap','GapHypothesis'),('method','Method'),('proposal','Proposal')]]
                summary=report['summary']
                nodes.append(node(fact_id,'Experiment',{'kind':'derived_development_comparison',
                    'partition':'development','valid':True,'status':'success','metric':summary['metric'],
                    'effect':summary['candidate_minus_comparator'][plan['baseline']],
                    'effect_gate_passed':summary['eligible_for_confirmation'],
                    'minimum_effect':summary['minimum_effect'],'independent_run_id':prefix+'-development-aggregate',
                    'seed_ids':sorted({row['seed'] for row in summary['slices']}),
                    'scientific_unit':'declared development seed aggregate; not additional independent runs or category slices',
                    'evidence_dependency_ids':sources,'report_ref':ref,'scientific_conclusion':'unknown',
                    'counterevidence':not summary['eligible_for_confirmation'],
                    'counterevidence_to':'candidate selection threshold; not a mechanism refutation'},task,protocol))
                edges=[edge(prefix+'-targets',ids['proposal'],ids['gap'],'TARGETS',[],task,store.version+1),
                       edge(prefix+'-uses',ids['proposal'],ids['method'],'USES',[],task,store.version+1),
                       edge(prefix+'-tests',fact_id,ids['proposal'],'TESTS',[fact_id],task,store.version+1)]
                for source in sources:
                    edges.append(edge(prefix+'-source-'+digest(source)[:12],fact_id,source,'DERIVED_FROM',[source],task,store.version+1))
                for row in summary['slices']:
                    identifier=prefix+'-slice-'+digest([row['dataset'],row['seed']])[:12]
                    nodes.append(node(identifier,'Observation',{**row,'partition':'development',
                        'independent_unit':False,'counterevidence':row['effects'][plan['baseline']]<0,
                        'evidence_dependency_ids':[fact_id],'report_ref':ref},task,protocol))
                    edges.append(edge(identifier+'-produced',fact_id,identifier,'PRODUCES',[fact_id],task,store.version+1))
                store.patch({'base_version':store.version,'nodes':nodes,'edges':edges})
            from .amazon_contract_graph import attach_contracts
            from .amazon_controls import load_config
            attach_contracts(root,store,load_config(root,store),report,task,protocol,fact_id)
            snapshot=store.snapshot()
            nodes=[n for n in snapshot['nodes'] if n['attrs']['task_id']==task]
            ids={n['id'] for n in nodes}
            result={'task_id':task,'graph_version':store.version,'readout_hash':report['hash'],'nodes':nodes,
                    'edges':[e for e in snapshot['edges'] if e['source'] in ids and e['target'] in ids],
                    'qualification':'development research subgraph; no confirmation evidence or raw user records',
                    'integrity':store.verify()}
            write_json(root/'research-readouts'/(proposal_id+'-graph.json'),result)
            return result
        finally:store.close()
