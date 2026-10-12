#!/usr/bin/env python3
"""Live role-model validation with explicitly simulated compute boundaries.

Runs real autoresearch, plan registration, source AST validation, SQLite graph
ingestion, analyst patch validation, cycle commit, archival audit and resume.
Executes CPU synthetic contracts only when --cpu-python is supplied; never trains. Artifacts belong to an isolated fixture
study and MUST NOT be imported as real experimental evidence.
"""
import argparse
from contextlib import ExitStack
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rec_harness.agents import CodexAgent,ROLES
from rec_harness.autonomous import autoresearch
from rec_harness.adapters.amazon_autonomous import AmazonAutoresearch
from rec_harness.adapters.amazon_controls import load_config
from rec_harness.core import read_json,write_json,digest,file_hash,ContractError
from rec_harness.runtime import node
from rec_harness.store import Store

NOTICE=('LIVE INTEGRATION VALIDATION, NOT A REAL MODEL EXPERIMENT. All new baseline, candidate and control '
        'scores and numerical/compute receipts in this isolated campaign are SIMULATED. No training, '
        'forward/backward computation, independent rescoring or confirmation is performed. Historical imported '
        'memory is reported history, not validation of current code. Do not conclude any real mechanism or performance '
        'benefit from the mock scores. Do not launch training, use GPUs, install packages, access network, or read '
        'credentials or files outside this role directory. Read supplied model source and context. Return honest '
        'limitations. Do not delegate. Keep reasoning/output concise.')


class LiveAgent:
    def __init__(self,root):
        self.root=root
        notice=read_json(root/'config.json')['runtime_validation']['notice']
        prompts={role:notice for role in ROLES}
        prompts['researcher']+=' Propose a small, falsifiable causal readout architecture change confined to NextItNetModel.encode_sequence; preserve all other functions/modules. Name exactly one generated mechanism control. Use relevant graph memory and measurement gaps in your rationale; scientific status unknown. The GPU budget is a simulated ledger.'
        prompts['analyst']+=' To exercise graph write/read-back, append at least one provisional Claim and one MeasurementGap explicitly about the limits of SIMULATED outcomes and the real validation still required. Ground evidence_refs in current graph IDs. Never claim a real metric gain. Empty edges is valid. Keep patch attributes concise.'
        self.agent=CodexAgent(timeout=360,web_search='disabled',prompts=prompts)

    def run(self,role,context,directory):
        print('LIVE ROLE',directory.parent.name,role,flush=True)
        return self.agent.run(role,context,directory)


def prepare(root,rounds=2,seed_campaign=None,cpu_python=None):
    notice=NOTICE
    if cpu_python:
        notice=notice.replace('scores and numerical/compute receipts', 'scores and training/compute receipts').replace('No training, forward/backward computation, independent rescoring or confirmation is performed.', 'No training, independent rescoring or confirmation is performed. The host executes real CPU synthetic forward/backward correctness contracts without parameter updates; role agents do not execute model code.')
    study=root/'study';study.mkdir(parents=True)
    archive=ROOT/'docs/experiments/amazon-autoresearch-v7-runtime/records'
    base=study/'baseline.py';base.write_bytes((archive/'research/rounds/round-1/researcher/model/model.py').read_bytes())
    memory=read_json(archive/'development-memory.json')
    # One real exported historical round exercises cross-campaign provenance,
    # without copying private records or future v7 contexts.
    entry=next(e for e in memory['entries'] if e['round']=='amazon-next-v5/round-1')
    pack={k:memory[k] for k in ('schema','protocol_hash')};pack['entries']=[entry];pack['hash']=digest(pack)
    cfg={'model_versions':{'selected_v3':{'path':str(base),'sha256':file_hash(base)}},
         'protocol_hash':memory['protocol_hash'],'protocol':{'metric':'HR@10','minimum_effect':0.005},
         'datasets':['simulation-only'],'development_seeds':[42],'confirmation_seeds':[17,83],
         'python':str(cpu_python.absolute()) if cpu_python else sys.executable,'simulation':True}
    write_json(study/'study.json',cfg);write_json(study/'study-lock.json',{'config_hash':digest(cfg)})
    store=Store(study/'memory');store.configure_budget(100000);store.close()
    memories=[pack]
    if seed_campaign:
        from rec_harness.adapters.amazon_memory import export_development_memory
        memories.append(export_development_memory(seed_campaign,root/'prior-validation-memory.json'))
    config={'adapter':'amazon-nextitnet-v1','study':str(study),'task_id':'evidence-live-validation',
            'base_model_id':'selected_v3','max_rounds':rounds,'role_call_limit':12,
            'role_layout':'compact-four-v1','evidence_graph_policy':'prefix-evidence-v2',
            'generated_control_count':1,'rescore_recipe':{'simulation':True},'uid_registry':'simulation-only',
            'development_memory':memories,'agent_backend':{'template':'unused','runtime_root':'unused'},
            'runtime_validation':{'simulation':True,'notice':notice,'real_cpu_contracts':bool(cpu_python)}}
    write_json(root/'config.json',config)
    (root/'SIMULATION-ONLY.txt').write_text(notice+'\nNever import this study into production evidence.\n')
    return study


def run(root,rounds=2,seed_campaign=None,cpu_python=None):
    root=root.resolve()
    if not (root/'config.json').exists():study=prepare(root,rounds,seed_campaign,cpu_python)
    else:study=root/'study'
    runtime=read_json(root/'config.json')['runtime_validation']
    notice=runtime['notice'];real_cpu=runtime.get('real_cpu_contracts',False)
    campaign=root/'campaign';agent=LiveAgent(root)
    training=[]

    def simulated_result(cfg,version,dataset,seed):
        # Always below baseline, so this experiment cannot release confirmation.
        value=.5 if version=='selected_v3' else (.47 if 'control' in version else .48)
        return {'status':'success','simulation':True,'metrics':{'HR@10':value},
                'model_version':version,'dataset':dataset,'seed':seed,
                'source_hash':cfg['model_versions'][version]['sha256'],
                'scope':NOTICE}

    def audit_job(path,cfg,version,dataset,seed):
        receipt=Path(path)/'jobs'/f'{version}-{dataset}-{seed}'/'completed.json'
        if version=='selected_v3' and not receipt.exists():write_json(receipt,simulated_result(cfg,version,dataset,seed))
        if not receipt.exists():raise ContractError('simulation outcome requested before compute boundary')
        return read_json(receipt),{'candidates_hash':'simulation-only','data_hashes':{}}

    def phase(path,pid,partition,execution_id=None):
        if partition!='development':raise AssertionError('simulation must not reach confirmation')
        cfg=load_config(path);store=Store(Path(path)/'memory')
        try:plan=next(e['payload'] for e in store.events() if e['kind']=='amazon_research_plan_registered' and e['payload']['proposal_id']==pid)
        finally:store.close()
        for job in plan['phases'][0]['jobs']:
            target=Path(path)/'jobs'/f"{job['model_version']}-{job['dataset']}-{job['seed']}"/'completed.json'
            if not target.exists():write_json(target,simulated_result(cfg,job['model_version'],job['dataset'],job['seed']))
        training.append(pid);write_json(root/'simulated-compute.json',{'training_executed':False,'boundary_calls':training})
        print('SIMULATED COMPUTE',pid,flush=True)

    def ingest(path):
        cfg=load_config(path);store=Store(Path(path)/'memory')
        try:
            known={n['id'] for n in store.snapshot()['nodes']};nodes=[]
            for receipt in sorted((Path(path)/'jobs').glob('*/completed.json')):
                key=receipt.parent.name
                if key not in known:
                    nodes.append(node(key,'Experiment',{**read_json(receipt),'partition':'development',
                        'valid':True,'independent_unit':False},'simulation-worker',cfg['protocol_hash']))
            if nodes:store.patch({'base_version':store.version,'nodes':nodes,'edges':[]})
        finally:store.close()

    def gate(python,source,model_id,directory,env,timeout):
        write_json(Path(directory)/'report.json',{'simulation':True,'checks':{'parameter_gradients':{'SIMULATED_NOT_MEASURED':{'finite':True}}},'scope':NOTICE})
        return {'status':'success','simulation':True}

    def smoke(python,source,directory,recipe,artifacts,registry):
        file=Path(directory)/'simulation.json';write_json(file,{'simulation':True,'training_executed':False})
        return {'status':'success','simulation':True,'report_ref':artifacts.put(file)}

    def rescore(controller,folder,version,seeds):
        marker=root/'interruption-injected.json'
        if folder.name=='round-1' and version.endswith('-model') and not marker.exists():
            write_json(marker,{'at':'before simulated rescore','role_calls_before_interrupt':3})
            print('INJECT INTERRUPT AFTER IMPLEMENTATION/REVIEW',flush=True)
            raise KeyboardInterrupt()
        return [{'simulation':True,'training_executed':False}]

    with ExitStack() as stack:
        if sys.platform=='darwin':
            # CLI subprocess wait/kill remains real. This harness only interrupts
            # BETWEEN completed processes, never asserts Linux orphan recovery.
            stack.enter_context(patch('rec_harness.agents.identity',side_effect=lambda pid:{'pid':int(pid),'observable':False,'platform':'darwin-validation'}))
        for module in ('amazon_autonomous','amazon_selection'):
            stack.enter_context(patch('rec_harness.adapters.'+module+'.audit_job',side_effect=audit_job))
        stack.enter_context(patch('rec_harness.adapters.amazon_autonomous.run_phase',side_effect=phase))
        stack.enter_context(patch('rec_harness.adapters.amazon_autonomous.run_study',side_effect=AssertionError('real study forbidden')))
        stack.enter_context(patch('rec_harness.adapters.amazon_research_graph.ingest',side_effect=ingest))
        if not real_cpu:stack.enter_context(patch('rec_harness.adapters.amazon_admission.run_gate',side_effect=gate))
        stack.enter_context(patch('rec_harness.adapters.amazon_admission.run_training_smoke',side_effect=smoke))
        stack.enter_context(patch.object(AmazonAutoresearch,'_rescore',rescore))
        try:result=autoresearch(campaign,None if (campaign/'amazon-auto.json').exists() else root/'config.json',agent=agent)
        except KeyboardInterrupt:
            print('RESUME SAME CONTROLLER / REUSE ACCEPTED ROLES',flush=True)
            result=autoresearch(campaign,agent=agent)
        before={str(p):file_hash(p) for p in campaign.glob('rounds/*/*/request.json')}
        resumed=autoresearch(campaign,agent=agent)
        assert result==resumed
        assert before=={str(p):file_hash(p) for p in campaign.glob('rounds/*/*/request.json')}
        controller=AmazonAutoresearch(campaign,load_agent=False)
        try:audit=controller.verify()
        finally:controller.close()
    rounds=[]
    for folder in sorted((campaign/'rounds').glob('round-*')):
        context=read_json(folder/'input.json');g=context['evidence_graph']
        record={'round':folder.name,'graph_version':g['graph_version'],'graph_hash':g['hash'],
                'nodes':len(g['nodes']),'edges':len(g['edges']),
                'native_measurement_gaps':[n['id'] for n in g['nodes'] if n['type']=='MeasurementGap' and n['attrs'].get('origin')=='agent_hypothesis'],
                'imported_interpretations':[n['id'] for n in g['nodes'] if n['attrs'].get('kind')=='reported_interpretation']}
        if (folder/'researcher/response.json').exists():record['researcher']=read_json(folder/'researcher/response.json')
        if (folder/'reviewer/response.json').exists():record['reviewer']=read_json(folder/'reviewer/response.json')
        if (folder/'analyst/response.json').exists():
            record['analyst']=read_json(folder/'analyst/response.json')
            record['analyst_history_hash']=read_json(folder/'analyst/context.json')['prior_research_evidence']['hash']
        if (folder/'failure.json').exists():record['failure']=read_json(folder/'failure.json')
        rounds.append(record)
    report={'scope':notice,'real_components':['Codex role calls','source AST checks','plan registration','development aggregation',
        'graph ingestion','knowledge validation and commit','SQLite event chain','archive audit','resume'],
        'real_cpu_contracts':real_cpu,
        'simulated_components':([] if real_cpu else ['numerical gate'])+['training smoke','training worker','independent scoring','worker evidence ingestion'],
        'process_identity':'macOS validation adapter; no claim of Linux orphan-process recovery',
        'interrupted':(root/'interruption-injected.json').exists(),'terminal_resume_no_new_calls':True,
        'result':result,'audit':audit,'rounds':rounds}
    write_json(root/'validation-report.json',report)
    print('COMPLETE',root/'validation-report.json',flush=True)
    print({'verified_roles':audit.get('verified_roles'),'verified_cycles':audit.get('verified_cycles'),'failed_rounds':result.get('failed_rounds')},flush=True)
    return report


def probe_readback(root,probe_name='readback-probe'):
    """One fresh live researcher, after committed cycles, without another fit."""
    from rec_harness.adapters.amazon_memory import export_development_memory
    from rec_harness.adapters.amazon_compact import design
    from rec_harness.core import validate_card
    root=root.resolve();campaign=root/'campaign';probe=root/probe_name
    pack=export_development_memory(campaign,root/'development-memory-export.json')
    config=read_json(root/'config.json')
    config.update(task_id='evidence-live-readback',max_rounds=1,role_call_limit=4,
                  development_memory=[*config.get('development_memory',[]),pack])
    if not (probe/'amazon-auto.json').exists():AmazonAutoresearch.init(probe,config)
    agent=LiveAgent(root)
    with ExitStack() as stack:
        if sys.platform=='darwin':
            stack.enter_context(patch('rec_harness.agents.identity',side_effect=lambda pid:{'pid':int(pid),'observable':False,'platform':'darwin-validation'}))
        def audit(path,cfg,version,dataset,seed):
            return read_json(Path(path)/'jobs'/f'{version}-{dataset}-{seed}'/'completed.json'),{'candidates_hash':'simulation-only','data_hashes':{}}
        stack.enter_context(patch('rec_harness.adapters.amazon_autonomous.audit_job',side_effect=audit))
        controller=AmazonAutoresearch(probe,agent=agent)
        try:
            folder=probe/'rounds/round-1';folder.mkdir(parents=True,exist_ok=True)
            context=controller._context({'round':0,'incumbent':'selected_v3'},folder)
            write_json(folder/'input.json',context);binding=context['bindings']
            native_gaps=[n['id'] for n in context['evidence_graph']['nodes'] if n['type']=='MeasurementGap' and n['attrs'].get('origin')=='agent_hypothesis']
            def gap_check(value,_):
                validate_card('gap',value)
                if value['model_version']!=binding['base_model_id'] or value['protocol_hash']!=binding['train_protocol_hash']:raise ContractError('probe gap binding mismatch')
            def proposal_check(value,_,gap):
                validate_card('method',value['method'],require_method_sources=True);validate_card('proposal',value['proposal'])
                p=value['proposal']
                for key in ('proposal_id','base_model_id','controls','train_protocol_hash','seeds'):
                    if p[key]!=binding[key]:raise ContractError('probe proposal binding mismatch')
                if p['gap_ids']!=[gap['gap_id']] or p['method_ids']!=[value['method']['method_id']]:raise ContractError('probe card references mismatch')
            source=Path(load_config(config['study'])['model_versions']['selected_v3']['path'])
            design(controller,folder,context,source,gap_check,proposal_check)
            response=read_json(folder/'researcher/response.json')
            result={'scope':'Fresh live researcher after committed simulated cycles; no implementation or training.',
                    'native_measurement_gaps_visible':native_gaps,
                    'native_measurement_gaps_cited':sorted(set(native_gaps)&set(response['evidence_refs'])),
                    'graph_hash':context['evidence_graph']['hash'],'response':response,
                    'ledger':controller.store.verify(),'role_cost':controller.store.budget()}
            write_json(root/('readback-report.json' if probe_name=='readback-probe' else probe_name+'-report.json'),result)
            print('READBACK',{'visible':native_gaps,'cited':result['native_measurement_gaps_cited']},flush=True)
            return result
        finally:controller.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu-python',type=Path,help='Interpreter with torch/numpy; enables actual CPU admission gates, no optimizer updates')
    parser.add_argument('--probe-name',default='readback-probe')
    parser.add_argument('--probe-only',action='store_true')
    parser.add_argument('--rounds',type=int,default=2);parser.add_argument('--seed-campaign',type=Path)
    args=parser.parse_args()
    if args.probe_only:probe_readback(args.output,args.probe_name)
    else:run(args.output,args.rounds,args.seed_campaign,args.cpu_python)
