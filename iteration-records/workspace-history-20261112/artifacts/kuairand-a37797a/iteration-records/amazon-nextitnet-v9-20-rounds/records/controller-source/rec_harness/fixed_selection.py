"""Fixed-pool decision comparisons; scoring outcomes never enter role requests."""
import argparse
import fcntl
import statistics
import math
from pathlib import Path
from .agents import schema_for,validate_schema,ROLES
from .core import Artifacts,ContractError,digest,file_hash,identifier,number,public_only,read_json,write_json
from .jailed_agent import JailedCodexAgent
from .store import Store
from .validation_case import settle_once

GROUPS=('no_tool','existing_tool','candidate_tool')


def implementation_hashes():
    return {p.name:file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def freeze_suite(specification,directory):
    path=Path(specification).resolve();spec=read_json(path);root=Path(directory).resolve()
    if root.exists() and any(root.iterdir()):raise ContractError('fixed selection directory must be empty')
    public_only(spec)
    if spec['direction'] not in {'maximize','minimize'}:raise ContractError('invalid selection direction')
    number(spec['call_cost'],'selection call cost',0);number(spec['budget'],'selection budget',0)
    backend=spec['agent_backend']
    if backend.pop('kind','jailed_codex')!='jailed_codex':raise ContractError('fixed selection requires jailed Codex')
    if set(backend)-{'model','template','runtime_root','timeout','web_search'}:raise ContractError('unsupported fixed selection backend option')
    number(backend.setdefault('timeout',300),'selection timeout',0)
    if backend['timeout']<=0:raise ContractError('positive selection timeout required')
    if not isinstance(backend.get('model'),str) or not backend['model'].strip():raise ContractError('explicit selection model required')
    for key in ['template','runtime_root']:
        backend[key]=str((path.parent/backend[key]).resolve())
    if backend.get('web_search','disabled')!='disabled':raise ContractError('fixed selection uses supplied information only')
    backend['web_search']='disabled'
    seen=set();stages=set()
    for case in spec['cases']:
        identifier(case['id'])
        if case['id'] in seen:raise ContractError('duplicate fixed selection condition')
        seen.add(case['id']);stages.add(case['stage'])
        if case['stage'] not in {'development','confirmation'}:raise ContractError('invalid selection stage')
        candidates=case['candidates'];ids=[]
        if not isinstance(candidates,list) or len(candidates)<2:raise ContractError('fixed selection requires at least two candidates')
        for candidate in candidates:
            if set(candidate)!={'candidate_id','public'}:raise ContractError('candidate requires ID and public description')
            identifier(candidate['candidate_id']);ids.append(candidate['candidate_id'])
        if len(set(ids))!=len(ids) or set(case['scores'])!=set(ids):raise ContractError('fixed candidate score identities differ')
        for value in case['scores'].values():number(value,'fixed candidate score')
        if not math.isfinite(max(case['scores'].values())-min(case['scores'].values())):raise ContractError('fixed score range must be finite')
        if set(case['observations'])!=set(GROUPS) or case['observations']['no_tool']!={}:raise ContractError('three observation groups and empty no-tool view required')
        for group in GROUPS[1:]:
            if set(case['observations'][group])!=set(ids):raise ContractError('observations must cover the same fixed candidate pool')
        if not isinstance(case['context'],dict):raise ContractError('fixed selection context must be an object')
    if stages!={'development','confirmation'}:raise ContractError('independent development and confirmation conditions required')
    if spec['budget']<len(spec['cases'])*len(GROUPS)*spec['call_cost']:raise ContractError('budget does not cover frozen selection calls')
    body={**spec,'schema':'fixed-selection-suite-v1','selector_schema_hash':digest(schema_for('selection')),
        'selector_instruction_hash':digest(ROLES['selection']),
        'runner_hash':file_hash(__file__),'implementation_hashes':implementation_hashes()}
    body['identity']={
        'base_llm':digest(backend),'tools':digest([c['observations'] for c in spec['cases']]),
        'start':digest([c['candidates'] for c in spec['cases']]),'graph':digest('no persistent cross-case memory'),
        'budget':digest([spec['budget'],spec['call_cost']]),'protocol':digest([spec['direction'],spec['cases']])}
    body['hash']=digest(body);write_json(root/'suite.json',body);return body


def load_suite(path):
    suite=read_json(path)
    if suite.get('schema')!='fixed-selection-suite-v1' or suite.get('hash')!=digest({k:v for k,v in suite.items() if k!='hash'}):
        raise ContractError('fixed selection suite changed')
    if (suite['implementation_hashes']!=implementation_hashes() or suite['runner_hash']!=file_hash(__file__) or suite['selector_schema_hash']!=digest(schema_for('selection'))
            or suite['selector_instruction_hash']!=digest(ROLES['selection'])):
        raise ContractError('fixed selection implementation changed')
    return suite


def decision(store,objects,folder,key,context,agent,cost):
    expected={'key':key,'context_hash':digest(context)}
    submitted=[e['payload'] for e in store.events() if e['kind']=='fixed_selection_submitted' and e['payload']['key']==key]
    recorded=[e['payload'] for e in store.events() if e['kind']=='fixed_selection_recorded' and e['payload']['key']==key]
    if submitted:
        if len(submitted)!=1 or any(submitted[0][k]!=v for k,v in expected.items()):raise ContractError('fixed selection request changed')
        if len(recorded)!=1 or recorded[0]['request_hash']!=digest(submitted[0]):
            raise ContractError('fixed selection outcome unknown; no automatic resubmission')
        saved=recorded[0]
    else:
        if recorded or folder.exists():raise ContractError('unbound fixed selection artifacts')
        store.reserve(key,cost)
        folder.mkdir(parents=True);write_json(folder/'host-context.json',context)
        submission={**expected,'context_ref':objects.put(folder/'host-context.json')}
        store.event('fixed_selection_submitted',submission)
        response_ref=None;error_kind=None
        try:
            response=agent.run('selection',context,folder)
            write_json(folder/'host-response.json',response);response_ref=objects.put(folder/'host-response.json')
            public_only(response);validate_schema(response,schema_for('selection'))
            if response['candidate_id'] not in {c['candidate_id'] for c in context['candidates']}:raise ContractError('selection outside fixed pool')
        except (ContractError,OSError,ValueError,TypeError,KeyError) as error:error_kind=type(error).__name__
        saved={'key':key,'request_hash':digest(submission),'response_ref':response_ref,'error_kind':error_kind,
            'receipts':objects.snapshot(folder,[p.name for p in folder.iterdir() if p.is_file() and p.name in {
                'request.json','schema.json','prompt.txt','response.json','resources.json','process.json','stdout.jsonl','stderr.txt'}])}
        store.event('fixed_selection_recorded',saved)
    request=submitted[0] if submitted else submission
    if read_json(objects.verify(request['context_ref']))!=context:raise ContractError('fixed selection context archive changed')
    for ref in saved['receipts'].values():objects.verify(ref)
    response=read_json(objects.verify(saved['response_ref'])) if saved['response_ref'] else None
    passed=saved['error_kind'] is None
    if passed:
        validate_schema(response,schema_for('selection'))
        if response['candidate_id'] not in {c['candidate_id'] for c in context['candidates']}:raise ContractError('archived selection outside fixed pool')
    settle_once(store,key,cost,'accepted' if passed else 'rejected')
    return {'passed':passed,'selected_candidate':response['candidate_id'] if passed else None,
            'response_ref':saved['response_ref'],'error_kind':saved['error_kind']}


def run_validation(suite_path,output,stage,*,agent_factory=None):
    suite=load_suite(suite_path)
    if stage not in {'development','confirmation'}:raise ContractError('invalid fixed selection stage')
    root=Path(output).resolve().parent/'fixed-selection';root.mkdir(parents=True,exist_ok=True)
    with (root/'controller.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise ContractError('fixed selection already running') from error
        binding={'suite_hash':suite['hash'],'stage':stage,'live_decisions':agent_factory is None}
        if (root/'binding.json').exists() and read_json(root/'binding.json')!=binding:raise ContractError('fixed selection binding changed')
        write_json(root/'binding.json',binding);store=Store(root/'memory');objects=Artifacts(root/'objects')
        try:
            store.verify();store.configure_budget(suite['budget']);rows=[]
            for case in suite['cases']:
                if case['stage']!=stage:continue
                results={};sign=1 if suite['direction']=='maximize' else -1
                best=max(sign*value for value in case['scores'].values())
                for group in GROUPS:
                    context={'evaluation_mode':'fixed_candidate_selection','evaluation_id':case['id'],
                        'direction':suite['direction'],'context':case['context'],
                        'candidates':case['candidates'],'observations':case['observations'][group]}
                    public_only(context)
                    agent=agent_factory(case['id'],group) if agent_factory else JailedCodexAgent(**suite['agent_backend'])
                    key='select-'+digest([case['id'],group])[:32]
                    result=decision(store,objects,root/'decisions'/key,key,context,agent,suite['call_cost'])
                    score=case['scores'][result['selected_candidate']] if result['passed'] else None
                    results[group]={**result,'score':score,'regret':best-sign*score if score is not None else None}
                rows.append({'evaluation_id':case['id'],**results})
            valid=all(row[group]['passed'] for row in rows for group in GROUPS)
            gains={group:statistics.mean((1 if suite['direction']=='maximize' else -1)*(row['candidate_tool']['score']-row[group]['score']) for row in rows)
                   for group in GROUPS[:2]} if valid else {}
            report={'passed':valid,'evaluation_mode':'fixed_candidate_selection','online':False,
                'live_decisions':agent_factory is None,'gain':min(gains.values()) if valid else 0,
                'gain_by_reference':gains,'regression':max([0]+[-(1 if suite['direction']=='maximize' else -1)*(row['candidate_tool']['score']-row[g]['score']) for row in rows for g in GROUPS[:2]]) if valid else 0,
                'total_cost':store.budget()['charged_or_reserved'],'evaluation_ids':[r['evaluation_id'] for r in rows],
                'comparison_groups':list(GROUPS),'identity':suite['identity'],'rows':rows,'binding':binding,'integrity':store.verify(),
                'qualification':'Fixed supplied candidate pools, diagnostic observations and host outcomes. Measures selection only, not new candidate generation, diagnostic execution, retraining, or online model utility. Data independence and observation provenance require separate qualification.'}
            write_json(output,report);return report
        finally:store.close()


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='action',required=True)
    freeze=sub.add_parser('freeze');freeze.add_argument('specification');freeze.add_argument('directory')
    run=sub.add_parser('run');run.add_argument('--suite',required=True);run.add_argument('--output',required=True)
    run.add_argument('--stage',choices=['development','confirmation'],required=True)
    args=parser.parse_args()
    if args.action=='freeze':freeze_suite(args.specification,args.directory)
    else:run_validation(args.suite,args.output,args.stage)


if __name__=='__main__':main()
