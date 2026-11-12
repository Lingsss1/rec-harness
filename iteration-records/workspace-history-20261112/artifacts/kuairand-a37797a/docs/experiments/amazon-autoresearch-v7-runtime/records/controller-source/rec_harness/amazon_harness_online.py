"""Independent Amazon development/confirmation pairs for research policies."""
import argparse
import copy
import fcntl
import statistics
from pathlib import Path

from .amazon_tool_online import freeze_cases,episode,read_result
from .amazon_tool_cost import check_binding,real_transports
from .adapters.amazon_execution import code_identity
from .core import Artifacts,ContractError,digest,file_hash,number,read_json,write_json
from .evolution import validate_policy

GROUPS=('baseline','candidate')


def identity(suite):
    cases=suite['cases']
    result = {'base_llm':digest([c['controller']['agent_backend'] for c in cases]),
        'tools':digest([c['controller'].get('activation_diagnostic') for c in cases]),
        'start':digest([c['models'] for c in cases]),
        'graph':digest('fresh independent Amazon graph overlays per policy'),
        'budget':digest([suite['cost_weights'],[{k:c['controller'].get(k) for k in
            ['max_rounds','role_call_limit','diagnostic_budget','activation_diagnostic']} for c in cases],
            [c['study']['budget_gpu_seconds'] for c in cases]]),
        'protocol':digest([c['study'] for c in cases])}
    if any('semantic_encoder' in c['controller'] for c in cases):
        result['tools']=digest([result['tools'],[c['controller'].get('semantic_encoder') for c in cases]])
    return result


def freeze_suite(specification,directory):
    path=Path(specification).resolve();spec=read_json(path);root=Path(directory).resolve()
    if root.exists() and any(root.iterdir()):raise ContractError('Amazon Harness suite directory must be empty')
    if set(spec)-{'cases','cost_weights','baseline_policy'}:raise ContractError('unsupported Amazon Harness suite option')
    weights=spec['cost_weights'];required={'role_calls','gpu_seconds'}
    if set(weights) not in [required,required|{'activation_units'}]:raise ContractError('explicit role and GPU cost weights required')
    for value in weights.values():
        if number(value,'ledger cost weight',0)<=0:raise ContractError('positive ledger cost weights required')
    policy=spec.get('baseline_policy',{});validate_policy(policy)
    cases=freeze_cases(spec['cases'],path.parent,Artifacts(root/'objects'),weights,require_diagnostics=False)
    body={'schema':'amazon-online-harness-suite-v1','cases':cases,'baseline_policy':policy,
        'cost_weights':weights,'code_hashes':code_identity(),
        'qualification':'Paired Amazon research with fixed built-in observations and independent empty memories. '
            'Scores terminal development HR@10 under the frozen internal confirmation rule. '
            'Different initialization seeds may share validation users; not final-test evidence. '
            'CPU admission/rescoring wall time and external optimizer development costs are not priced here.'}
    body['identity']=identity(body);body['hash']=digest(body)
    write_json(root/'suite.json',body);return body


def load_suite(path):
    path=Path(path).resolve();suite=read_json(path)
    if (suite.get('schema')!='amazon-online-harness-suite-v1'
            or suite.get('hash')!=digest({k:v for k,v in suite.items() if k!='hash'})
            or suite.get('identity')!=identity(suite)):raise ContractError('Amazon Harness suite changed')
    if suite['code_hashes']!=code_identity():raise ContractError('Amazon Harness implementation changed')
    validate_policy(suite['baseline_policy']);objects=Artifacts(path.parent/'objects')
    for case in suite['cases']:
        for ref in [case['worker'],*case['models'].values()]:objects.verify(ref)
        for name,sha in case['study']['data_hashes'].items():
            if file_hash(Path(case['study']['data'])/name)!=sha:raise ContractError('private data identity differs')
    return suite,objects


def policy_case(case,policy):
    result=copy.deepcopy(case);validate_policy(policy)
    result['controller']['research_policy']=copy.deepcopy(policy)
    return result


def verify_group(root,case,policy,*,scripted=False):
    """Check effective policy and frozen study inputs before accepting a score."""
    adapted=policy_case(case,policy)
    config=check_binding(root,adapted,{},'no_tool',{},study_group=root.name,scripted_agent=scripted)
    result=read_result(root)
    if read_json(root/'completed.json')!=result:raise ContractError('Amazon Harness completed result changed')
    if result['diagnostic_calls'] or result['diagnostic_failures']:
        raise ContractError('Amazon Harness comparison requires fixed built-in tools')
    if not scripted:real_transports(root,config)
    return result


def summarize(suite,binding,rows):
    if not rows:raise ContractError('no Amazon Harness comparison conditions')
    deltas=[]
    for row in rows:
        if len({r['protocol_hash'] for g in GROUPS for r in [row[g]]})!=1:
            raise ContractError('Amazon Harness paired protocols differ')
        if len({digest(row[g]['evaluation_conditions']) for g in GROUPS})!=1:
            raise ContractError('Amazon Harness paired evaluation populations differ')
        for group in GROUPS:
            if set(row[group]['costs'])-set(suite['cost_weights']):
                raise ContractError('Amazon Harness report has an unpriced cost axis')
        deltas.append(row['candidate']['score']-row['baseline']['score'])
    axes={axis:sum(row[g]['costs'].get(axis,{}).get('charged_or_reserved',0)
        for row in rows for g in GROUPS) for axis in suite['cost_weights']}
    return {'passed':True,'gain':statistics.mean(deltas),'regression':max([0]+[-d for d in deltas]),
        'total_cost':number(sum(axes[k]*v for k,v in suite['cost_weights'].items()),'weighted ledger cost',0),
        'cost_axes':axes,'cost_weights':suite['cost_weights'],'comparison_groups':list(GROUPS),
        'evaluation_ids':[row['evaluation_id'] for row in rows],'rows':rows,'identity':suite['identity'],
        'binding':binding,'online':not binding['scripted_agent'],
        'evaluation_mode':'end_to_end_candidate_generation','qualification':suite['qualification']}


def audit_report(suite_path,output):
    """Read-only recomputation; never invokes a role or repairs a ledger."""
    suite,_=load_suite(suite_path);output=Path(output).resolve();report=read_json(output)
    root=output.parent/'amazon-harness-online';binding=read_json(root/'binding.json')
    policy=read_json(root/'candidate-policy.json');validate_policy(policy)
    if (binding.get('suite_hash')!=suite['hash'] or binding.get('candidate_policy_hash')!=digest(policy)
            or binding.get('stage') not in {'development','confirmation'}
            or type(binding.get('scripted_agent')) is not bool):raise ContractError('Amazon Harness report binding differs')
    rows=[]
    for case in suite['cases']:
        if case['stage']!=binding['stage']:continue
        pair={g:verify_group(root/case['id']/g,case,patch,scripted=binding['scripted_agent'])
              for g,patch in [('baseline',suite['baseline_policy']),('candidate',policy)]}
        rows.append({'evaluation_id':case['id'],**pair})
    expected=summarize(suite,binding,rows)
    if report!=expected:raise ContractError('Amazon Harness report differs from committed evidence')
    return expected


def run_validation(suite_path,candidate,output,stage,*,agent_factory=None):
    if stage not in {'development','confirmation'}:raise ContractError('invalid Amazon Harness validation stage')
    suite,objects=load_suite(suite_path);policy=read_json(Path(candidate)/'policy.json');validate_policy(policy)
    output=Path(output).resolve();root=output.parent/'amazon-harness-online';root.mkdir(parents=True,exist_ok=True)
    binding={'suite_hash':suite['hash'],'candidate_policy_hash':digest(policy),'stage':stage,
        'scripted_agent':agent_factory is not None}
    with (root/'controller.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise ContractError('Amazon Harness comparison already running') from error
        if (root/'binding.json').exists() and read_json(root/'binding.json')!=binding:
            raise ContractError('Amazon Harness comparison binding changed')
        if (root/'candidate-policy.json').exists() and read_json(root/'candidate-policy.json')!=policy:
            raise ContractError('Amazon Harness archived policy changed')
        write_json(root/'binding.json',binding);write_json(root/'candidate-policy.json',policy)
        rows=[]
        for case in suite['cases']:
            if case['stage']!=stage:continue
            pair={}
            for group,patch in [('baseline',suite['baseline_policy']),('candidate',policy)]:
                folder=root/case['id']/group
                result=episode(policy_case(case,patch),folder,objects,None,
                    agent=agent_factory(case,group) if agent_factory else None)
                checked=verify_group(folder,case,patch,scripted=agent_factory is not None)
                if result!=checked:raise ContractError('Amazon Harness episode changed after completion')
                pair[group]=checked
            rows.append({'evaluation_id':case['id'],**pair})
        report=summarize(suite,binding,rows);write_json(output,report);return report


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='action',required=True)
    freeze=sub.add_parser('freeze');freeze.add_argument('specification');freeze.add_argument('directory')
    run=sub.add_parser('run');run.add_argument('--suite',required=True);run.add_argument('--candidate',required=True)
    run.add_argument('--output',required=True);run.add_argument('--stage',choices=['development','confirmation'],required=True)
    audit=sub.add_parser('audit');audit.add_argument('--suite',required=True);audit.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.action=='freeze':freeze_suite(args.specification,args.directory)
    elif args.action=='audit':audit_report(args.suite,args.output)
    else:run_validation(args.suite,args.candidate,args.output,args.stage)


if __name__=='__main__':main()
