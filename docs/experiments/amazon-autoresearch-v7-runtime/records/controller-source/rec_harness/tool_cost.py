"""Audit three-group utility receipts and include declared development costs."""
import argparse
import statistics
from pathlib import Path
from .core import Artifacts,ContractError,digest,read_json,write_json
from .cost_audit import audit_cost,verify_cost_prefix
from .store import Store
from .tool_exposure import audit_exposure
from .tool_online import GROUPS
from .online_validation import load_suite


def bound_suite(report,suite_path):
    if suite_path is None:raise ContractError('frozen utility suite required')
    path=Path(suite_path).resolve();suite=read_json(path)
    if suite.get('schema')!='online-tool-suite-v1' or suite['hash']!=digest({k:v for k,v in suite.items() if k!='hash'}):
        raise ContractError('cost audit suite changed')
    episodes,_=load_suite(path.parent/'episodes/suite.json')
    if episodes['hash']!=suite['episode_suite_hash'] or report['binding']['suite_hash']!=suite['hash'] or report['identity']!=suite['identity']:
        raise ContractError('utility report differs from frozen suite')
    stage=report['binding']['stage']
    if stage not in {'utility','confirmation'}:raise ContractError('invalid utility report stage')
    cases={c['id']:c for c in episodes['cases'] if c['stage']==('development' if stage=='utility' else 'confirmation')}
    if report['evaluation_ids']!=list(cases) or [r['evaluation_id'] for r in report['rows']]!=list(cases):
        raise ContractError('utility report omitted, duplicated or changed frozen cases')
    return suite,episodes,cases


def check_episode_binding(root,config,case,policy,suite,group,report):
    expected=case['config'];manifest=read_json(root/'research_archive/manifest.json')
    if manifest['root_workspace']!={name:case['inputs'][name] for name in expected['model_files']}:
        raise ContractError('utility initial model differs from frozen suite')
    for key,value in expected.items():
        if key in {'source','protocol','execution','evaluator_files'}:continue
        if config.get(key)!=value:raise ContractError('utility configuration changed: '+key)
    if config.get('research_policy')!=policy:raise ContractError('utility policy changed')
    if {k:v for k,v in config['protocol'].items() if k!='evaluator_snapshot'}!=expected['protocol']:
        raise ContractError('utility protocol changed')
    if config['protocol']['evaluator_snapshot']!={name:case['inputs'][name] for name in expected.get('evaluator_files',[])}:
        raise ContractError('utility evaluator changed')
    if {k:v for k,v in config['execution'].items() if k not in {'host_scorer_hash','worker_hash'}}!=expected['execution']:
        raise ContractError('utility execution changed')
    tool=config.get('diagnostic_tool')
    if group!='no_tool':
        if not tool:raise ContractError('missing utility diagnostic tool')
        wanted=digest(suite['existing_tool']) if group=='existing_tool' else report['binding']['candidate_hash']
        if tool['implementation_hash']!=wanted:raise ContractError('utility tool source changed')
        if tool['worker']['sha256']!=suite['worker_hash']:raise ContractError('utility diagnostic worker changed')
        for key,value in suite['tool_runtime'].items():
            if tool.get(key)!=value:raise ContractError('utility diagnostic runtime changed')


def audit_utility(report,suite_path=None):
    if report.get('online') is not True or report.get('comparison_groups')!=list(GROUPS):
        raise ContractError('real three-group utility report required')
    suite,episodes,cases=bound_suite(report,suite_path)
    receipts=[];used=set();deltas=[];total=0
    for row in report['rows']:
        scores={};protocols=set();directions=set()
        for group in GROUPS:
            recorded=row[group];memory=Path(recorded['graph_directory']).resolve()
            if memory in used:raise ContractError('comparison groups share a cost ledger')
            used.add(memory);root=memory.parent;config=read_json(root/'campaign.json')
            if config.get('agent_backend',{}).get('kind')!='jailed_codex':raise ContractError('utility was not configured for isolated Codex')
            check_episode_binding(root,config,cases[row['evaluation_id']],episodes['baseline_policy'],suite,group,report)
            exposure=audit_exposure(root);store=Store(memory)
            try:
                cost=audit_cost(store,config['budget'])
                if cost['pending']:raise ContractError('utility has unfinished reservations')
                if cost['integrity']['head']!=recorded['event_head'] or cost['charged_or_reserved']!=recorded['charged_cost']:
                    raise ContractError('utility receipt differs from committed cost or head')
                stops=[e['payload']['state'] for e in store.events() if e['kind']=='controller_stopped']
                if not stops or stops[-1]!=read_json(root/'state.json'):raise ContractError('terminal state differs from committed stop')
                state=stops[-1]
                if state['stop_reason'] not in {'budget','max_rounds'} or not state['baseline']['valid']:
                    raise ContractError('utility lacks valid terminal incumbent')
                if state['baseline']['score']!=recorded['score'] or state['incumbent']!=recorded['incumbent']:
                    raise ContractError('utility score differs from terminal incumbent')
                objects=Artifacts(root/'artifacts/objects')
                for ref in state['workspace'].values():objects.verify(ref)
                real_calls=0
                for event in store.events():
                    if event['kind']!='agent_completed' or event['payload']['status']!='success':continue
                    archive=event['payload']['archive']
                    if not all(name in archive for name in ['resources.json','request.json','response.json']):
                        raise ContractError('completed call lacks actual transport receipts')
                    resource=read_json(objects.verify(archive['resources.json']))
                    request=read_json(objects.verify(archive['request.json']))
                    objects.verify(archive['response.json'])
                    isolation=resource.get('isolation',{})
                    if resource.get('status')!='success' or isolation.get('kind')!='chroot-unprivileged' or isolation!=request.get('isolation'):
                        raise ContractError('unverified isolated role transport')
                    argv=request.get('argv',[]);model=config.get('codex',{}).get('model')
                    if not model or '--model' not in argv or argv[argv.index('--model')+1]!=model:
                        raise ContractError('role transport model differs from frozen model')
                    real_calls+=1
                if real_calls<1:raise ContractError('no completed real Codex decisions')
                scores[group]=recorded['score'];protocols.add(config['protocol_hash']);directions.add(config['protocol']['direction'])
                tool=config.get('diagnostic_tool')
                if group=='no_tool':
                    if tool or exposure['exposures']:raise ContractError('reference unexpectedly used a tool')
                else:
                    if not tool or not config.get('validation_only'):raise ContractError('missing experimental tool binding')
                    if group=='candidate_tool' and tool['implementation_hash']!=report['binding']['candidate_hash']:
                        raise ContractError('candidate utility used different source')
                    if report['passed'] and not any(e['measurement_status']=='success' and e['transport_completed'] for e in exposure['exposures']):
                        raise ContractError('no successful measurement reached a completed role')
                receipts.append({'evaluation_id':row['evaluation_id'],'group':group,'cost':cost,'exposure':exposure})
                total+=cost['charged_or_reserved']
            finally:store.close()
        if len(protocols)!=1 or len(directions)!=1:raise ContractError('utility protocols differ')
        sign=1 if directions.pop()=='maximize' else -1
        calculated={group:sign*(scores['candidate_tool']-scores[group]) for group in GROUPS[:2]}
        if calculated!=row['deltas']:raise ContractError('reported comparison delta changed')
        deltas.append(calculated)
    if not deltas:raise ContractError('empty utility report')
    gain=min(statistics.mean(row[group] for row in deltas) for group in GROUPS[:2])
    regression=max(0,max(-v for row in deltas for v in row.values()))
    if (gain,regression,total)!=(report['gain'],report['regression'],report['total_cost']):
        raise ContractError('utility aggregate differs from audited episodes')
    return receipts,used


def run_validation(utility,candidate,output,development_ledgers=(),evolution_context=None,*,suite_path=None):
    report=read_json(utility);receipts,used=audit_utility(report,suite_path)
    # Check the actual candidate supplied by the evolution runner, not merely
    # the candidate name printed by a validator.
    first=next(row for row in report['rows'])
    config=read_json(Path(first['candidate_tool']['graph_directory']).parent/'campaign.json')
    artifacts=Artifacts(Path(output).resolve().parent/'cost-audit-objects')
    snapshot=artifacts.snapshot(candidate,list(config['diagnostic_tool']['implementation']))
    if digest(snapshot)!=report['binding']['candidate_hash']:raise ContractError('cost audit candidate changed')
    extra=[]
    for directory in development_ledgers:
        memory=Path(directory).resolve()
        if memory in used:raise ContractError('cost ledger counted twice')
        used.add(memory)
        if not (memory/'memory.sqlite').is_file():raise ContractError('development ledger missing')
        store=Store(memory)
        try:
            cost=audit_cost(store,store.budget()['limit'])
            if cost['pending']:raise ContractError('development ledger still running')
            extra.append({'directory':str(memory),'cost':cost})
        finally:store.close()
    controller_cost=0;prefix=None
    if evolution_context is not None:
        prefix=read_json(evolution_context);memory=Path(prefix['memory']).resolve()
        if memory in used:raise ContractError('controller cost ledger counted twice')
        if not (memory/'memory.sqlite').is_file():raise ContractError('controller ledger missing')
        store=Store(memory)
        try:controller_cost=verify_cost_prefix(store,prefix)
        finally:store.close()
    result={key:report[key] for key in ['passed','gain','regression','evaluation_ids','identity','comparison_groups','online']}
    result.update(controller_cost_prefix=prefix,total_cost=report['total_cost']+controller_cost+sum(row['cost']['charged_or_reserved'] for row in extra),
        utility_report_hash=digest(report),binding=report['binding'],utility_receipts=receipts,development_costs=extra,
        qualification='Event-reconciled BCU costs for all utility groups and explicitly supplied development ledgers; no inference of mechanism or unrecorded costs')
    write_json(output,result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--utility',required=True);parser.add_argument('--suite',required=True);parser.add_argument('--candidate',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--evolution-context');parser.add_argument('--development-ledger',action='append',default=[])
    args=parser.parse_args();run_validation(args.utility,args.candidate,args.output,args.development_ledger,args.evolution_context,suite_path=args.suite)


if __name__=='__main__':main()
