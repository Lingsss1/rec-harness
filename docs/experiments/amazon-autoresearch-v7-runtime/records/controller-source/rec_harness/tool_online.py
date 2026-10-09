"""Host-scored, three-group online diagnostic utility experiments.

Unpromoted tools are confined to validation-only campaigns. Each group receives
an independent graph, the same model, evaluator, LLM, policy and total budget.
"""
import argparse
import copy
import fcntl
import statistics
from pathlib import Path
from .core import Artifacts,ContractError,digest,file_hash,number,read_json,write_json
from .online_validation import freeze_suite as freeze_episodes,load_suite,episode

GROUPS=('no_tool','existing_tool','candidate_tool')


def freeze_suite(specification,directory):
    path=Path(specification).resolve();spec=read_json(path);root=Path(directory).resolve()
    if root.exists() and any(root.iterdir()):raise ContractError('tool utility suite directory must be empty')
    runtime=spec['tool_runtime']
    for key in ['cost','timeout']:number(runtime[key],key,0)
    if runtime['timeout']<=0:raise ContractError('positive tool timeout required')
    for key in ['recipe','python','uid_registry']:
        if key not in runtime['execution']:raise ContractError('missing tool execution '+key)
    files=spec['tool_files']
    if 'diagnostic.py' not in files:raise ContractError('diagnostic.py entry required')
    episodes=freeze_episodes(path,root/'episodes')
    objects=Artifacts(root/'objects');existing=objects.snapshot(path.parent/spec['existing_tool'],files)
    body={'schema':'online-tool-suite-v1','episode_suite_hash':episodes['hash'],
        'existing_tool':existing,'tool_files':files,'tool_runtime':runtime,
        'worker_hash':file_hash(Path(__file__).with_name('diagnostic_worker.py')),
        'identity':{**episodes['identity'],'tools':digest({'existing':existing,'runtime':runtime,'files':files})},
        'qualification':'independent synthetic seed conditions, one task family; experimental tools are not promoted'}
    body['hash']=digest(body);write_json(root/'suite.json',body);return body


def run_validation(suite_path,candidate,output,stage,*,agent_factory=None):
    if stage not in {'utility','confirmation'}:raise ContractError('tool stage must be utility or confirmation')
    path=Path(suite_path).resolve();suite=read_json(path)
    if suite.get('schema')!='online-tool-suite-v1' or suite['hash']!=digest({k:v for k,v in suite.items() if k!='hash'}):
        raise ContractError('tool utility suite changed')
    episodes,inputs=load_suite(path.parent/'episodes/suite.json')
    if episodes['hash']!=suite['episode_suite_hash']:raise ContractError('bound episodes changed')
    if file_hash(Path(__file__).with_name('diagnostic_worker.py'))!=suite['worker_hash']:raise ContractError('diagnostic worker changed')
    objects=Artifacts(path.parent/'objects')
    for ref in suite['existing_tool'].values():objects.verify(ref)
    root=Path(output).resolve().parent/'tool-online-episodes';root.mkdir(parents=True,exist_ok=True)
    with (root/'controller.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise ContractError('tool utility validation already active') from error
        artifacts=Artifacts(root/'objects');snapshot=artifacts.snapshot(candidate,suite['tool_files'])
        binding={'suite_hash':suite['hash'],'candidate_hash':digest(snapshot),'stage':stage}
        if (root/'binding.json').exists() and read_json(root/'binding.json')!=binding:raise ContractError('tool utility inputs changed')
        write_json(root/'binding.json',binding)
        candidate_source=root/'candidate-inputs';artifacts.restore(snapshot,candidate_source)
        existing_source=root/'existing-inputs';objects.restore(suite['existing_tool'],existing_source)
        rows=[]
        for case in episodes['cases']:
            if case['stage']!=('development' if stage=='utility' else 'confirmation'):continue
            results={}
            for group in GROUPS:
                tool=None
                if group!='no_tool':
                    tool={**copy.deepcopy(suite['tool_runtime']),'files':suite['tool_files'],
                        'source':str(candidate_source if group=='candidate_tool' else existing_source)}
                agent=agent_factory(case,group) if agent_factory else None
                results[group]=episode(case,episodes['baseline_policy'],root/case['id']/group,inputs,
                    agent=agent,validation_tool=tool)
            if len({row['protocol_hash'] for row in results.values()})!=1:raise ContractError('comparison protocols differ')
            sign=1 if case['config']['protocol']['direction']=='maximize' else -1
            deltas={group:sign*(results['candidate_tool']['score']-results[group]['score']) for group in GROUPS[:2]}
            rows.append({'evaluation_id':case['id'],'deltas':deltas,**results})
        if not rows:raise ContractError('no independent tool utility cases')
        # Require improvement against both references; never cherry-pick the
        # weaker reference. Tool failures cannot count as positive evidence.
        valid=all(row[group]['diagnostic_calls']>0 and row[group]['diagnostic_failures']==0 for row in rows for group in GROUPS[1:])
        gains={group:statistics.mean(row['deltas'][group] for row in rows) for group in GROUPS[:2]}
        report={'passed':valid,'gain':min(gains.values()),'gain_by_reference':gains,
            'regression':max(0,max(-delta for row in rows for delta in row['deltas'].values())),
            'total_cost':sum(row[group]['charged_cost'] for row in rows for group in GROUPS),
            'evaluation_ids':[row['evaluation_id'] for row in rows],'identity':suite['identity'],
            'comparison_groups':list(GROUPS),'online':agent_factory is None,'evaluation_mode':'end_to_end_candidate_generation','rows':rows,
            'binding':binding,'qualification':suite['qualification']}
        write_json(output,report);return report


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='action',required=True)
    freeze=sub.add_parser('freeze');freeze.add_argument('specification');freeze.add_argument('directory')
    run=sub.add_parser('run');run.add_argument('--suite',required=True);run.add_argument('--candidate',required=True)
    run.add_argument('--output',required=True);run.add_argument('--stage',choices=['utility','confirmation'],required=True)
    args=parser.parse_args()
    if args.action=='freeze':freeze_suite(args.specification,args.directory)
    else:run_validation(args.suite,args.candidate,args.output,args.stage)


if __name__=='__main__':main()
