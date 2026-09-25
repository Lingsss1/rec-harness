"""Frozen integration of the six ToolEvolver validation stages."""
import argparse
from pathlib import Path
from .core import ContractError,digest,file_hash,read_json,write_json
from .evolution import STAGES
from .tool_validation import validate_suite


def freeze_suite(specification,directory):
    path=Path(specification).resolve();spec=read_json(path);root=Path(directory).resolve()
    if root.exists() and any(root.iterdir()):raise ContractError('pipeline directory must be empty')
    suites={};identity=None;files=None
    for kind in ['reference','shadow','online']:
        source=(path.parent/spec[kind]).resolve();body=read_json(source)
        if identity is None:identity=body['identity'];files=body['tool_files']
        if body['identity']!=identity or body['tool_files']!=files:
            raise ContractError('pipeline stages must share frozen identity and tool files')
        suites[kind]={'path':str(source),'sha256':file_hash(source)}
    for stage in ['correctness','synthetic','robustness']:validate_suite(read_json(suites['reference']['path']),stage)
    shadow=read_json(suites['shadow']['path']);online=read_json(suites['online']['path'])
    if shadow.get('schema')!='amazon-tool-shadow-v1' or online.get('schema')!='online-tool-suite-v1':
        raise ContractError('pipeline requires frozen shadow and online suites')
    body={'schema':'tool-validation-pipeline-v1','suites':suites,'identity':identity,'tool_files':files}
    body['hash']=digest(body);write_json(root/'suite.json',body);return body


def load_suite(path):
    suite=read_json(path)
    if suite.get('schema')!='tool-validation-pipeline-v1' or suite['hash']!=digest({k:v for k,v in suite.items() if k!='hash'}):
        raise ContractError('tool pipeline changed')
    for ref in suite['suites'].values():
        if file_hash(ref['path'])!=ref['sha256']:raise ContractError('bound pipeline stage suite changed')
    return suite


def accepted_report(directory,stage,implementation_hash):
    accepted=read_json(directory/stage/'accepted.json')
    if accepted['report_hash']!=digest(accepted['report']) or accepted['implementation_hash']!=implementation_hash:
        raise ContractError('pipeline predecessor report or implementation changed')
    return accepted['report']


def run_validation(suite_path,stage,candidate,output,cost_context=None):
    if stage not in STAGES['tool']:raise ContractError('unknown tool pipeline stage')
    suite=load_suite(suite_path);refs=suite['suites'];output=Path(output).resolve()
    if stage in {'correctness','synthetic','robustness'}:
        from .tool_validation import run_validation as run
        report=run(refs['reference']['path'],candidate,output,stage)
    elif stage=='shadow':
        from .tool_shadow import run_validation as run
        report=run(refs['shadow']['path'],candidate,output)
    elif stage=='utility':
        from .tool_online import run_validation as run
        report=run(refs['online']['path'],candidate,output,'utility')
    else:
        from .core import Artifacts
        from .tool_cost import run_validation as run
        if cost_context is None:raise ContractError('cost stage requires controller cost context')
        if output.parent.name!='cost_regression':raise ContractError('cost stage must use standard evolution stage directory')
        directory=output.parent.parent
        snapshot=Artifacts(output.parent/'pipeline-objects').snapshot(candidate,suite['tool_files'])
        implementation_hash=digest(snapshot);ledgers=[]
        for previous in STAGES['tool'][:-1]:
            report=accepted_report(directory,previous,implementation_hash)
            if report['identity']!=suite['identity']:raise ContractError('pipeline predecessor identity changed')
            if previous in {'correctness','synthetic','robustness'}:
                ledgers.append(directory/previous/'reference-validation/memory')
            elif previous=='shadow':ledgers.append(directory/previous/'shadow-validation/memory')
        utility=directory/'utility/result.json'
        if read_json(utility)!=accepted_report(directory,'utility',implementation_hash):
            raise ContractError('utility result differs from accepted report')
        # Retain costs of interrupted validation attempts, including incomplete
        # paired episodes. These are separate ledgers, never replacement costs.
        attempts=directory.parent/'validation-attempts'
        if attempts.exists():
            from .process_identity import assert_stopped
            from .store import Store
            assert_stopped(attempts)
            for path in sorted(attempts.rglob('memory.sqlite')):
                store=Store(path.parent)
                try:store.recover()  # Preserve reservations when actual cost is unknown.
                finally:store.close()
                ledgers.append(path.parent)
        report=run(utility,candidate,output,ledgers,cost_context,suite_path=refs['online']['path'])
    if report['identity']!=suite['identity']:raise ContractError('pipeline result identity differs')
    return report


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='action',required=True)
    freeze=sub.add_parser('freeze');freeze.add_argument('specification');freeze.add_argument('directory')
    run=sub.add_parser('run');run.add_argument('--suite',required=True);run.add_argument('--stage',required=True,choices=STAGES['tool'])
    run.add_argument('--candidate',required=True);run.add_argument('--output',required=True);run.add_argument('--cost-context')
    args=parser.parse_args()
    if args.action=='freeze':freeze_suite(args.specification,args.directory)
    else:run_validation(args.suite,args.stage,args.candidate,args.output,args.cost_context)


if __name__=='__main__':main()
