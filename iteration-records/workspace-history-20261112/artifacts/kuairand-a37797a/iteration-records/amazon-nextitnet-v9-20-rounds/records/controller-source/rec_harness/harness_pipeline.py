"""Frozen prefix, development and confirmation stages for Harness evolution."""
import argparse
from pathlib import Path
from .core import Artifacts,ContractError,digest,file_hash,read_json,write_json
from .cost_audit import audit_cost,verify_cost_prefix
from .evolution import STAGES
from .process_identity import assert_stopped
from .store import Store
from .tool_pipeline import accepted_report


def online_module(path):
    schema=read_json(path).get('schema')
    if schema=='amazon-online-harness-suite-v1':return 'amazon_harness_online'
    if schema=='online-harness-suite-v1':return 'online_validation'
    raise ContractError('unsupported Harness online suite')


def load_online(path):
    if online_module(path)=='amazon_harness_online':
        from .amazon_harness_online import load_suite
    else:
        from .online_validation import load_suite
    return load_suite(path)


def prefix_suite(path):
    body=read_json(path)
    if body.get('schema')!='harness-prefix-suite-v1' or body.get('hash')!=digest({k:v for k,v in body.items() if k!='hash'}):
        raise ContractError('prefix suite changed')
    objects=Artifacts(Path(path).parent/'objects')
    if not body['cases']:raise ContractError('empty prefix suite')
    if body.get('adapter')=='amazon-nextitnet-v1':
        from .adapters.amazon_execution import code_identity
        if body.get('code_hashes')!=code_identity():raise ContractError('Amazon prefix implementation changed')
    for case in body['cases']:
        if digest(case['context'])!=case['input_hash']:raise ContractError('prefix input changed')
        for ref in case['source'].values():objects.verify(ref)
    return body


def check_compatibility(prefix,online):
    if prefix['identity']!=online['identity'] or prefix['baseline_policy']!=online['baseline_policy']:
        raise ContractError('prefix and online reference identities differ')
    if online['schema']=='amazon-online-harness-suite-v1':
        from .amazon_prefix import conditions
        if prefix.get('adapter')!='amazon-nextitnet-v1':raise ContractError('Amazon Harness requires Amazon source prefixes')
        models={c['controller']['agent_backend']['model'] for c in online['cases']}
        if models!={prefix['agent_backend']['model']}:raise ContractError('prefix and online base models differ')
        used=set(prefix['source_conditions'])
        if not used:raise ContractError('Amazon prefix lacks source conditions')
        ids={prefix['trace_source']['task_id']}|{c['id'] for c in prefix['cases']}
        for case in online['cases']:
            if case['id'] in ids or used.intersection(conditions(case['study'])):
                raise ContractError('Amazon online conditions overlap prefix source')
        return
    if prefix.get('adapter')=='amazon-nextitnet-v1':raise ContractError('Amazon prefix requires Amazon online comparison')
    models={c['config']['codex']['model'] for c in online['cases']}
    if models!={prefix['agent_backend']['model']}:raise ContractError('prefix and online base models differ')
    used=set();ids={prefix['trace_source']['task_id']}|{c['id'] for c in prefix['cases']}
    for case in prefix['cases']:
        seeds=case['context'].get('protocol',{}).get('seeds')
        if not seeds:raise ContractError('prefix lacks source seed conditions')
        for seed in seeds:used.update([seed,seed+10000])
    for case in online['cases']:
        if case['id'] in ids or case['config']['task_id']==prefix['trace_source']['task_id']:
            raise ContractError('online tasks overlap optimizer source')
        for seed in case['config']['protocol']['seeds']:
            if {seed,seed+10000}&used:raise ContractError('online data overlap prefix source')


def freeze_suite(specification,directory):
    path=Path(specification).resolve();spec=read_json(path);root=Path(directory).resolve()
    if root.exists() and any(root.iterdir()):raise ContractError('pipeline directory must be empty')
    paths={name:(path.parent/spec[name]).resolve() for name in ['prefix','online']}
    prefix=prefix_suite(paths['prefix']);online,_=load_online(paths['online']);check_compatibility(prefix,online)
    body={'schema':'harness-validation-pipeline-v1','identity':online['identity'],
        'suites':{name:{'path':str(value),'sha256':file_hash(value)} for name,value in paths.items()}}
    body['hash']=digest(body);write_json(root/'suite.json',body);return body


def load_suite(path):
    suite=read_json(path)
    if suite.get('schema')!='harness-validation-pipeline-v1' or suite.get('hash')!=digest({k:v for k,v in suite.items() if k!='hash'}):
        raise ContractError('Harness pipeline changed')
    for ref in suite['suites'].values():
        if file_hash(ref['path'])!=ref['sha256']:raise ContractError('bound Harness stage suite changed')
    prefix=prefix_suite(suite['suites']['prefix']['path']);online,_=load_online(suite['suites']['online']['path'])
    check_compatibility(prefix,online)
    if suite['identity']!=online['identity']:raise ContractError('Harness pipeline identity changed')
    return suite


def costs(directory,stage,cost_context,*,cost_weights=None):
    """Audit actual child ledgers, retaining abandoned attempts without double counting."""
    ledgers=[]
    for previous in STAGES['harness'][:STAGES['harness'].index(stage)+1]:
        for path in sorted((directory/previous).rglob('memory.sqlite')):ledgers.append(path.parent)
    abandoned=directory.parent/'validation-attempts'
    if abandoned.exists():
        assert_stopped(abandoned)
        for path in sorted(abandoned.rglob('memory.sqlite')):
            ledger=Store(path.parent)
            try:ledger.recover()
            finally:ledger.close()
            ledgers.append(path.parent)
    seen=set();rows=[]
    for path in ledgers:
        resolved=str(path.resolve())
        if resolved in seen:raise ContractError('duplicate Harness child cost ledger')
        seen.add(resolved);store=Store(path)
        try:
            if cost_weights is None:axis,limit='development_bcu',None
            else:
                from .amazon_tool_cost import ledger_axis
                axis,limit=ledger_axis(path)
            cost=audit_cost(store,store.budget()['limit'] if limit is None else limit)
            if cost['pending']:raise ContractError('Harness validation child still pending')
            row={'directory':resolved,'cost':cost}
            if cost_weights is not None:
                if axis!='development_bcu' and axis not in cost_weights:raise ContractError('Harness ledger lacks frozen cost weight')
                weight=1 if axis=='development_bcu' else cost_weights[axis]
                row.update(axis=axis,weight=weight,weighted_cost=weight*cost['charged_or_reserved'])
            rows.append(row)
        finally:store.close()
    if not rows:raise ContractError('Harness stage produced no auditable child costs')
    prefix=read_json(cost_context)
    if str(Path(prefix['memory']).resolve()) in seen:raise ContractError('controller cost counted twice')
    store=Store(prefix['memory'])
    try:parent=verify_cost_prefix(store,prefix)
    finally:store.close()
    return {'children':rows,'controller_cost_prefix':prefix,
        'total_cost':parent+sum(row.get('weighted_cost',row['cost']['charged_or_reserved']) for row in rows)}


def run_validation(suite_path,stage,candidate,output,cost_context):
    if stage not in STAGES['harness']:raise ContractError('unknown Harness stage')
    suite=load_suite(suite_path);output=Path(output).resolve()
    if output.parent.name!=stage:raise ContractError('Harness pipeline requires standard stage directory')
    directory=output.parent.parent
    implementation=Artifacts(output.parent/'pipeline-objects').snapshot(candidate,['policy.json'])
    for previous in STAGES['harness'][:STAGES['harness'].index(stage)]:
        old=accepted_report(directory,previous,digest(implementation))
        if old.get('pipeline_hash')!=suite['hash'] or old['identity']!=suite['identity']:
            raise ContractError('Harness predecessor belongs to another pipeline')
    raw=output.parent/'raw-result.json'
    if stage=='prefix_audit':
        from .harness_prefix import run_validation as run
        report=run(suite['suites']['prefix']['path'],candidate,raw)
    else:
        if online_module(suite['suites']['online']['path'])=='amazon_harness_online':
            from .amazon_harness_online import run_validation as run
        else:
            from .online_validation import run_validation as run
        report=run(suite['suites']['online']['path'],candidate,raw,stage)
    if report['identity']!=suite['identity']:raise ContractError('Harness stage identity differs')
    weights=None
    if online_module(suite['suites']['online']['path'])=='amazon_harness_online':
        from .amazon_harness_online import audit_report
        online,_=load_online(suite['suites']['online']['path']);weights=online['cost_weights']
        for previous in STAGES['harness'][1:STAGES['harness'].index(stage)+1]:
            audited=audit_report(suite['suites']['online']['path'],directory/previous/'raw-result.json')
            if not audited['online']:raise ContractError('Harness pipeline requires real Amazon transports')
    audit=costs(directory,stage,cost_context,cost_weights=weights)
    report={**report,'stage_cost':report['total_cost'],'total_cost':audit['total_cost'],
        'cost_audit':audit,'pipeline_hash':suite['hash'],'raw_report_sha256':file_hash(raw)}
    write_json(output,report);return report


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='action',required=True)
    freeze=sub.add_parser('freeze');freeze.add_argument('specification');freeze.add_argument('directory')
    run=sub.add_parser('run');run.add_argument('--suite',required=True);run.add_argument('--stage',choices=STAGES['harness'],required=True)
    run.add_argument('--candidate',required=True);run.add_argument('--output',required=True);run.add_argument('--cost-context',required=True)
    args=parser.parse_args()
    if args.action=='freeze':freeze_suite(args.specification,args.directory)
    else:run_validation(args.suite,args.stage,args.candidate,args.output,args.cost_context)


if __name__=='__main__':main()
