"""Read-only diagnostic shadow runs on audited Amazon development conditions."""
import argparse
import fcntl
from pathlib import Path
from .core import Artifacts,ContractError,digest,file_hash,identifier,number,read_json,write_json
from .cpu_job import run_cpu_job
from .diagnostics import validate_measurement
from .diagnostic_inputs import CONTRACT,OBSERVATION_CONTRACT,CONTRACTS
from .process_identity import assert_stopped
from .store import Store
from .adapters.amazon_controls import load_config
from .adapters.amazon_report import audit_job


def freeze_suite(specification,directory):
    path=Path(specification).resolve();spec=read_json(path);study=(path.parent/spec['study']).resolve()
    config=load_config(study);root=Path(directory).resolve()
    if root.exists() and any(root.iterdir()):raise ContractError('shadow suite directory must be empty')
    contract=spec.get('tool_contract',CONTRACT)
    if contract not in CONTRACTS:raise ContractError('unsupported shadow tool contract')
    cases=[];ids=set();conditions=set()
    for case in spec['conditions']:
        identifier(case['id'])
        if case['id'] in ids:raise ContractError('duplicate shadow case ID')
        ids.add(case['id']);version=case['model_version'];seed=case['seed']
        if seed not in config['development_seeds']:raise ContractError('shadow inputs must be development conditions')
        condition=digest([config['model_versions'][version]['sha256'],seed,config['datasets']])
        if condition in conditions:raise ContractError('shadow conditions are aliases of the same source and seed')
        conditions.add(condition);runs=[];receipts={}
        for dataset in config['datasets']:
            result,manifest=audit_job(study,config,version,dataset,seed)
            job_id=f'{version}-{dataset}-{seed}'
            receipts[job_id]={'completed_hash':digest(result),'manifest_hash':digest(manifest)}
            runs.append({'dataset':dataset,'seed':seed,'valid':True,'metrics':result['metrics']})
        frozen={'id':case['id'],'condition_hash':condition,'source_receipts':receipts,
            'input':{'task_id':'tool-shadow','model_version':version,'protocol_hash':config['protocol_hash'],'runs':runs}}
        if contract==OBSERVATION_CONTRACT:
            from .shadow_observations import collect
            source=case.get('observation_source')
            if not isinstance(source,dict) or 'campaign' not in source:raise ContractError('v2 shadow requires an observation_source per condition')
            source={**source,'campaign':str((path.parent/source['campaign']).resolve())}
            frozen['input'],frozen['observation_source']=collect(source,study,config,version,runs)
        elif 'observation_source' in case:raise ContractError('observation_source requires v2 shadow contract')
        cases.append(frozen)
    if len(cases)<2:raise ContractError('shadow requires at least two distinct audited conditions')
    for key in ['budget','case_cost','repeat_tolerance','timeout']:number(spec[key],key,0)
    if spec['timeout']<=0:raise ContractError('positive shadow timeout required')
    if not spec.get('measurement_bounds'):raise ContractError('predeclared measurement bounds required')
    for name,bounds in spec['measurement_bounds'].items():
        if len(bounds)!=2 or number(bounds[0])>number(bounds[1]):raise ContractError('invalid measurement bounds')
    body={key:spec[key] for key in ['identity','execution','tool_files','budget','case_cost','repeat_tolerance','timeout','measurement_bounds']}
    body.update(schema='amazon-tool-shadow-v1',study=str(study),study_config_hash=digest(read_json(study/'study.json')),cases=cases,
        qualification='read-only development aggregates; repeatability and applicability, not decision utility')
    if contract==OBSERVATION_CONTRACT:body.update(tool_contract=contract)
    body['hash']=digest(body);write_json(root/'suite.json',body);return body


def audit_inputs(suite):
    if suite.get('schema')!='amazon-tool-shadow-v1' or suite['hash']!=digest({k:v for k,v in suite.items() if k!='hash'}):
        raise ContractError('shadow suite changed')
    study=Path(suite['study'])
    if digest(read_json(study/'study.json'))!=suite['study_config_hash']:raise ContractError('bound shadow study changed')
    config=load_config(study)
    contract=suite.get('tool_contract',CONTRACT)
    if contract not in CONTRACTS:raise ContractError('unsupported shadow tool contract')
    if len(suite['cases'])<2:raise ContractError('two shadow conditions required')
    identities=set();ids=set()
    for case in suite['cases']:
        identifier(case['id'])
        if case['id'] in ids:raise ContractError('duplicate shadow case ID')
        ids.add(case['id'])
        request=case['input'];version=request['model_version']
        rows=request['runs']
        if request['protocol_hash']!=config['protocol_hash'] or [row['dataset'] for row in rows]!=config['datasets'] or len({row['seed'] for row in rows})!=1:
            raise ContractError('shadow condition changed protocol or dataset coverage')
        condition=digest([config['model_versions'][version]['sha256'],rows[0]['seed'],config['datasets']])
        if condition!=case['condition_hash'] or condition in identities:raise ContractError('shadow condition identity changed or duplicated')
        identities.add(condition)
        for row in request['runs']:
            if row['seed'] not in config['development_seeds']:raise ContractError('non-development shadow input')
            result,manifest=audit_job(study,config,version,row['dataset'],row['seed'])
            ref=case['source_receipts'][f"{version}-{row['dataset']}-{row['seed']}"]
            if digest(result)!=ref['completed_hash'] or digest(manifest)!=ref['manifest_hash'] or result['metrics']!=row['metrics']:
                raise ContractError('shadow aggregate differs from audited job')
        if contract==OBSERVATION_CONTRACT:
            from .shadow_observations import collect
            source=case['observation_source']
            expected,binding=collect({key:source[key] for key in ['campaign','observation_ids','cutoff_seq']},study,config,version,rows)
            if expected!=request or binding!=source:raise ContractError('shadow observation prefix changed')
        elif 'observation_source' in case or 'observations' in request:
            raise ContractError('observation inputs require v2 shadow contract')


def run_validation(suite_path,candidate,output):
    suite=read_json(suite_path);audit_inputs(suite)
    root=Path(output).resolve().parent/'shadow-validation';root.mkdir(parents=True,exist_ok=True)
    with (root/'controller.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise ContractError('shadow validator already running') from error
        assert_stopped(root/'cases');artifacts=Artifacts(root/'objects');store=Store(root/'memory')
        try:
            source=artifacts.snapshot(candidate,suite['tool_files'])
            if 'diagnostic.py' not in source:raise ContractError('diagnostic.py required')
            worker=Path(__file__).with_name('diagnostic_worker.py')
            binding={'suite_hash':suite['hash'],'implementation_hash':digest(source),'worker_hash':file_hash(worker)}
            if (root/'binding.json').exists() and read_json(root/'binding.json')!=binding:raise ContractError('shadow validation identity changed')
            write_json(root/'binding.json',binding);store.configure_budget(suite['budget']);store.recover();rows=[]
            for case in suite['cases']:
                repetitions=[]
                for repetition in range(2):
                    key=case['id']+'-'+str(repetition)
                    prior=[e['payload'] for e in store.events() if e['kind']=='shadow_measurement_completed' and e['payload']['key']==key]
                    if prior:
                        row=prior[-1]
                        for ref in row.get('artifacts',{}).values():artifacts.verify(ref)
                        repetitions.append(row);continue
                    folder=root/'cases'/key;folder.mkdir(parents=True,exist_ok=True);write_json(folder/'input.json',case['input'])
                    files={'worker.py':{'path':str(worker),'sha256':binding['worker_hash']},
                        'request.json':{'path':str(folder/'input.json'),'sha256':file_hash(folder/'input.json')}}
                    for name,ref in source.items():files['tool/'+name]={'path':str(artifacts.verify(ref)),'sha256':ref['sha256']}
                    attempt=len(list(folder.glob('job-*')))+1;task=key+'-'+str(attempt);store.reserve(task,suite['case_cost'])
                    row={'key':key,'passed':False}
                    try:
                        execution=suite['execution'];job=folder/('job-'+str(attempt))
                        receipt=run_cpu_job(job,execution['recipe'],files,'worker.py',[],python=execution['python'],uid_registry=execution['uid_registry'],artifacts=artifacts,timeout=suite['timeout'])
                        row['artifacts']=receipt['collection']['artifacts'];row['wall_seconds']=receipt['wall_seconds']
                        if receipt['process']['status']!='success':raise ContractError('shadow process failed')
                        values=validate_measurement(read_json(job/'results/measurement.json'))['measurements']
                        if set(values)!=set(suite['measurement_bounds']):raise ContractError('shadow measurement names changed')
                        if not all(bounds[0]<=values[name]<=bounds[1] for name,bounds in suite['measurement_bounds'].items()):raise ContractError('measurement outside preregistered bounds')
                        row.update(passed=True,measurements=values)
                    except (ContractError,OSError,ValueError,KeyError,TypeError) as error:row['error']=str(error)
                    finally:store.settle(task,suite['case_cost'],'accepted' if row['passed'] else 'rejected')
                    store.event('shadow_measurement_completed',row);repetitions.append(row)
                passed=all(row['passed'] for row in repetitions)
                if passed:passed=all(abs(repetitions[0]['measurements'][name]-repetitions[1]['measurements'][name])<=suite['repeat_tolerance'] for name in suite['measurement_bounds'])
                rows.append({'evaluation_id':case['id'],'passed':passed,'repetitions':repetitions,'condition_hash':case['condition_hash']})
            accuracy=sum(row['passed'] for row in rows)/len(rows)
            report={'passed':all(row['passed'] for row in rows),'gain':accuracy,'regression':1-accuracy,
                'total_cost':store.budget()['charged_or_reserved'],'evaluation_ids':[row['evaluation_id'] for row in rows],
                'identity':suite['identity'],'comparison_groups':['candidate_tool'],'online':False,'rows':rows,
                'binding':binding,'integrity':store.verify(),'qualification':suite['qualification']}
            write_json(output,report);return report
        finally:store.close()


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='action',required=True)
    freeze=sub.add_parser('freeze');freeze.add_argument('specification');freeze.add_argument('directory')
    run=sub.add_parser('run');run.add_argument('--suite',required=True);run.add_argument('--candidate',required=True);run.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.action=='freeze':freeze_suite(args.specification,args.directory)
    else:run_validation(args.suite,args.candidate,args.output)


if __name__=='__main__':main()
