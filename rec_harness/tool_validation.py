"""Host-reference diagnostic validation. Reference values never enter candidate jails."""
import argparse
import fcntl
from pathlib import Path
from .core import Artifacts,ContractError,digest,file_hash,identifier,number,public_only,read_json,write_json
from .cpu_job import run_cpu_job
from .diagnostics import validate_measurement
from .process_identity import assert_stopped
from .store import Store
from .validation_case import prepare as prepare_case,settle_once


def validate_suite(suite,stage):
    if stage not in {'correctness','synthetic','robustness'}:raise ContractError('reference validator is offline only')
    cases=[c for c in suite['cases'] if c['stage']==stage]
    if not cases:raise ContractError('no reference cases for stage')
    seen=set()
    for case in suite['cases']:
        identifier(case['id'])
        if case['id'] in seen:raise ContractError('duplicate reference case ID')
        seen.add(case['id']);public_only(case['input'])
        if not isinstance(case['expected'],dict) or not case['expected']:raise ContractError('host reference measurements required')
        for value in case['expected'].values():number(value,'reference measurement')
        number(case['absolute_tolerance'],'absolute tolerance',0)
    number(suite['case_cost'],'case cost',0);number(suite['budget'],'validation budget',0)
    number(suite['timeout'],'timeout',0)
    if suite['timeout']<=0:raise ContractError('positive deadline required')
    return cases


def run_validation(suite_path,candidate,output,stage):
    suite=read_json(suite_path);cases=validate_suite(suite,stage)
    root=Path(output).resolve().parent/'reference-validation';root.mkdir(parents=True,exist_ok=True)
    with (root/'controller.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise ContractError('reference validator already running') from error
        assert_stopped(root/'cases');artifacts=Artifacts(root/'objects');store=Store(root/'memory')
        try:
            source=artifacts.snapshot(candidate,suite['tool_files'])
            if 'diagnostic.py' not in source:raise ContractError('diagnostic.py required')
            worker=Path(__file__).with_name('diagnostic_worker.py')
            binding={'suite_hash':digest(suite),'implementation_hash':digest(source),'worker_hash':file_hash(worker),'stage':stage}
            if (root/'binding.json').exists() and read_json(root/'binding.json')!=binding:raise ContractError('frozen reference validation changed')
            write_json(root/'binding.json',binding);store.configure_budget(suite['budget']);store.recover()
            rows=[]
            for case in cases:
                prior=[e['payload'] for e in store.events() if e['kind']=='reference_case_completed' and e['payload']['case_id']==case['id']]
                if prior:
                    row=prior[-1]
                    for ref in row.get('artifacts',{}).values():artifacts.verify(ref)
                    rows.append(row);continue
                folder=root/'cases'/case['id'];folder.mkdir(parents=True,exist_ok=True)
                write_json(folder/'input.json',case['input'])
                files={'request.json':{'path':str(folder/'input.json'),'sha256':file_hash(folder/'input.json')},
                    'worker.py':{'path':str(worker),'sha256':binding['worker_hash']}}
                for name,ref in source.items():files['tool/'+name]={'path':str(artifacts.verify(ref)),'sha256':ref['sha256']}
                task,job,receipt=prepare_case(store,folder,case['id'],files,suite,artifacts)
                row={'case_id':case['id'],'passed':False,'input_hash':digest(case['input'])}
                try:
                    execution=suite['execution']
                    if receipt is None:
                        receipt=run_cpu_job(job,execution['recipe'],files,'worker.py',[],python=execution['python'],
                            uid_registry=execution['uid_registry'],artifacts=artifacts,timeout=suite['timeout'])
                    row['artifacts']=receipt['collection']['artifacts'];row['wall_seconds']=receipt['wall_seconds']
                    if receipt['process']['status']!='success':raise ContractError('candidate measurement process failed')
                    value=validate_measurement(read_json(artifacts.verify(receipt['collection']['artifacts']['measurement.json'])))['measurements']
                    if set(value)!=set(case['expected']):raise ContractError('measurement names differ from reference contract')
                    errors={name:abs(value[name]-expected) for name,expected in case['expected'].items()}
                    for error in errors.values():number(error,'reference error',0)
                    row.update(errors=errors,passed=all(error<=case['absolute_tolerance'] for error in errors.values()))
                except (ContractError,OSError,ValueError,KeyError,TypeError) as error:row['error']=str(error)
                finally:settle_once(store,task,suite['case_cost'],'accepted' if row['passed'] else 'rejected')
                store.event('reference_case_completed',row);rows.append(row)
            accuracy=sum(row['passed'] for row in rows)/len(rows)
            report={'passed':all(row['passed'] for row in rows),'gain':accuracy,'regression':1-accuracy,
                'total_cost':store.budget()['charged_or_reserved'],'evaluation_ids':[c['id'] for c in cases],
                'identity':suite['identity'],'comparison_groups':['candidate_tool','host_reference'],'online':False,
                'measurement_accuracy':accuracy,'rows':rows,'binding':binding,'integrity':store.verify(),
                'qualification':'offline host-reference agreement; gain is case accuracy, not model or decision utility'}
            write_json(output,report);return report
        finally:store.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--suite',required=True);parser.add_argument('--candidate',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--stage',required=True,choices=['correctness','synthetic','robustness'])
    args=parser.parse_args();run_validation(args.suite,args.candidate,args.output,args.stage)


if __name__=='__main__':main()
