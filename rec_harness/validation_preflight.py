"""Read-only readiness checks for frozen validation plans, before research calls."""
import argparse
import json
from pathlib import Path
from .core import Artifacts,ContractError,digest,file_hash,read_json
from .evolution import freeze_validator_files


SUPPORTED={'rec_harness.'+name for name in ['tool_pipeline','harness_pipeline','tool_online',
    'online_validation','tool_shadow','tool_validation','harness_prefix']}


def check_suite(module,path,stage=None):
    path=Path(path).resolve()
    if module in {'rec_harness.online_validation','rec_harness.harness_prefix','rec_harness.tool_online'}:
        if not (path.parent/'objects').is_dir():raise ContractError('validation artifact directory missing')
    if module=='rec_harness.tool_pipeline':
        from .tool_pipeline import load_suite
        suite=load_suite(path)
        modules={'reference':'tool_validation','shadow':'tool_shadow','online':'tool_online'}
        for name,ref in suite['suites'].items():
            child=check_suite('rec_harness.'+modules[name],ref['path'])
            if child['identity']!=suite['identity']:raise ContractError('pipeline child identity differs')
            if child['tool_files']!=suite['tool_files']:raise ContractError('pipeline child tool files differ')
    elif module=='rec_harness.harness_pipeline':
        from .harness_pipeline import load_suite
        for ref in read_json(path)['suites'].values():
            if not (Path(ref['path']).parent/'objects').is_dir():raise ContractError('validation artifact directory missing')
        suite=load_suite(path)
    elif module=='rec_harness.online_validation':
        from .online_validation import load_suite
        suite,_=load_suite(path)
    elif module=='rec_harness.harness_prefix':
        from .harness_pipeline import prefix_suite
        suite=prefix_suite(path)
    elif module=='rec_harness.tool_shadow':
        from .tool_shadow import audit_inputs
        suite=read_json(path);audit_inputs(suite)
    elif module=='rec_harness.tool_validation':
        from .tool_validation import validate_suite
        suite=read_json(path)
        for item in ([stage] if stage else ['correctness','synthetic','robustness']):validate_suite(suite,item)
    elif module=='rec_harness.tool_online':
        from .online_validation import load_suite
        suite=read_json(path)
        if suite.get('schema')!='online-tool-suite-v1' or suite.get('hash')!=digest({k:v for k,v in suite.items() if k!='hash'}):
            raise ContractError('tool utility suite changed')
        if file_hash(Path(__file__).with_name('diagnostic_worker.py'))!=suite['worker_hash']:
            raise ContractError('diagnostic worker changed; freeze a new validation suite')
        if not (path.parent/'episodes/objects').is_dir():raise ContractError('validation artifact directory missing')
        episodes,_=load_suite(path.parent/'episodes/suite.json')
        if episodes['hash']!=suite['episode_suite_hash']:raise ContractError('bound episodes changed')
        if not (path.parent/'objects').is_dir():raise ContractError('validation artifact directory missing')
        objects=Artifacts(path.parent/'objects')
        for ref in suite['existing_tool'].values():objects.verify(ref)
    else:raise ContractError('unsupported suite preflight')
    return suite


def preflight(plan):
    hashes=freeze_validator_files(plan['stages']);checks=[];seen=set()
    for stage,rule in plan['stages'].items():
        command=rule['command'];module=command[command.index('-m')+1] if '-m' in command else None
        if module not in SUPPORTED:
            checks.append({'stage':stage,'scope':'validator files only; no built-in suite audit'})
            continue
        if '--suite' not in command:raise ContractError('built-in validator requires --suite for preflight')
        index=command.index('--suite')+1
        if index>=len(command) or '{' in command[index]:raise ContractError('preflight requires a concrete frozen suite path')
        path=Path(command[index]).resolve();key=(module,str(path),stage if module.endswith('.tool_validation') else None)
        if key not in seen:
            suite=check_suite(module,path,key[2]);seen.add(key)
            if suite['identity']!=plan['identity']:raise ContractError('validator suite differs from plan identity')
        checks.append({'stage':stage,'module':module,'suite':str(path),'suite_hash':file_hash(path),'scope':'frozen suite and bound inputs'})
    return {'status':'passed','validator_files_hash':digest(hashes),'checks':checks,
        'qualification':'Read-only readiness check; no candidate execution, Codex call, resource reservation or scientific utility result. External/custom validators receive file checks only.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('configuration');args=parser.parse_args()
    value=read_json(args.configuration)
    try:result=preflight(value.get('plan',value))
    except (ContractError,OSError,KeyError,TypeError,ValueError) as error:
        parser.exit(2,'validation preflight failed: '+str(error)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
