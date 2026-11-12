"""Recoverable host entry for the staged isolated synthetic NextItNet probe."""
import argparse
import fcntl
from pathlib import Path
from rec_harness.core import Artifacts,ContractError,digest,file_hash,number,read_json,write_json
from rec_harness.cpu_job import run_cpu_job
from rec_harness.diagnostics import validate_measurement
from rec_harness.process_identity import assert_stopped
from rec_harness.store import Store
from .activation_contract import public_measurement
from .activation_checkpoint import resolve_checkpoint
from .activation_histories import resolve_histories


def run_probe(config,directory,*,runner=None):
    root=Path(directory).resolve();root.mkdir(parents=True,exist_ok=True)
    number(config['budget'],'probe budget',0);number(config['cost'],'probe cost',0)
    if config['cost']>config['budget']:raise ContractError('probe cost exceeds declared envelope')
    if not isinstance(config['timeout'],(int,float)) or isinstance(config['timeout'],bool) or not 0<config['timeout']<=120:
        raise ContractError('probe timeout must be positive and at most 120 seconds')
    settings=config['settings']
    if set(settings)!={'HIDDEN_UNITS','NUM_BLOCKS','KERNEL_SIZE','DROPOUT_RATE','MAXLEN'}:
        raise ContractError('explicit NextItNet settings required')
    if type(settings['MAXLEN']) is not int or settings['MAXLEN']<16:raise ContractError('synthetic protocol needs length 16 support')
    base=Path(__file__).parent
    files={'model.py':config['source'],**{name:{'path':str(base/('activation_worker.py' if name=='worker.py' else name)),'sha256':file_hash(base/('activation_worker.py' if name=='worker.py' else name))} for name in ['activation_probe.py','worker.py']}}
    checkpoint=resolve_checkpoint(config)
    if checkpoint is not None:files['checkpoint.pt']=checkpoint
    histories=resolve_histories(config)
    binding={'checkpoint_hash':checkpoint['sha256'] if checkpoint else None,
        'checkpoint_adapter_hash':file_hash(base/'activation_checkpoint.py'),'config_hash':digest(config),'validator_hash':file_hash(base/'activation_contract.py'),
        'runtime_hash':file_hash(__file__),'probe_hash':files['activation_probe.py']['sha256'],'worker_hash':files['worker.py']['sha256']}
    if histories is not None:
        binding.update(history_input_hash=digest(histories),history_adapter_hash=file_hash(base/'activation_histories.py'))
    with (root/'controller.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise ContractError('activation probe already running') from error
        assert_stopped(root/'job')
        if (root/'binding.json').exists() and read_json(root/'binding.json')!=binding:raise ContractError('frozen activation probe changed')
        write_json(root/'binding.json',binding);write_json(root/'config.json',config)
        objects=Artifacts(root/'objects');store=Store(root/'memory')
        try:
            store.configure_budget(config['budget']);store.verify()
            terminal=[e['payload'] for e in store.events() if e['kind']=='activation_probe_completed']
            if terminal:
                result=terminal[-1]
                for ref in result.get('artifacts',{}).values():objects.verify(ref)
                if result.get('status')=='success':
                    measurement=read_json(objects.verify(result['artifacts']['measurement.json']))
                    if measurement!=result['measurement']:raise ContractError('committed probe measurement differs from archive')
                    write_json(root/'measurement.json',measurement)
                write_json(root/'result.json',result);return result
            recovered=store.recover()
            attempted=any(e['kind']=='budget_reserved' and e['payload']['task_id']=='activation-probe' for e in store.events())
            if recovered or attempted or (root/'job').exists():
                starts=[e['payload'] for e in store.events() if e['kind']=='activation_probe_started']
                refs=dict(starts[-1]['inputs']) if starts else {}
                if starts and starts[-1]['binding']!=binding:raise ContractError('interrupted probe input binding changed')
                for ref in refs.values():objects.verify(ref)
                for name in ['request.json','receipt.json','process/process.json','process/stdout.jsonl','process/stderr.txt']:
                    path=root/'job'/name
                    if path.is_file() and not path.is_symlink():refs['job/'+name]=objects.put(path)
                result={'status':'interrupted','reason':'Stopped unfinished probe retained; use a new run to retry.',
                    'binding':binding,'budget':store.budget(),'artifacts':refs}
                store.event('activation_probe_completed',result);write_json(root/'result.json',result);return result
            # Verify and freeze declared source bytes before starting any subprocess.
            if histories is not None:
                history_path=root/'private-histories.json';write_json(history_path,histories['payload'])
                history_path.chmod(0o600)
                files['histories.json']={'path':str(history_path),'sha256':file_hash(history_path)}
            frozen={name:objects.put(ref['path']) for name,ref in files.items()}
            if any(frozen[name]['sha256']!=ref['sha256'] for name,ref in files.items()):raise ContractError('activation probe source changed')
            staged={name:{'path':str(objects.verify(ref)),'sha256':ref['sha256']} for name,ref in frozen.items()}
            store.reserve('activation-probe',config['cost']);status='invalid';result={};refs=dict(frozen)
            store.event('activation_probe_started',{'inputs':frozen,'binding':binding})
            try:
                execution=config['execution']
                receipt=(runner or run_cpu_job)(root/'job',execution['recipe'],staged,'worker.py',[],
                    python=execution['python'],uid_registry=execution['uid_registry'],artifacts=objects,timeout=config['timeout'])
                refs.update({'output/'+name:ref for name,ref in receipt['collection']['artifacts'].items()})
                for ref in refs.values():objects.verify(ref)
                if receipt['process']['status']!='success':raise ContractError('isolated activation worker failed')
                if 'report.json' not in receipt['collection']['artifacts']:raise ContractError('activation report missing')
                report=read_json(objects.verify(receipt['collection']['artifacts']['report.json']))
                expected={'source_hash':config['source']['sha256'],'probe_hash':binding['probe_hash'],'settings':settings,
                    'initialization':('development checkpoint '+checkpoint['sha256'] if checkpoint else 'random seed 13, no checkpoint'),'slice_counts':{'all':5,'short':3,'long':2},'valid_positions':41}
                if histories is not None:
                    expected.update(history_hash=files['histories.json']['sha256'],
                        slice_counts=histories['slice_counts'],valid_positions=histories['valid_positions'])
                measurement=public_measurement(report,expected);validate_measurement(measurement)
                write_json(root/'measurement.json',measurement);refs['measurement.json']=objects.put(root/'measurement.json')
                status='success';result={'measurement':measurement}
            except (ContractError,OSError,ValueError,KeyError,TypeError) as error:result={'reason':str(error)}
            finally:store.settle('activation-probe',config['cost'],status)
            result.update(status=status,binding=binding,budget=store.budget(),artifacts=refs,
                qualification=('Sampled training-history' if histories is not None else 'Synthetic-history')+
                    ' NextItNet probe; optional audited development weights stay local. Not a promoted diagnostic, recommendation accuracy or scientific utility result.')
            store.event('activation_probe_completed',result);write_json(root/'result.json',result);return result
        finally:store.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('config');parser.add_argument('directory');args=parser.parse_args()
    result=run_probe(read_json(args.config),args.directory)
    print({'status':result['status'],'budget':result['budget']})


if __name__=='__main__':main()
