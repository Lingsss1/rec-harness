"""CPU correctness gates for reviewed models, inside a shared job deadline."""
from pathlib import Path

from rec_harness.agents import run_process
from rec_harness.core import ContractError, digest, file_hash, read_json, write_json


def run_gate(python, source, model_id, directory, env, timeout, *, checkpoint=None, checkpoint_hash=None):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    inspector = Path(__file__).with_name('model_contract.py')
    frozen = directory / 'inspector.py'
    frozen.write_bytes(inspector.read_bytes())
    inspector_hash = file_hash(frozen)
    report_path = directory / 'report.json'
    package_root = Path(__file__).resolve().parents[2]
    argv = [python, str(frozen), '--source', str(Path(source['path']).resolve()), '--expected-hash', source['sha256'],
            '--model-id', model_id, '--output', str(report_path)]
    if checkpoint is not None:
        argv += ['--checkpoint', str(Path(checkpoint).resolve()), '--checkpoint-hash', checkpoint_hash]
    environment = {**env, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': str(package_root)}
    write_json(directory / 'request.json', {'argv': argv, 'inspector_hash': inspector_hash,
               'source_hash': source['sha256'], 'checkpoint_hash': checkpoint_hash,
               'scope': 'reviewed source, CPU synthetic contract; not untrusted-code isolation'})
    if timeout <= 0:
        return {'status': 'timeout', 'reason': 'shared job deadline exhausted'}
    process = run_process(argv, package_root, directory / 'process', timeout=timeout, env=environment)
    result = {'status': process['status'], 'process': process, 'inspector_hash': inspector_hash}
    if process['status'] == 'success':
        try:
            report = read_json(report_path)
            if (digest({k: v for k, v in report.items() if k != 'hash'}) != report['hash']
                    or report['source_hash'] != source['sha256']
                    or report['inspector_hash'] != inspector_hash
                    or report.get('checkpoint_hash') != checkpoint_hash
                    or report['model_id'] != model_id):
                raise ContractError('gate report identity mismatch')
            checks = report['checks']
            if not all(checks.get(k) is True for k in ['causal_input_gradient', 'tail_readout', 'empty_history_finite', 'padding_gradient_zero']):
                raise ContractError('gate checks incomplete')
            from rec_harness.model_graph import tensor_graph_summary
            result['tensor_graph']=tensor_graph_summary(report,model_id,source['sha256'],required=True)
            result['report_hash'] = report['hash']
        except (OSError, ValueError, KeyError, TypeError, ContractError) as error:
            result.update(status='invalid', reason=str(error))
    write_json(directory / 'receipt.json', result)
    return result
