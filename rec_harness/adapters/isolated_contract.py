"""Isolated synthetic CPU probe. Success is not candidate admission or trusted scoring."""
from pathlib import Path
import shutil
import time

from rec_harness.agents import run_process
from rec_harness.core import ContractError, digest, file_hash, read_json, write_json
from rec_harness.gpu_job import allocate_uid, input_manifest
from rec_harness.gpu_outputs import collect_outputs
from rec_harness.isolation import prepare


def run_isolated_contract(python, source, model_id, directory, recipe, artifacts, uid_registry, *, timeout=90):
    if timeout <= 0:
        raise ContractError('positive CPU probe timeout required')
    inspector = Path(__file__).with_name('model_contract.py')
    files = input_manifest({'model.py': source, 'inspector.py': {'path': str(inspector), 'sha256': file_hash(inspector)}})
    directory = Path(directory).resolve(); directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); uid = allocate_uid(uid_registry)
    jail = directory / 'jail'
    request = {'source_hash': source['sha256'], 'inspector_hash': files['inspector.py']['sha256'],
               'model_id': model_id, 'recipe_hash': digest(recipe), 'uid': uid,
               'scope': 'isolated synthetic CPU probe; candidate shares inspector process and can forge its output'}
    write_json(directory / 'request.json', request)
    prepare(jail, recipe['runtime_dirs'], recipe['runtime_files'], uid=uid)
    (jail / 'inputs').mkdir()
    for name, ref in files.items():
        target = jail / 'inputs' / name
        shutil.copyfile(ref['path'], target); target.chmod(0o444)
        if file_hash(target) != ref['sha256']:
            raise ContractError('CPU input changed while staging')
    remaining = timeout - (time.monotonic() - started)
    if remaining <= 0:
        raise ContractError('CPU probe deadline exhausted during staging')
    argv = [python, '-m', 'rec_harness.cpu_launcher', '--root', str(jail), '--uid', str(uid),
            '--script', '/inputs/inspector.py', '--expected-hash', files['inspector.py']['sha256'], '--',
            '--source', '/inputs/model.py', '--expected-hash', source['sha256'], '--model-id', model_id,
            '--output', '/output/report.json', '--threads', '1']
    process = run_process(argv, Path(__file__).resolve().parents[2], directory / 'process', timeout=remaining,
                          env={'PATH': '/usr/bin:/bin', 'PYTHONHASHSEED': '0'})
    collection = collect_outputs(jail / 'output', directory / 'process', artifacts, max_bytes=32 * 1024**2)
    artifacts.restore(collection['artifacts'], directory / 'results')
    receipt = {'process': process, 'collection': collection, 'status': process['status'],
               'wall_seconds': time.monotonic() - started, 'qualification': 'untrusted_probe_output',
               'candidate_admitted': False}
    if process['status'] == 'success':
        try:
            report = read_json(directory / 'results/report.json')
            if report.get('hash') != digest({k: v for k, v in report.items() if k != 'hash'}) or any(
                    report.get(k) != request[k] for k in ('source_hash', 'inspector_hash', 'model_id')):
                raise ContractError('CPU probe report identity mismatch')
            receipt['report_hash'] = report['hash']
        except (OSError, ValueError, TypeError, KeyError, ContractError) as error:
            receipt.update(status='invalid', reason=str(error))
    write_json(directory / 'receipt.json', receipt)
    return receipt
