"""Isolated CPU inference jobs with explicit inputs and collected score outputs."""
from pathlib import Path
import shutil
import time

from .agents import run_process
from .core import ContractError, file_hash, read_json, write_json
from .gpu_job import allocate_uid, input_manifest
from .gpu_outputs import collect_outputs
from .isolation import prepare
from .process_identity import assert_stopped


def recover_completed_cpu_job(directory,recipe,files,entry,arguments,*,artifacts):
    """Recover collection after successful process cleanup; never relaunch inference."""
    directory=Path(directory);files=input_manifest(files)
    submitted=read_json(directory/'request.json')
    if any(submitted.get(key)!=value for key,value in {'inputs':files,'entry':entry,'arguments':arguments,'recipe':recipe}.items()):
        raise ContractError('CPU recovery input binding changed')
    assert_stopped(directory/'process')
    completion=directory/'process/process.json'
    if not completion.exists() or read_json(completion).get('status')!='success':
        raise ContractError('completed successful CPU process required for recovery')
    process=read_json(completion);receipt_path=directory/'receipt.json'
    if receipt_path.exists():
        receipt=read_json(receipt_path)
        if receipt['process']!=process:raise ContractError('CPU recovery process receipt changed')
        for ref in receipt['collection']['artifacts'].values():artifacts.verify(ref)
        return receipt
    collection=collect_outputs(directory/'jail/output',directory/'process',artifacts,max_bytes=32*1024**2)
    artifacts.restore(collection['artifacts'],directory/'results')
    receipt={'process':process,'collection':collection,'wall_seconds':None,
        'recovery':{'kind':'completed_process_collection','time_scope':'Process wall time preserved; total job time including staging is unknown.'}}
    write_json(receipt_path,receipt);return receipt


def run_cpu_job(directory, recipe, files, entry, arguments, *, python, uid_registry, artifacts, timeout=120):
    files = input_manifest(files)
    if entry not in files or timeout <= 0:
        raise ContractError('declared entry and positive CPU deadline required')
    directory = Path(directory).resolve(); directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); uid = allocate_uid(uid_registry)
    jail = directory / 'jail'
    write_json(directory / 'request.json', {'inputs': files, 'entry': entry, 'arguments': arguments,
               'uid': uid, 'recipe': recipe, 'qualification': 'isolated inference; outputs are untrusted'})
    prepare(jail, recipe['runtime_dirs'], recipe['runtime_files'], uid=uid)
    for name, ref in files.items():
        target = jail / 'inputs' / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ref['path'], target); target.chmod(0o444)
        if file_hash(target) != ref['sha256']:
            raise ContractError('CPU input changed during staging')
    remaining = timeout - (time.monotonic() - started)
    if remaining <= 0:
        raise ContractError('CPU inference deadline exhausted during staging')
    process = run_process([python, '-m', 'rec_harness.cpu_launcher', '--root', str(jail), '--uid', str(uid),
            '--script', '/inputs/' + entry, '--expected-hash', files[entry]['sha256'], '--', *arguments],
            Path(__file__).resolve().parents[1], directory / 'process', timeout=remaining,
            env={'PATH': '/usr/bin:/bin', 'PYTHONHASHSEED': '0'})
    collection = collect_outputs(jail / 'output', directory / 'process', artifacts, max_bytes=32 * 1024**2)
    artifacts.restore(collection['artifacts'], directory / 'results')
    receipt = {'process': process, 'collection': collection, 'wall_seconds': time.monotonic() - started}
    write_json(directory / 'receipt.json', receipt)
    return receipt
