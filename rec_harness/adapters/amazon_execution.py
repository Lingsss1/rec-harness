"""Explicit append-only registration of reviewed GPU execution versions."""
from pathlib import Path

from rec_harness.core import Artifacts, ContractError, digest, file_hash

PACKAGE = Path(__file__).resolve().parents[1]
# Freeze the whole controller package, including budget/recovery and indirect
# imports, instead of maintaining an incomplete hand-written dependency list.
CODE = sorted(str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob('*.py'))


def code_identity():
    package = Path(__file__).resolve().parents[1]
    return {name: file_hash(package / name) for name in CODE}


def register_execution(store, recipe, *, runtime_template=None):
    if set(recipe) != {'runtime_dirs', 'runtime_files'} or not all(
            isinstance(recipe[k], list) and recipe[k] and all(isinstance(p, str) and Path(p).is_absolute() for p in recipe[k])
            for k in recipe):
        raise ContractError('execution recipe needs explicit absolute runtime dirs/files')
    hashes = code_identity()
    artifacts = Artifacts(store.root.parent / 'artifacts/objects')
    package = Path(__file__).resolve().parents[1]
    refs = artifacts.snapshot(package, CODE)
    if {name: ref['sha256'] for name, ref in refs.items()} != hashes:
        raise ContractError('execution code changed during registration')
    body = {'kind': 'isolated-reviewed-v1', 'recipe': recipe, 'code_hashes': hashes,
            'code_artifacts': refs,
            'scope': 'reviewed models only; recipe identity is not a complete runtime package lock'}
    if runtime_template is not None:
        from tempfile import TemporaryDirectory
        from rec_harness.runtime_archive import freeze_runtime, validate_runtime
        with TemporaryDirectory() as temporary:
            archive = freeze_runtime(runtime_template, artifacts, Path(temporary) / 'runtime-lock.json')
        if archive['recipe_hash'] != digest(recipe) or not archive['files']:
            raise ContractError('runtime template must contain files and match execution recipe')
        validate_runtime(archive, artifacts)
        body.update(runtime_archive=archive,
                    scope='reviewed models; archived userspace runtime bytes; kernel and GPU driver fingerprinted, not archived')
    execution = {**body, 'execution_id': 'execution-' + digest(body)[:24]}
    store.verify()
    if not any(e['kind'] == 'amazon_execution_registered' and e['payload'] == execution for e in store.events()):
        store.event('amazon_execution_registered', execution)
    return execution


def preflight_execution(store, execution, python):
    """Reject unavailable frozen runtimes before reserving any GPU job."""
    if 'runtime_archive' not in execution:return None
    from uuid import uuid4
    from rec_harness.agents import run_process
    from rec_harness.core import read_json, write_json
    directory=store.root.parent/'runtime-preflight'/uuid4().hex
    directory.mkdir(parents=True)
    archive=execution['runtime_archive']
    lock=directory/'runtime-lock.json';output=directory/'result.json'
    write_json(lock,archive)
    process=run_process([python,'-m','rec_harness.runtime_loaded',str(lock),str(output)],
        PACKAGE.parent,directory/'process',timeout=120,
        env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','CUDA_VISIBLE_DEVICES':'',
             'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONHASHSEED':'0'})
    result=read_json(output) if output.is_file() else {'status':'invalid','reason':'startup check produced no result'}
    valid=(process['status']=='success' and result.get('status')=='success'
           and result.get('runtime_archive_hash')==archive['hash']
           and result.get('audit',{}).get('runtime_archive_hash')==archive['hash'])
    artifacts=Artifacts(store.root.parent/'artifacts/objects')
    receipt={'execution_id':execution['execution_id'],'runtime_archive_hash':archive['hash'],
        'status':'success' if valid else 'invalid','process':process,
        'result_ref':artifacts.put(output) if output.is_file() else None,
        'scope':'CPU preflight; no GPU job reserved; does not replace launcher audit'}
    write_json(directory/'receipt.json',receipt)
    store.event('amazon_runtime_preflight',receipt)
    if not valid:raise ContractError('runtime startup preflight failed before GPU reservation; inspect '+str(directory))
    return receipt


def resolve_execution(store, execution_id):
    store.verify()
    matches = [e['payload'] for e in store.events() if e['kind'] == 'amazon_execution_registered'
               and e['payload']['execution_id'] == execution_id]
    if len(matches) != 1:
        raise ContractError('execution version must be uniquely registered')
    execution = matches[0]
    body = {k: v for k, v in execution.items() if k != 'execution_id'}
    if 'execution-' + digest(body)[:24] != execution_id or execution['code_hashes'] != code_identity():
        raise ContractError('registered execution code changed; register a new version')
    artifacts = Artifacts(store.root.parent / 'artifacts/objects')
    refs = execution.get('code_artifacts')
    if not isinstance(refs, dict) or set(refs) != set(execution['code_hashes']):
        raise ContractError('execution code archive missing; register a new version')
    for name, ref in refs.items():
        if ref['sha256'] != execution['code_hashes'][name]:
            raise ContractError('execution code archive identity mismatch')
        artifacts.verify(ref)
    if 'runtime_archive' in execution:
        from rec_harness.runtime_archive import validate_runtime
        archive = execution['runtime_archive']
        if archive['recipe_hash'] != digest(execution['recipe']) or not archive['files']:
            raise ContractError('registered runtime archive differs from execution recipe')
        validate_runtime(archive, artifacts)
    return execution
