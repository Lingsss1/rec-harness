"""Per-job staging and collection for reviewed GPU training entry points."""
import os
from pathlib import Path
import pwd
import secrets
import shutil
import time

from .agents import run_process
from .core import Artifacts, ContractError, digest, file_hash, safe_file, write_json
from .gpu_outputs import collect_outputs
from .isolation import prepare


def allocate_uid(registry):
    registry = Path(registry); registry.mkdir(parents=True, exist_ok=True)
    for _ in range(100):
        uid = 200000 + secrets.randbelow(100000000)
        try:
            pwd.getpwuid(uid)
            continue
        except KeyError:
            pass
        try:
            (registry / str(uid)).mkdir()
            return uid
        except FileExistsError:
            continue
    raise ContractError('cannot allocate unused job UID')


def input_manifest(files):
    result = {}
    for relative, source in files.items():
        path = Path(relative)
        if not path.parts or path.is_absolute() or '..' in path.parts:
            raise ContractError('input paths must be relative under /inputs')
        actual = Path(source['path'])
        if actual.is_symlink() or not actual.is_file() or file_hash(actual) != source['sha256']:
            raise ContractError('declared input hash or file type mismatch')
        result[str(path)] = dict(source)
    if len(result) != len(files):
        raise ContractError('duplicate normalized input path')
    return result


def run_reviewed_job(directory, recipe, files, entry, arguments, *, python, device,
                     uid_registry, timeout, artifacts, max_output_bytes=2 * 1024**3,
                     runtime_archive=None):
    """Caller reserves GPU budget before calling and settles actual slot duration.

    Runtime recipe comes from a reviewed administrator configuration. Input bytes
    are frozen per job; complete runtime package/environment locking is separate.
    """
    if timeout <= 0:
        raise ContractError('positive GPU job timeout required')
    files = input_manifest(files)
    if entry not in files:
        raise ContractError('entry is not a declared input')
    directory = Path(directory).resolve(); directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    uid = allocate_uid(uid_registry)
    jail = directory / 'jail'
    write_json(directory / 'request.json', {'runtime_recipe': recipe, 'runtime_recipe_hash': digest(recipe),
               'inputs': files, 'entry': entry, 'arguments': arguments, 'device': device,
               'uid': uid, 'timeout': timeout, 'scope': 'reviewed-code execution',
               'runtime_archive_hash': runtime_archive['hash'] if runtime_archive else None})
    prepare(jail, recipe['runtime_dirs'], recipe['runtime_files'], uid=uid, include_gpu=True, gpu_devices=[device],
            runtime_archive=runtime_archive, runtime_artifacts=artifacts if runtime_archive is not None else None)
    for relative, source in files.items():
        destination = jail / 'inputs' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Snapshot inputs, never share mutable source inodes with the job.
        shutil.copyfile(source['path'], destination); destination.chmod(0o444)
        if file_hash(destination) != source['sha256']:
            raise ContractError('input changed while staging')
    remaining = timeout - (time.monotonic() - started)
    if remaining <= 0:
        raise ContractError('GPU job deadline exhausted during staging')
    archive_arguments = []
    if runtime_archive is not None:
        lock_path = directory / 'runtime-lock.json'
        write_json(lock_path, runtime_archive)
        archive_arguments = ['--runtime-archive', str(lock_path)]
    argv = [python, '-m', 'rec_harness.gpu_launcher', '--root', str(jail),
            '--script', '/inputs/' + entry, '--expected-hash', files[entry]['sha256'],
            '--device', str(device), '--uid', str(uid), *archive_arguments, '--', *arguments]
    process = run_process(argv, Path(__file__).resolve().parents[1], directory / 'process',
                          timeout=remaining, env={'PATH': '/usr/bin:/bin', 'PYTHONHASHSEED': '0'})
    collection = collect_outputs(jail / 'output', directory / 'process', artifacts,
                                 max_bytes=max_output_bytes)
    # Export verified CAS objects into the controller-owned result directory.
    artifacts.restore(collection['artifacts'], directory / 'results')
    report = {'process': process, 'collection': collection, 'uid': uid,
              'allocated_gpu_seconds': time.monotonic() - started, 'device': device,
              'runtime_archive_hash': runtime_archive['hash'] if runtime_archive else None}
    write_json(directory / 'completed.json', report)
    return report
