"""Archive the bytes of a prepared Python/library runtime, excluding job inputs."""
import platform
from pathlib import Path

from .core import ContractError, digest, file_hash, read_json, safe_file, write_json


def host_identity():
    driver = Path('/proc/driver/nvidia/version')
    return {'machine': platform.machine(), 'kernel_release': platform.release(),
            'nvidia_driver_version_hash': file_hash(driver) if driver.is_file() else None}


def check_path(name):
    p = Path(name)
    if not p.parts or p.is_absolute() or '..' in p.parts or p.parts[0] in {'dev', 'proc', 'inputs', 'output', 'workspace', 'tmp'} or name == 'jail-manifest.json':
        raise ContractError('runtime archive path overlaps job or kernel state')


def freeze_runtime(template, artifacts, output):
    template = Path(template).resolve()
    if Path(output).exists(): raise ContractError('runtime archive output already exists')
    recipe = read_json(template / 'jail-manifest.json')
    # The prepared tree has dereferenced aliases. Only declared runtime roots
    # and loader/library locations are eligible; job inputs never enter the lock.
    declared = [*recipe['runtime_dirs'], *recipe['runtime_files']]
    roots = [*declared,
             '/lib', '/lib64', '/usr/lib', '/etc/ld.so.cache']
    files = {}
    for root in roots:
        relative = str(Path(root)).lstrip('/')
        check_path(relative)
        path = template / relative
        if not path.exists():
            if root in declared:
                raise ContractError('declared runtime path missing from prepared template')
            continue
        paths = path.rglob('*') if path.is_dir() else [path]
        for candidate in paths:
            name = str(candidate.relative_to(template)); check_path(name)
            if candidate.is_symlink(): raise ContractError('prepared runtime must not contain symlinks')
            if candidate.is_file(): files[name] = candidate
    refs, modes, inode_cache = {}, {}, {}
    for name, path in sorted(files.items()):
        info = path.stat()
        key = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        if key not in inode_cache: inode_cache[key] = artifacts.put(safe_file(template, name))
        refs[name] = inode_cache[key]
        modes[name] = 0o555 if info.st_mode & 0o111 else 0o444
    body = {'schema': 'gpu-runtime-bytes-v1', 'files': refs, 'modes': modes,
            'recipe_hash': digest({k: recipe[k] for k in ['runtime_dirs', 'runtime_files']}),
            'host_identity': host_identity(),
            'scope': 'archived userspace files; kernel and GPU driver state are fingerprinted, not archived'}
    report = {**body, 'hash': digest(body)}
    write_json(output, report)
    return report


def validate_runtime(report, artifacts=None):
    if report.get('schema') != 'gpu-runtime-bytes-v1' or digest({k: v for k, v in report.items() if k != 'hash'}) != report['hash']:
        raise ContractError('runtime archive identity mismatch')
    if report['host_identity'] != host_identity():
        raise ContractError('runtime host kernel or driver identity changed')
    if set(report['files']) != set(report['modes']):
        raise ContractError('runtime archive mode map incomplete')
    for name, mode in report['modes'].items():
        check_path(name)
        if mode not in {0o444, 0o555}: raise ContractError('runtime files must be readonly')
        if artifacts is not None: artifacts.verify(report['files'][name])
    return report


def restore_runtime(report, artifacts, destination):
    # restore() verifies each object while copying; avoid a second full byte scan.
    validate_runtime(report)
    destination = Path(destination)
    if destination.exists(): raise ContractError('runtime destination must be new')
    artifacts.restore(report['files'], destination)
    for name, mode in report['modes'].items():
        safe_file(destination, name).chmod(mode)
    return destination
