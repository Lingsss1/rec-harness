"""Verify a Linux process's currently mapped native files against an archive.

This is a snapshot of loaded files, not a guarantee about future lazy loads or
kernel/driver memory. Run in the process being audited before entering a jail.
"""
import os
from pathlib import Path
import re

from .core import ContractError, digest, file_hash
from .runtime_archive import host_identity


def mapped_files(maps):
    files = {}
    for line in maps.splitlines():
        fields = line.split(None, 5)
        if len(fields) != 6 or not fields[5].startswith('/'):
            continue
        path = re.sub(r'\\([0-7]{3})', lambda match: chr(int(match[1], 8)), fields[5])
        if path.endswith(' (deleted)'):
            raise ContractError('runtime has a deleted file mapping')
        identity = (fields[3], int(fields[4]))
        if path in files and files[path] != identity:
            raise ContractError('runtime mapping has conflicting inode identities')
        files[path] = identity
    return files


def audit_loaded_runtime(report, *, maps=None):
    if report.get('schema') != 'gpu-runtime-bytes-v1' or report.get('hash') != digest({k: v for k, v in report.items() if k != 'hash'}):
        raise ContractError('runtime archive identity mismatch')
    if report['host_identity'] != host_identity():
        raise ContractError('runtime host kernel or driver identity changed')
    if maps is None:
        maps = Path('/proc/self/maps').read_text()
    paths = mapped_files(maps)
    if not paths:
        raise ContractError('runtime has no mapped files to audit')
    verified = {}
    for name, (device, inode) in sorted(paths.items()):
        ref = report['files'].get(name.lstrip('/'))
        if ref is None:
            raise ContractError('mapped runtime file is not archived: ' + name)
        path = Path(name)
        before = path.stat()
        major, minor = (int(part, 16) for part in device.split(':'))
        if before.st_ino != inode or (os.major(before.st_dev), os.minor(before.st_dev)) != (major, minor):
            raise ContractError('mapped runtime file inode changed: ' + name)
        sha = file_hash(path)
        after = path.stat()
        if (before.st_ino, before.st_dev, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_ino, after.st_dev, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ContractError('mapped runtime file changed during audit: ' + name)
        if sha != ref['sha256'] or before.st_size != ref['size']:
            raise ContractError('mapped runtime bytes differ from archive: ' + name)
        verified[name] = {'sha256': sha, 'size': before.st_size}
    body = {'schema': 'loaded-runtime-audit-v1', 'runtime_archive_hash': report['hash'],
            'files': verified, 'scope': 'current file mappings only; excludes future loads, Python source, kernel and driver memory'}
    return {**body, 'hash': digest(body)}


def main():
    """CPU-only startup check; the launcher must still audit after CUDA warmup."""
    import argparse
    from .core import read_json, write_json
    parser=argparse.ArgumentParser()
    parser.add_argument('archive');parser.add_argument('output')
    args=parser.parse_args()
    report=read_json(args.archive)
    try:
        import torch
        torch.set_num_threads(2)
        audit=audit_loaded_runtime(report)
        result={'status':'success','audit':audit}
    except Exception as error:
        result={'status':'invalid','reason':str(error),'error_type':type(error).__name__}
    result.update(runtime_archive_hash=report.get('hash'),
        scope='CPU startup mappings only; CUDA warmup and job-time audit still required')
    write_json(args.output,result)
    return 0 if result['status']=='success' else 1


if __name__=='__main__':
    raise SystemExit(main())
