"""Verify a Linux process's currently mapped native files against an archive.

This is a snapshot of loaded files, not a guarantee about future lazy loads or
kernel/driver memory. Run in the process being audited before entering a jail.
"""
import os
import ctypes
import hashlib
import mmap
from pathlib import Path
import re
import stat

from .core import ContractError, digest
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


def mapping_region(probe):
    byte=ctypes.c_char.from_buffer(probe);address=ctypes.addressof(byte);del byte
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields=line.split(None,5)
        start,end=(int(value,16) for value in fields[0].split('-'))
        if start<=address<end:return fields
    raise ContractError('runtime descriptor mapping is not observable')


def mapped_fd_identity(handle):
    """Ask the kernel which file this open descriptor maps, without stat aliases."""
    with mmap.mmap(handle.fileno(),1,access=mmap.ACCESS_COPY) as probe:
        fields=mapping_region(probe)
        if len(fields)!=6 or not fields[5].startswith('/') or fields[5].endswith(' (deleted)'):
            raise ContractError('runtime descriptor lacks a named live mapping')
        return fields[3],int(fields[4])


def userspace_mappings(maps,devices):
    """Explicit GPU mode separates non-executable driver/anonymous data mappings."""
    if not devices:return maps,{}
    from .isolation import gpu_device_paths
    allowed=set(gpu_device_paths(devices));kept=[];excluded={}
    with mmap.mmap(-1,1) as anonymous:
        reference=mapping_region(anonymous)
    for line in maps.splitlines():
        fields=line.split(None,5)
        if len(fields)!=6:kept.append(line);continue
        name=fields[5];kind=None
        if name=='/dev/zero (deleted)' and len(reference)==6 and reference[5]==name and fields[3]==reference[3]:
            kind='anonymous shared data'
        elif name in allowed:
            if not stat.S_ISCHR(Path(name).stat().st_mode):raise ContractError('GPU mapping path is not a character device')
            kind='driver device memory'
        if kind is None:kept.append(line);continue
        if 'x' in fields[1] or not fields[1].endswith('s'):
            raise ContractError('GPU data mapping has unexpected protection')
        start,end=(int(v,16) for v in fields[0].split('-'))
        row=excluded.setdefault(name,{'kind':kind,'regions':0,'bytes':0})
        row['regions']+=1;row['bytes']+=end-start
    return '\n'.join(kept),excluded


def audit_loaded_runtime(report, *, maps=None, devices=()):
    if report.get('schema') != 'gpu-runtime-bytes-v1' or report.get('hash') != digest({k: v for k, v in report.items() if k != 'hash'}):
        raise ContractError('runtime archive identity mismatch')
    if report['host_identity'] != host_identity():
        raise ContractError('runtime host kernel or driver identity changed')
    if maps is None:
        maps = Path('/proc/self/maps').read_text()
    maps,excluded=userspace_mappings(maps,devices)
    paths = mapped_files(maps)
    if not paths:
        raise ContractError('runtime has no mapped files to audit')
    verified = {}
    for name, (device, inode) in sorted(paths.items()):
        ref = report['files'].get(name.lstrip('/'))
        if ref is None:
            raise ContractError('mapped runtime file is not archived: ' + name)
        path = Path(name)
        with path.open('rb') as handle:
            before=os.fstat(handle.fileno())
            # stat can expose an overlay identity while maps exposes its backing
            # inode. Compare two kernel mapping identities, then hash that same FD.
            if mapped_fd_identity(handle)!=(device,inode):
                raise ContractError('mapped runtime file inode changed: '+name)
            hasher=hashlib.sha256()
            for chunk in iter(lambda:handle.read(1024*1024),b''):hasher.update(chunk)
            sha=hasher.hexdigest()
            stamp=lambda value:(value.st_ino,value.st_dev,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
            if stamp(before)!=stamp(os.fstat(handle.fileno())) or stamp(before)!=stamp(path.stat()):
                raise ContractError('mapped runtime file changed during audit: '+name)
        if sha != ref['sha256'] or before.st_size != ref['size']:
            raise ContractError('mapped runtime bytes differ from archive: ' + name)
        verified[name] = {'sha256': sha, 'size': before.st_size}
    body = {'schema': 'loaded-runtime-audit-v1', 'runtime_archive_hash': report['hash'],
            'files': verified, 'identity_method':'kernel mapping of the same open descriptor used for hashing',
            'excluded_data_mappings':excluded,
            'scope': 'current file mappings only; excludes future loads, Python source, kernel and driver memory'}
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
