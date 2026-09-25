"""Preinitialized GPU launcher for prepared, reviewed-code training jails.

The caller owns staging, timeout, accounting and output collection. This does
not authorize arbitrary generated models or provide a GPU-driver security proof.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import runpy
import sys

from .core import ContractError, file_hash, read_json, safe_file
from .gpu_descriptors import inventory, select_descriptors
from .policy_sandbox import restrict_syscalls


def validate_launch(root, script, expected_hash, device, uid):
    root = Path(root).resolve()
    if root == Path('/') or type(uid) is not int or uid <= 0:
        raise ContractError('prepared jail and nonprivileged UID required')
    manifest = read_json(root / 'jail-manifest.json')
    if manifest.get('gpu_devices') != [device] or manifest['uid'] != uid:
        raise ContractError('launch UID or device differs from prepared jail')
    path = Path(script)
    if not path.is_absolute() or '..' in path.parts:
        raise ContractError('script must be an absolute in-jail path')
    source = safe_file(root, str(path).lstrip('/'))
    if file_hash(source) != expected_hash:
        raise ContractError('launch script hash mismatch')
    if source.stat().st_uid != 0 or source.stat().st_mode & 0o022:
        raise ContractError('entry script must be root owned and not group/world writable')
    for directory in ('workspace', 'output', 'tmp'):
        target = root / directory
        if target.is_symlink() or not target.is_dir() or target.stat().st_uid != uid:
            raise ContractError('jail writable directory ownership mismatch')
    actual_devices = sorted(p.name for p in (root / 'dev').glob('nvidia*'))
    if actual_devices != sorted(['nvidia' + str(device), 'nvidiactl', 'nvidia-uvm']):
        raise ContractError('jail contains unexpected GPU device paths')
    return root


def launch(root, script, expected_hash, arguments, *, device=0, uid=65534, runtime_archive=None):
    if os.geteuid() != 0:
        raise ContractError('GPU jail requires privileged launcher')
    root = validate_launch(root, script, expected_hash, device, uid)
    archive = read_json(runtime_archive) if runtime_archive is not None else None
    frozen_hash = read_json(root / 'jail-manifest.json').get('runtime_archive_hash')
    if (archive['hash'] if archive else None) != frozen_hash:
        raise ContractError('launcher runtime archive differs from prepared jail')
    before = inventory()
    if before:
        raise ContractError('GPU launcher inherited unexpected non-stdio descriptors')
    os.environ.clear()
    os.environ.update(PATH='/usr/bin:/bin', HOME='/workspace', LANG='C.UTF-8',
                      CUDA_VISIBLE_DEVICES=str(device), OMP_NUM_THREADS='2',
                      MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONHASHSEED='0')
    import torch
    torch.set_num_threads(2)
    warmup = torch.ones((32, 32), device='cuda', requires_grad=True)
    (warmup @ warmup).sum().backward()
    torch.cuda.synchronize()
    del warmup
    if archive is not None:
        from .runtime_loaded import audit_loaded_runtime
        loaded = audit_loaded_runtime(archive)
        print(json.dumps({'loaded_runtime_audit': loaded}), flush=True)
    policy = select_descriptors(before, inventory(), [device])
    for fd in policy['close']:
        os.close(fd)
    restrict_syscalls(sync_threads=True)
    if ctypes.CDLL(None).prctl(38, 1, 0, 0, 0) != 0:
        raise ContractError('cannot set no_new_privs')
    # No exec: it would invalidate the preinitialized CUDA runtime and its pipes.
    os.chroot(root); os.chdir('/workspace')
    os.setgroups([]); os.setgid(uid); os.setuid(uid)
    if file_hash(script) != expected_hash:
        raise ContractError('entry script changed before isolated execution')
    print(json.dumps({'gpu_boundary': 'chroot-uid-seccomp-tsync-preinitialized',
                      'uid': os.getuid(), 'device': device, 'pipe_pairs': policy['pipe_pairs'],
                      'entry_hash': expected_hash}), flush=True)
    sys.argv = [script, *arguments]
    runpy.run_path(script, run_name='__main__')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True); parser.add_argument('--script', required=True)
    parser.add_argument('--expected-hash', required=True)
    parser.add_argument('--device', type=int, default=0); parser.add_argument('--uid', type=int, default=65534)
    parser.add_argument('--runtime-archive')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    launch(args.root, args.script, args.expected_hash, arguments, device=args.device, uid=args.uid, runtime_archive=args.runtime_archive)


if __name__ == '__main__':
    main()
