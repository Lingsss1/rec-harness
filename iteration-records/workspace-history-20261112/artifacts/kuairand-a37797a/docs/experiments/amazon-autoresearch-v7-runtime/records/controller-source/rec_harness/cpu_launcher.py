"""CPU-only process boundary for model probes; reports remain untrusted output."""
import argparse
import ctypes
import os
from pathlib import Path
import resource
import runpy
import sys

from .core import ContractError, file_hash, read_json, safe_file
from .policy_sandbox import restrict_syscalls


def validate_launch(root, script, expected_hash, uid):
    root = Path(root).resolve()
    if root == Path('/') or type(uid) is not int or uid <= 0:
        raise ContractError('prepared jail and nonprivileged UID required')
    manifest = read_json(root / 'jail-manifest.json')
    if manifest['uid'] != uid or manifest.get('gpu_devices'):
        raise ContractError('CPU probe requires matching UID and no GPU devices')
    if any((root / 'dev').glob('nvidia*')):
        raise ContractError('CPU probe must not expose GPU device nodes')
    path = Path(script)
    if not path.is_absolute() or '..' in path.parts:
        raise ContractError('entry must be an absolute in-jail path')
    source = safe_file(root, str(path).lstrip('/'))
    info = source.stat()
    if file_hash(source) != expected_hash or info.st_uid != 0 or info.st_mode & 0o022:
        raise ContractError('CPU entry identity or ownership mismatch')
    for name in ('workspace', 'output', 'tmp'):
        directory = root / name
        if directory.is_symlink() or not directory.is_dir() or directory.stat().st_uid != uid:
            raise ContractError('CPU writable directory ownership mismatch')
    return root


def launch(root, script, expected_hash, arguments, *, uid):
    if os.geteuid() != 0:
        raise ContractError('CPU boundary requires privileged launcher')
    root = validate_launch(root, script, expected_hash, uid)
    os.environ.clear()
    os.environ.update(PATH='/usr/bin:/bin', HOME='/workspace', LANG='C.UTF-8',
                      CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1', PYTHONHASHSEED='0')
    # Initialize trusted CPU libraries before denying thread/process creation.
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024**2, 32 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    restrict_syscalls(sync_threads=True)
    if ctypes.CDLL(None).prctl(38, 1, 0, 0, 0) != 0:
        raise ContractError('cannot set no_new_privs')
    os.chroot(root); os.chdir('/workspace')
    os.setgroups([]); os.setgid(uid); os.setuid(uid)
    os.closerange(3, 1048576)
    if file_hash(script) != expected_hash:
        raise ContractError('CPU entry changed before execution')
    sys.argv = [script, *arguments]
    runpy.run_path(script, run_name='__main__')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True); parser.add_argument('--script', required=True)
    parser.add_argument('--expected-hash', required=True); parser.add_argument('--uid', type=int, required=True)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    launch(args.root, args.script, args.expected_hash, arguments, uid=args.uid)


if __name__ == '__main__':
    main()
