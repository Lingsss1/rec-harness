"""Filesystem jail for GPU/agent workers where user namespaces are unavailable.

Preparation requires host privilege. The child chroots, closes outside fds, drops
all supplementary groups and UID/GID, then execs. No host /proc is mounted.
Runtime/input files remain root-owned; only declared output directories are
writable. This is a process/filesystem boundary, not protection against kernel
or GPU-driver vulnerabilities.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import stat

from .core import ContractError, digest, read_json, write_json


def target_path(root, absolute):
    path = Path(absolute)
    if not path.is_absolute() or '..' in path.parts:
        raise ContractError('jail paths must be absolute and normalized')
    return root / str(path).lstrip('/')


def copy_readonly(source, root, destination=None, seen=None):
    source = Path(source)
    destination = Path(destination or source)
    target = target_path(root, destination)
    seen = set() if seen is None else seen
    identity = (str(source), str(destination))
    if identity in seen:
        return
    seen.add(identity)
    if source.is_symlink():
        # Dereference into the jail; never traverse a staged absolute symlink on the host.
        copy_readonly(source.resolve(), root, destination, seen)
    elif source.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            copy_readonly(child, root, destination / child.name, seen)
    elif source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists(): return
        mode = source.stat().st_mode
        # Never chmod/chown a hardlink; shared host files must already be readonly to the child.
        if mode & 0o022 or source.stat().st_uid == 65534:
            shutil.copyfile(source, target)
            target.chmod(0o555 if mode & 0o111 else 0o444)
        else:
            try: os.link(source, target)
            except OSError: shutil.copy2(source, target)
    else:
        raise ContractError(f'unsupported runtime file: {source}')


def gpu_device_paths(device_ids):
    if not isinstance(device_ids, list) or not device_ids or any(type(i) is not int or i < 0 for i in device_ids) or len(set(device_ids)) != len(device_ids):
        raise ContractError('GPU jail requires explicit distinct nonnegative device IDs')
    return ['/dev/nvidiactl', '/dev/nvidia-uvm'] + ['/dev/nvidia' + str(i) for i in sorted(device_ids)]


def prepare(root, runtime_dirs, runtime_files, inputs=None, uid=65534, include_gpu=False, gpu_devices=None,
            runtime_archive=None, runtime_artifacts=None):
    gpu_paths = gpu_device_paths(gpu_devices) if include_gpu else []
    if not include_gpu and gpu_devices is not None:
        raise ContractError('gpu_devices requires include_gpu')
    if os.geteuid() != 0:
        raise ContractError('jail preparation requires privilege')
    root = Path(root).resolve()
    if (runtime_archive is None) != (runtime_artifacts is None):
        raise ContractError('runtime archive and artifact store must be supplied together')
    if runtime_archive is not None:
        from .runtime_archive import restore_runtime
        if runtime_archive.get('recipe_hash') != digest({'runtime_dirs': runtime_dirs, 'runtime_files': runtime_files}):
            raise ContractError('runtime archive does not match requested recipe')
        restore_runtime(runtime_archive, runtime_artifacts, root)
    else:
        root.mkdir(parents=True, exist_ok=False)
        for directory in runtime_dirs:
            copy_readonly(directory, root)
        for path in runtime_files:
            if not any(Path(path).is_relative_to(Path(d)) for d in runtime_dirs):
                copy_readonly(path, root)
        for path in runtime_files:
            if str(path).startswith('/usr/lib/'):
                copy_readonly(path, root, str(path)[4:])
        resolved = {str(Path(p).resolve()) for p in runtime_files}
        for directory in {Path(p).parent for p in runtime_files}:
            for alias in directory.glob('*.so*'):
                if alias.is_symlink() and str(alias.resolve()) in resolved:
                    copy_readonly(alias, root)
                    if str(alias).startswith('/usr/lib/'):
                        copy_readonly(alias, root, str(alias)[4:])
        for source in ('/lib64/ld-linux-x86-64.so.2', '/etc/ld.so.cache'):
            if Path(source).exists(): copy_readonly(source, root)
    for source, destination in (inputs or {}).items():
        copy_readonly(source, root, destination)
    for relative in ('tmp', 'output', 'workspace'):
        path = root / relative; path.mkdir(exist_ok=True)
        path.chmod(0o700); os.chown(path, uid, uid)
    (root / 'dev').mkdir(exist_ok=True)
    devices = ['/dev/null', '/dev/zero', '/dev/random', '/dev/urandom']
    devices += gpu_paths
    for source in devices:
        info = os.stat(source)
        destination = root / source.lstrip('/')
        os.mknod(destination, stat.S_IFCHR | 0o666, info.st_rdev)
        destination.chmod(0o666)
    # Safe snapshots only: no /proc/PID/root, memory or host file descriptors.
    for source in ('/proc/cpuinfo', '/proc/meminfo', '/proc/driver/nvidia/version'):
        if Path(source).exists():
            target = target_path(root, source); target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(source).read_bytes()); target.chmod(0o444)
    write_json(root / 'jail-manifest.json', {'runtime_dirs': runtime_dirs, 'runtime_files': runtime_files,
                                          'inputs': inputs or {}, 'uid': uid, 'gpu_devices': gpu_devices or [],
                                          'runtime_archive_hash': runtime_archive['hash'] if runtime_archive else None,
                                          'device_paths': devices, 'boundary': 'chroot-unprivileged-no-host-proc'})
    return root


def enter(root, command, uid=65534):
    if os.geteuid() != 0:
        raise ContractError('entering jail requires privilege before dropping uid')
    root = Path(root).resolve()
    if root == Path('/') or not (root / 'jail-manifest.json').is_file():
        raise ContractError('not a prepared jail')
    if uid <= 0:
        raise ContractError('worker uid must be nonprivileged')
    if ctypes.CDLL(None).prctl(38, 1, 0, 0, 0) != 0:
        raise ContractError('cannot set no_new_privs')
    os.chroot(root); os.chdir('/workspace')
    os.setgroups([]); os.setgid(uid); os.setuid(uid)
    # No descriptor referring to the host may survive exec.
    os.closerange(3, 1048576)
    os.execv(command[0], command)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'enter'])
    parser.add_argument('root')
    parser.add_argument('--config')
    parser.add_argument('--uid', type=int, default=65534)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.root, **read_json(args.config))
    else:
        command = args.command[1:] if args.command[:1] == ['--'] else args.command
        uid = args.uid
        if command[:1] == ['--uid']:
            uid = int(command[1]); command = command[2:]
            if command[:1] == ['--']: command = command[1:]
        enter(args.root, command, uid=uid)


if __name__ == '__main__':
    main()
