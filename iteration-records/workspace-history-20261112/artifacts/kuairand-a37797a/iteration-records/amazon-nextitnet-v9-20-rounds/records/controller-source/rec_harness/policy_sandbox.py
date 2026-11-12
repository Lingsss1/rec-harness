"""Local replay policy subprocesses: no world mount, network, credentials or forks.

Linux/root launcher required. Only Python's public runtime is staged. The
simulator retains the complete world and sends one visible prefix over stdin.
This boundary does not defend against kernel vulnerabilities.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import errno
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import sysconfig
import time
import uuid

from .agents import run_process
from .core import ContractError, canonical, file_hash, read_json, safe_file, write_json
from .isolation import enter, prepare


def restrict_syscalls(*, sync_threads=False):
    library = ctypes.util.find_library('seccomp')
    if not library:
        raise ContractError('libseccomp required; no unisolated fallback')
    lib = ctypes.CDLL(library, use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]; lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]; lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    lib.seccomp_attr_set.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32]
    context = lib.seccomp_init(0x7fff0000)  # SCMP_ACT_ALLOW
    if not context:
        raise ContractError('cannot allocate seccomp filter')
    try:
        # libseccomp's public enum SCMP_FLTATR_CTL_TSYNC = 4.
        # A preinitialized GPU runtime already has threads: filtering only the
        # calling thread would leave those threads outside the boundary.
        if sync_threads and lib.seccomp_attr_set(context, 4, 1) != 0:
            raise ContractError('cannot request thread-synchronized syscall filter')
        # Deny network and process creation, including alternate io_uring route.
        for name in ('socket', 'socketpair', 'connect', 'accept', 'accept4', 'bind', 'listen',
                     'sendto', 'sendmsg', 'sendmmsg', 'recvfrom', 'recvmsg', 'recvmmsg',
                     'clone', 'clone3', 'fork', 'vfork', 'io_uring_setup', 'ptrace',
                     'process_vm_readv', 'process_vm_writev', 'bpf', 'mount', 'unshare', 'setns'):
            call = lib.seccomp_syscall_resolve_name(name.encode())
            if call >= 0 and lib.seccomp_rule_add(context, 0x00050000 | errno.EPERM, call, 0) != 0:
                raise ContractError('cannot install syscall rule')
        if lib.seccomp_load(context) != 0:
            raise ContractError('cannot load seccomp filter')
    finally:
        lib.seccomp_release(context)


def prepare_runtime(destination):
    """Freeze public CPython + standard library; never stage a research project."""
    binary = Path(sys.executable).resolve()
    stdlib = Path(sysconfig.get_path('stdlib')).resolve()
    if not str(stdlib).startswith('/usr/lib/python') or not str(binary).startswith('/usr/bin/python'):
        raise ContractError('prepare using system /usr/bin/python3, not a project virtualenv')
    runtime_files = {str(binary)}
    for executable in [binary, *stdlib.glob('lib-dynload/*.so')]:
        result = subprocess.run(['ldd', str(executable)], text=True, capture_output=True, check=True)
        for line in result.stdout.splitlines():
            for word in line.split():
                if word.startswith('/') and Path(word).is_file():
                    runtime_files.add(word)
    root = prepare(destination, [str(stdlib)], sorted(runtime_files), include_gpu=False)
    manifest = read_json(root / 'jail-manifest.json')
    manifest.update(purpose='replay-policy-runtime', python=str(binary), stdlib=str(stdlib))
    manifest['runtime_hashes'] = {str(p.relative_to(root)): file_hash(p) for p in root.rglob('*')
                                  if p.is_file() and p.name != 'jail-manifest.json'}
    write_json(root / 'jail-manifest.json', manifest)
    return manifest


class PolicySandbox:
    def __init__(self, template, directory, *, timeout=10):
        self.template, self.directory = Path(template).resolve(), Path(directory).resolve()
        self.manifest = read_json(self.template / 'jail-manifest.json')
        if self.manifest.get('purpose') != 'replay-policy-runtime':
            raise ContractError('dedicated policy runtime required; agent/auth templates forbidden')
        if not isinstance(self.manifest.get('runtime_hashes'),dict) or not self.manifest['runtime_hashes']:
            raise ContractError('policy runtime lacks frozen file hashes; prepare a new runtime')
        self.timeout = timeout

    def decide(self, policy, visible, directory):
        if os.geteuid() != 0:
            raise ContractError('policy isolation requires privileged launcher')
        started = time.monotonic()
        for relative, expected in self.manifest['runtime_hashes'].items():
            if file_hash(safe_file(self.template, relative)) != expected:
                raise ContractError('frozen policy runtime changed')
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        run_id = uuid.uuid4().hex
        uid = 200000 + int(run_id[:7], 16)
        jail = self.directory / run_id
        # Template is runtime only: no data, world, historical graph or credentials.
        shutil.copytree(self.template, jail, symlinks=True,
                        ignore=shutil.ignore_patterns('dev', 'tmp', 'workspace', 'output'), copy_function=os.link)
        for relative in ('tmp', 'workspace'):
            path = jail / relative; path.mkdir(); path.chmod(0o700); os.chown(path, uid, uid)
        policy = Path(policy).resolve()
        shutil.copyfile(policy, jail / 'workspace/policy.py')
        (jail / 'workspace/policy.py').chmod(0o444)
        request = {'visible': visible, 'policy_sha256': file_hash(policy),
                   'boundary': 'chroot-unique-uid-seccomp-no-network-no-fork', 'uid': uid}
        write_json(directory / 'request.json', request)
        launch = [sys.executable, '-m', 'rec_harness.policy_sandbox', 'enter', str(jail),
                  '--uid', str(uid), '--python', self.manifest['python']]
        result = run_process(launch, Path(__file__).resolve().parents[1], directory,
                             stdin=canonical(visible), timeout=self.timeout,
                             env={'PATH': '/usr/bin:/bin', 'HOME': '/workspace', 'LANG': 'C.UTF-8', 'PYTHONHASHSEED': '0'})
        if result['status'] != 'success':
            raise ContractError(f'policy process {result["status"]}; see {directory}')
        path = safe_file(directory, 'stdout.jsonl')
        if path.stat().st_size > 1048576:
            raise ContractError('policy output exceeds 1 MiB')
        response = read_json(path)
        write_json(directory / 'response.json', response)
        write_json(directory / 'resources.json', {'runner_wall_seconds': time.monotonic() - started,
                   'process_wall_seconds': result['wall_seconds'], 'historical_training_cost': 'accounted separately by simulator'})
        return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'enter']); parser.add_argument('directory')
    parser.add_argument('--uid', type=int); parser.add_argument('--python')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare_runtime(args.directory)
    else:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        restrict_syscalls()
        # Environment is an allowlist; -s/-S also excludes user and site packages.
        # Unlike -I, these preserve the controller's fixed PYTHONHASHSEED.
        enter(args.directory, [args.python, '-s', '-S', '/workspace/policy.py'], uid=args.uid)


if __name__ == '__main__':
    main()
