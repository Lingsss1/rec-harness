"""Fail-closed descriptor selection for a preinitialized GPU launcher.

Only a trusted launcher may call this before executing model code. This policy
does not itself chroot, drop privileges, install seccomp, or certify a driver.
"""
import fcntl
import os
import stat

from .core import ContractError
from .isolation import gpu_device_paths


def inventory():
    result = {}
    for name in os.listdir('/proc/self/fd'):
        fd = int(name)
        if fd < 3:
            continue
        try:
            info = os.fstat(fd)
            target = os.readlink('/proc/self/fd/' + name)
            access = fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE
        except OSError:
            continue  # transient descriptor used by the directory listing
        kind = ('pipe' if stat.S_ISFIFO(info.st_mode) else
                'socket' if stat.S_ISSOCK(info.st_mode) else
                'device' if stat.S_ISCHR(info.st_mode) else
                'eventfd' if target == 'anon_inode:[eventfd]' else 'other')
        result[fd] = {'kind': kind, 'target': target, 'inode': info.st_ino,
                      'filesystem_device': info.st_dev, 'access': access}
    return result


def select_descriptors(before_cuda, after_cuda, device_ids):
    """Pipes must be newly created and have both endpoints in this process."""
    allowed = set(gpu_device_paths(device_ids))
    if before_cuda:
        raise ContractError('GPU launcher inherited unexpected non-stdio descriptors')
    keep, close, pipes = [], [], {}
    for fd, info in after_cuda.items():
        if type(fd) is not int or fd < 3:
            raise ContractError('descriptor inventory must exclude standard streams')
        kind = info['kind']
        if kind == 'pipe':
            identity = (info['filesystem_device'], info['inode'])
            pipes.setdefault(identity, []).append(info['access'])
            keep.append(fd)
        elif kind == 'eventfd' or (kind == 'device' and info['target'] in allowed):
            keep.append(fd)
        else:
            close.append(fd)
    if any(sorted(ends) != [os.O_RDONLY, os.O_WRONLY] for ends in pipes.values()):
        raise ContractError('CUDA pipe lacks exactly one local read and write endpoint')
    return {'retain': sorted(keep), 'close': sorted(close), 'pipe_pairs': len(pipes),
            'device_paths': sorted(allowed),
            'scope': 'local endpoint ownership; does not prove no other process holds a duplicate'}
