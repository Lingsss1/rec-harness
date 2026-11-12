"""Bounded collection of flat training outputs after controller-owned completion."""
import os
from pathlib import Path
import stat
import tempfile

from .core import ContractError, read_json
from .process_identity import assert_stopped


def collect_outputs(output, process_directory, artifacts, *, max_bytes=2 * 1024**3, max_files=32):
    if type(max_bytes) is not int or max_bytes < 1 or type(max_files) is not int or max_files < 1:
        raise ContractError('positive output limits required')
    process_directory = Path(process_directory)
    completion = process_directory / 'process.json'
    if not completion.is_file() or read_json(completion).get('status') not in {'success', 'crash', 'timeout'}:
        raise ContractError('controller-owned terminal process receipt required')
    assert_stopped(process_directory)
    output = Path(output).absolute()
    if any(p.is_symlink() for p in [output, *output.parents]):
        raise ContractError('output path contains a symlink')
    root_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    refs, total = {}, 0
    try:
        names = sorted(os.listdir(root_fd))
        if len(names) > max_files:
            raise ContractError('output file count exceeds limit')
        with tempfile.TemporaryDirectory(prefix='rec-harness-output-') as scratch:
            for name in names:
                info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ContractError('output must be a regular file without links')
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root_fd)
                with os.fdopen(fd, 'rb') as source:
                    before = os.fstat(source.fileno())
                    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                        raise ContractError('output changed to a nonregular or linked file')
                    if total + before.st_size > max_bytes:
                        raise ContractError('output bytes exceed limit')
                    staged = Path(scratch) / 'artifact'
                    copied = 0
                    with staged.open('wb') as destination:
                        while True:
                            block = source.read(min(1024**2, max_bytes - total - copied + 1))
                            if not block:
                                break
                            copied += len(block)
                            if total + copied > max_bytes:
                                raise ContractError('output bytes exceed limit')
                            destination.write(block)
                    after = os.fstat(source.fileno())
                    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns) or copied != before.st_size:
                        raise ContractError('output changed during collection')
                refs[name] = artifacts.put(staged)
                total += copied
        return {'artifacts': refs, 'total_bytes': total, 'file_count': len(refs),
                'process_status': read_json(completion)['status']}
    finally:
        os.close(root_fd)
