"""Linux process identity checks for conservative recovery after controller loss."""
from pathlib import Path
import os

from .core import ContractError, read_json


def identity(pid):
    if int(Path('/proc/self/stat').read_text().split(' ', 1)[0]) != os.getpid():
        return {'pid': int(pid), 'observable': False}
    # comm can contain spaces and parentheses; fields after its final ')' are stable.
    fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': int(pid), 'start_ticks': int(fields[19]), 'session': int(fields[3]),
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'state': fields[0]}


def session_live(record):
    """PID reuse cannot prove liveness; descendants in the original session can."""
    if record.get('observable') is False or identity(os.getpid()).get('observable') is False:
        raise ContractError('process identity unavailable in this PID namespace')
    if record['boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        return False
    try:
        leader = identity(record['pid'])
    except FileNotFoundError:
        leader = None
    if leader and leader['start_ticks'] != record['start_ticks']:
        return False
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            current = identity(int(path.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
        if current['session'] == record['session'] and current['state'] not in {'Z', 'X'}:
            return True
    return False


def assert_stopped(directory):
    """Fail closed if a launcher died between starting and journaling a worker."""
    directory = Path(directory)
    for path in directory.rglob('stdout.jsonl'):
        if not (path.parent / 'launch.json').exists() and not (path.parent / 'process.json').exists():
            raise ContractError(f'legacy worker identity unresolved: {path.parent}')
    for path in directory.rglob('launch.json'):
        record = read_json(path)
        if (path.parent / 'process.json').exists():
            # Receipt is written only after process-group cleanup and wait().
            continue
        if 'worker' in record:
            if session_live(record['worker']):
                raise ContractError(f'worker session still live: {path.parent}')
        elif not (path.parent / 'process.json').exists():
            raise ContractError(f'worker launch identity unresolved: {path.parent}')
