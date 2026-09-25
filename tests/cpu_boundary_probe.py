"""Adversarial import fixture for isolated_contract; intentionally has no model."""
import json
import os
from pathlib import Path
import socket
import stat

checks = {
    'nonprivileged': os.getuid() != 0,
    'host_proc_invisible': not Path('/proc/1/root').exists(),
    'no_gpu_devices': not list(Path('/dev').glob('nvidia*')),
    'host_study_invisible': not Path('/home/liuzhishan/lhr/rec-harness-runs/amazon-nextitnet-v5-20260923-r2/study.json').exists(),
    'credentials_env_absent': not any(k.endswith('_API_KEY') or k.endswith('_TOKEN') for k in os.environ),
}
for name, action in [
    ('network_denied', socket.socket),
    ('regain_root_denied', lambda: os.setuid(0)),
    ('device_creation_denied', lambda: os.mknod('/workspace/node', stat.S_IFCHR | 0o600, os.makedev(195, 0))),
    ('source_write_denied', lambda: Path('/inputs/model.py').write_text('changed')),
]:
    try:
        value = action()
        if name == 'network_denied': value.close()
        checks[name] = False
    except PermissionError:
        checks[name] = True
try:
    pid = os.fork()
    if pid == 0: os._exit(0)
    os.waitpid(pid, 0)
    checks['fork_denied'] = False
except PermissionError:
    checks['fork_denied'] = True
Path('/output/boundary.json').write_text(json.dumps(checks, sort_keys=True))
assert all(checks.values()), checks
# A clean process exit without inspector evidence must not admit a candidate.
raise SystemExit(0)
