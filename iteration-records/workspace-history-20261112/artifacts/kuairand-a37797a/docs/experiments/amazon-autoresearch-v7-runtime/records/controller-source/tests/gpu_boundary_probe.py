"""Reviewed standalone boundary probe. Run only in a prepared test jail."""
import json
import os
from pathlib import Path
import socket
import stat
import sys

checks = {}
checks['nonprivileged'] = os.getuid() != 0
checks['environment_removed'] = 'REC_HARNESS_SYNTHETIC_CANARY' not in os.environ
checks['host_canary_invisible'] = not Path(sys.argv[1]).exists()
checks['host_proc_invisible'] = not Path('/proc/1/root').exists()
checks['other_gpu_invisible'] = not Path('/dev/nvidia1').exists()
for name, action in [('network', socket.socket), ('regain_root', lambda: os.setuid(0)),
                     ('create_device', lambda: os.mknod('/workspace/forbidden-device', stat.S_IFCHR | 0o600, os.makedev(195, 1)))]:
    try:
        value = action()
        if name == 'network': value.close()
        checks[name] = False
    except PermissionError:
        checks[name] = True
try:
    pid = os.fork()
    if pid == 0: os._exit(0)
    os.waitpid(pid, 0)
    checks['fork'] = False
except PermissionError:
    checks['fork'] = True
try:
    Path('/inputs/gpu-boundary-probe.py').write_text('tampered')
    checks['entry_readonly'] = False
except PermissionError:
    checks['entry_readonly'] = True
print(json.dumps({'boundary_checks': checks}), flush=True)
assert all(checks.values()), checks
