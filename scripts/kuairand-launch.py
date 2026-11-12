"""Detach a reviewed KuaiRand controller and retain its launch identity/log."""
import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rec_harness.core import write_json
from rec_harness.process_identity import identity

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('kind', choices=['baseline', 'autoresearch'])
p.add_argument('config')
p.add_argument('output', type=Path)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
argv = ([sys.executable, '-m', 'rec_harness.kuairand_baseline', a.config] if a.kind == 'baseline' else
        [sys.executable, '-m', 'rec_harness', 'autoresearch', str(a.output/'campaign'), '--config', a.config])
with (a.output/'launch.log').open('a') as log:
    process = subprocess.Popen(argv, cwd=Path(__file__).resolve().parents[1],
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
write_json(a.output/'launch.json', {'argv': argv, 'worker': identity(process.pid)})
print(process.pid)
