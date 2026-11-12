import json
import os
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).absolute().parent
PROJECT=ROOT.parents[1]
read=lambda p:json.loads(p.read_text())
deadline=read(ROOT/'budget.json')['deadline_unix']
while time.time()<deadline:
    path=ROOT/'enhanced/knowledge-handoff.json'
    if path.exists():
        marker=read(path)
        if marker.get('resumed_unix'):raise RuntimeError('Checkpoint already consumed')
        subprocess.run([str(PROJECT/'benchmark_res/.venv/bin/python'),str(ROOT/'apply-knowledge-repair.py'),'enhanced'],cwd=PROJECT,check=True)
        print('Both arms have resumed on knowledge-access-v2',flush=True)
        break
    status=read(ROOT/'enhanced/status.json')
    if status['phase'] in ['completed','failed','failed_before_assessment']:raise RuntimeError('Peer stopped before handoff: '+str(status))
    time.sleep(0.2)
else:raise TimeoutError('Original deadline reached')
