"""Read-only experiment status; safe while the controller is running."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    a = p.parse_args()
    if (a.run / 'summary.json').exists():
        print((a.run / 'summary.json').read_text())
        return
    jobs, charged = [], 0
    for d in sorted((a.run / 'trials').glob('*')):
        if (d / 'receipt.json').exists():
            receipt = json.loads((d / 'receipt.json').read_text())
            charged += receipt['allocated_gpu_seconds']
            job = {k: receipt[k] for k in ('id', 'status', 'metrics', 'allocated_gpu_seconds') if k in receipt}
        else:
            job = {'id': d.name, 'status': 'started_without_receipt'}
            if (d / 'pid.json').exists():
                pid = json.loads((d / 'pid.json').read_text())['pid']
                job.update(pid=pid, process_exists=Path(f'/proc/{pid}').exists())
        if (d / 'curve.jsonl').exists():
            rows = (d / 'curve.jsonl').read_text().splitlines()
            if rows:
                job['latest_validation'] = json.loads(rows[-1])
        jobs.append(job)
    events = (a.run / 'events.jsonl').read_text().splitlines() if (a.run / 'events.jsonl').exists() else []
    print(json.dumps({'status': 'running_or_interrupted', 'settled_gpu_seconds': charged,
        'latest_event': json.loads(events[-1]) if events else None, 'trials': jobs}, indent=2))


if __name__ == '__main__':
    main()
