"""Small host utilities, shared by acquisition, fixed runs and the dashboard."""
import hashlib
import json
from pathlib import Path
import time
import uuid


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def aggregate(tasks, labels, allow_undefined=False):
    if set(tasks) != set(labels) or len(labels) < 2:
        raise ValueError('Scoring requires every native task exactly once')
    aucs = [tasks[label]['auc'] for label in labels]
    losses = [tasks[label]['logloss'] for label in labels]
    if not allow_undefined and any(value is None for value in aucs + losses):
        raise ValueError('An undefined task must not be silently omitted')
    return {'auc': sum(aucs) / len(aucs) if all(x is not None for x in aucs) else None,
            'logloss': sum(losses) / len(losses) if all(x is not None for x in losses) else None,
            'aggregation': 'equal mean across every native task', 'tasks': list(labels)}
