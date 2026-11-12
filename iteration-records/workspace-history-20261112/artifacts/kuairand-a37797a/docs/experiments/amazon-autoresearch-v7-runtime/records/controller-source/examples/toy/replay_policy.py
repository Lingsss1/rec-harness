"""Deterministic depth-first continuation; choose best visible successful model."""
import json
import sys

request = json.load(sys.stdin)
view = request['view']
memory = request['memory'] or {'exhausted': []}
for item in (request['previous_outcome'] or {}).get('outcomes', []):
    if item['status'] in {'exhausted', 'unsupported'}:
        memory['exhausted'].append(item['parent'])
legal = [p for p in view['legal_parents'] if p not in memory['exhausted']]
scored = [(o['score'], n) for n, o in view['observations'].items()
          if o.get('status') == 'success' and o.get('score') is not None]
best = (max(scored) if request['direction'] == 'maximize' else min(scored))[1] if scored else None
print(json.dumps({'parents': legal[-1:] if legal else [], 'action_kind': 'continue',
                  'select': best, 'memory': memory, 'rationale': 'Continue the last available frontier; stop when recorded history is exhausted.'}))
