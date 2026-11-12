"""Verify archive hashes and historical record counts without original data."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'manifest.json').read_text())
actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p!=root/'manifest.json'}
assert actual==set(manifest['files']), 'Missing or unindexed files'
for name,item in manifest['files'].items():
    raw=(root/name).read_bytes()
    assert len(raw)==item['bytes'],name
    assert hashlib.sha256(raw).hexdigest()==item['sha256'],name
rows=json.loads((root/'iterations.json').read_text());summary=json.loads((root/'summary.json').read_text())
assert len(rows)==summary['training_attempts']==1386
assert sum(x['rounds'] for x in summary['lanes'])==summary['rounds']==356
assert len(json.loads((root/'calls.json').read_text()))==summary['agent_calls']==413
for row in rows:
    assert hashlib.sha256((root/row['model_path']).read_bytes()).hexdigest()==row['source_sha256'],row['id']
print(json.dumps({'verified_files':len(manifest['files']),'rounds':356,'training_attempts':1386,'agent_calls':413}))
