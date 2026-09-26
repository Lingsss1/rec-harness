"""Typed legacy projections with lossless originals and no inherited evidence gates."""
import csv
import io
import json
import math
import tempfile
from pathlib import Path

from .core import Artifacts, ContractError, digest, public_only, read_json, safe_file
from .runtime import node, edge

FILES = ('results.tsv', 'episodes.jsonl', 'failures.jsonl', 'lessons.json', 'strategy.json')


def scalar(value):
    if value is None or isinstance(value, bool): return None
    try: result = float(value)
    except (ValueError, TypeError): return None
    return result if math.isfinite(result) else None


def metrics(value):
    """Only explicitly named aggregate numbers; the raw record is retained too."""
    result = {}
    supplied = value.get('metrics')
    if isinstance(supplied, dict):
        primary = supplied.get('primary')
        if isinstance(primary, dict) and isinstance(primary.get('name'), str):
            numeric = scalar(primary.get('value'))
            if numeric is not None: result[primary['name']] = numeric
        for key, number in supplied.items():
            if key not in {'primary', 'guardrails'} and scalar(number) is not None:
                result[key] = scalar(number)
        guardrails = supplied.get('guardrails', {})
        if isinstance(guardrails, dict):
            result.update({key: scalar(number) for key, number in guardrails.items() if scalar(number) is not None})
    for key in ('primary_metric', 'primary_delta', 'peak_vram_mb', 'runtime_sec'):
        if key in value and scalar(value[key]) is not None: result[key] = scalar(value[key])
    return result


def import_legacy(store, source, task_id, protocol_hash='unknown'):
    source = Path(source).resolve()
    if not source.is_dir(): raise ContractError('legacy source directory does not exist')
    snapshot = store.snapshot()
    known = {n['id']: n for n in snapshot['nodes']}
    known_edge_records = {e['id']:e for e in snapshot['edges']}
    known_edges = set(known_edge_records)
    archive = Artifacts(store.root.parent / 'artifacts/objects')
    records, source_refs = [], {}
    for name in FILES:
        if not (source / name).exists(): continue
        path = safe_file(source, name)
        data = path.read_bytes()
        if name.endswith('.tsv'):
            values = list(csv.DictReader(io.StringIO(data.decode()), delimiter='\t'))
        elif name.endswith('.jsonl'):
            values = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
        else:
            content = json.loads(data); public_only(content)
            if name == 'lessons.json' and isinstance(content, dict) and isinstance(content.get('lessons'), list):
                values = content['lessons']
            else: values = content if isinstance(content, list) else [content]
        public_only(values)
        # Archive precisely the bytes parsed, even if the legacy writer appends or
        # replaces its file between parsing and archival.
        with tempfile.NamedTemporaryFile(dir=archive.root) as frozen:
            frozen.write(data); frozen.flush()
            source_ref = archive.put(frozen.name)
        source_refs[name] = source_ref
        for index, value in enumerate(values):
            old_id = 'legacy-' + digest({'kind': name, 'value': value})[:32]
            prior = known.get(old_id)
            if prior and (prior['attrs']['task_id'], prior['attrs']['protocol_hash']) == (task_id, protocol_hash):
                raw_id = old_id
            else:
                raw_id = 'legacy-' + digest([task_id, protocol_hash, name, value])[:32]
            records.append((name, value, raw_id, source_ref, index))

    fresh, relations, raw_ids = {}, {}, set()
    common = {'status': 'legacy_unverified', 'replay_qualification': 'audit_only', 'qualification': 'unverified_source_summary'}
    def add(key, kind, attrs):
        candidate = node(key, kind, {**common, **attrs}, task_id, protocol_hash, 'legacy_unverified')
        if key not in known and key not in fresh:
            fresh[key] = candidate
        elif kind == 'Claim' and key in known and known[key] != candidate:
            # Resolve newly available source references; keep the original source
            # bytes and earlier graph snapshot, and never upgrade scientific status.
            fresh[key] = candidate
        return key
    def link(source_id, target_id, kind, suffix, source_record=None):
        key = 'legacy-edge-' + digest([task_id, protocol_hash, source_id, target_id, suffix])[:32]
        existing=relations.get(key) or known_edge_records.get(key)
        if existing:
            if source_record and source_record not in existing['attrs'].get('legacy_source_record_ids',[]):
                relations[key]={**existing,'attrs':{**existing['attrs'],
                    'legacy_source_record_ids':sorted(set(existing['attrs'].get('legacy_source_record_ids',[])+[source_record])),
                    'revision_id':str(store.version+1),
                    'qualification':'reported parent relation; no weight transfer or metric comparability asserted'}}
            return
        item = edge(key, source_id, target_id, kind, [], task_id, store.version + 1)
        item['attrs'].update(origin='legacy_unverified', status='legacy_unverified')
        if source_record:
            item['attrs'].update(legacy_source_record_ids=[source_record],
                qualification='reported parent relation; no weight transfer or metric comparability asserted')
        relations[key] = item
    def external_identity(value):
        supplied = value.get('experiment_id') or value.get('run_id')
        return str(supplied) if isinstance(supplied, (str, int)) and supplied != '' else None
    def experiment(external_id):
        key = 'legacy-exp-' + digest([task_id, protocol_hash, external_id])[:32]
        return add(key, 'Experiment', {'legacy_experiment_id': external_id,
                   'independent_run_id': None, 'valid': False, 'effect_gate_passed': False,
                   'seeds': None, 'protocol_verified': False})

    anchors = {n['attrs']['legacy_experiment_id']: n['id'] for n in known.values()
               if n['type'] == 'Experiment' and n['attrs'].get('origin') == 'legacy_unverified'
               and (n['attrs']['task_id'], n['attrs']['protocol_hash']) == (task_id, protocol_hash)
               and 'legacy_experiment_id' in n['attrs']}
    for name, value, raw_id, source_ref, index in records:
        raw_ids.add(raw_id)
        add(raw_id, 'LegacyRecord', {'source_file': name, 'source_hash': source_ref['sha256'],
            'source_file_ref': source_ref, 'source_record_index': index, 'original': value, 'seeds': None})
        if isinstance(value, dict) and name in {'results.tsv', 'episodes.jsonl', 'failures.jsonl'}:
            external_id = external_identity(value)
            if external_id is not None: anchors[external_id] = experiment(external_id)
            elif name in {'results.tsv', 'episodes.jsonl'}:
                anchors[raw_id] = experiment('unidentified-record:' + raw_id)

    for name, value, raw_id, source_ref, index in records:
        if not isinstance(value, dict): continue
        external_id = external_identity(value)
        anchor = anchors.get(external_id) if external_id is not None else anchors.get(raw_id)
        key = 'legacy-view-' + digest([task_id, protocol_hash, name, value])[:32]
        if name in {'results.tsv', 'episodes.jsonl'}:
            add(key, 'Observation', {'source_record_id': raw_id, 'metrics': metrics(value),
                'reported_validity': value.get('validity'), 'reported_status': value.get('status', value.get('decision')),
                'reported_budget': value.get('budget'), 'protocol_verified': False,
                'independent_run_id': None, 'valid': False})
            if anchor: link(anchor, key, 'PRODUCES', 'summary')
            parent = value.get('parent_experiment_id', value.get('parent_id'))
            if parent is not None and str(parent) in anchors and anchors[str(parent)] != anchor:
                link(anchor, anchors[str(parent)], 'DERIVED_FROM', 'reported-parent',raw_id)
        elif name == 'failures.jsonl':
            add(key, 'Failure', {'source_record_id': raw_id,
                'failure_kind': value.get('kind', 'unknown'), 'signature': value.get('signature'),
                'scope': value.get('scope', 'unknown'), 'reproduction': value.get('reproduction'),
                'reported_likely_cause': value.get('likely_cause'), 'avoid': value.get('avoid'),
                'execution_evidence_only': True, 'superseded_by_external_id': value.get('superseded_by')})
            if anchor: link(anchor, key, 'HAS_FAILURE', 'failure')
        elif name == 'lessons.json':
            supports, counters = value.get('supporting_episode_ids', []), value.get('counterevidence_episode_ids', [])
            if not isinstance(supports, list) or not isinstance(counters, list):
                raise ContractError('legacy lesson references must be arrays')
            resolved = [anchors[str(r)] for r in supports + counters if str(r) in anchors]
            add(key, 'Claim', {'source_record_id': raw_id, 'statement': value.get('statement', value.get('lesson', 'unknown')),
                'scope': value.get('scope', 'unknown'), 'evidence_refs': sorted(set(resolved)),
                'reported_status': value.get('status'), 'reported_confidence': value.get('confidence'),
                'unresolved_external_refs': [str(r) for r in supports + counters if str(r) not in anchors],
                'falsification_condition': value.get('falsification_condition')})
            for relation, refs in [('SUPPORTS', supports), ('CONTRADICTS', counters)]:
                for ref in refs:
                    if str(ref) in anchors: link(anchors[str(ref)], key, relation, relation)
        else:
            add(key, 'Context', {'source_record_id': raw_id, 'context_kind': 'legacy_strategy_working_state',
                                'working_state': value, 'historical_evidence': False})
        link(key, raw_id, 'DERIVED_FROM', 'lossless-source')

    if fresh or relations:
        store.patch({'base_version': store.version, 'nodes': list(fresh.values()), 'edges': list(relations.values())})
    manifest = {'task_id': task_id, 'protocol_hash': protocol_hash, 'files': source_refs}
    manifest_hash = digest(manifest)
    if not any(e['kind'] == 'legacy_sources_archived' and e['payload']['manifest_hash'] == manifest_hash for e in store.events()):
        store.event('legacy_sources_archived', {'manifest_hash': manifest_hash, **manifest})
    return {'imported': len(raw_ids & fresh.keys()), 'already_present': len(raw_ids - fresh.keys()),
            'typed_nodes_added': sum(n['type'] != 'LegacyRecord' and k not in known for k, n in fresh.items()),
            'typed_nodes_updated': sum(k in known for k in fresh),
            'relations_added': sum(key not in known_edges for key in relations),
            'relations_updated': sum(key in known_edges for key in relations), 'source_manifest_hash': manifest_hash}
