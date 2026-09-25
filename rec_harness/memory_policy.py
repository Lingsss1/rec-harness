"""Frozen, bounded source-only knowledge packages; transfer is not target evidence."""
from pathlib import Path
import fcntl

from .core import ContractError, canonical, digest, identifier, public_only, read_json, write_json
from .runtime import node, edge

SCHEMA = 'memory-policy-v3'
ROOT_TYPES = {'Claim', 'GapHypothesis', 'Method', 'Source'}
ALLOWED_TYPES = ROOT_TYPES | {'Experiment', 'Observation', 'Failure', 'Component', 'Context', 'ModelVersion', 'Proposal'}
TEXT_FIELDS = set('task_id protocol_hash origin status statement mechanism cost title url citation '
                  'independent_run_id dataset model_version failure_kind suspected_capability '
                  'cheapest_discriminating_test insertion_contract component_role shape_contract code_hash'.split())
LIST_FIELDS = set('evidence_refs observation_refs evidence_dependency_ids prerequisites insertion_points tensor_constraints '
                  'sources alternative_explanations predicted_observations limitations'.split())
SCALAR_FIELDS = {'seed', 'valid', 'effect_gate_passed', 'mechanism_control', 'counterevidence'}
SCOPE_FIELDS = {'task_id', 'model_version', 'dataset', 'protocol_hash', 'condition', 'split'}
RELATIONS = {'SUPPORTS', 'CONTRADICTS', 'ALTERNATIVE_TO', 'SUPPORTED_BY', 'BASED_ON', 'LOCATED_AT', 'TARGETS', 'USES'}


def project_node(original):
    if original['type'] not in ALLOWED_TYPES:
        raise ContractError('node type not transferable')
    attrs = original['attrs']
    if attrs['origin'] == 'legacy_unverified' or any(k in attrs for k in ('source_package_hash', 'source_node', 'transfer_status')):
        raise ContractError('unverified or transferred provenance cannot be exported')
    projected = {}
    for key in TEXT_FIELDS & attrs.keys():
        if attrs[key] is None:
            continue
        if not isinstance(attrs[key], str) or len(attrs[key]) > 8192:
            raise ContractError('transfer text field invalid')
        projected[key] = attrs[key]
    for key in LIST_FIELDS & attrs.keys():
        if not isinstance(attrs[key], list) or any(not isinstance(v, str) or len(v) > 2048 for v in attrs[key]):
            raise ContractError('transfer list field invalid')
        projected[key] = attrs[key]
    for key in SCALAR_FIELDS & attrs.keys():
        if attrs[key] is not None and not isinstance(attrs[key], (bool, int, float)):
            raise ContractError('transfer scalar field invalid')
        projected[key] = attrs[key]
    if 'scope' in attrs:
        scope = attrs['scope']
        if isinstance(scope, str):
            if len(scope) > 8192: raise ContractError('scope too long')
            projected['scope'] = scope
        elif isinstance(scope, dict):
            projected['scope'] = {k: v for k, v in scope.items() if k in SCOPE_FIELDS and isinstance(v, (str, int, float, bool))}
        else:
            raise ContractError('invalid scope')
    if 'metrics' in attrs:
        metrics = attrs['metrics']
        if metrics is not None:
            if not isinstance(metrics, dict) or any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in metrics.values()):
                raise ContractError('only scalar aggregate metrics can transfer')
            projected['metrics'] = metrics
    result = {'id': identifier(original['id']), 'type': original['type'], 'attrs': projected}
    public_only(result)
    return result


def _size(nodes, edges):
    return len(canonical({'nodes': nodes, 'edges': edges}).encode())


def export_memory(store, path, *, task_id, protocol_hash, version=None, limit=1000,
                  cutoff_seq=None, max_bytes=1048576):
    if type(limit) is not int or not 1 <= limit <= 10000 or type(max_bytes) is not int or not 1024 <= max_bytes <= 1048576:
        raise ContractError('invalid frozen transfer capacity')
    store.verify()
    events = store.events()
    cutoff_seq = len(events) if cutoff_seq is None else cutoff_seq
    if type(cutoff_seq) is not int or not 0 <= cutoff_seq <= len(events):
        raise ContractError('invalid source cutoff sequence')
    eligible = {0: None}
    for event in events[:cutoff_seq]:
        if event['kind'] in {'graph_committed', 'graph_rollback'}:
            eligible[event['payload']['version']] = event
    version = max(eligible) if version is None else version
    if version not in eligible:
        raise ContractError('graph version is after source cutoff')
    snapshot = store.snapshot(version)
    scoped = {n['id']: n for n in snapshot['nodes']
              if (n['attrs']['task_id'], n['attrs']['protocol_hash']) == (task_id, protocol_hash)}
    projected = {}
    for key, original in scoped.items():
        try: projected[key] = project_node(original)
        except ContractError: continue
    relations = [e for e in snapshot['edges'] if e['relation'] in RELATIONS
                 and (e['source'] in scoped or e['target'] in scoped)]
    chosen, skipped = set(), {'ineligible_or_incomplete': sum(n['type'] in ROOT_TYPES and k not in projected for k, n in scoped.items()), 'capacity': 0}
    roots = sorted((k for k, n in projected.items() if n['type'] in ROOT_TYPES),
                   key=lambda k: (projected[k]['attrs'].get('status') not in {'disputed', 'demoted'}, k))
    def contents(ids):
        nodes = [projected[k] for k in sorted(ids)]
        edges = [{'id': e['id'], 'source': e['source'], 'target': e['target'], 'relation': e['relation'],
                  'evidence_refs': e['attrs']['evidence_refs']} for e in relations
                 if e['source'] in ids and e['target'] in ids]
        return nodes, sorted(edges, key=lambda e: e['id'])
    for root in roots:
        group, pending = set(), {root}
        while pending:
            key = pending.pop()
            if key in group: continue
            group.add(key)
            if key not in projected: continue
            attrs = projected[key]['attrs']
            pending.update(set(attrs.get('evidence_refs', [])) - group)
            pending.update(set(attrs.get('observation_refs', [])) - group)
            # evidence_dependency_ids names replay attempts, not necessarily graph
            # nodes. Preserve those provenance labels without fabricating nodes.
            for relation in relations:
                if key in {relation['source'], relation['target']}:
                    pending.update(({relation['source'], relation['target']} | set(relation['attrs']['evidence_refs'])) - group)
        if not group.issubset(projected):
            skipped['ineligible_or_incomplete'] += 1; continue
        candidate = chosen | group
        nodes, edges = contents(candidate)
        if len(candidate) > limit or _size(nodes, edges) > max_bytes:
            skipped['capacity'] += 1; continue
        chosen = candidate
    nodes, edges = contents(chosen)
    cutoff = events[cutoff_seq - 1] if cutoff_seq else None
    body = {'schema': SCHEMA, 'source_task': task_id, 'source_protocol': protocol_hash,
            'source_graph_version': version, 'source_graph_hash': snapshot['hash'],
            'source_cutoff': {'seq': cutoff_seq, 'hash': cutoff['hash'] if cutoff else '0' * 64,
                              'created_at': cutoff['created_at'] if cutoff else None},
            'capacity': {'max_nodes': limit, 'max_payload_bytes': max_bytes, 'payload_bytes': _size(nodes, edges)},
            'nodes': nodes, 'edges': edges, 'omitted_roots': skipped,
            'policy': 'source-only-projected-fields-complete-evidence-and-counterevidence-closure'}
    public_only(body)
    package = {**body, 'hash': digest(body)}
    write_json(path, package)
    return package


def validate_package(package):
    fields = {'schema', 'source_task', 'source_protocol', 'source_graph_version', 'source_graph_hash',
              'source_cutoff', 'capacity', 'nodes', 'edges', 'omitted_roots', 'policy', 'hash'}
    if set(package) != fields:
        raise ContractError('unexpected memory package fields')
    if package.get('schema') not in {'memory-policy-v2', SCHEMA}:
        raise ContractError('unsupported memory schema; re-export with current policy')
    if digest({k: v for k, v in package.items() if k != 'hash'}) != package['hash']:
        raise ContractError('memory package hash mismatch')
    public_only(package)
    nodes, edges = package['nodes'], package['edges']
    capacity = package['capacity']
    if set(capacity) != {'max_nodes', 'max_payload_bytes', 'payload_bytes'} or any(type(v) is not int for v in capacity.values()):
        raise ContractError('invalid memory capacity declaration')
    if not 1 <= capacity['max_nodes'] <= 10000 or not 1024 <= capacity['max_payload_bytes'] <= 1048576:
        raise ContractError('invalid memory capacity limits')
    cutoff = package['source_cutoff']
    if set(cutoff) != {'seq', 'hash', 'created_at'} or type(cutoff['seq']) is not int or cutoff['seq'] < 0:
        raise ContractError('invalid source cutoff')
    if not isinstance(cutoff['hash'], str) or len(cutoff['hash']) != 64 or (cutoff['created_at'] is not None and not isinstance(cutoff['created_at'], str)):
        raise ContractError('invalid source cutoff identity')
    if len(nodes) > capacity['max_nodes'] or _size(nodes, edges) != capacity['payload_bytes'] or _size(nodes, edges) > capacity['max_payload_bytes']:
        raise ContractError('memory capacity mismatch')
    by_id = {n['id']: n for n in nodes}
    if len(by_id) != len(nodes): raise ContractError('duplicate source identity')
    for item in nodes:
        if package['schema']=='memory-policy-v2' and item['type']=='Proposal':
            raise ContractError('Proposal requires memory-policy-v3')
        if project_node(item) != item:
            raise ContractError('unapproved transfer field')
        attrs = item['attrs']
        if (attrs['task_id'], attrs['protocol_hash']) != (package['source_task'], package['source_protocol']):
            raise ContractError('package includes another task or protocol')
        if not (set(attrs.get('evidence_refs', [])) | set(attrs.get('observation_refs', []))).issubset(by_id):
            raise ContractError('package evidence closure incomplete')
    for relation in edges:
        if package['schema']=='memory-policy-v2' and relation['relation']=='USES':
            raise ContractError('USES requires memory-policy-v3')
        if set(relation) != {'id', 'source', 'target', 'relation', 'evidence_refs'}:
            raise ContractError('unexpected memory relation fields')
        if relation['relation'] not in RELATIONS or not ({relation['source'], relation['target']} | set(relation['evidence_refs'])).issubset(by_id):
            raise ContractError('package relation closure incomplete')


def import_memory(store, path, target_task, target_protocol):
    with (store.root.parent / '.controller.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ContractError('target controller is running; transfer must be frozen first') from error
        return _import_memory(store, path, target_task, target_protocol)


def _import_memory(store, path, target_task, target_protocol):
    if Path(path).stat().st_size > 2 * 1048576:
        raise ContractError('memory package exceeds file size limit')
    package = read_json(path); validate_package(package)
    if package['source_task'] == target_task:
        raise ContractError('transfer target must be held out from source')
    snapshot = store.snapshot()
    if any(n['attrs']['task_id'] == package['source_task'] for n in snapshot['nodes']):
        raise ContractError('target overlay must be separate from source graph')
    target_identity = digest([package['hash'], target_task, target_protocol])
    prefix = 'transfer-' + target_identity[:16] + '-'
    known = {n['id'] for n in snapshot['nodes']}
    mapping = {n['id']: prefix + digest(n['id'])[:24] for n in package['nodes']}
    fresh = set(mapping.values()) - known
    started = any(e['kind'] == 'round_started' for e in store.events())
    if fresh and (started or any(n['attrs']['task_id'] == target_task and n['type'] in {'Experiment', 'Observation', 'DecisionTrace'} for n in snapshot['nodes'])):
        raise ContractError('freeze transfer before target research starts')
    counter = {e['source'] for e in package['edges'] if e['relation'] in {'CONTRADICTS', 'ALTERNATIVE_TO'}}
    nodes = []
    for original in package['nodes']:
        kind = 'Method' if original['type'] == 'Method' else ('GapHypothesis' if original['type'] in {'Claim', 'GapHypothesis'} else 'Source')
        nodes.append(node(mapping[original['id']], kind, {'source_node': original,
            'source_package_hash': package['hash'], 'source_cutoff': package['source_cutoff'],
            'status': 'unknown', 'counterevidence': original['id'] in counter or original['type'] == 'Failure' or original['attrs'].get('status') in {'disputed', 'demoted'},
            'transfer_status': 'analogy_requires_target_validation', 'scope': {'target_task': target_task},
            'evidence_refs': [], 'source_reference_ids': [mapping[r] for r in sorted(set(original['attrs'].get('evidence_refs', [])) | set(original['attrs'].get('observation_refs', [])))],
        }, target_task, target_protocol, 'agent_hypothesis'))
    nodes = [n for n in nodes if n['id'] not in known]
    edges = []
    for relation in package['edges']:
        item = edge(prefix + digest(relation['id'])[:24], mapping[relation['source']], mapping[relation['target']],
                    'DERIVED_FROM', [], target_task, store.version + 1)
        item['attrs'].update(source_relation=relation['relation'], transfer_status='source_relation_only')
        edges.append(item)
    if nodes:
        store.patch({'base_version': store.version, 'nodes': nodes, 'edges': edges})
    return {'imported': len(nodes), 'package_hash': package['hash'], 'target_overlay_version': store.version}
