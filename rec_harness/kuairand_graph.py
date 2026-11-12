"""Append-only research facts and bounded evidence-closed views for KuaiRand."""
from .core import ContractError, canonical, digest
from .runtime import node, edge
from .retrieval import select, revision_visibility


def view(store, task, protocol, query='', limit=96):
    snapshot = store.snapshot()
    nodes = {n['id']: n for n in snapshot['nodes'] if n['attrs']['task_id'] == task
             and n['attrs']['protocol_hash'] == protocol
             and n['attrs'].get('partition', 'development') == 'development'}
    while True:
        bad = {k for k, n in nodes.items() if not set(n['attrs'].get('evidence_refs', [])) <= set(nodes)}
        if not bad: break
        for k in bad: nodes.pop(k)
    edges = [e for e in snapshot['edges'] if {e['source'], e['target'], *e['attrs']['evidence_refs']} <= set(nodes)]
    counter = {k for k, n in nodes.items() if n['type'] == 'Failure' or n['attrs'].get('counterevidence')}
    counter.update(e['source'] for e in edges if e['relation'] in {'CONTRADICTS', 'SUPERSEDES'})
    selected, audit = select(nodes, edges, counter, 'hypothesis', query, limit, canonical)
    visible = set(selected)
    # Close recursively, never silently detach an explanation from its basis.
    for key in list(selected):
        closure, pending = {key}, [key]
        while pending:
            refs = set(nodes[pending.pop()]['attrs'].get('evidence_refs', [])) - closure
            closure.update(refs); pending.extend(refs)
        if len(visible | closure) <= limit: visible |= closure
        else: visible.discard(key)
    while True:
        bad = {k for k in visible if not set(nodes[k]['attrs'].get('evidence_refs', [])) <= visible}
        if not bad: break
        visible -= bad
    return {'graph_snapshot_id': snapshot['version'], 'graph_hash': snapshot['hash'],
        'source_event_head': store.verify()['head'], 'nodes': [nodes[k] for k in sorted(visible)],
        'edges': [e for e in edges if {e['source'], e['target'], *e['attrs']['evidence_refs']} <= visible],
        'revision_visibility': revision_visibility(nodes, edges, sorted(visible)),
        'retrieval': {**audit, 'omitted_node_count': len(nodes)-len(visible)},
        'qualification': 'Development-only visible snapshot; single-seed mechanisms remain provisional.'}


def append(store, nodes, edges=()):
    existing = {n['id']: n for n in store.snapshot()['nodes']}
    for n in nodes:
        if n['id'] in existing and existing[n['id']] != n:
            raise ContractError('research graph records require new revision identities')
        if n['type'] == 'Claim' and n['attrs'].get('status') != 'provisional':
            raise ContractError('single-seed interpretation must remain provisional')
    prior_edges = {e['id']: e for e in store.snapshot()['edges']}
    new_edges = []
    for e in edges:
        old = prior_edges.get(e['id'])
        if old:
            if any(e[k] != old[k] for k in ('source', 'target', 'relation')) or e['attrs']['evidence_refs'] != old['attrs']['evidence_refs']:
                raise ContractError('graph relationship revisions require new identities')
        else: new_edges.append(e)
    new_nodes = [n for n in nodes if n['id'] not in existing]
    if not new_nodes and not new_edges: return store.snapshot()
    return store.patch({'base_version': store.version, 'nodes': new_nodes, 'edges': new_edges})


def observation(store, identity, result, task, protocol, artifact_refs):
    # Only host result fields; no free-form model output is promoted to fact.
    valid = result.get('kind') == 'kuairand_research_validation' and result.get('epochs') == 101
    attrs = {'partition': 'development', 'valid': valid, 'result': result,
             'artifact_refs': artifact_refs, 'seed': 42, 'independent_unit': False,
             'mechanism_status': 'single_seed_provisional'}
    return append(store, [node(identity, 'Experiment' if valid else 'Failure', attrs, task, protocol)])


def pareto(records):
    eligible = [r for r in records if r.get('valid') and r.get('result', {}).get('kind') == 'kuairand_research_validation']
    def vector(r):
        v = r['result']
        return (-v['metrics']['ndcg@10'], v['training_gpu_seconds'], v['scoring_seconds'])
    return sorted(r['id'] for r in eligible if not any(
        all(a <= b for a, b in zip(vector(other), vector(r)))
        and any(a < b for a, b in zip(vector(other), vector(r)))
        for other in eligible if other['id'] != r['id']))


def interaction(none, only_a, only_b, both):
    return both - only_a - only_b + none
