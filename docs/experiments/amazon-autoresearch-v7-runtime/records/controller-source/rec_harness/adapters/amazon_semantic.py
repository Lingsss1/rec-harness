"""Semantic seeds over public development memory, with readout closure."""
from rec_harness.core import canonical, digest
from rec_harness.retrieval import select
from .amazon_memory import compact_development_context


def retrieve(history, entries, failures, protocol, seeds, query, encoder, limit):
    # Validate every readout before any text reaches the encoder. Compact first
    # so reports cannot be duplicated inside design entries during ranking.
    history, entries = compact_development_context(history, entries, protocol, seeds, history)
    nodes = {}
    kinds = {}
    reports = {}
    for kind, values, node_type in [('history', history, 'Experiment'),
                                    ('memory', entries, 'Observation'),
                                    ('failure', failures, 'Failure')]:
        for value in values:
            key = kind + '-' + digest(value)
            nodes[key] = {'id': key, 'type': node_type, 'attrs': value}
            kinds[key] = kind
            if kind == 'history':
                reports[value['hash']] = key
    edges = [{'source': reports[value['attrs']['development_hash']], 'target': key,
              'relation': 'SUPPORTS'} for key, value in nodes.items()
             if kinds[key] == 'memory' and 'development_hash' in value['attrs']]
    counter = {key for key in nodes if kinds[key] == 'failure'
               or (kinds[key] == 'memory' and
                   nodes[key]['attrs'].get('development_decision', {}).get('selection') == 'discard')}
    scores = encoder(nodes, query)
    selected, audit = select(nodes, edges, counter, 'evidence', query, limit, canonical, semantic=scores)
    values = lambda kind: [nodes[key]['attrs'] for key in selected if kinds[key] == kind]
    selected_history, selected_entries = compact_development_context(
        values('history'), values('memory'), protocol, seeds, history)
    closure = [reports[value['hash']] for value in selected_history if reports[value['hash']] not in selected]
    return selected_history, selected_entries, values('failure'), {
        **audit, 'query': query, 'limit': limit, 'retrieved_node_ids': selected,
        'readout_closure_ids': closure, 'counterevidence_available': len(counter),
        'counterevidence_omitted': len(counter - set(selected)),
        'scope': 'Public development memory only. Discarded model decisions are retained as alternatives, not mechanism refutations.'}
