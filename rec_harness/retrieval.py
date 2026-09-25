"""Deterministic role priorities; evidence visibility is enforced by Store first."""

ROLE_TYPES = {
    'evidence': ('Observation', 'MeasurementGap', 'GapHypothesis', 'Experiment'),
    'hypothesis': ('GapHypothesis', 'Method', 'Component', 'Proposal', 'Source'),
    'experiment': ('Proposal', 'Component', 'ModelVersion', 'Method'),
    'evaluation': ('Experiment', 'Observation', 'Proposal', 'Claim'),
    'knowledge': ('Claim', 'Observation', 'Experiment', 'GapHypothesis'),
    'lead': ('DecisionTrace', 'Experiment', 'GapHypothesis', 'Proposal'),
    'tool': ('MeasurementGap', 'ToolValidation', 'ToolCandidate', 'Observation'),
    'optimizer': ('DecisionTrace', 'DecisionFailureHypothesis', 'HarnessValidation'),
}


def select(nodes, edges, counter, role, query, limit, canonical):
    priorities = ROLE_TYPES.get(role, ())
    terms = query.lower().split()

    def rank(key):
        kind = nodes[key]['type']
        return (-sum(t in canonical(nodes[key]).lower() for t in terms),
                priorities.index(kind) if kind in priorities else len(priorities), key)

    ranked = sorted(nodes, key=rank)
    quota = max(1, limit // 3)
    selected = sorted(counter, key=rank)[:quota]
    reasons = {key: 'counterevidence' for key in selected}
    for key in ranked[:quota]:
        if key not in reasons and len(selected) < limit:
            selected.append(key)
            reasons[key] = 'role_query_seed'
    frontier = set(selected)
    for hop in (1, 2):
        neighbors = set()
        for edge in edges:
            if edge['source'] in frontier:
                neighbors.add(edge['target'])
            if edge['target'] in frontier:
                neighbors.add(edge['source'])
        frontier = set()
        for key in sorted(neighbors - set(selected), key=rank):
            if len(selected) >= limit:
                break
            selected.append(key)
            frontier.add(key)
            reasons[key] = f'hop_{hop}'
    # Fill only after expansion; filling at hop 1 starves hop 2 entirely.
    for key in ranked:
        if len(selected) >= limit:
            break
        if key not in reasons:
            selected.append(key)
            reasons[key] = 'ranked_fill'
    return selected, {'method': 'role-lexical-two-hop-v2',
                      'role_type_priority': list(priorities), 'selection_reasons': reasons}
