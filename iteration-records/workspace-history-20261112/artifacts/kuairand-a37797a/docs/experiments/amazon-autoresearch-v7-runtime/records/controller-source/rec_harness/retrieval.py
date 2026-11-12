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


def select(nodes, edges, counter, role, query, limit, canonical, *, semantic=None):
    priorities = ROLE_TYPES.get(role, ())
    terms = query.lower().split()

    def rank(key):
        kind = nodes[key]['type']
        return (-sum(t in canonical(nodes[key]).lower() for t in terms),
                priorities.index(kind) if kind in priorities else len(priorities), key)

    ranked = sorted(nodes, key=rank)
    seeds=ranked
    if semantic is not None:
        from .semantic import validate
        scores=validate(semantic,nodes,query)
        seeds=sorted(nodes,key=lambda key:(-scores[key],rank(key)))
    quota = max(1, limit // 3)
    selected = sorted(counter, key=rank)[:quota]
    reasons = {key: 'counterevidence' for key in selected}
    for key in seeds[:quota]:
        if key not in reasons and len(selected) < limit:
            selected.append(key)
            reasons[key] = 'semantic_seed' if semantic is not None else 'role_query_seed'
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
    return selected, {'method': 'role-neural-two-hop-v1' if semantic is not None else 'role-lexical-two-hop-v2',
                      **({'semantic':semantic} if semantic is not None else {}),
                      'role_type_priority': list(priorities), 'selection_reasons': reasons}


def revision_visibility(nodes, edges, selected):
    """Describe revision truncation without changing a claim's scientific status.

    Store supplies only task/protocol/reveal-visible nodes and edges. Omitted
    source bodies stay omitted; at most one compact entry is added per returned
    node. A proposed SUPERSEDES link is not itself a host-confirmed retraction.
    """
    visible=set(selected);incoming={}
    for relation in edges:
        if relation['relation']=='SUPERSEDES':
            incoming.setdefault(relation['target'],set()).add(relation['source'])
    rows=[]
    for identity in selected:
        status=nodes[identity]['attrs'].get('status')
        sources=incoming.get(identity,set())
        if not sources and status not in {'superseded','retracted'}:continue
        rows.append({'node_id':identity,'declared_status':status,
            'visible_superseders':sorted(sources&visible),
            'omitted_superseder_count':len(sources-visible),
            'requires_revision_review':True})
    return {'entries':rows,
        'scope':'Declared retirement and incoming SUPERSEDES links in the allowed evidence snapshot. A revision link does not automatically retire its target or establish the replacement scientifically. Omitted superseders are not supplied as evidence.'}
