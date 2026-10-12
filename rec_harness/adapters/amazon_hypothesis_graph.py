"""Pre-design retrieval of native, public development evidence and relations."""
from rec_harness.core import canonical, digest
from rec_harness.retrieval import select, revision_visibility

RELATIONS = {'TESTS','TARGETS','USES','PRODUCES','SUPPORTS','CONTRADICTS',
             'BASED_ON','SUPERSEDES','ALTERNATIVE_TO','DERIVED_FROM'}
DESIGNS = {'Claim','GapHypothesis','Method','Proposal','Source'}


def project(snapshot, history, baseline, protocol, rows, *, limit=64, query='', include_measurement=False):
    if type(limit) is not int or limit < 2:
        from rec_harness.core import ContractError
        raise ContractError('hypothesis graph limit must be at least two')
    designs = DESIGNS | ({'MeasurementGap'} if include_measurement else set())
    all_nodes = {n['id']:n for n in snapshot['nodes']}
    facts = {}
    unavailable = []
    for report in history:
        task='amazon-research-'+digest(report['proposal_id'])[:16]
        key=task+'-'+report['hash'][:16]+'-development'
        n=all_nodes.get(key);a=n['attrs'] if n else {}
        if (not n or a.get('kind')!='derived_development_comparison' or
            a.get('partition')!='development' or a.get('protocol_hash')!=protocol or
            a.get('valid') is not True or a.get('origin')!='local_evidence' or
            a.get('visibility')!='public'):
            unavailable.append(report['hash']);continue
        facts[key]=n
    tasks={n['attrs']['task_id'] for n in facts.values()}
    nodes=dict(facts)
    for n in snapshot['nodes']:
        a=n['attrs']
        # Accepted knowledge uses public_only's default-public convention.
        # Apply it only to v2 interpretation nodes; native facts still require
        # explicit public provenance and all references must close below.
        visibility=a.get('visibility','public' if include_measurement and n['type'] in designs else None)
        if (a.get('task_id') not in tasks or a.get('protocol_hash')!=protocol or
            visibility!='public' or a.get('partition') not in (None,'development')):
            continue
        if (n['type'] in designs and a.get('origin') in ('agent_hypothesis','paper_claim') or
            n['type']=='Observation' and a.get('origin')=='local_evidence' and
            a.get('valid') is not False and a.get('evidence_dependency_ids') and
            set(a['evidence_dependency_ids'])<=set(facts)):
            nodes[n['id']]=n
    # Never return a claim whose supporting references are outside this allowed
    # development snapshot. In particular, private confirmation is not a seed.
    while True:
        invalid={key for key,n in nodes.items() if n['type'] in designs and
                 not set(n['attrs'].get('evidence_refs',[]))<=set(nodes)}
        if not invalid:break
        for key in invalid:del nodes[key]
    edges=[e for e in snapshot['edges'] if e['relation'] in RELATIONS and
           e['source'] in nodes and e['target'] in nodes and
           set(e['attrs'].get('evidence_refs',[]))<=set(nodes)]
    counter={key for key,n in nodes.items() if n['attrs'].get('counterevidence') or
             n['attrs'].get('status') in ('contradicted','superseded','retracted')}
    counter.update(e['source'] for e in edges if e['relation'] in ('CONTRADICTS','SUPERSEDES'))
    selected,audit=select(nodes,edges,counter,'hypothesis',query,limit-1,canonical)
    # Bounded reference closure: skip an explanation if its evidence cannot fit.
    visible=set(selected)
    for key in list(selected):
        missing=set(nodes[key]['attrs'].get('evidence_refs',[]))-visible
        missing.update(set(nodes[key]['attrs'].get('evidence_dependency_ids',[]))&set(nodes)-visible)
        if len(visible|missing)<=limit-1:visible.update(missing)
        elif nodes[key]['type'] in designs or nodes[key]['type']=='Observation':visible.discard(key)
    while True:
        invalid={key for key in visible if nodes[key]['type'] in designs and
                 not set(nodes[key]['attrs'].get('evidence_refs',[]))<=visible}
        if not invalid:break
        visible-=invalid
    current={'id':'current-baseline-'+digest([baseline,protocol,rows])[:24], 'type':'Observation',
             'attrs':{'model_version':baseline,'protocol_hash':protocol,'partition':'development',
                      'origin':'local_evidence','visibility':'public','kind':'current_audited_baseline',
                      'rows':rows,'independent_unit':False}}
    result={'schema':'amazon-hypothesis-evidence-graph-v1',
            'graph_version':snapshot.get('version',snapshot.get('graph_version')),
            'nodes':[current,*[nodes[key] for key in sorted(visible)]],
            'edges':[e for e in edges if e['source'] in visible and e['target'] in visible and
                     set(e['attrs'].get('evidence_refs',[]))<=visible],
            'retrieval':{**audit,'limit':limit,'selected_node_ids':sorted(visible),
                         'omitted_node_count':len(nodes)-len(visible),'unavailable_readout_hashes':unavailable},
            'revision_visibility':revision_visibility(nodes,edges,sorted(visible)),
            'scope':'Native public development graph selected before design. Current baseline is a derived view of audited rows. Imported readouts without local graph lineage stay in history; no fabricated graph edges. Selection is not proof that an agent used or understood evidence.'}
    return {**result,'hash':digest(result)}


def context_graph(controller, history, baseline, protocol, rows, *, available_history=None, entries=(), failures=()):
    from rec_harness.store import Store
    store=Store(controller.study/'memory',read_only=True)
    try:
        store.db.execute('BEGIN')
        integrity=store.verify();snapshot=store.snapshot()
        if controller.config.get('evidence_graph_policy') == 'prefix-evidence-v2':
            from .amazon_evidence_view import evidence_view
            result=evidence_view(snapshot,available_history if available_history is not None else history,
                entries,failures,baseline,protocol,rows,selected_history=history,
                limit=controller.config.get('hypothesis_graph_limit',64))
        else:
            result=project(snapshot,history,baseline,protocol,rows,
                           limit=controller.config.get('hypothesis_graph_limit',64),
                           query='mechanism evidence counterevidence '+baseline)
        result['source_event_head']=integrity['head']
        result['hash']=digest({k:v for k,v in result.items() if k!='hash'})
        return result
    finally:store.close()
