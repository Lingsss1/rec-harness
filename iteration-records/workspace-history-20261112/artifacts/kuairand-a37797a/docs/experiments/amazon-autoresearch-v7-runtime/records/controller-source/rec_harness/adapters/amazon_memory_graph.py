"""Expose existing local development explanations for append-only revision."""
from rec_harness.core import digest


def prior_explanations(graph, snapshot, entries):
    known={item['id']:item for item in snapshot['nodes']}
    selected={};facts={};unavailable=[]
    for entry in entries:
        proposal=entry.get('design',{}).get('proposal',{}).get('proposal_id')
        readout=entry.get('development_hash')
        if not proposal or not readout:continue
        task='amazon-research-'+digest(proposal)[:16]
        prefix=task+'-'+readout[:16]
        fact=known.get(prefix+'-development')
        if (not fact or fact['type']!='Experiment' or
            fact['attrs'].get('task_id')!=task or
            fact['attrs'].get('kind')!='derived_development_comparison' or
            fact['attrs'].get('partition')!='development' or
            fact['attrs'].get('valid') is not True or
            fact['attrs'].get('origin')!='local_evidence' or
            fact['attrs'].get('protocol_hash')!=graph['protocol_hash']):
            unavailable.append({'round':entry['round'],'reason':'no matching local development fact'})
            continue
        facts[fact['id']]=fact
        for item in snapshot['nodes']:
            attrs=item['attrs']
            if (item['type']=='Observation' and attrs.get('task_id')==task
                and attrs.get('protocol_hash')==graph['protocol_hash']
                and attrs.get('partition')=='development' and attrs.get('origin')=='local_evidence'
                and attrs.get('valid') is not False
                and attrs.get('evidence_dependency_ids')==[fact['id']]):
                facts[item['id']]=item
            if (attrs.get('task_id')!=task or attrs.get('protocol_hash')!=graph['protocol_hash']
                or attrs.get('partition') not in {None,'development'}
                or attrs.get('origin')!='agent_hypothesis'):continue
            if item['type']=='Claim' and attrs.get('status')=='provisional':
                selected[item['id']]=item
            elif item['type']=='GapHypothesis':
                selected[item['id']]=item
    current={item['id'] for item in graph['nodes']}
    selected={key:value for key,value in selected.items() if key not in current}
    visible={item['id']:item for item in graph['nodes']}|selected
    revisions=[];evidence={}
    existing={item['id'] for item in graph.get('edges',[])}
    for relation in snapshot.get('edges',[]):
        if relation['relation'] not in {'SUPERSEDES','BASED_ON','ALTERNATIVE_TO'}:continue
        if relation['source'] not in visible or relation['target'] not in selected:continue
        attrs=relation['attrs'];source=visible[relation['source']]
        refs=attrs.get('evidence_refs',[])
        if (attrs.get('origin')!='agent_hypothesis' or attrs.get('status')!='provisional'
            or attrs.get('scope')!=source['attrs']['task_id'] or not refs
            or not set(refs)<=set(facts)):continue
        if any(facts[ref]['attrs']['task_id']!=source['attrs']['task_id'] for ref in refs):continue
        if relation['id'] not in existing:revisions.append(relation)
        for ref in refs:
            for needed in [ref,*facts[ref]['attrs'].get('evidence_dependency_ids',[])]:
                if needed in facts and needed not in visible:evidence[needed]=facts[needed]
    return {**graph,'nodes':graph['nodes']+[selected[key] for key in sorted(selected)]+[evidence[key] for key in sorted(evidence)],
        'edges':graph.get('edges',[])+sorted(revisions,key=lambda item:item['id']),
        'historical_evidence_ids':sorted(evidence),
        'historical_revision_edge_ids':sorted(item['id'] for item in revisions),
        'historical_explanation_ids':sorted(selected),
        'historical_explanations_unavailable':unavailable,
        'historical_revision_scope':'Existing local development explanations only. New provisional interpretations may supersede or build on these nodes; no historical fact/status is rewritten and cross-task causal comparison is not established.'}
