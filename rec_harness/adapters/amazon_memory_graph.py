"""Expose existing local development explanations for append-only revision."""
from rec_harness.core import digest


def prior_explanations(graph, snapshot, entries):
    known={item['id']:item for item in snapshot['nodes']}
    selected={};unavailable=[]
    for entry in entries:
        proposal=entry.get('design',{}).get('proposal',{}).get('proposal_id')
        readout=entry.get('development_hash')
        if not proposal or not readout:continue
        task='amazon-research-'+digest(proposal)[:16]
        prefix=task+'-'+readout[:16]
        fact=known.get(prefix+'-development')
        if (not fact or fact['type']!='Experiment' or
            fact['attrs'].get('partition')!='development' or
            fact['attrs'].get('valid') is not True or
            fact['attrs'].get('origin')!='local_evidence' or
            fact['attrs'].get('protocol_hash')!=graph['protocol_hash']):
            unavailable.append({'round':entry['round'],'reason':'no matching local development fact'})
            continue
        for item in snapshot['nodes']:
            attrs=item['attrs']
            if (attrs.get('task_id')!=task or attrs.get('protocol_hash')!=graph['protocol_hash']
                or attrs.get('origin')!='agent_hypothesis'):continue
            if item['type']=='Claim' and attrs.get('status')=='provisional':
                selected[item['id']]=item
            elif item['type']=='GapHypothesis':
                selected[item['id']]=item
    current={item['id'] for item in graph['nodes']}
    selected={key:value for key,value in selected.items() if key not in current}
    return {**graph,'nodes':graph['nodes']+[selected[key] for key in sorted(selected)],
        'historical_explanation_ids':sorted(selected),
        'historical_explanations_unavailable':unavailable,
        'historical_revision_scope':'Existing local development explanations only. New provisional interpretations may supersede or build on these nodes; no historical fact/status is rewritten and cross-task causal comparison is not established.'}
