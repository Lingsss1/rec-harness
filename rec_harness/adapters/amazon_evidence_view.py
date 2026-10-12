"""Bounded read-only evidence federation. Exported memory is not a native audit.

Inputs must be the controller's already-visible development prefix. This module
does not import into Store, retrieve confirmation, or fabricate missing evidence.
"""
import json
import re

from rec_harness.core import ContractError, canonical, digest, public_only
from rec_harness.retrieval import select, revision_visibility
from .amazon_memory import validate_readout
from .amazon_hypothesis_graph import project


def fact_id(report):
    return 'amazon-research-'+digest(report['proposal_id'])[:16]+'-'+report['hash'][:16]+'-development'


def ordered_entries(entries):
    """Keep campaign availability order, with numeric rounds within each campaign."""
    groups={}
    def key(pair):
        index,entry=pair
        match=re.fullmatch(r'(.*?)(?:/?round-)(\d+)',entry['round'])
        group=match[1] if match else entry['round']
        if group not in groups:groups[group]=len(groups)
        return groups[group],int(match[2]) if match else index,index
    return [entry for _,entry in sorted(enumerate(entries),key=key)]


def prior_round_paths(paths, current):
    """A resumed earlier round must not see already-existing later directories."""
    number=int(current.name.removeprefix('round-'))
    return sorted((path for path in paths if int(path.parent.name.removeprefix('round-'))<number),
                  key=lambda path:int(path.parent.name.removeprefix('round-')))


def design_excerpt(design):
    """Exact bounded excerpts; retain a hash and disclose every clipped field."""
    fields={'gap':('suspected_capability',), 'method':('mechanism',),
            'proposal':('action','mechanism','falsification_condition','controls')}
    result={};clipped=[]
    for card,names in fields.items():
        for name in names:
            value=design.get(card,{}).get(name)
            if value is None:continue
            text=value if isinstance(value,str) else canonical(value)
            key=card+'.'+name
            result[key]=text[:1200]
            if len(text)>1200:clipped.append(key)
    return {'source_design_hash':digest(design),'excerpts':result,'truncated_fields':clipped,
            'qualification':'Exact excerpts for retrieval context. Full design remains in the source memory; omitted text is not available evidence.'}


def portable_records(history, entries, native_ids, protocol):
    """Expose hash-validated reports and explicitly *reported* interpretations.

    Documentary DERIVED_FROM edges record extraction, never scientific SUPPORTS.
    Original references remain reported strings, not invented graph endpoints.
    """
    nodes={};edges=[];by_hash={r['hash']:r for r in history}
    for report in history:
        key=fact_id(report)
        if key in native_ids:continue
        nodes[key]={'id':key,'type':'Experiment','attrs':{
            'kind':'imported_development_readout','origin':'imported_development',
            'visibility':'public','partition':'development','protocol_hash':protocol,
            'task_id':'amazon-research-'+digest(report['proposal_id'])[:16],
            'readout_hash':report['hash'],'proposal_id':report['proposal_id'],
            'summary':report['summary'],'independent_unit':False,
            'counterevidence':report['summary'].get('eligible_for_confirmation') is False,
            'qualification':'Hash-validated exported development aggregate. Original execution receipts have not been re-audited here. Gate rejection is not mechanism refutation.'}}
    for entry in entries:
        report=entry.get('development') or by_hash.get(entry.get('development_hash'))
        if not report or report['hash'] not in by_hash or fact_id(report) in native_ids:continue
        fact=fact_id(report);task=nodes[fact]['attrs']['task_id']
        identity=digest(entry);key='memory-entry-'+identity
        common={'task_id':task,'protocol_hash':protocol,'visibility':'public','partition':'development',
                'origin':'imported_memory','source_memory_hash':identity,'source_round':entry['round'],
                'independent_unit':False}
        nodes[key]={'id':key,'type':'Observation','attrs':{**common,
            'kind':'exported_research_record','design':design_excerpt(entry.get('design',{})),
            'evidence_dependency_ids':[fact],
            'qualification':'Record of a prior research design/interpretation, not independent experimental evidence.'}}
        edges.append(_derived(key,fact))
        knowledge=entry.get('provisional_knowledge',{})
        if not knowledge:continue
        try:patch=json.loads(knowledge['patch_json'])
        except (ValueError,KeyError,TypeError) as error:
            raise ContractError('malformed exported knowledge patch') from error
        public_only(patch)
        for original in patch.get('nodes',[]):
            attrs=original.get('attrs',{})
            if (original.get('type') not in {'Claim','GapHypothesis','MeasurementGap'} or
                attrs.get('task_id')!=task or attrs.get('protocol_hash')!=protocol or
                attrs.get('partition') not in (None,'development') or
                attrs.get('visibility') not in (None,'public') or
                attrs.get('origin')!='agent_hypothesis' or
                (original['type']=='Claim' and attrs.get('status')!='provisional')):continue
            identifier='reported-knowledge-'+digest([identity,original])
            nodes[identifier]={'id':identifier,'type':original['type'],'attrs':{**common,
                'kind':'reported_interpretation','status':'provisional',
                'reported_node_id':original['id'],'reported_content':attrs,
                'reported_evidence_refs':attrs.get('evidence_refs',[]),
                'evidence_dependency_ids':[key],
                'counterevidence':bool(attrs.get('counterevidence')),
                'qualification':'Exported agent interpretation only. Reported references and relations are not restored or verified native evidence; follow DERIVED_FROM for documentary provenance.'}}
            edges.append(_derived(identifier,key))
    return nodes,edges


def _derived(source,target):
    return {'id':'view-derived-'+digest([source,target]),'source':source,'target':target,
            'relation':'DERIVED_FROM','attrs':{'origin':'imported_memory','evidence_refs':[],
            'qualification':'Documentary extraction only; not causal support.'}}


def attach_analysis_history(request, prior):
    """Keep the frozen design-time history visible, outside writable patch refs."""
    if not prior or prior.get('schema')!='amazon-hypothesis-evidence-graph-v2':return request
    if prior.get('hash')!=digest({k:v for k,v in prior.items() if k!='hash'}):
        raise ContractError('design-time evidence view changed before analysis')
    return {**request,'prior_research_evidence':prior,
            'instructions':[*request.get('instructions',[]),
                'prior_research_evidence is the frozen pre-design history, not current-candidate measurements. Compare the new results with its relevant explanations and measurement gaps; reported imports remain unverified interpretations.',
                'Only graph / patch_contract define permitted knowledge-patch evidence_refs and relation endpoints. Do not use imported prior_research_evidence IDs as local experimental evidence. Prior interpretations may motivate a new current-scoped provisional explanation, grounded in the supplied current development graph.']}


def evidence_view(snapshot, history, entries, failures, baseline, protocol, rows, *,
                  selected_history=(), limit=64):
    if type(limit) is not int or limit<2:raise ContractError('hypothesis graph limit must be at least two')
    public_only([history,entries,failures])
    # Preserve supplied availability order; duplicate records do not become new evidence.
    reports={}
    for report in history:
        validate_readout(report,protocol)
        reports[report['hash']]=report
    history=list(reports.values())
    for entry in entries:
        if entry.get('development'):
            validate_readout(entry['development'],protocol)
    query='mechanism evidence counterevidence measurement '+baseline
    # Native eligibility is still protocol/task/public-development scoped. Selection
    # uses ALL prefix history, not only the first-stage semantic shortlist.
    native=project(snapshot,history,baseline,protocol,rows,
                   limit=max(2,len(snapshot['nodes'])+1),query=query,include_measurement=True)
    current=native['nodes'][0];nodes={n['id']:n for n in native['nodes'][1:]};edges=list(native['edges'])
    portable,links=portable_records(history,entries,set(nodes),protocol)
    nodes.update(portable);edges.extend(links)
    failure_ids=[]
    for failure in failures:
        key='execution-failure-'+digest(failure);failure_ids.append(key)
        nodes[key]={'id':key,'type':'Observation','attrs':{
            'kind':'execution_failure','origin':'execution_record','visibility':'public',
            'partition':'development','protocol_hash':protocol,'record':failure,
            'qualification':'Execution problem only, not a failed scientific hypothesis.',
            'independent_unit':False}}
    counter={key for key,n in nodes.items() if n['attrs'].get('counterevidence') or
             n['attrs'].get('status') in {'contradicted','superseded','retracted'}}
    counter.update(e['source'] for e in edges if e['relation'] in {'CONTRADICTS','SUPERSEDES'})
    ranked,audit=select(nodes,edges,counter,'hypothesis',query,max(1,len(nodes)),canonical)
    selected=[];visible=set();reasons={};omitted=[]

    def closure(key):
        pending=[key];found=set()
        while pending:
            item=pending.pop()
            if item in found:continue
            if item not in nodes:return None
            found.add(item);attrs=nodes[item]['attrs']
            pending.extend(attrs.get('evidence_refs',[]))
            # Native aggregate dependencies can refer to original run receipts
            # outside this derived view. Those are not independent new evidence.
            pending.extend(ref for ref in attrs.get('evidence_dependency_ids',[]) if ref in nodes)
        return found

    def add(key,reason,dependencies=()):
        bundle=closure(key)
        for dependency in dependencies:
            extra=closure(dependency)
            bundle=bundle | extra if bundle is not None and extra is not None else None
        if bundle is None or len(visible | bundle)>limit-1:
            omitted.append({'node_id':key,'reason':reason,'cause':'missing_reference' if bundle is None else 'node_budget'})
            return
        for item in [key,*sorted(bundle-{key})]:
            if item not in visible:
                selected.append(item);visible.add(item);reasons[item]=reason if item==key else 'reference_closure'

    # Mandatory tiers are explicit and bounded. An impossible budget is reported,
    # never silently exceeded or interpreted as absence of evidence.
    recent=[fact_id(r) for r in history[-2:]][::-1]
    for key in recent:add(key,'recent_development')
    for key in failure_ids[-1:]:add(key,'recent_execution_failure')
    for kind,reason in [('MeasurementGap','measurement_gap'),('Claim','recent_interpretation')]:
        for fact in recent:
            task=nodes[fact]['attrs']['task_id']
            choices=[key for key in ranked if nodes[key]['type']==kind and nodes[key]['attrs'].get('task_id')==task]
            if choices:add(choices[0],reason)
    task_order={'amazon-research-'+digest(r['proposal_id'])[:16]:i for i,r in enumerate(history)}
    revisions=[e for e in edges if e['relation']=='SUPERSEDES']
    revisions.sort(key=lambda e:(-task_order.get(nodes[e['source']]['attrs'].get('task_id'),-1),e['id']))
    for relation in revisions[:2]:
        add(relation['source'],'revision',[relation['target'],*relation['attrs'].get('evidence_refs',[])])
    for key in [k for k in ranked if k in counter][:max(1,(limit-1)//4)]:add(key,'counterevidence')
    for report in selected_history:
        key=fact_id(report)
        if key in nodes:add(key,'semantic_history')
    for key in ranked:
        if key not in visible:add(key,audit['selection_reasons'].get(key,'ranked_fill'))
    result={'schema':'amazon-hypothesis-evidence-graph-v2','graph_version':native['graph_version'],
            'nodes':[current,*[nodes[k] for k in selected]],
            'edges':[e for e in edges if e['source'] in visible and e['target'] in visible and
                     set(e['attrs'].get('evidence_refs',[]))<=visible],
            'retrieval':{'method':'prefix-federated-tiered-two-hop-v2','limit':limit,'query':query,
                'selected_node_ids':selected,'selection_reasons':reasons,
                'available_readout_hashes':list(reports),'recent_fact_ids':recent,
                'portable_readout_hashes':[r['hash'] for r in history if fact_id(r) in portable],
                'native_unavailable_readout_hashes':native['retrieval']['unavailable_readout_hashes'],
                'omitted_node_count':len(nodes)-len(visible),'omitted_requests':omitted},
            'revision_visibility':revision_visibility(nodes,edges,selected),
            'scope':'Read-only development prefix. Native evidence and exported/report-only memory remain distinguished. DERIVED_FROM on imported records is documentary provenance, not scientific support. Missing native references are not fabricated.'}
    result['coverage']={'history_available':bool(history),'historical_nodes_visible':bool(visible),
        'baseline_only':not visible,'imported_interpretations_verified':False,
        'instructions':'Assess relevant history, counterevidence and measurement gaps; explain exclusions in the gap alternatives or proposal rationale. Cite supplied IDs actually used. Reported interpretations are leads to verify, not established mechanisms. Baseline-only input must not be described as a review of historical graph evidence.'}
    return {**result,'hash':digest(result)}
