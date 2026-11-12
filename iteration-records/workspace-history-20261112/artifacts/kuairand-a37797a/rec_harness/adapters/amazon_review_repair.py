"""Bounded pre-registration design/code revision with crash-safe archives."""
from rec_harness.core import ContractError, digest, file_hash, read_json, write_json

ROLES = ('researcher', 'experiment', 'reviewer')


def recover(controller, folder):
    path = folder/'review-repair/state.json'
    if not path.exists():
        return None
    state = read_json(path)
    if type(state.get('count')) is not int or not 1<=state['count']<=2 or state.get('phase') not in {'archiving','ready'}:
        raise ContractError('invalid review revision state')
    if state['phase'] == 'archiving':
        event = state['archive']
        existing=[e['payload'] for e in controller.store.events() if e['kind']=='review_revision_started'
                  and e['payload']['round']==folder.name and e['payload']['revision']==state['count']]
        if existing and existing!=[event]:raise ContractError('review revision intent changed')
        if not existing:controller.store.event('review_revision_started',event)
        for name, refs in event['directories'].items():
            source=folder/name; target=folder/'review-repair'/('revision-'+str(state['count']))/name
            current=target if target.exists() else source
            actual={str(p.relative_to(current)) for p in current.rglob('*') if p.is_file()}
            if actual!=set(refs):raise ContractError('review revision archive file set changed')
            for key,ref in refs.items():
                controller.artifacts.verify(ref)
                if file_hash(current/key)!=ref['sha256']:raise ContractError('review revision archive changed')
            if not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True)
                source.rename(target)
        state['phase']='ready';write_json(path,state)
    verify(controller,folder)
    return state


def verify(controller, folder):
    events=[e['payload'] for e in controller.store.events() if e['kind']=='review_revision_started' and e['payload']['round']==folder.name]
    if not events:
        if (folder/'review-repair/state.json').exists():raise ContractError('review revision state has no ledger event')
        return
    state=read_json(folder/'review-repair/state.json')
    if [e['revision'] for e in events]!=list(range(1,len(events)+1)) or state['count']!=len(events) or state['archive']!=events[-1]:
        raise ContractError('review revision state differs from ledger')
    for event in events:
        for name,refs in event['directories'].items():
            target=folder/'review-repair'/('revision-'+str(event['revision']))/name
            if {str(p.relative_to(target)) for p in target.rglob('*') if p.is_file()}!=set(refs):
                raise ContractError('review revision archive file set changed')
            for key,ref in refs.items():
                controller.artifacts.verify(ref)
                if file_hash(target/key)!=ref['sha256']:raise ContractError('review revision archive changed')


def revise(controller, folder, review):
    state=recover(controller,folder)
    count=state['count'] if state else 0
    limit=controller.config.get('review_revision_limit',0)
    if count>=limit:
        raise ContractError('independent code review blocked training after '+str(count)+' revisions: '+
                            review['verdict']+'; '+'; '.join(review['blocking_findings']))
    if (folder/'preregistration.json').exists():raise ContractError('cannot revise a registered experiment')
    names=sorted(p.name for p in folder.iterdir() if p.is_dir() and
                 any(p.name==role or p.name.startswith(role+'-rejected-') for role in ROLES))
    refs={name:controller.artifacts.snapshot(folder/name,
        [str(p.relative_to(folder/name)) for p in (folder/name).rglob('*') if p.is_file()]) for name in names}
    event={'round':folder.name,'revision':count+1,'review_hash':digest(review),'directories':refs}
    write_json(folder/'review-repair/state.json',{'count':count+1,'phase':'archiving','archive':event})
    return recover(controller,folder)


def context_for_revision(folder, context, state):
    if not state:return context
    prior=folder/'review-repair'/('revision-'+str(state['count']))
    review=read_json(prior/'reviewer/response.json')
    design=read_json(prior/'researcher/response.json')
    sources={p.name:p.read_text() for p in (prior/'experiment/model').glob('*.py')}
    return {**context,'review_revision':{'number':state['count'],'review':review,
            'previous_design':design,'previous_sources':sources},
        'instructions':context['instructions']+[
            'Revise the prior proposal to address the independent reviewer. Preserve the research objective and all supplied IDs, seeds, protocol, controls and budget limits. Explain revisions in proposal.guardrails. You may correct an unnecessary or contradictory proposal-specific check with a reason; never remove host correctness gates or relabel evidence to obtain approval. Return the full revised design; the experiment role implements it and a fresh reviewer checks it before registration.']}
