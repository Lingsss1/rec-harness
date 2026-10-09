"""Freeze prior tool definitions and gate outcomes without exporting scores."""
from pathlib import Path
from .core import Artifacts,ContractError,digest,public_only,read_json
from .store import Store


def collect(sources):
    if not isinstance(sources,list) or not sources:raise ContractError('nonempty tool history sources required')
    bindings=[];rows=[];excluded=set();seen=set();paths=set()
    for source in sources:
        root=Path(source['campaign']).resolve()
        if root in paths:raise ContractError('duplicate tool history campaign')
        paths.add(root);store=Store(root/'memory',read_only=True)
        try:
            store.db.execute('BEGIN');store.verify();events=store.events()
            cutoff=source.get('cutoff_seq',len(events))
            if type(cutoff) is not int or not 0<cutoff<=len(events):raise ContractError('invalid tool history cutoff')
            events=events[:cutoff];head=events[-1]['hash']
            if source.get('event_head',head)!=head:raise ContractError('tool history frozen event head changed')
            if head in seen:raise ContractError('duplicate tool history evidence prefix')
            seen.add(head);source_id='tool-history-'+digest([str(root),head])[:24]
            selected=source.get('candidate_ids')
            if selected is not None and (not isinstance(selected,list) or not selected or len(set(selected))!=len(selected)):
                raise ContractError('distinct tool history candidate IDs required')
            candidates={e['payload']['candidate_id']:e['payload'] for e in events
                if e['kind']=='candidate_proposed' and e['payload']['kind']=='tool'}
            if selected is not None:
                if not set(selected)<=set(candidates):raise ContractError('tool history candidate absent from prefix')
                candidates={key:candidates[key] for key in selected}
            if not candidates:raise ContractError('tool history contains no tool candidates')
            if not (root/'objects').is_dir():raise ContractError('tool history artifact directory missing')
            objects=Artifacts(root/'objects')
            for key,candidate in sorted(candidates.items()):
                plan=read_json(root/'candidates'/key/'plan.json')
                if digest(plan)!=candidate['plan_hash']:raise ContractError('tool history plan changed')
                excluded.update(plan['development_ids'])
                row={'history_id':source_id+'-'+key,'candidate_id':key,'source_id':source_id,
                    'plan_hash':candidate['plan_hash'],'validation_identity':plan.get('identity'),
                    'tool_contract':plan.get('tool_contract'),'status':'proposed','method_card':None,'validation':[]}
                for event in events:
                    value=event['payload']
                    if event['kind']=='candidate_rollback' and value.get('from')==key:row['status']='rolled_back'
                    if event['kind']=='candidate_promoted' and (value.get('previous') or {}).get('candidate_id')==key:
                        row['status']='superseded'
                    if value.get('candidate_id')!=key:continue
                    kind=event['kind']
                    if kind=='candidate_implemented':
                        row.update(status='implemented',implementation_hash=value['implementation_hash'])
                        if plan.get('tool_method_contract'):
                            from .tool_method import validate
                            card=read_json(objects.verify(value['proposal_ref']));validate(card,value.get('source_hash'))
                            row.update(method_card=card,method_card_hash=value['proposal_ref']['sha256'])
                    elif kind in {'validation_passed','validation_rejected'}:
                        validation={'stage':value['stage'],'status':'passed' if kind=='validation_passed' else 'rejected'}
                        if kind=='validation_rejected':
                            reason=value.get('rejection_kind')
                            validation['rejection_kind']=reason if reason in {'validation_gate','validation_contract'} else 'unknown'
                        row['validation'].append(validation)
                        if 'report' in value:
                            if digest(value['report'])!=value['report_hash']:raise ContractError('tool history report changed')
                            report=value['report'];ids=report.get('evaluation_ids') if isinstance(report,dict) else None
                            if not isinstance(ids,list) or not ids or any(not isinstance(item,str) or not item.strip() for item in ids):
                                raise ContractError('tool history validation has invalid evaluation IDs')
                            excluded.update(ids)
                            if isinstance(report.get('evaluation_mode'),str) and report['evaluation_mode'] in {'fixed_candidate_selection','end_to_end_candidate_generation','offline_measurement'}:
                                validation['evaluation_mode']=report['evaluation_mode']
                            if kind=='validation_rejected' and 'failed_gates' in value:
                                from .evolution import numerical_gate_failures
                                expected=numerical_gate_failures(report,plan['stages'][value['stage']])
                                if value.get('rejection_kind')!='validation_gate' or not expected or value['failed_gates']!=expected:
                                    raise ContractError('tool history failed gates differ from frozen report and plan')
                                validation['failed_gates']=expected
                        else:raise ContractError('tool history validation lacks evaluation IDs; cannot exclude prior conditions')
                    elif kind=='candidate_promoted':row['status']='promoted'
                    elif kind=='evolution_stopped':row['status']=value['status']
                rows.append(row)
            bindings.append({'source_id':source_id,'campaign':str(root),'cutoff_seq':cutoff,'event_head':head})
        finally:store.close()
    result={'schema':'tool-development-history-v1','sources':bindings,'candidates':rows,
        'instructions':'Review these existing and failed tool candidates before choosing a method. Compare definitions, assumptions, failure modes and validation conditions. Avoid repeating rejected designs without a specific new rationale. Missing cards and different conditions are not evidence of applicability.',
        'qualification':'Prior tool author declarations and gate outcomes only. Missing method cards remain unknown. No scores, raw outputs or validation examples. Prior conditions are excluded from new validation; these records do not prove current-task utility.'}
    public_only(result)
    return result,sorted(excluded)


def project_sources(project):
    """Only committed project tool episodes, bounded by their terminal reports."""
    sources=[];seen=set()
    for event in project.store.events():
        entry=event['payload'].get('episode') if event['kind']=='project_transition' else None
        if not entry or entry['kind']!='tool':continue
        root=Path(entry['campaign']).resolve()
        if not root.is_relative_to(project.root/'episodes'):raise ContractError('tool history child outside project')
        report=read_json(root/'run-report.json')
        if digest(report)!=entry['result_hash']:raise ContractError('tool history project report changed')
        store=Store(root/'memory',read_only=True)
        try:
            store.verify()
            matches=[e for e in store.events() if e['kind']=='evolution_stopped' and digest(e['payload'])==entry['result_hash']]
            if not matches:raise ContractError('tool history has no committed terminal report')
            terminal=matches[0]
        finally:store.close()
        if root in seen:continue
        seen.add(root)
        sources.append({'campaign':str(root),'cutoff_seq':terminal['seq'],'event_head':terminal['hash']})
    return sources
