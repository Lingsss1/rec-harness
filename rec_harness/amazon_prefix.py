"""Freeze and check Amazon role prefixes without revealing later outcomes."""
from pathlib import Path
from .agents import schema_for,validate_schema
from .core import Artifacts,ContractError,digest,number,read_json,validate_card,write_json
from .decision_traces import collect_traces
from .evolution import validate_policy
from .adapters.amazon_execution import code_identity


def conditions(study):
    return sorted(digest([dataset,[study['data_hashes'][dataset+'_'+split+'.txt']
        for split in ['train','valid']],seed]) for dataset in study['datasets']
        for seed in study['development_seeds']+study['confirmation_seeds'])


def freeze_suite(path,root,spec):
    from .harness_prefix import ROLES
    campaign=(path.parent/spec['campaign']).resolve();config=read_json(campaign/'amazon-auto.json')
    if digest(config)!=read_json(campaign/'auto-lock.json')['config_hash']:raise ContractError('Amazon prefix controller changed')
    study=read_json(Path(config['study'])/'study.json')
    if (digest(study)!=read_json(Path(config['study'])/'study-lock.json')['config_hash']
            or digest(study)!=config['study_config_hash']):raise ContractError('Amazon prefix study changed')
    roles=spec.get('roles',sorted(ROLES))
    if not roles or not set(roles).issubset(ROLES):raise ContractError('unsupported Amazon prefix role')
    if not spec['agent_backend'].get('model'):raise ContractError('prefix replay requires explicit model')
    for key in ['budget','call_cost']:number(spec[key],key,0)
    traces=collect_traces(campaign,roles=roles,cutoff_seq=spec.get('cutoff_seq'))
    source=Artifacts(campaign/'artifacts/objects');objects=Artifacts(root/'objects');cases=[]
    for trace in traces['decision_traces']:
        if trace.get('role_admission')=='rejected':continue
        if trace['visible_source_snapshot'] is None:raise ContractError('exact pre-call source snapshot required')
        cases.append({'adapter':'amazon-nextitnet-v1','graph':None,'id':trace['trace_id'],
            'role':trace['role'],'context':trace['visible_state'],'input_hash':trace['input_hash'],
            'source':{name:objects.put(source.verify(ref)) for name,ref in trace['visible_source_snapshot'].items()},
            'submitted_event':trace['submitted_event'],
            'require_method_sources':config.get('require_method_sources',False)})
    if not cases:raise ContractError('no admitted Amazon prefix decisions')
    baseline=config.get('research_policy',{});validate_policy(baseline)
    body={'schema':'harness-prefix-suite-v1','adapter':'amazon-nextitnet-v1','cases':cases,
        'baseline_policy':baseline,'source_conditions':conditions(study),'code_hashes':code_identity(),
        **{key:spec[key] for key in ['agent_backend','identity','budget','call_cost']},
        'trace_source':traces['source']}
    body['hash']=digest(body);write_json(root/'suite.json',body);return body


def check_response(case,context,response):
    role=case['role'];validate_schema(response,schema_for(role))
    if role=='reviewer':
        from .code_review import validate_code_review
        validate_code_review(response,context['implementation_hash'])
        return
    if role=='analyst':
        check_response({**case,'role':'lead'},context,response['decision'])
        from .knowledge import validate_knowledge_patch
        import json
        validate_knowledge_patch(json.loads(response['knowledge']['patch_json']),context['graph'])
        return
    if role=='researcher':
        check_response({**case,'role':'evidence'},context,response['gap'])
        check_response({**case,'role':'hypothesis'},{**context,'gap':response['gap']},
                       {k:response[k] for k in ('method','proposal')})
        allowed={n['id'] for n in context['evidence_graph']['nodes']}
        if not response['evidence_refs'] or not set(response['evidence_refs'])<=allowed or not response['gap']['observation_refs'] or not set(response['gap']['observation_refs'])<=allowed:
            raise ContractError('researcher prefix evidence outside visible graph')
        return
    binding=context['bindings']
    if role=='evidence':
        validate_card('gap',response)
        if response['model_version']!=binding['base_model_id'] or response['protocol_hash']!=binding['train_protocol_hash']:
            raise ContractError('Amazon prefix gap changed incumbent identity')
    elif role=='hypothesis':
        validate_card('method',response['method'],require_method_sources=case['require_method_sources'])
        proposal=response['proposal'];validate_card('proposal',proposal)
        for key in ['proposal_id','base_model_id','controls','train_protocol_hash','seeds']:
            if proposal[key]!=binding[key]:raise ContractError('Amazon prefix proposal binding differs: '+key)
        if proposal['gap_ids']!=[context['gap']['gap_id']] or proposal['method_ids']!=[response['method']['method_id']]:
            raise ContractError('Amazon prefix proposal references differ')
        if not 0<proposal['budget']<=context['budget']['remaining']:raise ContractError('Amazon prefix proposal exceeds budget')
        if context.get('insertion_contract_format'):
            from .adapters.insertion_contract import parse_contract
            parse_contract(proposal['insertion_contract'],required=True,
                available_methods=context['insertion_contract_format']['allowed_methods'])
    else:
        validate_card('decision',response);development=context['development']
        if development['hash'] not in response['evidence_refs']:raise ContractError('Amazon prefix exact development reference required')
        if response['conclusion_status'] not in {'unknown','provisional'}:raise ContractError('Amazon prefix cannot establish confirmation')
        if response['selection']=='keep' and development.get('summary',{}).get('eligible_for_confirmation') is not True:
            raise ContractError('Amazon prefix keep bypassed development gate')
