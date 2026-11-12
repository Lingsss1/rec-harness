"""Constrain mechanical role fields before generation, without rewriting replies."""
from .agents import schema_for


def bound_schema(role, context):
    schema = schema_for(role)
    if not context.get('host_bound_output'):
        return schema
    if role == 'researcher':
        schema['properties']['gap']=bound_schema('evidence',context)
        proposal=bound_schema('hypothesis',context)['properties']
        schema['properties'].update(proposal)
        allowed=[n['id'] for n in context.get('evidence_graph',{}).get('nodes',[])]
        if allowed:
            schema['properties']['evidence_refs']['items']['enum']=allowed
            schema['properties']['gap']['properties']['observation_refs']['items']['enum']=allowed
        return schema
    if role == 'analyst':
        schema['properties']['decision']=bound_schema('lead',context)
        return schema
    if role == 'reviewer':
        schema['properties']['evidence_refs']['enum']=[[context['implementation_hash']]]
        return schema
    props = schema['properties']
    binding = context.get('bindings', {})
    def exact(fields, key, value):
        fields[key] = {**fields[key], 'enum': [value]}
    if role == 'evidence' and binding:
        exact(props, 'model_version', binding['base_model_id'])
        exact(props, 'protocol_hash', binding['train_protocol_hash'])
    elif role == 'hypothesis' and binding:
        proposal = props['proposal']['properties']
        for key in ('proposal_id', 'base_model_id', 'controls', 'train_protocol_hash', 'seeds'):
            exact(proposal, key, binding[key])
        if context.get('gap'):
            exact(proposal, 'gap_ids', [context['gap']['gap_id']])
        proposal['budget'].update(exclusiveMinimum=0, maximum=context['budget']['remaining'])
    elif role == 'experiment':
        exact(props, 'changed_files', ['model.py', *[c['file'] for c in context.get('generated_controls', [])]])
    elif role in ('evaluation', 'lead') and context.get('development'):
        exact(props, 'evidence_refs', [context['development']['hash']])
        props['conclusion_status']['enum'] = ['unknown', 'provisional']
    return schema
