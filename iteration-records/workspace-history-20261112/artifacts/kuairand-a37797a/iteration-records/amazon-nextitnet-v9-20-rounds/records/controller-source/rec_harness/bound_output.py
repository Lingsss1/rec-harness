"""Constrain mechanical role fields before generation, without rewriting replies."""
import json
from .agents import schema_for


def bound_schema(role, context):
    schema = json.loads(json.dumps(schema_for(role)))
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


def service_schema(schema):
    """Project host contracts into Codex's structured-output subset.

    Whole-array enums and exclusive bounds remain enforced against the original
    schema after generation. No returned evidence is repaired or rewritten.
    """
    result=json.loads(json.dumps(schema))
    def visit(node):
        if not isinstance(node,dict):return
        if node.get('type')=='array' and 'enum' in node:
            values=node.pop('enum')
            node['minItems']=min(map(len,values));node['maxItems']=max(map(len,values))
            allowed=list(dict.fromkeys(item for value in values for item in value))
            if allowed:node['items']['enum']=allowed
        if 'exclusiveMinimum' in node:node['minimum']=node.pop('exclusiveMinimum')
        if 'exclusiveMaximum' in node:node['maximum']=node.pop('exclusiveMaximum')
        for child in node.get('properties',{}).values():visit(child)
        if 'items' in node:visit(node['items'])
    visit(result)
    return result
