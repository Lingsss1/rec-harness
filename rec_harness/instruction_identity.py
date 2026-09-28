"""Identity of actual transmitted instructions, not inferred Skill execution."""
from .core import ContractError,digest,read_json


def manifest(prompt,schema):
    return {'schema':'role-instruction-identity-v1','prompt_hash':digest(prompt),
        'response_schema_hash':digest(schema),
        'skill_visibility':'No Skill files explicitly supplied by this transport; CLI automatic Skill discovery and execution are not recorded.'}


def from_archive(archive,artifacts):
    if 'request.json' not in archive:return None
    request=read_json(artifacts.verify(archive['request.json']))
    recorded=request.get('instruction_identity')
    if recorded is None:return None  # Preserve legacy trace identity.
    if 'schema.json' not in archive or not isinstance(request.get('prompt'),str):
        raise ContractError('instruction identity lacks archived prompt or schema')
    schema=read_json(artifacts.verify(archive['schema.json']))
    if recorded!=manifest(request['prompt'],schema):
        raise ContractError('instruction identity differs from archived request')
    return {**recorded,'request_ref':archive['request.json'],'schema_ref':archive['schema.json']}
