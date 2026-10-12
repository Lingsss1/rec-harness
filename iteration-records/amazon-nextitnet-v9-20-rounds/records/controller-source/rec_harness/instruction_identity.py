"""Identity of actual transmitted instructions, not inferred Skill execution."""
import copy
from .core import ContractError,digest,identifier,read_json


def validate_skills(value):
    if not isinstance(value,list):raise ContractError('role skills must be a list')
    names=set()
    for skill in value:
        if not isinstance(skill,dict) or set(skill)!={'name','content'}:
            raise ContractError('role skill requires name and content only')
        identifier(skill['name'])
        if skill['name'] in names:raise ContractError('duplicate role skill name')
        names.add(skill['name'])
        if not isinstance(skill['content'],str) or not skill['content'].strip():
            raise ContractError('role skill content must be nonempty text')
    return copy.deepcopy(value)


def role_skills(value,roles):
    if value is None:return {}
    if not isinstance(value,dict) or set(value)-set(roles):raise ContractError('unknown role in role_skills')
    return {role:validate_skills(skills) for role,skills in value.items()}


def skill_suffix(skills):
    skills=validate_skills(skills)
    if not skills:return ''
    # The exact transmitted content is archived, not read from mutable host paths.
    return '\nExplicit frozen research skills (subordinate to the frozen protocol):\n'+''.join(
        '\nSkill '+item['name']+' ['+digest(item['content'])+']:\n'+item['content']+'\n' for item in skills)


def manifest(prompt,schema,skills=None):
    skills=validate_skills([] if skills is None else skills)
    if skills:
        if not prompt.endswith(skill_suffix(skills)):raise ContractError('role skills differ from transmitted prompt')
        return {'schema':'role-instruction-identity-v2','prompt_hash':digest(prompt),
            'response_schema_hash':digest(schema),
            'explicit_skills':[{'name':item['name'],'content_hash':digest(item['content'])} for item in skills],
            'skill_visibility':'Explicit skill content supplied in the prompt; execution is not attested. CLI automatic Skill discovery and execution are not recorded.'}

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
    if recorded!=manifest(request['prompt'],schema,request.get('role_skills')):
        raise ContractError('instruction identity differs from archived request')
    return {**recorded,'request_ref':archive['request.json'],'schema_ref':archive['schema.json']}
