"""Source-review decisions: positive checks never imply or erase blockers."""
from .agents import schema_for, validate_schema
from .core import ContractError


def validate_code_review(value, identity=None, *, allow_legacy=False, require_approval=False):
    schema = schema_for('reviewer')
    if allow_legacy and 'blocking_findings' not in value:
        # Old accepted records retain their original, conservative semantics.
        props = schema['properties']
        props['findings'] = props.pop('blocking_findings')
        props.pop('checks')
        schema['required'] = list(props)
    validate_schema(value, schema)
    if identity is not None and value['evidence_refs'] != [identity]:
        raise ContractError('reviewer implementation identity mismatch')
    blockers = value.get('blocking_findings', value.get('findings', []))
    if value['verdict'] == 'approve' and blockers:
        raise ContractError('reviewer approval has blocking findings; use checks for successful checks and limitations for caveats')
    if require_approval and value['verdict'] != 'approve':
        raise ContractError('independent code review blocked training: ' + value['verdict'] + '; ' + '; '.join(blockers))
