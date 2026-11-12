"""Structured, attributed tool definitions; author claims are not verification."""
from .core import ContractError,public_only

CONTRACT='diagnostic-method-v1'
FILENAME='tool-method.json'


def freeze_plan(plan):
    """New autonomous tasks require a definition; frozen legacy plans stay intact."""
    result={**plan,'tool_method_contract':plan.get('tool_method_contract',CONTRACT)}
    if result['tool_method_contract']!=CONTRACT:
        raise ContractError('new autonomous tools require diagnostic-method-v1')
    contract(result)
    return result


def contract(plan):
    value=plan.get('tool_method_contract')
    if value is None:return None
    if value!=CONTRACT:raise ContractError('unsupported tool method contract')
    return {'schema':CONTRACT,'file':FILENAME,'instructions':
        'Write tool-method.json under model/ and include it in changed_files. It is archived as a method card, not executable tool code. '
        'Choose route: existing_tool, paper, or original. Supply measurement, definition (precise quantity/formula), '
        'inputs, assumptions, complexity, failure_modes, positive_control, null_control, independent_reference. '
        'All are nonempty strings except assumptions/failure_modes (nonempty string lists). '
        'Supply sources as a list of {locator, method_mapping} strings: identify the primary paper and map its formula or section to this definition. '
        'Paper route requires sources. Supply predecessor_implementation_hash as the inherited implementation hash for existing_tool, otherwise null. '
        'Do not claim source content or scientific utility was verified merely because a source is cited.'}


def validate(card,source_hash=None):
    public_only(card)
    strings=['measurement','definition','inputs','complexity','positive_control','null_control','independent_reference']
    expected={'schema','route','sources','predecessor_implementation_hash','assumptions','failure_modes',*strings}
    if not isinstance(card,dict) or set(card)!=expected or card['schema']!=CONTRACT:
        raise ContractError('tool method card fields differ from contract')
    if card['route'] not in {'existing_tool','paper','original'}:raise ContractError('unknown tool method route')
    for key in strings:
        if not isinstance(card[key],str) or not card[key].strip():raise ContractError('tool method requires '+key)
    for key in ['assumptions','failure_modes']:
        if not isinstance(card[key],list) or not card[key] or any(not isinstance(v,str) or not v.strip() for v in card[key]):
            raise ContractError('tool method requires '+key)
    sources=card['sources']
    if not isinstance(sources,list):raise ContractError('tool method sources must be a list')
    for source in sources:
        if (not isinstance(source,dict) or set(source)!={'locator','method_mapping'} or
                any(not isinstance(v,str) or not v.strip() for v in source.values())):
            raise ContractError('tool method source requires locator and method mapping')
    if card['route']=='paper' and not sources:raise ContractError('paper tool method requires primary source mapping')
    predecessor=card['predecessor_implementation_hash']
    if card['route']=='existing_tool':
        if source_hash is None or predecessor!=source_hash:raise ContractError('tool method predecessor differs from inherited source')
    elif predecessor is not None:raise ContractError('only existing_tool route declares predecessor')
    return card
