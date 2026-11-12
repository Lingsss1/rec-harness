"""Preregistered executable method scope; scientific mechanism remains a hypothesis."""
import ast
import json
from rec_harness.core import ContractError,digest
from .model_source_contract import CLASSES,METHODS

SCHEMA='nextitnet-method-scope-v1'


def method_catalog(source):
    tree=ast.parse(source)
    return sorted(cls.name+'.'+fn.name for cls in tree.body
        if isinstance(cls,ast.ClassDef) and cls.name in CLASSES
        for fn in cls.body if isinstance(fn,ast.FunctionDef))


def parse_contract(text, *, required=False, available_methods=None):
    if not isinstance(text,str):raise ContractError('insertion contract must be a JSON string')
    try:value=json.loads(text)
    except (ValueError,TypeError):
        if required or text.lstrip().startswith('{'):raise ContractError('structured insertion contract JSON required')
        return None
    if not isinstance(value,dict) or set(value)!={'schema','allowed_methods','rationale'} or value['schema']!=SCHEMA:
        raise ContractError('unsupported structured insertion contract')
    methods=value['allowed_methods']
    available=set(available_methods) if available_methods is not None else {c+'.'+m for c in CLASSES for m in METHODS}
    if not isinstance(methods,list) or not methods or any(not isinstance(m,str) for m in methods) or len(set(methods))!=len(methods) or not set(methods)<=available:
        raise ContractError('insertion contract methods must be distinct existing model methods')
    if not isinstance(value['rationale'],str) or not value['rationale'].strip():raise ContractError('insertion rationale required')
    return value


def verify_scope(contract,review):
    if contract is None:return None
    delta=review['structural_changes'];changed=[row['method'] for row in delta['methods']]
    if not changed:raise ContractError('insertion contract requires an actual model AST change')
    if not set(changed)<=set(contract['allowed_methods']):
        raise ContractError('model changed methods outside frozen insertion scope: '+', '.join(sorted(set(changed)-set(contract['allowed_methods']))))
    return {'status':'verified','contract_hash':digest(contract),'changed_methods':changed,
        'baseline_ast_hash':delta['baseline_ast_hash'],'candidate_ast_hash':delta['candidate_ast_hash'],
        'scope':'Actual AST changes confined to preregistered methods. Not proof of insertion semantics, mechanism or control adequacy.'}
