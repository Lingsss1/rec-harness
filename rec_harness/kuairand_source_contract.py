"""Mechanical model-only surface checks; not a substitute for an OS boundary."""
import ast

from .core import ContractError, digest

IMPORTS = {'torch', 'math', 'dataclasses', 'typing', 'collections', 'functools',
           'longseq_bench.models', 'longseq_bench.ccformer'}
BANNED_NAMES = {'open', 'eval', 'exec', 'compile', '__import__', 'getattr', 'setattr', 'delattr',
                'globals', 'locals', 'vars', 'breakpoint', 'input'}
BANNED_ATTRS = {'save', 'load', 'load_library', 'load_inline', 'set_default_device', 'set_default_dtype',
                'set_num_threads', 'set_num_interop_threads', 'from_file', 'hub', 'distributed',
                'multiprocessing', 'subprocess', 'storage', 'untyped_storage', 'data', 'system', 'popen'}


def review_source(source):
    try: tree = ast.parse(source)
    except SyntaxError as error: raise ContractError('candidate syntax error: '+str(error)) from error
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for alias in n.names:
                if alias.name not in IMPORTS and not alias.name.startswith('torch.nn'):
                    raise ContractError('candidate import outside model surface: '+alias.name)
        if isinstance(n, ast.ImportFrom):
            module = n.module or ''
            if n.level or (module not in IMPORTS and not module.startswith(('torch.nn', 'torch.utils.checkpoint'))):
                raise ContractError('candidate import outside model surface: '+module)
            if any(a.name == '*' or a.name.startswith('_') or a.name in BANNED_ATTRS for a in n.names):
                raise ContractError('candidate wildcard/private/capability import prohibited')
        if isinstance(n, ast.Name) and n.id in BANNED_NAMES:
            raise ContractError('candidate dynamic execution or IO prohibited: '+n.id)
        if isinstance(n, ast.Attribute):
            if n.attr in BANNED_ATTRS or n.attr.startswith('__') and n.attr != '__init__':
                raise ContractError('candidate introspection or external capability prohibited: '+n.attr)
            if isinstance(n.ctx, ast.Store):
                root = n.value
                while isinstance(root, ast.Attribute): root = root.value
                if not isinstance(root, ast.Name) or root.id != 'self':
                    raise ContractError('candidate cannot assign external library/host attributes')
        if isinstance(n, (ast.Global, ast.Nonlocal)):
            raise ContractError('candidate cannot rebind external state')
        if isinstance(n, ast.FunctionDef) and n.name.startswith('__') and n.name != '__init__':
            raise ContractError('candidate special execution hooks prohibited')
    if not any(isinstance(n, ast.FunctionDef) and n.name == 'build' for n in tree.body):
        raise ContractError('candidate must define build(config)')
    return {'contract': 'kuairand-model-only-v1', 'ast_hash': digest(ast.dump(tree, include_attributes=False)),
        'qualification': 'Mechanical executable-surface check; trusted process mode, not proof of sandbox isolation or mechanism validity.'}
