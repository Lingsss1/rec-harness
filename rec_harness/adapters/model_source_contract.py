"""Mechanical admission of model-class edits under a fixed training module.

This limits the executable surface; resource/filesystem isolation remains required.
It does not infer scientific correctness from syntax.
"""
import ast

from rec_harness.core import ContractError,digest

CLASSES={'NextItNetModel','NextItNetBlock'}
METHODS={'__init__','forward','encode_sequence','score_candidates','_causal_conv'}
GLOBALS={'torch','nn','math','range','len','max','min','abs','int','float','enumerate','zip','super',*CLASSES}
CALLS={'range','len','max','min','abs','int','float','enumerate','zip','super','NextItNetBlock'}
MODULE_CALLS={
    'nn.Embedding','nn.Conv1d','nn.Linear','nn.LayerNorm','nn.Dropout','nn.ModuleList','nn.Parameter',
    'nn.ReLU','nn.GELU','nn.Sigmoid','nn.Tanh','nn.Identity','nn.Sequential',
    'nn.init.xavier_uniform_','nn.init.zeros_','nn.init.ones_','nn.init.constant_',
    'nn.functional.pad','nn.functional.normalize','nn.functional.relu','nn.functional.gelu',
    'torch.no_grad','torch.arange','torch.zeros','torch.ones','torch.full','torch.tensor',
    'torch.zeros_like','torch.ones_like','torch.where','torch.cat','torch.stack','torch.matmul',
    'torch.bmm','torch.sigmoid','torch.tanh','torch.softmax','torch.relu','torch.exp','torch.sqrt',
    'math.ceil','math.sqrt','math.log','math.log1p','math.exp',
}
TENSOR_METHODS={'float','long','to','unsqueeze','squeeze','expand','reshape','view','transpose',
    'permute','contiguous','sum','mean','cumsum','clamp','clamp_min','clamp_max','matmul','softmax',
    'sigmoid','tanh','relu','pow','sqrt','exp','norm','masked_fill','any','bool','tril','zero_',
    'new_zeros','new_ones','size','clone'}
NODES=(ast.Module,ast.ClassDef,ast.FunctionDef,ast.arguments,ast.arg,ast.Assign,ast.AnnAssign,
       ast.AugAssign,ast.Expr,ast.Return,ast.If,ast.For,ast.With,ast.withitem,ast.ListComp,
       ast.comprehension,ast.Call,ast.keyword,ast.Name,ast.Attribute,ast.Subscript,ast.Slice,
       ast.Constant,ast.List,ast.Tuple,ast.Dict,ast.BinOp,ast.UnaryOp,ast.BoolOp,ast.Compare,
       ast.IfExp,ast.Load,ast.Store,ast.operator,ast.unaryop,ast.boolop,ast.cmpop,ast.Pass)


def dotted(expr):
    if isinstance(expr,ast.Name):return expr.id
    if isinstance(expr,ast.Attribute):return dotted(expr.value)+'.'+expr.attr
    return ''


def review_source(baseline,candidate):
    try:before,after=ast.parse(baseline),ast.parse(candidate)
    except SyntaxError as error:raise ContractError('invalid model syntax: '+str(error)) from error
    def outside(tree):
        return ast.dump(ast.Module(body=[n for n in tree.body if not isinstance(n,ast.ClassDef) or n.name not in CLASSES],type_ignores=[]),include_attributes=False)
    if outside(before)!=outside(after):raise ContractError('candidate changed frozen imports, training, data or module configuration')
    classes=[n for n in after.body if isinstance(n,ast.ClassDef) and n.name in CLASSES]
    original={n.name:n for n in before.body if isinstance(n,ast.ClassDef) and n.name in CLASSES}
    if {n.name for n in classes}!=CLASSES or len(classes)!=2:raise ContractError('exactly the declared model classes required')
    block_class=next(n for n in classes if n.name=='NextItNetBlock')
    block_fields={target.attr for n in ast.walk(block_class) if isinstance(n,ast.Assign)
                  and isinstance(n.value,ast.Call) and dotted(n.value.func) in MODULE_CALLS
                  and dotted(n.value.func).startswith('nn.') for target in n.targets
                  if isinstance(target,ast.Attribute) and dotted(target.value)=='self'}
    for cls in classes:
        if cls.decorator_list or cls.keywords or len(cls.bases)!=1 or dotted(cls.bases[0])!='nn.Module':
            raise ContractError('model inheritance/decorators outside contract')
        if any(not isinstance(n,ast.FunctionDef) for n in cls.body):raise ContractError('model classes may only declare methods')
        if sorted(n.name for n in cls.body)!=sorted(n.name for n in original[cls.name].body if isinstance(n,ast.FunctionDef)):
            raise ContractError('model methods missing or duplicated')
        module_fields={dotted(target) for n in ast.walk(cls) if isinstance(n,ast.Assign)
                       and isinstance(n.value,ast.Call) and dotted(n.value.func).startswith('nn.')
                       and dotted(n.value.func) in MODULE_CALLS for target in n.targets
                       if isinstance(target,ast.Attribute) and dotted(target).startswith('self.')}
        for fn in cls.body:
            if fn.name not in METHODS or fn.decorator_list:raise ContractError('unsupported model method or decorator')
            original_fn=next((n for n in original[cls.name].body if isinstance(n,ast.FunctionDef) and n.name==fn.name),None)
            if original_fn is None or ast.dump(fn.args)!=ast.dump(original_fn.args):
                raise ContractError('frozen model method signature changed')
            locals_={n.id for n in ast.walk(fn) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
            args={n.arg for n in ast.walk(fn.args) if isinstance(n,ast.arg)}
            if locals_ & (GLOBALS|{'self'}) or (args-{'self'}) & GLOBALS:
                raise ContractError('cannot shadow model execution globals')
            blocks={n.target.id for n in ast.walk(fn) if isinstance(n,ast.For) and isinstance(n.target,ast.Name)
                    and dotted(n.iter)=='self.blocks'}
            block_calls={name+'.'+field for name in blocks for field in block_fields}
            block_helpers={name+'._causal_conv' for name in blocks
                           if any(isinstance(m,ast.FunctionDef) and m.name=='_causal_conv' for m in block_class.body)}
            if fn.name=='_causal_conv' and 'conv' in args:blocks.add('conv')
            for n in ast.walk(fn):
                if not isinstance(n,NODES):raise ContractError('unsupported source syntax: '+type(n).__name__)
                if isinstance(n,ast.FunctionDef) and n is not fn:raise ContractError('nested functions prohibited')
                if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id not in locals_|args|GLOBALS:
                    raise ContractError('undeclared model capability: '+n.id)
                if isinstance(n,ast.Attribute):
                    path=dotted(n)
                    if path.startswith(('torch.','nn.','math.')):
                        allowed=MODULE_CALLS|{'nn.Module','torch.Tensor','torch.float16','torch.bfloat16',
                                             'torch.float32','torch.float64','torch.int64','torch.bool'}
                        if path not in allowed and not any(v.startswith(path+'.') for v in allowed):
                            raise ContractError('unapproved library capability: '+path)
                    if n.attr.startswith('_') and not ((n.attr=='_causal_conv' and dotted(n.value)=='self') or path in block_helpers):
                        if not (n.attr=='__init__' and isinstance(n.value,ast.Call) and dotted(n.value.func)=='super'):
                            raise ContractError('private/introspection attribute prohibited')
                    if isinstance(n.ctx,ast.Store) and not (isinstance(n.value,ast.Name) and n.value.id=='self'):
                        raise ContractError('only model-owned attributes may be assigned')
                if isinstance(n,ast.Call):
                    path=dotted(n.func)
                    if isinstance(n.func,ast.Name):
                        if n.func.id not in CALLS|blocks:raise ContractError('unapproved callable: '+n.func.id)
                    elif isinstance(n.func,ast.Attribute):
                        if path.startswith(('torch.','nn.','math.')):
                            if path not in MODULE_CALLS:raise ContractError('unapproved library call: '+path)
                        elif path.startswith('self.'):
                            if path not in module_fields|{'self.'+m for m in METHODS} and n.func.attr not in TENSOR_METHODS:
                                raise ContractError('unapproved model call: '+path)
                        elif path in block_calls|block_helpers:pass
                        elif n.func.attr=='__init__' and isinstance(n.func.value,ast.Call) and dotted(n.func.value.func)=='super':pass
                        elif n.func.attr not in TENSOR_METHODS:raise ContractError('unapproved tensor call: '+n.func.attr)
                    else:raise ContractError('indirect calls prohibited')
    return {'contract':'nextitnet-model-classes-v1','source_ast_hash':digest(ast.dump(after,include_attributes=False)),
            'scope':'model-class edits only; fixed module/training protocol; isolated numerical checks still required'}
