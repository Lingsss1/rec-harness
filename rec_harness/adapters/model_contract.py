"""CPU structure and gradient probes for reviewed NextItNet-compatible sources.

Synthetic inputs only. Hooks record shapes/identities, never activation values.
Direct tensor identity edges are partial dataflow, not a fabricated full graph.
"""
import argparse
import importlib.util
import os
from pathlib import Path

from rec_harness.core import ContractError, digest, file_hash, write_json


def tensor_metadata(value):
    import torch
    if isinstance(value, torch.Tensor):
        return {'shape': list(value.shape), 'dtype': str(value.dtype), 'requires_grad': value.requires_grad}
    if isinstance(value, (list, tuple)): return [tensor_metadata(v) for v in value]
    if isinstance(value, dict): return {str(k): tensor_metadata(v) for k, v in value.items()}
    return {'python_type': type(value).__name__}


def tensors(value):
    import torch
    if isinstance(value, torch.Tensor): yield value
    elif isinstance(value, (tuple, list)):
        for v in value: yield from tensors(v)
    elif isinstance(value, dict):
        for v in value.values(): yield from tensors(v)


def trace_encode(model, sequence, model_id, source_hash):
    calls, handles, counts, stacks, producers, retained = [], [], {}, {}, {}, []
    def before(path):
        def hook(module, args):
            ordinal = counts.get(path, 0) + 1; counts[path] = ordinal
            callsite = f'encode_sequence:{ordinal}'
            component_id = 'component-' + digest([model_id, source_hash, path, callsite])[:32]
            inputs = list(tensors(args))
            record = {'id': component_id, 'module_path': path, 'callsite': callsite,
                      'module_type': type(module).__name__, 'input': tensor_metadata(args),
                      'direct_input_producers': sorted({producers[id(t)] for t in inputs if id(t) in producers}),
                      'parameters': {n: {'shape': list(p.shape), 'trainable': p.requires_grad}
                                     for n, p in module.named_parameters(recurse=False)}}
            stacks.setdefault(path, []).append(record); retained.extend(inputs)
        return hook
    def after(path):
        def hook(module, args, output):
            record = stacks[path].pop(); record['output'] = tensor_metadata(output)
            for t in tensors(output):
                producers[id(t)] = record['id']; retained.append(t)
            calls.append(record)
        return hook
    for path, module in model.named_modules():
        if path:
            handles += [module.register_forward_pre_hook(before(path)), module.register_forward_hook(after(path))]
    try:
        encoded = model.encode_sequence(sequence)
        shape = list(encoded.shape)
    finally:
        for handle in handles: handle.remove()
    return {'components': calls, 'encoded_shape': shape,
            'dataflow_coverage': 'direct_tensor_identity_only; functional transforms and residual arithmetic are not fully traced'}


def trace_operations(model, sequence, model_id, source_hash, *, max_operations=10000):
    """Record dispatched tensor operations on one concrete encode execution."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    if type(max_operations) is not int or max_operations < 1:
        raise ContractError('positive tensor graph operation limit required')
    values, operations, identities, retained = [], [], {}, []
    storage_ids, writes, module_stack, handles = {}, {}, [], []
    def storage(value):
        # Storage identity binds views without leaking process addresses into artifacts.
        key=value.untyped_storage()._cdata
        if key not in storage_ids:storage_ids[key]='storage-'+str(len(storage_ids))
        return storage_ids[key]
    def define(value, producer=None, name=None):
        ref='tensor-'+str(len(values))
        row={'id':ref,'metadata':tensor_metadata(value),'storage':storage(value),'producer':producer}
        if name is not None:row['external_name']=name
        values.append(row);identities[id(value)]=ref;retained.append(value)
        return ref
    def reference(value):
        return identities.get(id(value)) or define(value,name='unregistered_external')
    define(sequence,name='sequence')
    for name,value in [*model.named_parameters(),*model.named_buffers()]:
        if id(value) not in identities:define(value,name=name)
    class Recorder(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs=kwargs or {}
            if len(operations)>=max_operations:raise ContractError('tensor graph operation limit exceeded')
            inputs=list(tensors((args,kwargs)))
            refs=[reference(value) for value in inputs]
            dependencies=sorted({writes[storage(value)] for value in inputs if storage(value) in writes})
            ordinal=len(operations)
            operation_id='operation-'+digest([model_id,source_hash,ordinal,str(func)])[:32]
            mutated=[]
            for index,argument in enumerate(func._schema.arguments):
                if argument.alias_info is not None and argument.alias_info.is_write:
                    value=args[index] if index<len(args) else kwargs.get(argument.name)
                    mutated.extend(storage(tensor) for tensor in tensors(value))
            output=func(*args,**kwargs)
            output_refs=[define(value,producer=operation_id) for value in tensors(output)]
            operations.append({'id':operation_id,'ordinal':ordinal,'operator':str(func),
                'module_path':module_stack[-1] if module_stack else '',
                'inputs':refs,'outputs':output_refs,'mutation_dependencies':dependencies,
                'written_storages':sorted(set(mutated))})
            for key in mutated:writes[key]=operation_id
            return output
    def before(path):
        def hook(module,args):module_stack.append(path)
        return hook
    def after(module,args,output):module_stack.pop()
    for path,module in model.named_modules():
        if path:handles.extend([module.register_forward_pre_hook(before(path)),module.register_forward_hook(after)])
    try:
        with torch.no_grad(), Recorder():
            encoded=model.encode_sequence(sequence)
        outputs=[reference(value) for value in tensors(encoded)]
    finally:
        for handle in handles:handle.remove()
    result={'schema':'observed-tensor-graph-v1','model_id':model_id,'source_hash':source_hash,
        'tensors':values,'operations':operations,'outputs':outputs,
        'scope':'Dispatched tensor operations for this concrete encode_sequence input and execution mode. Includes functional and mutation dependencies; excludes unexecuted branches, Python scalar/control flow and operator internals. Not a universal static graph or semantic insertion proof.'}
    result['hash']=digest(result)
    return result


def check_contract(model, *, itemnum, length=16):
    import torch
    from torch.nn import functional as F
    if itemnum < 8 or length < 4: raise ContractError('probe needs itemnum>=8 and length>=4')
    model.cpu().eval()
    dense = (torch.arange(2 * length).reshape(2, length) % (itemnum - 2)) + 1
    with torch.no_grad():
        reference = model.encode_sequence(dense)
    if reference.ndim != 3 or reference.shape[:2] != dense.shape or not torch.isfinite(reference).all():
        raise ContractError('encode shape or finiteness contract failed')
    boundaries = sorted({0, length // 2, length - 2})
    for cutoff in boundaries:
        changed = dense.clone(); changed[:, cutoff + 1:] = itemnum
        with torch.no_grad(): alternate = model.encode_sequence(changed)
        if not torch.allclose(reference[:, :cutoff + 1], alternate[:, :cutoff + 1], atol=1e-5, rtol=1e-5):
            raise ContractError('future token perturbation changed causal prefix')
    sequences = dense.clone(); sequences[0, :-3] = 0; sequences[1, :-1] = 0
    with torch.no_grad():
        encoded = model.encode_sequence(sequences)
        candidates = torch.tensor([1, 2, itemnum])
        for row in range(2):
            predicted = model.score_candidates(sequences[row:row + 1], candidates)
            expected = model.item_emb(candidates) @ encoded[row, -1]
            if predicted.shape != expected.shape or not torch.allclose(predicted, expected, atol=1e-5, rtol=1e-5):
                raise ContractError('tail candidate readout contract failed')
        for size in (1, 2, length):
            empty = model.encode_sequence(torch.zeros((2, size), dtype=torch.long))
            if empty.shape != (2, size, reference.shape[-1]) or not torch.isfinite(empty).all():
                raise ContractError('empty history shape/finiteness contract failed')
    model.zero_grad(set_to_none=True)
    positive = torch.where(sequences != 0, sequences + 1, 0)
    negative = torch.where(sequences != 0, itemnum, 0)
    pos_score, neg_score = model(sequences, positive, negative)
    if pos_score.shape != sequences.shape or neg_score.shape != sequences.shape:
        raise ContractError('training score shape contract failed')
    mask = positive != 0
    loss = F.softplus(neg_score - pos_score)[mask].mean()
    if not torch.isfinite(loss): raise ContractError('training loss is nonfinite')
    loss.backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        if parameter.grad is None: raise ContractError(f'trainable parameter disconnected: {name}')
        if not torch.isfinite(parameter.grad).all(): raise ContractError(f'nonfinite gradient: {name}')
        gradients[name] = {'shape': list(parameter.grad.shape), 'norm': float(parameter.grad.norm()), 'finite': True}
    if not any(v['norm'] > 0 for v in gradients.values()): raise ContractError('all training gradients are zero')
    if not torch.equal(model.item_emb.weight.grad[0], torch.zeros_like(model.item_emb.weight.grad[0])):
        raise ContractError('padding embedding receives training gradient')
    captured = []
    def capture(module, args, output):
        output.retain_grad(); captured.append(output)
    handle = model.item_emb.register_forward_hook(capture)
    try:
        model.zero_grad(set_to_none=True)
        encoded = model.encode_sequence(dense)
        cutoff = length // 2
        objective = (encoded[0, cutoff] * torch.linspace(0.1, 1, encoded.shape[-1])).sum()
        objective.backward()
        gradient = captured[0].grad
        if gradient is None or not torch.isfinite(gradient).all(): raise ContractError('input gradient unavailable or nonfinite')
        if gradient[0, :cutoff + 1].abs().max() == 0: raise ContractError('causal gradient probe has no signal')
        if gradient[0, cutoff + 1:].abs().max() > 1e-7: raise ContractError('future embedding receives prefix gradient')
    finally:
        handle.remove(); model.zero_grad(set_to_none=True)
    return {'causal_boundaries': boundaries, 'causal_input_gradient': True, 'tail_readout': True,
            'empty_history_finite': True, 'padding_gradient_zero': True, 'training_loss': float(loss.detach()),
            'parameter_gradients': gradients, 'scope': 'fresh initialization, synthetic CPU inputs; not trained-checkpoint certification'}


def load_checkpoint(path, expected_hash):
    import torch
    if not expected_hash or file_hash(path) != expected_hash:
        raise ContractError('checkpoint hash mismatch')
    weights = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(weights, dict) or 'item_emb.weight' not in weights:
        raise ContractError('checkpoint must be model state_dict with item_emb.weight')
    for name, value in weights.items():
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ContractError(f'checkpoint tensor invalid or nonfinite: {name}')
    embedding = weights['item_emb.weight']
    if embedding.ndim != 2 or embedding.shape[0] < 9:
        raise ContractError('checkpoint embedding shape invalid')
    if file_hash(path) != expected_hash:
        raise ContractError('checkpoint changed during load')
    return weights, embedding.shape[0] - 1


def inspect_source(source, expected_hash, model_id, output, *, checkpoint=None, checkpoint_hash=None, threads=2):
    import torch
    if type(threads) is not int or threads not in (1, 2):
        raise ContractError('probe threads must be 1 or 2')
    source = Path(source).resolve()
    output = Path(output)
    if output.exists(): raise ContractError('model contract output already exists; use a new probe directory')
    if file_hash(source) != expected_hash: raise ContractError('source hash mismatch')
    if (checkpoint is None) != (checkpoint_hash is None):
        raise ContractError('checkpoint and checkpoint_hash must be supplied together')
    weights, itemnum = load_checkpoint(checkpoint, checkpoint_hash) if checkpoint else (None, 127)
    for name in list(os.environ):
        if name.startswith('NEXTITNET_'): del os.environ[name]
    torch.set_num_threads(threads); torch.manual_seed(20260923)
    spec = importlib.util.spec_from_file_location('reviewed_model', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    # Legacy modules seed at import time; reset after import so the recorded probe
    # seed, rather than the model file's global SEED, controls initialization.
    torch.manual_seed(20260923)
    settings = {key: getattr(module, key) for key in ['HIDDEN_UNITS', 'NUM_BLOCKS', 'KERNEL_SIZE', 'DROPOUT_RATE', 'MAXLEN']}
    model = module.NextItNetModel(itemnum, hidden_units=settings['HIDDEN_UNITS'], num_blocks=settings['NUM_BLOCKS'],
                kernel_size=settings['KERNEL_SIZE'], dropout_rate=settings['DROPOUT_RATE'], maxlen=settings['MAXLEN'])
    if weights is not None:
        model.load_state_dict(weights, strict=True)
    length = min(16, settings['MAXLEN'])
    model.eval()
    sequence = (torch.arange(2 * length).reshape(2, length) % itemnum) + 1
    report = {'model_id': model_id, 'source_hash': expected_hash, 'settings': settings,
              'inspector_hash': file_hash(__file__),
              'torch_version': torch.__version__, 'input_protocol': {'itemnum': itemnum, 'length': length, 'batch': 2, 'seed': 20260923, 'cpu_threads': threads},
              'structure': trace_encode(model, sequence, model_id, expected_hash),
              'tensor_graph': trace_operations(model, sequence, model_id, expected_hash),
              'checks': check_contract(model, itemnum=itemnum, length=length)}
    if checkpoint:
        if file_hash(checkpoint) != checkpoint_hash:
            raise ContractError('checkpoint changed during probe')
        report['checkpoint_hash'] = checkpoint_hash
        report['checks']['scope'] = 'trained checkpoint, synthetic CPU inputs; not a held-out quality evaluation'
    output.parent.mkdir(parents=True, exist_ok=True)
    for label, original, expected in [('source', source, expected_hash), ('inspector', Path(__file__), report['inspector_hash'])]:
        filename = expected + '.py'
        frozen = output.parent / filename
        if not frozen.exists(): frozen.write_bytes(original.read_bytes())
        if file_hash(frozen) != expected or file_hash(original) != expected:
            raise ContractError(f'{label} changed during model probe')
        report[label + '_filename'] = filename
    report['hash'] = digest(report); write_json(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True); parser.add_argument('--expected-hash', required=True)
    parser.add_argument('--model-id', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--threads', type=int, choices=[1, 2], default=2)
    parser.add_argument('--checkpoint'); parser.add_argument('--checkpoint-hash')
    args = parser.parse_args()
    report = inspect_source(args.source, args.expected_hash, args.model_id, args.output,
                            checkpoint=args.checkpoint, checkpoint_hash=args.checkpoint_hash, threads=args.threads)
    print({'model_id': report['model_id'], 'components': len(report['structure']['components']), 'hash': report['hash'], 'checks': 'passed'})
