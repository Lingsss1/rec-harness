"""Label-free model inference entry. The host computes all rankings and metrics."""
import argparse
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path

from rec_harness.core import ContractError, digest, file_hash, read_json, write_json


@contextmanager
def bypass_block(model,index):
    """A fixed-weight intervention; always remove only the hook installed here."""
    if index is None:
        yield
        return
    blocks=getattr(model,'blocks',())
    if type(index) is not int or not 0<=index<len(blocks):
        raise ContractError('bypass block index outside model')
    handle=blocks[index].register_forward_hook(lambda module,inputs,output:inputs[0])
    try:yield
    finally:handle.remove()


def main():
    import torch
    parser = argparse.ArgumentParser()
    for name in ('source', 'source-hash', 'checkpoint', 'checkpoint-hash', 'request', 'request-hash', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--bypass-block',type=int)
    args = parser.parse_args()
    for path, expected in [(args.source,args.source_hash),(args.checkpoint,args.checkpoint_hash),(args.request,args.request_hash)]:
        if file_hash(path) != expected: raise ContractError('inference input hash mismatch')
    request = read_json(args.request)
    if request['schema'] != 'candidate-scores-request-v1': raise ContractError('unsupported score request')
    weights = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(weights, dict) or 'item_emb.weight' not in weights:
        raise ContractError('invalid checkpoint state_dict')
    if any(not isinstance(v,torch.Tensor) or not torch.isfinite(v).all() for v in weights.values()):
        raise ContractError('nonfinite checkpoint tensors')
    itemnum = weights['item_emb.weight'].shape[0] - 1
    for name in list(os.environ):
        if name.startswith('NEXTITNET_'): del os.environ[name]
    spec = importlib.util.spec_from_file_location('candidate_model',args.source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    torch.set_num_threads(1)
    model = module.NextItNetModel(itemnum, hidden_units=module.HIDDEN_UNITS, num_blocks=module.NUM_BLOCKS,
                                 kernel_size=module.KERNEL_SIZE, dropout_rate=module.DROPOUT_RATE, maxlen=module.MAXLEN)
    model.load_state_dict(weights,strict=True); model.eval()
    outputs = []
    with torch.inference_mode(),bypass_block(model,args.bypass_block):
        for start in range(0,len(request['rows']),64):
            rows = request['rows'][start:start+64]
            sequence = torch.zeros((len(rows),module.MAXLEN),dtype=torch.long)
            for i,row in enumerate(rows):
                history = row['history'][-module.MAXLEN:]
                sequence[i,-len(history):] = torch.tensor(history,dtype=torch.long)
            candidates = torch.tensor([r['candidates'] for r in rows],dtype=torch.long)
            encoded = model.encode_sequence(sequence)[:,-1]
            scores = (model.item_emb(candidates)*encoded[:,None,:]).sum(-1).tolist()
            outputs.extend({'row_id':r['row_id'],'scores':s} for r,s in zip(rows,scores))
    write_json(args.output,{'schema':'candidate-scores-response-v1','request_hash':digest(request),'rows':outputs})


if __name__ == '__main__': main()
