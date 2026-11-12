from __future__ import annotations
import os
import importlib.util
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_CLASS = None
_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _load(dataset):
    global _CLASS
    if dataset in _MODELS:
        return _MODELS[dataset]
    if _CLASS is None:
        path = os.path.join(_HERE, '..', 'train.py')
        spec = importlib.util.spec_from_file_location('bertrec_trained_architecture', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CLASS = module.BERT4Rec
    checkpoint = torch.load(os.path.join(_HERE, dataset + '_model.pt'), map_location='cpu', weights_only=True)
    config = checkpoint['config']
    model = _CLASS(**config)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(_DEVICE).eval()
    _MODELS[dataset] = model
    return model


def predict(user_id, history, candidates, *, dataset_name=None):
    if not candidates:
        return []
    dataset = dataset_name if dataset_name is not None else _DATASETS[0]
    if dataset not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset))
    model = _load(dataset)
    tokens = list(history)[-(model.maxlen - 1):] + [model.mask_id]
    seq = [0] * (model.maxlen - len(tokens)) + tokens
    with torch.no_grad():
        sequence = torch.tensor([seq], dtype=torch.long, device=_DEVICE)
        items = torch.tensor([list(candidates)], dtype=torch.long, device=_DEVICE)
        valid = (items >= 1) & (items <= model.itemnum)
        safe_items = items.clamp(1, model.itemnum)
        scores = model.predict(sequence, safe_items).masked_fill(~valid, -1e9)
        return scores[0].cpu().tolist()
