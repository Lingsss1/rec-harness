from __future__ import annotations
import importlib.util
import os
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_DEVICE = None
_CLASS = None


def _load(dataset):
    global _DEVICE, _CLASS
    if dataset in _MODELS:
        return _MODELS[dataset]
    if _CLASS is None:
        spec = importlib.util.spec_from_file_location('gsasrec_trial_model', os.path.join(_HERE, '..', 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CLASS = module.SASRec
        _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(os.path.join(_HERE, dataset + '_model.pt'), map_location='cpu', weights_only=True)
    model = _CLASS(**checkpoint['config'])
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
    hist = list(history[-model.maxlen:])
    seq = [0] * (model.maxlen - len(hist)) + hist
    with torch.inference_mode():
        seq_t = torch.tensor([seq], dtype=torch.long, device=_DEVICE)
        cand_t = torch.tensor([candidates], dtype=torch.long, device=_DEVICE)
        return model.predict(seq_t, cand_t)[0].cpu().tolist()
