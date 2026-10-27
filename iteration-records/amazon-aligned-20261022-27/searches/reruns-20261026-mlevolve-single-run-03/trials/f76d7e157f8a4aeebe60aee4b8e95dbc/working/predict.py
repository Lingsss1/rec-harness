from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific',
             'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_module = None
_device = None


def _load(dataset_name):
    global _module, _device
    if dataset_name in _models:
        return
    if _module is None:
        spec = importlib.util.spec_from_file_location(
            'masked_sasrec_architecture', os.path.join(_HERE, '..', 'train.py'))
        _module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_module)
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    meta = metadata['constructor']
    model = _module.MaskedSASRec(**meta).to(_device)
    state = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'),
                       map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id, history, candidates, *, dataset_name=None):
    name = dataset_name if dataset_name is not None else _DATASETS[0]
    if name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(name))
    _load(name)
    meta = _metas[name]
    sequence = _module.prediction_sequence(history, meta['itemnum'], meta['maxlen'])
    sequence = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor([candidates], dtype=torch.long, device=_device)
    with torch.inference_mode():
        return _models[name].predict(sequence, items)[0].cpu().tolist()
