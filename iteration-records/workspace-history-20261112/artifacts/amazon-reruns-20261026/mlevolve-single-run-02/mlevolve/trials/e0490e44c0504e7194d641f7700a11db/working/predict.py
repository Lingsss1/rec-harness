from __future__ import annotations
import importlib.util
import os
import pickle
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific',
             'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None
_pack_history = None


def _ensure_loaded():
    global _device, _pack_history
    if _models:
        return
    torch.set_num_threads(4)
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = os.path.join(_HERE, '..', 'train.py')
    spec = importlib.util.spec_from_file_location('convrec_training_module', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _pack_history = module.pack_history
    for name in _DATASETS:
        with open(os.path.join(_HERE, name + '_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        model = module.ConvRec(**meta).to(_device)
        state = torch.load(os.path.join(_HERE, name + '_model.pt'),
                           map_location=_device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _models[name] = model
        _metas[name] = meta


def predict(user_id, history, candidates, *, dataset_name=None):
    _ensure_loaded()
    name = dataset_name if dataset_name is not None else _DATASETS[0]
    if name not in _models:
        raise ValueError('Unknown dataset: ' + str(name))
    model = _models[name]
    seq = _pack_history(history, _metas[name]['maxlen'])
    seq = torch.as_tensor(seq, dtype=torch.long, device=_device).unsqueeze(0)
    cand = torch.as_tensor(candidates, dtype=torch.long, device=_device).unsqueeze(0)
    with torch.inference_mode():
        return model.predict(seq, cand)[0].float().cpu().tolist()
