from __future__ import annotations
import importlib.util
import os
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific',
             'Electronics', 'CDs_and_Vinyl']
_models = {}
_module = None
_device = None


def _load(dataset_name):
    global _module, _device
    if _module is None:
        spec = importlib.util.spec_from_file_location(
            'trained_multvae_module', os.path.join(_HERE, '..', 'train.py'))
        _module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_module)
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_num_threads(4)
    if dataset_name not in _models:
        saved = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'),
                           map_location='cpu', weights_only=True)
        model = _module.MultVAE(**saved['config'])
        model.load_state_dict(saved['state_dict'])
        _models[dataset_name] = model.to(_device).eval()
    return _models[dataset_name]


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    ds = dataset_name if dataset_name is not None else _DATASETS[0]
    if ds not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(ds))
    model = _load(ds)
    if not candidates:
        return []
    # Share uniform unique-item normalization with training/validation.
    row = _module.transform_history([int(i) for i in history])
    if row[0].size and (row[0].min() < 1 or row[0].max() > model.itemnum):
        raise ValueError('History item outside the trained catalog')
    ids, offsets, _, input_weights = _module.pack_rows([row])
    candidate_tensor = torch.tensor([candidates], dtype=torch.long, device=_device)
    scores = model.predict(
        ids.to(_device), offsets.to(_device), candidate_tensor,
        input_weights.to(_device))
    return scores[0].cpu().tolist()
