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
            'trained_ensemble_module', os.path.join(_HERE, '..', 'train.py'))
        _module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_module)
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_num_threads(4)
    if dataset_name not in _models:
        _models[dataset_name] = _module.load_ensemble(_HERE, dataset_name, _device)
    return _models[dataset_name]


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    ds = dataset_name if dataset_name is not None else _DATASETS[0]
    if ds not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(ds))
    model = _load(ds)
    # The imported SASRec reconstructs independent context item_emb and candidate
    # output_emb tables and strictly reloads both from the trained checkpoint.
    # Auxiliary prefixes are training-only; saved configurations and full-history
    # candidate scoring remain shared with validation through the model helpers.
    # Local validation and host inference share uniform-history preparation and
    # weighted reciprocal-rank fusion: MultVAE 0.75, SASRec 0.25,
    # constant 10 and one-based ranks, implemented by the shared training helper.
    return model.predict(user_id, history, candidates, dataset_name=ds)
