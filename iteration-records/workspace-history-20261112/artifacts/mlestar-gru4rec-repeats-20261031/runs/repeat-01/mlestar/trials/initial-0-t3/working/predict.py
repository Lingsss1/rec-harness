from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded():
    global _device
    if len(_models) == len(_DATASETS):
        return
    _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    spec = importlib.util.spec_from_file_location('rdrop_gru_training_definition', os.path.join(_HERE, '..', 'train.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for dataset in _DATASETS:
        if dataset in _models:
            continue
        with open(os.path.join(_HERE, dataset + '_meta.pkl'), 'rb') as handle:
            meta = pickle.load(handle)
        model = module.GRU4RecModel(meta['itemnum'], meta['hidden_units'], meta['num_layers'], meta['dropout_rate']).to(_device)
        weights = torch.load(os.path.join(_HERE, dataset + '_model.pt'), map_location=_device, weights_only=True)
        model.load_state_dict(weights, strict=True)
        model.eval()
        _models[dataset] = model
        _metas[dataset] = meta


@torch.no_grad()
def predict(user_id, history, candidates, *, dataset_name=None):
    _ensure_loaded()
    if dataset_name is None:
        dataset_name = _DATASETS[0]
    if dataset_name not in _models:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    if not candidates:
        return []
    model = _models[dataset_name]
    meta = _metas[dataset_name]
    maxlen = meta['maxlen']
    seq = [0] * maxlen
    recent = list(history[-maxlen:])
    if recent:
        seq[-len(recent):] = recent
    sequence = torch.tensor([seq], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    return [float(value) for value in model.score_candidates(sequence, items).cpu().tolist()]
