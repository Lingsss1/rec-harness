from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific',
             'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_MAPS = {}
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_CLASS = None


def _load(dataset):
    global _CLASS
    if dataset in _MODELS:
        return _MODELS[dataset]
    if _CLASS is None:
        spec = importlib.util.spec_from_file_location(
            'trained_graph_architecture', os.path.join(_HERE, '..', 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CLASS = module.GraphRecommender
    with open(os.path.join(_HERE, dataset + '_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    state = torch.load(os.path.join(_HERE, dataset + '_model.pt'),
                       map_location=_DEVICE, weights_only=True)
    model = _CLASS(len(meta['user_ids']), meta['itemnum'], meta['dim'],
                   meta['layers'], state['adjacency'],
                   history_len=meta['history_len'],
                   gate_bias=meta['gate_bias']).to(_DEVICE)
    model.load_state_dict(state, strict=True)
    model.prepare_inference()
    _MAPS[dataset] = {int(u): j for j, u in enumerate(meta['user_ids'])}
    _MODELS[dataset] = model
    return model


def predict(user_id, history, candidates, *, dataset_name=None):
    dataset = dataset_name if dataset_name is not None else _DATASETS[0]
    if dataset not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset))
    model = _load(dataset)
    if len(candidates) == 0:
        return []
    cand = torch.tensor(candidates, dtype=torch.long, device=_DEVICE)
    if bool(((cand < 1) | (cand > model.itemnum)).any()):
        raise ValueError('Candidate outside dataset item vocabulary')
    with torch.inference_mode():
        # Both routes apply the architecture's identical filtering, truncation
        # and padding. Attention is recomputed for every supplied history.
        valid_history = [int(i) for i in history
                         if 1 <= int(i) <= model.itemnum]
        valid_history = valid_history[-model.history_len:]
        index = _MAPS[dataset].get(int(user_id))
        if index is not None:
            users = torch.tensor([index], dtype=torch.long, device=_DEVICE)
            hist = model.pack_histories([valid_history], _DEVICE)
            scores = model.predict_indices(users, cand[None], hist)[0]
        else:
            scores = model.predict_history(valid_history, cand)
    return scores.cpu().tolist()
