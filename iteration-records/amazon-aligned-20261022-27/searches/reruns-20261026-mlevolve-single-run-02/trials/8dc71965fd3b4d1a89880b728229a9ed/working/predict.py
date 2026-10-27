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
    if meta['architecture_version'] != 'candidate_conditioned_history_v4':
        raise ValueError('Unsupported candidate-conditioned history architecture')
    if meta['history_candidate_query_dim'] != meta['dim']:
        raise ValueError('Unsupported candidate-query dimension')
    if meta['candidate_chunk_size'] < 1:
        raise ValueError('Invalid candidate chunk size')
    if meta['depth_attention_hidden_dim'] != meta['dim']:
        raise ValueError('Unsupported graph-depth attention hidden dimension')
    state = torch.load(os.path.join(_HERE, dataset + '_model.pt'),
                       map_location=_DEVICE, weights_only=True)
    model = _CLASS(len(meta['user_ids']), meta['itemnum'], meta['dim'],
                   meta['layers'], state['adjacency'],
                   history_len=meta['history_len'],
                   gate_bias=meta['gate_bias'],
                   history_conv_width=meta['history_conv_width'],
                   history_kernel_size=meta['history_kernel_size'],
                   architecture_version=meta['architecture_version']).to(_DEVICE)
    # Reconstruct exactly one category network, including depth attention and the
    # jointly trained candidate query. Strict loading restores its sole selected
    # checkpoint, tied item vectors, and full training adjacency.
    model.candidate_chunk_size = int(meta['candidate_chunk_size'])
    model.load_state_dict(state, strict=True)
    model._topology_keys = None
    model._cached = None
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
