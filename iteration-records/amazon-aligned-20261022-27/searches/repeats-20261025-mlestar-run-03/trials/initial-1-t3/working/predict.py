from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_METAS = {}
_CLASS = None
_DEVICE = None


def _load(dataset):
    global _CLASS, _DEVICE
    if dataset in _MODELS:
        return
    if dataset not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset))
    if _CLASS is None:
        spec = importlib.util.spec_from_file_location('fmlprec_trial_train', os.path.join(_HERE, '..', 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CLASS = module.FMLPRec
        _DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, dataset + '_meta.pkl'), 'rb') as handle:
        meta = pickle.load(handle)
    model = _CLASS(**meta).to(_DEVICE)
    state = torch.load(os.path.join(_HERE, dataset + '_model.pt'), map_location=_DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    _MODELS[dataset] = model
    _METAS[dataset] = meta


def predict(user_id, history, candidates, *, dataset_name=None):
    dataset = dataset_name if dataset_name is not None else _DATASETS[0]
    _load(dataset)
    if len(candidates) == 0:
        return []
    model = _MODELS[dataset]
    meta = _METAS[dataset]
    maxlen = meta['maxlen']
    hist = list(history)[-maxlen:]
    seq = [0] * maxlen
    if hist:
        seq[-len(hist):] = [int(i) if 0 < int(i) <= meta['itemnum'] else 0 for i in hist]
    cand = [int(i) for i in candidates]
    valid = [0 < i <= meta['itemnum'] for i in cand]
    safe = [i if ok else 0 for i, ok in zip(cand, valid)]
    with torch.inference_mode():
        seq_t = torch.tensor([seq], dtype=torch.long, device=_DEVICE)
        cand_t = torch.tensor([safe], dtype=torch.long, device=_DEVICE)
        scores = model.predict(seq_t, cand_t)[0].cpu().tolist()
    return [float(score) if ok else -1e9 for score, ok in zip(scores, valid)]
