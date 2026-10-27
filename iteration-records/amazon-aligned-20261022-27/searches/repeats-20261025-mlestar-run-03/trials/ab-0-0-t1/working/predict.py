from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    spec = importlib.util.spec_from_file_location('sampled_softmax_sasrec_runtime', os.path.join(_HERE, '..', 'train.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in DATASETS:
        with open(os.path.join(_HERE, name + '_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        # These saved options include normalization and training temperature.
        model = module.SASRec(**meta).to(_device)
        state = torch.load(os.path.join(_HERE, name + '_model.pt'), map_location=_device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _models[name] = model
        _metas[name] = meta


def predict(user_id, history, candidates, *, dataset_name=None):
    _ensure_loaded()
    if not len(candidates):
        return []
    name = dataset_name if dataset_name is not None else DATASETS[0]
    if name not in _models:
        raise ValueError('Unknown dataset: ' + str(name))
    model, meta = _models[name], _metas[name]
    maxlen, itemnum = meta['maxlen'], meta['itemnum']
    hist = [int(item) if 0 < int(item) <= itemnum else 0 for item in history[-maxlen:]]
    sequence = [0] * (maxlen - len(hist)) + hist
    candidate_ids = [int(item) for item in candidates]
    valid = [0 < item <= itemnum for item in candidate_ids]
    safe_candidates = [item if ok else 0 for item, ok in zip(candidate_ids, valid)]
    with torch.inference_mode():
        seq = torch.tensor([sequence], dtype=torch.long, device=_device)
        cand = torch.tensor([safe_candidates], dtype=torch.long, device=_device)
        scores = model.predict(seq, cand)[0].cpu().tolist()
    return [float(score) if ok else -1e9 for score, ok in zip(scores, valid)]
