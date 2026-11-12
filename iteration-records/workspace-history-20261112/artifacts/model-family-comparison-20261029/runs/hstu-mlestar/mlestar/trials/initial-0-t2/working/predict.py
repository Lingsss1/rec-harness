from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_model_class = None


def _load(name):
    global _model_class
    if name in _models:
        return _models[name], _metas[name]
    if _model_class is None:
        spec = importlib.util.spec_from_file_location('hstu_gbce_architecture', os.path.join(_HERE, '..', 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _model_class = module.HSTURec
    with open(os.path.join(_HERE, name + '_meta.pkl'), 'rb') as stream:
        meta = pickle.load(stream)
    model = _model_class(**meta).to(_device)
    state = torch.load(os.path.join(_HERE, name + '_model.pt'), map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[name] = model
    _metas[name] = meta
    return model, meta


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    if not candidates:
        return []
    if dataset_name is None:
        dataset_name = _DATASETS[0]
    if dataset_name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    model, meta = _load(dataset_name)
    maxlen = meta['maxlen']
    history = [int(item) if 0 < int(item) <= meta['itemnum'] else 0 for item in history[-maxlen:]]
    sequence = [0] * (maxlen - len(history)) + history
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    legal = (items > 0) & (items <= meta['itemnum'])
    safe_items = items.clamp(0, meta['itemnum'])
    features = model.log2feats(seq)
    # Locate the final actual item, including histories with interior padding.
    positions = torch.arange(maxlen, device=_device)
    final = int(positions.masked_fill(seq[0].eq(0), -1).max().item())
    if final < 0:
        return [0.0 if bool(ok) else -1e9 for ok in legal.cpu().tolist()]
    scores = model.predict_score(features[0, final], safe_items)
    return scores.masked_fill(~legal, -1e9).cpu().tolist()
