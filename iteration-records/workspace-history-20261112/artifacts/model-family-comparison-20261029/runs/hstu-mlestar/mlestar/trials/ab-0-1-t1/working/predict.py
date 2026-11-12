from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_model_class = None
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _ensure_loaded(dataset_name):
    global _model_class
    if dataset_name in _models:
        return
    if _model_class is None:
        spec = importlib.util.spec_from_file_location('trained_hstu_architecture', os.path.join(_HERE, 'hstu_architecture.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _model_class = module.HSTURec
    with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as stream:
        meta = pickle.load(stream)
    model = _model_class(**meta['config']).to(_device)
    state = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'), map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id, history, candidates, *, dataset_name=None):
    if not candidates:
        return []
    if dataset_name is None:
        dataset_name = _DATASETS[0]
    if dataset_name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    _ensure_loaded(dataset_name)
    model = _models[dataset_name]
    history = list(history)[-model.maxlen:]
    if any(int(item) < 0 or int(item) > model.itemnum for item in history):
        raise ValueError('History item outside dataset catalog')
    if any(int(item) <= 0 or int(item) > model.itemnum for item in candidates):
        raise ValueError('Candidate outside dataset catalog')
    seq = torch.zeros((1, model.maxlen), dtype=torch.long, device=_device)
    if history:
        seq[0, -len(history):] = torch.tensor(history, dtype=torch.long, device=_device)
    with torch.inference_mode():
        features = model.readout(seq)
        scores = model.predict_score(features, candidates)[0]
    return scores.cpu().tolist()
