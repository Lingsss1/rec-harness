from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
_model_class = None


def _load(dataset_name):
    global _model_class
    if dataset_name in _models:
        return _models[dataset_name]
    if _model_class is None:
        spec = importlib.util.spec_from_file_location('trained_hstu_definition', os.path.join(_HERE, '..', 'train.py'))
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
    return model


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    name = _DATASETS[0] if dataset_name is None else dataset_name
    if name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(name))
    if not candidates:
        return []
    model = _load(name)
    history = list(history[-model.maxlen:])
    sequence = [0] * (model.maxlen - len(history)) + history
    sequence = torch.tensor([sequence], dtype=torch.long, device=_device)
    features = model.log2feats(sequence)[0, -1]
    return model.predict_score(features, candidates).float().cpu().tolist()
