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
_model_class = None

def _load(dataset):
    global _device, _model_class
    if dataset in _models:
        return
    if _device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if _model_class is None:
        spec = importlib.util.spec_from_file_location('gsasrec_trial_model', os.path.join(_HERE, '..', 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _model_class = module.SASRec
    with open(os.path.join(_HERE, dataset + '_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = _model_class(**meta).to(_device)
    state = torch.load(os.path.join(_HERE, dataset + '_model.pt'), map_location=_device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    _models[dataset] = model
    _metas[dataset] = meta

def predict(user_id, history, candidates, *, dataset_name=None):
    if not candidates:
        return []
    dataset = dataset_name if dataset_name is not None else _DATASETS[0]
    if dataset not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset))
    _load(dataset)
    meta = _metas[dataset]
    maxlen = meta['maxlen']
    hist = list(history)[-maxlen:]
    seq = [0] * (maxlen - len(hist)) + hist
    with torch.no_grad():
        sequence = torch.tensor([seq], dtype=torch.long, device=_device)
        items = torch.tensor([list(candidates)], dtype=torch.long, device=_device)
        return _models[dataset].predict(sequence, items)[0].float().cpu().tolist()
