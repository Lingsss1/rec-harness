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


def _load_class():
    global _model_class
    if _model_class is None:
        path = os.path.abspath(os.path.join(_HERE, '..', 'train.py'))
        spec = importlib.util.spec_from_file_location('fmlprec_architecture_' + str(os.getpid()), path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _model_class = module.FMLPRec
    return _model_class


def _ensure_loaded(dataset):
    global _device
    if dataset in _models:
        return
    if dataset not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset))
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    with open(os.path.join(_HERE, dataset + '_meta.pkl'), 'rb') as handle:
        meta = pickle.load(handle)
    config = dict(meta['config'])
    config['dropout'] = 0.0
    model = _load_class()(**config).to(_device)
    state = torch.load(os.path.join(_HERE, dataset + '_model.pt'), map_location=_device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    _models[dataset] = model
    _metas[dataset] = meta


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    if len(candidates) == 0:
        return []
    dataset = dataset_name if dataset_name is not None else _DATASETS[0]
    _ensure_loaded(dataset)
    model = _models[dataset]
    maxlen = _metas[dataset]['config']['maxlen']
    hist = [int(item) for item in list(history)[-maxlen:]]
    hist = [item if 0 < item <= model.itemnum else 0 for item in hist]
    seq = torch.tensor([[0] * (maxlen - len(hist)) + hist], dtype=torch.long, device=_device)
    ids = torch.tensor([[int(item) for item in candidates]], dtype=torch.long, device=_device)
    valid = (ids > 0) & (ids <= model.itemnum)
    scores = model.predict(seq, ids.clamp(0, model.itemnum))
    scores = scores.masked_fill(~valid, float('-inf'))
    return scores[0].cpu().tolist()
