from __future__ import annotations
import os
import pickle
import importlib.util
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_device = None
_class = None


def _load(dataset_name):
    global _device, _class
    if dataset_name in _models:
        return _models[dataset_name]
    if dataset_name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    if _class is None:
        spec = importlib.util.spec_from_file_location(
            'trained_single_network', os.path.join(_HERE, '..', 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _class = module.SASRec
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_num_threads(4)
    with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as f:
        config = pickle.load(f)
    model = _class(**config).to(_device)
    model.load_state_dict(torch.load(os.path.join(_HERE, dataset_name + '_model.pt'),
                                    map_location=_device, weights_only=True))
    model.eval()
    _models[dataset_name] = model
    return model


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    name = dataset_name if dataset_name is not None else _DATASETS[0]
    model = _load(name)
    if len(candidates) == 0:
        return []
    h = list(history[-model.maxlen:])
    seq = torch.tensor([[0] * (model.maxlen - len(h)) + h],
                       dtype=torch.long, device=_device)
    cand = torch.tensor([list(candidates)], dtype=torch.long, device=_device)
    return model.predict(seq, cand)[0].float().cpu().tolist()
