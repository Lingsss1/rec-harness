from __future__ import annotations
import importlib.util
import os
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_NAMES = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_model_class = None


def _load(name):
    global _model_class
    if name not in _NAMES:
        raise ValueError('Unknown dataset: ' + str(name))
    if name not in _models:
        if _model_class is None:
            spec = importlib.util.spec_from_file_location(
                'trained_sasrec_architecture', os.path.join(_HERE, '..', 'train.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _model_class = module.SASRec
        checkpoint = torch.load(os.path.join(_HERE, name + '_model.pt'),
                                map_location='cpu', weights_only=True)
        model = _model_class(**checkpoint['config'])
        model.load_state_dict(checkpoint['state'])
        model.to(_device).eval()
        _models[name] = model
    return _models[name]


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    if not candidates:
        return []
    name = dataset_name if dataset_name is not None else _NAMES[0]
    model = _load(name)
    h = list(history)[-model.maxlen:]
    sequence = [0] * (model.maxlen - len(h)) + h
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    cand = torch.tensor([list(candidates)], dtype=torch.long, device=_device)
    return model.predict(seq, cand)[0].float().cpu().tolist()
