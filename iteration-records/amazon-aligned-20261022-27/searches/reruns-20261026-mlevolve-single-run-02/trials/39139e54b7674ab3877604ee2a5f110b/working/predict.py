from __future__ import annotations
import importlib.util
import json
import os
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_NAMES = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_CLASS = None


def _load(name):
    global _CLASS
    if name not in _NAMES:
        raise ValueError('Unknown dataset: ' + str(name))
    if name not in _MODELS:
        if _CLASS is None:
            spec = importlib.util.spec_from_file_location(
                'history_training_architecture', os.path.join(_HERE, '..', 'train.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _CLASS = module.HistoryModel
        with open(os.path.join(_HERE, name + '_meta.json')) as f:
            config = json.load(f)
        model = _CLASS(**config).to(_DEVICE)
        state = torch.load(os.path.join(_HERE, name + '_model.pt'),
                           map_location=_DEVICE, weights_only=True)
        assert model.input_emb.weight is model.output_emb.weight
        model.load_state_dict(state, strict=True)
        assert model.input_emb.weight is model.output_emb.weight
        model.eval()
        _MODELS[name] = model
    return _MODELS[name]


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    model = _load(dataset_name if dataset_name is not None else _NAMES[0])
    history = list(history)[-model.maxlen:]
    seq = [0] * (model.maxlen - len(history)) + history
    seq = torch.tensor([seq], dtype=torch.long, device=_DEVICE)
    items = torch.tensor([list(candidates)], dtype=torch.long, device=_DEVICE)
    return model.predict(seq, items)[0].cpu().tolist()
