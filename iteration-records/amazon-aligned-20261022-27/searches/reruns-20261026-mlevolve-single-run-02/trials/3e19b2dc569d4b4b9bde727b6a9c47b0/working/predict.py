from __future__ import annotations
import importlib.util
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
            spec = importlib.util.spec_from_file_location('recurrent_trial_model', os.path.join(_HERE, '..', 'train.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _CLASS = module.GRURec
        checkpoint = torch.load(os.path.join(_HERE, name + '_model.pt'), map_location=_DEVICE, weights_only=True)
        model = _CLASS(**checkpoint['meta']).to(_DEVICE)
        # Reconstruct the single network, including its jointly trained residual.
        if 'relative_bias' not in checkpoint['state_dict']:
            raise ValueError('Checkpoint lacks relative_bias: ' + name)
        if 'residual_gate' not in checkpoint['state_dict']:
            raise ValueError('Checkpoint lacks residual_gate: ' + name)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.eval()
        _MODELS[name] = model
    return _MODELS[name]


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    name = dataset_name if dataset_name is not None else _NAMES[0]
    model = _load(name)
    if len(candidates) == 0:
        return []
    hist = list(history)[-model.maxlen:]
    sequence = hist + [0] * (model.maxlen - len(hist))
    seq = torch.tensor([sequence], dtype=torch.long, device=_DEVICE)
    cand = torch.tensor([list(candidates)], dtype=torch.long, device=_DEVICE)
    return model.predict(seq, cand)[0].float().cpu().tolist()
