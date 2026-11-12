from __future__ import annotations
import os
import importlib.util
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_CLASS = None


def _load(ds):
    global _CLASS
    if ds in _MODELS:
        return _MODELS[ds]
    if _CLASS is None:
        spec = importlib.util.spec_from_file_location(
            'graph_training_definition', os.path.join(_HERE, '..', 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CLASS = module.PredictionModel
    state = torch.load(os.path.join(_HERE, ds + '_model.pt'),
                       map_location=_DEVICE, weights_only=True)
    model = _CLASS(state['user_ids'], state['user_vectors'], state['item_vectors'])
    model.to(_DEVICE).eval()
    _MODELS[ds] = model
    return model


def predict(user_id, history, candidates, *, dataset_name=None):
    ds = dataset_name if dataset_name is not None else _DATASETS[0]
    if ds not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(ds))
    model = _load(ds)
    # Preserve the complete chronological history and its validity mask.
    history = list(history)
    hist = history if history else [0]
    history_mask = [True] * len(history) if history else [False]
    with torch.inference_mode():
        scores = model(
            torch.tensor([user_id], dtype=torch.long, device=_DEVICE),
            torch.tensor([hist], dtype=torch.long, device=_DEVICE),
            torch.tensor([history_mask], dtype=torch.bool, device=_DEVICE),
            torch.tensor([candidates], dtype=torch.long, device=_DEVICE))
    return scores[0].cpu().tolist()
