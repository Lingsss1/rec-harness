from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    spec = importlib.util.spec_from_file_location('trained_nextitnet_definition', os.path.join(_HERE, '..', 'train.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for ds in DATASETS:
        with open(os.path.join(_HERE, ds + '_meta.pkl'), 'rb') as f:
            config = pickle.load(f)
        model = module.NextItNetModel(**config).to(_device)
        model.load_state_dict(torch.load(os.path.join(_HERE, ds + '_model.pt'), map_location=_device, weights_only=True))
        model.eval()
        _models[ds] = model
        _metas[ds] = config


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ds = DATASETS[0] if dataset_name is None else dataset_name
    if ds not in _models:
        raise ValueError('Unknown dataset: ' + str(ds))
    if not candidates:
        return []
    model = _models[ds]
    maxlen = _metas[ds]['maxlen']
    sequence = [0] * maxlen
    h = history[-maxlen:]
    if h:
        sequence[-len(h):] = h
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        return [float(x) for x in model.score_candidates(seq, items).cpu().tolist()]
