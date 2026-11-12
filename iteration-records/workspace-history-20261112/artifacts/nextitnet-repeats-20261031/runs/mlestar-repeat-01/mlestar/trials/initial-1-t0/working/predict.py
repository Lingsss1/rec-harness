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


def _load_model_class():
    path = os.path.join(_HERE, '..', 'train.py')
    spec = importlib.util.spec_from_file_location(f'nextitnet_train_{os.getpid()}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NextItNetModel


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Model = _load_model_class()
    for ds in DATASETS:
        with open(os.path.join(_HERE, f'{ds}_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        model = Model(**meta).to(_device)
        state = torch.load(os.path.join(_HERE, f'{ds}_model.pt'), map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ds = DATASETS[0] if dataset_name is None else dataset_name
    if ds not in _models:
        raise ValueError(f'Unknown dataset: {ds}')
    if not candidates:
        return []
    model = _models[ds]
    maxlen = _metas[ds]['maxlen']
    seq_arr = [0] * maxlen
    tail = history[-maxlen:]
    if tail:
        seq_arr[-len(tail):] = tail
    seq = torch.tensor([seq_arr], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, items).cpu().tolist()
    return [float(score) for score in scores]
