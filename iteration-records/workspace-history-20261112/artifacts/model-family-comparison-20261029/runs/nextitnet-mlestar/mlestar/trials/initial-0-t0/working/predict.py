"""One trained NextItNet checkpoint per Amazon category."""
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
_ModelClass = None


def _ensure_loaded(dataset_name):
    global _device, _ModelClass
    if dataset_name in _models:
        return
    if dataset_name not in DATASETS:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if _ModelClass is None:
        path = os.path.join(_HERE, '..', 'train.py')
        spec = importlib.util.spec_from_file_location(f'nextitnet_training_{os.getpid()}', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _ModelClass = module.NextItNetModel
    with open(os.path.join(_HERE, f'{dataset_name}_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    kwargs = {key: meta[key] for key in ('itemnum', 'hidden_units', 'num_blocks', 'kernel_size', 'dropout_rate', 'maxlen')}
    model = _ModelClass(**kwargs).to(_device)
    state = torch.load(os.path.join(_HERE, f'{dataset_name}_model.pt'), map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    if not candidates:
        return []
    ds = dataset_name if dataset_name is not None else DATASETS[0]
    _ensure_loaded(ds)
    model, meta = _models[ds], _metas[ds]
    maxlen = meta['maxlen']
    sequence = [0] * maxlen
    recent = history[-maxlen:]
    if recent:
        sequence[-len(recent):] = recent
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, items).cpu().tolist()
    return [float(score) for score in scores]
