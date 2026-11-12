"""Route each Amazon category to its single trained NextItNet checkpoint."""
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


def _load_model_class():
    global _ModelClass
    if _ModelClass is None:
        path = os.path.join(_HERE, '..', 'train.py')
        spec = importlib.util.spec_from_file_location('nextitnet_bprmax_model_definition', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _ModelClass = module.NextItNetModel
    return _ModelClass


def _ensure_loaded(dataset_name):
    global _device
    if dataset_name in _models:
        return
    if dataset_name not in DATASETS:
        raise ValueError(f'Unknown dataset_name: {dataset_name}')
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, f'{dataset_name}_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    kwargs = {k: meta[k] for k in ('itemnum', 'hidden_units', 'num_blocks', 'kernel_size', 'dropout_rate', 'maxlen')}
    model = _load_model_class()(**kwargs).to(_device)
    state = torch.load(os.path.join(_HERE, f'{dataset_name}_model.pt'), map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    ds = dataset_name if dataset_name is not None else DATASETS[0]
    _ensure_loaded(ds)
    if not candidates:
        return []
    model = _models[ds]
    meta = _metas[ds]
    maxlen = int(meta['maxlen'])
    seq_arr = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        seq_arr[-len(tail):] = tail
    seq = torch.tensor([seq_arr], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, items).float().cpu().tolist()
    return [float(s) for s in scores]
