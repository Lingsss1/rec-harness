"""One NextItNet checkpoint per category; no score fusion."""
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
        spec = importlib.util.spec_from_file_location('nextitnet_bprmax_training_definition', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _ModelClass = module.NextItNetModel
    return _ModelClass


def _ensure_loaded(dataset_name=None):
    global _device
    if _device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    requested = DATASETS if dataset_name is None else [dataset_name]
    ModelClass = _load_model_class()
    for ds in requested:
        if ds in _models:
            continue
        if ds not in DATASETS:
            raise ValueError(f'unknown dataset: {ds}')
        with open(os.path.join(_HERE, f'{ds}_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        model = ModelClass(
            itemnum=meta['itemnum'], hidden_units=meta['hidden_units'],
            num_blocks=meta['num_blocks'], kernel_size=meta['kernel_size'],
            dropout_rate=meta['dropout_rate'], maxlen=meta['maxlen']
        ).to(_device)
        state = torch.load(os.path.join(_HERE, f'{ds}_model.pt'), map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    ds = DATASETS[0] if dataset_name is None else dataset_name
    _ensure_loaded(ds)
    if not candidates:
        return []
    model = _models[ds]
    maxlen = int(_metas[ds]['maxlen'])
    sequence = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        sequence[-len(tail):] = tail
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, items).cpu().tolist()
    return [float(score) for score in scores]
