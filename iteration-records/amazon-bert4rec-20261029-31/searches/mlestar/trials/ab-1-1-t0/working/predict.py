"""Load exactly one trained BERT4Rec checkpoint per category."""
from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None
_ModelClass = None


def _ensure_loaded(dataset_name):
    global _device, _ModelClass
    if dataset_name in _models:
        return
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if _ModelClass is None:
        spec = importlib.util.spec_from_file_location(
            'bert4rec_trial_model', os.path.join(_HERE, '..', 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _ModelClass = module.BERT4RecModel
    with open(os.path.join(_HERE, f'{dataset_name}_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = _ModelClass(**meta).to(_device)
    state = torch.load(os.path.join(_HERE, f'{dataset_name}_model.pt'),
                       map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    dataset_name = dataset_name or _DATASETS[0]
    if dataset_name not in _DATASETS:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    _ensure_loaded(dataset_name)
    if not candidates:
        return []
    model = _models[dataset_name]
    maxlen = int(_metas[dataset_name]['maxlen'])
    seq = [0] * maxlen
    items = history[-maxlen:]
    if items:
        seq[-len(items):] = items
    seq_t = torch.tensor([seq], dtype=torch.long, device=_device)
    cand_t = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq_t, cand_t).cpu().tolist()
    return [float(score) for score in scores]
