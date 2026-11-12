"""Single-checkpoint per-category BERT4Rec inference."""
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
_model_class = None


def _load_model_class():
    global _model_class
    if _model_class is None:
        path = os.path.join(_HERE, '..', 'train.py')
        spec = importlib.util.spec_from_file_location('bert4rec_contrastive_architecture', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _model_class = module.BERT4RecModel
    return _model_class


def _ensure_loaded(dataset_name):
    global _device
    if dataset_name in _models:
        return
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, f'{dataset_name}_meta.pkl'), 'rb') as handle:
        meta = pickle.load(handle)
    model = _load_model_class()(
        itemnum=meta['itemnum'], hidden_units=meta['hidden_units'],
        num_blocks=meta['num_blocks'], num_heads=meta['num_heads'],
        dropout_rate=meta['dropout_rate'], maxlen=meta['maxlen'],
    ).to(_device)
    state = torch.load(os.path.join(_HERE, f'{dataset_name}_model.pt'),
                       map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    if dataset_name is None:
        dataset_name = DATASETS[0]
    if dataset_name not in DATASETS:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    _ensure_loaded(dataset_name)
    if not candidates:
        return []
    model = _models[dataset_name]
    maxlen = int(_metas[dataset_name]['maxlen'])
    array = [0] * maxlen
    items = list(history[-maxlen:])
    if items:
        array[-len(items):] = items
    sequence = torch.tensor([array], dtype=torch.long, device=_device)
    candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(sequence, candidate_tensor).cpu().tolist()
    return [float(score) for score in scores]
