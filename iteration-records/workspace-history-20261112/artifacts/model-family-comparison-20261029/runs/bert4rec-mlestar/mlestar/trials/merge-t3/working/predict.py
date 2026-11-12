"""Load exactly one trained BERT4Rec checkpoint per category."""
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
        spec = importlib.util.spec_from_file_location(
            'bert4rec_trial_architecture', os.path.join(_HERE, '..', 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _model_class = module.BERT4RecModel
    return _model_class


def _ensure_loaded(dataset):
    global _device
    if dataset in _models:
        return
    if dataset not in DATASETS:
        raise ValueError(f'Unknown dataset: {dataset}')
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    Model = _load_model_class()
    with open(os.path.join(_HERE, f'{dataset}_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = Model(itemnum=meta['itemnum'], hidden_units=meta['hidden_units'],
                  num_blocks=meta['num_blocks'], num_heads=meta['num_heads'],
                  dropout_rate=0.0, maxlen=meta['maxlen']).to(_device)
    state = torch.load(os.path.join(_HERE, f'{dataset}_model.pt'),
                       map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset] = model
    _metas[dataset] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    dataset = dataset_name if dataset_name is not None else DATASETS[0]
    _ensure_loaded(dataset)
    if not candidates:
        return []
    model, meta = _models[dataset], _metas[dataset]
    maxlen = int(meta['maxlen'])
    sequence = [0] * maxlen
    tail = history[-maxlen:]
    if tail:
        sequence[-len(tail):] = tail
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, items).cpu().tolist()
    return [float(score) for score in scores]
