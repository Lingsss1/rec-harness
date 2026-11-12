"""One trained GRU4Rec checkpoint per benchmark category."""
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
    spec = importlib.util.spec_from_file_location(f'gru4rec_train_{os.getpid()}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GRU4RecModel


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ModelClass = _load_model_class()
    loaded = {}
    metas = {}
    for ds in DATASETS:
        with open(os.path.join(_HERE, f'{ds}_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        model = ModelClass(itemnum=meta['itemnum'], hidden_units=meta['hidden_units'],
                           num_layers=meta['num_layers'], dropout_rate=0.0).to(_device)
        state = torch.load(os.path.join(_HERE, f'{ds}_model.pt'),
                           map_location=_device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        loaded[ds] = model
        metas[ds] = meta
    _models.update(loaded)
    _metas.update(metas)


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    if dataset_name is None:
        dataset_name = DATASETS[0]
    if dataset_name not in _models:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    model = _models[dataset_name]
    meta = _metas[dataset_name]
    maxlen = int(meta['maxlen'])
    seq_arr = [0] * maxlen
    for i, item in enumerate(reversed(history[-maxlen:])):
        seq_arr[maxlen - 1 - i] = item
    seq = torch.tensor([seq_arr], dtype=torch.long, device=_device)
    cand = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.no_grad():
        scores = model.score_candidates(seq, cand).detach().cpu().tolist()
    return [float(score) for score in scores]
