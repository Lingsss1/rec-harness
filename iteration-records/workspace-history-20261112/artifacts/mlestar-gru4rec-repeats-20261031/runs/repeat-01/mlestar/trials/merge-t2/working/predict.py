"""One locked-dropout GRU4Rec checkpoint per category."""
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
    if len(_models) == len(DATASETS):
        return
    _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    spec = importlib.util.spec_from_file_location(
        'merged_gru4rec_architecture', os.path.join(_HERE, '..', 'train.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for ds in DATASETS:
        if ds in _models:
            continue
        with open(os.path.join(_HERE, f'{ds}_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        model = module.GRU4RecModel(
            itemnum=meta['itemnum'], hidden_units=meta['hidden_units'],
            num_layers=meta['num_layers'], dropout_rate=meta['dropout_rate'],
            input_dropout=meta['input_dropout'], output_dropout=meta['output_dropout'],
        ).to(_device)
        state = torch.load(os.path.join(_HERE, f'{ds}_model.pt'),
                           map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[ds], _metas[ds] = model, meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ds = DATASETS[0] if dataset_name is None else dataset_name
    if ds not in _models:
        raise ValueError(f'unknown dataset_name: {ds}')
    if len(candidates) == 0:
        return []
    model, meta = _models[ds], _metas[ds]
    maxlen = int(meta['maxlen'])
    seq_arr = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        seq_arr[-len(tail):] = tail
    seq = torch.tensor([seq_arr], dtype=torch.long, device=_device)
    cand = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, cand).cpu().tolist()
    return [float(score) for score in scores]
