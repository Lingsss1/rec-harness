"""One independently trained GRU4Rec model per Amazon category."""
from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["Movies_and_TV", "Industrial_and_Scientific", "Electronics", "CDs_and_Vinyl"]
_models = {}
_metas = {}
_device = None


def _ensure_loaded():
    global _device
    if len(_models) == len(DATASETS):
        return
    _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path = os.path.join(_HERE, "..", "train.py")
    spec = importlib.util.spec_from_file_location(f"gru4rec_dropout_train_{os.getpid()}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for dataset in DATASETS:
        if dataset in _models:
            continue
        with open(os.path.join(_HERE, f"{dataset}_meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        model = module.GRU4RecModel(
            itemnum=meta["itemnum"], hidden_units=meta["hidden_units"],
            num_layers=meta["num_layers"], input_dropout=meta["input_dropout"],
            output_dropout=meta["output_dropout"]).to(_device)
        state = torch.load(os.path.join(_HERE, f"{dataset}_model.pt"),
                           map_location=_device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _models[dataset] = model
        _metas[dataset] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    if dataset_name is None:
        dataset_name = DATASETS[0]
    if dataset_name not in _models:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")
    if not candidates:
        return []
    model = _models[dataset_name]
    meta = _metas[dataset_name]
    maxlen = meta["maxlen"]
    sequence = [0] * maxlen
    tail = history[-maxlen:]
    if tail:
        sequence[-len(tail):] = tail
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    cand = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, cand).cpu().tolist()
    return [float(score) for score in scores]
