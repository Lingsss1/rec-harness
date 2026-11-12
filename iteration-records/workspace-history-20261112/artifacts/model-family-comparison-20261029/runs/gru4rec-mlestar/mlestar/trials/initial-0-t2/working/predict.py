"""Route each category to exactly one trained GRU4Rec checkpoint."""
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
_ModelClass = None


def _ensure_loaded(dataset_name):
    global _device, _ModelClass
    if dataset_name in _models:
        return
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _ModelClass is None:
        spec = importlib.util.spec_from_file_location(
            f"gru4rec_train_{os.getpid()}", os.path.join(_HERE, "..", "train.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _ModelClass = module.GRU4RecModel
    with open(os.path.join(_HERE, f"{dataset_name}_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    model = _ModelClass(meta["itemnum"], meta["hidden_units"],
                        meta["num_layers"], meta["dropout_rate"]).to(_device)
    state = torch.load(os.path.join(_HERE, f"{dataset_name}_model.pt"),
                       map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    dataset_name = dataset_name or DATASETS[0]
    _ensure_loaded(dataset_name)
    model = _models[dataset_name]
    meta = _metas[dataset_name]
    maxlen = int(meta["maxlen"])
    seq_arr = [0] * maxlen
    tail = history[-maxlen:]
    if tail:
        seq_arr[-len(tail):] = tail
    seq = torch.tensor([seq_arr], dtype=torch.long, device=_device)
    cand = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.no_grad():
        scores = model.score_candidates(seq, cand).cpu().tolist()
    return [float(score) for score in scores]
