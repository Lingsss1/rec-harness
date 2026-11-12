"""One trained GRU4Rec checkpoint per category; output dropout ablated."""
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


def _load_model_class():
    path = os.path.join(_HERE, "..", "train.py")
    spec = importlib.util.spec_from_file_location("locked_gru4rec_architecture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GRU4RecModel


def _ensure_loaded():
    global _device
    if len(_models) == len(DATASETS):
        return
    _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Model = _load_model_class()
    for ds in DATASETS:
        if ds in _models:
            continue
        with open(os.path.join(_HERE, f"{ds}_meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        model = Model(
            itemnum=meta["itemnum"], hidden_units=meta["hidden_units"],
            num_layers=meta["num_layers"], dropout_rate=meta["dropout_rate"],
            input_dropout=meta["input_dropout"], output_dropout=meta["output_dropout"],
        ).to(_device)
        checkpoint = torch.load(os.path.join(_HERE, f"{ds}_model.pt"),
                                map_location=_device, weights_only=True)
        model.load_state_dict(checkpoint, strict=True)
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ds = dataset_name if dataset_name is not None else DATASETS[0]
    if ds not in _models:
        raise ValueError(f"unknown dataset: {ds}")
    if len(candidates) == 0:
        return []
    model, meta = _models[ds], _metas[ds]
    maxlen = int(meta["maxlen"])
    seq = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        seq[-len(tail):] = tail
    seq_t = torch.tensor([seq], dtype=torch.long, device=_device)
    cand_t = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq_t, cand_t).cpu().tolist()
    return [float(score) for score in scores]
