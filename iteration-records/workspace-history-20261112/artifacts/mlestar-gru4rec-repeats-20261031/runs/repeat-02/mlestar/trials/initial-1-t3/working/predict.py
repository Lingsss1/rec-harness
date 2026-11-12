"""Single-checkpoint per-category GRU4Rec prediction."""
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
_model_class = None


def _load_model_class():
    global _model_class
    if _model_class is None:
        path = os.path.join(_HERE, "..", "train.py")
        spec = importlib.util.spec_from_file_location(f"gru4rec_dropout_train_{os.getpid()}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _model_class = module.GRU4RecModel
    return _model_class


def _ensure_loaded(dataset_name):
    global _device
    if dataset_name in _models:
        return
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")
    if _device is None:
        _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(_HERE, f"{dataset_name}_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    model = _load_model_class()(
        itemnum=meta["itemnum"], hidden_units=meta["hidden_units"],
        num_layers=meta["num_layers"], dropout_rate=meta["dropout_rate"],
        input_dropout=meta["input_dropout"], output_dropout=meta["output_dropout"],
    ).to(_device)
    state = torch.load(os.path.join(_HERE, f"{dataset_name}_model.pt"),
                       map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    ds = dataset_name if dataset_name is not None else DATASETS[0]
    _ensure_loaded(ds)
    if not candidates:
        return []
    model, meta = _models[ds], _metas[ds]
    maxlen = meta["maxlen"]
    seq = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        seq[-len(tail):] = tail
    seq_t = torch.tensor([seq], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq_t, items).cpu().tolist()
    return [float(score) for score in scores]
