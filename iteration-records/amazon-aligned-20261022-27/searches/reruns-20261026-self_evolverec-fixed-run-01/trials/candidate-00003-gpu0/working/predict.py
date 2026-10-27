"""Per-dataset SASRec predict — routes by dataset_name kwarg.

Supports both legacy signature:
    predict(user_id, history, candidates) -> list[float]
and per-dataset signature (used by the RecHarness harness when available):
    predict(user_id, history, candidates, *, dataset_name) -> list[float]
"""
from __future__ import annotations

import importlib.util
import os
import pickle

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
### >>> Self_EvolveRec-BLOCK-START: Preserve all four predictor routes
DATASETS = list(DEFAULT_DATASETS)

_models: dict = {}
### <<< Self_EvolveRec-BLOCK-END
_metas: dict = {}
_device = None


def _load_model_class():
    spec = importlib.util.spec_from_file_location(
        f"perdataset_train_{os.getpid()}", os.path.join(_HERE, "..", "train.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SASRec


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SASRec = _load_model_class()
    for ds in DATASETS:
        meta_path = os.path.join(_HERE, f"{ds}_meta.pkl")
        ckpt_path = os.path.join(_HERE, f"{ds}_model.pt")
        ### >>> Self_EvolveRec-BLOCK-START: Fail explicitly when a required dataset artifact is missing
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Missing required model artifacts for {ds}: {meta_path}, {ckpt_path}"
            )
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        _metas[ds] = meta
        m = SASRec(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            ### >>> Self_EvolveRec-BLOCK-START: Reconstruct the trained pooling and gate configuration
            dropout_rate=0.0,
            maxlen=meta["maxlen"],
            pooling_enabled=meta["pooling_enabled"],
            pool_window=meta["pool_window"],
            fixed_gate=meta["fixed_gate"],
        ).to(_device)
            ### <<< Self_EvolveRec-BLOCK-END
        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        m.eval()
        _models[ds] = m


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    # Select model: use dataset_name kwarg if provided, else fall back to first available
    ### >>> Self_EvolveRec-BLOCK-START: Reject missing explicit dataset routes
    if dataset_name is not None and dataset_name not in _models:
        raise ValueError(f"No trained model for dataset {dataset_name!r}")
    if dataset_name and dataset_name in _models:
        model = _models[dataset_name]
        meta  = _metas[dataset_name]
    ### <<< Self_EvolveRec-BLOCK-END
    elif _models:
        ds    = next(iter(_models))
        model = _models[ds]
        meta  = _metas[ds]
    else:
        return [0.0] * len(candidates)

    maxlen = meta["maxlen"]
    hist   = history[-maxlen:]
    seq_arr = [0] * maxlen
    for i, item in enumerate(reversed(hist)):
        seq_arr[maxlen - 1 - i] = item

    seq_t = torch.tensor([seq_arr], dtype=torch.long, device=_device)

    with torch.no_grad():
        cand_t   = torch.tensor([candidates], dtype=torch.long, device=_device)
        scores   = model.predict(seq_t, cand_t)[0].cpu().tolist()

    return scores
