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
DATASETS = [
    part.strip()
    for part in os.environ.get("GAGC_DATASETS", ",".join(DEFAULT_DATASETS)).split(",")
    if part.strip()
]

_models: dict = {}
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
            raise FileNotFoundError(f"Missing trained artifacts for {ds}")
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        _metas[ds] = meta
        ### >>> Self_EvolveRec-BLOCK-START: Reload the exact sparse residual architecture in the exported predictor
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        buffer_names = (
            "edge_keys", "edge_counts", "source_mass", "target_mass", "total_mass",
            "training_users", "user_edge_keys", "user_source_keys", "user_source_values",
            "user_target_keys", "user_target_values", "user_mass",
        )
        m = SASRec(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            dropout_rate=0.0,
            maxlen=meta["maxlen"],
            transition_anchors=meta["transition_anchors"],
            transition_statistics={name: state[name] for name in buffer_names},
        )
        m.load_state_dict(state, strict=True)
        m = m.to(_device)
        ### <<< Self_EvolveRec-BLOCK-END
        m.eval()
        _models[ds] = m


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    # Select model: use dataset_name kwarg if provided, else fall back to first available
    ### >>> Self_EvolveRec-BLOCK-START: Preserve explicit category routing and reject unknown routes
    if dataset_name is not None:
        if dataset_name not in _models:
            raise ValueError(f"Unknown or unloaded dataset: {dataset_name!r}")
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

    ### >>> Self_EvolveRec-BLOCK-START: Reuse causal features during full-catalog chunked inference
    if not candidates:
        return []
    with torch.inference_mode():
        features = model.log2feats(seq_t)
        positions = torch.arange(seq_t.shape[1], device=_device).expand_as(seq_t)
        last = positions.masked_fill(seq_t == 0, -1).amax(dim=1).clamp_min(0)
        final = features[torch.arange(seq_t.shape[0], device=_device), last]
        scores = []
        for start in range(0, len(candidates), 4096):
            cand_t = torch.tensor([candidates[start:start + 4096]],
                                  dtype=torch.long, device=_device)
            scores.extend(model.predict_score(final, cand_t)[0].cpu().tolist())
    return scores
    ### <<< Self_EvolveRec-BLOCK-END
