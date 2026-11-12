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


### >>> Self_EvolveRec-BLOCK-START: Preserve evaluator randomness when importing the seeded training module
def _load_model_class():
    import random
    import numpy as np

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        spec = importlib.util.spec_from_file_location(
            f"perdataset_train_{os.getpid()}", os.path.join(_HERE, "..", "train.py")
        )
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load the trained SASRec architecture from train.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.SASRec
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
### <<< Self_EvolveRec-BLOCK-END


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SASRec = _load_model_class()
    for ds in DATASETS:
        meta_path = os.path.join(_HERE, f"{ds}_meta.pkl")
        ckpt_path = os.path.join(_HERE, f"{ds}_model.pt")
        ### >>> Self_EvolveRec-BLOCK-START: Require complete artifacts for every dataset route
        for required_path in (meta_path, ckpt_path):
            if not os.path.isfile(required_path):
                raise FileNotFoundError(required_path)
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        _metas[ds] = meta
        m = SASRec(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            dropout_rate=0.0,
            maxlen=meta["maxlen"],
        ).to(_device)
        ### >>> Self_EvolveRec-BLOCK-START: Restore the selected transition residual alongside the neural checkpoint
        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        graph_path = os.path.join(_HERE, f"{ds}_transition.pt")
        alpha = float(meta.get("transition_alpha", 0.0))
        if os.path.isfile(graph_path):
            graph_state = torch.load(graph_path, map_location="cpu", weights_only=True)
            m.set_transition(graph_state, alpha=alpha)
        elif alpha != 0.0:
            raise FileNotFoundError(graph_path)
        m.eval()
        _models[ds] = m
        ### <<< Self_EvolveRec-BLOCK-END


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    ### >>> Self_EvolveRec-BLOCK-START: Reject invalid explicit routes while retaining the legacy default
    if not _models:
        raise RuntimeError("No trained dataset models were loaded")
    ds = dataset_name if dataset_name is not None else next(iter(_models))
    if ds not in _models:
        raise ValueError(f"Unknown dataset route: {ds!r}")
    model = _models[ds]
    meta = _metas[ds]
    ### <<< Self_EvolveRec-BLOCK-END

    maxlen = meta["maxlen"]
    hist   = history[-maxlen:]
    seq_arr = [0] * maxlen
    for i, item in enumerate(reversed(hist)):
        seq_arr[maxlen - 1 - i] = item

    seq_t = torch.tensor([seq_arr], dtype=torch.long, device=_device)

    ### >>> Self_EvolveRec-BLOCK-START: Exact itemwise fusion across complete candidate lists in bounded chunks
    if not candidates:
        return []
    with torch.no_grad():
        # Encode once; every chunk uses the same prefix and itemwise residual.
        final_feat = model.log2feats(seq_t)[:, -1, :]
        transition = getattr(model, "transition", None)
        scores = []
        for start in range(0, len(candidates), 4096):
            cand_t = torch.tensor([candidates[start:start + 4096]], dtype=torch.long, device=_device)
            chunk_scores = model.item_emb(cand_t).matmul(final_feat.unsqueeze(-1)).squeeze(-1)
            if transition is not None and transition.alpha != 0.0:
                chunk_scores = chunk_scores + transition.alpha * transition(seq_t, cand_t)
            scores.extend(chunk_scores[0].cpu().tolist())
    return scores
    ### <<< Self_EvolveRec-BLOCK-END
