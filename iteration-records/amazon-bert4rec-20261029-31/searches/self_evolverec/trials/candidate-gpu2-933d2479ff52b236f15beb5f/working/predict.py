"""Per-dataset BERT4Rec predict function."""
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
_models = {}
_metas = {}
_device = None


def _load_model_class():
    spec = importlib.util.spec_from_file_location(
        f"bert4rec_perdataset_train_{os.getpid()}", os.path.join(_HERE, "..", "train.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BERT4RecModel


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ModelClass = _load_model_class()
    for ds in DATASETS:
        working_dir = _HERE if os.path.basename(_HERE) == "working" else os.path.join(_HERE, "working")
        meta_path = os.path.join(working_dir, f"{ds}_meta.pkl")
        ckpt_path = os.path.join(working_dir, f"{ds}_model.pt")
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            continue
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        model = ModelClass(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            ### >>> Self_EvolveRec-BLOCK-START: Reconstruct the saved single BERT4Rec configuration
            dropout_rate=meta["dropout_rate"],
            maxlen=meta["maxlen"],
            ### <<< Self_EvolveRec-BLOCK-END
        ).to(_device)
        model.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ### >>> Self_EvolveRec-BLOCK-START: Reject missing category checkpoints instead of silently misrouting predictions
    if dataset_name is not None:
        if dataset_name not in _models:
            raise RuntimeError(f"No trained BERT4Rec checkpoint for {dataset_name!r}")
        ds = dataset_name
    elif len(_models) == 1:
        ds = next(iter(_models))
    else:
        raise ValueError("dataset_name is required when multiple category models are loaded")
    model = _models[ds]
    meta = _metas[ds]
    ### <<< Self_EvolveRec-BLOCK-END

    maxlen = int(meta.get("maxlen", 128))
    ### >>> Self_EvolveRec-BLOCK-START: Emit history validation consistent with appended-mask model inference
    if maxlen < 1:
        raise ValueError("saved maxlen must be positive")
    seq_arr = [0] * maxlen
    # score_candidates shifts once to reserve the final position for MASK,
    # retaining exactly the most recent maxlen-1 history items.
    for i, item in enumerate(reversed(history[-maxlen:])):
        item = int(item)
        if not 1 <= item <= model.itemnum:
            raise ValueError(f"history contains a noncatalog item: {item}")
        seq_arr[maxlen - 1 - i] = item
    seq_t = torch.tensor([seq_arr], dtype=torch.long, device=_device)
    ### <<< Self_EvolveRec-BLOCK-END
    cand_t = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.no_grad():
        scores = model.score_candidates(seq_t, cand_t).detach().cpu().tolist()
    return [float(s) for s in scores]
