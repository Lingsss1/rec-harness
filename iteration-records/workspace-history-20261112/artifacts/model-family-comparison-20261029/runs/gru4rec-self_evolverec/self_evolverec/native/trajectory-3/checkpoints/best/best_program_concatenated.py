# === main_code.py ===
"""Per-dataset GRU4Rec baseline for Amazon Reviews.

Trains one independent GRU4Rec checkpoint per dataset while sharing one default
hyper-parameter profile across datasets. The generated predict.py routes by the
optional dataset_name keyword used by the RecHarness Amazon Reviews harness.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

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

### >>> Self_EvolveRec-BLOCK-START: Preserve required data boundary and training seed
DATA_DIR = os.environ["GAGC_DATA_DIR"]

SEED = 42
### <<< Self_EvolveRec-BLOCK-END
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def _read_split(path: str) -> dict[int, list[int]]:
    data: dict[int, list[int]] = defaultdict(list)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                data[int(parts[0])].append(int(parts[1]))
    return dict(data)


def _load_itemnum(data_dir: str, dataset_name: str) -> int:
    stats_path = os.path.join(data_dir, "dataset_stats.json")
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        if dataset_name in stats:
            return int(stats[dataset_name]["itemnum"])
    itemnum = 0
    for split in ("train", "valid"):
        path = os.path.join(data_dir, f"{dataset_name}_{split}.txt")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    itemnum = max(itemnum, int(parts[1]))
    return itemnum


def load_dataset(dataset_name: str, data_dir: str):
    user_train = _read_split(os.path.join(data_dir, f"{dataset_name}_train.txt"))
    user_valid = _read_split(os.path.join(data_dir, f"{dataset_name}_valid.txt"))
    itemnum = _load_itemnum(data_dir, dataset_name)
    return user_train, user_valid, itemnum


def sample_batch(
    user_train: dict[int, list[int]], itemnum: int, batch_size: int, maxlen: int
):
    users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise RuntimeError("no users with at least two training interactions")
    batch_seq, batch_pos, batch_neg = [], [], []
    for _ in range(batch_size):
        user = random.choice(users)
        history = user_train[user]
        seq = np.zeros(maxlen, dtype=np.int64)
        pos = np.zeros(maxlen, dtype=np.int64)
        neg = np.zeros(maxlen, dtype=np.int64)
        nxt = history[-1]
        idx = maxlen - 1
        interacted = set(history)
        for item in reversed(history[:-1]):
            seq[idx] = item
            pos[idx] = nxt
            if nxt != 0:
                neg_item = random.randint(1, itemnum)
                while neg_item in interacted:
                    neg_item = random.randint(1, itemnum)
                neg[idx] = neg_item
            nxt = item
            idx -= 1
            if idx < 0:
                break
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg)
    return np.asarray(batch_seq), np.asarray(batch_pos), np.asarray(batch_neg)


def _history_to_tensor(
    history: list[int], maxlen: int, device: torch.device
) -> torch.Tensor:
    seq_arr = np.zeros(maxlen, dtype=np.int64)
    for i, item in enumerate(reversed(history[-maxlen:])):
        seq_arr[maxlen - 1 - i] = item
    return torch.tensor([seq_arr], dtype=torch.long, device=device)


def evaluate_val(
    model: nn.Module,
    user_train: dict[int, list[int]],
    user_valid: dict[int, list[int]],
    itemnum: int,
    device: torch.device,
    maxlen: int,
    num_users: int = 200,
) -> dict[str, float]:
    ### >>> Self_EvolveRec-BLOCK-START: Make local checkpoint validation reproducible without consuming training randomness
    rng = random.Random(SEED)
    val_users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(val_users) > num_users:
        val_users = rng.sample(val_users, num_users)
    ### <<< Self_EvolveRec-BLOCK-END
    if not val_users:
        return {"HR@10": 0.0, "HR@20": 0.0, "NDCG@10": 0.0, "NDCG@20": 0.0}

    hr10 = hr20 = ndcg10 = ndcg20 = 0.0
    model.eval()
    with torch.no_grad():
        for user in val_users:
            target = user_valid[user][0]
            history = user_train[user]
            interacted = set(history) | {target, 0}
            ### >>> Self_EvolveRec-BLOCK-START: Use fixed unique validation negatives and reject impossible candidate counts
            available = itemnum - sum(1 <= item <= itemnum for item in interacted)
            if available < 99:
                raise ValueError(
                    f"Validation user {user} has only {available} unseen negatives; "
                    "99 unique negatives are required"
                )
            candidates = [target]
            while len(candidates) < 100:
                neg_item = rng.randint(1, itemnum)
                if neg_item not in interacted:
                    interacted.add(neg_item)
                    candidates.append(neg_item)
            ### <<< Self_EvolveRec-BLOCK-END
            seq_t = _history_to_tensor(history, maxlen, device)
            cand_t = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq_t, cand_t).detach().cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
            hr10 += 1.0 if rank < 10 else 0.0
            hr20 += 1.0 if rank < 20 else 0.0
            ndcg10 += (1.0 / math.log2(rank + 2)) if rank < 10 else 0.0
            ndcg20 += (1.0 / math.log2(rank + 2)) if rank < 20 else 0.0
    n = float(len(val_users))
    model.train()
    return {
        "HR@10": hr10 / n,
        "HR@20": hr20 / n,
        "NDCG@10": ndcg10 / n,
        "NDCG@20": ndcg20 / n,
    }


def bpr_sequence_loss(
    model: nn.Module, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor
) -> torch.Tensor:
    pos_scores, neg_scores = model(seq, pos, neg)
    mask = (pos != 0).float()
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


# Hyper-parameters
MAXLEN = int(os.environ.get("GRU4REC_MAXLEN", "128"))
HIDDEN_UNITS = int(os.environ.get("GRU4REC_HIDDEN", "128"))
NUM_LAYERS = int(os.environ.get("GRU4REC_LAYERS", "1"))
DROPOUT_RATE = float(os.environ.get("GRU4REC_DROPOUT", "0.1"))
LR = float(os.environ.get("GRU4REC_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("GRU4REC_WD", "0.0"))
BATCH_SIZE = int(os.environ.get("GRU4REC_BATCH", "128"))
### >>> Self_EvolveRec-BLOCK-START: Preserve the fixed training budget
NUM_BATCHES = 200
NUM_EPOCHS = min(
    200,
    max(
        1, int(os.environ.get("GRU4REC_EPOCHS", os.environ.get("SASREC_EPOCHS", "200")))
    ),
)
### <<< Self_EvolveRec-BLOCK-END
EVAL_EVERY = int(os.environ.get("GRU4REC_EVAL_EVERY", "5"))
PATIENCE = int(os.environ.get("GRU4REC_PATIENCE", "20"))


class GRU4RecModel(nn.Module):
    ### >>> Self_EvolveRec-BLOCK-START: Explicit retention architecture configuration
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 128,
        num_layers: int = 1,
        dropout_rate: float = 0.1,
        architecture_variant: str = "learned_pre_gate",
        retention_init: float = 0.99,
    ):
        ### <<< Self_EvolveRec-BLOCK-END
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        dropout = dropout_rate if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            hidden_units,
            hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        ### >>> Self_EvolveRec-BLOCK-START: Embedding-conditioned coordinate retention before ordinary GRU updates
        self.dropout = nn.Dropout(dropout_rate)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
        if architecture_variant != "learned_pre_gate":
            raise ValueError(f"Unsupported architecture: {architecture_variant}")
        if not 0.0 < retention_init < 1.0:
            raise ValueError("retention_init must lie strictly between zero and one")
        self.architecture_config = {
            "itemnum": itemnum,
            "hidden_units": hidden_units,
            "num_layers": num_layers,
            "dropout_rate": dropout_rate,
            "architecture_variant": architecture_variant,
            "retention_init": retention_init,
        }
        # Shared across layers when a multilayer baseline is requested. For the
        # default single layer this adds H * (2 * H + 1) learned parameters.
        self.retention_x = nn.Linear(hidden_units, hidden_units, bias=True)
        self.retention_h = nn.Linear(hidden_units, hidden_units, bias=False)
        nn.init.zeros_(self.retention_x.weight)
        nn.init.constant_(
            self.retention_x.bias, math.log(retention_init / (1.0 - retention_init))
        )
        nn.init.zeros_(self.retention_h.weight)
        # A diagnostic hook may observe this module without changing feature
        # return values. Its input/output shape is layers x batch x hidden.
        self.retention_gate = nn.Sigmoid()
        ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Skip shared leading padding while preserving causal gated features and interfaces
    def log2feats(self, seq: torch.Tensor) -> torch.Tensor:
        emb = self.dropout(self.item_emb(seq))
        occupied = seq.ne(0).any(dim=0).nonzero(as_tuple=False).flatten()
        if occupied.numel() == 0:
            return emb * 0.0
        # One host synchronization per batch avoids recurrent calls over the
        # shared padding prefix. Retain the original B x L x H feature shape.
        start = int(occupied[0].item())
        projected_input = self.retention_x(emb[:, start:])
        hidden = emb.new_zeros(self.gru.num_layers, seq.size(0), self.hidden_units)
        outputs = []
        for offset, t in enumerate(range(start, seq.size(1))):
            valid = seq[:, t].ne(0)
            retention = self.retention_gate(
                projected_input[:, offset].unsqueeze(0) + self.retention_h(hidden)
            )
            # Keep the ordinary GRU equations and all per-row padding masks.
            _, proposed = self.gru(emb[:, t : t + 1], (retention * hidden).contiguous())
            hidden = torch.where(valid[None, :, None], proposed, hidden)
            outputs.append(
                torch.where(valid[:, None], hidden[-1], torch.zeros_like(hidden[-1]))
            )
        suffix = torch.stack(outputs, dim=1)
        if start == 0:
            return suffix
        prefix = emb.new_zeros(seq.size(0), start, self.hidden_units)
        return torch.cat((prefix, suffix), dim=1)

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        out = self.log2feats(seq)
        if seq.size(1) == 0:
            return self.item_emb.weight.new_zeros(seq.size(0), self.hidden_units)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values
        features = out[torch.arange(seq.size(0), device=seq.device), last.clamp(min=0)]
        return torch.where(last[:, None].ge(0), features, torch.zeros_like(features))

    def predict_score(
        self, features: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        item_features = self.item_emb(items)
        if item_features.ndim == features.ndim + 1:
            features = features.unsqueeze(-2)
        return (features * item_features).sum(dim=-1)

    def forward(self, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        out = self.log2feats(seq)
        return self.predict_score(out, pos), self.predict_score(out, neg)

    ### <<< Self_EvolveRec-BLOCK-END

    def score_candidates(
        self, seq: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        user_emb = self.encode(seq)[0]
        cand_emb = self.item_emb(candidates)
        return torch.matmul(cand_emb, user_emb)


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str) -> float:
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f"empty dataset or item vocabulary for {dataset_name}")

    model = GRU4RecModel(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT_RATE,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    best_hr10 = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, neg = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg, dtype=torch.long, device=device)
            loss = bpr_sequence_loss(model, seq_t, pos_t, neg_t)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(NUM_BATCHES, 1)
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            val_m = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            hr10 = val_m["HR@10"]
            print(
                f"  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}"
                f"  loss={avg_loss:.4f}  val_score: {hr10:.4f}"
                f"  HR@20={val_m['HR@20']:.4f}  NDCG@10={val_m['NDCG@10']:.4f}",
                flush=True,
            )
            if hr10 > best_hr10:
                best_hr10 = hr10
                best_epoch = epoch
                ### >>> Self_EvolveRec-BLOCK-START: Save one selected model with its architecture configuration
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "architecture_config": model.architecture_config,
                    },
                    ckpt_path,
                )
                ### <<< Self_EvolveRec-BLOCK-END
            if epoch - best_epoch >= PATIENCE:
                print(f"  [{dataset_name}] Early stopping at epoch {epoch}", flush=True)
                break
        elif epoch % 10 == 0:
            print(
                f"  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}  loss={avg_loss:.4f}",
                flush=True,
            )

    ### >>> Self_EvolveRec-BLOCK-START: Preserve checkpoint format in the fallback save path
    if not os.path.exists(ckpt_path):
        torch.save(
            {
                "state_dict": model.state_dict(),
                "architecture_config": model.architecture_config,
            },
            ckpt_path,
        )
    ### <<< Self_EvolveRec-BLOCK-END

    meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "itemnum": itemnum,
                "maxlen": MAXLEN,
                "hidden_units": HIDDEN_UNITS,
                "num_layers": NUM_LAYERS,
                ### >>> Self_EvolveRec-BLOCK-START: Record inherited training settings and retention architecture
                "dropout_rate": DROPOUT_RATE,
                "architecture_config": model.architecture_config,
                "training_config": {
                    "seed": SEED,
                    "max_epochs": NUM_EPOCHS,
                    "batches_per_epoch": NUM_BATCHES,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LR,
                    "weight_decay": WEIGHT_DECAY,
                    "optimizer": "Adam",
                    "loss": "baseline_sampled_bpr",
                    "eval_every": EVAL_EVERY,
                    "patience": PATIENCE,
                    "best_epoch": best_epoch,
                    "gradient_clip_norm": 5.0,
                },
                ### <<< Self_EvolveRec-BLOCK-END
            },
            f,
        )
    print(
        f"  [{dataset_name}] Best val HR@10={max(best_hr10, 0.0):.4f}  ckpt={ckpt_path}",
        flush=True,
    )
    return max(best_hr10, 0.0)


def write_predict_script(working_dir: str) -> None:
    script = r'''"""Per-dataset GRU4Rec predict function."""
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
        f"gru4rec_perdataset_train_{os.getpid()}", os.path.join(_HERE, "..", "train.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GRU4RecModel


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
        ### >>> Self_EvolveRec-BLOCK-START: Reconstruct the single selected retention-gated checkpoint exactly
        checkpoint = torch.load(ckpt_path, map_location=_device, weights_only=True)
        config = checkpoint["architecture_config"]
        if config != meta["architecture_config"]:
            raise RuntimeError(f"Checkpoint/metadata architecture mismatch for {ds}")
        model = ModelClass(**config).to(_device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        ### <<< Self_EvolveRec-BLOCK-END
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    ### >>> Self_EvolveRec-BLOCK-START: Reject missing category checkpoints instead of silently scoring with another category
    _ensure_loaded()
    if dataset_name is not None:
        if dataset_name not in _models:
            raise RuntimeError(f"No trained GRU4Rec checkpoint for {dataset_name!r}")
        ds = dataset_name
    elif len(_models) == 1:
        ds = next(iter(_models))
    else:
        raise ValueError("dataset_name is required when routing multiple category models")
    model = _models[ds]
    meta = _metas[ds]
    ### <<< Self_EvolveRec-BLOCK-END

    maxlen = int(meta.get("maxlen", 128))
    seq_arr = [0] * maxlen
    for i, item in enumerate(reversed(history[-maxlen:])):
        seq_arr[maxlen - 1 - i] = item
    seq_t = torch.tensor([seq_arr], dtype=torch.long, device=_device)
    cand_t = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.no_grad():
        scores = model.score_candidates(seq_t, cand_t).detach().cpu().tolist()
    return [float(s) for s in scores]
'''
    predict_path = os.path.join(working_dir, "predict.py")
    with open(predict_path, "w") as f:
        f.write(script)
    print(f"predict.py written -> {predict_path}", flush=True)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Per-dataset GRU4Rec training on {device}", flush=True)
    scores = []
    for dataset_name in DATASETS:
        scores.append(train_one_dataset(dataset_name, device, DATA_DIR))
    working_dir = os.path.join(os.path.dirname(__file__), "working")
    write_predict_script(working_dir)
    avg = sum(scores) / max(len(scores), 1)
    print(f"\nAll datasets done. Average val HR@10 = {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()


# === self_evolverec_interface.py ===
def self_evolverec_interface():
    raise RuntimeError("Training and grading are executed by the common host boundary.")


# === diagnosis_tools.py ===
import torch
import numpy as np
import torch.nn.functional as F
import json

### >>> Self_EvolveRec-BLOCK-START: Reproducible conditional temporal sensitivity diagnostics
import random
import time
from collections import Counter

### <<< Self_EvolveRec-BLOCK-END


def diagnosis_interpreter_prompt(raw_diagnosis):
    return f"""
    You are a Senior AI Researcher analyzing the mathematical health of a Recommendation System.
    
    Here are the raw mathematical metrics measured from the current model:
    {json.dumps(raw_diagnosis['metrics'], indent=2)}
    
    Here are the definitions of what these metrics mean:
    {json.dumps(raw_diagnosis['metric_definitions'], indent=2)}
    
    Your task: produce a concise diagnosis summary that captures ONLY the core findings and implications.
    Do NOT propose web searches or action plans. Do NOT list "what to look up".

    Strictly follow the JSON format below.

    Output Format (JSON):
    {{
    "status": <CRITICAL | NEEDS_IMPROVEMENT | STABLE>,
    "core_findings": [
        "<1-2 sentences: the most important interpretation of the metrics>",
        "<1-2 sentences: the second most important interpretation (only if truly necessary)>"
    ],
    "key_implications": [
        "<1 sentence: what this implies about model behavior>",
        "<1 sentence: what this implies about training dynamics or representation>",
        "<optional 1 sentence: risk/trade-off if relevant>"
    ],
    "evidence": {{
        "headline_metrics": {{
        "<metric_name>": <value>,
        "<metric_name>": <value>
        }},
        "brief_metric_read": {{
        "<metric_name>": "<very short interpretation tied to its definition>",
        "<metric_name>": "<very short interpretation tied to its definition>"
        }}
    }}
    }}

    Rules:
    - Keep it short: core_findings <= 3 bullets, key_implications <= 3 bullets.
    - Use technical terms only when they are essential (e.g., 'gradient', 'loss landscape', 'inductive bias'), and keep them brief.
    ### >>> Self_EvolveRec-BLOCK-START: Restrict interpretation to measured conditional training-prefix effects
    - Every statement must be grounded in the provided metric values/definitions and include relevant counts and correctness checks.
    - Temporal effects are conditional in-sample sensitivity of fitted checkpoints to selected training prefixes and fixed candidates. They do not establish generalization, causal user intent, or validation gains.
    - Failed correctness checks invalidate temporal interpretation. Null metrics mean unavailable evidence, not zero effects. Missing categories invalidate four-category estimates.
    - State movement without ranking damage establishes representation sensitivity only.
    - Distinguish measurements from alternative explanations: memorization, distribution shift, redundant histories, and scoring bottlenecks.
    - Do not infer exposure, compatibility, dislikes, current intent, mood, ratings, review text, or simulator traits.
    - Confidence intervals condition on the checkpoint, prefixes, and candidates; they omit training-seed variability and validation-search selection optimism.
    - Only independently measured official macro validation HR@10 selects models. Do not claim an improvement from these diagnostics.
    # Review checks: inspected shapes, final-position readout, predictor parity,
    # finite outcomes, RNG restoration, stale-state handling and GPU batch caps.
    # Static review only; no execution or validation improvement is claimed.
    # Four-category aggregation still requires all four labeled category records;
    # a single ModelView cannot produce that measurement independently.
    ### <<< Self_EvolveRec-BLOCK-END
    - If metrics look healthy, status should remain STABLE and implications should focus on "what is already working" plus one potential risk to monitor.
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve ModelView and use only available training information
    def __init__(self, view, device):
        self.model = view
        self.module = view.model
        self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train
        self._observed_maxlen = None
        self._margin_count = 0
        self._margin_frequency = {}

    def _get_categories(self, item_id: int):
        # Item-category metadata is unavailable in this matched task.
        return []

    def _histories(self):
        histories = self.user_train.get("History", self.user_train)
        return {
            u: [int(i) for i in history if int(i) != 0]
            for u, history in histories.items()
        }

    def _features(self, sequence):
        # Preserve the supplied ModelView's numpy-input adapter.
        return self.model.log2feats(sequence.detach().cpu().numpy())

    def _last(self, features, sequence):
        positions = torch.arange(sequence.size(1), device=sequence.device)
        last = positions.expand_as(sequence).masked_fill(sequence.eq(0), -1).amax(1)
        states = features[
            torch.arange(sequence.size(0), device=sequence.device), last.clamp_min(0)
        ]
        return torch.where(last[:, None].ge(0), states, torch.zeros_like(states))

    def _pad(self, histories, width):
        sequence = torch.zeros(
            len(histories), width, dtype=torch.long, device=self.device
        )
        for row, history in enumerate(histories):
            if history:
                sequence[row, -len(history) :] = torch.as_tensor(
                    history, dtype=torch.long, device=self.device
                )
        return sequence

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()
            emb = emb[
                1 : self.model.model.itemnum + 1
            ]  # exclude padding and reserved mask tokens

            n = emb.size(0)
            if n <= 2:
                return 0.0

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx)
            x = F.normalize(x, p=2, dim=-1)

            gram = x @ x.t()
            offdiag_sum = gram.sum() - gram.diag().sum()
            offdiag_cnt = k * (k - 1)
            mean_cos = (offdiag_sum / offdiag_cnt).clamp(-1, 1).item()

            collapse_score = (mean_cos + 1.0) * 0.5
            return round(float(collapse_score), 4)

    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(
        self,
        dataloader,
        num_batches: int = 10,
        eps: float = 0.5,
        topk_categories: int = 20,
    ):
        ### >>> Self_EvolveRec-BLOCK-START: Measure actual recurrent model with frequency proxy counts
        self.module.eval()
        frequency = Counter(i for h in self._histories().values() for i in h)
        frequency_margins = {"1-4": [], "5-19": [], "20+": []}
        margins_all = []
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        for step, batch in enumerate(dataloader):
            if step >= num_batches:
                break
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 4):
                continue

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            ### >>> Self_EvolveRec-BLOCK-START: Record existing window and preserve family-specific feature adapter
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            if seq_t.ndim != 2 or seq_t.size(1) < 1:
                raise ValueError("Training sequences must have shape B x L with L >= 1")
            self._observed_maxlen = int(seq_t.size(1))
            log_feats = torch.cat(
                [self._features(chunk) for chunk in seq_t.split(64)], dim=0
            )
            ### <<< Self_EvolveRec-BLOCK-END

            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)

            ### >>> Self_EvolveRec-BLOCK-START: Bound sampled-margin scoring and reject invalid measurements
            if pos_t.shape != seq_t.shape or neg_t.shape != seq_t.shape:
                raise ValueError(
                    "Sequence, positive and negative tensors must share B x L shape"
                )
            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = torch.cat(
                [
                    self.model.predict_score(features, items)
                    for features, items in zip(log_feats.split(64), all_items.split(64))
                ],
                dim=0,
            )
            if all_logits.shape != all_items.shape:
                raise ValueError("Per-position candidate scoring must return B x L x K")
            if not torch.isfinite(all_logits).all():
                raise ValueError("Sampled-margin scoring returned nonfinite values")
            margin = all_logits[..., 0] - all_logits[..., 1]
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Count valid sampled margins and stratify by training frequency
            if all_logits.shape != all_items.shape:
                raise ValueError("Per-position candidate scoring must return B x L x K")
            valid = (pos_t != 0) & (seq_t != 0) & (neg_t != 0)
            if valid.any():
                values = margin[valid].detach().float().cpu()
                margins_all.append(values)
                ids = pos_t[valid].detach().cpu().tolist()
                for item, value in zip(ids, values.tolist()):
                    count = frequency[item]
                    label = "1-4" if count < 5 else "5-19" if count < 20 else "20+"
                    frequency_margins[label].append(value)
            ### <<< Self_EvolveRec-BLOCK-END

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Apply the same valid-position mask to all margin summaries
            valid_cpu = valid.detach().cpu().numpy()
            ### <<< Self_EvolveRec-BLOCK-END

            low_mask = valid_cpu & (margin_cpu < float(eps))
            wrong_mask = valid_cpu & (margin_cpu < 0.0)

            for b, l in np.argwhere(low_mask):
                low_total += 1
                pid = int(pos_cpu[b, l])
                nid = int(neg_cpu[b, l])
                for c in self._get_categories(pid):
                    low_pos_cat[c] += 1
                for c in self._get_categories(nid):
                    low_neg_cat[c] += 1

            for b, l in np.argwhere(wrong_mask):
                wrong_total += 1
                pid = int(pos_cpu[b, l])
                nid = int(neg_cpu[b, l])
                for c in self._get_categories(pid):
                    wrong_pos_cat[c] += 1
                for c in self._get_categories(nid):
                    wrong_neg_cat[c] += 1

        if len(margins_all) == 0:
            margin_metrics = {
                ### >>> Self_EvolveRec-BLOCK-START: Missing sampled margins are unavailable measurements
                "mean": None,
                "p10": None,
                "neg_beats_pos_rate": None,
                "small_margin_rate": None,
                ### <<< Self_EvolveRec-BLOCK-END
            }
        else:
            m = torch.cat(margins_all, dim=0)
            N = m.numel()

            k = max(1, int(np.ceil(0.10 * N)))
            p10 = float(torch.kthvalue(m, k).values.item())

            margin_metrics = {
                "mean": round(float(m.mean().item()), 4),
                "p10": round(p10, 4),
                "neg_beats_pos_rate": round(float((m < 0.0).float().mean().item()), 4),
                "small_margin_rate": round(
                    float((m < float(eps)).float().mean().item()), 4
                ),
            }

        def _topk(counter: Counter, k: int):
            return [{"category": c, "count": int(n)} for c, n in counter.most_common(k)]

        breakdown = {
            "eps": float(eps),
            "low_margin_total_positions": int(low_total),
            "wrong_rank_total_positions": int(wrong_total),
            "low_margin_top_categories": {
                "pos_item": _topk(low_pos_cat, topk_categories),
                "neg_item": _topk(low_neg_cat, topk_categories),
            },
            "wrong_rank_top_categories": {
                "pos_item": _topk(wrong_pos_cat, topk_categories),
                "neg_item": _topk(wrong_neg_cat, topk_categories),
            },
        }

        ### >>> Self_EvolveRec-BLOCK-START: Report sampled-margin denominators and frequency strata
        self._margin_count = sum(m.numel() for m in margins_all)
        self._margin_frequency = {
            label: {
                "positions": len(values),
                "mean": float(np.mean(values)) if values else None,
                "neg_beats_pos_rate": (
                    float(np.mean(np.asarray(values) < 0)) if values else None
                ),
            }
            for label, values in frequency_margins.items()
        }
        breakdown["valid_positions"] = self._margin_count
        breakdown["metadata_status"] = (
            "Item-category metadata unavailable; breakdown lists intentionally empty"
        )
        if not self._margin_count:
            breakdown["unavailable_reason"] = (
                "No valid sampled positive/negative positions"
            )
        return margin_metrics, breakdown
        ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Verify prediction parity and reproducible bounded temporal diagnostics
    def _correctness_checks(self, width):
        report = {
            "passed": False,
            "atol": 1e-5,
            "rtol": 1e-4,
            "max_valid_feature_error": 0.0,
            "max_final_feature_error": 0.0,
            "max_candidate_score_error": 0.0,
            "max_position_score_error": 0.0,
        }
        try:
            if width < 1:
                raise ValueError("Existing model input window is unavailable")
            itemnum = int(self.module.itemnum)
            if itemnum < 1:
                raise ValueError("Empty item vocabulary")
            history = [1 + i % itemnum for i in range(min(5, width))]
            histories = [history, [history[-1]], []]
            sequence = self._pad(histories, width)
            features = self._features(sequence)
            if features.shape[:2] != sequence.shape or features.ndim != 3:
                raise ValueError("log2feats must return B x L x H")
            report["dtype"] = str(features.dtype)
            states = self._last(features, sequence)
            candidates = torch.arange(1, min(3, itemnum) + 1, device=self.device)
            ids = candidates[None, :].expand(len(histories), -1)
            scores = self.model.predict_score(states, ids)
            if scores.shape != ids.shape:
                raise ValueError("Batched candidate scoring must return B x K")

            def compare(a, b, key):
                if (
                    a.shape != b.shape
                    or not torch.isfinite(a).all()
                    or not torch.isfinite(b).all()
                ):
                    raise ValueError(f"Nonfinite values or shape mismatch: {key}")
                error = float((a - b).abs().max().item()) if a.numel() else 0.0
                report[key] = max(report.get(key, 0.0), error)
                if not torch.allclose(a, b, atol=1e-5, rtol=1e-4):
                    raise ValueError(f"Execution consistency failed: {key}")

            for row, h in enumerate(histories):
                if h:
                    # Compare aligned valid features at every supported leading-pad width.
                    widths = sorted({len(h), min(width, len(h) + 1), width})
                    reference = self._features(self._pad([h], len(h)))
                    for test_width in widths:
                        test_seq = self._pad([h], test_width)
                        test_features = self._features(test_seq)
                        compare(
                            reference[0],
                            test_features[0, -len(h) :],
                            "max_valid_feature_error",
                        )
                        compare(
                            states[row],
                            self._last(test_features, test_seq)[0],
                            "max_final_feature_error",
                        )
                    compare(
                        reference[0],
                        features[row, -len(h) :],
                        "max_valid_feature_error",
                    )
                    individual = self.model.predict_score(reference[0, -1], candidates)
                else:
                    compare(
                        features[row],
                        torch.zeros_like(features[row]),
                        "max_valid_feature_error",
                    )
                    individual = self.model.predict_score(
                        torch.zeros_like(states[row]), candidates
                    )
                compare(scores[row], individual, "max_candidate_score_error")
            compare(
                features[sequence.eq(0)],
                torch.zeros_like(features[sequence.eq(0)]),
                "max_valid_feature_error",
            )
            # Explicit broadcasting of a shared list; verify every selected position.
            position_ids = candidates[None, None, :].expand(sequence.size(0), width, -1)
            position_scores = self.model.predict_score(features, position_ids)
            if position_scores.shape != position_ids.shape:
                raise ValueError("Per-position scoring must return B x L x K")
            for row in range(sequence.size(0)):
                for position in sorted({0, width // 2, width - 1}):
                    reference = self.model.predict_score(
                        features[row, position], candidates
                    )
                    compare(
                        position_scores[row, position],
                        reference,
                        "max_position_score_error",
                    )
            # Compare against the actual prediction entry point as well as the
            # ModelView adapter. This detects a wrong readout or scoring adapter.
            predictor = getattr(self.module, "score_candidates", None)
            if not callable(predictor):
                raise ValueError(
                    "GRU4Rec prediction entry point score_candidates is unavailable"
                )
            for row in range(sequence.size(0)):
                predicted = predictor(sequence[row : row + 1], candidates)
                if predicted.shape != candidates.shape:
                    raise ValueError(
                        "Prediction entry point must return one score per candidate"
                    )
                compare(scores[row], predicted, "max_prediction_score_error")
            # Also execute a standalone all-padding batch: its path can differ
            # from an empty row sharing a batch with nonempty histories.
            empty_sequence = self._pad([[]], width)
            empty_features = self._features(empty_sequence)
            compare(
                empty_features,
                torch.zeros_like(empty_features),
                "max_valid_feature_error",
            )
            empty_scores = self.model.predict_score(
                self._last(empty_features, empty_sequence), candidates[None, :]
            )
            compare(empty_scores[0], scores[-1], "max_candidate_score_error")
            report["passed"] = True
        except Exception as exc:
            report["reason"] = f"{type(exc).__name__}: {exc}"
        return report

    def _temporal_sensitivity(self, width):
        category_names = [
            "Movies_and_TV",
            "Industrial_and_Scientific",
            "Electronics",
            "CDs_and_Vinyl",
        ]
        category = getattr(self.model, "dataset_name", None)
        if category is None:
            category = getattr(self.module, "dataset_name", None)
        if category not in category_names:
            category = None
        result = {
            "population": "Selected training-prefix targets; conditional in-sample sensitivity",
            "dataset_category": category,
            "category_label_reason": (
                None
                if category
                else "Dataset routing label not exposed by this ModelView"
            ),
            "seed": 42,
            "user_cap": 200,
            "bootstrap_replicates": 2000,
            "eligible_users": 0,
            "sampled_users": 0,
            "retained_users": 0,
            "exclusions": {},
            "unique_keys": True,
            "user_outcomes": [],
            "category_results": {},
            "available_category_summary": None,
            "four_category_mean": None,
            "four_category_interval": None,
            "four_category_unavailable_reason": "This probe receives one category ModelView; all four independent category records are required",
        }
        if width < 5:
            result["unavailable_reason"] = (
                "Existing input window has fewer than five positions"
            )
            return result
        histories = self._histories()
        eligible = sorted(
            (u for u, h in histories.items() if len(h) >= 6), key=lambda u: str(u)
        )
        rng = np.random.default_rng(42)
        result["eligible_users"] = len(eligible)
        selected = rng.choice(
            len(eligible), size=min(200, len(eligible)), replace=False
        )
        users = [eligible[int(i)] for i in selected]
        result["sampled_users"] = len(users)
        itemnum = int(self.module.itemnum)
        excluded = Counter()
        keys = set()
        effects = []
        for user in users:
            original = histories[user]
            target_index = int(rng.integers(5, len(original)))
            target = original[target_index]
            window = original[:target_index][-width:]
            if len(window) < 5:
                excluded["short_retained_prefix"] += 1
                continue
            if not all(1 <= item <= itemnum for item in original):
                excluded["invalid_training_item_id"] += 1
                continue
            # The unchanged full training history defines exclusions, just as in
            # the supplied validation sampler. Perturbations never change them.
            forbidden = set(original) | {0, target}
            available = itemnum - sum(1 <= item <= itemnum for item in forbidden)
            if available < 99:
                excluded["fewer_than_99_unique_negatives"] += 1
                continue
            candidates = [target]
            used = set(forbidden)
            # Complement sampling avoids rejection blowups for dense histories.
            if available * 5 < itemnum:
                pool = np.asarray(
                    [i for i in range(1, itemnum + 1) if i not in forbidden]
                )
                candidates.extend(int(i) for i in rng.choice(pool, 99, replace=False))
            else:
                while len(candidates) < 100:
                    item = int(rng.integers(1, itemnum + 1))
                    if item not in used:
                        candidates.append(item)
                        used.add(item)
            n = len(window)
            k = max(1, (n - 1) // 4)
            variants = [
                ("unchanged", list(window)),
                ("oldest_delete", window[k:]),
                ("recent_delete", window[: n - 1 - k] + window[-1:]),
                ("final_item_remove", window[:-1]),
                ("last_item_only", window[-1:]),
            ]
            for _ in range(5):
                removed = set(int(i) for i in rng.choice(n - 1, k, replace=False))
                variants.append(
                    (
                        "random_delete",
                        [item for i, item in enumerate(window) if i not in removed],
                    )
                )
                shuffled = []
                for start in range(0, n - 1, 3):
                    block = window[start : min(start + 3, n - 1)]
                    shuffled.extend(int(i) for i in rng.permutation(block))
                variants.append(("block_shuffle", shuffled + window[-1:]))
            sequence = self._pad([h for _, h in variants], width)
            features = self._features(sequence)
            states = self._last(features, sequence)
            ids = torch.tensor(candidates, device=self.device, dtype=torch.long)
            # Fifteen examples, one hundred candidates: below both GPU caps.
            scores = self.model.predict_score(
                states, ids[None, :].expand(len(variants), -1)
            )
            if scores.shape != (len(variants), 100) or not torch.isfinite(scores).all():
                raise ValueError(
                    "Nonfinite temporal scores or invalid B x K score shape"
                )
            scores = scores.detach().float().cpu().numpy()
            ranks = []
            candidate_array = np.asarray(candidates)
            for row in scores:
                order = np.lexsort((candidate_array, -row))
                ranks.append(int(np.flatnonzero(order == 0)[0]) + 1)
            if not torch.isfinite(states).all():
                raise ValueError("Temporal features contain nonfinite values")
            state_array = states.detach().float().cpu().numpy().astype(np.float64)
            base = state_array[0]
            base_norm = float(np.linalg.norm(base))
            condition_values = {}
            for index, (condition, _) in enumerate(variants):
                h = state_array[index]
                norm = float(np.linalg.norm(h))
                cosine = (
                    None
                    if min(norm, base_norm) < 1e-8
                    else float(
                        1.0 - np.clip(np.dot(h, base) / (norm * base_norm), -1.0, 1.0)
                    )
                )
                values = {
                    "rank_change": float(ranks[index] - ranks[0]),
                    "hr10_change": float(int(ranks[index] <= 10) - int(ranks[0] <= 10)),
                    "cosine_distance": cosine,
                    "normalized_displacement": float(
                        np.linalg.norm(h - base) / max(base_norm, 1e-8)
                    ),
                }
                condition_values.setdefault(condition, []).append(values)
            averaged = {}
            for condition, repeats in condition_values.items():
                averaged[condition] = {}
                for metric in repeats[0]:
                    values = [
                        entry[metric] for entry in repeats if entry[metric] is not None
                    ]
                    averaged[condition][metric] = (
                        float(np.mean(values)) if values else None
                    )
                    averaged[condition][metric + "_valid_repeats"] = len(values)
            key = (category, str(user), target_index)
            if key in keys:
                raise ValueError("Duplicate category/user/target-index diagnostic key")
            keys.add(key)
            length_slice = "5-9" if n < 10 else "10-19" if n < 20 else "20+"
            effects.append(
                {
                    "user": str(user),
                    "target_index": target_index,
                    "history_length": n,
                    "length_slice": length_slice,
                    "target_status": (
                        "repeat" if target in original[:target_index] else "novel"
                    ),
                    "intact_rank": ranks[0],
                    "intact_hr10": int(ranks[0] <= 10),
                    "effects": averaged,
                }
            )
        result["exclusions"] = dict(excluded)
        result["retained_users"] = len(effects)
        result["user_outcomes"] = effects
        summary = self._summarize_users(effects)
        result["available_category_summary"] = summary
        if category:
            result["category_results"][category] = summary
        if not effects:
            result["unavailable_reason"] = (
                "No retained eligible training-prefix examples"
            )
        return result

    @staticmethod
    def _summarize_users(records):
        bootstrap_rng = np.random.default_rng(42)

        def summarize(rows):
            report = {
                "users": len(rows),
                "sparse": len(rows) < 30,
                "intact_hr10": (
                    float(np.mean([r["intact_hr10"] for r in rows])) if rows else None
                ),
                "conditions": {},
            }
            if not rows:
                report["unavailable_reason"] = "No users in this population"
                return report
            # The same resampled users are used for all paired outcome columns.
            draws = (
                bootstrap_rng.integers(0, len(rows), size=(2000, len(rows)))
                if len(rows) >= 30
                else None
            )
            for condition in rows[0]["effects"]:
                report["conditions"][condition] = {}
                for metric in (
                    "rank_change",
                    "hr10_change",
                    "cosine_distance",
                    "normalized_displacement",
                ):
                    values = np.asarray(
                        [
                            (
                                np.nan
                                if r["effects"][condition][metric] is None
                                else r["effects"][condition][metric]
                            )
                            for r in rows
                        ],
                        dtype=np.float64,
                    )
                    finite = np.isfinite(values)
                    count = int(finite.sum())
                    entry = {
                        "users": count,
                        "unavailable_users": len(rows) - count,
                        "mean": float(values[finite].mean()) if count else None,
                        "constant": (
                            bool(np.ptp(values[finite]) == 0) if count else None
                        ),
                        "interval": None,
                    }
                    if draws is not None and count == len(rows):
                        means = values[draws].mean(1)
                        entry["interval"] = [
                            float(v) for v in np.percentile(means, [2.5, 97.5])
                        ]
                    else:
                        entry["interval_reason"] = (
                            "Requires at least 30 unique users and complete outcomes in this population"
                        )
                    report["conditions"][condition][metric] = entry
            return report

        result = summarize(records)
        result["slices"] = {}
        for field, labels in (
            ("length_slice", ["5-9", "10-19", "20+"]),
            ("target_status", ["repeat", "novel"]),
        ):
            result["slices"][field] = {
                label: summarize([r for r in records if r[field] == label])
                for label in labels
            }
        return result

    def run_full_diagnosis(self, train_loader):
        previous_mode = self.module.training
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        cpu_state = torch.random.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        metrics = {}
        try:
            # Reset per-run state so a failed or empty loader cannot expose
            # counts or a window retained from a previous diagnostic call.
            self._observed_maxlen = None
            self._margin_count = 0
            self._margin_frequency = {}
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(42)
            self.module.eval()
            with torch.inference_mode():
                collapse_score = self.analyze_embeddings_collapse_cosine(
                    sample_size=2048
                )
                eps = 0.5
                try:
                    margin_metrics, breakdown = (
                        self.analyze_pos_neg_margin_with_category_breakdown(
                            train_loader, num_batches=10, eps=eps, topk_categories=20
                        )
                    )
                except Exception as exc:
                    margin_metrics = dict.fromkeys(
                        ["mean", "p10", "neg_beats_pos_rate", "small_margin_rate"]
                    )
                    self._margin_count = 0
                    self._margin_frequency = {}
                    reason = f"{type(exc).__name__}: {exc}"
                    print(
                        f"Diagnosis sampled-margin probe unavailable: {reason}",
                        flush=True,
                    )
                    breakdown = {
                        "unavailable_reason": reason,
                        "valid_positions": 0,
                        "low_margin_top_categories": {"pos_item": [], "neg_item": []},
                        "wrong_rank_top_categories": {"pos_item": [], "neg_item": []},
                    }
                metrics.update(
                    {
                        "embedding_collapse_score": collapse_score,
                        "embedding_sample_count": min(2048, int(self.module.itemnum)),
                        "pos_neg_margin_mean": margin_metrics["mean"],
                        "pos_neg_margin_p10": margin_metrics["p10"],
                        "pos_neg_neg_beats_pos_rate": margin_metrics[
                            "neg_beats_pos_rate"
                        ],
                        "pos_neg_small_margin_rate": margin_metrics[
                            "small_margin_rate"
                        ],
                        "pos_neg_margin_category_breakdown": breakdown,
                        "pos_neg_margin_frequency_breakdown": self._margin_frequency,
                    }
                )
                width = self._observed_maxlen
                if width is None:
                    width = getattr(self.model, "maxlen", None)
                if width is None:
                    width = getattr(getattr(self.model, "args", None), "maxlen", 0)
                width = int(width or 0)
                checks = self._correctness_checks(width)
                metrics["execution_checks"] = checks
                if checks["passed"]:
                    try:
                        metrics["temporal_sensitivity"] = self._temporal_sensitivity(
                            width
                        )
                    except Exception as exc:
                        metrics["temporal_sensitivity"] = None
                        metrics["temporal_unavailable_reason"] = (
                            f"{type(exc).__name__}: {exc}"
                        )
                else:
                    metrics["temporal_sensitivity"] = None
                    metrics["temporal_unavailable_reason"] = (
                        "Execution checks failed; original checkpoint semantics were not modified"
                    )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            metrics["diagnostic_inference_seconds"] = float(
                time.perf_counter() - started
            )
        finally:
            self.module.train(previous_mode)
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
        return {
            "metrics": metrics,
            "metric_definitions": {
                "embedding_collapse_score": "Sample up to 2048 real-item embeddings, seed 42, excluding padding/reserved tokens. Map mean off-diagonal pairwise cosine to [0,1]; denominator k*(k-1). Higher indicates directional alignment, not measured recommendation diversity. Existing convention returns zero for at most two items.",
                "embedding_sample_count": "Number of real-item embeddings sampled for the collapse probe; pairwise denominator is k*(k-1).",
                "pos_neg_margin_mean": "Mean positive-minus-negative model score across valid training-loader positions in at most ten batches; positive, negative and sequence IDs must be nonzero. Larger means stronger sampled separation; denominator is valid_positions in the breakdown.",
                "pos_neg_margin_p10": "Nearest-rank lower tenth percentile of the same valid training-position margins, index ceil(0.1*N); lower indicates weaker sampled separation. Null if N=0.",
                "pos_neg_neg_beats_pos_rate": "Fraction of the same N valid training positions whose sampled negative score exceeds positive score; lower is better. Null if N=0.",
                "pos_neg_small_margin_rate": "Fraction of the same N valid training positions with positive-minus-negative score below eps=0.5; lower indicates fewer small margins. Null if N=0.",
                "pos_neg_margin_category_breakdown": "Counts low-margin and wrong-rank valid training positions and their denominator. Item-category metadata is unavailable, so category lists remain empty; these lists are unrelated to dataset routing.",
                "pos_neg_margin_frequency_breakdown": "Stratify the same sampled training positions by positive-item training-interaction frequency: 1-4, 5-19, 20+. Each stratum reports its position denominator, mean margin and negative-beats-positive fraction. Frequency is not exposure.",
                "execution_checks": "At atol=1e-5, rtol=1e-4, compare individual unpadded and batched left-padded aligned valid features, maximum-nonzero-position readouts and fixed candidate scores, including singleton/empty rows, a standalone all-padding batch and added leading zeros within the existing window. Compare ModelView scores with the actual trained module's score_candidates prediction entry point for every checked row. Empty sequence features must be zero; trained score biases remain included. Check BxLxK scoring against loop references. Report dtype, maximum discrepancies and failures; no checkpoint repair occurs here.",
                "temporal_sensitivity": {
                    "population": "At most 200 uniformly sampled eligible training users without replacement in sorted ID order, RNG 42, per actual dataset ModelView. One uniformly selected training target index with at least five preceding interactions; truncate its prefix once to the existing model window. Require five retained items. Report eligible, sampled, retained, exclusion and unique-key counts. These targets may have been used in training.",
                    "candidates": "Target plus 99 unique negatives drawn independently of official evaluation. Exclude the original full training history and target before perturbation, as in the supplied validation sampler; preserve prior target occurrences in inputs. Freeze candidate IDs for all conditions. Exclude insufficient-candidate users. Target rank is one-based, sorting by descending model score then ascending item ID.",
                    "interventions": "For retained length n, k=max(1,floor((n-1)/4)). Unchanged, oldest-k deletion, most-recent-k deletion among the first n-1 positions, five uniform random-k deletions among those positions, five independent within-block permutations of the first n-1 positions with block size three, final-item removal, and last-item-only. Preserve survivor chronology and the final item except in explicit removal. Never refill; left-pad every variant to the same width. Average repeated scalar effects within each user before aggregation. Local shuffling can create implausible chronology.",
                    "rank_change": "Perturbed rank minus intact rank per retained user; positive means ranking damage. Means divide by reported contributing user counts.",
                    "hr10_change": "Perturbed indicator(rank<=10) minus intact indicator(rank<=10) per retained user; negative means ranking damage. Means divide by reported contributing user counts.",
                    "cosine_distance": "One minus cosine of perturbed and intact final states; larger means state movement. Null when either norm is below 1e-8. Report valid-repeat and user denominators. Representation movement alone does not establish utility damage.",
                    "normalized_displacement": "L2 state difference divided by max(intact-state L2 norm,1e-8), averaged over repeats then users; larger means relative state movement.",
                    "summaries": "Descriptive category means, intact diagnostic HR@10 and history-length 5-9/10-19/20+ and repeat/novel slices. Repeat means target occurred before its training index, even outside the retained window. Denominators and sparse/constant flags accompany each result. user_outcomes caches paired scalar outcomes for host aggregation; it contains no validation or test labels.",
                    "uncertainty": "2000 CPU bootstrap replicates, separate RNG 42, resampling users with replacement within this category and preserving paired conditions. Percentile 2.5th-97.5th intervals require at least 30 unique users with complete outcomes. Four-category point estimates require all four category records and equal category weights; four-category intervals additionally require at least 30 contributing users in every category, including within each slice. This single-view API cannot manufacture missing categories, so macro fields are null with reasons. An available-category summary is explicitly descriptive and separate.",
                    "limits": "Conditional in-sample sensitivity given fitted checkpoints, selected prefixes and candidates; intervals omit seed variability, new-category generalization and validation-search selection optimism. Simulator users and their required full-catalog recommendations remain separate. No inference about intent, exposure, ratings, reviews or simulator traits is supported. Official macro validation HR@10 alone selects models.",
                },
                "temporal_unavailable_reason": "Reason the entire temporal probe is null; failed execution checks or missing measurements must not be interpreted as zero effects.",
                "diagnostic_inference_seconds": "Elapsed wall seconds for diagnostic probes, including CPU summaries and synchronized GPU work; excludes model training. Diagnostics consume the shared search budget.",
            },
        }

    ### <<< Self_EvolveRec-BLOCK-END
