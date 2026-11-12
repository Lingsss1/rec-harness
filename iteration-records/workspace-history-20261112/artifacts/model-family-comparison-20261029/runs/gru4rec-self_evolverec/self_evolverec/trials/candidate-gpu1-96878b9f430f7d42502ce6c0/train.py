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
### >>> Self_EvolveRec-BLOCK-START: Preserve the four category routes and required training seed
DATASETS = list(DEFAULT_DATASETS)
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
            ### >>> Self_EvolveRec-BLOCK-START: Bound negative rejection sampling and handle exhausted catalogs
            if nxt != 0 and len(interacted) < itemnum:
                neg_item = random.randint(1, itemnum)
                for _attempt in range(32):
                    if neg_item not in interacted:
                        break
                    neg_item = random.randint(1, itemnum)
                if neg_item in interacted:
                    available = [
                        i for i in range(1, itemnum + 1) if i not in interacted
                    ]
                    neg_item = random.choice(available)
                neg[idx] = neg_item
            ### <<< Self_EvolveRec-BLOCK-END
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
    ### >>> Self_EvolveRec-BLOCK-START: Use reproducible validation sampling independent of training RNG
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
            ### >>> Self_EvolveRec-BLOCK-START: Freeze validation candidates and bound their count by catalog availability
            candidates = [target]
            candidate_count = 1 + min(
                99, itemnum - sum(1 <= item <= itemnum for item in interacted)
            )
            while len(candidates) < candidate_count:
                neg_item = rng.randint(1, itemnum)
                ### <<< Self_EvolveRec-BLOCK-END
                if neg_item not in interacted:
                    interacted.add(neg_item)
                    candidates.append(neg_item)
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
    ### >>> Self_EvolveRec-BLOCK-START: Preserve BPR with stable log-sigmoid and valid negative masking
    pos_scores, neg_scores = model(seq, pos, neg)
    mask = seq.ne(0) & pos.ne(0) & neg.ne(0)
    loss = -torch.nn.functional.logsigmoid(pos_scores.float() - neg_scores.float())
    return loss.masked_fill(~mask, 0.0).sum() / mask.sum().clamp(min=1)
    ### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Require mapped public text and preserve embedding semantics during bounded encoding
def build_item_text(user_train, itemnum, data_dir, dataset_name):
    import re
    import unicodedata
    from collections import Counter

    counts = torch.zeros(itemnum + 1, dtype=torch.long)
    for history in user_train.values():
        for item in history:
            if 1 <= item <= itemnum:
                counts[item] += 1

    # Only read public item documents, never interaction or held-out files.
    paths = [
        os.path.join(data_dir, f"{dataset_name}_metadata.json"),
        os.path.join(data_dir, f"{dataset_name}_meta.json"),
        os.path.join(data_dir, f"{dataset_name}_metadata.jsonl"),
    ]
    explicit = os.environ.get("GAGC_ITEM_TEXT_FILE")
    if explicit:
        path = os.path.realpath(explicit.format(dataset=dataset_name))
        root = os.path.realpath(data_dir)
        if os.path.commonpath([root, path]) != root:
            raise ValueError("GAGC_ITEM_TEXT_FILE must be beneath GAGC_DATA_DIR")
        paths = [path]
    path = next((p for p in paths if os.path.isfile(p)), None)
    if path is None:
        raise FileNotFoundError(
            f"[{dataset_name}] Text-fusion training requires public title/description "
            f"metadata using the supplied integer item mapping. Checked: {paths}. "
            "Set GAGC_ITEM_TEXT_FILE to the mapped public metadata path beneath "
            "GAGC_DATA_DIR; {dataset} may be used in the path."
        )

    def text_value(value):
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return " ".join(text_value(v) for v in value)
        return ""

    pattern = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)

    def tokenize(value):
        return pattern.findall(unicodedata.normalize("NFKC", text_value(value)).lower())

    documents = {}

    def accept(key, record):
        if not isinstance(record, dict):
            return
        # IDs must already use the supplied split's integer item mapping.
        try:
            item = int(key)
        except (TypeError, ValueError):
            return
        if not 1 <= item <= itemnum:
            return
        documents[item] = (
            tokenize(record.get("title", "")),
            tokenize(record.get("description", "")),
        )

    with open(path, encoding="utf-8") as handle:
        if path.endswith(".jsonl"):
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if isinstance(record, dict):
                        accept(record.get("item_id"), record)
        else:
            payload = json.load(handle)
            if isinstance(payload, dict):
                for key, record in payload.items():
                    accept(key, record)
            elif isinstance(payload, list):
                for record in payload:
                    if isinstance(record, dict):
                        accept(record.get("item_id"), record)
            else:
                raise ValueError("Public item text must be an item-ID mapping or list")

    frequencies = Counter()
    for item, (title, description) in documents.items():
        if counts[item] > 0:
            frequencies.update(title)
            frequencies.update(description)
    words = sorted(
        (word for word, frequency in frequencies.items() if frequency >= 2),
        key=lambda word: (-frequencies[word], word),
    )[:29998]
    vocabulary = {word: index + 2 for index, word in enumerate(words)}
    tokens = torch.zeros(itemnum + 1, 128, dtype=torch.long)
    for item, (title, description) in documents.items():
        ids = [vocabulary.get(word, 1) for word in title[:32] + description[:96]]
        if ids:
            tokens[item, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    supported = counts.gt(0)
    if not bool(tokens[supported].ne(0).any()):
        raise ValueError(
            "Public text file has no nonempty documents matching training item IDs; "
            "provide metadata using the supplied integer item mapping"
        )
    log_counts = counts.float().log1p()
    mean = log_counts[supported].mean()
    std = log_counts[supported].std(unbiased=False).clamp_min(1e-8)
    config = {
        "vocab_size": len(vocabulary) + 2,
        "token_dim": 64,
        "token_width": 128,
    }
    return {
        "config": config,
        "vocabulary": vocabulary,
        "tokens": tokens,
        "counts": counts,
        "count_mean": mean,
        "count_std": std,
        "standardized_log_count": (log_counts - mean) / std,
        "tokenizer": {
            "normalization": "NFKC",
            "lowercase": True,
            "pattern": pattern.pattern,
            "engine": "Python re",
            "unicode_version": unicodedata.unidata_version,
            "minimum_frequency": 2,
            "title_limit": 32,
            "description_limit": 96,
            "reserved_tokens": {"PAD": 0, "UNK": 1},
        },
    }


class FusedItemEmbedding(nn.Module):
    def __init__(self, id_embedding, config):
        super().__init__()
        self.id_embedding = id_embedding
        self.num_embeddings = id_embedding.num_embeddings
        self.embedding_dim = id_embedding.embedding_dim
        self.padding_idx = id_embedding.padding_idx
        self.scale_grad_by_freq = id_embedding.scale_grad_by_freq
        self.max_norm = id_embedding.max_norm
        if self.padding_idx != 0:
            raise ValueError("Text fusion requires the supplied padding index 0")
        if self.scale_grad_by_freq or self.max_norm is not None:
            raise ValueError(
                "Deduplicated text fusion requires scale_grad_by_freq=False "
                "and max_norm=None to preserve occurrence-level gradients"
            )
        self.token_embedding = nn.Embedding(
            config["vocab_size"], config["token_dim"], padding_idx=0
        )
        # Match the original ID embedding's Xavier-uniform bounds.
        bound = math.sqrt(6.0 / (self.num_embeddings + self.embedding_dim))
        nn.init.uniform_(self.token_embedding.weight, -bound, bound)
        with torch.no_grad():
            self.token_embedding.weight[0].zero_()
        self.text_projection = nn.Linear(
            config["token_dim"], self.embedding_dim, bias=False
        )
        self.gate_slope = nn.Parameter(torch.zeros(()))
        self.gate_bias = nn.Parameter(torch.full((), -3.0))
        self.register_buffer(
            "item_tokens",
            torch.zeros(self.num_embeddings, config["token_width"], dtype=torch.long),
        )
        self.register_buffer(
            "training_counts", torch.zeros(self.num_embeddings, dtype=torch.long)
        )
        self.register_buffer("standardized_log_count", torch.zeros(self.num_embeddings))
        self.register_buffer("count_mean", torch.zeros(()))
        self.register_buffer("count_std", torch.ones(()))

    def _encode_unique(self, items):
        ids = self.id_embedding(items)
        tokens = self.item_tokens[items]
        lengths = tokens.ne(0).sum(dim=-1, keepdim=True)
        width = self.item_tokens.size(1)
        offsets = torch.arange(items.numel(), device=items.device) * width
        # EmbeddingBag avoids an occurrences x tokens x dimensions allocation.
        pooled = torch.nn.functional.embedding_bag(
            tokens.reshape(-1),
            self.token_embedding.weight,
            offsets,
            mode="sum",
            padding_idx=0,
            scale_grad_by_freq=False,
        ) / lengths.clamp_min(1)
        text = self.text_projection(pooled)
        gate = torch.sigmoid(
            self.gate_slope * self.standardized_log_count[items] + self.gate_bias
        ).unsqueeze(-1)
        residual = gate * text * lengths.gt(0)
        return (ids + residual).masked_fill(items.eq(0).unsqueeze(-1), 0.0)

    def forward(self, items):
        if items.numel() == 0:
            return self.id_embedding.weight.new_empty(*items.shape, self.embedding_dim)
        unique, inverse = torch.unique(items.reshape(-1), return_inverse=True)
        encoded = torch.cat(
            [self._encode_unique(chunk) for chunk in unique.split(512)], dim=0
        )
        return encoded[inverse].reshape(*items.shape, self.embedding_dim)

    def _apply(self, fn, recurse=True):
        # No learned-representation cache is retained across device/dtype changes.
        return super()._apply(fn, recurse=recurse)

    @property
    def weight(self):
        # ModelView diagnostics see the actual current representation. Training
        # uses sampled forward lookups and never constructs this complete table.
        items = torch.arange(
            self.num_embeddings, device=self.id_embedding.weight.device
        )
        return self(items)


### <<< Self_EvolveRec-BLOCK-END


# Hyper-parameters
MAXLEN = int(os.environ.get("GRU4REC_MAXLEN", "128"))
HIDDEN_UNITS = int(os.environ.get("GRU4REC_HIDDEN", "128"))
NUM_LAYERS = int(os.environ.get("GRU4REC_LAYERS", "1"))
DROPOUT_RATE = float(os.environ.get("GRU4REC_DROPOUT", "0.1"))
LR = float(os.environ.get("GRU4REC_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("GRU4REC_WD", "0.0"))
BATCH_SIZE = int(os.environ.get("GRU4REC_BATCH", "128"))
### >>> Self_EvolveRec-BLOCK-START: Honor the host epoch setting within the fixed training limits
NUM_BATCHES = 200
NUM_EPOCHS = min(
    200,
    max(
        1, int(os.environ.get("SASREC_EPOCHS", os.environ.get("GRU4REC_EPOCHS", "200")))
    ),
)
### <<< Self_EvolveRec-BLOCK-END
EVAL_EVERY = int(os.environ.get("GRU4REC_EVAL_EVERY", "5"))
PATIENCE = int(os.environ.get("GRU4REC_PATIENCE", "20"))


### >>> Self_EvolveRec-BLOCK-START: Preserve shared text-fused GRU scoring with bounded candidate memory and allocation-only empty paths
class GRU4RecModel(nn.Module):
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 128,
        num_layers: int = 1,
        dropout_rate: float = 0.1,
        text_config: dict | None = None,
    ):
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
        self.dropout = nn.Dropout(dropout_rate)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
        # Construct original modules first to preserve their initialization RNG.
        if text_config is not None:
            self.item_emb = FusedItemEmbedding(self.item_emb, text_config)

    def log2feats(self, seq: torch.Tensor) -> torch.Tensor:
        if seq.ndim != 2:
            raise ValueError("sequence must have shape B x L")
        batch_size, width = seq.shape
        if width == 0 or batch_size == 0:
            # FusedItemEmbedding.weight encodes the entire catalog; use an
            # existing parameter only as a device/dtype allocation reference.
            return self.gru.weight_ih_l0.new_zeros(batch_size, width, self.hidden_units)
        valid = seq.ne(0)
        lengths = valid.sum(dim=1)
        positions = torch.arange(width, device=seq.device).expand_as(seq)
        # Compact observed items in their original order. Packed recurrence never
        # processes leading padding, so GRU biases cannot alter the initial state.
        order = (positions + (~valid).long() * width).argsort(dim=1)
        embeddings = self.dropout(self.item_emb(seq))
        compact = embeddings.gather(
            1, order.unsqueeze(-1).expand(-1, -1, self.hidden_units)
        )
        packed = nn.utils.rnn.pack_padded_sequence(
            compact, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        compact_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=width
        )
        out = torch.zeros_like(compact_out).scatter(
            1, order.unsqueeze(-1).expand_as(compact_out), compact_out
        )
        # The synthetic one-step input for an empty history must emit no feature.
        return out.masked_fill(~valid.unsqueeze(-1), 0.0)

    def predict_score(
        self, features: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.item_emb(items)
        if items.ndim == features.ndim - 1:
            return (features * embeddings).sum(dim=-1)
        return (features.unsqueeze(-2) * embeddings).sum(dim=-1)

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        if seq.size(1) == 0:
            return self.gru.weight_ih_l0.new_zeros(seq.size(0), self.hidden_units)
        out = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).max(dim=1).values
        features = out[torch.arange(seq.size(0), device=seq.device), last.clamp(min=0)]
        return features.masked_fill(last.lt(0).unsqueeze(-1), 0.0)

    def forward(self, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        features = self.log2feats(seq)
        return self.predict_score(features, pos), self.predict_score(features, neg)

    def score_candidates(
        self, seq: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        if seq.ndim != 2 or seq.size(0) != 1 or candidates.ndim != 1:
            raise ValueError(
                "score_candidates expects one B=1 history and a flat candidate list; "
                "use predict_score for aligned batched diagnostic scores"
            )
        feature = self.encode(seq)[0]
        if candidates.numel() == 0:
            return feature.new_empty(0)
        # Encode the history once and bound full-catalog candidate allocations.
        # Candidate order and repeated occurrences remain unchanged.
        return torch.cat(
            [self.predict_score(feature, chunk) for chunk in candidates.split(4096)],
            dim=0,
        )


### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Seed each independent category model reproducibly
def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str) -> float:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    ### <<< Self_EvolveRec-BLOCK-END
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f"empty dataset or item vocabulary for {dataset_name}")

    ### >>> Self_EvolveRec-BLOCK-START: Initialize and checkpoint one jointly trained text-fused GRU per category
    text_data = build_item_text(user_train, itemnum, data_dir, dataset_name)
    text_config = None if text_data is None else text_data["config"]
    model = GRU4RecModel(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT_RATE,
        text_config=text_config,
    )
    if text_data is not None:
        with torch.no_grad():
            model.item_emb.item_tokens.copy_(text_data["tokens"])
            model.item_emb.training_counts.copy_(text_data["counts"])
            for name in ("standardized_log_count", "count_mean", "count_std"):
                getattr(model.item_emb, name).copy_(text_data[name])
        available = text_data["tokens"].ne(0).any(dim=1)
        training_items = text_data["counts"].gt(0)
        all_unknown = available & (
            text_data["tokens"].eq(0) | text_data["tokens"].eq(1)
        ).all(dim=1)
        print(
            f"  [{dataset_name}] text_config={json.dumps(text_config)} "
            f"training_item_text_support={int((available & training_items).sum())}/"
            f"{int(training_items.sum())} "
            f"training_all_UNK_documents={int((all_unknown & training_items).sum())}",
            flush=True,
        )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ### <<< Self_EvolveRec-BLOCK-END

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
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= PATIENCE:
                print(f"  [{dataset_name}] Early stopping at epoch {epoch}", flush=True)
                break
        elif epoch % 10 == 0:
            print(
                f"  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}  loss={avg_loss:.4f}",
                flush=True,
            )

    if not os.path.exists(ckpt_path):
        torch.save(model.state_dict(), ckpt_path)

    meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "itemnum": itemnum,
                "maxlen": MAXLEN,
                "hidden_units": HIDDEN_UNITS,
                "num_layers": NUM_LAYERS,
                ### >>> Self_EvolveRec-BLOCK-START: Persist encoder configuration and vocabulary with the category artifacts
                "dropout_rate": DROPOUT_RATE,
                "text_config": text_config,
                "text_vocabulary": (
                    None if text_data is None else text_data["vocabulary"]
                ),
                "text_tokenizer": None if text_data is None else text_data["tokenizer"],
                "training_config": {
                    "seed": SEED,
                    "epochs": NUM_EPOCHS,
                    "batches_per_epoch": NUM_BATCHES,
                    "batch_size": BATCH_SIZE,
                    "optimizer": "Adam",
                    "lr": LR,
                    "weight_decay": WEIGHT_DECAY,
                    "objective": "BPR",
                    "gradient_clip_norm": 5.0,
                    "eval_every": EVAL_EVERY,
                    "patience": PATIENCE,
                    "torch_version": str(torch.__version__),
                    "numpy_version": np.__version__,
                    "text_enabled": text_config is not None,
                    "token_initialization": "uniform using original ID Xavier bounds",
                    "projection_initialization": "PyTorch bias-free Linear default",
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


### >>> Self_EvolveRec-BLOCK-START: Load the embedded trained GRU4Rec architecture
# DEBUG: Replace the unexpectedly indented placeholder with the missing loader.
# DEBUG: The predictor writer embeds GRU4RecModel before this function is called.
def _load_model_class():
    return GRU4RecModel
### <<< Self_EvolveRec-BLOCK-END


def _ensure_loaded():
    global _device
    ### >>> Self_EvolveRec-BLOCK-START: Require every category to finish loading
    if all(ds in _models for ds in DEFAULT_DATASETS):
        return
    ### <<< Self_EvolveRec-BLOCK-END
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ModelClass = _load_model_class()
    ### >>> Self_EvolveRec-BLOCK-START: Preserve all four inference routes
    for ds in DEFAULT_DATASETS:
        working_dir = _HERE if os.path.basename(_HERE) == "working" else os.path.join(_HERE, "working")
    ### <<< Self_EvolveRec-BLOCK-END
        meta_path = os.path.join(working_dir, f"{ds}_meta.pkl")
        ckpt_path = os.path.join(working_dir, f"{ds}_model.pt")
        ### >>> Self_EvolveRec-BLOCK-START: Reject missing category artifacts instead of silently misrouting
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing GRU4Rec artifacts for {ds}")
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        model = ModelClass(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_layers=meta["num_layers"],
            ### >>> Self_EvolveRec-BLOCK-START: Reconstruct the saved GRU and jointly trained text encoder
            dropout_rate=meta["dropout_rate"],
            text_config=meta.get("text_config"),
            ### <<< Self_EvolveRec-BLOCK-END
        ).to(_device)
        model.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ### >>> Self_EvolveRec-BLOCK-START: Enforce explicit category routing when a category is supplied
    if dataset_name is not None:
        if dataset_name not in _models:
            raise ValueError(f"Unknown dataset_name: {dataset_name!r}")
        ds = dataset_name
    elif _models:
        ds = next(iter(_models))
    else:
        raise RuntimeError("No GRU4Rec category models were loaded")
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
    ### >>> Self_EvolveRec-BLOCK-START: Embed both jointly trained representation and GRU definitions in the predictor
    import inspect

    model_source = (
        "import math\n\n"
        + inspect.getsource(FusedItemEmbedding)
        + "\n\n"
        + inspect.getsource(GRU4RecModel)
    )
    import_anchor = "import torch\n"
    if script.count(import_anchor) != 1:
        raise RuntimeError("Predictor template must contain exactly one torch import")
    script = script.replace(
        import_anchor,
        import_anchor + "import torch.nn as nn\n\n" + model_source + "\n",
        1,
    )
    predict_path = os.path.join(working_dir, "predict.py")
    compiled = compile(script, predict_path, "exec")
    # Check the actual embedded definitions against the selected checkpoints.
    # Replicas are temporary verification objects, never fitted or deployed.
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        namespace = {"__file__": predict_path, "__name__": "_predictor_audit"}
        exec(compiled, namespace)
        with torch.inference_mode():
            for dataset_name in DEFAULT_DATASETS:
                with open(
                    os.path.join(working_dir, f"{dataset_name}_meta.pkl"), "rb"
                ) as handle:
                    meta = pickle.load(handle)
                kwargs = dict(
                    itemnum=meta["itemnum"],
                    hidden_units=meta["hidden_units"],
                    num_layers=meta["num_layers"],
                    dropout_rate=meta["dropout_rate"],
                    text_config=meta.get("text_config"),
                )
                state = torch.load(
                    os.path.join(working_dir, f"{dataset_name}_model.pt"),
                    map_location="cpu",
                    weights_only=True,
                )
                reference = GRU4RecModel(**kwargs).eval()
                embedded = namespace["GRU4RecModel"](**kwargs).eval()
                reference.load_state_dict(state, strict=True)
                embedded.load_state_dict(state, strict=True)
                candidates = torch.arange(1, min(meta["itemnum"], 16) + 1)
                for width in (3, 8):
                    for history in ([], [1], [1, min(2, meta["itemnum"]), 1]):
                        sequence = torch.zeros(1, width, dtype=torch.long)
                        if history:
                            sequence[0, -len(history) :] = torch.tensor(history)
                        torch.testing.assert_close(
                            embedded.log2feats(sequence),
                            reference.log2feats(sequence),
                            atol=1e-5,
                            rtol=1e-4,
                        )
                        direct = reference.predict_score(
                            reference.encode(sequence)[0], candidates
                        )
                        actual = embedded.score_candidates(sequence, candidates)
                        torch.testing.assert_close(actual, direct, atol=1e-5, rtol=1e-4)
                del reference, embedded, state
                print(
                    f"  [{dataset_name}] embedded checkpoint feature/score parity passed",
                    flush=True,
                )
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
    ### <<< Self_EvolveRec-BLOCK-END
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


