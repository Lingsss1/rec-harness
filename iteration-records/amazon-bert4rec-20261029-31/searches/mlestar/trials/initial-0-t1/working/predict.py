from __future__ import annotations
import os
import pickle
import torch
import torch.nn as nn

class BERT4RecModel(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2,
                 num_heads=2, dropout_rate=0.2, maxlen=128):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.mask_token = itemnum + 1
        self.item_emb = nn.Embedding(itemnum + 2, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_units, nhead=num_heads,
            dim_feedforward=hidden_units * 4, dropout=dropout_rate,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_blocks)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

    def encode_sequence(self, seq):
        device = self.item_emb.weight.device
        seq = torch.as_tensor(seq, dtype=torch.long, device=device)
        batch, length = seq.shape
        positions = torch.arange(1, length + 1, device=device).unsqueeze(0).expand(batch, -1)
        positions = positions * seq.ne(0).long()
        x = self.item_emb(seq) * (self.hidden_units ** 0.5) + self.pos_emb(positions)
        x = self.dropout(x)
        padding = seq.eq(0)
        safe_padding = padding.clone()
        safe_padding[padding.all(dim=1), -1] = False
        x = self.encoder(x, src_key_padding_mask=safe_padding)
        return self.norm(x)

    def log2feats(self, sequence):
        return self.encode_sequence(sequence)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=self.item_emb.weight.device)
        embeddings = self.item_emb(items)
        if features.ndim == 3 and embeddings.ndim == 2:
            return torch.matmul(features, embeddings.transpose(0, 1))
        if features.ndim == 2 and embeddings.ndim == 2:
            return torch.matmul(features, embeddings.transpose(0, 1))
        if embeddings.ndim == features.ndim + 1:
            return (features.unsqueeze(-2) * embeddings).sum(dim=-1)
        return (features * embeddings).sum(dim=-1)

    def forward(self, seq, pos, neg):
        features = self.encode_sequence(seq)
        return ((features * self.item_emb(pos)).sum(dim=-1),
                (features * self.item_emb(neg)).sum(dim=-1))

    def score_candidates(self, seq, candidates):
        seq = torch.as_tensor(seq, dtype=torch.long, device=self.item_emb.weight.device)
        prediction_seq = torch.zeros_like(seq)
        prediction_seq[:, :-1] = seq[:, 1:]
        prediction_seq[:, -1] = self.mask_token
        # All histories are left padded; appended MASK is the final nonpadding token.
        user_embedding = self.encode_sequence(prediction_seq)[0, -1]
        return self.predict_score(user_embedding, candidates)


_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["Movies_and_TV", "Industrial_and_Scientific", "Electronics", "CDs_and_Vinyl"]
_models = {}
_metas = {}
_device = None


def _ensure_loaded(dataset_name):
    global _device
    if dataset_name in _models:
        return
    if dataset_name not in DATASETS:
        raise ValueError(f"unknown dataset_name: {dataset_name}")
    if _device is None:
        _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(_HERE, f"{dataset_name}_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    model = BERT4RecModel(**meta).to(_device)
    state = torch.load(os.path.join(_HERE, f"{dataset_name}_model.pt"),
                       map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    if dataset_name is None:
        dataset_name = DATASETS[0]
    _ensure_loaded(dataset_name)
    if not candidates:
        return []
    model = _models[dataset_name]
    maxlen = int(_metas[dataset_name]["maxlen"])
    sequence = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        sequence[-len(tail):] = tail
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.no_grad():
        scores = model.score_candidates(seq, items).cpu().tolist()
    return [float(score) for score in scores]
