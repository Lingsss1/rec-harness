"""One GRU4Rec checkpoint per Amazon category."""
from __future__ import annotations
import os
import pickle
import torch
import torch.nn as nn

class GRU4RecModel(nn.Module):
    def __init__(self, itemnum, hidden_units=128, num_layers=1, dropout_rate=0.2):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.gru = nn.GRU(
            hidden_units, hidden_units, num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.residual_gate = nn.Parameter(torch.full((hidden_units,), -2.197224577))
        self.output_norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()

    def log2feats(self, sequence):
        if not torch.is_tensor(sequence):
            sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        else:
            sequence = sequence.to(device=self.item_emb.weight.device, dtype=torch.long)
        batch_size, width = sequence.shape
        mask = sequence.ne(0)
        lengths = mask.sum(dim=1)
        positions = torch.arange(width, device=sequence.device).unsqueeze(0)
        offsets = width - lengths.unsqueeze(1)
        # Compact left-padded histories for packing, preserving chronological order.
        compact_indices = (positions + offsets).clamp(max=width - 1)
        compact_items = sequence.gather(1, compact_indices)
        compact_mask = positions < lengths.unsqueeze(1)
        compact_items = compact_items.masked_fill(~compact_mask, 0)
        embeddings = self.dropout(self.item_emb(compact_items))
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths.clamp(min=1).detach().cpu(),
            batch_first=True, enforce_sorted=False,
        )
        packed_features, _ = self.gru(packed)
        recurrent, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_features, batch_first=True, total_length=width,
        )
        features = self.output_norm(
            recurrent + torch.sigmoid(self.residual_gate) * embeddings
        )
        # Return features in the original left-padded layout for shifted targets.
        restore_indices = (positions - offsets).clamp(min=0)
        features = features.gather(
            1, restore_indices.unsqueeze(-1).expand(batch_size, width, self.hidden_units)
        )
        return features.masked_fill(~mask.unsqueeze(-1), 0.0)

    def encode(self, seq):
        out = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        return out[torch.arange(seq.size(0), device=seq.device), last]

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=self.item_emb.weight.device)
        embeddings = self.item_emb(items)
        if items.ndim == 1:
            return features @ embeddings.transpose(0, 1)
        if items.ndim == features.ndim - 1 and items.shape == features.shape[:-1]:
            return (features * embeddings).sum(dim=-1)
        if features.ndim == 2 and items.ndim == 2:
            return torch.einsum('bd,bkd->bk', features, embeddings)
        return (features.unsqueeze(-2) * embeddings).sum(dim=-1)

    def forward(self, seq, pos, neg):
        out = self.log2feats(seq)
        return (out * self.item_emb(pos)).sum(-1), (out * self.item_emb(neg)).sum(-1)

    def score_candidates(self, seq, candidates):
        return self.predict_score(self.encode(seq), candidates)[0]


_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded(dataset_name):
    global _device
    if dataset_name in _models:
        return
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, f'{dataset_name}_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = GRU4RecModel(
        itemnum=meta['itemnum'], hidden_units=meta['hidden_units'],
        num_layers=meta['num_layers'], dropout_rate=meta['dropout_rate'],
    ).to(_device)
    state = torch.load(os.path.join(_HERE, f'{dataset_name}_model.pt'),
                       map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    ds = dataset_name if dataset_name is not None else DATASETS[0]
    if ds not in DATASETS:
        raise ValueError(f'Unknown dataset: {ds}')
    if len(candidates) == 0:
        return []
    _ensure_loaded(ds)
    model, meta = _models[ds], _metas[ds]
    maxlen = int(meta['maxlen'])
    seq = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        seq[-len(tail):] = tail
    seq_t = torch.tensor([seq], dtype=torch.long, device=_device)
    cand_t = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq_t, cand_t)
    return [float(score) for score in scores.cpu().tolist()]
