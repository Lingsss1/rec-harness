from __future__ import annotations
import os
import pickle
import torch
import torch.nn as nn

class GRU4RecModel(nn.Module):
    def __init__(self, itemnum, hidden_units=128, num_layers=1, dropout_rate=0.1):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.dropout = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(hidden_units, hidden_units, num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout_rate if num_layers > 1 else 0.0)
        self.item_bias = nn.Parameter(torch.zeros(itemnum + 1))
        nn.init.xavier_uniform_(self.item_emb.weight)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def log2feats(self, sequence):
        # Compact LEFT-padded histories before recurrence, then restore the
        # original positions for causal sequence-wide supervision.
        valid = sequence.ne(0)
        lengths = valid.sum(dim=1)
        batch, steps = sequence.shape
        positions = torch.arange(steps, device=sequence.device).unsqueeze(0)
        offsets = steps - lengths
        indices = (positions + offsets.unsqueeze(1)).clamp(max=steps - 1)
        compact = sequence.gather(1, indices)
        compact = compact.masked_fill(positions >= lengths.unsqueeze(1), 0)
        embeddings = self.dropout(self.item_emb(compact))
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths.clamp(min=1).cpu(),
            batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=steps)
        restore = (positions - offsets.unsqueeze(1)).clamp(min=0)
        output = output.gather(1, restore.unsqueeze(-1).expand(batch, steps, self.hidden_units))
        return output * valid.unsqueeze(-1)

    def encode(self, sequence):
        features = self.log2feats(sequence)
        positions = torch.arange(sequence.shape[1], device=sequence.device).expand_as(sequence)
        last = positions.masked_fill(sequence.eq(0), -1).max(dim=1).values.clamp(min=0)
        return features[torch.arange(sequence.shape[0], device=sequence.device), last]

    def predict_score(self, features, items):
        if features.ndim == 3:
            # log2feats restores LEFT padding; the final column is the latest
            # item for every nonempty history.
            features = features[:, -1, :]
        embeddings = self.item_emb(items)
        if items.ndim == 1:
            return features @ embeddings.T + self.item_bias[items]
        return (features.unsqueeze(1) * embeddings).sum(-1) + self.item_bias[items]

    def score_candidates(self, sequence, candidates):
        return self.predict_score(self.encode(sequence), candidates).squeeze(0)

    def forward(self, sequence, positive, negative):
        features = self.log2feats(sequence)
        pos_scores = (features * self.item_emb(positive)).sum(-1) + self.item_bias[positive]
        neg_scores = (features * self.item_emb(negative)).sum(-1) + self.item_bias[negative]
        return pos_scores, neg_scores


_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded(dataset_name):
    global _device
    if dataset_name in _models:
        return
    if dataset_name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = GRU4RecModel(**meta['model_config']).to(_device)
    state = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'),
                       map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


@torch.inference_mode()
def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    ds = dataset_name if dataset_name is not None else _DATASETS[0]
    _ensure_loaded(ds)
    if not candidates:
        return []
    model = _models[ds]
    maxlen = _metas[ds]['maxlen']
    tail = list(history[-maxlen:])
    sequence = torch.tensor([[0] * (maxlen - len(tail)) + tail],
                            dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    scores = model.score_candidates(sequence, items)
    return [float(value) for value in scores.cpu().tolist()]
