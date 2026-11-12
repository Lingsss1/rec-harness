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
        self.gru = nn.GRU(hidden_units, hidden_units, num_layers=num_layers,
                          batch_first=True, dropout=dropout_rate if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout_rate)
        nn.init.xavier_uniform_(self.item_emb.weight)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def log2feats(self, sequence):
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        output, _ = self.gru(self.dropout(self.item_emb(sequence)))
        return output

    def encode(self, sequence):
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        output = self.log2feats(sequence)
        positions = torch.arange(sequence.shape[1], device=sequence.device).expand_as(sequence)
        last = positions.masked_fill(sequence.eq(0), -1).amax(dim=1)
        result = output[torch.arange(sequence.shape[0], device=sequence.device), last.clamp_min(0)]
        return result * last.ge(0).unsqueeze(-1).to(result.dtype)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=self.item_emb.weight.device)
        embedding = self.item_emb(items)
        if items.ndim == 1:
            return features @ embedding.transpose(0, 1)
        return (features.unsqueeze(-2) * embedding).sum(dim=-1)

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        return ((features * self.item_emb(pos)).sum(-1),
                (features * self.item_emb(neg)).sum(-1))

    def score_candidates(self, seq, candidates):
        scores = self.predict_score(self.encode(seq), candidates)
        return scores[0] if scores.shape[0] == 1 else scores


_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _ensure_loaded(dataset_name):
    if dataset_name not in DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    if dataset_name in _models:
        return
    with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = GRU4RecModel(meta['itemnum'], meta['hidden_units'], meta['num_layers'], meta['dropout_rate']).to(_device)
    state = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'), map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    dataset_name = dataset_name if dataset_name is not None else DATASETS[0]
    _ensure_loaded(dataset_name)
    if len(candidates) == 0:
        return []
    model = _models[dataset_name]
    maxlen = _metas[dataset_name]['maxlen']
    tail = list(history[-maxlen:])
    sequence = [0] * (maxlen - len(tail)) + tail
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    return [float(x) for x in model.score_candidates(seq, items).cpu().tolist()]
