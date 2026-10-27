from __future__ import annotations
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, 1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, 1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        outputs = self.conv1(inputs.transpose(-1, -2))
        outputs = self.relu(self.dropout1(outputs))
        outputs = self.dropout2(self.conv2(outputs))
        return inputs + outputs.transpose(-1, -2)


class SASRec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, num_heads=1,
                 dropout_rate=0.2, maxlen=128, normalize=True, temperature=0.1):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.normalize = normalize
        self.temperature = temperature
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=1.0)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.attention_layers.append(nn.MultiheadAttention(hidden_units, num_heads, dropout=dropout_rate, batch_first=True))
            self.forward_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(hidden_units, dropout_rate))
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.register_buffer('causal_mask', torch.ones(maxlen, maxlen, dtype=torch.bool).triu(1), persistent=False)

    def log2feats(self, seq):
        batch_size, length = seq.shape
        positions = torch.arange(length, device=seq.device).unsqueeze(0)
        values = self.item_emb(seq) * math.sqrt(self.hidden_units)
        values = self.emb_dropout(values + self.pos_emb(positions))
        mask = (seq != 0).unsqueeze(-1)
        values = values * mask
        for attention_norm, attention, forward_norm, forward_layer in zip(
                self.attention_layernorms, self.attention_layers,
                self.forward_layernorms, self.forward_layers):
            queries = attention_norm(values)
            attended, _ = attention(queries, values, values,
                                   attn_mask=self.causal_mask[:length, :length],
                                   need_weights=False)
            values = queries + attended
            values = forward_layer(forward_norm(values)) * mask
        return self.last_layernorm(values)

    def forward(self, seq):
        return self.log2feats(seq)

    def predict(self, seq, candidates):
        features = self.log2feats(seq)[:, -1, :].float()
        embeddings = self.item_emb(candidates).float()
        if self.normalize:
            features = F.normalize(features, dim=-1)
            embeddings = F.normalize(embeddings, dim=-1)
        return (embeddings * features.unsqueeze(1)).sum(-1) / self.temperature


_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _get_model(dataset_name):
    name = dataset_name if dataset_name is not None else _DATASETS[0]
    if name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(name))
    if name not in _models:
        path = os.path.join(_HERE, name + '_model.pt')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        config = dict(checkpoint['config'])
        config['dropout_rate'] = 0.0
        model = SASRec(**config)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(_device).eval()
        _models[name] = model
    return _models[name]


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    if len(candidates) == 0:
        return []
    model = _get_model(dataset_name)
    history = [int(item) if 0 <= int(item) <= model.itemnum else 0
               for item in history[-model.maxlen:]]
    sequence = [0] * (model.maxlen - len(history)) + history
    ids = [int(item) for item in candidates]
    legal = [1 <= item <= model.itemnum for item in ids]
    safe_ids = [item if ok else 0 for item, ok in zip(ids, legal)]
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    cand = torch.tensor([safe_ids], dtype=torch.long, device=_device)
    scores = model.predict(seq, cand)[0].cpu().tolist()
    return [float(score) if ok else -1e9 for score, ok in zip(scores, legal)]
