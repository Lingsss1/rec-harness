from __future__ import annotations
import math
import os
import torch
import torch.nn as nn

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
        return outputs.transpose(-1, -2) + inputs


class SASRec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, num_heads=1,
                 dropout_rate=0.5, maxlen=128):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=1.0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.attention_layers.append(nn.MultiheadAttention(hidden_units, num_heads, dropout=dropout_rate))
            self.forward_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(hidden_units, dropout_rate))
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.register_buffer('causal_mask', torch.triu(torch.ones(maxlen, maxlen, dtype=torch.bool), diagonal=1), persistent=False)

    def log2feats(self, seq):
        batch, length = seq.shape
        positions = torch.arange(length, device=seq.device).unsqueeze(0).expand(batch, -1)
        x = self.item_emb(seq) * math.sqrt(self.hidden_units)
        x = self.emb_dropout(x + self.pos_emb(positions))
        keep = seq.ne(0).unsqueeze(-1)
        x = x * keep
        mask = self.causal_mask[:length, :length]
        for norm, attention, ff_norm, ff in zip(self.attention_layernorms, self.attention_layers, self.forward_layernorms, self.forward_layers):
            x = x.transpose(0, 1)
            queries = norm(x)
            output, _ = attention(queries, x, x, attn_mask=mask, need_weights=False)
            x = (queries + output).transpose(0, 1)
            x = ff(ff_norm(x)) * keep
        return self.last_layernorm(x)

    def forward(self, seq):
        return self.log2feats(seq)

    def predict(self, seq, candidates):
        features = self.log2feats(seq)[:, -1, :]
        return (self.item_emb(candidates) * features.unsqueeze(1)).sum(-1)



_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(4)


def _load(name):
    if name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(name))
    if name not in _MODELS:
        path = os.path.join(_HERE, name + '_model.pt')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        model = SASRec(**payload['config'])
        model.load_state_dict(payload['state_dict'])
        model.to(_DEVICE).eval()
        _MODELS[name] = model
    return _MODELS[name]


def predict(user_id, history, candidates, *, dataset_name=None):
    if not candidates:
        return []
    name = dataset_name if dataset_name is not None else _DATASETS[0]
    model = _load(name)
    hist = list(history)[-model.maxlen:]
    sequence = [0] * (model.maxlen - len(hist)) + hist
    seq_t = torch.tensor([sequence], dtype=torch.long, device=_DEVICE)
    cand_t = torch.tensor([list(candidates)], dtype=torch.long, device=_DEVICE)
    with torch.inference_mode():
        return model.predict(seq_t, cand_t)[0].float().cpu().tolist()
