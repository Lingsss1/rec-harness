from __future__ import annotations
import os
import torch
from torch import nn

class GRURec(nn.Module):
    def __init__(self, itemnum, dim=128, layers=2, dropout=0.2, maxlen=64):
        super().__init__()
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.embedding = nn.Embedding(itemnum + 1, dim, padding_idx=0)
        self.input_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            dim,
            dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.context_output = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(2 * dim, dim)
        nn.init.constant_(self.gate.bias, -2.0)
        self.norm = nn.LayerNorm(dim)
        self.output_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(dim, dim, bias=False)
        self.item_bias = nn.Embedding(itemnum + 1, 1, padding_idx=0)
        nn.init.normal_(self.embedding.weight, std=0.05)
        nn.init.zeros_(self.item_bias.weight)
        with torch.no_grad():
            self.embedding.weight[0].zero_()
        for name, parameter in self.gru.named_parameters():
            if "weight_hh" in name:
                for block in parameter.chunk(3, dim=0):
                    nn.init.orthogonal_(block)
            elif "bias" in name:
                nn.init.zeros_(parameter)

    def forward(self, seq):
        item_memory = self.embedding(seq)
        embedded = self.input_dropout(item_memory)
        features, _ = self.gru(embedded)
        queries = self.query(features)
        # Atomic change: retrieve purchase identities directly, rather than
        # retrieving recurrently compressed states. These embeddings and all
        # retrieval projections train jointly with the recurrent network.
        keys = self.key(item_memory)
        values = self.value(item_memory)
        valid = seq.ne(0)
        length = seq.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=seq.device).tril()
        allowed = causal.unsqueeze(0) & valid.unsqueeze(1) & valid.unsqueeze(2)
        scores = (queries @ keys.transpose(1, 2)) * (features.shape[-1] ** -0.5)
        scores = scores.masked_fill(~allowed, float("-inf"))
        scores = scores.masked_fill(~valid.unsqueeze(-1), 0.0)
        weights = torch.softmax(scores, dim=-1).masked_fill(~allowed, 0.0)
        context = self.context_output(weights @ values)
        context = context.masked_fill(~valid.unsqueeze(-1), 0.0)
        gate = torch.sigmoid(self.gate(torch.cat([features, context], dim=-1)))
        features = features + gate * context
        features = self.output_dropout(self.norm(features))
        return self.output(features)

    def score_features(self, features, candidates):
        embeddings = self.embedding(candidates)
        bias = self.item_bias(candidates).squeeze(-1)
        return (features.unsqueeze(-2) * embeddings).sum(-1) + bias

    def predict(self, seq, candidates):
        features = self.forward(seq)
        last = seq.ne(0).sum(1).clamp(min=1) - 1
        features = features[torch.arange(seq.shape[0], device=seq.device), last]
        return self.score_features(features, candidates)


_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific',
             'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _load(dataset_name):
    name = dataset_name if dataset_name is not None else _DATASETS[0]
    if name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(name))
    if name not in _MODELS:
        checkpoint = torch.load(os.path.join(_HERE, name + '_model.pt'),
                                map_location=_DEVICE, weights_only=True)
        model = GRURec(**checkpoint['config']).to(_DEVICE)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.eval()
        _MODELS[name] = model
    return _MODELS[name]


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    model = _load(dataset_name)
    if len(candidates) == 0:
        return []
    hist = list(history[-model.maxlen:])
    seq = [0] * model.maxlen
    seq[:len(hist)] = hist
    seq_t = torch.tensor([seq], dtype=torch.long, device=_DEVICE)
    cand_t = torch.tensor([candidates], dtype=torch.long, device=_DEVICE)
    return model.predict(seq_t, cand_t)[0].float().cpu().tolist()
