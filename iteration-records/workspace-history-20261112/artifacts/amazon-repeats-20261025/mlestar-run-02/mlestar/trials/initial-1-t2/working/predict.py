from __future__ import annotations
import os
import pickle
import torch
from torch import nn
import torch.nn.functional as F

class FilterLayer(nn.Module):
    def __init__(self, maxlen, hidden_units, dropout_rate):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(maxlen // 2 + 1, hidden_units, 2) * 0.02)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units, eps=1e-12)

    def forward(self, x):
        # FFT must stay float32, including under mixed precision.
        with torch.autocast(device_type=x.device.type, enabled=False):
            spectrum = torch.fft.rfft(x.float(), dim=1, norm='ortho')
            weight = torch.view_as_complex(self.complex_weight.float().contiguous())
            filtered = torch.fft.irfft(spectrum * weight.unsqueeze(0), n=x.shape[1], dim=1, norm='ortho')
        return self.norm(x + self.dropout(filtered.to(x.dtype)))


class Intermediate(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.dense1 = nn.Linear(hidden_units, 4 * hidden_units)
        self.dense2 = nn.Linear(4 * hidden_units, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units, eps=1e-12)

    def forward(self, x):
        y = self.dense2(F.gelu(self.dense1(x)))
        return self.norm(x + self.dropout(y))


class FMLPRec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, dropout_rate=0.3, maxlen=128):
        super().__init__()
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.input_norm = nn.LayerNorm(hidden_units, eps=1e-12)
        self.input_dropout = nn.Dropout(dropout_rate)
        self.filters = nn.ModuleList([FilterLayer(maxlen, hidden_units, dropout_rate) for _ in range(num_blocks)])
        self.intermediates = nn.ModuleList([Intermediate(hidden_units, dropout_rate) for _ in range(num_blocks)])
        self.apply(self._initialize)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode(self, seq):
        mask = seq.ne(0).unsqueeze(-1)
        positions = torch.arange(seq.shape[1], device=seq.device)
        x = self.item_emb(seq) + self.pos_emb(positions).unsqueeze(0)
        x = self.input_dropout(self.input_norm(x)) * mask
        for frequency_filter, intermediate in zip(self.filters, self.intermediates):
            x = intermediate(frequency_filter(x)) * mask
        return x[:, -1, :]

    def forward(self, seq, positive, negatives):
        features = self.encode(seq)
        positive_logits = (features * self.item_emb(positive)).sum(-1)
        negative_logits = (features.unsqueeze(1) * self.item_emb(negatives)).sum(-1)
        return positive_logits, negative_logits

    def predict(self, seq, candidates):
        features = self.encode(seq)
        return (self.item_emb(candidates) * features.unsqueeze(1)).sum(-1)



_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_METAS = {}
_DEVICE = None


def _load(dataset_name):
    global _DEVICE
    if dataset_name in _MODELS:
        return _MODELS[dataset_name], _METAS[dataset_name]
    if dataset_name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    if _DEVICE is None:
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        _DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    if meta['architecture'] != 'fmlp':
        raise ValueError('Unsupported checkpoint architecture')
    model = FMLPRec(itemnum=meta['itemnum'], hidden_units=meta['hidden_units'],
                    num_blocks=meta['num_blocks'], dropout_rate=meta['dropout_rate'],
                    maxlen=meta['maxlen']).to(_DEVICE)
    weights = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'),
                         map_location=_DEVICE, weights_only=True)
    model.load_state_dict(weights)
    model.eval()
    _MODELS[dataset_name] = model
    _METAS[dataset_name] = meta
    return model, meta


@torch.no_grad()
def predict(user_id, history, candidates, *, dataset_name=None):
    if dataset_name is None:
        dataset_name = _DATASETS[0]
    model, meta = _load(dataset_name)
    if len(candidates) == 0:
        return []
    maxlen = meta['maxlen']
    hist = [int(item) for item in history[-maxlen:]]
    seq = [0] * (maxlen - len(hist)) + hist
    seq_t = torch.tensor([seq], dtype=torch.long, device=_DEVICE)
    cand_t = torch.tensor([candidates], dtype=torch.long, device=_DEVICE)
    return model.predict(seq_t, cand_t)[0].cpu().tolist()
