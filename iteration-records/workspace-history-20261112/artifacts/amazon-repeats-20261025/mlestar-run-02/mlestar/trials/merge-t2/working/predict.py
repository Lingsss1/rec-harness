from __future__ import annotations
import os
import pickle
import torch
from torch import nn
from torch.nn import functional as F

class FilterLayer(nn.Module):
    def __init__(self, width, maxlen, dropout):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.empty(1, maxlen // 2 + 1, width, 2))
        nn.init.normal_(self.complex_weight, std=0.02)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(width, eps=1e-12)

    def forward(self, inputs):
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            spectrum = torch.fft.rfft(inputs.float(), dim=1, norm='ortho')
            weight = torch.view_as_complex(self.complex_weight.float().contiguous())
            filtered = torch.fft.irfft(spectrum * weight, n=inputs.shape[1], dim=1, norm='ortho')
        return self.norm(inputs + self.dropout(filtered.to(inputs.dtype)))


class Intermediate(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.dense1 = nn.Linear(width, 4 * width)
        self.dense2 = nn.Linear(4 * width, width)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(width, eps=1e-12)

    def forward(self, inputs):
        outputs = self.dense2(F.gelu(self.dense1(inputs)))
        return self.norm(inputs + self.dropout(outputs))


class FMLPRec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, maxlen=128, dropout_rate=0.3):
        super().__init__()
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_norm = nn.LayerNorm(hidden_units, eps=1e-12)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.filters = nn.ModuleList([FilterLayer(hidden_units, maxlen, dropout_rate) for _ in range(num_blocks)])
        self.intermediates = nn.ModuleList([Intermediate(hidden_units, dropout_rate) for _ in range(num_blocks)])
        self.apply(self._initialize)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode(self, seq):
        mask = seq.ne(0).unsqueeze(-1)
        positions = torch.arange(seq.shape[1], device=seq.device).unsqueeze(0)
        x = self.emb_dropout(self.emb_norm(self.item_emb(seq) + self.pos_emb(positions)))
        x = x * mask
        for frequency_filter, feed_forward in zip(self.filters, self.intermediates):
            x = frequency_filter(x)
            x = feed_forward(x) * mask
        return x[:, -1, :]

    def forward(self, seq, positive, negative):
        features = self.encode(seq)
        positive_logits = (features * self.item_emb(positive)).sum(-1)
        negative_logits = torch.einsum('bd,bkd->bk', features, self.item_emb(negative))
        return positive_logits, negative_logits

    def predict(self, seq, candidates):
        features = self.encode(seq)
        return torch.einsum('bd,bkd->bk', features, self.item_emb(candidates))



_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_METAS = {}
_DEVICE = None


def _load(dataset):
    global _DEVICE
    if dataset in _MODELS:
        return _MODELS[dataset], _METAS[dataset]
    if dataset not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset))
    if _DEVICE is None:
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        _DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, dataset + '_meta.pkl'), 'rb') as handle:
        meta = pickle.load(handle)
    if meta['architecture'] != 'fmlp':
        raise ValueError('Unsupported model architecture')
    config = {key: meta[key] for key in ('itemnum', 'hidden_units', 'num_blocks', 'maxlen', 'dropout_rate')}
    model = FMLPRec(**config).to(_DEVICE)
    weights = torch.load(os.path.join(_HERE, dataset + '_model.pt'), map_location=_DEVICE, weights_only=True)
    model.load_state_dict(weights)
    model.eval()
    _MODELS[dataset] = model
    _METAS[dataset] = meta
    return model, meta


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    if len(candidates) == 0:
        return []
    dataset = dataset_name if dataset_name is not None else _DATASETS[0]
    model, meta = _load(dataset)
    prefix = list(history)[-meta['maxlen']:]
    prefix = [int(item) if 0 < int(item) <= meta['itemnum'] else 0 for item in prefix]
    sequence = [0] * (meta['maxlen'] - len(prefix)) + prefix
    safe_candidates = [int(item) if 0 < int(item) <= meta['itemnum'] else 0 for item in candidates]
    seq = torch.tensor([sequence], dtype=torch.long, device=_DEVICE)
    cand = torch.tensor([safe_candidates], dtype=torch.long, device=_DEVICE)
    scores = model.predict(seq, cand)[0].cpu().tolist()
    return [score if item != 0 else -1e9 for score, item in zip(scores, safe_candidates)]
