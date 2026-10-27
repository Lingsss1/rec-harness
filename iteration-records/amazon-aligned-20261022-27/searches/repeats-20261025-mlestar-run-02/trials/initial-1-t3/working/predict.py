
import math
import torch
from torch import nn

class BertBlock(nn.Module):
    def __init__(self, hidden, heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(hidden, eps=1e-6)
        self.ff = nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.GELU(),
                                nn.Linear(4 * hidden, hidden), nn.Dropout(dropout))
        self.ff_norm = nn.LayerNorm(hidden, eps=1e-6)

    def forward(self, x, padding):
        a = self.attn(x, x, x, key_padding_mask=padding, need_weights=False)[0]
        x = self.attn_norm(x + self.attn_dropout(a))
        x = self.ff_norm(x + self.ff(x))
        return x.masked_fill(padding.unsqueeze(-1), 0)

class BERT4Rec(nn.Module):
    def __init__(self, itemnum, hidden=64, blocks=2, heads=2,
                 dropout=0.2, maxlen=128, mask_id=None):
        super().__init__()
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.mask_id = itemnum + 1 if mask_id is None else mask_id
        self.item_emb = nn.Embedding(itemnum + 2, hidden, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden)
        self.input_norm = nn.LayerNorm(hidden, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([BertBlock(hidden, heads, dropout) for _ in range(blocks)])
        self.output_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                         nn.LayerNorm(hidden, eps=1e-6))
        self.output_bias = nn.Parameter(torch.zeros(itemnum))
        self.apply(self._initialize)
        # MultiheadAttention's packed projection is not an nn.Linear module.
        for block in self.blocks:
            nn.init.normal_(block.attn.in_proj_weight, std=0.02)
            nn.init.zeros_(block.attn.in_proj_bias)
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
        padding = seq.eq(0)
        positions = torch.arange(seq.shape[1], device=seq.device)
        x = self.dropout(self.input_norm(self.item_emb(seq) + self.pos_emb(positions)[None]))
        x = x.masked_fill(padding.unsqueeze(-1), 0)
        for block in self.blocks:
            x = block(x, padding)
        return x

    def predict(self, seq, candidates):
        state = self.output_head(self.encode(seq)[:, -1])
        embeddings = self.item_emb(candidates)
        return (embeddings * state[:, None]).sum(-1) + self.output_bias[candidates - 1]


import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_DEVICE = None

def _load(dataset):
    global _DEVICE
    if dataset in _MODELS:
        return _MODELS[dataset]
    if _DEVICE is None:
        torch.set_num_threads(2)
        _DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(os.path.join(_HERE, dataset + '_model.pt'),
                            map_location='cpu', weights_only=True)
    model = BERT4Rec(**checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'])
    model.to(_DEVICE).eval()
    _MODELS[dataset] = model
    return model

def predict(user_id, history, candidates, *, dataset_name=None):
    if not candidates:
        return []
    dataset = dataset_name if dataset_name is not None else _DATASETS[0]
    if dataset not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset))
    model = _load(dataset)
    context = list(history)[-(model.maxlen - 1):]
    sequence = [0] * (model.maxlen - len(context) - 1) + context + [model.mask_id]
    seq = torch.tensor([sequence], dtype=torch.long, device=_DEVICE)
    cand = torch.tensor([list(candidates)], dtype=torch.long, device=_DEVICE)
    with torch.inference_mode():
        return model.predict(seq, cand)[0].float().cpu().tolist()
