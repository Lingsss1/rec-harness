from __future__ import annotations
import os
import pickle
import torch
import torch.nn as nn

class BERT4RecModel(nn.Module):
    def __init__(self, itemnum: int, hidden_units: int = 64, num_blocks: int = 2,
                 num_heads: int = 2, dropout_rate: float = 0.2, maxlen: int = 128):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.mask_token = itemnum + 1
        self.item_emb = nn.Embedding(itemnum + 2, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_units, nhead=num_heads,
            dim_feedforward=hidden_units * 4, dropout=dropout_rate,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

    def encode_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        batch, length = seq.shape
        pos = torch.arange(length, device=seq.device).unsqueeze(0).expand(batch, -1) + 1
        pos = pos * (seq != 0).long()
        x = self.item_emb(seq) * (self.hidden_units ** 0.5) + self.pos_emb(pos)
        x = self.dropout(x)
        padding = seq == 0
        safe_padding = padding.clone()
        safe_padding[padding.all(dim=1), -1] = False
        x = self.encoder(x, src_key_padding_mask=safe_padding)
        return self.norm(x)

    def log2feats(self, sequence):
        if not torch.is_tensor(sequence):
            sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        else:
            sequence = sequence.to(device=self.item_emb.weight.device, dtype=torch.long)
        if sequence.ndim == 1:
            sequence = sequence.unsqueeze(0)
        return self.encode_sequence(sequence)

    def predict_score(self, features, items):
        if not torch.is_tensor(items):
            items = torch.as_tensor(items, dtype=torch.long, device=self.item_emb.weight.device)
        else:
            items = items.to(device=self.item_emb.weight.device, dtype=torch.long)
        return (features * self.item_emb(items)).sum(dim=-1)

    def forward(self, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        out = self.encode_sequence(seq)
        return self.predict_score(out, pos), self.predict_score(out, neg)

    def score_candidates(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        prediction_seq = torch.zeros_like(seq)
        prediction_seq[:, :-1] = seq[:, 1:]
        prediction_seq[:, -1] = self.mask_token
        out = self.encode_sequence(prediction_seq)
        # Histories are left padded; the appended mask is the final nonpadding token.
        user_emb = out[0, -1]
        return torch.matmul(self.item_emb(candidates), user_emb)


_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASETS = ["Movies_and_TV", "Industrial_and_Scientific", "Electronics", "CDs_and_Vinyl"]
DATASETS = [x.strip() for x in os.environ.get("GAGC_DATASETS", ",".join(DEFAULT_DATASETS)).split(",") if x.strip()]
_models = {}
_metas = {}
_device = None


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ds in DATASETS:
        with open(os.path.join(_HERE, f"{ds}_meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        model = BERT4RecModel(
            itemnum=meta["itemnum"], hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"], num_heads=meta["num_heads"],
            dropout_rate=0.0, maxlen=meta["maxlen"],
        ).to(_device)
        state = torch.load(os.path.join(_HERE, f"{ds}_model.pt"), map_location=_device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    if dataset_name is not None:
        if dataset_name not in _models:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")
        ds = dataset_name
    else:
        ds = next(iter(_models))
    model, meta = _models[ds], _metas[ds]
    maxlen = int(meta["maxlen"])
    seq_arr = [0] * maxlen
    for i, item in enumerate(reversed(history[-maxlen:])):
        seq_arr[maxlen - 1 - i] = item
    seq_t = torch.tensor([seq_arr], dtype=torch.long, device=_device)
    cand_t = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.no_grad():
        scores = model.score_candidates(seq_t, cand_t).detach().cpu().tolist()
    return [float(s) for s in scores]
