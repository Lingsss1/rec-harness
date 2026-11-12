"""Per-dataset NextItNet predict function."""
from __future__ import annotations

### >>> Self_EvolveRec-BLOCK-START: Supply dependencies for the embedded trained architecture
import math
import os
import pickle

import torch
import torch.nn as nn
### <<< Self_EvolveRec-BLOCK-END

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
DATASETS = [
    part.strip()
    for part in os.environ.get("GAGC_DATASETS", ",".join(DEFAULT_DATASETS)).split(",")
    if part.strip()
]
_models = {}
_metas = {}
_device = None


### >>> Self_EvolveRec-BLOCK-START: Load the architecture embedded by the predictor writer
class NextItNetBlock(nn.Module):
    def __init__(
        self, hidden_units: int, dilation: int, kernel_size: int, dropout_rate: float
    ):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(
            hidden_units, hidden_units, kernel_size, dilation=dilation, padding=0
        )
        self.conv2 = nn.Conv1d(
            hidden_units, hidden_units, kernel_size, dilation=dilation, padding=0
        )
        self.norm1 = nn.LayerNorm(hidden_units)
        self.norm2 = nn.LayerNorm(hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        ### >>> Self_EvolveRec-BLOCK-START: Restore the corrected ungated NextItNet residual backbone
        self.activation = nn.GELU()
        ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Omit convolution taps that reach only causal zero padding
    def _causal_conv(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        dilation = conv.dilation[0]
        kernel = conv.kernel_size[0]
        active_taps = min(kernel, 1 + (x.size(-1) - 1) // dilation)
        pad = (active_taps - 1) * dilation
        padded = nn.functional.pad(x, (pad, 0))
        # Conv1d stores the current-position tap last. Earlier omitted taps
        # multiply only zeros and have zero gradient in the original operation.
        if active_taps == kernel:
            return conv(padded)
        return nn.functional.conv1d(
            padded,
            conv.weight[..., kernel - active_taps :].contiguous(),
            conv.bias,
            stride=conv.stride,
            dilation=conv.dilation,
            groups=conv.groups,
        )

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Preserve ungated causal updates and padding masks
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x
        y = (self.norm1(x) * mask).transpose(1, 2)
        y = self._causal_conv(self.conv1, y).transpose(1, 2)
        y = self.dropout(self.activation(y))
        y = (self.norm2(y) * mask).transpose(1, 2)
        y = self._causal_conv(self.conv2, y).transpose(1, 2)
        y = self.dropout(y)
        return (residual + y) * mask

    ### <<< Self_EvolveRec-BLOCK-END


class NextItNetModel(nn.Module):
    ### >>> Self_EvolveRec-BLOCK-START: Configure one shared NextItNet with a fixed ten-item raw context
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 8,
        kernel_size: int = 3,
        dropout_rate: float = 0.2,
        maxlen: int = 128,
        context_window: int = 10,
        window_batch_size: int = 256,
    ):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.context_window = int(context_window)
        self.window_batch_size = int(window_batch_size)
        if self.context_window != 10 or self.window_batch_size < 1 or maxlen < 1:
            raise ValueError(
                "Require context_window=10, positive maxlen and window batch size"
            )
        ### <<< Self_EvolveRec-BLOCK-END
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        dilations = [1, 2, 4, 8] * max(1, math.ceil(num_blocks / 4))
        self.blocks = nn.ModuleList(
            [
                NextItNetBlock(hidden_units, dilations[i], kernel_size, dropout_rate)
                for i in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

    ### >>> Self_EvolveRec-BLOCK-START: Independently encode raw ten-item windows with preserved corrected-baseline absolute positions
    def raw_windows(self, sequence: torch.Tensor, target_positions: torch.Tensor):
        if sequence.ndim != 2 or not 0 < sequence.size(1) <= self.maxlen:
            raise ValueError("Expected B x L histories with 0 < L <= maxlen")
        if target_positions.ndim != 2 or target_positions.size(1) != 2:
            raise ValueError("Expected N x 2 row/position indices")
        rows, terminal = target_positions.unbind(dim=1)
        histories = sequence.index_select(0, rows)
        indices = torch.arange(sequence.size(1), device=sequence.device)
        eligible = (histories != 0) & (indices[None, :] <= terminal[:, None])
        # Select raw input IDs, never states computed from an older context.
        # Sorting the selected indices restores chronological order, with PAD
        # indices at the left. No aligned next-item target enters this window.
        selected = indices.expand_as(histories).masked_fill(~eligible, -1)
        selected = selected.topk(
            min(self.context_window, sequence.size(1)), dim=1, sorted=True
        ).values.flip(dims=(1,))
        if selected.size(1) < self.context_window:
            selected = nn.functional.pad(
                selected, (self.context_window - selected.size(1), 0), value=-1
            )
        occupied = selected >= 0
        windows = histories.gather(1, selected.clamp(min=0))
        windows = windows.masked_fill(~occupied, 0)
        # The supplied corrected ungated baseline uses absolute input indices.
        # Keep those IDs instead of renumbering each independently encoded window.
        position_ids = (selected + 1).masked_fill(~occupied, 0)
        return windows, position_ids

    def _encode_raw_windows(
        self, windows: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        mask = (windows != 0).unsqueeze(-1).to(self.item_emb.weight.dtype)
        x = self.item_emb(windows) * (self.hidden_units**0.5)
        x = self.dropout(x + self.pos_emb(position_ids)) * mask
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x) * mask

    def encode_window_terminal(
        self, sequence: torch.Tensor, target_positions: torch.Tensor
    ) -> torch.Tensor:
        windows, position_ids = self.raw_windows(sequence, target_positions)
        features = self._encode_raw_windows(windows, position_ids)
        indices = torch.arange(windows.size(1), device=windows.device)
        last = indices.expand_as(windows).masked_fill(windows == 0, -1).amax(dim=1)
        rows = torch.arange(windows.size(0), device=windows.device)
        values = features[rows, last.clamp(min=0)]
        return values * (last >= 0).unsqueeze(-1).to(values.dtype)

    def encode_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        if seq.ndim != 2 or not 0 < seq.size(1) <= self.maxlen:
            raise ValueError("Expected B x L histories with 0 < L <= maxlen")
        positions = (seq != 0).nonzero(as_tuple=False)
        output = self.item_emb.weight.new_zeros((seq.numel(), self.hidden_units))
        # Compatibility path for ModelView and probes. Training calls the
        # terminal helper directly and releases each microbatch graph promptly.
        for chunk in positions.split(self.window_batch_size):
            if chunk.size(0) == 0:
                continue
            values = self.encode_window_terminal(seq, chunk)
            flat = chunk[:, 0] * seq.size(1) + chunk[:, 1]
            output = output.index_copy(0, flat, values)
        return output.reshape(seq.size(0), seq.size(1), self.hidden_units)

    def log2feats(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.encode_sequence(sequence)

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Preserve aligned and multi-negative forward compatibility
    def forward(self, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        out = self.log2feats(seq)
        pos_scores = self.predict_score(out, pos)
        neg_scores = self.predict_score(out, neg)
        return pos_scores, neg_scores

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Use the identical terminal-window path for prediction and support family diagnostic shapes
    def predict_score(
        self, features: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.item_emb(items)
        if features.shape[:-1] == items.shape:
            return (features * embeddings).sum(dim=-1)
        if items.ndim == features.ndim and items.shape[:-1] == features.shape[:-1]:
            return (features.unsqueeze(-2) * embeddings).sum(dim=-1)
        return (features * embeddings).sum(dim=-1)

    def score_candidates(
        self, seq: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        if seq.ndim != 2 or seq.size(0) == 0 or not 0 < seq.size(1) <= self.maxlen:
            raise ValueError("Expected nonempty B x L histories with L <= maxlen")
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).amax(dim=1)
        rows = torch.arange(seq.size(0), device=seq.device)
        selected = torch.stack((rows, last.clamp(min=0)), dim=1)
        user_emb = self.encode_window_terminal(seq, selected)
        user_emb = user_emb * (last >= 0).unsqueeze(-1).to(user_emb.dtype)
        if candidates.ndim == 1:
            scores = user_emb @ self.item_emb(candidates).transpose(0, 1)
        elif candidates.ndim == 2 and candidates.size(0) == seq.size(0):
            scores = self.predict_score(user_emb[:, None, :], candidates)
        else:
            raise ValueError("Expected shared K or per-history B x K candidates")
        return scores[0] if seq.size(0) == 1 else scores

    ### <<< Self_EvolveRec-BLOCK-END


def _load_model_class():
    return NextItNetModel
### <<< Self_EvolveRec-BLOCK-END


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ModelClass = _load_model_class()
    for ds in DATASETS:
        working_dir = _HERE if os.path.basename(_HERE) == "working" else os.path.join(_HERE, "working")
        meta_path = os.path.join(working_dir, f"{ds}_meta.pkl")
        ckpt_path = os.path.join(working_dir, f"{ds}_model.pt")
        ### >>> Self_EvolveRec-BLOCK-START: Require a checkpoint and architecture metadata for every routed category
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Missing NextItNet artifacts for {ds}: {meta_path}, {ckpt_path}"
            )
        ### <<< Self_EvolveRec-BLOCK-END
        ### >>> Self_EvolveRec-BLOCK-START: Reject incompatible predictor architecture metadata explicitly
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        if meta.get("architecture") != "nextitnet_raw_window10_v1":
            raise RuntimeError(f"Incompatible NextItNet architecture for {ds}")
        if meta.get("context_window") != 10:
            raise RuntimeError(f"Expected ten-interaction context for {ds}")
        ### <<< Self_EvolveRec-BLOCK-END
        model = ModelClass(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            kernel_size=meta["kernel_size"],
            ### >>> Self_EvolveRec-BLOCK-START: Reconstruct the identical raw-window architecture in the generated predictor
            dropout_rate=meta["dropout_rate"],
            maxlen=meta["maxlen"],
            context_window=meta["context_window"],
            window_batch_size=meta["window_batch_size"],
            ### <<< Self_EvolveRec-BLOCK-END
        ).to(_device)
        model.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ### >>> Self_EvolveRec-BLOCK-START: Prevent cross-category fallback and fabricated scores
    if dataset_name is None:
        if len(_models) != 1:
            raise ValueError("dataset_name is required when multiple categories are loaded")
        dataset_name = next(iter(_models))
    if dataset_name not in _models:
        raise ValueError(f"No trained NextItNet model for category {dataset_name!r}")
    model = _models[dataset_name]
    meta = _metas[dataset_name]
    ### <<< Self_EvolveRec-BLOCK-END

    maxlen = int(meta.get("maxlen", 128))
    seq_arr = [0] * maxlen
    for i, item in enumerate(reversed(history[-maxlen:])):
        seq_arr[maxlen - 1 - i] = item
    seq_t = torch.tensor([seq_arr], dtype=torch.long, device=_device)
    cand_t = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.no_grad():
        scores = model.score_candidates(seq_t, cand_t).detach().cpu().tolist()
    return [float(s) for s in scores]
