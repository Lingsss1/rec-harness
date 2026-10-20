# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2025-2026 USTC-StarTeam.
# Copyright 2026 longseq-kuairand-bench contributors.
# SPDX-License-Identifier: Apache-2.0
# HSTU equations, initialization and relative bias are adapted from the pinned
# FuXi-Linear implementation; see NOTICE. DARE is independently implemented from
# its published mechanism; no unlicensed DARE repository source is copied here.
"""Portable PyTorch models. No production FP8/custom CUDA performance claims."""
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .ccformer import CCFormerEncoder
from .registry import MODEL_NAMES


@dataclass
class ModelConfig:
    name: str = "hstu"
    num_items: int = 131091
    max_history: int = 1024
    dim: int = 128
    blocks: int = 4
    heads: int = 4
    dqk: int = 32
    dv: int = 128
    input_dropout: float = 0.5
    linear_dropout: float = 0.5
    # The reference pads the 1024 history by gr_output_length + 1 = 11.
    reference_output_padding: int = 11
    query_chunk: int = 64
    checkpoint_blocks: bool = True
    sla_local: int = 256
    sla_global: int = 32
    truncate_keep: int = 512
    # One post-truncation layer gives exact per-prefix all-position supervision.
    truncate_after: int = 3
    dare_top_k: int = 20
    cc_group_size: int = 8
    cc_channel_size: int = 16
    cc_mixing_expansion: int = 2
    cc_kernel_size: int = 3
    cc_stride: int = 2
    cc_time_unit_ms: float = 1000.0

    def __post_init__(self):
        if self.name not in MODEL_NAMES:
            raise ValueError("Unknown model: " + self.name)
        if min(self.num_items, self.max_history, self.dim, self.query_chunk) <= 0:
            raise ValueError("Item count, history length, dimension and chunks must be positive")
        if self.name == "ccformer":
            if min(self.blocks, self.heads, self.cc_group_size, self.cc_channel_size,
                   self.cc_mixing_expansion, self.cc_kernel_size, self.cc_stride, self.cc_time_unit_ms) <= 0:
                raise ValueError("CCFormer block, head, grouping and compression settings must be positive")
            if self.dim % self.heads or self.dim % self.cc_channel_size:
                raise ValueError("CCFormer dim must be divisible by heads and channel group size")
        if self.name == "ultra_hstu":
            if self.truncate_after != self.blocks - 1:
                raise ValueError("Exact all-position adapter requires exactly one post-truncation layer")
            if self.truncate_keep <= 0 or self.sla_local <= 0 or self.sla_global <= 0:
                raise ValueError("ULTRA window sizes must be positive")


def reference_truncated_normal(tensor, std):
    with torch.no_grad():
        samples = tensor.new_empty(tensor.shape + (4,)).normal_()
        valid = (samples < 2) & (samples > -2)
        index = valid.max(-1, keepdim=True)[1]
        tensor.copy_(samples.gather(-1, index).squeeze(-1) * std)


class RelativeBias(nn.Module):
    def __init__(self, padded_length):
        super().__init__()
        self.padded_length = padded_length
        self.ts_weight = nn.Parameter(torch.empty(129).normal_(0, 0.02))
        self.pos_weight = nn.Parameter(torch.empty(2 * padded_length - 1).normal_(0, 0.02))

    def forward(self, timestamps, q_positions, k_positions):
        # Intentionally use timestamp(q+1), as in the reference HSTU bias.
        delta = timestamps[:, q_positions + 1, None] - timestamps[:, None, k_positions]
        buckets = (torch.log(delta.abs().clamp(min=1)) / 0.301).long().clamp(0, 128)
        indices = self.padded_length - 1 + k_positions[None, :] - q_positions[:, None]
        return self.pos_weight[indices][None, :, :] + self.ts_weight[buckets]


def attention_mask(q, k, local=0, global_size=0, tail=0, origin=0):
    """Chronological SLA; when truncated, rebase global anchors to each prefix."""
    causal = k[None, :] <= q[:, None]
    base = (q - tail + 1).clamp(min=origin) if tail else torch.full_like(q, origin)
    if local or global_size:
        nearby = (q[:, None] - k[None, :]) < local
        global_part = (k[None, :] >= base[:, None]) & (k[None, :] < base[:, None] + global_size)
        causal = causal & (nearby | global_part)
    if tail:
        causal = causal & (k[None, :] >= base[:, None])
    return causal


class STUBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.uvqk = nn.Parameter(torch.empty(cfg.dim, 2 * cfg.heads * (cfg.dv + cfg.dqk)).normal_(0, 0.02))
        self.output = nn.Linear(cfg.heads * cfg.dv, cfg.dim)
        nn.init.xavier_uniform_(self.output.weight)
        self.bias = RelativeBias(cfg.max_history + cfg.reference_output_padding)

    def forward(self, x, timestamps, positions=None, local=0, global_size=0, tail=0):
        c = self.cfg
        b, length, _ = x.shape
        if positions is None:
            positions = torch.arange(length, device=x.device)
        projected = F.silu(F.layer_norm(x, (c.dim,), eps=1e-6) @ self.uvqk)
        u, v, q, k = projected.split([c.heads * c.dv, c.heads * c.dv,
                                    c.heads * c.dqk, c.heads * c.dqk], dim=-1)
        q = q.reshape(b, length, c.heads, c.dqk)
        k = k.reshape(b, length, c.heads, c.dqk)
        v = v.reshape(b, length, c.heads, c.dv)
        pieces = []
        for left in range(0, length, c.query_chunk):
            right = min(left + c.query_chunk, length)
            qp = positions[left:right]
            # Union of allowed keys for this query tile. Sparse models avoid a
            # full LxL materialization and do not gather a [B,L,window,D] tensor.
            allowed = attention_mask(qp, positions, local, global_size, tail, positions[0])
            selected = allowed.any(dim=0).nonzero(as_tuple=True)[0]
            kp = positions[selected]
            logits = torch.einsum("bqhd,bkhd->bhqk", q[:, left:right], k[:, selected])
            logits = logits + self.bias(timestamps, qp, kp).unsqueeze(1)
            weights = F.silu(logits) / self.bias.padded_length
            weights = weights * allowed[:, selected][None, None, :, :]
            attended = torch.einsum("bhqk,bkhd->bqhd", weights, v[:, selected])
            pieces.append(attended.reshape(b, right - left, c.heads * c.dv))
        attended = torch.cat(pieces, dim=1)
        gated = u * F.layer_norm(attended, (c.heads * c.dv,), eps=1e-6)
        return x + self.output(F.dropout(gated, c.linear_dropout, self.training))


class HSTUEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.item_embedding = nn.Embedding(cfg.num_items + 1, cfg.dim, padding_idx=0)
        reference_truncated_normal(self.item_embedding.weight, 0.02)
        self.position_embedding = nn.Embedding(cfg.max_history + cfg.reference_output_padding, cfg.dim)
        nn.init.xavier_normal_(self.position_embedding.weight)
        self.layers = nn.ModuleList([STUBlock(cfg) for _ in range(cfg.blocks)])

    def initial(self, ids):
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = self.item_embedding(ids) * math.sqrt(self.cfg.dim) + self.position_embedding(pos)[None]
        return F.dropout(x, self.cfg.input_dropout, self.training) * ids.ne(0).unsqueeze(-1)

    def run_layer(self, layer, x, times, tail=0):
        ultra = self.cfg.name == "ultra_hstu"
        def operation(values, timestamps):
            return layer(values, timestamps, local=self.cfg.sla_local if ultra else 0,
                         global_size=self.cfg.sla_global if ultra else 0, tail=tail)
        if self.training and self.cfg.checkpoint_blocks and torch.is_grad_enabled():
            return checkpoint(operation, x, times, use_reentrant=False)
        return operation(x, times)

    def encode_all(self, ids, times):
        x = self.initial(ids)
        valid = ids.ne(0).unsqueeze(-1)
        for i, layer in enumerate(self.layers):
            tail = self.cfg.truncate_keep if self.cfg.name == "ultra_hstu" and i == self.cfg.truncate_after else 0
            x = self.run_layer(layer, x, times, tail) * valid
        return F.normalize(x, dim=-1, eps=1e-6)

    def encode_current(self, ids, times, lengths):
        if self.cfg.name != "ultra_hstu":
            full = self.encode_all(ids, times)
            return full[torch.arange(len(ids), device=ids.device), lengths - 1]
        # Physically truncate before the final layer at inference. The training
        # band formulation is exactly this operation for every supervised prefix.
        x = self.initial(ids)
        valid = ids.ne(0).unsqueeze(-1)
        for layer in self.layers[:-1]:
            x = self.run_layer(layer, x, times) * valid
        outputs = []
        for b, end in enumerate(lengths.tolist()):
            start = max(0, end - self.cfg.truncate_keep)
            positions = torch.arange(start, end, device=ids.device)
            y = self.layers[-1](x[b:b + 1, start:end], times[b:b + 1], positions,
                                local=self.cfg.sla_local, global_size=self.cfg.sla_global)
            outputs.append(y[0, -1])
        return F.normalize(torch.stack(outputs), dim=-1, eps=1e-6)

    def score_queries(self, context, ids, rows, positions, candidates):
        queries = context[rows, positions]
        items = F.normalize(self.item_embedding(candidates), dim=-1, eps=1e-6)
        return torch.einsum("md,mcd->mc", queries, items)

    def score_current(self, context, candidates):
        items = F.normalize(self.item_embedding(candidates), dim=-1, eps=1e-6)
        return context @ items.T


class DAREEncoder(nn.Module):
    """ID-only DARE task adaptation; scoring remains candidate-conditioned."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.attention_embedding = nn.Embedding(cfg.num_items + 1, cfg.dim, padding_idx=0)
        self.representation_embedding = nn.Embedding(cfg.num_items + 1, cfg.dim, padding_idx=0)
        nn.init.normal_(self.attention_embedding.weight, std=0.02)
        nn.init.normal_(self.representation_embedding.weight, std=0.02)
        # No batch norm: a candidate's score must not depend on evaluation tiles.
        self.mlp = nn.Sequential(nn.Linear(3 * cfg.dim, 200), nn.PReLU(init=0.1),
                                 nn.Linear(200, 80), nn.PReLU(init=0.1), nn.Linear(80, 2))

    def encode_all(self, ids, times):
        del times
        return (self.attention_embedding(ids),
                F.normalize(self.representation_embedding(ids), dim=-1, eps=1e-6))

    def score_queries(self, context, ids, rows, positions, candidates):
        keys, values = context
        target_key = self.attention_embedding(candidates)
        target_value = F.normalize(self.representation_embedding(candidates), dim=-1, eps=1e-6)
        scores = torch.einsum("mcd,mld->mcl", target_key, keys[rows]) / math.sqrt(self.cfg.dim)
        available = ids[rows].ne(0) & (torch.arange(ids.shape[1], device=ids.device)[None] <= positions[:, None])
        scores = scores.masked_fill(~available[:, None], -torch.inf)
        top_scores, top_indices = scores.topk(min(self.cfg.dare_top_k, ids.shape[1]), dim=-1)
        valid = torch.isfinite(top_scores)
        # At least the first history item exists for every supervised query.
        weights = torch.softmax(top_scores, dim=-1).masked_fill(~valid, 0)
        m = torch.arange(len(rows), device=ids.device)[:, None, None]
        selected = values[rows][m, top_indices]
        history = F.normalize((weights.unsqueeze(-1) * selected).sum(dim=-2), dim=-1, eps=1e-6)
        features = torch.cat((target_value, history, target_value * history), dim=-1)
        logits = self.mlp(features)
        return logits[..., 1] - logits[..., 0]


class BenchmarkModel(nn.Module):
    def __init__(self, cfg, negatives=128, temperature=0.05, loss_chunk=64):
        super().__init__()
        self.cfg, self.negatives, self.temperature, self.loss_chunk = cfg, negatives, temperature, loss_chunk
        self.encoder = (DAREEncoder(cfg) if cfg.name == "dare" else
                        CCFormerEncoder(cfg) if cfg.name == "ccformer" else HSTUEncoder(cfg))

    def _chunk_loss(self, *args):
        # context tensors, then ids, rows, positions, candidate IDs.
        *ctx, ids, rows, positions, candidates = args
        context = tuple(ctx) if self.cfg.name in {"dare", "ccformer"} else ctx[0]
        logits = self.encoder.score_queries(context, ids, rows, positions, candidates).float() / self.temperature
        collision = candidates[:, 1:].eq(candidates[:, :1])
        logits = torch.cat((logits[:, :1], logits[:, 1:].masked_fill(collision, -5e4)), dim=1)
        return -F.log_softmax(logits, dim=-1)[:, 0].sum()

    def forward(self, batch):
        ids, times, labels = batch["ids"], batch["times"], batch["labels"]
        context = self.encoder.encode_all(ids, times)
        context_tensors = context if isinstance(context, tuple) else (context,)
        rows, positions = labels.ne(0).nonzero(as_tuple=True)
        if not len(rows):
            raise ValueError("No supervised next-item positions")
        loss = ids.new_zeros((), dtype=torch.float32)
        for left in range(0, len(rows), self.loss_chunk):
            r, p = rows[left:left + self.loss_chunk], positions[left:left + self.loss_chunk]
            positive = labels[r, p, None]
            negative = torch.randint(1, self.cfg.num_items + 1, (len(r), self.negatives), device=ids.device)
            candidates = torch.cat((positive, negative), dim=-1)
            args = (*context_tensors, ids, r, p, candidates)
            if self.training and torch.is_grad_enabled():
                value = checkpoint(self._chunk_loss, *args, use_reentrant=False)
            else:
                value = self._chunk_loss(*args)
            loss = loss + value
        return loss / len(rows)
