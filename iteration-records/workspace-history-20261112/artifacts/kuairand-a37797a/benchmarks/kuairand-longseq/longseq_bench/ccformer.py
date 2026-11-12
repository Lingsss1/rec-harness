# Copyright 2026 longseq-kuairand-bench contributors.
# SPDX-License-Identifier: Apache-2.0
"""Independent causal ID-only adaptation of CCFormer, arXiv:2607.28070v1.

The paper's CTR fields become two history-derived user tokens, history ID
tokens, and one token per candidate. Causal subspace projections and left-padded
strided convolutions allow shared, leak-free all-position supervision. These
are task adaptations, not an author-code or industrial reproduction.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        scale = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return (x.float() * scale).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim, 2 * dim, bias=False)
        self.value = nn.Linear(dim, 2 * dim, bias=False)
        self.output = nn.Linear(2 * dim, dim, bias=False)

    def forward(self, x):
        return self.output(F.silu(self.gate(x)) * self.value(x))


class CausalSubspaceMixer(nn.Module):
    """Eqs. 9--15 with triangular temporal and token-projection masks."""
    def __init__(self, dim, group, channels, expansion, time_unit_ms):
        super().__init__()
        self.group, self.channels, self.time_unit_ms = group, channels, time_unit_ms
        width, hidden = group * channels, group * channels * expansion
        count = dim // channels
        self.gate = nn.Parameter(torch.empty(count, width, hidden))
        self.value = nn.Parameter(torch.empty(count, width, hidden))
        self.output = nn.Parameter(torch.empty(count, hidden, width))
        for weight in (self.gate, self.value, self.output):
            nn.init.normal_(weight, std=1 / math.sqrt(weight.shape[-2]))
        tokens = torch.arange(width) // channels
        hidden_tokens = torch.arange(hidden) // (channels * expansion)
        self.register_buffer("input_mask", tokens[:, None] <= hidden_tokens[None, :], persistent=False)
        self.register_buffer("output_mask", hidden_tokens[:, None] <= tokens[None, :], persistent=False)
        self.register_buffer("local_causal", torch.ones(group, group, dtype=torch.bool).tril(), persistent=False)
        self.position = nn.Parameter(torch.zeros(2 * group - 1))
        self.raw_alpha = nn.Parameter(torch.tensor(0.0))
        self.raw_decay = nn.Parameter(torch.tensor(0.0))  # beta = exp(-softplus(raw_decay))
        self.raw_gamma = nn.Parameter(torch.tensor(0.0))
        self.norm = RMSNorm(dim)

    def forward(self, x, times, valid):
        b, length, dim = x.shape
        m, n = self.group, self.channels
        padding = (-length) % m
        values = F.pad(x * valid[..., None], (0, 0, 0, padding)).reshape(b, -1, m, dim)
        grouped_times = F.pad(times[:, :length], (0, padding)).reshape(b, -1, m).double()
        delta = (grouped_times[..., :, None] - grouped_times[..., None, :]).abs().float() / self.time_unit_ms
        # Stable beta ** (delta ** gamma), including identical timestamps. The
        # exponent clamp only limits already saturated decay and avoids NaN grads.
        power = torch.exp((F.softplus(self.raw_gamma) * delta.clamp_min(1e-6).log()).clamp(-30, 30))
        power = torch.where(delta == 0, 0.0, power)
        temporal = F.softplus(self.raw_alpha) * torch.exp(-F.softplus(self.raw_decay) * power)
        p = torch.arange(m, device=x.device)
        relative = self.position[p[:, None] - p[None, :] + m - 1]
        weights = (temporal + relative) * self.local_causal
        mixed = torch.einsum("bgij,bgjd->bgid", weights.to(values.dtype), values)
        mixed = self.norm(mixed)
        groups = dim // n
        subspaces = mixed.reshape(b, -1, m, groups, n).permute(0, 1, 3, 2, 4).flatten(-2)
        gate = torch.einsum("bgci,cih->bgch", subspaces, self.gate * self.input_mask)
        value = torch.einsum("bgci,cih->bgch", subspaces, self.value * self.input_mask)
        result = torch.einsum("bgch,chi->bgci", F.silu(gate) * value, self.output * self.output_mask)
        result = result.reshape(b, -1, groups, m, n).permute(0, 1, 3, 2, 4).reshape(b, -1, dim)
        return result[:, :length] * valid[..., None]


class CCFormerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        self.user_norm, self.target_norm, self.sequence_norm = (RMSNorm(d) for _ in range(3))
        self.user_ffn, self.target_ffn = SwiGLU(d), SwiGLU(d)
        self.user_qkv = nn.Linear(d, 3 * d, bias=False)
        self.sequence_kv = nn.Linear(d, 2 * d, bias=False)
        self.target_to_sequence = nn.Linear(d, d, bias=False)
        self.target_to_user = nn.Linear(d, d, bias=False)
        self.user_output = nn.Linear(d, d, bias=False)
        self.target_output = nn.Linear(2 * d, d, bias=False)
        self.mixer = CausalSubspaceMixer(d, cfg.cc_group_size, cfg.cc_channel_size,
                                         cfg.cc_mixing_expansion, cfg.cc_time_unit_ms)
        self.compression = nn.Conv1d(d, d, kernel_size=cfg.cc_kernel_size,
                                     stride=cfg.cc_stride, bias=False)

    def drop(self, x):
        return F.dropout(x, self.cfg.linear_dropout, self.training)

    def attend(self, q, k, v, allowed=None):
        b, queries, d = q.shape
        h = self.cfg.heads
        q, k, v = (value.reshape(b, -1, h, d // h).transpose(1, 2) for value in (q, k, v))
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(d // h)
        if allowed is not None:
            logits = logits.masked_fill(~allowed[:, None], -torch.inf)
        weights = torch.softmax(logits.float(), dim=-1).to(v.dtype)
        return (weights @ v).transpose(1, 2).reshape(b, queries, d)

    def history(self, x, users, times, valid, anchors, prefix_positions, prefix_valid):
        b, prefixes, user_tokens, d = users.shape
        bar_users = users + self.drop(self.user_ffn(self.user_norm(users)))
        uq, uk, uv = self.user_qkv(self.user_norm(bar_users)).chunk(3, -1)
        sk, sv = self.sequence_kv(self.sequence_norm(x)).chunk(2, -1)
        pieces = []
        for left in range(0, prefixes, self.cfg.query_chunk):
            right = min(left + self.cfg.query_chunk, prefixes)
            available = (anchors[None, None] <= prefix_positions[None, left:right, None]) & valid[:, None]
            available = available[:, :, None].expand(-1, -1, user_tokens, -1).flatten(1, 2)
            q = uq[:, left:right].reshape(b, -1, d)
            pieces.append(self.attend(q, sk, sv, available).reshape(b, right - left, user_tokens, d))
        next_users = (bar_users + self.drop(self.user_output(torch.cat(pieces, dim=1)))) * prefix_valid[:, :, None, None]
        updated = (x + self.drop(self.mixer(x, times, valid))) * valid[..., None]
        compressed = self.compression(F.pad(updated.transpose(1, 2), (self.cfg.cc_kernel_size - 1, 0))).transpose(1, 2)
        compressed = compressed * valid[:, ::self.cfg.cc_stride, None]
        return compressed, next_users, sk, sv, uk, uv

    def target(self, target, sk, sv, uk, uv, available):
        bar_target = target + self.drop(self.target_ffn(self.target_norm(target)))
        from_history = self.attend(self.target_to_sequence(bar_target), sk, sv,
                                   available[:, None].expand(-1, target.shape[1], -1))
        from_users = self.attend(self.target_to_user(bar_target), uk, uv)
        return bar_target + self.drop(self.target_output(torch.cat((from_history, from_users), dim=-1)))


class CCFormerEncoder(nn.Module):
    """Candidate-independent history/user cache; target-conditioned exact scores."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.item_embedding = nn.Embedding(cfg.num_items + 1, cfg.dim, padding_idx=0)
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        self.user_seed = nn.Parameter(torch.empty(2, cfg.dim).normal_(std=0.02))
        self.layers = nn.ModuleList([CCFormerBlock(cfg) for _ in range(cfg.blocks)])
        self.head = nn.Sequential(nn.Linear(4 * cfg.dim, 2 * cfg.dim), nn.SiLU(), nn.Linear(2 * cfg.dim, 1))

    def run_history(self, layer, *inputs):
        if self.training and self.cfg.checkpoint_blocks and torch.is_grad_enabled():
            return checkpoint(layer.history, *inputs, use_reentrant=False)
        return layer.history(*inputs)

    def encode_all(self, ids, times):
        b, length = ids.shape
        valid = ids.ne(0)
        x = F.dropout(self.item_embedding(ids) * math.sqrt(self.cfg.dim), self.cfg.input_dropout, self.training)
        x = x * valid[..., None]
        means = x.cumsum(1) / valid.cumsum(1).clamp_min(1)[..., None]
        users = (torch.stack((means, x), dim=2) + self.user_seed) * valid[:, :, None, None]
        prefixes = torch.arange(length, device=ids.device)
        anchors = prefixes
        history_times = times[:, :length]
        contexts = []
        for layer in self.layers:
            x, users, sk, sv, uk, uv = self.run_history(layer, x, users, history_times, valid,
                                                      anchors, prefixes, ids.ne(0))
            contexts.extend((sk, sv, uk, uv))
            anchors = anchors[::self.cfg.cc_stride]
            valid = valid[:, ::self.cfg.cc_stride]
            history_times = history_times[:, ::self.cfg.cc_stride]
        means = x.cumsum(1) / valid.cumsum(1).clamp_min(1)[..., None]
        last_anchor = (torch.searchsorted(anchors.contiguous(), prefixes, right=True) - 1).clamp_min(0)
        contexts.extend((F.normalize(users.flatten(2), dim=-1, eps=1e-6),
                         F.normalize(means[:, last_anchor], dim=-1, eps=1e-6)))
        return tuple(contexts)

    def score_queries(self, context, ids, rows, positions, candidates):
        target = self.item_embedding(candidates) * math.sqrt(self.cfg.dim)
        for index, layer in enumerate(self.layers):
            sk, sv, uk, uv = context[4 * index:4 * index + 4]
            anchors = torch.arange(sk.shape[1], device=ids.device) * self.cfg.cc_stride ** index
            available = (anchors[None] <= positions[:, None]) & ids[rows[:, None], anchors[None]].ne(0)
            target = layer.target(target, sk[rows], sv[rows], uk[rows, positions], uv[rows, positions], available)
        users, sequence = context[-2:]
        shared = torch.cat((users[rows, positions], sequence[rows, positions]), dim=-1)
        features = torch.cat((shared[:, None].expand(-1, candidates.shape[1], -1),
                              F.normalize(target, dim=-1, eps=1e-6)), dim=-1)
        return self.head(features).squeeze(-1)
