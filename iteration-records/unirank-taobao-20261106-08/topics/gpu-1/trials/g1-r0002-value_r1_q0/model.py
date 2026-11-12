"""Prospective value-preservation / query-refresh factorial.

The accepted mass-exchange ancestor remains the comparison lineage. Both arms
read original history values with the second reader's attention, expose the same
learned rank-32 reconstruction, and instantiate identical learned modules. The
R treatment adds an identity residual around that rank bottleneck. The Q factor
is inherited query refresh. No outcome, priority, or pure-rank causal claim.
Reference provenance and falsifiers are recorded in proposal.json.
"""
import math
import torch
from torch import nn
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbeddingDict
from unirank.pytorch.layers.activations import Dice

CONFIG = {'embedding_dim': 16, 'token_dim': 32,
          'refresh_query': False, 'mass_scaling': True,
          'raw_value_residual': True}


class FieldMix(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.ffnorm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                nn.Linear(2 * width, width))

    def forward(self, fields):
        q, k, v = self.qkv(self.norm(fields)).chunk(3, dim=-1)
        weights = torch.softmax(torch.bmm(q.float(), k.float().transpose(1, 2))
                                / math.sqrt(q.shape[-1]), dim=-1).to(v.dtype)
        fields = fields + self.out(torch.bmm(weights, v))
        return fields + self.ff(self.ffnorm(fields))


class Read(nn.Module):
    def __init__(self, width, mass_scaling):
        super().__init__()
        self.mass_scaling = bool(mass_scaling)
        self.qnorm = nn.LayerNorm(width)
        self.enorm = nn.LayerNorm(width)
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(width, 2 * width)
        self.out = nn.Linear(width, width, bias=False)

    def forward(self, fields, events, mask):
        if events.shape[1] == 0:
            weights = fields.new_zeros((fields.shape[0], fields.shape[1], 0))
            return torch.zeros_like(fields), weights
        q = self.q(self.qnorm(fields))
        k, v = self.kv(self.enorm(events)).chunk(2, dim=-1)
        scores = torch.bmm(q.float(), k.float().transpose(1, 2)) / math.sqrt(q.shape[-1])
        valid = mask[:, None, :].bool()
        # Finite masked scores avoid NaNs for all-empty histories; explicitly
        # zero and renormalize after softmax so empty histories read exactly zero.
        weights = torch.softmax(scores.masked_fill(~valid, -1e9), dim=-1)
        weights = weights * valid.to(weights.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
        # Both branches compute identical counts. The treatment changes only
        # evidence magnitude, retaining relative event weights and strict masks.
        # sqrt(n) lies in [0,10] for the frozen 100-event prefix; n=0 stays zero.
        count = mask.float().sum(dim=-1, keepdim=True).unsqueeze(1)
        scale = count.sqrt() if self.mass_scaling else torch.ones_like(count)
        weights = weights * scale
        return self.out(torch.bmm(weights.to(v.dtype), v)), weights


class Write(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.local = nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                   nn.Linear(width, width))
        self.condition = nn.Linear(width, 2 * width)

    def forward(self, events, fields, mask):
        scale, shift = self.condition(fields.mean(dim=1)).chunk(2, dim=-1)
        delta = self.local(self.norm(events)) * torch.sigmoid(scale[:, None, :])
        delta = delta + torch.tanh(shift[:, None, :])
        return (events + delta) * mask[:, :, None].to(events.dtype)


class Model(MultiTaskModel):
    def __init__(self, feature_map, model_id='ValuePreservingExchange', gpu=-1,
                 embedding_dim=16, token_dim=32, refresh_query=False,
                 mass_scaling=True, raw_value_residual=True, num_tasks=None,
                 task='binary_classification', **kwargs):
        if num_tasks is not None and num_tasks != len(feature_map.labels):
            raise ValueError('Task count mismatch')
        super().__init__(feature_map, model_id=model_id, gpu=gpu,
                         num_tasks=len(feature_map.labels), task=task, **kwargs)
        self.refresh_query = bool(refresh_query)
        self.mass_scaling = bool(mass_scaling)
        self.raw_value_residual = bool(raw_value_residual)
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        self.item_fields = [n for n, s in feature_map.features.items()
                            if s.get('source') == 'item' and s['type'] != 'meta']
        self.user_fields = [n for n, s in feature_map.features.items()
                            if s.get('source') not in ['item', 'action'] and s['type'] != 'meta']
        self.action_field = next((n for n, s in feature_map.features.items()
                                  if s.get('source') == 'action'), None)
        self.field_count = len(self.item_fields) + len(self.user_fields)
        # Separate projections retain field identity without expanding sparse tables.
        self.field_projs = nn.ModuleList([nn.Linear(embedding_dim, token_dim)
                                          for _ in range(self.field_count)])
        event_width = (len(self.item_fields) + int(self.action_field is not None)) * embedding_dim
        # One relative position scalar derived only from the visible prefix order.
        self.event_proj = nn.Linear(event_width + 1, token_dim)
        self.mixers = nn.ModuleList([FieldMix(token_dim) for _ in range(2)])
        self.readers = nn.ModuleList([Read(token_dim, self.mass_scaling) for _ in range(2)])
        self.writer = Write(token_dim)
        self.value_width = event_width
        # Both factors instantiate these exact active, bias-free maps. The
        # reconstruction is linear with rank <= token_dim, and maps zero to zero.
        self.value_down = nn.Linear(event_width, token_dim, bias=False)
        self.value_up = nn.Linear(token_dim, event_width, bias=False)
        tower_width = self.field_count * (embedding_dim + token_dim) + 1 + event_width
        self.towers = nn.ModuleList([
            nn.Sequential(nn.Linear(tower_width, 128), Dice(128),
                          nn.Linear(128, 64), Dice(64), nn.Linear(64, 1))
            for _ in feature_map.labels])
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        user, item, mask = self.get_inputs(inputs)
        user_emb = self.embedding_layer(user)
        item_emb = self.embedding_layer(item)
        raw_fields = [user_emb[n] for n in self.user_fields]
        raw_fields += [item_emb[n][:, -1] for n in self.item_fields]
        fields = torch.stack([proj(x) for proj, x in zip(self.field_projs, raw_fields)], dim=1)
        history = [item_emb[n][:, :-1] for n in self.item_fields]
        if self.action_field is not None:
            history.append(item_emb[self.action_field][:, :-1])
        history = torch.cat(history, dim=-1)
        length = history.shape[1]
        # Prepadding positions are erased before any mixing or reduction.
        position = torch.arange(length, device=history.device, dtype=history.dtype)
        position = ((position + 1) / max(length, 1)).view(1, length, 1)
        position = position.expand(history.shape[0], -1, -1)
        events = self.event_proj(torch.cat([history, position], dim=-1))
        events = events * mask[:, :, None].to(events.dtype)
        first_query = self.mixers[0](fields)
        first_read, _ = self.readers[0](first_query, events, mask)
        first_fields = first_query + first_read
        # Keep the best prior write condition fixed in this follow-up. A changed
        # read magnitude can affect the subsequent broadcast as well as readout.
        events = self.writer(events, first_fields, mask)
        second_fields = self.mixers[1](first_fields)
        query = first_fields if self.refresh_query else first_query
        second_read, weights = self.readers[1](query, events, mask)
        final_fields = second_fields + second_read
        # Compact read of ORIGINAL embedding values, not the rewritten latent
        # events. Query/event exchange changes selection; this route preserves
        # content. Every cell computes the same pooled vector before treatment.
        if length:
            selected = weights.mean(dim=1, keepdim=True).to(history.dtype)
            masked_history = history * mask[:, :, None].to(history.dtype)
            summary = torch.bmm(selected, masked_history).squeeze(1)
        else:
            summary = history.new_zeros((history.shape[0], self.value_width))
        value_features = self.value_up(self.value_down(summary))
        if self.raw_value_residual:
            value_features = value_features + summary
        # No bias, normalization, or label-dependent operation can create a
        # nonzero value summary for an empty history. Same head width in R0/R1.
        # Identical scalar and active tower capacity in all four cells controls
        # for learning the history-length prior directly. This is visible-prefix
        # length, not full historical exposure count or a timestamp proxy.
        log_count = torch.log1p(mask.float().sum(dim=-1, keepdim=True)) / math.log(101.0)
        log_count = log_count.to(final_fields.dtype)
        features = torch.cat([torch.cat(raw_fields, dim=-1),
                              final_fields.flatten(1), log_count,
                              value_features.to(final_fields.dtype)], dim=-1)
        return {label + '_pred': torch.sigmoid(tower(features))
                for label, tower in zip(self.feature_map.labels, self.towers)}
