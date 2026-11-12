"""Wide historical values with evolving field/event routing.

Prospective 2x2: uncompressed versus learned low-rank historical values,
crossed with an enabled versus suppressed field-to-event write. This inherits
the local exchange implementation; it is not a new-method priority claim.
The final route uses evolved task queries and event keys, but its values are
only the original visible historical field/action embeddings. A value bypass
does not undo the compressed routing keys. Both value paths are computed in
all arms, with equal registered parameters and forward operation counts.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbeddingDict
from unirank.pytorch.layers.activations import Dice

CONFIG = {
    'embedding_dim': 16,
    'token_dim': 32,
    'num_heads': 2,
    'dynamic_query': True,
    'raw_values': False,
    'event_feedback': True,
    'tower_hidden_units': [128, 64],
}


class CrossAttention(nn.Module):
    """Pre-normalized attention; a zero null handles empty histories."""
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        # Bias-free projections preserve exactly zero empty-history evidence.
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, query, memory, key_mask=None):
        batch, queries, width = query.shape
        query = self.query_norm(query)
        memory = self.memory_norm(memory)
        attention_mask = None
        if key_mask is not None:
            valid = key_mask.bool()
            memory = torch.where(valid.unsqueeze(-1), memory,
                                 torch.zeros_like(memory))
            # Append after normalization so LN biases cannot affect the null.
            memory = torch.cat([memory, memory.new_zeros(batch, 1, width)], dim=1)
            valid = torch.cat([valid, ~valid.any(dim=1, keepdim=True)], dim=1)
            attention_mask = valid[:, None, None, :]
        length = memory.shape[1]
        q = self.query(query).reshape(batch, queries, self.heads,
                                      self.head_width).transpose(1, 2)
        k = self.key(memory).reshape(batch, length, self.heads,
                                    self.head_width).transpose(1, 2)
        v = self.value(memory).reshape(batch, length, self.heads,
                                      self.head_width).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        attended = attended.transpose(1, 2).contiguous().reshape(batch, queries, width)
        return self.output(attended)


    def pool_values(self, query, memory, values, key_mask):
        """Route wide values with the same learned Q/K as the narrow reads.

        Average attention heads before aggregation, so each original field
        keeps its own embedding coordinates. The null is valid only for an
        empty history; it makes all-padding and zero-length batches finite.
        The final read has only one query per task, not one per field.
        """
        batch, queries, width = query.shape
        valid = key_mask.bool()
        query = self.query_norm(query)
        memory = self.memory_norm(memory)
        memory = torch.where(valid.unsqueeze(-1), memory,
                             torch.zeros_like(memory))
        values = torch.where(valid.unsqueeze(-1), values,
                             torch.zeros_like(values))
        memory = torch.cat([memory, memory.new_zeros(batch, 1, width)], dim=1)
        values = torch.cat([values, values.new_zeros(batch, 1, values.shape[-1])], dim=1)
        valid = torch.cat([valid, ~valid.any(dim=1, keepdim=True)], dim=1)
        length = memory.shape[1]
        q = self.query(query).reshape(batch, queries, self.heads,
                                      self.head_width).transpose(1, 2)
        k = self.key(memory).reshape(batch, length, self.heads,
                                     self.head_width).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_width)
        logits = logits.masked_fill(~valid[:, None, None, :], float('-inf'))
        weights = torch.softmax(logits, dim=-1, dtype=torch.float32).mean(dim=1)
        return torch.bmm(weights.to(values.dtype), values)


class FieldMixer(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.attention = CrossAttention(width, heads)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                 nn.Linear(2 * width, width))

    def forward(self, fields):
        fields = fields + 0.5 * self.attention(fields, fields)
        return fields + 0.5 * self.ffn(self.ffn_norm(fields))


class Model(MultiTaskModel):
    def __init__(self, feature_map, model_id='WideValueExchange', gpu=-1,
                 embedding_dim=16, token_dim=32, num_heads=2,
                 dynamic_query=CONFIG['dynamic_query'],
                 raw_values=CONFIG['raw_values'],
                 event_feedback=CONFIG['event_feedback'],
                 tower_hidden_units=(128, 64), num_tasks=None,
                 task='binary_classification', **kwargs):
        labels = feature_map.labels
        if num_tasks is not None and num_tasks != len(labels):
            raise ValueError('Task count mismatch')
        if token_dim < 1 or num_heads < 1 or token_dim % num_heads:
            raise ValueError('token_dim must be divisible by num_heads')
        if not all(isinstance(flag, bool) for flag in (dynamic_query, raw_values, event_feedback)):
            raise ValueError('Path controls must be boolean')
        super().__init__(feature_map, model_id=model_id, gpu=gpu,
                         num_tasks=len(labels), task=task, **kwargs)
        self.dynamic_query = float(dynamic_query)
        self.raw_values = float(raw_values)
        self.event_feedback = float(event_feedback)
        self.token_dim = token_dim
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        self.item_fields = [name for name, spec in feature_map.features.items()
                            if spec.get('source') == 'item' and spec['type'] != 'meta']
        self.user_fields = [name for name, spec in feature_map.features.items()
                            if spec.get('source') not in ['item', 'action']
                            and spec['type'] != 'meta']
        self.action_field = next((name for name, spec in feature_map.features.items()
                                  if spec.get('source') == 'action'), None)
        self.field_count = len(self.user_fields) + len(self.item_fields)
        if not self.item_fields or self.field_count == 0:
            raise ValueError('Requires item and non-sequential feature fields')
        self.field_projection = nn.Linear(embedding_dim, token_dim, bias=False)
        event_width = (len(self.item_fields) + int(self.action_field is not None)) * embedding_dim
        self.event_projection = nn.Linear(event_width, token_dim, bias=False)
        # Independent value compression avoids tying this replacement to the
        # routing-key projection. These modules exist and run in every arm.
        self.value_down = nn.Linear(event_width, token_dim, bias=False)
        self.value_up = nn.Linear(token_dim, event_width, bias=False)
        # Equalize pooled-value scale without trainable normalization capacity.
        self.value_norm = nn.LayerNorm(event_width, elementwise_affine=False)
        self.field_identity = nn.Parameter(torch.empty(1, self.field_count + len(labels), token_dim))
        # Valid rank is derived only from the prefix mask, with no timestamps.
        self.history_position = nn.Parameter(torch.empty(101, token_dim))
        self.initial_mix = FieldMixer(token_dim, num_heads)
        # Tied read weights: static/static retrieves the same summary twice.
        self.read = CrossAttention(token_dim, num_heads)
        self.write = CrossAttention(token_dim, num_heads)
        self.after_read = nn.ModuleList([FieldMixer(token_dim, num_heads)
                                         for _ in range(2)])
        self.final_norm = nn.LayerNorm(token_dim)
        self.towers = nn.ModuleList()
        readout_width = self.field_count * embedding_dim + 2 * token_dim + 1 + event_width
        for _ in labels:
            tower = []
            width = readout_width
            for hidden in tower_hidden_units:
                tower.extend([nn.Linear(width, hidden), Dice(hidden)])
                width = hidden
            tower.append(nn.Linear(width, 1))
            self.towers.append(nn.Sequential(*tower))
        self.reset_parameters()
        nn.init.normal_(self.field_identity, std=0.02)
        nn.init.normal_(self.history_position, std=0.02)
        self.model_to_device()

    def forward(self, inputs):
        user, item, mask = self.get_inputs(inputs)
        user_emb = self.embedding_layer(user)
        item_emb = self.embedding_layer(item)
        batch, length = mask.shape
        if length > 100:
            raise ValueError('History exceeds the frozen 100-event contract')
        valid = mask.bool()
        raw_fields = [user_emb[name].reshape(batch, -1) for name in self.user_fields]
        raw_fields += [item_emb[name][:, -1] for name in self.item_fields]
        field_inputs = torch.stack(raw_fields, dim=1)
        raw_skip = field_inputs.flatten(start_dim=1)
        fields = self.field_projection(field_inputs)
        fields = torch.cat([fields, fields.new_zeros(batch, self.num_tasks, self.token_dim)], dim=1)
        fields = self.initial_mix(fields + self.field_identity)
        query_seed = fields

        history_parts = [item_emb[name][:, :-1] for name in self.item_fields]
        if self.action_field is not None:
            # The unknown current-target action is never used.
            history_parts.append(item_emb[self.action_field][:, :-1])
        event_inputs = torch.cat(history_parts, dim=-1)
        event_inputs = torch.where(valid.unsqueeze(-1), event_inputs,
                                   torch.zeros_like(event_inputs))
        rank = valid.long().cumsum(dim=1)
        events = self.event_projection(event_inputs) + self.history_position[rank]
        events = torch.where(valid.unsqueeze(-1), events, torch.zeros_like(events))

        for round_index, mixer in enumerate(self.after_read):
            # Scalar path interventions keep parameter shapes, initial weights,
            # module order and forward operation counts identical in all arms.
            query = query_seed + self.dynamic_query * (fields - query_seed)
            evidence = self.read(query, events, valid)
            fields = mixer(fields + 0.5 * evidence)
            if round_index == 0 and length > 0:
                delta = self.write(events, fields)
                events = events + 0.5 * self.event_feedback * delta
                events = torch.where(valid.unsqueeze(-1), events, torch.zeros_like(events))

        fields = self.final_norm(fields)
        task_states = fields[:, self.field_count:]
        wide_summary = self.read.pool_values(task_states, events, event_inputs, valid)
        # Linear compression commutes with weighted summation. This is the
        # same low-rank value replacement as encoding every event first,
        # implemented after pooling to avoid an extra sequence-wide matmul.
        reconstructed = self.value_up(self.value_down(wide_summary))
        wide_summary = self.value_norm(wide_summary)
        reconstructed = self.value_norm(reconstructed)
        summary = (self.raw_values * wide_summary
                   + (1.0 - self.raw_values) * reconstructed)
        # Keep empty-history evidence exactly zero, independently of biases.
        summary = torch.where(valid.any(dim=1)[:, None, None], summary,
                              torch.zeros_like(summary))
        global_state = fields[:, :self.field_count].mean(dim=1)
        count = torch.log1p(valid.sum(dim=1, keepdim=True).float()) / math.log(101.0)
        common = torch.cat([raw_skip, global_state, count.to(raw_skip.dtype)], dim=-1)
        return {label + '_pred': torch.sigmoid(tower(torch.cat([common, task_states[:, index], summary[:, index]], dim=-1)))
                for index, (label, tower) in enumerate(zip(self.feature_map.labels, self.towers))}
