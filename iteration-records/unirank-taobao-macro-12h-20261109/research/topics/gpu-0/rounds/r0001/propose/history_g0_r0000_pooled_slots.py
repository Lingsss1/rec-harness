"""Controlled history representation study under the frozen host protocol.

DIN local activation is inherited from baseline_reference.py. Field tokens and
five history slots interact in one small attention block. Only slot construction
changes across behavior, counts, pooled, and recency conditions. This is an
experimental adaptation, not a reproduction or originality claim.
"""
import math
import torch
from torch import nn
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbeddingDict
from unirank.pytorch.layers.activations import Dice


CONFIG = {
    'embedding_dim': 16,
    'attention_hidden_units': [64, 32],
    'tower_hidden_units': [128, 64],
    'token_dim': 32,
    'num_heads': 2,
    'slot_mode': 'pooled',
}


class JointBlock(nn.Module):
    """Pre-norm attention with invalid slots excluded as keys and outputs."""

    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 2 * width), nn.SiLU(),
                                 nn.Linear(2 * width, width))

    def forward(self, tokens, valid):
        batch, count, width = tokens.shape
        qkv = self.qkv(self.norm1(tokens)).reshape(
            batch, count, 3, self.heads, self.head_width)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # FP32 scores/softmax avoid overflow under the frozen BF16 autocast.
        # Explicitly disable autocast: .float() alone does not protect matmul.
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            scores = torch.matmul(query.float(), key.float().transpose(-1, -2))
            scores = scores / math.sqrt(self.head_width)
            scores = scores.masked_fill(~valid[:, None, None, :], float('-inf'))
            attention = torch.softmax(scores, dim=-1)
        mixed = torch.matmul(attention.to(value.dtype), value)
        mixed = mixed.transpose(1, 2).reshape(batch, count, width)
        tokens = tokens + self.output(mixed)
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens * valid.unsqueeze(-1).to(tokens.dtype)


class Model(MultiTaskModel):
    def __init__(self, feature_map, model_id='BehaviorSlotStudy', gpu=-1,
                 embedding_dim=16, attention_hidden_units=(64, 32),
                 tower_hidden_units=(128, 64), token_dim=32, num_heads=2,
                 slot_mode='pooled', num_tasks=None,
                 task='binary_classification', **kwargs):
        if num_tasks is not None and num_tasks != len(feature_map.labels):
            raise ValueError('Task count mismatch')
        if list(feature_map.labels) != ['is_click', 'cart', 'fav', 'buy']:
            raise ValueError('This study requires the frozen four-task order')
        if slot_mode not in ('behavior', 'counts', 'pooled', 'recency'):
            raise ValueError('Unknown slot construction')
        if token_dim <= 0 or num_heads <= 0 or token_dim % num_heads:
            raise ValueError('token_dim must be divisible by num_heads')
        super().__init__(feature_map, model_id=model_id, gpu=gpu,
                         num_tasks=len(feature_map.labels), task=task, **kwargs)
        self.slot_mode = slot_mode
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        self.item_fields = [name for name, spec in feature_map.features.items()
                            if spec.get('source') == 'item' and spec['type'] != 'meta']
        self.user_fields = [name for name, spec in feature_map.features.items()
                            if spec.get('source') not in ('item', 'action')
                            and spec['type'] != 'meta']
        self.action_field = next((name for name, spec in feature_map.features.items()
                                  if spec.get('source') == 'action'), None)
        if not self.item_fields or not self.user_fields or self.action_field is None:
            raise ValueError('Required released feature groups are missing')
        self.key_width = len(self.item_fields) * embedding_dim
        self.value_width = self.key_width + embedding_dim
        self.static_count = len(self.user_fields) + len(self.item_fields)
        self.token_count = self.static_count + 5
        layers = []
        width = 4 * self.key_width
        for hidden in attention_hidden_units:
            layers.extend([nn.Linear(width, hidden), Dice(hidden)])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.local_activation = nn.Sequential(*layers)
        self.field_projection = nn.Linear(embedding_dim, token_dim)
        self.slot_projection = nn.Linear(self.value_width + 1, token_dim)
        self.token_identity = nn.Parameter(torch.empty(1, self.token_count, token_dim))
        self.interaction = JointBlock(token_dim, num_heads)
        # Retain raw static fields and the original global DIN sum in all arms.
        input_width = (self.static_count * embedding_dim + self.value_width
                       + self.token_count * token_dim + 5)
        self.towers = nn.ModuleList()
        for _ in feature_map.labels:
            tower = []
            width = input_width
            for hidden in tower_hidden_units:
                tower.extend([nn.Linear(width, hidden), Dice(hidden)])
                width = hidden
            tower.append(nn.Linear(width, 1))
            self.towers.append(nn.Sequential(*tower))
        self.reset_parameters()
        nn.init.normal_(self.token_identity, std=0.02)
        self.model_to_device()

    def _slot_masks(self, valid, task_masks):
        """Return content masks and counts, intersecting visible history."""
        all_events = valid[:, None, :]
        behavior = torch.stack([m.bool() & valid for m in task_masks], dim=1)
        behavior = torch.cat([all_events, behavior], dim=1)
        if self.slot_mode == 'behavior':
            content = behavior
            counts = behavior.sum(dim=-1)
        elif self.slot_mode == 'counts':
            # Keep semantic counts/occupancy but remove semantic assignment:
            # use global content, scaled to the corresponding group's count.
            content = all_events.expand(-1, 5, -1)
            counts = behavior.sum(dim=-1)
        elif self.slot_mode == 'pooled':
            content = all_events.expand(-1, 5, -1)
            counts = content.sum(dim=-1)
        else:
            # Four equal-rank bins in the visible prefix, oldest to newest.
            ranks = valid.long().cumsum(dim=-1) - 1
            length = valid.long().sum(dim=-1, keepdim=True).clamp_min(1)
            bins = torch.div(4 * ranks, length, rounding_mode='floor').clamp(0, 3)
            parts = [(bins == index) & valid for index in range(4)]
            content = torch.cat([all_events, torch.stack(parts, dim=1)], dim=1)
            counts = content.sum(dim=-1)
        return content, counts

    def forward(self, inputs):
        if len(inputs) != 4 or inputs[3] is None:
            raise ValueError('Frozen historical task masks are required')
        user, item, mask, task_masks = self.get_inputs(inputs, return_multi_masks=True)
        if len(task_masks) != 4:
            raise ValueError('Expected four historical task masks')
        valid = mask.bool()
        batch, length = valid.shape
        # Sanitize padded IDs before embedding and before batch-coupled Dice.
        # Exclude the current action even if an upstream caller changes it.
        clean_item = {}
        for name in self.item_fields + [self.action_field]:
            history = item[name][:, :-1].masked_fill(~valid, 0)
            target = (torch.zeros_like(item[name][:, -1:])
                      if name == self.action_field else item[name][:, -1:])
            clean_item[name] = torch.cat([history, target], dim=1)
        user_emb = self.embedding_layer(user)
        item_emb = self.embedding_layer(clean_item)
        static = torch.stack([user_emb[name] for name in self.user_fields]
                             + [item_emb[name][:, -1] for name in self.item_fields], dim=1)
        target = torch.cat([item_emb[name][:, -1] for name in self.item_fields], dim=-1)
        keys = torch.cat([item_emb[name][:, :-1] for name in self.item_fields], dim=-1)
        keys = keys * valid.unsqueeze(-1).to(keys.dtype)
        action = item_emb[self.action_field][:, :-1]
        values = torch.cat([keys, action], dim=-1)
        values = values * valid.unsqueeze(-1).to(values.dtype)
        content_mask, counts = self._slot_masks(valid, task_masks)
        if length:
            query = target[:, None, :].expand(-1, length, -1)
            interactions = torch.cat([query, keys, query - keys, query * keys], dim=-1)
            weights = self.local_activation(interactions.reshape(batch * length, -1))
            weights = weights.reshape(batch, length) * valid.to(weights.dtype)
            weighted_masks = content_mask.to(weights.dtype) * weights[:, None, :]
            pooled = torch.bmm(weighted_masks, values)
        else:
            pooled = target.new_zeros((batch, 5, self.value_width))
        if self.slot_mode == 'counts':
            proportions = counts.float() / counts[:, :1].clamp_min(1).float()
            pooled = pooled * proportions.unsqueeze(-1).to(pooled.dtype)
        slot_valid = counts > 0
        pooled = pooled * slot_valid.unsqueeze(-1).to(pooled.dtype)
        log_counts = torch.log1p(counts.float()) / math.log(101.0)
        log_counts = log_counts.to(pooled.dtype)
        slot_tokens = self.slot_projection(torch.cat([pooled, log_counts.unsqueeze(-1)], dim=-1))
        tokens = torch.cat([self.field_projection(static), slot_tokens], dim=1)
        tokens = tokens + self.token_identity.to(tokens.dtype)
        static_valid = torch.ones((batch, self.static_count), dtype=torch.bool, device=valid.device)
        token_valid = torch.cat([static_valid, slot_valid], dim=1)
        tokens = tokens * token_valid.unsqueeze(-1).to(tokens.dtype)
        tokens = self.interaction(tokens, token_valid)
        features = torch.cat([static.flatten(1), pooled[:, 0], tokens.flatten(1), log_counts], dim=-1)
        return {label + '_pred': torch.sigmoid(tower(features))
                for label, tower in zip(self.feature_map.labels, self.towers)}
