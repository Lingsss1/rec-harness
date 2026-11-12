"""Prospective field/feedback summary study, not an evaluated model.

Shares the released field embeddings with the DIN adaptation. Target/context
queries retrieve either each item field or the entire event with shared weights.
Five summary slots (all exposures and four overlapping feedback groups) join
static field tokens and four learned task tokens. The two switches form a 2x2
replacement study with identical trainable tensors. No new data are accessed.
Nearest inspected implementations: pinned UniRank HiFormer, EST, TokenFormer.
This is an adaptation; originality and effectiveness remain unverified.
"""
import math
import torch
from torch import nn
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbeddingDict
from unirank.pytorch.layers.activations import Dice

CONFIG = {'embedding_dim': 16, 'token_dim': 32, 'num_heads': 2,
          'num_layers': 2, 'field_retrieval': True, 'feedback_groups': False,
          'tower_hidden_units': [128, 64]}


class Interaction(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.width = width
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                 nn.Linear(2 * width, width))

    def forward(self, tokens):
        b, n, d = tokens.shape
        qkv = self.qkv(self.norm1(tokens)).reshape(
            b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d // self.heads)
        attention = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        mixed = torch.matmul(attention, v).transpose(1, 2).reshape(b, n, d)
        tokens = tokens + self.out(mixed)
        return tokens + self.ffn(self.norm2(tokens))


class Model(MultiTaskModel):
    def __init__(self, feature_map, model_id='FieldFeedbackSummary', gpu=-1,
                 embedding_dim=16, token_dim=32, num_heads=2, num_layers=2,
                 field_retrieval=True, feedback_groups=False,
                 tower_hidden_units=(128, 64), num_tasks=None,
                 task='binary_classification', **kwargs):
        if len(feature_map.labels) != 4 or num_tasks not in (None, 4):
            raise ValueError('This study requires the frozen four tasks')
        if token_dim % num_heads:
            raise ValueError('token_dim must be divisible by num_heads')
        super().__init__(feature_map, model_id=model_id, gpu=gpu,
                         num_tasks=4, task=task, **kwargs)
        self.field_retrieval = field_retrieval
        self.feedback_groups = feedback_groups
        self.item_fields = [name for name, spec in feature_map.features.items()
                            if spec.get('source') == 'item' and spec['type'] != 'meta']
        self.user_fields = [name for name, spec in feature_map.features.items()
                            if spec.get('source') not in ['item', 'action']
                            and spec['type'] != 'meta' and name not in feature_map.labels]
        self.action_field = next(name for name, spec in feature_map.features.items()
                                 if spec.get('source') == 'action' and spec['type'] != 'meta')
        fields = self.user_fields + self.item_fields + [self.action_field]
        if any(feature_map.features[name].get('embedding_dim', embedding_dim)
               != embedding_dim for name in fields):
            raise ValueError('Study requires uniform field embedding dimensions')
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        f = len(self.item_fields)
        self.query = nn.ModuleList([nn.Linear(2 * embedding_dim, embedding_dim)
                                    for _ in range(f)])
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        static_count = len(self.user_fields) + f
        self.static_proj = nn.ModuleList([nn.Linear(embedding_dim, token_dim)
                                         for _ in range(static_count)])
        self.summary_proj = nn.Linear((f + 1) * embedding_dim, token_dim)
        self.count_proj = nn.Linear(1, token_dim, bias=False)
        self.identities = nn.Parameter(torch.empty(1, static_count + 5 + 4, token_dim))
        self.blocks = nn.ModuleList([Interaction(token_dim, num_heads)
                                     for _ in range(num_layers)])
        # All variants retain raw static fields, all five summaries and counts.
        # This direct path also keeps empty-history prediction well defined.
        input_width = static_count * embedding_dim + 6 * token_dim + 5
        self.towers = nn.ModuleList()
        for _ in feature_map.labels:
            layers = []
            width = input_width
            for hidden in tower_hidden_units:
                layers.extend([nn.Linear(width, hidden), Dice(hidden)])
                width = hidden
            layers.append(nn.Linear(width, 1))
            self.towers.append(nn.Sequential(*layers))
        self.reset_parameters()
        nn.init.normal_(self.identities, std=0.02)
        self.model_to_device()

    @staticmethod
    def _masked_weights(scores, mask):
        # Finite sentinel, then explicit renormalization: an empty group gives
        # exactly zero, including in BF16. Masked positions never contribute.
        valid = mask[:, :, None, :].bool()
        weights = torch.softmax(scores[:, None].float().masked_fill(~valid, -1e9), dim=-1)
        weights = weights * valid.to(weights.dtype)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    def forward(self, inputs):
        if len(inputs) != 4 or inputs[3] is None:
            raise ValueError('Frozen task-specific historical masks are required')
        user, item, mask, multi_masks = self.get_inputs(inputs, return_multi_masks=True)
        if len(multi_masks) != 4 or any(m.shape != mask.shape for m in multi_masks):
            raise ValueError('Expected four historical masks in label order')
        ue = self.embedding_layer(user)
        ie = self.embedding_layer(item)
        targets = [ie[name][:, -1] for name in self.item_fields]
        static = [ue[name] for name in self.user_fields] + targets
        context = torch.stack([ue[name] for name in self.user_fields], dim=1).mean(dim=1)
        history = torch.stack([ie[name][:, :-1] for name in self.item_fields], dim=2)
        action = ie[self.action_field][:, :-1]
        b, length, f, d = history.shape
        # Intersect every host feedback mask with exposure visibility. Groups
        # overlap for composite action tokens; they are not disjoint outcomes.
        groups = torch.stack([mask] + [m * mask for m in multi_masks], dim=1).float()
        counts = groups.sum(dim=-1)
        count_features = torch.log1p(counts) / math.log(101.0)
        present = counts.gt(0).to(history.dtype)
        if length:
            query = torch.stack([torch.tanh(layer(torch.cat([target, context], dim=-1)))
                                 for layer, target in zip(self.query, targets)], dim=1)
            keys = self.key(history)
            scores = (keys * query[:, None]).sum(dim=-1).transpose(1, 2) / math.sqrt(d)
            if not self.field_retrieval:
                # One event score shared by every field, using the same queries.
                scores = scores.mean(dim=1, keepdim=True).expand(-1, f, -1)
            retrieval_masks = groups if self.feedback_groups else mask[:, None].expand(-1, 5, -1)
            weights = self._masked_weights(scores, retrieval_masks).to(history.dtype)
            pooled_fields = torch.einsum('bgfl,blfd->bgfd', weights, history)
            pooled_action = torch.bmm(weights.mean(dim=2), action)
            summaries = torch.cat([pooled_fields.flatten(start_dim=2), pooled_action], dim=-1)
        else:
            summaries = history.new_zeros((b, 5, (f + 1) * d))
        # The ungrouped controls retain identical group counts and absence flags;
        # only the contents retrieved for a nonempty group change.
        summaries = summaries * present[:, :, None]
        history_tokens = self.summary_proj(summaries) * present[:, :, None]
        history_tokens = history_tokens + self.count_proj(count_features[:, :, None].to(history_tokens.dtype))
        static_tokens = torch.stack([proj(value) for proj, value in zip(self.static_proj, static)], dim=1)
        task_tokens = static_tokens.new_zeros((b, 4, static_tokens.shape[-1]))
        tokens = torch.cat([static_tokens, history_tokens, task_tokens], dim=1) + self.identities
        for block in self.blocks:
            tokens = block(tokens)
        common = torch.cat([torch.cat(static, dim=-1), history_tokens.flatten(start_dim=1),
                            count_features.to(history_tokens.dtype)], dim=-1)
        return {label + '_pred': torch.sigmoid(tower(torch.cat([common, tokens[:, -4 + i]], dim=-1)))
                for i, (label, tower) in enumerate(zip(self.feature_map.labels, self.towers))}
