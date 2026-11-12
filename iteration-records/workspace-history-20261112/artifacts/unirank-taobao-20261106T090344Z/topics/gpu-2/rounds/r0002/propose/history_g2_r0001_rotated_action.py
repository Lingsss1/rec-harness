"""Feedback-preserving compression with target/action local activation.

Prospective 2x2 study: true versus occupancy-matched rotated auxiliary-feedback
groups, crossed with action-conditioned versus item-only activation. This is
an adaptation of the supplied DIN reference, not a novelty or benefit claim.
Only released historical task masks and historical action embeddings are used.
"""
import torch
from torch import nn
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbeddingDict
from unirank.pytorch.layers.activations import Dice


CONFIG = {
    'embedding_dim': 16,
    'attention_hidden_units': [64, 32],
    'tower_hidden_units': [128, 64],
    'attention_normalization': 'none',
    'routing_mode': 'rotated',
    'action_conditioning': True,
}


class Model(MultiTaskModel):
    def __init__(self, feature_map, model_id='FeedbackBlockDIN', gpu=-1,
                 embedding_dim=16, attention_hidden_units=(64, 32),
                 tower_hidden_units=(128, 64), attention_normalization='none',
                 routing_mode='rotated', action_conditioning=True,
                 num_tasks=None, task='binary_classification', **kwargs):
        if num_tasks is not None and num_tasks != len(feature_map.labels):
            raise ValueError('Task count mismatch')
        if set(feature_map.labels) != {'is_click', 'cart', 'fav', 'buy'}:
            raise ValueError('This study requires the frozen four-task schema')
        if attention_normalization != 'none':
            raise ValueError('This study preserves unnormalized DIN activation')
        if routing_mode not in ('feedback', 'rotated'):
            raise ValueError('Unknown routing mode')
        if not isinstance(action_conditioning, bool):
            raise ValueError('action_conditioning must be boolean')
        super().__init__(feature_map, model_id=model_id, gpu=gpu,
                         num_tasks=len(feature_map.labels), task=task, **kwargs)
        self.routing_mode = routing_mode
        self.action_conditioning = action_conditioning
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        self.item_fields = [name for name, spec in feature_map.features.items()
                            if spec.get('source') == 'item' and spec['type'] != 'meta']
        self.user_fields = [name for name, spec in feature_map.features.items()
                            if spec.get('source') not in ('item', 'action')
                            and spec['type'] != 'meta']
        self.action_field = next((name for name, spec in feature_map.features.items()
                                  if spec.get('source') == 'action'), None)
        if self.action_field is None:
            raise ValueError('Historical action field is required')
        self.auxiliary_indices = [list(feature_map.labels).index(label)
                                  for label in ('cart', 'fav', 'buy')]
        self.key_width = len(self.item_fields) * embedding_dim
        self.value_width = self.key_width + embedding_dim

        # Identical parameter shapes and initialization order in all four cells.
        # Item-only controls zero the added input, keeping action in the values.
        layers = []
        width = 4 * self.key_width + embedding_dim
        for hidden in attention_hidden_units:
            layers.extend([nn.Linear(width, hidden), Dice(hidden)])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.local_activation = nn.Sequential(*layers)
        input_width = len(self.user_fields) * embedding_dim + self.key_width + self.value_width
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
        self.model_to_device()

    def _represent(self, inputs):
        # Reject missing masks: the framework's fallback would silently turn
        # every historical event into a positive event for every task.
        if len(inputs) != 4:
            raise ValueError('Released historical task masks are required')
        supplied = inputs[3]
        if isinstance(supplied, (list, tuple)):
            if len(supplied) != self.num_tasks:
                raise ValueError('Wrong number of historical task masks')
        elif torch.is_tensor(supplied):
            if supplied.ndim != 3 or supplied.shape[0] != self.num_tasks:
                raise ValueError('Expected historical masks with shape [4,B,100]')
        else:
            raise ValueError('Released historical task masks are required')
        user, item, mask, task_masks = self.get_inputs(inputs, return_multi_masks=True)
        if mask.ndim != 2 or mask.shape[1] != 100:
            raise ValueError('Frozen protocol requires 100 history positions')
        auxiliary = torch.zeros_like(mask, dtype=torch.bool)
        for index in self.auxiliary_indices:
            if task_masks[index].shape != mask.shape:
                raise ValueError('Historical task mask shape mismatch')
            auxiliary = auxiliary | (task_masks[index] > 0)
        auxiliary = auxiliary & mask.bool()
        user_emb = self.embedding_layer(user)
        item_emb = self.embedding_layer(item)
        target = torch.cat([item_emb[name][:, -1] for name in self.item_fields], dim=-1)
        keys = torch.cat([item_emb[name][:, :-1] for name in self.item_fields], dim=-1)
        # The unknown current action never enters the scorer or representation.
        values = torch.cat([keys, item_emb[self.action_field][:, :-1]], dim=-1)
        context = torch.cat([user_emb[name] for name in self.user_fields], dim=-1)
        return context, target, values, mask, auxiliary

    def _compress(self, values, mask, auxiliary):
        batch, length, width = values.shape
        if length != 100:
            raise ValueError('Frozen protocol requires 100 history positions')
        groups, block_size = 10, 10
        valid = mask.bool().reshape(batch, groups, block_size)
        membership = auxiliary.reshape(batch, groups, block_size) & valid
        if self.routing_mode == 'rotated':
            # Released input is pre-padded. Each partial block has a valid
            # suffix. Rotate labels among those valid positions, never padding;
            # this preserves both stratum counts exactly for every block.
            n = valid.long().sum(-1, keepdim=True)
            rank = valid.long().cumsum(-1) - 1
            source = block_size - n + (rank + n // 2).remainder(n.clamp_min(1))
            membership = membership.gather(-1, source.clamp(0, block_size - 1)) & valid
        assignment = torch.stack((valid & ~membership, membership), dim=-2).to(values.dtype)
        assignment = assignment.reshape(batch * groups, 2, block_size)
        counts = assignment.sum(-1)
        grouped = values.reshape(batch * groups, block_size, width)
        sums = torch.bmm(assignment, grouped)
        means = sums / counts.clamp_min(1).unsqueeze(-1)
        return means.reshape(batch, groups * 2, width), counts.reshape(batch, groups * 2)

    def forward(self, inputs):
        context, target, values, mask, auxiliary = self._represent(inputs)
        values, counts = self._compress(values, mask, auxiliary)
        keys = values[:, :, :self.key_width]
        action = values[:, :, self.key_width:]
        if not self.action_conditioning:
            action = torch.zeros_like(action)
        batch, slots, _ = keys.shape
        query = target[:, None, :].expand(-1, slots, -1)
        interactions = torch.cat([query, keys, query - keys, query * keys, action], dim=-1)
        weights = self.local_activation(interactions.reshape(batch * slots, -1)).reshape(batch, slots)
        weights = weights * counts.to(weights.dtype)
        pooled = torch.bmm(weights[:, None, :], values).squeeze(1)
        features = torch.cat([context, target, pooled], dim=-1)
        return {label + '_pred': torch.sigmoid(tower(features))
                for label, tower in zip(self.feature_map.labels, self.towers)}
