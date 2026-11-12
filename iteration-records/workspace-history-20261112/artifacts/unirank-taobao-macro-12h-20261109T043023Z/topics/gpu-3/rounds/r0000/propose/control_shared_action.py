"""Prospective task/action relevance study; no local effectiveness claim.

DIN-style shared local activation trunk, four score channels and four towers.
The 2x2 controls change routing and action input, with identical module shapes.
Channel order always follows feature_map.labels. Only historical actions enter
the scorer or values; the current target action is never consumed.
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
    'task_specific': False,
    'action_in_score': True,
}


class Model(MultiTaskModel):
    def __init__(self, feature_map, model_id='TaskActionRelevance', gpu=-1,
                 embedding_dim=16, attention_hidden_units=(64, 32),
                 tower_hidden_units=(128, 64), attention_normalization='none',
                 task_specific=False, action_in_score=True, num_tasks=None,
                 task='binary_classification', **kwargs):
        labels = list(feature_map.labels)
        if len(labels) != 4 or set(labels) != {'is_click', 'cart', 'fav', 'buy'}:
            raise ValueError('Expected all four frozen feedback tasks')
        if num_tasks is not None and num_tasks != len(labels):
            raise ValueError('Task count mismatch')
        if attention_normalization != 'none':
            raise ValueError('This study fixes unnormalized DIN weights')
        super().__init__(feature_map, model_id=model_id, gpu=gpu,
                         num_tasks=len(labels), task=task, **kwargs)
        self.task_specific = bool(task_specific)
        self.action_in_score = bool(action_in_score)
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        self.item_fields = [name for name, spec in feature_map.features.items()
                            if spec.get('source') == 'item' and spec['type'] != 'meta']
        self.user_fields = [name for name, spec in feature_map.features.items()
                            if spec.get('source') not in ['item', 'action']
                            and spec['type'] != 'meta']
        action_fields = [name for name, spec in feature_map.features.items()
                         if spec.get('source') == 'action' and spec['type'] != 'meta']
        if not self.item_fields or len(action_fields) != 1:
            raise ValueError('Expected item fields and one historical action field')
        self.action_field = action_fields[0]
        self.key_width = len(self.item_fields) * embedding_dim
        self.value_width = self.key_width + embedding_dim

        # Every cell allocates the action columns and four score rows.
        layers = []
        width = 4 * self.key_width + embedding_dim
        for hidden in attention_hidden_units:
            layers.extend([nn.Linear(width, hidden), Dice(hidden)])
            width = hidden
        layers.append(nn.Linear(width, self.num_tasks))
        self.local_activation = nn.Sequential(*layers)

        input_width = len(self.user_fields) * embedding_dim + self.key_width + self.value_width
        self.towers = nn.ModuleList()
        for _ in labels:
            tower = []
            width = input_width
            for hidden in tower_hidden_units:
                tower.extend([nn.Linear(width, hidden), Dice(hidden)])
                width = hidden
            tower.append(nn.Linear(width, 1))
            self.towers.append(nn.Sequential(*tower))
        self.reset_parameters()
        # Identical initial channels remove a pooling-scale difference between
        # shared averaging and task routing. Task gradients can specialize them.
        score = self.local_activation[-1]
        with torch.no_grad():
            score.weight.copy_(score.weight[:1].clone().expand_as(score.weight))
            score.bias.copy_(score.bias[:1].clone().expand_as(score.bias))
        self.model_to_device()

    def _represent(self, inputs):
        user, item, mask = self.get_inputs(inputs)
        user_emb = self.embedding_layer(user)
        item_emb = self.embedding_layer(item)
        target = torch.cat([item_emb[name][:, -1] for name in self.item_fields], dim=-1)
        keys = torch.cat([item_emb[name][:, :-1] for name in self.item_fields], dim=-1)
        actions = item_emb[self.action_field][:, :-1]
        valid = mask.to(torch.bool)
        # Mask before Dice and after scoring. Padding IDs cannot enter the
        # normalization statistics or pooled values. Padding count is retained.
        keys = keys.masked_fill(~valid.unsqueeze(-1), 0)
        actions = actions.masked_fill(~valid.unsqueeze(-1), 0)
        values = torch.cat([keys, actions], dim=-1)
        if self.user_fields:
            context = torch.cat([user_emb[name] for name in self.user_fields], dim=-1)
        else:
            context = target.new_zeros((target.shape[0], 0))
        return context, target, keys, actions, values, valid

    def _pool(self, target, keys, actions, values, valid):
        batch, length, _ = keys.shape
        if length == 0:
            return target.new_zeros((batch, self.num_tasks, self.value_width))
        query = target[:, None, :].expand(-1, length, -1)
        action_input = actions if self.action_in_score else torch.zeros_like(actions)
        interactions = torch.cat([query, keys, query - keys, query * keys, action_input], dim=-1)
        scores = self.local_activation(interactions.reshape(batch * length, -1))
        scores = scores.reshape(batch, length, self.num_tasks)
        if not self.task_specific:
            # Every row gets gradients, but averaging reduces effective capacity.
            scores = scores.mean(dim=-1, keepdim=True).expand(-1, -1, self.num_tasks)
        scores = scores.masked_fill(~valid.unsqueeze(-1), 0)
        # Four summaries cost O(B * T * L * value_width), with no L^2 term.
        # All-masked histories give zero summaries without softmax.
        return torch.bmm(scores.transpose(1, 2), values)

    def forward(self, inputs):
        context, target, keys, actions, values, valid = self._represent(inputs)
        pooled = self._pool(target, keys, actions, values, valid)
        outputs = {}
        for index, (label, tower) in enumerate(zip(self.feature_map.labels, self.towers)):
            features = torch.cat([context, target, pooled[:, index]], dim=-1)
            outputs[label + '_pred'] = torch.sigmoid(tower(features))
        return outputs
