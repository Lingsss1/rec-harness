"""Prospective click query/history gradient factorial; performance unmeasured.

Keep private relevance scorers, shared embeddings, values and readouts.
Independently stop click relevance gradients into the target query or into
historical keys/actions BEFORE forming differences and products. All other
task gradients, pooled-value gradients and direct target paths stay live.
The two switches change backward connectivity, never visible information.
"""
import copy
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
    'task_specific': True,
    'action_in_score': True,
    'scorer_sharing': 0.0,
    'block_click_query_grad': True,
    'block_click_history_grad': False,
}


class Model(MultiTaskModel):
    def __init__(self, feature_map, model_id='ClickRoleGradientRouting', gpu=-1,
                 embedding_dim=16, attention_hidden_units=(64, 32),
                 tower_hidden_units=(128, 64), attention_normalization='none',
                 task_specific=True, action_in_score=True, scorer_sharing=0.0,
                 block_click_query_grad=True, block_click_history_grad=False,
                 num_tasks=None,
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
        if not self.task_specific or not self.action_in_score:
            raise ValueError('This study retains task/action relevance')
        if float(scorer_sharing) != 0.0:
            raise ValueError('Gradient routing study fixes private scorers')
        if not isinstance(block_click_query_grad, bool) or not isinstance(block_click_history_grad, bool):
            raise ValueError('Gradient routing switches must be booleans')
        self.block_click_query_grad = block_click_query_grad
        self.block_click_history_grad = block_click_history_grad
        if not attention_hidden_units:
            raise ValueError('A nonlinear scorer trunk is required')
        self.scorer_sharing = float(scorer_sharing)
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

        # Keep the incumbent parameter layout and initialization sequence.
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
        # Clone only AFTER the incumbent's construction/reset/equalization.
        # Deepcopy draws no RNG and copies Dice buffers to independent storage.
        # Never reset the expanded model: that would alter its initial function.
        trunk = nn.Sequential(*list(self.local_activation.children())[:-1])
        self.score_output = self.local_activation[-1]
        self.relevance_trunks = nn.ModuleList(
            [trunk] + [copy.deepcopy(trunk) for _ in range(self.num_tasks - 1)])
        del self.local_activation
        self.model_to_device()

    def _represent(self, inputs):
        user, item, mask = self.get_inputs(inputs)
        user_emb = self.embedding_layer(user)
        item_emb = self.embedding_layer(item)
        target = torch.cat([item_emb[name][:, -1] for name in self.item_fields], dim=-1)
        keys = torch.cat([item_emb[name][:, :-1] for name in self.item_fields], dim=-1)
        actions = item_emb[self.action_field][:, :-1]
        valid = mask.to(torch.bool)
        # Zero history IDs/actions before Dice and mask scores afterward.
        # Target queries at padding positions still affect Dice statistics;
        # this inherited behavior is held fixed, not padding-free normalization.
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
        flat = interactions.reshape(batch * length, -1)
        # Form a second, numerically identical interaction tensor in EVERY
        # factorial cell, so the role switches do not change forward work.
        # Detach operands before difference/product construction: detaching
        # only the raw query/key columns would leave interaction bypasses.
        click_query = query.detach() if self.block_click_query_grad else query
        click_keys = keys.detach() if self.block_click_history_grad else keys
        click_actions = action_input.detach() if self.block_click_history_grad else action_input
        click_interactions = torch.cat([
            click_query, click_keys, click_query - click_keys,
            click_query * click_keys, click_actions], dim=-1)
        click_flat = click_interactions.reshape(batch * length, -1)
        # Shared embedding TABLES remain shared. These are role-specific paths,
        # not disjoint parameter partitions. Scorer parameters still learn.
        states = torch.stack([
            trunk(click_flat if label == 'is_click' else flat)
            for trunk, label in zip(self.relevance_trunks, self.feature_map.labels)
        ], dim=1)
        center = states.float().mean(dim=1, keepdim=True).to(states.dtype)
        mixed = states + self.scorer_sharing * (center - states)
        columns = []
        for index in range(self.num_tasks):
            # Functional linear keeps autocast consistent with the parent.
            columns.append(torch.nn.functional.linear(
                mixed[:, index], self.score_output.weight[index:index + 1],
                self.score_output.bias[index:index + 1]))
        scores = torch.cat(columns, dim=-1).reshape(batch, length, self.num_tasks)
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
