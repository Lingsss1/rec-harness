"""Prospective feedback-group residual routing; effectiveness is unmeasured.

Keep the incumbent's task/action relevance and all-event summaries. Add a
small task-specific residual from five group summaries (all events plus four
overlapping historical feedback masks). A 2x2 study changes mask alignment
and the router's direct context input, with identical parameter shapes.
Historical feedback only; current target action and labels are not consumed.
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
    'attention_normalization': 'none',
    'task_specific': True,
    'action_in_score': True,
    'aligned_groups': False,
    'context_routing': True,
    'routing_hidden_units': 32,
}


class Model(MultiTaskModel):
    def __init__(self, feature_map, model_id='FeedbackGroupRouting', gpu=-1,
                 embedding_dim=16, attention_hidden_units=(64, 32),
                 tower_hidden_units=(128, 64), attention_normalization='none',
                 task_specific=True, action_in_score=True, aligned_groups=False,
                 context_routing=True, routing_hidden_units=32, num_tasks=None,
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
        self.aligned_groups = bool(aligned_groups)
        self.context_routing = bool(context_routing)
        self.group_count = len(labels) + 1
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
        # Keep incumbent modules together. Added construction/reset consumes
        # RNG, so initialization is matched across this factorial, not parent.
        routing_width = len(self.user_fields) * embedding_dim + self.key_width
        self.group_router = nn.Sequential(
            nn.Linear(routing_width + self.group_count, routing_hidden_units),
            nn.ReLU(),
            nn.Linear(routing_hidden_units, self.num_tasks * self.group_count),
        )
        self.residual_gain = nn.Parameter(torch.full((self.num_tasks, 1), 0.1))
        self.reset_parameters()
        # Identical initial channels remove a pooling-scale difference between
        # shared averaging and task routing. Task gradients can specialize them.
        score = self.local_activation[-1]
        with torch.no_grad():
            score.weight.copy_(score.weight[:1].clone().expand_as(score.weight))
            score.bias.copy_(score.bias[:1].clone().expand_as(score.bias))
        self.model_to_device()

    def _represent(self, inputs):
        if len(inputs) != 4 or inputs[3] is None:
            raise ValueError('The study requires authorized historical feedback masks')
        user, item, mask, multi_masks = self.get_inputs(inputs, return_multi_masks=True)
        if len(multi_masks) != self.num_tasks:
            raise ValueError('Expected four historical feedback masks')
        user_emb = self.embedding_layer(user)
        item_emb = self.embedding_layer(item)
        target = torch.cat([item_emb[name][:, -1] for name in self.item_fields], dim=-1)
        keys = torch.cat([item_emb[name][:, :-1] for name in self.item_fields], dim=-1)
        actions = item_emb[self.action_field][:, :-1]
        valid = mask.to(torch.bool)
        # Keep the incumbent masking. Invalid key/action IDs do not affect the
        # scorer, but padded positions still enter Dice statistics through the
        # target query. This study does not change that inherited limitation.
        keys = keys.masked_fill(~valid.unsqueeze(-1), 0)
        actions = actions.masked_fill(~valid.unsqueeze(-1), 0)
        values = torch.cat([keys, actions], dim=-1)
        if self.user_fields:
            context = torch.cat([user_emb[name] for name in self.user_fields], dim=-1)
        else:
            context = target.new_zeros((target.shape[0], 0))
        feedback_masks = torch.stack([m.to(torch.bool) for m in multi_masks], dim=1)
        if feedback_masks.shape != (target.shape[0], self.num_tasks, keys.shape[1]):
            raise ValueError('Historical feedback mask shape mismatch')
        feedback_masks = feedback_masks & valid[:, None, :]
        return context, target, keys, actions, values, valid, feedback_masks

    def _pool(self, context, target, keys, actions, values, valid, feedback_masks):
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
        pooled = torch.bmm(scores.transpose(1, 2), values)

        # Reverse assignments only among valid positions; preserve every group
        # count and all mask intersections, including histories with holes.
        # Compute the same index map in all cells for comparable overhead.
        positions = torch.arange(length, device=valid.device)[None, :].expand(batch, -1)
        packed_positions = positions.masked_fill(~valid, length).sort(dim=1).values
        ranks = valid.long().cumsum(dim=1) - 1
        reversed_ranks = (valid.long().sum(dim=1, keepdim=True) - 1 - ranks).clamp(0, length - 1)
        reverse_index = packed_positions.gather(1, reversed_ranks).clamp_max(length - 1)
        reversed_masks = feedback_masks.gather(
            2, reverse_index[:, None, :].expand(-1, self.num_tasks, -1))
        reversed_masks = reversed_masks & valid[:, None, :]
        group_masks = feedback_masks if self.aligned_groups else reversed_masks
        group_masks = torch.cat([valid[:, None, :], group_masks], dim=1)
        counts = group_masks.float().sum(dim=-1)

        # Weighted group sums / sqrt(n) retain some frequency evidence while
        # avoiding a linear scaling advantage for large groups. Identical in
        # every cell. No event is replicated as an independent training sample.
        summaries = []
        for task_index in range(self.num_tasks):
            weights = group_masks.to(scores.dtype) * scores[:, :, task_index][:, None, :]
            group_values = torch.bmm(weights, values)
            summaries.append(group_values / counts.clamp_min(1).sqrt().to(group_values.dtype)[:, :, None])
        summaries = torch.stack(summaries, dim=1)  # B,T,G,V

        routing_context = torch.cat([context, target], dim=-1)
        if not self.context_routing:
            # Support-only control retains count conditioning and the same
            # allocated network. Effective capacity is lower, explicitly.
            routing_context = torch.zeros_like(routing_context)
        support = torch.log1p(counts) / math.log1p(length)
        routing_inputs = torch.cat([routing_context, support.to(routing_context.dtype)], dim=-1)
        logits = self.group_router(routing_inputs).reshape(batch, self.num_tasks, self.group_count)
        # All-event group is a zero sentinel when history is empty, so softmax
        # always has a finite entry. Missing feedback groups get zero mass.
        available = torch.cat([torch.ones_like(counts[:, :1], dtype=torch.bool), counts[:, 1:] > 0], dim=1)
        logits = logits.masked_fill(~available[:, None, :], float('-inf'))
        routing = torch.softmax(logits.float(), dim=-1).to(summaries.dtype)
        residual = (routing[:, :, :, None] * summaries).sum(dim=2)
        return pooled + self.residual_gain[None, :, :] * residual

    def forward(self, inputs):
        context, target, keys, actions, values, valid, feedback_masks = self._represent(inputs)
        pooled = self._pool(context, target, keys, actions, values, valid, feedback_masks)
        outputs = {}
        for index, (label, tower) in enumerate(zip(self.feature_map.labels, self.towers)):
            features = torch.cat([context, target, pooled[:, index]], dim=-1)
            outputs[label + '_pred'] = torch.sigmoid(tower(features))
        return outputs
