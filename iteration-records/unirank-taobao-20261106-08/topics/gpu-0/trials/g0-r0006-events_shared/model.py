"""Prospective factorial: fixed history compression x task score readout.

All cells allocate the same modules in the source task-selector RNG order.
The trunk and item/action embeddings receive click gradients only. Context
embeddings receive all task gradients. Shared scoring uses the click row;
the other three readout rows are allocated but inactive in that control.
Compression retains source UF routing: click uses ten uniform block slots,
auxiliaries use twenty feedback-stratified slots, with count-weighted sums.
No measured benefit or architecture novelty is asserted before host execution.
"""
import torch
from torch import nn
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbeddingDict
from unirank.pytorch.layers.activations import Dice


CONFIG = {
    'embedding_dim': 16,
    'attention_hidden_units': [128, 32],
    'tower_hidden_units': [128, 64],
    'history_mode': 'events',
    'routing': 'shared',
    'feedback_in_score': True,
    'selector_gradient': 'primary',
    'block_aux_context_gradient': False,
    'block_aux_item_gradient': True,
}


class Model(MultiTaskModel):
    def __init__(self, feature_map, model_id='CompressedTaskReadout', gpu=-1,
                 embedding_dim=16, attention_hidden_units=(128, 32),
                 tower_hidden_units=(128, 64), history_mode='events', routing='shared',
                 feedback_in_score=True, selector_gradient='primary',
                 block_aux_context_gradient=False, block_aux_item_gradient=True,
                 num_tasks=None, task='binary_classification', **kwargs):
        labels = list(feature_map.labels)
        if num_tasks is not None and num_tasks != len(labels):
            raise ValueError('Task count mismatch')
        if len(labels) != 4 or set(labels) != {'is_click', 'cart', 'fav', 'buy'}:
            raise ValueError('The frozen four-task schema is required')
        if history_mode not in ('uf', 'events') or routing not in ('task', 'shared'):
            raise ValueError('Unknown factorial cell')
        if (feedback_in_score is not True or selector_gradient != 'primary'
                or block_aux_context_gradient is not False
                or block_aux_item_gradient is not True):
            raise ValueError('The source gradient ownership is fixed in this study')
        super().__init__(feature_map, model_id=model_id, gpu=gpu,
                         num_tasks=len(labels), task=task, **kwargs)
        self.history_mode = history_mode
        self.routing = routing
        self.primary_index = labels.index('is_click')
        self.auxiliary_indices = [labels.index(label) for label in ('cart', 'fav', 'buy')]
        self.register_buffer('_primary_channel', torch.tensor(
            [label == 'is_click' for label in labels], dtype=torch.bool).view(1, -1),
            persistent=False)
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        self.item_fields = [name for name, spec in feature_map.features.items()
                            if spec.get('source') == 'item' and spec['type'] != 'meta']
        self.user_fields = [name for name, spec in feature_map.features.items()
                            if spec.get('source') not in ('item', 'action')
                            and spec['type'] != 'meta']
        self.action_field = next((name for name, spec in feature_map.features.items()
                                  if spec.get('source') == 'action'), None)
        if self.action_field is None:
            raise ValueError('Released historical actions are required')
        self.key_width = len(self.item_fields) * embedding_dim
        self.value_width = self.key_width + embedding_dim
        # Module construction/reset order matches the frozen full-event source.
        width = 4 * self.key_width + embedding_dim
        layers = []
        for hidden in attention_hidden_units:
            layers.extend([nn.Linear(width, hidden), Dice(hidden)])
            width = hidden
        self.score_trunk = nn.Sequential(*layers)
        self.score_readout = nn.Linear(width, len(labels))
        self.towers = nn.ModuleList()
        input_width = len(self.user_fields) * embedding_dim + self.key_width + self.value_width
        for _ in labels:
            tower = []
            width = input_width
            for hidden in tower_hidden_units:
                tower.extend([nn.Linear(width, hidden), Dice(hidden)])
                width = hidden
            tower.append(nn.Linear(width, 1))
            self.towers.append(nn.Sequential(*tower))
        self.reset_parameters()
        with torch.no_grad():
            initial_weight = self.score_readout.weight[:1].clone()
            initial_bias = self.score_readout.bias[:1].clone()
            self.score_readout.weight.copy_(initial_weight.expand_as(self.score_readout.weight))
            self.score_readout.bias.copy_(initial_bias.expand_as(self.score_readout.bias))
        self.model_to_device()

    def _represent(self, inputs):
        if len(inputs) != 4:
            raise ValueError('Released historical task masks are required')
        supplied = inputs[3]
        if isinstance(supplied, (list, tuple)):
            if len(supplied) != self.num_tasks:
                raise ValueError('Wrong historical task-mask count')
        elif torch.is_tensor(supplied):
            if supplied.ndim != 3 or supplied.shape[0] != self.num_tasks:
                raise ValueError('Expected historical masks [4,B,100]')
        else:
            raise ValueError('Historical task masks cannot be omitted')
        user, item, mask, task_masks = self.get_inputs(inputs, return_multi_masks=True)
        if mask.ndim != 2 or mask.shape[1] != 100:
            raise ValueError('The frozen history has 100 positions')
        auxiliary = torch.zeros_like(mask, dtype=torch.bool)
        for index in self.auxiliary_indices:
            if task_masks[index].shape != mask.shape:
                raise ValueError('Historical task-mask shape mismatch')
            auxiliary = auxiliary | (task_masks[index] > 0)
        auxiliary = auxiliary & mask.bool()
        user_emb = self.embedding_layer(user)
        item_emb = self.embedding_layer(item)
        target = torch.cat([item_emb[name][:, -1] for name in self.item_fields], dim=-1)
        keys = torch.cat([item_emb[name][:, :-1] for name in self.item_fields], dim=-1)
        valid = mask.bool().unsqueeze(-1)
        keys = keys.masked_fill(~valid, 0)
        action = item_emb[self.action_field][:, :-1].masked_fill(~valid, 0)
        values = torch.cat([keys, action], dim=-1)
        context = torch.cat([user_emb[name] for name in self.user_fields], dim=-1)
        return context, target, values, mask, auxiliary

    def _slots(self, values, mask, auxiliary):
        if self.history_mode == 'events':
            return values, mask.to(values.dtype)
        batch, _, width = values.shape
        valid = mask.bool().reshape(batch, 10, 10)
        membership = auxiliary.reshape(batch, 10, 10) & valid
        assignment = torch.stack((valid, valid & ~membership, membership), dim=-2)
        assignment = assignment.to(values.dtype).reshape(batch * 10, 3, 10)
        counts = assignment.sum(-1)
        sums = torch.bmm(assignment, values.reshape(batch * 10, 10, width))
        means = sums / counts.clamp_min(1).unsqueeze(-1)
        return means.reshape(batch, 30, width), counts.reshape(batch, 30)

    def _scores(self, target, values, counts):
        batch, slots, _ = values.shape
        keys = values[:, :, :self.key_width]
        action = values[:, :, self.key_width:]
        query = target[:, None, :].expand(-1, slots, -1)
        interactions = torch.cat([query, keys, query - keys, query * keys, action], dim=-1)
        # Zero inputs precede Dice statistics. Empty slots still change the
        # fraction of fixed zero rows; this is a recorded compression confound.
        interactions = interactions.masked_fill(~(counts > 0).unsqueeze(-1), 0)
        hidden = self.score_trunk(interactions.reshape(batch * slots, -1))
        primary_scores = self.score_readout(hidden)
        if not torch.is_grad_enabled():
            if self.routing == 'shared':
                primary_scores = primary_scores[:, self.primary_index:self.primary_index + 1]
                primary_scores = primary_scores.expand(-1, self.num_tasks)
            return primary_scores.reshape(batch, slots, self.num_tasks)
        if self.routing == 'task':
            # Auxiliary rows learn; shared trunk and its inputs are primary-owned.
            auxiliary_scores = self.score_readout(hidden.detach())
        else:
            primary_scores = primary_scores[:, self.primary_index:self.primary_index + 1]
            primary_scores = primary_scores.expand(-1, self.num_tasks)
            # A shared click scorer must remain protected from auxiliary losses.
            auxiliary_scores = primary_scores.detach()
        scores = torch.where(self._primary_channel, primary_scores, auxiliary_scores)
        return scores.reshape(batch, slots, self.num_tasks)

    def _pool(self, scores, values, counts):
        valid = (counts > 0).unsqueeze(-1)
        scores = scores.masked_fill(~valid, 0)
        if self.history_mode == 'events':
            weighted_scores = scores
        else:
            # Block-major slots: [uniform, no-aux-feedback, aux-feedback].
            slot_index = torch.arange(values.shape[1], device=values.device)
            uniform = (slot_index % 3 == 0).view(1, -1, 1)
            routes = torch.where(self._primary_channel.unsqueeze(1), uniform, ~uniform)
            weighted_scores = scores * counts.to(scores.dtype).unsqueeze(-1)
            weighted_scores = weighted_scores * routes.to(scores.dtype)
        pooled = torch.bmm(weighted_scores.transpose(1, 2), values)
        if not torch.is_grad_enabled():
            return pooled
        auxiliary_pooled = torch.bmm(weighted_scores.transpose(1, 2), values.detach())
        return torch.where(self._primary_channel.unsqueeze(-1), pooled, auxiliary_pooled)

    def forward(self, inputs):
        context, target, values, mask, auxiliary = self._represent(inputs)
        values, counts = self._slots(values, mask, auxiliary)
        scores = self._scores(target, values, counts)
        pooled = self._pool(scores, values, counts)
        outputs = {}
        for index, (label, tower) in enumerate(zip(self.feature_map.labels, self.towers)):
            task_target = target if label == 'is_click' else target.detach()
            features = torch.cat([context, task_target, pooled[:, index]], dim=-1)
            outputs[label + '_pred'] = torch.sigmoid(tower(features))
        return outputs
