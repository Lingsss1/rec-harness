"""Source-anchored UF compression x task readout / empty-input factorial.

All variants preserve the incumbent scalar primary scorer initialization.
Auxiliary rows are added after reset with RNG state restored and initialized
as copies of the primary row. Shared controls allocate the same rows unused.
Auxiliary losses update context and their own readout/tower, but do not update
primary scorer or target/history item/action paths. Dice state updates once.
Empty-slot query/zero policies differ only before the same scorer population;
zero-count slots always have zero pooling mass. Host numerical gates pending.
No new measured gain, mechanism confirmation or novelty is asserted.
"""
import torch
from torch import nn
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbeddingDict
from unirank.pytorch.layers.activations import Dice


CONFIG = {'embedding_dim': 16,
 'attention_hidden_units': [64, 32],
 'tower_hidden_units': [128, 64],
 'attention_normalization': 'none',
 'compression_routes': 'uf',
 'block_aux_scorer_grad': True,
 'block_aux_context_grad': False,
 'block_aux_item_grad': True,
 'routing': 'shared',
 'empty_slot_inputs': 'zero'}


class Model(MultiTaskModel):
    def __init__(self, feature_map, model_id='AnchoredTaskCompression', gpu=-1,
                 embedding_dim=16, attention_hidden_units=(64, 32),
                 tower_hidden_units=(128, 64), attention_normalization='none',
                 compression_routes='uf', block_aux_scorer_grad=True,
                 block_aux_context_grad=False, block_aux_item_grad=True,
                 routing='shared', empty_slot_inputs='zero', num_tasks=None,
                 task='binary_classification', **kwargs):
        if num_tasks is not None and num_tasks != len(feature_map.labels):
            raise ValueError('Task count mismatch')
        if set(feature_map.labels) != {'is_click', 'cart', 'fav', 'buy'}:
            raise ValueError('This study requires the frozen four-task schema')
        if attention_normalization != 'none':
            raise ValueError('Unnormalized DIN activation is required')
        if compression_routes != 'uf':
            raise ValueError('This source-path study freezes click/auxiliary routes to uf')
        super().__init__(feature_map, model_id=model_id, gpu=gpu,
                         num_tasks=len(feature_map.labels), task=task, **kwargs)
        self.compression_routes = compression_routes
        if block_aux_scorer_grad is not True:
            raise ValueError('Auxiliary scorer parameter gradients must stay blocked')
        if not isinstance(block_aux_context_grad, bool) or not isinstance(block_aux_item_grad, bool):
            raise ValueError('Source-path switches must be booleans')
        if block_aux_context_grad is not False or block_aux_item_grad is not True:
            raise ValueError('Frozen source-path ownership is required')
        if routing not in ('task', 'shared') or empty_slot_inputs not in ('query', 'zero'):
            raise ValueError('Unknown factorial cell')
        self.routing = routing
        self.empty_slot_inputs = empty_slot_inputs
        self.auxiliary_labels = [label for label in feature_map.labels if label != 'is_click']
        self.block_aux_scorer_grad = block_aux_scorer_grad
        self.block_aux_context_grad = block_aux_context_grad
        self.block_aux_item_grad = block_aux_item_grad
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

        # Inherit the action-aware scorer and towers from the previous factorial.
        # All cells instantiate exactly the same modules in exactly this order.
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
        # Preserve all inherited RNG draws and the RNG state used by host
        # data iteration. nn.Linear is constructed on CPU, so no CUDA RNG
        # is used in this scope. Allocate identical rows even when inactive.
        with torch.random.fork_rng(devices=[]):
            primary_readout = self.local_activation[-1]
            self.auxiliary_readout = nn.Linear(primary_readout.in_features, 3)
            with torch.no_grad():
                self.auxiliary_readout.weight.copy_(primary_readout.weight.expand_as(
                    self.auxiliary_readout.weight))
                self.auxiliary_readout.bias.copy_(primary_readout.bias.expand_as(
                    self.auxiliary_readout.bias))
        self.model_to_device()

    def _represent(self, inputs):
        # Do not allow the framework fallback that treats every event as
        # positive for every task when historical masks are missing.
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
        # Current action is unknown and never enters either summary or scorer.
        values = torch.cat([keys, item_emb[self.action_field][:, :-1]], dim=-1)
        context = torch.cat([user_emb[name] for name in self.user_fields], dim=-1)
        return context, target, values, mask, auxiliary

    def _compress_both(self, values, mask, auxiliary):
        batch, length, width = values.shape
        if length != 100:
            raise ValueError('Frozen protocol requires 100 history positions')
        groups, block_size = 10, 10
        valid = mask.bool().reshape(batch, groups, block_size)
        membership = auxiliary.reshape(batch, groups, block_size) & valid
        # Per temporal block: one uniform slot and two feedback strata.
        # Each branch independently covers every valid event exactly once.
        assignment = torch.stack((valid, valid & ~membership, membership), dim=-2)
        assignment = assignment.to(values.dtype).reshape(batch * groups, 3, block_size)
        counts = assignment.sum(-1)
        grouped = values.reshape(batch * groups, block_size, width)
        sums = torch.bmm(assignment, grouped)
        means = sums / counts.clamp_min(1).unsqueeze(-1)
        return means.reshape(batch, groups * 3, width), counts.reshape(batch, groups * 3)

    def _interactions(self, target, values):
        keys = values[:, :, :self.key_width]
        action = values[:, :, self.key_width:]
        batch, slots, _ = keys.shape
        query = target[:, None, :].expand(-1, slots, -1)
        return torch.cat([query, keys, query - keys, query * keys, action], dim=-1).reshape(batch * slots, -1)

    def _pool(self, weights, values, counts):
        batch, slots, _ = values.shape
        weights = weights.reshape(batch, slots) * counts.to(weights.dtype)
        weighted = (weights.unsqueeze(-1) * values).reshape(batch, 10, 3, self.value_width)
        return weighted.sum(dim=1)

    def forward(self, inputs):
        context, target, values, mask, auxiliary = self._represent(inputs)
        values, counts = self._compress_both(values, mask, auxiliary)
        batch, slots, _ = values.shape
        interactions = self._interactions(target, values)
        if self.empty_slot_inputs == 'zero':
            interactions = interactions.masked_fill(
                ~(counts.reshape(batch * slots) > 0).unsqueeze(-1), 0)
        # Source query policy includes target interactions at empty slots.
        # Every cell scores all thirty slots before routing/count masking.
        hidden = interactions
        for layer in list(self.local_activation.children())[:-1]:
            hidden = layer(hidden)
        primary_weights = self.local_activation[-1](hidden)
        primary_strata = self._pool(primary_weights, values, counts)
        uniform = primary_strata[:, 0, :]

        aux_context = context
        aux_target = target.detach() if self.training else target
        aux_values = values.detach() if self.training else values
        if self.routing == 'task':
            # Protect primary scorer parameters and all upstream item/action
            # paths. Each auxiliary row alone learns its task's score map.
            aux_hidden = hidden.detach() if self.training else hidden
            auxiliary_weights = self.auxiliary_readout(aux_hidden)
        else:
            # Same primary scalar scorer, frozen for every auxiliary task.
            auxiliary_weights = primary_weights.detach() if self.training else primary_weights
            auxiliary_weights = auxiliary_weights.expand(-1, 3)

        predictions = {}
        for label, tower in zip(self.feature_map.labels, self.towers):
            if label == 'is_click':
                features = torch.cat([context, target, uniform], dim=-1)
            else:
                column = self.auxiliary_labels.index(label)
                aux_strata = self._pool(auxiliary_weights[:, column:column + 1],
                                        aux_values, counts)
                feedback = aux_strata[:, 1, :] + aux_strata[:, 2, :]
                features = torch.cat([aux_context, aux_target, feedback], dim=-1)
            predictions[label + '_pred'] = torch.sigmoid(tower(features))
        return predictions
