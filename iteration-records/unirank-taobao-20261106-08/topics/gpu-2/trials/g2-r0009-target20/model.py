"""Prospective target-conditioned compression versus history-conditioned retrieval.

Four cells cross target/history queries with 10/20 temporal blocks. A shared
rank-16 cosine router computes one bounded event score, then separately
normalizes inside uniform/negative-feedback/positive-feedback block slots.
Every summary is half uniform mean and half attention mean. Counts stay exact;
the inherited downstream DIN activation remains unnormalized.

Router modules and inherited tensor shapes/initialization match across cells.
Auxiliary losses see detached compressed values and target embeddings; only
click trains the router and shared item/action/scorer parameters. Context
updates remain shared. There are no new loss hooks, sidecars, ID tables or
normalization buffers. This implements a controlled adaptation, not a new
attention primitive or a reproduction of a complete external architecture.
Numerical execution and training are host-only prerequisites.
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
    'compression_routes': 'uf',
    'block_aux_scorer_grad': True,
    'block_aux_context_grad': False,
    'block_aux_item_grad': True,
    'query_source': 'target',
    'compression_groups': 20,
    'router_width': 16,
}


class Model(MultiTaskModel):
    def __init__(self, feature_map, model_id='ConditionalBlockCompression', gpu=-1,
                 embedding_dim=16, attention_hidden_units=(64, 32),
                 tower_hidden_units=(128, 64), attention_normalization='none',
                 compression_routes='uf', block_aux_scorer_grad=True,
                 block_aux_context_grad=False, block_aux_item_grad=True,
                 query_source='target', compression_groups=20, router_width=16, num_tasks=None,
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
        self.block_aux_scorer_grad = block_aux_scorer_grad
        self.block_aux_context_grad = block_aux_context_grad
        self.block_aux_item_grad = block_aux_item_grad
        if block_aux_context_grad or not block_aux_item_grad:
            raise ValueError('This study freezes incumbent context/item gradient gates')
        if query_source not in ('target', 'history'):
            raise ValueError('query_source must be target or history')
        if compression_groups not in (10, 20) or router_width != 16:
            raise ValueError('Frozen factorial requires 10/20 blocks and rank 16')
        self.query_source = query_source
        self.compression_groups = compression_groups
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

        # Preserve the incumbent scorer/tower construction and initialization.
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
        # Allocate only after the inherited reset. Restore the CPU RNG stream
        # so adding the router does not change the host training sample order.
        with torch.random.fork_rng(devices=[]):
            self.router_key = nn.Linear(self.value_width, router_width, bias=False)
            self.router_query = nn.Linear(self.key_width, router_width, bias=False)
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

    def _compress_both(self, target, values, mask, auxiliary):
        batch, length, width = values.shape
        if length != 100:
            raise ValueError('Frozen protocol requires 100 history positions')
        groups = self.compression_groups
        block_size = length // groups
        valid = mask.bool()
        # Sanitize padding before both query construction and value projection.
        clean = torch.where(valid.unsqueeze(-1), values, torch.zeros_like(values))
        history_query = clean[:, :, :self.key_width].sum(dim=1)
        history_query = history_query / valid.sum(dim=1, keepdim=True).clamp_min(1)
        # Execute both query projections in every cell. Each chosen query has
        # the same width, active parameter count, and authorized sample scope.
        target_code = self.router_query(target)
        history_code = self.router_query(history_query)
        query = target_code if self.query_source == 'target' else history_code
        keys = self.router_key(clean)
        # Small FP32 normalization/reductions avoid BF16 underflow and make
        # all-empty strata finite. The projections still use host autocast.
        with torch.autocast(device_type=values.device.type, enabled=False):
            query = torch.nn.functional.normalize(query.float(), dim=-1, eps=1e-6)
            keys = torch.nn.functional.normalize(keys.float(), dim=-1, eps=1e-6)
            scores = 2.0 * (keys * query.unsqueeze(1)).sum(dim=-1)
            valid_blocks = valid.reshape(batch, groups, block_size)
            member = auxiliary.reshape(batch, groups, block_size) & valid_blocks
            assignment = torch.stack((valid_blocks, valid_blocks & ~member, member), dim=-2)
            counts = assignment.float().sum(dim=-1)
            logits = scores.reshape(batch, groups, 1, block_size).expand_as(assignment)
            probability = torch.softmax(logits.masked_fill(~assignment, -1e4), dim=-1)
            probability = probability * assignment.float()
            probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            uniform = assignment.float() / counts.clamp_min(1).unsqueeze(-1)
            coefficients = 0.5 * (uniform + probability)
        # Each nonempty coefficient row sums to one. Empty rows are exactly
        # zero. The 1/2 uniform component keeps every valid event represented.
        coefficients = coefficients.reshape(batch * groups, 3, block_size).to(clean.dtype)
        grouped = clean.reshape(batch * groups, block_size, width)
        means = torch.bmm(coefficients, grouped)
        return means.reshape(batch, groups * 3, width), counts.reshape(batch, groups * 3)

    def _interactions(self, target, values):
        keys = values[:, :, :self.key_width]
        action = values[:, :, self.key_width:]
        batch, slots, _ = keys.shape
        query = target[:, None, :].expand(-1, slots, -1)
        return torch.cat([query, keys, query - keys, query * keys, action], dim=-1).reshape(batch * slots, -1)

    def _auxiliary_activation(self, interactions):
        # Functional evaluation uses the SAME learned scorer with parameter
        # stop-gradient fixed on. Do not use no_grad: gradients into each
        # allowed embedding source must survive the shared scoring operator.
        parameters = {
            name: (parameter.detach() if self.block_aux_scorer_grad else parameter)
            for name, parameter in self.local_activation.named_parameters()
        }
        # Dice contains BatchNorm running state. Only the primary call may
        # update persistent buffers. Cloned auxiliary buffers also prevent an
        # in-place second update of state referenced by the first graph.
        buffers = {name: buffer.detach().clone()
                   for name, buffer in self.local_activation.named_buffers()}
        return torch.func.functional_call(
            self.local_activation, (parameters, buffers), (interactions,), strict=True)

    def _pool(self, weights, values, counts):
        batch, slots, _ = values.shape
        weights = weights.reshape(batch, slots) * counts.to(weights.dtype)
        weighted = (weights.unsqueeze(-1) * values).reshape(batch, self.compression_groups, 3, self.value_width)
        return weighted.sum(dim=1)

    def forward(self, inputs):
        context, target, values, mask, auxiliary = self._represent(inputs)
        values, counts = self._compress_both(target, values, mask, auxiliary)
        # Both scorer calls include all 3*groups slots. Within a block-count
        # pair, query source is the only changed operator. Across block counts,
        # granularity, scorer cost and Dice slot population all change.
        primary_weights = self.local_activation(self._interactions(target, values))
        primary_strata = self._pool(primary_weights, values, counts)
        uniform = primary_strata[:, 0, :]

        if self.training:
            # Detach AFTER compression: auxiliary losses cannot train either
            # the router or target/history item/action tables through routing.
            # Supplied context embeddings remain open to auxiliary gradients.
            aux_context = context.detach() if self.block_aux_context_grad else context
            aux_target = target.detach() if self.block_aux_item_grad else target
            aux_values = values.detach() if self.block_aux_item_grad else values
            aux_weights = self._auxiliary_activation(self._interactions(aux_target, aux_values))
            aux_strata = self._pool(aux_weights, aux_values, counts)
        else:
            # Identical values/parameters and no state update: inference needs
            # just one scorer pass for every factorial cell.
            aux_context, aux_target = context, target
            aux_strata = primary_strata
        feedback = aux_strata[:, 1, :] + aux_strata[:, 2, :]
        predictions = {}
        for label, tower in zip(self.feature_map.labels, self.towers):
            if label == 'is_click':
                features = torch.cat([context, target, uniform], dim=-1)
            else:
                features = torch.cat([aux_context, aux_target, feedback], dim=-1)
            predictions[label + '_pred'] = torch.sigmoid(tower(features))
        return predictions
