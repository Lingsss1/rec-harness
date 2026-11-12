"""Prospective task/target-conditioned relational content compression.

Keep the exact-count UF ancestor and append twelve masked content means:
all exposures/cart/fav/buy crossed with category/brand/joint target matches.
An auxiliary residual combines six means, target products and exact counts.
Aligned and rotated membership cells have identical capacity and count inputs;
rotation is within each example's valid history, never across examples. Cross
this contrast with the auxiliary item/action gradient gate. No new ID banks.

New final layers are zero initialized under a restored CPU RNG stream, after
all inherited initialization. No new normalization state or scoring passes.
This source has only static checks in the coding workspace. Host numerical,
masking, ownership and initialization checks are required before training.
No improvement, sufficient-statistic property or architecture novelty asserted.
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
    'compression_routes': 'uf',
    'block_aux_scorer_grad': True,
    'block_aux_context_grad': False,
    'block_aux_item_grad': True,
    'count_view': 'exact',
    'count_hidden_units': 32,
    'moment_alignment': 'rotated',
    'moment_hidden_units': 32,
}


class Model(MultiTaskModel):
    def __init__(self, feature_map, model_id='RelationalContentCompression', gpu=-1,
                 embedding_dim=16, attention_hidden_units=(64, 32),
                 tower_hidden_units=(128, 64), attention_normalization='none',
                 compression_routes='uf', block_aux_scorer_grad=True,
                 block_aux_context_grad=False, block_aux_item_grad=True,
                 count_view='exact', count_hidden_units=32,
                 moment_alignment='rotated', moment_hidden_units=32, num_tasks=None,
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
        if block_aux_context_grad:
            raise ValueError('Shared context updates stay open in this study')
        if count_view != 'exact':
            raise ValueError('The exact-count sidecar is frozen in every cell')
        if moment_alignment not in ('aligned', 'rotated'):
            raise ValueError('moment_alignment must be aligned or rotated')
        if moment_hidden_units != 32:
            raise ValueError('Matched moment head capacity is fixed at 32')
        self.moment_alignment = moment_alignment
        if count_hidden_units != 32:
            raise ValueError('Matched sidecar capacity is fixed to 32 hidden units')
        self.count_view = count_view
        self.count_fields = ('cate_id', 'brand')
        for name in self.count_fields:
            spec = feature_map.features.get(name)
            if spec is None or spec.get('source') != 'item' or spec['type'] != 'categorical':
                raise ValueError('Exact joins require categorical category and brand fields')
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

        # Preserve incumbent action-aware scorer/towers and initialization.
        # The count sidecar is allocated only after the inherited reset.
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
        # Allocate after inherited reset and restore the CPU RNG stream. This
        # preserves parent initialization and subsequent host sample ordering.
        # Keep these heads outside embedding_layer: host dense AdamW owns them.
        with torch.random.fork_rng(devices=[]):
            self.count_heads = nn.ModuleDict({
                label: nn.Sequential(nn.Linear(40, count_hidden_units), nn.ReLU(),
                                     nn.Linear(count_hidden_units, 1))
                for label in feature_map.labels if label != 'is_click'
            })
            for head in self.count_heads.values():
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
        # The inherited count-head fork restores the same CPU RNG state as
        # its ancestor. Allocate the new heads in another restoring scope.
        # There are six moments per task: three exposure relations plus three
        # matching relations for that task. Target action is never included.
        self.moment_input_width = 40 + self.key_width + 6 * (self.value_width + self.key_width)
        with torch.random.fork_rng(devices=[]):
            self.moment_heads = nn.ModuleDict({
                label: nn.Sequential(nn.Linear(self.moment_input_width, moment_hidden_units),
                                     nn.ReLU(), nn.Linear(moment_hidden_units, 1))
                for label in feature_map.labels if label != 'is_click'
            })
            for head in self.moment_heads.values():
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
        self.aux_moment_groups = {'cart': 1, 'fav': 2, 'buy': 3}
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
        exact_stats, support_stats, assignment = self._count_statistics(item, mask, task_masks)
        moments = self._content_moments(values, assignment, mask)
        return context, target, values, mask, auxiliary, exact_stats, support_stats, moments

    def _count_statistics(self, item, mask, task_masks):
        # Historical masks come from released action tokens; neither current
        # labels nor unknown target action contribute. Feedback groups overlap
        # intentionally when a historical event has several action tokens.
        valid = mask.bool()
        groups = [valid]
        for task_mask in task_masks:
            if task_mask.shape != mask.shape:
                raise ValueError('Historical task mask shape mismatch')
            groups.append((task_mask > 0) & valid)
        group_mask = torch.stack(groups, dim=1).float()  # [B,5,100]
        match_membership, support_membership = [], []
        for name in self.count_fields:
            ids = item[name]
            if ids.ndim != 2 or ids.shape != (mask.shape[0], 101):
                raise ValueError('Exact joins require 100 history IDs plus target')
            history, target_id = ids[:, :-1], ids[:, -1:]
            # Zero is padding/missing under the released convention. Exclude
            # it from joins even at a valid exposure; do not count zero==zero.
            support = (history > 0) & valid
            match = support & (target_id > 0) & (history == target_id)
            match_membership.append(match)
            support_membership.append(support)
        # Preserve the conjunction as well as each marginal: matching both
        # fields in the same historical event cannot be recovered from two
        # separate aggregate match counts.
        match_membership.append(match_membership[0] & match_membership[1])
        support_membership.append(support_membership[0] & support_membership[1])
        match_membership = torch.stack(match_membership, dim=-1).float()
        support_membership = torch.stack(support_membership, dim=-1).float()
        # Disable autocast only for tiny integer-count reductions and ratios;
        # the maximum count is 100. Both feature views are computed in every
        # cell to keep operator cost matched without dense history scoring.
        with torch.autocast(device_type=mask.device.type, enabled=False):
            totals = group_mask.sum(dim=-1)  # [B,5]
            matches = torch.bmm(group_mask, match_membership)  # [B,5,3]
            supports = torch.bmm(group_mask, support_membership)
            scale = math.log(101.0)
            common = torch.cat((torch.log1p(totals) / scale,
                                totals / totals[:, :1].clamp_min(1)), dim=-1)
            def vector(counts):
                return torch.cat((common,
                                  (torch.log1p(counts) / scale).flatten(1),
                                  (counts / totals.clamp_min(1).unsqueeze(-1)).flatten(1)), dim=-1)
            # Keep all five groups for the identical count sidecar. Only all
            # exposures and the three auxiliary types need content moments.
            selected_groups = [0] + [index + 1 for index in self.auxiliary_indices]
            task_groups = group_mask[:, selected_groups, :]
            assignment = (task_groups[:, :, None, :]
                          * match_membership.transpose(1, 2)[:, None, :, :])
            return vector(matches), vector(supports), assignment.reshape(mask.shape[0], 12, 100)

    def _content_moments(self, values, assignment, mask):
        # Rotate the ENTIRE joint membership by half the valid-event count.
        # Padding may be arbitrary: build rank->physical-position using only
        # valid events rather than rolling the physical 100 padded positions.
        # The same index/gather/reduction operators execute in aligned cells.
        valid = mask.bool()
        batch, length = valid.shape
        positions = torch.arange(length, device=mask.device).expand(batch, -1)
        ranks = (valid.long().cumsum(dim=1) - 1).clamp_min(0)
        lengths = valid.long().sum(dim=1, keepdim=True)
        physical_by_rank = torch.zeros_like(positions).scatter_add(
            1, ranks, positions * valid.long())
        shift = lengths // 2
        if self.moment_alignment == 'aligned':
            shift = torch.zeros_like(shift)
        source_ranks = (ranks + shift).remainder(lengths.clamp_min(1))
        source_positions = physical_by_rank.gather(1, source_ranks)
        assignment = assignment.gather(
            2, source_positions[:, None, :].expand(-1, 12, -1))
        assignment = assignment * valid[:, None, :].to(assignment.dtype)
        counts = assignment.sum(dim=-1)
        # No per-event learned scorer or new embeddings. Values retain their
        # original chronological positions; only membership changes in the
        # negative control. Counts/overlaps are exactly preserved by rotation.
        sums = torch.bmm(assignment.to(values.dtype), values)
        means = sums / counts.to(sums.dtype).clamp_min(1).unsqueeze(-1)
        return means.reshape(batch, 4, 3, self.value_width)

    def _moment_features(self, label, target, moments, exact_stats):
        group = self.aux_moment_groups[label]
        means = torch.cat((moments[:, 0], moments[:, group]), dim=1)
        keys = means[:, :, :self.key_width]
        products = target[:, None, :] * keys
        return torch.cat((exact_stats.to(target.dtype), target,
                          means.flatten(1), products.flatten(1)), dim=-1)


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
        weighted = (weights.unsqueeze(-1) * values).reshape(batch, 10, 3, self.value_width)
        return weighted.sum(dim=1)

    def forward(self, inputs):
        context, target, values, mask, auxiliary, exact_stats, support_stats, moments = self._represent(inputs)
        count_stats = exact_stats if self.count_view == 'exact' else support_stats
        values, counts = self._compress_both(values, mask, auxiliary)
        # Both calls include all 30 slots: each BatchNorm sees the identical
        # population. Selecting ten versus twenty slots BEFORE scoring would
        # change the forward operator and invalidate the gradient-only claim.
        primary_weights = self.local_activation(self._interactions(target, values))
        primary_strata = self._pool(primary_weights, values, counts)
        uniform = primary_strata[:, 0, :]

        if self.training:
            # Item fields share tables across current target and past events:
            # gate both uses together. Historical action joins this group.
            # Context includes only supplied non-item/non-action features.
            aux_context = context.detach() if self.block_aux_context_grad else context
            aux_target = target.detach() if self.block_aux_item_grad else target
            aux_values = values.detach() if self.block_aux_item_grad else values
            aux_moments = moments.detach() if self.block_aux_item_grad else moments
            aux_weights = self._auxiliary_activation(self._interactions(aux_target, aux_values))
            aux_strata = self._pool(aux_weights, aux_values, counts)
        else:
            # Identical values/parameters and no state update: inference needs
            # just one scorer pass for every factorial cell.
            aux_context, aux_target = context, target
            aux_moments = moments
            aux_strata = primary_strata
        feedback = aux_strata[:, 1, :] + aux_strata[:, 2, :]
        predictions = {}
        for label, tower in zip(self.feature_map.labels, self.towers):
            if label == 'is_click':
                features = torch.cat([context, target, uniform], dim=-1)
            else:
                features = torch.cat([aux_context, aux_target, feedback], dim=-1)
            logit = tower(features)
            if label != 'is_click':
                logit = logit + self.count_heads[label](count_stats)
                moment_features = self._moment_features(label, aux_target, aux_moments, exact_stats)
                logit = logit + self.moment_heads[label](moment_features)
            predictions[label + '_pred'] = torch.sigmoid(logit)
        return predictions
