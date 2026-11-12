"""Prospective compression-timing x source-gradient factorial.

Encode each visible event using its fields/action, target-field products and
supplied context/target. The candidate averages ReLU codes; the replacement
control applies ReLU after averaging the same affine codes. Both execute the
same 100-event projections and twelve masked reductions with the same weights.
Only the location/count of elementwise ReLU operations differs. Cross this
contrast with blocked/shared auxiliary item-and-action embedding gradients.

Keep the ancestor UF scorer, towers, exact-count branch and source gates.
The new encoder learns from auxiliary losses even in the blocked cell: detach
source embeddings BEFORE encoding, never the learned codes. No new ID tables,
normalization state, current-label inputs or training-loss hook are introduced.
The three new readout final layers start at zero; inherited initialization and
CPU RNG stream are preserved. Numerical checks are host prerequisites.
No measured gain, sufficient-statistic property or novelty is claimed.
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
    'code_timing': 'after_pool',
    'code_width': 32,
    'code_head_hidden_units': 32,
}


class Model(MultiTaskModel):
    def __init__(self, feature_map, model_id='CompressionTiming', gpu=-1,
                 embedding_dim=16, attention_hidden_units=(64, 32),
                 tower_hidden_units=(128, 64), attention_normalization='none',
                 compression_routes='uf', block_aux_scorer_grad=True,
                 block_aux_context_grad=False, block_aux_item_grad=True,
                 count_view='exact', count_hidden_units=32,
                 code_timing='after_pool', code_width=32,
                 code_head_hidden_units=32, num_tasks=None,
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
        if code_timing not in ('before_pool', 'after_pool'):
            raise ValueError('code_timing must be before_pool or after_pool')
        if code_width != 32 or code_head_hidden_units != 32:
            raise ValueError('Matched encoder and readout widths are fixed at 32')
        self.code_timing = code_timing
        self.code_width = code_width
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
        # A separate restoring scope leaves every inherited tensor and the
        # host CPU RNG stream untouched. All factorial cells allocate modules
        # in this exact order, independently of their two switches.
        context_width = len(self.user_fields) * embedding_dim
        self.code_input_width = 40 + self.key_width + 6 * code_width
        with torch.random.fork_rng(devices=[]):
            self.event_projection = nn.Linear(self.value_width, code_width, bias=False)
            self.product_projection = nn.Linear(self.key_width, code_width, bias=False)
            self.condition_projection = nn.Linear(context_width + self.key_width, code_width)
            self.code_heads = nn.ModuleDict({
                label: nn.Sequential(nn.Linear(self.code_input_width, code_head_hidden_units),
                                     nn.ReLU(), nn.Linear(code_head_hidden_units, 1))
                for label in feature_map.labels if label != 'is_click'
            })
            for head in self.code_heads.values():
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
        self.aux_code_groups = {'cart': 1, 'fav': 2, 'buy': 3}
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
        return context, target, values, mask, auxiliary, exact_stats, support_stats, assignment

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
            # exposures and the three auxiliary types need content codes.
            selected_groups = [0] + [index + 1 for index in self.auxiliary_indices]
            task_groups = group_mask[:, selected_groups, :]
            assignment = (task_groups[:, :, None, :]
                          * match_membership.transpose(1, 2)[:, None, :, :])
            return vector(matches), vector(supports), assignment.reshape(mask.shape[0], 12, 100)

    def _encoded_moments(self, context, target, values, assignment):
        # Affine in each event for a fixed context/target. Thus for nonempty
        # groups mean(z) equals encoding the raw mean BEFORE the ReLU. This
        # equivalence does not require an extra trained model or extra labels.
        # Both cells deliberately keep projections at 100 events to match the
        # major matrix operations; report cost rather than assume equality.
        products = values[:, :, :self.key_width] * target[:, None, :]
        condition = self.condition_projection(torch.cat((context, target), dim=-1))
        codes = (self.event_projection(values) + self.product_projection(products)
                 + condition[:, None, :])
        if self.code_timing == 'before_pool':
            codes = torch.relu(codes)
        counts = assignment.sum(dim=-1)
        sums = torch.bmm(assignment.to(codes.dtype), codes)
        means = sums / counts.to(sums.dtype).clamp_min(1).unsqueeze(-1)
        if self.code_timing == 'after_pool':
            means = torch.relu(means)
        # Empty groups are zero in both cells, including the query/bias part.
        # No stochastic layers or batch statistics enter this new branch.
        return means.reshape(values.shape[0], 4, 3, self.code_width)

    def _code_features(self, label, target, codes, exact_stats):
        group = self.aux_code_groups[label]
        selected = torch.cat((codes[:, 0], codes[:, group]), dim=1)
        return torch.cat((exact_stats.to(target.dtype), target, selected.flatten(1)), dim=-1)

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
        context, target, values, mask, auxiliary, exact_stats, support_stats, assignment = self._represent(inputs)
        raw_values = values
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
            aux_raw_values = raw_values.detach() if self.block_aux_item_grad else raw_values
            aux_weights = self._auxiliary_activation(self._interactions(aux_target, aux_values))
            aux_strata = self._pool(aux_weights, aux_values, counts)
        else:
            # Identical values/parameters and no state update: inference needs
            # just one scorer pass for every factorial cell.
            aux_context, aux_target = context, target
            aux_raw_values = raw_values
            aux_strata = primary_strata
        # Detach only the sources above: these new projection parameters
        # must still learn in the blocked cells. Context gradients stay open.
        codes = self._encoded_moments(aux_context, aux_target, aux_raw_values, assignment)
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
                code_features = self._code_features(label, aux_target, codes, exact_stats)
                logit = logit + self.code_heads[label](code_features)
            predictions[label + '_pred'] = torch.sigmoid(logit)
        return predictions
