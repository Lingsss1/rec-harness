"""Count-matched attribution of temporal compression resolution.

A 2x2 factorial crosses recent/equal widths with chronological/stratum_reverse
content placement. Reversal moves complete item/action vectors only among valid
positions in the SAME inherited auxiliary-union stratum. Padding, every stratum
mask, and all 30 per-sample summary counts are exactly unchanged. This tests
content recency conditional on that stratum schedule, not every effect of time.

The rank lookup executes in chronological cells too, matching operator overhead.
A reversal is an involution, so gathering the assignment columns routes original
values without materializing a second B x 100 x value_width tensor. No events
are discarded. Occupied Dice, parameters, initialization and source gates are
inherited. Identity cells are same-seed replays, not independent repetitions.
Host numerical validation and training remain required. No novelty claim.
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
    'history_partition': 'equal',
    'scorer_population': 'occupied',
    'content_order': 'stratum_reverse',
}


class PopulationScorer(nn.Sequential):
    """Same layers and state keys as the ancestor; optional row compaction.

    Native BatchNorm computes statistics over the compacted rows, preserving
    its biased training variance and unbiased running-variance convention.
    Masked rows never enter any layer or statistic. A degenerate zero/one-row
    call does not update running state; one row uses stored statistics.
    """

    def forward(self, interactions, active=None):
        if active is None:
            return super().forward(interactions)
        indices = torch.nonzero(active.reshape(-1), as_tuple=False).reshape(-1)
        selected = interactions.index_select(0, indices)
        if selected.shape[0] == 0:
            # Keep a zero derivative to supplied inputs; scorer state is idle.
            return interactions[:, :1] * 0.0
        if self.training and selected.shape[0] == 1:
            # Avoid changing module modes or shared running buffers. This is
            # only a degenerate-call rule, not a change to the host recipe.
            for layer in self:
                if isinstance(layer, Dice):
                    normalized = torch.nn.functional.batch_norm(
                        selected, layer.bn.running_mean, layer.bn.running_var,
                        layer.bn.weight, layer.bn.bias, training=False,
                        momentum=0.0, eps=layer.bn.eps)
                    probability = torch.sigmoid(normalized)
                    selected = (probability * selected
                                + layer.alpha * (1 - probability) * selected)
                else:
                    selected = layer(selected)
            scores = selected
        else:
            scores = super().forward(selected)
        output = scores.new_zeros((interactions.shape[0], 1))
        return output.index_copy(0, indices, scores)


class Model(MultiTaskModel):
    def __init__(self, feature_map, model_id='CountMatchedTemporalCompression', gpu=-1,
                 embedding_dim=16, attention_hidden_units=(64, 32),
                 tower_hidden_units=(128, 64), attention_normalization='none',
                 compression_routes='uf', block_aux_scorer_grad=True,
                 block_aux_context_grad=False, block_aux_item_grad=True,
                 history_partition='equal', scorer_population='occupied',
                 content_order='stratum_reverse', num_tasks=None,
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
            raise ValueError('Incumbent auxiliary source gates are fixed')
        if history_partition not in ('recent', 'equal'):
            raise ValueError('Unknown temporal partition')
        self.history_partition = history_partition
        if scorer_population != 'occupied':
            raise ValueError('Unknown scorer population')
        self.scorer_population = scorer_population
        if content_order not in ('chronological', 'stratum_reverse'):
            raise ValueError('Unknown content order')
        self.content_order = content_order
        widths = (20, 20, 20, 20, 10, 5, 2, 1, 1, 1)
        if history_partition == 'equal':
            widths = (10,) * 10
        # Constants consume no RNG and add no trainable parameters. This
        # nonpersistent buffer follows device moves but is derived from CONFIG.
        boundaries = [0]
        for width in widths:
            boundaries.append(boundaries[-1] + width)
        positions = torch.arange(100)
        partition = torch.stack([(positions >= left) & (positions < right)
                                 for left, right in zip(boundaries[:-1], boundaries[1:])])
        self.register_buffer('partition', partition, persistent=False)
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
        self.local_activation = PopulationScorer(*layers)
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

    def _content_permutation(self, valid, auxiliary):
        # Build a per-sample bijection; no cross-sample input or random choice.
        # Within each stratum, chronological rank r maps to r or n-1-r.
        # Invalid positions are fixed, including arbitrary holes in the mask.
        length = valid.shape[1]
        positions = torch.arange(length, device=valid.device).unsqueeze(0).expand_as(valid)
        permutation = positions
        for member in (valid & ~auxiliary, valid & auxiliary):
            rank = member.long().cumsum(dim=1) - 1
            count = member.long().sum(dim=1, keepdim=True)
            ordered = torch.where(member, positions, torch.full_like(positions, length))
            ordered = ordered.sort(dim=1).values
            source_rank = count - 1 - rank if self.content_order == 'stratum_reverse' else rank
            selected = ordered.gather(1, source_rank.clamp(min=0, max=length - 1))
            # Only active ranks are used, hence selected < length on members.
            permutation = torch.where(member, selected, permutation)
        return permutation

    def _compress_both(self, values, mask, auxiliary):
        batch, length, width = values.shape
        if length != 100:
            raise ValueError('Frozen protocol requires 100 history positions')
        valid = mask.bool()
        auxiliary = auxiliary & valid
        clean = torch.where(valid.unsqueeze(-1), values, torch.zeros_like(values))
        valid_blocks = self.partition.unsqueeze(0) & valid.unsqueeze(1)
        member = auxiliary.unsqueeze(1) & valid_blocks
        assignment = torch.stack((valid_blocks, valid_blocks & ~member, member), dim=2)
        assignment = assignment.reshape(batch, 30, 100).to(clean.dtype)
        permutation = self._content_permutation(valid, auxiliary)
        # Both permitted permutations are involutions. Reindex assignments,
        # not values: new_assignment[source] = old_assignment[perm[source]].
        assignment = assignment.gather(2, permutation[:, None, :].expand(-1, 30, -1))
        counts = assignment.sum(dim=-1)
        # Content order preserves these counts exactly within a partition.
        # Partition widths themselves still change counts and mean variance.
        sums = torch.bmm(assignment, clean)
        means = sums / counts.clamp_min(1).unsqueeze(-1)
        return means, counts

    def _interactions(self, target, values):
        keys = values[:, :, :self.key_width]
        action = values[:, :, self.key_width:]
        batch, slots, _ = keys.shape
        query = target[:, None, :].expand(-1, slots, -1)
        return torch.cat([query, keys, query - keys, query * keys, action], dim=-1).reshape(batch * slots, -1)

    def _auxiliary_activation(self, interactions, active=None):
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
            self.local_activation, (parameters, buffers), (interactions, active), strict=True)

    def _pool(self, weights, values, counts):
        batch, slots, _ = values.shape
        weights = weights.reshape(batch, slots)
        weights = weights * counts.to(weights.dtype)
        weighted = (weights.unsqueeze(-1) * values).reshape(batch, 10, 3, self.value_width)
        return weighted.sum(dim=1)

    def forward(self, inputs):
        context, target, values, mask, auxiliary = self._represent(inputs)
        values, counts = self._compress_both(values, mask, auxiliary)
        # Within each partition, both content orders score the exact same
        # occupied rows, although their input distributions may differ.
        active = counts > 0
        primary_weights = self.local_activation(self._interactions(target, values), active)
        primary_strata = self._pool(primary_weights, values, counts)
        uniform = primary_strata[:, 0, :]

        if self.training:
            # Item fields share tables across current target and past events:
            # gate both uses together. Historical action joins this group.
            # Context includes only supplied non-item/non-action features.
            aux_context = context.detach() if self.block_aux_context_grad else context
            aux_target = target.detach() if self.block_aux_item_grad else target
            aux_values = values.detach() if self.block_aux_item_grad else values
            aux_weights = self._auxiliary_activation(self._interactions(aux_target, aux_values), active)
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
