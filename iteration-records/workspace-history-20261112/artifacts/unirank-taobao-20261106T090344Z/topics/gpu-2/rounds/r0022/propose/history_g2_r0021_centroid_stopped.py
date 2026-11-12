"""Prospective slot-local prototype transport, study g2-r0021-query_gradient.

A 2x2 replacement tests current-target versus history-centroid transport
queries, and live versus stopped score derivatives. Only value arguments to
_pool change. Scorer inputs, angular correction, identity readout, all learned
state and inherited auxiliary gradient gates stay fixed. Both query sources
are computed with the same FP32 statistics path in all cells; selected backward
paths intentionally differ. Coefficient 0.5 preserves count mass and bounded
nonnegative event weights in exact arithmetic. No training result, mechanism
confirmation, originality or measured efficiency gain is claimed here.
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
 'history_partition': 'recent',
 'scorer_population': 'occupied',
 'content_order': 'chronological',
 'temporal_readout': 'separate',
 'behavior_readout': 'separate',
 'residual_hidden_units': 32,
 'identity_content': 'union',
 'identity_hidden_units': 32,
 'moment_timing': 'centroid',
 'moment_query': 'aligned',
 'moment_hidden_units': 32,
 'angular_population': 'equal',
 'concentration_readout': 'retained',
 'transport_query': 'centroid',
 'transport_score_grad': False}


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
    def __init__(self, feature_map, model_id='PrototypeValueTransport', gpu=-1,
                 embedding_dim=16, attention_hidden_units=(64, 32),
                 tower_hidden_units=(128, 64), attention_normalization='none',
                 compression_routes='uf', block_aux_scorer_grad=True,
                 block_aux_context_grad=False, block_aux_item_grad=True,
                 history_partition='recent', scorer_population='occupied',
                 content_order='chronological', temporal_readout='separate',
                 behavior_readout='separate', residual_hidden_units=32,
                 identity_content='union', identity_hidden_units=32,
                 moment_timing='centroid', moment_query='aligned',
                 moment_hidden_units=32, angular_population='equal',
                 concentration_readout='retained', transport_query='centroid',
                 transport_score_grad=False,
                 num_tasks=None,
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
        if content_order != 'chronological':
            raise ValueError('This study keeps content chronological')
        self.content_order = content_order
        if temporal_readout not in ('separate', 'collapsed'):
            raise ValueError('Unknown temporal readout')
        if residual_hidden_units != 32:
            raise ValueError('Residual width is fixed at 32 for attribution')
        self.temporal_readout = temporal_readout
        if behavior_readout not in ('separate', 'collapsed'):
            raise ValueError('Unknown behavior readout')
        self.behavior_readout = behavior_readout
        if temporal_readout != 'separate' or behavior_readout != 'separate':
            raise ValueError('Freeze the incumbent marginal readouts')
        if identity_content not in ('exact', 'centroid', 'union'):
            raise ValueError('Unknown task-moment replacement')
        if identity_hidden_units != 32:
            raise ValueError('Identity residual width is fixed at 32')
        self.identity_content = identity_content
        if identity_content != 'union':
            raise ValueError('Freeze the union parent for moment attribution')
        if history_partition != 'recent':
            raise ValueError('Freeze the recent partition')
        if moment_timing != 'centroid' or moment_query != 'aligned':
            raise ValueError('Freeze the squared aligned first-moment channel')
        if angular_population not in ('equal', 'norm_weighted'):
            raise ValueError('Unknown angular population')
        if concentration_readout not in ('retained', 'unit'):
            raise ValueError('Unknown concentration readout')
        if angular_population != 'equal' or concentration_readout != 'retained':
            raise ValueError('Freeze the equal-event retained angular reference')
        if transport_query not in ('target', 'centroid'):
            raise ValueError('Unknown within-slot transport query')
        if not isinstance(transport_score_grad, bool):
            raise ValueError('Transport score-gradient switch must be boolean')
        self.transport_query = transport_query
        self.transport_score_grad = transport_score_grad
        self.angular_population = angular_population
        self.concentration_readout = concentration_readout
        if moment_hidden_units != 32 or embedding_dim < 2:
            raise ValueError('Require width32 and at least two embedding coordinates')
        self.moment_timing = moment_timing
        self.moment_query = moment_query
        self.moment_embedding_dim = embedding_dim
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
        # Both layouts have a boundary at index 90; no summary crosses it.
        if 90 not in boundaries:
            raise ValueError('The last-ten boundary must align with summaries')
        age_sign = torch.tensor([1.0 if left >= 90 else -1.0
                                 for left in boundaries[:-1]])
        self.register_buffer('age_sign', age_sign, persistent=False)
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
        # Preserve ancestor initialization AND the post-construction CPU RNG.
        # New modules are made on CPU and moved by model_to_device below.
        # No shared reset is called after adding the residual heads.
        with torch.random.fork_rng(devices=[]):
            self.residual_heads = nn.ModuleList([
                nn.Sequential(nn.Linear(input_width + 3 * self.value_width,
                                        residual_hidden_units),
                              nn.ReLU(), nn.Linear(residual_hidden_units, 1))
                for _ in feature_map.labels
            ])
            for head in self.residual_heads:
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
        # A separate RNG fork leaves every inherited parameter and the caller's
        # CPU RNG unchanged. All replacement cells have identical new shapes,
        # initialization and module order. Zero output reproduces the parent
        # predictions initially; it does not promise trained-model equality.
        with torch.random.fork_rng(devices=[]):
            identity_width = (len(self.user_fields) * embedding_dim
                              + self.key_width + 6 * self.value_width + 30)
            self.identity_heads = nn.ModuleList([
                nn.Sequential(nn.Linear(identity_width, identity_hidden_units),
                              nn.ReLU(), nn.Linear(identity_hidden_units, 1))
                for _ in feature_map.labels
            ])
            for head in self.identity_heads:
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
        # Separate fork preserves all inherited parameters and caller RNG.
        # Input is C + target + slot item/action mean + two cosine means
        # + selected second moment + normalized count + age sign.
        with torch.random.fork_rng(devices=[]):
            moment_width = (len(self.user_fields) * embedding_dim
                            + self.key_width + self.value_width
                            + 3 * len(self.item_fields) + 2)
            self.moment_correction = nn.Sequential(
                nn.Linear(moment_width, moment_hidden_units), nn.ReLU(),
                nn.Linear(moment_hidden_units, 1))
            nn.init.zeros_(self.moment_correction[-1].weight)
            nn.init.zeros_(self.moment_correction[-1].bias)
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
        # Preserve all three authorized task masks, including overlaps. The
        # canonical order is independent of the FeatureMap label order.
        task_members = torch.stack([
            (task_masks[index] > 0) & mask.bool()
            for index in self.auxiliary_indices
        ], dim=1)
        return context, target, values, mask, auxiliary, task_members

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

    def _transport_values(self, target, raw_values, mask, auxiliary, values):
        """Query-source x score-gradient replacement in the value path only.

        A slot's history query is the direction of the equal-event mean of
        per-field normalized keys. Both query sources are computed in every
        cell. A single [B,slot,event] score tensor avoids event/value expansion
        and within-slot reversal indices. Stopping scores cuts their entire
        derivative, while gradients through the clean values remain live.
        """
        batch, length, _ = raw_values.shape
        valid = mask.bool()
        valid_blocks = self.partition[None, :, :] & valid[:, None, :]
        member = auxiliary[:, None, :] & valid_blocks
        assignment = torch.stack((valid_blocks, valid_blocks & ~member, member), dim=2)
        assignment = assignment.reshape(batch, 30, length)
        fields = len(self.item_fields)
        # This is a local statistics boundary, not a host-precision change.
        with torch.autocast(device_type=raw_values.device.type, enabled=False):
            clean = torch.where(valid[:, :, None], raw_values.float(),
                                torch.zeros_like(raw_values, dtype=torch.float32))
            keys = clean[:, :, :self.key_width].reshape(
                batch, length, fields, self.moment_embedding_dim)
            unit_keys = torch.nn.functional.normalize(keys, p=2, dim=-1, eps=1e-8)
            flat_keys = unit_keys.reshape(batch, length, self.key_width)
            a = assignment.float()
            counts = a.sum(dim=-1, keepdim=True)
            n = counts.clamp_min(1.0)
            # Means use only members of this exact all/nonfeedback/feedback
            # slot; no transfer across users, age partitions or strata.
            direction_means = (torch.bmm(a, flat_keys) / n).reshape(
                batch, 30, fields, self.moment_embedding_dim)
            history_query = torch.nn.functional.normalize(
                direction_means, p=2, dim=-1, eps=1e-8)
            target_query = target.float().reshape(
                batch, fields, self.moment_embedding_dim)
            target_query = torch.nn.functional.normalize(
                target_query, p=2, dim=-1, eps=1e-8)
            target_query = target_query[:, None, :, :].expand(-1, 30, -1, -1)
            query = history_query if self.transport_query == 'centroid' else target_query
            score = torch.bmm(query.reshape(batch, 30, self.key_width),
                              flat_keys.transpose(1, 2)) / float(fields)
            score = score.clamp(min=-1.0, max=1.0)
            score = torch.where(assignment, score, torch.zeros_like(score))
            if not self.transport_score_grad:
                score = score.detach()
            mean_value = torch.bmm(a, clean) / n
            mean_score = score.sum(dim=-1, keepdim=True) / n
            covariance = torch.bmm(score, clean) / n - mean_score * mean_value
            # Match the inherited exact zero rule, including holey masks.
            covariance = torch.where(counts > 1, covariance,
                                     torch.zeros_like(covariance))
            # Equivalent to count-preserving weights in [0,2] in exact
            # arithmetic. Inherited base means may already be BF16-rounded.
            tilted = values.float() + 0.5 * covariance
        return tilted.to(values.dtype)

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
        total = weighted.sum(dim=1)
        contrast = (weighted * self.age_sign.to(weighted.dtype).view(1, 10, 1, 1)).sum(dim=1)
        # Compute both reductions in every cell. The replacement control
        # duplicates total: no age contrast reaches its logit branch.
        readout = contrast if self.temporal_readout == 'separate' else total
        return total, readout

    def _identity_statistics(self, raw_values, mask, task_members, values, counts):
        # [B,task,partition,event]; no extra scorer rows or parameters.
        clean = torch.where(mask.bool().unsqueeze(-1), raw_values,
                            torch.zeros_like(raw_values))
        assignment = (task_members[:, :, None, :]
                      & self.partition[None, None, :, :])
        assignment = assignment.reshape(raw_values.shape[0], 30, 100).to(clean.dtype)
        task_counts = assignment.sum(dim=-1, dtype=torch.float32).reshape(-1, 3, 10)
        exact = torch.bmm(assignment, clean).float().reshape(-1, 3, 10, self.value_width)
        union_means = values.reshape(-1, 10, 3, self.value_width)[:, :, 2, :]
        union_counts = counts.reshape(-1, 10, 3)[:, :, 2].float()
        centroid = task_counts.unsqueeze(-1) * union_means[:, None, :, :]
        duplicated = (union_counts.unsqueeze(-1) * union_means)[:, None, :, :].expand(-1, 3, -1, -1)
        # Compute all alternatives before replacement in every cell. Exact
        # task counts stay visible in the exact/centroid pair. The union cell
        # contains no resolved mask information on any path to a logit.
        if self.identity_content == 'exact':
            sums, exposed_counts = exact, task_counts
        elif self.identity_content == 'centroid':
            sums, exposed_counts = centroid, task_counts
        else:
            sums = duplicated
            exposed_counts = union_counts[:, None, :].expand(-1, 3, -1)
        return sums, exposed_counts

    def _identity_readout(self, primary_weights, sums, counts):
        # Each partition's existing union-positive score is reused for each
        # task sum. Counts are already present in sums; never multiply twice.
        weights = primary_weights.reshape(-1, 10, 3)[:, :, 2].float()
        # Common FP32 elementwise/reduction path across replacements; the
        # BMM itself still follows the frozen host autocast recipe.
        weighted = sums.float() * weights[:, None, :, None]
        total = weighted.sum(dim=2)
        age = (weighted * self.age_sign.to(weighted.dtype)[None, None, :, None]).sum(dim=2)
        # Counts are unweighted and FP32, on the fixed 100-event scale.
        # Do not interpret them as rates per unit time: timestamps unavailable.
        return torch.cat([total.flatten(1), age.flatten(1),
                          counts.flatten(1) / 100.0], dim=-1)

    def _similarity_statistics(self, target, raw_values, mask, auxiliary):
        """Return three [B,30,F] channels from target-independent slot vectors.

        All four cells measure the same per-event norm and pool its scalar
        mean. Equal weighting additionally divides keys before pooling; norm
        weighting divides only slot vectors by the pooled norm. Unit cells
        expose the direction of the corresponding mean before that scalar
        division, preserving the raw-unit ancestor's epsilon convention.
        The unused scalar mean in unit/equal cells matches forward norm work;
        backward paths and normalization kernels necessarily differ.
        """
        batch, length, _ = raw_values.shape
        valid = mask.bool()
        valid_blocks = self.partition[None, :, :] & valid[:, None, :]
        member = auxiliary[:, None, :] & valid_blocks
        assignment = torch.stack((valid_blocks, valid_blocks & ~member, member), dim=2)
        assignment = assignment.reshape(batch, 30, length)
        fields = len(self.item_fields)
        # Same local FP32-statistics boundary as both authorized ancestors.
        # Learned modules still follow the host's frozen BF16 autocast recipe.
        with torch.autocast(device_type=raw_values.device.type, enabled=False):
            keys = raw_values[:, :, :self.key_width].float()
            keys = torch.where(valid[:, :, None], keys, torch.zeros_like(keys))
            keys = keys.reshape(batch, length, fields, self.moment_embedding_dim)
            event_norms = torch.linalg.vector_norm(keys, ord=2, dim=-1).clamp_min(1e-8)
            assignment = assignment.float()
            n = assignment.sum(dim=-1, keepdim=True).clamp_min(1.0)
            mean_norm = torch.bmm(assignment, event_norms) / n
            if self.angular_population == 'equal':
                event_vectors = keys / event_norms.unsqueeze(-1)
            else:
                event_vectors = keys
            means = (torch.bmm(assignment, event_vectors.reshape(
                batch, length, self.key_width)) / n).reshape(
                    batch, 30, fields, self.moment_embedding_dim)
            if self.concentration_readout == 'unit':
                descriptors = torch.nn.functional.normalize(means, p=2, dim=-1, eps=1e-8)
            elif self.angular_population == 'norm_weighted':
                # mean_norm is a masked mean of pointwise clamped norms,
                # NOT the norm of the mean. Triangle inequality bounds this
                # vector by one, including zero vectors and empty slots.
                descriptors = means / mean_norm.clamp_min(1e-8).unsqueeze(-1)
            else:
                descriptors = means
            query = target.float().reshape(batch, fields, self.moment_embedding_dim)
            query = torch.nn.functional.normalize(query, p=2, dim=-1, eps=1e-8)
            rotated_query = torch.roll(query, shifts=1, dims=-1)
            mean_aligned = (descriptors * query[:, None, :, :]).sum(dim=-1)
            mean_rotated = (descriptors * rotated_query[:, None, :, :]).sum(dim=-1)
            squared_mean = mean_aligned.square()
        return mean_aligned, mean_rotated, squared_mean

    def _moment_weights(self, context, target, raw_values, mask, auxiliary,
                        values, counts):
        ma, mr, second = self._similarity_statistics(target, raw_values, mask, auxiliary)
        batch, slots, _ = values.shape
        age = self.age_sign[None, :, None].expand(batch, -1, 3).reshape(batch, slots, 1)
        inputs = torch.cat([
            context[:, None, :].expand(-1, slots, -1).float(),
            target[:, None, :].expand(-1, slots, -1).float(), values.float(),
            ma, mr, second, counts.float().unsqueeze(-1) / 100.0, age.float()
        ], dim=-1)
        correction = self.moment_correction(inputs.reshape(batch * slots, -1))
        # No new BatchNorm/Dice population; inactive rows cannot change a weight.
        return correction * (counts > 0).reshape(-1, 1).to(correction.dtype)

    def forward(self, inputs):
        context, target, raw_values, mask, auxiliary, task_members = self._represent(inputs)
        values, counts = self._compress_both(raw_values, mask, auxiliary)
        transported = self._transport_values(target, raw_values, mask, auxiliary, values)
        identity_sums, identity_counts = self._identity_statistics(
            raw_values, mask, task_members, values, counts)
        # The readout replacement changes no compression counts, occupied
        # scorer rows, input ordering or inherited Dice populations.
        active = counts > 0
        correction = self._moment_weights(context, target, raw_values, mask,
                                          auxiliary, values, counts)
        primary_weights = (self.local_activation(self._interactions(target, values), active)
                           + correction)
        primary_strata, primary_readout = self._pool(primary_weights, transported, counts)
        identity = self._identity_readout(primary_weights, identity_sums, identity_counts)
        # Scorer and both item/action sources are frozen to auxiliary losses.
        # Computing this descriptor once also avoids another Dice population.
        aux_identity = identity.detach() if self.training else identity
        uniform = primary_strata[:, 0, :]
        click_readout = primary_readout[:, 0, :]
        # F stays visible in every cell, including click. Then B vs duplicate
        # F tests stratum identity conditional on their common total, rather
        # than comparing two different nonlinear scoring/aggregation routes.
        primary_feedback = primary_strata[:, 1, :] + primary_strata[:, 2, :]
        primary_difference = primary_strata[:, 2, :] - primary_strata[:, 1, :]
        primary_behavior = (primary_difference if self.behavior_readout == 'separate'
                            else primary_feedback)

        if self.training:
            # Item fields share tables across current target and past events:
            # gate both uses together. Historical action joins this group.
            # Context includes only supplied non-item/non-action features.
            aux_context = context.detach() if self.block_aux_context_grad else context
            aux_target = target.detach() if self.block_aux_item_grad else target
            aux_values = values.detach() if self.block_aux_item_grad else values
            aux_weights = (self._auxiliary_activation(self._interactions(aux_target, aux_values), active)
                           + correction.detach())
            aux_transported = transported.detach() if self.block_aux_item_grad else transported
            aux_strata, aux_readout = self._pool(aux_weights, aux_transported, counts)
        else:
            # Identical values/parameters and no state update: inference needs
            # just one scorer pass for every factorial cell.
            aux_context, aux_target = context, target
            aux_strata, aux_readout = primary_strata, primary_readout
        feedback = aux_strata[:, 1, :] + aux_strata[:, 2, :]
        feedback_readout = aux_readout[:, 1, :] + aux_readout[:, 2, :]
        feedback_difference = aux_strata[:, 2, :] - aux_strata[:, 1, :]
        feedback_behavior = (feedback_difference if self.behavior_readout == 'separate'
                             else feedback)
        predictions = {}
        for label, tower, head, identity_head in zip(
                self.feature_map.labels, self.towers, self.residual_heads,
                self.identity_heads):
            if label == 'is_click':
                features = torch.cat([context, target, uniform], dim=-1)
                extra = (click_readout, primary_feedback, primary_behavior)
                identity_input = torch.cat([context, target, identity], dim=-1)
            else:
                features = torch.cat([aux_context, aux_target, feedback], dim=-1)
                extra = (feedback_readout, feedback, feedback_behavior)
                identity_input = torch.cat([aux_context, aux_target, aux_identity], dim=-1)
            logits = (tower(features) + head(torch.cat((features,) + extra, dim=-1))
                      + identity_head(identity_input))
            predictions[label + '_pred'] = torch.sigmoid(logits)
        return predictions
