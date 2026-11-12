"""Prospective target-match by block-order interaction replacement study.

Keep all four incumbent summaries and append one residualized joint moment.
Ordered/scrambled interactions and two duplicate directions share support,
coefficient norm, allocation and contraction shape. Ordinals are released
array positions, not elapsed time. Performance and novelty remain unverified.
"""
import torch
from torch import nn
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbeddingDict
from unirank.pytorch.layers.activations import Dice


CONFIG = {'embedding_dim': 16, 'attention_hidden_units': [64, 32], 'tower_hidden_units': [128, 64], 'attention_normalization': 'none', 'compression_routes': 'uf', 'block_aux_scorer_grad': True, 'block_aux_context_grad': False, 'block_aux_item_grad': True, 'residual_input': 'match_detail', 'residual_routing': 'task', 'residual_hidden': 16, 'temporal_detail': 'position', 'block_detail': 'ordinal', 'curvature_detail': 'quadratic', 'joint_detail': 'duplicate_match'}


class Model(MultiTaskModel):
    def __init__(self, feature_map, model_id='TargetMatchOrderMoment', gpu=-1,
                 embedding_dim=16, attention_hidden_units=(64, 32),
                 tower_hidden_units=(128, 64), attention_normalization='none',
                 compression_routes='uf', block_aux_scorer_grad=True,
                 block_aux_context_grad=False, block_aux_item_grad=True, num_tasks=None,
                 residual_input='match_detail', residual_routing='task', residual_hidden=16,
                 temporal_detail='position', block_detail='ordinal',
                 curvature_detail='quadratic', joint_detail='duplicate_match',
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
        if block_aux_context_grad is not False or block_aux_item_grad is not True:
            raise ValueError('The incumbent source-path ownership is required')
        if residual_input != 'match_detail':
            raise ValueError('The original match-detail channel must be retained')
        if temporal_detail != 'position':
            raise ValueError('The incumbent within-block ordinal moment is frozen')
        if block_detail != 'ordinal':
            raise ValueError('The incumbent coarse-linear summary is frozen')
        if curvature_detail != 'quadratic':
            raise ValueError('The incumbent quadratic fourth summary is frozen')
        if joint_detail not in ('ordered', 'scrambled', 'duplicate_match', 'duplicate_curvature'):
            raise ValueError('Unknown fifth-summary replacement')
        if residual_routing != 'task':
            raise ValueError('This replacement study freezes task-specific routing')
        if residual_routing not in ('task', 'shared'):
            raise ValueError('Unknown residual parameter-sharing control')
        if not isinstance(residual_hidden, int) or residual_hidden < 1:
            raise ValueError('Residual width must be a positive integer')
        self.residual_input = residual_input
        self.residual_routing = residual_routing
        self.temporal_detail = temporal_detail
        self.block_detail = block_detail
        self.curvature_detail = curvature_detail
        self.joint_detail = joint_detail
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
        # Only supplied encoded fields are compared; no external lookups.
        # Zero and declared special IDs never constitute positive evidence.
        self.match_fields = ('item_id', 'cate_id', 'brand')
        self.match_excluded_ids = {}
        for name in self.match_fields:
            if name not in self.item_fields:
                raise ValueError('Required target-match field is absent: ' + name)
            spec = feature_map.features[name]
            excluded = {0}
            for special in ('padding_idx', 'oov_idx'):
                if spec.get(special) is not None:
                    excluded.add(int(spec[special]))
            self.match_excluded_ids[name] = tuple(sorted(excluded))

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
        # No change to inherited modules, their initialization or host RNG.
        # Zero final layers give source-equivalent predictions initially.
        with torch.random.fork_rng(devices=[]):
            self.residual_bank = nn.ModuleList([
                nn.Sequential(nn.Linear(input_width + len(self.match_fields) + 5 * self.value_width,
                                        residual_hidden), nn.ReLU(),
                              nn.Linear(residual_hidden, self.value_width))
                for _ in feature_map.labels
            ])
            for adapter in self.residual_bank:
                nn.init.zeros_(adapter[-1].weight)
                nn.init.zeros_(adapter[-1].bias)
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
        detail, position_detail, block_detail, curvature_detail, joint_detail, match_counts = self._target_match_summary(
            item, values, mask)
        return context, target, values, mask, auxiliary, detail, position_detail, block_detail, curvature_detail, joint_detail, match_counts

    def _target_match_summary(self, item, values, mask):
        """Five masked value summaries; only the fifth varies across cells.

        Quadratic/scrambled coefficients are orthogonal to valid constants
        and coarse linear ranks under event-count weighting. A common gate
        excludes fewer than three occupied blocks and degenerate residuals.
        All fourth coefficients match the inherited coarse coefficient norm.
        Duplicate coefficients copy that coarse vector and add no new moment.
        FP32 coefficient construction precedes the inherited bf16 contraction.
        """
        matches = []
        for name in self.match_fields:
            ids = item[name]
            if ids.ndim != 2 or ids.shape[1] != 101 or ids.shape[0] != mask.shape[0]:
                raise ValueError('Expected target-match IDs with shape [B,101]')
            history_ids, target_id = ids[:, :-1], ids[:, -1:]
            valid = mask.bool() & (history_ids > 0) & (target_id > 0)
            for special in self.match_excluded_ids[name]:
                valid = valid & (history_ids != special) & (target_id != special)
            matches.append(valid & (history_ids == target_id))
        match_matrix = torch.stack(matches, dim=-1).to(torch.float32)
        match_counts = torch.log1p(match_matrix.sum(dim=1))
        batch, length, width = values.shape
        if length != 100:
            raise ValueError('Frozen target-match compression requires 100 history positions')
        valid = mask.bool().reshape(batch, 10, 10).to(torch.float32)
        counts = valid.sum(dim=-1, keepdim=True)
        affinity = match_matrix.mean(dim=-1).reshape(batch, 10, 10)
        centered = affinity - valid * (affinity.sum(dim=-1, keepdim=True)
                                      / counts.clamp_min(1))
        # Retain the incumbent within-block ordinal moment without changes.
        position = torch.arange(10, device=values.device, dtype=torch.float32).view(1, 1, 10) / 9.0
        position_sum = (position * valid).sum(dim=-1, keepdim=True)
        centered_position = valid * (position - position_sum / counts.clamp_min(1))
        # Coarse ordinals add only information missing under whole-block
        # permutations. Center using event counts, including partial blocks.
        total = counts.sum(dim=1).clamp_min(1)
        block_rank = torch.arange(10, device=values.device, dtype=torch.float32).view(1, 10, 1) / 9.0
        block_mean = (block_rank * valid).sum(dim=(1, 2), keepdim=True) / total.unsqueeze(-1)
        centered_block = valid * (block_rank - block_mean)
        local_norm2 = centered_position.square().sum(dim=(1, 2), keepdim=True)
        block_norm2 = centered_block.square().sum(dim=(1, 2), keepdim=True)
        centered_block = (centered_block * (local_norm2 / block_norm2.clamp_min(1e-12)).sqrt()
                          * (block_norm2 > 1e-12).to(valid.dtype))
        # Retain the inherited coarse coefficient construction, including its
        # unused local duplicate, identically in all fourth-channel cells.
        duplicated_position = centered_position * (block_norm2 > 1e-12).to(valid.dtype)
        block_coefficients = (centered_block if self.block_detail == 'ordinal'
                              else duplicated_position)
        # Work with the unscaled centered block rank. Its constant and
        # linear span is independent of the incumbent norm matching above.
        linear = valid * (block_rank - block_mean)
        linear_norm2 = linear.square().sum(dim=(1, 2), keepdim=True)
        quadratic_raw = (block_rank - block_mean).square()
        # One fixed, predeclared permutation; no sample identities or RNG.
        scrambled_rank = torch.tensor(
            [0, 7, 2, 9, 4, 1, 8, 3, 6, 5],
            device=values.device, dtype=torch.float32).view(1, 10, 1) / 9.0
        scrambled_raw = scrambled_rank.square()
        quadratic, quadratic_norm2 = self._orthogonal_block_residual(
            quadratic_raw, linear, linear_norm2, valid, total)
        scrambled, scrambled_norm2 = self._orthogonal_block_residual(
            scrambled_raw, linear, linear_norm2, valid, total)
        occupied = (counts > 0).sum(dim=1, keepdim=True)
        support = ((occupied >= 3) & (quadratic_norm2 > 1e-10)
                   & (scrambled_norm2 > 1e-10)).to(valid.dtype)
        target_norm2 = centered_block.square().sum(dim=(1, 2), keepdim=True)
        quadratic = (quadratic * (target_norm2 / quadratic_norm2.clamp_min(1e-10)).sqrt()
                     * support)
        scrambled = (scrambled * (target_norm2 / scrambled_norm2.clamp_min(1e-10)).sqrt()
                     * support)
        duplicate = centered_block * support
        if self.curvature_detail == 'quadratic':
            fourth = quadratic
        elif self.curvature_detail == 'scrambled':
            fourth = scrambled
        else:
            fourth = duplicate
        fifth = self._joint_coefficients(
            centered, centered_position, fourth, valid, counts,
            block_rank, scrambled_rank)
        coefficients = torch.stack(
            (centered, centered_position, block_coefficients, fourth, fifth), dim=-2)
        summaries = torch.bmm(
            coefficients.to(values.dtype).reshape(batch * 10, 5, 10),
            values.reshape(batch * 10, 10, width))
        summaries = summaries.reshape(batch, 10, 5, width).sum(dim=1)
        summaries = summaries / total.sqrt().unsqueeze(-1).to(summaries.dtype)
        return (summaries[:, 0, :], summaries[:, 1, :], summaries[:, 2, :],
                summaries[:, 3, :], summaries[:, 4, :], match_counts)

    def _joint_coefficients(self, affinity, position, fourth, valid, counts,
                            rank, scrambled_rank):
        """Explicit match/order cross-moment beyond the marginal coefficients.

        All basis construction is FP32 and independent of learned values.
        Within-block centering gives orthogonality to every block-constant
        coefficient, including the inherited linear and quadratic moments.
        Two projection sweeps reduce finite-precision leakage into the
        remaining local-ordinal and target-match directions. The common
        support gate is shared even by duplicates; it can itself carry new
        availability information, so duplicates isolate direction beyond it.
        """
        position_norm2 = position.square().sum(dim=(1, 2), keepdim=True)
        match_norm2 = affinity.square().sum(dim=(1, 2), keepdim=True)
        fourth_norm2 = fourth.square().sum(dim=(1, 2), keepdim=True)
        # Orthogonalize affinity against position first. A zero basis is
        # harmless; no inverse of a near-singular two-by-two matrix is used.
        basis = affinity
        for _ in range(2):
            slope = ((basis * position).sum(dim=(1, 2), keepdim=True)
                     / position_norm2.clamp_min(1e-10))
            basis = valid * (basis - slope * position)
            basis = valid * (basis - basis.sum(dim=-1, keepdim=True)
                             / counts.clamp_min(1))
        basis_norm2 = basis.square().sum(dim=(1, 2), keepdim=True)
        basis = basis * (basis_norm2 > 1e-10).to(valid.dtype)
        basis_norm2 = basis.square().sum(dim=(1, 2), keepdim=True)

        ordered = self._joint_residual(
            affinity * rank, position, position_norm2,
            basis, basis_norm2, valid, counts)
        scrambled = self._joint_residual(
            affinity * scrambled_rank, position, position_norm2,
            basis, basis_norm2, valid, counts)
        ordered_norm2 = ordered.square().sum(dim=(1, 2), keepdim=True)
        scrambled_norm2 = scrambled.square().sum(dim=(1, 2), keepdim=True)
        support = ((ordered_norm2 > 1e-10) & (scrambled_norm2 > 1e-10)
                   & (position_norm2 > 1e-10) & (match_norm2 > 1e-10)
                   & (fourth_norm2 > 1e-10)).to(valid.dtype)
        ordered = ordered * (position_norm2 / ordered_norm2.clamp_min(1e-10)).sqrt() * support
        scrambled = scrambled * (position_norm2 / scrambled_norm2.clamp_min(1e-10)).sqrt() * support
        duplicate_match = affinity * (position_norm2 / match_norm2.clamp_min(1e-10)).sqrt() * support
        duplicate_curvature = fourth * (position_norm2 / fourth_norm2.clamp_min(1e-10)).sqrt() * support
        if self.joint_detail == 'ordered':
            return ordered
        if self.joint_detail == 'scrambled':
            return scrambled
        if self.joint_detail == 'duplicate_match':
            return duplicate_match
        return duplicate_curvature

    @staticmethod
    def _joint_residual(raw, position, position_norm2, basis, basis_norm2, valid, counts):
        residual = valid * (raw - (raw * valid).sum(dim=-1, keepdim=True)
                            / counts.clamp_min(1))
        for _ in range(2):
            slope = ((residual * position).sum(dim=(1, 2), keepdim=True)
                     / position_norm2.clamp_min(1e-10))
            residual = valid * (residual - slope * position)
            slope = ((residual * basis).sum(dim=(1, 2), keepdim=True)
                     / basis_norm2.clamp_min(1e-10))
            residual = valid * (residual - slope * basis)
            residual = valid * (residual - residual.sum(dim=-1, keepdim=True)
                                / counts.clamp_min(1))
        return residual

    @staticmethod
    def _orthogonal_block_residual(raw, linear, linear_norm2, valid, total):
        mean = (raw * valid).sum(dim=(1, 2), keepdim=True) / total.unsqueeze(-1)
        residual = valid * (raw - mean)
        slope = ((residual * linear).sum(dim=(1, 2), keepdim=True)
                 / linear_norm2.clamp_min(1e-12))
        residual = valid * (residual - slope * linear)
        # Recenter once more to reduce finite-precision constant leakage.
        residual_mean = residual.sum(dim=(1, 2), keepdim=True) / total.unsqueeze(-1)
        residual = valid * (residual - residual_mean)
        return residual, residual.square().sum(dim=(1, 2), keepdim=True)

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
        context, target, values, mask, auxiliary, detail, position_detail, block_detail, curvature_detail, joint_detail, match_counts = self._represent(inputs)
        values, counts = self._compress_both(values, mask, auxiliary)
        # Both scorer calls include all 30 slots: BatchNorm sees the identical
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
            aux_detail = detail.detach() if self.block_aux_item_grad else detail
            aux_position_detail = position_detail.detach() if self.block_aux_item_grad else position_detail
            aux_block_detail = block_detail.detach() if self.block_aux_item_grad else block_detail
            aux_curvature_detail = curvature_detail.detach() if self.block_aux_item_grad else curvature_detail
            aux_joint_detail = joint_detail.detach() if self.block_aux_item_grad else joint_detail
            aux_weights = self._auxiliary_activation(self._interactions(aux_target, aux_values))
            aux_strata = self._pool(aux_weights, aux_values, counts)
        else:
            # Identical values/parameters and no state update: inference needs
            # just one scorer pass for every replacement cell.
            aux_context, aux_target = context, target
            aux_detail = detail
            aux_position_detail = position_detail
            aux_block_detail = block_detail
            aux_curvature_detail = curvature_detail
            aux_joint_detail = joint_detail
            aux_strata = primary_strata
        feedback = aux_strata[:, 1, :] + aux_strata[:, 2, :]
        grouped_counts = counts.reshape(counts.shape[0], 10, 3)
        total_count = grouped_counts[:, :, 0].sum(dim=1, keepdim=True)
        has_history = (total_count > 0).to(uniform.dtype)
        predictions = {}
        for index, (label, tower) in enumerate(zip(self.feature_map.labels, self.towers)):
            if label == 'is_click':
                task_context, task_target, base = context, target, uniform
                task_detail = detail
                task_position_detail = position_detail
                task_block_detail = block_detail
                task_curvature_detail = curvature_detail
                task_joint_detail = joint_detail
            else:
                task_context, task_target, base = aux_context, aux_target, feedback
                task_detail = aux_detail
                task_position_detail = aux_position_detail
                task_block_detail = aux_block_detail
                task_curvature_detail = aux_curvature_detail
                task_joint_detail = aux_joint_detail
            signal = torch.cat([task_detail, task_position_detail, task_block_detail,
                                task_curvature_detail, task_joint_detail], dim=-1)
            residual_features = torch.cat([task_context, task_target, base,
                                           match_counts.to(base.dtype), signal], dim=-1)
            # Equal bank execution in all cells; no task-specific scorer rows
            # or extra stateful normalization is introduced.
            bank = torch.stack([adapter(residual_features)
                                for adapter in self.residual_bank], dim=1)
            correction = bank[:, index, :] if self.residual_routing == 'task' else bank.mean(dim=1)
            pooled = base + has_history * correction
            features = torch.cat([task_context, task_target, pooled], dim=-1)
            predictions[label + '_pred'] = torch.sigmoid(tower(features))
        return predictions
