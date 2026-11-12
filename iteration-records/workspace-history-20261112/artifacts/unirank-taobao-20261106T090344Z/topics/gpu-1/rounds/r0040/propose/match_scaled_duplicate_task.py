"""Duplicate one observed match-action summary into both action slots.

Raw/raw and scaled/scaled controls retain action content in two active slots
without retaining two distinct scales/support patterns. Cross these controls
with task/shared residual routing. Selected action-input derivatives are
stopped; pool, ordinal, item, scorer and auxiliary ownership are inherited.
Existing observed raw/scaled endpoints remain available as exact bridges.
This is a mechanism attribution study, without an architecture novelty claim.
"""
import torch
from torch import nn
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbeddingDict
from unirank.pytorch.layers.activations import Dice


CONFIG = {'embedding_dim': 16, 'attention_hidden_units': [64, 32], 'tower_hidden_units': [128, 64], 'attention_normalization': 'none', 'compression_routes': 'uf', 'block_aux_scorer_grad': True, 'block_aux_context_grad': False, 'block_aux_item_grad': True, 'residual_input': 'match_detail', 'residual_routing': 'task', 'residual_hidden': 16, 'temporal_detail': 'position', 'block_detail': 'ordinal', 'curvature_detail': 'quadratic', 'joint_detail': 'duplicate_match', 'match_support': 'available', 'match_scaling': 'norm', 'coarse_extension': 'duplicate', 'primary_partition': 'uniform', 'normalization_support': 'rotated_duplicate', 'support_gradient': 'attached', 'action_in_scorer': False, 'action_in_pool': True, 'action_in_summaries': True, 'summary_action_group': 'all', 'summary_action_alignment': 'rotated', 'summary_rotation_group': 'none', 'match_action_availability': 'both', 'match_action_content': 'scaled_duplicate', 'match_replacement_gradient': 'blocked', 'match_observed_gradient': 'blocked'}


class Model(MultiTaskModel):
    def __init__(self, feature_map, model_id='TargetMatchOrderMoment', gpu=-1,
                 embedding_dim=16, attention_hidden_units=(64, 32),
                 tower_hidden_units=(128, 64), attention_normalization='none',
                 compression_routes='uf', block_aux_scorer_grad=True,
                 block_aux_context_grad=False, block_aux_item_grad=True, num_tasks=None,
                 residual_input='match_detail', residual_routing='task', residual_hidden=16,
                 temporal_detail='position', block_detail='ordinal',
                 curvature_detail='quadratic', joint_detail='duplicate_match',
                 match_support='available', match_scaling='norm', coarse_extension='duplicate',
                 primary_partition='uniform', normalization_support='rotated_duplicate',
                 support_gradient='attached', action_in_scorer=False,
                 action_in_pool=True, action_in_summaries=True,
                 summary_action_group='all', summary_action_alignment='rotated',
                 summary_rotation_group='none', match_action_availability='both',
                 match_action_content='scaled_duplicate',
                 match_replacement_gradient='blocked',
                 match_observed_gradient='blocked',
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
        if joint_detail != 'duplicate_match':
            raise ValueError('This factorial retains the fifth match direction')
        if match_support not in ('available', 'joint'):
            raise ValueError('Unknown fifth match-support control')
        if match_scaling not in ('norm', 'raw'):
            raise ValueError('Unknown fifth match-scaling control')
        if residual_routing not in ('task', 'shared'):
            raise ValueError('Unknown residual parameter-sharing control')
        if not isinstance(residual_hidden, int) or residual_hidden < 1:
            raise ValueError('Residual width must be a positive integer')
        if action_in_scorer is not False:
            raise ValueError('This study freezes scorer action coordinates to zero')
        if not isinstance(action_in_pool, bool) or not isinstance(action_in_summaries, bool):
            raise ValueError('Action value-path switches must be booleans')
        self.action_in_scorer = action_in_scorer
        self.action_in_pool = action_in_pool
        self.action_in_summaries = action_in_summaries
        if action_in_summaries is not True:
            raise ValueError('Select summary action content with summary_action_group')
        # Submitted cells restore pooled action; False is an ancestor bridge.
        # Keep the Boolean pool switch independent of the family selector.
        if summary_action_group not in ('match', 'ordinal', 'all', 'none'):
            raise ValueError('Unknown summary action group')
        self.summary_action_group = summary_action_group
        if summary_action_alignment not in ('aligned', 'rotated'):
            raise ValueError('Unknown within-block summary action alignment')
        self.summary_action_alignment = summary_action_alignment
        if summary_rotation_group not in ('match', 'ordinal', 'all', 'none', 'raw_match', 'scaled_match'):
            raise ValueError('Unknown rotated summary action family')
        self.summary_rotation_group = summary_rotation_group
        if match_action_availability not in ('both', 'raw', 'scaled', 'none'):
            raise ValueError('Unknown match-summary action availability')
        self.match_action_availability = match_action_availability
        if match_action_content not in ('observed', 'item_duplicate',
                                         'raw_observed', 'scaled_observed',
                                         'raw_duplicate', 'scaled_duplicate'):
            raise ValueError('Unknown match-action content replacement')
        self.match_action_content = match_action_content
        if match_replacement_gradient not in ('attached', 'blocked'):
            raise ValueError('Unknown match-replacement gradient control')
        self.match_replacement_gradient = match_replacement_gradient
        if match_observed_gradient not in ('attached', 'blocked'):
            raise ValueError('Unknown observed match-action gradient control')
        self.match_observed_gradient = match_observed_gradient
        self.residual_input = residual_input
        self.residual_routing = residual_routing
        self.temporal_detail = temporal_detail
        self.block_detail = block_detail
        self.curvature_detail = curvature_detail
        self.joint_detail = joint_detail
        self.match_support = match_support
        self.match_scaling = match_scaling
        if match_support != 'available' or match_scaling != 'norm':
            raise ValueError('This study freezes the incumbent fifth support and scale')
        if coarse_extension not in ('cubic', 'scrambled', 'duplicate', 'bridge'):
            raise ValueError('Unknown coarse ordinal replacement')
        self.coarse_extension = coarse_extension
        if primary_partition not in ('aligned', 'rotated', 'uniform'):
            raise ValueError('Unknown primary partition readout')
        if primary_partition != 'uniform':
            raise ValueError('This study fixes the primary uniform readout')
        self.primary_partition = primary_partition
        if normalization_support not in ('aligned_rotated', 'rotated_duplicate'):
            raise ValueError('Unknown Dice support population')
        if support_gradient not in ('attached', 'blocked'):
            raise ValueError('Unknown support preactivation gradient')
        self.normalization_support = normalization_support
        self.support_gradient = support_gradient
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
        embedding_item = item
        # The pool requires observed actions even for the none-summary
        # endpoint. With pool off and family none, preserve the inherited
        # zero-token lookup and disconnected action-parameter semantics.
        needs_action = self.action_in_pool or self.summary_action_group != 'none'
        if not needs_action:
            embedding_item = dict(item)
            embedding_item[self.action_field] = torch.zeros_like(item[self.action_field])
        item_emb = self.embedding_layer(embedding_item)
        if not needs_action:
            item_emb[self.action_field] = torch.zeros_like(item_emb[self.action_field])
        target = torch.cat([item_emb[name][:, -1] for name in self.item_fields], dim=-1)
        keys = torch.cat([item_emb[name][:, :-1] for name in self.item_fields], dim=-1)
        # Current action is unknown and never enters either summary or scorer.
        values = torch.cat([keys, item_emb[self.action_field][:, :-1]], dim=-1)
        context = torch.cat([user_emb[name] for name in self.user_fields], dim=-1)
        # Only the summary consumer sees reindexed action coordinates.
        # The observed values returned below still feed the unchanged pool
        # and scorer. Neither item IDs nor original task masks are reindexed.
        summary_values = self._summary_values(values, mask)
        detail, position_detail, block_detail, curvature_detail, joint_detail, match_counts, match_union = self._target_match_summary(
            item, self._action_channel(summary_values, self.action_in_summaries), mask,
            self._action_channel(values, self.action_in_summaries))
        return context, target, values, mask, auxiliary, detail, position_detail, block_detail, curvature_detail, joint_detail, match_counts, match_union

    def _summary_values(self, values, mask):
        """Gather action embeddings by valid rank within each ten-event block.

        The rotated cell shifts by floor(n_valid/2); aligned uses zero shift.
        Invalid positions retain their old values and cannot be gather sources
        for valid destinations. No action token crosses a block or sample.
        Empty and singleton blocks are identities. The gather is bijective on
        valid positions, so action multisets and their sums are preserved per
        block. Original item values, masks and action lookup are never changed.
        """
        batch, length, width = values.shape
        if length != 100 or mask.shape != values.shape[:2]:
            raise ValueError('Expected one mask per frozen history position')
        valid = mask.bool().reshape(batch, 10, 10)
        ranks = (valid.long().cumsum(dim=-1) - 1).clamp_min(0)
        counts = valid.long().sum(dim=-1, keepdim=True)
        positions = torch.arange(10, device=values.device, dtype=torch.long).view(1, 1, 10)
        packed_positions = torch.zeros_like(ranks).scatter_add(
            -1, ranks, positions * valid.long())
        shift = (counts // 2) * int(self.summary_action_alignment == 'rotated')
        source_rank = (ranks + shift).remainder(counts.clamp_min(1))
        source_position = packed_positions.gather(-1, source_rank)
        actions = values[:, :, self.key_width:].reshape(batch, 10, 10, width - self.key_width)
        gathered = actions.gather(
            2, source_position.unsqueeze(-1).expand(-1, -1, -1, width - self.key_width))
        actions = torch.where(valid.unsqueeze(-1), gathered, actions)
        return torch.cat([values[:, :, :self.key_width],
                          actions.reshape(batch, length, width - self.key_width)], dim=-1)

    def _target_match_summary(self, item, values, mask, aligned_values):
        """Five masked value summaries; only the fifth direction/gate varies.

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
        incumbent_fifth = self._joint_coefficients(
            centered, centered_position, fourth, valid, counts,
            block_rank, scrambled_rank)
        fifth = self._coarse_extension_coefficients(
            incumbent_fifth, centered_position, fourth, linear,
            block_rank, scrambled_rank, valid, counts, total)
        coefficients = torch.stack(
            (centered, centered_position, block_coefficients, fourth, fifth), dim=-2)
        summaries = torch.bmm(
            coefficients.to(values.dtype).reshape(batch * 10, 5, 10),
            values.reshape(batch * 10, 10, width))
        summaries = summaries.reshape(batch, 10, 5, width).sum(dim=1)
        summaries = summaries / total.sqrt().unsqueeze(-1).to(summaries.dtype)
        # Execute the same full-width contraction for the aligned stream.
        # Both contractions and the gather execute in every new factorial cell.
        # No coefficients, reductions or inherited rotated item values change.
        aligned_summaries = torch.bmm(
            coefficients.to(values.dtype).reshape(batch * 10, 5, 10),
            aligned_values.reshape(batch * 10, 10, width))
        aligned_summaries = aligned_summaries.reshape(batch, 10, 5, width).sum(dim=1)
        aligned_summaries = aligned_summaries / total.sqrt().unsqueeze(-1).to(aligned_summaries.dtype)
        summaries = self._select_summary_alignment(summaries, aligned_summaries)
        summaries = self._summary_action_channels(summaries)
        return (summaries[:, 0, :], summaries[:, 1, :], summaries[:, 2, :],
                summaries[:, 3, :], summaries[:, 4, :], match_counts,
                match_matrix.bool().any(dim=-1))

    def _select_summary_alignment(self, rotated, aligned):
        """Select alignment, then gate only raw/scaled match action rows.

        The availability switch is independent of the legacy alignment
        switch. Submitted cells retain alignment and vary availability.

        Raw row 0 and scaled/gated row 4 can be rotated independently.
        Match selects both, none neither; ordinal rows stay aligned in the
        four submitted cells. Legacy selectors remain for host bridges.
        Ordinal rows 1/2/3 use local, coarse-linear and quadratic directions.
        Coarse directions are invariant to within-block rotation in exact
        arithmetic; host checks must bound the finite-precision residual.
        Item coordinates always retain the inherited rotated contraction.
        The all/none endpoints execute this identical graph for bridge checks.
        """
        groups = {
            # Separate a raw match direction from its inherited scaled,
            # support-gated copy. No coefficient or learned width changes.
            'raw_match': (True, False, False, False, False),
            'scaled_match': (False, False, False, False, True),
            'match': (True, False, False, False, True),
            'ordinal': (False, True, True, True, False),
            'all': (True, True, True, True, True),
            'none': (False, False, False, False, False),
        }
        active = torch.tensor(groups[self.summary_rotation_group],
                              device=rotated.device, dtype=torch.bool).view(1, 5, 1)
        actions = torch.where(active, rotated[:, :, self.key_width:],
                              aligned[:, :, self.key_width:])
        # Replacement reads only already-masked item summaries. The observed
        # branch is an identity, preserving the exact ancestor graph. Apply
        # availability afterwards so a disabled row can never be restored.
        actions = self._match_action_content(rotated[:, :, :self.key_width], actions)
        # Zero only the selected match-action inputs after both full-width
        # contractions. Item and ordinal paths and observed pool values stay
        # intact. Identical gate shapes execute even for the all-kept cell.
        keep = {
            'both': (True, True, True, True, True),
            'raw': (True, True, True, True, False),
            'scaled': (False, True, True, True, True),
            'none': (False, True, True, True, False),
        }
        available = torch.tensor(keep[self.match_action_availability],
                                 device=actions.device, dtype=torch.bool).view(1, 5, 1)
        actions = torch.where(available, actions, torch.zeros_like(actions))
        return torch.cat([rotated[:, :, :self.key_width], actions], dim=-1)

    def _match_action_content(self, item_summaries, actions):
        """Replace or duplicate match-action rows, preserving all other inputs.

        raw_duplicate copies observed row 0 to both rows 0/4;
        scaled_duplicate copies observed row 4 to both rows 0/4. Their local
        derivatives obey match_observed_gradient; match_replacement_gradient
        applies only to the inherited item-derived replacement modes below.

        raw_observed replaces scaled row 4; scaled_observed replaces raw
        row 0. Observed/attached is the exact ancestor identity endpoint;
        observed/blocked retains its values but stops only rows 0/4. The fixed
        FP32 sum/sqrt(F) adds no input information, parameters or randomness.
        It changes active inputs and gradient geometry; it does not match
        action information rank or variance. Blocked removes only the extra
        replacement derivative into item summaries; ordinary item paths stay
        live. Attached retains the original derivative. Original row
        availability gates execute AFTER replacement. Rows 1/2/3 are identity.

        For mixed content, stop only the retained observed match row when
        requested. Earlier mixed-content runs used attached derivatives;
        the full observed/item_duplicate endpoints keep their exact code paths.
        """
        # Copy a single already-masked, already-contracted action summary
        # into both match-action slots. No item content, fitting, norm
        # adjustment, new state or RNG enters this replacement. The source
        # is aligned in submitted cells (summary_rotation_group='none').
        # Stop the source derivative for BOTH selected outputs; pool and
        # ordinal action paths remain live. The original availability and
        # summary-family gates still execute after this helper.
        if self.match_action_content in ('raw_duplicate', 'scaled_duplicate'):
            source_row = 0 if self.match_action_content == 'raw_duplicate' else 4
            duplicate = actions[:, source_row:source_row + 1, :]
            if self.match_observed_gradient == 'blocked':
                duplicate = duplicate.detach()
            selected = torch.tensor((True, False, False, False, True),
                                    device=actions.device, dtype=torch.bool).view(1, 5, 1)
            return torch.where(selected, duplicate, actions)
        if self.match_action_content == 'observed':
            if self.match_observed_gradient == 'attached':
                return actions
            # Stop only the residual consumer's raw/scaled match-action
            # inputs. Direct pools, ordinal rows and item summaries retain
            # their original paths. No in-place update or value change.
            selected = torch.tensor((True, False, False, False, True),
                                    device=actions.device, dtype=torch.bool).view(1, 5, 1)
            return torch.where(selected, actions.detach(), actions)
        batch, rows, action_width = actions.shape
        fields = len(self.item_fields)
        duplicate = item_summaries.reshape(batch, rows, fields, action_width)
        duplicate = duplicate.float().sum(dim=2) / (fields ** 0.5)
        # Stop only the added replacement-to-item-summary path. The ordinary
        # item summary returned by the caller remains attached and unchanged.
        # Detaching preserves exact forward values, dtype and reduction order.
        if self.match_replacement_gradient == 'blocked':
            duplicate = duplicate.detach()
        replace_rows = {
            'item_duplicate': (True, False, False, False, True),
            'raw_observed': (False, False, False, False, True),
            'scaled_observed': (True, False, False, False, False),
        }
        selected = torch.tensor(replace_rows[self.match_action_content],
                                device=actions.device, dtype=torch.bool).view(1, 5, 1)
        if (self.match_observed_gradient == 'blocked'
                and self.match_action_content in ('raw_observed', 'scaled_observed')):
            retained_rows = {
                'raw_observed': (True, False, False, False, False),
                'scaled_observed': (False, False, False, False, True),
            }
            retained = torch.tensor(retained_rows[self.match_action_content],
                                    device=actions.device, dtype=torch.bool).view(1, 5, 1)
            actions = torch.where(retained, actions.detach(), actions)
        return torch.where(selected, duplicate.to(actions.dtype), actions)

    def _coarse_extension_coefficients(self, incumbent, position, quadratic,
                                       linear, rank, scrambled_rank,
                                       valid, counts, total):
        """Replace the redundant fifth direction at fixed learned capacity.

        Event-weighted projection uses the actual validity mask, including
        partial/noncontiguous blocks. Every candidate coefficient is block
        constant on valid events, hence orthogonal to the first target-match
        and second local-position coefficients as well. A common gate and
        norm make information content the main replacement contrast. The
        gate can itself carry information; the duplicate isolates that.
        """
        cubic = self._coarse_residual(
            rank * rank * rank, linear, quadratic, valid, total)
        scrambled = self._coarse_residual(
            scrambled_rank * scrambled_rank * scrambled_rank,
            linear, quadratic, valid, total)
        cubic_norm2 = cubic.square().sum(dim=(1, 2), keepdim=True)
        scrambled_norm2 = scrambled.square().sum(dim=(1, 2), keepdim=True)
        incumbent_norm2 = incumbent.square().sum(dim=(1, 2), keepdim=True)
        target_norm2 = position.square().sum(dim=(1, 2), keepdim=True)
        occupied = (counts > 0).sum(dim=1, keepdim=True)
        support = ((occupied >= 4) & (cubic_norm2 > 1e-10)
                   & (scrambled_norm2 > 1e-10)
                   & (incumbent_norm2 > 1e-10)
                   & (target_norm2 > 1e-10)).to(valid.dtype)
        cubic = (cubic * (target_norm2 / cubic_norm2.clamp_min(1e-10)).sqrt()
                 * support)
        scrambled = (scrambled
                     * (target_norm2 / scrambled_norm2.clamp_min(1e-10)).sqrt()
                     * support)
        duplicate = incumbent * support
        # All alternatives above execute in all cells; no modules or RNG added.
        if self.coarse_extension == 'cubic':
            return cubic
        if self.coarse_extension == 'scrambled':
            return scrambled
        if self.coarse_extension == 'duplicate':
            return duplicate
        return incumbent

    @staticmethod
    def _coarse_residual(raw, linear, quadratic, valid, total):
        """Two projection sweeps in FP32; zero bases are harmless."""
        residual = valid * raw
        linear_norm2 = linear.square().sum(dim=(1, 2), keepdim=True)
        quadratic_norm2 = quadratic.square().sum(dim=(1, 2), keepdim=True)
        for _ in range(2):
            mean = residual.sum(dim=(1, 2), keepdim=True) / total.unsqueeze(-1)
            residual = valid * (residual - mean)
            slope = ((residual * linear).sum(dim=(1, 2), keepdim=True)
                     / linear_norm2.clamp_min(1e-12))
            residual = valid * (residual - slope * linear)
            slope = ((residual * quadratic).sum(dim=(1, 2), keepdim=True)
                     / quadratic_norm2.clamp_min(1e-12))
            residual = valid * (residual - slope * quadratic)
        mean = residual.sum(dim=(1, 2), keepdim=True) / total.unsqueeze(-1)
        return valid * (residual - mean)

    def _joint_coefficients(self, affinity, position, fourth, valid, counts,
                            rank, scrambled_rank):
        """Retain the incumbent gate; vary only duplicate-match support/scale.

        All basis construction is FP32 and independent of learned values.
        Within-block centering gives orthogonality to every block-constant
        coefficient, including the inherited linear and quadratic moments.
        Two projection sweeps reduce finite-precision leakage into the
        remaining local-ordinal and target-match directions. The common
        support gate can carry availability information even in a duplicate
        direction. Build every inherited coefficient in all factorial cells;
        basic availability additionally admits matches lacking joint moments.
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
        # Gate and rescaling depend only on the authorized masked input IDs
        # and positions. No labels, learned values, new state or RNG enter.
        available = ((position_norm2 > 1e-10)
                     & (match_norm2 > 1e-10)).to(valid.dtype)
        available_norm = affinity * (position_norm2 / match_norm2.clamp_min(1e-10)).sqrt() * available
        available_raw = affinity * available
        joint_raw = affinity * support
        # All four cells build all four alternatives. The joint/norm cell
        # returns exactly the old expression for a same-seed parity bridge.
        if self.match_support == 'available':
            return available_norm if self.match_scaling == 'norm' else available_raw
        return duplicate_match if self.match_scaling == 'norm' else joint_raw

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

    def _compress_with_match(self, values, mask, auxiliary, match_union):
        """Seven rows with population-matched counts and support emptiness.

        Valid-rank rotation operates inside a block, including noncontiguous
        validity masks, and preserves target-match cardinality. Special IDs
        have already been excluded by _target_match_summary. No labels,
        external identities, timestamps, state or RNG enter this assignment.
        """
        batch, length, width = values.shape
        if length != 100:
            raise ValueError('Frozen protocol requires 100 history positions')
        valid = mask.bool().reshape(batch, 10, 10)
        feedback = auxiliary.bool().reshape(batch, 10, 10) & valid
        matched = match_union.bool().reshape(batch, 10, 10) & valid
        rotated = self._rotate_valid_membership(matched, valid)
        # Both support pairs have identical per-block cardinalities and
        # empty-slot patterns. Duplicate control removes only aligned value
        # grouping; the target-match counts and rotated pair remain present.
        first_support = (matched if self.normalization_support == 'aligned_rotated'
                         else rotated)
        assignment = torch.stack(
            (valid, valid & ~feedback, feedback,
             valid & ~first_support, first_support,
             valid & ~rotated, rotated), dim=-2)
        assignment = assignment.to(values.dtype).reshape(batch * 10, 7, 10)
        counts = assignment.sum(dim=-1)
        sums = torch.bmm(assignment, values.reshape(batch * 10, 10, width))
        means = sums / counts.clamp_min(1).unsqueeze(-1)
        return means.reshape(batch, 70, width), counts.reshape(batch, 70)

    @staticmethod
    def _rotate_valid_membership(matched, valid):
        # Pack membership by valid-event rank, rotate by half the valid
        # count, then unpack to the original valid positions. Invalid
        # entries contribute zero even when scatter indices coincide.
        matched = matched & valid
        ranks = (valid.long().cumsum(dim=-1) - 1).clamp_min(0)
        counts = valid.long().sum(dim=-1, keepdim=True)
        packed = torch.zeros_like(ranks).scatter_add(-1, ranks, matched.long())
        source = (ranks + counts // 2).remainder(counts.clamp_min(1))
        return packed.gather(-1, source).bool() & valid

    def _pool_seven(self, weights, values, counts):
        batch = values.shape[0]
        weights = weights.reshape(batch, 70) * counts.to(weights.dtype)
        weighted = (weights.unsqueeze(-1) * values).reshape(
            batch, 10, 7, self.value_width)
        return weighted.sum(dim=1)

    def _summary_action_channels(self, summaries):
        """Select action content only; preserve five full-width summary slots.

        Summary indices 0/4 carry the inherited target-match coefficients;
        1/2/3 carry local position, coarse linear and quadratic coefficients.
        The fifth match direction retains its existing support/scale gates.
        All learned modules and all coefficient computations execute unchanged.
        """
        if self.summary_action_group == 'all':
            return summaries
        if self.summary_action_group == 'match':
            keep = (1, 0, 0, 0, 1)
        elif self.summary_action_group == 'ordinal':
            keep = (0, 1, 1, 1, 0)
        else:
            keep = (0, 0, 0, 0, 0)
        # torch.where eliminates disabled gradients without rescaling the
        # surviving action content. No new learned state, RNG or scorer pass.
        active = torch.tensor(keep, device=summaries.device,
                              dtype=torch.bool).view(1, 5, 1)
        action = summaries[:, :, self.key_width:]
        action = torch.where(active, action, torch.zeros_like(action))
        return torch.cat([summaries[:, :, :self.key_width], action], dim=-1)

    def _action_channel(self, values, enabled):
        """Preserve item coordinates and optionally erase action coordinates.

        The pool and five-summary consumers use independent switches. All
        learned widths, compression populations and count weights stay fixed.
        No detach, learned module, RNG or persistent state is introduced.
        """
        if enabled:
            return values
        return torch.cat([values[:, :, :self.key_width],
                          torch.zeros_like(values[:, :, self.key_width:])], dim=-1)

    def _interactions(self, target, values):
        keys = values[:, :, :self.key_width]
        action = values[:, :, self.key_width:]
        # Preserve the original scorer width and its Dice population.
        if not self.action_in_scorer:
            action = torch.zeros_like(action)
        batch, slots, _ = keys.shape
        query = target[:, None, :].expand(-1, slots, -1)
        return torch.cat([query, keys, query - keys, query * keys, action], dim=-1).reshape(batch * slots, -1)

    def _primary_activation(self, interactions):
        """Forward-identical stop-gradient on nonreadout support rows.

        Flattened order is batch, block, seven strata. The first three strata
        retain the incumbent gradient paths. Before EACH Dice, rows 3..6
        become constants for moment estimation in the blocked cell. This
        removes their backward contribution to the preceding affine layer
        and embedding sources, while preserving their forward activations,
        batch moments and running-buffer updates at a fixed parameter state.
        Selected rows still differentiate through their own batch moments;
        shared scorer parameters and later updates can consequently differ.
        No extra scorer pass, parameters, persistent state or RNG is added.
        """
        if not self.training or self.support_gradient == 'attached':
            return self.local_activation(interactions)
        retained = (torch.arange(interactions.shape[0], device=interactions.device)
                    .remainder(7) < 3).unsqueeze(-1)
        hidden = interactions
        for layer in self.local_activation:
            if isinstance(layer, Dice):
                hidden = torch.where(retained, hidden, hidden.detach())
            hidden = layer(hidden)
        return hidden

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
        context, target, values, mask, auxiliary, detail, position_detail, block_detail, curvature_detail, joint_detail, match_counts, match_union = self._represent(inputs)
        batch = values.shape[0]
        all_values, all_counts = self._compress_with_match(values, mask, auxiliary, match_union)
        # Compress the observed stream once. Only _interactions can expose
        # its action coordinates to the scorer; only the pool projection can
        # expose them to pooling. Counts/assignments stay unchanged.
        primary_weights = self._primary_activation(self._interactions(target, all_values))
        all_strata = self._pool_seven(primary_weights, self._action_channel(all_values, self.action_in_pool), all_counts)
        uniform = all_strata[:, 0, :]
        # Auxiliary training retains the original ordered 30-slot population
        # and the original scorer/item stop-gradient ownership. No extra
        # slots enter its training Dice batch statistics. During evaluation
        # Dice acts pointwise, so the first three primary strata are reused.
        values = all_values.reshape(batch, 10, 7, self.value_width)[:, :, :3, :].reshape(
            batch, 30, self.value_width)
        counts = all_counts.reshape(batch, 10, 7)[:, :, :3].reshape(batch, 30)
        primary_strata = all_strata[:, :3, :]

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
            aux_strata = self._pool(aux_weights, self._action_channel(aux_values, self.action_in_pool), counts)
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
