"""Prospective centering-order by reduction-precision diagnostic.

One three-output identity collects cart/fav/buy product-context cotangents.
The same explicit task summation topology is used in all four cells; only
centering order and FP32/FP64 backward reduction vary. Forward architecture,
private task gradients, true labels, click graph and optimizer are inherited.
This prescribed first-order VJP is not a scalar-objective derivative, a new
forward architecture, or a verified numerical repair. Host preflight required.
"""
import torch
from torch import nn
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbeddingDict
from unirank.pytorch.layers.activations import Dice


CONFIG = {'embedding_dim': 16, 'attention_hidden_units': [64, 32], 'tower_hidden_units': [128, 64], 'attention_normalization': 'none', 'compression_routes': 'uf', 'block_aux_scorer_grad': True, 'block_aux_context_grad': False, 'block_aux_item_grad': True, 'residual_input': 'match_detail', 'residual_routing': 'task', 'residual_hidden': 16, 'temporal_detail': 'position', 'block_detail': 'ordinal', 'curvature_detail': 'quadratic', 'joint_detail': 'duplicate_match', 'match_support': 'available', 'match_scaling': 'norm', 'coarse_extension': 'duplicate', 'primary_partition': 'uniform', 'normalization_support': 'rotated_duplicate', 'support_gradient': 'attached', 'inference_support': 'fully_pruned', 'training_support': 'folded', 'residual_execution': 'direct', 'auxiliary_train_population': 'primary', 'auxiliary_eval_population': 'primary', 'block_aux_tower_context_grad': True, 'block_aux_adapter_context_grad': False, 'context_fusion': 'product', 'interaction_rank': 16, 'block_aux_residual_context_grad': True, 'block_aux_product_context_grad': False, 'product_context_assignment': 'aligned', 'product_context_component': 'centered', 'product_context_magnitude': 'native', 'product_context_orientation': 'aligned', 'parallel_orientation': 'aligned', 'orthogonal_orientation': 'aligned', 'centering_scope': 'task', 'click_context_collection': 'direct', 'auxiliary_reduction_precision': 'float32'}

# Identical true-label joint loss hook inherited from g3-r0032 sources.
AUXILIARY_OBJECTIVE = 'joint'


def _reduce_auxiliary_context(gradients, occupied, centering_scope, precision):
    """Pure reduction helper for authorized host numerical preflight.

    Inputs are ordered cart, fav, buy. Promotion occurs before any task sum
    or row reduction; casting back occurs once at the collection boundary.
    No precision change to upstream task VJPs or embedding accumulation is
    implied. Both scopes implement the same linear map in exact arithmetic.
    """
    if len(gradients) != 3:
        raise ValueError('Expected cart/fav/buy cotangents')
    if centering_scope not in ('task', 'aggregate'):
        raise ValueError('Unknown centering scope')
    if precision not in ('float32', 'float64'):
        raise ValueError('Unknown auxiliary reduction precision')
    dtype = torch.float64 if precision == 'float64' else torch.float32
    with torch.autocast(device_type=gradients[0].device.type, enabled=False):
        active = [torch.where(occupied, g.to(dtype=dtype), 0.0) for g in gradients]
        count = occupied.to(dtype=dtype).sum().clamp_min(1.0)
        if centering_scope == 'task':
            centered = [torch.where(
                occupied, g - g.sum(dim=0, keepdim=True) / count, 0.0)
                for g in active]
            transfer = (centered[0] + centered[1]) + centered[2]
        else:
            total = (active[0] + active[1]) + active[2]
            transfer = torch.where(
                occupied, total - total.sum(dim=0, keepdim=True) / count, 0.0)
    return transfer.to(dtype=gradients[0].dtype)


class _CollectedAuxiliaryTransfer(torch.autograd.Function):
    """Three identity edges; one ordered, prescribed auxiliary backward.

    Click uses the original context outside this function. Auxiliary tower
    and residual context edges remain blocked; private modules never detach.
    This fixes collection topology across the four cells. It differs from
    historical task-edge execution, so historical AUC is an anchor only.
    Unused cotangents are materialized as zeros by the autograd default.
    """
    @staticmethod
    def forward(ctx, context, occupied, centering_scope, precision):
        if context.ndim != 2 or context.shape[0] < 2 or context.shape[0] % 2:
            raise ValueError('Frozen training requires even context batches [B,D]')
        if occupied.shape != (context.shape[0], 1) or occupied.dtype != torch.bool:
            raise ValueError('Expected history occupancy bool[B,1]')
        if centering_scope not in ('task', 'aggregate'):
            raise ValueError('Unknown centering scope')
        if precision not in ('float32', 'float64'):
            raise ValueError('Unknown auxiliary reduction precision')
        ctx.centering_scope = centering_scope
        ctx.precision = precision
        ctx.save_for_backward(occupied)
        return context.view_as(context), context.view_as(context), context.view_as(context)

    @staticmethod
    def backward(ctx, cart_gradient, fav_gradient, buy_gradient):
        (occupied,) = ctx.saved_tensors
        transfer = _reduce_auxiliary_context(
            (cart_gradient, fav_gradient, buy_gradient), occupied,
            ctx.centering_scope, ctx.precision)
        return transfer, None, None, None


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
                 support_gradient='attached', inference_support='fully_pruned',
                 training_support='folded', residual_execution='direct',
                 auxiliary_train_population='primary', auxiliary_eval_population='primary',
                 block_aux_tower_context_grad=True,
                 block_aux_adapter_context_grad=False,
                 context_fusion='product', interaction_rank=16,
                 block_aux_residual_context_grad=True,
                 block_aux_product_context_grad=False,
                 product_context_assignment='aligned',
                 product_context_component='centered',
                 product_context_magnitude='native',
                 product_context_orientation='aligned',
                 parallel_orientation='aligned',
                 orthogonal_orientation='aligned',
                 centering_scope='task',
                 click_context_collection='direct',
                 auxiliary_reduction_precision='float32',
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
            raise ValueError('Use the separate context-path gates; item/action gradients stay blocked')
        if (not isinstance(block_aux_tower_context_grad, bool)
                or not isinstance(block_aux_adapter_context_grad, bool)):
            raise ValueError('Context-path gates must be booleans')
        self.block_aux_tower_context_grad = block_aux_tower_context_grad
        self.block_aux_adapter_context_grad = block_aux_adapter_context_grad
        if (not isinstance(block_aux_residual_context_grad, bool)
                or not isinstance(block_aux_product_context_grad, bool)):
            raise ValueError('Adapter-specific context gates must be booleans')
        # Plain metadata only: no modules, buffers, RNG or parameter changes.
        self.block_aux_residual_context_grad = block_aux_residual_context_grad
        self.block_aux_product_context_grad = block_aux_product_context_grad
        if product_context_assignment != 'aligned':
            raise ValueError('Direction factorial fixes example assignment to aligned')
        if product_context_orientation != 'aligned':
            raise ValueError('Use separate component orientations in this study')
        if parallel_orientation not in ('aligned', 'reversed'):
            raise ValueError('Unknown parallel orientation')
        if orthogonal_orientation not in ('aligned', 'reversed'):
            raise ValueError('Unknown orthogonal orientation')
        if parallel_orientation != 'aligned' or orthogonal_orientation != 'aligned':
            raise ValueError('Replay-integrity factorial freezes both orientations to aligned')
        if centering_scope not in ('task', 'aggregate'):
            raise ValueError('Unknown centering scope')
        if click_context_collection != 'direct':
            raise ValueError('Collected auxiliary study freezes direct click context')
        if auxiliary_reduction_precision not in ('float32', 'float64'):
            raise ValueError('Unknown auxiliary reduction precision')
        self.auxiliary_reduction_precision = auxiliary_reduction_precision
        self.centering_scope = centering_scope
        self.click_context_collection = click_context_collection
        self.parallel_orientation = parallel_orientation
        self.orthogonal_orientation = orthogonal_orientation
        self.product_context_orientation = product_context_orientation
        if product_context_component != 'centered':
            raise ValueError('This factorial retains centered auxiliary transfer')
        if block_aux_residual_context_grad is not True:
            raise ValueError('Component study isolates product-only context transfer')
        self.product_context_component = product_context_component
        if product_context_magnitude != 'native':
            raise ValueError('Direction study freezes native component magnitude')
        self.product_context_magnitude = product_context_magnitude
        if (block_aux_tower_context_grad is not True
                or block_aux_adapter_context_grad is not False
                or block_aux_product_context_grad is not False):
            raise ValueError('Freeze other context gates; auxiliary product edge must be open')
        if context_fusion != 'product':
            raise ValueError('Direction attribution freezes product fusion')
        self.product_context_assignment = product_context_assignment
        if context_fusion not in ('product', 'sum'):
            raise ValueError('Unknown context/history fusion operator')
        if (not isinstance(interaction_rank, int) or isinstance(interaction_rank, bool)
                or interaction_rank < 1):
            raise ValueError('Interaction rank must be a positive integer')
        self.context_fusion = context_fusion
        self.interaction_rank = interaction_rank
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
        if normalization_support != 'rotated_duplicate' or support_gradient != 'attached':
            raise ValueError('This study retains native attached rotated support during training')
        if inference_support not in ('full', 'score_pruned', 'fully_pruned'):
            raise ValueError('Unknown inference support execution mode')
        self.inference_support = inference_support
        if training_support not in ('folded', 'compression_only', 'full', 'unweighted'):
            raise ValueError('Unknown support computation control')
        if normalization_support != 'rotated_duplicate' or support_gradient != 'attached':
            raise ValueError('Folding requires identical attached support pairs')
        if inference_support != 'fully_pruned':
            raise ValueError('This suite shares the validated pruned eval path')
        self.training_support = training_support
        if residual_execution not in ('direct', 'all'):
            raise ValueError('Unknown task-adapter execution mode')
        self.residual_execution = residual_execution
        if auxiliary_train_population not in ('feedback', 'primary'):
            raise ValueError('Unknown auxiliary training normalization population')
        if auxiliary_eval_population not in ('primary', 'feedback'):
            raise ValueError('Unknown auxiliary evaluation normalization population')
        self.auxiliary_train_population = auxiliary_train_population
        self.auxiliary_eval_population = auxiliary_eval_population
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
            # Appended after every inherited learned module is initialized.
            # This same fork preserves the host RNG after construction.
            context_width = len(self.user_fields) * embedding_dim
            history_width = 6 * self.value_width + len(self.match_fields)
            self.context_queries = nn.ModuleList([
                nn.Linear(context_width, interaction_rank, bias=False)
                for _ in feature_map.labels
            ])
            self.history_projections = nn.ModuleList([
                nn.Linear(history_width, interaction_rank, bias=False)
                for _ in feature_map.labels
            ])
            self.interaction_outputs = nn.ModuleList([
                nn.Linear(interaction_rank, self.value_width, bias=False)
                for _ in feature_map.labels
            ])
            for output in self.interaction_outputs:
                nn.init.zeros_(output.weight)
        # Add only nonlearned state, AFTER every inherited learned module and
        # initialization. Native Dice has running mean/variance and counters.
        # Registered clones move with the model and round-trip in checkpoints;
        # they never alias the primary running buffers. No RNG is consumed.
        self._auxiliary_state_names = {}
        for index, (name, buffer) in enumerate(self.local_activation.named_buffers()):
            storage_name = '_auxiliary_moment_' + str(index)
            self.register_buffer(storage_name, buffer.detach().clone())
            self._auxiliary_state_names[name] = storage_name
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
        detail, position_detail, block_detail, curvature_detail, joint_detail, match_counts, match_union = self._target_match_summary(
            item, values, mask)
        return context, target, values, mask, auxiliary, detail, position_detail, block_detail, curvature_detail, joint_detail, match_counts, match_union

    def _target_match_summary(self, item, values, mask):
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
        return (summaries[:, 0, :], summaries[:, 1, :], summaries[:, 2, :],
                summaries[:, 3, :], summaries[:, 4, :], match_counts,
                match_matrix.bool().any(dim=-1))

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

    def _compress_unique_support(self, values, mask, auxiliary, match_union):
        """Five representatives of the inherited seven assignment rows.

        Empty slots remain zero rows and retain their normalization mass.
        Counts are used only by readout, never as Dice multiplicities.
        """
        batch, length, width = values.shape
        if length != 100:
            raise ValueError('Frozen protocol requires 100 history positions')
        valid = mask.bool().reshape(batch, 10, 10)
        feedback = auxiliary.bool().reshape(batch, 10, 10) & valid
        matched = match_union.bool().reshape(batch, 10, 10) & valid
        rotated = self._rotate_valid_membership(matched, valid)
        assignment = torch.stack(
            (valid, valid & ~feedback, feedback, valid & ~rotated, rotated), dim=-2)
        assignment = assignment.to(values.dtype).reshape(batch * 10, 5, 10)
        counts = assignment.sum(dim=-1)
        sums = torch.bmm(assignment, values.reshape(batch * 10, 10, width))
        means = sums / counts.clamp_min(1).unsqueeze(-1)
        return means.reshape(batch, 50, width), counts.reshape(batch, 50)

    @staticmethod
    def _expand_support_rows(rows):
        """[B,50,D] -> [B,70,D], preserving block and original row order.

        index_select is differentiable: two copies accumulate gradients back
        into one representative. No detach, state, RNG or mask changes.
        """
        batch, slots, width = rows.shape
        if slots != 50:
            raise ValueError('Expected five representatives per temporal block')
        index = torch.tensor([0, 1, 2, 3, 4, 3, 4], dtype=torch.long,
                             device=rows.device)
        return rows.reshape(batch, 10, 5, width).index_select(2, index).reshape(
            batch, 70, width)

    def _folded_support_activation(self, interactions, batch):
        """Affine maps on 50 rows; the SAME Dice sees all original 70 rows.

        After Dice, duplicate support outputs are identical in real
        arithmetic. Retain the first five representatives and accumulate
        their downstream multiplicities at the next expansion. Autograd
        through both gather and selection preserves the derivative of the
        tied computation. This relies on the audited Linear/Dice scorer;
        stochastic or cross-row layers would require a new derivation.
        Native BatchNorm owns its epsilon, biased batch variance, unbiased
        running variance, momentum, counter updates and coupled derivatives.
        """
        hidden = interactions
        for layer in self.local_activation:
            if isinstance(layer, Dice):
                expanded = self._expand_support_rows(hidden.reshape(batch, 50, -1))
                expanded = layer(expanded.reshape(batch * 70, -1))
                hidden = expanded.reshape(batch, 10, 7, -1)[:, :, :5, :].reshape(
                    batch * 50, -1)
            elif isinstance(layer, nn.Linear):
                hidden = layer(hidden)
            else:
                raise ValueError('Support folding only supports the audited Linear/Dice scorer')
        return hidden

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

    def _interactions(self, target, values):
        keys = values[:, :, :self.key_width]
        action = values[:, :, self.key_width:]
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
        # Independent registered state is updated by native BatchNorm during
        # training and read during eval. Fixed momentum0.01 makes train output
        # independent of previous running moments/counter values. The original
        # primary buffers are never supplied to this functional call.
        buffers = {name: getattr(self, storage_name)
                   for name, storage_name in self._auxiliary_state_names.items()}
        return torch.func.functional_call(
            self.local_activation, (parameters, buffers), (interactions,), strict=True)

    def _pool(self, weights, values, counts):
        batch, slots, _ = values.shape
        weights = weights.reshape(batch, slots) * counts.to(weights.dtype)
        weighted = (weights.unsqueeze(-1) * values).reshape(batch, 10, 3, self.value_width)
        return weighted.sum(dim=1)

    def _context_history_correction(self, index, context, history):
        # Only supplied context and already-masked compressed history enter.
        # FP32 bounded projections and a shared RMS denominator reduce gross
        # scale confounding between product and sum. Their geometry and
        # optimization still differ; equal norms do not imply equal capacity.
        query = torch.tanh(self.context_queries[index](context).float())
        summary = torch.tanh(self.history_projections[index](history).float())
        if self.context_fusion == 'product':
            fused = query * summary
        else:
            fused = (query + summary) * (2.0 ** -0.5)
        fused = fused * torch.rsqrt(fused.square().mean(dim=-1, keepdim=True) + 1e-4)
        return self.interaction_outputs[index](fused.to(history.dtype))

    def forward(self, inputs):
        context, target, values, mask, auxiliary, detail, position_detail, block_detail, curvature_detail, joint_detail, match_counts, match_union = self._represent(inputs)
        batch = values.shape[0]
        if self.training:
            occupied = mask.bool().any(dim=1, keepdim=True)
            collected_contexts = _CollectedAuxiliaryTransfer.apply(
                context, occupied, self.centering_scope,
                self.auxiliary_reduction_precision)
            product_context_by_label = dict(zip(('cart', 'fav', 'buy'), collected_contexts))
            # Other auxiliary context uses are blocked below. The original
            # context stays on click's unchanged graph.
            transfer_context = context
        if (self.training and self.training_support == 'full') or self.inference_support == 'full':
            # This training path is exactly the incumbent's operator sequence:
            # all 70 rows enter Dice and retain native support gradients.
            all_values, all_counts = self._compress_with_match(values, mask, auxiliary, match_union)
            primary_weights = self._primary_activation(self._interactions(target, all_values))
            all_strata = self._pool_seven(primary_weights, all_values, all_counts)
            uniform = all_strata[:, 0, :]
            values = all_values.reshape(batch, 10, 7, self.value_width)[:, :, :3, :].reshape(
                batch, 30, self.value_width)
            counts = all_counts.reshape(batch, 10, 7)[:, :, :3].reshape(batch, 30)
            primary_strata = all_strata[:, :3, :]
        elif self.training:
            # Folded and unweighted use five unique assignment rows. The
            # compression-only control restores seven BEFORE every scorer
            # operation, isolating compression arithmetic from affine work.
            unique_values, unique_counts = self._compress_unique_support(
                values, mask, auxiliary, match_union)
            if self.training_support == 'compression_only':
                all_values = self._expand_support_rows(unique_values)
                all_counts = self._expand_support_rows(unique_counts.unsqueeze(-1)).squeeze(-1)
                primary_weights = self._primary_activation(self._interactions(target, all_values))
                all_strata = self._pool_seven(primary_weights, all_values, all_counts)
                primary_strata = all_strata[:, :3, :]
            else:
                interactions = self._interactions(target, unique_values)
                if self.training_support == 'folded':
                    primary_weights = self._folded_support_activation(interactions, batch)
                else:
                    # Ablation removes two support copies from every Dice
                    # population. It is not an equivalence-preserving path.
                    primary_weights = self.local_activation(interactions)
                weights = primary_weights.reshape(batch, 10, 5)[:, :, :3].reshape(batch, 30)
                read_values = unique_values.reshape(batch, 10, 5, self.value_width)[:, :, :3, :].reshape(
                    batch, 30, self.value_width)
                read_counts = unique_counts.reshape(batch, 10, 5)[:, :, :3].reshape(batch, 30)
                primary_strata = self._pool(weights, read_values, read_counts)
            uniform = primary_strata[:, 0, :]
            values = unique_values.reshape(batch, 10, 5, self.value_width)[:, :, :3, :].reshape(
                batch, 30, self.value_width)
            counts = unique_counts.reshape(batch, 10, 5)[:, :, :3].reshape(batch, 30)
        else:
            # Pruning is valid only in eval mode. Never change train/eval
            # flags, running buffers, parameters, or the host batch policy.
            if self.inference_support == 'score_pruned':
                # Retain the original compression arithmetic to isolate the
                # effect of reducing scorer GEMM rows from 70 to 30.
                all_values, all_counts = self._compress_with_match(values, mask, auxiliary, match_union)
                values = all_values.reshape(batch, 10, 7, self.value_width)[:, :, :3, :].reshape(
                    batch, 30, self.value_width)
                counts = all_counts.reshape(batch, 10, 7)[:, :, :3].reshape(batch, 30)
            else:
                # The first three assignment rows of seven-row compression
                # are exactly the inherited uniform/feedback assignment.
                values, counts = self._compress_both(values, mask, auxiliary)
            primary_weights = self._primary_activation(self._interactions(target, values))
            primary_strata = self._pool(primary_weights, values, counts)
            uniform = primary_strata[:, 0, :]

        if self.training:
            # Item fields share tables across current target and past events:
            # gate both uses together. Historical action joins this group.
            # Context includes only supplied non-item/non-action features.
            aux_context = transfer_context.detach() if self.block_aux_context_grad else transfer_context
            aux_target = target.detach() if self.block_aux_item_grad else target
            aux_values = values.detach() if self.block_aux_item_grad else values
            aux_detail = detail.detach() if self.block_aux_item_grad else detail
            aux_position_detail = position_detail.detach() if self.block_aux_item_grad else position_detail
            aux_block_detail = block_detail.detach() if self.block_aux_item_grad else block_detail
            aux_curvature_detail = curvature_detail.detach() if self.block_aux_item_grad else curvature_detail
            aux_joint_detail = joint_detail.detach() if self.block_aux_item_grad else joint_detail
            aux_weights = self._auxiliary_activation(self._interactions(aux_target, aux_values))
            feedback_population_strata = self._pool(aux_weights, aux_values, counts)
            # Execute feedback scoring in every training cell to match work
            # and collect a complete independent running-state trajectory.
            # Primary strata have all support rows in their native moments;
            # detach the entire result to preserve auxiliary source ownership.
            if self.auxiliary_train_population == 'primary':
                aux_strata = primary_strata.detach()
            else:
                aux_strata = feedback_population_strata
        else:
            # The eval factor never changes training operations. Each choice
            # uses only frozen running moments; eval cannot update either set.
            aux_context, aux_target = context, target
            aux_detail = detail
            aux_position_detail = position_detail
            aux_block_detail = block_detail
            aux_curvature_detail = curvature_detail
            aux_joint_detail = joint_detail
            if self.auxiliary_eval_population == 'feedback':
                aux_weights = self._auxiliary_activation(self._interactions(target, values))
                aux_strata = self._pool(aux_weights, values, counts)
            else:
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
            # Keep the two uses numerically identical. Detach only the
            # selected context edge; every auxiliary adapter/tower parameter
            # remains in its task loss graph. Eval has no gradient gating.
            adapter_context = task_context
            tower_context = task_context
            if self.training and label != 'is_click':
                if self.block_aux_adapter_context_grad:
                    adapter_context = adapter_context.detach()
                if self.block_aux_tower_context_grad:
                    tower_context = tower_context.detach()
            # Split only the two context input edges. The master adapter gate
            # above remains effective; private adapter parameters never detach.
            residual_context = adapter_context
            product_context = adapter_context
            if self.training and label != 'is_click':
                if self.block_aux_residual_context_grad:
                    residual_context = residual_context.detach()
                product_context = product_context_by_label[label]
                if self.block_aux_product_context_grad:
                    product_context = product_context.detach()
                # Only input context cotangents change. Private product-module
                # gradients and all true-label task losses stay ordinary.
            signal = torch.cat([task_detail, task_position_detail, task_block_detail,
                                task_curvature_detail, task_joint_detail], dim=-1)
            residual_features = torch.cat([residual_context, task_target, base,
                                           match_counts.to(base.dtype), signal], dim=-1)
            # Task routing is required by __init__. Each adapter still
            # receives its own task's features once; every parameter remains
            # in the loss graph. Unselected adapters are stateless and their
            # outputs have zero loss derivative in the all-bank control.
            if self.residual_execution == 'direct':
                correction = self.residual_bank[index](residual_features)
            else:
                bank = torch.stack([adapter(residual_features)
                                    for adapter in self.residual_bank], dim=1)
                correction = bank[:, index, :] if self.residual_routing == 'task' else bank.mean(dim=1)
            history_features = torch.cat([base, match_counts.to(base.dtype), signal], dim=-1)
            correction = correction + self._context_history_correction(
                index, product_context, history_features)
            pooled = base + has_history * correction
            features = torch.cat([tower_context, task_target, pooled], dim=-1)
            predictions[label + '_pred'] = torch.sigmoid(tower(features))
        return predictions

    def training_loss(self, inputs, probabilities, train_labels):
        """Training-only objective; never reads labels in forward.

        Keep the four per-task BCE graphs even in click-only mode. Zero-weight
        auxiliary terms yield zero gradients rather than removing parameters
        or skipping their forward/state updates. The host keeps the original
        labels for scoring. A fixed half-batch cyclic shift preserves the
        auxiliary label triple and its marginal counts, but does not prove
        feature/label independence. It consumes no RNG or persistent state.
        """
        if (probabilities.ndim != 2 or train_labels.ndim != 2
                or probabilities.shape != train_labels.shape
                or probabilities.shape[1] != len(self.feature_map.labels)):
            raise ValueError('Expected aligned prediction and training-label matrices [B,4]')
        if AUXILIARY_OBJECTIVE not in ('joint', 'click_only', 'batch_roll'):
            raise ValueError('Unknown auxiliary training objective')
        if probabilities.shape[0] < 2:
            raise ValueError('This frozen full-batch training study requires at least two examples')
        # BCE is unsafe inside CUDA autocast. FP32 arithmetic here leaves
        # gradients into the original probabilities and parameters intact.
        with torch.autocast(device_type=probabilities.device.type, enabled=False):
            truth = train_labels.detach().to(dtype=torch.float32)
            shifted = torch.roll(truth, shifts=truth.shape[0] // 2, dims=0)
            losses = []
            for index, label in enumerate(self.feature_map.labels):
                target = (shifted[:, index]
                          if AUXILIARY_OBJECTIVE == 'batch_roll' and label != 'is_click'
                          else truth[:, index])
                task_loss = nn.functional.binary_cross_entropy(
                    probabilities[:, index].float(), target, reduction='mean')
                weight = (0.0 if AUXILIARY_OBJECTIVE == 'click_only'
                          and label != 'is_click' else 1.0)
                losses.append(weight * task_loss)
            return torch.stack(losses).sum()
