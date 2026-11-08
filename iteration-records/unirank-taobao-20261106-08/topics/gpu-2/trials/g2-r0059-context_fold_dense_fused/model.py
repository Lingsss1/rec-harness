"""Prediction-time folding of the inherited buy-context affine.

Training, Parameters, initialization, gradient ownership and masks are inherited.
The folded modes form fresh derived weights each eval call, with no cache or
state mutation. Real-arithmetic identities do not imply BF16 equivalence.
Full-rank factors are training capacity, not a parameter-reduction claim.
Prospective controls and quality/cost gates are recorded in proposal.json.
"""
import torch
from torch import nn
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbeddingDict
from unirank.pytorch.layers.activations import Dice


CONFIG = {'embedding_dim': 16, 'attention_hidden_units': [64, 32], 'tower_hidden_units': [128, 64], 'attention_normalization': 'none', 'compression_routes': 'uf', 'block_aux_scorer_grad': True, 'block_aux_context_grad': False, 'block_aux_item_grad': True, 'residual_input': 'match_detail', 'residual_routing': 'task', 'residual_hidden': 14, 'temporal_detail': 'position', 'block_detail': 'ordinal', 'curvature_detail': 'quadratic', 'joint_detail': 'duplicate_match', 'match_support': 'available', 'match_scaling': 'norm', 'coarse_extension': 'duplicate', 'primary_partition': 'uniform', 'normalization_support': 'rotated_duplicate', 'support_gradient': 'attached', 'auxiliary_residual_hidden': 10, 'evaluation_projection': 'merged_hoist', 'evaluation_dice': 'both', 'click_tower_hidden_units': [128, 64], 'auxiliary_tower_hidden_units': [112, 64], 'auxiliary_context_tasks': ['buy'], 'auxiliary_tower_overrides': {'cart': [42, 32], 'fav': [44, 32]}, 'auxiliary_residual_overrides': {'cart': 2, 'fav': 2}, 'nonowner_residual': 'retained', 'nonowner_tower_depth': 1, 'nonowner_residual_overrides': {'cart': 'retained', 'fav': 'removed'}, 'nonowner_tower_depth_overrides': {'cart': 1, 'fav': 2}, 'private_hidden_scale': 'inherited', 'private_readout_scale': 'inherited', 'owner_initialization_scale': 'inherited', 'owner_affine_rank': 0, 'owner_factor_balance': 'balanced', 'owner_input_partition': 'semantic', 'owner_context_projection': 'separate', 'owner_context_rank': 0, 'owner_context_scale': 'inherited', 'owner_context_balance': 'balanced', 'owner_context_evaluation': 'fused'}


class _RankAffine(nn.Module):
    """Two trainable linear factors with no intervening nonlinearity.

    This module is installed only after the parent's final reset/reduction.
    SVD is constructor-only CPU FP32 work; no spectral operations, full-sized
    backing weights, caches or rank selection occur during forward. All
    output channels, the inherited bias and following Dice remain present.
    A failed decomposition propagates to the host; there is no silent fallback.
    """
    def __init__(self, linear, rank, balance):
        super().__init__()
        if not isinstance(linear, nn.Linear) or linear.bias is None:
            raise ValueError('Expected the inherited biased first affine')
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise ValueError('Rank must be an integer')
        if not 1 <= rank < min(linear.in_features, linear.out_features):
            raise ValueError('Rank must strictly reduce the affine rank ceiling')
        if rank * (linear.in_features + linear.out_features) >= linear.weight.numel():
            raise ValueError('Factorization must reduce actual weight storage')
        if balance not in ('balanced', 'output_heavy'):
            raise ValueError('Unknown factor balance')
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank
        with torch.no_grad():
            weight = linear.weight.detach().to(device='cpu', dtype=torch.float32)
            left, singular, right = torch.linalg.svd(weight, full_matrices=False)
            # Canonicalize paired signs without changing the product. No RNG.
            left = left[:, :rank]
            right = right[:rank, :]
            singular = singular[:rank]
            pivots = left.abs().argmax(dim=0, keepdim=True)
            signs = left.gather(0, pivots).squeeze(0).sign()
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            left = left * signs.unsqueeze(0)
            right = right * signs.unsqueeze(1)
            exponent = 0.5 if balance == 'balanced' else 1.0
            output_factor = left * singular.pow(exponent).unsqueeze(0)
            input_factor = singular.pow(1.0 - exponent).unsqueeze(1) * right
            # Clone after conversion: no full template/SVD tensor storage is
            # retained by Parameters, including when device/dtype is unchanged.
            self.input_factor = nn.Parameter(
                input_factor.to(device=linear.weight.device,
                                dtype=linear.weight.dtype).clone(),
                requires_grad=linear.weight.requires_grad)
            self.output_factor = nn.Parameter(
                output_factor.to(device=linear.weight.device,
                                 dtype=linear.weight.dtype).clone(),
                requires_grad=linear.weight.requires_grad)
            self.bias = nn.Parameter(linear.bias.detach().clone(),
                                     requires_grad=linear.bias.requires_grad)

    def forward(self, features):
        hidden = torch.nn.functional.linear(features, self.input_factor)
        return torch.nn.functional.linear(hidden, self.output_factor, self.bias)


class _PartitionedAffine(nn.Module):
    """Two packed affine row groups with common context and local inputs.

    Parameter slices come only from a freshly initialized parent. Constructor
    indexing consumes no RNG; no checkpoint/data/label inspection occurs.
    Full dense template weights are not retained, masked or rebuilt. Both
    partitions execute the same two gathers/affines/concatenation in train
    and eval. The following Dice and second dense layer belong to the parent.
    """
    def __init__(self, linear, context_width, key_width, value_width,
                 embedding_dim, partition):
        super().__init__()
        if not isinstance(linear, nn.Linear) or linear.bias is None:
            raise ValueError('Expected the inherited biased first affine')
        if partition not in ('semantic', 'mixed'):
            raise ValueError('Unknown packed input partition')
        if linear.in_features != context_width + key_width + value_width:
            raise ValueError('First affine inputs must be context,target,pooled')
        if linear.out_features != 112:
            raise ValueError('Partition contrast must retain112 nonlinear channels')
        if embedding_dim < 2 or embedding_dim % 2 or key_width % embedding_dim:
            raise ValueError('Mixed control requires even field dimensions')
        if value_width != key_width + embedding_dim:
            raise ValueError('Expected item fields followed by historical action')
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.row_split = linear.out_features // 2
        target_start = context_width
        pooled_start = context_width + key_width
        if partition == 'semantic':
            local_first = list(range(target_start, pooled_start))
        else:
            # Half of every target field and the corresponding pooled field.
            # Action coordinates stay in the complementary local group.
            local_first = [start + field + coordinate
                           for start in (target_start, pooled_start)
                           for field in range(0, key_width, embedding_dim)
                           for coordinate in range(embedding_dim // 2)]
        local_set = set(local_first)
        local_second = [index for index in range(context_width, self.in_features)
                        if index not in local_set]
        if (len(local_first) != key_width or len(local_second) != value_width
                or len(local_set) != key_width
                or local_set.intersection(local_second)
                or local_set.union(local_second) != set(range(context_width, self.in_features))):
            raise ValueError('Local groups must exactly partition target and pool')
        common = list(range(context_width))
        first_columns = torch.tensor(common + local_first, dtype=torch.long,
                                     device=linear.weight.device)
        second_columns = torch.tensor(common + local_second, dtype=torch.long,
                                      device=linear.weight.device)
        self.register_buffer('first_columns', first_columns)
        self.register_buffer('second_columns', second_columns)
        with torch.no_grad():
            self.first_weight = nn.Parameter(
                linear.weight[:self.row_split].index_select(1, first_columns).detach().clone(),
                requires_grad=linear.weight.requires_grad)
            self.second_weight = nn.Parameter(
                linear.weight[self.row_split:].index_select(1, second_columns).detach().clone(),
                requires_grad=linear.weight.requires_grad)
            self.bias = nn.Parameter(linear.bias.detach().clone(),
                                     requires_grad=linear.bias.requires_grad)

    def forward(self, features):
        first = torch.nn.functional.linear(
            features.index_select(-1, self.first_columns), self.first_weight,
            self.bias[:self.row_split])
        second = torch.nn.functional.linear(
            features.index_select(-1, self.second_columns), self.second_weight,
            self.bias[self.row_split:])
        return torch.cat((first, second), dim=-1)


class _ContextAffine(nn.Module):
    """Separate the common context map; optionally constrain only its rank.

    Installed after every inherited construction/reset/slice. All new weights
    derive from the fresh untrained packed affine, without new RNG draws.
    Local target/pool weights and bias are cloned unchanged. No full source
    affine, checkpoint, data, spectral tensor or derived cache persists.
    """
    def __init__(self, source, context_width, projection, rank, scale, balance,
                 evaluation='native'):
        super().__init__()
        if not isinstance(source, _PartitionedAffine):
            raise ValueError('Expected a packed semantic affine')
        if projection not in ('separate', 'rank', 'full_rank_control'):
            raise ValueError('Expected separate, reduced-rank or full-rank context')
        if scale not in ('inherited', 'frobenius'):
            raise ValueError('Unknown context initialization scale')
        if balance not in ('balanced', 'output_heavy'):
            raise ValueError('Unknown context factor balance')
        if projection == 'separate' and balance != 'balanced':
            raise ValueError('Dense context has no factor-balance intervention')
        if projection != 'rank' and scale != 'inherited':
            raise ValueError('Only reduced rank accepts energy restoration')
        if projection == 'separate' and rank != 0:
            raise ValueError('Separate dense context must have rank zero')
        if projection == 'rank' and not 1 <= rank < min(context_width, source.out_features):
            raise ValueError('Context rank must strictly reduce its rank ceiling')
        if projection == 'rank' and rank * (context_width + source.out_features) >= context_width * source.out_features:
            raise ValueError('Context factors must reduce parameter storage')
        if projection == 'full_rank_control' and rank != min(context_width, source.out_features):
            raise ValueError('Full-rank diagnostic must retain every singular direction')
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.row_split = source.row_split
        self.context_width = context_width
        self.projection = projection
        self.rank = rank
        if evaluation not in ('native', 'folded', 'fused'):
            raise ValueError('Unknown prediction-time context implementation')
        self.evaluation = evaluation
        with torch.no_grad():
            expected = torch.arange(context_width, device=source.first_columns.device)
            if (not torch.equal(source.first_columns[:context_width], expected)
                    or not torch.equal(source.second_columns[:context_width], expected)):
                raise ValueError('Both branches must read every context coordinate in order')
            self.register_buffer('first_columns', source.first_columns[context_width:].clone())
            self.register_buffer('second_columns', source.second_columns[context_width:].clone())
            self.first_weight = nn.Parameter(
                source.first_weight[:, context_width:].detach().clone(),
                requires_grad=source.first_weight.requires_grad)
            self.second_weight = nn.Parameter(
                source.second_weight[:, context_width:].detach().clone(),
                requires_grad=source.second_weight.requires_grad)
            self.bias = nn.Parameter(source.bias.detach().clone(),
                                     requires_grad=source.bias.requires_grad)
            context = torch.cat((source.first_weight[:, :context_width],
                                 source.second_weight[:, :context_width]), dim=0).detach()
            if projection == 'separate':
                self.context_weight = nn.Parameter(context.clone(),
                    requires_grad=source.first_weight.requires_grad)
            else:
                # CPU FP32 SVD only at construction. Full-rank mode is an
                # attribution control, not a reduction. It retains all
                # singular directions; rank mode retains only the top r.
                left, singular, right = torch.linalg.svd(
                    context.to(device='cpu', dtype=torch.float32), full_matrices=False)
                if scale == 'frobenius':
                    total_energy = singular.square().sum()
                    retained_energy = singular[:rank].square().sum()
                    if (not torch.isfinite(total_energy) or not torch.isfinite(retained_energy)
                            or total_energy <= 0 or retained_energy <= 0):
                        raise ValueError('Cannot normalize degenerate context energy')
                    energy_gain = (total_energy / retained_energy).sqrt()
                left, singular, right = left[:, :rank], singular[:rank], right[:rank, :]
                if scale == 'frobenius':
                    # Restores ||U_r diag(s_r) V_r||_F to ||W||_F.
                    # Following sqrt splits the gain across both factors.
                    # No persistent scalar or forward-time scale operation.
                    singular = singular * energy_gain
                pivots = left.abs().argmax(dim=0, keepdim=True)
                signs = left.gather(0, pivots).squeeze(0).sign()
                signs = torch.where(signs == 0, torch.ones_like(signs), signs)
                if balance == 'balanced':
                    # Exact inherited balanced construction, including operation order.
                    root = singular.sqrt()
                    output_factor = (left * signs.unsqueeze(0)) * root.unsqueeze(0)
                    input_factor = root.unsqueeze(1) * (right * signs.unsqueeze(1))
                else:
                    # Same U_r diag(s_r) V_r product over real arithmetic.
                    # Only factor coordinates change: all singular magnitude
                    # is in the output factor; input rows are orthonormal.
                    # Optimizers, clipping and BF16 are not gauge invariant.
                    output_factor = (left * signs.unsqueeze(0)) * singular.unsqueeze(0)
                    input_factor = right * signs.unsqueeze(1)
                self.context_input = nn.Parameter(input_factor.to(
                    device=context.device, dtype=context.dtype).clone(),
                    requires_grad=source.first_weight.requires_grad)
                self.context_output = nn.Parameter(output_factor.to(
                    device=context.device, dtype=context.dtype).clone(),
                    requires_grad=source.first_weight.requires_grad)

    def _evaluation_forward(self, features):
        """Fold linear factors, then optionally absorb context into both branches.

        Derive W = context_output @ context_input in FP32, outside autocast.
        Rebuild it every call, so load_state_dict, optimizer updates, device
        moves and eval/train transitions cannot leave a stale derived cache.
        There is no detach, persistent tensor, state update or hidden fallback.
        The per-call product, concatenations and gathers count toward timing.
        """
        if self.training:
            raise ValueError('Context folding is evaluation-only')
        if self.projection == 'separate':
            weight = self.context_weight
        else:
            with torch.autocast(device_type=features.device.type, enabled=False):
                weight = self.context_output.float() @ self.context_input.float()
            weight = weight.to(dtype=self.context_input.dtype)
        context = features[..., :self.context_width]
        first_features = features.index_select(-1, self.first_columns)
        second_features = features.index_select(-1, self.second_columns)
        if self.evaluation == 'fused':
            # Match the inherited packed semantic connectivity, with no dense
            # zero-filled backing matrix and no new trainable coefficients.
            first = torch.nn.functional.linear(
                torch.cat((context, first_features), dim=-1),
                torch.cat((weight[:self.row_split], self.first_weight), dim=-1),
                self.bias[:self.row_split])
            second = torch.nn.functional.linear(
                torch.cat((context, second_features), dim=-1),
                torch.cat((weight[self.row_split:], self.second_weight), dim=-1),
                self.bias[self.row_split:])
            return torch.cat((first, second), dim=-1)
        context = torch.nn.functional.linear(context, weight)
        first = torch.nn.functional.linear(
            first_features, self.first_weight, self.bias[:self.row_split])
        second = torch.nn.functional.linear(
            second_features, self.second_weight, self.bias[self.row_split:])
        return torch.cat((first, second), dim=-1) + context

    def forward(self, features):
        if not self.training and self.evaluation != 'native':
            return self._evaluation_forward(features)
        context = features[..., :self.context_width]
        if self.projection == 'separate':
            context = torch.nn.functional.linear(context, self.context_weight)
        else:
            context = torch.nn.functional.linear(context, self.context_input)
            context = torch.nn.functional.linear(context, self.context_output)
        first = torch.nn.functional.linear(
            features.index_select(-1, self.first_columns), self.first_weight,
            self.bias[:self.row_split])
        second = torch.nn.functional.linear(
            features.index_select(-1, self.second_columns), self.second_weight,
            self.bias[self.row_split:])
        return torch.cat((first, second), dim=-1) + context


class Model(MultiTaskModel):
    def __init__(self, feature_map, model_id='TargetMatchOrderMoment', gpu=-1,
                 embedding_dim=16, attention_hidden_units=(64, 32),
                 tower_hidden_units=(128, 64), attention_normalization='none',
                 compression_routes='uf', block_aux_scorer_grad=True,
                 block_aux_context_grad=False, block_aux_item_grad=True, num_tasks=None,
                 residual_input='match_detail', residual_routing='task', residual_hidden=14,
                 auxiliary_residual_hidden=10,
                 temporal_detail='position', block_detail='ordinal',
                 curvature_detail='quadratic', joint_detail='duplicate_match',
                 match_support='available', match_scaling='norm', coarse_extension='duplicate',
                 primary_partition='uniform', normalization_support='rotated_duplicate',
                 support_gradient='attached', evaluation_projection='merged_hoist',
                 evaluation_dice='both',
                 click_tower_hidden_units=(128, 64),
                 auxiliary_tower_hidden_units=(112, 64),
                 auxiliary_tower_overrides=None,
                 auxiliary_residual_overrides=None,
                 auxiliary_context_tasks=('buy',),
                 nonowner_residual='retained', nonowner_tower_depth=2,
                 nonowner_residual_overrides=None,
                 nonowner_tower_depth_overrides=None,
                 private_hidden_scale='inherited',
                 private_readout_scale='inherited',
                 owner_initialization_scale='inherited',
                 owner_affine_rank=0, owner_factor_balance='balanced',
                 owner_input_partition='dense',
                 owner_context_projection='fused', owner_context_rank=0,
                 owner_context_scale='inherited',
                 owner_context_balance='balanced',
                 owner_context_evaluation='native',
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
        if owner_context_evaluation not in ('native', 'folded', 'fused'):
            raise ValueError('Unknown prediction-time context implementation')
        if owner_context_evaluation != 'native' and owner_context_projection == 'fused':
            raise ValueError('Context folding requires an explicit context map')
        if owner_context_projection not in ('fused', 'separate', 'rank', 'full_rank_control'):
            raise ValueError('Unknown context projection')
        if owner_context_scale not in ('inherited', 'frobenius'):
            raise ValueError('Unknown context initialization scale')
        if owner_context_balance not in ('balanced', 'output_heavy'):
            raise ValueError('Unknown context factor balance')
        if owner_context_projection in ('fused', 'separate') and owner_context_balance != 'balanced':
            raise ValueError('Factor balance requires a factored context map')
        if owner_context_projection != 'rank' and owner_context_scale != 'inherited':
            raise ValueError('Energy restoration requires reduced context rank')
        if isinstance(owner_context_rank, bool) or not isinstance(owner_context_rank, int) or owner_context_rank < 0:
            raise ValueError('Context rank must be a nonnegative integer')
        if owner_context_projection in ('fused', 'separate') and owner_context_rank != 0:
            raise ValueError('Only rank context projection accepts positive rank')
        if owner_context_projection in ('rank', 'full_rank_control') and owner_context_rank == 0:
            raise ValueError('Rank context projection requires positive rank')
        if owner_context_projection != 'fused' and owner_input_partition != 'semantic':
            raise ValueError('Context intervention fixes semantic local partition')
        if owner_input_partition not in ('dense', 'semantic', 'mixed'):
            raise ValueError('Unknown owner input partition')
        if owner_input_partition != 'dense' and owner_affine_rank != 0:
            raise ValueError('Partition and rank interventions cannot be combined')
        if owner_input_partition != 'dense' and owner_initialization_scale != 'inherited':
            raise ValueError('Partition contrast freezes inherited initialization scale')
        if isinstance(owner_affine_rank, bool) or not isinstance(owner_affine_rank, int) or owner_affine_rank < 0:
            raise ValueError('Owner affine rank must be a nonnegative integer; zero disables it')
        if owner_factor_balance not in ('balanced', 'output_heavy'):
            raise ValueError('Unknown owner factor balance')
        if owner_initialization_scale not in ('inherited', 'xavier'):
            raise ValueError('Unknown owner tower initialization scale')
        if private_hidden_scale not in ('inherited', 'xavier'):
            raise ValueError('Unknown private hidden initialization scale')
        if private_readout_scale not in ('inherited', 'xavier'):
            raise ValueError('Unknown private readout initialization scale')
        if tuple(auxiliary_context_tasks) != ('buy',):
            raise ValueError('This factorial fixes buy-only auxiliary context ownership')
        if nonowner_residual not in ('retained', 'removed'):
            raise ValueError('Unknown nonowner residual intervention')
        if isinstance(nonowner_tower_depth, bool) or nonowner_tower_depth not in (1, 2):
            raise ValueError('Nonowner tower depth must be 1 or 2')
        self.nonowner_residual = nonowner_residual
        self.nonowner_tower_depth = nonowner_tower_depth
        # Only the two context-detached private heads can override stages.
        # Resolve maps before learned construction: no RNG/tensor operations.
        if nonowner_residual_overrides is None:
            nonowner_residual_overrides = {}
        if nonowner_tower_depth_overrides is None:
            nonowner_tower_depth_overrides = {}
        if not isinstance(nonowner_residual_overrides, dict):
            raise ValueError('Private residual overrides must be a task-to-mode dict')
        if not isinstance(nonowner_tower_depth_overrides, dict):
            raise ValueError('Private depth overrides must be a task-to-depth dict')
        if any(label not in ('cart', 'fav') for label in nonowner_residual_overrides):
            raise ValueError('Residual-stage overrides are limited to cart/fav')
        if any(label not in ('cart', 'fav') for label in nonowner_tower_depth_overrides):
            raise ValueError('Depth-stage overrides are limited to cart/fav')
        if any(mode not in ('retained', 'removed')
               for mode in nonowner_residual_overrides.values()):
            raise ValueError('Unknown private residual-stage mode')
        if any(isinstance(depth, bool) or not isinstance(depth, int) or depth not in (1, 2)
               for depth in nonowner_tower_depth_overrides.values()):
            raise ValueError('Private tower depths must be integers 1 or 2')
        self.nonowner_residual_modes = {
            label: nonowner_residual_overrides.get(label, nonowner_residual)
            for label in ('cart', 'fav')
        }
        nonowner_tower_depths = {
            label: nonowner_tower_depth_overrides.get(label, nonowner_tower_depth)
            for label in ('cart', 'fav')
        }
        # The full-width template fixes the common initialization draws.
        # Effective widths are architecture choices; host training is unchanged.
        if tuple(tower_hidden_units) != (128, 64):
            raise ValueError('The initialization template must remain (128, 64)')
        if auxiliary_tower_overrides is None:
            auxiliary_tower_overrides = {}
        if not isinstance(auxiliary_tower_overrides, dict):
            raise ValueError('Auxiliary tower overrides must be a task-to-widths dict')
        if any(label not in ('cart', 'fav', 'buy')
               for label in auxiliary_tower_overrides):
            raise ValueError('Only auxiliary task head dimensions may be overridden')
        for widths in (click_tower_hidden_units, auxiliary_tower_hidden_units,
                       *auxiliary_tower_overrides.values()):
            if not isinstance(widths, (list, tuple)) or len(widths) != 2:
                raise ValueError('Each task tower requires two hidden widths')
            for width, maximum in zip(widths, tower_hidden_units):
                if isinstance(width, bool) or not isinstance(width, int) or not 1 <= width <= maximum:
                    raise ValueError('Effective widths must be positive integers within the template')
        # Copy only validated immutable dimensions. This mapping creates no
        # Parameters, buffers or RNG draws and is never consulted in forward.
        auxiliary_tower_overrides = {
            label: tuple(widths) for label, widths in auxiliary_tower_overrides.items()
        }
        if evaluation_projection not in ('split_repeat', 'split_hoist', 'merged_hoist'):
            raise ValueError('Unknown evaluation affine factoring control')
        self.evaluation_projection = evaluation_projection
        if evaluation_dice not in ('scorer', 'towers', 'both'):
            raise ValueError('Unknown evaluation Dice location')
        self.evaluation_dice = evaluation_dice
        if block_aux_scorer_grad is not True:
            raise ValueError('Auxiliary scorer parameter gradients must stay blocked')
        if not isinstance(block_aux_context_grad, bool) or not isinstance(block_aux_item_grad, bool):
            raise ValueError('Source-path switches must be booleans')
        self.block_aux_scorer_grad = block_aux_scorer_grad
        self.block_aux_context_grad = block_aux_context_grad
        # A fixed task subset gates BOTH residual and direct head context uses.
        # This adds no tensor state, Parameters, RNG draws or loss reweighting.
        if block_aux_context_grad is not False:
            raise ValueError('Use auxiliary_context_tasks for context ownership')
        if not isinstance(auxiliary_context_tasks, (list, tuple)):
            raise ValueError('Context task subset must be a list or tuple')
        if (any(not isinstance(label, str) or label not in ('cart', 'fav', 'buy')
                for label in auxiliary_context_tasks)
                or len(set(auxiliary_context_tasks)) != len(auxiliary_context_tasks)):
            raise ValueError('Context task subset must contain distinct auxiliary labels')
        self.auxiliary_context_tasks = tuple(auxiliary_context_tasks)
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
        if residual_routing != 'task':
            raise ValueError('This replacement study freezes task-specific routing')
        if residual_routing not in ('task', 'shared'):
            raise ValueError('Unknown residual parameter-sharing control')
        if not isinstance(residual_hidden, int) or residual_hidden < 1:
            raise ValueError('Residual width must be a positive integer')
        if not isinstance(auxiliary_residual_hidden, int) or isinstance(auxiliary_residual_hidden, bool) or auxiliary_residual_hidden < 1:
            raise ValueError('Auxiliary residual width must be a positive integer')
        if auxiliary_residual_overrides is None:
            auxiliary_residual_overrides = {}
        if not isinstance(auxiliary_residual_overrides, dict):
            raise ValueError('Residual overrides must map auxiliary labels to widths')
        if any(label not in ('cart', 'fav', 'buy')
               for label in auxiliary_residual_overrides):
            raise ValueError('Only auxiliary residual widths may be overridden')
        for hidden in auxiliary_residual_overrides.values():
            if (isinstance(hidden, bool) or not isinstance(hidden, int)
                    or not 1 <= hidden <= auxiliary_residual_hidden):
                raise ValueError('Residual widths must fit the inherited template')
        auxiliary_residual_overrides = dict(auxiliary_residual_overrides)
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
                                        residual_hidden if label == 'is_click'
                                        else auxiliary_residual_hidden), nn.ReLU(),
                              nn.Linear(residual_hidden if label == 'is_click'
                                        else auxiliary_residual_hidden, self.value_width))
                for label in feature_map.labels
            ])
            for adapter in self.residual_bank:
                nn.init.zeros_(adapter[-1].weight)
                nn.init.zeros_(adapter[-1].bias)
        # Apply only after EVERY parent initialization operation, including
        # residual-bank creation inside fork_rng. Common Parameters and the
        # external RNG state then match the original source at this point.
        # All discarded channels are removed, including Dice state, before
        # optimizer construction; no full-sized backing tensors are retained.
        for index, (label, tower) in enumerate(zip(feature_map.labels, self.towers)):
            widths = (click_tower_hidden_units if label == 'is_click'
                      else auxiliary_tower_overrides.get(label, auxiliary_tower_hidden_units))
            if label in ('cart', 'fav') and nonowner_tower_depths[label] == 1:
                # The full template output has 64 inputs. Retain its first
                # h1 coefficients as a direct readout of the first hidden
                # layer. The original second hidden layer and Dice, including
                # their running buffers, are discarded before optimization.
                # This changes the function and effective initialization scale.
                # No checkpoint, labels, new RNG draw or empirical selection.
                if widths[0] > tower[-1].in_features:
                    raise ValueError('Shallow readout exceeds the template output width')
                self._reduce_tower(tower, (widths[0], widths[0]))
                self.towers[index] = nn.Sequential(tower[0], tower[1], tower[-1])
            else:
                self._reduce_tower(tower, widths)
        # Keep every parent RNG draw, including all full-width residuals.
        # This is an untrained channel subnetwork, not checkpoint pruning.
        # Clone the slices so removed channels retain no backing storage.
        # Retained first-layer weights keep the original initialization scale.
        for label, adapter in zip(feature_map.labels, self.residual_bank):
            if label not in auxiliary_residual_overrides:
                continue
            hidden = auxiliary_residual_overrides[label]
            first, output = adapter[0], adapter[2]
            if not isinstance(first, nn.Linear) or not isinstance(output, nn.Linear):
                raise ValueError('Unexpected inherited residual structure')
            if first.out_features != auxiliary_residual_hidden or output.in_features != auxiliary_residual_hidden:
                raise ValueError('Unexpected inherited residual width')
            if hidden == auxiliary_residual_hidden:
                continue
            with torch.no_grad():
                first.weight = nn.Parameter(
                    first.weight[:hidden, :].detach().clone(),
                    requires_grad=first.weight.requires_grad)
                first.bias = nn.Parameter(
                    first.bias[:hidden].detach().clone(),
                    requires_grad=first.bias.requires_grad)
                output.weight = nn.Parameter(
                    output.weight[:, :hidden].detach().clone(),
                    requires_grad=output.weight.requires_grad)
                first.out_features = hidden
                output.in_features = hidden
        # Remove storage and optimizer parameters, not merely output values.
        # Identity is a parameterless slot marker and is never invoked below.
        for index, label in enumerate(feature_map.labels):
            if label in ('cart', 'fav') and self.nonowner_residual_modes[label] == 'removed':
                self.residual_bank[index] = nn.Identity()
        # Every parent draw and reduction has finished. Only the two fully
        # upstream-detached private heads change; shared dense clipping still
        # makes exact equality of subsequent click updates an empirical check.
        for label, tower in zip(feature_map.labels, self.towers):
            if label in ('cart', 'fav'):
                self._rescale_private_initialization(
                    tower, input_width, tower_hidden_units,
                    private_hidden_scale, private_readout_scale)
        # All inherited draws, slicing and private-head initialization finish
        # first. Correct only the owner tower's retained marginal variance.
        # Common state and external RNG stay unchanged at construction; buy
        # predictions/context gradients and later shared updates can change.
        if owner_initialization_scale == 'xavier':
            for label, tower in zip(feature_map.labels, self.towers):
                if label == 'buy':
                    self._rescale_private_initialization(
                        tower, input_width, tower_hidden_units, 'xavier', 'xavier')
        # Every parent reset, random draw, slice and rescale is complete.
        # Rank zero is an exact no-op. Only buy's first affine is replaced;
        # retained nonlinear width, Dice buffers and second/readout layers
        # are untouched. The discarded affine has no persistent references.
        if owner_affine_rank:
            for label, tower in zip(feature_map.labels, self.towers):
                if label == 'buy':
                    tower[0] = _RankAffine(tower[0], owner_affine_rank,
                                           owner_factor_balance)
        # Install after ALL inherited initialization and reductions. Dense
        # mode is a strict no-op. Only packed Parameters and integer buffers
        # persist; the old affine and discarded coefficients have no owner.
        if owner_input_partition != 'dense':
            for label, tower in zip(feature_map.labels, self.towers):
                if label == 'buy':
                    tower[0] = _PartitionedAffine(
                        tower[0], len(self.user_fields) * embedding_dim,
                        self.key_width, self.value_width, embedding_dim,
                        owner_input_partition)
        # Only the buy first-affine context block changes. Fused mode is an
        # exact no-op. Replacement removes all old context backing storage.
        if owner_context_projection != 'fused':
            for label, tower in zip(feature_map.labels, self.towers):
                if label == 'buy':
                    tower[0] = _ContextAffine(
                        tower[0], len(self.user_fields) * embedding_dim,
                        owner_context_projection, owner_context_rank,
                        owner_context_scale, owner_context_balance,
                        owner_context_evaluation)
        self.model_to_device()

    @staticmethod
    def _rescale_private_initialization(tower, input_width, template_widths,
                                        hidden_scale, readout_scale):
        """Match retained weights' marginal variance without drawing weights.

        The upstream reset initializes tower Linear weights with Xavier normal
        and zero biases before slicing. A sliced coefficient retains variance
        2/(template fan_in + template fan_out). Multiply by the square root of
        template fan-sum / actual fan-sum to get dimension-specific variance.
        A shallow readout comes from the original final affine, hence its
        source fan-sum is template_widths[-1]+1 even though it now reads the
        first hidden layer. This does not restore the removed layer/function.
        Dice alpha/running buffers, biases and all residual weights are intact.
        This constructor-only operation adds no prediction-time arithmetic.
        """
        if hidden_scale == 'inherited' and readout_scale == 'inherited':
            return
        linears = [layer for layer in tower if isinstance(layer, nn.Linear)]
        if len(linears) not in (2, 3):
            raise ValueError('Expected shallow or deep inherited private tower')
        source_inputs = (input_width, template_widths[0])
        with torch.no_grad():
            if hidden_scale == 'xavier':
                for index, linear in enumerate(linears[:-1]):
                    source_sum = source_inputs[index] + template_widths[index]
                    actual_sum = linear.in_features + linear.out_features
                    linear.weight.mul_((source_sum / actual_sum) ** 0.5)
            if readout_scale == 'xavier':
                output = linears[-1]
                source_sum = template_widths[-1] + 1
                actual_sum = output.in_features + output.out_features
                output.weight.mul_((source_sum / actual_sum) ** 0.5)

    @staticmethod
    def _reduce_tower(tower, widths):
        """Fixed channel subnetwork of a fresh, untrained random template.

        Clone slices so a reduced Parameter owns only its reduced storage.
        Unchanged modules keep their original Parameter objects. The retained
        random weights use the original Xavier scale; no rescaling is applied.
        Initialization scale remains a stated alternative to capacity effects.
        This constructor helper is never called during forward or training.
        """
        previous = tower[0].in_features
        with torch.no_grad():
            for index, hidden in enumerate(widths):
                linear, dice = tower[2 * index], tower[2 * index + 1]
                if not isinstance(linear, nn.Linear) or not isinstance(dice, Dice):
                    raise ValueError('Unexpected parent tower structure')
                if linear.in_features != previous or linear.out_features != hidden:
                    linear.weight = nn.Parameter(
                        linear.weight[:hidden, :previous].detach().clone(),
                        requires_grad=linear.weight.requires_grad)
                    linear.bias = nn.Parameter(
                        linear.bias[:hidden].detach().clone(),
                        requires_grad=linear.bias.requires_grad)
                    linear.in_features = previous
                    linear.out_features = hidden
                if dice.bn.num_features != hidden:
                    dice.alpha = nn.Parameter(
                        dice.alpha[:hidden].detach().clone(),
                        requires_grad=dice.alpha.requires_grad)
                    dice.bn.running_mean = dice.bn.running_mean[:hidden].detach().clone()
                    dice.bn.running_var = dice.bn.running_var[:hidden].detach().clone()
                    dice.bn.num_features = hidden
                previous = hidden
            output = tower[-1]
            if not isinstance(output, nn.Linear) or output.out_features != 1:
                raise ValueError('Unexpected parent output layer')
            if output.in_features != previous:
                output.weight = nn.Parameter(
                    output.weight[:, :previous].detach().clone(),
                    requires_grad=output.weight.requires_grad)
                output.in_features = previous

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
        """Specialize the inherited available/norm branch without changing it.

        The constructor already requires match_support='available' and
        match_scaling='norm'. The discarded projections affect neither this
        returned coefficient nor state, RNG, learned values or gradients.
        Keep the live expression and operation order unchanged. The caller's
        coarse-extension gate (including both residual norms) is retained.
        """
        position_norm2 = position.square().sum(dim=(1, 2), keepdim=True)
        match_norm2 = affinity.square().sum(dim=(1, 2), keepdim=True)
        available = ((position_norm2 > 1e-10)
                     & (match_norm2 > 1e-10)).to(valid.dtype)
        available_norm = affinity * (position_norm2 / match_norm2.clamp_min(1e-10)).sqrt() * available
        return available_norm

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

    def _evaluation_activation(self, target, values):
        """Factor only the evaluation first affine, retaining its Parameters.

        Training never enters this method. For keys k, target q and action a,
        the original affine is Wq*q + Wk*k + Wd*(q-k) + Wp*(q*k) + Wa*a + b.
        Merging gives (Wq+Wd)*q + (Wk-Wd)*k + Wp*(q*k) + Wa*a + b.
        These identities hold over real arithmetic; BF16 evaluation differs
        in rounding and GEMM shapes. No equality claim is made for BF16.
        Derived weights and projections are local tensors rebuilt each call,
        so every update/device move remains visible and no cache is retained.
        """
        if self.training:
            raise ValueError('Affine factoring is evaluation-only')
        batch, slots, _ = values.shape
        key_width = self.key_width
        keys = values[:, :, :key_width]
        action = values[:, :, key_width:]
        query = target[:, None, :].expand(-1, slots, -1)
        first = self.local_activation[0]
        weight = first.weight
        if self.evaluation_projection == 'merged_hoist':
            # Preserve all original Parameter objects and initialization.
            # These sums are differentiable expressions, not new Parameters.
            query_weight = weight[:, :key_width] + weight[:, 2 * key_width:3 * key_width]
            history_weight = torch.cat([
                weight[:, key_width:2 * key_width] - weight[:, 2 * key_width:3 * key_width],
                weight[:, 3 * key_width:]], dim=-1)
            query_projection = torch.nn.functional.linear(target, query_weight)[:, None, :]
            history_features = torch.cat([keys, query * keys, action], dim=-1)
        else:
            # Both split cells have the same decomposition and history GEMM.
            # Only the query GEMM row count differs: B*slots versus B.
            query_weight = weight[:, :key_width]
            history_weight = weight[:, key_width:]
            if self.evaluation_projection == 'split_repeat':
                query_projection = torch.nn.functional.linear(
                    query.reshape(batch * slots, key_width), query_weight).reshape(
                        batch, slots, first.out_features)
            else:
                query_projection = torch.nn.functional.linear(target, query_weight)[:, None, :]
            history_features = torch.cat([keys, query - keys, query * keys, action], dim=-1)
        history_projection = torch.nn.functional.linear(
            history_features.reshape(batch * slots, -1), history_weight, first.bias).reshape(
                batch, slots, first.out_features)
        hidden = (query_projection + history_projection).reshape(batch * slots, first.out_features)
        # Retain each Dice and later affine in the same order, with the same
        # buffers, masks and 30-row evaluation population as the reference.
        for index in range(1, len(self.local_activation)):
            layer = self.local_activation[index]
            if self.evaluation_dice in ('scorer', 'both') and isinstance(layer, Dice):
                hidden = self._evaluation_dice(layer, hidden)
            else:
                hidden = layer(hidden)
        return hidden

    def _evaluation_dice(self, layer, hidden):
        """Factor only the mixture, preserving the original frozen BN gate.

        Upstream Dice: p*x + alpha*(1-p)*x, p=sigmoid(BN(x)).
        Here: x * (alpha + p*(1-alpha)), using one broadcast addcmul.
        Original BN, epsilon, running statistics and sigmoid are unchanged.
        No constants/derived tensors persist between calls. This never runs
        during training or substitutes for any training Dice population.
        Alpha remains FP32 under the frozen autocast recipe. Compared with
        the upstream BF16 expression, the complement and p*x rounding sites
        change; real-arithmetic identity is not a BF16 equality claim.
        """
        if self.training or layer.training or layer.bn.training:
            raise ValueError('Dice mixture factoring is evaluation-only')
        probability = torch.sigmoid(layer.bn(hidden))
        mixture = torch.addcmul(layer.alpha, probability, 1 - layer.alpha)
        return hidden * mixture

    def forward(self, inputs):
        context, target, values, mask, auxiliary, detail, position_detail, block_detail, curvature_detail, joint_detail, match_counts, match_union = self._represent(inputs)
        batch = values.shape[0]
        if self.training:
            all_values, all_counts = self._compress_with_match(values, mask, auxiliary, match_union)
            # All four cells score 70 slots; only uniform slots enter the primary
            # readout. Gradient pairs share exact forward populations. Population
            # pairs preserve counts/empty rows, but change aligned support means.
            primary_weights = self._primary_activation(self._interactions(target, all_values))
            all_strata = self._pool_seven(primary_weights, all_values, all_counts)
            uniform = all_strata[:, 0, :]
            # Auxiliary training retains the original ordered 30-slot population
            # and the original scorer/item stop-gradient ownership. No extra
            # slots enter its training Dice batch statistics. During evaluation
            # Dice acts pointwise, so the first three primary strata are reused.
            values = all_values.reshape(batch, 10, 7, self.value_width)[:, :, :3, :].reshape(
                batch, 30, self.value_width)
            counts = all_counts.reshape(batch, 10, 7)[:, :, :3].reshape(batch, 30)
            primary_strata = all_strata[:, :3, :]

        else:
            # Evaluation Dice uses frozen running statistics, so scoring
            # discarded support rows cannot affect retained row activations.
            # Keep all three readout strata, including empty rows, in order.
            values, counts = self._compress_both(values, mask, auxiliary)
            primary_weights = self._evaluation_activation(target, values)
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
                # Stop only the nonselected task's upstream context gradient.
                # detach is forward-identical and changes neither this head's
                # own parameter gradients at a fixed state nor its loss weight.
                task_context = (aux_context if not self.training
                                or label in self.auxiliary_context_tasks
                                else aux_context.detach())
                task_target, base = aux_target, feedback
                task_detail = aux_detail
                task_position_detail = aux_position_detail
                task_block_detail = aux_block_detail
                task_curvature_detail = aux_curvature_detail
                task_joint_detail = aux_joint_detail
            if label in ('cart', 'fav') and self.nonowner_residual_modes[label] == 'removed':
                # These branches retain the inherited feedback-stratified
                # base, context and target, with their original detach gates.
                # Do not build unused residual inputs or allocate a correction.
                pooled = base
            else:
                signal = torch.cat([task_detail, task_position_detail, task_block_detail,
                                    task_curvature_detail, task_joint_detail], dim=-1)
                residual_features = torch.cat([task_context, task_target, base,
                                               match_counts.to(base.dtype), signal], dim=-1)
                correction = self.residual_bank[index](residual_features)
                pooled = base + has_history * correction
            features = torch.cat([task_context, task_target, pooled], dim=-1)
            if self.training or self.evaluation_dice not in ('towers', 'both'):
                logits = tower(features)
            else:
                logits = features
                for layer in tower:
                    if isinstance(layer, Dice):
                        logits = self._evaluation_dice(layer, logits)
                    else:
                        logits = layer(logits)
            predictions[label + '_pred'] = torch.sigmoid(logits)
        return predictions
