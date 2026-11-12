"""Prospective separation of effective item capacity from arithmetic layout.

Native15, zero-lifted15 and masked16 retain exactly the same fifteen item
coordinates in real arithmetic, while table storage and dense contraction
widths differ. All active initial coordinates come from the unchanged full16
source. Padding is constant, has no learnable map, and precedes compression.
The full16 cell retains all coordinates. BF16 trajectories need not agree.
No numerical equivalence, predictive benefit or novelty is asserted here.
"""

CONFIG = {'embedding_dim': 16, 'attention_hidden_units': [64, 32], 'tower_hidden_units': [128, 64], 'attention_normalization': 'none', 'compression_routes': 'uf', 'block_aux_scorer_grad': True, 'block_aux_context_grad': False, 'block_aux_item_grad': True, 'id_embedding_dim': 15, 'attribute_embedding_dim': 15, 'action_embedding_dim': 16, 'source_coupled_initialization': True, 'item_coordinate_mode': 'zero_lift16'}

import copy
import torch
from torch import nn
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbeddingDict
from unirank.pytorch.layers.activations import Dice


def _capacity_layout(feature_map, id_width, attribute_width, action_width):
    """Copy public metadata and honor ordered upstream table aliases.

    Aliases keep the maximum requested width. The constructor rejects a cell
    if sharing prevents its exact ID/attribute/context/action allocation.
    """
    if id_width not in (15, 16) or attribute_width not in (15, 16) or action_width != 16:
        raise ValueError('ID/attribute widths must be fifteen or sixteen; action stays sixteen')
    if ('item_id' not in feature_map.features
            or feature_map.features['item_id'].get('source') != 'item'
            or feature_map.features['item_id']['type'] == 'meta'):
        raise ValueError('The released advertisement item_id field is required')
    local_map = copy.copy(feature_map)
    local_map.features = copy.deepcopy(feature_map.features)
    owner_for, owner_widths = {}, {}
    for name, spec in feature_map.features.items():
        if spec['type'] == 'meta':
            continue
        if spec['type'] not in ('categorical', 'sequence'):
            raise ValueError('This contrast requires supplied categorical tables')
        if spec.get('feature_encoder') or 'pretrained_emb' in spec:
            raise ValueError('Encoded or pretrained fields need a separate audit')
        shared = spec.get('share_embedding')
        owner = owner_for[shared] if shared in owner_for else name
        owner_for[name] = owner
        if spec.get('source') == 'item':
            requested = id_width if name == 'item_id' else attribute_width
        elif spec.get('source') == 'action':
            requested = action_width
        else:
            requested = 16
        owner_widths[owner] = max(owner_widths.get(owner, 0), requested)
    widths = {name: owner_widths[owner] for name, owner in owner_for.items()}
    for name, width in widths.items():
        local_map.features[name]['embedding_dim'] = width
    return local_map, widths


class _NativeModel(MultiTaskModel):
    def __init__(self, feature_map, model_id='ItemActionCapacity', gpu=-1,
                 embedding_dim=16, id_embedding_dim=16, attribute_embedding_dim=16,
                 action_embedding_dim=16, item_coordinate_mode='native',
                 attention_hidden_units=(64, 32),
                 tower_hidden_units=(128, 64), attention_normalization='none',
                 compression_routes='uf', block_aux_scorer_grad=True,
                 block_aux_context_grad=False, block_aux_item_grad=True, num_tasks=None,
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
        if embedding_dim != 16:
            raise ValueError('Keep fallback embedding_dim=16; use ID/attribute capacity factors')
        if item_coordinate_mode not in ('native', 'zero_lift16', 'mask16'):
            raise ValueError('Unknown item coordinate mode')
        expected = 15 if item_coordinate_mode == 'zero_lift16' else 16
        if item_coordinate_mode != 'native' and (id_embedding_dim != expected or attribute_embedding_dim != expected):
            raise ValueError('Lift requires uniform15 tables; masking requires uniform16 tables')
        self.item_coordinate_mode = item_coordinate_mode
        local_map, self.table_widths = _capacity_layout(
            feature_map, id_embedding_dim, attribute_embedding_dim, action_embedding_dim)
        self.embedding_layer = FeatureEmbeddingDict(local_map, embedding_dim)
        for name, width in self.table_widths.items():
            if self.embedding_layer.embedding_layers[name].embedding_dim != width:
                raise ValueError('Shared embedding width does not match local layout')
        self.item_fields = [name for name, spec in feature_map.features.items()
                            if spec.get('source') == 'item' and spec['type'] != 'meta']
        self.user_fields = [name for name, spec in feature_map.features.items()
                            if spec.get('source') not in ('item', 'action')
                            and spec['type'] != 'meta']
        self.action_field = next((name for name, spec in feature_map.features.items()
                                  if spec.get('source') == 'action'), None)
        if self.action_field is None:
            raise ValueError('Historical action field is required')
        if any(self.table_widths[name] != (id_embedding_dim if name == 'item_id'
                                         else attribute_embedding_dim)
               for name in self.item_fields):
            raise ValueError('Embedding aliases prevent the requested ID/attribute widths')
        if self.table_widths[self.action_field] != action_embedding_dim:
            raise ValueError('Embedding aliases prevent the requested action width')
        if any(self.table_widths[name] != 16 for name in self.user_fields):
            raise ValueError('Context width must remain sixteen')
        self.auxiliary_indices = [list(feature_map.labels).index(label)
                                  for label in ('cart', 'fav', 'buy')]
        self.key_width = (sum(self.table_widths[name] for name in self.item_fields)
                          if self.item_coordinate_mode == 'native'
                          else 16 * len(self.item_fields))
        self.action_width = self.table_widths[self.action_field]
        self.value_width = self.key_width + self.action_width
        self.context_width = sum(self.table_widths[name] for name in self.user_fields)

        # Field-specific native widths feed scorer and towers. Hidden widths, Dice,
        # module order, compression and gradient gates remain inherited.
        # Fixed padding adds no map; zero-input dense columns remain counted.
        layers = []
        width = 4 * self.key_width + self.action_width
        for hidden in attention_hidden_units:
            layers.extend([nn.Linear(width, hidden), Dice(hidden)])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.local_activation = nn.Sequential(*layers)
        input_width = self.context_width + self.key_width + self.value_width
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
        if self.item_coordinate_mode != 'native':
            # Fixed zero insertion, identical for current target and history.
            # Source coordinate fifteen has no output or gradient path.
            item_emb = dict(item_emb)
            for name in self.item_fields:
                embedded = item_emb[name]
                item_emb[name] = torch.cat(
                    (embedded[..., :15], embedded.new_zeros(embedded.shape[:-1] + (1,))), dim=-1)
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
        context, target, values, mask, auxiliary = self._represent(inputs)
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

def _source_columns(item_widths, context_field_count, action_width=16):
    """Map variable-width fields into all semantic full16 input blocks."""
    if (not item_widths or any(width not in (15, 16) for width in item_widths)
            or context_field_count < 0 or action_width != 16):
        raise ValueError('Invalid ID/attribute source layout')
    full_key = 16 * len(item_widths)
    full_context = 16 * context_field_count
    key_columns = [16 * field + coordinate
                   for field, width in enumerate(item_widths)
                   for coordinate in range(width)]
    scorer_columns = [block * full_key + column
                      for block in range(4) for column in key_columns]
    scorer_columns += [4 * full_key + coordinate for coordinate in range(action_width)]
    tower_columns = list(range(full_context))
    tower_columns += [full_context + column for column in key_columns]
    tower_columns += [full_context + full_key + column for column in key_columns]
    tower_columns += [full_context + 2 * full_key + coordinate
                      for coordinate in range(action_width)]
    return scorer_columns, tower_columns


class Model(_NativeModel):
    def __init__(self, feature_map, model_id='ItemCapacityLayout', gpu=-1,
                 id_embedding_dim=15, attribute_embedding_dim=15,
                 action_embedding_dim=16, source_coupled_initialization=True,
                 item_coordinate_mode='zero_lift16', **kwargs):
        if not isinstance(source_coupled_initialization, bool):
            raise ValueError('Initialization coupling must be boolean')
        # Reject unsupported metadata before any RNG-consuming construction.
        _capacity_layout(feature_map, id_embedding_dim, attribute_embedding_dim,
                         action_embedding_dim)
        incoming_cpu_rng = torch.get_rng_state()
        reference = _NativeModel(
            feature_map, model_id=model_id, gpu=-1,
            id_embedding_dim=16, attribute_embedding_dim=16,
            action_embedding_dim=16, **kwargs)
        # Keep the same ending full16 CPU stream for the host's data sampler.
        # Construction uses no device random draws or manually chosen seeds.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(incoming_cpu_rng)
            super().__init__(
                feature_map, model_id=model_id, gpu=gpu,
                id_embedding_dim=id_embedding_dim,
                attribute_embedding_dim=attribute_embedding_dim,
                action_embedding_dim=action_embedding_dim,
                item_coordinate_mode=item_coordinate_mode, **kwargs)
        if source_coupled_initialization:
            self._copy_source_initialization(reference)
        del reference

    def _copy_source_initialization(self, reference):
        item_widths = ([self.table_widths[field] for field in self.item_fields]
                       if self.item_coordinate_mode == 'native'
                       else [16] * len(self.item_fields))
        scorer_columns, tower_columns = _source_columns(
            item_widths, len(self.user_fields), self.action_width)
        source_state = reference.state_dict()
        target_state = self.state_dict()
        if set(source_state) != set(target_state):
            raise ValueError('Source and candidate state names differ')
        tower_inputs = {'towers.' + str(index) + '.0.weight'
                        for index in range(self.num_tasks)}
        item_tables = {'embedding_layer.embedding_layers.' + field + '.weight':
                       self.table_widths[field] for field in self.item_fields}
        with torch.no_grad():
            for name, target in target_state.items():
                value = source_state[name]
                if name.startswith('embedding_layer.embedding_layers.') and name.endswith('.weight'):
                    if value.ndim != 2 or target.ndim != 2:
                        raise ValueError('Expected categorical embedding matrices')
                    width = item_tables.get(name, 16)
                    value = value[:, :width]
                elif name == 'local_activation.0.weight':
                    value = value[:, scorer_columns]
                elif name in tower_inputs:
                    value = value[:, tower_columns]
                if value.shape != target.shape:
                    raise ValueError('Unmapped source tensor shape: ' + name)
                target.copy_(value)
