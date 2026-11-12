"""Source-group table-capacity factorial; interaction width remains sixteen.

Only architecture-local embedding dimensions vary. The supplied feature map,
inputs, labels, table aliases, history compression, scorer, task towers and
auxiliary gradient gates are preserved. Narrow fields are copied within field
after masked compression; there are no learned projection operators.

Frozen unused maps are deliberately retained, as in the previous copy controls,
to preserve endpoint initialization/RNG consumption and count their storage.
Mixed-width cells need host numerical/masking/alias tests and measurements.
No novelty, retained-quality or speed claim is made by this proposal.
"""
import math
import copy
import torch
from torch import nn
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbeddingDict
from unirank.pytorch.layers.activations import Dice


CONFIG = {
    'embedding_dim': 16,
    'table_embedding_dim': 16,
    'context_table_embedding_dim': 8,
    'item_table_embedding_dim': 16,
    'attention_hidden_units': [64, 32],
    'tower_hidden_units': [128, 64],
    'attention_normalization': 'none',
    'compression_routes': 'uf',
    'block_aux_scorer_grad': True,
    'block_aux_context_grad': False,
    'block_aux_item_grad': True,
}


def _capacity_layout(feature_map, context_width, item_width):
    """Copy architecture metadata; never modify the host-owned feature map.

    Mirror the embedding library's ordered sharing rule. If an alias group
    crosses the two sources, keep the larger requested width for the whole
    group, preserving the table identity and both computational paths.
    """
    if context_width not in (8, 16) or item_width not in (8, 16):
        raise ValueError('Table widths must be 8 or 16')
    local_map = copy.copy(feature_map)
    local_map.features = copy.deepcopy(feature_map.features)
    owner_for = {}
    owner_widths = {}
    for name, spec in feature_map.features.items():
        if spec['type'] == 'meta':
            continue
        if spec['type'] not in ('categorical', 'sequence'):
            raise ValueError('This contrast requires the supplied categorical tables')
        if spec.get('feature_encoder') or 'pretrained_emb' in spec:
            raise ValueError('Encoded or pretrained fields require a separate audit')
        shared = spec.get('share_embedding')
        owner = owner_for[shared] if shared in owner_for else name
        owner_for[name] = owner
        requested = item_width if spec.get('source') in ('item', 'action') else context_width
        owner_widths[owner] = max(owner_widths.get(owner, 0), requested)
    widths = {name: owner_widths[owner] for name, owner in owner_for.items()}
    for name, width in widths.items():
        local_map.features[name]['embedding_dim'] = width
    return local_map, widths


class Model(MultiTaskModel):
    def __init__(self, feature_map, model_id='SourceGroupCapacity', gpu=-1,
                 embedding_dim=16, table_embedding_dim=16,
                 context_table_embedding_dim=None, item_table_embedding_dim=None,
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
        if embedding_dim != 16 or table_embedding_dim not in (8, 16):
            raise ValueError('Aligned maps require interaction width 16 and table width 8 or 16')
        self.table_embedding_dim = table_embedding_dim
        self.interaction_embedding_dim = embedding_dim
        context_width = (table_embedding_dim if context_table_embedding_dim is None
                         else context_table_embedding_dim)
        item_width = (table_embedding_dim if item_table_embedding_dim is None
                      else item_table_embedding_dim)
        local_map, self.table_widths = _capacity_layout(feature_map, context_width, item_width)
        self.embedding_layer = FeatureEmbeddingDict(local_map, table_embedding_dim)
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

        # Keep the source scorer/tower hidden widths. Only their input widths
        # follow the interaction width; table storage varies independently.
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
        # Construct after the source reset so projection initialization does not
        # change the source's embedding/tower initialization at matched widths.
        # Projection aliases follow actual shared table objects, including any
        # cross-source aliases; the host must still audit gradient ownership.
        self.embedding_projections = nn.ModuleDict()
        owners = []
        required = set(self.user_fields + self.item_fields + [self.action_field])
        for name, spec in feature_map.features.items():
            if name not in required:
                continue
            raw_width = self.table_widths[name]
            if name in self.embedding_layer.feature_encoders:
                raise ValueError('Encoded fields need a separate projection audit')
            table = self.embedding_layer.embedding_layers[name]
            projection = next((project for owner, project in owners if owner is table), None)
            if projection is None:
                projection = nn.Linear(raw_width, embedding_dim, bias=False)
                # For independent isotropic input coordinates, keep mean output
                # coordinate variance equal across rectangular/square maps.
                gain = math.sqrt(max(1.0, embedding_dim / raw_width))
                nn.init.orthogonal_(projection.weight, gain=gain)
                # Keep the identical RNG consumption as the random ancestor,
                # then set a coordinate-aligned map without drawing randomness.
                # For 8->16, both copies initially expose every stored coordinate;
                # no output coordinate is initialized to zero. The paired arms
                # share these initial values; this arm freezes maps below.
                with torch.no_grad():
                    aligned = torch.eye(raw_width,
                                        dtype=projection.weight.dtype,
                                        device=projection.weight.device)
                    aligned = aligned.repeat(embedding_dim // raw_width, 1)
                    projection.weight.copy_(aligned)
                owners.append((table, projection))
            self.embedding_projections[name] = projection
        # Freeze map weights only; do not detach projected values.
        self.embedding_projections.requires_grad_(False)
        self.model_to_device()

    def _project_fields(self, values, fields):
        widths = [self.table_widths[name] for name in fields]
        if values.shape[-1] != sum(widths):
            raise ValueError('Unexpected raw embedding width')
        if all(width == self.interaction_embedding_dim for width in widths):
            return values
        # Preserve field order and incoming dtype. An eight-coordinate field
        # becomes [x, x]; a sixteen-coordinate field passes through unchanged.
        pieces = torch.split(values, widths, dim=-1)
        return torch.cat([piece for piece, width in zip(pieces, widths)
                          for _ in range(self.interaction_embedding_dim // width)], dim=-1)

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
        # Masked arithmetic means commute with a bias-free field projection.
        # Apply it to the 30 compressed slots, not all 100 historical events.
        values, counts = self._compress_both(values, mask, auxiliary)
        context = self._project_fields(context, self.user_fields)
        target = self._project_fields(target, self.item_fields)
        values = self._project_fields(values, self.item_fields + [self.action_field])
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
