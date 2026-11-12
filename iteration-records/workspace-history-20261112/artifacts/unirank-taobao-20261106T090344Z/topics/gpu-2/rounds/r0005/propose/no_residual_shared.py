"""Prospective private residuals for task-specific compressed histories.

The primary path retains the incumbent uniform summaries. Auxiliary tasks use
feedback-stratified summaries of base item/action embeddings plus optional
zero-initialized ID residuals. An independent gate controls auxiliary updates
to the base item/action tables. Residuals are shared among the three auxiliary
tasks but never enter click predictions. Context updates remain shared, and
auxiliary scorer parameter updates remain blocked. No new efficacy or priority
claim is made. Numerical checks and all training belong to the host.
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
    'block_aux_item_grad': False,
    'use_aux_item_residual': False,
}


class Model(MultiTaskModel):
    def __init__(self, feature_map, model_id='PrivateResidualCompression', gpu=-1,
                 embedding_dim=16, attention_hidden_units=(64, 32),
                 tower_hidden_units=(128, 64), attention_normalization='none',
                 compression_routes='uf', block_aux_scorer_grad=True,
                 block_aux_context_grad=False, block_aux_item_grad=False,
                 use_aux_item_residual=False, num_tasks=None,
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
            raise ValueError('This study retains shared auxiliary context updates')
        if not isinstance(use_aux_item_residual, bool):
            raise ValueError('Residual switch must be boolean')
        self.use_aux_item_residual = use_aux_item_residual
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

        # Inherit scorer and tower construction and RNG order. Both residual
        # cells add identical banks; no-residual controls retain original capacity.
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
        # Initialize the inherited network before allocating residual tables:
        # base embeddings, scorer and towers retain the incumbent RNG order.
        self.reset_parameters()
        # Register under embedding_layer for the host embedding optimizer.
        # Preserve any item/action table aliases; reject cross-source aliases
        # because they invalidate the intended parameter ownership contrast.
        residual_layers = nn.ModuleDict()
        if self.use_aux_item_residual:
            selected = self.item_fields + [self.action_field]
            base_layers = self.embedding_layer.embedding_layers
            context_ids = {id(base_layers[name]) for name in self.user_fields}
            aliases = {}
            for name in selected:
                base = base_layers[name]
                if not isinstance(base, nn.Embedding):
                    raise ValueError('Residual bank requires categorical item/action tables')
                if base.embedding_dim != embedding_dim:
                    raise ValueError('Residual bank requires uniform field width')
                if id(base) in context_ids:
                    raise ValueError('Cross-source embedding alias invalidates ownership')
                if id(base) not in aliases:
                    # Construction normally consumes CPU RNG even though
                    # weights are immediately zeroed. Restore that stream so
                    # the added bank cannot change subsequent host shuffling.
                    with torch.random.fork_rng(devices=[]):
                        residual = nn.Embedding(base.num_embeddings, base.embedding_dim,
                                                padding_idx=base.padding_idx)
                        nn.init.zeros_(residual.weight)
                    aliases[id(base)] = residual
                residual_layers[name] = aliases[id(base)]
        self.embedding_layer.auxiliary_residual_layers = residual_layers
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
        residual_target, residual_values = None, None
        if self.use_aux_item_residual:
            residual_layers = self.embedding_layer.auxiliary_residual_layers
            # All IDs come from the same filtered item dictionary as the base
            # embeddings. Never look up the unknown current action.
            residual_items = {name: residual_layers[name](item[name].long())
                              for name in self.item_fields}
            residual_target = torch.cat([residual_items[name][:, -1]
                                         for name in self.item_fields], dim=-1)
            residual_keys = torch.cat([residual_items[name][:, :-1]
                                       for name in self.item_fields], dim=-1)
            residual_action = residual_layers[self.action_field](
                item[self.action_field][:, :-1].long())
            residual_values = torch.cat([residual_keys, residual_action], dim=-1)
        return context, target, values, mask, auxiliary, residual_target, residual_values

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
        context, target, values, mask, auxiliary, residual_target, residual_values = self._represent(inputs)
        values, counts = self._compress_both(values, mask, auxiliary)
        if self.use_aux_item_residual:
            # Arithmetic compression commutes with residual addition; masks,
            # strata and event counts are identical for base and residuals.
            residual_values, _ = self._compress_both(residual_values, mask, auxiliary)
        # Both scorer calls include all 30 slots with identical memberships.
        # Residuals may change auxiliary values and their Dice distribution.
        # Retain the inherited slot population for the no-residual endpoints.
        primary_weights = self.local_activation(self._interactions(target, values))
        primary_strata = self._pool(primary_weights, values, counts)
        uniform = primary_strata[:, 0, :]

        if self.training or self.use_aux_item_residual:
            # Gate the BASE table path before adding residuals. The residual
            # bank always receives auxiliary gradients and no click gradients.
            aux_context = context
            aux_target = target.detach() if self.block_aux_item_grad else target
            aux_values = values.detach() if self.block_aux_item_grad else values
            if self.use_aux_item_residual:
                aux_target = aux_target + residual_target
                aux_values = aux_values + residual_values
            # Inference also needs this call when representations differ.
            # Functional detached parameters preserve input gradients, and
            # cloned buffers prevent auxiliary updates to persistent Dice state.
            aux_weights = self._auxiliary_activation(self._interactions(aux_target, aux_values))
            aux_strata = self._pool(aux_weights, aux_values, counts)
        else:
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
