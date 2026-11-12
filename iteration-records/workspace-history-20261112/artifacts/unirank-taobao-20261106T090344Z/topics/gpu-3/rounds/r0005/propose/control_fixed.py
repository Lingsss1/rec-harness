"""Auxiliary-owned item/action tables under a primary-owned history selector.

The primary click path and shared context tables follow the incumbent. Auxiliary
heads may learn one private item/action representation shared by cart/fav/buy.
A four-mode replacement suite separates adaptive private values from a frozen
initial basis and from tied primary-only/joint embedding updates. All modes have
identical allocated capacity and initialization, but active capacity and backward
cost differ. Clones start from the inherited initialization, never a checkpoint.
Related work: STEM (Su et al., AAAI 2024), arXiv:2308.13537v2, and the local DIN
adaptation. This is neither a STEM reproduction nor an originality claim.
Forward uses only the frozen host inputs; current target action is never used.
"""
import copy
import torch
from torch import nn
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbeddingDict
from unirank.pytorch.layers.activations import Dice

CONFIG = {'embedding_dim': 16, 'attention_hidden_units': [128, 32], 'tower_hidden_units': [128, 64], 'routing': 'task', 'feedback_in_score': True, 'selector_gradient': 'primary', 'auxiliary_selector': 'none', 'block_aux_context_gradient': False, 'auxiliary_item_mode': 'fixed'}


class Model(MultiTaskModel):
    def __init__(self, feature_map, model_id='PrivateAuxiliaryItemRepresentation', gpu=-1,
                 embedding_dim=16, attention_hidden_units=(128, 32),
                 tower_hidden_units=(128, 64), routing='task',
                 feedback_in_score=True, selector_gradient='primary',
                 auxiliary_selector='none', block_aux_context_gradient=False,
                 auxiliary_item_mode='fixed', num_tasks=None,
                 task='binary_classification', **kwargs):
        labels = list(feature_map.labels)
        if num_tasks is not None and num_tasks != len(labels):
            raise ValueError('Task count mismatch')
        if routing not in ('task', 'shared'):
            raise ValueError('routing must be task or shared')
        super().__init__(feature_map, model_id=model_id, gpu=gpu,
                         num_tasks=len(labels), task=task, **kwargs)
        if selector_gradient not in ('joint', 'primary'):
            raise ValueError('selector_gradient must be joint or primary')
        if selector_gradient == 'primary' and routing != 'task':
            raise ValueError('Primary selector gradients require task routing')
        if 'is_click' not in labels:
            raise ValueError('The primary task is_click is required')
        if auxiliary_selector not in ('none', 'detached', 'live', 'frozen'):
            raise ValueError('Unknown auxiliary_selector')
        if auxiliary_selector != 'none' and (routing != 'task' or selector_gradient != 'primary'):
            raise ValueError('Private auxiliary selectors require primary task routing')
        if routing != 'task' or selector_gradient != 'primary' or auxiliary_selector != 'none':
            raise ValueError('Embedding ownership study requires the incumbent primary selector')
        if not isinstance(block_aux_context_gradient, bool):
            raise ValueError('Context gradient switch must be boolean')
        if auxiliary_item_mode not in ('private', 'fixed', 'tied_primary', 'tied_joint'):
            raise ValueError('Unknown auxiliary item representation mode')
        if block_aux_context_gradient:
            raise ValueError('This study fixes context sharing; auxiliary_item_mode controls item ownership')
        self.auxiliary_item_mode = auxiliary_item_mode
        self.block_aux_context_gradient = block_aux_context_gradient
        self.auxiliary_selector = auxiliary_selector
        self.selector_gradient = selector_gradient
        self.register_buffer('_primary_channel', torch.tensor(
            [label == 'is_click' for label in labels], dtype=torch.bool).view(1, -1),
            persistent=False)
        self.routing = routing
        self.feedback_in_score = feedback_in_score
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        self.item_fields = [name for name, spec in feature_map.features.items()
                            if spec.get('source') == 'item' and spec['type'] != 'meta']
        self.user_fields = [name for name, spec in feature_map.features.items()
                            if spec.get('source') not in ('item', 'action')
                            and spec['type'] != 'meta']
        self.action_field = next((name for name, spec in feature_map.features.items()
                                  if spec.get('source') == 'action'), None)
        self.key_width = len(self.item_fields) * embedding_dim
        self.action_width = embedding_dim if self.action_field else 0
        self.value_width = self.key_width + self.action_width
        width = 4 * self.key_width + self.action_width
        layers = []
        for hidden in attention_hidden_units:
            layers.extend([nn.Linear(width, hidden), Dice(hidden)])
            width = hidden
        self.score_trunk = nn.Sequential(*layers)
        self.score_readout = nn.Linear(width, len(labels))
        self.towers = nn.ModuleList()
        input_width = len(self.user_fields) * embedding_dim + self.key_width + self.value_width
        for _ in labels:
            tower = []
            width = input_width
            for hidden in tower_hidden_units:
                tower.extend([nn.Linear(width, hidden), Dice(hidden)])
                width = hidden
            tower.append(nn.Linear(width, 1))
            self.towers.append(nn.Sequential(*tower))
        self.reset_parameters()
        # Equal initial score functions remove a task/shared initial-scale confound.
        # Task-specific rows subsequently diverge under their respective losses.
        with torch.no_grad():
            initial_weight = self.score_readout.weight[:1].clone()
            initial_bias = self.score_readout.bias[:1].clone()
            self.score_readout.weight.copy_(initial_weight.expand_as(self.score_readout.weight))
            self.score_readout.bias.copy_(initial_bias.expand_as(self.score_readout.bias))
        # Copy after the inherited reset and row tying: the original trunk,
        # embeddings, readout and towers keep exactly the parent's RNG stream.
        # Each copy has separate parameters AND Dice/BatchNorm buffers.
        self.aux_score_trunk = (copy.deepcopy(self.score_trunk)
                                if auxiliary_selector != 'none' else None)
        # Clone only item/action tables AFTER the full inherited initialization.
        # deepcopy preserves aliasing within the selected tables and consumes no
        # random numbers. No new call to reset_parameters is made after cloning.
        # All four modes allocate the same tables and execute the same lookups.
        self.aux_item_fields = list(self.item_fields)
        if self.action_field:
            self.aux_item_fields.append(self.action_field)
        for name in self.aux_item_fields:
            spec = feature_map.features[name]
            if spec['type'] not in ('categorical', 'sequence'):
                raise ValueError('Private item path requires categorical inputs')
        self.aux_item_layers = nn.ModuleDict(copy.deepcopy({
            name: self.embedding_layer.embedding_layers[name]
            for name in self.aux_item_fields}))
        self.aux_item_encoders = nn.ModuleDict(copy.deepcopy({
            name: self.embedding_layer.feature_encoders[name]
            for name in self.aux_item_fields
            if name in self.embedding_layer.feature_encoders}))
        self.model_to_device()

    def _history_scores(self, hidden, interactions):
        scores = self.score_readout(hidden)
        if self.aux_score_trunk is not None:
            if self.auxiliary_selector == 'frozen':
                # Same cloned weights and label-free BN updates, but no label
                # gradients into this trunk or its embedding inputs.
                with torch.no_grad():
                    auxiliary_hidden = self.aux_score_trunk(interactions.detach())
            else:
                auxiliary_inputs = (interactions if self.auxiliary_selector == 'live'
                                    else interactions.detach())
                auxiliary_hidden = self.aux_score_trunk(auxiliary_inputs)
            auxiliary_scores = self.score_readout(auxiliary_hidden)
            # Click learns only its trunk; auxiliaries share their own trunk.
            # Readout rows are still trained by their corresponding task loss.
            return torch.where(self._primary_channel, scores, auxiliary_scores)
        if self.selector_gradient == 'primary' and torch.is_grad_enabled():
            auxiliary_scores = self.score_readout(hidden.detach())
            scores = torch.where(self._primary_channel, scores, auxiliary_scores)
        return scores

    def _private_item_embeddings(self, item):
        # Separate target/history tables are NOT introduced: the same private
        # table is reused for both positions, as in the original item path.
        result = {}
        for name in self.aux_item_fields:
            value = self.aux_item_layers[name](item[name].long())
            if name in self.aux_item_encoders:
                value = self.aux_item_encoders[name](value)
            result[name] = value
        return result

    def forward(self, inputs):
        user, item, mask = self.get_inputs(inputs)
        user_emb = self.embedding_layer(user)
        item_emb = self.embedding_layer(item)
        target = torch.cat([item_emb[name][:, -1] for name in self.item_fields], dim=-1)
        keys = torch.cat([item_emb[name][:, :-1] for name in self.item_fields], dim=-1)
        valid = mask.to(dtype=torch.bool).unsqueeze(-1)
        keys = keys.masked_fill(~valid, 0)
        values = keys
        action = None
        if self.action_field:
            # The current unknown action position is never consumed.
            action = item_emb[self.action_field][:, :-1].masked_fill(~valid, 0)
            values = torch.cat([keys, action], dim=-1)
        context = torch.cat([user_emb[name] for name in self.user_fields], dim=-1)
        # No private representation is supplied to the selector in this study.
        # Selectors retain the inherited primary-owned input geometry, avoiding
        # a simultaneous selector rewrite or extra Dice statistics updates.
        if self.auxiliary_item_mode == 'private':
            private_emb = self._private_item_embeddings(item)
        else:
            with torch.no_grad():
                private_emb = self._private_item_embeddings(item)
        private_target = torch.cat([private_emb[name][:, -1]
                                    for name in self.item_fields], dim=-1)
        private_values = torch.cat([private_emb[name][:, :-1]
                                    for name in self.item_fields], dim=-1)
        private_values = private_values.masked_fill(~valid, 0)
        if self.action_field:
            private_action = private_emb[self.action_field][:, :-1].masked_fill(~valid, 0)
            private_values = torch.cat([private_values, private_action], dim=-1)
        if self.auxiliary_item_mode in ('private', 'fixed'):
            auxiliary_target, auxiliary_values = private_target, private_values
        elif self.auxiliary_item_mode == 'tied_primary':
            auxiliary_target, auxiliary_values = target.detach(), values.detach()
        else:
            auxiliary_target, auxiliary_values = target, values
        batch, length, _ = keys.shape
        if length:
            query = target[:, None, :].expand(-1, length, -1)
            interactions = torch.cat([query, keys, query - keys, query * keys], dim=-1)
            if action is not None:
                score_action = action if self.feedback_in_score else torch.zeros_like(action)
                interactions = torch.cat([interactions, score_action], dim=-1)
            # Mask before Dice statistics as well as after scoring. Padding IDs
            # cannot influence other rows through training-mode BatchNorm.
            interactions = interactions.masked_fill(~valid, 0)
            flat_interactions = interactions.reshape(batch * length, -1)
            hidden = self.score_trunk(flat_interactions)
            scores = self._history_scores(hidden, flat_interactions).reshape(batch, length, -1)
            if self.routing == 'shared':
                scores = scores.mean(dim=-1, keepdim=True).expand(-1, -1, len(self.towers))
            scores = scores.masked_fill(~valid, 0)
            pooled = torch.bmm(scores.transpose(1, 2), values)
            # Detach the representation, if requested, before pooling; never
            # detach scores or pooled auxiliary outputs. Every auxiliary score
            # readout row and prediction tower must continue to learn.
            auxiliary_pooled = torch.bmm(scores.transpose(1, 2), auxiliary_values)
            pooled = torch.where(self._primary_channel.unsqueeze(-1),
                                 pooled, auxiliary_pooled)
        else:
            pooled = target.new_zeros((batch, len(self.towers), self.value_width))
        outputs = {}
        for index, (label, tower) in enumerate(zip(self.feature_map.labels, self.towers)):
            task_context, task_target = context, target
            if label != 'is_click':
                if self.block_aux_context_gradient:
                    task_context = context.detach()
                task_target = auxiliary_target
            features = torch.cat([task_context, task_target, pooled[:, index]], dim=-1)
            outputs[label + '_pred'] = torch.sigmoid(tower(features))
        return outputs
