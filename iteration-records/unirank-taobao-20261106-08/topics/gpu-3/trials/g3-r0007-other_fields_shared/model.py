"""Field-selective shared/private item values with a primary-owned selector.

Two independently switched groups are cate_id/brand and all remaining item/action
fields. A shared group supplies the current primary tables to auxiliary heads,
allowing auxiliary gradients into those tables. A private group supplies learned
auxiliary copies. Both target and historical values obey the same ownership.
All four factorial cells allocate and look up the same copied tables and preserve
the inherited initialization. Unused copies receive no loss gradient; effective
capacity and backward cost therefore differ and must be reported by the host.
The 00 cell reproduces g3-r0005-private_aux_items; 11 recovers tied-joint values.
There is no late cross-task residual. Context sharing and selector ownership stay
fixed. These are local tests of label-granularity-motivated field ownership.
STEM already studies field-selective task embeddings (Su et al., AAAI 2024,
https://arxiv.org/html/2308.13537v2, section 5.3); no originality is claimed.
Only frozen host inputs are consumed; the target action is never consumed.
"""
import copy
import torch
from torch import nn
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbeddingDict
from unirank.pytorch.layers.activations import Dice

CONFIG = {'embedding_dim': 16, 'attention_hidden_units': [128, 32], 'tower_hidden_units': [128, 64], 'routing': 'task', 'feedback_in_score': True, 'selector_gradient': 'primary', 'auxiliary_selector': 'none', 'block_aux_context_gradient': False, 'auxiliary_item_mode': 'private', 'share_label_fields': False, 'share_other_fields': True}


class Model(MultiTaskModel):
    def __init__(self, feature_map, model_id='FieldSelectiveAuxiliaryRepresentation', gpu=-1,
                 embedding_dim=16, attention_hidden_units=(128, 32),
                 tower_hidden_units=(128, 64), routing='task',
                 feedback_in_score=True, selector_gradient='primary',
                 auxiliary_selector='none', block_aux_context_gradient=False,
                 auxiliary_item_mode='private', share_label_fields=False,
                 share_other_fields=True, num_tasks=None,
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
        if auxiliary_item_mode != 'private':
            raise ValueError('Field ownership requires learned private fallback tables')
        if not isinstance(share_label_fields, bool) or not isinstance(share_other_fields, bool):
            raise ValueError('Field ownership switches must be boolean')
        self.share_label_fields = share_label_fields
        self.share_other_fields = share_other_fields
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
        label_fields = {'cate_id', 'brand'}
        if not label_fields.issubset(self.item_fields):
            raise ValueError('Frozen category and brand item fields are required')
        self.shared_auxiliary_fields = tuple(
            name for name in self.aux_item_fields
            if (share_label_fields if name in label_fields else share_other_fields))
        # An embedding alias crossing ownership groups would silently leak
        # updates. Reject it instead of changing the host feature definition.
        for name in self.aux_item_fields:
            for other in self.embedding_layer.embedding_layers:
                if name == other:
                    continue
                if (self.embedding_layer.embedding_layers[name]
                        is self.embedding_layer.embedding_layers[other]):
                    if (other not in self.aux_item_fields or
                            ((name in self.shared_auxiliary_fields) !=
                             (other in self.shared_auxiliary_fields))):
                        raise ValueError('Embedding alias crosses field ownership boundary')
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
        # Always execute the same lookups. A table's value is selected once per
        # field and reused for target and history. No blends, new dimensions,
        # extra Dice calls, loss changes, or new random draws are introduced.
        private_emb = self._private_item_embeddings(item)
        auxiliary_emb = {
            name: (item_emb[name] if name in self.shared_auxiliary_fields
                   else private_emb[name])
            for name in self.aux_item_fields}
        auxiliary_target = torch.cat([auxiliary_emb[name][:, -1]
                                      for name in self.item_fields], dim=-1)
        auxiliary_values = torch.cat([auxiliary_emb[name][:, :-1]
                                      for name in self.item_fields], dim=-1)
        auxiliary_values = auxiliary_values.masked_fill(~valid, 0)
        if self.action_field:
            auxiliary_action = auxiliary_emb[self.action_field][:, :-1].masked_fill(~valid, 0)
            auxiliary_values = torch.cat([auxiliary_values, auxiliary_action], dim=-1)
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
