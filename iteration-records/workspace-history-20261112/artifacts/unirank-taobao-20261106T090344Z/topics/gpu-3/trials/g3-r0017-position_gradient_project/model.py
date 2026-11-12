"""Position-route task sharing and gradient ownership; prospective, unmeasured.

Inherit g3-r0014-position_collapsed without new parameters or RNG draws.
Factor 1: task-private position logits or the click row broadcast to all tasks.
Factor 2: joint loss gradients or stop only auxiliary position-logit gradients.
All cells execute the inherited four projections and both rank contractions.
Shared cells allocate all 320 position coefficients but use only the 80 in
the click row; allocated capacity is matched, functional capacity is not.
The shared joint/primary pair differs only in auxiliary backward ownership
at fixed weights. In task/primary, unused auxiliary positional rows remain
zero from initialization, while all other auxiliary pathways still train.

The self donor still multiplies click gate logits by 1.5. Click coefficient
L1 mass is preserved. Context and dense gradient clipping remain indirect
task couplings. Rank counts supplied events, not time. Feedback visibility,
labels, loss weights and host training are unchanged. No novelty or measured
performance claim is made. Tensor/autograd checks belong to the host.
"""
import copy
import math
import torch
from torch import nn
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbeddingDict
from unirank.pytorch.layers.activations import Dice

CONFIG = {'embedding_dim': 16, 'attention_hidden_units': [128, 32], 'tower_hidden_units': [128, 64], 'routing': 'task', 'feedback_in_score': True, 'selector_gradient': 'primary', 'auxiliary_selector': 'none', 'block_aux_context_gradient': False, 'auxiliary_item_mode': 'private', 'target_matches': True, 'feedback_identity': True, 'gate_scope': 'event', 'gate_feedback_identity': True, 'gate_transfer_source': 'self', 'gate_transfer_gradient': 'live', 'primary_gate_component': 'redistribute', 'gate_conditioning': 'local', 'conditioning_feedback_identity': False, 'position_scope': 'event', 'position_feedback_identity': False, 'position_sharing': 'shared', 'position_aux_gradient': 'joint', 'position_gradient_rule': 'project'}


class Model(MultiTaskModel):
    def __init__(self, feature_map, model_id='HistoryCompositionGate', gpu=-1,
                 embedding_dim=16, attention_hidden_units=(128, 32),
                 tower_hidden_units=(128, 64), routing='task',
                 feedback_in_score=True, selector_gradient='primary',
                 auxiliary_selector='none', block_aux_context_gradient=False,
                 auxiliary_item_mode='private', target_matches=True,
                 feedback_identity=True, gate_scope='event',
                 gate_feedback_identity=True, gate_transfer_source='self',
                 gate_transfer_gradient='live', primary_gate_component='redistribute',
                 gate_conditioning='local', conditioning_feedback_identity=False,
                 position_scope='event', position_feedback_identity=False,
                 position_sharing='shared', position_aux_gradient='joint',
                 position_gradient_rule='project',
                 num_tasks=None,
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
            raise ValueError('This evidence study fixes learned private auxiliary tables')
        if not isinstance(target_matches, bool) or not isinstance(feedback_identity, bool):
            raise ValueError('Evidence switches must be boolean')
        if set(labels) != {'is_click', 'cart', 'fav', 'buy'} or len(labels) != 4:
            raise ValueError('Exactly the four contracted tasks are required')
        if not all(name in feature_map.features for name in ('cate_id', 'brand')):
            raise ValueError('Category and brand fields are required')
        if not target_matches or not feedback_identity:
            raise ValueError('The inherited additive evidence must remain resolved and target matched')
        if gate_scope not in ('event', 'mean'):
            raise ValueError('gate_scope must be event or mean')
        if not isinstance(gate_feedback_identity, bool):
            raise ValueError('Gate feedback identity switch must be boolean')
        if gate_scope != 'event' or not gate_feedback_identity:
            raise ValueError('This transfer contrast fixes the resolved event gate')
        if gate_transfer_source not in ('auxiliary', 'self'):
            raise ValueError('Gate donor must be auxiliary or self')
        if gate_transfer_gradient not in ('detached', 'live'):
            raise ValueError('Gate donor gradient must be detached or live')
        if gate_transfer_source != 'self' or gate_transfer_gradient != 'live':
            raise ValueError('Component attribution fixes the incumbent live self gate')
        if primary_gate_component not in ('full', 'redistribute', 'scale', 'none'):
            raise ValueError('Unknown primary gate component')
        if primary_gate_component != 'redistribute':
            raise ValueError('Composition study fixes click mass-preserving redistribution')
        if gate_conditioning not in ('history', 'local'):
            raise ValueError('gate_conditioning must be history or local')
        if not isinstance(conditioning_feedback_identity, bool):
            raise ValueError('Conditioning identity switch must be boolean')
        self.gate_conditioning = gate_conditioning
        self.conditioning_feedback_identity = conditioning_feedback_identity
        if gate_conditioning != 'local' or conditioning_feedback_identity:
            raise ValueError('Position study fixes the local-collapsed incumbent route')
        if position_scope not in ('event', 'marginal'):
            raise ValueError('position_scope must be event or marginal')
        if not isinstance(position_feedback_identity, bool):
            raise ValueError('Position feedback identity switch must be boolean')
        self.position_scope = position_scope
        self.position_feedback_identity = position_feedback_identity
        if position_scope != 'event' or position_feedback_identity:
            raise ValueError('Ownership study fixes event position and collapsed feedback')
        if position_sharing not in ('task', 'shared'):
            raise ValueError('Position sharing must be task or shared')
        if position_aux_gradient not in ('joint', 'primary'):
            raise ValueError('Position gradient ownership must be joint or primary')
        self.position_sharing = position_sharing
        self.position_aux_gradient = position_aux_gradient
        if position_sharing != 'shared' or position_aux_gradient != 'joint':
            raise ValueError('Gradient study requires the shared joint forward graph')
        if position_gradient_rule not in ('project', 'norm', 'joint', 'primary'):
            raise ValueError('Unknown positional gradient rule')
        self.position_gradient_rule = position_gradient_rule
        self.position_primary_index = labels.index('is_click')
        self.primary_gate_component = primary_gate_component
        self.gate_transfer_source = gate_transfer_source
        self.gate_transfer_gradient = gate_transfer_gradient
        self.gate_scope = gate_scope
        self.gate_feedback_identity = gate_feedback_identity
        self.target_matches = target_matches
        self.feedback_identity = feedback_identity
        # Zero is the released missing-row sentinel. Also honor any declared
        # padding sentinel; do not infer vocabulary frequencies or raw IDs.
        self.match_padding = {name: feature_map.features[name].get('padding_idx')
                              for name in ('cate_id', 'brand')}
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
        self.evidence_width = (1 + len(labels)) * 4 * 2
        # A raw zero Parameter consumes no RNG and is registered after the
        # inherited reset. It is a dense parameter under the host optimizer.
        # No extra bias: empty histories receive exactly zero logit correction.
        self.evidence_weight = nn.Parameter(torch.zeros(len(labels), self.evidence_width))
        self.gate_width = (1 + len(labels)) * 4
        self.event_gate_weight = nn.Parameter(torch.zeros(len(labels), self.gate_width))
        # Registered after the complete inherited initialization. The new
        # route is zero at step zero; neither allocation nor controls draw RNG.
        self.composition_gate_weight = nn.Parameter(torch.zeros(
            len(labels), self.gate_width, self.gate_width))
        # Allocate after every inherited initialization without drawing RNG.
        self.position_gate_weight = nn.Parameter(torch.zeros(
            len(labels), self.gate_width, 4))
        self.model_to_device()

    def _history_evidence(self, item, mask, multi_masks):
        """Return ([B,40] statistics, [B,L,5] channels, [B,L,4] matches).

        Channel order is exposure, then feature_map.labels. Feedback masks may
        overlap; their counts are not mutually exclusive conversion counts.
        The identity control averages *features* across four feedback channels,
        preserving exposure features and the sum across feedback slots. It is
        not a union of feedback events and not a shuffled-label control.
        """
        valid = mask.to(dtype=torch.bool)
        if multi_masks is None or len(multi_masks) != len(self.towers):
            raise ValueError('Four host-provided historical feedback masks are required')
        if any(m.shape != mask.shape for m in multi_masks):
            raise ValueError('Feedback mask shape mismatch')
        matches = [valid]
        for name in ('cate_id', 'brand'):
            history = item[name][:, :-1]
            target = item[name][:, -1:]
            known = valid & history.ne(0) & target.ne(0)
            padding = self.match_padding[name]
            if padding is not None:
                known = known & history.ne(padding) & target.ne(padding)
            matches.append(known & history.eq(target) if self.target_matches else known)
        matches.append(matches[1] & matches[2])
        match_tensor = torch.stack(matches, dim=-1).float()  # [B,L,4]
        channels = torch.stack([valid] + [m.to(dtype=torch.bool) & valid
                                        for m in multi_masks], dim=-1).float()
        # These are input-only statistics. Force FP32 accumulation even under
        # BF16 host autocast; casts alone would still allow autocast bmm.
        with torch.autocast(device_type=mask.device.type, enabled=False):
            counts = torch.bmm(channels.transpose(1, 2), match_tensor)
            denominator = channels.sum(dim=1).unsqueeze(-1).clamp_min(1.0)
            log_counts = torch.log1p(counts) / math.log1p(max(mask.shape[1], 1))
            fractions = counts / denominator
            evidence = torch.stack([log_counts, fractions], dim=-1)  # [B,5,4,2]
            if not self.feedback_identity:
                shared = evidence[:, 1:].mean(dim=1, keepdim=True)
                evidence = torch.cat([evidence[:, :1],
                                      shared.expand(-1, len(self.towers), -1, -1)], dim=1)
        # Return the same input-only channel/match tensors to the routing path.
        # Counts and the original additive readout are identical to the parent.
        return evidence.reshape(mask.shape[0], self.evidence_width), channels, match_tensor

    def _composition_gate_logits(self, channels, matches, mask):
        """Input-only bilinear route; caller disables autocast for FP32.

        History means are within each example and use valid positions only.
        No cross-example statistics, learned embeddings, or current actions
        enter this route. Collapsing preserves exposure and the sum of the
        four feedback slots, including overlapping feedback masks.
        """
        if not self.conditioning_feedback_identity:
            collapsed = channels[:, :, 1:].mean(dim=-1, keepdim=True)
            channels = torch.cat([channels[:, :, :1],
                                  collapsed.expand(-1, -1, len(self.towers))], dim=-1)
        valid = mask.bool().unsqueeze(-1)
        features = (channels.unsqueeze(-1) * matches.unsqueeze(-2)).flatten(2)
        features = features.masked_fill(~valid, 0)
        count = valid.float().sum(dim=1).clamp_min(1.0)
        history_mean = features.sum(dim=1) / count
        history_logits, local_logits = [], []
        scale = math.sqrt(self.gate_width)
        # Avoid allocating [B,L,T,D] projected features at the frozen batch
        # size. Both replacements execute the same task loop in all cells.
        for task_index in range(len(self.towers)):
            projected = torch.nn.functional.linear(
                features, self.composition_gate_weight[task_index].transpose(0, 1)) / scale
            history_logits.append((projected * history_mean[:, None, :]).sum(dim=-1))
            local_logits.append((projected * features).sum(dim=-1))
        history_logits = torch.stack(history_logits, dim=-1)
        local_logits = torch.stack(local_logits, dim=-1)
        result = history_logits if self.gate_conditioning == 'history' else local_logits
        return result.masked_fill(~valid, 0)

    def _position_gate_logits(self, channels, matches, mask):
        """FP32 caller; input-only rank features and matched replacement.

        Mask ranks ignore padding even if there are holes or a different
        amount of left padding. Four fixed scales are in event counts, not
        hours. Empty histories have zero features and zero route correction.
        No cross-example statistics or current target actions are consumed.
        """
        if not self.position_feedback_identity:
            collapsed = channels[:, :, 1:].mean(dim=-1, keepdim=True)
            channels = torch.cat([channels[:, :, :1],
                                  collapsed.expand(-1, -1, len(self.towers))], dim=-1)
        valid = mask.bool()
        valid_float = valid.float()
        features = (channels.unsqueeze(-1) * matches.unsqueeze(-2)).flatten(2)
        features = features.masked_fill(~valid.unsqueeze(-1), 0)
        count = valid_float.sum(dim=1, keepdim=True)
        rank_from_right = (count - valid_float.cumsum(dim=1)).clamp_min(0)
        scales = features.new_tensor((1.0, 4.0, 16.0, 64.0))
        basis = torch.exp(-rank_from_right.unsqueeze(-1) / scales)
        basis = basis.masked_fill(~valid.unsqueeze(-1), 0)
        marginal = basis.sum(dim=1, keepdim=True) / count.clamp_min(1).unsqueeze(-1)
        event_logits, marginal_logits = [], []
        # Project to four kernels per task, avoiding [B,L,T,20,4].
        # Both replacements run in all cells, with identical allocated state.
        for task_index in range(len(self.towers)):
            projected = torch.nn.functional.linear(
                features, self.position_gate_weight[task_index].transpose(0, 1)) / 2.0
            event_logits.append((projected * basis).sum(dim=-1))
            marginal_logits.append((projected * marginal).sum(dim=-1))
        event_logits = torch.stack(event_logits, dim=-1)
        marginal_logits = torch.stack(marginal_logits, dim=-1)
        result = event_logits if self.position_scope == 'event' else marginal_logits
        # Always calculate the shared alternative. Select the click row by
        # label rather than assuming FeatureMap task order. Broadcast logits,
        # not an average of matrices, so the click gradient scale is retained.
        shared_result = (result * self._primary_channel).sum(dim=-1, keepdim=True)
        if self.position_sharing == 'shared':
            result = shared_result.expand_as(result)
        if self.position_aux_gradient == 'primary':
            # Only this input-only positional correction loses auxiliary
            # gradients; auxiliary predictions, towers, private embeddings,
            # score readout rows, and other gate parameters keep learning.
            result = torch.where(self._primary_channel, result, result.detach())
        return result.masked_fill(~valid.unsqueeze(-1), 0)

    def _event_route_gain(self, channels, matches, mask):
        """Bounded [B,L,T] gains with a donor added only to click logits.

        Masked positions have neutral gain and zero pooling scores. Empty
        histories still have zero additive evidence and zero pooled vectors.
        No current action or current label participates in donor computation.
        """
        with torch.autocast(device_type=mask.device.type, enabled=False):
            if not self.gate_feedback_identity:
                collapsed = channels[:, :, 1:].mean(dim=-1, keepdim=True)
                channels = torch.cat([channels[:, :, :1],
                                      collapsed.expand(-1, -1, len(self.towers))], dim=-1)
            event_features = (channels.unsqueeze(-1) * matches.unsqueeze(-2)).flatten(2)
            gate_logits = torch.nn.functional.linear(
                event_features, self.event_gate_weight)
            gate_logits = gate_logits + self._composition_gate_logits(
                channels, matches, mask)
            gate_logits = gate_logits + self._position_gate_logits(
                channels, matches, mask)
            # Both donor alternatives are calculated in every variant. The
            # contracted four labels may be ordered arbitrarily by FeatureMap.
            primary_logits = (gate_logits * self._primary_channel).sum(
                dim=-1, keepdim=True)
            auxiliary_logits = (gate_logits * (~self._primary_channel)).sum(
                dim=-1, keepdim=True) / (len(self.towers) - 1)
            donor = (auxiliary_logits if self.gate_transfer_source == 'auxiliary'
                     else primary_logits)
            if self.gate_transfer_gradient == 'detached':
                donor = donor.detach()
            # Detachment changes only backward flow. In auxiliary/detached,
            # click cannot rewrite donor gate rows through this transfer path.
            # Shared context and host clipping remain indirect coupling paths.
            mixed_logits = torch.where(
                self._primary_channel, gate_logits + 0.5 * donor, gate_logits)
            modulation = torch.tanh(mixed_logits)
            valid = mask.to(dtype=torch.bool).unsqueeze(-1)
            modulation = modulation.masked_fill(~valid, 0)
            denominator = valid.float().sum(dim=1, keepdim=True).clamp_min(1.0)
            mean_modulation = modulation.sum(dim=1, keepdim=True) / denominator
            applied = modulation if self.gate_scope == 'event' else mean_modulation
            return (1.0 + applied).expand_as(modulation).masked_fill(~valid, 1.0)

    def _apply_gate_components(self, scores, gain, mask):
        """Factor coefficient L1 mass and event redistribution for click.

        All four cells compute the same intermediates without additional RNG
        or state. Auxiliaries always use the inherited full gate. No detach:
        each intervention includes its induced optimization geometry.
        """
        with torch.autocast(device_type=mask.device.type, enabled=False):
            base = scores.float().masked_fill(~mask.bool().unsqueeze(-1), 0)
            gated = base * gain
            base_mass = base.abs().sum(dim=1, keepdim=True)
            gated_mass = gated.abs().sum(dim=1, keepdim=True)
            # Normalize coefficients before restoring mass, avoiding an
            # unbounded gain ratio at zero-weight positions. Never divide by
            # zero even in an unselected torch.where branch.
            safe_base = torch.where(base_mass > 0, base_mass,
                                    torch.ones_like(base_mass))
            safe_gated = torch.where(gated_mass > 0, gated_mass,
                                     torch.ones_like(gated_mass))
            redistributed = torch.where(gated_mass > 0,
                                        (gated / safe_gated) * base_mass, base)
            scaled = (base / safe_base) * gated_mass
            if self.primary_gate_component == 'full':
                primary_scores = gated
            elif self.primary_gate_component == 'redistribute':
                primary_scores = redistributed
            elif self.primary_gate_component == 'scale':
                primary_scores = scaled
            else:
                primary_scores = base
            result = torch.where(self._primary_channel, primary_scores, gated)
            # BF16 rounding may slightly perturb FP32 mass conservation.
            return result.to(dtype=scores.dtype)

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
        if len(inputs) != 4 or inputs[3] is None:
            raise ValueError('Historical feedback masks must be supplied by the host')
        user, item, mask, multi_masks = self.get_inputs(inputs, return_multi_masks=True)
        evidence, channels, matches = self._history_evidence(item, mask, multi_masks)
        correction = torch.nn.functional.linear(evidence, self.evidence_weight)
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
            gain = self._event_route_gain(channels, matches, mask)
            # Factor only click's gate. Private item tables, auxiliary gates,
            # additive evidence and primary selector ownership stay inherited.
            # Shared context and host clipping still couple the four losses.
            scores = self._apply_gate_components(scores, gain, mask)
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
            logits = tower(features)
            outputs[label + '_pred'] = torch.sigmoid(
                logits + correction[:, index:index + 1].to(dtype=logits.dtype))
        return outputs


    def training_loss(self, inputs, probabilities, train_labels):
        """Four BCE values; alter only one tensor's auxiliary gradient.

        Host owns backward, clipping, optimizer steps, scheduling and labels.
        autograd.grad probes retain the original graph without writing .grad
        or replaying forward/BatchNorm. Detached corrections add no second
        derivatives. Every policy computes both probes for a matched recipe.
        """
        if probabilities.ndim != 2 or probabilities.shape[1] != len(self.towers):
            raise ValueError('Expected probabilities [B,4] in FeatureMap label order')
        if train_labels.shape != probabilities.shape:
            raise ValueError('Expected training labels matching probabilities')
        with torch.autocast(device_type=probabilities.device.type, enabled=False):
            # Use the supplied probabilities without changing host clipping.
            element_losses = torch.nn.functional.binary_cross_entropy(
                probabilities.float(), train_labels.float(), reduction='none')
            task_losses = element_losses.mean(dim=0)
            total = task_losses.sum()
            if not torch.is_grad_enabled() or not probabilities.requires_grad:
                return total
            primary_loss = task_losses[self.position_primary_index]
            auxiliary_loss = torch.stack([
                task_losses[i] for i in range(len(self.towers))
                if i != self.position_primary_index]).sum()
            primary_gradient = torch.autograd.grad(
                primary_loss, self.position_gate_weight,
                retain_graph=True, create_graph=False, allow_unused=True)[0]
            auxiliary_gradient = torch.autograd.grad(
                auxiliary_loss, self.position_gate_weight,
                retain_graph=True, create_graph=False, allow_unused=True)[0]
            # A zero-length history skips the complete route in forward.
            if primary_gradient is None:
                primary_gradient = torch.zeros_like(self.position_gate_weight)
            if auxiliary_gradient is None:
                auxiliary_gradient = torch.zeros_like(self.position_gate_weight)
            primary_gradient = primary_gradient.detach().float()
            auxiliary_gradient = auxiliary_gradient.detach().float()
            primary_norm = torch.linalg.vector_norm(primary_gradient)
            safe_primary_norm = torch.where(primary_norm > 0, primary_norm,
                                            torch.ones_like(primary_norm))
            primary_unit = primary_gradient / safe_primary_norm
            opposing_component = (auxiliary_gradient * primary_unit).sum().clamp(max=0)
            projected = auxiliary_gradient - opposing_component * primary_unit
            auxiliary_norm = torch.linalg.vector_norm(auxiliary_gradient)
            safe_auxiliary_norm = torch.where(auxiliary_norm > 0, auxiliary_norm,
                                              torch.ones_like(auxiliary_norm))
            retention = (torch.linalg.vector_norm(projected) /
                         safe_auxiliary_norm).clamp(min=0, max=1)
            norm_matched = auxiliary_gradient * retention
            if self.position_gradient_rule == 'project':
                desired_auxiliary = projected
            elif self.position_gradient_rule == 'norm':
                desired_auxiliary = norm_matched
            elif self.position_gradient_rule == 'joint':
                desired_auxiliary = auxiliary_gradient
            else:
                desired_auxiliary = torch.zeros_like(auxiliary_gradient)
            correction = (desired_auxiliary - auxiliary_gradient).detach()
            surrogate = (self.position_gate_weight.float() * correction).sum()
            # Parentheses ensure an exactly zero scalar at finite values.
            # d(loss)/d(position) = g_click + desired_auxiliary; all other
            # leaves retain their original gradients before global clipping.
            return total + (surrogate - surrogate.detach())
