"""Prospective positive/negative auxiliary ownership of shared context gradients.

Parent: g3-r0024-context_all_control. All four task towers, input values,
private item representations and inherited history routing are retained.
Two switches allow positive and negative auxiliary examples, respectively,
to update shared context through their own tower input. The primary path
always stays live. All task losses and all tower gradients remain active.

Forward never reads labels. Each auxiliary task receives its own value-equal
clone of shared context. Only training_loss receives current training labels
and computes a blocked loss-to-context partial VJP. A zero-valued surrogate
subtracts only this derivative, preserving logits, BCE values and tower
parameter gradients, including with cross-example Dice/BatchNorm coupling. The gate is not a gradient of a reweighted scalar
objective: it is an explicit partial-derivative intervention. Dense clipping
and adaptive embedding accumulators can still mediate indirect effects.

All four cells allocate identical state and use the same FP32 BCE hook,
including a fixed [1e-7, 1-1e-7] probability clamp. Existing references lack
this declared loss hook, so historical endpoint reproduction is a host gate,
not assumed bitwise equivalence. Each minibatch has exactly one forward/loss
pair; ephemeral graph references are cleared by training_loss and on forward.
No labels enter prediction, no parameters are added, and no training has been
performed in the coding workspace. No originality or improvement is claimed.
"""
import copy
import math
import torch
from torch import nn
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbeddingDict
from unirank.pytorch.layers.activations import Dice

CONFIG = {'embedding_dim': 16, 'attention_hidden_units': [128, 32], 'tower_hidden_units': [128, 64], 'routing': 'task', 'feedback_in_score': True, 'selector_gradient': 'primary', 'auxiliary_selector': 'none', 'block_aux_context_gradient': False, 'auxiliary_item_mode': 'private', 'target_matches': True, 'feedback_identity': True, 'gate_scope': 'event', 'gate_feedback_identity': True, 'gate_transfer_source': 'self', 'gate_transfer_gradient': 'live', 'primary_gate_component': 'redistribute', 'gate_conditioning': 'local', 'conditioning_feedback_identity': False, 'position_scope': 'event', 'position_feedback_identity': False, 'position_sharing': 'task', 'position_aux_gradient': 'primary', 'rank_evidence_scope': 'event', 'rank_evidence_location': 'route', 'rank_feedback_identity': False, 'retain_position_route': False, 'rank_primary_gain': 2.0, 'rank_aux_gradient': 'primary', 'selector_input_space': 'private', 'selector_input_gradient': 'detached', 'selector_transfer_source': 'none', 'selector_transfer_gradient': 'detached', 'aux_context_space': 'shared', 'aux_context_gradient': 'live', 'context_cart_fav_live': True, 'context_buy_live': True, 'context_positive_live': True, 'context_negative_live': False}


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
                 position_sharing='task', position_aux_gradient='primary',
                 rank_evidence_scope='event', rank_evidence_location='route',
                 rank_feedback_identity=False, retain_position_route=False,
                 rank_primary_gain=2.0, rank_aux_gradient='primary',
                 selector_input_space='private', selector_input_gradient='detached',
                 selector_transfer_source='none',
                 selector_transfer_gradient='detached',
                 aux_context_space='shared', aux_context_gradient='live',
                 context_cart_fav_live=True, context_buy_live=True,
                 context_positive_live=True, context_negative_live=False,
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
        if position_sharing != 'task' or position_aux_gradient != 'primary':
            raise ValueError('Rank evidence study fixes inherited task/primary positions')
        if rank_evidence_scope not in ('event', 'marginal'):
            raise ValueError('Rank evidence scope must be event or marginal')
        if rank_evidence_location not in ('readout', 'route'):
            raise ValueError('Rank evidence location must be readout or route')
        if rank_evidence_scope != 'event' or rank_evidence_location != 'route':
            raise ValueError('Replacement study fixes event-linked rank routing')
        if not isinstance(rank_feedback_identity, bool) or not isinstance(retain_position_route, bool):
            raise ValueError('Identity and route-retention switches must be boolean')
        if rank_feedback_identity:
            raise ValueError('This study fixes collapsed rank feedback features')
        if isinstance(rank_primary_gain, bool) or rank_primary_gain not in (1.0, 2.0):
            raise ValueError('Click rank gain must be one or two')
        if rank_aux_gradient not in ('joint', 'primary'):
            raise ValueError('Rank gradient ownership must be joint or primary')
        if selector_input_space not in ('primary', 'private'):
            raise ValueError('Selector input space must be primary or private')
        if selector_input_gradient not in ('detached', 'live'):
            raise ValueError('Selector input gradient must be detached or live')
        if selector_transfer_source not in ('auxiliary', 'self', 'none'):
            raise ValueError('Unknown selector transfer source')
        if selector_transfer_gradient not in ('detached', 'live'):
            raise ValueError('Unknown selector transfer gradient')
        if selector_input_space != 'private' or selector_input_gradient != 'detached':
            raise ValueError('Transfer study fixes detached private selector inputs')
        if selector_transfer_source != 'none' or selector_transfer_gradient != 'detached':
            raise ValueError('Context study fixes the disabled selector transfer')
        if aux_context_space not in ('shared', 'private'):
            raise ValueError('Auxiliary context space must be shared or private')
        if aux_context_gradient not in ('live', 'detached'):
            raise ValueError('Auxiliary context gradients must be live or detached')
        if aux_context_space != 'shared' or aux_context_gradient != 'live':
            raise ValueError('Task ownership study requires shared live context before task boundaries')
        if not isinstance(context_cart_fav_live, bool) or not isinstance(context_buy_live, bool):
            raise ValueError('Task-group context derivative switches must be boolean')
        if not context_cart_fav_live or not context_buy_live:
            raise ValueError('Outcome study requires all auxiliary task groups live')
        if not isinstance(context_positive_live, bool) or not isinstance(context_negative_live, bool):
            raise ValueError('Outcome context derivative switches must be boolean')
        self.context_positive_live = context_positive_live
        self.context_negative_live = context_negative_live
        # Plain ephemeral graph state, never a parameter or persistent buffer.
        self._context_loss_inputs = None
        self.context_cart_fav_live = context_cart_fav_live
        self.context_buy_live = context_buy_live
        self.aux_context_space = aux_context_space
        self.aux_context_gradient = aux_context_gradient
        self.selector_transfer_source = selector_transfer_source
        self.selector_transfer_gradient = selector_transfer_gradient
        self.transfer_aux_indices = [labels.index(name) for name in ('cart', 'fav', 'buy')]
        self.selector_input_space = selector_input_space
        self.selector_input_gradient = selector_input_gradient
        self.rank_primary_gain = float(rank_primary_gain)
        self.rank_aux_gradient = rank_aux_gradient
        self.rank_feedback_identity = rank_feedback_identity
        self.retain_position_route = retain_position_route
        self.rank_evidence_scope = rank_evidence_scope
        self.rank_evidence_location = rank_evidence_location
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
        # Append after all inherited initialization; zero allocation draws no
        # RNG. Same parameter count/state/initial function in every cell.
        self.rank_evidence_weight = nn.Parameter(torch.zeros(
            len(labels), self.gate_width, 4))
        # Allocate in every factorial cell, after all inherited initialization.
        # deepcopy draws no RNG; parameters and Dice buffers are independent.
        self.aligned_aux_score_trunk = copy.deepcopy(self.score_trunk)
        # Appended after inherited initialization; dense, zero and RNG-neutral.
        self.selector_transfer_weight = nn.Parameter(torch.zeros(3))
        # Copy the context group once, retaining any aliases WITHIN the group.
        # No extra FeatureEmbeddingDict construction or reset: inherited RNG
        # draws, original state and initial predictions remain undisturbed.
        # Copy layers and encoders together to preserve their internal aliases.
        context_copies = copy.deepcopy({
            'layers': {name: self.embedding_layer.embedding_layers[name]
                       for name in self.user_fields},
            'encoders': {name: self.embedding_layer.feature_encoders[name]
                         for name in self.user_fields
                         if name in self.embedding_layer.feature_encoders}})
        self.aux_context_layers = nn.ModuleDict(context_copies['layers'])
        self.aux_context_encoders = nn.ModuleDict(context_copies['encoders'])
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

    def _rank_evidence_logits(self, channels, matches, mask):
        """Return task-private route [B,L,T] and readout [B,T] logits.

        The new ownership switch stops auxiliary derivatives through only
        this input-only positional correction. At a fixed parameter state it
        preserves forward values. Click derivatives and all other auxiliary
        paths remain live. Task rows are selected by label, never by index.
        """
        with torch.autocast(device_type=mask.device.type, enabled=False):
            valid = mask.bool()
            valid_float = valid.float()
            rank_channels = channels.float()
            common_feedback = rank_channels[:, :, 1:].mean(dim=-1, keepdim=True)
            collapsed_channels = torch.cat([
                rank_channels[:, :, :1],
                common_feedback.expand(-1, -1, len(self.towers))], dim=-1)
            if not self.rank_feedback_identity:
                rank_channels = collapsed_channels
            features = (rank_channels.unsqueeze(-1) *
                        matches.float().unsqueeze(-2)).flatten(2)
            features = features.masked_fill(~valid.unsqueeze(-1), 0)
            count = valid_float.sum(dim=1, keepdim=True)
            rank = (count - valid_float.cumsum(dim=1)).clamp_min(0)
            scales = features.new_tensor((1.0, 4.0, 16.0, 64.0))
            basis = torch.exp(-rank.unsqueeze(-1) / scales)
            basis = basis.masked_fill(~valid.unsqueeze(-1), 0)
            marginal = basis.sum(dim=1, keepdim=True) / count.clamp_min(1).unsqueeze(-1)
            event_logits, marginal_logits = [], []
            for task_index in range(len(self.towers)):
                projected = torch.nn.functional.linear(
                    features, self.rank_evidence_weight[task_index].transpose(0, 1)) / 2.0
                event_logits.append((projected * basis).sum(dim=-1))
                marginal_logits.append((projected * marginal).sum(dim=-1))
            event_logits = torch.stack(event_logits, dim=-1)
            marginal_logits = torch.stack(marginal_logits, dim=-1)
            selected = (event_logits if self.rank_evidence_scope == 'event'
                        else marginal_logits)
            selected = selected.masked_fill(~valid.unsqueeze(-1), 0)
            if self.rank_aux_gradient == 'primary':
                selected = torch.where(self._primary_channel,
                                       selected, selected.detach())
            # A valid-event average gives a bounded linear statistic per task,
            # without a learned bias; zero-length/all-empty histories are zero.
            readout = selected.sum(dim=1) / count.clamp_min(1)
            route = selected
            if self.rank_evidence_location == 'readout':
                route = torch.zeros_like(route)
            else:
                readout = torch.zeros_like(readout)
            return route, readout

    def _event_route_gain(self, channels, matches, mask, rank_route):
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
            inherited_position = self._position_gate_logits(channels, matches, mask)
            # Both cells execute this path. A zero factor retains a zero
            # gradient tensor, without mutating optimizer state or ownership.
            position_factor = 1.0 if self.retain_position_route else 0.0
            gate_logits = gate_logits + position_factor * inherited_position
            # Gain changes only the click positional correction, before the
            # inherited live self-donor factor and bounded modulation. All
            # auxiliary forward routes retain unit gain. A single route with
            # gain two matches the duplicate click function span, not its raw
            # parameter gradient norm or necessarily its optimization path.
            applied_rank = torch.where(self._primary_channel,
                                       self.rank_primary_gain * rank_route,
                                       rank_route)
            gate_logits = gate_logits + applied_rank
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

    def _history_scores(self, hidden, target, values, private_target,
                        private_values, valid):
        """Learned private selector; two explicit, local information-flow factors.

        The primary path is exactly the parent's score_readout(hidden). Auxiliary
        selector inputs can differ from auxiliary prediction values by design.
        Stop gradients before pairwise features to block every input derivative,
        while preserving gradients to the cloned trunk and auxiliary readout.
        No losses/labels, auxiliary prediction outputs or target actions are read.
        """
        scores = self.score_readout(hidden)
        if self.selector_input_space == 'private':
            aux_target, aux_values = private_target, private_values
        else:
            aux_target, aux_values = target, values
        if self.selector_input_gradient == 'detached':
            aux_target, aux_values = aux_target.detach(), aux_values.detach()
        aux_keys = aux_values[:, :, :self.key_width]
        batch, length, _ = aux_keys.shape
        aux_query = aux_target[:, None, :].expand(-1, length, -1)
        aux_interactions = torch.cat([
            aux_query, aux_keys, aux_query - aux_keys, aux_query * aux_keys], dim=-1)
        if self.action_field:
            aux_action = aux_values[:, :, self.key_width:]
            score_action = aux_action if self.feedback_in_score else torch.zeros_like(aux_action)
            aux_interactions = torch.cat([aux_interactions, score_action], dim=-1)
        # Padding never enters Dice via arbitrary item/action IDs. The same
        # zero-padding convention as the inherited primary trunk is retained.
        aux_interactions = aux_interactions.masked_fill(~valid, 0)
        aux_hidden = self.aligned_aux_score_trunk(
            aux_interactions.reshape(batch * length, -1))
        auxiliary_scores = self.score_readout(aux_hidden)
        scores = torch.where(self._primary_channel, scores, auxiliary_scores)
        return scores

    def _apply_selector_transfer(self, scores, raw_scores, mask):
        """Click-only score transfer with within-example, padding-aware statistics.

        Raw donor scores precede all event gates and coefficient mass controls.
        No batch-level normalization is introduced. Both donor alternatives
        are calculated in every cell, with no extra learned representation.
        The correction is exactly zero at the common zero initialization.
        """
        with torch.autocast(device_type=mask.device.type, enabled=False):
            valid = mask.bool().unsqueeze(-1)
            count = valid.float().sum(dim=1, keepdim=True).clamp_min(1.0)
            raw = raw_scores.float().masked_fill(~valid, 0)
            mean = raw.sum(dim=1, keepdim=True) / count
            centered = (raw - mean).masked_fill(~valid, 0)
            rms = (centered.square().sum(dim=1, keepdim=True) / count + 1e-6).sqrt()
            normalized = torch.tanh(centered / rms).masked_fill(~valid, 0)
            auxiliary = torch.stack([normalized[:, :, i]
                                     for i in self.transfer_aux_indices], dim=-1)
            primary_donor = (normalized * self._primary_channel).sum(dim=-1, keepdim=True)
            self_donor = primary_donor.expand_as(auxiliary)
            donor = self_donor if self.selector_transfer_source == 'self' else auxiliary
            if self.selector_transfer_gradient == 'detached':
                donor = donor.detach()
            correction = torch.nn.functional.linear(
                donor, torch.tanh(self.selector_transfer_weight).unsqueeze(0)) / 3.0
            enabled = 0.0 if self.selector_transfer_source == 'none' else 1.0
            # Range [0.5,1.5] prevents sign flips or a zero denominator for
            # nonzero coefficients. The disabled cell retains a zero gradient
            # tensor on the three new parameters, rather than unused state.
            gain = 1.0 + enabled * 0.5 * correction
            primary = (scores.float() * self._primary_channel).sum(dim=-1, keepdim=True)
            modulated = primary * gain
            mass = primary.abs().sum(dim=1, keepdim=True)
            modulated_mass = modulated.abs().sum(dim=1, keepdim=True)
            safe_mass = torch.where(mass > 0, mass, torch.ones_like(mass))
            safe_modulated = torch.where(modulated_mass > 0, modulated_mass,
                                         torch.ones_like(modulated_mass))
            # Residual form keeps an exactly zero correction at unit gain.
            residual = mass * (modulated / safe_modulated - primary / safe_mass)
            result = scores.float() + residual * self._primary_channel
            return result.masked_fill(~valid, 0).to(dtype=scores.dtype)

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

    def _private_context_embeddings(self, user):
        # Match the pinned FeatureEmbeddingDict.forward casts and encoders.
        # Consume only host-filtered features already used by the parent.
        result = {}
        for name in self.user_fields:
            feature_type = self.feature_map.features[name]['type']
            if feature_type == 'numeric':
                inp = user[name].float().view(-1, 1)
            elif feature_type in ('categorical', 'sequence'):
                inp = user[name].long()
            elif feature_type == 'embedding':
                inp = user[name].float()
            else:
                raise ValueError('Unsupported context feature type')
            value = self.aux_context_layers[name](inp)
            if name in self.aux_context_encoders:
                value = self.aux_context_encoders[name](value)
            result[name] = value
        return result

    def forward(self, inputs):
        # Discard a prior unused graph, including after failed diagnostics.
        self._context_loss_inputs = None
        if len(inputs) != 4 or inputs[3] is None:
            raise ValueError('Historical feedback masks must be supplied by the host')
        user, item, mask, multi_masks = self.get_inputs(inputs, return_multi_masks=True)
        evidence, channels, matches = self._history_evidence(item, mask, multi_masks)
        rank_route, rank_readout = self._rank_evidence_logits(channels, matches, mask)
        correction = torch.nn.functional.linear(evidence, self.evidence_weight)
        correction = correction.float() + rank_readout
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
        # All cells execute the same extra lookups and encoder calls. Detach
        # only auxiliary context inputs, never complete tower outputs/losses.
        private_context_emb = self._private_context_embeddings(user)
        private_context = torch.cat([private_context_emb[name]
                                     for name in self.user_fields], dim=-1)
        auxiliary_context = (private_context if self.aux_context_space == 'private'
                             else context)
        if self.aux_context_gradient == 'detached':
            auxiliary_context = auxiliary_context.detach()
        # Auxiliary target/history values remain private in every cell. The
        # auxiliary selector can separately read primary or private geometry.
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
            scores = self._history_scores(
                hidden, target, values, private_target, private_values, valid
            ).reshape(batch, length, -1)
            if self.routing == 'shared':
                scores = scores.mean(dim=-1, keepdim=True).expand(-1, -1, len(self.towers))
            scores = scores.masked_fill(~valid, 0)
            raw_selector_scores = scores
            gain = self._event_route_gain(channels, matches, mask, rank_route)
            # Factor only click's gate. Private item tables, auxiliary gates,
            # additive evidence and primary selector ownership stay inherited.
            # Context ownership is selected at the task tower below. Host
            # dense clipping still couples losses across all factorial cells.
            scores = self._apply_gate_components(scores, gain, mask)
            scores = self._apply_selector_transfer(scores, raw_selector_scores, mask)
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
        context_loss_inputs = []
        for index, (label, tower) in enumerate(zip(self.feature_map.labels, self.towers)):
            task_context, task_target = context, target
            if label != 'is_click':
                # Isolate this task's backward edge before concatenation.
                # Clone keeps exact forward values and avoids gating click or
                # another task when a loss-source VJP is removed at training loss.
                task_context = auxiliary_context.clone()
                if self.training and torch.is_grad_enabled():
                    context_loss_inputs.append((index, task_context))
                task_target = auxiliary_target
            features = torch.cat([task_context, task_target, pooled[:, index]], dim=-1)
            logits = tower(features)
            outputs[label + '_pred'] = torch.sigmoid(
                logits + correction[:, index:index + 1].to(dtype=logits.dtype))
        if self.training and torch.is_grad_enabled():
            self._context_loss_inputs = tuple(context_loss_inputs)
        return outputs

    def training_loss(self, inputs, probabilities, train_labels):
        """Gate loss-source contributions to context, including through Dice BN.

        The host invokes this once after the matching training forward. A
        partial VJP computes the blocked auxiliary loss contribution to its
        context edge without accumulating parameter .grad values. Subtracting
        a zero-valued, stopped-gradient surrogate removes exactly that edge
        contribution on the final backward. The complete BCE still updates
        every tower and every other live path. This also handles cross-example
        coupling inside the inherited Dice/BatchNorm tower correctly.
        """
        context_inputs = self._context_loss_inputs
        self._context_loss_inputs = None
        if not self.training or not torch.is_grad_enabled() or context_inputs is None:
            raise RuntimeError('training_loss requires one matching gradient-enabled training forward')
        if (probabilities.ndim != 2 or probabilities.shape != train_labels.shape
                or probabilities.shape[1] != len(self.towers)):
            raise ValueError('Expected aligned [B,4] probabilities and training labels')
        if len(context_inputs) != len(self.towers) - 1:
            raise RuntimeError('Every auxiliary context edge must be recorded')
        targets = train_labels.detach()
        with torch.autocast(device_type=probabilities.device.type, enabled=False):
            prediction = probabilities.float().clamp(1e-7, 1.0 - 1e-7)
            losses = torch.nn.functional.binary_cross_entropy(
                prediction, targets.float(), reduction='none')
            objective = losses.mean(dim=0).sum()
            surrogate = objective.new_zeros(())
            for index, context_input in context_inputs:
                if not context_input.requires_grad:
                    raise RuntimeError('Shared context must be differentiable during training')
                # Mask the originating loss, not its destination input row:
                # Dice batch statistics can mix samples before this VJP.
                positive = targets[:, index].gt(0.5)
                allowed = torch.where(positive,
                                      1.0 if self.context_positive_live else 0.0,
                                      1.0 if self.context_negative_live else 0.0)
                blocked_loss = (losses[:, index] * (1.0 - allowed)).mean()
                blocked_gradient, = torch.autograd.grad(
                    blocked_loss, context_input, retain_graph=True,
                    create_graph=False, allow_unused=False)
                # Exactly zero value at finite states; its derivative w.r.t.
                # context equals the detached blocked loss-to-context VJP.
                # Stopping the VJP prevents second derivatives and therefore
                # prevents this correction from changing tower gradients.
                displacement = context_input.float() - context_input.detach().float()
                surrogate = surrogate + (
                    displacement * blocked_gradient.detach().float()).sum()
            return objective - surrogate
