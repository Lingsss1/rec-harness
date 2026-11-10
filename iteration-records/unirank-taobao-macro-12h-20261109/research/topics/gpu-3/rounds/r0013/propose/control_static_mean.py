"""Prospective feedback-memory conditioning of task relevance scorers.

Inherit the committed g3-r0012 private/linear compatibility architecture.
Condition its shared scorer preactivation on the mean of six target-field
query tokens, before or after the existing masked feedback-memory read.
A zero-start bias-free token_dim -> scorer_hidden_width map preserves the
initial inherited function, parameter states, and CPU RNG stream.
Cross memory/static source with private/mean task routing. All four arms own
identical parameters and execute both source summaries and the same operators.
The g0 memory, relation/moment paths, joint block and prediction heads remain.
Numerical host checks are pending; no new training results or novelty claimed.
"""
import math
import torch
from torch import nn
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbeddingDict
from unirank.pytorch.layers.activations import Dice


CONFIG = {'embedding_dim': 16, 'attention_hidden_units': [64, 32], 'tower_hidden_units': [128, 64], 'token_dim': 32, 'num_heads': 2, 'slot_mode': 'recency', 'representation_mode': 'replicas', 'content_mode': 'relation', 'count_mode': 'relation', 'moment_mode': 'linear', 'moment_component': 'full', 'residual_mode': 'copy', 'bank_mode': 'relation', 'content_bank_mode': 'relation', 'feedback_content_mode': 'factorized', 'memory_source': 'joint', 'memory_layout': 'collapsed', 'task_routing': 'mean', 'block_click_score_grad': False, 'block_aux_score_grad': False, 'scorer_hidden_width': 64, 'task_extra_basis': 'linear', 'scorer_context_source': 'static'}


class FeedbackMemory(nn.Module):
    """One-way, strictly masked read of target-weighted behavior summaries.

Validity depends only on the temporal bin, so every arm exposes identical
key masks. Per-feedback counts remain separate even in the collapsed arm.
Overlapping task masks are allowed. No signed-weight normalization is used.
"""

    def __init__(self, item_width, width, heads, source, layout):
        super().__init__()
        self.source = source
        self.layout = layout
        self.item_width = item_width
        self.heads = heads
        self.head_width = width // heads
        self.input_projection = nn.Linear(item_width + 1, width)
        self.identity = nn.Parameter(torch.empty(1, 5, 4, width))
        self.query_norm = nn.LayerNorm(width)
        self.key_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        nn.init.normal_(self.identity, std=0.02)
        nn.init.zeros_(self.output.weight)

    def _summaries(self, values, weights, temporal_masks, valid, task_masks):
        batch, length, _ = values.shape
        # Empty tensors still have five bins, four feedbacks and fixed width.
        if not length:
            content = values.new_zeros((batch, 5, 4, self.item_width),
                                       dtype=torch.float32)
            counts = values.new_zeros((batch, 5, 4, 1), dtype=torch.float32)
            return content, counts
        with torch.autocast(device_type=values.device.type, enabled=False):
            temporal = (temporal_masks.bool() & valid[:, None, :]).float()
            items = values[..., :self.item_width].float()
            weighted = temporal * weights.float()[:, None, :]
            total = torch.bmm(weighted, items) / math.sqrt(7.0)
            bin_count = temporal.sum(dim=-1, keepdim=True).clamp_min(1.0)
            joint_parts, rate_parts, count_parts = [], [], []
            for task_mask in task_masks:
                feedback = (task_mask.bool() & valid).float()[:, None, :]
                selected = temporal * feedback
                count = selected.sum(dim=-1, keepdim=True)
                joint = torch.bmm(weighted * feedback, items) / math.sqrt(7.0)
                joint_parts.append(joint)
                rate_parts.append(count / bin_count)
                count_parts.append(count)
            # Both inputs are computed in all arms for a matched forward path.
            joint = torch.stack(joint_parts, dim=2)
            rates = torch.stack(rate_parts, dim=2)
            counts = torch.stack(count_parts, dim=2)
            factorized = rates * total[:, :, None, :]
            content = joint if self.source == 'joint' else factorized
            collapsed = content.mean(dim=2, keepdim=True).expand_as(content)
            if self.layout == 'collapsed':
                content = collapsed
            log_counts = torch.log1p(counts) / math.log(101.0)
        return content, log_counts

    def forward(self, tokens, token_valid, values, weights, temporal_masks,
                valid, task_masks):
        content, counts = self._summaries(values, weights, temporal_masks,
                                         valid, task_masks)
        memory = self.input_projection(torch.cat([content, counts], dim=-1))
        memory = (memory + self.identity.to(memory.dtype)).flatten(1, 2)
        memory_valid = temporal_masks.bool().any(dim=-1)
        memory_valid = memory_valid[:, :, None].expand(-1, -1, 4).flatten(1)
        memory = memory * memory_valid.unsqueeze(-1).to(memory.dtype)
        batch, count, width = tokens.shape
        query = self.query(self.query_norm(tokens)).reshape(
            batch, count, self.heads, self.head_width).transpose(1, 2)
        key = self.key(self.key_norm(memory)).reshape(
            batch, 20, self.heads, self.head_width).transpose(1, 2)
        value = self.value(memory).reshape(
            batch, 20, self.heads, self.head_width).transpose(1, 2)
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            scores = torch.matmul(query.float(), key.float().transpose(-1, -2))
            scores = scores / math.sqrt(self.head_width)
            # Finite sentinel plus exact mask/renormalization handles all-empty
            # histories without NaNs or a dummy key carrying unavailable data.
            key_mask = memory_valid[:, None, None, :]
            scores = scores.masked_fill(~key_mask, -1e9)
            attention = torch.softmax(scores, dim=-1) * key_mask.float()
            attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        mixed = torch.matmul(attention.to(value.dtype), value)
        mixed = mixed.transpose(1, 2).reshape(batch, count, width)
        update = self.output(mixed)
        active = token_valid & memory_valid.any(dim=-1, keepdim=True)
        update = update * active.unsqueeze(-1).to(update.dtype)
        return (tokens + update.to(tokens.dtype)) * token_valid.unsqueeze(-1).to(tokens.dtype)


class JointBlock(nn.Module):
    """Pre-norm attention with invalid slots excluded as keys and outputs."""

    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 2 * width), nn.SiLU(),
                                 nn.Linear(2 * width, width))

    def forward(self, tokens, valid):
        batch, count, width = tokens.shape
        qkv = self.qkv(self.norm1(tokens)).reshape(
            batch, count, 3, self.heads, self.head_width)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # FP32 scores/softmax avoid overflow under the frozen BF16 autocast.
        # Explicitly disable autocast: .float() alone does not protect matmul.
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            scores = torch.matmul(query.float(), key.float().transpose(-1, -2))
            scores = scores / math.sqrt(self.head_width)
            scores = scores.masked_fill(~valid[:, None, None, :], float('-inf'))
            attention = torch.softmax(scores, dim=-1)
        mixed = torch.matmul(attention.to(value.dtype), value)
        mixed = mixed.transpose(1, 2).reshape(batch, count, width)
        tokens = tokens + self.output(mixed)
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens * valid.unsqueeze(-1).to(tokens.dtype)


class Model(MultiTaskModel):
    def __init__(self, feature_map, model_id='MemoryConditionedTaskScorerStudy', gpu=-1,
                 embedding_dim=16, attention_hidden_units=(64, 32),
                 tower_hidden_units=(128, 64), token_dim=32, num_heads=2,
                 slot_mode='recency', representation_mode='replicas',
                 content_mode='relation', count_mode='relation', moment_mode='linear',
                 moment_component='full', residual_mode='copy', bank_mode='relation',
                 content_bank_mode='relation', feedback_content_mode='factorized',
                 memory_source='joint', memory_layout='collapsed',
                 task_routing='mean', block_click_score_grad=False,
                 block_aux_score_grad=False, scorer_hidden_width=64,
                 task_extra_basis='linear', scorer_context_source='static',
                 num_tasks=None,
                 task='binary_classification', **kwargs):
        if num_tasks is not None and num_tasks != len(feature_map.labels):
            raise ValueError('Task count mismatch')
        if list(feature_map.labels) != ['is_click', 'cart', 'fav', 'buy']:
            raise ValueError('This study requires the frozen four-task order')
        if slot_mode not in ('pooled', 'recency'):
            raise ValueError('Unknown slot construction')
        if token_dim <= 0 or num_heads <= 0 or token_dim % num_heads:
            raise ValueError('token_dim must be divisible by num_heads')
        if task_routing not in ('private', 'mean', 'parent'):
            raise ValueError('Unknown task readout routing')
        if not isinstance(block_click_score_grad, bool) or not isinstance(block_aux_score_grad, bool):
            raise ValueError('Scorer-input stop switches must be booleans')
        if block_click_score_grad or block_aux_score_grad:
            raise ValueError('Shared-trunk study requires open scorer-input gradients')
        if len(attention_hidden_units) != 2:
            raise ValueError('Shared-trunk study requires two relevance hidden layers')
        if not isinstance(scorer_hidden_width, int) or isinstance(scorer_hidden_width, bool) or scorer_hidden_width <= 0:
            raise ValueError('scorer_hidden_width must be a positive integer')
        if task_extra_basis not in ('linear', 'copy'):
            raise ValueError('Extra task summary requires linear or copy basis')
        if scorer_context_source not in ('memory', 'static'):
            raise ValueError('Scorer context requires memory or static source')
        self.scorer_context_source = scorer_context_source
        self.task_routing = task_routing
        self.task_extra_basis = task_extra_basis
        super().__init__(feature_map, model_id=model_id, gpu=gpu,
                         num_tasks=len(feature_map.labels), task=task, **kwargs)
        if representation_mode != 'replicas':
            raise ValueError('This factorial study starts from replica_seven')
        if content_mode not in ('relation', 'replicas'):
            raise ValueError('Unknown summary content')
        if count_mode not in ('relation', 'temporal'):
            raise ValueError('Unknown explicit count channel')
        if moment_mode not in ('linear', 'quadratic', 'copy', 'zero'):
            raise ValueError('Unknown position moment')
        if moment_mode != 'linear':
            raise ValueError('Decomposition study freezes the linear rank basis')
        if moment_component not in ('within', 'occupancy', 'full', 'zero'):
            raise ValueError('Unknown relation position component')
        if moment_component != 'full':
            raise ValueError('This study retains the full parent moment')
        if residual_mode not in ('occupancy', 'copy', 'zero'):
            raise ValueError('Unknown residual summary')
        if residual_mode != 'copy':
            raise ValueError('Relation-bank study freezes the duplicate-full parent')
        if bank_mode not in ('relation', 'rotated', 'shared', 'zero'):
            raise ValueError('Unknown relation bank sharing rule')
        if bank_mode != 'relation':
            raise ValueError('Content study freezes the relation-specific rank bank')
        if content_bank_mode not in ('relation', 'rotated', 'moment', 'zero'):
            raise ValueError('Unknown content bank intervention')
        if content_bank_mode != 'relation':
            raise ValueError('Feedback study freezes relation-specific content maps')
        if feedback_content_mode not in ('joint', 'factorized', 'duplicate', 'zero'):
            raise ValueError('Unknown feedback-content intervention')
        if feedback_content_mode != 'factorized':
            raise ValueError('Memory study retains the factorized incumbent path')
        if memory_source not in ('joint', 'factorized'):
            raise ValueError('Unknown memory source')
        if memory_layout not in ('separate', 'collapsed'):
            raise ValueError('Unknown memory layout')
        self.feedback_content_mode = feedback_content_mode
        self.content_bank_mode = content_bank_mode
        self.bank_mode = bank_mode
        self.residual_mode = residual_mode
        self.moment_component = moment_component
        if content_mode != 'relation' or count_mode != 'relation':
            raise ValueError('Position study requires the frozen relation parent')
        self.moment_mode = moment_mode
        self.slot_mode = slot_mode
        self.representation_mode = representation_mode
        self.content_mode = content_mode
        self.count_mode = count_mode
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        self.item_fields = [name for name, spec in feature_map.features.items()
                            if spec.get('source') == 'item' and spec['type'] != 'meta']
        self.user_fields = [name for name, spec in feature_map.features.items()
                            if spec.get('source') not in ('item', 'action')
                            and spec['type'] != 'meta']
        self.action_field = next((name for name, spec in feature_map.features.items()
                                  if spec.get('source') == 'action'), None)
        if not self.item_fields or not self.user_fields or self.action_field is None:
            raise ValueError('Required released feature groups are missing')
        self.key_width = len(self.item_fields) * embedding_dim
        self.value_width = self.key_width + embedding_dim
        self.static_count = len(self.user_fields) + len(self.item_fields)
        # All moment arms keep the parent's 35 summary tokens and equal parameters.
        value_fields = self.item_fields + [self.action_field]
        expected = {'item_id', 'cate_id', 'campaign_id', 'customer_id',
                    'brand', 'price_bucket', 'action'}
        if set(value_fields) != expected or len(value_fields) != 7:
            raise ValueError('Expected the frozen six item fields plus action')
        self.group_count = len(value_fields)
        self.token_count = self.static_count + 5 * self.group_count
        # Keep the parent's nonrandom buffer and module construction order.
        selection = torch.zeros(self.group_count, self.value_width)
        selection.fill_(1.0 / math.sqrt(self.group_count))
        self.register_buffer('content_selection', selection)
        self.match_fields = list(self.item_fields)
        self.match_padding = {name: feature_map.features[name].get('padding_idx', 0)
                              for name in self.match_fields}
        layers = []
        width = 4 * self.key_width
        for hidden in attention_hidden_units:
            layers.extend([nn.Linear(width, hidden), Dice(hidden)])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.local_activation = nn.Sequential(*layers)
        self.field_projection = nn.Linear(embedding_dim, token_dim)
        self.slot_projection = nn.Linear(self.value_width + 1, token_dim)
        self.token_identity = nn.Parameter(torch.empty(1, self.token_count, token_dim))
        self.interaction = JointBlock(token_dim, num_heads)
        # Retain raw static fields and the original global DIN sum in all arms.
        input_width = (self.static_count * embedding_dim + self.value_width
                       + self.token_count * token_dim + 5)
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
        nn.init.normal_(self.token_identity, std=0.02)
        # Append AFTER every inherited initialization. The new path starts at
        # exactly zero, and its constructor cannot advance the parent's CPU
        # RNG stream (including any later host sampling using that stream).
        with torch.random.fork_rng(devices=[]):
            self.moment_projection = nn.Linear(self.value_width, token_dim, bias=False)
            nn.init.zeros_(self.moment_projection.weight)
        # A separate fork preserves all inherited initialization and the host
        # CPU RNG stream. Every arm has this same extra zero-initialized map.
        with torch.random.fork_rng(devices=[]):
            self.residual_projection = nn.Linear(self.value_width, token_dim, bias=False)
            nn.init.zeros_(self.residual_projection.weight)
        # Append after every inherited initialization; zeros consume no RNG.
        # All arms own exactly seven [value_width, token_dim] matrices.
        self.relation_bank = nn.Parameter(
            torch.zeros(self.group_count, self.value_width, token_dim))
        self.register_buffer('bank_routes', torch.tensor(
            self._bank_assignments(), dtype=torch.long))
        # Append AFTER all inherited initialization; zeros consume no RNG.
        # Same nominal parameter count and contraction in every new arm.
        self.content_bank = nn.Parameter(
            torch.zeros(self.group_count, self.value_width, token_dim))
        self.register_buffer('content_bank_routes', torch.tensor(
            self._content_bank_assignments(), dtype=torch.long))
        # Append after all inherited initialization, without consuming RNG.
        # Four maps shared over every bin/relation; only item fields enter.
        self.feedback_content_bank = nn.Parameter(
            torch.zeros(4, self.key_width, token_dim))
        # New modules are constructed only after inherited reset/initialization.
        # Forking CPU RNG preserves inherited weights and subsequent host draws.
        # Every cell has exactly the same added modules and zero-start output.
        with torch.random.fork_rng(devices=[]):
            self.feedback_memory = FeedbackMemory(
                self.key_width, token_dim, num_heads, memory_source, memory_layout)
        # Only the residual scorer architecture changes. Keep every inherited
        # initialization and the host CPU RNG stream intact. Separate head
        # hidden weights and Dice buffers avoid a redundant symmetric mean.
        # All output layers start at zero; the trunk learns after they open.
        with torch.random.fork_rng(devices=[]):
            self.task_trunk = nn.Sequential(
                nn.Linear(4 * self.key_width + embedding_dim, scorer_hidden_width),
                Dice(scorer_hidden_width))
            self.task_scorers = nn.ModuleList([
                nn.Sequential(nn.Linear(scorer_hidden_width, attention_hidden_units[1]),
                              Dice(attention_hidden_units[1]),
                              nn.Linear(attention_hidden_units[1], 1))
                for _ in feature_map.labels])
            for module in self.task_trunk.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight)
                    nn.init.zeros_(module.bias)
            for scorer in self.task_scorers:
                for module in scorer.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_normal_(module.weight)
                        nn.init.zeros_(module.bias)
                nn.init.zeros_(scorer[-1].weight)
                nn.init.zeros_(scorer[-1].bias)
        # Append after every inherited module initialization. This changes
        # neither inherited states nor the host CPU sampling RNG stream.
        # One shared map preserves identical extra capacity in both routing
        # arms; zero bias is structural, so empty history stays exactly zero.
        with torch.random.fork_rng(devices=[]):
            self.task_extra_projection = nn.Linear(
                self.value_width, self.value_width, bias=False)
            nn.init.zeros_(self.task_extra_projection.weight)
        # Append only after ALL inherited initialization. Both context sources
        # have the same zero-start map and preserve the host CPU RNG stream.
        # There is no bias: a zero source always gives a zero preactivation.
        with torch.random.fork_rng(devices=[]):
            self.task_context_projection = nn.Linear(
                token_dim, scorer_hidden_width, bias=False)
            nn.init.zeros_(self.task_context_projection.weight)
        self.model_to_device()

    def _task_pool(self, target, keys, actions, values, valid, scorer_context):
        """Retain inherited content and projected rank summaries.

        Scorers see target, visible item fields and historical actions, with
        a context shift before shared-trunk Dice. Routing still precedes both
        summaries. Rank uses visible order rather than timestamps. Wc=0
        preserves the corresponding R12 function at arbitrary inherited state.
        Inherited scorer outputs and P, as well as the new context map, start
        at zero. Scorer outputs must open before context/trunk/P gradients
        can become nonzero; validate gradient connectivity in a cloned state.
        """
        batch, length, _ = keys.shape
        if not length:
            return target.new_zeros((batch, self.num_tasks, self.value_width))
        query = target[:, None, :].expand(-1, length, -1)
        actions = actions.masked_fill(~valid.unsqueeze(-1), 0)
        interactions = torch.cat([query, keys, query - keys, query * keys, actions], dim=-1)
        flat = interactions.reshape(batch * length, -1)
        # Keep the inherited Dice input shape [B*L,H] and all event masks.
        # A sample-level history-conditioned shift is broadcast to its events;
        # it can change the nonlinear relevance function without L new maps.
        hidden = self.task_trunk[0](flat).reshape(batch, length, -1)
        context_bias = self.task_context_projection(scorer_context).to(hidden.dtype)
        hidden = hidden + context_bias[:, None, :]
        hidden = self.task_trunk[1](hidden.reshape(batch * length, -1))
        columns = [scorer(hidden) for scorer in self.task_scorers]
        scores = torch.cat(columns, dim=-1).reshape(batch, length, self.num_tasks)
        scores = scores.masked_fill(~valid.unsqueeze(-1), 0)
        # Same score/routing arithmetic in all factorial cells. FP32 averaging
        # avoids an unnecessary BF16 accumulation loss. Explicit broadcasting
        # makes mean routing exactly task-independent, including under BF16.
        center = scores.float().mean(dim=-1, keepdim=True).to(scores.dtype)
        mixing = 1.0 if self.task_routing == 'mean' else 0.0
        scores = (1.0 - mixing) * scores + mixing * center
        # Content path keeps the R9 content control's FP32 arithmetic.
        # Both summaries and the dense projection execute in every arm.
        # Protect signed cancellation and projection/addition from BF16
        # autocast, then restore the inherited value dtype once at the end.
        with torch.autocast(device_type=values.device.type, enabled=False):
            content_basis = valid.float()
            extra_basis = self._moment_coefficients(
                valid[:, None, :], mode=self.task_extra_basis).squeeze(1)
            content_weights = scores.float() * content_basis[:, :, None]
            extra_weights = scores.float() * extra_basis[:, :, None]
            content = torch.bmm(content_weights.transpose(1, 2), values.float())
            extra = torch.bmm(extra_weights.transpose(1, 2), values.float())
            result = content + self.task_extra_projection(extra)
        return result.to(values.dtype)

    def _feedback_source(self, joint, factorized, total):
        if self.feedback_content_mode == 'factorized':
            return factorized
        if self.feedback_content_mode == 'duplicate':
            return total
        return joint

    def _feedback_content_tokens(self, values, weights, relation_masks,
                                 valid, task_masks):
        """Preserve item-feedback binding or replace it by its marginals.

        Rates use unweighted counts, never a signed-weight denominator.
        Task masks can overlap and are intersected with valid history.
        All arms execute every summary and contraction; effective backward
        cost differs. Accumulate one feedback at a time to avoid materializing
        [B,5,7,4,key_width]. FP32 avoids cancellation/overflow in sums.
        """
        batch, length, _ = values.shape
        output = values.new_zeros((batch, 5, self.group_count,
                                   self.feedback_content_bank.shape[-1]),
                                  dtype=torch.float32)
        if not length:
            return output
        with torch.autocast(device_type=values.device.type, enabled=False):
            relation = relation_masks.float()
            counts = relation.sum(dim=-1, keepdim=True)
            item_values = values[..., :self.key_width].float()
            weighted = relation * weights.float()[:, None, None, :]
            total = torch.bmm(weighted.flatten(1, 2), item_values)
            total = total.reshape(batch, 5, self.group_count, self.key_width)
            total = total / math.sqrt(self.group_count)
            for index, task_mask in enumerate(task_masks):
                feedback = (task_mask.bool() & valid).float()[:, None, None, :]
                selected = relation * feedback
                rate = selected.sum(dim=-1, keepdim=True) / counts.clamp_min(1.0)
                joint = torch.bmm((weighted * feedback).flatten(1, 2), item_values)
                joint = joint.reshape(batch, 5, self.group_count, self.key_width)
                joint = joint / math.sqrt(self.group_count)
                factorized = rate * total
                source = self._feedback_source(joint, factorized, total)
                projected = torch.matmul(source, self.feedback_content_bank[index].float())
                if self.feedback_content_mode == 'zero':
                    projected = projected * 0.0
                output = output + projected
        return output

    def _content_bank_assignments(self):
        """Balance five uses per bank; vary sharing, never token positions."""
        return tuple(tuple((relation + window) % self.group_count
                           if self.content_bank_mode == 'rotated' else relation
                           for relation in range(self.group_count))
                     for window in range(5))

    def _content_bank_input(self, grouped, moments):
        """S versus already available M; both have scale 1/sqrt(7)."""
        if self.content_bank_mode == 'moment':
            return moments
        return grouped

    def _project_content_bank(self, grouped, moments):
        """Bias-free residual on an existing summary, before JointBlock.

        Moment is a redundant active parameterization of the parent's C[r]*M:
        equal nominal parameters, but smaller functional capacity than D[r]*S.
        Zero is an inactive-path replay, not an active capacity control.
        """
        source = self._content_bank_input(grouped, moments)
        routed = self.content_bank[self.content_bank_routes].to(source.dtype)
        projected = torch.einsum('btgv,tgvd->btgd', source, routed)
        if self.content_bank_mode == 'zero':
            projected = projected * 0.0
        return projected

    def _bank_assignments(self):
        """Seven banks, five uses each; rotate by window only in the control.

        A single global bank relabeling cannot undo the rotated assignment:
        relation r uses bank r in the global bin and r+t in window t.
        Only parameter sharing changes. No tokens, values or masks move.
        """
        return tuple(tuple((relation + window) % self.group_count
                           if self.bank_mode == 'rotated' else relation
                           for relation in range(self.group_count))
                     for window in range(5))

    def _project_relation_bank(self, moments):
        """Project existing M with C[r], a balanced rotation, or a shared mean.

        Zero weights start every arm at the duplicate-full parent function.
        The primary capacity-matched contrast is relation versus rotated.
        Shared has equal nominal capacity but a smaller functional subspace;
        zero is a replay check with an inactive path, not an active control.
        """
        bank = self.relation_bank
        if self.bank_mode == 'shared':
            bank = bank.mean(dim=0, keepdim=True).expand(self.group_count, -1, -1)
        routed = bank[self.bank_routes].to(moments.dtype)
        projected = torch.einsum('btgv,tgvd->btgd', moments, routed)
        if self.bank_mode == 'zero':
            projected = projected * 0.0
        return projected

    def _slot_masks(self, valid):
        """Global plus four rank quartiles, or five global copies."""
        all_events = valid[:, None, :]
        if self.slot_mode == 'pooled':
            content = all_events.expand(-1, 5, -1)
        else:
            ranks = valid.long().cumsum(dim=-1) - 1
            length = valid.long().sum(dim=-1, keepdim=True).clamp_min(1)
            bins = torch.div(4 * ranks, length, rounding_mode='floor').clamp(0, 3)
            parts = [(bins == index) & valid for index in range(4)]
            content = torch.cat([all_events, torch.stack(parts, dim=1)], dim=1)
        return content, content.sum(dim=-1)

    def _relation_masks(self, clean_item, valid, temporal_masks):
        """[B,5,7,L]: global plus six overlapping encoded-field matches.

        Zero/padding IDs never create equality evidence. Equality of any other
        encoded ID is not guaranteed to establish true entity identity (OOV or
        publisher missing-value codes may coincide). No action/label matching.
        """
        relations = [valid]
        for name in self.match_fields:
            history = clean_item[name][:, :-1]
            target = clean_item[name][:, -1:]
            match = (history == target) & (history > 0) & (target > 0) & valid
            padding = self.match_padding[name]
            if padding is not None:
                match = match & (history != padding) & (target != padding)
            relations.append(match)
        relations = torch.stack(relations, dim=1)
        return temporal_masks[:, :, None, :] & relations[:, None, :, :]

    def _moment_coefficients(self, temporal_masks, mode=None):
        """Relative valid rank, independent of padding IDs and absolute offsets.

        Center BEFORE restricting by a target relation, so a relation present
        only late versus early in a bin remains distinguishable. RMS is over
        that temporal bin, not over relation occupancy. The global bin also
        carries a global moment. No real timestamp/delay is inferred.
        """
        visible = temporal_masks.float()
        ranks = temporal_masks.long().cumsum(dim=-1).float() - 1.0
        counts = visible.sum(dim=-1, keepdim=True)
        safe_counts = counts.clamp_min(1.0)
        centered = (2.0 * ranks - counts + 1.0) * visible
        linear_rms = (centered.square().sum(dim=-1, keepdim=True)
                      / safe_counts).sqrt().clamp_min(1e-6)
        linear = centered / linear_rms
        squared = centered.square()
        quadratic = (squared - squared.sum(dim=-1, keepdim=True)
                     / safe_counts) * visible
        quadratic_rms = (quadratic.square().sum(dim=-1, keepdim=True)
                         / safe_counts).sqrt().clamp_min(1e-6)
        quadratic = quadratic / quadratic_rms
        # Always calculate both bases in all arms. In particular, a reversed
        # linear basis is NOT a distinct mechanism; its sign is absorbable.
        mode = self.moment_mode if mode is None else mode
        if mode == 'linear':
            return linear
        if mode == 'quadratic':
            return quadratic
        if mode == 'copy':
            return visible
        return visible * 0.0

    def _residual_moment(self, moments, occupancy):
        """Choose the input to B; all arms retain A applied to the full M."""
        if self.residual_mode == 'occupancy':
            return occupancy
        if self.residual_mode == 'copy':
            return moments
        return moments * 0.0

    def _position_moments(self, values, weights, relation_masks, temporal_masks):
        """M=W+O with an unweighted matched-event rank centroid.

        Keep the original temporal basis scale. Centering separately within
        a relation removes its mean placement, without dividing by signed
        learned weights or rescaling sparse relations. No-match input is zero;
        singleton/constant weighted-content W vanishes in exact arithmetic.
        """
        batch, length, _ = values.shape
        if not length:
            empty = values.new_zeros((batch, 5, self.group_count, self.value_width))
            return empty, self._residual_moment(empty, empty)
        # Preserve the parent's raw M accumulation for the full replay.
        # Common forward work in every arm; effective backward work can differ.
        with torch.autocast(device_type=values.device.type, enabled=False):
            coefficients = self._moment_coefficients(temporal_masks)
            moment_weights = (relation_masks.float() * weights.float()[:, None, None, :]
                              * coefficients[:, :, None, :])
            moments = torch.bmm(moment_weights.flatten(1, 2), values.float())
            moments = moments.reshape(batch, 5, self.group_count, self.value_width)
            moments = moments / math.sqrt(self.group_count)
            visible = relation_masks.float()
            centroid = (visible * coefficients[:, :, None, :]).sum(dim=-1, keepdim=True)
            centroid = centroid / visible.sum(dim=-1, keepdim=True).clamp_min(1.0)
            content_weights = visible * weights.float()[:, None, None, :]
            content = torch.bmm(content_weights.flatten(1, 2), values.float())
            content = content.reshape(batch, 5, self.group_count, self.value_width)
            content = content / math.sqrt(self.group_count)
            occupancy = centroid * content
            within = moments - occupancy
            if self.moment_component == 'full':
                result = moments
            elif self.moment_component == 'within':
                result = within
            elif self.moment_component == 'occupancy':
                result = occupancy
            else:
                result = moments * 0.0
        return result, self._residual_moment(moments, occupancy)

    def forward(self, inputs):
        if len(inputs) != 4 or inputs[3] is None:
            raise ValueError('Frozen historical task masks are required')
        user, item, mask, task_masks = self.get_inputs(inputs, return_multi_masks=True)
        if len(task_masks) != 4:
            raise ValueError('Expected four historical task masks')
        valid = mask.bool()
        batch, length = valid.shape
        # Sanitize padded IDs before embedding and before batch-coupled Dice.
        # Exclude the current action even if an upstream caller changes it.
        clean_item = {}
        for name in self.item_fields + [self.action_field]:
            history = item[name][:, :-1].masked_fill(~valid, 0)
            target = (torch.zeros_like(item[name][:, -1:])
                      if name == self.action_field else item[name][:, -1:])
            clean_item[name] = torch.cat([history, target], dim=1)
        user_emb = self.embedding_layer(user)
        item_emb = self.embedding_layer(clean_item)
        static = torch.stack([user_emb[name] for name in self.user_fields]
                             + [item_emb[name][:, -1] for name in self.item_fields], dim=1)
        target = torch.cat([item_emb[name][:, -1] for name in self.item_fields], dim=-1)
        keys = torch.cat([item_emb[name][:, :-1] for name in self.item_fields], dim=-1)
        keys = keys * valid.unsqueeze(-1).to(keys.dtype)
        action = item_emb[self.action_field][:, :-1]
        values = torch.cat([keys, action], dim=-1)
        values = values * valid.unsqueeze(-1).to(values.dtype)
        content_mask, counts = self._slot_masks(valid)
        relation_masks = self._relation_masks(clean_item, valid, content_mask)
        relation_counts = relation_masks.sum(dim=-1)
        if length:
            query = target[:, None, :].expand(-1, length, -1)
            interactions = torch.cat([query, keys, query - keys, query * keys], dim=-1)
            weights = self.local_activation(interactions.reshape(batch * length, -1))
            weights = weights.reshape(batch, length) * valid.to(weights.dtype)
            weighted_masks = content_mask.to(weights.dtype) * weights[:, None, :]
            pooled = torch.bmm(weighted_masks, values)
        else:
            pooled = target.new_zeros((batch, 5, self.value_width))
        slot_valid = counts > 0
        pooled = pooled * slot_valid.unsqueeze(-1).to(pooled.dtype)
        log_counts = torch.log1p(counts.float()) / math.log(101.0)
        log_counts = log_counts.to(pooled.dtype)
        if self.content_mode == 'relation':
            if length:
                relation_weights = (relation_masks.to(weights.dtype)
                                    * weights[:, None, None, :])
                grouped = torch.bmm(relation_weights.flatten(1, 2), values)
                grouped = grouped.reshape(batch, 5, self.group_count, self.value_width)
                grouped = grouped / math.sqrt(self.group_count)
            else:
                grouped = pooled.new_zeros((batch, 5, self.group_count, self.value_width))
        else:
            grouped = (pooled[:, :, None, :]
                       * self.content_selection[None, None, :, :].to(pooled.dtype))
        if self.count_mode == 'relation':
            group_counts = torch.log1p(relation_counts.float()) / math.log(101.0)
            group_counts = group_counts.to(pooled.dtype).unsqueeze(-1)
        else:
            group_counts = log_counts[:, :, None, None].expand(-1, -1, self.group_count, 1)
        slot_tokens = self.slot_projection(torch.cat([grouped, group_counts], dim=-1))
        # The inherited content/count projection and token validity remain
        # intact. The second summary enters the SAME token, before interaction.
        if not length:
            weights = target.new_zeros((batch, 0))
        moments, residual = self._position_moments(values, weights, relation_masks, content_mask)
        moment_tokens = self.moment_projection(moments.to(slot_tokens.dtype))
        slot_tokens = slot_tokens + moment_tokens
        residual_tokens = self.residual_projection(residual.to(slot_tokens.dtype))
        slot_tokens = slot_tokens + residual_tokens
        # Preserve temporal/relation identities and all existing summary paths.
        bank_tokens = self._project_relation_bank(moments.to(slot_tokens.dtype))
        slot_tokens = slot_tokens + bank_tokens
        content_tokens = self._project_content_bank(
            grouped.to(slot_tokens.dtype), moments.to(slot_tokens.dtype))
        slot_tokens = slot_tokens + content_tokens
        feedback_tokens = self._feedback_content_tokens(
            values, weights, relation_masks, valid, task_masks)
        slot_tokens = slot_tokens + feedback_tokens.to(slot_tokens.dtype)
        slot_tokens = slot_tokens.flatten(1, 2)
        # Validity is temporal in EVERY arm, so group occupancy is not leaked
        # by the attention mask when the explicit count input is ablated.
        # A no-match relation has zero content but an active identity token.
        group_valid = slot_valid[:, :, None].expand(-1, -1, self.group_count).flatten(1)
        tokens = torch.cat([self.field_projection(static), slot_tokens], dim=1)
        tokens = tokens + self.token_identity.to(tokens.dtype)
        static_valid = torch.ones((batch, self.static_count), dtype=torch.bool, device=valid.device)
        token_valid = torch.cat([static_valid, group_valid], dim=1)
        tokens = tokens * token_valid.unsqueeze(-1).to(tokens.dtype)
        # Static query tokens include the SAME field projection and identity
        # as their post-memory counterparts. Only this source selection differs
        # across controls. Both means execute, including in the static arms.
        target_start = len(self.user_fields)
        static_task_context = tokens[:, target_start:self.static_count].float().mean(dim=1)
        static_task_context = static_task_context.to(tokens.dtype)
        tokens = self.feedback_memory(tokens, token_valid, values, weights,
                                      content_mask, valid, task_masks)
        memory_task_context = tokens[:, target_start:self.static_count].float().mean(dim=1)
        memory_task_context = memory_task_context.to(tokens.dtype)
        scorer_context = (memory_task_context if self.scorer_context_source == 'memory'
                          else static_task_context)
        # Shared prediction representation follows the unchanged g0 path.
        tokens = self.interaction(tokens, token_valid)
        task_pooled = pooled[:, 0:1] + self._task_pool(
            target, keys, action, values, valid, scorer_context)
        # Parent routing is for initial-function/replay diagnostics only. It
        # still allocates and executes the new scorer family. No proposed
        # factorial arm uses this path, and private gradients are unused here.
        if self.task_routing == 'parent':
            task_pooled = pooled[:, 0:1].expand(-1, self.num_tasks, -1)
        outputs = {}
        for index, (label, tower) in enumerate(zip(self.feature_map.labels, self.towers)):
            features = torch.cat([static.flatten(1), task_pooled[:, index],
                                  tokens.flatten(1), log_counts], dim=-1)
            outputs[label + '_pred'] = torch.sigmoid(tower(features))
        return outputs
