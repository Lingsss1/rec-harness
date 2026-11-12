"""Attribute the small g2 feedback-residual gain on the g0 joint backbone.

Cross centered-product versus content replacement with feedback fractions
versus repeated exposure counts. The archived incumbent supplies the fourth
(centered-product, feedback) cell. Three new standalone cells keep identical
modules, initialization, summary work, widths and token count. All inherited
feedback summaries are computed before choosing readout inputs. This tests
representation and added-readout explanations; it claims no new result.
"""
import math
import torch
from torch import nn
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbeddingDict
from unirank.pytorch.layers.activations import Dice


CONFIG = {'embedding_dim': 16, 'attention_hidden_units': [64, 32], 'tower_hidden_units': [128, 64], 'token_dim': 32, 'num_heads': 2, 'slot_mode': 'recency', 'representation_mode': 'replicas', 'content_mode': 'relation', 'count_mode': 'relation', 'moment_mode': 'linear', 'moment_component': 'full', 'residual_mode': 'copy', 'bank_mode': 'relation', 'content_bank_mode': 'relation', 'feedback_content_mode': 'factorized', 'memory_source': 'joint', 'memory_layout': 'collapsed', 'pair_content_source': 'joint', 'pair_count_source': 'joint', 'feedback_dim': 8, 'residual_hidden': 64, 'recent_window': 16, 'feedback_mode': 'product', 'rate_interpolation': 'count', 'shrinkage_strength': 4.0, 'feedback_moment_mode': 'factorized', 'centering_mode': 'user', 'support_mode': 'exposure', 'residual_content_source': 'centered_product', 'residual_rate_source': 'exposure'}


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
    def __init__(self, feature_map, model_id='TargetRelationStudy', gpu=-1,
                 embedding_dim=16, attention_hidden_units=(64, 32),
                 tower_hidden_units=(128, 64), token_dim=32, num_heads=2,
                 slot_mode='recency', representation_mode='replicas',
                 content_mode='relation', count_mode='relation', moment_mode='linear',
                 moment_component='full', residual_mode='copy', bank_mode='relation',
                 content_bank_mode='relation', feedback_content_mode='factorized',
                 memory_source='joint', memory_layout='collapsed',
                 pair_content_source=CONFIG['pair_content_source'],
                 pair_count_source=CONFIG['pair_count_source'],
                 feedback_dim=8, residual_hidden=64, recent_window=16,
                 feedback_mode='product', rate_interpolation='count',
                 shrinkage_strength=4.0, feedback_moment_mode='factorized',
                 centering_mode='user', support_mode=CONFIG['support_mode'],
                 residual_content_source=CONFIG['residual_content_source'],
                 residual_rate_source=CONFIG['residual_rate_source'],
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
        if pair_content_source not in ('joint', 'factorized'):
            raise ValueError('Unknown pair content source')
        if pair_count_source not in ('joint', 'factorized'):
            raise ValueError('Unknown pair count source')
        # Transfer only the inspected committed g2 residual configuration.
        # The g0 moment_mode remains the inherited linear rank basis.
        if (feedback_mode != 'product' or rate_interpolation != 'count'
                or feedback_moment_mode != 'factorized' or centering_mode != 'user'):
            raise ValueError('The transferred residual freezes the committed g2 mechanism')
        if feedback_dim <= 0 or residual_hidden <= 0 or recent_window <= 0:
            raise ValueError('Residual dimensions and window must be positive')
        if not math.isfinite(shrinkage_strength) or shrinkage_strength <= 0:
            raise ValueError('Shrinkage strength must be finite and positive')
        if support_mode not in ('diversity', 'exposure'):
            raise ValueError('Unknown residual reliability support')
        if residual_content_source not in ('centered_product', 'content'):
            raise ValueError('Unknown residual content source')
        if residual_rate_source not in ('feedback', 'exposure'):
            raise ValueError('Unknown residual rate source')
        self.residual_content_source = residual_content_source
        self.residual_rate_source = residual_rate_source
        self.feedback_dim = feedback_dim
        self.feedback_tasks = 4
        self.feedback_slots = 8
        self.recent_window = recent_window
        self.feedback_mode = feedback_mode
        self.rate_interpolation = rate_interpolation
        self.shrinkage_strength = float(shrinkage_strength)
        self.feedback_moment_mode = feedback_moment_mode
        self.centering_mode = centering_mode
        self.support_mode = support_mode
        self.pair_content_source = pair_content_source
        self.pair_count_source = pair_count_source
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
        # No constructor or random draw follows inherited initialization.
        # Relation zero is global; pair indices address only the six fields.
        pairs = [(left + 1, right + 1)
                 for left in range(len(self.match_fields))
                 for right in range(left + 1, len(self.match_fields))]
        self.pair_bank = nn.Parameter(
            torch.zeros(len(pairs), self.value_width + 1, token_dim))
        self.register_buffer('pair_left', torch.tensor(
            [left for left, right in pairs], dtype=torch.long))
        self.register_buffer('pair_right', torch.tensor(
            [right for left, right in pairs], dtype=torch.long))
        routes = torch.zeros(len(pairs), self.group_count)
        scale = 1.0 / math.sqrt(len(self.match_fields) - 1)
        for index, (left, right) in enumerate(pairs):
            routes[index, left] = scale
            routes[index, right] = scale
        self.register_buffer('pair_routes', routes)
        # Append after every g0 module/reset. Common weights and the caller's
        # CPU RNG stream stay equal to g0; residual modules match across cells.
        with torch.random.fork_rng(devices=[]):
            self.feedback_projection = nn.Linear(self.key_width, feedback_dim, bias=False)
            residual_width = self.feedback_slots * (self.feedback_tasks * (feedback_dim + 1) + 1)
            self.feedback_residual = nn.Sequential(
                nn.Linear(residual_width, residual_hidden), nn.SiLU(),
                nn.Linear(residual_hidden, self.feedback_tasks))
            nn.init.zeros_(self.feedback_residual[-1].weight)
            nn.init.zeros_(self.feedback_residual[-1].bias)
        self.model_to_device()

    def _pair_tokens(self, values, weights, relation_masks, relation_counts):
        """Conjunctive content/counts versus symmetric marginal estimates.

        Let n be temporal support, n_r a relation count, and S_r the parent's
        signed content sum (scaled by sqrt(7)). The marginal estimates are
        n_rs=n_r*n_s/n and S_rs=(n_s*S_r+n_r*S_s)/(2*n).
        These controls use only information already in the parent. They are
        approximations, not asserted data-generating independence models.
        All arms compute both sources; parameters and contractions match.
        """
        batch, length, _ = values.shape
        if not length:
            return values.new_zeros((batch, 5, self.group_count,
                                     self.pair_bank.shape[-1]))
        with torch.autocast(device_type=values.device.type, enabled=False):
            left = relation_masks.index_select(2, self.pair_left)
            right = relation_masks.index_select(2, self.pair_right)
            joint_mask = left & right
            joint_count = joint_mask.sum(dim=-1, keepdim=True).float()
            pair_weights = joint_mask.float() * weights.float()[:, None, None, :]
            joint_content = torch.bmm(pair_weights.flatten(1, 2), values.float())
            joint_content = joint_content.reshape(
                batch, 5, self.pair_bank.shape[0], self.value_width)
            joint_content = joint_content / math.sqrt(self.group_count)
            counts = relation_counts.float().unsqueeze(-1)
            support = counts[:, :, :1, :].clamp_min(1.0)
            left_count = counts.index_select(2, self.pair_left)
            right_count = counts.index_select(2, self.pair_right)
            factorized_count = left_count * right_count / support
            # Recompute marginal sums in the SAME FP32 arithmetic as joint
            # sums, rather than granting the joint arm higher precision than
            # the parent's BF16 grouped tensor. This adds no sample inputs.
            marginal_weights = relation_masks.float() * weights.float()[:, None, None, :]
            marginal_content = torch.bmm(marginal_weights.flatten(1, 2), values.float())
            marginal_content = marginal_content.reshape(
                batch, 5, self.group_count, self.value_width)
            marginal_content = marginal_content / math.sqrt(self.group_count)
            left_content = marginal_content.index_select(2, self.pair_left)
            right_content = marginal_content.index_select(2, self.pair_right)
            factorized_content = 0.5 * (
                (right_count / support) * left_content
                + (left_count / support) * right_content)
            content = (joint_content if self.pair_content_source == 'joint'
                       else factorized_content)
            count = (joint_count if self.pair_count_source == 'joint'
                     else factorized_count)
            log_count = torch.log1p(count) / math.log(101.0)
            source = torch.cat([content, log_count], dim=-1)
        # Summary accumulation is FP32. Projection follows host BF16 autocast,
        # like the incumbent content banks. No signed denominator is used.
        projected = torch.einsum('btpv,pvd->btpd', source, self.pair_bank)
        return torch.einsum('btpd,pg->btgd', projected,
                            self.pair_routes.to(projected.dtype))

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

    def _moment_coefficients(self, temporal_masks):
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
        if self.moment_mode == 'linear':
            return linear
        if self.moment_mode == 'quadratic':
            return quadratic
        if self.moment_mode == 'copy':
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

    def _feedback_membership(self, clean_item, valid):
        """Eight disjoint groups: old/recent x category/brand match bits.

        Recent means last sixteen prefix POSITIONS, matching the source;
        no event-time deltas are available. Padding never creates a match.
        """
        batch, length = valid.shape
        pos = torch.arange(length, device=valid.device)[None, :].expand(batch, -1)
        recent = pos >= max(length - self.recent_window, 0)
        matches = []
        for name in ('cate_id', 'brand'):
            history = clean_item[name][:, :-1]
            target = clean_item[name][:, -1:]
            match = (history == target) & (history > 0) & (target > 0) & valid
            padding = self.match_padding[name]
            if padding is not None:
                match = match & (history != padding) & (target != padding)
            matches.append(match)
        group = 2 * matches[0].long() + matches[1].long() + 4 * recent.long()
        return torch.stack([(group == i) & valid for i in range(self.feedback_slots)], dim=1)


    def _reliability_support(self, clean_item, valid, members, counts):
        """Return n^2 / sum_i n_i^2, or the original exposure count n.

        Histograms are slot-local and use only historical advertisement IDs.
        Sorting once and scattering into at most L run bins avoids an L x L
        equality matrix. Missing/nonpositive/padding IDs are singleton groups;
        other encoded-ID collisions remain an inherited benchmark limitation.
        Both modes execute the histogram, adding no modules, RNG, or gradients.
        Empty slots yield zero; unique IDs give n; one repeated ID gives one.
        This changes reliability only, never moment denominators or raw counts.
        """
        batch, length = valid.shape
        if not length:
            return counts
        ids = clean_item['item_id'][:, :-1]
        order = ids.argsort(dim=-1)
        sorted_ids = ids.gather(1, order)
        identified = (sorted_ids > 0) & valid.gather(1, order)
        padding = self.match_padding['item_id']
        if padding is not None:
            identified = identified & (sorted_ids != padding)
        starts = torch.ones_like(sorted_ids, dtype=torch.bool)
        starts[:, 1:] = ((sorted_ids[:, 1:] != sorted_ids[:, :-1])
                         | ~identified[:, 1:] | ~identified[:, :-1])
        groups = starts.long().cumsum(dim=-1) - 1
        with torch.autocast(device_type=members.device.type, enabled=False):
            sorted_members = members.float().gather(
                2, order[:, None, :].expand(-1, self.feedback_slots, -1))
            histogram = torch.zeros_like(members, dtype=torch.float32).scatter_add(
                2, groups[:, None, :].expand(-1, self.feedback_slots, -1),
                sorted_members)
            collisions = histogram.square().sum(dim=-1)
            diversity = counts.float().square() / collisions.clamp_min(1.0)
        return diversity if self.support_mode == 'diversity' else counts


    def _residual_rates(self, fractions, support, feedback, valid):
        """Support or fixed convex mixing with the same visible-user background.

        This background includes each slot; it is not an independent prior.
        Raw local fractions are retained in both product variants.
        """
        total = valid.sum(dim=-1).clamp_min(1)
        background = feedback.sum(dim=-1) / total[:, None]
        weight = support / (support + self.shrinkage_strength)
        if self.rate_interpolation == 'fixed':
            weight = weight * 0.0 + 0.5
        return (weight[:, :, None] * fractions
                + (1.0 - weight[:, :, None]) * background[:, None, :])


    def _event_covariance(self, keys, members, feedback, fractions, counts):
        """Within-slot E[(W x) y] - E[W x] E[y], using all-event denominators.

        This uses the SAME bias-free map as the source mean-key product.
        Project each event once, then accumulate small 8D moments in FP32.
        Center with the mean of those same projected events so constant
        content has zero covariance even under rounded event projection.
        Historical masks may overlap. No current labels or action are read.
        """
        batch, length, _ = keys.shape
        if not length:
            return keys.new_zeros((batch, self.feedback_slots,
                                   self.feedback_tasks, self.feedback_dim)).float()
        event_projection = self.feedback_projection(keys)
        with torch.autocast(device_type=keys.device.type, enabled=False):
            projected_events = event_projection.float()
            divisor = counts.clamp_min(1).unsqueeze(-1)
            mean_events = torch.bmm(members, projected_events) / divisor
            joint_members = (members[:, :, None, :] * feedback[:, None, :, :])
            joint = torch.bmm(joint_members.flatten(1, 2), projected_events)
            joint = joint.reshape(batch, self.feedback_slots,
                                  self.feedback_tasks, self.feedback_dim)
            joint = joint / divisor.unsqueeze(-1)
            covariance = (joint - mean_events[:, :, None, :]
                          * fractions[:, :, :, None])
            return covariance * (counts > 0)[:, :, None, None].to(covariance.dtype)


    def _center_moments(self, moments, projected, feedback, valid):
        """Subtract the visible-user background moment in FP32.

        Shapes: moments [B,8,4,D], projected [B,8,D], feedback [B,4,L].
        Feedback is already intersected with valid by the caller, and empty
        slots have zero projected content. In exact arithmetic the centered
        joint feature is alpha * (E[Wx*y | slot] - E[Wx | slot] * b_user).
        BF16 projection/mean rounding makes that identity approximate.
        Both modes compute the offset; none is the matched zero multiplier.
        No detach: the inherited projection learns through this transform.
        """
        with torch.autocast(device_type=projected.device.type, enabled=False):
            total = valid.sum(dim=-1).clamp_min(1)
            background = feedback.float().sum(dim=-1) / total[:, None]
            offset = (projected.float()[:, :, None, :]
                      * background[:, None, :, None])
            centering = 1.0 if self.centering_mode == 'user' else 0.0
            return moments.float() - centering * offset


    def _residual_features(self, clean_item, valid, task_masks, keys):
        batch, length = valid.shape
        members = self._feedback_membership(clean_item, valid).float()
        counts = members.sum(dim=-1)
        support = self._reliability_support(clean_item, valid, members, counts)
        divisor = counts.clamp_min(1).unsqueeze(-1)
        feedback = torch.stack([(m > 0) & valid for m in task_masks], dim=1).float()
        if length:
            mean_keys = torch.bmm(members.to(keys.dtype), keys) / divisor
            with torch.autocast(device_type=keys.device.type, enabled=False):
                fractions = torch.bmm(members, feedback.transpose(1, 2)) / divisor
        else:
            mean_keys = keys.new_zeros((batch, self.feedback_slots, self.key_width))
            fractions = members.new_zeros((batch, self.feedback_slots, self.feedback_tasks))
        occupied = (counts > 0).unsqueeze(-1)
        projected = self.feedback_projection(mean_keys)
        projected = projected * occupied.to(projected.dtype)
        gate = self._residual_rates(fractions, support, feedback, valid)
        if self.feedback_mode != 'product':
            # Active content parameters and unchanged readout width in controls.
            gate = torch.ones_like(gate)
        moments = projected[:, :, None, :] * gate[:, :, :, None].to(projected.dtype)
        # Factorized arms perform the same arithmetic, then suppress only
        # the covariance. Thus parameters, initialization and operation shapes
        # are shared. They also calibrate equality to the archived products.
        covariance = self._event_covariance(keys, members, feedback, fractions, counts)
        local_weight = support / (support + self.shrinkage_strength)
        if self.rate_interpolation == 'fixed':
            local_weight = local_weight * 0.0 + 0.5
        coupling = 1.0 if self.feedback_moment_mode == 'joint' else 0.0
        with torch.autocast(device_type=keys.device.type, enabled=False):
            moments = moments.float() + coupling * local_weight[:, :, None, None] * covariance
            moments = self._center_moments(moments, projected, feedback, valid).to(projected.dtype)
        # Compute the complete inherited product before the replacement.
        # Content replacement keeps the projection active without any local
        # feedback-dependent multiplier. Same shape, but effective input rank
        # and scale can differ: nominal capacity matching is not equal rank.
        if self.residual_content_source == 'content':
            moments = projected[:, :, None, :].expand(
                -1, -1, self.feedback_tasks, -1)
        count_features = counts.unsqueeze(-1) / 100.0
        if self.residual_rate_source == 'exposure':
            fractions = count_features.expand(-1, -1, self.feedback_tasks)
        if self.feedback_mode == 'content':
            # Additional task-feedback removed; repeat already-authorized counts
            # in the same four columns. Core action embeddings remain intact.
            fractions = count_features.expand(-1, -1, self.feedback_tasks)
        raw = torch.cat([moments.flatten(2), fractions.to(projected.dtype),
                         count_features.to(projected.dtype)], dim=-1)
        return raw.flatten(1)


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
        pair_tokens = self._pair_tokens(values, weights, relation_masks,
                                        relation_counts)
        slot_tokens = slot_tokens + pair_tokens.to(slot_tokens.dtype)
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
        tokens = self.feedback_memory(tokens, token_valid, values, weights,
                                      content_mask, valid, task_masks)
        tokens = self.interaction(tokens, token_valid)
        features = torch.cat([static.flatten(1), pooled[:, 0], tokens.flatten(1), log_counts], dim=-1)
        residual_features = self._residual_features(clean_item, valid, task_masks, keys)
        residual = self.feedback_residual(residual_features)
        # Empty prefixes receive exactly zero added logits in train/eval.
        residual = residual * valid.any(dim=-1, keepdim=True).to(residual.dtype)
        logits = torch.cat([tower(features) for tower in self.towers], dim=-1) + residual
        return {label + '_pred': torch.sigmoid(logits[:, index:index + 1])
                for index, label in enumerate(self.feature_map.labels)}
