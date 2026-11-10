"""Prospective user-background centering of shrunk joint feedback moments.

Factorial: centered versus absolute joint moments x count versus fixed mixing.
All arms compute the same background offset; only a scalar multiplier changes.
Raw fractions, counts, all registered modules and initialization stay matched.
The background is the same visible history, not an independent prior or a
causal counterfactual. This is an untested representation hypothesis.
"""
import math
import torch
from torch import nn
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbeddingDict
from unirank.pytorch.layers.activations import Dice


CONFIG = {'embedding_dim': 16, 'attention_hidden_units': [64, 32], 'tower_hidden_units': [128, 64], 'token_dim': 32, 'num_heads': 2, 'slot_mode': 'recency', 'representation_mode': 'replicas', 'content_mode': 'relation', 'count_mode': 'relation', 'feedback_dim': 8, 'residual_hidden': 64, 'recent_window': 16, 'feedback_mode': 'product', 'rate_interpolation': 'fixed', 'shrinkage_strength': 4.0, 'moment_mode': 'joint', 'centering_mode': 'user'}


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
    def __init__(self, feature_map, model_id='ResidualHistoryFeedback', gpu=-1,
                 embedding_dim=16, attention_hidden_units=(64, 32),
                 tower_hidden_units=(128, 64), token_dim=32, num_heads=2,
                 slot_mode='recency', representation_mode='replicas',
                 content_mode='relation', count_mode='relation',
                 feedback_dim=8, residual_hidden=64, recent_window=16,
                 feedback_mode=CONFIG['feedback_mode'],
                 rate_interpolation=CONFIG['rate_interpolation'],
                 shrinkage_strength=4.0, moment_mode=CONFIG['moment_mode'],
                 centering_mode=CONFIG['centering_mode'],
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
        if feedback_mode not in ('product', 'marginal', 'content'):
            raise ValueError('Unknown residual feature replacement')
        if rate_interpolation not in ('count', 'fixed'):
            raise ValueError('Unknown residual rate interpolation')
        if feedback_dim <= 0 or residual_hidden <= 0 or recent_window <= 0:
            raise ValueError('Residual dimensions and window must be positive')
        if not math.isfinite(shrinkage_strength) or shrinkage_strength <= 0:
            raise ValueError('Shrinkage strength must be finite and positive')
        if feedback_mode != 'product':
            raise ValueError('This factorial study requires product feedback')
        if moment_mode not in ('joint', 'factorized'):
            raise ValueError('Unknown event-feedback moment replacement')
        if centering_mode not in ('user', 'none'):
            raise ValueError('Unknown joint-moment centering')
        self.centering_mode = centering_mode
        self.moment_mode = moment_mode
        self.feedback_mode = feedback_mode
        self.rate_interpolation = rate_interpolation
        self.shrinkage_strength = float(shrinkage_strength)
        self.recent_window = recent_window
        self.feedback_tasks = 4
        self.feedback_slots = 8
        self.feedback_dim = feedback_dim
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
        # All four arms have 35 summary tokens and identical parameters.
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
        # Construct AFTER the complete source reset/identity initialization:
        # common modules keep the source's RNG consumption and initial state.
        self.feedback_projection = nn.Linear(self.key_width, feedback_dim, bias=False)
        residual_width = self.feedback_slots * (self.feedback_tasks * (feedback_dim + 1) + 1)
        self.feedback_residual = nn.Sequential(
            nn.Linear(residual_width, residual_hidden), nn.SiLU(),
            nn.Linear(residual_hidden, self.feedback_tasks))
        nn.init.zeros_(self.feedback_residual[-1].weight)
        nn.init.zeros_(self.feedback_residual[-1].bias)
        self.model_to_device()

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

    def _residual_rates(self, fractions, counts, feedback, valid):
        """Count or fixed convex mixing with the same visible-user background.

        This background includes each slot; it is not an independent prior.
        Raw local fractions are retained in both product variants.
        """
        total = valid.sum(dim=-1).clamp_min(1)
        background = feedback.sum(dim=-1) / total[:, None]
        weight = counts / (counts + self.shrinkage_strength)
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
        gate = self._residual_rates(fractions, counts, feedback, valid)
        if self.feedback_mode != 'product':
            # Active content parameters and unchanged readout width in controls.
            gate = torch.ones_like(gate)
        moments = projected[:, :, None, :] * gate[:, :, :, None].to(projected.dtype)
        # Factorized arms perform the same arithmetic, then suppress only
        # the covariance. Thus parameters, initialization and operation shapes
        # are shared. They also calibrate equality to the archived products.
        covariance = self._event_covariance(keys, members, feedback, fractions, counts)
        local_weight = counts / (counts + self.shrinkage_strength)
        if self.rate_interpolation == 'fixed':
            local_weight = local_weight * 0.0 + 0.5
        coupling = 1.0 if self.moment_mode == 'joint' else 0.0
        with torch.autocast(device_type=keys.device.type, enabled=False):
            moments = moments.float() + coupling * local_weight[:, :, None, None] * covariance
            moments = self._center_moments(moments, projected, feedback, valid).to(projected.dtype)
        count_features = counts.unsqueeze(-1) / 100.0
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
        tokens = self.interaction(tokens, token_valid)
        features = torch.cat([static.flatten(1), pooled[:, 0], tokens.flatten(1), log_counts], dim=-1)
        residual_features = self._residual_features(clean_item, valid, task_masks, keys)
        residual = self.feedback_residual(residual_features)
        # No residual bias for empty histories, in train or eval mode.
        residual = residual * valid.any(dim=-1, keepdim=True).to(residual.dtype)
        logits = torch.cat([tower(features) for tower in self.towers], dim=-1) + residual
        return {label + '_pred': torch.sigmoid(logits[:, index:index + 1])
                for index, label in enumerate(self.feature_map.labels)}
