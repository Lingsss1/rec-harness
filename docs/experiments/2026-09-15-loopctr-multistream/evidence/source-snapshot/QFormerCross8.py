# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# =========================================================================

import math

import torch
from torch import nn
import torch.nn.functional as F

from unirank.pytorch.layers import FeatureEmbedding, MLP_Block
from unirank.pytorch.models import MultiTaskModel


class FeedForward(nn.Module):
    """Two-layer feed-forward block shared by both QFormer stages."""

    def __init__(self, token_dim, ffn_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(token_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, token_dim)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, inputs):
        return self.dropout(self.fc2(F.gelu(self.fc1(inputs))))


class SDPAMultiHeadAttention(nn.Module):
    """Unmasked multi-head attention backed by PyTorch SDPA."""

    def __init__(self, token_dim, num_heads, qk_norm=False):
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.w_q = nn.Linear(token_dim, token_dim)
        self.w_k = nn.Linear(token_dim, token_dim)
        self.w_v = nn.Linear(token_dim, token_dim)
        self.w_o = nn.Linear(token_dim, token_dim)
        self.q_norm = (
            nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        )
        self.k_norm = (
            nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        )

    def forward(self, query, key, value):
        batch_size, query_length, _ = query.shape
        key_length = key.size(1)
        q = self.w_q(query).view(
            batch_size, query_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.w_k(key).view(
            batch_size, key_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.w_v(value).view(
            batch_size, key_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale,
        )
        output = output.transpose(1, 2).contiguous().view(
            batch_size, query_length, -1
        )
        return self.w_o(output)


class MaskedSDPAMultiHeadAttention(nn.Module):
    """Multi-head SDPA with an optional key/value padding mask."""

    def __init__(self, token_dim, num_heads, qk_norm=False):
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.w_q = nn.Linear(token_dim, token_dim)
        self.w_k = nn.Linear(token_dim, token_dim)
        self.w_v = nn.Linear(token_dim, token_dim)
        self.w_o = nn.Linear(token_dim, token_dim)
        self.q_norm = (
            nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        )
        self.k_norm = (
            nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        )

    def forward(self, query, key, value, mask=None):
        batch_size, query_length, _ = query.shape
        key_length = key.size(1)
        q = self.w_q(query).view(
            batch_size, query_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.w_k(key).view(
            batch_size, key_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.w_v(value).view(
            batch_size, key_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)

        key_mask = None
        has_valid_key = None
        if mask is not None:
            valid = mask.to(dtype=torch.bool, device=q.device)
            key_mask = valid.view(batch_size, 1, 1, key_length)
            has_valid_key = valid.any(dim=1).view(batch_size, 1, 1)

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=key_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale,
        )
        output = output.transpose(1, 2).contiguous().view(
            batch_size, query_length, self.token_dim
        )
        output = self.w_o(output)
        if has_valid_key is not None:
            output = output * has_valid_key.to(dtype=output.dtype)
        return output


class SDPACrossValueAttention(nn.Module):
    """Head-local SDPA followed by an explicit query-feature product."""

    def __init__(self, token_dim, num_heads, qk_norm=False,
                 attention_temperature=1.0):
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        if attention_temperature <= 0:
            raise ValueError("attention_temperature must be positive")
        self.attention_temperature = float(attention_temperature)
        self.scale = (
            1.0 / math.sqrt(self.head_dim) / self.attention_temperature
        )
        self.w_q = nn.Linear(token_dim, token_dim)
        self.w_k = nn.Linear(token_dim, token_dim)
        self.w_qi = nn.Linear(token_dim, token_dim)
        self.w_fi = nn.Linear(token_dim, token_dim)
        self.w_v_pair = nn.Linear(token_dim, token_dim)
        self.w_o = nn.Linear(token_dim, token_dim)
        self.q_norm = (
            nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        )
        self.k_norm = (
            nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        )

    def forward(self, queries, keys, values, mask=None):
        batch_size, num_queries, _ = queries.shape
        sequence_length = keys.size(1)
        q = self.w_q(queries).view(
            batch_size, num_queries, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.w_k(keys).view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.w_fi(values).view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)

        key_mask = None
        if mask is not None:
            key_mask = mask.to(dtype=torch.bool, device=q.device).view(
                batch_size, 1, 1, sequence_length
            )
        attended_features = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=key_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale,
        )

        q_interaction = self.w_qi(queries).view(
            batch_size, num_queries, self.num_heads, self.head_dim
        ).transpose(1, 2)
        pair_summary = q_interaction * attended_features
        pair_summary = pair_summary.transpose(1, 2).contiguous().view(
            batch_size, num_queries, self.token_dim
        )
        return self.w_o(self.w_v_pair(pair_summary))


class RecursiveCrossValueLayer(nn.Module):
    """One exact residual step: Q_(l+1) = Q_l + CrossValue(Q_l, X)."""

    def __init__(self, token_dim, num_heads, dropout=0.0, qk_norm=False,
                 residual_scale=1.0, attention_temperature=1.0,
                 learned_scale_shape=None, learned_scale_bound=0.0):
        super().__init__()
        self.cross_attn = SDPACrossValueAttention(
            token_dim, num_heads, qk_norm, attention_temperature
        )
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = float(residual_scale)
        self.learned_scale_bound = float(learned_scale_bound)
        self.learned_scale_logits = (
            nn.Parameter(torch.zeros(learned_scale_shape))
            if learned_scale_shape is not None else None
        )

    def forward(self, queries, features, mask=None):
        cross_output = self.cross_attn(queries, features, features, mask)
        scale = self.residual_scale
        if self.learned_scale_logits is not None:
            scale = scale * (
                1.0 + self.learned_scale_bound
                * torch.tanh(self.learned_scale_logits)
            )
        return queries + scale * self.dropout(cross_output)


class NormalizedRecursiveCrossValueLayer(RecursiveCrossValueLayer):
    """Sequence recurrence with a bounded normalization placement."""

    MODES = {
        "pre_layernorm",
        "post_layernorm",
        "post_rmsnorm",
        "pre_layernorm_post_rmsnorm",
    }

    def __init__(self, token_dim, num_heads, dropout=0.0, qk_norm=False,
                 residual_scale=1.0, update_mode="post_rmsnorm",
                 attention_temperature=1.0, learned_scale_shape=None,
                 learned_scale_bound=0.0):
        if update_mode not in self.MODES:
            raise ValueError(f"unsupported sequence_cross_update_mode: {update_mode}")
        super().__init__(
            token_dim, num_heads, dropout, qk_norm, residual_scale,
            attention_temperature, learned_scale_shape, learned_scale_bound
        )
        self.update_mode = update_mode
        self.pre_norm = (
            nn.LayerNorm(token_dim)
            if update_mode.startswith("pre_layernorm") else nn.Identity()
        )
        if update_mode.endswith("post_rmsnorm"):
            self.post_norm = nn.RMSNorm(token_dim)
        elif update_mode == "post_layernorm":
            self.post_norm = nn.LayerNorm(token_dim)
        else:
            self.post_norm = nn.Identity()

    def forward(self, queries, features, mask=None):
        cross_output = self.cross_attn(
            self.pre_norm(queries), features, features, mask
        )
        updated = queries + self.residual_scale * self.dropout(cross_output)
        return self.post_norm(updated)


class RecursiveCrossValueStage(nn.Module):
    """Read fields once, recursively cross them, then mix the queries."""

    def __init__(self, token_dim, num_heads, num_layers, num_queries, ffn_dim,
                 dropout=0.0, qk_norm=False, residual_scale=1.0,
                 learned_scale_mode="none", learned_scale_bound=0.0):
        super().__init__()
        scale_shapes = {
            "none": None,
            "layer_scalar": (1, 1, 1),
            "layer_channel": (1, 1, token_dim),
            "layer_query": (1, num_queries, 1),
            "layer_query_channel": (1, num_queries, token_dim),
        }
        if learned_scale_mode not in scale_shapes:
            raise ValueError("unsupported non-sequential learned scale mode")
        if learned_scale_bound < 0 or learned_scale_bound > 0.5:
            raise ValueError(
                "non-sequential learned scale bound must be in [0, 0.5]"
            )
        self.token_dim = token_dim
        self.num_queries = num_queries
        self.learnable_queries = nn.Parameter(
            torch.empty(num_queries, token_dim)
        )
        self.initial_read = MaskedSDPAMultiHeadAttention(
            token_dim, num_heads, qk_norm
        )
        self.initial_norm = nn.LayerNorm(token_dim)
        self.layers = nn.ModuleList([
            RecursiveCrossValueLayer(
                token_dim=token_dim,
                num_heads=num_heads,
                dropout=dropout,
                qk_norm=qk_norm,
                residual_scale=residual_scale,
                learned_scale_shape=scale_shapes[learned_scale_mode],
                learned_scale_bound=learned_scale_bound,
            )
            for _ in range(num_layers)
        ])
        self.post_self_attn = SDPAMultiHeadAttention(
            token_dim, num_heads, qk_norm
        )
        self.post_self_norm = nn.LayerNorm(token_dim)
        self.post_ffn = FeedForward(token_dim, ffn_dim, dropout)
        self.post_ffn_norm = nn.LayerNorm(token_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_normal_(self.learnable_queries)

    def forward(self, features, mask=None):
        batch_size = features.size(0)
        queries = self.learnable_queries.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        initial_delta = self.initial_read(
            queries, features, features, mask
        )
        queries = self.initial_norm(
            queries + self.dropout(initial_delta)
        )
        for layer in self.layers:
            queries = layer(queries, features, mask)

        mixed = self.post_self_attn(queries, queries, queries)
        queries = self.post_self_norm(queries + self.dropout(mixed))
        ffn_output = self.post_ffn(queries)
        return self.post_ffn_norm(
            queries + self.dropout(ffn_output)
        )


class RecursiveSequenceCrossValueStage(nn.Module):
    """Apply the same residual CrossValue recurrence to sequence tokens."""

    def __init__(self, token_dim, num_heads, num_layers, ffn_dim,
                 dropout=0.0, qk_norm=False, residual_scale=1.0,
                 update_mode="residual", attention_temperature=1.0,
                 num_queries=None, learned_scale_mode="none",
                 learned_scale_bound=0.0, post_ffn_extra_layers=0):
        super().__init__()
        scale_shapes = {
            "none": None,
            "layer_scalar": (1, 1, 1),
            "layer_channel": (1, 1, token_dim),
            "layer_query": (1, num_queries, 1) if num_queries else None,
            "layer_query_channel": (
                (1, num_queries, token_dim) if num_queries else None
            ),
        }
        if learned_scale_mode not in scale_shapes:
            raise ValueError("unsupported sequence learned scale mode")
        if learned_scale_mode.startswith("layer_query") and num_queries is None:
            raise ValueError("query-local learned scales require num_queries")
        if learned_scale_bound < 0 or learned_scale_bound > 0.5:
            raise ValueError("sequence learned scale bound must be in [0, 0.5]")
        layer_cls = (
            RecursiveCrossValueLayer
            if update_mode == "residual"
            else NormalizedRecursiveCrossValueLayer
        )
        self.layers = nn.ModuleList([
            layer_cls(
                token_dim=token_dim,
                num_heads=num_heads,
                dropout=dropout,
                qk_norm=qk_norm,
                residual_scale=residual_scale,
                attention_temperature=attention_temperature,
                learned_scale_shape=scale_shapes[learned_scale_mode],
                learned_scale_bound=learned_scale_bound,
                **({} if update_mode == "residual" else {
                    "update_mode": update_mode
                }),
            )
            for _ in range(num_layers)
        ])
        self.post_self_attn = SDPAMultiHeadAttention(
            token_dim, num_heads, qk_norm
        )
        self.post_self_norm = nn.LayerNorm(token_dim)
        self.post_ffn = FeedForward(token_dim, ffn_dim, dropout)
        self.post_ffn_norm = nn.LayerNorm(token_dim)
        self.extra_post_ffns = nn.ModuleList([
            FeedForward(token_dim, ffn_dim, dropout)
            for _ in range(post_ffn_extra_layers)
        ])
        self.extra_post_ffn_norms = nn.ModuleList([
            nn.LayerNorm(token_dim) for _ in range(post_ffn_extra_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward_components(self, queries, sequence_features,
                           sequence_mask=None):
        for layer in self.layers:
            queries = layer(queries, sequence_features, sequence_mask)

        mixed = self.post_self_attn(queries, queries, queries)
        queries = self.post_self_norm(queries + self.dropout(mixed))
        ffn_output = self.post_ffn(queries)
        return queries, self.dropout(ffn_output)

    def combine_post_ffn(self, queries, ffn_output, scale=1.0):
        output = self.post_ffn_norm(
            queries + float(scale) * ffn_output
        )
        for extra_ffn, extra_norm in zip(
                self.extra_post_ffns, self.extra_post_ffn_norms):
            output = extra_norm(output + self.dropout(extra_ffn(output)))
        return output

    def forward(self, queries, sequence_features, sequence_mask=None):
        queries, ffn_output = self.forward_components(
            queries, sequence_features, sequence_mask
        )
        return self.combine_post_ffn(queries, ffn_output)


class QFormerCross8(MultiTaskModel):
    """Symmetric CrossValue recurrences for fields and behavior sequences.

    The first stage initializes latent queries from user/context and target
    item fields, then recursively crosses the queries with the unchanged field
    tokens. The second stage applies the same recurrence to the unchanged
    sequence-token stream::

        Q_ns_(l+1)  = Q_ns_l  + CrossValue(Q_ns_l, X)
        Q_seq_(l+1) = Q_seq_l + CrossValue(Q_seq_l, S)

    Each stage mixes queries and applies an FFN only after its complete cross
    stack. This file contains the complete model implementation and does not
    depend on another QFormerCross variant.
    """

    def __init__(self,
                 feature_map,
                 model_id="QFormerCross8",
                 task=["binary_classification"],
                 gpu=-1,
                 tower_activations="ReLU",
                 tower_hidden_units=[256, 128],
                 embedding_dim=16,
                 token_dim=256,
                 num_heads=4,
                 num_queries=8,
                 num_ns_layers=2,
                 num_unified_layers=3,
                 ffn_ratio=4.0,
                 unified_ffn_ratio=None,
                 unified_post_ffn_extra_layers=0,
                 task_unified_post_ffn_adapter_rank=0,
                 qk_norm=True,
                 max_len=100,
                 num_tasks=4,
                 net_dropout=0,
                 accumulation_steps=1,
                 _ns_qformer_cls=None,
                 _unified_qformer_cls=None,
                 tower_output_bias_init=None,
                 cross_residual_scale=1.0,
                 ns_cross_residual_scale=None,
                 ns_cross_learned_scale_mode="none",
                 ns_cross_learned_scale_bound=0.0,
                 sequence_cross_residual_scale=None,
                 sequence_cross_layer_residual_scales=None,
                 sequence_cross_learned_scale_mode="none",
                 sequence_cross_learned_scale_bound=0.0,
                 unified_cross_attention_temperature=1.0,
                 sequence_cross_update_mode="residual",
                 unified_layer_gradient_scales=None,
                 unified_final_layer_component_gradient_scales=None,
                 position_embedding_scale=1.0,
                 query_dropout=0.0,
                 query_fusion_learned_scale_mode="none",
                 query_fusion_learned_scale_bound=0.0,
                 query_fusion_fixed_scales=None,
                 query_fusion_training_scales=None,
                 query_fusion_zero_recovery_mode="none",
                 query_pooling="flatten",
                 target_pool_residual_scale=0.1,
                 ns_field_rmsnorm=False,
                 ns_source_balance=False,
                 ns_token_dropout=0.0,
                 ns_energy_gate_strength=0.0,
                 input_projection_learned_scale_mode="none",
                 input_projection_learned_scale_bound=0.0,
                 task_residual_gate_strength=0.0,
                 task_unified_post_ffn_scales=None,
                 task_unified_post_ffn_eval_scales=None,
                 **kwargs):
        super().__init__(
            feature_map,
            model_id=model_id,
            task=task,
            num_tasks=num_tasks,
            gpu=gpu,
            **kwargs,
        )
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        if num_queries < 1:
            raise ValueError("num_queries must be positive")
        if num_ns_layers < 1 or num_unified_layers < 1:
            raise ValueError("both QFormer stages require at least one layer")
        if ffn_ratio <= 0:
            raise ValueError("ffn_ratio must be positive")
        if unified_ffn_ratio is None:
            unified_ffn_ratio = ffn_ratio
        if unified_ffn_ratio <= 0:
            raise ValueError("unified_ffn_ratio must be positive")
        if unified_post_ffn_extra_layers not in {0, 1}:
            raise ValueError("unified_post_ffn_extra_layers must be 0 or 1")
        if task_unified_post_ffn_adapter_rank < 0:
            raise ValueError(
                "task_unified_post_ffn_adapter_rank must be nonnegative"
            )
        if max_len < 1:
            raise ValueError("max_len must be positive")
        if cross_residual_scale <= 0:
            raise ValueError("cross_residual_scale must be positive")
        ns_cross_residual_scale = (
            cross_residual_scale if ns_cross_residual_scale is None
            else ns_cross_residual_scale
        )
        sequence_cross_residual_scale = (
            cross_residual_scale if sequence_cross_residual_scale is None
            else sequence_cross_residual_scale
        )
        if ns_cross_residual_scale <= 0:
            raise ValueError("ns_cross_residual_scale must be positive")
        if sequence_cross_residual_scale <= 0:
            raise ValueError("sequence_cross_residual_scale must be positive")
        if sequence_cross_layer_residual_scales is not None:
            if len(sequence_cross_layer_residual_scales) != num_unified_layers:
                raise ValueError(
                    "sequence_cross_layer_residual_scales must contain exactly "
                    f"{num_unified_layers} values"
                )
            if any(float(value) <= 0
                   for value in sequence_cross_layer_residual_scales):
                raise ValueError(
                    "sequence_cross_layer_residual_scales values must be positive"
                )
        if unified_cross_attention_temperature <= 0:
            raise ValueError(
                "unified_cross_attention_temperature must be positive"
            )
        valid_learned_scale_modes = {
            "none", "layer_scalar", "layer_channel", "layer_query",
            "layer_query_channel",
        }
        if ns_cross_learned_scale_mode not in valid_learned_scale_modes:
            raise ValueError("unsupported ns_cross_learned_scale_mode")
        if not 0 <= ns_cross_learned_scale_bound <= 0.5:
            raise ValueError(
                "ns_cross_learned_scale_bound must be in [0, 0.5]"
            )
        if (ns_cross_learned_scale_mode == "none") != (
            ns_cross_learned_scale_bound == 0
        ):
            raise ValueError(
                "learned non-sequential scaling requires both a mode and positive bound"
            )
        if sequence_cross_learned_scale_mode not in valid_learned_scale_modes:
            raise ValueError("unsupported sequence_cross_learned_scale_mode")
        if not 0 <= sequence_cross_learned_scale_bound <= 0.5:
            raise ValueError(
                "sequence_cross_learned_scale_bound must be in [0, 0.5]"
            )
        if (sequence_cross_learned_scale_mode == "none") != (
            sequence_cross_learned_scale_bound == 0
        ):
            raise ValueError(
                "learned sequence scaling requires both a mode and positive bound"
            )
        valid_sequence_update_modes = {
            "residual", *NormalizedRecursiveCrossValueLayer.MODES
        }
        if sequence_cross_update_mode not in valid_sequence_update_modes:
            raise ValueError(
                "unsupported sequence_cross_update_mode: "
                f"{sequence_cross_update_mode}"
            )
        if unified_layer_gradient_scales is None:
            unified_layer_gradient_scales = [1.0] * num_unified_layers
        if len(unified_layer_gradient_scales) != num_unified_layers:
            raise ValueError(
                "unified_layer_gradient_scales must contain exactly "
                f"{num_unified_layers} values"
            )
        if any(float(value) <= 0 for value in unified_layer_gradient_scales):
            raise ValueError(
                "unified_layer_gradient_scales values must be positive"
            )
        valid_unified_components = {
            "routing_qk", "interaction", "output", "all_cross_attention"
        }
        unified_final_layer_component_gradient_scales = dict(
            unified_final_layer_component_gradient_scales or {}
        )
        unknown_components = (
            set(unified_final_layer_component_gradient_scales)
            - valid_unified_components
        )
        if unknown_components:
            raise ValueError(
                "unsupported unified component gradient scales: "
                f"{sorted(unknown_components)}"
            )
        if any(
            float(value) <= 0
            for value in unified_final_layer_component_gradient_scales.values()
        ):
            raise ValueError(
                "unified_final_layer_component_gradient_scales values must be positive"
            )
        if position_embedding_scale < 0:
            raise ValueError("position_embedding_scale must be nonnegative")
        if not 0 <= query_dropout < 1:
            raise ValueError("query_dropout must be in [0, 1)")
        query_fusion_scale_shapes = {
            "none": None,
            "shared_query": (1, num_queries, 1),
            "task_query": (num_tasks, num_queries, 1),
            "shared_query_channel": (1, num_queries, token_dim),
            "task_query_channel": (num_tasks, num_queries, token_dim),
        }
        if query_fusion_learned_scale_mode not in query_fusion_scale_shapes:
            raise ValueError("unsupported query fusion learned scale mode")
        if not 0 <= query_fusion_learned_scale_bound <= 0.5:
            raise ValueError("query fusion learned scale bound must be in [0, 0.5]")
        if (query_fusion_learned_scale_mode == "none") != (
            query_fusion_learned_scale_bound == 0
        ):
            raise ValueError(
                "learned query fusion scaling requires both a mode and positive bound"
            )
        if query_fusion_fixed_scales is None:
            query_fusion_fixed_scales = [1.0] * num_queries
        if len(query_fusion_fixed_scales) != num_queries:
            raise ValueError(
                "query_fusion_fixed_scales must contain exactly num_queries values"
            )
        if any(not 0 <= float(value) <= 1 for value in query_fusion_fixed_scales):
            raise ValueError("query_fusion_fixed_scales values must be in [0, 1]")
        if query_fusion_training_scales is None:
            query_fusion_training_scales = query_fusion_fixed_scales
        if len(query_fusion_training_scales) != num_queries:
            raise ValueError(
                "query_fusion_training_scales must contain exactly num_queries values"
            )
        if any(not 0 <= float(value) <= 1 for value in query_fusion_training_scales):
            raise ValueError("query_fusion_training_scales values must be in [0, 1]")
        if query_fusion_zero_recovery_mode not in {"none", "shared", "task"}:
            raise ValueError("unsupported query_fusion_zero_recovery_mode")
        zero_indices = [
            index for index, value in enumerate(query_fusion_fixed_scales)
            if float(value) == 0
        ]
        if query_fusion_zero_recovery_mode != "none" and not zero_indices:
            raise ValueError(
                "query fusion zero recovery requires at least one fixed-zero query"
            )
        if query_pooling not in {
            "flatten", "target_attention", "target_residual"
        }:
            raise ValueError("unsupported query_pooling")
        if target_pool_residual_scale < 0:
            raise ValueError("target_pool_residual_scale must be nonnegative")
        if not 0 <= ns_token_dropout < 1:
            raise ValueError("ns_token_dropout must be in [0, 1)")
        if not 0 <= ns_energy_gate_strength <= 1:
            raise ValueError("ns_energy_gate_strength must be in [0, 1]")
        valid_input_projection_scale_modes = {
            "none", "global", "field", "channel", "field_channel",
        }
        if input_projection_learned_scale_mode not in (
            valid_input_projection_scale_modes
        ):
            raise ValueError("unsupported input projection learned scale mode")
        if not 0 <= input_projection_learned_scale_bound <= 0.5:
            raise ValueError(
                "input projection learned scale bound must be in [0, 0.5]"
            )
        if (input_projection_learned_scale_mode == "none") != (
            input_projection_learned_scale_bound == 0
        ):
            raise ValueError(
                "learned input projection scaling requires both a mode and "
                "positive bound"
            )
        if not 0 <= task_residual_gate_strength <= 1:
            raise ValueError("task_residual_gate_strength must be in [0, 1]")

        self.feature_map = feature_map
        self.num_tasks = num_tasks
        self.embedding_dim = embedding_dim
        self.token_dim = token_dim
        self.num_queries = num_queries
        self.max_len = max_len
        self.accumulation_steps = accumulation_steps
        self.position_embedding_scale = float(position_embedding_scale)
        self.unified_cross_attention_temperature = float(
            unified_cross_attention_temperature
        )
        self.unified_layer_gradient_scales = tuple(
            float(value) for value in unified_layer_gradient_scales
        )
        self.unified_final_layer_component_gradient_scales = {
            key: float(value)
            for key, value in unified_final_layer_component_gradient_scales.items()
        }
        self.query_pooling = query_pooling
        self.target_pool_residual_scale = float(target_pool_residual_scale)
        self.query_dropout = nn.Dropout(float(query_dropout))
        self.query_fusion_learned_scale_mode = query_fusion_learned_scale_mode
        self.query_fusion_learned_scale_bound = float(
            query_fusion_learned_scale_bound
        )
        query_fusion_shape = query_fusion_scale_shapes[
            query_fusion_learned_scale_mode
        ]
        self.query_fusion_scale_logits = (
            nn.Parameter(torch.zeros(query_fusion_shape))
            if query_fusion_shape is not None else None
        )
        self.register_buffer(
            "query_fusion_fixed_scales",
            torch.tensor(query_fusion_fixed_scales, dtype=torch.float32),
            # Persist inference-time query routing so a promoted artifact is a
            # self-contained deployable checkpoint rather than a checkpoint
            # whose behavior depends on an out-of-band config override.
            persistent=True,
        )
        self.register_buffer(
            "query_fusion_training_scales",
            torch.tensor(query_fusion_training_scales, dtype=torch.float32),
            persistent=False,
        )
        self.query_fusion_zero_recovery_mode = query_fusion_zero_recovery_mode
        recovery_tasks = num_tasks if query_fusion_zero_recovery_mode == "task" else 1
        self.query_fusion_zero_recovery_logits = (
            nn.Parameter(torch.zeros(recovery_tasks, len(zero_indices), 1))
            if query_fusion_zero_recovery_mode != "none" else None
        )
        self.query_fusion_zero_indices = tuple(zero_indices)
        self.ns_field_rmsnorm = bool(ns_field_rmsnorm)
        self.ns_source_balance = bool(ns_source_balance)
        self.ns_token_dropout = nn.Dropout(float(ns_token_dropout))
        self.ns_energy_gate_strength = float(ns_energy_gate_strength)
        self.input_projection_learned_scale_mode = (
            input_projection_learned_scale_mode
        )
        self.input_projection_learned_scale_bound = float(
            input_projection_learned_scale_bound
        )
        self.task_residual_gate_strength = float(
            task_residual_gate_strength
        )
        if task_unified_post_ffn_scales is not None:
            if len(task_unified_post_ffn_scales) != num_tasks:
                raise ValueError(
                    "task_unified_post_ffn_scales must contain exactly "
                    f"{num_tasks} values"
                )
            if any(float(value) < 0 for value in task_unified_post_ffn_scales):
                raise ValueError("task_unified_post_ffn_scales must be nonnegative")
            self.task_unified_post_ffn_scales = [
                float(value) for value in task_unified_post_ffn_scales
            ]
        else:
            self.task_unified_post_ffn_scales = None
        if task_unified_post_ffn_eval_scales is not None:
            if len(task_unified_post_ffn_eval_scales) != num_tasks:
                raise ValueError(
                    "task_unified_post_ffn_eval_scales must contain exactly "
                    f"{num_tasks} values"
                )
            if any(float(value) < 0 for value in task_unified_post_ffn_eval_scales):
                raise ValueError(
                    "task_unified_post_ffn_eval_scales must be nonnegative"
                )
            eval_scales = [
                float(value) for value in task_unified_post_ffn_eval_scales
            ]
        else:
            eval_scales = [1.0] * num_tasks
        # Persistent so the exact validation-time calibration is preserved in
        # the winning checkpoint without adding an optimizer parameter or
        # changing the training trajectory.
        self.register_buffer(
            "task_unified_post_ffn_eval_scales",
            torch.tensor(eval_scales, dtype=torch.float32),
        )

        self.target_item_features = []
        self.sequence_features = []
        self.context_features = []
        for feature, spec in feature_map.features.items():
            if feature in feature_map.labels or spec.get("type") == "meta":
                continue
            if spec.get("source") == "item":
                self.target_item_features.append(feature)
                self.sequence_features.append(feature)
            elif spec.get("source") == "action":
                # Target actions are zero placeholders and only belong to the
                # history stream.
                self.sequence_features.append(feature)
            else:
                self.context_features.append(feature)
        if not self.target_item_features:
            raise ValueError("QFormerCross8 requires target item features")
        if not self.sequence_features:
            raise ValueError(
                "QFormerCross8 requires item/action sequence features"
            )
        if not self.context_features:
            raise ValueError("QFormerCross8 requires user/context features")

        self.ns_features = self.context_features + self.target_item_features
        input_projection_scale_shapes = {
            "none": None,
            "global": (1, 1, 1),
            "field": (1, len(self.ns_features), 1),
            "channel": (1, 1, token_dim),
            "field_channel": (1, len(self.ns_features), token_dim),
        }
        input_projection_scale_shape = input_projection_scale_shapes[
            input_projection_learned_scale_mode
        ]
        self.input_projection_scale_logits = (
            nn.Parameter(torch.zeros(input_projection_scale_shape))
            if input_projection_scale_shape is not None else None
        )
        self.item_info_dim = sum(
            feature_map.features[feature].get(
                "embedding_dim", embedding_dim
            )
            for feature in self.sequence_features
        )

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.field_projection = nn.ModuleDict({
            feature: nn.Linear(
                feature_map.features[feature].get(
                    "embedding_dim", embedding_dim
                ),
                token_dim,
                bias=False,
            )
            for feature in self.ns_features
        })
        self.ns_field_embedding = nn.Parameter(torch.empty(
            len(self.ns_features), token_dim
        ))
        self.ns_input_norm = nn.LayerNorm(token_dim)

        self.sequence_projection = nn.Linear(self.item_info_dim, token_dim)
        # A Parameter keeps this small position table in the dense optimizer.
        self.position_embedding = nn.Parameter(torch.empty(max_len, token_dim))
        self.sequence_input_norm = nn.LayerNorm(token_dim)

        ffn_dim = int(token_dim * ffn_ratio)
        unified_ffn_dim = int(token_dim * unified_ffn_ratio)
        ns_qformer_cls = _ns_qformer_cls or RecursiveCrossValueStage
        self.ns_qformer = ns_qformer_cls(
            token_dim=token_dim,
            num_heads=num_heads,
            num_layers=num_ns_layers,
            num_queries=num_queries,
            ffn_dim=ffn_dim,
            dropout=net_dropout,
            qk_norm=qk_norm,
            residual_scale=ns_cross_residual_scale,
            learned_scale_mode=ns_cross_learned_scale_mode,
            learned_scale_bound=ns_cross_learned_scale_bound,
        )
        unified_qformer_cls = (
            _unified_qformer_cls or RecursiveSequenceCrossValueStage
        )
        unified_qformer_kwargs = dict(
            token_dim=token_dim,
            num_heads=num_heads,
            num_layers=num_unified_layers,
            ffn_dim=unified_ffn_dim,
            dropout=net_dropout,
            qk_norm=qk_norm,
            residual_scale=sequence_cross_residual_scale,
            update_mode=sequence_cross_update_mode,
            attention_temperature=unified_cross_attention_temperature,
            num_queries=num_queries,
            learned_scale_mode=sequence_cross_learned_scale_mode,
            learned_scale_bound=sequence_cross_learned_scale_bound,
        )
        if unified_qformer_cls is RecursiveSequenceCrossValueStage:
            unified_qformer_kwargs["post_ffn_extra_layers"] = (
                unified_post_ffn_extra_layers
            )
        elif unified_post_ffn_extra_layers:
            raise ValueError(
                "unified_post_ffn_extra_layers requires the default unified stage"
            )
        self.unified_qformer = unified_qformer_cls(**unified_qformer_kwargs)
        if sequence_cross_layer_residual_scales is not None:
            for layer, scale in zip(
                    self.unified_qformer.layers,
                    sequence_cross_layer_residual_scales):
                layer.residual_scale = float(scale)
        # These hooks alter only optimizer-facing gradients. Unit scales are
        # exactly incumbent behavior and forward activations stay unchanged.
        self._unified_gradient_scale_handles = []
        for layer, scale in zip(
                self.unified_qformer.layers,
                self.unified_layer_gradient_scales):
            if scale == 1.0:
                continue
            for parameter in layer.parameters():
                self._unified_gradient_scale_handles.append(
                    parameter.register_hook(
                        lambda gradient, multiplier=scale:
                        gradient * multiplier
                    )
                )
        component_modules = {
            "routing_qk": ("w_q", "w_k"),
            "interaction": ("w_qi", "w_fi", "w_v_pair"),
            "output": ("w_o",),
            "all_cross_attention": (
                "w_q", "w_k", "w_qi", "w_fi", "w_v_pair", "w_o"
            ),
        }
        for component, scale in (
            self.unified_final_layer_component_gradient_scales.items()
        ):
            if scale == 1.0:
                continue
            cross_attn = self.unified_qformer.layers[-1].cross_attn
            for module_name in component_modules[component]:
                for parameter in getattr(cross_attn, module_name).parameters():
                    self._unified_gradient_scale_handles.append(
                        parameter.register_hook(
                            lambda gradient, multiplier=scale:
                            gradient * multiplier
                        )
                    )

        self.tower_input_dim = num_queries * token_dim
        self.task_unified_post_ffn_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(token_dim, task_unified_post_ffn_adapter_rank,
                          bias=False),
                nn.GELU(),
                nn.Linear(task_unified_post_ffn_adapter_rank, token_dim,
                          bias=False),
            )
            for _ in range(num_tasks)
        ]) if task_unified_post_ffn_adapter_rank else nn.ModuleList()
        self.tower = nn.ModuleList([
            MLP_Block(
                input_dim=self.tower_input_dim,
                output_dim=1,
                hidden_units=tower_hidden_units,
                hidden_activations=tower_activations,
                output_activation=None,
                dropout_rates=net_dropout,
            )
            for _ in range(num_tasks)
        ])
        self.task_residual_gate = nn.ModuleList([
            nn.Linear(self.tower_input_dim, self.tower_input_dim)
            for _ in range(num_tasks)
        ])
        if isinstance(task, list):
            if len(task) != num_tasks:
                raise ValueError(
                    "the number of tasks must equal the length of task"
                )
            self.output_activation = nn.ModuleList([
                self.get_output_activation(str(task_type))
                for task_type in task
            ])
        else:
            self.output_activation = nn.ModuleList([
                self.get_output_activation(task)
                for _ in range(num_tasks)
            ])

        self.compile(
            kwargs.get("dense_optimizer"),
            kwargs["loss"],
            kwargs.get("dense_learning_rate"),
        )
        self.reset_parameters()
        nn.init.normal_(self.ns_field_embedding, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)
        for gate in self.task_residual_gate:
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)
        for adapter in self.task_unified_post_ffn_adapters:
            nn.init.zeros_(adapter[-1].weight)
        if tower_output_bias_init is not None:
            for tower in self.tower:
                output_layer = next(
                    module for module in reversed(tower.mlp)
                    if isinstance(module, nn.Linear)
                )
                nn.init.constant_(output_layer.bias, float(tower_output_bias_init))
        self.model_to_device()

    def _build_non_sequence_tokens(self, context_embedding_dict,
                                   item_embedding_dict):
        tokens = []
        for feature in self.context_features:
            embedding = context_embedding_dict[feature]
            if embedding.dim() != 2:
                raise ValueError(
                    f"context feature {feature} must produce B x D embeddings, "
                    f"got shape={tuple(embedding.shape)}"
                )
            tokens.append(self.field_projection[feature](embedding))

        for feature in self.target_item_features:
            embedding = item_embedding_dict[feature]
            if embedding.dim() != 3:
                raise ValueError(
                    f"item feature {feature} must produce B x (S+1) x D "
                    f"embeddings, got shape={tuple(embedding.shape)}"
                )
            tokens.append(self.field_projection[feature](embedding[:, -1, :]))

        ns_tokens = torch.stack(tokens, dim=1)
        if self.input_projection_scale_logits is not None:
            ns_tokens = ns_tokens * (
                1.0 + self.input_projection_learned_scale_bound
                * torch.tanh(self.input_projection_scale_logits)
            )
        if self.ns_field_rmsnorm:
            ns_tokens = F.rms_norm(ns_tokens, (ns_tokens.size(-1),))
        if self.ns_energy_gate_strength:
            rms = ns_tokens.float().square().mean(dim=-1, keepdim=True).add(
                1e-6
            ).rsqrt().to(dtype=ns_tokens.dtype)
            strength = self.ns_energy_gate_strength
            ns_tokens = ns_tokens * ((1.0 - strength) + strength * rms)
        if self.ns_source_balance:
            context_count = len(self.context_features)
            item_count = len(self.target_item_features)
            total_count = context_count + item_count
            source_weights = ns_tokens.new_empty(total_count)
            source_weights[:context_count] = total_count / (2 * context_count)
            source_weights[context_count:] = total_count / (2 * item_count)
            ns_tokens = ns_tokens * source_weights.view(1, -1, 1)
        ns_tokens = self.ns_token_dropout(ns_tokens)
        ns_tokens = ns_tokens + self.ns_field_embedding.unsqueeze(0)
        return self.ns_input_norm(ns_tokens)

    def _build_sequence_tokens(self, item_embedding_dict, history_mask):
        history_fields = [
            item_embedding_dict[feature][:, :-1, :]
            for feature in self.sequence_features
        ]
        history_embedding = torch.cat(history_fields, dim=-1)
        sequence_length = history_embedding.size(1)
        if sequence_length > self.max_len:
            raise ValueError(
                f"sequence length {sequence_length} exceeds "
                f"max_len={self.max_len}"
            )

        sequence_tokens = self.sequence_projection(history_embedding)
        # Count valid chronological events so left padding does not alter the
        # position assigned to real behavior tokens.
        valid_mask = history_mask.to(dtype=torch.bool)
        position_ids = valid_mask.long().cumsum(dim=1).sub(1).clamp_min(0)
        position_tokens = self.position_embedding[position_ids]
        sequence_tokens = self.sequence_input_norm(
            sequence_tokens + self.position_embedding_scale * position_tokens
        )
        return sequence_tokens * valid_mask.unsqueeze(-1).to(
            dtype=sequence_tokens.dtype
        )

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        batch_size = mask.size(0)
        if mask.size(1) > self.max_len:
            raise ValueError(
                f"input history length {mask.size(1)} exceeds "
                f"max_len={self.max_len}"
            )

        context_embedding_dict = self.embedding_layer.embedding_layer(
            batch_dict
        )
        item_embedding_dict = self.embedding_layer.embedding_layer(item_dict)

        ns_field_tokens = self._build_non_sequence_tokens(
            context_embedding_dict, item_embedding_dict
        )
        ns_mask = torch.ones(
            batch_size,
            ns_field_tokens.size(1),
            dtype=mask.dtype,
            device=mask.device,
        )
        ns_queries = self.activation_checkpoint(
            self.ns_qformer, ns_field_tokens, ns_mask
        )

        history_mask = mask.to(device=ns_queries.device)
        sequence_tokens = self._build_sequence_tokens(
            item_embedding_dict, history_mask
        )
        eval_only_scales = (
            self.task_unified_post_ffn_eval_scales.tolist()
            if not self.training and not torch.all(
                self.task_unified_post_ffn_eval_scales == 1
            )
            else None
        )
        effective_scales = (
            self.task_unified_post_ffn_scales
            if self.task_unified_post_ffn_scales is not None
            else eval_only_scales
        )
        if effective_scales is None:
            unified_queries = self.activation_checkpoint(
                self.unified_qformer,
                ns_queries,
                sequence_tokens,
                history_mask,
            )
            # Apply query-interface dropout to the tensor that is actually
            # flattened below.  Keeping the dropped value only in
            # ``task_queries`` made query_dropout a no-op on the default
            # (unscaled) forward path.
            unified_queries = self.query_dropout(unified_queries)
            task_queries = [unified_queries] * self.num_tasks
        else:
            pre_ffn, ffn_output = self.unified_qformer.forward_components(
                ns_queries, sequence_tokens, history_mask
            )
            task_queries = [
                self.query_dropout(self.unified_qformer.combine_post_ffn(
                    pre_ffn, ffn_output, scale
                ))
                for scale in effective_scales
            ]
            unified_queries = task_queries[0]

        if self.task_unified_post_ffn_adapters:
            task_queries = [
                queries + self.task_unified_post_ffn_adapters[index](queries)
                for index, queries in enumerate(task_queries)
            ]
            unified_queries = task_queries[0]

        if self.query_pooling in {"target_attention", "target_residual"}:
            target_count = len(self.target_item_features)
            target_query = ns_field_tokens[:, -target_count:, :].mean(dim=1)
            query_scores = torch.einsum(
                "bqd,bd->bq", unified_queries, target_query
            ) / math.sqrt(self.token_dim)
            query_weights = torch.softmax(query_scores, dim=1).unsqueeze(-1)
            pooled = torch.sum(query_weights * unified_queries, dim=1)
            repeated_pool = pooled.repeat(1, self.num_queries)
            if self.query_pooling == "target_attention":
                bottom_output = repeated_pool
            else:
                bottom_output = unified_queries.flatten(start_dim=1)
                bottom_output = bottom_output + (
                    self.target_pool_residual_scale * repeated_pool
                )
        else:
            unscaled_task_queries = task_queries
            route_scales = (
                self.query_fusion_training_scales
                if self.training else self.query_fusion_fixed_scales
            )
            fixed_scales = route_scales.to(
                device=task_queries[0].device, dtype=task_queries[0].dtype
            ).view(1, self.num_queries, 1)
            task_queries = [queries * fixed_scales for queries in task_queries]
            if self.query_fusion_zero_recovery_logits is not None:
                recovered_queries = []
                for index, queries in enumerate(task_queries):
                    recovered = queries.clone()
                    recovery = torch.tanh(self.query_fusion_zero_recovery_logits[
                        index if self.query_fusion_zero_recovery_mode == "task" else 0
                    ])
                    recovered[:, self.query_fusion_zero_indices, :] = (
                        unscaled_task_queries[index][
                            :, self.query_fusion_zero_indices, :
                        ] * recovery
                    )
                    recovered_queries.append(recovered)
                task_queries = recovered_queries
                unified_queries = task_queries[0]
            if self.query_fusion_scale_logits is not None:
                scales = 1.0 + self.query_fusion_learned_scale_bound * torch.tanh(
                    self.query_fusion_scale_logits
                )
                task_queries = [
                    queries * scales[
                        index if scales.size(0) == self.num_tasks else 0
                    ].unsqueeze(0)
                    for index, queries in enumerate(task_queries)
                ]
                unified_queries = task_queries[0]
            if effective_scales is None:
                tower_inputs = [
                    queries.reshape(batch_size, -1) for queries in task_queries
                ]
                bottom_output = tower_inputs[0]
            else:
                tower_inputs = [
                    queries.reshape(batch_size, -1) for queries in task_queries
                ]
                bottom_output = tower_inputs[0]
        if self.query_pooling in {"target_attention", "target_residual"}:
            tower_inputs = [bottom_output for _ in range(self.num_tasks)]
        if self.task_residual_gate_strength:
            strength = self.task_residual_gate_strength
            tower_inputs = [
                bottom_output * (
                    1.0 + strength * torch.tanh(
                        self.task_residual_gate[index](bottom_output)
                    )
                )
                for index in range(self.num_tasks)
            ]
        tower_output = [
            self.tower[index](tower_inputs[index])
            for index in range(self.num_tasks)
        ]
        y_pred = [
            self.output_activation[index](tower_output[index])
            for index in range(self.num_tasks)
        ]
        labels = self.feature_map.labels
        return {
            f"{labels[index]}_pred": y_pred[index]
            for index in range(self.num_tasks)
        }
