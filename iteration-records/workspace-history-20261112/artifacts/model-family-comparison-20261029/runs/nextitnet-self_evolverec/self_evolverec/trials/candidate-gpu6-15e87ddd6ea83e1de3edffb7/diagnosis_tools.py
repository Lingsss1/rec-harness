### >>> Self_EvolveRec-BLOCK-START: Configure numerical prerequisites without repairing an initialized CUDA process
import os
import sys

_torch_module = sys.modules.get("torch")
_cuda_already_initialized = bool(
    _torch_module is not None and _torch_module.cuda.is_initialized()
)
_workspace_at_import = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
_workspace_error = None
if _workspace_at_import not in (":4096:8", ":16:8"):
    if _cuda_already_initialized:
        _workspace_error = (
            "CUDA was already initialized with absent or unsupported "
            "CUBLAS_WORKSPACE_CONFIG="
            + repr(_workspace_at_import)
            + "; deterministic measurements require a fresh process"
        )
    elif _workspace_at_import is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        _workspace_error = (
            "Unsupported CUBLAS_WORKSPACE_CONFIG="
            + repr(_workspace_at_import)
            + "; restart with ':4096:8' or ':16:8'"
        )

import torch
import numpy as np
import torch.nn.functional as F
import json
import random
from collections import Counter

### <<< Self_EvolveRec-BLOCK-END


def diagnosis_interpreter_prompt(raw_diagnosis):
    return f"""
    You are a Senior AI Researcher analyzing the mathematical health of a Recommendation System.
    
    Here are the raw mathematical metrics measured from the current model:
    {json.dumps(raw_diagnosis['metrics'], indent=2)}
    
    Here are the definitions of what these metrics mean:
    {json.dumps(raw_diagnosis['metric_definitions'], indent=2)}
    
    Your task: produce a concise diagnosis summary that captures ONLY the core findings and implications.
    Do NOT propose web searches or action plans. Do NOT list "what to look up".

    Strictly follow the JSON format below.

    Output Format (JSON):
    {{
    "status": <CRITICAL | NEEDS_IMPROVEMENT | STABLE>,
    "core_findings": [
        "<1-2 sentences: the most important interpretation of the metrics>",
        "<1-2 sentences: the second most important interpretation (only if truly necessary)>"
    ],
    "key_implications": [
        "<1 sentence: what this implies about model behavior>",
        "<1 sentence: what this implies about training dynamics or representation>",
        "<optional 1 sentence: risk/trade-off if relevant>"
    ],
    "evidence": {{
        "headline_metrics": {{
        "<metric_name>": <value>,
        "<metric_name>": <value>
        }},
        "brief_metric_read": {{
        "<metric_name>": "<very short interpretation tied to its definition>",
        "<metric_name>": "<very short interpretation tied to its definition>"
        }}
    }}
    }}

    Rules:
    - Keep it short: core_findings <= 3 bullets, key_implications <= 3 bullets.
    - Use technical terms only when they are essential (e.g., 'gradient', 'loss landscape', 'inductive bias'), and keep them brief.
    - Every statement must be grounded in the provided metric values/definitions.
    ### >>> Self_EvolveRec-BLOCK-START: Interpret measured training diagnostics with explicit limitations
    - Null measurements are unavailable, never evidence of healthy behavior.
    - Separate implementation contract failures from in-sample history sensitivity.
    - Embedding alignment does not establish recommendation diversity or preferences.
    - Training-prefix outcomes are not validation results or evidence of improvement.
    - Simulator traits, ratings, reviews, prices, stores and item categories are unobserved.
    - Bootstrap intervals are exploratory and conditional on checkpoint, cached events, candidates and realized edits; they exclude training, search, candidate-sampling and permutation-distribution uncertainty.
    - Unverified export parity or causal controls must remain explicitly unverified.
    - Arrangement sensitivity does not establish temporal reasoning, harmful history, current intent or preferences. Invariance does not establish that order is never used.
    - Structural support is conditional on preserving the nonpadding mask; reachable offsets need not influence this checkpoint.
    - Report condition-specific and common eligibility, actual token changes and numerical failures.
    - Repeat-capable and novelty-only candidate protocols describe different training populations.
    - Do not infer a four-category macro estimate from a subset of categories.
    - Selection remains exclusively pinned host four-category macro validation HR@10.
    ### <<< Self_EvolveRec-BLOCK-END
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the ModelView contract and actual network ownership
    def __init__(self, view, device):
        self.view = view
        self.model = view
        self.network = view.model
        if not isinstance(self.network, torch.nn.Module):
            raise TypeError("view.model must be the actual trained torch.nn.Module")
        self.device = torch.device(device)
        self.meta_data = getattr(view, "meta_data", {}) or {}
        self.review_data = getattr(view, "review_data", {}) or {}
        self.user_train = view.user_train
        self.embedding_sample_count = 0
        self.category = getattr(view, "dataset_name", None)
        if self.category not in (
            "Movies_and_TV",
            "Industrial_and_Scientific",
            "Electronics",
            "CDs_and_Vinyl",
        ):
            self.category = None

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Do not infer unavailable item categories
    def _get_categories(self, item_id: int):
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()
            emb = emb[
                1 : self.model.model.itemnum + 1
            ]  # exclude padding and reserved mask tokens

            n = emb.size(0)
            ### >>> Self_EvolveRec-BLOCK-START: Report unavailable embedding estimates and their denominators
            self.embedding_sample_count = min(sample_size, n)
            if min(sample_size, n) < 2:
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx)
            x = F.normalize(x, p=2, dim=-1)

            gram = x @ x.t()
            offdiag_sum = gram.sum() - gram.diag().sum()
            offdiag_cnt = k * (k - 1)
            mean_cos = (offdiag_sum / offdiag_cnt).clamp(-1, 1).item()

            collapse_score = (mean_cos + 1.0) * 0.5
            return round(float(collapse_score), 4)

    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(
        self,
        dataloader,
        num_batches: int = 10,
        eps: float = 0.5,
        topk_categories: int = 20,
    ):
        ### >>> Self_EvolveRec-BLOCK-START: Require the diagnostic evaluation context
        if self.network.training:
            raise RuntimeError(
                "Run margin probes through run_full_diagnosis in evaluation mode"
            )
        margins_all = []
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        for step, batch in enumerate(dataloader):
            if step >= num_batches:
                break
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 4):
                continue

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            ### >>> Self_EvolveRec-BLOCK-START: Convert diagnostic batches through the family adapter
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            log_feats = self._features(seq_t)
            ### <<< Self_EvolveRec-BLOCK-END

            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)

            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
            margin = all_logits[..., 0] - all_logits[..., 1]

            ### >>> Self_EvolveRec-BLOCK-START: Exclude padding and invalid margin positions
            valid = (seq_t != 0) & (pos_t != 0) & (neg_t != 0)
            valid &= torch.isfinite(margin)
            ### <<< Self_EvolveRec-BLOCK-END
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Use identical margin denominators throughout
            valid_cpu = valid.detach().cpu().numpy()
            ### <<< Self_EvolveRec-BLOCK-END

            low_mask = valid_cpu & (margin_cpu < float(eps))
            wrong_mask = valid_cpu & (margin_cpu < 0.0)

            for b, l in np.argwhere(low_mask):
                low_total += 1
                pid = int(pos_cpu[b, l])
                nid = int(neg_cpu[b, l])
                for c in self._get_categories(pid):
                    low_pos_cat[c] += 1
                for c in self._get_categories(nid):
                    low_neg_cat[c] += 1

            for b, l in np.argwhere(wrong_mask):
                wrong_total += 1
                pid = int(pos_cpu[b, l])
                nid = int(neg_cpu[b, l])
                for c in self._get_categories(pid):
                    wrong_pos_cat[c] += 1
                for c in self._get_categories(nid):
                    wrong_neg_cat[c] += 1

        if len(margins_all) == 0:
            ### >>> Self_EvolveRec-BLOCK-START: Empty margin measurements are unavailable
            margin_metrics = {
                "mean": None,
                "p10": None,
                "neg_beats_pos_rate": None,
                "small_margin_rate": None,
            }
            ### <<< Self_EvolveRec-BLOCK-END
        else:
            m = torch.cat(margins_all, dim=0)
            N = m.numel()

            k = max(1, int(np.ceil(0.10 * N)))
            p10 = float(torch.kthvalue(m, k).values.item())

            margin_metrics = {
                "mean": round(float(m.mean().item()), 4),
                "p10": round(p10, 4),
                "neg_beats_pos_rate": round(float((m < 0.0).float().mean().item()), 4),
                "small_margin_rate": round(
                    float((m < float(eps)).float().mean().item()), 4
                ),
            }

        def _topk(counter: Counter, k: int):
            return [{"category": c, "count": int(n)} for c, n in counter.most_common(k)]

        ### >>> Self_EvolveRec-BLOCK-START: Record margin coverage and unavailable category information
        breakdown = {
            "valid_positions": sum(int(m.numel()) for m in margins_all),
            "category_breakdown_reason": "Item categories are unavailable in this task",
            "eps": float(eps),
            ### <<< Self_EvolveRec-BLOCK-END
            "low_margin_total_positions": int(low_total),
            "wrong_rank_total_positions": int(wrong_total),
            "low_margin_top_categories": {
                "pos_item": _topk(low_pos_cat, topk_categories),
                "neg_item": _topk(low_neg_cat, topk_categories),
            },
            "wrong_rank_top_categories": {
                "pos_item": _topk(wrong_pos_cat, topk_categories),
                "neg_item": _topk(wrong_neg_cat, topk_categories),
            },
        }

        return margin_metrics, breakdown

    ### >>> Self_EvolveRec-BLOCK-START: Bounded scoring with stricter dependency certification and explicit audit failures
    def _features(self, sequence):
        # ModelView implementations may convert NumPy inputs themselves. The
        # supplied NextItNet network accepts tensors; do not reduce B x L x H.
        try:
            features = self.view.log2feats(sequence)
        except (TypeError, AttributeError):
            features = self.view.log2feats(sequence.detach().cpu().numpy())
        if not isinstance(features, torch.Tensor) or features.ndim != 3:
            raise ValueError("log2feats must return B x L x H tensors")
        if features.shape[:2] != sequence.shape:
            raise ValueError("log2feats changed sequence dimensions")
        if features.dtype != torch.float32 or not torch.isfinite(features).all():
            raise ValueError("Diagnostic features must be finite FP32 values")
        return features

    def _tensor(self, histories):
        length = int(self.network.maxlen)
        result = torch.zeros(
            (len(histories), length), device=self.device, dtype=torch.long
        )
        for row, history in enumerate(histories):
            tail = history[-length:]
            if tail:
                result[row, -len(tail) :] = torch.as_tensor(
                    tail, device=self.device, dtype=torch.long
                )
        return result

    def _scores(self, sequence, candidates):
        features = self._features(sequence)
        positions = torch.arange(sequence.size(1), device=self.device).expand_as(
            sequence
        )
        last = positions.masked_fill(sequence == 0, -1).max(dim=1).values
        if (last < 0).any():
            raise ValueError("Cannot score an empty diagnostic history")
        items = torch.as_tensor(candidates, device=self.device, dtype=torch.long)
        if items.ndim == 1:
            items = items.unsqueeze(0).expand(sequence.size(0), -1)
        if items.ndim != 2 or items.size(0) != sequence.size(0):
            raise ValueError("Candidates must have shape N or B x N")
        if items.size(1) < 1 or ((items < 1) | (items > self.network.itemnum)).any():
            raise ValueError("Candidates must contain valid nonpadding catalog IDs")
        # Retain the sequence axis required by ModelView, but score only the
        # original final occupied feature. This avoids allocating B x L x N x H
        # candidate embeddings. Every paired comparison uses this same path.
        rows = torch.arange(sequence.size(0), device=features.device)
        readout = features[rows, last.to(features.device)].unsqueeze(1)
        items = items.unsqueeze(1)
        logits = self.view.predict_score(readout, items)
        if not isinstance(logits, torch.Tensor) or logits.shape != items.shape:
            raise ValueError("predict_score must preserve B x 1 x N candidate axes")
        if logits.dtype != torch.float32 or not torch.isfinite(logits).all():
            raise ValueError("Diagnostic scores must be finite FP32 values")
        return logits[:, 0, :]

    @staticmethod
    def _seed(*parts):
        import hashlib

        message = json.dumps([42, *parts], ensure_ascii=True, separators=(",", ":"))
        return int.from_bytes(hashlib.sha256(message.encode()).digest()[:8], "little")

    def _dependencies(self):
        import ast
        import inspect
        import textwrap

        # Certify only the supplied implementation. Module metadata alone cannot
        # reveal functional pooling, gates, direct lookups or normalization axes.
        expected = {
            "encode_sequence": """
                def encode_sequence(self, seq: torch.Tensor) -> torch.Tensor:
                    valid = seq != 0
                    mask = valid.unsqueeze(-1).to(dtype=self.item_emb.weight.dtype)
                    pos = valid.long().cumsum(dim=1) * valid.long()
                    x = self.item_emb(seq) * (self.hidden_units**0.5) + self.pos_emb(pos)
                    x = self.dropout(x * mask) * mask
                    for block in self.blocks:
                        x = block(x, mask)
                    return self.norm(x) * mask
            """,
            "log2feats": """
                def log2feats(self, sequence: torch.Tensor) -> torch.Tensor:
                    return self.encode_sequence(sequence)
            """,
            "_causal_conv": """
                def _causal_conv(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
                    pad = (self.kernel_size - 1) * self.dilation
                    x = nn.functional.pad(x, (pad, 0))
                    return conv(x)
            """,
            "block_forward": """
                def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                    residual = x * mask
                    y = (self.norm1(residual) * mask).transpose(1, 2)
                    y = self._causal_conv(self.conv1, y).transpose(1, 2)
                    y = self.dropout(self.activation(y)) * mask
                    y = (self.norm2(y) * mask).transpose(1, 2)
                    y = self._causal_conv(self.conv2, y).transpose(1, 2)
                    y = self.dropout(y) * mask
                    if self.gate_mode == "fixed_one":
                        return (residual + y) * mask
                    if self.gate_mode == "contextual":
                        logits = self.gate(torch.cat((residual, y), dim=-1))
                    else:
                        logits = self.gate_bias
                    numerator = torch.sigmoid(logits)
                    denominator = torch.sigmoid(numerator.new_tensor(2.0))
                    gate = numerator / denominator
                    return (residual + gate * y) * mask
            """,
        }

        def check_source(bound, key):
            function = getattr(bound, "__func__", None)
            if function is None:
                raise ValueError("Unknown or overridden inference method: " + key)
            observed = ast.parse(textwrap.dedent(inspect.getsource(function)))
            wanted = ast.parse(textwrap.dedent(expected[key]))
            if ast.dump(observed, include_attributes=False) != ast.dump(
                wanted, include_attributes=False
            ):
                raise ValueError(
                    "Inference source requires a new dependency audit: " + key
                )

        model = self.network
        if type(model).__name__ != "NextItNetModel":
            raise ValueError("Unknown NextItNet inference implementation")
        check_source(model.encode_sequence, "encode_sequence")
        check_source(model.log2feats, "log2feats")
        allowed = {
            torch.nn.ModuleList,
            torch.nn.Embedding,
            torch.nn.Conv1d,
            torch.nn.LayerNorm,
            torch.nn.Dropout,
            torch.nn.GELU,
            torch.nn.Linear,
            type(model),
        }
        allowed.update(type(block) for block in model.blocks)
        for name, module in model.named_modules():
            if type(module) not in allowed:
                raise ValueError("Unknown functional module: " + name)
            if module._forward_hooks or module._forward_pre_hooks:
                raise ValueError("Forward hooks invalidate source tracing: " + name)
            # Exact types alone do not reject instance-level forward overrides
            # or custom module dispatch. Those paths bypass the audited source.
            if (
                type(module).__call__ is not torch.nn.Module.__call__
                or getattr(module._call_impl, "__func__", None)
                is not torch.nn.Module._call_impl
                or getattr(module._wrapped_call_impl, "__func__", None)
                is not torch.nn.Module._wrapped_call_impl
                or getattr(module, "_compiled_call_impl", None) is not None
            ):
                raise ValueError("Unaudited module dispatch: " + name)
            if (
                type(module)
                in {
                    torch.nn.Embedding,
                    torch.nn.Conv1d,
                    torch.nn.LayerNorm,
                    torch.nn.Dropout,
                    torch.nn.GELU,
                    torch.nn.Linear,
                }
                and getattr(module.forward, "__func__", None)
                is not type(module).forward
            ):
                raise ValueError("Overridden standard-module forward: " + name)
            if isinstance(module, torch.nn.Embedding) and module.max_norm is not None:
                raise ValueError(
                    "Embedding lookup can mutate checkpoint weights: " + name
                )
            if isinstance(module, torch.nn.LayerNorm):
                if tuple(module.normalized_shape) != (model.hidden_units,):
                    raise ValueError("Normalization is not channel-only: " + name)
            if module.training:
                raise ValueError("Support tracing requires evaluation mode")
        import torch.nn.modules.module as module_runtime

        if (
            module_runtime._global_forward_hooks
            or module_runtime._global_forward_pre_hooks
        ):
            raise ValueError("Global forward hooks invalidate source tracing")
        if type(model.item_emb) is not torch.nn.Embedding:
            raise ValueError("Unknown item lookup")
        if type(model.pos_emb) is not torch.nn.Embedding:
            raise ValueError("Unknown position lookup")
        offsets, description = {0}, []
        for index, block in enumerate(model.blocks):
            if type(block).__name__ != "NextItNetBlock":
                raise ValueError("Unknown residual branch implementation")
            check_source(block.forward, "block_forward")
            check_source(block._causal_conv, "_causal_conv")
            if type(block.activation) is not torch.nn.GELU:
                raise ValueError("Unknown block activation")
            if block.gate_mode not in {
                "fixed_one",
                "scalar",
                "bias_only",
                "contextual",
            }:
                raise ValueError("Unknown gate dependency path")
            if (
                block.gate_mode == "contextual"
                and type(block.gate) is not torch.nn.Linear
            ):
                raise ValueError("Unknown contextual gate")
            residual, branch = set(offsets), set(offsets)
            for name in ("conv1", "conv2"):
                conv = getattr(block, name)
                if (
                    type(conv) is not torch.nn.Conv1d
                    or conv.stride != (1,)
                    or conv.padding != (0,)
                    or conv.kernel_size != (block.kernel_size,)
                    or conv.dilation != (block.dilation,)
                    or conv.padding_mode != "zeros"
                ):
                    raise ValueError(
                        "Convolution differs from audited left-padding path"
                    )
                branch = {
                    lag + tap * conv.dilation[0]
                    for lag in branch
                    for tap in range(conv.kernel_size[0])
                }
                description.append(
                    {
                        "block": index,
                        "name": name,
                        "kernel": conv.kernel_size[0],
                        "dilation": conv.dilation[0],
                    }
                )
            # The contextual gate consumes both residual and completed update.
            # Its product with the update unions those same temporal supports.
            offsets = residual | branch
        return sorted(offsets), description

    def _controls(self, offsets):
        length, vocab = int(self.network.maxlen), int(self.network.itemnum)
        result = {"atol": 1e-6, "rtol": 1e-5, "failures": []}
        if vocab < 2 or length < 3:
            result["failures"].append("Insufficient vocabulary or sequence width")
            return result
        candidates = list(range(1, min(vocab, 7) + 1))
        items = torch.tensor(candidates, device=self.device)
        checks, shape_checks = [], []
        for history in ([1], [1, 2], [1] * length):
            sequence = self._tensor([history])
            first_features = self._features(sequence)
            second_features = self._features(sequence.clone())
            direct_features = self.network.log2feats(sequence)
            first = self._scores(sequence, candidates)
            second = self._scores(sequence.clone(), candidates)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            if not torch.equal(first_features, second_features) or not torch.equal(
                first, second
            ):
                raise ValueError("Actual same-shape feature/score repeatability failed")
            deployed = self.network.score_candidates(sequence, items)
            valid = (
                deployed.shape == first[0].shape
                and torch.isfinite(deployed).all().item()
                and torch.allclose(deployed, first[0], atol=1e-6, rtol=1e-5)
                and torch.allclose(
                    first_features, direct_features, atol=1e-6, rtol=1e-5
                )
            )
            checks.append({"history_length": len(history), "match": bool(valid)})
            if not valid:
                result["failures"].append("ModelView/direct-network readout mismatch")
            batched = self._scores(sequence.expand(2, -1).clone(), candidates)
            error = (batched - first).abs()
            floor = 1e-8
            relative = error / first.abs().clamp_min(floor)
            reference = first[0].cpu().tolist()
            rows = batched.cpu().tolist()
            rank = lambda row: sorted(
                range(len(row)), key=lambda j: (-row[j], candidates[j])
            )
            shape_checks.append(
                {
                    "history_length": len(history),
                    "max_abs_error": float(error.max()),
                    "max_relative_error": float(relative.max()),
                    "relative_denominator_floor": floor,
                    "within_tolerance": bool(
                        torch.allclose(
                            batched, first.expand_as(batched), atol=1e-6, rtol=1e-5
                        )
                    ),
                    "candidate_order_changes": sum(
                        rank(row) != rank(reference) for row in rows
                    ),
                    "near_target_ties": int(
                        (
                            (first[0, 1:] - first[0, 0]).abs()
                            <= 1e-6 + 1e-5 * first[0, 0].abs()
                        ).sum()
                    ),
                }
            )
        result["left_padded_readout_checks"] = checks
        result["same_shape_repeatability"] = True
        result["batch_shape_sensitivity"] = shape_checks
        # No copied models or altered parameters are required for causal checks.
        x = self._tensor([[1] * length])
        reference = self._features(x)
        causal = []
        for position in sorted({0, length // 2, length - 2}):
            edited = x.clone()
            edited[:, position + 1 :] = 2
            changed = self._features(edited)
            a, b = reference[:, : position + 1], changed[:, : position + 1]
            passed = bool(torch.allclose(a, b, atol=1e-6, rtol=1e-5))
            causal.append(
                {
                    "position": position,
                    "max_abs_error": float((a - b).abs().max()),
                    "invariant": passed,
                }
            )
            if not passed:
                result["failures"].append("Future-token invariance failed")
        result["causality"] = causal
        result["outside_dependency"] = None
        if offsets is not None:
            outside = [p for p in range(length - 1) if length - 1 - p not in offsets]
            if outside:
                edited = x.clone()
                edited[:, outside] = 2
                a, b = self._scores(x, candidates), self._scores(edited, candidates)
                passed = bool(torch.allclose(a, b, atol=1e-6, rtol=1e-5))
                result["outside_dependency"] = {
                    "positions": outside,
                    "invariant": passed,
                    "max_abs_error": float((a - b).abs().max()),
                }
                if not passed:
                    result["failures"].append("Outside-support invariance failed")
        return result

    def _export_parity(self, records):
        import importlib.util
        import inspect
        import os

        if not records or self.category is None:
            return {
                "available": False,
                "reason": "An explicit category route and eligible prefix are required",
            }
        try:
            root = os.path.dirname(os.path.abspath(inspect.getfile(type(self.network))))
            path = os.path.join(root, "working", "predict.py")
            if not os.path.isfile(path):
                return {
                    "available": False,
                    "reason": "The trial predictor has not been written",
                }
            spec = importlib.util.spec_from_file_location(
                "nextitnet_diagnostic_predictor", path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module._ensure_loaded()
            exported = module._models[self.category]
            live_state, export_state = self.network.state_dict(), exported.state_dict()
            identical = live_state.keys() == export_state.keys() and all(
                torch.equal(value.detach().cpu(), export_state[key].detach().cpu())
                for key, value in live_state.items()
            )
            if not identical:
                return {
                    "available": False,
                    "reason": "Live model differs from the exported selected checkpoint",
                }
            checks = []
            for record in records[:2]:
                history, candidates = record["history"], record["candidates"]
                expected = (
                    self._scores(self._tensor([history]), candidates)[0].cpu().numpy()
                )
                observed = np.asarray(
                    module.predict(
                        record["user"], history, candidates, dataset_name=self.category
                    ),
                    dtype=np.float64,
                )
                order = record["tie_order"]
                checks.append(
                    {
                        "prefix_id": record["prefix_id"],
                        "max_abs_error": float(np.max(np.abs(expected - observed))),
                        "scores_match": bool(
                            np.allclose(expected, observed, atol=1e-5, rtol=1e-5)
                        ),
                        "rank_order_match": sorted(
                            range(len(expected)), key=lambda j: (-expected[j], order[j])
                        )
                        == sorted(
                            range(len(observed)), key=lambda j: (-observed[j], order[j])
                        ),
                    }
                )
            return {
                "available": True,
                "checks": checks,
                "atol": 1e-5,
                "rtol": 1e-5,
                "batched_export": None,
                "batched_export_reason": "The supplied public predictor accepts one history per call",
            }
        except Exception as exc:
            return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    def _prefixes(self, offsets, cached):
        source = self.user_train
        source = source.get("History", source) if isinstance(source, dict) else {}
        if not isinstance(source, dict):
            raise ValueError("user_train histories must be a user-to-history mapping")
        excluded, wanted, histories = Counter(), {}, {}
        vocab, width = int(self.network.itemnum), int(self.network.maxlen)
        for user, raw in source.items():
            values = np.asarray(raw)
            if (
                values.ndim != 1
                or not np.issubdtype(values.dtype, np.number)
                or not np.isfinite(values).all()
                or not np.equal(values, np.floor(values)).all()
                or ((values < 1) | (values > vocab)).any()
            ):
                excluded["invalid_stored_history"] += 1
                continue
            histories[str(user)] = values.astype(np.int64).tolist()
        for batch in cached:
            if not isinstance(batch, (tuple, list)) or len(batch) != 4:
                excluded["malformed_batch"] += 1
                continue
            arrays = [torch.as_tensor(x).detach().cpu().numpy() for x in batch[1:4]]
            seq, pos, neg = arrays
            if (
                seq.ndim != 2
                or seq.shape[1] != width
                or pos.shape != seq.shape
                or neg.shape != seq.shape
            ):
                excluded["malformed_batch"] += 1
                continue
            for s, p in zip(seq, pos):
                excluded["cached_rows"] += 1
                if (
                    not np.isfinite(s).all()
                    or not np.isfinite(p).all()
                    or not np.equal(s, np.floor(s)).all()
                    or not np.equal(p, np.floor(p)).all()
                ):
                    excluded["noninteger_or_nonfinite_row"] += 1
                    continue
                occupied = np.flatnonzero(s != 0)
                if not len(occupied):
                    excluded["empty_row"] += 1
                    continue
                if (
                    ((s < 0) | (s > vocab)).any()
                    or not np.array_equal(occupied, np.arange(occupied[0], width))
                    or not 1 <= p[occupied[-1]] <= vocab
                ):
                    excluded["invalid_prefix_or_target"] += 1
                    continue
                key = (tuple(int(i) for i in s[occupied]), int(p[occupied[-1]]))
                if key in wanted:
                    excluded["duplicate_encoded_row"] += 1
                wanted.setdefault(key, [])
        for user, history in sorted(histories.items()):
            for endpoint in range(1, len(history)):
                key = (
                    tuple(history[max(0, endpoint - width) : endpoint]),
                    history[endpoint],
                )
                if key in wanted and len(wanted[key]) < 2:
                    wanted[key].append((user, endpoint))
        by_user = {}
        for matches in wanted.values():
            if len(matches) != 1:
                excluded["ambiguous_event" if matches else "unidentified_event"] += 1
                continue
            user, endpoint = matches[0]
            by_user.setdefault(user, set()).add(endpoint)
        rng = np.random.default_rng(42)
        pool_users = sorted(by_user)
        chosen = sorted(
            rng.choice(
                len(pool_users), min(32, len(pool_users)), replace=False
            ).tolist()
        )
        records, edits, prefix_counts = [], {}, {}
        for user_index in chosen:
            user = pool_users[user_index]
            endpoints = sorted(by_user[user])
            picked = sorted(
                rng.choice(
                    len(endpoints), min(2, len(endpoints)), replace=False
                ).tolist()
            )
            prefix_counts[user] = len(picked)
            for index in picked:
                endpoint = endpoints[index]
                history = histories[user]
                prefix, target = history[:endpoint], history[endpoint]
                retained = prefix[-width:]
                interior = list(range(len(retained) - 1))
                accessible = (
                    None
                    if offsets is None
                    else [j for j in interior if len(retained) - 1 - j in offsets]
                )
                inaccessible = (
                    None
                    if offsets is None
                    else [j for j in interior if j not in accessible]
                )
                variants, manifest = {"unchanged": retained.copy()}, {}

                def add(name, positions, origins):
                    changed = retained.copy()
                    for destination, origin in zip(positions, origins):
                        changed[destination] = retained[origin]
                    assert changed[-1] == retained[-1]
                    assert len(changed) == len(retained) and sorted(changed) == sorted(
                        retained
                    )
                    variants[name] = changed
                    manifest[name] = {
                        "destination_origin_pairs": [
                            [int(a), int(b)] for a, b in zip(positions, origins)
                        ],
                        "selected_positions": len(positions),
                        "unchanged_draw": changed == retained,
                        "changed_tokens": sum(
                            a != b for a, b in zip(changed, retained)
                        ),
                        "mean_assignment_displacement": (
                            float(
                                np.mean(
                                    [abs(a - b) for a, b in zip(positions, origins)]
                                )
                            )
                            if positions
                            else 0.0
                        ),
                    }

                add("identity", interior, interior)
                for condition, positions in (
                    ("accessible", accessible),
                    ("unrestricted", interior),
                    ("inaccessible", inaccessible),
                ):
                    if positions is None or len(positions) < 2:
                        excluded[condition + "_unavailable"] += 1
                        continue
                    for draw in range(3):
                        local = np.random.default_rng(
                            self._seed(self.category, user, endpoint, condition, draw)
                        )
                        add(
                            f"{condition}/{draw}",
                            positions,
                            local.permutation(positions).tolist(),
                        )
                duplicate = next(
                    (
                        (a, b)
                        for a in interior
                        for b in interior
                        if a < b and retained[a] == retained[b]
                    ),
                    None,
                )
                if duplicate is not None:
                    add("duplicate_swap", list(duplicate), list(reversed(duplicate)))
                else:
                    excluded["duplicate_swap_unavailable"] += 1
                adjacent = (
                    []
                    if accessible is None
                    else [
                        j
                        for j in accessible
                        if j + 1 in accessible and retained[j] != retained[j + 1]
                    ]
                )
                if adjacent:
                    local = np.random.default_rng(
                        self._seed(self.category, user, endpoint, "local_swap")
                    )
                    j = adjacent[int(local.integers(len(adjacent)))]
                    add("local_swap", [j, j + 1], [j + 1, j])
                else:
                    excluded["local_swap_unavailable"] += 1
                identifier = json.dumps(
                    [self.category, user, endpoint], separators=(",", ":")
                )
                edits[identifier] = manifest
                for protocol in ("repeat_capable", "novelty_only"):
                    repeated = target in prefix
                    if protocol == "novelty_only" and repeated:
                        excluded["novelty_only_repeat_target"] += 1
                        continue
                    forbidden = {target} | (
                        set(prefix) if protocol == "novelty_only" else set()
                    )
                    pool = [i for i in range(1, vocab + 1) if i not in forbidden]
                    if not pool:
                        excluded[protocol + "_no_alternative"] += 1
                        continue
                    local = random.Random(
                        self._seed(
                            self.category, user, endpoint, protocol, "candidates"
                        )
                    )
                    candidates = [target] + local.sample(pool, min(99, len(pool)))
                    records.append(
                        {
                            "user": user,
                            "prefix_id": identifier,
                            "endpoint": endpoint,
                            "history": prefix,
                            "target": target,
                            "protocol": protocol,
                            "repeated_target": repeated,
                            "candidates": candidates,
                            "realized_alternatives": len(candidates) - 1,
                            "reduced_candidate_pool": len(candidates) < 100,
                            "tie_order": candidates.copy(),
                            "variants": variants,
                            "edit_metadata": manifest,
                        }
                    )
        return (
            records,
            dict(excluded),
            edits,
            {
                "identifiable_user_pool": len(pool_users),
                "identifiable_event_pool": sum(map(len, by_user.values())),
                "selected_users": len(chosen),
                "prefix_counts": prefix_counts,
                "selected_events": sum(prefix_counts.values()),
                "protocol_records": len(records),
            },
        )

    @staticmethod
    def _outcome(scores, tie_order):
        scores = np.asarray(scores, dtype=np.float64)
        greater = int((scores[1:] > scores[0]).sum())
        tied = scores[1:] == scores[0]
        lower, upper = 1 + greater, 1 + greater + int(tied.sum())
        ids = np.asarray(tie_order)
        rank = lower + int((tied & (ids[1:] < ids[0])).sum())
        return np.asarray(
            [
                lower,
                upper,
                rank,
                scores[0] - scores[1:].max(),
                float(upper <= 10),
                float(lower <= 10),
            ],
            dtype=np.float64,
        )

    def _summaries(self, records):
        conditions = (
            "accessible",
            "unrestricted",
            "inaccessible",
            "identity",
            "duplicate_swap",
            "local_swap",
        )
        outcomes, controls = [], []
        for record in records:
            intact_sequence = self._tensor([record["variants"]["unchanged"]])
            occupied = torch.nonzero(intact_sequence[0] != 0, as_tuple=False).flatten()
            if not occupied.numel():
                raise ValueError("Empty event reached the order audit")
            last = int(occupied[-1])
            intact_features = self._features(intact_sequence)[:, last]
            repeated_features = self._features(intact_sequence.clone())[:, last]
            intact = self._scores(intact_sequence, record["candidates"])[0]
            repeated = self._scores(intact_sequence.clone(), record["candidates"])[0]
            if not torch.equal(intact_features, repeated_features) or not torch.equal(
                intact, repeated
            ):
                raise ValueError(
                    "Event features/scores failed same-shape repeatability"
                )
            original = self._outcome(intact.cpu().numpy(), record["tie_order"])
            results, changed_results = {}, {}
            for condition in conditions:
                names = [
                    name
                    for name in record["variants"]
                    if name == condition or name.startswith(condition + "/")
                ]
                deltas, changed_deltas = [], []
                for name in names:
                    sequence = self._tensor([record["variants"][name]])
                    if not torch.equal(sequence != 0, intact_sequence != 0):
                        raise ValueError(
                            "Order edit changed the occupied-position mask"
                        )
                    scores = self._scores(sequence, record["candidates"])[0]
                    edited = self._outcome(scores.cpu().numpy(), record["tie_order"])
                    delta = edited - original
                    # Bounds for edited rank minus original rank under unknown ties.
                    delta = np.concatenate(
                        (delta, [edited[0] - original[1], edited[1] - original[0]])
                    )
                    deltas.append(delta)
                    if record["edit_metadata"][name]["changed_tokens"]:
                        changed_deltas.append(delta)
                    if condition in ("identity", "duplicate_swap", "inaccessible"):
                        features = self._features(sequence)[:, last]
                        score_error = float((scores - intact).abs().max())
                        feature_error = float((features - intact_features).abs().max())
                        passed = bool(
                            torch.allclose(scores, intact, atol=1e-6, rtol=1e-5)
                            and torch.allclose(
                                features, intact_features, atol=1e-6, rtol=1e-5
                            )
                        )
                        controls.append(
                            {
                                "prefix_id": record["prefix_id"],
                                "protocol": record["protocol"],
                                "condition": name,
                                "passed": passed,
                                "score_max_abs_error": score_error,
                                "feature_max_abs_error": feature_error,
                            }
                        )
                if deltas:
                    results[condition] = np.mean(deltas, axis=0)
                if changed_deltas:
                    changed_results[condition] = np.mean(changed_deltas, axis=0)
            outcomes.append((record, results, changed_results))
        failed = any(not row["passed"] for row in controls)
        summaries = {}
        common_names = {"accessible", "unrestricted", "inaccessible"}
        for protocol in ("repeat_capable", "novelty_only"):
            for stratum in ("all", "repeated", "first_time"):
                for population in (
                    "condition_specific",
                    "common_permutation_eligibility",
                    "changed_only_exploratory",
                ):
                    for condition in conditions:
                        per_user, prefixes = {}, 0
                        for record, results, changed in outcomes:
                            if record["protocol"] != protocol:
                                continue
                            if stratum != "all" and record["repeated_target"] != (
                                stratum == "repeated"
                            ):
                                continue
                            if (
                                population == "common_permutation_eligibility"
                                and not common_names.issubset(results)
                            ):
                                continue
                            values = (
                                changed
                                if population == "changed_only_exploratory"
                                else results
                            )
                            if condition not in values:
                                continue
                            per_user.setdefault(record["user"], []).append(
                                values[condition]
                            )
                            prefixes += 1
                        key = "/".join((protocol, stratum, population, condition))
                        blocked = failed and condition in (
                            "accessible",
                            "inaccessible",
                            "local_swap",
                        )
                        result = {
                            "users": len(per_user),
                            "prefixes": prefixes,
                            "mean_delta": None,
                            "percentile_95_interval": None,
                            "missing_reason": (
                                "Support controls failed"
                                if blocked
                                else (None if per_user else "No eligible users")
                            ),
                        }
                        if per_user and not blocked:
                            values = np.stack(
                                [np.mean(per_user[u], axis=0) for u in sorted(per_user)]
                            )
                            result["mean_delta"] = values.mean(0).tolist()
                            if len(values) >= 2:
                                rng = np.random.default_rng(42)
                                draws = rng.integers(
                                    len(values), size=(2000, len(values))
                                )
                                result["percentile_95_interval"] = np.quantile(
                                    values[draws].mean(1), [0.025, 0.975], axis=0
                                ).T.tolist()
                            else:
                                result["interval_missing_reason"] = (
                                    "Fewer than two independent users"
                                )
                        summaries[key] = result
        return {
            "summaries": summaries,
            "invariance_controls": controls,
            "support_interpretation_blocked": failed,
        }

    def _write_manifest(self, manifest):
        import inspect
        import os

        try:
            root = os.path.dirname(os.path.abspath(inspect.getfile(type(self.network))))
            directory = os.path.join(root, "working")
            os.makedirs(directory, exist_ok=True)
            name = self.category or "unidentified_category"
            path = os.path.join(directory, f"{name}_diagnostic_manifest.json")
            with open(path, "w") as stream:
                json.dump(manifest, stream, indent=2, allow_nan=False)
            return {"written": True, "path": path}
        except Exception as exc:
            return {"written": False, "reason": f"{type(exc).__name__}: {exc}"}

    def run_full_diagnosis(self, train_loader):
        import time
        from itertools import islice

        started = time.perf_counter()
        modes = [(module, module.training) for module in self.network.modules()]
        python_state, numpy_state = random.getstate(), np.random.get_state()
        cpu_state = torch.random.get_rng_state()
        cuda_state = None
        previous = {
            "precision": torch.get_float32_matmul_precision(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        }
        reductions = {}
        for name in (
            "allow_fp16_reduced_precision_reduction",
            "allow_bf16_reduced_precision_reduction",
        ):
            if hasattr(torch.backends.cuda.matmul, name):
                reductions[name] = getattr(torch.backends.cuda.matmul, name)
        metrics = {
            "scope": "Frozen-model training sensitivity; not validation or held-out accuracy",
            "category": self.category,
            "training_prefix_sensitivity": None,
            "deterministic_measurements_available": False,
            "missing_reason": None,
            "four_category_macro": None,
            "four_category_macro_reason": "One category invocation has no four-category paired validation data",
            "export_parity": {
                "available": False,
                "reason": "No non-loading exported-model handle is supplied; no extra category checkpoints were loaded",
            },
        }
        definitions = {
            "scope": "Only public training histories and cached training targets are used. One frozen jointly trained NextItNet is measured; no parameters, training, predictor writer, evaluator or checkpoint selection are changed.",
            "numerical_configuration": "Recorded FP32 evaluation with autocast and TF32 disabled, deterministic algorithms requested with warn_only=False. Workspace readiness and repeated nontrivial GEMM and actual feature/score calls are tested. Failures discard affected summaries. This does not promise reproducibility across hardware or shapes.",
            "dependency_trace": "Source-AST verification of the supplied feature, convolution and residual-gate paths plus exact standard module checks. Convolution support composes all kernel taps sequentially; residual and contextual-gate operands are unioned. Channel LayerNorm is position-local. Positional cumsum is fixed only because every edit preserves the nonpadding mask. Reachability is structural, not measured influence. Unknown implementations yield unavailable support and unrestricted-only order measurements.",
            "sampling": "At most ten cached B x L loader batches. Encoded truncated pre-target contexts and aligned positives must uniquely identify training user and endpoint. Ambiguous matches are excluded. Seed-42 sampling selects at most 32 eligible users without replacement and at most two distinct cached events per user. Counts describe this cached sample.",
            "candidates": "Primary repeat-capable pools contain target plus up to 99 unique uniformly sampled catalog alternatives, including seen items. Novelty-only excludes repeated-target events and all complete strictly pre-target seen items. Pools are fixed across edits; reduced pools are disclosed. Unobserved alternatives are not confirmed dislikes. These are not host validation candidates.",
            "edits": "Three independent condition-keyed uniform permutations of accessible, unrestricted or inaccessible interior positions. Endpoint, padding, occupied mask, length and selected multiset are preserved. Identity, equal-ID swap and one uniformly chosen distinct-ID adjacent accessible swap are controls. SHA256 of JSON [42, category, user, endpoint, condition, draw] supplies stable seeds. Unchanged draws remain in primary means. Assignment displacement and actual token changes are recorded; condition differences are not matched corruption effects.",
            "training_prefix_sensitivity": "Vectors contain edited-minus-intact [rank_lower, rank_upper, item_ID_order_rank, target_minus_best_alternative_margin, hit10_lower, hit10_upper, paired_rank_change_lower, paired_rank_change_upper]. For target score s, g alternatives score strictly above s and t exactly equal s; ranks are [1+g,1+g+t]. Ascending item ID breaks exact ties diagnostically. Paired rank bounds are [edited_lower-original_upper, edited_upper-original_lower]. Positive rank change is worse; positive margin/hit change is better. All scoring uses singleton feature/candidate shapes and the original final occupied position.",
            "aggregation": "Average draws within prefix, prefixes within user, then equal-weight users. Report condition-specific populations, intersection eligibility of all three permutation conditions, changed-only exploratory results and original repeated/first-time strata. Main contrast is accessible permutation minus intact under repeat-capable pools. Other contrasts are controls/exploratory. Whole-user percentile bootstrap uses 2000 seed-42 replicates; fewer than two users gives no interval. Intervals condition on checkpoint, cached sample, fixed candidates and realized edits, exclude training/search/candidate/permutation-distribution uncertainty, and have no simultaneous coverage guarantee.",
            "contract_controls": "Short left-padded readouts, finite outputs, exact same-shape repeats and future-token invariance are checked on the actual frozen network. Batch-shape sensitivity is separate: atol=1e-6, rtol=1e-5, relative denominator floor=1e-8. Cross-shape differences do not establish nondeterminism. Sampled invariance does not prove source completeness.",
            "export_parity": "Unavailable unless the actual exported checkpoint, route and supported loaded handle can be compared. Direct-network/ModelView readout agreement does not establish exported-predictor equivalence.",
            "limitations": "Arrangement sensitivity does not establish temporal reasoning, obsolete interests, preference dynamics, intent, ownership or compatibility. No ratings, reviews, categories, prices, stores or simulator traits are used. Selection remains pinned four-category macro validation HR@10. No validation candidates are reconstructed. Native twenty-user full-catalog recommendation behavior is unchanged.",
        }
        try:
            if self.network.item_emb.weight.device != self.device:
                if not (
                    self.device.type == "cuda"
                    and self.device.index is None
                    and self.network.item_emb.weight.device.type == "cuda"
                ):
                    raise ValueError(
                        "Probe device differs from the trained checkpoint device"
                    )
                self.device = self.network.item_emb.weight.device
            if self.device.type == "cuda":
                if _workspace_error:
                    raise ValueError(_workspace_error)
                if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (
                    ":4096:8",
                    ":16:8",
                ):
                    raise ValueError(
                        "Invalid CuBLAS workspace; restart with a supported configuration"
                    )
                cuda_state = torch.cuda.get_rng_state(self.device)
            if any(
                p.is_floating_point() and p.dtype != torch.float32
                for p in self.network.parameters()
            ):
                raise ValueError("Diagnostics require the original FP32 checkpoint")
            self.network.eval()
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True, warn_only=False)
            for name in reductions:
                setattr(torch.backends.cuda.matmul, name, False)
            metrics["numerical_configuration"] = {
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "device": str(self.device),
                "hardware": (
                    torch.cuda.get_device_name(self.device)
                    if self.device.type == "cuda"
                    else "CPU"
                ),
                "workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "cuda_initialized_at_adapter_import": _cuda_already_initialized,
                "dtype": "float32",
                "autocast": False,
                "tf32": False,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "deterministic": True,
                "warn_only": False,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "reduced_precision_reduction": {name: False for name in reductions},
            }
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type, enabled=False
            ):
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                a = (
                    torch.arange(
                        17 * 23, device=self.device, dtype=torch.float32
                    ).reshape(17, 23)
                    / 391.0
                )
                b = (
                    torch.arange(
                        23 * 19, device=self.device, dtype=torch.float32
                    ).reshape(23, 19)
                    / 437.0
                )
                first, second = a @ b, a @ b
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                if not torch.isfinite(first).all() or not torch.equal(first, second):
                    raise ValueError(
                        "Nontrivial GEMM preflight failed finite/exact-repeat checks"
                    )
                metrics["gemm_preflight_passed"] = True
                cached = []
                for batch in islice(train_loader, 10):
                    if isinstance(batch, (tuple, list)) and len(batch) == 4:
                        cached.append(
                            (
                                None,
                                *(
                                    torch.as_tensor(x).detach().cpu().clone()
                                    for x in batch[1:4]
                                ),
                            )
                        )
                    else:
                        cached.append(batch)
                metrics["cached_batches"] = len(cached)
                try:
                    offsets, convolutions = self._dependencies()
                    trace_reason = None
                except (ValueError, TypeError, OSError, SyntaxError) as error:
                    offsets, convolutions = None, []
                    trace_reason = f"{type(error).__name__}: {error}"
                metrics["dependency_trace"] = {
                    "available": offsets is not None,
                    "offsets": offsets,
                    "receptive_width": None if offsets is None else 1 + max(offsets),
                    "convolutions": convolutions,
                    "missing_reason": trace_reason,
                    "scope": "Item-content dependencies conditional on unchanged occupied-position mask",
                }
                controls = self._controls(offsets)
                metrics["contract_controls"] = controls
                if controls["failures"]:
                    raise ValueError(
                        "Contract checks failed: " + "; ".join(controls["failures"])
                    )
                records, exclusions, edits, coverage = self._prefixes(offsets, cached)
                metrics["coverage"], metrics["exclusions"] = coverage, exclusions
                if not records:
                    raise ValueError(
                        "No eligible identifiable training events with alternatives"
                    )
                measured = self._summaries(records)
                metrics["training_prefix_sensitivity"] = measured
                metrics["deterministic_measurements_available"] = True
                metrics["support_interpretations_available"] = (
                    offsets is not None
                    and not measured["support_interpretation_blocked"]
                )
                metrics["support_interpretations_missing_reason"] = (
                    trace_reason
                    if offsets is None
                    else (
                        "Sampled support invariance controls failed"
                        if measured["support_interpretation_blocked"]
                        else None
                    )
                )
                metrics["manifest"] = self._write_manifest(
                    {
                        "seed": 42,
                        "category": self.category,
                        "hypothesis": "Accessible interior arrangement can affect fixed-candidate training rankings; no validation direction is inferred",
                        "primary_contrast": "repeat_capable/accessible_permutation_minus_intact",
                        "caps": {
                            "loader_batches": 10,
                            "users": 32,
                            "prefixes_per_user": 2,
                            "permutation_draws": 3,
                            "alternatives": 99,
                            "bootstrap": 2000,
                        },
                        "readout": "largest original occupied position",
                        "dependency_trace": metrics["dependency_trace"],
                        "coverage": coverage,
                        "exclusions": exclusions,
                        "records": [
                            {
                                k: v
                                for k, v in record.items()
                                if k not in ("variants", "edit_metadata")
                            }
                            for record in records
                        ],
                        "edits": edits,
                        "numerical_configuration": metrics["numerical_configuration"],
                    }
                )
        except (RuntimeError, ValueError, TypeError) as error:
            # Never turn a failed or partial audit into apparently valid results.
            metrics["training_prefix_sensitivity"] = None
            metrics["deterministic_measurements_available"] = False
            metrics["missing_reason"] = f"{type(error).__name__}: {error}"
            metrics["affected_partial_summaries_discarded"] = True
        finally:
            try:
                if self.device.type == "cuda" and torch.cuda.is_initialized():
                    torch.cuda.synchronize(self.device)
            except RuntimeError as error:
                # A delayed CUDA failure also invalidates completed summaries.
                # Preserve its exact message while still restoring caller state.
                metrics["training_prefix_sensitivity"] = None
                metrics["deterministic_measurements_available"] = False
                metrics["support_interpretations_available"] = False
                metrics["final_synchronization_error"] = (
                    f"{type(error).__name__}: {error}"
                )
                metrics["missing_reason"] = metrics["final_synchronization_error"]
                metrics["affected_partial_summaries_discarded"] = True
            finally:
                metrics["elapsed_synchronized_seconds"] = time.perf_counter() - started
                torch.set_float32_matmul_precision(previous["precision"])
                torch.backends.cuda.matmul.allow_tf32 = previous["matmul_tf32"]
                torch.backends.cudnn.allow_tf32 = previous["cudnn_tf32"]
                torch.backends.cudnn.benchmark = previous["benchmark"]
                torch.backends.cudnn.deterministic = previous["cudnn_deterministic"]
                for name, value in reductions.items():
                    setattr(torch.backends.cuda.matmul, name, value)
                torch.use_deterministic_algorithms(
                    previous["deterministic"], warn_only=previous["warn_only"]
                )
                torch.random.set_rng_state(cpu_state)
                if cuda_state is not None:
                    torch.cuda.set_rng_state(cuda_state, self.device)
                random.setstate(python_state)
                np.random.set_state(numpy_state)
                for module, training in modes:
                    module.training = training
        # Review summary: statically checked scoring shapes, final occupied
        # readouts, dependency dispatch, candidate eligibility, repeated feature
        # checks, empty-sample handling and delayed CUDA error restoration.
        # Training, hyperparameters and predictor architecture are unchanged.
        # No execution, diagnostic result or validation improvement is claimed.
        return {"metrics": metrics, "metric_definitions": definitions}

    ### <<< Self_EvolveRec-BLOCK-END
