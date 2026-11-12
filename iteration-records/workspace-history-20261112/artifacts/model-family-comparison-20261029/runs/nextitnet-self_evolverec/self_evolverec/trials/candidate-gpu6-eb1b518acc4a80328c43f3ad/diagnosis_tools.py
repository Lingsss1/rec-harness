import torch
import numpy as np
import torch.nn.functional as F
import json
import random
from collections import Counter


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
    ### >>> Self_EvolveRec-BLOCK-START: Interpret staged activation interventions within their measured populations
    - Missing measurements and failed contracts are unknown, never evidence of health.
    - Separate input sensitivity, unconditional completed activation patches, structural reachability, and descriptive score-responsive subsets. Score responsiveness never determines whether a patch is executed.
    - Compare exploration and confirmation separately. Confirmation remains training sensitivity, not independent validation.
    - Local, coherent-window and whole-output patches are different interventions. Direct input-embedding paths remain outside the patched residual boundary.
    - Restoration and disruption are separate directions, each relative to its own unpatched recipient.
    - Intermediate patches change only one position and may create unfamiliar hybrid states; do not infer universal necessity, sufficiency, causal locality, or memorization.
    - Endpoint replacement removes a legitimate signal and does not establish shortcut learning.
    - Donor controls measure assignment sensitivity, not a permutation null distribution; report donor reuse without recipient-only confidence intervals.
    - Bootstrap intervals condition on this checkpoint, selected users, candidates, gate and realized draws. They exclude training, search and candidate/corruption sampling uncertainty and have no simultaneous guarantees.
    - Training outcomes are not host validation, generalization or evidence of improvement. Four-category macro validation uncertainty remains unavailable.
    - Do not infer intent, ownership, compatibility, satisfaction, simulator traits or unavailable metadata.
    - Unverified export parity and dependency paths must remain explicitly unverified.
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

    ### >>> Self_EvolveRec-BLOCK-START: Audited bidirectional patches with efficient bootstrap and explicit execution coverage
    def _features(self, sequence):
        # Use the host adapter without replacing or fitting its score head.
        features = self.view.log2feats(sequence.detach().cpu().numpy())
        if (
            not isinstance(features, torch.Tensor)
            or features.ndim != 3
            or tuple(features.shape[:2]) != tuple(sequence.shape)
            or features.dtype != torch.float32
            or not torch.isfinite(features).all()
        ):
            raise ValueError("Expected finite FP32 B x L x H ModelView features")
        return features

    @staticmethod
    def _outcome(scores, candidates):
        values = scores.detach().double().cpu().numpy().reshape(-1)
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("Outcomes require a target and finite alternatives")
        ids = np.asarray(candidates, dtype=np.int64)
        # Exact computed ties, with item ID ascending as the diagnostic tie rule.
        rank = int(np.flatnonzero(np.lexsort((ids, -values)) == 0)[0]) + 1
        return np.array([values[0] - values[1:].max(), rank, float(rank <= 10)])

    @staticmethod
    def _user_summary(rows, field="delta", intervals=True, predicate=None):
        grouped = {}
        for row in rows:
            grouped.setdefault(row["user"], []).append(row)
        users = sorted(grouped)
        predicate = predicate or (lambda row: True)
        selected = {u: [r[field] for r in grouped[u] if predicate(r)] for u in users}
        contributing = [u for u in users if selected[u]]
        result = {
            "users": len(contributing),
            "cohort_users": len(users),
            "observations": sum(len(selected[u]) for u in users),
            "user_ids": contributing,
            "mean": None,
            "percentile_95_interval": None,
            "undefined_bootstrap_replicates": 0,
            "interval_missing_reason": None,
        }
        if not contributing:
            result["missing_reason"] = "No completed eligible observations"
            result["interval_missing_reason"] = "No contributing users"
            return result
        means = {u: np.mean(selected[u], axis=0) for u in contributing}
        result["mean"] = np.mean(list(means.values()), axis=0).tolist()
        if intervals and len(contributing) >= 10:
            # Resample complete user records, including nonresponsive users.
            # Recompute subset membership and its denominator in each replicate.
            rng = np.random.default_rng(42)
            # Resample every cohort user, then recompute the responsive-subset
            # denominator. Zero-filled noncontributors never enter that mean.
            values = np.zeros((len(users), *np.asarray(means[contributing[0]]).shape))
            eligible = np.zeros(len(users), dtype=bool)
            for index, user in enumerate(users):
                if user in means:
                    values[index] = means[user]
                    eligible[index] = True
            draw = rng.integers(len(users), size=(2000, len(users)))
            denominators = eligible[draw].sum(axis=1)
            defined = denominators > 0
            result["undefined_bootstrap_replicates"] = int((~defined).sum())
            if defined.any():
                totals = values[draw[defined]].sum(axis=1)
                divisor = denominators[defined].reshape(
                    (-1,) + (1,) * (values.ndim - 1)
                )
                boot = totals / divisor
                result["percentile_95_interval"] = np.quantile(
                    boot, [0.025, 0.975], axis=0
                ).T.tolist()
        else:
            result["interval_missing_reason"] = (
                "Fewer than ten contributing users or intervals disabled"
            )
        return result

    def _audit_dependencies(self):
        import ast
        import inspect
        import operator
        import textwrap
        from torch.fx import symbolic_trace

        actual = self.network
        expected = """
        def encode_sequence(self, seq: torch.Tensor) -> torch.Tensor:
            valid = seq != 0
            mask = valid.unsqueeze(-1).to(dtype=self.item_emb.weight.dtype)
            pos = valid.long().cumsum(dim=1) * valid.long()
            current_input_embedding = self.item_emb(seq) * mask
            x = current_input_embedding * (self.hidden_units**0.5) + self.pos_emb(pos)
            x = self.dropout(x * mask) * mask
            for block in self.blocks:
                x = block(x, mask, current_input_embedding)
            return self.norm(x) * mask
        """

        def syntax(source):
            return ast.dump(
                ast.parse(textwrap.dedent(source)), include_attributes=False
            )

        if syntax(inspect.getsource(type(actual).encode_sequence)) != syntax(expected):
            raise ValueError(
                "Functional encoder audit unavailable: unsupported encoder implementation"
            )
        if actual._forward_hooks or actual._forward_pre_hooks:
            raise ValueError(
                "Preexisting network hooks invalidate the dependency audit"
            )
        for module in actual.modules():
            if module._forward_hooks or module._forward_pre_hooks:
                raise ValueError(
                    "Preexisting module hooks invalidate the dependency audit"
                )
        if (
            type(actual.item_emb) is not torch.nn.Embedding
            or type(actual.pos_emb) is not torch.nn.Embedding
        ):
            raise ValueError("Unsupported embedding implementation")
        if type(
            actual.norm
        ) is not torch.nn.LayerNorm or actual.norm.normalized_shape != (
            actual.hidden_units,
        ):
            raise ValueError("Unsupported final normalization")
        if type(actual.dropout) is not torch.nn.Dropout:
            raise ValueError("Unsupported input dropout")
        blocks = list(actual.blocks)
        if len(blocks) < 2:
            raise ValueError(
                "A strictly intermediate boundary requires two residual blocks"
            )
        audited = []
        for block in blocks:
            if type(block).__name__ != "NextItNetBlock":
                raise ValueError("Unsupported residual block implementation")
            graph = symbolic_trace(block)
            deps = {}
            for node in graph.graph.nodes:
                args = list(node.all_input_nodes)
                incoming = set().union(*(deps[a] for a in args)) if args else set()
                if node.op == "placeholder":
                    if node.target not in ("x", "mask", "current_input_embedding"):
                        raise ValueError("Unknown residual input")
                    deps[node] = (
                        set() if node.target == "mask" else {(str(node.target), 0)}
                    )
                elif node.op == "get_attr":
                    deps[node] = set()
                elif node.op == "call_module":
                    module = graph.get_submodule(node.target)
                    if type(module) is torch.nn.Conv1d:
                        if (
                            module.stride != (1,)
                            or module.padding != (0,)
                            or module.padding_mode != "zeros"
                        ):
                            raise ValueError("Unsupported convolution indexing")
                        deps[node] = {
                            (source, offset - j * module.dilation[0])
                            for source, offset in incoming
                            for j in range(module.kernel_size[0])
                        }
                    elif type(module) in (
                        torch.nn.LayerNorm,
                        torch.nn.Linear,
                        torch.nn.Dropout,
                        torch.nn.GELU,
                    ):
                        if type(
                            module
                        ) is torch.nn.LayerNorm and module.normalized_shape != (
                            actual.hidden_units,
                        ):
                            raise ValueError("Normalization mixes temporal positions")
                        deps[node] = incoming
                    else:
                        raise ValueError(
                            "Unsupported residual module: " + type(module).__name__
                        )
                elif node.op == "call_function":
                    if node.target is F.pad:
                        padding = node.args[1]
                        if len(padding) != 2 or padding[1] != 0 or padding[0] < 0:
                            raise ValueError("Unsupported temporal padding")
                        if node.kwargs.get(
                            "mode", "constant"
                        ) != "constant" or node.kwargs.get("value", 0) not in (0, None):
                            raise ValueError("Unsupported padding value")
                        deps[node] = {(s, o + padding[0]) for s, o in incoming}
                    elif node.target in (
                        operator.add,
                        operator.mul,
                        operator.truediv,
                        operator.pow,
                        torch.sigmoid,
                        torch.cat,
                    ):
                        if node.target is torch.cat and node.kwargs.get(
                            "dim", 0
                        ) not in (-1, 2):
                            raise ValueError("Concatenation changes temporal indexing")
                        deps[node] = incoming
                    else:
                        raise ValueError(
                            "Unsupported functional operation: " + str(node.target)
                        )
                elif (
                    node.op == "call_method"
                    and node.target == "transpose"
                    and tuple(node.args[1:]) in ((1, 2), (2, 1))
                ):
                    deps[node] = incoming
                elif node.op == "output":
                    deps[node] = incoming
                else:
                    raise ValueError(
                        "Unsupported functional dependency node: " + str(node)
                    )
            offsets = deps[list(graph.graph.nodes)[-1]]
            if any(o < 0 for _, o in offsets):
                raise ValueError("Future dependency found")
            audited.append(
                {
                    "x": sorted(o for s, o in offsets if s == "x"),
                    "embedding": sorted(
                        o for s, o in offsets if s == "current_input_embedding"
                    ),
                    "graph": str(graph.graph),
                }
            )
        boundary = len(blocks) // 2 - 1
        upstream = {0}
        for paths in audited[: boundary + 1]:
            upstream = {a + b for a in upstream for b in paths["x"]} | set(
                paths["embedding"]
            )
        downstream = {0}
        for paths in audited[boundary + 1 :]:
            downstream = {a + b for a in downstream for b in paths["x"]}
        return boundary, upstream, downstream, audited

    def _write_manifest(self, manifest):
        import inspect
        import os

        try:
            root = os.path.dirname(os.path.abspath(inspect.getfile(type(self.network))))
            directory = os.path.join(root, "working")
            os.makedirs(directory, exist_ok=True)
            # An unknown category must not overwrite another category's replay.
            if self.category is None:
                raise ValueError("Explicit category route unavailable for replay file")
            path = os.path.join(directory, f"{self.category}_patch_manifest.json")
            with open(path, "w") as stream:
                json.dump(manifest, stream, indent=2, allow_nan=False)
            return {"written": True, "path": path}
        except (OSError, ValueError, TypeError) as exc:
            return {"written": False, "reason": f"{type(exc).__name__}: {exc}"}

    def run_full_diagnosis(self, train_loader):
        import os
        import time

        class BudgetReached(Exception):
            pass

        class PatchContractError(ValueError):
            pass

        actual = self.network
        device = actual.item_emb.weight.device
        modes = [(m, m.training) for m in actual.modules()]
        python_state, numpy_state = random.getstate(), np.random.get_state()
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        old = {
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        }
        handles, events, stage1, stage2, donor_rows = [], [], [], [], []
        counts, donor_reuse = Counter(), Counter()
        errors, site_failures, controls = [], {}, []
        sites, captured_donors = {}, {}
        atol, rtol, cap = 1e-6, 1e-5, 120.0
        kinds = ("interior_permutation",)
        extra_work = [True]
        longest_forward = [0.01]
        directions = ("restoration", "disruption")
        user_rng = np.random.default_rng(42)
        candidate_rng = np.random.default_rng(42)
        corruption_rng = np.random.default_rng(42)
        donor_rng = np.random.default_rng(42)

        def sync():
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        sync()
        started = time.perf_counter()

        def elapsed():
            sync()
            return time.perf_counter() - started

        def check_budget():
            if elapsed() >= cap - 2.0:
                raise BudgetReached(
                    "Cumulative category budget reached; reserve CPU reporting time"
                )

        def reserve_bundle(size):
            check_budget()
            counter = (
                "extra_forward_attempts" if extra_work[0] else "main_forward_attempts"
            )
            limit = 32 if extra_work[0] else 448
            if counts[counter] + size > limit:
                raise BudgetReached("Insufficient forward slots for a complete bundle")
            if elapsed() + size * max(0.02, 2.0 * longest_forward[0]) >= cap - 2.0:
                raise BudgetReached(
                    "Insufficient estimated budget for a complete condition bundle"
                )

        metrics = {
            "category": self.category,
            "status": "initializing",
            "scope": "Conditional training sensitivity; ModelView only",
            "missing_metadata": "Reviews, ratings, categories, prices, stores and simulator traits unavailable",
            "export_parity": {
                "available": False,
                "reason": "No extra checkpoint loaded by this diagnostic",
            },
            "four_category_macro_validation": None,
            "four_category_macro_reason": "Pinned host validation candidates and hit indicators are unavailable",
            "numerical_configuration": {
                "torch_version": str(torch.__version__),
                "cuda_version": torch.version.cuda,
                "device": str(device),
                "dtype": str(actual.item_emb.weight.dtype),
                "hardware": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else "CPU"
                ),
                "inherited_backend_policy": old,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "workspace_note": "Recorded only; this adapter cannot establish a pre-initialization CUDA setting",
                "requested": {
                    "autocast": False,
                    "tf32": False,
                    "deterministic_algorithms": True,
                },
                "candidate_shapes": [],
            },
        }
        manifest = {
            "category": self.category,
            "seed": 42,
            "max_batches": 10,
            "max_users": 32,
            "events_per_user": 1,
            "draws_per_kind": 3,
            "max_alternatives": 99,
            "protocol_version": "nextitnet_earlier_position_bidirectional_v3",
            "max_distinct_donors": 0,
            "main_forward_cap": 448,
            "extra_forward_cap": 32,
            "time_cap_seconds": cap,
            "identity_atol": atol,
            "identity_rtol": rtol,
            "gate": "Descriptive only; all eligible draws are patched at thresholds 0.1, 1, 10 times (atol + rtol*abs(clean))",
            "rng": "Separate local NumPy Generator(42) streams for users, candidates, corruptions, donors and bootstrap",
            "events": events,
            "controls": controls,
            "site_failures": site_failures,
            "stage1": stage1,
            "stage2": stage2,
            "donor_controls": donor_rows,
        }

        def require_close(a, b, label):
            if (
                a.shape != b.shape
                or not torch.isfinite(a).all()
                or not torch.isfinite(b).all()
            ):
                raise PatchContractError(label + ": invalid shapes or nonfinite values")
            discrepancy = float((a - b).abs().max().item())
            if not torch.allclose(a, b, atol=atol, rtol=rtol):
                raise PatchContractError(f"{label}: max_abs_error={discrepancy}")
            return discrepancy

        def forward(sequence, candidates, capture=False, patch=None):
            check_budget()
            counter = (
                "extra_forward_attempts" if extra_work[0] else "main_forward_attempts"
            )
            limit = 32 if extra_work[0] else 448
            if counts[counter] >= limit:
                raise BudgetReached(counter + " cap reached")
            counts[counter] += 1
            counts["forward_attempts"] += 1
            forward_started = time.perf_counter()
            seq = torch.as_tensor(sequence, dtype=torch.long, device=device)
            if seq.ndim == 1:
                seq = seq.unsqueeze(0)
            occupied = seq.ne(0)
            last = torch.arange(seq.shape[1], device=device).expand_as(seq)
            last = last.masked_fill(~occupied, -1).max(1).values
            if (last < 0).any():
                raise ValueError("Empty diagnostic sequence")
            cache, calls, local_handles = {}, Counter(), []

            def hook_for(name):
                def hook(module, args, output):
                    calls[name] += 1
                    if (
                        not isinstance(output, torch.Tensor)
                        or output.ndim != 3
                        or tuple(output.shape)
                        != (seq.shape[0], seq.shape[1], actual.hidden_units)
                        or output.dtype != torch.float32
                        or output.device != device
                        or not torch.isfinite(output).all()
                    ):
                        raise PatchContractError(
                            f"{name}: invalid residual-stream output"
                        )
                    if occupied.logical_not().any() and torch.count_nonzero(
                        output[~occupied]
                    ):
                        raise PatchContractError(f"{name}: nonzero padding activation")
                    if capture:
                        cache[name] = output.detach().clone()
                    if patch is None or patch[0] != name:
                        return None
                    donor = patch[1]
                    # None is an explicit no-op hook, independently checked.
                    result = output.clone()
                    if donor is not None:
                        positions = patch[2]
                        recipient = result[:, positions, :]
                        if (
                            donor.shape != recipient.shape
                            or donor.dtype != result.dtype
                            or donor.device != result.device
                            or not torch.isfinite(donor).all()
                        ):
                            raise PatchContractError(f"{name}: donor layout mismatch")
                        saved_donor = donor.clone()
                        result[:, positions, :] = donor
                        if not torch.equal(result[:, positions, :], saved_donor):
                            raise PatchContractError(
                                f"{name}: replacement verification failed"
                            )
                        if not torch.equal(donor, saved_donor):
                            raise PatchContractError(f"{name}: donor cache mutated")
                        if patch[3] == "deliberate" and torch.equal(
                            recipient, saved_donor
                        ):
                            raise PatchContractError(
                                f"{name}: deliberate alteration did not change the intended tensor"
                            )
                    return result

                return hook

            watched = list(sites) if capture else ([patch[0]] if patch else [])
            try:
                for name in watched:
                    handle = sites[name].register_forward_hook(hook_for(name))
                    handles.append(handle)
                    local_handles.append(handle)
                feats = self._features(seq)
                ids = torch.as_tensor(candidates, dtype=torch.long, device=feats.device)
                if (
                    ids.ndim != 1
                    or ids.numel() == 0
                    or ((ids < 1) | (ids > actual.itemnum)).any()
                    or ids.unique().numel() != ids.numel()
                ):
                    raise ValueError("Candidates must be unique valid catalog IDs")
                # Keep the adapter's sequence axis, but score only the original
                # readout. Avoid materializing B x L x N x H candidate lookups.
                rows = torch.arange(len(seq), device=feats.device)
                readout = feats[rows, last.to(feats.device)].unsqueeze(1)
                items = ids.view(1, 1, -1).expand(seq.shape[0], 1, -1)
                logits = self.view.predict_score(readout, items)
                if not isinstance(logits, torch.Tensor) or logits.shape != items.shape:
                    raise ValueError("ModelView scorer must return B x 1 x N scores")
                scores = logits[:, 0, :]
                if scores.dtype != torch.float32 or not torch.isfinite(scores).all():
                    raise ValueError("Expected finite FP32 candidate scores")
                if any(calls[name] != 1 for name in watched):
                    raise PatchContractError(
                        "Residual site did not execute exactly once"
                    )
                shape = list(items.shape)
                shapes = metrics["numerical_configuration"]["candidate_shapes"]
                if shape not in shapes:
                    shapes.append(shape)
                counts["forward_completed"] += 1
                return scores.detach().clone(), feats.detach(), cache
            finally:
                for handle in local_handles:
                    handle.remove()
                    handles.remove(handle)
                sync()
                longest_forward[0] = max(
                    longest_forward[0], time.perf_counter() - forward_started
                )

        def identity_events():
            vocab = int(actual.itemnum)
            source = self.user_train.get("History", self.user_train)
            if not isinstance(source, dict):
                raise ValueError("Training History must be a user-to-items mapping")
            histories = {}
            canonical_counts = Counter(str(user) for user in source)
            for user, values in source.items():
                if canonical_counts[str(user)] != 1:
                    counts["ambiguous_canonical_user_ids"] += 1
                    continue
                check_budget()
                try:
                    array = np.asarray(values)
                    if (
                        array.ndim != 1
                        or not np.isfinite(array).all()
                        or not np.equal(array, np.floor(array)).all()
                    ):
                        raise ValueError("Noninteger history")
                    history = array.astype(np.int64).tolist()
                    if any(i < 1 or i > vocab for i in history):
                        raise ValueError("Invalid history IDs")
                    if str(user) in histories:
                        raise ValueError("Ambiguous canonical user ID")
                    histories[str(user)] = history
                except (TypeError, ValueError):
                    counts["invalid_stored_history"] += 1
                    continue
            wanted = {}
            for _, batch in zip(range(10), train_loader):
                check_budget()
                counts["loader_batches"] += 1
                if not isinstance(batch, (tuple, list)) or len(batch) != 4:
                    counts["malformed_batches"] += 1
                    continue
                arrays = [torch.as_tensor(x).detach().cpu().numpy() for x in batch[1:]]
                seq, pos, neg = arrays
                if (
                    seq.ndim != 2
                    or seq.shape[1] == 0
                    or pos.shape != seq.shape
                    or neg.shape != seq.shape
                ):
                    counts["malformed_batches"] += 1
                    continue
                for s, p, n in zip(seq, pos, neg):
                    counts["loader_rows"] += 1
                    if any(
                        not np.isfinite(a).all() or not np.equal(a, np.floor(a)).all()
                        for a in (s, p, n)
                    ):
                        counts["noninteger_rows"] += 1
                        continue
                    s, p, n = s.astype(np.int64), p.astype(np.int64), n.astype(np.int64)
                    occupied = np.flatnonzero(s)
                    if not len(occupied):
                        counts["empty_rows"] += 1
                        continue
                    last = int(occupied[-1])
                    if np.any((s < 0) | (s > vocab)) or not np.array_equal(
                        occupied, np.arange(occupied[0], len(s))
                    ):
                        counts["invalid_or_non_left_padded_rows"] += 1
                        continue
                    if not 1 <= p[last] <= vocab:
                        counts["invalid_targets"] += 1
                        continue
                    key = (len(s), tuple(s[occupied].tolist()), int(p[last]))
                    if key in wanted:
                        counts["duplicate_encoded_rows"] += 1
                    else:
                        wanted[key] = (s.tolist(), int(n[last]), last)
            matches = {key: [] for key in wanted}
            widths = {}
            for width, encoded, target in wanted:
                widths.setdefault(target, set()).add(width)
            # Finish the entire identity audit before accepting any match.
            for user in sorted(histories):
                history = histories[user]
                for endpoint in range(1, len(history)):
                    if endpoint % 256 == 0:
                        check_budget()
                    for width in sorted(widths.get(history[endpoint], ())):
                        key = (
                            width,
                            tuple(history[max(0, endpoint - width) : endpoint]),
                            history[endpoint],
                        )
                        if key in matches and len(matches[key]) < 2:
                            matches[key].append((user, endpoint))
                check_budget()
            by_user = {}
            for key, found in matches.items():
                if len(found) != 1:
                    counts["ambiguous_events" if found else "unidentified_events"] += 1
                    continue
                user, endpoint = found[0]
                sequence, negative, last = wanted[key]
                prefix, target = histories[user][:endpoint], histories[user][endpoint]
                # Eligibility needs only a count. Materialize catalog pools only
                # for the at most 32 selected users below.
                if vocab - len(set(histories[user]) | {target}) == 0:
                    counts["no_alternative_events"] += 1
                    continue
                by_user.setdefault(user, []).append(
                    {
                        "user": user,
                        "endpoint": endpoint,
                        "sequence": sequence,
                        "prefix": prefix,
                        "observed_history": histories[user],
                        "target": target,
                        "negative": negative,
                        "last": last,
                        "length": len(key[1]),
                        "repeat": target in prefix,
                        "donor_pool_size": None,
                        "donors": [],
                        "donor_assignment_status": "pending",
                    }
                )
            users = sorted(by_user)
            counts["eligible_users"] = len(users)
            selected = user_rng.permutation(len(users))[:32]
            for selection_index, index in enumerate(selected.tolist()):
                options = sorted(by_user[users[index]], key=lambda e: e["endpoint"])
                event = options[int(user_rng.integers(len(options)))]
                pool = np.setdiff1d(
                    np.arange(1, vocab + 1),
                    list(set(event["observed_history"]) | {event["target"]}),
                )
                event["candidates"] = [event["target"]] + candidate_rng.choice(
                    pool, min(99, len(pool)), replace=False
                ).tolist()
                event["split"] = (
                    "confirmation"
                    if selection_index >= max(0, len(selected) - 8)
                    else "exploration"
                )
                event["candidate_source"] = (
                    "diagnostic_uniform_without_replacement; loader exposes only one negative"
                )
                event["negative_count"] = len(event["candidates"]) - 1
                event["candidate_shortage"] = 99 - event["negative_count"]
                event["degenerate_hr10"] = len(event["candidates"]) <= 10
                event["edits"] = []
                original = np.asarray(event["sequence"], dtype=np.int64)
                earlier = np.flatnonzero(original)[:-1]
                for draw in range(3):
                    permutation = corruption_rng.permutation(earlier)
                    changed = original.copy()
                    changed[earlier] = original[permutation]
                    event["edits"].append(
                        {
                            "kind": kinds[0],
                            "draw": draw,
                            "sequence": changed.tolist(),
                            "permutation_destination": earlier.tolist(),
                            "permutation_source": permutation.tolist(),
                            "changed_positions": np.flatnonzero(
                                changed != original
                            ).tolist(),
                            "identity_input": bool(np.array_equal(changed, original)),
                            "missing_reason": None,
                            "status": "pending",
                        }
                    )
                event["donor_assignment_status"] = "not_scheduled"
                events.append(event)

        try:
            actual.eval()
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
            if any(
                p.is_floating_point() and p.dtype != torch.float32
                for p in actual.parameters()
            ):
                raise ValueError("FP32 diagnostic requires FP32 checkpoint parameters")
            if type(actual).__name__ != "NextItNetModel":
                raise ValueError("Expected the supplied NextItNet model family")
            blocks = list(actual.blocks)
            boundary, upstream, downstream, graph = self._audit_dependencies()
            name = f"blocks.{boundary}"
            sites[name] = blocks[boundary]
            metrics["sites"] = [
                {
                    "name": name,
                    "index": boundary,
                    "invocation": 1,
                    "layout": "B x L x H",
                    "boundary": "post residual addition and padding mask",
                    "position": "largest eligible occupied index strictly earlier than readout",
                    "cropping": "none",
                    "remaining_blocks": len(blocks) - boundary - 1,
                }
            ]
            metrics["patching_missing_reason"] = None
            metrics["dependency_audit"] = {
                "bounded_future_invariance": "not_completed",
                "complete_functional_dependency_paths": "audited for supplied encoder and FX-supported residual operations, conditional on fixed occupancy masks",
                "upstream_offsets": sorted(upstream),
                "downstream_offsets": sorted(downstream),
                "block_graphs": graph,
                "whole_boundary_captures_all_suffix_inputs": False,
                "bypass": "Each downstream gate also receives original current_input_embedding; whole-output patches do not replace this bypass",
                "mask_scope": "Permutation and activation interventions preserve IDs' occupancy; cumsum positions and padding masks are fixed covariates",
            }
            manifest["dependency_audit"] = metrics["dependency_audit"]
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, enabled=False
            ):
                identity_events()
                lookup = {e["user"]: e for e in events}
                counts["selected_users"] = len(events)
                # Explicit short left-padding and future-invariance preflight.
                vocab, width = int(actual.itemnum), int(actual.maxlen)
                for length in sorted({1, min(2, width), width}):
                    check_budget()
                    sequence = [0] * (width - length) + [1] * length
                    candidates = list(range(1, min(vocab, 4) + 1))
                    expected, feats, _ = forward(sequence, candidates)
                    check_budget()
                    if counts["extra_forward_attempts"] >= 32:
                        raise BudgetReached("Extra-forward cap reached")
                    counts["extra_forward_attempts"] += 1
                    counts["forward_attempts"] += 1
                    deployed = actual.score_candidates(
                        torch.tensor([sequence], device=device),
                        torch.tensor(candidates, device=device),
                    ).reshape(1, -1)
                    sync()
                    error = require_close(expected, deployed, "original readout")
                    controls.append(
                        {"kind": "readout", "length": length, "max_abs_error": error}
                    )
                    batched, _, _ = forward([sequence, sequence], candidates)
                    error = require_close(
                        batched, expected.expand_as(batched), "batch consistency"
                    )
                    controls.append(
                        {
                            "kind": "batch_consistency",
                            "length": length,
                            "max_abs_error": error,
                        }
                    )
                if vocab >= 2 and width >= 2:
                    sequence = [1] * width
                    _, before, _ = forward(sequence, [1, 2])
                    midpoint = (width - 1) // 2
                    changed = sequence[: midpoint + 1] + [2] * (width - midpoint - 1)
                    _, after, _ = forward(changed, [1, 2])
                    error = require_close(
                        before[:, : midpoint + 1],
                        after[:, : midpoint + 1],
                        "future invariance",
                    )
                    controls.append(
                        {"kind": "bounded_future_invariance", "max_abs_error": error}
                    )
                    metrics["dependency_audit"]["bounded_future_invariance"] = "passed"
                else:
                    metrics["dependency_audit"][
                        "bounded_future_invariance"
                    ] = "unavailable: requires at least two items and two positions"
                metrics["status"] = "running"
                metrics["embedding_collapse_score"] = (
                    self.analyze_embeddings_collapse_cosine()
                )
                metrics["embedding_observations"] = self.embedding_sample_count
                metrics["helpers_run"] = [
                    "analyze_embeddings_collapse_cosine",
                    "_features",
                    "_outcome",
                    "_user_summary",
                ]
                metrics["legacy_margin_helper"] = (
                    "Retained callable; not run in this bounded patch protocol"
                )
                # Hash the actual frozen state; no other checkpoint is loaded.
                import hashlib

                digest = hashlib.sha256()
                for key, tensor in actual.state_dict().items():
                    check_budget()
                    digest.update(key.encode())
                    digest.update(str(tuple(tensor.shape)).encode())
                    digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
                manifest["checkpoint_state_sha256"] = digest.hexdigest()
                eligible_control_users = []
                for event in events:
                    occupied = np.flatnonzero(event["sequence"]).tolist()
                    r = event["last"]
                    possible = [
                        p
                        for p in occupied
                        if p < r
                        and r - p in downstream
                        and any(i < r and p - i in upstream for i in occupied)
                    ]
                    event["position"] = max(possible) if possible else None
                    event["eligibility_missing_reason"] = (
                        None
                        if possible
                        else "No audited earlier occupied site connecting input to readout"
                    )
                    if possible and len(eligible_control_users) < 2:
                        eligible_control_users.append(event["user"])
                manifest["extra_control_users"] = eligible_control_users
                extra_work[0] = False
                for event in events:
                    p = event["position"]
                    if p is None:
                        counts["site_ineligible_users"] += 1
                        continue
                    reserve_bundle(2)
                    candidates = event["candidates"]
                    clean, clean_feats, ca = forward(
                        event["sequence"], candidates, capture=True
                    )
                    same, same_feats, _ = forward(
                        event["sequence"],
                        candidates,
                        patch=(name, ca[name][:, [p], :].clone(), [p], "identity"),
                    )
                    clean_error = require_close(same, clean, "clean identity scores")
                    feature_error = require_close(
                        same_feats, clean_feats, "clean identity features"
                    )
                    controls.append(
                        {
                            "kind": "clean_identity",
                            "user": event["user"],
                            "score_error": clean_error,
                            "feature_error": feature_error,
                        }
                    )
                    baseline = self._outcome(clean, candidates)
                    event["clean_scores"] = clean.cpu().tolist()
                    for edit in event["edits"]:
                        reserve_bundle(4)
                        counts["patch_pairs_attempted"] += 1
                        corrupt, corrupt_feats, co = forward(
                            edit["sequence"], candidates, capture=True
                        )
                        identity, identity_feats, _ = forward(
                            edit["sequence"],
                            candidates,
                            patch=(name, co[name][:, [p], :].clone(), [p], "identity"),
                        )
                        score_error = require_close(
                            identity, corrupt, "corrupt identity scores"
                        )
                        feat_error = require_close(
                            identity_feats, corrupt_feats, "corrupt identity features"
                        )
                        restored, restored_feats, _ = forward(
                            edit["sequence"],
                            candidates,
                            patch=(
                                name,
                                ca[name][:, [p], :].clone(),
                                [p],
                                "restoration",
                            ),
                        )
                        disrupted, disrupted_feats, _ = forward(
                            event["sequence"],
                            candidates,
                            patch=(
                                name,
                                co[name][:, [p], :].clone(),
                                [p],
                                "disruption",
                            ),
                        )
                        altered = self._outcome(corrupt, candidates)
                        difference = (corrupt - clean).abs()
                        gates = {
                            str(scale): bool(
                                (difference > scale * (atol + rtol * clean.abs())).any()
                            )
                            for scale in (0.1, 1, 10)
                        }
                        reachable = any(
                            p - i in upstream for i in edit["changed_positions"]
                        )
                        common = {
                            "user": event["user"],
                            "repeat": event["repeat"],
                            "split": event["split"],
                            "negative_count": event["negative_count"],
                            "degenerate_hr10": event["degenerate_hr10"],
                            "kind": kinds[0],
                            "draw": edit["draw"],
                            "site": name,
                            "position": p,
                            "gate": gates["1"],
                            "gates": gates,
                            "reachable": reachable,
                            "identity_input": edit["identity_input"],
                        }
                        input_row = dict(
                            common,
                            clean=baseline.tolist(),
                            corrupt=altered.tolist(),
                            delta=(altered - baseline).tolist(),
                            corruption_gap=(baseline - altered).tolist(),
                            max_abs_score_difference=float(difference.max()),
                            tie_pairs=int(
                                torch.triu(
                                    corrupt[0, :, None] == corrupt[0, None, :],
                                    diagonal=1,
                                ).sum()
                            ),
                            corrupt_scores=corrupt.cpu().tolist(),
                        )
                        pending = []
                        for direction, patched, feats, recipient, reference_feats in (
                            (
                                "restoration",
                                restored,
                                restored_feats,
                                corrupt,
                                corrupt_feats,
                            ),
                            (
                                "disruption",
                                disrupted,
                                disrupted_feats,
                                clean,
                                clean_feats,
                            ),
                        ):
                            before, after = self._outcome(
                                recipient, candidates
                            ), self._outcome(patched, candidates)
                            pending.append(
                                dict(
                                    common,
                                    direction=direction,
                                    recipient=before.tolist(),
                                    patched=after.tolist(),
                                    delta=(after - before).tolist(),
                                    identity_max_abs_error=max(
                                        clean_error,
                                        score_error,
                                        feature_error,
                                        feat_error,
                                    ),
                                    max_abs_score_difference=float(
                                        (patched - recipient).abs().max()
                                    ),
                                    final_feature_distance=float(
                                        (
                                            feats[:, event["last"]]
                                            - reference_feats[:, event["last"]]
                                        ).norm()
                                    ),
                                    final_feature_norm=float(
                                        feats[:, event["last"]].norm()
                                    ),
                                    natural_clean_norm=float(
                                        clean_feats[:, event["last"]].norm()
                                    ),
                                    natural_corrupt_norm=float(
                                        corrupt_feats[:, event["last"]].norm()
                                    ),
                                )
                            )
                        if edit["identity_input"]:
                            require_close(corrupt, clean, "identity permutation scores")
                            require_close(
                                corrupt_feats,
                                clean_feats,
                                "identity permutation features",
                            )
                            # Input intervention is exactly unchanged; numerical identity
                            # tolerances do not redefine the zero input effect.
                            input_row["delta"] = [0.0, 0.0, 0.0]
                            input_row["corruption_gap"] = [0.0, 0.0, 0.0]
                        stage1.append(input_row)
                        stage2.extend(pending)
                        edit["status"] = "pair_complete"
                        counts["patch_pairs_completed"] += 1
                        counts["stage1_completed"] += 1
                        counts["gated_in" if gates["1"] else "gated_out"] += 1
                        # Additional controls are prespecified for the first two
                        # eligible users and their first realized nonidentity draw.
                        if (
                            event["user"] in eligible_control_users
                            and not edit["identity_input"]
                            and not event.get("controls_attempted")
                        ):
                            event["controls_attempted"] = True
                            extra_work[0] = True
                            try:
                                occupied = np.flatnonzero(event["sequence"]).tolist()
                                r = event["last"]
                                window = [
                                    i
                                    for i in occupied
                                    if i >= max(0, r - max(downstream))
                                ]
                                unreachable = [
                                    i for i in occupied if r - i not in downstream
                                ]
                                groups = [
                                    ("coherent_window", window),
                                    (
                                        "whole_output",
                                        list(range(len(event["sequence"]))),
                                    ),
                                ]
                                if unreachable:
                                    groups.append(
                                        ("unreachable_site", [max(unreachable)])
                                    )
                                else:
                                    controls.append(
                                        {
                                            "user": event["user"],
                                            "kind": "unreachable_site",
                                            "missing_reason": "Every occupied position can reach readout",
                                        }
                                    )
                                reserve_bundle(1 + 2 * len(groups))
                                if (
                                    counts["extra_forward_attempts"]
                                    + 1
                                    + 2 * len(groups)
                                    > 32
                                ):
                                    raise BudgetReached(
                                        "Insufficient remaining extra-forward slots"
                                    )
                                altered_vector = ca[name][:, [p], :].clone()
                                altered_vector[..., 0] += 1.0
                                forward(
                                    event["sequence"],
                                    candidates,
                                    patch=(name, altered_vector, [p], "deliberate"),
                                )
                                controls.append(
                                    {
                                        "user": event["user"],
                                        "kind": "deliberate_tensor_change",
                                        "passed": True,
                                    }
                                )
                                for label, positions in groups:
                                    for direction, sequence, donor, recipient in (
                                        (
                                            "restoration",
                                            edit["sequence"],
                                            ca[name],
                                            corrupt,
                                        ),
                                        (
                                            "disruption",
                                            event["sequence"],
                                            co[name],
                                            clean,
                                        ),
                                    ):
                                        patched, _, _ = forward(
                                            sequence,
                                            candidates,
                                            patch=(
                                                name,
                                                donor[:, positions, :].clone(),
                                                positions,
                                                label,
                                            ),
                                        )
                                        if label == "unreachable_site":
                                            require_close(
                                                patched,
                                                recipient,
                                                "unreachable control",
                                            )
                                        controls.append(
                                            {
                                                "user": event["user"],
                                                "draw": edit["draw"],
                                                "kind": label,
                                                "direction": direction,
                                                "positions": positions,
                                                "extra_unreachable_positions": [
                                                    i
                                                    for i in positions
                                                    if r - i not in downstream
                                                ],
                                                "delta": (
                                                    self._outcome(patched, candidates)
                                                    - self._outcome(
                                                        recipient, candidates
                                                    )
                                                ).tolist(),
                                                "donor_restoration_certificate": (
                                                    "unavailable: unpatched current-input-embedding bypass"
                                                    if label == "whole_output"
                                                    else None
                                                ),
                                            }
                                        )
                            except BudgetReached as exc:
                                controls.append(
                                    {
                                        "user": event["user"],
                                        "kind": "optional_controls",
                                        "missing_reason": str(exc),
                                    }
                                )
                            finally:
                                extra_work[0] = False
                metrics["status"] = "completed"
        except BudgetReached as exc:
            metrics["status"] = "budget_limited"
            errors.append(str(exc))
        except (ValueError, TypeError, AttributeError, RuntimeError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            # Stage 1 remains separately usable if only downstream patching failed.
            metrics["status"] = "contract_or_execution_failure"
            if metrics.get("status_before_failure") is None:
                metrics["status_before_failure"] = (
                    "stage1_available" if stage1 else "no_valid_input_effects"
                )
            if stage2 or donor_rows:
                for name in sites:
                    site_failures.setdefault(
                        name, "Patch execution interrupted by contract/runtime failure"
                    )
        finally:
            for handle in handles:
                handle.remove()
            handles.clear()
            # A CUDA execution error can surface during synchronization. Record
            # it without allowing timing to skip mode/backend/RNG restoration.
            try:
                metrics["elapsed_seconds"] = elapsed()
                metrics["timing_synchronized"] = True
            except RuntimeError as exc:
                metrics["elapsed_seconds"] = time.perf_counter() - started
                metrics["timing_synchronized"] = False
                errors.append(f"Final synchronization failed: {exc}")
                metrics["status"] = "contract_or_execution_failure"
                for name in sites:
                    site_failures.setdefault(name, "Final CUDA synchronization failed")
            metrics["budget_overrun_seconds"] = max(
                0.0, metrics["elapsed_seconds"] - cap
            )
            for module, mode in modes:
                module.training = mode
            torch.backends.cuda.matmul.allow_tf32 = old["matmul_tf32"]
            torch.backends.cudnn.allow_tf32 = old["cudnn_tf32"]
            torch.backends.cudnn.benchmark = old["benchmark"]
            torch.backends.cudnn.deterministic = old["cudnn_deterministic"]
            torch.use_deterministic_algorithms(
                old["deterministic"], warn_only=old["warn_only"]
            )
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state, device)

        complete_users = {
            e["user"]
            for e in events
            if all(d["status"] == "pair_complete" for d in e["edits"])
        }
        valid_stage1 = [r for r in stage1 if r["user"] in complete_users]
        valid_stage2 = [
            r
            for r in stage2
            if r["user"] in complete_users and r["site"] not in site_failures
        ]
        summaries, comparisons, donor_summaries = {}, {}, {}
        summary_started = time.perf_counter()
        predicates = {
            "all": lambda r: True,
            "reachable": lambda r: r["reachable"],
            "nonidentity": lambda r: not r["identity_input"],
            **{
                f"responsive_{scale}": (lambda r, s=str(scale): r["gates"][s])
                for scale in (0.1, 1, 10)
            },
        }
        # Bound reporting too: primary full-candidate, unconditional estimates first.
        for split in ("all", "exploration", "confirmation"):
            for candidate_group in (
                "99_negatives",
                "short_nondegenerate",
                "degenerate_hr10",
            ):
                for stratum in ("all", "repeat", "novel"):

                    def accepts(r):
                        return (
                            (split == "all" or r["split"] == split)
                            and (
                                stratum == "all" or r["repeat"] == (stratum == "repeat")
                            )
                            and (
                                (
                                    candidate_group == "99_negatives"
                                    and r["negative_count"] == 99
                                )
                                or (
                                    candidate_group == "short_nondegenerate"
                                    and 10 < r["negative_count"] + 1 < 100
                                )
                                or (
                                    candidate_group == "degenerate_hr10"
                                    and r["degenerate_hr10"]
                                )
                            )
                        )

                    populations = [
                        ("input", [r for r in valid_stage1 if accepts(r)], "delta")
                    ]
                    populations += [
                        (
                            direction,
                            [
                                r
                                for r in valid_stage2
                                if accepts(r) and r["direction"] == direction
                            ],
                            "delta",
                        )
                        for direction in directions
                    ]
                    for label, rows, field in populations:
                        for subset, predicate in predicates.items():
                            key = (
                                f"{split}/{candidate_group}/{stratum}/{label}/{subset}"
                            )
                            if time.perf_counter() - started >= cap:
                                summaries[key] = {
                                    "missing_reason": "Reporting budget exhausted"
                                }
                                continue
                            summaries[key] = self._user_summary(
                                rows, field, predicate=predicate
                            )
                            summaries[key]["full_scheduled_cohort_available"] = (
                                len(complete_users)
                                == sum(e.get("position") is not None for e in events)
                                and not site_failures
                            )
        metrics["score_difference_quantiles_0_10_50_90_100"] = (
            np.quantile(
                [r["max_abs_score_difference"] for r in valid_stage1],
                [0, 0.1, 0.5, 0.9, 1],
            ).tolist()
            if valid_stage1
            else None
        )
        metrics["corruption_gap_summary"] = self._user_summary(
            valid_stage1, "corruption_gap", intervals=False
        )
        metrics["complete_users"] = sorted(complete_users)
        metrics["split_sizes"] = dict(Counter(e["split"] for e in events))
        metrics["completed_split_sizes"] = dict(
            Counter(e["split"] for e in events if e["user"] in complete_users)
        )
        metrics["optional_control_coverage"] = [
            {
                "user": user,
                "attempted": bool(
                    next(e for e in events if e["user"] == user).get(
                        "controls_attempted"
                    )
                ),
                "missing_reason": (
                    None
                    if any(c.get("user") == user for c in controls)
                    else "No optional control completed; inspect pending draws and execution errors"
                ),
            }
            for user in manifest.get("extra_control_users", [])
        ]
        metrics["confirmation_missing_reason"] = (
            None
            if any(
                e["split"] == "confirmation" and e["user"] in complete_users
                for e in events
            )
            else "No complete confirmation users"
        )
        metrics["reporting_seconds"] = time.perf_counter() - summary_started
        pending = sum(
            e["status"] == "pending" for event in events for e in event["edits"]
        )
        metrics.update(
            {
                "summaries": summaries,
                "common_user_endpoint_minus_ordering": comparisons,
                "donor_summaries": donor_summaries,
                "controls": controls,
                "site_failures": site_failures,
                "errors": errors,
                "counts": dict(counts),
                "coverage": {
                    "selected_users": len(events),
                    "planned_corruptions": 3 * len(events),
                    "eligible_scheduled_corruptions": 3
                    * sum(e.get("position") is not None for e in events),
                    "completed_all_draw_users": len(complete_users),
                    "partial_user_records_excluded_from_summaries": len(stage1)
                    - len(valid_stage1),
                    "pending_input_conditions": pending,
                    "gated_site_direction_opportunities": 2
                    * sum(r["gate"] for r in stage1),
                    "unconditional_site_direction_opportunities": 6
                    * sum(e.get("position") is not None for e in events),
                    "valid_patch_direction_observations": len(valid_stage2),
                    "matched_donor_eligible_recipients": sum(
                        bool(e["donors"]) for e in events
                    ),
                    "matched_donor_total_recipients": len(events),
                    "pending_donor_assignments": sum(
                        e["donor_assignment_status"] != "completed" for e in events
                    ),
                    "invalidated_patch_direction_observations": len(stage2)
                    - len(valid_stage2),
                    "missing_patch_direction_observations": (
                        6 * sum(e.get("position") is not None for e in events)
                        - len(valid_stage2)
                    ),
                    "donor_assignments": {e["user"]: e["donors"] for e in events},
                    "realized_donor_pair_reuse_including_invalidated_sites": dict(
                        donor_reuse
                    ),
                },
            }
        )
        manifest["results"] = metrics
        manifest["incomplete_reason"] = (
            None if metrics["status"] == "completed" else "; ".join(errors)
        )
        definitions = {
            "scope": "One frozen category model, public training histories and targets only; targets may have influenced fitting. No additional scorer, checkpoint or ensemble is fitted or loaded.",
            "identity": "At most ten loader batches. Require integer B x L aligned tensors, valid IDs, nonempty contiguous left padding. Match encoded suffix plus aligned target against every training endpoint; reject ambiguous matches. Canonically sorted eligible user IDs are sampled without replacement, cap 32; one event per user, local seed 42.",
            "candidates": "Target plus up to 99 unique uniformly sampled alternatives, excluding padding, target and complete strictly pre-target history. Repeat targets retained. Same candidates across every condition; not host validation candidates. Supplied-negative margin is a separate two-item call and is unavailable for invalid or identical IDs.",
            "outcome_vector": "Every outcome/delta vector is [raw target-minus-best-alternative margin, one-based target rank, diagnostic HR@10]. Descending computed FP32 scores, then ascending integer item ID; exact equality defines ties. Positive margin/hit change is better; positive rank change is worse.",
            "stage1": "Corrupted minus clean on all completed eligible input draws, before numerical gating. Three draws/kind. Ordering uniformly permutes earlier occupied IDs, preserves endpoint and padding, requires two distinct earlier IDs and permits ten unchanged rejection attempts. Endpoint replacement is uniform over catalog IDs excluding original endpoint, candidates and complete pre-target history. Operational stress tests, not semantic counterfactuals.",
            "stage2": "Conditions on any candidate absolute score difference exceeding 1e-6 + 1e-5*abs(clean score). Sites floor((N-1)/3) and floor(2*(N-1)/3) must be distinct and strictly before final block. Patch only the original maximum occupied position of a cloned residual output; recompute downstream. Restoration: clean donor into corrupted recipient. Disruption: corrupted donor into clean recipient. Each delta is patched minus its own recipient. No final-representation localization claim.",
            "contracts": "Preflight finite FP32 features/scores, short-history original readout, duplicate-batch consistency, bounded future invariance, fixed-shape repetition. Capture hooks verify residual B x L x H layout, finite activations, zero padding and exactly one invocation. Each patch pair requires clean/corrupt no-op and identity-patch score agreement at atol=1e-6, rtol=1e-5. Failure invalidates all summaries at the affected site; stage 1 is retained separately. Complete dependency paths and exported predictor parity remain unverified.",
            "summaries": "Average realized draws within each user, then equally weight users. All/repeat/novel use original complete-prefix target repetition. Clean/corrupt and recipient/patched baselines are reported alongside changes, each on the same rows as its delta. Zero coverage is unavailable, not zero effect.",
            "common_user_endpoint_minus_ordering": "Endpoint-replacement user-mean delta minus ordering user-mean delta, restricted to their intersection of completed eligible users. Stage 2 intersections additionally condition on site, direction and numerical gate. No comparison of ungated and gated populations as interchangeable samples.",
            "intervals": "2000 paired user-bootstrap replicates, local seed 42, percentile 95% intervals; fewer than ten users gives no interval. Same user indices resample all vector components and paired differences. Conditional on checkpoint, recipients, candidates, gate and realized draws; excludes training, search and candidate/corruption sampling uncertainty. No simultaneous guarantees.",
            "donor_summaries": "Other selected users matched exactly on original encoded occupied length and original endpoint ID; no relaxed matches. Up to three distinct donors sampled uniformly without replacement, separate seed 42. Assignments reused across kinds/sites/directions when completed. During endpoint corruption, matching refers to the original clean endpoint. User-weighted descriptive means and pooled draw quantiles only; donor reuse prevents recipient-only intervals. Not an exchangeable permutation null or proof of in-distribution hybrids. Missing donors do not remove primary within-event results.",
            "coverage": "Counts distinguish planned, attempted, completed, unavailable, gated-out and invalidated work. Remaining pending conditions are missing under interruption. Replay retains input IDs, target endpoints, candidates, draws, donors, raw completed outcomes and errors, including invalidated records.",
            "numerics": "FP32 checkpoint required, autocast/TF32 disabled, deterministic algorithms requested and original backend settings restored. Existing CuBLAS workspace configuration is recorded, never established retroactively. Unsupported deterministic operations produce explicit failures; no automatic relaxed retry. Fixed-shape checks do not establish hardware/version/batch-shape invariance.",
            "budget": "Cumulative synchronized wall time capped at 120 seconds/category, including preparation and GPU work, checked before bounded forwards and during identity reconstruction. A running operation is not preempted; elapsed time and overrun reported. CPU summaries and manifest writing occur after GPU timing. All computation consumes the host search budget.",
            "interpretation": "Conditional training dependence only; no inferred intent, ownership, memorization, universal necessity/sufficiency, harmful stale interest, user causality or validation gain. Other temporal positions and direct embedding paths remain unpatched. Single checkpoint, training, architecture, score head, predictor writer, data splits and category routing are unchanged.",
        }
        # Replace obsolete staged/gated estimands; retain helper APIs but do not
        # describe the previous endpoint or independent-donor experiments as run.
        definitions.update(
            {
                "identity": "Cache at most ten batches; uniquely match encoded prefix plus aligned target against public training endpoints. Deterministic seed-42 shuffle of eligible user IDs, cap 32, one event each; last eight selected users reserved for confirmation. Selection does not inspect scores or patch results.",
                "candidates": "Diagnostic target plus up to 99 unique uniform alternatives excluding the target and every observed training-history item. Repeat targets are retained exactly once. IDs, shortage and degeneracy are in replay. These are not pinned host candidates; primary summaries require 99 negatives. At most ten candidates makes HR@10 degenerate.",
                "stage1": "Three independent permutations of earlier occupied positions, with endpoint, padding and earlier-item multiset fixed. No rejection sampling. Identity inputs remain scheduled and have exactly zero input-intervention effect. Input effects are corrupted minus clean; corruption gap is clean minus corrupted.",
                "stage2": "Every eligible draw is patched regardless of score responsiveness. One boundary after block floor(B/2)-1. Choose largest earlier occupied p with r-p in downstream offsets and p-i in upstream offsets for an earlier occupied input i. Patch the complete vector at p after residual addition/masking. Restoration is clean vector into corrupted input; disruption is corrupted vector into clean input. Each delta uses its own recipient baseline.",
                "dependencies": "Encoder AST must match the supplied fixed-occupancy implementation. FX audits residual functional operations; serial offsets use Minkowski sums, residual and multiplicative/gate paths use unions. Explicit left padding and valid convolution indexing are composed. Unknown operations or preexisting hooks produce unavailable measurements. Structural reachability includes actual changed IDs and does not guarantee score effects. Positional-cumsum dependencies are conditioned on unchanged occupancy.",
                "contracts": "Capture/patch hooks verify exactly one invocation, finite FP32 B x L x H layout, zero padding, donor integrity and exact replacement values. Clean and corrupted identity patches must reproduce features and candidate scores at atol=1e-6, rtol=1e-5. Runtime/contract failures invalidate previously collected site results. Readout, duplicate batch and bounded future checks have limited tested scope; export parity remains unavailable.",
                "summaries": "Only users completing all three scheduled paired draws enter summaries. Average eligible draws within user, then weight contributing users equally. Exploration, confirmation, repetition and candidate-count strata are separate. Responsive subsets are descriptive at 0.1/1/10 tolerance multipliers; all eligible draws are executed. Full scheduled-cohort estimates are unavailable under incomplete coverage; complete-user summaries may be selected by timeout.",
                "intervals": "2000 seed-42 paired user bootstrap replicates only with ten contributing users. Resample full completed user records, retaining nonresponsive users, and recompute subset denominators; report undefined empty-subset replicates. Eight confirmation users therefore have no interval. Conditional on checkpoint, users, candidates and realized draws; excludes training/search/candidate-draw uncertainty and simultaneous guarantees.",
                "controls": "First two structurally eligible users, first nonidentity draw: deliberate tensor alteration, bidirectional coherent occupied window and whole-output patches, plus an unreachable occupied site if one exists. Window is a contiguous superset of downstream-reachable positions. Whole-output patches do not capture direct current-input-embedding bypasses and are not certified donor-restoration controls. These users support descriptive comparisons only; hybrid-state norms/distances cannot certify distributional support.",
                "coverage": "Replay distinguishes selected, site-ineligible, pending, complete and partial users/draws; includes candidates, permutations, changed IDs, boundary graphs, sites, tolerances, frozen-state SHA256 and raw paired outcomes. Partial pairs never enter estimates. Exact alternative-alternative score ties are recorded per completed corrupted candidate vector; ranking always uses ascending item-ID ties.",
                "budget": "Synchronized 120-second cumulative category budget, 448 main forwards plus at most 32 extra forwards including preflight. Reserve estimated time before complete main bundles. CPU preparation, hashing and summaries count; reporting stops when the budget is exhausted. An in-flight operation/bootstrap/manifest write is not preempted; final elapsed time and overrun disclose this limitation. Optional incomplete controls are separate from main coverage.",
                "common_user_endpoint_minus_ordering": "Not scheduled: endpoint replacements are excluded from this protocol.",
                "donor_summaries": "Not scheduled: no independent-user donor experiment or additional model is used.",
                "embedding_collapse_score": "(mean off-diagonal normalized item-embedding cosine + 1)/2, at most 2048 nonpadding items, seed 42. Alignment only; not diversity or a dimensional-collapse certificate.",
            }
        )
        definitions["readout_scoring"] = (
            "ModelView receives B x 1 x H features gathered at the original maximum "
            "occupied index and B x 1 x N candidates. This preserves its sequence "
            "axis while avoiding candidate scoring at unused positions. Readout "
            "and duplicate-batch preflight checks gate subsequent measurements."
        )
        definitions["pending_donor_assignments"] = (
            "Independent donor assignments are not scheduled."
        )
        definitions["missing_patch_direction_observations"] = (
            "All structurally eligible scheduled site/direction opportunities minus valid complete-user observations, including incomplete and invalidated work; never interpreted as zero effects."
        )
        metrics["elapsed_seconds"] = time.perf_counter() - started
        metrics["budget_overrun_seconds"] = max(0.0, metrics["elapsed_seconds"] - cap)
        # Persist once after all summary fields and definitions are finalized.
        manifest["metric_definitions"] = definitions
        metrics["manifest"] = self._write_manifest(manifest)
        metrics["elapsed_seconds"] = time.perf_counter() - started
        metrics["budget_overrun_seconds"] = max(0.0, metrics["elapsed_seconds"] - cap)
        # Review summary: statically checked residual tensor dimensions, complete
        # bundle forward limits, bootstrap subset denominators, confirmation
        # coverage, unscheduled donor counts and optional-control reporting.
        # Vectorized bootstrap retains equal-user weighting and 2000 replicates.
        # No execution performed; runtime and validation gains remain unmeasured.
        # Training, checkpoints, predictor and all four routes are unchanged.
        return {"metrics": metrics, "metric_definitions": definitions}

    ### <<< Self_EvolveRec-BLOCK-END
