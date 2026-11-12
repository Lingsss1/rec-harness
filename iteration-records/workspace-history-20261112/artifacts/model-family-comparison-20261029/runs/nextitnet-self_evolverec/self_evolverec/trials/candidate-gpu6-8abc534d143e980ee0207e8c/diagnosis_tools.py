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
    - Separate ungated input sensitivity, numerically gated intermediate patches, and matched-donor coverage.
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

    ### >>> Self_EvolveRec-BLOCK-START: Bounded patching with isolated site failures, efficient readout scoring and reliable cleanup
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
    def _user_summary(rows, field="delta", intervals=True):
        grouped = {}
        for row in rows:
            grouped.setdefault(row["user"], []).append(row[field])
        users = sorted(grouped)
        result = {
            "users": len(users),
            "observations": len(rows),
            "user_ids": users,
            "mean": None,
            "percentile_95_interval": None,
            "interval_missing_reason": None,
        }
        if not users:
            result["missing_reason"] = "No completed eligible observations"
            result["interval_missing_reason"] = "No eligible users"
            return result
        values = np.stack([np.mean(grouped[u], axis=0) for u in users])
        result["mean"] = values.mean(0).tolist()
        if intervals and len(users) >= 10:
            rng = np.random.default_rng(42)
            indices = rng.integers(len(users), size=(2000, len(users)))
            result["percentile_95_interval"] = np.quantile(
                values[indices].mean(1), [0.025, 0.975], axis=0
            ).T.tolist()
        else:
            result["interval_missing_reason"] = (
                "Fewer than ten eligible users"
                if intervals
                else "Donor reuse dependence: descriptive results only"
            )
        return result

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
        kinds = ("interior_permutation", "endpoint_replacement")
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
            if elapsed() >= cap:
                raise BudgetReached("Cumulative 120-second category budget reached")

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
            "max_distinct_donors": 3,
            "time_cap_seconds": cap,
            "identity_atol": atol,
            "identity_rtol": rtol,
            "gate": "any(abs(corrupt-clean) > atol + rtol*abs(clean))",
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
            counts["forward_attempts"] += 1
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
                        or tuple(output.shape[:2]) != tuple(seq.shape)
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
                        rows = torch.arange(len(seq), device=device)
                        recipient = result[rows, last]
                        if (
                            donor.shape != recipient.shape
                            or donor.dtype != result.dtype
                            or donor.device != result.device
                            or not torch.isfinite(donor).all()
                        ):
                            raise PatchContractError(f"{name}: donor layout mismatch")
                        result[rows, last] = donor.clone()
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
                if vocab - len(set(prefix) | {target}) == 0:
                    counts["no_alternative_events"] += 1
                    continue
                by_user.setdefault(user, []).append(
                    {
                        "user": user,
                        "endpoint": endpoint,
                        "sequence": sequence,
                        "prefix": prefix,
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
            selected = user_rng.choice(len(users), min(32, len(users)), replace=False)
            for index in sorted(selected.tolist()):
                options = sorted(by_user[users[index]], key=lambda e: e["endpoint"])
                event = options[int(user_rng.integers(len(options)))]
                pool = np.setdiff1d(
                    np.arange(1, vocab + 1),
                    list(set(event["prefix"]) | {event["target"]}),
                )
                event["candidates"] = [event["target"]] + candidate_rng.choice(
                    pool, min(99, len(pool)), replace=False
                ).tolist()
                event["edits"] = []
                original = np.asarray(event["sequence"], dtype=np.int64)
                earlier = np.flatnonzero(original)[:-1]
                replacement_pool = np.setdiff1d(
                    np.arange(1, vocab + 1),
                    list(
                        set(event["prefix"])
                        | set(event["candidates"])
                        | {int(original[event["last"]])}
                    ),
                )
                for kind in kinds:
                    for draw in range(3):
                        changed, reason = original.copy(), None
                        if kind == "interior_permutation":
                            if len(np.unique(original[earlier])) < 2:
                                reason = "Fewer than two distinct earlier item IDs"
                            else:
                                for attempt in range(10):
                                    changed[earlier] = corruption_rng.permutation(
                                        original[earlier]
                                    )
                                    if not np.array_equal(changed, original):
                                        break
                                else:
                                    reason = "Ten permutation attempts were unchanged"
                        elif len(replacement_pool):
                            changed[event["last"]] = int(
                                corruption_rng.choice(replacement_pool)
                            )
                        else:
                            reason = "No eligible endpoint replacement"
                        event["edits"].append(
                            {
                                "kind": kind,
                                "draw": draw,
                                "sequence": (
                                    changed.tolist() if reason is None else None
                                ),
                                "missing_reason": reason,
                                "status": (
                                    "pending" if reason is None else "unavailable"
                                ),
                            }
                        )
                events.append(event)
            for event in events:
                pool = [
                    e["user"]
                    for e in events
                    if e["user"] != event["user"]
                    and e["length"] == event["length"]
                    and e["sequence"][e["last"]] == event["sequence"][event["last"]]
                ]
                event["donor_pool_size"] = len(pool)
                event["donors"] = donor_rng.choice(
                    sorted(pool), min(3, len(pool)), replace=False
                ).tolist()
                event["donor_assignment_status"] = "completed"

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
            blocks = list(getattr(actual, "blocks", []))
            indices = [(len(blocks) - 1) // 3, 2 * (len(blocks) - 1) // 3]
            if len(set(indices)) == 2 and all(
                0 <= i < len(blocks) - 1 for i in indices
            ):
                if all(type(blocks[i]).__name__ == "NextItNetBlock" for i in indices):
                    for i in indices:
                        sites[f"blocks.{i}"] = blocks[i]
                else:
                    # Unsupported patch sites must not prevent unhooked stage 1.
                    errors.append("Patching unavailable: unknown residual block layout")
            metrics["sites"] = [
                {
                    "name": name,
                    "index": int(name.split(".")[-1]),
                    "layout": "B x L x H",
                    "position": "original maximum nonzero index",
                    "remaining_blocks": len(blocks) - int(name.split(".")[-1]) - 1,
                }
                for name in sites
            ]
            metrics["patching_missing_reason"] = (
                None
                if sites
                else "Two distinct strictly intermediate residual sites unavailable"
            )
            metrics["dependency_audit"] = {
                "bounded_future_invariance": "not_completed",
                "residual_layout_and_mask_checks": "checked only on completed capture/patch calls",
                "complete_functional_dependency_paths": "unverified",
            }
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
                # Finish all inexpensive input conditions before any patching.
                clean_cache = {}
                for event in events:
                    candidates = event["candidates"]
                    clean, _, _ = forward(event["sequence"], candidates)
                    repeat, _, _ = forward(event["sequence"], candidates)
                    discrepancy = require_close(clean, repeat, "clean repeatability")
                    controls.append(
                        {
                            "kind": "repeatability",
                            "user": event["user"],
                            "max_abs_error": discrepancy,
                        }
                    )
                    clean_cache[event["user"]] = clean
                    negative = event["negative"]
                    if 1 <= negative <= vocab and negative != event["target"]:
                        pair, _, _ = forward(
                            event["sequence"], [event["target"], negative]
                        )
                        event["supplied_negative_margin"] = float(
                            pair[0, 0].double() - pair[0, 1].double()
                        )
                    else:
                        event["supplied_negative_margin"] = None
                        counts["invalid_supplied_negatives"] += 1
                    for edit in event["edits"]:
                        if edit["sequence"] is None:
                            counts["unavailable_corruptions"] += 1
                            continue
                        counts["stage1_attempted"] += 1
                        corrupt, _, _ = forward(edit["sequence"], candidates)
                        gated = bool(
                            ((corrupt - clean).abs() > atol + rtol * clean.abs())
                            .any()
                            .item()
                        )
                        baseline, altered = self._outcome(
                            clean, candidates
                        ), self._outcome(corrupt, candidates)
                        row = {
                            "user": event["user"],
                            "repeat": event["repeat"],
                            "kind": edit["kind"],
                            "draw": edit["draw"],
                            "gate": gated,
                            "clean": baseline.tolist(),
                            "corrupt": altered.tolist(),
                            "delta": (altered - baseline).tolist(),
                            "max_abs_score_difference": float(
                                (corrupt - clean).abs().max()
                            ),
                            "corrupt_scores": corrupt.cpu().tolist(),
                        }
                        stage1.append(row)
                        edit["status"] = "stage1_complete"
                        counts["stage1_completed"] += 1
                        counts["gated_in" if gated else "gated_out"] += 1
                for row in stage1:
                    if not row["gate"] or not sites:
                        continue
                    if all(name in site_failures for name in sites):
                        break
                    event = lookup[row["user"]]
                    edit = next(
                        e
                        for e in event["edits"]
                        if e["kind"] == row["kind"] and e["draw"] == row["draw"]
                    )
                    candidates, last = event["candidates"], event["last"]
                    clean = clean_cache[event["user"]]
                    corrupt = torch.tensor(
                        row["corrupt_scores"], device=device, dtype=torch.float32
                    )
                    # Capture hooks observe only; baseline stage 1 used no hooks.
                    try:
                        observed_clean, _, ca = forward(
                            event["sequence"], candidates, capture=True
                        )
                        observed_corrupt, _, co = forward(
                            edit["sequence"], candidates, capture=True
                        )
                        require_close(observed_clean, clean, "clean capture hook")
                        require_close(observed_corrupt, corrupt, "corrupt capture hook")
                    except PatchContractError as exc:
                        # A capture failure cannot safely identify a specific site.
                        for name in sites:
                            site_failures[name] = str(exc)
                        continue
                    for name in sites:
                        if name in site_failures:
                            continue
                        counts["patch_pairs_attempted"] += 1
                        try:
                            identity_errors = []
                            for sequence, base, activation in (
                                (event["sequence"], clean, ca[name]),
                                (edit["sequence"], corrupt, co[name]),
                            ):
                                noop, _, _ = forward(
                                    sequence, candidates, patch=(name, None)
                                )
                                same, _, _ = forward(
                                    sequence,
                                    candidates,
                                    patch=(name, activation[:, last]),
                                )
                                identity_errors.extend(
                                    [
                                        require_close(noop, base, "no-op hook"),
                                        require_close(same, base, "identity patch"),
                                    ]
                                )
                            restored, _, _ = forward(
                                edit["sequence"],
                                candidates,
                                patch=(name, ca[name][:, last]),
                            )
                            disrupted, _, _ = forward(
                                event["sequence"],
                                candidates,
                                patch=(name, co[name][:, last]),
                            )
                            pending = []
                            for direction, patched, base in (
                                ("restoration", restored, corrupt),
                                ("disruption", disrupted, clean),
                            ):
                                baseline = self._outcome(base, candidates)
                                result = self._outcome(patched, candidates)
                                pending.append(
                                    {
                                        "user": event["user"],
                                        "repeat": event["repeat"],
                                        "kind": row["kind"],
                                        "draw": row["draw"],
                                        "site": name,
                                        "direction": direction,
                                        "recipient": baseline.tolist(),
                                        "patched": result.tolist(),
                                        "delta": (result - baseline).tolist(),
                                        "identity_max_abs_error": max(identity_errors),
                                        "max_abs_score_difference": float(
                                            (patched - base).abs().max()
                                        ),
                                    }
                                )
                            stage2.extend(pending)
                            counts["patch_pairs_completed"] += 1
                            for donor_user in event["donors"]:
                                donor = lookup[donor_user]
                                if donor_user not in captured_donors:
                                    _, _, donor_acts = forward(
                                        donor["sequence"],
                                        donor["candidates"],
                                        capture=True,
                                    )
                                    captured_donors[donor_user] = donor_acts
                                vector = captured_donors[donor_user][name][
                                    :, donor["last"]
                                ]
                                donor_pending = []
                                for direction, sequence, base in (
                                    ("restoration", edit["sequence"], corrupt),
                                    ("disruption", event["sequence"], clean),
                                ):
                                    patched, _, _ = forward(
                                        sequence, candidates, patch=(name, vector)
                                    )
                                    donor_pending.append(
                                        {
                                            "user": event["user"],
                                            "donor": donor_user,
                                            "kind": row["kind"],
                                            "draw": row["draw"],
                                            "site": name,
                                            "direction": direction,
                                            "delta": (
                                                self._outcome(patched, candidates)
                                                - self._outcome(base, candidates)
                                            ).tolist(),
                                        }
                                    )
                                donor_rows.extend(donor_pending)
                                donor_reuse[donor_user] += 1
                        except PatchContractError as exc:
                            site_failures[name] = str(exc)
                    edit["status"] = "patch_processing_complete"
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

        # Invalidate every previously collected result at a failed patch site.
        valid_stage2 = [r for r in stage2 if r["site"] not in site_failures]
        valid_donors = [r for r in donor_rows if r["site"] not in site_failures]
        summaries = {}
        for stratum in ("all", "repeat", "novel"):
            accept = lambda r: stratum == "all" or r["repeat"] == (stratum == "repeat")
            for kind in kinds:
                rows = [r for r in stage1 if r["kind"] == kind and accept(r)]
                summaries[f"stage1/{stratum}/{kind}"] = {
                    field: self._user_summary(rows, field)
                    for field in ("clean", "corrupt", "delta")
                }
                for name in sites:
                    for direction in directions:
                        rows = [
                            r
                            for r in valid_stage2
                            if r["kind"] == kind
                            and r["site"] == name
                            and r["direction"] == direction
                            and accept(r)
                        ]
                        summaries[f"stage2/{stratum}/{kind}/{name}/{direction}"] = {
                            field: self._user_summary(rows, field)
                            for field in ("recipient", "patched", "delta")
                        }
        # Compare condition means only on the same users, with paired resampling.
        comparisons = {}
        for label, population in [("stage1", stage1)] + [
            (
                f"stage2/{name}/{direction}",
                [
                    r
                    for r in valid_stage2
                    if r["site"] == name and r["direction"] == direction
                ],
            )
            for name in sites
            for direction in directions
        ]:
            by_kind = {}
            for kind in kinds:
                grouped = {}
                for row in population:
                    if row["kind"] == kind:
                        grouped.setdefault(row["user"], []).append(row["delta"])
                by_kind[kind] = {u: np.mean(v, axis=0) for u, v in grouped.items()}
            common = sorted(set(by_kind[kinds[0]]) & set(by_kind[kinds[1]]))
            comparisons[label] = self._user_summary(
                [
                    {"user": u, "delta": by_kind[kinds[1]][u] - by_kind[kinds[0]][u]}
                    for u in common
                ]
            )
        donor_summaries = {}
        for kind in kinds:
            for name in sites:
                for direction in directions:
                    rows = [
                        r
                        for r in valid_donors
                        if r["kind"] == kind
                        and r["site"] == name
                        and r["direction"] == direction
                    ]
                    result = self._user_summary(rows, intervals=False)
                    result["distinct_donors"] = len({r["donor"] for r in rows})
                    result["draw_delta_quantiles_10_50_90"] = (
                        np.quantile(
                            [r["delta"] for r in rows], [0.1, 0.5, 0.9], axis=0
                        ).tolist()
                        if rows
                        else None
                    )
                    donor_summaries[f"{kind}/{name}/{direction}"] = result
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
                    "planned_corruptions": 6 * len(events),
                    "pending_input_conditions": pending,
                    "gated_site_direction_opportunities": 2
                    * len(sites)
                    * sum(r["gate"] for r in stage1),
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
                        2 * len(sites) * sum(r["gate"] for r in stage1)
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
        metrics["manifest"] = self._write_manifest(manifest)
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
        definitions["readout_scoring"] = (
            "ModelView receives B x 1 x H features gathered at the original maximum "
            "occupied index and B x 1 x N candidates. This preserves its sequence "
            "axis while avoiding candidate scoring at unused positions. Readout "
            "and duplicate-batch preflight checks gate subsequent measurements."
        )
        definitions["pending_donor_assignments"] = (
            "Recipients whose matching was interrupted before donor assignment. "
            "Their empty donor list indicates missing work, not an empty donor stratum."
        )
        definitions["missing_patch_direction_observations"] = (
            "Numerically gated site/direction opportunities minus valid completed "
            "observations; includes interrupted and invalidated work. Raw records "
            "remain in the replay manifest."
        )
        # Review summary: statically traced scoring shapes, left-padded readout,
        # interrupted donor initialization, ambiguous identities, site isolation,
        # hook cleanup and final synchronization failure handling. Training,
        # checkpoints and predictor architecture are unchanged. No execution or
        # validation improvement is claimed.
        return {"metrics": metrics, "metric_definitions": definitions}

    ### <<< Self_EvolveRec-BLOCK-END
