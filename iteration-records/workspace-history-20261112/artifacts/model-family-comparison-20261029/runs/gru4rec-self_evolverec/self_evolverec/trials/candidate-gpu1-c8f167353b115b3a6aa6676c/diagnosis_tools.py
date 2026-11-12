import torch
import numpy as np
import torch.nn.functional as F
import json

### >>> Self_EvolveRec-BLOCK-START: Persist bounded diagnostic draws without changing training RNG
import random
import os
import uuid
import copy
import io
from itertools import islice
from collections import Counter, defaultdict

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
    ### >>> Self_EvolveRec-BLOCK-START: Constrain interpretations to measured training-history reliance
    - Cite metric names, sample counts, missing evidence and available uncertainty intervals.
    - Treat failed correctness checks as CRITICAL; incomplete predictor parity is unresolved evidence.
    - Training margins measure sampled training separability, not validation accuracy.
    - Perturbations measure reliance on fixed observed histories and candidates. They do not measure current intent, compatibility, realistic behavior, causal effects or expected accuracy gains.
    - Do not infer unavailable ratings, reviews, categories, mood, ownership needs or simulator traits.
    - Embedding cosine similarity is a directional concentration statistic, not recommendation diversity.
    - Bootstrap intervals condition on fixed histories and candidates; they exclude training-seed and candidate-sampling uncertainty. Do not treat anonymous prefixes as independent users.
    - Mention prefix memorization, sparse strata and repeated-validation selection limits when relevant.
    - Do not recommend optimizing sensitivity itself. Any later intervention requires predeclared comparison through the unchanged host macro validation HR@10.
    - Use STABLE only when measured checks support it; explicitly acknowledge unavailable checks.
    ### <<< Self_EvolveRec-BLOCK-END
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Keep the host view and actual trained GRU module distinct
    def __init__(self, view, device):
        self.model = view
        self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train
        if not isinstance(view.model, torch.nn.Module):
            raise TypeError("view.model must be the actual trained nn.Module")

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Respect unavailable category metadata in the shared task
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

            ### >>> Self_EvolveRec-BLOCK-START: Report undefined embedding concentration instead of a healthy zero
            n = emb.size(0)
            if n < 2:
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            ### >>> Self_EvolveRec-BLOCK-START: Exclude zero-norm vectors from cosine diagnostics
            x = emb.index_select(0, idx)
            x = x[x.norm(dim=-1).gt(0)]
            k = x.size(0)
            if k < 2:
                return None
            x = F.normalize(x, p=2, dim=-1)
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Compute identical cosine concentration without a quadratic Gram matrix
            x = x.float()
            summed = x.sum(dim=0)
            offdiag_sum = summed.square().sum() - x.square().sum()
            offdiag_cnt = k * (k - 1)
            mean_cos = (offdiag_sum / offdiag_cnt).clamp(-1, 1).item()
            if not np.isfinite(mean_cos):
                raise RuntimeError("Nonfinite embedding concentration statistic")
            ### <<< Self_EvolveRec-BLOCK-END

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
        ### >>> Self_EvolveRec-BLOCK-START: Evaluate the actual trained module through the host view
        self.model.model.eval()
        ### <<< Self_EvolveRec-BLOCK-END

        margins_all = []
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        for step, batch in enumerate(dataloader):
            if step >= num_batches:
                break
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 4):
                continue

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            ### >>> Self_EvolveRec-BLOCK-START: Preserve GRU timestep alignment with tensor histories
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            log_feats = self.model.log2feats(seq_t)
            ### <<< Self_EvolveRec-BLOCK-END

            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)

            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
            margin = all_logits[..., 0] - all_logits[..., 1]

            ### >>> Self_EvolveRec-BLOCK-START: Validate aligned finite margins and exclude missing diagnostic negatives
            valid = seq_t.ne(0) & pos_t.ne(0) & neg_t.ne(0)
            if margin.shape != valid.shape:
                raise RuntimeError(
                    f"Margin shape {margin.shape} does not match aligned positions {valid.shape}"
                )
            if not bool(torch.isfinite(margin[valid]).all()):
                raise RuntimeError(
                    "Nonfinite positive/negative scores at valid training positions"
                )
            ### <<< Self_EvolveRec-BLOCK-END
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Apply the same validity mask to category counts
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
            margin_metrics = {
                ### >>> Self_EvolveRec-BLOCK-START: Keep undefined margin statistics explicit
                "mean": None,
                "p10": None,
                "neg_beats_pos_rate": None,
                "small_margin_rate": None,
                ### <<< Self_EvolveRec-BLOCK-END
            }
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

        breakdown = {
            ### >>> Self_EvolveRec-BLOCK-START: Expose margin support and metadata availability
            "eps": float(eps),
            "valid_positions": int(sum(m.numel() for m in margins_all)),
            "undefined_reason": (
                None if margins_all else "No valid nonpadding positive/negative pairs"
            ),
            "category_availability": "Unavailable in this shared task; category lists are empty",
            "low_margin_total_positions": int(low_total),
            ### <<< Self_EvolveRec-BLOCK-END
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

    ### >>> Self_EvolveRec-BLOCK-START: Validate GRU audit shapes and precision, bound identities, and accelerate paired summaries
    def _features(self, sequence):
        features = self.model.log2feats(sequence)
        expected = (*sequence.shape, self.model.item_emb.embedding_dim)
        if tuple(features.shape) != expected:
            raise RuntimeError(
                f"Unaligned GRU features: {features.shape}, expected {expected}"
            )
        if not torch.isfinite(features).all():
            raise RuntimeError("Nonfinite diagnostic features")
        return features

    def _last(self, sequence, features):
        positions = torch.arange(sequence.size(1), device=self.device).expand_as(
            sequence
        )
        final = positions.masked_fill(sequence.eq(0), -1).max(dim=1).values
        result = features[
            torch.arange(sequence.size(0), device=self.device), final.clamp(min=0)
        ]
        return result * final.ge(0).unsqueeze(-1)

    def _sequence(self, history, width):
        sequence = torch.zeros((1, width), dtype=torch.long, device=self.device)
        if history:
            sequence[0, -len(history) :] = torch.as_tensor(history, device=self.device)
        return sequence

    def _audit(self):
        module = self.model.model
        itemnum = int(module.itemnum)
        if itemnum < 1:
            raise RuntimeError("Cannot audit an empty item vocabulary")
        a, b = 1, min(2, itemnum)
        history = [a, b, a]
        wide = self._sequence(history, 10)
        reference = self._features(wide)
        checks = {}

        def check(name, actual, expected):
            if actual.shape != expected.shape:
                raise RuntimeError(
                    f"GRU correctness audit failed: {name}; "
                    f"shape mismatch {tuple(actual.shape)} versus {tuple(expected.shape)}"
                )
            if actual.dtype != torch.float32 or expected.dtype != torch.float32:
                raise RuntimeError(
                    f"GRU correctness audit requires FP32 outputs: {name}; "
                    f"received {actual.dtype} and {expected.dtype}. "
                    "Casting reduced-precision results would not constitute an FP32 audit."
                )
            actual, expected = actual.float(), expected.float()
            finite = bool(
                torch.isfinite(actual).all() and torch.isfinite(expected).all()
            )
            passed = finite and torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)
            checks[name] = {
                "passed": bool(passed),
                "max_abs_difference": (
                    float((actual - expected).abs().max()) if finite else None
                ),
            }
            if not passed:
                raise RuntimeError(
                    f"GRU correctness audit failed: {name}; {checks[name]}. "
                    "Repair training and predictor semantics together and retrain before validation."
                )

        for width in (3, 6, 10):
            seq = self._sequence(history, width)
            check(
                f"left_padding_width_{width}",
                self._features(seq)[seq.ne(0)],
                reference[wide.ne(0)],
            )
        check(
            "padding_features_zero",
            reference[wide.eq(0)],
            torch.zeros_like(reference[wide.eq(0)]),
        )
        empty = self._sequence([], 10)
        check("empty_history", self._features(empty), torch.zeros_like(reference))
        single = self._sequence([a], 10)
        check(
            "single_item_padding",
            self._features(single)[:, -1],
            self._features(single[:, -1:])[:, -1],
        )
        neighbor = self._sequence([b] * 7, 10)
        batch = torch.cat([wide, neighbor, empty, single], dim=0)
        batched = self._features(batch)
        individual = torch.cat(
            [self._features(row.unsqueeze(0)) for row in batch], dim=0
        )
        check("batched_vs_individual", batched, individual)
        permutation = torch.tensor([2, 0, 3, 1], device=self.device)
        check(
            "batch_permutation",
            self._features(batch[permutation]),
            batched[permutation],
        )
        changed = batch.clone()
        changed[1] = single[0]
        check("changed_neighbor", self._features(changed)[0], reference[0])
        self._features(neighbor)
        check("fresh_state_after_unrelated_batch", self._features(wide), reference)
        for end in range(1, len(history) + 1):
            prefix = self._sequence(history[:end], 10)
            check(
                f"timestep_alignment_{end}",
                reference[:, 6 + end],
                self._features(prefix)[:, -1],
            )
        candidates = torch.arange(1, min(itemnum, 7) + 1, device=self.device)
        final = self._last(wide, reference)[0]
        scores = self.model.predict_score(final, candidates)
        check("view_vs_module_features", reference, module.log2feats(wide))
        check(
            "view_vs_predictor_score_path",
            scores,
            module.score_candidates(wide, candidates),
        )
        # Reload the exact trained state without writing another category checkpoint.
        buffer = io.BytesIO()
        torch.save(module.state_dict(), buffer)
        buffer.seek(0)
        replica = copy.deepcopy(module).to(self.device).eval()
        try:
            replica.load_state_dict(
                torch.load(buffer, map_location=self.device, weights_only=True)
            )
            check(
                "checkpoint_roundtrip_score_parity",
                scores,
                replica.score_candidates(wide, candidates),
            )
        finally:
            del replica
            buffer.close()
        return {
            "status": "passed_module_checks_writer_parity_unverified",
            "rtol": 1e-5,
            "atol": 1e-6,
            "tolerance_adjustment": None,
            "checks": checks,
            "generated_predictor_parity": {
                "status": "unavailable",
                "reason": "The adapter has no host-provided callable for the generated checkpoint-loaded predictor. Module roundtrip parity does not verify routing or the writer.",
            },
        }

    def _collect_batches(self, loader):
        batches, identities = [], []
        users, counts = set(), Counter()
        # Optional sidecar: diagnostic_manifest[batch_index][row_index] contains
        # user_id and optionally prefix_id; the loader's four-element yield stays fixed.
        manifest = getattr(loader, "diagnostic_manifest", None)
        stop = False
        for step, batch in enumerate(islice(loader, 10)):
            if not isinstance(batch, (tuple, list)) or len(batch) != 4:
                raise ValueError(
                    "Diagnostic loader must yield (None, sequence, positive, negative)"
                )
            tensors = [
                torch.as_tensor(x, dtype=torch.long).detach().cpu() for x in batch[1:]
            ]
            if tensors[0].ndim != 2 or any(
                x.shape != tensors[0].shape for x in tensors
            ):
                raise ValueError("Diagnostic tensors must have matching B x L shapes")
            entries = manifest[step] if manifest is not None else None
            if entries is not None and len(entries) != tensors[0].size(0):
                raise ValueError(
                    "Diagnostic sampling manifest does not align with its batch"
                )
            selected, ids = [], []
            for row in range(tensors[0].size(0)):
                entry = entries[row] if entries is not None else {}
                uid = entry.get("user_id")
                uid = None if uid is None else str(uid)
                # Count each anonymous row as a distinct possible user. This
                # conservatively bounds mixed identified/anonymous samples too.
                represented_users = len(users) + counts["anonymous_rows"]
                if (uid is None or uid not in users) and represented_users >= 256:
                    stop = True
                    break
                if uid is not None:
                    if uid not in users and len(users) >= 256:
                        stop = True
                        break
                    users.add(uid)
                selected.append(row)
                ids.append(
                    {
                        "user_id": uid,
                        "prefix_id": str(
                            entry.get("prefix_id", f"batch_{step}_row_{row}")
                        ),
                        "batch_index": step,
                        "row_index": row,
                    }
                )
                counts["rows"] += 1
                counts["anonymous_rows"] += int(uid is None)
                if len(users) + counts["anonymous_rows"] >= 256:
                    stop = True
                    break
            if selected:
                batches.append((None, *(x[selected] for x in tensors)))
                identities.extend(ids)
            if stop:
                break
        counts["batches"] = len(batches)
        counts["known_distinct_users"] = len(users)
        counts["anonymous_rows"] = sum(x["user_id"] is None for x in identities)
        return batches, identities, dict(counts)

    def _evaluate_history(self, history, width, candidates, target):
        sequence = self._sequence(history, width)
        feature = self._last(sequence, self._features(sequence))[0]
        items = torch.as_tensor(candidates, dtype=torch.long, device=self.device)
        scores = (
            self.model.predict_score(feature, items)
            .detach()
            .float()
            .cpu()
            .numpy()
            .reshape(-1)
        )
        if scores.size != len(candidates) or not np.isfinite(scores).all():
            raise RuntimeError("Invalid diagnostic candidate scores")
        order = sorted(
            range(len(candidates)), key=lambda j: (-float(scores[j]), candidates[j])
        )
        ranked = [candidates[j] for j in order]
        target_index = candidates.index(target)
        target_score = float(scores[target_index])
        boundary_tie = None
        if len(order) > 10:
            boundary_tie = bool(scores[order[9]] == scores[order[10]])
        return feature, {
            "rank": ranked.index(target) + 1,
            "target_score": target_score,
            "scores": scores.tolist(),
            "ranking": ranked,
            "target_score_ties": int(np.count_nonzero(scores == target_score) - 1),
            "top10_boundary_tie": boundary_tie,
        }

    def _summarize(self, records):
        keys = ("delta_rank", "delta_target_score", "overlap_at_10", "state_distance")
        result = {"prefix_count": len(records)}
        known = bool(records) and all(r["user_id"] is not None for r in records)
        result["aggregation"] = (
            "user_means" if known else "prefix_means_without_user_inference"
        )
        result["user_count"] = len({r["user_id"] for r in records}) if known else None
        result["interval_limitation"] = (
            "Conditional on fixed histories/candidates; excludes training-seed and candidate-sampling uncertainty"
            if known
            else "Stable user identifiers unavailable; user-bootstrap intervals omitted"
        )
        for key in keys:
            grouped = defaultdict(list)
            for index, record in enumerate(records):
                value = record[key]
                if value is not None:
                    grouped[record["user_id"] if known else index].append(value)
            values = np.asarray([np.mean(v) for v in grouped.values()], dtype=float)
            interval = None
            if known and len(values) >= 2:
                rng = np.random.default_rng(42)
                draws = rng.choice(values, size=(1000, len(values)), replace=True).mean(
                    axis=1
                )
                interval = np.quantile(draws, [0.025, 0.975]).tolist()
            result[key] = {
                "mean": float(values.mean()) if len(values) else None,
                "valid_prefix_count": sum(r[key] is not None for r in records),
                "valid_aggregation_units": len(values),
                "user_bootstrap_95_percentile_interval": interval,
                "undefined_reason": (
                    "No defined observations"
                    if not len(values)
                    else (
                        "Fewer than two identified users; interval omitted"
                        if known and len(values) < 2
                        else None
                    )
                ),
            }
        return result

    def _perturb(self, batches, identities):
        rng = np.random.default_rng(42)
        histories = self.user_train.get("History", self.user_train)
        frequencies = Counter(
            int(item)
            for history in histories.values()
            for item in history
            if 1 <= int(item) <= int(self.model.model.itemnum)
        )
        catalog = sorted(frequencies)
        popularity_order = sorted(catalog, key=lambda item: (-frequencies[item], item))
        popularity = {
            item: ("head", "middle", "tail")[min(2, 3 * rank // max(1, len(catalog)))]
            for rank, item in enumerate(popularity_order)
        }
        counts, draws, paired = Counter(), [], []
        identity_index = 0
        for _, sequences, positives, _ in batches:
            width = sequences.size(1)
            for sequence, positive in zip(sequences, positives):
                identity = identities[identity_index]
                identity_index += 1
                positions = sequence.ne(0).nonzero(as_tuple=True)[0]
                if not positions.numel():
                    counts["empty_history"] += 1
                    continue
                target = int(positive[int(positions.max())])
                if target == 0:
                    counts["missing_target"] += 1
                    continue
                history = sequence[positions].tolist()
                if not 1 <= target <= int(self.model.model.itemnum):
                    raise ValueError("Diagnostic target outside the model vocabulary")
                exclusions = set(history) | {0, target}
                eligible = [item for item in catalog if item not in exclusions]
                negative_count = min(99, len(eligible))
                negatives = (
                    rng.choice(eligible, size=negative_count, replace=False).tolist()
                    if negative_count
                    else []
                )
                candidates = [target] + [int(item) for item in negatives]
                original_feature, original = self._evaluate_history(
                    history, width, candidates, target
                )
                n = len(history)
                length_bin = (
                    "1" if n == 1 else "2-5" if n <= 5 else "6-20" if n <= 20 else "21+"
                )
                strata = {
                    "history_length": length_bin,
                    "target_popularity": popularity.get(
                        target, "unobserved_in_training"
                    ),
                    "repeat_target": target in set(history),
                    "candidate_count": len(candidates),
                }
                entry = {
                    **identity,
                    "history": history,
                    "target": target,
                    "candidates": candidates,
                    "requested_negative_count": 99,
                    "actual_negative_count": negative_count,
                    "exclusions": sorted(exclusions),
                    "strata": strata,
                    "original": original,
                    "variants": {},
                }
                draws.append(entry)
                counts["valid_original_prefixes"] += 1
                counts["original_target_tie_prefixes"] += int(
                    original["target_score_ties"] > 0
                )
                counts["original_top10_boundary_tie_prefixes"] += int(
                    original["top10_boundary_tie"] is True
                )
                if n < 2:
                    counts["insufficient_history_for_deletion"] += 1
                    continue
                k = min(3, max(1, n // 4))
                older = history[:-1]
                shuffled = [
                    older[int(i)] for i in rng.permutation(len(older))
                ] + history[-1:]
                if len(older) < 2:
                    counts["shuffle_fewer_than_two_older_items"] += 1
                if shuffled == history:
                    counts["shuffle_no_op"] += 1
                    if len(older) >= 2 and len(set(older)) < len(older):
                        counts["shuffle_no_op_with_duplicate_older_items"] += 1
                variants = {
                    "remove_oldest": history[k:],
                    "remove_latest": history[:-k],
                    "shuffle_older": shuffled,
                }
                entry["deletion_count"] = k
                for name, items in variants.items():
                    feature, result = self._evaluate_history(
                        items, width, candidates, target
                    )
                    norm = original_feature.norm() * feature.norm()
                    distance = None
                    if float(norm) > 0:
                        cosine = (original_feature * feature).sum() / norm
                        distance = float(1.0 - cosine.clamp(-1, 1))
                    overlap = None
                    if len(candidates) >= 10:
                        overlap = (
                            len(
                                set(original["ranking"][:10])
                                & set(result["ranking"][:10])
                            )
                            / 10.0
                        )
                    record = {
                        **identity,
                        **strata,
                        "variant": name,
                        "delta_rank": result["rank"] - original["rank"],
                        "delta_target_score": result["target_score"]
                        - original["target_score"],
                        "overlap_at_10": overlap,
                        "state_distance": distance,
                        "no_op": items == history,
                    }
                    paired.append(record)
                    entry["variants"][name] = {
                        "history": items,
                        **result,
                        "paired_metrics": record,
                    }
                    counts[f"{name}_no_op"] += int(items == history)
                    counts[f"{name}_target_tie_prefixes"] += int(
                        result["target_score_ties"] > 0
                    )
                    counts[f"{name}_top10_boundary_tie_prefixes"] += int(
                        result["top10_boundary_tie"] is True
                    )
        summaries, joint = {}, []
        for name in ("remove_oldest", "remove_latest", "shuffle_older"):
            rows = [r for r in paired if r["variant"] == name]
            summaries[name] = self._summarize(rows)
            groups = defaultdict(list)
            for row in rows:
                key = tuple(
                    row[field]
                    for field in (
                        "history_length",
                        "target_popularity",
                        "repeat_target",
                        "candidate_count",
                    )
                )
                groups[key].append(row)
            for key, group in groups.items():
                joint.append(
                    {
                        "variant": name,
                        "history_length": key[0],
                        "target_popularity": key[1],
                        "repeat_target": key[2],
                        "candidate_count": key[3],
                        "summary": self._summarize(group),
                    }
                )
        return {
            "counts": dict(counts),
            "paired_category_means": summaries,
            "joint_strata": joint,
            "original_candidate_count_strata": dict(
                Counter(str(len(d["candidates"])) for d in draws)
            ),
            "original_repeat_target_strata": dict(
                Counter(str(d["strata"]["repeat_target"]) for d in draws)
            ),
            "training_catalog_size": len(catalog),
        }, draws

    def run_full_diagnosis(self, train_loader):
        module = self.model.model
        # Preserve mixed submodule modes as well as the root training flag.
        prior_modes = [(child, child.training) for child in module.modules()]
        python_state, numpy_state = random.getstate(), np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
            module.eval()
            with torch.no_grad(), torch.autocast(
                device_type=self.device.type, enabled=False
            ):
                audit = self._audit()
                batches, identities, sample_counts = self._collect_batches(train_loader)
                collapse_score = self.analyze_embeddings_collapse_cosine(
                    sample_size=2048
                )
                margin_metrics, breakdown = (
                    self.analyze_pos_neg_margin_with_category_breakdown(
                        batches, num_batches=10, eps=0.5, topk_categories=20
                    )
                )
                perturbation, draws = self._perturb(batches, identities)
            protocol = {
                "seed": 42,
                "sample_limits": {
                    "batches": 10,
                    "distinct_users": 256,
                    "anonymous_row_limit": 256,
                },
                "identity_source": "optional train_loader.diagnostic_manifest; unknown IDs remain null",
                "candidate_distribution": "uniform over observed training-catalog IDs after exclusions",
                "replacement": False,
                "requested_negatives": 99,
                "exclusions": "padding, target, and original history; frozen across all variants",
                "reserved_ids": [0],
                "target_inclusion": "exactly once, even for repeat targets",
                "state": "fresh zero recurrent state for each independent history",
                "ranking": "descending score, then ascending item ID",
                "bootstrap": {
                    "draws": 1000,
                    "seed": 42,
                    "unit": "identified user",
                    "interval": "95% percentile",
                },
                "strata": "length 1/2-5/6-20/21+; training-frequency rank thirds with ascending item-ID tie break; repeat target; candidate count",
                "unavailable_fields": [
                    "ratings",
                    "reviews",
                    "categories",
                    "prices",
                    "stores",
                    "simulator traits",
                ],
                "predictor_parity_limitation": audit["generated_predictor_parity"],
            }
            # The module-local working directory is beneath this trial directory.
            artifact_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "working", "diagnostics"
            )
            os.makedirs(artifact_dir, exist_ok=True)
            category = str(getattr(self.model, "dataset_name", "category_unreported"))
            safe_category = "".join(
                c if c.isalnum() or c in "_-" else "_" for c in category
            )
            artifact = os.path.join(
                artifact_dir, f"{safe_category}_{uuid.uuid4().hex}.json"
            )
            with open(artifact, "w") as handle:
                json.dump(
                    {
                        "category": category,
                        "protocol": protocol,
                        "audit": audit,
                        "sample_counts": sample_counts,
                        "sampling_manifest": identities,
                        "draws": draws,
                    },
                    handle,
                    indent=2,
                    allow_nan=False,
                )
            return {
                "metrics": {
                    "correctness_audit": audit,
                    "diagnostic_sample_counts": sample_counts,
                    "diagnostic_artifact": artifact,
                    "embedding_collapse_score": collapse_score,
                    "embedding_undefined_reason": (
                        "Fewer than two sampled nonzero-norm item embeddings"
                        if collapse_score is None
                        else None
                    ),
                    "pos_neg_margin_mean": margin_metrics["mean"],
                    "pos_neg_margin_p10": margin_metrics["p10"],
                    "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                    "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                    "pos_neg_margin_category_breakdown": breakdown,
                    "history_perturbations": perturbation,
                },
                "metric_definitions": {
                    "correctness_audit": "FP32 allclose at rtol=1e-5, atol=1e-6. Failures raise before reliance analysis. Serialization parity uses the same trained weights; actual generated-predictor parity remains explicitly unverified.",
                    "diagnostic_sample_counts": "Bounded training-loader sample; anonymous rows are capped at 256 and are not assumed to identify distinct users.",
                    "diagnostic_artifact": "Trial-local JSON with actual histories, manifest IDs or null, frozen candidates, exclusions, scores, rankings and protocol.",
                    "embedding_collapse_score": "(Mean off-diagonal cosine + 1)/2 over up to 2048 sampled nonzero-norm catalog embeddings, excluding padding. Directional concentration only; not recommendation diversity. Null when undefined.",
                    "pos_neg_margin_mean": "Mean positive-minus-negative score over aligned positions with nonzero input, positive and negative; training separability only.",
                    "pos_neg_margin_p10": "Nearest-rank lower 10th percentile of the same valid training margins; null without observations.",
                    "pos_neg_neg_beats_pos_rate": "Fraction of valid training margins strictly below zero.",
                    "pos_neg_small_margin_rate": "Fraction of valid training margins below eps=0.5.",
                    "pos_neg_margin_category_breakdown": "Valid-position and low/wrong-margin counts; category lists remain empty because categories are unavailable.",
                    "history_perturbations": {
                        "protocol": protocol,
                        "delta_rank": "Perturbed target rank minus original rank; rank is one plus the number preceding the target under deterministic ordering. Sampled diagnostic rank, not benchmark HR@10 or full-catalog position.",
                        "delta_target_score": "Perturbed target score minus original target score on identical candidates.",
                        "overlap_at_10": "Intersection size of original and perturbed top ten divided by ten; null for candidate pools smaller than ten.",
                        "state_distance": "One minus final-feature cosine similarity; null if either norm is zero.",
                        "deletion": "For n>=2, remove exactly k=min(3,max(1,n//4)) oldest versus latest observed items, retaining the same target.",
                        "shuffle": "Permute all but the latest input. Short older histories, unchanged shuffles and duplicate-containing unchanged shuffles are counted separately; no-ops remain in summaries.",
                        "ties": "Exact score equality; target tie counts exclude the target itself; boundary ties compare ranks ten and eleven and are undefined when no rank eleven exists.",
                        "aggregation": "Average prefixes within each identified user before category means and 1000 user-bootstrap draws. Anonymous samples receive descriptive prefix means without intervals. Each metric reports its own defined support.",
                        "limits": "Training-only reliance evidence subject to prefix memorization. Fixed-target perturbations may be implausible; shuffles may violate unobserved catalog availability. No intent, compatibility, causal or accuracy-gain claims. Sparse joint cells need caution. Repeat targets are separate from strict unseen-item simulator recommendations. The benchmark and its candidates are unchanged; repeated validation selection can overfit.",
                    },
                },
            }
        finally:
            for child, training in prior_modes:
                child.training = training
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    ### <<< Self_EvolveRec-BLOCK-END
