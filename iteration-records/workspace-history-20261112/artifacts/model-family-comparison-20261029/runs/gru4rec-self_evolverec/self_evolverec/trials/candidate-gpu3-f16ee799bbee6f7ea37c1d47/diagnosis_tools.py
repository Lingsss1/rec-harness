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
    ### >>> Self_EvolveRec-BLOCK-START: Constrain interpretation to measured training-prefix sensitivity
    - Every statement must be grounded in the provided metric values/definitions.
    - These are training-prefix diagnostics, not pinned validation HR@10 or evidence of improvement.
    - Distinguish composite deletion, final-item sensitivity, and context sensitivity with the final item preserved.
    - Inspect execution_audit, skipped counts, paired differences, and final-item flags before interpreting interventions.
    - If shortened_history_padding_confounded is true, withhold recency conclusions from deletion controls.
    - Final-item sensitivity alone does not establish a harmful shortcut or user intent.
    - Loader rows have no reliable user IDs; descriptive aggregates may overweight frequent users.
    - Null uncertainty intervals mean unavailable uncertainty, not zero uncertainty.
    - Do not infer ownership, compatibility, simulator traits, or unavailable metadata.
    - Embedding cosine geometry does not directly measure recommendation diversity.
    ### <<< Self_EvolveRec-BLOCK-END
    - If metrics look healthy, status should remain STABLE and implications should focus on "what is already working" plus one potential risk to monitor.
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve ModelView and prohibit unavailable category evidence
    def __init__(self, view, device):
        self.model = view
        self.device = device
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        # Category metadata is unavailable in this matched task.
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

            ### >>> Self_EvolveRec-BLOCK-START: Report unavailable embedding geometry without inventing a collapse value
            n = emb.size(0)
            if n < 2 or sample_size < 2:
                return None
            if not bool(torch.isfinite(emb).all()):
                raise RuntimeError(
                    "Nonfinite real-item embeddings encountered in diagnosis"
                )
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
        ### >>> Self_EvolveRec-BLOCK-START: Let the diagnostic runner manage actual module evaluation mode
        # run_full_diagnosis sets view.model.eval() and restores its original flag.
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

            ### >>> Self_EvolveRec-BLOCK-START: Validate IDs before embedding lookup and reject nonfinite margins
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or seq_t.size(1) == 0
                or seq_t.shape != pos_t.shape
                or seq_t.shape != neg_t.shape
            ):
                raise ValueError(
                    "Diagnostic loader must yield matching nonempty B x L tensors"
                )
            itemnum = int(self.model.model.itemnum)
            history_valid = ((seq_t >= 0) & (seq_t <= itemnum)).all(dim=1)
            valid = (
                history_valid[:, None]
                & (seq_t > 0)
                & (pos_t > 0)
                & (pos_t <= itemnum)
                & (neg_t > 0)
                & (neg_t <= itemnum)
                & (pos_t != neg_t)
            )
            # Invalid histories are excluded entirely. Safe placeholders prevent
            # invalid IDs from reaching embeddings before the validity mask applies.
            safe_seq = seq_t.masked_fill(~history_valid[:, None], 0)
            log_feats = self.model.log2feats(safe_seq.detach().cpu().numpy())
            if log_feats.ndim != 3 or tuple(log_feats.shape[:2]) != tuple(seq_t.shape):
                raise RuntimeError("ModelView.log2feats must return B x L x D features")
            all_items = torch.stack(
                [pos_t.masked_fill(~valid, 0), neg_t.masked_fill(~valid, 0)], dim=-1
            )
            all_logits = self.model.predict_score(log_feats, all_items)
            if tuple(all_logits.shape) != tuple(all_items.shape):
                raise RuntimeError(
                    "ModelView.predict_score must return B x L x 2 scores"
                )
            margin = all_logits[..., 0] - all_logits[..., 1]
            if not bool(torch.isfinite(margin[valid]).all()):
                raise RuntimeError(
                    "Nonfinite scores encountered in valid diagnostic training pairs"
                )
            ### <<< Self_EvolveRec-BLOCK-END
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Use the same validity mask for margin breakdown counts
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

        ### >>> Self_EvolveRec-BLOCK-START: Distinguish missing sampled-margin evidence from healthy zero metrics
        if len(margins_all) == 0:
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

        ### >>> Self_EvolveRec-BLOCK-START: Count the exact observations used by margin summaries
        breakdown = {
            "valid_positions": sum(int(values.numel()) for values in margins_all),
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

    ### >>> Self_EvolveRec-BLOCK-START: Validated matched GRU interventions with padding audits and exact observation counts
    def _replay(self, histories, width, candidates):
        """Use ModelView's sequential scoring contract without changing recurrence."""
        if width <= 0 or not histories:
            raise ValueError("Replay requires positive width and at least one history")
        itemnum = int(self.model.model.itemnum)
        if len(candidates) != 2 or not all(1 <= int(x) <= itemnum for x in candidates):
            raise ValueError("Replay requires two real-item candidate IDs")
        if int(candidates[0]) == int(candidates[1]):
            raise ValueError("Replay candidates must be distinct")
        sequence = np.zeros((len(histories), width), dtype=np.int64)
        for row, history in enumerate(histories):
            if not 0 < len(history) <= width:
                raise ValueError(
                    "Replay history must be nonempty and fit the external width"
                )
            if not all(1 <= int(x) <= itemnum for x in history):
                raise ValueError("Replay history contains an invalid real-item ID")
            sequence[row, -len(history) :] = history
        features = self.model.log2feats(sequence)
        if features.ndim != 3 or tuple(features.shape[:2]) != sequence.shape:
            raise RuntimeError("ModelView.log2feats must return B x L x D features")
        # External histories remain LEFT padded. Never gather at count-1.
        positions = torch.arange(width, device=features.device).expand(
            len(histories), -1
        )
        occupied = torch.as_tensor(sequence != 0, device=features.device)
        last = positions.masked_fill(~occupied, -1).max(dim=1).values
        if (last < 0).any():
            raise RuntimeError("Empty diagnostic history")
        states = features[torch.arange(len(histories), device=features.device), last]
        # DEBUG: Score through the same B x L x D / B x L x 2 contract
        # used by the working margin probe. A singleton sequence dimension
        # triggers incompatible ModelView broadcasting; reshaping cannot fix it.
        score_features = features
        if width == 1:
            # DEBUG: Duplicate already-computed features for scoring only.
            # Do not append input items or change the GRU recurrence/state.
            score_features = features.expand(-1, 2, -1)
        items = torch.as_tensor(candidates, device=features.device, dtype=torch.long)
        items = items.view(1, 1, 2).expand(len(histories), score_features.size(1), 2)
        all_scores = self.model.predict_score(score_features, items)
        if tuple(all_scores.shape) != tuple(items.shape):
            raise RuntimeError(
                "ModelView.predict_score must return B x L x 2 scores; "
                f"received {tuple(all_scores.shape)} for "
                f"features {tuple(score_features.shape)} and items {tuple(items.shape)}"
            )
        # DEBUG: Gather the actual final nonpadding position after scoring,
        # preserving left-padding semantics and the actual model scorer.
        scores = all_scores[torch.arange(len(histories), device=features.device), last]
        return states.detach().float(), scores.detach().float()

    def _history_panel(self, batches):
        rng = np.random.default_rng(42)
        counts = Counter()
        observations = {}
        paired = {"recent_nonfinal_minus_earlier": [], "final_minus_earlier": []}
        audit = {
            "model_class": type(self.model.model).__name__,
            "recurrent_module": type(getattr(self.model.model, "gru", None)).__name__,
            "initial_state_contract": (
                "Supplied GRU4Rec calls gru(emb) without hx on every invocation: "
                "implicit zero initial state, with no carried recurrent state. "
                "The diagnostic does not modify recurrence."
            ),
            "supplied_recurrence": (
                "The supplied baseline runs an ordinary GRU over leading zero embeddings; "
                "padding_idx=0 does not skip GRU updates."
            ),
            "repeated_call_max_abs_score_difference": None,
            "batch_permutation_max_abs_score_difference": None,
            "batched_vs_single_max_abs_score_difference": None,
            "unpadded_max_abs_score_difference": None,
            "unpadded_max_abs_state_difference": None,
            "shortened_history_padding_confounded": True,
            "exported_predictor_score_agreement": None,
            "exported_predictor_audit_reason": (
                "No exported predictor callable is supplied to DiagnosisProbe; "
                "training/export agreement cannot be measured here."
            ),
            "sampling": "First up to ten yielded batches; loader order is not controlled by this probe.",
            "perturbation_seed": 42,
            "random_control_draws": 5,
            "audit_observations": 0,
        }
        audit_done = False
        itemnum = int(self.model.model.itemnum)
        for _, sequence, positive, negative in batches:
            seq = torch.as_tensor(sequence).cpu().numpy()
            pos = torch.as_tensor(positive).cpu().numpy()
            neg = torch.as_tensor(negative).cpu().numpy()
            if (
                seq.ndim != 2
                or seq.shape[1] == 0
                or seq.shape != pos.shape
                or seq.shape != neg.shape
            ):
                raise ValueError(
                    "Diagnostic loader must yield matching nonempty B x L tensors"
                )
            width = seq.shape[1]
            for row in range(seq.shape[0]):
                counts["rows_seen"] += 1
                valid_positions = np.flatnonzero((seq[row] > 0) & (pos[row] > 0))
                if not len(valid_positions):
                    counts["skipped_no_valid_position"] += 1
                    continue
                t = int(valid_positions[-1])
                prefix = seq[row, : t + 1]
                history = prefix[prefix != 0].astype(np.int64).tolist()
                if any(x < 1 or x > itemnum for x in history):
                    counts["skipped_invalid_history"] += 1
                    continue
                candidates = [int(pos[row, t]), int(neg[row, t])]
                if not all(1 <= x <= itemnum for x in candidates):
                    counts["skipped_invalid_candidates"] += 1
                    continue
                if candidates[0] == candidates[1]:
                    counts["skipped_identical_candidates"] += 1
                    continue
                n = len(history)
                if n < 4:
                    counts["skipped_short_history"] += 1
                    continue
                counts["eligible_rows"] += 1
                k = min(3, (n - 1) // 3)
                indices = list(range(n))
                variants = []

                def add(name, removed):
                    retained = [i for i in indices if i not in removed]
                    variants.append((name, retained, [history[i] for i in retained]))

                add("original", set())
                add("composite_recent", set(range(n - k, n)))
                add("composite_old", set(range(k)))
                add("recent_nonfinal", set(range(n - 1 - k, n - 1)))
                # These removal regions are disjoint and both preserve the final item.
                for _draw in range(5):
                    removed = rng.choice(n - 1 - k, size=k, replace=False).tolist()
                    add("earlier_matched", set(removed))
                add("final_deleted", {n - 1})
                for _draw in range(5):
                    add("earlier_single", {int(rng.integers(n - 1))})
                add("final_only", set(range(n - 1)))
                seen_lengths = set()
                for requested in (2, 4, 8, 16, n):
                    length = min(requested, n)
                    if length in seen_lengths:
                        continue
                    seen_lengths.add(length)
                    # Name the actual retained length. A clipped full history
                    # must not be pooled with genuinely longer suffixes.
                    name = "suffix_full" if length == n else f"suffix_{length}"
                    add(name, set(range(n - length)))
                states, scores = self._replay(
                    [v[2] for v in variants], width, candidates
                )
                if not bool(
                    torch.isfinite(states).all() and torch.isfinite(scores).all()
                ):
                    counts["skipped_nonfinite_replay"] += 1
                    continue
                if not audit_done:
                    original = [history]
                    state1, score1 = self._replay(original, width, candidates)
                    _, score2 = self._replay(original, width, candidates)
                    unpad_state, unpad_score = self._replay(original, n, candidates)
                    _, reverse_scores = self._replay(
                        [v[2] for v in reversed(variants)], width, candidates
                    )
                    audit.update(
                        {
                            "repeated_call_max_abs_score_difference": float(
                                (score1 - score2).abs().max().item()
                            ),
                            "batch_permutation_max_abs_score_difference": float(
                                (scores - reverse_scores.flip(0)).abs().max().item()
                            ),
                            "batched_vs_single_max_abs_score_difference": float(
                                (scores[:1] - score1).abs().max().item()
                            ),
                            "unpadded_max_abs_score_difference": float(
                                (score1 - unpad_score).abs().max().item()
                            ),
                            "unpadded_max_abs_state_difference": float(
                                (state1 - unpad_state).abs().max().item()
                            ),
                            "audit_observations": 1,
                            "original_audit_history_length": n,
                            "original_external_width": width,
                        }
                    )
                    # Even observed invariance at one prefix does not establish padding-free recurrence.
                    audit_done = True
                wins = (scores[:, 0] > scores[:, 1]).float()
                wins += 0.5 * (scores[:, 0] == scores[:, 1]).float()
                original_win = float(wins[0].item())
                grouped = {}
                original_norm = states[0].norm()
                for index, (name, retained, retained_history) in enumerate(variants):
                    group = grouped.setdefault(
                        name,
                        {
                            "wins": [],
                            "distances": [],
                            "zero": 0,
                            "final_retained": [],
                            "final_id_changed": [],
                        },
                    )
                    group["wins"].append(float(wins[index].item()))
                    group["final_retained"].append(float(n - 1 in retained))
                    group["final_id_changed"].append(
                        float(retained_history[-1] != history[-1])
                    )
                    if (
                        float(original_norm.item()) == 0.0
                        or float(states[index].norm().item()) == 0.0
                    ):
                        group["zero"] += 1
                    else:
                        # Normalize nonzero states explicitly: cosine_similarity's
                        # default epsilon can distort very small, nonzero vectors.
                        base = states[0].double()
                        current = states[index].double()
                        cosine = torch.dot(
                            base / torch.linalg.vector_norm(base),
                            current / torch.linalg.vector_norm(current),
                        ).clamp(-1.0, 1.0)
                        distance = 1.0 - float(cosine.item())
                        group["distances"].append(float(distance))
                per_row = {}
                for name, group in grouped.items():
                    record = observations.setdefault(
                        name,
                        {
                            "wins": [],
                            "deltas": [],
                            "cosine": [],
                            "zero_norm_replays": 0,
                            "final_retained": [],
                            "final_id_changed": [],
                        },
                    )
                    win = float(np.mean(group["wins"]))
                    per_row[name] = win
                    record["wins"].append(win)
                    record["deltas"].append(win - original_win)
                    if group["distances"]:
                        record["cosine"].append(float(np.mean(group["distances"])))
                    record["zero_norm_replays"] += group["zero"]
                    record["final_retained"].append(
                        float(np.mean(group["final_retained"]))
                    )
                    record["final_id_changed"].append(
                        float(np.mean(group["final_id_changed"]))
                    )
                paired["recent_nonfinal_minus_earlier"].append(
                    per_row["recent_nonfinal"] - per_row["earlier_matched"]
                )
                paired["final_minus_earlier"].append(
                    per_row["final_deleted"] - per_row["earlier_single"]
                )
                counts["analyzed_rows"] += 1

        def mean(values):
            return float(np.mean(values)) if values else None

        summaries = {}
        for name, record in observations.items():
            summaries[name] = {
                "observations": len(record["wins"]),
                "mean_pairwise_win": mean(record["wins"]),
                "mean_win_minus_original": mean(record["deltas"]),
                "mean_one_minus_state_cosine": mean(record["cosine"]),
                "cosine_valid_observations": len(record["cosine"]),
                "zero_norm_replays": record["zero_norm_replays"],
                "final_position_retained_fraction": mean(record["final_retained"]),
                "final_item_id_changed_fraction": mean(record["final_id_changed"]),
            }
        audit["recency_conclusions_withheld"] = True
        audit["padding_confound_reason"] = (
            "Supplied ordinary leading-padding GRU updates make deletion change both real context "
            "and the number of initial padding updates. One unpadded replay is an execution check, "
            "not proof of padding invariance. No recurrence correction or retraining is performed."
        )
        return {
            "counts": dict(counts),
            "conditions": summaries,
            "paired_differences": {
                name: {"mean": mean(values), "observations": len(values)}
                for name, values in paired.items()
            },
            "execution_audit": audit,
            "user_bootstrap_macro_95_intervals": None,
            "uncertainty_unavailable_reason": (
                "Loader supplies no reliable user IDs, and this probe has no explicitly user-indexed "
                "observations from all four categories. Rows are not bootstrapped as independent users. "
                "Aggregates are descriptive and may overweight frequent users."
            ),
        }

    def run_full_diagnosis(self, train_loader):
        module = self.model.model
        saved_training = module.training
        try:
            module.eval()
            with torch.inference_mode():
                # Cache the same bounded panel for margins and perturbations, including one-shot loaders.
                batches = []
                iterator = iter(train_loader)
                for _ in range(10):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        break
                    if not isinstance(batch, (list, tuple)) or len(batch) < 4:
                        raise ValueError(
                            "Expected (None, sequence, positive, negative)"
                        )
                    batches.append(
                        (None,)
                        + tuple(
                            torch.as_tensor(x).detach().cpu().clone()
                            for x in batch[1:4]
                        )
                    )
                collapse_score = self.analyze_embeddings_collapse_cosine(
                    sample_size=2048
                )
                margin_metrics, breakdown = (
                    self.analyze_pos_neg_margin_with_category_breakdown(
                        batches, num_batches=10, eps=0.5, topk_categories=20
                    )
                )
                breakdown["unavailable_reason"] = (
                    "Item category metadata is unavailable in this task."
                )
                panel = self._history_panel(batches)
                return {
                    "metrics": {
                        "embedding_collapse_score": collapse_score,
                        "pos_neg_margin_mean": margin_metrics["mean"],
                        "pos_neg_margin_p10": margin_metrics["p10"],
                        "pos_neg_neg_beats_pos_rate": margin_metrics[
                            "neg_beats_pos_rate"
                        ],
                        "pos_neg_small_margin_rate": margin_metrics[
                            "small_margin_rate"
                        ],
                        "pos_neg_margin_category_breakdown": breakdown,
                        "margin_valid_positions": breakdown["valid_positions"],
                        "history_sensitivity": panel,
                    },
                    "metric_definitions": {
                        "embedding_collapse_score": (
                            "(Mean off-diagonal cosine of up to 2048 seed-42 sampled real-item embeddings + 1)/2. "
                            "Null means fewer than two sampled real-item embeddings. "
                            "0.5 corresponds to mean cosine zero. Directional alignment alone does not establish "
                            "collapse, poor recommendation diversity, or prediction quality; zero vectors follow F.normalize semantics."
                        ),
                        "pos_neg_margin_mean": "Mean raw positive-minus-negative logit over valid distinct training pairs; score-scale dependent.",
                        "pos_neg_margin_p10": "Nearest-rank tenth percentile of raw sampled training-pair margins.",
                        "pos_neg_neg_beats_pos_rate": "Fraction of valid sampled training pairs with negative logit strictly higher.",
                        "pos_neg_small_margin_rate": "Fraction of valid sampled training pairs with raw margin < 0.5; scale dependent.",
                        "margin_valid_positions": "Exact number of real, distinct candidate training pairs used for margin summaries. Rows containing invalid history IDs are excluded. Zero count yields null margin summaries; nonfinite valid scores raise an explicit error.",
                        "pos_neg_margin_category_breakdown": "Category lists remain empty because category metadata is unavailable; total counts describe valid sampled training positions only.",
                        "history_sensitivity": (
                            "Last position with sequence>0 and positive>0 per row; real distinct fixed positive/negative IDs; "
                            "at least four real history items. Pairwise win is 1/0/0.5 for win/loss/tie. "
                            "k=min(3,floor((n-1)/3)). Composite controls remove k newest or oldest items. "
                            "Recent-nonfinal removes the k positions immediately before the final item; earlier_matched "
                            "removes k uniformly sampled positions from the disjoint preceding region. final_deleted "
                            "removes the final position; earlier_single removes one uniformly sampled nonfinal position. "
                            "Five random controls are averaged within each row, using a local seed-42 generator. "
                            "Suffix lengths 2/4/8/16/n are clipped and deduplicated in that order; "
                            "suffix_full always denotes the complete original history, while other suffix names "
                            "give the actual retained length. Condition counts can therefore differ. All histories retain "
                            "their original external width and chronological order. Mean deltas are intervention-minus-original; "
                            "direct paired differences subtract the matched earlier control. Cosine distances are reported "
                            "separately, average valid draws within rows, and exclude only zero-norm cosine comparisons. "
                            "Final-position retention differs from final-item-ID change when histories contain repeats. "
                            "Execution checks use one eligible history; null checks mean no observation. "
                            "The baseline explicitly initializes each GRU call at zero but processes leading padding. "
                            "Deletion effects therefore confound padding and context; recency conclusions are withheld. "
                            "No user intervals or four-category macro intervals are estimated from anonymous rows. "
                            "These are descriptive training-prefix sensitivity measurements, not held-out HR@10, "
                            "evidence of intent, proof of harmful shortcut use, or improvement estimates."
                        ),
                    },
                }
        finally:
            module.train(saved_training)

    ### <<< Self_EvolveRec-BLOCK-END
