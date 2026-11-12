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
    ### >>> Self_EvolveRec-BLOCK-START: Separate measured training diagnostics from validation evidence
    - Training margins measure in-sample sampled-negative separation, not held-out ranking or user preferences.
    - Embedding cosine measures alignment; it cannot establish collapse or recommendation diversity by itself.
    - Suffix interventions measure frozen-model sensitivity, not the effect of retraining with stochastic suffix removal.
    - Endpoint diagnostics and dense-position diagnostics have different supervision weights; do not conflate them.
    - Null metrics mean unavailable measurements, never measured zero effects.
    - Ratings, reviews, item categories/prices/stores and simulator traits are unavailable.
    - Only the pinned host four-category macro validation HR@10 selects model trials.
    - These diagnostics alone cannot establish healthy generalization or justify CRITICAL status.
    ### <<< Self_EvolveRec-BLOCK-END
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the family-specific ModelView contract
    def __init__(self, view, device):
        self.model = view
        self.network = view.model
        if not isinstance(self.network, torch.nn.Module):
            raise TypeError("view.model must be the actual trained nn.Module")
        self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    ### <<< Self_EvolveRec-BLOCK-END

    def _get_categories(self, item_id: int):
        # DEBUG: Item categories are unavailable in this benchmark.
        # Keep the compatibility helper without inferring category labels.
        return []

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            ### >>> Self_EvolveRec-BLOCK-START: Read catalog embeddings from the actual trained network
            emb = self.network.item_emb.weight.detach()
            emb = emb[1 : int(self.network.itemnum) + 1]
            if emb.ndim != 2 or not torch.isfinite(emb).all():
                raise ValueError("Expected finite catalog item embeddings")
            ### <<< Self_EvolveRec-BLOCK-END

            n = emb.size(0)
            ### >>> Self_EvolveRec-BLOCK-START: Represent unavailable embedding alignment explicitly
            if n < 2 or sample_size < 2:
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            ### >>> Self_EvolveRec-BLOCK-START: Compute identical mean cosine without a quadratic Gram matrix
            x = F.normalize(emb.index_select(0, idx).double(), p=2, dim=-1)
            # Sum of all pairwise inner products equals the squared norm of
            # the vector sum. Subtract actual diagonal norms, including zeros.
            offdiag_sum = x.sum(dim=0).square().sum() - x.square().sum()
            offdiag_cnt = k * (k - 1)
            mean_cos = (offdiag_sum / offdiag_cnt).clamp(-1, 1).item()
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
        ### >>> Self_EvolveRec-BLOCK-START: Diagnose dense margins and matched endpoint suffix sensitivity
        margins_all = []
        endpoint_records = []
        augmentation_rng = torch.Generator(device="cpu").manual_seed(42)
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Bound loader consumption without drawing an unused diagnostic batch
        if num_batches < 0:
            raise ValueError("num_batches must be nonnegative")
        # range is advanced first, so exhausting it never requests another batch.
        for step, batch in zip(range(num_batches), dataloader):
            ### <<< Self_EvolveRec-BLOCK-END
            ### >>> Self_EvolveRec-BLOCK-START: Validate aligned scores and measure endpoint causality with identical labels and negatives
            if not (isinstance(batch, (list, tuple)) and len(batch) == 4):
                raise ValueError(
                    "Diagnostic loader must yield (None, sequence, positive, negative)"
                )
            _, seq, pos, neg = batch
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
            ):
                raise ValueError("Expected aligned B x L diagnostic tensors")
            log_feats = self.model.log2feats(seq_t)
            if (
                log_feats.ndim != 3
                or log_feats.shape[:2] != seq_t.shape
                or not torch.isfinite(log_feats).all()
            ):
                raise ValueError(
                    "log2feats must return finite aligned B x L x D features"
                )
            positive_scores = self.model.predict_score(log_feats, pos_t)
            negative_scores = self.model.predict_score(log_feats, neg_t)
            if (
                positive_scores.shape != seq_t.shape
                or negative_scores.shape != seq_t.shape
                or not torch.isfinite(positive_scores).all()
                or not torch.isfinite(negative_scores).all()
            ):
                raise ValueError("Expected finite aligned B x L diagnostic scores")
            margin = positive_scores - negative_scores
            valid = (seq_t != 0) & (pos_t != 0) & (neg_t != 0)

            # One deterministic endpoint per eligible loader row. These are
            # frozen-model probes, not a replacement training objective.
            rows, endpoints = [], []
            for row, flags in enumerate(valid.detach().cpu()):
                positions = flags.nonzero(as_tuple=False).flatten()
                if positions.numel():
                    index = (step + row) % positions.numel()
                    rows.append(row)
                    endpoints.append(int(positions[index]))
            if rows:
                row_ids = torch.tensor(rows, device=self.device, dtype=torch.long)
                terminal = torch.tensor(endpoints, device=self.device, dtype=torch.long)
                prefixes = seq_t.index_select(0, row_ids).clone()
                indices = torch.arange(seq_t.size(1), device=self.device)
                prefixes.masked_fill_(indices[None, :] > terminal[:, None], 0)
                positive = pos_t[row_ids, terminal]
                negative = neg_t[row_ids, terminal]
                variants = {
                    "matched_full": prefixes,
                    "matched_half_suffix": prefixes.clone(),
                    "matched_random_suffix": prefixes.clone(),
                }
                lengths = []
                random_lengths = []
                for row in range(prefixes.size(0)):
                    occupied = prefixes[row].nonzero(as_tuple=False).flatten()
                    n = occupied.numel()
                    minimum = (n + 1) // 2
                    retained = n
                    if (
                        n > 1
                        and torch.rand((), generator=augmentation_rng).item() >= 0.5
                    ):
                        retained = int(
                            torch.randint(
                                minimum, n + 1, (), generator=augmentation_rng
                            ).item()
                        )
                    variants["matched_half_suffix"][row, occupied[: n - minimum]] = 0
                    variants["matched_random_suffix"][row, occupied[: n - retained]] = 0
                    lengths.append(n)
                    random_lengths.append(retained)
                measured = {}
                endpoint_causality_errors = None
                for name, inputs in variants.items():
                    features = self.model.log2feats(inputs)
                    if (
                        features.shape != log_feats.index_select(0, row_ids).shape
                        or not torch.isfinite(features).all()
                    ):
                        raise ValueError(
                            "Suffix encoding returned invalid aligned features"
                        )
                    last = (
                        indices.expand_as(inputs)
                        .masked_fill(inputs == 0, -1)
                        .amax(dim=1)
                    )
                    if not torch.equal(last, terminal):
                        raise AssertionError("Suffix probe changed endpoint positions")
                    features = features[
                        torch.arange(inputs.size(0), device=self.device), last
                    ]
                    if name == "matched_full":
                        # A causal encoder must agree at the endpoint when only
                        # later inputs are removed; retained position IDs agree.
                        endpoint_causality_errors = (
                            (features - log_feats[row_ids, terminal])
                            .abs()
                            .amax(dim=-1)
                            .detach()
                            .double()
                            .cpu()
                            .tolist()
                        )
                    positive_scores = self.model.predict_score(features, positive)
                    negative_scores = self.model.predict_score(features, negative)
                    if (
                        positive_scores.shape != positive.shape
                        or negative_scores.shape != negative.shape
                        or not torch.isfinite(positive_scores).all()
                        or not torch.isfinite(negative_scores).all()
                    ):
                        raise ValueError("Expected one finite score per endpoint item")
                    measured[name] = (
                        (positive_scores - negative_scores)
                        .detach()
                        .float()
                        .cpu()
                        .tolist()
                    )
                for row, n in enumerate(lengths):
                    endpoint_records.append(
                        {
                            "batch": step,
                            "row": rows[row],
                            "endpoint": endpoints[row],
                            "available_items": n,
                            "future_input_removal_feature_max_abs_error": (
                                endpoint_causality_errors[row]
                            ),
                            "future_input_removal_changed_tokens": int(
                                (seq_t[rows[row]] != prefixes[row]).sum().item()
                            ),
                            "half_retained_items": (n + 1) // 2,
                            "random_retained_items": random_lengths[row],
                            "random_full_probability": (
                                1.0 if n <= 1 else 0.5 + 0.5 / (n - (n + 1) // 2 + 1)
                            ),
                            "margins": {
                                name: values[row] for name, values in measured.items()
                            },
                        }
                    )
            ### <<< Self_EvolveRec-BLOCK-END
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            valid_cpu = valid.detach().cpu().numpy()

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
            ### >>> Self_EvolveRec-BLOCK-START: Missing margins are not measured zeros
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

        breakdown = {
            # DEBUG: Distinguish unavailable category labels from measured counts.
            "category_breakdown_available": False,
            "category_breakdown_unavailable_reason": (
                "Item categories are not provided by this benchmark."
            ),
            "valid_positions": sum(int(m.numel()) for m in margins_all),
            "eps": float(eps),
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

        ### >>> Self_EvolveRec-BLOCK-START: Report suffix support and measured causal consistency without claiming retraining
        breakdown["endpoint_suffix_probe"] = {
            "protocol": "frozen-nextitnet-endpoint-suffix-v1",
            "endpoint_schedule": "valid_positions[(loader_batch_index + row_index) % count]",
            "augmentation_seed": 42,
            "endpoint_count": len(endpoint_records),
            "records": endpoint_records,
            "summary": {},
            "scope": "Frozen-model training-loader sensitivity; no models fitted",
            "training_intervention_verified": False,
            "training_intervention_unverified_reason": (
                "This diagnostic does not implement or verify endpoint-only "
                "training, matched supervision streams, or checkpoint selection."
            ),
            "future_input_removal_feature_max_abs_error": (
                max(
                    r["future_input_removal_feature_max_abs_error"]
                    for r in endpoint_records
                )
                if endpoint_records
                else None
            ),
            "future_input_removal_changed_tokens": sum(
                r["future_input_removal_changed_tokens"] for r in endpoint_records
            ),
            "causality_definition": (
                "Maximum absolute endpoint-feature difference between the dense "
                "input and its fixed-position prefix after removing future inputs. "
                "Changed-token counts distinguish actual interventions from no-ops. "
                "This local numerical check does not prove causality for all inputs."
            ),
        }
        for name in ("matched_full", "matched_half_suffix", "matched_random_suffix"):
            values = np.asarray(
                [r["margins"][name] for r in endpoint_records], dtype=np.float64
            )
            differences = np.asarray(
                [
                    r["margins"][name] - r["margins"]["matched_full"]
                    for r in endpoint_records
                ],
                dtype=np.float64,
            )
            breakdown["endpoint_suffix_probe"]["summary"][name] = {
                "mean_margin": float(values.mean()) if values.size else None,
                "negative_beats_positive_rate": (
                    float((values < 0).mean()) if values.size else None
                ),
                "mean_margin_change_from_full": (
                    float(differences.mean()) if differences.size else None
                ),
            }
        return margin_metrics, breakdown
        ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Restore RNG streams and individual module modes after diagnosis
    def run_full_diagnosis(self, train_loader):
        states = [(module, module.training) for module in self.network.modules()]
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        cpu_state = torch.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            self.network.eval()
            with torch.no_grad():
                return self._run_full_diagnosis(train_loader)
        finally:
            for module, training in states:
                module.training = training
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    def _run_full_diagnosis(self, train_loader):
        print(">>> Running NextItNet training-margin and endpoint-suffix diagnosis...")
        collapse_score = self.analyze_embeddings_collapse_cosine(sample_size=2048)
        eps = 0.5
        margin_metrics, breakdown = self.analyze_pos_neg_margin_with_category_breakdown(
            train_loader, num_batches=10, eps=eps, topk_categories=20
        )
        ### <<< Self_EvolveRec-BLOCK-END
        return {
            "metrics": {
                "embedding_collapse_score": collapse_score,
                "pos_neg_margin_mean": margin_metrics["mean"],
                "pos_neg_margin_p10": margin_metrics["p10"],
                "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                "pos_neg_margin_category_breakdown": breakdown,
            },
            "metric_definitions": {
                "embedding_collapse_score": (
                    ### >>> Self_EvolveRec-BLOCK-START: Interpret alignment without unsupported collapse claims
                    "Mean sampled off-diagonal item cosine mapped to [0,1]. "
                    "Higher values indicate stronger directional alignment, which alone "
                    "does not establish collapse, diversity or recommendation quality. "
                    "Null means fewer than two embeddings were available."
                    ### <<< Self_EvolveRec-BLOCK-END
                ),
                "pos_neg_margin_mean": "Mean of (pos_logit - neg_logit) over valid positions; higher means clearer separation.",
                "pos_neg_margin_p10": "10th percentile of (pos_logit - neg_logit); lower suggests many hard/confusing cases.",
                "pos_neg_neg_beats_pos_rate": "Fraction where neg_logit > pos_logit (margin < 0); lower is better.",
                "pos_neg_small_margin_rate": "Fraction where (pos_logit - neg_logit) < eps; higher means weak separation.",
                # DEBUG: Describe measured counts without claiming unavailable metadata.
                "pos_neg_margin_category_breakdown": (
                    "Reports valid training-position counts and counts of low-margin "
                    "(margin<eps) and wrong-rank (margin<0) cases. Category breakdowns "
                    "are unavailable because item category labels are not provided; "
                    "empty category lists are not measured zero category counts. "
                    ### >>> Self_EvolveRec-BLOCK-START: Define dense separation, endpoint sensitivity and causal consistency separately
                    "Dense margins use all valid loader positions and describe sampled "
                    "training separation. If valid_positions is zero, margin summaries "
                    "are null. endpoint_suffix_probe selects one deterministic valid "
                    "endpoint per loader row, removes future inputs, preserves absolute "
                    "positions, terminal items and identical positive/negative IDs, then "
                    "compares full, ceil(n/2) and stochastic suffix contexts. Random "
                    "suffixes retain full context with probability 0.5 plus the uniform "
                    "branch's full-length probability, recorded per endpoint. Summaries "
                    "weight endpoint records equally; repeated rows are not independent "
                    "users. Positive margin changes mean greater sampled separation. "
                    "These are frozen-model sensitivity measurements, not retrained "
                    "condition results, validation HR@10 or evidence of user intent. "
                    "Future-input-removal feature errors compare endpoint features "
                    "before and after removing later inputs at fixed positions; "
                    "changed-token counts expose no-op comparisons. They are local "
                    "causal-consistency measurements, not a general proof. "
                    "No training alignment or predictor-equivalence test is implied. "
                    "The supplied training workflow still uses dense BPR and does "
                    "not invoke endpoint selection or stochastic suffix removal; "
                    "these probes alone do not implement the research experiment."
                    ### <<< Self_EvolveRec-BLOCK-END
                ),
            },
        }
