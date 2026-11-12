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
    ### >>> Self_EvolveRec-BLOCK-START: Evidence-limited temporal diagnosis interpretation
    - Distinguish position-weighted training separation, training-user score sensitivity,
      and held-out validation ranking. Only the first two are measured by this adapter.
    - Cite eligible counts, controls, missingness, and uncertainty for every finding.
    - A score change is not evidence of relevance, improved HR, user intent, simulator
      traits, or feedback effects. Angular alignment and covariance rank do not establish
      useful ranking. The margin threshold 0.5 is scale dependent.
    - Stale-content hypotheses require corroboration across deletion, geometry controls,
      zero replacement, and frequency-matched nonzero replacement. Otherwise label them
      exploratory. Interior zeros and replacement items introduce distribution shifts.
    - Bootstrap intervals condition on fitted models, users, and fixed candidates;
      they exclude training-seed, candidate-sampling, and adaptive-search uncertainty.
      Do not infer p-values or confirmatory significance from these intervals.
    - Missing categories, validation interventions, or unavailable metadata are unavailable
      evidence. Do not invent measurements or recommend ensembles or fitted score fusion.
    ### <<< Self_EvolveRec-BLOCK-END
    - Keep it short: core_findings <= 3 bullets, key_implications <= 3 bullets.
    - Use technical terms only when they are essential (e.g., 'gradient', 'loss landscape', 'inductive bias'), and keep them brief.
    - Every statement must be grounded in the provided metric values/definitions.
    - If metrics look healthy, status should remain STABLE and implications should focus on "what is already working" plus one potential risk to monitor.
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the family-specific ModelView boundary
    def __init__(self, view, device):
        self.view = view
        self.model = view
        self.device = torch.device(device)
        self.meta_data = getattr(view, "meta_data", {})
        self.review_data = getattr(view, "review_data", {})
        self.user_train = view.user_train

    ### <<< Self_EvolveRec-BLOCK-END

    def _get_categories(self, item_id: int):
        key = str(int(item_id))
        info = self.meta_data.get(key, None)
        if not isinstance(info, dict):
            return []
        cats = info.get("categories", [])
        if isinstance(cats, list):
            return [c for c in cats if isinstance(c, str)]
        return []

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()
            emb = emb[
                1 : self.model.model.itemnum + 1
            ]  # exclude padding and reserved mask tokens

            n = emb.size(0)
            if n <= 2:
                return 0.0

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
        self.model.eval()

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

            seq_np = seq.cpu().numpy() if isinstance(seq, torch.Tensor) else seq
            log_feats = self.model.log2feats(seq_np)

            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)

            # DEBUG: Score B x L item tensors separately to match NextItNet's
            # B x L x D features and avoid an incompatible candidate dimension.
            pos_logits = self.model.predict_score(log_feats, pos_t)
            neg_logits = self.model.predict_score(log_feats, neg_t)
            margin = pos_logits - neg_logits

            valid = pos_t != 0
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            valid_cpu = pos_cpu != 0

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
                "mean": 0.0,
                "p10": 0.0,
                "neg_beats_pos_rate": 0.0,
                "small_margin_rate": 0.0,
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

        return margin_metrics, breakdown

    ### >>> Self_EvolveRec-BLOCK-START: Padding-aware NextItNet structural and paired temporal diagnostics
    def run_full_diagnosis(self, train_loader):
        from itertools import islice

        view, network = self.view, self.view.model
        rng = np.random.default_rng(42)
        user_rng = np.random.default_rng(42)
        candidate_rng = np.random.default_rng(42)
        previous_modes = [(module, module.training) for module in network.modules()]
        python_state, numpy_state = random.getstate(), np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_state = (
            torch.cuda.get_rng_state(self.device)
            if self.device.type == "cuda"
            else None
        )
        metrics = {}
        definitions = {
            "training_positions": "Descriptive position-weighted training statistics from at most ten batches, requiring nonzero sequence, positive and negative IDs. No independent-user inference.",
            "pos_neg_margin_mean": "Mean positive-minus-negative training score; scale dependent.",
            "pos_neg_margin_p10": "Nearest-rank tenth percentile of training margins.",
            "pos_neg_neg_beats_pos_rate": "Fraction of valid training margins below zero.",
            "pos_neg_small_margin_rate": "Fraction of valid training margins below 0.5; this threshold is scale dependent.",
            "pairwise_accuracy": "P(positive score > negative score) + 0.5 P(tie), conditional on sampled negative difficulty; invariant to strictly increasing score transformations.",
            "embedding_angular_alignment": "(Mean off-diagonal cosine + 1)/2; approximately 0.5 means zero mean cosine. Not a ranking-quality or semantic-diversity metric.",
            "representations": "Centered covariance eigenvalues for raw and unit-normalized vectors; total variance, largest eigenvalue share and exp(-sum(p log p)), where p is covariance eigenvalue mass. This is not RankMe. Zero variance has null effective rank.",
            "structure": "Inspection of actual NextItNet residual blocks, temporal convolutions, positionwise norms, biases, explicit causal padding and masks. Reachability is conditional on this recognized graph and absence of unrecognized operations.",
            "checks": "Feature comparisons in eval mode: atol=1e-6, rtol=1e-5; report maximum absolute difference. Width checks retain item content but may change absolute positions.",
            "training_endpoint_interventions": "Fixed positive/negative endpoint pair, largest nonzero index, descriptive loader-row margin changes; rows are not identified independent users.",
            "user_score_sensitivity": "Identified public training users, fixed 99 training-observed unseen candidates per user. Mean absolute score change and signed mean score change; no relevance labels, HR or target ranks.",
            "temporal_geometry": "Identified-user strengths 0.25 and 0.5 use k=max(1,floor(fraction*original_nonpadding_length)), before baseline truncation. Interventions are unavailable when retained history cannot support the requested size or preserve the newest k items. Loader-row probes use their observed nonpadding lengths because full histories are unidentified. Deletions compact and left-pad; replacements preserve positions; shuffles preserve identities and multiplicities. Unchanged draws are excluded and counted, so shuffle estimates condition on changed draws.",
            "user_intervals": "Pointwise 95% percentile intervals from 2000 seed-42 user bootstrap resamples. Each complete paired vector is resampled together within the current category and original-length stratum. Conditional on fitted model, fixed users/candidates; excludes training-seed, candidate-sampling and adaptive-search uncertainty. Exploratory, no p-values.",
            "missingness": "Unavailable observations are null with reasons, never zero-effect evidence. Cross-category macro estimates require all four categories.",
        }

        def summary(values):
            x = np.asarray(values, dtype=np.float64).reshape(-1)
            if not x.size:
                return {"n": 0, "mean": None, "reason": "No eligible observations"}
            if not np.isfinite(x).all():
                return {
                    "n": int(x.size),
                    "mean": None,
                    "reason": "Nonfinite observations",
                }
            return {
                "n": int(x.size),
                "mean": float(x.mean()),
                "p10": float(np.quantile(x, 0.1)),
                "p50": float(np.quantile(x, 0.5)),
                "p90": float(np.quantile(x, 0.9)),
                "min": float(x.min()),
                "max": float(x.max()),
            }

        def representation(x):
            x = x.detach().float().cpu().double()
            if not torch.isfinite(x).all():
                return {
                    "count": int(x.shape[0]),
                    "value": None,
                    "reason": "Nonfinite representation values; covariance unavailable",
                }
            result = {
                "count": int(x.shape[0]),
                "norms": summary(x.norm(dim=-1).numpy()),
            }
            for name, z in (("raw", x), ("unit_normalized", F.normalize(x, dim=-1))):
                if len(z) < 2:
                    result[name] = {
                        "total_variance": None,
                        "effective_rank": None,
                        "reason": "At least two vectors required",
                    }
                    continue
                z = z - z.mean(0, keepdim=True)
                eig = torch.linalg.eigvalsh(z.T @ z / (len(z) - 1)).clamp_min(0)
                total = float(eig.sum())
                entry = {
                    "eigenvalues": eig.tolist(),
                    "total_variance": total,
                    "leading_share": None,
                    "effective_rank": None,
                }
                if total > 0:
                    p = eig[eig > 0] / total
                    entry.update(
                        leading_share=float(eig[-1] / total),
                        effective_rank=float(torch.exp(-(p * p.log()).sum())),
                    )
                else:
                    entry["reason"] = "Zero centered variance"
                result[name] = entry
            return result

        def encode(seq):
            return view.log2feats(
                torch.as_tensor(seq, dtype=torch.long, device=self.device)
            )

        def endpoint(seq):
            positions = torch.arange(seq.shape[1], device=seq.device).expand_as(seq)
            return positions.masked_fill(seq == 0, -1).max(1).values

        def final_features(sequences):
            seq = torch.as_tensor(
                (
                    sequences
                    if isinstance(sequences, torch.Tensor)
                    else np.asarray(sequences)
                ),
                dtype=torch.long,
                device=self.device,
            )
            if seq.ndim != 2 or seq.shape[1] == 0:
                raise ValueError("Diagnostic sequences must have nonempty B x L shape")
            last = endpoint(seq)
            if (last < 0).any():
                raise ValueError("All-padding history cannot be scored by diagnostics")
            feats = encode(seq)
            return feats[torch.arange(len(seq), device=self.device), last]

        def score_vectors(sequences, candidates):
            output = []
            for start in range(0, len(sequences), 32):
                feats = final_features(sequences[start : start + 32])
                # One feature vector and one candidate list per call preserves
                # ModelView implementations accepting arbitrary item-list shapes.
                output.extend(
                    view.predict_score(f, candidates).detach().float().cpu().numpy()
                    for f in feats
                )
            return np.asarray(output)

        def compare(a, b):
            return {
                "max_absolute_difference": float((a - b).abs().max()),
                "within_tolerance": bool(torch.allclose(a, b, atol=1e-6, rtol=1e-5)),
            }

        def inspect_structure():
            convs = [
                {
                    "name": name,
                    "kernel": list(m.kernel_size),
                    "dilation": list(m.dilation),
                    "stride": list(m.stride),
                    "padding": list(m.padding),
                    "bias": m.bias is not None,
                }
                for name, m in network.named_modules()
                if isinstance(m, torch.nn.Conv1d)
            ]
            norms = [
                {
                    "name": name,
                    "type": type(m).__name__,
                    "normalized_shape": (
                        list(m.normalized_shape)
                        if isinstance(m, torch.nn.LayerNorm)
                        else None
                    ),
                }
                for name, m in network.named_modules()
                if isinstance(
                    m,
                    (
                        torch.nn.LayerNorm,
                        torch.nn.BatchNorm1d,
                        torch.nn.InstanceNorm1d,
                        torch.nn.GroupNorm,
                    ),
                )
            ]
            recognized = type(network).__name__ == "NextItNetModel"
            recognized &= hasattr(network, "blocks") and hasattr(network, "depth_norms")
            offsets, all_depths = {0}, set()
            if recognized:
                recognized &= len(convs) == 2 * len(network.blocks)
                for block in network.blocks:
                    recognized &= type(block).__name__ == "NextItNetBlock"
                    branch = set(offsets)
                    for conv in (block.conv1, block.conv2):
                        recognized &= (
                            conv.stride == (1,)
                            and conv.padding == (0,)
                            and conv.kernel_size == (block.kernel_size,)
                            and conv.dilation == (block.dilation,)
                        )
                        branch = {
                            o + j * conv.dilation[0]
                            for o in branch
                            for j in range(conv.kernel_size[0])
                        }
                    offsets |= branch
                    all_depths |= offsets
                recognized &= all(
                    n["type"] == "LayerNorm"
                    and n["normalized_shape"] == [network.hidden_units]
                    for n in norms
                )
            return {
                "convolutions": convs,
                "normalizations": norms,
                "recognized_baseline_graph": bool(recognized),
                "reachable_endpoint_offsets": (
                    sorted(all_depths) if recognized else None
                ),
                "receptive_field_span": (
                    max(all_depths) + 1 if recognized and all_depths else None
                ),
                "padding_and_masks": (
                    "Recognized baseline explicitly left-pads each convolution and masks before/after convolutions, normalization and residual paths"
                    if recognized
                    else None
                ),
                "locality_caveat": "Conditional on supplied baseline forward graph; empirical checks below corroborate but do not prove locality",
            }

        try:
            network.eval()
            with torch.no_grad():
                metrics["structure"] = inspect_structure()
                itemnum = int(network.itemnum)
                weights = view.item_emb.weight.detach()
                real_count = min(itemnum, len(weights) - 1)
                indices = rng.choice(
                    np.arange(1, real_count + 1), min(2048, real_count), replace=False
                )
                sample = weights[
                    torch.as_tensor(indices, dtype=torch.long, device=weights.device)
                ]
                metrics["item_representations"] = representation(sample)
                if len(sample) > 1:
                    unit = F.normalize(sample.float(), dim=-1)
                    cosine = (unit.sum(0).square().sum() - unit.square().sum()) / (
                        len(unit) * (len(unit) - 1)
                    )
                    metrics["embedding_angular_alignment"] = float(
                        (cosine.clamp(-1, 1) + 1) / 2
                    )
                else:
                    metrics["embedding_angular_alignment"] = None
                margins, positives, negatives, final_states = [], [], [], []
                endpoint_effects = []
                batch_count, valid_count = 0, 0
                for batch in islice(iter(train_loader), 10):
                    _, sequence, positive, negative = batch
                    seq, pos, neg = [
                        torch.as_tensor(x, dtype=torch.long, device=self.device)
                        for x in (sequence, positive, negative)
                    ]
                    feats = encode(seq)
                    # Score the two B x L tensors separately: the actual NextItNet
                    # scorer multiplies B x L x D features and item embeddings.
                    ps, ns = view.predict_score(feats, pos), view.predict_score(
                        feats, neg
                    )
                    valid = (seq != 0) & (pos != 0) & (neg != 0)
                    margins.extend((ps - ns)[valid].float().cpu().tolist())
                    positives.extend(ps[valid].float().cpu().tolist())
                    negatives.extend(ns[valid].float().cpu().tolist())
                    valid_count += int(valid.sum())
                    batch_count += 1
                    last = endpoint(seq)
                    rows = torch.where(last >= 0)[0]
                    final_states.append(feats[rows, last[rows]].detach().cpu())
                    if batch_count == 1 and len(rows):
                        checks = {
                            "identity": compare(feats, encode(seq)),
                            "batch_reordering": compare(
                                feats, encode(seq.flip(0)).flip(0)
                            ),
                            "input_width": int(seq.shape[1]),
                        }
                        row = int(rows[0])
                        nonzero = torch.where(seq[row] != 0)[0]
                        checks["retained_history"] = int(len(nonzero))
                        if len(nonzero) >= 2 and real_count >= 2:
                            pivot = int(nonzero[len(nonzero) // 2 - 1])
                            altered = seq[row : row + 1].clone()
                            future = (
                                torch.arange(seq.shape[1], device=self.device) > pivot
                            )
                            future &= altered[0] != 0
                            altered[0, future] = altered[0, future] % real_count + 1
                            checks["future_token_causality"] = dict(
                                compare(
                                    feats[row, : pivot + 1],
                                    encode(altered)[0, : pivot + 1],
                                ),
                                pivot=pivot,
                                changed_tokens=int(future.sum()),
                            )
                        offsets = metrics["structure"]["reachable_endpoint_offsets"]
                        if offsets is not None and real_count >= 2:
                            end = int(last[row])
                            outside = [
                                int(i) for i in nonzero if end - int(i) not in offsets
                            ]
                            if outside:
                                altered = seq[row : row + 1].clone()
                                altered[0, outside] = (
                                    altered[0, outside] % real_count + 1
                                )
                                checks["outside_reachable_offsets"] = dict(
                                    compare(feats[row, end], encode(altered)[0, end]),
                                    changed_lags=[end - i for i in outside],
                                )
                            else:
                                checks["outside_reachable_offsets"] = {
                                    "value": None,
                                    "reason": "No retained tokens outside reachable offsets",
                                }
                        width_checks = []
                        first = int(nonzero[0])
                        for remove in sorted({1, first // 2, first}):
                            if not 0 < remove <= first:
                                continue
                            shortened = seq[row : row + 1, remove:]
                            try:
                                width_checks.append(
                                    dict(
                                        compare(
                                            feats[row, last[row]],
                                            final_features(shortened.cpu().numpy())[0],
                                        ),
                                        input_width=int(shortened.shape[1]),
                                        removed_leading_padding=remove,
                                        absolute_positions_shifted=hasattr(
                                            network, "pos_emb"
                                        ),
                                    )
                                )
                            except (RuntimeError, ValueError, IndexError) as exc:
                                width_checks.append(
                                    {
                                        "value": None,
                                        "input_width": int(shortened.shape[1]),
                                        "reason": str(exc),
                                    }
                                )
                        checks["leading_padding_widths"] = width_checks or {
                            "value": None,
                            "reason": "No removable leading padding",
                        }
                        metrics["checks"] = checks
                        for r in rows[:8].tolist():
                            e = int(last[r])
                            if not bool(valid[r, e]):
                                continue
                            history = seq[r][seq[r] != 0].cpu().numpy()
                            if len(history) < 2:
                                continue
                            pair = torch.stack((pos[r, e], neg[r, e]))
                            baseline = float(ps[r, e] - ns[r, e])
                            for fraction in (0.25, 0.5):
                                k = max(1, int(fraction * len(history)))
                                for region, kept in (
                                    ("old", history[k:]),
                                    ("recent", history[:-k]),
                                ):
                                    altered = np.zeros(seq.shape[1], dtype=np.int64)
                                    altered[-len(kept) :] = kept
                                    scores = score_vectors([altered], pair)[0]
                                    endpoint_effects.append(
                                        {
                                            "fraction": fraction,
                                            "region": region,
                                            "retained_length": len(history),
                                            "deleted": k,
                                            "margin_change": float(
                                                scores[0] - scores[1] - baseline
                                            ),
                                        }
                                    )
                m = np.asarray(margins)
                metrics["training_positions"] = {
                    "batches": batch_count,
                    "valid_positions": valid_count,
                    "margins": summary(m),
                    "positive_scores": summary(positives),
                    "negative_scores": summary(negatives),
                }
                metrics["pos_neg_margin_mean"] = float(m.mean()) if len(m) else None
                metrics["pos_neg_margin_p10"] = (
                    float(np.sort(m)[max(0, int(np.ceil(0.1 * len(m))) - 1)])
                    if len(m)
                    else None
                )
                metrics["pos_neg_neg_beats_pos_rate"] = (
                    float((m < 0).mean()) if len(m) else None
                )
                metrics["pos_neg_small_margin_rate"] = (
                    float((m < 0.5).mean()) if len(m) else None
                )
                metrics["pairwise_accuracy"] = (
                    float(((m > 0) + 0.5 * (m == 0)).mean()) if len(m) else None
                )
                metrics["training_endpoint_interventions"] = endpoint_effects
                metrics["loader_endpoint_representations"] = (
                    representation(torch.cat(final_states))
                    if final_states
                    else {"count": 0, "reason": "No nonpadding rows"}
                )

                histories = self.user_train.get("History", self.user_train)
                if not isinstance(histories, dict):
                    raise TypeError(
                        "ModelView user_train must expose mapped item histories"
                    )
                # Histories must already use the baseline's integer item mapping.
                ordered = sorted(histories, key=lambda x: (type(x).__name__, str(x)))
                chosen = sorted(
                    user_rng.choice(
                        len(ordered), min(200, len(ordered)), replace=False
                    ).tolist()
                )
                frequencies = Counter(
                    int(i)
                    for h in histories.values()
                    for i in h
                    if 1 <= int(i) <= real_count
                )
                observed = np.asarray(sorted(frequencies), dtype=np.int64)
                ordered_frequency = sorted(
                    observed.tolist(), key=lambda i: (frequencies[i], i)
                )
                decile = {
                    item: min(9, 10 * rank // max(1, len(ordered_frequency)))
                    for rank, item in enumerate(ordered_frequency)
                }
                pools = {
                    d: np.asarray(
                        [i for i in observed if decile[int(i)] == d], dtype=np.int64
                    )
                    for d in range(10)
                }
                maxlen = int(network.maxlen)
                user_rows, skips, intact_user_states = [], Counter(), []

                def padded(history):
                    history = np.asarray(history, dtype=np.int64)
                    if history.ndim != 1 or not 0 < len(history) <= maxlen:
                        raise ValueError(
                            "History must be nonempty and fit the diagnostic input width"
                        )
                    array = np.zeros(maxlen, dtype=np.int64)
                    array[-len(history) :] = history
                    return array

                for index in chosen:
                    uid = ordered[index]
                    original = np.asarray(histories[uid], dtype=np.int64)
                    if not len(original) or np.any(
                        (original < 1) | (original > real_count)
                    ):
                        skips["empty_or_unmapped_history"] += 1
                        continue
                    h = original[-maxlen:]
                    unseen = observed[~np.isin(observed, original)]
                    if len(unseen) < 99:
                        skips["fewer_than_99_training_observed_unseen_candidates"] += 1
                        continue
                    candidates = candidate_rng.choice(unseen, 99, replace=False)
                    cand_t = torch.as_tensor(
                        candidates, dtype=torch.long, device=self.device
                    )
                    intact = padded(h)
                    intact_features = final_features([intact])[0]
                    intact_user_states.append(intact_features.detach().cpu())
                    baseline = (
                        view.predict_score(intact_features, cand_t)
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                    )
                    if baseline.shape != (99,) or not np.isfinite(baseline).all():
                        raise RuntimeError(
                            "Diagnostic scorer must return 99 finite candidate scores"
                        )
                    variants, labels, geometry = (
                        [intact.copy()],
                        ["identity"],
                        [{"changed": 0}],
                    )
                    unavailable = Counter()

                    def add(label, history, metadata):
                        array = padded(history)
                        if np.array_equal(array, intact):
                            unavailable[label + ":unchanged"] += 1
                            return
                        variants.append(array)
                        labels.append(label)
                        geometry.append(metadata)

                    for fraction in (0.25, 0.5):
                        n = len(h)
                        k = max(1, int(fraction * len(original)))
                        prefix = str(fraction) + ":"
                        if n <= k:
                            unavailable[prefix + "insufficient_remaining_history"] += 1
                            continue
                        regions = {"recent": np.arange(n - k, n)}
                        if n - k >= k:
                            regions["old"] = np.arange(k)
                        else:
                            unavailable[prefix + "old:cannot_preserve_newest_k"] += 1
                        if n - k >= k:
                            regions["random"] = np.sort(
                                rng.choice(n - k, k, replace=False)
                            )
                        else:
                            unavailable[prefix + "random:cannot_preserve_newest_k"] += 1
                        if n - k - 1 >= k:
                            regions["interior"] = np.sort(
                                rng.choice(np.arange(1, n - k), k, replace=False)
                            )
                        else:
                            unavailable[
                                prefix + "interior:cannot_preserve_newest_k"
                            ] += 1
                        for name, positions in regions.items():
                            metadata = {
                                "k": k,
                                "original_length": len(original),
                                "retained_length": n,
                                "input_width": maxlen,
                                "lags": (n - 1 - positions).tolist(),
                                "newest_k_preserved": bool(np.all(positions < n - k)),
                            }
                            add(
                                prefix + name + "_delete",
                                np.delete(h, positions),
                                metadata,
                            )
                            zeros = h.copy()
                            zeros[positions] = 0
                            add(prefix + name + "_zero", zeros, metadata)
                            replacements = h.copy()
                            unmatched = 0
                            for p in positions:
                                forbidden = np.concatenate((h, candidates))
                                options = pools[decile[int(h[p])]]
                                options = options[~np.isin(options, forbidden)]
                                if not len(options):
                                    unmatched += 1
                                    options = observed[~np.isin(observed, forbidden)]
                                if not len(options):
                                    break
                                replacements[p] = rng.choice(options)
                            else:
                                add(
                                    prefix + name + "_replace",
                                    replacements,
                                    dict(
                                        metadata, unmatched_frequency_decile=unmatched
                                    ),
                                )
                                continue
                            unavailable[prefix + name + "_replace:no_alternative"] += 1
                        for name, positions in (
                            ("recent", np.arange(n - k, n)),
                            ("old", np.arange(n - k)),
                        ):
                            if len(positions) < 2 or len(np.unique(h[positions])) < 2:
                                unavailable[
                                    prefix
                                    + name
                                    + "_shuffle:insufficient_distinct_items"
                                ] += 1
                                continue
                            for draw in range(20):
                                altered = h.copy()
                                altered[positions] = rng.permutation(h[positions])
                                add(
                                    prefix + name + "_shuffle",
                                    altered,
                                    {
                                        "k": k,
                                        "region_size": len(positions),
                                        "draw": draw,
                                        "lags": (n - 1 - positions).tolist(),
                                        "retained_length": n,
                                        "input_width": maxlen,
                                    },
                                )
                    scores = score_vectors(variants, cand_t)
                    if (
                        scores.shape != (len(variants), 99)
                        or not np.isfinite(scores).all()
                    ):
                        raise RuntimeError(
                            "Diagnostic intervention scores have invalid shape or nonfinite values"
                        )
                    effects = {}
                    for label in dict.fromkeys(labels):
                        indices = [
                            i for i, label_i in enumerate(labels) if label_i == label
                        ]
                        delta = scores[indices] - baseline
                        effects[label] = {
                            "mean_absolute_score_change": float(np.abs(delta).mean()),
                            "signed_mean_score_change": float(delta.mean()),
                            "changed_draws": len(indices),
                            "geometry": [geometry[i] for i in indices],
                        }
                    user_rows.append(
                        {
                            "user_id": str(uid),
                            "original_length": len(original),
                            "retained_length": len(h),
                            "effects": effects,
                            "ineligible": dict(unavailable),
                        }
                    )

                metrics["intact_user_endpoint_representations"] = (
                    representation(torch.stack(intact_user_states))
                    if intact_user_states
                    else {
                        "count": 0,
                        "value": None,
                        "reason": "No eligible identified training users",
                    }
                )
                definitions["intact_user_endpoint_representations"] = (
                    "Raw and unit-normalized covariance statistics and norm quantiles for "
                    "one intact final-nonpadding feature per identified training user "
                    "eligible for the fixed-candidate sensitivity probe. Loader endpoint "
                    "statistics remain separate and may contain repeated users."
                )
                definitions["paired_contrasts"] = (
                    "Within-user difference between each named method and the first method "
                    "in its declared comparison family, computed on common eligible users. "
                    "Bootstrap intervals resample each complete paired vector together. "
                    "These are score-sensitivity contrasts, not ranking improvements."
                )

                def aggregate(rows, names):
                    if not rows:
                        return {
                            "eligible_users": 0,
                            "estimates": None,
                            "reason": "No common eligible users",
                        }
                    columns = [
                        (name, field)
                        for name in names
                        for field in (
                            "mean_absolute_score_change",
                            "signed_mean_score_change",
                        )
                    ]
                    values = np.asarray(
                        [
                            [r["effects"][name][field] for name, field in columns]
                            for r in rows
                        ]
                    )
                    # Reset locally for each eligible cohort: paired columns share
                    # exactly the same identified-user bootstrap draws.
                    cohort_rng = np.random.default_rng(42)
                    draws = cohort_rng.integers(0, len(rows), size=(2000, len(rows)))
                    means = values[draws].mean(axis=1)
                    bounds = np.quantile(means, [0.025, 0.975], axis=0)
                    contrasts = {}
                    for j, (name, field) in enumerate(columns):
                        if name == names[0]:
                            continue
                        reference = columns.index((names[0], field))
                        paired_difference = values[:, j] - values[:, reference]
                        bootstrap_difference = means[:, j] - means[:, reference]
                        contrasts[name + "_minus_" + names[0] + ":" + field] = {
                            "mean": float(paired_difference.mean()),
                            "ci95": np.quantile(
                                bootstrap_difference, [0.025, 0.975]
                            ).tolist(),
                        }
                    return {
                        "eligible_users": len(rows),
                        "paired_contrasts": contrasts,
                        "estimates": {
                            name
                            + ":"
                            + field: {
                                "mean": float(values[:, j].mean()),
                                "ci95": bounds[:, j].tolist(),
                            }
                            for j, (name, field) in enumerate(columns)
                        },
                    }

                groups = {
                    "all": lambda r: True,
                    "1-10": lambda r: r["original_length"] <= 10,
                    "11-25": lambda r: 11 <= r["original_length"] <= 25,
                    "26+": lambda r: r["original_length"] >= 26,
                }
                methods = ["identity"] + [
                    str(fraction) + ":" + suffix
                    for fraction in (0.25, 0.5)
                    for suffix in (
                        "old_delete",
                        "recent_delete",
                        "random_delete",
                        "interior_delete",
                        "old_zero",
                        "recent_zero",
                        "random_zero",
                        "interior_zero",
                        "old_replace",
                        "recent_replace",
                        "random_replace",
                        "interior_replace",
                        "old_shuffle",
                        "recent_shuffle",
                    )
                ]
                aggregates, common = {}, {}
                for group, predicate in groups.items():
                    rows = [r for r in user_rows if predicate(r)]
                    aggregates[group] = {
                        name: aggregate(
                            [r for r in rows if name in r["effects"]], [name]
                        )
                        for name in methods
                    }
                    common[group] = {}
                    for fraction in (0.25, 0.5):
                        prefix = str(fraction) + ":"
                        families = {
                            "deletions": [
                                "old_delete",
                                "recent_delete",
                                "random_delete",
                                "interior_delete",
                            ],
                            "old_geometry_controls": [
                                "old_delete",
                                "old_zero",
                                "old_replace",
                            ],
                            "recent_geometry_controls": [
                                "recent_delete",
                                "recent_zero",
                                "recent_replace",
                            ],
                            "random_geometry_controls": [
                                "random_delete",
                                "random_zero",
                                "random_replace",
                            ],
                            "interior_geometry_controls": [
                                "interior_delete",
                                "interior_zero",
                                "interior_replace",
                            ],
                            "shuffles": ["old_shuffle", "recent_shuffle"],
                        }
                        for family, suffixes in families.items():
                            names = [prefix + suffix for suffix in suffixes]
                            paired = [
                                r
                                for r in rows
                                if all(name in r["effects"] for name in names)
                            ]
                            common[group][prefix + family] = aggregate(paired, names)
                metrics["user_score_sensitivity"] = {
                    "category": str(
                        getattr(
                            view,
                            "dataset_name",
                            getattr(network, "dataset_name", "current_category"),
                        )
                    ),
                    "selected_users": len(chosen),
                    "eligible_users": len(user_rows),
                    "skipped_users": dict(skips),
                    "maxlen": maxlen,
                    "candidate_source": "99 fixed training-observed unseen IDs per identified user",
                    "by_original_length": aggregates,
                    "common_eligible_comparisons": common,
                    "paired_user_records": user_rows,
                }
                metrics["four_category_macro"] = {
                    "value": None,
                    "reason": "This ModelView contains one category; aggregate four category reports externally only when all four have eligible observations",
                }
                metrics["validation_interventions"] = {
                    "value": None,
                    "reason": "No authorized pinned validation user/target/candidate interface supplied",
                }
                metrics["pos_neg_margin_category_breakdown"] = {
                    "value": None,
                    "reason": "Item categories are unavailable in this benchmark",
                }
                metrics["unobserved_traits"] = [
                    "ratings",
                    "review text",
                    "prices",
                    "stores",
                    "item categories",
                    "simulator diversity",
                    "simulator conformity",
                ]
                metrics["diagnostic_scope"] = (
                    "Exploratory diagnostics only; no model, checkpoint, sampling, training, predictor or evaluator changes"
                )
        finally:
            for module, mode in previous_modes:
                module.training = mode
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state, self.device)
        return {"metrics": metrics, "metric_definitions": definitions}

    ### <<< Self_EvolveRec-BLOCK-END
