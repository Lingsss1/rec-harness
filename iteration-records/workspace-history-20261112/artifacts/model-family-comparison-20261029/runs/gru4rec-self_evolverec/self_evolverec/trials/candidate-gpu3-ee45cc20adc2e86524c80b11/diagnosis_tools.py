import torch
import numpy as np
import torch.nn.functional as F
import json

### >>> Self_EvolveRec-BLOCK-START: Reproducible temporal panel and trial-local artifacts
import os
import random
import tempfile
import time
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
    ### >>> Self_EvolveRec-BLOCK-START: Restrict interpretation to measured conditional training-prefix effects
    - Every statement must be grounded in the provided metric values/definitions and include relevant counts and correctness checks.
    - Temporal effects are conditional in-sample sensitivity of fitted checkpoints to selected training prefixes and fixed candidates. They do not establish generalization, causal user intent, or validation gains.
    - Inspect audit comparisons individually: exact same-tensor indexing failures differ from tolerance failures between independent forward executions. Skipped, empty and errored comparisons are unavailable coverage.
    - Report development and held-out diagnostic calibration separately, including unchanged-control failures, witnessed detections, survivors, inactive and invalid instances. Held-out variants belong to the same three operator families; this is limited custom diagnostic coverage.
    - A witnessed survivor is not a confirmed equivalent. Successful mutation detection establishes neither general correctness nor recommendation improvement.
    - Activation compares clean and altered outputs captured within one wrapper invocation. Detection requires the targeted assertion to reject the mutation while its frozen baseline and invocation controls pass; unrelated assertion failures do not establish detection. Failed clean controls leave attribution unresolved.
    - Export adapter score agreement does not establish strict checkpoint or parameter/buffer parity. Missing export coverage remains unresolved.
    - Failed required checks invalidate temporal interpretation. Null metrics mean unavailable evidence, not zero effects. Missing categories invalidate four-category estimates.
    - State or score movement without measured ranking damage establishes sensitivity only.
    - Full-minus-fixed-final compares two intervention distributions; it does not isolate final-item importance. Weak shuffle effects do not exclude unordered history use.
    - Include unchanged draws in primary estimates. Label changed-draw summaries separately and report endpoint preservation against the multiplicity-based expectation.
    - Rank and HR are unavailable unless the pinned evaluator supplies its exact ranking/tie helper. Never substitute an assumed ordering.
    - Percentile intervals describe empirical selected-user reweighting conditional on realized draws. Deterministic user selection provides no population-coverage justification.
    - Finite-permutation MCSE is separate from user reweighting intervals. Do not add their variances or treat independent intervention draws as paired permutations.
    - Distinguish measurements from alternative explanations: memorization, distribution shift, redundant histories, and scoring bottlenecks.
    - Do not infer exposure, compatibility, dislikes, current intent, mood, ratings, review text, or simulator traits.
    - Uncertainty conditions on the checkpoint, selected prefixes and frozen candidates; it excludes retraining, training-seed variation, candidate resampling and validation-search selection optimism.
    - Only independently measured official macro validation HR@10 selects models. Do not claim an improvement from these diagnostics.
    # Required checks cover shapes, final-position readout, actual-module scoring,
    # repeated calls, batch order, reset state, finite outputs and RNG restoration.
    # Exported-predictor parity is measured only when its callable is supplied.
    # Review summary: statically checked adapter shapes and finite values,
    # item-ID bounds, distinct sampled pairs, fixed diagnostic width,
    # same-tensor readout, actual-module parity and explicit error reporting.
    # Training, routing and one checkpoint per category remain unchanged.
    # No execution, timing result or validation improvement is claimed.
    # Four-category aggregation still requires all four labeled category records;
    # a single ModelView cannot produce that measurement independently.
    ### <<< Self_EvolveRec-BLOCK-END
    - If metrics look healthy, status should remain STABLE and implications should focus on "what is already working" plus one potential risk to monitor.
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve ModelView and use only available training information
    def __init__(self, view, device):
        self.model = view
        self.module = view.model
        self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train
        self._observed_maxlen = None
        self._margin_count = 0
        self._margin_frequency = {}

    def _get_categories(self, item_id: int):
        # Item-category metadata is unavailable in this matched task.
        return []

    def _histories(self):
        histories = self.user_train.get("History", self.user_train)
        return {
            u: [int(i) for i in history if int(i) != 0]
            for u, history in histories.items()
        }

    def _features(self, sequence):
        # Preserve the supplied ModelView's numpy-input adapter and score dtype.
        features = self.model.log2feats(sequence.detach().cpu().numpy())
        features = torch.as_tensor(features, device=self.device)
        if features.ndim != 3 or features.shape[:2] != sequence.shape:
            raise ValueError("ModelView.log2feats must return B x L x H")
        if not torch.isfinite(features).all():
            raise ValueError("ModelView.log2feats returned nonfinite features")
        return features

    def _last(self, features, sequence):
        if sequence.size(1) == 0:
            return features.new_zeros(sequence.size(0), features.size(-1))
        positions = torch.arange(sequence.size(1), device=sequence.device)
        last = positions.expand_as(sequence).masked_fill(sequence.eq(0), -1).amax(1)
        states = features[
            torch.arange(sequence.size(0), device=sequence.device), last.clamp_min(0)
        ]
        return torch.where(last[:, None].ge(0), states, torch.zeros_like(states))

    def _pad(self, histories, width):
        sequence = torch.zeros(
            len(histories), width, dtype=torch.long, device=self.device
        )
        for row, history in enumerate(histories):
            if history:
                sequence[row, -len(history) :] = torch.as_tensor(
                    history, dtype=torch.long, device=self.device
                )
        return sequence

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
        ### >>> Self_EvolveRec-BLOCK-START: Measure actual recurrent model with frequency proxy counts
        self.module.eval()
        frequency = Counter(i for h in self._histories().values() for i in h)
        frequency_margins = {"1-4": [], "5-19": [], "20+": []}
        margins_all = []
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        for step, batch in enumerate(dataloader):
            if step >= num_batches:
                break
            ### >>> Self_EvolveRec-BLOCK-START: Reject malformed diagnostic batches explicitly
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 4):
                raise ValueError("Expected (None, sequence, positive, negative)")
            ### <<< Self_EvolveRec-BLOCK-END

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            ### >>> Self_EvolveRec-BLOCK-START: Record existing window and preserve family-specific feature adapter
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            if seq_t.ndim != 2 or seq_t.size(0) < 1 or seq_t.size(1) < 1:
                raise ValueError("Training sequences must have nonempty B x L shape")
            if not ((seq_t >= 0) & (seq_t <= int(self.module.itemnum))).all():
                raise ValueError("Diagnostic sequence contains an invalid item ID")
            if self._observed_maxlen not in (None, int(seq_t.size(1))):
                raise ValueError(
                    "Diagnostic loader changed its external sequence width"
                )
            self._observed_maxlen = int(seq_t.size(1))
            log_feats = torch.cat(
                [self._features(chunk) for chunk in seq_t.split(64)], dim=0
            )
            ### <<< Self_EvolveRec-BLOCK-END

            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)

            ### >>> Self_EvolveRec-BLOCK-START: Bound sampled-margin scoring and reject invalid measurements
            if pos_t.shape != seq_t.shape or neg_t.shape != seq_t.shape:
                raise ValueError(
                    "Sequence, positive and negative tensors must share B x L shape"
                )
            itemnum = int(self.module.itemnum)
            if not (
                ((pos_t >= 0) & (pos_t <= itemnum)).all()
                and ((neg_t >= 0) & (neg_t <= itemnum)).all()
            ):
                raise ValueError("Diagnostic targets contain an invalid item ID")
            if ((pos_t == neg_t) & pos_t.ne(0) & seq_t.ne(0)).any():
                raise ValueError(
                    "Diagnostic positive and negative IDs must be distinct"
                )
            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = torch.cat(
                [
                    self.model.predict_score(features, items)
                    for features, items in zip(log_feats.split(64), all_items.split(64))
                ],
                dim=0,
            )
            if all_logits.shape != all_items.shape:
                raise ValueError("Per-position candidate scoring must return B x L x K")
            if not torch.isfinite(all_logits).all():
                raise ValueError("Sampled-margin scoring returned nonfinite values")
            margin = all_logits[..., 0] - all_logits[..., 1]
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Count valid sampled margins and stratify by training frequency
            if all_logits.shape != all_items.shape:
                raise ValueError("Per-position candidate scoring must return B x L x K")
            valid = (pos_t != 0) & (seq_t != 0) & (neg_t != 0)
            if valid.any():
                values = margin[valid].detach().float().cpu()
                margins_all.append(values)
                ids = pos_t[valid].detach().cpu().tolist()
                for item, value in zip(ids, values.tolist()):
                    count = frequency[item]
                    label = "1-4" if count < 5 else "5-19" if count < 20 else "20+"
                    frequency_margins[label].append(value)
            ### <<< Self_EvolveRec-BLOCK-END

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Apply the same valid-position mask to all margin summaries
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
                ### >>> Self_EvolveRec-BLOCK-START: Missing sampled margins are unavailable measurements
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

        ### >>> Self_EvolveRec-BLOCK-START: Report sampled-margin denominators and frequency strata
        self._margin_count = sum(m.numel() for m in margins_all)
        self._margin_frequency = {
            label: {
                "positions": len(values),
                "mean": float(np.mean(values)) if values else None,
                "neg_beats_pos_rate": (
                    float(np.mean(np.asarray(values) < 0)) if values else None
                ),
            }
            for label, values in frequency_margins.items()
        }
        breakdown["valid_positions"] = self._margin_count
        breakdown["metadata_status"] = (
            "Item-category metadata unavailable; breakdown lists intentionally empty"
        )
        if not self._margin_count:
            breakdown["unavailable_reason"] = (
                "No valid sampled positive/negative positions"
            )
        return margin_metrics, breakdown
        ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Calibrate independent readout, row and candidate assertions before temporal replay
    def _correctness_checks(self, width):
        import hashlib

        report = {
            "passed": False,
            "audit_version": 2,
            "atol": 1e-5,
            "rtol": 1e-4,
            "checks": [],
            "mutations": [],
        }
        category = getattr(self.model, "dataset_name", None)
        category = category or getattr(self.module, "dataset_name", None)
        allowed = {
            "Movies_and_TV",
            "Industrial_and_Scientific",
            "Electronics",
            "CDs_and_Vinyl",
        }
        label = category if category in allowed else "unlabeled"
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working")
        checkpoint = os.path.join(root, f"{label}_model.pt")
        directory = os.path.join(root, "diagnostics")

        def digest():
            if label == "unlabeled" or not os.path.isfile(checkpoint):
                return None
            value = hashlib.sha256()
            with open(checkpoint, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    value.update(chunk)
            return value.hexdigest()

        def persist(name, value):
            os.makedirs(directory, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", dir=directory, suffix=".json", delete=False
                ) as handle:
                    temporary = handle.name
                    json.dump(value, handle, allow_nan=False)
                os.replace(temporary, os.path.join(directory, f"{label}_{name}.json"))
            finally:
                if temporary and os.path.exists(temporary):
                    os.unlink(temporary)

        def compare(name, actual, reference, exact=False, required=True, sink=None):
            entry = {
                "name": name,
                "reference": "frozen independent oracle",
                "status": "error",
                "required": required,
                "elements": 0,
                "finite": None,
                "violations": None,
                "max_abs_error": None,
                "max_relative_error": None,
            }
            (report["checks"] if sink is None else sink).append(entry)
            a, b = torch.as_tensor(actual).detach(), torch.as_tensor(reference).detach()
            entry.update(
                shape=list(a.shape),
                reference_shape=list(b.shape),
                dtype=str(a.dtype),
                elements=a.numel(),
                exact=exact,
            )
            if a.shape != b.shape:
                entry["reason"] = "Shape mismatch"
                return entry
            if not a.numel():
                entry.update(status="skipped", reason="No compared elements")
                return entry
            entry["finite"] = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            if not entry["finite"]:
                entry["reason"] = "Nonfinite comparison"
                return entry
            a, b = a.to(device="cpu", dtype=torch.float64), b.to(
                device="cpu", dtype=torch.float64
            )
            error = (a - b).abs()
            bad = error.ne(0) if exact else error.gt(1e-5 + 1e-4 * b.abs())
            entry.update(
                status="fail" if bool(bad.any()) else "pass",
                violations=int(bad.sum()),
                max_abs_error=float(error.max()),
                max_relative_error=float((error / b.abs().clamp_min(1e-8)).max()),
            )
            return entry

        before = digest()
        report["checkpoint_sha256_before"] = before
        try:
            if width < 2 or int(self.module.itemnum) < 3:
                raise ValueError(
                    "Calibration requires width >= 2 and at least three items"
                )
            readout = getattr(self.module, "_readout", None)
            if not callable(readout):
                raise ValueError(
                    "Actual _readout(out, sequence) unavailable; indexing coverage unresolved"
                )
            public = sorted(self._histories().items(), key=lambda pair: str(pair[0]))
            public = [
                (str(u), h)
                for u, h in public
                if h and all(1 <= i <= self.module.itemnum for i in h)
            ][:12]
            if len({u for u, _ in public}) != len(public) or len(public) < 4:
                raise ValueError(
                    "Need four distinct serialized training users for disjoint audit groups"
                )
            manifest = {
                "audit_version": 2,
                "seed": 42,
                "width": width,
                "selection": "First 12 valid nonempty histories in sorted serialized-ID order",
                "partition": "Even indices development; odd indices held-out diagnostic",
                "assertions_frozen_before_assessment": True,
                "held_out_variants": "count index, row roll, candidate roll by two; same operator families as development",
                "candidates": list(range(1, min(17, self.module.itemnum) + 1)),
                "groups": {},
            }
            for split, offset in (("development", 0), ("held_out_diagnostic", 1)):
                group = public[offset::2]
                a, b = group[0][1][-width:], group[1][1][-width:]
                longest = max(group, key=lambda pair: len(pair[1]))[1][-width:]
                panel = [
                    [],
                    a[-1:],
                    a[-min(5, len(a)) :],
                    b[-min(5, len(b)) :],
                    [a[-1]] * min(4, width),
                    longest,
                    a[-min(5, len(a)) :],
                    b[-1:],
                ]
                manifest["groups"][split] = {
                    "users": [u for u, _ in group],
                    "histories": panel,
                }
            # Persist the complete fixture and mutation specification before any audit forward.
            persist("audit_manifest", manifest)
            report["manifest"] = manifest
            for split, fixture in manifest["groups"].items():
                seq = self._pad(fixture["histories"], width)
                features = self._features(seq).detach().clone()
                states = self._last(features, seq)
                candidates = torch.tensor(manifest["candidates"], device=self.device)
                scores = self.module.predict_score(states, candidates)
                compare(
                    f"{split}/same_tensor_readout",
                    readout(features, seq),
                    states,
                    exact=True,
                )
                compare(
                    f"{split}/padding",
                    features[seq.eq(0)],
                    torch.zeros_like(features[seq.eq(0)]),
                    exact=True,
                )
                compare(f"{split}/view_features", self.module.log2feats(seq), features)
                compare(f"{split}/repeat", self._features(seq), features)
                order = torch.arange(seq.size(0) - 1, -1, -1, device=self.device)
                compare(
                    f"{split}/row_permutation",
                    self._features(seq[order])[order.argsort()],
                    features,
                )
                compare(
                    f"{split}/duplicates",
                    self._features(seq.repeat(2, 1)),
                    features.repeat(2, 1, 1),
                )
                compare(
                    f"{split}/partition",
                    torch.cat([self._features(x) for x in seq.split(3)]),
                    features,
                )
                individual = torch.cat([self._features(x) for x in seq.split(1)])
                compare(f"{split}/singleton", individual, features)
                for row, history in enumerate(fixture["histories"]):
                    if history:
                        compact = self._pad([history], len(history))
                        compare(
                            f"{split}/padding_row_{row}",
                            self._features(compact)[0],
                            features[row, -len(history) :],
                        )
                compare(
                    f"{split}/empty",
                    self._features(seq[:1]),
                    torch.zeros_like(features[:1]),
                )
                compare(
                    f"{split}/prediction",
                    self.module.score_candidates(seq, candidates),
                    scores,
                )
                ids = candidates[None, None, :].expand(seq.size(0), width, -1)
                position_scores = self.model.predict_score(features, ids)
                # Preserve the explicit item-ID oracle while avoiding B*L
                # separate scorer calls and their GPU launch overhead.
                loop = torch.stack(
                    [
                        self.module.predict_score(features, item.reshape(1)).squeeze(-1)
                        for item in candidates
                    ],
                    dim=-1,
                )
                compare(f"{split}/position_scores", position_scores, loop)
                explicit = torch.stack(
                    [
                        torch.stack(
                            [
                                self.module.predict_score(
                                    state, item.reshape(1)
                                ).reshape(())
                                for item in candidates
                            ]
                        )
                        for state in states
                    ]
                )
                compare(f"{split}/explicit_candidate_ids", scores, explicit)
                compare(
                    f"{split}/view_scores",
                    self.model.predict_score(
                        states, candidates[None].expand(seq.size(0), -1)
                    ),
                    explicit,
                )
                compare(
                    f"{split}/candidate_chunks",
                    torch.cat(
                        [
                            self.module.predict_score(states, chunk)
                            for chunk in candidates.split(5)
                        ],
                        -1,
                    ),
                    explicit,
                )
                compare(
                    f"{split}/candidate_permutation",
                    self.module.predict_score(states, candidates.flip(0)).flip(-1),
                    explicit,
                )
                sorted_scores = scores.double().sort(dim=-1, descending=True).values
                fixture["score_observations"] = {
                    "exact_tied_pairs": int(
                        torch.triu(
                            scores[:, :, None].eq(scores[:, None, :]), diagonal=1
                        ).sum()
                    ),
                    "tenth_minus_eleventh": (
                        (sorted_scores[:, 9] - sorted_scores[:, 10]).cpu().tolist()
                        if candidates.numel() > 10
                        else None
                    ),
                    "ranks": None,
                    "reason": "Execution audit does not assume a pinned ranking adapter",
                }
                synthetic = torch.arange(
                    features.numel(), device=self.device, dtype=features.dtype
                ).reshape_as(features)
                synthetic_reference = self._last(synthetic, seq)
                compare(
                    f"{split}/structural_readout",
                    readout(synthetic, seq),
                    synthetic_reference,
                    exact=True,
                )
                originals = {
                    name: getattr(self.module, name)
                    for name in ("_readout", "log2feats", "predict_score")
                }
                own = {name: self.module.__dict__.get(name) for name in originals}
                present = {name: name in self.module.__dict__ for name in originals}

                def observations():
                    return [
                        self.module._readout(synthetic, seq),
                        self.module.log2feats(seq),
                        self.module.predict_score(states, candidates),
                    ]

                baseline = observations()
                references = [synthetic_reference, individual, explicit]
                names = ["readout", "row_identity", "candidate_identity"]
                # Frozen assertions and references never call a mutated helper.
                for fault in (
                    "unchanged",
                    "count_index",
                    "row_swap",
                    "candidate_mismatch",
                ):
                    local = []
                    record = {
                        "split": split,
                        "fault": fault,
                        "checks": local,
                        "status": "unresolved",
                        "eligible_witnessed": False,
                    }
                    report["mutations"].append(record)
                    try:
                        # Capture clean and altered outputs within the SAME
                        # wrapper invocation. Independent forward differences
                        # must not count as mutation activation.
                        witness_pair = []
                        if fault == "count_index":

                            def wrong_readout(out, sequence):
                                count = sequence.ne(0).sum(1)
                                index = (
                                    count - (1 if split == "development" else 0)
                                ).clamp(0, sequence.size(1) - 1)
                                value = out[
                                    torch.arange(
                                        sequence.size(0), device=sequence.device
                                    ),
                                    index,
                                ]
                                altered = torch.where(
                                    count[:, None].gt(0), value, torch.zeros_like(value)
                                )
                                clean = originals["_readout"](out, sequence)
                                witness_pair[:] = [
                                    clean.detach().clone(),
                                    altered.detach().clone(),
                                ]
                                return altered

                            self.module._readout = wrong_readout
                        elif fault == "row_swap":

                            def wrong_rows(sequence):
                                value = originals["log2feats"](sequence)
                                index = torch.arange(value.size(0), device=value.device)
                                if split == "development":
                                    index[1], index[2] = (
                                        index[2].clone(),
                                        index[1].clone(),
                                    )
                                else:
                                    index = index.roll(1)
                                altered = value[index]
                                witness_pair[:] = [
                                    value.detach().clone(),
                                    altered.detach().clone(),
                                ]
                                return altered

                            self.module.log2feats = wrong_rows
                        elif fault == "candidate_mismatch":

                            def wrong_candidates(value, items):
                                clean = originals["predict_score"](value, items)
                                altered = originals["predict_score"](
                                    value,
                                    items.roll(1 if split == "development" else 2, -1),
                                )
                                witness_pair[:] = [
                                    clean.detach().clone(),
                                    altered.detach().clone(),
                                ]
                                return altered

                            self.module.predict_score = wrong_candidates
                        observed = observations()
                        for name, actual, reference in zip(names, observed, references):
                            compare(
                                name,
                                actual,
                                reference,
                                exact=name == "readout",
                                sink=local,
                            )
                        failed = [
                            entry["name"]
                            for entry in local
                            if entry["status"] != "pass"
                        ]
                        record["fault_by_check"] = {
                            name: name in failed for name in names
                        }
                        if fault == "unchanged":
                            record["status"] = (
                                "control_fail" if failed else "control_pass"
                            )
                        else:
                            target = {
                                "count_index": 0,
                                "row_swap": 1,
                                "candidate_mismatch": 2,
                            }[fault]
                            if len(witness_pair) != 2:
                                raise RuntimeError("Mutation wrapper was not invoked")
                            clean, altered = witness_pair
                            if (
                                clean.shape != altered.shape
                                or not clean.numel()
                                or not bool(torch.isfinite(clean).all())
                                or not bool(torch.isfinite(altered).all())
                            ):
                                raise ValueError("Invalid mutation activation witness")
                            changed = not torch.equal(clean, altered)
                            record.update(
                                activation_witness=changed,
                                witness_max_abs_change=float(
                                    (altered.double() - clean.double()).abs().max()
                                ),
                                eligible_witnessed=changed,
                            )
                            # An already-failing oracle cannot establish that
                            # the injected fault caused assertion rejection.
                            baseline_check = compare(
                                names[target] + "_clean_control",
                                baseline[target],
                                references[target],
                                exact=target == 0,
                                sink=local,
                            )
                            invocation_check = compare(
                                names[target] + "_invocation_control",
                                clean,
                                references[target],
                                exact=target == 0,
                                sink=local,
                            )
                            if not changed:
                                record["status"] = "inactive"
                            elif any(
                                entry["status"] != "pass"
                                for entry in (baseline_check, invocation_check)
                            ):
                                record.update(
                                    status="unresolved",
                                    reason="Clean target assertion failed; mutation detection is not attributable",
                                )
                            elif local[target]["status"] in {"error", "skipped"}:
                                record.update(
                                    status="invalid",
                                    reason="Target assertion did not execute a valid comparison",
                                )
                            else:
                                record["status"] = (
                                    "detected"
                                    if local[target]["status"] == "fail"
                                    else "survivor"
                                )
                    except Exception as exc:
                        record.update(
                            status="invalid", reason=f"{type(exc).__name__}: {exc}"
                        )
                    finally:
                        for name in originals:
                            if present[name]:
                                setattr(self.module, name, own[name])
                            else:
                                self.module.__dict__.pop(name, None)
                exported = getattr(self.model, "diagnostic_exported_predictor", None)
                if callable(exported):
                    try:
                        compare(
                            f"{split}/export_adapter_scores",
                            torch.as_tensor(
                                exported(seq.cpu().numpy(), candidates.cpu().tolist()),
                                device=self.device,
                            ),
                            scores,
                            required=False,
                        )
                    except Exception as exc:
                        # Optional export coverage remains explicitly unresolved;
                        # retain completed required execution measurements.
                        reason = f"{type(exc).__name__}: {exc}"
                        report["checks"].append(
                            {
                                "name": f"{split}/export_adapter_scores",
                                "status": "error",
                                "required": False,
                                "reference": "Frozen actual-module candidate scores",
                                "elements": 0,
                                "finite": None,
                                "violations": None,
                                "max_abs_error": None,
                                "max_relative_error": None,
                                "reason": reason,
                            }
                        )
                        print(
                            f"Diagnosis export adapter unavailable: {reason}",
                            flush=True,
                        )
            report["export_coverage"] = {
                "parameter_buffer_verification": "unavailable: explicit checkpoint identity adapter not supplied",
                "score_adapter": (
                    "executed"
                    if callable(
                        getattr(self.model, "diagnostic_exported_predictor", None)
                    )
                    else "skipped"
                ),
                "strict_export_parity": None,
            }
            report["calibration_summary"] = {
                split: dict(
                    Counter(
                        r["status"] for r in report["mutations"] if r["split"] == split
                    )
                )
                for split in manifest["groups"]
            }
            required = [entry for entry in report["checks"] if entry["required"]]
            report["passed"] = bool(required) and all(
                entry["status"] == "pass" for entry in required
            )
            report["passed"] &= all(
                r["status"] in {"control_pass", "detected"} for r in report["mutations"]
            )
        except Exception as exc:
            report["reason"] = f"{type(exc).__name__}: {exc}"
        finally:
            after = digest()
            report["checkpoint_sha256_after"] = after
            report["checkpoint_unchanged"] = (
                before == after if before is not None else None
            )
            if before is not None and before != after:
                report["passed"] = False
                report["checkpoint_error"] = "Checkpoint bytes changed during audit"
            report["coverage"] = dict(
                Counter(entry["status"] for entry in report["checks"])
            )
            try:
                persist("readout_audit", report)
            except Exception as exc:
                report["artifact_error"] = f"{type(exc).__name__}: {exc}"
                report["passed"] = False
        return report

    def _temporal_sensitivity(self, width):
        category_names = [
            "Movies_and_TV",
            "Industrial_and_Scientific",
            "Electronics",
            "CDs_and_Vinyl",
        ]
        category = getattr(self.model, "dataset_name", None)
        if category is None:
            category = getattr(self.module, "dataset_name", None)
        if category not in category_names:
            category = None
        result = {
            "population": "Selected training-prefix targets; conditional in-sample sensitivity",
            "dataset_category": category,
            "category_label_reason": (
                None
                if category
                else "Dataset routing label not exposed by this ModelView"
            ),
            "seed": 42,
            "user_cap": 200,
            "bootstrap_replicates": 2000,
            "eligible_users": 0,
            "sampled_users": 0,
            "retained_users": 0,
            "exclusions": {},
            "unique_keys": True,
            "user_outcomes": [],
            "category_results": {},
            "available_category_summary": None,
            "four_category_mean": None,
            "four_category_interval": None,
            "four_category_unavailable_reason": "This probe receives one category ModelView; all four independent category records are required",
        }
        if width < 5:
            result["unavailable_reason"] = (
                "Existing input window has fewer than five positions"
            )
            return result
        rank_helper = getattr(self.model, "diagnostic_rank_helper", None)
        ranking_rule = getattr(self.model, "diagnostic_ranking_rule", None)
        if not (
            callable(rank_helper) and isinstance(ranking_rule, str) and ranking_rule
        ):
            rank_helper = None
            ranking_rule = None
        result["ranking_rule"] = ranking_rule
        result["ranking_unavailable_reason"] = (
            None if rank_helper else "Pinned evaluator ranking/tie adapter unavailable"
        )
        result["configuration"] = {
            "width": width,
            "minimum_retained_history": 5,
            "target_rule": "final_training_interaction",
            "selection_rule": "first_200_eligible_in_lexicographic_serialized_id_order",
            "serialization": "str(user_id); collisions rejected",
            "truncation": "latest_width_items_of_preceding_training_prefix",
            "candidate_rule": "target_first_plus_99_unique_uniform_training_unseen_items",
            "seed": 42,
            "seed_namespaces": {
                "category": category_names.index(category) if category else None,
                "candidates": 100,
                "full_shuffle": 200,
                "final_item_fixed": 201,
                "bootstrap": 300,
            },
            "permutations_per_intervention": 5,
            "bootstrap_replicates": 2000,
            "minimum_interval_users": 30,
            "primary_includes_unchanged_draws": True,
        }
        if category is None:
            result["unavailable_reason"] = (
                "Category identity required for independent reproducible random streams"
            )
            return result
        category_seed = category_names.index(category)
        histories = self._histories()
        serialized = [str(u) for u in histories]
        if len(set(serialized)) != len(serialized):
            raise ValueError("Serialized user IDs are not unique")
        itemnum = int(self.module.itemnum)
        excluded = Counter()
        eligible = []
        for user in sorted(histories, key=lambda u: str(u)):
            original = histories[user]
            if len(original) < 6:
                excluded["short_training_history"] += 1
            elif not all(1 <= item <= itemnum for item in original):
                excluded["invalid_training_item_id"] += 1
            elif itemnum - len(set(original)) < 99:
                excluded["fewer_than_99_unique_negatives"] += 1
            else:
                eligible.append(user)
        users = eligible[:200]
        result["total_training_users"] = len(histories)
        result["eligible_users"] = len(eligible)
        result["not_selected_eligible_users"] = len(eligible) - len(users)
        result["sampled_users"] = len(users)
        result["selected_user_order"] = [str(u) for u in users]
        keys = set()
        effects = []
        for stable_user_index, user in enumerate(users):
            rng = np.random.default_rng(
                np.random.SeedSequence([42, category_seed, stable_user_index, 100])
            )
            original = histories[user]
            target_index = len(original) - 1
            target = original[target_index]
            window = original[:target_index][-width:]
            if len(window) < 5:
                excluded["short_retained_prefix"] += 1
                continue
            if not all(1 <= item <= itemnum for item in original):
                excluded["invalid_training_item_id"] += 1
                continue
            # The unchanged full training history defines exclusions, just as in
            # the supplied validation sampler. Perturbations never change them.
            forbidden = set(original) | {0, target}
            available = itemnum - sum(1 <= item <= itemnum for item in forbidden)
            if available < 99:
                excluded["fewer_than_99_unique_negatives"] += 1
                continue
            candidates = [target]
            used = set(forbidden)
            # Complement sampling avoids rejection blowups for dense histories.
            if available * 5 < itemnum:
                pool = np.asarray(
                    [i for i in range(1, itemnum + 1) if i not in forbidden]
                )
                candidates.extend(int(i) for i in rng.choice(pool, 99, replace=False))
            else:
                while len(candidates) < 100:
                    item = int(rng.integers(1, itemnum + 1))
                    if item not in used:
                        candidates.append(item)
                        used.add(item)
            n = len(window)
            variants = [("unchanged", list(window))]
            permutation_records = [None]
            for intervention_id, condition in enumerate(
                ("full_shuffle", "final_item_fixed")
            ):
                seed_sequence = [
                    42,
                    category_seed,
                    stable_user_index,
                    200 + intervention_id,
                ]
                permutation_rng = np.random.default_rng(
                    np.random.SeedSequence(seed_sequence)
                )
                movable = n if condition == "full_shuffle" else n - 1
                for repetition in range(5):
                    permutation = permutation_rng.permutation(movable)
                    # Every assignment reads the unchanged original history.
                    altered = [window[int(i)] for i in permutation] + window[movable:]
                    changed_positions = sum(a != b for a, b in zip(window, altered))
                    variants.append((condition, altered))
                    permutation_records.append(
                        {
                            "repetition": repetition,
                            "seed_sequence": seed_sequence,
                            "movable_positions": list(
                                range(width - n, width - n + movable)
                            ),
                            "permutation": permutation.tolist(),
                            "identity_permutation": bool(
                                np.array_equal(permutation, np.arange(movable))
                            ),
                            "changed_positions": changed_positions,
                            "changed_fraction": changed_positions / n,
                            "unchanged": changed_positions == 0,
                            "endpoint_identity_changed": altered[-1] != window[-1],
                        }
                    )
            sequence = self._pad([h for _, h in variants], width)
            features = self._features(sequence)
            states = self._last(features, sequence)
            ids = torch.tensor(candidates, device=self.device, dtype=torch.long)
            # Eleven fixed-width histories, each with the same 100 candidates.
            scores = self.model.predict_score(
                states, ids[None, :].expand(len(variants), -1)
            )
            if scores.shape != (len(variants), 100) or not torch.isfinite(scores).all():
                raise ValueError(
                    "Nonfinite temporal scores or invalid B x K score shape"
                )
            # Preserve the native scoring dtype for the pinned ranking adapter.
            score_rows = scores.detach().cpu()
            scores = score_rows.to(torch.float64).numpy()
            ranks, hits, ties = [], [], []
            for index, row in enumerate(scores):
                rank = hit = None
                if rank_helper is not None:
                    rank, hit = rank_helper(
                        score_rows[index].clone(), target, list(candidates), k=10
                    )
                    if (
                        not np.isfinite(rank)
                        or int(rank) != rank
                        or not 1 <= rank <= 100
                    ):
                        raise ValueError(
                            "Pinned helper must return a one-based rank in [1,100]"
                        )
                    if hit not in (0, 1) or int(hit) != int(rank <= 10):
                        raise ValueError("Pinned helper returned inconsistent HR@10")
                    rank, hit = int(rank), int(hit)
                ranks.append(rank)
                hits.append(hit)
                boundary = np.sort(row)[-10]
                above, equal = int(np.sum(row > boundary)), int(np.sum(row == boundary))
                ties.append(
                    {
                        "target_exact_other_ties": int(np.sum(row == row[0])) - 1,
                        "boundary_tied_candidates": equal,
                        "boundary_spanning_tie": bool(above < 10 < above + equal),
                        "target_near_ties_atol_1e-8": int(
                            np.sum(np.abs(row - row[0]) <= 1e-8)
                        )
                        - 1,
                    }
                )
            if not torch.isfinite(states).all():
                raise ValueError("Temporal features contain nonfinite values")
            state_array = states.detach().to(torch.float64).cpu().numpy()
            base = state_array[0]
            base_norm = float(np.linalg.norm(base))
            condition_values = {}
            for index, (condition, _) in enumerate(variants):
                h = state_array[index]
                norm = float(np.linalg.norm(h))
                cosine = (
                    None
                    if min(norm, base_norm) < 1e-8
                    else float(
                        1.0 - np.clip(np.dot(h, base) / (norm * base_norm), -1.0, 1.0)
                    )
                )
                values = {
                    "rank_damage": (
                        None if ranks[index] is None else float(ranks[index] - ranks[0])
                    ),
                    "hr_damage": (
                        None if hits[index] is None else float(hits[0] - hits[index])
                    ),
                    "cosine_distance": cosine,
                    "normalized_displacement": float(
                        np.linalg.norm(h - base) / max(base_norm, 1e-8)
                    ),
                    "score_rms_change": float(
                        np.sqrt(np.mean((scores[index] - scores[0]) ** 2))
                    ),
                    "mean_positive_negative_margin": float(
                        np.mean(scores[index, 0] - scores[index, 1:])
                    ),
                    "target_highest_negative_margin": float(
                        scores[index, 0] - scores[index, 1:].max()
                    ),
                }
                condition_values.setdefault(condition, []).append(values)
            averaged = {}
            for condition, repeats in condition_values.items():
                averaged[condition] = {}
                for metric in repeats[0]:
                    values = [
                        entry[metric] for entry in repeats if entry[metric] is not None
                    ]
                    complete = len(values) == len(repeats) and np.isfinite(values).all()
                    averaged[condition][metric] = (
                        float(np.mean(values)) if complete else None
                    )
                    averaged[condition][metric + "_valid_repeats"] = sum(
                        bool(np.isfinite(value)) for value in values
                    )
            key = (category, str(user), target_index)
            if key in keys:
                raise ValueError("Duplicate category/user/target-index diagnostic key")
            keys.add(key)
            length_slice = "5-9" if n < 10 else "10-19" if n < 20 else "20+"
            effects.append(
                {
                    "user": str(user),
                    "stable_user_index": stable_user_index,
                    "target": target,
                    "target_index": target_index,
                    "raw_training_history_length": len(original),
                    "raw_prefix_length": target_index,
                    "truncated_items": target_index - n,
                    "sequence": sequence[0].detach().cpu().tolist(),
                    "candidate_ids": candidates,
                    "candidate_seed_sequence": [
                        42,
                        category_seed,
                        stable_user_index,
                        100,
                    ],
                    "score_dtype": str(score_rows.dtype),
                    "original_scores": scores[0].tolist(),
                    "endpoint_item_multiplicity": window.count(window[-1]),
                    "full_shuffle_endpoint_preservation_expectation": window.count(
                        window[-1]
                    )
                    / n,
                    "draws": [
                        {
                            "condition": condition,
                            "permutation": permutation_records[index],
                            "sequence": sequence[index].detach().cpu().tolist(),
                            "scores": scores[index].tolist(),
                            "score_differences": (scores[index] - scores[0]).tolist(),
                            "positive_negative_margins": (
                                scores[index, 0] - scores[index, 1:]
                            ).tolist(),
                            "rank": ranks[index],
                            "hit": hits[index],
                            "ties": ties[index],
                            **condition_values[condition][
                                sum(name == condition for name, _ in variants[:index])
                            ],
                        }
                        for index, (condition, _) in enumerate(variants)
                    ],
                    "history_length": n,
                    "length_slice": length_slice,
                    "target_status": (
                        "repeat" if target in original[:target_index] else "novel"
                    ),
                    "intact_rank": ranks[0],
                    "intact_hr10": hits[0],
                    "effects": averaged,
                }
            )
        result["exclusions"] = dict(excluded)
        result["retained_users"] = len(effects)
        result["user_outcomes"] = effects
        summary = self._summarize_users(effects)
        result["available_category_summary"] = summary
        if category:
            result["category_results"][category] = summary
        if not effects:
            result["unavailable_reason"] = (
                "No retained eligible training-prefix examples"
            )
        artifact_root = os.path.realpath(os.path.dirname(__file__))
        artifact_dir = os.path.join(artifact_root, "working", "diagnostics")
        os.makedirs(artifact_dir, exist_ok=True)
        artifact_path = os.path.join(
            artifact_dir, f"{category or 'unlabeled'}_temporal.json"
        )
        result["artifact_path"] = artifact_path
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=artifact_dir, suffix=".json", delete=False
            ) as handle:
                temporary_path = handle.name
                json.dump(result, handle, allow_nan=False)
            os.replace(temporary_path, artifact_path)
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)
        return result

    @staticmethod
    def _summarize_users(records):
        bootstrap_rng = np.random.default_rng(np.random.SeedSequence([42, 300]))
        metric_names = (
            "rank_damage",
            "hr_damage",
            "cosine_distance",
            "normalized_displacement",
            "score_rms_change",
            "mean_positive_negative_margin",
            "target_highest_negative_margin",
        )

        def summarize(rows):
            report = {
                "users": len(rows),
                "sparse": len(rows) < 30,
                "intact_hr10": (
                    float(np.mean([r["intact_hr10"] for r in rows]))
                    if rows and all(r["intact_hr10"] is not None for r in rows)
                    else None
                ),
                "conditions": {},
            }
            if not rows:
                report["unavailable_reason"] = "No users in this population"
                return report
            # The same resampled users are used for all paired outcome columns.
            draws = (
                bootstrap_rng.integers(0, len(rows), size=(2000, len(rows)))
                if len(rows) >= 30
                else None
            )
            for condition in rows[0]["effects"]:
                report["conditions"][condition] = {}
                for metric in metric_names:
                    values = np.asarray(
                        [
                            (
                                np.nan
                                if r["effects"][condition][metric] is None
                                else r["effects"][condition][metric]
                            )
                            for r in rows
                        ],
                        dtype=np.float64,
                    )
                    finite = np.isfinite(values)
                    count = int(finite.sum())
                    entry = {
                        "users": count,
                        "unavailable_users": len(rows) - count,
                        "mean": float(values[finite].mean()) if count else None,
                        "constant": (
                            bool(np.ptp(values[finite]) == 0) if count else None
                        ),
                        "interval": None,
                    }
                    if draws is not None and count == len(rows):
                        means = values[draws].mean(1)
                        entry["interval"] = [
                            float(v) for v in np.percentile(means, [2.5, 97.5])
                        ]
                    else:
                        entry["interval_reason"] = (
                            "Requires at least 30 unique users and complete outcomes in this population"
                        )
                    entry["interval_kind"] = (
                        "empirical_selected_user_reweighting_given_realized_draws"
                    )
                    report["conditions"][condition][metric] = entry
            report["change_and_tie_accounting"] = {}
            report["changed_draw_conditional"] = {}
            report["permutation_mc_se"] = {}
            blocks_by_condition = {}
            for condition in ("full_shuffle", "final_item_fixed"):
                blocks = [
                    [d for d in r["draws"] if d["condition"] == condition] for r in rows
                ]
                blocks_by_condition[condition] = blocks
                flat = [d for block in blocks for d in block]
                changed_blocks = [
                    [d for d in block if not d["permutation"]["unchanged"]]
                    for block in blocks
                ]
                changed = [d for block in changed_blocks for d in block]
                report["change_and_tie_accounting"][condition] = {
                    "users": len(rows),
                    "draws": len(flat),
                    "changed_draws": len(changed),
                    "unchanged_input_fraction": sum(
                        d["permutation"]["unchanged"] for d in flat
                    )
                    / len(flat),
                    "users_with_any_changed_draw": sum(
                        bool(block) for block in changed_blocks
                    ),
                    "identity_permutations": sum(
                        d["permutation"]["identity_permutation"] for d in flat
                    ),
                    "duplicate_induced_unchanged_draws": sum(
                        d["permutation"]["unchanged"]
                        and not d["permutation"]["identity_permutation"]
                        for d in flat
                    ),
                    "mean_changed_positions": float(
                        np.mean([d["permutation"]["changed_positions"] for d in flat])
                    ),
                    "mean_changed_fraction": float(
                        np.mean([d["permutation"]["changed_fraction"] for d in flat])
                    ),
                    "observed_endpoint_preservation": float(
                        np.mean(
                            [
                                not d["permutation"]["endpoint_identity_changed"]
                                for d in flat
                            ]
                        )
                    ),
                    "expected_endpoint_preservation": (
                        float(
                            np.mean(
                                [
                                    r["full_shuffle_endpoint_preservation_expectation"]
                                    for r in rows
                                ]
                            )
                        )
                        if condition == "full_shuffle"
                        else 1.0
                    ),
                    "draws_with_exact_target_ties": sum(
                        d["ties"]["target_exact_other_ties"] > 0 for d in flat
                    ),
                    "draws_with_boundary_spanning_ties": sum(
                        d["ties"]["boundary_spanning_tie"] for d in flat
                    ),
                    "original_users_with_exact_target_ties": sum(
                        r["draws"][0]["ties"]["target_exact_other_ties"] > 0
                        for r in rows
                    ),
                    "original_users_with_boundary_spanning_ties": sum(
                        r["draws"][0]["ties"]["boundary_spanning_tie"] for r in rows
                    ),
                }
                report["changed_draw_conditional"][condition] = {}
                report["permutation_mc_se"][condition] = {}
                for metric in metric_names:
                    conditional = [
                        float(np.mean([d[metric] for d in block]))
                        for block in changed_blocks
                        if block
                        and all(
                            d[metric] is not None and np.isfinite(d[metric])
                            for d in block
                        )
                    ]
                    report["changed_draw_conditional"][condition][metric] = {
                        "users": len(conditional),
                        "mean": float(np.mean(conditional)) if conditional else None,
                        "estimand": "equal_user_mean_among_users_with_changed_draws",
                    }
                    complete = all(
                        len(block) == 5
                        and all(
                            d[metric] is not None and np.isfinite(d[metric])
                            for d in block
                        )
                        for block in blocks
                    )
                    report["permutation_mc_se"][condition][metric] = {
                        "users": len(rows) if complete else 0,
                        "draws_per_user": 5,
                        "standard_error": (
                            float(
                                np.sqrt(
                                    sum(
                                        np.var([d[metric] for d in block], ddof=1) / 5
                                        for block in blocks
                                    )
                                )
                                / len(rows)
                            )
                            if complete
                            else None
                        ),
                        "unavailable_reason": (
                            None
                            if complete
                            else "Requires five finite draws for every selected user"
                        ),
                    }
            report["full_minus_final_fixed"] = {}
            for metric in metric_names:
                pairs = [
                    (
                        r["effects"]["full_shuffle"][metric],
                        r["effects"]["final_item_fixed"][metric],
                    )
                    for r in rows
                ]
                complete = all(
                    a is not None
                    and b is not None
                    and np.isfinite(a)
                    and np.isfinite(b)
                    for a, b in pairs
                )
                values = np.asarray([a - b for a, b in pairs]) if complete else None
                # The independent permutation streams have zero design covariance.
                mc_variance = 0.0
                mc_complete = complete
                for condition in ("full_shuffle", "final_item_fixed"):
                    for block in blocks_by_condition[condition]:
                        if len(block) != 5 or any(
                            d[metric] is None or not np.isfinite(d[metric])
                            for d in block
                        ):
                            mc_complete = False
                        else:
                            mc_variance += (
                                float(np.var([d[metric] for d in block], ddof=1)) / 5
                            )
                report["full_minus_final_fixed"][metric] = {
                    "users": len(rows) if complete else 0,
                    "mean": float(values.mean()) if complete else None,
                    "interval": (
                        [
                            float(v)
                            for v in np.percentile(values[draws].mean(1), [2.5, 97.5])
                        ]
                        if complete and draws is not None
                        else None
                    ),
                    "interval_reason": (
                        None
                        if complete and draws is not None
                        else "Requires at least 30 complete paired users"
                    ),
                    "interval_kind": "empirical_selected_user_reweighting_given_realized_draws",
                    "permutation_mc_standard_error": (
                        float(np.sqrt(mc_variance) / len(rows)) if mc_complete else None
                    ),
                    "interpretation": "Difference between intervention distributions; not isolated endpoint importance",
                }
            return report

        result = summarize(records)
        result["slices"] = {}
        for field, labels in (
            ("length_slice", ["5-9", "10-19", "20+"]),
            ("target_status", ["repeat", "novel"]),
        ):
            result["slices"][field] = {
                label: summarize([r for r in records if r[field] == label])
                for label in labels
            }
        return result

    def run_full_diagnosis(self, train_loader):
        previous_modes = [(module, module.training) for module in self.module.modules()]
        backend_state = (
            torch.get_float32_matmul_precision(),
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark,
        )
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        cpu_state = torch.random.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        metrics = {}
        try:
            # Reset per-run state so a failed or empty loader cannot expose
            # counts or a window retained from a previous diagnostic call.
            self._observed_maxlen = None
            self._margin_count = 0
            self._margin_frequency = {}
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(42)
            self.module.eval()
            with torch.inference_mode():
                collapse_score = self.analyze_embeddings_collapse_cosine(
                    sample_size=2048
                )
                eps = 0.5
                try:
                    margin_metrics, breakdown = (
                        self.analyze_pos_neg_margin_with_category_breakdown(
                            train_loader, num_batches=10, eps=eps, topk_categories=20
                        )
                    )
                except Exception as exc:
                    margin_metrics = dict.fromkeys(
                        ["mean", "p10", "neg_beats_pos_rate", "small_margin_rate"]
                    )
                    self._margin_count = 0
                    self._margin_frequency = {}
                    reason = f"{type(exc).__name__}: {exc}"
                    print(
                        f"Diagnosis sampled-margin probe unavailable: {reason}",
                        flush=True,
                    )
                    breakdown = {
                        "unavailable_reason": reason,
                        "valid_positions": 0,
                        "low_margin_top_categories": {"pos_item": [], "neg_item": []},
                        "wrong_rank_top_categories": {"pos_item": [], "neg_item": []},
                    }
                metrics.update(
                    {
                        "embedding_collapse_score": collapse_score,
                        "embedding_sample_count": min(2048, int(self.module.itemnum)),
                        "pos_neg_margin_mean": margin_metrics["mean"],
                        "pos_neg_margin_p10": margin_metrics["p10"],
                        "pos_neg_neg_beats_pos_rate": margin_metrics[
                            "neg_beats_pos_rate"
                        ],
                        "pos_neg_small_margin_rate": margin_metrics[
                            "small_margin_rate"
                        ],
                        "pos_neg_margin_category_breakdown": breakdown,
                        "pos_neg_margin_frequency_breakdown": self._margin_frequency,
                    }
                )
                width = self._observed_maxlen
                if width is None:
                    width = getattr(self.model, "maxlen", None)
                if width is None:
                    width = getattr(getattr(self.model, "args", None), "maxlen", 0)
                width = int(width or 0)
                checks = self._correctness_checks(width)
                metrics["execution_checks"] = checks
                if checks["passed"]:
                    try:
                        metrics["temporal_sensitivity"] = self._temporal_sensitivity(
                            width
                        )
                    except Exception as exc:
                        reason = f"{type(exc).__name__}: {exc}"
                        print(
                            f"Diagnosis temporal probe unavailable: {reason}",
                            flush=True,
                        )
                        metrics["temporal_sensitivity"] = None
                        metrics["temporal_unavailable_reason"] = reason
                else:
                    metrics["temporal_sensitivity"] = None
                    metrics["temporal_unavailable_reason"] = (
                        "Execution checks failed; original checkpoint semantics were not modified"
                    )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            metrics["diagnostic_inference_seconds"] = float(
                time.perf_counter() - started
            )
        finally:
            torch.set_float32_matmul_precision(backend_state[0])
            torch.backends.cuda.matmul.allow_tf32 = backend_state[1]
            torch.backends.cudnn.allow_tf32 = backend_state[2]
            torch.backends.cudnn.deterministic = backend_state[3]
            torch.backends.cudnn.benchmark = backend_state[4]
            for module, was_training in previous_modes:
                module.training = was_training
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
        return {
            "metrics": metrics,
            "metric_definitions": {
                "embedding_collapse_score": "Sample up to 2048 real-item embeddings, seed 42, excluding padding/reserved tokens. Map mean off-diagonal pairwise cosine to [0,1]; denominator k*(k-1). Higher indicates directional alignment, not measured recommendation diversity. Existing convention returns zero for at most two items.",
                "embedding_sample_count": "Number of real-item embeddings sampled for the collapse probe; pairwise denominator is k*(k-1).",
                "pos_neg_margin_mean": "Mean positive-minus-negative model score across valid training-loader positions in at most ten batches; positive, negative and sequence IDs must be nonzero. Larger means stronger sampled separation; denominator is valid_positions in the breakdown.",
                "pos_neg_margin_p10": "Nearest-rank lower tenth percentile of the same valid training-position margins, index ceil(0.1*N); lower indicates weaker sampled separation. Null if N=0.",
                "pos_neg_neg_beats_pos_rate": "Fraction of the same N valid training positions whose sampled negative score exceeds positive score; lower is better. Null if N=0.",
                "pos_neg_small_margin_rate": "Fraction of the same N valid training positions with positive-minus-negative score below eps=0.5; lower indicates fewer small margins. Null if N=0.",
                "pos_neg_margin_category_breakdown": "Counts low-margin and wrong-rank valid training positions and their denominator. Item-category metadata is unavailable, so category lists remain empty; these lists are unrelated to dataset routing.",
                "pos_neg_margin_frequency_breakdown": "Stratify the same sampled training positions by positive-item training-interaction frequency: 1-4, 5-19, 20+. Each stratum reports its position denominator, mean margin and negative-beats-positive fraction. Frequency is not exposure.",
                "execution_checks": "Audit version 2 freezes and persists public-training fixtures before audit forwards. First 12 eligible serialized user IDs are partitioned by even/odd index into development and held-out diagnostic groups. Each has empty, singleton, mixed, repeated, duplicate, tied-length and longest-available histories; longest available need not fill the model window. Exact same-tensor actual _readout checks use independently gathered maximum-nonzero positions and position-distinct structural tensors, with zero empty representations. Separate-forward comparisons use abs(error)<=1e-5+1e-4*abs(reference); relative errors divide by max(abs(reference),1e-8). Reports include shapes, dtype, finiteness, coverage, violations and nullable error statistics. Repeat, row permutation, duplicates, partition/singleton batches, changed padding, explicit item-ID scoring, position scoring, partial candidate chunks and candidate permutations are separate checks. Temporary actual-module wrappers inject count-index, row-identity and candidate-ID faults one at a time; references are frozen independently. Development uses count-1, row swap and candidate roll by one; held-out reserves count, row roll and candidate roll by two from the same operator families. Exact changed outputs establish activation before detection classification; below-tolerance survivors remain survivors. Inactive, invalid and unchanged-control outcomes are reported separately. No finite-suite survival establishes equivalence. Calibration covers actual module methods, not arbitrary adapters. Required comparison failures, calibration survivors/inactive/invalid instances or unchanged-control failures disable temporal replay. Optional export adapter scores are separate nonrequired coverage; strict parameter/buffer and checkpoint identity verification remains unavailable. Existing checkpoint bytes are hashed before/after, with missing files reported as unavailable. No model semantics or checkpoint weights are changed. Wrappers are restored in finally; outer diagnosis restores module flags, RNG streams and backend settings. Successful calibration establishes no validation gain.",
                "temporal_sensitivity": {
                    "population": "First at most 200 eligible users in lexicographic str(user_id) order; serialized-ID collisions raise an error. Eligibility requires at least six real training interactions, valid IDs and at least 99 training-unseen real items. Target is the final training interaction; retain the latest width preceding items, requiring at least five. Selection is deterministic. Raw/prefix/retained lengths, truncation, selection order, category and exclusion counts are saved. Targets may have been used during fitting.",
                    "candidates": "Target first plus 99 unique uniform training-unseen negatives, frozen across conditions. SeedSequence([42, category_index, stable_user_index, 100]), with category_index in the fixed four-category routing order. No official evaluation candidates or labels are read. Insufficient pools are excluded.",
                    "interventions": "Five uniform occurrence-position permutations of all n real positions and five of the first n-1 positions, fixing the final item. Independent streams SeedSequence([42, category_index, stable_user_index, 200/201]); draw order within each stream is recorded. Each draw reads the original history; width, padding and multiplicities remain fixed. Identity and duplicate-induced unchanged histories are retained. Full-shuffle endpoint preservation expectation is endpoint multiplicity divided by n; fixed-final expectation is one. These interventions can create distribution shift and are not null-hypothesis permutation tests.",
                    "ranking": "Rank/HR are null unless ModelView supplies diagnostic_rank_helper(scores, target, candidates, k=10) and a nonempty diagnostic_ranking_rule from the pinned evaluator. Helper receives native-dtype scores and returns one-based rank and binary hit. No substitute tie rule is assumed. Diagnostic HR is not official validation HR.",
                    "rank_damage": "Perturbed rank minus intact rank; positive means damage. Average all five draws within each user, then users equally. Null without the pinned ranking adapter.",
                    "hr_damage": "Intact hit minus perturbed hit at ten; positive means damage. Average all five draws within each user, then users equally. Null without the pinned ranking adapter.",
                    "scores_and_ties": "Save native score dtype, candidate IDs, full float64-converted score rows, paired score differences and 99 positive-minus-negative margins per draw. mean_positive_negative_margin averages those 99 margins; target_highest_negative_margin subtracts the maximum negative score. score_rms_change is RMS paired score movement over all 100 candidates. Exact target-score ties exclude the target. Boundary-spanning ties have fewer than ten scores strictly greater than the tenth-highest value and more than ten at or above it. Supplementary near ties use absolute tolerance 1e-8 and never alter ranks.",
                    "cosine_distance": "One minus cosine between perturbed and intact final states after direct conversion to float64. Null when either norm is below 1e-8. Primary user means require every prescribed draw to be finite; partial-repeat averages are unavailable. State movement alone does not establish ranking damage.",
                    "normalized_displacement": "L2 state difference divided by max(intact-state norm,1e-8), averaged within users then equally across users. Denominators and unavailable counts accompany each metric.",
                    "contrast": "full_minus_final_fixed is the within-user difference between expected outcomes under the two intervention distributions. It does not isolate final-item importance. Weak shuffle effects do not exclude unordered history use.",
                    "permutation_mc_se": "For U users and five draws, intervention MCSE=sqrt(sum_u(s_u^2/5))/U using sample variance ddof=1. Independent-stream contrast MCSE=sqrt(sum_u((s_full,u^2+s_fixed,u^2)/5))/U. No draw-wise covariance is estimated or assumed from user pairing. All users require five finite outcomes. Five-draw variance estimates are noisy; no precise tail-coverage claim is made.",
                    "changed_draw_conditional": "Separately labeled equal-user mean among users with any changed draws and complete changed-draw outcomes. This changes the population and estimand; primary means always retain unchanged draws.",
                    "summaries": "Equal selected-user category means and descriptive retained-length 5-9/10-19/20+ and repeat/novel slices. Repeat means the final training target occurred earlier, including outside the retained window. Report counts and sparse/constant flags. User records, exact sequences, candidates, draws and configuration are persisted atomically beneath this trial's working/diagnostics directory. They contain no held-out labels.",
                    "uncertainty": "2000 CPU bootstrap replicates from SeedSequence([42,300]), resampling whole user records with all realized draws retained; identical resampled indices across conditions and contrasts. Never resample permutations within users. Percentile 2.5th-97.5th intervals require at least 30 complete unique users. These are empirical selected-user reweighting intervals conditional on realized draws, fitted checkpoint and fixed candidates. Deterministic selection provides no probability-sampling justification for population coverage. MCSE is reported separately and is not added to reweighting variance.",
                    "macro": "Category-local probes return macro fields null. External aggregation requires exactly Movies_and_TV, Industrial_and_Scientific, Electronics and CDs_and_Vinyl, with equal category weights. Reweight users independently within each category; intervals require at least 30 complete users in every contributing category/slice. For independent category streams, macro MCSE=sqrt(sum_c(MCSE_c^2))/4. No missing category is imputed.",
                    "limits": "Conditional fitted-model training-prefix sensitivity, not generalization, causal user intent, harmful-shortcut evidence or validation improvement. Uncertainty excludes retraining, training-seed variation, candidate resampling and adaptive validation selection. No ratings, reviews, prices, stores, item categories or simulator traits are inferred. The separate twenty-user full-catalog simulator workflow is unchanged. Official four-category macro validation HR@10 alone selects models.",
                },
                "mutation_calibration": "Calibration revision: activation compares clean and altered outputs captured in one wrapper invocation, with matching shapes, positive coverage and finite values. Detection requires rejection by the targeted assertion and passing clean baseline and invocation controls against the frozen independent reference. Unrelated failures cannot detect a fault. Failed clean controls leave attribution unresolved; below-tolerance activated faults remain survivors. Optional export errors retain explicit unavailable coverage. Review summary: statically checked wrapper invocation, oracle independence, target-specific attribution, restoration paths, candidate tensor shapes and unchanged single-checkpoint model semantics. No execution or validation gain is claimed.",
                "temporal_unavailable_reason": "Reason the entire temporal probe is null; failed execution checks or missing measurements must not be interpreted as zero effects.",
                "diagnostic_inference_seconds": "Elapsed wall seconds for diagnostic probes, including CPU summaries and synchronized GPU work; excludes model training. Diagnostics consume the shared search budget.",
            },
        }

    ### <<< Self_EvolveRec-BLOCK-END
