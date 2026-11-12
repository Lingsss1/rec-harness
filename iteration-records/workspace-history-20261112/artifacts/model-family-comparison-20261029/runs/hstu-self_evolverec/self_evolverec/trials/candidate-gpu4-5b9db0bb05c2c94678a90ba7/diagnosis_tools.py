import torch
import numpy as np
import torch.nn.functional as F
import json
import random

### >>> Self_EvolveRec-BLOCK-START: Independent reproducible diagnostic sampling
from collections import Counter
import hashlib
from math import isfinite as math_isfinite
from itertools import islice


def _diagnostic_worker_init(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


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
    ### >>> Self_EvolveRec-BLOCK-START: Constrain interpretation to measured diagnostic evidence
    - Null measurements are unavailable, not evidence of healthy behavior.
    - Explain counts, unavailable reasons, and structural confounds before attributing effects to recency or content importance.
    - Deletion and replacement contrasts measure frozen-network responses, not causal importance or attention attribution.
    - Rows are not independent users; do not invent confidence intervals or cross-category macro estimates.
    - Item categories, prices, stores, ratings, reviews, and simulator traits are unavailable.
    - Do not infer recommendation diversity, satisfaction, ownership, or improved benchmark performance from these probes.
    - No numerical threshold here establishes overall model health; qualify the requested status using the available descriptive evidence.
    ### <<< Self_EvolveRec-BLOCK-END
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the HSTU ModelView boundary and public data limits
    def __init__(self, view, device):
        self.view = view
        self.model = view
        self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train
        self._cached_batches = None
        self._cache_metadata = None

    def _get_categories(self, item_id: int):
        # Category metadata is explicitly unavailable in the shared task.
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
            ### >>> Self_EvolveRec-BLOCK-START: Empty embedding probes are unavailable
            if n <= 2:
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

            ### >>> Self_EvolveRec-BLOCK-START: Report nonfinite embedding geometry as unavailable
            if not torch.isfinite(x).all().item() or not math_isfinite(mean_cos):
                return None
            collapse_score = (mean_cos + 1.0) * 0.5
            return round(float(collapse_score), 4)
            ### <<< Self_EvolveRec-BLOCK-END

    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(
        self,
        dataloader,
        num_batches: int = 10,
        eps: float = 0.5,
        topk_categories: int = 20,
    ):
        ### >>> Self_EvolveRec-BLOCK-START: Evaluate the actual trained HSTU module
        self.view.model.eval()
        ### <<< Self_EvolveRec-BLOCK-END

        margins_all = []
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Bound iteration without fetching an extra batch
        for step, batch in enumerate(islice(dataloader, num_batches)):
            ### <<< Self_EvolveRec-BLOCK-END
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 4):
                continue

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            ### >>> Self_EvolveRec-BLOCK-START: Tensor-based family scoring with valid sampled targets
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            itemnum = int(self.view.model.itemnum)
            valid = (
                seq_t.gt(0)
                & seq_t.le(itemnum)
                & pos_t.gt(0)
                & pos_t.le(itemnum)
                & neg_t.gt(0)
                & neg_t.le(itemnum)
                & pos_t.ne(neg_t)
            )
            log_feats = self.view.log2feats(seq_t)
            all_items = torch.stack(
                [pos_t.clamp(0, itemnum), neg_t.clamp(0, itemnum)], dim=-1
            )
            all_logits = self.view.predict_score(log_feats, all_items)
            margin = all_logits[..., 0] - all_logits[..., 1]
            valid = valid & torch.isfinite(margin)
            ### <<< Self_EvolveRec-BLOCK-END
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Keep category counters aligned with measured margins
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
                ### >>> Self_EvolveRec-BLOCK-START: Report unavailable margins with explicit counts
                "mean": None,
                "p10": None,
                "neg_beats_pos_rate": None,
                "small_margin_rate": None,
                "count": 0,
                ### <<< Self_EvolveRec-BLOCK-END
            }
        else:
            m = torch.cat(margins_all, dim=0)
            N = m.numel()

            k = max(1, int(np.ceil(0.10 * N)))
            p10 = float(torch.kthvalue(m, k).values.item())

            margin_metrics = {
                ### >>> Self_EvolveRec-BLOCK-START: Record the measured margin denominator
                "count": int(N),
                "mean": round(float(m.mean().item()), 4),
                ### <<< Self_EvolveRec-BLOCK-END
                "p10": round(p10, 4),
                "neg_beats_pos_rate": round(float((m < 0.0).float().mean().item()), 4),
                "small_margin_rate": round(
                    float((m < float(eps)).float().mean().item()), 4
                ),
            }

        def _topk(counter: Counter, k: int):
            return [{"category": c, "count": int(n)} for c, n in counter.most_common(k)]

        breakdown = {
            ### >>> Self_EvolveRec-BLOCK-START: Label unavailable category attribution
            "status": "unavailable",
            "reason": "Item category metadata is unavailable; category lists remain empty.",
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

    ### >>> Self_EvolveRec-BLOCK-START: Isolated HSTU perturbations with complete nonfinite accounting and explicit RNG restoration
    def _histories(self):
        histories = self.user_train.get("History", self.user_train)
        if not isinstance(histories, dict):
            raise TypeError("Training histories must be a mapping")
        return histories

    def _independent_examples(self, train_loader):
        if self._cached_batches is not None:
            return self._cached_batches
        itemnum = int(self.view.model.itemnum)
        width = int(self.view.model.maxlen)
        supplied_size = getattr(train_loader, "batch_size", None)
        batch_size = int(supplied_size) if supplied_size else 128
        if width < 1 or batch_size < 1:
            raise ValueError("Diagnostic width and batch size must be positive")
        histories = self._histories()
        users = sorted(histories, key=lambda key: (str(type(key)), str(key)))
        eligible = []
        for user in users:
            history = [int(item) for item in histories[user]]
            if any(item < 1 or item > itemnum for item in history):
                raise ValueError("Training history contains a non-real item ID")
            if len(history) >= 2 and len(set(history)) < itemnum:
                eligible.append(history)
        seqs, positives, negatives = [], [], []
        rng = random.Random(42)
        if eligible:
            for _ in range(10 * batch_size):
                history = rng.choice(eligible)
                interacted = set(history)
                seq = torch.zeros(width, dtype=torch.long)
                pos = torch.zeros_like(seq)
                neg = torch.zeros_like(seq)
                nxt = history[-1]
                for index, item in zip(
                    range(width - 1, -1, -1), reversed(history[:-1])
                ):
                    seq[index], pos[index] = item, nxt
                    candidate = rng.randint(1, itemnum)
                    while candidate in interacted:
                        candidate = rng.randint(1, itemnum)
                    neg[index] = candidate
                    nxt = item
                seqs.append(seq)
                positives.append(pos)
                negatives.append(neg)
        shape = (0, width)
        tensors = tuple(
            torch.stack(rows) if rows else torch.empty(shape, dtype=torch.long)
            for rows in (seqs, positives, negatives)
        )
        generator = torch.Generator().manual_seed(42)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*tensors),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            generator=generator,
            worker_init_fn=_diagnostic_worker_init,
        )
        batches = [(None, seq, pos, neg) for seq, pos, neg in islice(loader, 10)]
        fingerprint = hashlib.sha256()
        fingerprint.update(f"hstu-diagnostic-v1:{batch_size}:{width}:42".encode())
        for batch in batches:
            for tensor in batch[1:]:
                fingerprint.update(tensor.contiguous().numpy().astype("<i8").tobytes())
        self._cached_batches = batches
        self._cache_metadata = {
            "source": "Independent training-history sampler; live loader never iterated",
            "seed": 42,
            "batch_size": batch_size,
            "width": width,
            "batch_size_source": (
                "loader.batch_size" if supplied_size else "fallback 128"
            ),
            "batches": len(batches),
            "rows": sum(len(batch[1]) for batch in batches),
            "fingerprint_sha256": fingerprint.hexdigest(),
            "example_ids": "Ordered diagnostic row indices; stable user IDs not supplied",
            "eligible_training_users": len(eligible),
            "sampling": "Uniform eligible user, chronological shifted targets, unseen uniform negatives",
            "status": "available" if batches else "unavailable",
            "unavailable_reason": (
                None if batches else "No history with two events and an unseen negative"
            ),
        }
        return batches

    @staticmethod
    def _generator_states(*roots):
        # Inspect stored attributes and containers without invoking properties,
        # iterators, dataset accessors, or worker queues.
        import copy

        saved, seen = [], set()
        stack = list(roots)
        while stack:
            obj = stack.pop()
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            if isinstance(obj, torch.Generator):
                saved.append(("torch", obj, obj.get_state().clone()))
            elif isinstance(obj, random.Random):
                saved.append(("python", obj, obj.getstate()))
            elif isinstance(obj, np.random.Generator):
                saved.append(
                    ("numpy_generator", obj, copy.deepcopy(obj.bit_generator.state))
                )
            elif isinstance(obj, np.random.RandomState):
                saved.append(("numpy_state", obj, obj.get_state()))
            elif isinstance(obj, dict):
                stack.extend(obj.values())
            elif isinstance(obj, (list, tuple)):
                stack.extend(obj)
            elif not isinstance(obj, (torch.Tensor, np.ndarray, type)):
                attributes = getattr(obj, "__dict__", None)
                if isinstance(attributes, dict):
                    stack.extend(attributes.values())
        return saved

    @staticmethod
    def _summary(values):
        values = np.asarray(values, dtype=np.float64)
        finite = values[np.isfinite(values)]
        result = {
            "count": int(len(finite)),
            "nonfinite_count": int(len(values) - len(finite)),
            "status": "available" if len(finite) else "unavailable",
            "unavailable_reason": (
                None
                if len(finite)
                else (
                    "All attempted measurements were nonfinite"
                    if len(values)
                    else "No eligible measurements"
                )
            ),
            "mean": None,
            "median": None,
            "p10": None,
            "positive_contrast_fraction": None,
        }
        if len(finite):
            result.update(
                mean=float(finite.mean()),
                median=float(np.median(finite)),
                p10=float(np.quantile(finite, 0.1)),
                positive_contrast_fraction=float((finite > 0).mean()),
            )
        return result

    def _paired_perturbations(self, batches):
        itemnum = int(self.view.model.itemnum)
        counts = Counter(
            int(item) for history in self._histories().values() for item in history
        )
        ordered = sorted(range(1, itemnum + 1), key=lambda item: (counts[item], item))
        bins, item_bin = [], {}
        for group in np.array_split(np.asarray(ordered, dtype=np.int64), 10):
            group = [int(item) for item in group]
            bins.append(group)
            for item in group:
                item_bin[item] = len(bins) - 1
        deletion_rng = random.Random(42)
        replacement_rng = random.Random(42)
        effects = {
            name: []
            for name in (
                "deletion_recent_score_drop",
                "deletion_older_score_drop",
                "deletion_score_contrast",
                "deletion_recent_margin_drop",
                "deletion_older_margin_drop",
                "deletion_margin_contrast",
                "replacement_recent_score_drop",
                "replacement_older_score_drop",
                "replacement_recent_margin_drop",
                "replacement_older_margin_drop",
                "replacement_score_contrast",
                "replacement_margin_contrast",
            )
        }
        accounting = Counter(
            {
                "rows_seen": 0,
                "eligible_rows": 0,
                "skipped_invalid_targets": 0,
                "skipped_insufficient_history": 0,
                "skipped_invalid_history": 0,
                "deletion_nonfinite_rows": 0,
                "replacement_paired_rows": 0,
                "replacement_recent_eligible_rows": 0,
                "replacement_older_eligible_rows": 0,
                "replacement_recent_draws": 0,
                "replacement_older_draws": 0,
                "replacement_recent_unavailable_rows": 0,
                "replacement_older_unavailable_rows": 0,
                "replacement_recent_nonfinite_rows": 0,
                "replacement_older_nonfinite_rows": 0,
            }
        )
        for _, sequences, positives, negatives in batches:
            variants, candidates, records = [], [], []
            width = sequences.shape[1]

            def add(history, p, n):
                seq = torch.zeros(width, dtype=torch.long)
                if history:
                    seq[-len(history) :] = torch.tensor(history, dtype=torch.long)
                index = len(variants)
                variants.append(seq)
                candidates.append([p, n])
                return index

            for seq, pos, neg in zip(sequences, positives, negatives):
                accounting["rows_seen"] += 1
                valid = (
                    seq.gt(0)
                    & pos.gt(0)
                    & pos.le(itemnum)
                    & neg.gt(0)
                    & neg.le(itemnum)
                    & pos.ne(neg)
                )
                positions = valid.nonzero(as_tuple=False).flatten()
                if not len(positions):
                    accounting["skipped_invalid_targets"] += 1
                    continue
                t = int(positions[-1])
                history = seq[: t + 1]
                history = history[history.ne(0)].tolist()
                if any(item < 1 or item > itemnum for item in history):
                    accounting["skipped_invalid_history"] += 1
                    continue
                if len(history) < 2:
                    accounting["skipped_insufficient_history"] += 1
                    continue
                accounting["eligible_rows"] += 1
                p, n = int(pos[t]), int(neg[t])
                older = deletion_rng.randrange(len(history) - 1)
                record = {
                    "original": add(history, p, n),
                    "recent": add(history[:-1], p, n),
                    "older": add(history[:older] + history[older + 1 :], p, n),
                }
                excluded = set(history) | {p, n, 0}
                for label, event in (("recent", len(history) - 1), ("older", older)):
                    pool = [
                        item
                        for item in bins[item_bin[history[event]]]
                        if item not in excluded
                    ]
                    replacements = replacement_rng.sample(pool, min(3, len(pool)))
                    record[label + "_replacements"] = []
                    accounting[f"replacement_{label}_draws"] += len(replacements)
                    accounting[f"replacement_{label}_eligible_rows"] += bool(
                        replacements
                    )
                    accounting[
                        f"replacement_{label}_unavailable_rows"
                    ] += not replacements
                    for replacement in replacements:
                        changed = history.copy()
                        changed[event] = replacement
                        record[label + "_replacements"].append(add(changed, p, n))
                records.append(record)
            if not variants:
                continue
            outputs = []
            for start in range(0, len(variants), 128):
                seq = torch.stack(variants[start : start + 128]).to(self.device)
                items = torch.tensor(
                    candidates[start : start + 128], device=self.device
                )
                features = self.view.log2feats(seq)
                positions = torch.arange(width, device=self.device).expand_as(seq)
                last = positions.masked_fill(seq.eq(0), -1).amax(dim=1)
                if bool(last.lt(0).any()):
                    raise RuntimeError(
                        "Perturbation unexpectedly produced an empty history"
                    )
                selected = features[torch.arange(len(seq), device=self.device), last]
                scores = self.view.predict_score(selected, items)
                if tuple(scores.shape) != (len(seq), 2):
                    raise ValueError(
                        "ModelView.predict_score must return B x 2 candidate scores"
                    )
                outputs.append(scores.detach().double().cpu())
            scores = torch.cat(outputs).numpy()
            # Columns are fixed positive score and sampled positive-negative margin.
            response = np.stack((scores[:, 0], scores[:, 0] - scores[:, 1]), axis=1)
            for record in records:
                base = response[record["original"]]
                recent = response[record["recent"]]
                older = response[record["older"]]
                if np.isfinite(np.stack((base, recent, older))).all():
                    for column, kind in enumerate(("score", "margin")):
                        effects[f"deletion_recent_{kind}_drop"].append(
                            float(base[column] - recent[column])
                        )
                        effects[f"deletion_older_{kind}_drop"].append(
                            float(base[column] - older[column])
                        )
                        effects[f"deletion_{kind}_contrast"].append(
                            float(older[column] - recent[column])
                        )
                else:
                    accounting["deletion_nonfinite_rows"] += 1
                    for kind in ("score", "margin"):
                        for name in ("recent_", "older_", ""):
                            suffix = "drop" if name else "contrast"
                            effects[f"deletion_{name}{kind}_{suffix}"].append(
                                float("nan")
                            )
                drops = {}
                for label in ("recent", "older"):
                    indices = record[label + "_replacements"]
                    if not indices:
                        continue
                    changed = response[indices]
                    if not np.isfinite(base).all() or not np.isfinite(changed).all():
                        accounting[f"replacement_{label}_nonfinite_rows"] += 1
                        for kind in ("score", "margin"):
                            effects[f"replacement_{label}_{kind}_drop"].append(
                                float("nan")
                            )
                        continue
                    drops[label] = base - changed.mean(axis=0)
                    for column, kind in enumerate(("score", "margin")):
                        effects[f"replacement_{label}_{kind}_drop"].append(
                            float(drops[label][column])
                        )
                if len(drops) == 2:
                    accounting["replacement_paired_rows"] += 1
                    contrast = drops["recent"] - drops["older"]
                    for column, kind in enumerate(("score", "margin")):
                        effects[f"replacement_{kind}_contrast"].append(
                            float(contrast[column])
                        )
                elif all(
                    record[label + "_replacements"] for label in ("recent", "older")
                ):
                    for kind in ("score", "margin"):
                        effects[f"replacement_{kind}_contrast"].append(float("nan"))
        return {
            "counts": dict(accounting),
            "statistics": {
                name: self._summary(values) for name, values in effects.items()
            },
            "temporal_bias_control": {
                "status": "unavailable",
                "reason": "Verified HSTU interface accepts item sequences only; no aligned timestamp input or temporal-bias hook",
            },
            "retained_pair_bias_control": {
                "status": "unavailable",
                "reason": "No verified position-ID and retained-pair attention-bias interface",
            },
            "user_uncertainty": {
                "status": "unavailable",
                "reason": "No stable row-to-user identifiers supplied",
            },
            "four_category_macro": {
                "status": "unavailable",
                "reason": "This probe measures one frozen category model",
            },
            "candidate_ranking": {
                "status": "unavailable",
                "reason": "One positive and one negative cannot support benchmark ranking diagnostics",
            },
        }

    def run_full_diagnosis(self, train_loader):
        module = self.view.model
        modes = [(child, child.training) for child in module.modules()]
        python_state, numpy_state = random.getstate(), np.random.get_state()
        generators = self._generator_states(train_loader, self.view, module)
        devices = []
        if self.device.type == "cuda":
            devices = [
                (
                    self.device.index
                    if self.device.index is not None
                    else torch.cuda.current_device()
                )
            ]
        try:
            with torch.random.fork_rng(devices=devices):
                module.eval()
                with torch.inference_mode():
                    batches = self._independent_examples(train_loader)
                    collapse_score = self.analyze_embeddings_collapse_cosine(
                        sample_size=2048
                    )
                    margin_metrics, breakdown = (
                        self.analyze_pos_neg_margin_with_category_breakdown(
                            batches, num_batches=10, eps=0.5, topk_categories=20
                        )
                    )
                    perturbations = self._paired_perturbations(batches)
        finally:
            try:
                random.setstate(python_state)
                np.random.set_state(numpy_state)
                for kind, generator, state in generators:
                    if kind == "torch":
                        generator.set_state(state)
                    elif kind == "python":
                        generator.setstate(state)
                    elif kind == "numpy_generator":
                        generator.bit_generator.state = state
                    else:
                        generator.set_state(state)
            finally:
                # Restore mixed submodule modes even if RNG restoration fails.
                for child, flag in modes:
                    child.training = flag
        embedding_n = min(int(module.itemnum), self.view.item_emb.weight.shape[0] - 1)
        metrics = {
            "embedding_collapse_score": collapse_score,
            "embedding_probe": {
                "real_items": embedding_n,
                "sample_cap": 2048,
                "sampled_items": min(2048, embedding_n) if embedding_n > 2 else 0,
                "status": "available" if collapse_score is not None else "unavailable",
                "unavailable_reason": (
                    None
                    if collapse_score is not None
                    else "At most two real item embeddings or nonfinite sampled embedding geometry"
                ),
            },
            "pos_neg_margin_mean": margin_metrics["mean"],
            "pos_neg_margin_p10": margin_metrics["p10"],
            "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
            "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
            "pos_neg_margin_count": margin_metrics["count"],
            "pos_neg_margin_status": (
                "available" if margin_metrics["count"] else "unavailable"
            ),
            "pos_neg_margin_category_breakdown": breakdown,
            "diagnostic_examples": self._cache_metadata,
            "paired_event_perturbations": perturbations,
        }
        definitions = {
            "embedding_collapse_score": (
                "Mean off-diagonal cosine of at most 2048 real raw ID embeddings, sampled with local seed 42, "
                "mapped by (mean_cosine+1)/2. Padding/reserved IDs excluded. Directional similarity only; "
                "does not establish recommendation diversity, text-fused geometry, or overall health. Null when unavailable."
            ),
            "embedding_probe": "Embedding sampling denominator and availability; original 2048-item cap retained.",
            "pos_neg_margin_mean": "Mean positive logit minus sampled-negative logit over finite valid nonpadding training positions.",
            "pos_neg_margin_p10": "Nearest-rank tenth percentile of those position margins, preserving the existing margin definition.",
            "pos_neg_neg_beats_pos_rate": "Fraction of valid measured position margins below zero.",
            "pos_neg_small_margin_rate": "Fraction of valid measured position margins below eps=0.5; logits are uncalibrated.",
            "pos_neg_margin_count": "Number of finite valid positions used by the existing margin summaries; null summaries when zero.",
            "pos_neg_margin_status": "Availability of the measured margin summaries.",
            "pos_neg_margin_category_breakdown": "Position counts retained, but category lists are empty because item categories are unavailable.",
            "diagnostic_examples": (
                "Up to ten independent diagnostic batches sampled solely from public training histories. "
                "Separate seed-42 Python RNG and DataLoader torch.Generator, explicit worker initialization, "
                "num_workers=0, preserved order and batch size, SHA256 over dimensions and little-endian int64 tensors. "
                "The live training loader and sampler are never iterated. Cache fixed before scoring."
            ),
            "paired_event_perturbations": {
                "selection": (
                    "Per row use final occupied position t with valid distinct real positive/negative IDs, "
                    "and only nonpadding history in sequence[:t+1]. Require at least two events. "
                    "Choose one older event uniformly excluding the latest, using a separate local seed-42 RNG. "
                    "All variants use identical supplied positive/negative IDs and final nonpadding position readout."
                ),
                "deletion": (
                    "Delete latest versus selected older event, compact chronologically and left-pad to original width. "
                    "D_recent=s_original-s_recent_removed; D_older=s_original-s_older_removed. "
                    "Contrast=D_recent-D_older=s_older_removed-s_recent_removed. "
                    "Apply identically to positive-negative margins. Positive contrast means greater sensitivity to latest deletion."
                ),
                "replacement": (
                    "Sort real item IDs by (training interaction count,item ID), split into ten equally sized bins "
                    "with size difference at most one. For each selected event sample up to three distinct IDs "
                    "uniformly from its bin with a separate local seed-42 RNG, excluding observed prefix and both targets. "
                    "Replace item identity at the same occupied position; masks, width and absolute positions remain fixed. "
                    "Average score/margin drops within each row before aggregation. Paired contrasts require both controls. "
                    "Frequency matching uses training counts only, including zero-count real IDs."
                ),
                "statistics": (
                    "Each statistic reports finite count, nonfinite count, mean, median, linearly interpolated tenth percentile "
                    "and strict positive fraction. Empty statistics are null/unavailable. Nonfinite variants exclude the whole "
                    "corresponding paired row and increment explicit counters. Effects are descriptive row-level measurements; "
                    "rows may repeat users. Marginal replacement summaries can have different row sets; only paired contrast "
                    "summaries compare the common eligible rows."
                ),
                "structural_confounds": (
                    "Supplied HSTU uses absolute position embeddings and causal attention; no temporal input is exposed. "
                    "Latest deletion changes readout event and shifts retained absolute positions under left padding. "
                    "Interior deletion changes distances across the removed event and can shift earlier absolute positions. "
                    "Other HSTU query-time constructions could change temporal biases, which this interface cannot control. "
                    "These operations do not isolate content importance. Identity replacement preserves positions but changes "
                    "readout content when latest, and measures total network response rather than attention attribution. "
                    "Replacements are frequency-matched perturbations, not observed irrelevant events or confirmed negatives; "
                    "they can create unfamiliar combinations. No interior-padding approximation is used."
                ),
                "limits": (
                    "Frozen single category model, fixed candidates and perturbations. No stable user identifiers, bootstrap "
                    "intervals, calibrated cross-model effects, four-category macro diagnostic, or candidate-ranking metrics. "
                    "No aligned timestamps or verified retained-pair bias controls. Ratings/reviews, categories/prices/stores, "
                    "simulator traits, ownership and satisfaction are unobserved. No experiment or performance gain is claimed. "
                    "Do not select checkpoints for recency sensitivity; selection remains unchanged macro validation HR@10."
                ),
                "isolation": (
                    "Actual trained module evaluated under inference_mode. CPU and active-CUDA torch RNG forked; "
                    "Python/NumPy global states, stored reachable explicit Python/NumPy/torch generators and every individual "
                    "submodule training flag restored in finally, including exceptions. Independent local RNGs do not touch "
                    "training randomness. Training workers, queues and sampler progress are never consumed."
                ),
            },
        }
        return {"metrics": metrics, "metric_definitions": definitions}

    ### <<< Self_EvolveRec-BLOCK-END
