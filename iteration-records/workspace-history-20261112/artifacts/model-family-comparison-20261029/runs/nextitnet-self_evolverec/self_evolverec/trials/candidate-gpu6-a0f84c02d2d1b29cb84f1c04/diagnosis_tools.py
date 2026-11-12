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
    ### >>> Self_EvolveRec-BLOCK-START: Interpret measured training diagnostics with explicit limitations
    - Null measurements are unavailable, never evidence of healthy behavior.
    - Separate implementation contract failures from in-sample history sensitivity.
    - Embedding alignment does not establish recommendation diversity or preferences.
    - Training-prefix outcomes are not validation results or evidence of improvement.
    - Simulator traits, ratings, reviews, prices, stores and item categories are unobserved.
    - Bootstrap intervals exclude training-seed and candidate-sampling uncertainty.
    - Unverified export parity or causal controls must remain explicitly unverified.
    - Do not infer a four-category macro estimate from a subset of categories.
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

    ### >>> Self_EvolveRec-BLOCK-START: Memory-bounded NextItNet controls and contract-gated paired sensitivity
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
        # The supplied ModelView supports [B,L,H] with [B,L,N]. In particular,
        # avoid assuming that it accepts a reduced [B,H] feature tensor.
        items = items.unsqueeze(1).expand(-1, sequence.size(1), -1)
        logits = self.view.predict_score(features, items)
        if logits.shape != items.shape:
            raise ValueError("predict_score must preserve B x L x N candidate axes")
        scores = logits[torch.arange(sequence.size(0), device=self.device), last]
        if not torch.isfinite(scores).all():
            raise ValueError("Nonfinite diagnostic scores")
        return scores

    @staticmethod
    def _seed(*parts):
        import hashlib

        message = json.dumps([42, *parts], ensure_ascii=True, separators=(",", ":"))
        return int.from_bytes(hashlib.sha256(message.encode()).digest()[:8], "little")

    def _dependencies(self):
        offsets = {0}
        description = []
        if type(self.network).__name__ != "NextItNetModel":
            raise ValueError(
                "This diagnostic adapter requires the supplied NextItNet family"
            )
        for block in self.network.blocks:
            residual = set(offsets)
            for name in ("conv1", "conv2"):
                conv = getattr(block, name)
                if not isinstance(conv, torch.nn.Conv1d):
                    raise ValueError("Unknown NextItNet convolution structure")
                if conv.stride != (1,) or conv.padding != (0,):
                    raise ValueError(
                        "Dependency derivation requires causal stride-one convolutions"
                    )
                taps = {i * conv.dilation[0] for i in range(conv.kernel_size[0])}
                offsets = {a + b for a in offsets for b in taps}
                description.append(
                    {"kernel": conv.kernel_size[0], "dilation": conv.dilation[0]}
                )
            offsets |= residual
        return sorted(offsets), description

    def _controls(self, offsets):
        import copy

        length = int(self.network.maxlen)
        itemnum = int(self.network.itemnum)
        tolerance = 1e-5
        result = {"atol": tolerance, "rtol": 1e-5}
        normalization = []
        for name, module in self.network.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                safe = tuple(module.normalized_shape) == (
                    int(self.network.hidden_units),
                )
                normalization.append(
                    {"name": name, "kind": "LayerNorm", "channel_only_shape": safe}
                )
            elif isinstance(
                module,
                (
                    torch.nn.modules.batchnorm._BatchNorm,
                    torch.nn.modules.instancenorm._InstanceNorm,
                    torch.nn.GroupNorm,
                ),
            ):
                normalization.append(
                    {
                        "name": name,
                        "kind": type(module).__name__,
                        "potential_temporal_or_batch_coupling": True,
                    }
                )
        result["normalization"] = normalization
        if itemnum < 2 or length < 3:
            result["causality"] = None
            result["reason"] = "At least two items and three positions are required"
            return result
        x = torch.ones((1, length), device=self.device, dtype=torch.long)
        candidates = list(range(1, min(itemnum, 7) + 1))
        # Include short left-padded histories: a full-length history alone cannot
        # expose the count-minus-one readout bug.
        readout_checks = []
        for history in ([1], [1, 2], [1] * length):
            sequence = self._tensor([history])
            expected = self._scores(sequence, candidates)[0]
            observed = self.network.score_candidates(
                sequence,
                torch.as_tensor(candidates, device=self.device, dtype=torch.long),
            )
            if observed.shape != expected.shape:
                raise ValueError("Deployed candidate scorer returned an invalid shape")
            readout_checks.append(
                {
                    "history_length": len(history),
                    "max_abs_error": float((expected - observed).abs().max().item()),
                    "match": bool(
                        torch.allclose(expected, observed, atol=tolerance, rtol=1e-5)
                    ),
                }
            )
        result["left_padded_readout_checks"] = readout_checks
        scores = self._scores(x, candidates)[0]
        deployed = self.network.score_candidates(
            x, torch.as_tensor(candidates, device=self.device, dtype=torch.long)
        )
        result["readout_score_max_abs_error"] = float(
            (scores - deployed).abs().max().item()
        )
        result["readout_score_match"] = bool(
            torch.allclose(scores, deployed, atol=tolerance, rtol=1e-5)
        )
        paired = self._scores(x.expand(2, -1).clone(), candidates)
        result["batched_feature_score_max_abs_error"] = float(
            (paired - scores).abs().max().item()
        )
        result["batched_feature_score_match"] = bool(
            torch.allclose(paired, scores.expand_as(paired), atol=tolerance, rtol=1e-5)
        )

        # Controls use only item IDs 0, 1 and 2. Substitute a detached miniature
        # embedding while copying, avoiding several full-vocabulary GPU copies.
        # Every other parameter and buffer retains its trained value.
        def control_copy():
            miniature = torch.nn.Embedding.from_pretrained(
                self.network.item_emb.weight[:3].detach().clone(),
                freeze=True,
                padding_idx=0,
            )
            return copy.deepcopy(self.network, {id(self.network.item_emb): miniature})

        checks = []
        for training in (False, True):
            # All probes that enter training mode use disposable network copies.
            probe = control_copy()
            probe.train(training)
            for module in probe.modules():
                if isinstance(module, torch.nn.modules.dropout._DropoutNd):
                    module.eval()
            for position in sorted({0, length // 2, length - 2}):
                edited = x.clone()
                edited[:, position + 1 :] = 2
                # Separate copies give both forwards identical initial buffers.
                reference = copy.deepcopy(probe)
                other = copy.deepcopy(probe)
                a = reference.log2feats(x)[:, : position + 1]
                b = other.log2feats(edited)[:, : position + 1]
                checks.append(
                    {
                        "training": training,
                        "dropout_disabled": True,
                        "position": position,
                        "max_abs_error": float((a - b).abs().max().item()),
                        "invariant": bool(
                            torch.allclose(a, b, atol=tolerance, rtol=1e-5)
                        ),
                    }
                )
                del reference, other, a, b
            del probe
        result["causality"] = checks
        outside = [p for p in range(length) if length - 1 - p not in offsets]
        if outside:
            edited = x.clone()
            edited[:, outside] = 2
            changed = self._scores(edited, candidates)[0]
            result["outside_dependency"] = {
                "edited_positions": outside,
                "max_abs_error": float((scores - changed).abs().max().item()),
                "invariant": bool(
                    torch.allclose(scores, changed, atol=tolerance, rtol=1e-5)
                ),
            }
        else:
            result["outside_dependency"] = None
            result["outside_dependency_reason"] = (
                "Every retained position is structurally accessible"
            )
        # Establish boundary reachability on a controlled disposable network.
        # Identity normalizers remove trained cancellation; these are structural
        # controls, not measurements of the trained model's item preferences.
        synthetic = control_copy().double().eval()
        synthetic.norm = torch.nn.Identity()
        with torch.no_grad():
            synthetic.item_emb.weight.zero_()
            synthetic.item_emb.weight[2, 0] = 1.0
            synthetic.pos_emb.weight.zero_()
            for block in synthetic.blocks:
                block.norm1 = torch.nn.Identity()
                block.norm2 = torch.nn.Identity()
                block.activation = torch.nn.Identity()
                block.gate.weight.zero_()
                block.gate.bias.zero_()
                for conv in (block.conv1, block.conv2):
                    conv.weight.zero_()
                    if conv.bias is not None:
                        conv.bias.zero_()
                    for channel in range(conv.in_channels):
                        conv.weight[channel, channel, :] = 1.0 / conv.kernel_size[0]
            accessible = [d for d in offsets if d < length]
            tested = sorted({accessible[0], accessible[-1]})
            reference = synthetic.log2feats(x)[0, -1, 0]
            result["synthetic_reachability"] = []
            for distance in tested:
                edited = x.clone()
                edited[0, length - 1 - distance] = 2
                delta = float(
                    (synthetic.log2feats(edited)[0, -1, 0] - reference).item()
                )
                result["synthetic_reachability"].append(
                    {
                        "distance": distance,
                        "controlled_response": delta,
                        "reachable": delta > 0.0,
                    }
                )
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

    def _prefixes(self, offsets):
        source = self.user_train
        histories = source.get("History", source) if isinstance(source, dict) else {}
        if not isinstance(histories, dict):
            return [], {"invalid_history_mapping": 1}, {}, {}
        # Stable user sampling does not depend on model outcomes or loader order.
        users = sorted(
            histories, key=lambda u: (self._seed(self.category, str(u), "user"), str(u))
        )
        users = users[:32]
        excluded = Counter()
        transitions = Counter()
        for history in histories.values():
            if isinstance(history, (list, tuple, np.ndarray)):
                transitions.update(
                    (int(a), int(b)) for a, b in zip(history[:-1], history[1:])
                )
        records, edits = [], {}
        itemnum, length = int(self.network.itemnum), int(self.network.maxlen)
        for user in users:
            try:
                history = [int(i) for i in histories[user]]
            except (TypeError, ValueError):
                excluded["invalid_history"] += 1
                continue
            if not history or any(i < 1 or i > itemnum for i in history):
                excluded["empty_or_invalid_history"] += 1
                continue
            # Fixed target positions; later events never enter the input.
            targets = sorted({len(history) - 1, (len(history) - 1) // 2})
            for target_index in targets:
                if target_index < 1:
                    excluded["empty_history"] += 1
                    continue
                prefix, target = history[:target_index], history[target_index]
                retained = prefix[-length:]
                forbidden = set(prefix) | {0, target}
                if itemnum - len(forbidden - {0}) < 99:
                    excluded["insufficient_unique_negatives"] += 1
                    continue
                rng = random.Random(
                    self._seed(self.category, str(user), target_index, "candidates")
                )
                # Sampling without replacement with bounded work, even for dense users.
                pool = [i for i in range(1, itemnum + 1) if i not in forbidden]
                candidates = [target] + rng.sample(pool, 99)
                tie_rng = random.Random(
                    self._seed(self.category, str(user), target_index, "ties")
                )
                shuffled = sorted(candidates)
                tie_rng.shuffle(shuffled)
                tie = {item: j for j, item in enumerate(shuffled)}
                accessible = [
                    j
                    for j in range(len(retained) - 1)
                    if len(retained) - 1 - j in offsets
                ]
                variants = {"unchanged": retained}
                manifest = {"accessible_nonendpoint_indices": accessible}
                # Match genuine interior deletions. Deleting index zero removes
                # one edge; deleting an interior event removes two and joins a
                # new transition, so those interventions are not matched.
                deletable = [j for j in accessible if j > 0]
                manifest["matched_deletion_indices"] = deletable
                if len(deletable) >= 2:
                    older, recent = deletable[0], deletable[-1]
                    for condition, index in (
                        ("older_delete", older),
                        ("recent_delete", recent),
                    ):
                        variants[condition] = retained[:index] + retained[index + 1 :]
                        # Original event indices distinguish repeats and newly joined edges.
                        kept = [j for j in range(len(retained)) if j != index]
                        newly_reachable = [
                            j
                            for rank, j in enumerate(kept)
                            if len(kept) - 1 - rank in offsets
                            and len(retained) - 1 - j not in offsets
                        ]
                        manifest[condition] = {
                            "deleted_index": index,
                            "edit_count": 1,
                            "result_length": len(kept),
                            "endpoint_preserved": True,
                            "pattern": "one_contiguous_event",
                            "newly_reachable_original_indices": newly_reachable,
                            "distance_changes": [
                                [j, len(retained) - 1 - j, len(kept) - 1 - rank]
                                for rank, j in enumerate(kept)
                            ],
                            "newly_joined_transitions": [
                                [retained[a], retained[b]]
                                for a, b in zip(kept[:-1], kept[1:])
                                if b != a + 1
                            ],
                        }
                else:
                    excluded["insufficient_matched_accessible_positions"] += 1
                # Ordering remains measurable when a matched deletion pair is
                # unavailable. Record the exact permutation for replay.
                if len(accessible) >= 2:
                    permuted = retained.copy()
                    rotated = accessible[1:] + accessible[:1]
                    for destination, origin in zip(accessible, rotated):
                        permuted[destination] = retained[origin]
                    if permuted != retained:
                        variants["endpoint_fixed_reorder"] = permuted
                        manifest["endpoint_fixed_reorder"] = {
                            "destination_origin_pairs": [
                                [destination, origin]
                                for destination, origin in zip(accessible, rotated)
                            ],
                            "endpoint_preserved": True,
                        }
                    else:
                        excluded["reorder_identical_due_to_repeats"] += 1
                # Never insert a scored negative into the history: that creates
                # a candidate-overlap confound through the tied item embedding.
                replacement_forbidden = forbidden | set(candidates)
                replacement = None
                replacement_rng = random.Random(
                    self._seed(self.category, str(user), target_index, "replacement")
                )
                if itemnum > len(replacement_forbidden - {0}):
                    # A random cyclic starting point gives bounded work without
                    # materializing another full-catalog list.
                    start = replacement_rng.randrange(itemnum)
                    for offset in range(itemnum):
                        item = 1 + (start + offset) % itemnum
                        if item not in replacement_forbidden:
                            replacement = item
                            break
                if replacement is None:
                    excluded["no_candidate_disjoint_replacement"] += 1
                else:
                    variants["endpoint_replace_stress"] = retained[:-1] + [replacement]
                    manifest["endpoint_replace_stress"] = {
                        "replaced_index": len(retained) - 1,
                        "replacement_item": replacement,
                        "candidate_disjoint": True,
                    }
                    if accessible:
                        index = accessible[-1]
                        edited = retained.copy()
                        edited[index] = replacement
                        variants["interior_replace_stress"] = edited
                        manifest["interior_replace_stress"] = {
                            "replaced_index": index,
                            "replacement_item": replacement,
                            "candidate_disjoint": True,
                        }
                manifest["transition_support"] = {
                    condition: {
                        "edge_count": max(len(items) - 1, 0),
                        "supported_edges": sum(
                            transitions[(a, b)] > 0
                            for a, b in zip(items[:-1], items[1:])
                        ),
                    }
                    for condition, items in variants.items()
                }
                identifier = f"{str(user)}:{target_index}"
                records.append(
                    {
                        "user": str(user),
                        "prefix_id": identifier,
                        "history": prefix,
                        "target": target,
                        "repeated_target": target in prefix,
                        "candidates": candidates,
                        "tie_order": [tie[i] for i in candidates],
                        "variants": variants,
                    }
                )
                edits[identifier] = manifest
        return (
            records,
            dict(excluded),
            edits,
            {"selected_users": len(users), "source_users": len(histories)},
        )

    @staticmethod
    def _outcome(scores, tie_order):
        order = sorted(
            range(len(scores)), key=lambda j: (-float(scores[j]), tie_order[j])
        )
        rank = order.index(0)
        return np.asarray(
            [float(rank + 1), float(scores[0] - max(scores[1:])), float(rank < 10)]
        )

    def _summaries(self, records):
        outcomes = []
        for record in records:
            names = list(record["variants"])
            sequence = self._tensor([record["variants"][name] for name in names])
            scores = self._scores(sequence, record["candidates"]).cpu().numpy()
            results = {
                name: self._outcome(row, record["tie_order"])
                for name, row in zip(names, scores)
            }
            outcomes.append((record, results))
        summaries = {}
        # Keep absent conditions visible with null estimates and zero coverage.
        conditions = [
            "older_delete",
            "recent_delete",
            "endpoint_fixed_reorder",
            "endpoint_replace_stress",
            "interior_replace_stress",
        ]
        for stratum in ("all", "repeated", "novel"):
            selected = [
                (r, o)
                for r, o in outcomes
                if stratum == "all" or r["repeated_target"] == (stratum == "repeated")
            ]
            for condition in conditions + ["recent_minus_older_delete"]:
                per_user = {}
                prefixes = 0
                for record, results in selected:
                    if condition == "recent_minus_older_delete":
                        if not {"recent_delete", "older_delete"}.issubset(results):
                            continue
                        delta = results["recent_delete"] - results["older_delete"]
                    else:
                        if condition not in results:
                            continue
                        delta = results[condition] - results["unchanged"]
                    per_user.setdefault(record["user"], []).append(delta)
                    prefixes += 1
                key = f"{stratum}/{condition}"
                if not per_user:
                    summaries[key] = {
                        "mean_delta": None,
                        "percentile_95_interval": None,
                        "users": 0,
                        "prefixes": 0,
                    }
                    continue
                values = np.stack(
                    [np.mean(per_user[u], axis=0) for u in sorted(per_user)]
                )
                rng = np.random.default_rng(self._seed(self.category, key, "bootstrap"))
                # Resample users with all their paired prefixes retained. Chunk
                # replicates to bound memory independently of the user count.
                boot = []
                for start in range(0, 10000, 250):
                    indices = rng.integers(
                        0, len(values), size=(min(250, 10000 - start), len(values))
                    )
                    boot.append(values[indices].mean(axis=1))
                interval = np.percentile(np.concatenate(boot), [2.5, 97.5], axis=0).T
                summaries[key] = {
                    "mean_delta": values.mean(axis=0).tolist(),
                    "percentile_95_interval": interval.tolist(),
                    "users": len(values),
                    "prefixes": prefixes,
                }
        return summaries

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
        # Restore every module's mode, not just the root's training flag. Preserve
        # RNG state even when a loader or exported predictor initializes models.
        modes = [(module, module.training) for module in self.network.modules()]
        python_state, numpy_state = random.getstate(), np.random.get_state()
        # Preserve RNG on the single execution GPU without initializing every
        # visible GPU in a multi-slot host.
        devices = (
            [
                (
                    self.device.index
                    if self.device.index is not None
                    else torch.cuda.current_device()
                )
            ]
            if self.device.type == "cuda"
            else []
        )
        try:
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                self.network.eval()
                offsets, convolutions = self._dependencies()
                controls = self._controls(offsets)
                collapse = self.analyze_embeddings_collapse_cosine(sample_size=2048)
                margins, breakdown = (
                    self.analyze_pos_neg_margin_with_category_breakdown(
                        train_loader, num_batches=10, eps=0.5, topk_categories=20
                    )
                )
                records, exclusions, edits, coverage = self._prefixes(offsets)
                export = self._export_parity(records)
                failures = []
                if controls.get("readout_score_match") is not True:
                    failures.append("readout_score_match_failed_or_unavailable")
                if controls.get("batched_feature_score_match") is not True:
                    failures.append("batched_feature_score_match_failed_or_unavailable")
                readouts = controls.get("left_padded_readout_checks", [])
                if not readouts or any(c["match"] is not True for c in readouts):
                    failures.append("left_padded_readout_failed_or_unavailable")
                causal = controls.get("causality")
                if not causal or any(c["invariant"] is not True for c in causal):
                    failures.append("causality_failed_or_unavailable")
                outside = controls.get("outside_dependency")
                if outside is not None and outside["invariant"] is not True:
                    failures.append("outside_dependency_invariance_failed")
                boundaries = controls.get("synthetic_reachability", [])
                if not boundaries or any(
                    c["reachable"] is not True for c in boundaries
                ):
                    failures.append("synthetic_reachability_failed_or_unavailable")
                normalization = controls.get("normalization", [])
                if any(
                    c.get("potential_temporal_or_batch_coupling", False)
                    or c.get("channel_only_shape") is False
                    for c in normalization
                ):
                    failures.append("normalization_requires_dependency_audit")
                if export.get("available") is True:
                    if not export.get("checks") or any(
                        not c["scores_match"] or not c["rank_order_match"]
                        for c in export["checks"]
                    ):
                        failures.append("export_score_or_rank_parity_failed")
                controls["sensitivity_blocking_failures"] = failures
                controls["sensitivity_scope"] = (
                    "blocked"
                    if failures
                    else (
                        "export_verified"
                        if export.get("available") is True
                        else "ModelView_only_export_unverified"
                    )
                )
                # Missing exports remain explicitly unverified; observed contract
                # failures block interpretation of intervention outcomes.
                summaries = None if failures else self._summaries(records)
                manifest = {
                    "seed": 42,
                    "category": self.category,
                    "source": "public training histories only",
                    "maxlen": int(self.network.maxlen),
                    "readout": "maximum nonzero position",
                    "user_cap": 32,
                    "prefix_cap_per_user": 2,
                    "target_indices": "unique sorted {last event, floor(last event / 2)}",
                    "margin_batch_cap": 10,
                    "embedding_sample_cap": 2048,
                    "negative_count": 99,
                    "edit_count_per_deletion": 1,
                    "candidate_exclusion": "entire original prefix, target and padding",
                    "matching": "same accessible prefix, one strictly interior event, endpoint fixed, equal resulting lengths and one newly joined transition",
                    "replacement_status": "candidate-disjoint real-item plausibility-unverified stress tests",
                    "replacement_sampling": "independent seed-42-derived random cyclic catalog scan; not uniform over eligible items",
                    "sensitivity_scope": controls["sensitivity_scope"],
                    "sensitivity_blocking_failures": failures,
                    "internal_padding_edits": False,
                    "dependency_offsets": offsets,
                    "convolutions": convolutions,
                    "receptive_width": 1 + max(offsets),
                    "bootstrap_replicates": 10000,
                    "bootstrap_seed": "SHA256-derived from 42, category, condition and stratum",
                    "coverage": coverage,
                    "exclusions": exclusions,
                    "prefixes": [
                        {k: v for k, v in r.items() if k != "variants"} for r in records
                    ],
                    "edits": edits,
                    "coherence_limits": "Transition support is training-only and in-sample; timestamp and text coherence are unmeasured",
                    "loader_identity": "Loader rows are not assigned user identities; user aggregation uses the explicit training-prefix index",
                }
                artifact = self._write_manifest(manifest)
                return {
                    "metrics": {
                        "embedding_collapse_score": collapse,
                        "embedding_sample_count": self.embedding_sample_count,
                        "pos_neg_margin_mean": margins["mean"],
                        "pos_neg_margin_p10": margins["p10"],
                        "pos_neg_neg_beats_pos_rate": margins["neg_beats_pos_rate"],
                        "pos_neg_small_margin_rate": margins["small_margin_rate"],
                        "pos_neg_margin_category_breakdown": breakdown,
                        "contract_controls": controls,
                        "export_parity": export,
                        "training_prefix_sensitivity": summaries,
                        "training_prefix_count": len(records),
                        "repeated_target_prefixes": sum(
                            r["repeated_target"] for r in records
                        ),
                        "novel_target_prefixes": sum(
                            not r["repeated_target"] for r in records
                        ),
                        "exclusions": exclusions,
                        "category": self.category,
                        "four_category_macro": None,
                        "four_category_macro_reason": "One category view per invocation; all four category user means are required",
                        "manifest": artifact,
                    },
                    "metric_definitions": {
                        "embedding_collapse_score": "(Mean off-diagonal cosine + 1)/2 on up to 2048 nonpadding item embeddings, seed 42. Near 0.5 indicates near-zero average cosine, not failure. Measures directional alignment, not rank collapse or recommendation diversity. Null with fewer than two items.",
                        "embedding_sample_count": "Number of sampled nonpadding embeddings; pair denominator is k*(k-1).",
                        "pos_neg_margin_mean": "Mean positive-minus-negative logit over finite, nonpadding sequence/positive/negative positions in at most ten loader batches. Scale depends on embedding norms.",
                        "pos_neg_margin_p10": "Nearest-rank 10th percentile of the same valid training margins.",
                        "pos_neg_neg_beats_pos_rate": "Fraction of those valid positions with margin below zero; not sampled-candidate HR@10.",
                        "pos_neg_small_margin_rate": "Fraction of those valid positions with margin below 0.5; this threshold is score-scale dependent.",
                        "pos_neg_margin_category_breakdown": "Contains valid-position counts and empty category lists: item categories are unavailable.",
                        "contract_controls": "Structural offsets compose both convolutions in every residual block. Readout tests include short left-padded histories. Causality tests compare intermediate positions in independent disposable copies with dropout disabled and identical initial buffers. Copies retain trained parameters but use only the three embedding rows exercised by controls. Synthetic boundary controls use disposable weights. Observed contract failures block sensitivity summaries; absent export parity limits conclusions to ModelView. These checks do not prove all inputs causal.",
                        "export_parity": "Actual trial predict.py versus ModelView, with identical weights, category route, full prefix, truncation and fixed candidates. Null/unavailable if the export, route or matching checkpoint is absent. Public exported prediction is singleton-only; batched ModelView parity is checked separately.",
                        "training_prefix_sensitivity": "Each vector is [target rank, target-minus-best-negative margin, HR@10]. Positive rank delta is worse; positive margin/hit delta is better. Edited minus unchanged, except recent_minus_older_delete. Fixed 100-item pools and target-independent seeded item tie order. Prefix deltas average within user, then equally across users of this category. All/repeated/novel strata use original-prefix target repetition. Intervals resample paired users 10000 times and exclude training-seed/candidate-sampling uncertainty. Single-user intervals are degenerate. These are in-sample sensitivity estimates, not preferences or validation improvements.",
                        "training_prefix_count": "Eligible prefixes after fixed user/target-position caps and unique-negative checks; no outcome-dependent selection.",
                        "repeated_target_prefixes": "Eligible prefixes whose target occurred earlier in the original prefix; earlier occurrences are retained.",
                        "novel_target_prefixes": "Eligible prefixes whose target did not occur earlier; interpret alongside total coverage.",
                        "exclusions": "Counts of unavailable histories, candidate pools or interventions. An unavailable intervention does not remove other eligible paired conditions.",
                        "four_category_macro": "Unavailable within a single-category invocation. A macro estimate requires equally weighting all four category means; missing categories must not be reweighted.",
                        "manifest": "Trial-local JSON recording predeclared caps, seeds, candidates, event indices, dependency offsets, edit matching, coverage and exclusions. Transition support is only an in-sample coherence proxy. Reviews, ratings, item categories, prices, stores and simulator traits are not used.",
                    },
                }
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            for module, training in modes:
                module.training = training

    ### <<< Self_EvolveRec-BLOCK-END
