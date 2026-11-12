import torch
import numpy as np
import torch.nn.functional as F
import json
import random

### >>> Self_EvolveRec-BLOCK-START: Bound diagnostic iteration without consuming an extra batch
from collections import Counter
from itertools import islice

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
    ### >>> Self_EvolveRec-BLOCK-START: Separate measured sensitivity from accuracy and causal hypotheses
    - Every statement must be grounded in the provided metric values/definitions.
    - Report support and exclusions alongside each finding. Null means unavailable, never zero.
    - Separate implementation checks, measured sensitivities, and hypotheses explicitly.
    - Sensitivity alone establishes neither accuracy, user intent, compatibility, nor causal temporal importance.
    - Raw margins depend on score scale. Pairwise negative-win rates are not benchmark HR@10.
    - A mapped embedding cosine near 0.5 means mean cosine near zero, not a calibrated collapse probability.
    - Repeated draws and loader rows are not independent users; do not invent confidence intervals.
    - Missing deployment parity or validation measurements are unverified, not successful checks.
    - Separate structural mismatch, exercised hooks, local activation effects, logit propagation and alarms.
    - Report calibration controls, frozen tolerances, row/site support, exclusions and omitted checks.
    - Held-out sites do not establish held-out-family generalization. Surviving faults do not establish equivalence.
    - A local convolution reference does not verify whole-block semantics. Tensor reload controls do not verify deployment.
    - Without an independently verified clean implementation, report observed clean violations, not established false positives.
    - Ignore unavailable categories, ratings, reviews, and simulator diversity/conformity traits.
    ### <<< Self_EvolveRec-BLOCK-END
    - If metrics look healthy, status should remain STABLE and implications should focus on "what is already working" plus one potential risk to monitor.
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve ModelView while managing the actual trained module
    def __init__(self, view, device):
        self.model = view
        self.network = view.model
        self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train
        self._hooks = []

    def _get_categories(self, item_id: int):
        # Item categories are unavailable in this matched task.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()
            ### >>> Self_EvolveRec-BLOCK-START: Report undefined embedding measurements as null
            emb = emb[1 : self.network.itemnum + 1]

            n = emb.size(0)
            if min(sample_size, n) < 2 or not torch.isfinite(emb).all():
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Preserve mean cosine while avoiding the quadratic Gram matrix
            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx).float()
            x = F.normalize(x, p=2, dim=-1)

            # Sum of off-diagonal dot products, including zero-vector handling.
            summed = x.sum(dim=0)
            offdiag_sum = summed.square().sum() - x.square().sum()
            offdiag_cnt = k * (k - 1)
            mean_cos = (offdiag_sum / offdiag_cnt).clamp(-1, 1).item()

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
        ### >>> Self_EvolveRec-BLOCK-START: Let the outer diagnostic own mode restoration
        margins_all = []
        valid_total = nonfinite_total = 0
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Stop before requesting an extra diagnostic batch
        for batch in islice(iter(dataloader), max(0, num_batches)):
            ### <<< Self_EvolveRec-BLOCK-END
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 4):
                continue

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            ### >>> Self_EvolveRec-BLOCK-START: Use the NextItNet tensor compatibility contract
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            log_feats = self.model.log2feats(seq_t)
            ### <<< Self_EvolveRec-BLOCK-END

            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)

            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
            margin = all_logits[..., 0] - all_logits[..., 1]

            ### >>> Self_EvolveRec-BLOCK-START: Count finite supported all-position comparisons explicitly
            valid = seq_t.ne(0) & pos_t.ne(0) & neg_t.ne(0)
            finite = torch.isfinite(all_logits).all(dim=-1) & torch.isfinite(margin)
            valid_total += int(valid.sum().item())
            nonfinite_total += int((valid & ~finite).sum().item())
            valid = valid & finite
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
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
            ### >>> Self_EvolveRec-BLOCK-START: Distinguish absent comparisons from perfect margins
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

        ### >>> Self_EvolveRec-BLOCK-START: Attach denominators and finite exclusions to margin summaries
        margin_metrics.update(
            {
                "valid_positions": valid_total,
                "comparisons": valid_total - nonfinite_total,
                "excluded_nonfinite": nonfinite_total,
                "distinct_users": None,
            }
        )
        ### <<< Self_EvolveRec-BLOCK-END

        def _topk(counter: Counter, k: int):
            return [{"category": c, "count": int(n)} for c, n in counter.most_common(k)]

        ### >>> Self_EvolveRec-BLOCK-START: Explicitly mark item-category evidence unavailable
        breakdown = {
            "availability": "unavailable: item categories are not provided",
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

    ### >>> Self_EvolveRec-BLOCK-START: Validate clean sites before bounded fault calibration and verify exception cleanup
    @staticmethod
    def _last(sequence):
        indices = torch.arange(sequence.size(1), device=sequence.device).expand_as(
            sequence
        )
        last = indices.masked_fill(sequence.eq(0), -1).amax(dim=1)
        return last.clamp(min=0), last.ge(0)

    def _features(self, sequence, ablate=None):
        if ablate is None:
            return self.model.log2feats(sequence)

        def hook(module, args, output):
            if output.shape[:-1] != ablate.shape:
                raise RuntimeError(
                    "unexpected embedding shape during diagnostic ablation"
                )
            return output * (~ablate).unsqueeze(-1).to(output.dtype)

        handle = self.model.item_emb.register_forward_hook(hook)
        self._hooks.append(handle)
        try:
            return self.model.log2feats(sequence)
        finally:
            handle.remove()
            self._hooks.remove(handle)

    def _scores(self, sequence, targets, ablate=None):
        # The hook is removed before target embeddings are looked up.
        features = self._features(sequence, ablate)
        last, nonempty = self._last(sequence)
        rows = torch.arange(sequence.size(0), device=sequence.device)
        final = features[rows, last] * nonempty.unsqueeze(-1).to(features.dtype)
        return self.model.predict_score(final, targets)

    def _structure(self):
        net = self.network
        records, reachable = [], {0}
        limitations = []
        blocks = getattr(net, "blocks", [])
        if not blocks or type(net).__name__ != "NextItNetModel":
            limitations.append("unrecognized backbone topology")
        for index, block in enumerate(blocks):
            branch = set(reachable)
            for name in ("conv1", "conv2"):
                conv = getattr(block, name, None)
                if not isinstance(conv, torch.nn.Conv1d):
                    limitations.append(f"blocks.{index}.{name} is not Conv1d")
                    continue
                k, d, stride = conv.kernel_size[0], conv.dilation[0], conv.stride[0]
                records.append(
                    {
                        "name": f"blocks.{index}.{name}",
                        "kernel": k,
                        "dilation": d,
                        "stride": stride,
                        "module_padding": list(conv.padding),
                    }
                )
                if stride != 1 or conv.padding != (0,):
                    limitations.append(
                        "unsupported stride or internal convolution padding"
                    )
                branch = {s + j * d for s in branch for j in range(k)}
            reachable |= branch
        known_containers = {id(net), *(id(block) for block in blocks)}
        allowed = (
            torch.nn.ModuleList,
            torch.nn.Conv1d,
            torch.nn.Embedding,
            torch.nn.Linear,
            torch.nn.Dropout,
            torch.nn.GELU,
            torch.nn.LayerNorm,
        )
        for name, module in net.named_modules():
            if id(module) not in known_containers and not isinstance(module, allowed):
                limitations.append(
                    f"dependency analysis required for {name}: {type(module).__name__}"
                )
            if isinstance(module, torch.nn.LayerNorm):
                if tuple(module.normalized_shape) != (net.hidden_units,):
                    limitations.append(f"non-channel-only normalization: {name}")
        if len(records) != sum(isinstance(m, torch.nn.Conv1d) for m in net.modules()):
            limitations.append("convolutions outside the recognized residual branches")
        verified = bool(records) and not limitations
        return {
            "analysis": "supplied causal residual topology; empirical checks reported separately",
            "convolutions": records,
            "reachable_offsets": sorted(reachable) if verified else None,
            "maximum_offset": max(reachable) if verified else None,
            "span": max(reachable) + 1 if verified else None,
            "limitations": limitations,
        }

    @staticmethod
    def _summary(values):
        if not values:
            return {
                "comparisons": 0,
                "mean": None,
                "p10": None,
                "neg_beats_pos_rate": None,
                "small_margin_rate": None,
            }
        x = torch.cat(values).float()
        return {
            "comparisons": x.numel(),
            "mean": x.mean().item(),
            "p10": x.kthvalue(max(1, int(np.ceil(0.1 * x.numel())))).values.item(),
            "neg_beats_pos_rate": (x < 0).float().mean().item(),
            "small_margin_rate": (x < 0.5).float().mean().item(),
        }

    @torch.no_grad()
    def run_full_diagnosis(self, train_loader):
        previous_modes = [
            (module, module.training) for module in self.network.modules()
        ]
        python_state, numpy_state = random.getstate(), np.random.get_state()
        cuda_devices = []
        if self.device.type == "cuda":
            cuda_devices = [
                (
                    self.device.index
                    if self.device.index is not None
                    else torch.cuda.current_device()
                )
            ]
        try:
            self.network.eval()
            with torch.random.fork_rng(devices=cuda_devices):
                random.seed(42)
                np.random.seed(42)
                torch.random.default_generator.manual_seed(42)
                if cuda_devices:
                    with torch.cuda.device(cuda_devices[0]):
                        torch.cuda.manual_seed(42)
                return self._run_bounded(train_loader)
        finally:
            for handle in self._hooks:
                handle.remove()
            self._hooks.clear()
            for module, training in previous_modes:
                module.training = training
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    def _calibrate_operators(self, batches):
        import hashlib

        net = self.network
        # Frozen from the supplied source, never from mutated layer attributes.
        spec = {
            "source": "supplied NextItNetBlock source",
            "version": "explicit_conv_reference_v1",
            "dilation_cycle": [1, 2, 4, 8],
            "convolutions_per_block": 2,
            "stride": 1,
            "groups": 1,
            "internal_padding": 0,
            "padding": "left=(kernel-1)*dilation; right=0",
        }
        digest = lambda x: hashlib.sha256(
            json.dumps(x, sort_keys=True).encode()
        ).hexdigest()
        manifests, calibration, evaluation = [], [], []
        for b, (_, seq, pos, neg) in enumerate(batches):
            for r in range(seq.size(0)):
                occupied = seq[r].ne(0).nonzero().flatten().tolist()
                record = {
                    "batch": b,
                    "row": r,
                    "occupied_positions": occupied,
                    "length": len(occupied),
                    "sha256": digest([t[r].cpu().tolist() for t in (seq, pos, neg)]),
                    "partition": (
                        "calibration" if (b * 32 + r) % 2 == 0 else "evaluation"
                    ),
                }
                manifests.append(record)
                if occupied:
                    destination = (
                        calibration
                        if record["partition"] == "calibration"
                        else evaluation
                    )
                    destination.append((b, r))
        result = {
            "specification": spec,
            "specification_sha256": digest(spec),
            "row_manifest": manifests,
            "controls": [],
            "mutations": [],
            "configuration": {
                "seed": 42,
                "maximum_rows_per_partition": 2,
                "rtol": 1e-4,
                "threshold_rule": "max(1e-5,10*clean_max_abs_error)",
                "clean_discrepancy_ceiling": 1e-3,
                "site_partition": "first seeded site calibrates; remaining sites held out",
                "heldout_operator_families": [],
                "operator_version": "conv_hook_faults_v1",
                "device": str(self.device),
                "torch": torch.__version__,
            },
            "limitations": [
                "Local convolution reference only; whole-block normalization/skip reference unavailable.",
                "No independently verified clean implementation; alarms are observed clean violations.",
                "No primitive distinct-weight, normalization-axis or misplaced-mask calibration.",
                "Rows are training inputs without stable user identities; partitions can share users.",
                "Held-out sites are not held-out operator families; detectors were designed for these faults.",
                "No generated-predictor reload interface; clone controls do not establish deployment parity.",
            ],
        }
        if not calibration or not evaluation or not hasattr(net, "blocks"):
            return {
                **result,
                "available": False,
                "reason": "insufficient partition support or unsupported backbone",
            }
        calibration, evaluation = calibration[:2], evaluation[:2]
        result["configuration"]["used_calibration_rows"] = calibration
        result["configuration"]["used_evaluation_rows"] = evaluation

        # Register and execute the actual model; do not capture cached features.
        class Execution(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, sequence):
                features = self.model.log2feats(sequence)
                last, nonempty = DiagnosisProbe._last(sequence)
                final = features[
                    torch.arange(sequence.size(0), device=sequence.device), last
                ]
                ids = torch.arange(
                    1, min(self.model.itemnum, 32) + 1, device=sequence.device
                )
                return self.model.predict_score(final * nonempty[:, None], ids)

        execute = Execution(net)

        # Indexed sums are independent of torch convolution kernels.
        def reference(x, weight, bias, dilation):
            length = x.size(-1) - (weight.size(-1) - 1) * dilation
            y = x.new_zeros((x.size(0), weight.size(0), length))
            for tap in range(weight.size(-1)):
                values = x[:, :, tap * dilation : tap * dilation + length]
                y += torch.einsum("bct,oc->bot", values, weight[:, :, tap])
            return y if bias is None else y + bias[None, :, None]

        def error(a, b):
            if a.shape != b.shape:
                return {
                    "finite": False,
                    "elements": a.numel(),
                    "actual_shape": list(a.shape),
                    "reference_shape": list(b.shape),
                    "max_abs_error": None,
                    "max_relative_error": None,
                }
            finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            delta = (a - b).abs()
            return {
                "finite": finite,
                "elements": a.numel(),
                "max_abs_error": float(delta.max()) if finite and a.numel() else None,
                "max_relative_error": (
                    float((delta / b.abs().clamp_min(1e-8)).max())
                    if finite and a.numel()
                    else None
                ),
            }

        sites = [
            (i, name) for i in range(len(net.blocks)) for name in ("conv1", "conv2")
        ]
        order = np.random.default_rng(42).permutation(len(sites)).tolist()
        sites = [sites[i] for i in order[:3]]
        if not sites:
            return {**result, "available": False, "reason": "no convolution sites"}
        result["configuration"]["frozen_sites"] = [list(site) for site in sites]
        result["configuration"]["candidate_ids"] = list(
            range(1, min(net.itemnum, 32) + 1)
        )
        result["configuration"]["precision"] = str(net.item_emb.weight.dtype)
        result["configuration"]["numpy"] = np.__version__
        result["configuration"]["operator_version"] = "conv_hook_faults_v2"
        result["configuration"][
            "ranking_scope"
        ] = "ordered top-10 within the fixed candidate subset; not catalog HR@10"
        result["clean_site_checks"] = []
        controls_max = 0.0
        atol = 1e-5
        for site_index, (block_index, name) in enumerate(sites):
            conv = getattr(net.blocks[block_index], name)
            expected_d = spec["dilation_cycle"][block_index % 4]
            if (
                not isinstance(conv, torch.nn.Conv1d)
                or conv.groups != 1
                or conv.stride != (1,)
                or conv.padding != (0,)
                or conv.padding_mode != "zeros"
            ):
                result["mutations"].append(
                    {
                        "site": [block_index, name],
                        "excluded": "unsupported convolution layout",
                    }
                )
                continue
            weight = conv.weight.detach().clone()
            bias = None if conv.bias is None else conv.bias.detach().clone()
            kernel = weight.size(-1)
            expected_pad = (kernel - 1) * expected_d
            structural_match = (
                conv.dilation == (expected_d,)
                and net.blocks[block_index].dilation == expected_d
                and net.blocks[block_index].kernel_size == kernel
            )
            rows = calibration if site_index == 0 else evaluation
            if not structural_match:
                result["clean_site_checks"].append(
                    {
                        "site": [block_index, name],
                        "structural_match": False,
                        "actual_dilation": list(conv.dilation),
                        "expected_dilation": expected_d,
                        "block_dilation": net.blocks[block_index].dilation,
                        "kernel": kernel,
                    }
                )
                return {
                    **result,
                    "available": False,
                    "reason": "clean structural mismatch; mutation calibration stopped",
                }
            # Identity and faults share the same actual-model execution pipeline.
            operators = [
                ("unchanged_repeat", 0),
                ("identity", 0),
                ("dilation", 1),
                ("dilation", 2),
                ("padding", 1),
                ("padding", 2),
            ]
            if site_index == 0:
                operators = [
                    ("unchanged_repeat", 0),
                    ("ephemeral_tensor_reload", 0),
                    ("identity", 0),
                ]
            for operator, severity in operators:
                for b, r in rows:
                    sequence = batches[b][1][r : r + 1].clone()
                    before = execute(sequence).detach().clone()
                    observed = []

                    def hook(module, args, output):
                        x = args[0]
                        # Use the original sequence length, rather than accepting
                        # the caller's padding as part of the numerical oracle.
                        length = sequence.size(1)
                        if x.size(-1) < length:
                            raise RuntimeError(
                                "convolution input shorter than sequence"
                            )
                        unpadded = x[:, :, -length:]
                        expected_input = F.pad(unpadded, (expected_pad, 0))
                        clean = reference(expected_input, weight, bias, expected_d)
                        changed = output
                        new_d, right = expected_d, 0
                        if operator == "dilation":
                            new_d = (
                                expected_d + 1 if severity == 1 else 2 * expected_d + 1
                            )
                        if operator == "padding":
                            right = (
                                min(1, expected_pad)
                                if severity == 1
                                else expected_pad // 2
                            )
                        if severity:
                            total = (kernel - 1) * new_d
                            shifted = F.pad(unpadded, (total - right, right))
                            changed = F.conv1d(shifted, weight, bias, dilation=new_d)
                        elif operator == "ephemeral_tensor_reload":
                            changed = F.conv1d(
                                x,
                                weight.clone(),
                                None if bias is None else bias.clone(),
                                dilation=expected_d,
                            )
                        observed.append(
                            (
                                error(changed, clean),
                                bool(
                                    changed.shape == clean.shape
                                    and torch.isfinite(changed).all()
                                    and torch.isfinite(clean).all()
                                    and torch.isclose(
                                        changed, clean, atol=atol, rtol=1e-4
                                    ).all()
                                ),
                                new_d,
                                right,
                            )
                        )
                        return changed

                    original_hooks = tuple(conv._forward_hooks.items())
                    handle = conv.register_forward_hook(hook)
                    try:
                        after = execute(sequence).detach().clone()
                    finally:
                        handle.remove()
                        restored = execute(sequence)
                        if tuple(conv._forward_hooks.items()) != original_hooks:
                            raise RuntimeError("calibration changed existing hooks")
                        if not torch.allclose(
                            before, restored, atol=1e-5, rtol=1e-4, equal_nan=True
                        ):
                            raise RuntimeError(
                                "calibration failed clean-output restoration"
                            )
                    if not observed:
                        raise RuntimeError(
                            "calibration injection site was not executed"
                        )
                    local, close, new_d, right = observed[0]
                    record = {
                        "operator": operator,
                        "severity": severity,
                        "site": [block_index, name],
                        "batch_row": [b, r],
                        "heldout_site": site_index > 0,
                        "structural_match": structural_match,
                        "hook_calls": len(observed),
                        "local": local,
                        "local_alarm": not close,
                        "logits": error(after, before),
                        "top10_disagreement": not torch.equal(
                            torch.argsort(after, descending=True, stable=True)[:, :10],
                            torch.argsort(before, descending=True, stable=True)[:, :10],
                        ),
                        "requested_value_changed": bool(
                            new_d != expected_d or right != 0
                        ),
                        "actual_dilation": new_d,
                        "actual_right_padding": right,
                        "restoration_verified": True,
                    }
                    is_control = severity == 0
                    destination = "controls" if is_control else "mutations"
                    record["threshold_atol"] = atol
                    record["threshold_rtol"] = 1e-4
                    record["logit_alarm"] = not bool(
                        torch.isfinite(after).all()
                        and torch.isfinite(before).all()
                        and torch.isclose(after, before, atol=atol, rtol=1e-4).all()
                    )
                    record["detection_scope"] = "local numerical discrepancy"
                    result[destination].append(record)
                    if is_control:
                        result["clean_site_checks"].append(record.copy())
                        if (
                            not local["finite"]
                            or local["max_abs_error"] > 1e-3
                            or record["logit_alarm"]
                            or (site_index > 0 and not close)
                        ):
                            return {
                                **result,
                                "available": False,
                                "reason": "clean reference/control discrepancy; mutation calibration stopped",
                            }
                        if site_index == 0:
                            controls_max = max(controls_max, local["max_abs_error"])
            if site_index == 0:
                atol = max(1e-5, 10 * controls_max)
        result.update(
            {
                "available": bool(result["controls"]),
                "atol": atol,
                "clean_max_abs_error": controls_max,
            }
        )
        result["readout_mutation"] = {
            "available": False,
            "reason": "count-minus-one wrapper mutation not implemented; not counted as tested",
        }
        # Exercise cleanup after a hook actually ran, then deliberately fail.
        b, r = evaluation[0]
        sequence = batches[b][1][r : r + 1].clone()
        baseline = execute(sequence).detach().clone()
        block_index, name = sites[0]
        conv = getattr(net.blocks[block_index], name)
        original_hooks = tuple(conv._forward_hooks.items())
        calls = []

        def identity_hook(module, args, output):
            calls.append(1)
            return output

        class DeliberateDiagnosticException(RuntimeError):
            pass

        try:
            handle = conv.register_forward_hook(identity_hook)
            try:
                execute(sequence)
                raise DeliberateDiagnosticException("exercise diagnostic cleanup")
            finally:
                handle.remove()
        except DeliberateDiagnosticException:
            if not calls:
                raise RuntimeError("exception-path hook was not exercised")
        restored = execute(sequence)
        if tuple(conv._forward_hooks.items()) != original_hooks or not torch.allclose(
            baseline, restored, atol=1e-5, rtol=1e-4, equal_nan=True
        ):
            raise RuntimeError("exception-path restoration failed")
        result["exception_restoration"] = {
            "available": True,
            "hook_calls": len(calls),
            "verified": True,
            "scope": "temporary identity hook removal and fixed-input output agreement",
            "full_execution_state_verified": False,
        }
        result["severity_summary"] = [
            {
                "operator": operator,
                "severity": severity,
                "attempted": len(records),
                "finite_local_comparisons": sum(x["local"]["finite"] for x in records),
                "local_alarms": sum(x["local_alarm"] for x in records),
                "logit_alarms": sum(x["logit_alarm"] for x in records),
                "ordered_top10_disagreements": sum(
                    x["top10_disagreement"] for x in records
                ),
            }
            for operator, severity in sorted(
                {(x["operator"], x["severity"]) for x in result["mutations"]}
            )
            for records in [
                [
                    x
                    for x in result["mutations"]
                    if (x["operator"], x["severity"]) == (operator, severity)
                ]
            ]
        ]
        result["observed_clean_alarm_support"] = {
            "comparisons": len(result["controls"]),
            "local_alarms": sum(x["local_alarm"] for x in result["controls"]),
            "logit_alarms": sum(x["logit_alarm"] for x in result["controls"]),
            "calibration_controls_use_initial_tolerance": True,
        }
        result["limitations"].append(
            "Kernel width follows the fitted tensor shape; this is not a full independently frozen architecture oracle. Full execution-state restoration, primitive semantics, whole-block reference and readout mutation remain unverified."
        )
        return result

    def _run_bounded(self, train_loader):
        batches = []
        for batch in islice(iter(train_loader), 4):
            if not isinstance(batch, (tuple, list)) or len(batch) != 4:
                raise ValueError("expected (None, sequence, positive, negative)")
            tensors = [
                torch.as_tensor(x, device=self.device, dtype=torch.long)[:32].clone()
                for x in batch[1:]
            ]
            if any(t.ndim != 2 for t in tensors) or any(
                t.shape != tensors[0].shape for t in tensors
            ):
                raise ValueError("diagnostic tensors must have matching B x L shapes")
            if tensors[0].size(0) and tensors[0].size(1):
                batches.append((None, *tensors))
        calibration = self._calibrate_operators(batches)
        structure = self._structure()
        offsets = structure["reachable_offsets"]
        draws = [torch.Generator().manual_seed(42 + i) for i in range(5)]
        # Preserve explicit null summaries even when the loader is empty.
        perturbation_names = [
            name + suffix
            for name in ("recent", "older")
            for suffix in ("_repacked", "_embedding_ablation", "_repacked_vs_ablation")
        ] + ["prefix_shuffle"]
        paired = {
            name: {"eligible_comparisons": 0, "excluded_nonfinite": 0, "values": []}
            for name in perturbation_names
        }
        final_margins, checks, offset_checks = [], [], []
        counts = Counter()
        shuffle_support = [Counter() for _ in draws]
        structural_rows = 0

        def compare(name, baseline, changed, eligible):
            entry = paired.setdefault(
                name, {"eligible_comparisons": 0, "excluded_nonfinite": 0, "values": []}
            )
            entry["eligible_comparisons"] += int(eligible.sum().item())
            finite = torch.isfinite(baseline).all(-1) & torch.isfinite(changed).all(-1)
            before = baseline[:, 0] - baseline[:, 1]
            after = changed[:, 0] - changed[:, 1]
            finite &= torch.isfinite(before) & torch.isfinite(after)
            finite &= torch.isfinite(after - before)
            finite &= torch.isfinite(changed - baseline).all(-1)
            entry["excluded_nonfinite"] += int((eligible & ~finite).sum().item())
            keep = eligible & finite
            if keep.any():
                delta = changed[keep] - baseline[keep]
                values = torch.stack(
                    (
                        after[keep] - before[keep],
                        delta[:, 0],
                        delta[:, 1],
                        delta.abs().mean(-1),
                        (after[keep] < 0).float() - (before[keep] < 0).float(),
                    ),
                    -1,
                )
                entry["values"].append(values.float().cpu())

        def check(name, a, b):
            finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            checks.append(
                {
                    "name": name,
                    "elements": a.numel(),
                    "finite": finite,
                    "max_abs_error": (
                        (a - b).abs().max().item() if finite and a.numel() else None
                    ),
                    "close": (
                        bool(torch.isclose(a, b, rtol=1e-4, atol=1e-5).all())
                        if finite
                        else False
                    ),
                }
            )

        for _, sequence, positive, negative in batches:
            batch_size, length = sequence.shape
            rows = torch.arange(batch_size, device=self.device)
            last, nonempty = self._last(sequence)
            targets = torch.stack((positive[rows, last], negative[rows, last]), -1)
            valid = nonempty & targets.ne(0).all(-1)
            base = self._scores(sequence, targets)
            margin = base[:, 0] - base[:, 1]
            finite = torch.isfinite(base).all(-1) & torch.isfinite(margin)
            counts["sampled_rows"] += batch_size
            counts["all_padding_rows"] += int((~nonempty).sum().item())
            counts["nonempty_rows_without_both_targets"] += int(
                (nonempty & ~valid).sum().item()
            )
            counts["valid_final_targets"] += int(valid.sum().item())
            counts["nonfinite_final_pairs"] += int((valid & ~finite).sum().item())
            if (valid & finite).any():
                final_margins.append(margin[valid & finite].float().cpu())

            recent = torch.zeros_like(sequence, dtype=torch.bool)
            older = torch.zeros_like(recent)
            eligible = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            indices = [torch.where(row.ne(0))[0] for row in sequence]
            for row, slots in enumerate(indices):
                n = slots.numel()
                if n >= 3:
                    m = min(3, (n - 1) // 2)
                    recent[row, slots[-m - 1 : -1]] = True
                    older[row, slots[:m]] = True
                    eligible[row] = True
            counts["matched_history_rows"] += int(eligible.sum().item())
            for name, selected in (("recent", recent), ("older", older)):
                repacked = torch.zeros_like(sequence)
                for row in range(batch_size):
                    survivors = sequence[row][sequence[row].ne(0) & ~selected[row]]
                    if survivors.numel():
                        repacked[row, -survivors.numel() :] = survivors
                packed = self._scores(repacked, targets)
                fixed = self._scores(sequence, targets, ablate=selected)
                compare(name + "_repacked", base, packed, valid & eligible)
                compare(name + "_embedding_ablation", base, fixed, valid & eligible)
                compare(name + "_repacked_vs_ablation", fixed, packed, valid & eligible)

            any_shuffle = torch.zeros_like(eligible)
            for draw, generator in enumerate(draws):
                shuffled = sequence.clone()
                changed_rows = torch.zeros_like(eligible)
                for row, slots in enumerate(indices):
                    prefix = slots[:-1]
                    if prefix.numel() < 2 or sequence[row, prefix].unique().numel() < 2:
                        shuffle_support[draw]["no_nontrivial_prefix"] += 1
                        continue
                    shuffle_support[draw]["eligible_rows"] += 1
                    order = torch.randperm(prefix.numel(), generator=generator).to(
                        self.device
                    )
                    values = sequence[row, prefix[order]]
                    if torch.equal(values, sequence[row, prefix]):
                        shuffle_support[draw]["identity_item_permutations"] += 1
                        continue
                    shuffled[row, prefix] = values
                    changed_rows[row] = True
                    shuffle_support[draw]["nonidentity_rows"] += 1
                any_shuffle |= changed_rows
                compare(
                    "prefix_shuffle",
                    base,
                    self._scores(shuffled, targets),
                    valid & changed_rows,
                )
            counts["rows_with_any_nonidentity_shuffle"] += int(any_shuffle.sum().item())

            for row, slots in enumerate(indices):
                if (
                    structural_rows >= 4
                    or slots.numel() < 2
                    or self.network.itemnum < 2
                ):
                    continue
                structural_rows += 1
                original = sequence[row : row + 1]
                features = self._features(original)
                t = int(slots[(slots.numel() - 1) // 2].item())
                future = original.clone()
                future_slots = slots[slots > t]
                future[0, future_slots] = (
                    future[0, future_slots] % self.network.itemnum + 1
                )
                check(
                    "future_invariance_through_t",
                    features[:, : t + 1],
                    self._features(future)[:, : t + 1],
                )
                check(
                    "aligned_prefix_equivalence",
                    features[:, : t + 1],
                    self._features(original[:, : t + 1]),
                )
                if valid[row]:
                    direct = self.network.score_candidates(original, targets[row])
                    check("view_vs_network_candidate_scores", base[row], direct)
                if offsets is None:
                    continue
                readout = int(last[row].item())
                # Choose distances with observed inputs, so short histories do not
                # spend the bounded probe budget only on padding positions.
                offset_set = set(offsets)
                observed_distances = sorted(
                    readout - int(position) for position in slots.tolist()
                )
                reachable = [d for d in observed_distances if d in offset_set]
                unreachable = [d for d in observed_distances if d not in offset_set]
                tested = {0}
                if reachable:
                    tested.update(
                        (reachable[0], reachable[len(reachable) // 2], reachable[-1])
                    )
                if unreachable:
                    tested.update((unreachable[0], unreachable[-1]))
                for distance in sorted(tested):
                    position = readout - distance
                    record = {
                        "structural_row_index": structural_rows - 1,
                        "readout_position": readout,
                        "input_position": position,
                        "offset": distance,
                        "reachable": distance in offset_set,
                        "support": 0,
                        "replacement_differs": None,
                        "feature_max_abs_change": None,
                        "finite": None,
                    }
                    if original[0, position] != 0:
                        perturbed = original.clone()
                        perturbed[0, position] = (
                            original[0, position] % self.network.itemnum + 1
                        )
                        changed = self._features(perturbed)[0, readout]
                        reference = features[0, readout]
                        finite_pair = bool(
                            torch.isfinite(changed).all()
                            and torch.isfinite(reference).all()
                        )
                        record.update(
                            {
                                "support": 1,
                                "replacement_differs": True,
                                "finite": finite_pair,
                                "feature_max_abs_change": (
                                    (changed - reference).abs().max().item()
                                    if finite_pair
                                    else None
                                ),
                                "close": (
                                    bool(
                                        torch.isclose(
                                            changed, reference, rtol=1e-4, atol=1e-5
                                        ).all()
                                    )
                                    if finite_pair
                                    else False
                                ),
                            }
                        )
                    offset_checks.append(record)

        paired_summary = {}
        names = (
            "margin_change",
            "positive_score_change",
            "negative_score_change",
            "mean_absolute_score_change",
            "negative_win_rate_change",
        )
        for name, entry in paired.items():
            values = entry.pop("values")
            x = torch.cat(values) if values else None
            paired_summary[name] = {
                **entry,
                "comparisons": 0 if x is None else x.size(0),
                "distinct_users": None,
                **{
                    key: None if x is None else x[:, i].mean().item()
                    for i, key in enumerate(names)
                },
            }
        collapse = self.analyze_embeddings_collapse_cosine(sample_size=2048)
        margins, breakdown = self.analyze_pos_neg_margin_with_category_breakdown(
            batches, num_batches=4, eps=0.5, topk_categories=20
        )
        embedding = self.model.item_emb.weight.detach()[1 : self.network.itemnum + 1]
        metrics = {
            "embedding_collapse_score": collapse,
            "embedding_support": {
                "catalog_items": embedding.size(0),
                "sampled_items": (
                    min(2048, embedding.size(0)) if collapse is not None else 0
                ),
                "nonfinite_items": int(
                    (~torch.isfinite(embedding).all(-1)).sum().item()
                ),
            },
            "pos_neg_margin_mean": margins["mean"],
            "pos_neg_margin_p10": margins["p10"],
            "pos_neg_neg_beats_pos_rate": margins["neg_beats_pos_rate"],
            "pos_neg_small_margin_rate": margins["small_margin_rate"],
            "all_position_support": {
                k: margins[k]
                for k in (
                    "valid_positions",
                    "comparisons",
                    "excluded_nonfinite",
                    "distinct_users",
                )
            },
            "pos_neg_margin_category_breakdown": breakdown,
            "final_position_margins": self._summary(final_margins),
            "row_support": {
                key: int(counts[key])
                for key in (
                    "sampled_rows",
                    "all_padding_rows",
                    "nonempty_rows_without_both_targets",
                    "valid_final_targets",
                    "nonfinite_final_pairs",
                    "matched_history_rows",
                    "rows_with_any_nonidentity_shuffle",
                )
            },
            "paired_perturbations": paired_summary,
            "shuffle_support_by_draw": [
                {
                    key: int(s[key])
                    for key in (
                        "eligible_rows",
                        "no_nontrivial_prefix",
                        "identity_item_permutations",
                        "nonidentity_rows",
                    )
                }
                for s in shuffle_support
            ],
            "operator_fault_calibration": calibration,
            "structure": structure,
            "implementation_checks": checks,
            "offset_checks": offset_checks,
            "configuration": {
                "seed": 42,
                "shuffle_draw_seeds": list(range(42, 47)),
                "maximum_batches": 4,
                "maximum_rows_per_batch": 32,
                "maximum_structural_rows": 4,
                "selection": "first bounded loader rows",
                "batch_row_counts": [batch[1].size(0) for batch in batches],
                "dtype": str(embedding.dtype),
                "rtol": 1e-4,
                "atol": 1e-5,
                "distinct_users": None,
                "user_identifiers_available": False,
            },
            "deployment_reload_parity": {
                "available": False,
                "reason": "adapter has no generated-predictor reload interface; in-memory parity is separate",
            },
            "validation_rank_metrics": {
                "available": False,
                "reason": "loader supplies neither stable user IDs nor pinned validation candidate sets",
            },
        }
        definitions = {
            "embedding_collapse_score": "(mean sampled off-diagonal item cosine + 1)/2; 0.5 means mean cosine zero. Not a collapse probability or accuracy metric. Null if undefined or embeddings are nonfinite.",
            "embedding_support": "Catalog size, sampled item count, and nonfinite catalog item count; padding excluded.",
            "pos_neg_margin_mean": "Scale-dependent mean positive-minus-negative score across finite nonpadding positions with both targets nonzero.",
            "pos_neg_margin_p10": "Nearest-rank 10th percentile of the same margins.",
            "pos_neg_neg_beats_pos_rate": "Fraction of supported margins strictly below zero; ties are not negative wins. Not HR@10.",
            "pos_neg_small_margin_rate": "Fraction of supported margins below the scale-dependent threshold 0.5.",
            "all_position_support": "Valid positions before finite exclusions, finite comparisons, nonfinite exclusions; distinct users unavailable.",
            "pos_neg_margin_category_breakdown": "Item categories unavailable; category lists remain empty. Position totals are measured without category information.",
            "final_position_margins": "Same margin statistics at each row's largest nonzero index, requiring both original targets nonzero. Empty summaries have null values and zero comparisons.",
            "row_support": "Bounded loader-row counts, which can include repeated users. All-padding and missing-target rows are excluded from final comparisons.",
            "paired_perturbations": "Means of changed-minus-baseline margins and scores on identical original targets. Recent/older delete equal disjoint sets of m=min(3,(n-1)//2), retaining the final event. Repacking left-pads survivors; embedding ablation zeros only selected input embedding outputs while preserving positions. Their difference measures intervention implementation sensitivity, not an isolated causal distance effect. Shuffle pools nonidentity comparisons across five draws; repeated rows are correlated. Null means no finite supported comparisons.",
            "shuffle_support_by_draw": "Rows with at least two distinct prefix item values, skipped identity item permutations, and actual nonidentity draws; final event, padding, length and item multiset are preserved.",
            "structure": "Exact offset-set propagation through both actual stride-one convolution dilations and residual unions for the supplied causal NextItNet template. Span is maximum offset plus one, distinct from reachable count. Unsupported modules invalidate offset claims. Assumes supplied forward topology; empirical checks do not prove completeness.",
            "operator_fault_calibration": "Bounded local indexed-sum convolution reference with supplied-source dilation semantics and independently reconstructed left padding. Kernel width follows fitted tensor shape, so this is not a complete independent architecture oracle. Calibration rows at the first seeded site determine the frozen tolerance; held-out sites receive clean controls before mutations and cannot inflate it. Unsupported sites are excluded explicitly; clean mismatches stop calibration claims. Controls are separate from mutation denominators. Operator/severity summaries separate finite local comparisons, local alarms, logit alarms and tie-sensitive ordered top-10 disagreements within fixed IDs 1..min(itemnum,32). These candidates need not contain training targets; results are not HR@10. Tensor recomputation is not full checkpoint reload or deployment parity. A deliberate exception verifies temporary-hook cleanup and fixed-input output agreement only, not full execution-state restoration. Whole-block reference, synthetic primitive calibration, readout faults and held-out-family generalization remain unavailable. Rows may share users; no confidence interval or validation accuracy is inferred.",
            "implementation_checks": "Finite checks and numerical comparisons with recorded tolerances: future changes versus features through an interior position, aligned prefix versus full features, and view versus actual-module candidate scoring. No observations means untested, not passed.",
            "offset_checks": "Bounded input replacements at observed nonpadding offsets relative to final readout; structural row indices identify sampled rows, not users. Up to three reachable and two unreachable distances are selected per row. Missing unreachable observations mean untested, not passed. Unreachable effects beyond tolerance suggest implementation or dependency-analysis problems. No response at reachable offsets can reflect inactive nonlinearities; it does not prove irrelevance.",
            "configuration": "Bounded first-row selection, RNG seeds, tolerances and feature dtype. Global RNG states, module modes and temporary hooks are restored. No stable user identifiers or user-level confidence intervals.",
            "deployment_reload_parity": "Unavailable here. Requires separate host execution of generated-predictor reload on identical category routes, mappings, truncation, left padding, histories and candidates; in-memory scoring checks do not establish it.",
            "validation_rank_metrics": "Unavailable without permitted fixed validation candidates and stable IDs. No benchmark HR@10, ranks, overlap or user bootstrap is inferred from sampled positive/negative pairs.",
        }
        return {"metrics": metrics, "metric_definitions": definitions}

    ### <<< Self_EvolveRec-BLOCK-END
