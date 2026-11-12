import torch
import numpy as np
import torch.nn.functional as F
import json
import random

### >>> Self_EvolveRec-BLOCK-START: Use the supplied inline diagnostic calibration
from collections import Counter

### <<< Self_EvolveRec-BLOCK-END
### >>> Self_EvolveRec-BLOCK-START: Independent references and isolated single-fault calibration
"""Bounded diagnostic calibration; never fits or saves a model."""
import copy
import hashlib
import random
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

CATEGORIES = (
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
)
ATOL, RTOL = 1e-6, 1e-5
OPERATORS = (
    ("endpoint_count", "development", "readout index = nonzero count minus one"),
    ("future_access", "development", "first block output adds next-position input"),
    ("input_offset", "development", "cyclic input-only item index shift"),
    ("input_swap", "held_out", "input-only swap of first and last valid item IDs"),
    ("score_scale", "held_out", "multiply final candidate logits by 1.01"),
)


def unavailable(reason):
    return {"value": None, "reason": reason}


def endpoint_reference(sequence, features):
    # Deliberately use Python enumeration rather than the production selector.
    rows = []
    for row, ids in zip(features, sequence.detach().cpu().tolist()):
        indices = [i for i, item in enumerate(ids) if item != 0]
        rows.append(row[indices[-1]] if indices else torch.zeros_like(row[0]))
    return torch.stack(rows)


def different(a, b):
    return a.shape != b.shape or not bool(
        torch.allclose(a, b, atol=ATOL, rtol=RTOL, equal_nan=False)
    )


def convolution_reference(conv, x):
    # Explicit cross-correlation: no production padding or convolution helper.
    if conv.groups != 1 or conv.stride != (1,) or conv.padding != (0,):
        raise ValueError("Unsupported convolution reference configuration")
    result = x.new_zeros((x.shape[0], conv.out_channels, x.shape[-1]))
    weight = conv.weight.detach()
    for t in range(x.shape[-1]):
        for k in range(conv.kernel_size[0]):
            source = t - (conv.kernel_size[0] - 1 - k) * conv.dilation[0]
            if source >= 0:
                result[:, :, t] += x[:, :, source] @ weight[:, :, k].T
    if conv.bias is not None:
        result += conv.bias.detach()[None, :, None]
    return result


def block_reference(block, x, mask):
    def norm(layer, value):
        return F.layer_norm(
            value, layer.normalized_shape, layer.weight, layer.bias, layer.eps
        )

    residual = x * mask
    y = (norm(block.norm1, residual) * mask).transpose(1, 2)
    y = convolution_reference(block.conv1, y).transpose(1, 2) * mask
    y = F.gelu(y, approximate=block.activation.approximate) * mask
    y = (norm(block.norm2, y) * mask).transpose(1, 2)
    y = convolution_reference(block.conv2, y).transpose(1, 2) * mask
    return (residual + y) * mask


def summarize(records):
    valid = [r for r in records if r.get("valid")]

    def ratio(numerator, denominator):
        return (
            numerator / denominator if denominator else unavailable("Zero denominator")
        )

    counts = {
        key: sum(bool(r.get(key)) for r in records)
        for key in (
            "attempted",
            "valid",
            "reached",
            "infected",
            "output_changing",
            "detected",
            "old_detected",
        )
    }
    changed = [r for r in valid if r["output_changing"]]
    infected = [r for r in valid if r["infected"]]
    counts.update(
        activation_coverage=ratio(len(infected), len(valid)),
        conditional_score_fault_sensitivity=ratio(
            sum(r["detected"] for r in changed), len(changed)
        ),
        state_infection_sensitivity=ratio(
            sum(r["detected"] for r in infected), len(infected)
        ),
        end_to_end_yield=ratio(sum(r["detected"] for r in valid), len(valid)),
        additional_detections=sum(
            r["detected"] and not r["old_detected"] for r in valid
        ),
        alarms_without_score_change=sum(
            r["detected"] and not r["output_changing"] for r in valid
        ),
    )
    return counts


@torch.no_grad()
def calibrate_adapter(view, device):
    network = view.model
    if not isinstance(network, torch.nn.Module):
        raise TypeError("view.model must be the actual trained nn.Module")
    device = torch.device(device)
    modes = [(m, m.training) for m in network.modules()]
    py_state, np_state = random.getstate(), np.random.get_state()
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    handles, mutant = [], None
    calls = {"encoder": 0, "scorer": 0, "readout": 0}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_end = torch.cuda.Event(enable_timing=True)
        gpu_start.record(torch.cuda.current_stream(device))
    started = time.perf_counter()
    report = {
        "rules": {
            "atol": ATOL,
            "rtol": RTOL,
            "integer_comparison": "exact",
            "tie_order": "ascending item ID",
            "threshold_adaptation": False,
        },
        "operators": [
            {"name": n, "partition": p, "behavior": b} for n, p, b in OPERATORS
        ],
        "real_fault_utility": unavailable(
            "No independently confirmed real faults supplied"
        ),
        "operator_generalization": unavailable(
            "Only one held-out implementation per supported class"
        ),
        "checkpoint_reload": unavailable(
            "No explicit checkpoint reload interface supplied"
        ),
        "export_agreement": unavailable(
            "No actual exported-predictor callable supplied"
        ),
        "semantic_mapping": unavailable(
            "No independent preprocessing mapping anchors supplied"
        ),
        "mapping_round_trip": unavailable(
            "No raw-ID encoder/decoder contract supplied"
        ),
        "complete_network_reference": unavailable(
            "Reference covers residual blocks and readout only"
        ),
        "intervals": unavailable(
            "Dependent operator cases; no independent Bernoulli outcome declared"
        ),
        "artifact_writing_seconds": 0.0,
        "shared_dependencies": [
            "trained weights",
            "Torch arithmetic",
            "mapped training histories",
            "clean-model features and logits used as a paired differential oracle",
        ],
        "detector_scope": (
            "Augmented detection includes a clean-model differential oracle. "
            "It does not estimate standalone detection of unknown production faults. "
            "Input mapping mutations test disagreement with the supplied clean "
            "pipeline, not independently anchored semantic mapping correctness."
        ),
        "reference_policy": (
            "Fixed float32 tolerances; no threshold fitting or adaptation. "
            "Witness users are reported separately and do not tune detectors."
        ),
    }
    try:
        network.eval()
        category = getattr(view, "dataset_name", getattr(network, "dataset_name", None))
        report["category"] = category if category in CATEGORIES else None
        if category not in CATEGORIES:
            report["calibration"] = unavailable(
                "Explicit supported category identity unavailable"
            )
            return report
        if (
            type(network).__name__ != "NextItNetModel"
            or not callable(getattr(network, "score_candidates", None))
            or not callable(getattr(network, "encode_sequence", None))
        ):
            report["calibration"] = unavailable(
                "Unsupported production readout/encoder interface"
            )
            return report
        dtype = view.item_emb.weight.dtype
        report["provenance"] = {
            "architecture": type(network).__name__,
            "precision": str(dtype),
            "width": int(network.maxlen),
            "padding": "left; reserved item 0",
            "checkpoint_identity": getattr(view, "checkpoint_identity", None),
            "checkpoint_identity_reason": "Host value only; never inferred from paths",
            "configuration": {
                k: getattr(network, k, None)
                for k in ("itemnum", "maxlen", "hidden_units")
            },
        }
        if dtype != torch.float32:
            report["calibration"] = unavailable(
                "Only the preregistered float32 tolerance is supported"
            )
            return report
        limit, width = int(network.itemnum), int(network.maxlen)
        if limit < 2 or width < 4:
            report["calibration"] = unavailable(
                "Fixtures require at least two items and width four"
            )
            return report
        histories = view.user_train.get("History", view.user_train)
        if not isinstance(histories, dict):
            raise TypeError("Expected mapped training histories")
        canonical = [str(uid) for uid in histories]
        if len(set(canonical)) != len(canonical):
            report["calibration"] = unavailable(
                "Canonical string user IDs are not unique"
            )
            return report
        eligible, excluded = [], CounterLike()
        observed = set()
        for uid in sorted(histories, key=str):
            h = np.asarray(histories[uid])
            if h.ndim != 1 or h.dtype.kind not in "iu" or np.any((h < 1) | (h > limit)):
                excluded.add("invalid_mapping")
                continue
            observed.update(map(int, h))
            if len(h) >= 2:
                eligible.append(uid)
            else:
                excluded.add("empty_prefix")
        rows_match = len(view.item_emb.weight) == limit + 1
        report["internal_mapping"] = {
            "embedding_rows": len(view.item_emb.weight),
            "expected_rows": limit + 1,
            "embedding_rows_match": rows_match,
            "excluded_histories": dict(excluded),
            "padding_row_zero": bool(view.item_emb.weight[0].eq(0).all()),
            "semantic_correspondence": "unavailable without independent anchors",
        }
        if not rows_match:
            report["calibration"] = unavailable(
                "Embedding rows disagree with the declared item vocabulary"
            )
            return report
        rng = np.random.default_rng(42)
        chosen = sorted(
            rng.choice(len(eligible), min(32, len(eligible)), replace=False).tolist()
        )
        users = [eligible[i] for i in chosen]
        split = len(users) // 2
        report["selected_ids"] = [str(u) for u in users]
        report["witness_ids"] = [str(u) for u in users[:split]]
        report["evaluation_ids"] = [str(u) for u in users[split:]]

        # Exercise the actual readout on a disposable network with known features.
        mutant = copy.deepcopy(network).eval()
        fixtures = torch.tensor(
            [[0, 0, 1, 2], [0, 0, 0, 1], [1, 2, 1, 2], [0, 0, 0, 0]], device=device
        )
        known = (
            torch.arange(1, 1 + 4 * network.hidden_units, device=device, dtype=dtype)
            .reshape(1, 4, -1)
            .expand(4, -1, -1)
        )
        mutant.encode_sequence = types.MethodType(lambda self, seq: known, mutant)
        items = torch.tensor([1, limit], device=device)
        expected = endpoint_reference(fixtures, known) @ network.item_emb(items).T
        actual = mutant.score_candidates(fixtures, items)
        report["endpoint_fixture"] = {
            "expected_indices": [3, 3, 3, -1],
            "production_readout_alarm": different(actual, expected),
            "empty_convention": "zero logits",
            "scope": "actual score_candidates with injected encoder features",
        }
        del mutant
        mutant = None

        # Compare every supported residual block on tiny deterministic states.
        block_checks = []
        for index, block in enumerate(network.blocks):
            if type(block).__name__ != "NextItNetBlock" or not isinstance(
                block.activation, torch.nn.GELU
            ):
                block_checks.append(unavailable("Unsupported block equations"))
                continue
            # Cover both dilated convolutions, including nonzero earlier taps.
            length = 2 * (block.kernel_size - 1) * block.dilation + 3
            x = torch.linspace(
                -0.5,
                0.5,
                length * network.hidden_units,
                device=device,
                dtype=dtype,
            ).reshape(1, length, -1)
            mask = x.new_ones((1, length, 1))
            mask[:, :2] = 0
            expected = block_reference(block, x, mask)
            actual = block(x, mask)
            block_checks.append(
                {
                    "block": index,
                    "alarm": different(actual, expected),
                    "max_absolute_error": float((actual - expected).abs().max()),
                }
            )
        report["block_reference"] = block_checks
        cases = []
        catalog = np.asarray(sorted(observed), dtype=np.int64)
        for index, uid in enumerate(users):
            history = list(map(int, histories[uid]))
            retained = history[:-1][-width:]
            seq = torch.zeros((1, width), device=device, dtype=torch.long)
            seq[0, -len(retained) :] = torch.tensor(retained, device=device)
            digest = hashlib.sha256(str(uid).encode()).digest()
            local = np.random.default_rng([42, int.from_bytes(digest[:8], "little")])
            pool = catalog[~np.isin(catalog, history)]
            negatives = local.choice(pool, min(99, len(pool)), replace=False).tolist()
            candidates = torch.tensor([history[-1]] + negatives, device=device)
            cases.append((uid, seq, candidates, index < split, history))

        def encode(model, seq):
            calls["encoder"] += 1
            return model.log2feats(seq)

        def score(model, features, candidates):
            calls["scorer"] += 1
            return model.predict_score(features, candidates)

        def assess(model, seq, candidates, operator=None):
            features = encode(model, seq)
            if operator in ("input_offset", "input_swap"):
                # Input mapping mutation is confined to the encoder pipeline.
                mapped = seq.clone()
                valid = mapped.ne(0)
                if operator == "input_offset":
                    mapped[valid] = mapped[valid] % limit + 1
                else:
                    first, last = mapped.eq(1), mapped.eq(limit)
                    mapped[first], mapped[last] = limit, 1
                features = encode(model, mapped)
                infected = bool(mapped.ne(seq).any())
            else:
                infected = None
            selected = endpoint_reference(seq, features)
            if operator == "endpoint_count":
                wrong = seq.ne(0).sum(1).sub(1).clamp_min(0)
                selected = features[torch.arange(len(seq), device=device), wrong]
                expected_indices = torch.tensor(
                    [
                        max((i for i, item in enumerate(row) if item != 0), default=-1)
                        for row in seq.detach().cpu().tolist()
                    ],
                    device=device,
                )
                infected = bool(wrong.ne(expected_indices).any())
            # Preserve the production method's broadcasting convention.
            logits = score(model, selected.unsqueeze(1), candidates).reshape(-1)
            if operator == "score_scale":
                before = logits.clone()
                logits = logits * 1.01
                infected = not torch.equal(logits, before)
            return features, selected, logits, infected

        clean, records = [], {name: [] for name, _, _ in OPERATORS}
        references = {}
        for uid, seq, candidates, witness, history in cases:
            features, selected, logits, _ = assess(network, seq, candidates)
            expected_head = selected @ network.item_emb(candidates).T
            actual_readout = network.score_candidates(seq, candidates).reshape(-1)
            calls["readout"] += 1
            production_features = view.log2feats(seq)
            view_scores = view.predict_score(selected, candidates).reshape(-1)
            repeated = assess(network, seq, candidates)[2]
            reverse = assess(network, seq, candidates.flip(0))[2].flip(0)
            batch = encode(network, seq.repeat(2, 1))
            valid_positions = seq[0].nonzero().flatten()
            causal_alarm = False
            if len(valid_positions) >= 2:
                pivot = int(valid_positions[-2])
                altered = seq.clone()
                future = torch.arange(seq.size(1), device=device) > pivot
                future &= seq[0].ne(0)
                altered[0, future] = altered[0, future] % limit + 1
                altered_features = encode(network, altered)
                causal_alarm = different(
                    features[:, : pivot + 1], altered_features[:, : pivot + 1]
                )
            alarms = {
                "causal_prefix": causal_alarm,
                "repeatability": different(logits, repeated),
                "candidate_permutation": different(logits, reverse),
                "batch_consistency": different(features, batch[:1])
                or different(batch[:1], batch[1:]),
                "explicit_head": different(logits, expected_head.reshape(-1)),
                "actual_readout": different(actual_readout, expected_head.reshape(-1)),
                "view_encoder": different(features, production_features),
                "view_scorer": different(view_scores, expected_head.reshape(-1)),
            }
            references[str(uid)] = (features, selected, logits)
            clean.append({"user_id": str(uid), "witness": witness, "alarms": alarms})
        report["clean_records"] = clean
        evaluation_clean = [r for r in clean if not r["witness"]]
        old_names = ("repeatability", "batch_consistency", "causal_prefix")
        report["clean_controls"] = {
            "eligible_cases": len(evaluation_clean),
            "old_alarm_cases": sum(
                any(r["alarms"][k] for k in old_names) for r in evaluation_clean
            ),
            "augmented_alarm_cases": sum(
                any(r["alarms"].values()) for r in evaluation_clean
            ),
        }
        nclean = len(evaluation_clean)
        report["clean_controls"]["specificity"] = (
            1 - report["clean_controls"]["augmented_alarm_cases"] / nclean
            if nclean
            else unavailable("No evaluation users")
        )
        for operator, partition, behavior in OPERATORS:
            mutant = copy.deepcopy(network).eval()
            trace = {}
            try:
                if operator == "future_access":

                    def future_hook(module, inputs, output):
                        x, mask = inputs
                        following = torch.cat((x[:, 1:], torch.zeros_like(x[:, :1])), 1)
                        changed = (output + following) * mask
                        trace["reached"] = True
                        trace["infected"] = not torch.equal(changed, output)
                        return changed

                    handles.append(mutant.blocks[0].register_forward_hook(future_hook))
                for uid, seq, candidates, witness, history in cases:
                    row = {
                        "user_id": str(uid),
                        "witness": witness,
                        "attempted": True,
                        "valid": False,
                        "partition": partition,
                        "candidate_ids": candidates.tolist(),
                        "original_length": len(history),
                        "retained_length": min(width, len(history) - 1),
                        "repeated_target": history[-1] in history[:-1],
                    }
                    try:
                        clean_features, clean_selected, clean_logits = references[
                            str(uid)
                        ]
                        trace.clear()
                        features, selected, logits, infected = assess(
                            mutant, seq, candidates, operator
                        )
                        reached = True
                        if operator == "future_access":
                            reached = trace.get("reached", False)
                            infected = trace.get("infected", False)
                        # Old causal detector uses a future replacement at an intermediate position.
                        valid_positions = seq[0].nonzero().flatten()
                        causal_alarm = False
                        if len(valid_positions) >= 2:
                            pivot = int(valid_positions[-2])
                            altered = seq.clone()
                            altered[0, pivot + 1 :] = (
                                altered[0, pivot + 1 :] % limit + 1
                            )
                            altered_features = assess(
                                mutant, altered, candidates, operator
                            )[0]
                            causal_alarm = different(
                                features[:, : pivot + 1],
                                altered_features[:, : pivot + 1],
                            )
                        repeated = assess(mutant, seq, candidates, operator)[2]
                        doubled = encode(mutant, seq.repeat(2, 1))
                        # Mapping mutations are applied by assess, so test their batch path identically.
                        if operator in ("input_offset", "input_swap"):
                            doubled = assess(
                                mutant, seq.repeat(2, 1), candidates, operator
                            )[0]
                        old = (
                            different(logits, repeated)
                            or different(features, doubled[:1])
                            or causal_alarm
                        )
                        reference_logits = (
                            clean_selected @ network.item_emb(candidates).T
                        )
                        # Apply the clean candidate-permutation control to mutants too.
                        reversed_logits = assess(
                            mutant, seq, candidates.flip(0), operator
                        )[2].flip(0)
                        augmented = (
                            old
                            or different(logits, reversed_logits)
                            or different(selected, clean_selected)
                            or different(logits, reference_logits.reshape(-1))
                        )
                        order = lambda s: candidates[
                            torch.argsort(s, descending=True, stable=True)
                        ]
                        # Sort item IDs first so stable score sorting has the declared tie rule.
                        item_order = torch.argsort(candidates)
                        rank_ids = lambda s: candidates[item_order][
                            torch.argsort(s[item_order], descending=True, stable=True)
                        ]
                        row.update(
                            valid=True,
                            reached=bool(reached),
                            infected=bool(infected),
                            output_changing=different(logits, clean_logits),
                            detected=bool(augmented),
                            old_detected=bool(old),
                            rank_changed=not torch.equal(
                                rank_ids(logits), rank_ids(clean_logits)
                            ),
                            top10_changed=set(rank_ids(logits)[:10].tolist())
                            != set(rank_ids(clean_logits)[:10].tolist()),
                            max_score_difference=float(
                                (logits - clean_logits).abs().max()
                            ),
                            intermediate_feature_changed=different(
                                features, clean_features
                            ),
                            oracle_dependency="paired clean-model features and scores",
                        )
                    except (RuntimeError, ValueError, IndexError, TypeError) as exc:
                        row["reason"] = type(exc).__name__ + ": " + str(exc)
                    records[operator].append(row)
            finally:
                for handle in handles:
                    handle.remove()
                handles.clear()
                del mutant
                mutant = None
        report["case_records"] = records
        report["evaluation_summary"] = {
            name: summarize([r for r in rows if not r["witness"]])
            for name, rows in records.items()
        }
        report["witness_summary"] = {
            name: summarize([r for r in rows if r["witness"]])
            for name, rows in records.items()
        }
        report["scope"] = (
            "Single-component wrapper/hook mutants; no fitting, saving or production mutation"
        )
        return report
    finally:
        for handle in handles:
            handle.remove()
        mutant = None
        for module, mode in modes:
            module.training = mode
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)
            gpu_end.record(torch.cuda.current_stream(device))
            torch.cuda.synchronize(device)
            report["gpu_elapsed_seconds"] = gpu_start.elapsed_time(gpu_end) / 1000.0
            report["memory"] = {
                "process_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "scope": "Process lifetime peak; counters not reset by diagnostic",
                "incremental_peak_bytes": None,
            }
        else:
            report["gpu_elapsed_seconds"] = None
        report["elapsed_seconds"] = time.perf_counter() - started
        report["operation_counts"] = calls
        report["operation_count_scope"] = (
            "Explicit helper calls; readout-internal forwards and fixtures excluded"
        )


class CounterLike(dict):
    def add(self, key):
        self[key] = self.get(key, 0) + 1


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
    ### >>> Self_EvolveRec-BLOCK-START: Evidence-limited temporal diagnosis interpretation
    - Distinguish position-weighted training separation, training-user score sensitivity,
      in-sample training-prefix ranking audits, and held-out validation ranking.
      Held-out ranking is unavailable through this adapter.
    - Under nested negative sets, target rank cannot improve and hit-at-10 cannot
      increase; a strict decline is not inevitable and does not establish poor
      generalization. Exact hypergeometric expectations describe
      uniform sampling conditional on the fitted scores and declared tie order.
    - Candidate repetitions are not independent users. Paired bootstrap intervals
      condition on repetition zero; repetition SD measures candidate variability,
      and SD/sqrt(10) is only Monte Carlo error with users and models fixed.
    - Report depth eligibility, common-user counts, strata and exclusions. Raw
      tenth-negative margins depend on score scale. Training-prefix targets may
      have been optimized during training; do not infer validation gains, current
      intent, causality, memorization or exposure bias from these diagnostics.
    - This category report cannot establish a four-category macro result. Combine
      categories equally only when all four category reports are available.
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
    ### >>> Self_EvolveRec-BLOCK-START: Interpret calibration with explicit denominators
    - Calibration measures synthetic fault detection, not validation quality.
    - Distinguish valid, reached, infected, output-changing and detected cases.
      Sample inactivity does not establish semantic equivalence.
    - Report clean alarms separately from additional mutant detections. State
      denominators, missing interfaces and shared reference dependencies.
    - Fixed operator implementations do not estimate real-fault detection,
      repair success or generalization across implementations.
    - Missing calibration or reference measurements cannot establish health.
    - Augmented calibration uses clean-model features and scores as a paired
      differential oracle. Detection against this oracle does not establish that
      the adapter can identify unknown faults without a correct model reference.
    - Mapping perturbation detection does not verify raw-item semantic mapping;
      that requires independent preprocessing anchors.
    - State infection uses exact changed indices or intermediate tensors;
      output-changing cases use the frozen floating-point tolerance.
    - Keep it short: core_findings <= 3 bullets, key_implications <= 3 bullets.
    ### <<< Self_EvolveRec-BLOCK-END
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

    ### >>> Self_EvolveRec-BLOCK-START: Validated candidate-depth calibration with bounded bootstrap memory
    @torch.no_grad()
    def _candidate_depth_audit(self, exact_limit=50):
        import hashlib
        import math

        # DEBUG: Replace unavailable SciPy with the exact hypergeometric CDF
        # using standard-library integer combinations. This preserves the
        # conditional sampling expectation without floating-point underflow.
        def hypergeom_cdf(k, N, M, m):
            if not (0 <= M <= N and 0 <= m <= N):
                raise ValueError("Invalid hypergeometric population or sample size")
            lower = max(0, m - (N - M))
            upper = min(m, M)
            cutoff = math.floor(k)
            if cutoff < lower:
                return 0.0
            if cutoff >= upper:
                return 1.0
            favorable = sum(
                math.comb(M, successes) * math.comb(N - M, m - successes)
                for successes in range(lower, cutoff + 1)
            )
            return favorable / math.comb(N, m)

        view, network = self.view, self.view.model
        histories = self.user_train.get("History", self.user_train)
        if not isinstance(histories, dict):
            raise TypeError("Candidate audit requires mapped training histories")
        if int(network.maxlen) <= 0:
            raise ValueError("Candidate audit requires a positive model.maxlen")
        if not isinstance(exact_limit, (int, np.integer)) or exact_limit < 0:
            raise ValueError("exact_limit must be a nonnegative integer")
        category = str(
            getattr(view, "dataset_name", getattr(network, "dataset_name", "unknown"))
        )
        categories = [
            "Movies_and_TV",
            "Industrial_and_Scientific",
            "Electronics",
            "CDs_and_Vinyl",
        ]
        if category not in categories:
            return {
                "value": None,
                "reason": "Explicit supported category identity is unavailable",
            }
        category_index = categories.index(category)

        def local_rng(uid, repetition):
            digest = hashlib.sha256(str(uid).encode("utf-8")).digest()
            return np.random.default_rng(
                np.random.SeedSequence(
                    [
                        42,
                        category_index,
                        int.from_bytes(digest[:8], "little"),
                        repetition,
                    ]
                )
            )

        real_limit = min(int(network.itemnum), len(view.item_emb.weight) - 1)
        frequencies, eligible, exclusions = Counter(), [], Counter()
        ordered = sorted(histories, key=lambda uid: (type(uid).__name__, str(uid)))
        for uid in ordered:
            history = np.asarray(histories[uid], dtype=np.int64)
            if history.ndim != 1 or np.any((history < 1) | (history > real_limit)):
                exclusions["unmapped_or_invalid_history"] += 1
                continue
            frequencies.update(history.tolist())
            if len(history) < 2:
                exclusions["no_training_prefix"] += 1
            else:
                eligible.append(uid)
        selection_rng = np.random.default_rng(
            np.random.SeedSequence([42, category_index])
        )
        chosen = sorted(
            selection_rng.choice(
                len(eligible), min(200, len(eligible)), replace=False
            ).tolist()
        )
        users = [eligible[i] for i in chosen]
        exact_indices = set(
            selection_rng.choice(
                len(users), min(max(0, exact_limit), 50, len(users)), replace=False
            ).tolist()
        )
        catalog = np.asarray(sorted(frequencies), dtype=np.int64)
        by_frequency = sorted(frequencies, key=lambda item: (-frequencies[item], item))
        bands = {
            item: (
                "head"
                if rank < int(np.ceil(0.2 * len(catalog)))
                else "middle" if rank < int(np.ceil(0.5 * len(catalog))) else "tail"
            )
            for rank, item in enumerate(by_frequency)
        }
        depths, records, exact, unavailable = (99, 499, 999), [], [], Counter()
        recommended, recommendation_frequencies = set(), []

        def scores(feature, items):
            chunks = []
            for start in range(0, len(items), 4096):
                item_tensor = torch.as_tensor(
                    items[start : start + 4096], device=self.device, dtype=torch.long
                )
                value = (
                    view.predict_score(feature, item_tensor)
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
                if len(value) != len(item_tensor) or not np.isfinite(value).all():
                    raise RuntimeError(
                        "Candidate audit requires one finite score per item"
                    )
                chunks.append(value)
            return np.concatenate(chunks) if chunks else np.empty(0)

        def preceding(items, values, target, positive):
            return (values > positive) | ((values == positive) & (items < target))

        for user_index, uid in enumerate(users):
            history = np.asarray(histories[uid], dtype=np.int64)
            prefix, target = history[:-1], int(history[-1])
            sequence = torch.zeros(
                (1, int(network.maxlen)), dtype=torch.long, device=self.device
            )
            retained = prefix[-sequence.shape[1] :]
            sequence[0, -len(retained) :] = torch.as_tensor(
                retained, device=self.device
            )
            endpoint = torch.where(sequence[0] != 0)[0][-1]
            feature = view.log2feats(sequence)[0, endpoint]
            positive = float(scores(feature, np.asarray([target]))[0])
            pool = catalog[~np.isin(catalog, history)]
            N = len(pool)
            row = {
                "user_id": str(uid),
                "prefix_length": len(prefix),
                "retained_length": len(retained),
                "cutoff_training_index": len(history) - 1,
                "target": target,
                "pool_size": N,
                "frequency_band": bands[target],
                "length_bin": (
                    "1-5"
                    if len(prefix) <= 5
                    else "6-20" if len(prefix) <= 20 else "21+"
                ),
                "repeated_target": bool(target in prefix),
                "repetitions": [],
            }
            for repetition in range(10):
                negatives = local_rng(uid, repetition).choice(
                    pool, min(999, N), replace=False
                )
                negative_scores = scores(feature, negatives)
                measurements = {}
                previous_rank = 1
                for depth in depths:
                    if N < depth:
                        unavailable[str(depth) + ":insufficient_unique_negatives"] += 1
                        continue
                    rank = 1 + int(
                        preceding(
                            negatives[:depth], negative_scores[:depth], target, positive
                        ).sum()
                    )
                    if rank < previous_rank:
                        raise RuntimeError(
                            "Nested candidate ranks must be nondecreasing"
                        )
                    previous_rank = rank
                    measurements[str(depth)] = {
                        "rank": rank,
                        "hit_at_10": int(rank <= 10),
                        "margin_to_tenth_negative": positive
                        - float(np.partition(negative_scores[:depth], -10)[-10]),
                    }
                row["repetitions"].append(measurements)
            records.append(row)
            if user_index in exact_indices:
                M, top_items, top_scores = 0, np.empty(0, dtype=np.int64), np.empty(0)
                for start in range(0, N, 4096):
                    items = pool[start : start + 4096]
                    values = scores(feature, items)
                    M += int(preceding(items, values, target, positive).sum())
                    top_items, top_scores = np.concatenate(
                        (top_items, items)
                    ), np.concatenate((top_scores, values))
                    keep = np.lexsort((top_items, -top_scores))[:10]
                    top_items, top_scores = top_items[keep], top_scores[keep]
                recommended.update(top_items.tolist())
                recommendation_frequencies.extend(
                    frequencies[int(item)] for item in top_items
                )
                exact.append(
                    {
                        "user_id": str(uid),
                        "N": N,
                        "M": M,
                        "full_eligible_pool_rank": 1 + M,
                        "expected": {
                            str(m): {
                                # DEBUG: Execute the dependency-free exact CDF.
                                "hit_at_10": hypergeom_cdf(9, N, M, m),
                                "rank": 1 + m * M / N,
                            }
                            for m in depths
                            if m <= N
                        },
                        "top10_eligible_negative_items": top_items.tolist(),
                    }
                )
        fields = ("hit_at_10", "rank", "margin_to_tenth_negative")
        bootstrap_weights = {}

        def aggregate(rows, requested):
            common = [
                r for r in rows if all(str(d) in r["repetitions"][0] for d in requested)
            ]
            if not common:
                return {
                    "users": 0,
                    "estimates": None,
                    "reason": "No common eligible users",
                }
            values = np.asarray(
                [
                    [
                        [r["repetitions"][rep][str(d)][field] for field in fields]
                        for d in requested
                    ]
                    for r in common
                    for rep in range(10)
                ]
            )
            values = values.reshape(len(common), 10, len(requested), len(fields))
            n = len(common)
            if n not in bootstrap_weights:
                draws = np.random.default_rng(42).integers(n, size=(2000, n))
                counts = np.zeros((2000, n), dtype=np.float64)
                np.add.at(counts, (np.arange(2000)[:, None], draws), 1.0)
                bootstrap_weights[n] = counts / n
            # Multiplicities reproduce the same paired user resamples without
            # materializing a resample x user x depth x field tensor.
            bootstrap = (bootstrap_weights[n] @ values[:, 0].reshape(n, -1)).reshape(
                2000, len(requested), len(fields)
            )
            repetition_means = values.mean(axis=0)
            estimates, contrasts = {}, {}
            for j, depth in enumerate(requested):
                estimates[str(depth)] = {
                    field: {
                        "mean_over_ten_repetitions": float(values[:, :, j, k].mean()),
                        "repetition_zero_mean": float(values[:, 0, j, k].mean()),
                        "repetition_zero_user_ci95": np.quantile(
                            bootstrap[:, j, k], [0.025, 0.975]
                        ).tolist(),
                        "repetition_sd": float(repetition_means[:, j, k].std(ddof=1)),
                        "monte_carlo_se_of_ten_repetition_mean": float(
                            repetition_means[:, j, k].std(ddof=1) / np.sqrt(10)
                        ),
                    }
                    for k, field in enumerate(fields)
                }
                if j:
                    contrasts[str(depth) + "_minus_" + str(requested[0])] = {
                        field: {
                            "repetition_zero_mean_difference": float(
                                (values[:, 0, j, k] - values[:, 0, 0, k]).mean()
                            ),
                            "paired_user_ci95": np.quantile(
                                bootstrap[:, j, k] - bootstrap[:, 0, k], [0.025, 0.975]
                            ).tolist(),
                        }
                        for k, field in enumerate(fields)
                    }
            return {
                "users": len(common),
                "estimates": estimates,
                "paired_contrasts": contrasts,
            }

        by_id = {r["user_id"]: r for r in records}
        calibration = {}
        for depth in depths:
            key = str(depth)
            cohort = [e for e in exact if key in e["expected"]]
            calibration[key] = {
                "users": len(cohort),
                "expected_hit_at_10": None,
                "observed_mean_hit_at_10": None,
            }
            if cohort:
                expected = float(
                    np.mean([e["expected"][key]["hit_at_10"] for e in cohort])
                )
                observed = float(
                    np.mean(
                        [
                            by_id[e["user_id"]]["repetitions"][rep][key]["hit_at_10"]
                            for e in cohort
                            for rep in range(10)
                        ]
                    )
                )
                calibration[key].update(
                    expected_hit_at_10=expected,
                    observed_mean_hit_at_10=observed,
                    observed_minus_expected=observed - expected,
                )
        return {
            "category": category,
            "category_seed_index": category_index,
            "eligible_training_users": len(eligible),
            "selected_users": len(users),
            "selected_fraction_of_eligible": (
                len(users) / len(eligible) if eligible else None
            ),
            "selected_user_ids": [str(uid) for uid in users],
            "exclusions": dict(exclusions),
            "unavailable_user_repetition_depths": dict(unavailable),
            "nested_monotonicity_violations": 0,
            "depth_specific": {str(d): aggregate(records, [d]) for d in depths},
            "common_eligible_comparisons": {
                str(a) + ":" + str(b): aggregate(records, [a, b])
                for a, b in ((99, 499), (99, 999), (499, 999))
            },
            "all_depth_common_users": aggregate(records, depths),
            "strata": {
                field: {
                    str(value): {
                        "selected_users": sum(r[field] == value for r in records),
                        "depths": {
                            str(d): aggregate(
                                [r for r in records if r[field] == value], [d]
                            )
                            for d in depths
                        },
                    }
                    for value in sorted(set(r[field] for r in records), key=str)
                }
                for field in ("frequency_band", "length_bin", "repeated_target")
            },
            "exact_pool_records": exact,
            "sampling_calibration": calibration,
            "eligible_pool_recommendations": {
                "audited_users": len(exact),
                "unique_top10_items": len(recommended),
                "training_catalog_size": len(catalog),
                "catalog_coverage": (
                    len(recommended) / len(catalog) if len(catalog) else None
                ),
                "mean_training_frequency": (
                    float(np.mean(recommendation_frequencies))
                    if recommendation_frequencies
                    else None
                ),
            },
            "paired_user_records": records,
        }

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Padding-aware diagnostics with bounded paired bootstrap memory and synchronized timing
    def run_full_diagnosis(self, train_loader):
        from itertools import islice
        import time

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        diagnostic_started = time.perf_counter()
        view, network = self.view, self.view.model
        rng = np.random.default_rng(42)
        user_rng = np.random.default_rng(42)
        candidate_rng = np.random.default_rng(42)
        previous_modes = [(module, module.training) for module in network.modules()]
        python_state, numpy_state = random.getstate(), np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_state = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
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
                # The supplied gate includes a cumulative prefix mean. Its
                # endpoint can depend on every retained earlier input token.
                if hasattr(network, "prefix_norm") and hasattr(network, "depth_gate"):
                    all_depths.update(range(int(network.maxlen)))
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

                sensitivity_bootstrap_weights = {}

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
                    # Reuse deterministic user multiplicities by cohort size.
                    # Every column still shares the same paired resample.
                    n = len(rows)
                    if n not in sensitivity_bootstrap_weights:
                        draws = np.random.default_rng(42).integers(n, size=(2000, n))
                        counts = np.zeros((2000, n), dtype=np.float64)
                        np.add.at(counts, (np.arange(2000)[:, None], draws), 1.0)
                        sensitivity_bootstrap_weights[n] = counts / n
                    means = sensitivity_bootstrap_weights[n] @ values
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
                calibration = calibrate_adapter(view, self.device)
                metrics["adapter_calibration"] = calibration
                definitions["adapter_calibration"] = (
                    "Diagnostic-only synthetic fault calibration on at most 32 identified "
                    "training users, with lexicographically sorted canonical IDs and local "
                    "seed-42 selection. The first floor(n/2) selected users are witnesses; "
                    "the remainder are evaluation users. Tolerances are fixed before "
                    "execution. Valid, reached, infected, output-changing and detected "
                    "are separate case indicators. Old checks are repeatability, batch "
                    "reordering and intermediate causal-prefix invariance; augmented "
                    "checks additionally compare independently selected endpoint features "
                    "and explicit tied-dot-product scores. Conditional sensitivity uses "
                    "output-changing cases, infection sensitivity uses infected cases, "
                    "and yield uses valid cases. Zero denominators are unavailable. "
                    "Clean alarms and additional detections are reported separately. "
                    "Mapping mutations affect input lookup only. Copies are unfitted, "
                    "never saved and never used for production. These are training-prefix "
                    "cases, not validation observations. Fixed synthetic operators do not "
                    "establish real-fault utility or operator generalization. Reference "
                    "and export limitations are recorded explicitly."
                )
                metrics["candidate_depth_audit"] = self._candidate_depth_audit()
                definitions["candidate_depth_audit"] = (
                    "In-sample exploratory training-prefix audit; no validation labels. "
                    "Up to 200 uniformly sampled eligible users without replacement, sorted "
                    "before selection; seed 42 plus declared category index and stable hashed "
                    "user ID for ten independent candidate repetitions. Unknown category uses "
                    "index 4 and must not be treated as a named category macro report. Each "
                    "target is the final permitted training interaction; its cutoff is a zero-based "
                    "training index, and it may have been optimized during fitting. Prefixes are "
                    "left-padded/truncated to model.maxlen and read at the largest nonzero index. "
                    "The catalog contains only mapped training-observed items. Negative pools "
                    "exclude every full training-history item, including target; this is retrospective "
                    "hindsight rather than a prospective exclusion policy. Repeated targets are "
                    "recorded before exclusion. Ten uniform ordered samples without replacement "
                    "share nested prefixes of 99/499/999 negatives. Missing depths are unavailable, "
                    "with counts in user-repetition-depth units. Ties favor ascending item ID. "
                    "Rank includes the positive; hit means rank<=10. Margin subtracts the tenth "
                    "largest negative score and is scale dependent. Nested ranks cannot improve "
                    "with depth; declining hit rates alone imply nothing about generalization. "
                    "Users receive equal weight; depth-specific populations may differ. Paired "
                    "comparisons use common eligible users and 2000 seed-42 user-cluster bootstrap "
                    "resamples conditional on candidate repetition zero. Intervals exclude training "
                    "seed, adaptive search and candidate-resampling uncertainty. Repetition SD is "
                    "computed across ten category-level summaries; SD/sqrt(10) is solely Monte "
                    "Carlo error of their average, with model and users fixed. Repetitions are never "
                    "independent users. Frequency bands use all valid permitted training interactions, "
                    "descending frequency then ascending item ID: first ceil(20%) head, next through "
                    "ceil(50%) middle, remainder tail. Length bins use prefixes before truncation. "
                    "At most 50 randomly selected audit users have exact full-eligible-pool ranks; "
                    "scores stream in chunks<=4096 and retain only the top ten negative items. "
                    "N is eligible negative count and M counts negatives preceding the target. "
                    "Conditional expected hit is hypergeom.cdf(9,N,M,m); expected rank is "
                    "1+m*M/N, applicable only to declared uniform sampling and tie order. "
                    "Expected/observed residuals measure sampling variability, not generalization. "
                    "Recommendation popularity and catalog coverage describe top-ten scores on "
                    "these same restricted negative pools, excluding known training items and the "
                    "target; they are not unrestricted-catalog recommendation quality. Existing "
                    "controlled shuffle/truncation probes remain descriptive controls. Only one "
                    "category is measured here; four-category macro summaries require all four "
                    "reports and equal category weights. No auxiliary scorer or model is fitted."
                )
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
                torch.cuda.set_rng_state_all(cuda_state)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        metrics["diagnostic_elapsed_seconds"] = time.perf_counter() - diagnostic_started
        definitions["diagnostic_elapsed_seconds"] = (
            "Wall-clock seconds for all probes, including exact eligible-pool scoring "
            "and final CUDA synchronization; prior queued CUDA work is excluded by "
            "synchronizing before timing. Diagnostic execution consumes the shared "
            "experiment budget."
        )
        return {"metrics": metrics, "metric_definitions": definitions}

    ### <<< Self_EvolveRec-BLOCK-END
