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
    ### >>> Self_EvolveRec-BLOCK-START: Evidence boundaries for BERT4Rec diagnosis
    - Missing metrics and unavailable checks are unknown, never evidence of health.
    - Structural failures must be distinguished from training-exposed recovery and sensitivity.
    - Training-history targets were exposed during training. Their recovery does not measure held-out generalization or prove memorization alone.
    - History interventions change content, positions or attention normalization. Do not infer causal preferences from them.
    - Auxiliary candidate draws do not replace or modify the host validation protocol.
    - Inspect sample counts, repeat/explore strata, candidate counts, bootstrap intervals and unavailable reasons before drawing conclusions.
    - No review text, ratings, item categories, prices, stores or simulator traits are available. Do not infer them.
    - Separate consumed-pipeline consistency, synthetic calibration, training recovery, deployment parity and host validation.
    - Inspect assertion denominators, verified injections, equivalent and unresolved cases, clean false positives and unavailable reasons.
    - Passing bounded replay does not certify the entire pipeline, gradients, convergence, deployed checkpoints or held-out gains.
    - Export score parity requires an explicitly reloaded selected checkpoint and actual public-entry-point execution; live-model identity is insufficient.
    - Cite export manifest identity, completed fixture counts, score violations, exact rank disagreements, runtime and unavailable reasons separately.
    - SCORE_PARITY_OBSERVED does not certify internal readout, parameter/buffer parity, fault detection coverage or full-catalog simulator selection.
    - Missing fault activation evidence means calibration is unavailable; never infer detected faults, false-alarm rates or semantic correctness from clean parity.
    - A supplied export runner that fails is an audit failure, not an unavailable successful check. Runner-reported provenance is conditional evidence.
    - Repeated item IDs do not establish exposure of the same target event. Surviving mutations do not establish equivalence.
    - Retain exactly one independently trained BERT4Rec checkpoint per category; do not propose independent predictor fusion.
    ### <<< Self_EvolveRec-BLOCK-END
    """


### >>> Self_EvolveRec-BLOCK-START: Bounded consumed-event replay with explicit availability and verified restoration


def audit_consumed_records(records, histories, itemnum):
    """CPU-only checks; never call the producer or run another model forward."""
    if records is None:
        return {
            "available": False,
            "reason": "No explicit consumed-record side channel supplied",
        }
    if not isinstance(records, (list, tuple)):
        raise TypeError("consumed_records must be an explicit bounded list or tuple")
    histories = histories.get("History", histories)
    counts, failures, loss_errors = Counter(), [], []
    record_metadata = []

    def check(record):
        checks = {}
        try:
            sequence = np.asarray(record["sequence"], dtype=np.int64)
            positive = np.asarray(record["positive"], dtype=np.int64)
            negative = np.asarray(record["negative"], dtype=np.int64)
            if negative.ndim == 2:
                negative = negative[..., None]
            if (
                sequence.ndim != 2
                or positive.shape != sequence.shape
                or negative.shape[:2] != sequence.shape
                or negative.ndim != 3
                or negative.shape[-1] < 1
                or len(record["trace"]) != len(sequence)
            ):
                raise ValueError("unaligned record shapes")
            length = sequence.shape[1]
            expected_seq = np.zeros_like(sequence)
            expected_pos = np.zeros_like(positive)
            expected_neg = np.zeros_like(negative)
            for row, trace in enumerate(record["trace"]):
                user = trace["user"]
                history = histories[user] if user in histories else histories[str(user)]
                history = [int(i) for i in history]
                start, stop = trace["start"], trace["stop"]
                if not (
                    0 <= start < stop <= len(history)
                    and stop >= 2
                    and start == max(0, stop - length)
                ):
                    raise ValueError("invalid crop bounds")
                if any(i < 1 or i > itemnum for i in history):
                    raise ValueError("invalid public history IDs")
                selected = trace["selected"]
                offset = length - (stop - start)
                source = [start + p - offset for p in selected]
                if (
                    not selected
                    or len(set(selected)) != len(selected)
                    or any(p < offset or p >= length for p in selected)
                    or source != trace["source_events"]
                    or trace["original_labels"] != [history[i] for i in source]
                    or len(trace["negative_seeds"]) != len(selected)
                    or trace["policy"] != "selected_targets_always_reserved_mask"
                ):
                    raise ValueError("invalid selected-event trace")
                branch = trace["branch"]
                if branch not in {"terminal", "random"}:
                    raise ValueError("unknown corruption branch")
                if branch == "terminal" and selected != [length - 1]:
                    raise ValueError("terminal target must be final source event")
                expected_seq[row, offset:] = history[start:stop]
                excluded = set(history)
                if len(excluded) >= itemnum:
                    raise ValueError("no eligible negatives")
                for p, event, seed in zip(selected, source, trace["negative_seeds"]):
                    expected_pos[row, p] = history[event]
                    expected_seq[row, p] = itemnum + 1
                    rng = random.Random(seed)
                    for j in range(negative.shape[-1]):
                        candidate = rng.randint(1, itemnum)
                        while candidate in excluded:
                            candidate = rng.randint(1, itemnum)
                        expected_neg[row, p, j] = candidate
                # Event identities, rather than item-ID inequality, establish
                # which occurrences remain visible. Earlier repeats are legal.
                visible_events = set(range(start, stop)) - set(source)
                if branch == "terminal" and any(i >= source[0] for i in visible_events):
                    raise ValueError("terminal target or suffix event exposed")
            checks["event_trace"] = True
            checks["sequence_replay"] = bool(np.array_equal(sequence, expected_seq))
            checks["same_event_labels"] = bool(np.array_equal(positive, expected_pos))
            checks["negative_replay"] = bool(np.array_equal(negative, expected_neg))
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as error:
            checks["event_trace"] = False
            checks["reason"] = str(error)
        return checks

    def passed(checks):
        return all(
            checks.get(k) is True
            for k in (
                "event_trace",
                "sequence_replay",
                "same_event_labels",
                "negative_replay",
            )
        )

    calibration = {
        name: Counter() for name in ("shift_label_index", "restore_masked_target")
    }
    controls = Counter()
    for index, record in enumerate(records[:4]):
        if not isinstance(record, dict):
            raise TypeError("Each consumed record must be a dictionary")
        record_metadata.append(
            {
                key: record.get(key)
                for key in (
                    "category",
                    "epoch",
                    "batch_index",
                    "forward_step",
                    "update_outcome",
                )
            }
        )
        checks = check(record)
        for name, value in checks.items():
            if name == "reason":
                continue
            counts[name + ":checked"] += 1
            counts[name + ":passed"] += int(value)
            if not value:
                failures.append({"record": index, "assertion": name})
        payload = record.get("loss")
        if payload is None:
            counts["actual_loss_wiring:unavailable"] += 1
        else:
            try:
                p = torch.tensor(payload["positive_scores"], dtype=torch.float64)
                n = torch.tensor(payload["negative_scores"], dtype=torch.float64)
                selected = np.argwhere(np.asarray(record["positive"]) != 0).tolist()
                if (
                    p.ndim != 1
                    or n.ndim != 2
                    or n.shape[0] != len(p)
                    or n.shape[1] < 1
                    or not len(p)
                    or selected != payload["selected_indices"]
                    or len(selected) != len(p)
                    or payload["reduction"] != "pooled_valid_target_mean"
                    or payload["auxiliary_loss"] != 0
                    or payload["regularization_in_scalar"] != 0
                ):
                    raise ValueError("unsupported or misaligned loss specification")
                if not bool(torch.isfinite(p).all() & torch.isfinite(n).all()):
                    raise ValueError("Nonfinite captured same-forward logits")
                differences = n - p[:, None]
                if (
                    payload["objective"] == "ce"
                    and payload["negative_reduction"] == "logsumexp"
                ):
                    terms = torch.logsumexp(
                        torch.cat((torch.zeros_like(p[:, None]), differences), 1), 1
                    )
                elif (
                    payload["objective"] == "bpr"
                    and payload["negative_reduction"] == "mean"
                ):
                    terms = torch.logaddexp(
                        torch.zeros_like(differences), differences
                    ).mean(1)
                else:
                    raise ValueError("unknown objective")
                recorded_terms = torch.tensor(
                    payload["target_losses"], dtype=torch.float64
                )
                actual = float(payload["actual_loss"])
                expected = float(terms.mean())
                dtype = payload["dtype"]
                tolerance = {
                    "torch.float64": 1e-10,
                    "torch.float32": 1e-5,
                    "torch.float16": 5e-3,
                    "torch.bfloat16": 5e-2,
                }[dtype]
                agreement = (
                    recorded_terms.shape == terms.shape
                    and bool(torch.isfinite(terms).all())
                    and np.isfinite(actual)
                    and bool(
                        torch.allclose(
                            terms, recorded_terms, rtol=tolerance, atol=tolerance
                        )
                    )
                    and abs(expected - actual) <= tolerance * (1 + abs(expected))
                )
                loss_errors.append(
                    {
                        "record": index,
                        "absolute": abs(expected - actual),
                        "relative": abs(expected - actual) / max(abs(expected), 1e-12),
                        "rtol": tolerance,
                        "atol": tolerance,
                        "targets": len(p),
                    }
                )
                counts["actual_loss_wiring:checked"] += 1
                counts["actual_loss_wiring:passed"] += int(agreement)
                # This verifies the captured selection metadata. It cannot
                # independently certify absent terms in the executed loss graph.
                counts["loss_selection_metadata:checked"] += 1
                counts["loss_selection_metadata:passed"] += int(
                    selected == payload["selected_indices"]
                )
                counts["ignored_positions_zero_direct_terms:unavailable"] += 1
                if not agreement:
                    failures.append(
                        {"record": index, "assertion": "actual_loss_wiring"}
                    )
            except (KeyError, TypeError, ValueError, RuntimeError) as error:
                counts["actual_loss_wiring:unavailable"] += 1
                failures.append(
                    {
                        "record": index,
                        "assertion": "loss_record_invalid",
                        "reason": str(error),
                    }
                )
        # Predeclared calibration subset: first retained record only.
        if index != 0:
            continue
        import copy

        if not passed(checks):
            controls["unresolved_clean_records"] += 1
            continue
        controls["verified_clean_controls"] += 1
        controls["false_positives"] += int(not passed(check(record)))
        # Restrict this control to event/tensor assertions. Historical loss
        # payload ordering is deliberately excluded from the control's scope.
        harmless = copy.deepcopy(record)
        harmless.pop("loss", None)
        keys = ("sequence", "positive", "negative", "trace")
        for key in keys:
            harmless[key] = list(reversed(harmless[key]))
        inverse = copy.deepcopy(harmless)
        for key in keys:
            inverse[key] = list(reversed(inverse[key]))
        if not all(inverse[key] == record[key] for key in keys):
            controls["invalid_row_permutation_controls"] += 1
        else:
            controls["verified_row_permutation_controls"] += 1
            controls["false_positives"] += int(not passed(check(harmless)))
        for operator, report in calibration.items():
            report["attempted"] += 1
            mutant = copy.deepcopy(record)
            trace = mutant["trace"][0]
            position = trace["selected"][0]
            original = record["positive"][0][position]
            if operator == "restore_masked_target":
                if record["sequence"][0][position] != itemnum + 1:
                    report["inapplicable"] += 1
                    continue
                mutant["sequence"][0][position] = original
                verified = original != itemnum + 1 and 1 <= original <= itemnum
                assertion = "sequence_replay"
            else:
                user = trace["user"]
                history = histories[user] if user in histories else histories[str(user)]
                event = trace["source_events"][0]
                other_event = event - 1 if event > 0 else event + 1
                if not 0 <= other_event < len(history):
                    report["inapplicable"] += 1
                    continue
                changed = int(history[other_event])
                if changed == original:
                    report["proven_equivalent"] += 1
                    continue
                mutant["positive"][0][position] = changed
                verified = 1 <= changed <= itemnum and changed != original
                assertion = "same_event_labels"
            if not verified:
                report["unresolved"] += 1
                continue
            field = "sequence" if operator == "restore_masked_target" else "positive"
            before = record[field][0][position]
            after = mutant[field][0][position]
            # Verify an isolated, reversible change without consulting the
            # detector. Equality with a copy of the original proves nothing.
            restored = copy.deepcopy(mutant)
            restored[field][0][position] = before
            if before == after or restored != record:
                report["invalid"] += 1
                continue
            report.setdefault("injections", []).append(
                {
                    "record": index,
                    "row": 0,
                    "position": position,
                    "field": field,
                    "before": before,
                    "after": after,
                    "violated_assertion": assertion,
                    "restoration_exact": True,
                }
            )
            report["verified_corruption"] += 1
            outcome = check(mutant).get(assertion)
            if outcome is None:
                report["detector_unavailable"] += 1
            detected = outcome is False
            report["detected"] += int(detected)
            report["surviving"] += int(not detected)
            report["restoration_exact"] += 1
    result_calibration = {}
    for name, report in calibration.items():
        summary = {
            key: int(report[key])
            for key in (
                "attempted",
                "inapplicable",
                "invalid",
                "proven_equivalent",
                "unresolved",
                "verified_corruption",
                "detected",
                "surviving",
            )
        }
        summary["restoration_exact"] = int(report["restoration_exact"])
        summary["detector_unavailable"] = int(report["detector_unavailable"])
        summary["injections"] = report.get("injections", [])
        denominator = report["verified_corruption"]
        summary["detection_coverage"] = (
            report["detected"] / denominator if denominator else None
        )
        summary["assertion"] = (
            "same_event_labels" if name == "shift_label_index" else "sequence_replay"
        )
        result_calibration[name] = summary
    denominator = (
        controls["verified_clean_controls"]
        + controls["verified_row_permutation_controls"]
    )
    return {
        "available": bool(records),
        "records": min(len(records), 4),
        "reason": None if records else "Explicit consumed-record buffer is empty",
        "record_metadata": record_metadata,
        "capture_schedule": "first two batches of first and final executed epochs; deduplicated",
        "scope": "bounded consumed batches; trace consistency, not authenticity or exhaustive certification",
        "control_scope": "event/tensor assertions only; clean validity is conditional on public-history replay, not an independent correctness oracle",
        "ignored_positions_zero_direct_terms_reason": "Selection metadata alone does not independently certify every contribution to the executed scalar loss",
        "assertion_counts": dict(counts),
        "failures": failures,
        "loss_errors": loss_errors,
        "calibration": result_calibration,
        "controls": dict(controls),
        "clean_false_positive_rate": (
            controls["false_positives"] / denominator if denominator else None
        ),
        "unsupported_edge_cases": [
            "empty training windows",
            "singleton training windows",
        ],
        "deployment_parity": None,
        "host_validation_gain": None,
    }


### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Bounded native-device export audit with explicit coverage and numerical boundaries


def audit_exported_predictor(view, exported_runner=None, frozen_manifest=None):
    """The optional runner must execute the real exported public entry point.

    No live-model scores substitute for an explicitly reloaded checkpoint.
    Runner protocol: runner(manifest=<JSON copy>, reference_records=<fixture metadata>)
    returns invocation metadata and aligned records with binary NumPy scores.
    Reference scores are withheld from the runner. Fixture metadata contains
    only IDs and candidate order; the runner must execute exported preprocessing
    from raw histories in the manifest.
    The runner owns subprocess termination, timeout and isolated fault execution.
    Missing observation capabilities remain unavailable rather than passing.
    """
    import hashlib
    import time
    from pathlib import Path

    result = {
        "available": False,
        "status": "UNAVAILABLE",
        "reason": None,
        "failures": [],
        "completed_fixtures": 0,
        "rank_disagreements": 0,
        "score_tolerance_violations": 0,
        "maximum_absolute_error": None,
        "fault_calibration": None,
        "fault_calibration_reason": "No independently verified internal fault activation records",
        "simulator_full_catalog": None,
        "simulator_full_catalog_reason": "No designated simulator-user selection runner supplied",
        "readout_observation": None,
        "readout_observation_reason": "Public score parity alone does not observe the deployed readout index",
        "parameter_buffer_parity": None,
        "parameter_buffer_parity_reason": "No deployed parameter/buffer observation supplied",
        "host_validation_gain": None,
    }
    if not callable(exported_runner):
        result["reason"] = "Actual exported-predictor runner not supplied"
        return result
    if frozen_manifest is None:
        result["reason"] = "Frozen training artifact manifest not supplied"
        return result

    started = time.monotonic()
    reference = None
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    )
    module_modes = [(module, module.training) for module in view.model.modules()]
    backend_state = {
        "matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
        "memory_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_sdp": torch.backends.cuda.math_sdp_enabled(),
    }
    result["available"] = True
    result["status"] = "FAILED"
    result["fixture_results"] = []
    result["robust_rank_inversions"] = 0
    result["maximum_relative_error"] = None
    result["relative_error_convention"] = (
        "abs(export-reference)/max(abs(reference), 1e-8)"
    )
    result["score_observations"] = 0
    result["score_tolerance_passes"] = 0
    result["feature_observations"] = 0
    result["feature_tolerance_violations"] = 0
    result["maximum_feature_absolute_error"] = None
    result["observed_readout_fixtures"] = 0
    result["native_rank_disagreements"] = 0
    result["native_rank_observations"] = 0
    result["native_rank_reason"] = "No native ranking observations supplied"
    result["aggregate_wall_cap_seconds"] = 300.0
    result["deadline_enforcement"] = (
        "Cooperative checks between operations only. The supplied runner must "
        "enforce child timeouts and termination; this adapter cannot interrupt "
        "a blocked callback or GPU operation."
    )
    result["query_log"] = []
    result["selection_fixture_checks"] = []
    result["fixture_partitions"] = {}
    result["scope"] = (
        "Same selected parameters reloaded for auditing; no additional fitted predictor or fused recommendations"
    )
    result["nonzero_errors_at_zero_reference"] = 0
    result["observed_token_fixtures"] = 0
    result["gpu_seconds"] = None
    result["gpu_seconds_reason"] = "No GPU-time accounting interface supplied"

    def canonical(value):
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

    def digest_file(path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def check_deadline():
        require(time.monotonic() - started < 300.0, "Aggregate audit deadline exceeded")

    def native_settings(device):
        return {
            "device_type": device.type,
            "dtype": "torch.float32",
            "autocast": False,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
            "flash_sdp": bool(torch.backends.cuda.flash_sdp_enabled()),
            "memory_efficient_sdp": bool(
                torch.backends.cuda.mem_efficient_sdp_enabled()
            ),
            "math_sdp": bool(torch.backends.cuda.math_sdp_enabled()),
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        }

    # Independently specified expected selections exercise tied cutoff membership,
    # exclusions, irregular final chunks and fewer than k eligible items.
    def select_fixture(ids, scores, excluded, k, width):
        retained = []
        for start in range(0, len(ids), width):
            chunk = [
                (float(score), int(item))
                for item, score in zip(
                    ids[start : start + width], scores[start : start + width]
                )
                if item not in excluded
            ]
            chunk.sort(key=lambda pair: (-pair[0], pair[1]))
            retained = sorted(
                retained + chunk[:k], key=lambda pair: (-pair[0], pair[1])
            )[:k]
        return [item for _, item in retained]

    try:
        for ids, scores, excluded, k, expected in (
            ([5, 2, 9, 1, 7], [3, 3, 2, 3, 1], set(), 2, [1, 2]),
            ([5, 2, 9, 1, 7], [3, 3, 2, 3, 1], {1, 5}, 3, [2, 9, 7]),
            ([3, 1], [0, 0], {3}, 10, [1]),
        ):
            for width in (1, 2, 4):
                observed = select_fixture(ids, scores, excluded, k, width)
                require(
                    observed == expected, "Hand-calculated selection fixture failed"
                )
                result["selection_fixture_checks"].append(
                    {
                        "chunk_size": width,
                        "expected_ids": expected,
                        "passed": True,
                    }
                )
        # Freeze a JSON copy before inspecting any outcomes. Paths are supplied
        # explicitly by the host; never search for artifacts or other trials.
        manifest = json.loads(canonical(frozen_manifest))
        result["manifest_sha256"] = hashlib.sha256(canonical(manifest)).hexdigest()
        result["diagnostic_source_sha256"] = digest_file(__file__)
        require(manifest["family"] == "bert4rec", "Unexpected model family")
        require(manifest["preprocessing_seed"] == 0, "Wrong preprocessing seed")
        require(manifest["training_seed"] == 42, "Wrong training seed")
        require(manifest["evaluation_seed"] == 42, "Wrong evaluation seed")
        require(manifest["selected_epoch"] >= 1, "Missing selected checkpoint epoch")
        require(manifest["selected_step"] >= 1, "Missing selected checkpoint step")
        require(manifest["mapping"] == "identity_1_based", "Unsupported item mapping")
        device = view.model.item_emb.weight.device
        require(device.type in {"cpu", "cuda"}, "Unsupported reference device")
        require(
            torch.device(manifest["reference_device"]) == device,
            "Reference must use the actual model device, including its CUDA index",
        )
        require(manifest["dtype"] == "torch.float32", "Unsupported audit dtype")
        require(
            all(
                not p.is_floating_point() or p.dtype == torch.float32
                for p in view.model.parameters()
            ),
            "Native non-FP32 execution requires a separately declared audit protocol",
        )
        settings = native_settings(device)
        result["reference_execution_settings"] = settings
        result["primary_cuda_comparison"] = device.type == "cuda"
        # Frozen before observing scores; never widen after a discrepancy.
        rtol, atol = 1.3e-6, 1e-5
        require(
            manifest["rtol"] == rtol and manifest["atol"] == atol,
            "Manifest tolerance differs from frozen FP32 protocol",
        )
        timeout = float(manifest["timeout_seconds"])
        require(0 < timeout <= 60, "Invalid child timeout")
        fixtures = manifest["fixtures"]
        require(0 < len(fixtures) <= 8, "Fixture cap exceeded")
        for fixture in fixtures:
            partition = fixture.get("partition")
            require(
                partition in {"calibration", "reserved"},
                "Explicit fixture partition required",
            )
            result["fixture_partitions"][fixture["id"]] = partition
        result["reserved_evaluation_scope"] = (
            "Partition recorded before execution; independence from prior detector development is unverified"
        )
        require(
            manifest["forward_cap"] == len(fixtures), "Reference forward cap mismatch"
        )
        require(manifest["child_cap"] == 1, "Exactly one child is supported")
        require(
            len({f["id"] for f in fixtures}) == len(fixtures), "Duplicate fixture IDs"
        )
        root = Path(manifest["trial_directory"]).resolve()
        artifacts = manifest["artifacts"]
        for name in ("checkpoint", "predictor", "configuration", "mapping"):
            path = Path(artifacts[name]["path"]).resolve()
            require(root in path.parents, "Artifact outside trial directory")
            require(
                digest_file(path) == artifacts[name]["sha256"],
                "Artifact hash mismatch: " + name,
            )
        require(
            hashlib.sha256(canonical(fixtures)).hexdigest()
            == manifest["fixtures_sha256"],
            "Fixture payload hash mismatch",
        )
        configuration = manifest["configuration"]
        require(
            hashlib.sha256(canonical(configuration)).hexdigest()
            == artifacts["configuration"]["sha256"],
            "Configuration serialization mismatch",
        )
        itemnum, length = int(configuration["itemnum"]), int(configuration["maxlen"])
        require(itemnum >= 1 and length >= 2, "Invalid vocabulary or capacity")
        require(
            manifest["special_tokens"] == {"pad": 0, "mask": itemnum + 1},
            "Special-token mismatch",
        )
        # Construct from immutable selected artifacts, never from live weights.
        # Release this reload before invoking the deployed predictor.
        with torch.random.fork_rng(devices=[]):
            reference = type(view.model)(**configuration).to(device)
        state = torch.load(
            artifacts["checkpoint"]["path"], map_location="cpu", weights_only=True
        )
        reference.load_state_dict(state, strict=True)
        del state
        reference.eval()
        result["reference_device"] = str(device)
        result["precision_scope"] = (
            "Native settings retained; no kernel or tolerance adaptation"
        )
        result["reference_dtype"] = str(next(reference.parameters()).dtype)
        result["configuration_mapping_parity"] = None
        result["configuration_mapping_parity_reason"] = (
            "Artifact hashes verify frozen files, but the runner must separately "
            "observe the configuration and mapping actually loaded by the export"
        )
        references = []
        with torch.inference_mode():
            for fixture in fixtures:
                check_deadline()
                require(
                    fixture["dataset_name"] == manifest["dataset_name"],
                    "Fixture routing mismatch",
                )
                history, candidates = fixture["history"], fixture["candidates"]
                require(
                    all(type(i) is int and 1 <= i <= itemnum for i in history),
                    "Invalid fixture history",
                )
                require(
                    candidates
                    and len(candidates) <= 256
                    and len(set(candidates)) == len(candidates)
                    and all(type(i) is int and 1 <= i <= itemnum for i in candidates),
                    "Invalid fixture candidates",
                )
                retained = history[-(length - 1) :]
                tokens = [0] * (length - len(retained) - 1) + retained + [itemnum + 1]
                sequence = torch.tensor([tokens], dtype=torch.long, device=device)
                readout = max(index for index, token in enumerate(tokens) if token != 0)
                with torch.autocast(device_type=device.type, enabled=False):
                    features = reference.log2feats(sequence)[0, readout]
                    # Preserve the exported single-user feature batch dimension.
                    # Flatten only the returned scores, including K=1.
                    score_tensor = reference.predict_score(
                        features.unsqueeze(0),
                        torch.tensor(candidates, dtype=torch.long, device=device),
                    )
                    require(
                        score_tensor.numel() == len(candidates),
                        "Reference scorer returned an unexpected number of scores",
                    )
                    scores = score_tensor.reshape(-1).detach().cpu().numpy().copy()
                    del score_tensor
                require(
                    scores.shape == (len(candidates),) and np.isfinite(scores).all(),
                    "Invalid reference scores",
                )
                references.append(
                    {
                        "id": fixture["id"],
                        "tokens": tokens,
                        "candidates": candidates,
                        "readout": readout,
                        "features": features.detach().float().cpu().numpy().copy(),
                        "scores": scores,
                    }
                )
        # Discard every reference tensor before the child uses the same GPU.
        del features, sequence
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        del reference
        reference = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
        check_deadline()
        result["query_log"].append(
            {
                "kind": "clean_public_entry_point",
                "fixtures": len(fixtures),
                "timeout_seconds": timeout,
                "adapted_thresholds": False,
            }
        )
        deployed = exported_runner(
            manifest=json.loads(canonical(manifest)),
            reference_records=[
                {"id": record["id"], "candidates": list(record["candidates"])}
                for record in references
            ],
        )
        check_deadline()
        invocation = deployed["invocation"]
        result["invocation"] = invocation
        require(
            deployed.get("execution_settings") == settings,
            "Missing or mismatched observed native execution settings",
        )
        require(
            deployed.get("loaded_dataset_name") == manifest["dataset_name"],
            "Missing or incorrect observed category route",
        )
        result["execution_settings_scope"] = (
            "Conditional on actual-runner observations; child timeout and cleanup remain runner responsibilities"
        )
        require(
            invocation["fresh_interpreter"] is True, "Fresh interpreter not recorded"
        )
        require(
            invocation["actual_public_entry_point"] == "predict",
            "Wrong export entry point",
        )
        require(invocation["exit_code"] == 0, "Exported predictor process failed")
        require(
            invocation["terminated"] is True, "Export child termination unconfirmed"
        )
        require(invocation["child_count"] == 1, "Child cap exceeded")
        require(
            0 <= invocation["elapsed_seconds"] <= timeout, "Export timeout exceeded"
        )
        require(
            deployed["loaded_checkpoint_sha256"] == artifacts["checkpoint"]["sha256"],
            "Deployed checkpoint identity mismatch",
        )
        require(
            deployed["loaded_predictor_sha256"] == artifacts["predictor"]["sha256"],
            "Deployed predictor identity mismatch",
        )
        require(
            deployed["manifest_sha256"] == result["manifest_sha256"],
            "Runner manifest identity mismatch",
        )
        loaded_configuration = deployed.get("loaded_configuration")
        loaded_mapping = deployed.get("loaded_mapping")
        if loaded_configuration is not None:
            require(
                canonical(loaded_configuration) == canonical(configuration),
                "Deployed configuration mismatch",
            )
        if loaded_mapping is not None:
            require(
                loaded_mapping == "identity_1_based",
                "Deployed item mapping mismatch",
            )
        if loaded_configuration is not None and loaded_mapping is not None:
            result["configuration_mapping_parity"] = True
            result["configuration_mapping_parity_reason"] = (
                "Conditional on runner observations of loaded configuration and mapping"
            )
        records = deployed["records"]
        require(len(records) == len(references), "Missing export fixtures")
        maximum_error = 0.0
        maximum_relative_error = 0.0
        for fixture, expected, observed in zip(fixtures, references, records):
            require(observed["id"] == expected["id"], "Export fixture order mismatch")
            require(
                observed["candidates"] == expected["candidates"],
                "Candidate alignment mismatch",
            )
            # These must be observations from the actual export, not tokens
            # supplied to bypass its preprocessing. Absent hooks stay unknown.
            if observed.get("observed_tokens") is not None:
                require(
                    observed["observed_tokens"] == expected["tokens"],
                    "Export token mismatch",
                )
                result["observed_token_fixtures"] += 1
            if observed.get("observed_readout") is not None:
                require(
                    observed["observed_readout"] == expected["readout"],
                    "Export readout mismatch",
                )
                result["observed_readout_fixtures"] += 1
            if observed.get("observed_features") is not None:
                actual_features = np.asarray(observed["observed_features"])
                expected_features = expected["features"]
                require(
                    actual_features.shape == expected_features.shape
                    and actual_features.dtype == np.float32
                    and np.isfinite(actual_features).all(),
                    "Invalid observed exported features",
                )
                feature_error = np.abs(
                    actual_features.astype(np.float64)
                    - expected_features.astype(np.float64)
                )
                result["feature_observations"] += int(feature_error.size)
                result["feature_tolerance_violations"] += int(
                    (
                        feature_error
                        > atol + rtol * np.abs(expected_features.astype(np.float64))
                    ).sum()
                )
                result["maximum_feature_absolute_error"] = max(
                    result["maximum_feature_absolute_error"] or 0.0,
                    float(feature_error.max()),
                )
            actual = observed["scores"]
            require(
                isinstance(actual, np.ndarray) and actual.dtype == np.float32,
                "Binary FP32 NumPy score transport required",
            )
            target = expected["scores"]
            require(
                actual.shape == target.shape and np.isfinite(actual).all(),
                "Invalid exported score shape or values",
            )
            target64 = target.astype(np.float64)
            actual64 = actual.astype(np.float64)
            error = np.abs(actual64 - target64)
            permitted = atol + rtol * np.abs(target64)
            violations = int((error > permitted).sum())
            result["score_tolerance_violations"] += violations
            maximum_error = max(maximum_error, float(error.max()))
            nonzero = target64 != 0
            relative = error / np.maximum(np.abs(target64), 1e-8)
            result["score_observations"] += int(error.size)
            result["score_tolerance_passes"] += int(error.size) - violations
            maximum_relative_error = max(maximum_relative_error, float(relative.max()))
            result["nonzero_errors_at_zero_reference"] += int(
                ((~nonzero) & (error != 0)).sum()
            )
            ids = np.asarray(fixture["candidates"])
            ref_order = np.lexsort((ids, -target64))
            out_order = np.lexsort((ids, -actual64))
            # Canonical order is diagnostic only. The public predictor returns
            # scores, so simulator-native selection needs its own observation.
            native_ids = observed.get("native_ranked_ids")
            if native_ids is not None:
                require(
                    invocation.get("native_tie_rule") == "stable_candidate_order",
                    "Unsupported or unspecified observed native tie semantics",
                )
                require(
                    len(native_ids) == len(ids)
                    and set(native_ids) == set(ids.tolist()),
                    "Native ranked IDs are not the aligned candidate universe",
                )
                native_reference = ids[np.argsort(-target64, kind="stable")].tolist()
                result["native_rank_observations"] += 1
                result["native_rank_disagreements"] += int(
                    native_ids != native_reference
                )
                result["native_rank_reason"] = None
            k = min(10, len(ids))
            epsilon = float(error.max())
            gap = (
                float(target64[ref_order[k - 1]] - target64[ref_order[k]])
                if len(ids) > k
                else None
            )
            boundary = {
                "eligible_universe": "this complete fixture candidate list only",
                "epsilon": epsilon,
                "k": k,
                "boundary_gap": gap,
                "membership_protected": (
                    bool(gap > 2 * epsilon) if gap is not None else None
                ),
                "topk_set_equal": set(ids[ref_order[:k]]) == set(ids[out_order[:k]]),
            }
            disagree = not np.array_equal(ref_order, out_order)
            result["rank_disagreements"] += int(disagree)
            export_positions = np.empty(len(ids), dtype=np.int64)
            export_positions[out_order] = np.arange(len(ids))
            robust_inversions = 0
            for position, left in enumerate(ref_order[:-1]):
                right = ref_order[position + 1 :]
                reversed_pairs = export_positions[left] > export_positions[right]
                separated = (
                    target64[left] - target64[right]
                    > permitted[left] + permitted[right]
                )
                robust_inversions += int((reversed_pairs & separated).sum())
            result["robust_rank_inversions"] += robust_inversions
            gaps = target64[ref_order[:-1]] - target64[ref_order[1:]]
            result["fixture_results"].append(
                {
                    "id": fixture["id"],
                    "candidate_count": len(ids),
                    "score_tolerance_violations": violations,
                    "maximum_absolute_error": float(error.max()),
                    "reference_ranked_ids": ids[ref_order].tolist(),
                    "export_ranked_ids": ids[out_order].tolist(),
                    "reference_adjacent_score_gaps": gaps.tolist(),
                    "rank_disagreement": bool(disagree),
                    "robust_rank_inversions": robust_inversions,
                    "boundary": boundary,
                    "partition": fixture["partition"],
                }
            )
            result["completed_fixtures"] += 1
        result["maximum_absolute_error"] = maximum_error
        result["maximum_relative_error"] = maximum_relative_error
        readout_complete = result["observed_readout_fixtures"] == len(fixtures)
        result["readout_observation"] = True if readout_complete else None
        result["readout_observation_reason"] = (
            "Conditional on actual-path tracing supplied by the runner"
            if readout_complete
            else "Missing readout traces for one or more fixtures"
        )
        result["feature_observation_reason"] = (
            None
            if result["feature_observations"]
            else "No actual-path selected features supplied"
        )
        # Recheck immutable artifacts after execution; runner mutations must
        # occur on isolated copies and may never change selected artifacts.
        for artifact in artifacts.values():
            require(
                digest_file(artifact["path"]) == artifact["sha256"],
                "Artifact changed during export audit",
            )
        require(
            result["score_tolerance_violations"] == 0, "Export score tolerance failure"
        )
        require(
            result["feature_tolerance_violations"] == 0,
            "Export feature tolerance failure",
        )
        check_deadline()
        result["score_parity_observed"] = True
        result["canonical_rank_parity_observed"] = result["rank_disagreements"] == 0
        result["native_rank_parity_observed"] = (
            result["native_rank_disagreements"] == 0
            if result["native_rank_observations"] == len(fixtures)
            else None
        )
        rank_differences = (
            result["rank_disagreements"] > 0 or result["native_rank_disagreements"] > 0
        )
        result["status"] = (
            "CUDA_SCORE_PARITY_WITH_RANK_DIFFERENCES"
            if device.type == "cuda" and rank_differences
            else (
                "CUDA_SCORE_PARITY_OBSERVED"
                if device.type == "cuda"
                else (
                    "CPU_SECONDARY_PARITY_WITH_RANK_DIFFERENCES"
                    if rank_differences
                    else "CPU_SECONDARY_PARITY_OBSERVED"
                )
            )
        )
        result["reason"] = (
            "Bounded clean-score agreement under the declared tolerances. "
            "Inspect native and canonical rank differences separately. "
            "Fault activation and designated simulator selection remain unmeasured."
        )
    except Exception as error:
        result["failures"].append({"type": type(error).__name__, "reason": str(error)})
        result["reason"] = (
            "Supplied export audit failed; this is not an unavailable passing check"
        )
    finally:
        del reference
        for module, training in module_modes:
            module.training = training
        torch.set_float32_matmul_precision(backend_state["matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = backend_state["matmul_tf32"]
        torch.backends.cudnn.allow_tf32 = backend_state["cudnn_tf32"]
        torch.backends.cuda.enable_flash_sdp(backend_state["flash_sdp"])
        torch.backends.cuda.enable_mem_efficient_sdp(
            backend_state["memory_efficient_sdp"]
        )
        torch.backends.cuda.enable_math_sdp(backend_state["math_sdp"])
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        result["elapsed_wall_seconds"] = time.monotonic() - started
    return result


### <<< Self_EvolveRec-BLOCK-END


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the family-specific ModelView boundary
    def __init__(self, view, device):
        self.view = view
        self.model = view
        self.network = view.model
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

            n = emb.size(0)
            ### >>> Self_EvolveRec-BLOCK-START: Undefined collapse is not a healthy zero
            if min(sample_size, n) < 2:
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx)
            x = F.normalize(x, p=2, dim=-1)

            # Sum pairwise dot products without allocating a k x k matrix.
            accumulated = x.to(torch.float64)
            offdiag_sum = (
                accumulated.sum(dim=0).square().sum() - accumulated.square().sum()
            )
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
        ### >>> Self_EvolveRec-BLOCK-START: Mask-aligned training margin counts
        margins_all = []
        inspected_batches = 0
        invalid_targets = 0
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Bound diagnostic sampling and reject malformed batches
        from itertools import islice

        for batch in islice(dataloader, num_batches):
            if not (isinstance(batch, (list, tuple)) and len(batch) == 4):
                raise ValueError("Expected (None, sequence, positive, negative)")
            ### <<< Self_EvolveRec-BLOCK-END

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            ### >>> Self_EvolveRec-BLOCK-START: Validate family-specific diagnostic tensors and scores
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
            ):
                raise ValueError(
                    "Diagnostic sequence, positive and negative must have matching B x L shapes"
                )
            if bool(((seq_t < 0) | (seq_t > self.network.mask_token)).any()):
                raise ValueError("Diagnostic input contains an invalid token ID")
            if bool(((pos_t < 0) | (pos_t > self.network.itemnum)).any()):
                raise ValueError("Diagnostic positive contains a noncatalog target")
            if bool(((neg_t < 0) | (neg_t > self.network.itemnum)).any()):
                raise ValueError("Diagnostic negative contains a noncatalog item")
            seq_np = seq_t.detach().cpu().numpy()
            log_feats = self.view.log2feats(seq_np)
            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.view.predict_score(log_feats, all_items)
            if all_logits.shape != all_items.shape:
                raise ValueError("ModelView.predict_score must return B x L x 2 scores")
            if not bool(torch.isfinite(all_logits).all()):
                raise ValueError("Nonfinite training-margin diagnostic scores")
            margin = all_logits[..., 0] - all_logits[..., 1]
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Score only reserved-mask reconstruction targets
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            selected = pos_t != 0
            valid = (
                selected
                & (seq_t == self.network.mask_token)
                & (pos_t >= 1)
                & (pos_t <= self.network.itemnum)
                & (neg_t >= 1)
                & (neg_t <= self.network.itemnum)
                & (neg_t != pos_t)
            )
            inspected_batches += 1
            invalid_targets += int((selected & ~valid).sum().item())
            if valid.any():
                ### <<< Self_EvolveRec-BLOCK-END
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Match breakdown eligibility to margin eligibility
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
                ### >>> Self_EvolveRec-BLOCK-START: Report empty margin observations explicitly
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

        ### >>> Self_EvolveRec-BLOCK-START: Record training exposure and structural exclusions
        margin_metrics["count"] = sum(int(m.numel()) for m in margins_all)
        margin_metrics["reason"] = (
            None if margins_all else "No valid reserved-mask training targets observed"
        )
        breakdown = {
            "exposure": "training-exposed masked-target margins",
            "inspected_batches": inspected_batches,
            "invalid_target_positions": invalid_targets,
            "category_metadata_available": False,
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

    ### >>> Self_EvolveRec-BLOCK-START: Observe inference tokens, gate sensitivity and vectorize user bootstrap
    def _pack_inference(self, history):
        length = int(self.network.maxlen)
        if length < 1:
            raise ValueError("BERT4Rec maxlen must be positive")
        retained = list(history)[-(length - 1) :] if length > 1 else []
        tokens = retained + [int(self.network.mask_token)]
        return [0] * (length - len(tokens)) + tokens

    def _integrity(self):
        model = self.network
        length, itemnum = int(model.maxlen), int(model.itemnum)
        result = {
            "sampler_provenance": None,
            "sampler_provenance_reason": (
                "The loader supplies corrupted inputs and labels, but no original "
                "event/position provenance. Same-position labels, corruption branches "
                "and loss weights cannot be independently certified."
            ),
            "fresh_process_predictor_parity": None,
            "fresh_process_predictor_parity_reason": (
                "No fresh-process predictor/checkpoint runner is supplied through ModelView. "
                "Live inference checks below do not establish checkpoint, mapping or writer parity."
            ),
            "attention_deletion": None,
            "attention_deletion_reason": (
                "The supplied encode_sequence interface has no diagnostic key/value mask hook."
            ),
        }
        if itemnum < 1 or length < 1:
            result["failure"] = "Invalid item vocabulary or sequence length"
            return result
        other = min(2, itemnum)
        histories = [[], [1], [1] * length, [1, other, 1]]
        raw = torch.zeros((4, length), dtype=torch.long, device=self.device)
        for row, history in enumerate(histories):
            retained = history[-length:]
            if retained:
                raw[row, -len(retained) :] = torch.tensor(retained, device=self.device)
        expected = torch.tensor(
            [self._pack_inference(h) for h in histories],
            dtype=torch.long,
            device=self.device,
        )
        shifted = torch.zeros_like(raw)
        shifted[:, :-1] = raw[:, 1:]
        shifted[:, -1] = model.mask_token
        candidates = torch.arange(1, min(itemnum, 32) + 1, device=self.device)
        # Match the live single-user batch shape to avoid attributing differences
        # between batched kernels to preprocessing or checkpoint inconsistency.
        reference = torch.stack(
            [
                model.encode_sequence(expected[i : i + 1])[0, -1]
                @ model.item_emb(candidates).T
                for i in range(4)
            ]
        )
        observed_inputs = []

        def capture_sequence(module, args):
            tokens = args[0]
            if tokens.ndim == 2 and tuple(tokens.shape) == (1, length):
                observed_inputs.append(tokens.detach().clone())

        handle = model.item_emb.register_forward_pre_hook(capture_sequence)
        try:
            live = torch.stack(
                [model.score_candidates(raw[i : i + 1], candidates) for i in range(4)]
            )
        finally:
            handle.remove()
        ids_match = len(observed_inputs) == 4 and all(
            torch.equal(observed_inputs[i], expected[i : i + 1]) for i in range(4)
        )
        finite = bool(torch.isfinite(reference).all() & torch.isfinite(live).all())
        result.update(
            {
                "reserved_mask_id_valid": int(model.mask_token) == itemnum + 1,
                "fixture_count": 4,
                "fixture_names": ["empty", "short", "full_length", "repeated_id"],
                "inference_ids_match": bool(ids_match),
                "constructed_shift_ids_match": bool(torch.equal(expected, shifted)),
                "observed_sequence_calls": len(observed_inputs),
                "readout_absolute_index": length - 1,
                "finite_scores": finite,
                "live_inference_allclose": finite
                and bool(torch.allclose(reference, live, rtol=1.3e-6, atol=1e-5)),
                "live_inference_max_abs_error": (
                    float((reference - live).abs().max()) if finite else None
                ),
                "rtol": 1.3e-6,
                "atol": 1e-5,
            }
        )
        # A synthetic provenance control rejects shifted labels by position,
        # even when repeated IDs make their item identities indistinguishable.
        selected = np.array([False, True, False])
        source_positions = np.arange(3)
        correct = source_positions.copy()
        shifted_labels = np.array([1, 2, 2])
        result["shifted_provenance_control_rejected"] = bool(
            np.array_equal(correct[selected], source_positions[selected])
            and not np.array_equal(shifted_labels[selected], source_positions[selected])
        )
        result["shifted_provenance_control_scope"] = (
            "Synthetic validator only; not sampler certification"
        )
        if length < 3:
            result["padding_and_bidirectionality_reason"] = (
                "Fixtures require maxlen >= 3"
            )
            return result
        fixture = torch.tensor(
            [self._pack_inference([1])], dtype=torch.long, device=self.device
        )
        original = model.encode_sequence(fixture)
        saved_padding = model.item_emb.weight[0].detach().clone()
        try:
            model.item_emb.weight[0].copy_(
                saved_padding
                + torch.arange(
                    model.hidden_units,
                    device=self.device,
                    dtype=saved_padding.dtype,
                )
                + 1
            )
            perturbed = model.encode_sequence(fixture)
        finally:
            model.item_emb.weight[0].copy_(saved_padding)
        valid = fixture != 0
        result["padding_embedding_invariant"] = bool(
            torch.allclose(original[valid], perturbed[valid], rtol=1.3e-6, atol=1e-5)
        )
        result["padding_max_abs_error"] = float(
            (original[valid] - perturbed[valid]).abs().max()
        )
        # Perturb a visible token right of an interior query. A causal encoder
        # cannot pass this influence check; degeneracy is reported as inconclusive.
        changed = fixture.clone()
        changed[:, -1] = 1
        changed_features = model.encode_sequence(changed)
        right_delta = float((original[:, -2] - changed_features[:, -2]).abs().max())
        result["right_context_max_abs_influence"] = right_delta
        result["right_context_influence_detected"] = right_delta > 1e-5
        result["right_context_scope"] = (
            "Observed right-token influence; zero influence is inconclusive or a structural failure"
        )
        omitted = fixture.clone()
        omitted[:, -1] = 0
        omitted_scores = (
            model.encode_sequence(omitted)[:, -1] @ model.item_emb(candidates).T
        )
        wrong_index = int((fixture[0] != 0).sum()) - 1
        wrong_scores = original[:, wrong_index] @ model.item_emb(candidates).T
        result["omitted_mask_control_rejected"] = not bool(
            torch.allclose(
                omitted_scores,
                original[:, -1] @ model.item_emb(candidates).T,
                rtol=1.3e-6,
                atol=1e-5,
            )
        )
        result["count_minus_one_control_rejected"] = (
            not bool(
                torch.allclose(
                    wrong_scores,
                    original[:, -1] @ model.item_emb(candidates).T,
                    rtol=1.3e-6,
                    atol=1e-5,
                )
            )
            if wrong_index != length - 1
            else None
        )
        return result

    @staticmethod
    def _summary(values):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return {
                "mean": None,
                "count": 0,
                "ci95": None,
                "reason": "No eligible user observations",
            }
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite observations in diagnostic bootstrap")
        rng = np.random.default_rng(42)
        indices = rng.integers(0, values.size, size=(1000, values.size))
        means = values[indices].mean(axis=1)
        return {
            "mean": float(values.mean()),
            "count": int(values.size),
            "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
            "reason": None,
        }

    def _history_sensitivity(self):
        histories = self.user_train.get("History", self.user_train)
        itemnum = int(self.network.itemnum)
        length = int(self.network.maxlen)
        eligible = []
        popularity = Counter()
        for user, events in histories.items():
            events = list(events)
            if any(int(x) < 1 or int(x) > itemnum for x in events):
                raise ValueError("Training history contains a noncatalog item ID")
            events = [int(x) for x in events]
            popularity.update(events)
            # One last training event per user; all later events are absent.
            prefix = events[:-1][-(length - 1) :] if length > 1 else []
            if len(prefix) >= 2:
                eligible.append((user, events, prefix, events[-1]))
        eligible.sort(key=lambda row: str(row[0]))
        if len(eligible) > 200:
            eligible = random.Random(42).sample(eligible, 200)
        length_cuts = (
            np.quantile([len(row[2]) for row in eligible], [0.25, 0.5, 0.75]).tolist()
            if eligible
            else []
        )
        pop_cuts = (
            np.quantile(
                [popularity[i] for i in range(1, itemnum + 1)], [0.25, 0.5, 0.75]
            ).tolist()
            if itemnum
            else []
        )
        rows = {seed: [] for seed in (42, 43, 44)}
        names = [
            "original",
            "compact_oldest",
            "compact_newest",
            "mask_oldest",
            "mask_newest",
        ]
        for user, events, prefix, target in eligible:
            k = min(3, len(prefix) // 2)
            base = self._pack_inference(prefix)
            old_mask, new_mask = base.copy(), base.copy()
            first = length - 1 - len(prefix)
            old_mask[first : first + k] = [self.network.mask_token] * k
            new_mask[length - 1 - k : length - 1] = [self.network.mask_token] * k
            inputs = [
                base,
                self._pack_inference(prefix[k:]),
                self._pack_inference(prefix[:-k]),
                old_mask,
                new_mask,
            ]
            seq = torch.tensor(inputs, dtype=torch.long, device=self.device)
            features = self.network.encode_sequence(seq)[:, -1]
            excluded = set(events) | {0, target, int(self.network.mask_token)}
            available_count = itemnum - sum(1 <= x <= itemnum for x in excluded)
            for seed in (42, 43, 44):
                # Stable user-specific RNG; candidate sets are frozen for all interventions.
                rng = random.Random(f"{seed}:{user}")
                count = min(99, available_count)
                if available_count < 198:
                    pool = [x for x in range(1, itemnum + 1) if x not in excluded]
                    negatives = rng.sample(pool, count)
                else:
                    negatives, used = [], set()
                    while len(negatives) < count:
                        candidate = rng.randint(1, itemnum)
                        if candidate not in excluded and candidate not in used:
                            used.add(candidate)
                            negatives.append(candidate)
                candidates = torch.tensor([target] + negatives, device=self.device)
                scores = (
                    (features @ self.network.item_emb(candidates).T)
                    .float()
                    .cpu()
                    .numpy()
                )
                if not np.isfinite(scores).all():
                    raise ValueError("Nonfinite diagnostic candidate scores")
                measured, tops = {}, {}
                for index, name in enumerate(names):
                    order = np.argsort(-scores[index], kind="stable")
                    rank = int(np.flatnonzero(order == 0)[0]) + 1
                    tops[name] = set(order[:10].tolist())
                    measured[name] = {
                        "target_score": float(scores[index, 0]),
                        "rank": rank,
                        "hr10": float(rank <= 10),
                        "margin": (
                            float(scores[index, 0] - scores[index, 1:].max())
                            if count
                            else None
                        ),
                    }
                for name in names[1:]:
                    measured[name]["top10_overlap"] = len(
                        tops[name] & tops["original"]
                    ) / min(len(tops[name]), len(tops["original"]))
                    for metric in ("target_score", "rank", "hr10", "margin"):
                        value = measured[name][metric]
                        measured[name]["delta_" + metric] = (
                            value - measured["original"][metric]
                            if value is not None
                            else None
                        )
                for mode in ("compact", "mask"):
                    measured[mode + "_recent_minus_older"] = {
                        metric: (
                            (
                                measured[mode + "_newest"][metric]
                                - measured[mode + "_oldest"][metric]
                            )
                            if measured[mode + "_newest"][metric] is not None
                            else None
                        )
                        for metric in (
                            "target_score",
                            "rank",
                            "hr10",
                            "margin",
                            "top10_overlap",
                        )
                    }
                rows[seed].append(
                    {
                        "user": str(user),
                        "repeat": target in prefix,
                        "length_quartile": int(
                            np.searchsorted(length_cuts, len(prefix), side="right")
                        ),
                        "popularity_quartile": int(
                            np.searchsorted(pop_cuts, popularity[target], side="right")
                        ),
                        "candidate_count": count + 1,
                        "measurements": measured,
                    }
                )

        # Keep metric keys and explicit null summaries in empty strata.
        metric_keys = {
            name + "." + metric
            for records in rows.values()
            for record in records
            for name, measurements in record["measurements"].items()
            for metric in measurements
        }
        if not metric_keys:
            for name in names:
                metrics = ["target_score", "rank", "hr10", "margin"]
                if name != "original":
                    metrics += [
                        "top10_overlap",
                        "delta_target_score",
                        "delta_rank",
                        "delta_hr10",
                        "delta_margin",
                    ]
                metric_keys.update(name + "." + metric for metric in metrics)
            for mode in ("compact", "mask"):
                metric_keys.update(
                    mode + "_recent_minus_older." + metric
                    for metric in (
                        "target_score",
                        "rank",
                        "hr10",
                        "margin",
                        "top10_overlap",
                    )
                )

        def aggregate(records):
            output = {key: [] for key in sorted(metric_keys)}
            for record in records:
                for name, measurements in record["measurements"].items():
                    for metric, value in measurements.items():
                        if value is not None:
                            output[name + "." + metric].append(value)
            return {key: self._summary(values) for key, values in output.items()}

        draws = {}
        for seed, records in rows.items():
            draws[str(seed)] = {
                "users": len(records),
                "candidate_counts": dict(
                    Counter(r["candidate_count"] for r in records)
                ),
                "summary": aggregate(records),
                "repeat_explore": {
                    label: {
                        "users": sum(r["repeat"] == repeat for r in records),
                        "summary": aggregate(
                            [r for r in records if r["repeat"] == repeat]
                        ),
                    }
                    for label, repeat in (("repeat", True), ("explore", False))
                },
                "strata": {
                    field: {
                        str(q): {
                            "users": sum(r[field] == q for r in records),
                            "summary": aggregate([r for r in records if r[field] == q]),
                        }
                        for q in range(4)
                    }
                    for field in ("length_quartile", "popularity_quartile")
                },
            }
        return {
            "exposure": "training-exposed recovery; not held-out generalization",
            "category": getattr(self.view, "dataset_name", None),
            "eligible_selected_users": len(eligible),
            "reason": (
                None
                if eligible
                else "No training prefix with at least two retained items"
            ),
            "selection_seed": 42,
            "bootstrap_seed": 42,
            "bootstrap_resamples": 1000,
            "length_quartile_cutpoints": length_cuts,
            "item_training_popularity_quartile_cutpoints": pop_cuts,
            "candidate_policy": (
                "Positive exactly once; up to 99 unique catalog negatives excluding all "
                "permitted training-history items, positive, padding and mask. Frozen across interventions."
            ),
            "draws": draws,
            "category_metadata_breakdowns": {},
            "equal_category_macro": None,
            "equal_category_macro_reason": "A single ModelView cannot certify eligibility of all four categories",
        }

    def run_full_diagnosis(self, train_loader):
        modes = [(module, module.training) for module in self.network.modules()]
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            self.network.eval()
            with torch.inference_mode():
                integrity = self._integrity()
                records = getattr(train_loader, "consumed_records", None)
                if records is None:
                    records = getattr(self.view, "consumed_records", None)
                if records is None:
                    records = getattr(self.network, "consumed_records", None)
                consumed_audit = audit_consumed_records(
                    records, self.user_train, int(self.network.itemnum)
                )
                # The supplied loader is the diagnostic loader from the public
                # contract. Consumed-record replay is a separate provenance audit.
                # Without captures, measure disposable diagnostic batches while
                # leaving consumed-pipeline provenance explicitly unavailable.
                # DEBUG: Keep consumed-record replay separate from the supplied
                # disposable diagnostic loader. Replacing that loader with at
                # most four captures silently reduces the ten-batch margin
                # protocol and changes the population being measured.
                # audit_consumed_records above already audits those captures.
                integrity["sampler_provenance"] = consumed_audit
                integrity["sampler_provenance_reason"] = (
                    None
                    if consumed_audit["available"]
                    else consumed_audit.get("reason", "No captured records")
                )
                collapse = self.analyze_embeddings_collapse_cosine(
                    sample_size=2048, seed=42
                )
                margins, breakdown = (
                    self.analyze_pos_neg_margin_with_category_breakdown(
                        train_loader, num_batches=10, eps=0.5, topk_categories=20
                    )
                )
                required_checks = (
                    "reserved_mask_id_valid",
                    "inference_ids_match",
                    "finite_scores",
                    "live_inference_allclose",
                    "padding_embedding_invariant",
                    "omitted_mask_control_rejected",
                    "count_minus_one_control_rejected",
                )
                failed_checks = [
                    key for key in required_checks if integrity.get(key) is False
                ]
                unavailable_checks = [
                    key for key in required_checks if integrity.get(key) is None
                ]
                if integrity.get("failure"):
                    failed_checks.append("failure")
                # DEBUG: Nonempty batches do not establish target coverage.
                # Zero valid masked targets must remain explicitly unavailable.
                if breakdown["invalid_target_positions"]:
                    failed_checks.append("masked_target_alignment")
                elif margins["count"] == 0:
                    unavailable_checks.append("masked_target_alignment")
                # DEBUG: Report partial loader coverage rather than treating
                # fewer than ten inspected batches as complete diagnostics.
                integrity["masked_target_expected_batches"] = 10
                integrity["masked_target_inspected_batches"] = breakdown[
                    "inspected_batches"
                ]
                integrity["masked_target_valid_positions"] = margins["count"]
                if breakdown["inspected_batches"] < 10:
                    unavailable_checks.append("masked_target_batch_coverage")
                if consumed_audit.get("failures"):
                    failed_checks.append("consumed_pipeline_audit")
                if not consumed_audit.get("available"):
                    unavailable_checks.append("consumed_pipeline_audit")
                elif consumed_audit.get("assertion_counts", {}).get(
                    "actual_loss_wiring:unavailable", 0
                ):
                    unavailable_checks.append("actual_loss_wiring")
                integrity["failed_required_checks"] = failed_checks
                integrity["unavailable_required_checks"] = unavailable_checks
                integrity["structural_status"] = (
                    "FAILED"
                    if failed_checks
                    else (
                        "INCOMPLETE" if unavailable_checks else "OBSERVED_CHECKS_PASSED"
                    )
                )
                integrity["certification_scope"] = (
                    "Observed live fixtures and explicitly captured consumed-batch consistency only. "
                    "Missing records or same-forward loss payloads remain unavailable. "
                    "Fresh-process predictor parity, complete gradient wiring and effective "
                    "attention-mask inspection remain uncertified."
                )
                sensitivity = (
                    {
                        "exposure": "training-exposed recovery; not held-out generalization",
                        "result": None,
                        "reason": "Skipped because required structural checks failed",
                        "failed_checks": failed_checks,
                        "category_metadata_breakdowns": {},
                    }
                    if failed_checks
                    else self._history_sensitivity()
                )
                export_audit = audit_exported_predictor(
                    self.view,
                    getattr(self.view, "exported_predictor_runner", None),
                    getattr(self.view, "frozen_training_manifest", None),
                )
                metrics = {
                    "exported_predictor_audit": export_audit,
                    "bert4rec_integrity": integrity,
                    "embedding_collapse_score": collapse,
                    "embedding_collapse_sample_count": min(
                        2048, int(self.network.itemnum)
                    ),
                    "embedding_collapse_reason": (
                        None
                        if collapse is not None
                        else "Fewer than two item embeddings"
                    ),
                    "pos_neg_margin_mean": margins["mean"],
                    "pos_neg_margin_p10": margins["p10"],
                    "pos_neg_neg_beats_pos_rate": margins["neg_beats_pos_rate"],
                    "pos_neg_small_margin_rate": margins["small_margin_rate"],
                    "pos_neg_margin_sample_count": margins["count"],
                    "pos_neg_margin_reason": margins["reason"],
                    "pos_neg_margin_category_breakdown": breakdown,
                    "training_history_sensitivity": sensitivity,
                    "consumed_pipeline_audit": consumed_audit,
                }
                definitions = {
                    "exported_predictor_audit": (
                        "Optional host-runner audit against a strictly reloaded selected category checkpoint. "
                        "Reference inference uses the actual ModelView device, with at most eight frozen fixtures and 256 distinct "
                        "candidates per fixture. Canonical JSON uses sorted keys, UTF-8, compact separators "
                        "and no NaN. Artifact and fixture hashes identify the frozen manifest, not its correctness. "
                        "Binary FP32 scores use rtol=1.3e-6 and atol=1e-5 without adaptive widening. "
                        "Reference scores are withheld from the runner. Observed export settings must match "
                        "native reference settings; CPU evidence cannot establish CUDA parity. The child timeout "
                        "is at most 60 seconds and the aggregate wall cap is 300 seconds. Runner termination "
                        "remains the runner's responsibility; the adapter checks elapsed time after return. "
                        "Relative errors divide by max(abs(reference),1e-8). Optional feature and readout "
                        "observations have separate coverage. Native stable-candidate-order rankings require "
                        "explicit observations and do not replace diagnostic canonical ID-based rankings. "
                        "Hand-calculated tied selections test chunk truncation independently of the network. "
                        "Boundary protection requires complete aligned fixture scores and gap>2*epsilon; "
                        "it certifies neither full-catalog membership nor within-top-k ordering. "
                        "Fixture partitions are recorded, but independence from previous development is unverified. "
                        "Robust inversions reverse a reference pair separated by more than the sum "
                        "of its two permitted errors. Per-fixture ranked IDs and adjacent reference gaps are retained. "
                        "Loaded configuration/mapping comparisons require explicit runner observations. "
                        "Auxiliary ranks use descending score then ascending canonical item ID; exact rank "
                        "disagreements remain visible when score tolerances pass. Missing runner or manifest "
                        "is unavailable; supplied-path execution, artifact or comparison errors are failures. "
                        "Runner invocation and loaded-artifact observations are supplied evidence, not independently "
                        "authenticated process provenance. Internal fault calibration, parameter/buffer parity, "
                        "deployed readout observation and simulator full-catalog selection remain explicitly "
                        "uncertified. Wall time is recorded separately from unavailable GPU-time accounting. "
                        "This audit neither fits another model nor measures held-out recommendation quality."
                    ),
                    "consumed_pipeline_audit": (
                        "At most four explicitly supplied production records. Independently reconstructs crops, "
                        "source events, reserved-mask corruption, labels and seeded negative draws from public "
                        "training histories. Same-forward detached logits check pooled CE/BPR scalar reduction "
                        "without a second forward; tolerances are recorded per dtype. No gradient certification. "
                        "Calibration uses only the first retained record, verified changed-label and restored-mask "
                        "faults, plus clean and consistent row-permutation controls. Coverage is detected divided "
                        "by verified non-equivalent corruptions; zero denominators are null. Repeated-ID shifts "
                        "are equivalent for the label-only operator. These bounded controls do not estimate real "
                        "fault prevalence. Captures need not belong to the selected checkpoint. Missing transport, "
                        "deployment parity and held-out gains remain unavailable. No production iterator is "
                        "advanced when captured replay is supplied; worker state and exact restart are not certified."
                    ),
                    "bert4rec_integrity": (
                        "Independent appended-mask fixtures compare exact IDs and live candidate scores "
                        "at rtol=1.3e-6, atol=1e-5. Includes padding perturbation and negative controls. "
                        "False checks require investigation; unavailable checks are not passes. "
                        "Synthetic provenance controls do not certify the actual sampler."
                    ),
                    "embedding_collapse_score": (
                        "(Mean off-diagonal pairwise cosine + 1)/2 over up to 2048 seed-42 sampled "
                        "catalog embeddings, excluding padding and mask. Measures angular alignment, "
                        "not recommendation diversity or representation quality by itself."
                    ),
                    "embedding_collapse_sample_count": "Number of catalog embeddings sampled",
                    "embedding_collapse_reason": "Reason the collapse statistic is undefined, or null",
                    "pos_neg_margin_mean": "Training-exposed mean positive-minus-negative logit at valid masked targets",
                    "pos_neg_margin_p10": "Training-exposed nearest-rank tenth percentile of masked-target margins",
                    "pos_neg_neg_beats_pos_rate": "Fraction of valid masked targets with margin < 0",
                    "pos_neg_small_margin_rate": "Fraction of valid masked targets with margin < 0.5",
                    "pos_neg_margin_sample_count": "Number of valid supervised reserved-mask positions observed",
                    "pos_neg_margin_reason": "Reason no valid training margins were observed, or null",
                    "pos_neg_margin_category_breakdown": (
                        "Observed batch, invalid-target and low-margin counts. Category lists remain empty "
                        "because item category metadata is unavailable. The supplied sampler always masks selected targets."
                    ),
                    "training_history_sensitivity": (
                        "One last training target/user with strictly earlier prefix truncated to L-1; "
                        "earlier repeats are retained. Up to 200 users selected with seed 42. "
                        "Candidate draws 42/43/44 are reported separately. Rank is one-based with stable "
                        "candidate-order ties; HR10 is rank<=10; margin is target minus hardest negative. "
                        "Each delta is intervention minus original. Recent-minus-older is newest-block "
                        "intervention minus oldest-block intervention; k=min(3,floor(n/2)). "
                        "Overlap is intersection/minimum returned top-list length. Compact deletion shifts "
                        "survivor positions; mask replacement retains positions and introduces learned mask content. "
                        "Attention deletion is unavailable without a supported hook. Means have 1000-resample "
                        "seed-42 user-bootstrap percentile 95% intervals, conditional on the selected users and "
                        "candidate draw. These intervals exclude checkpoint-selection uncertainty. "
                        "Popularity quartiles use catalog training counts; length quartiles use selected retained "
                        "prefix lengths; tied cutpoints may produce empty strata. Results describe training-exposed "
                        "sensitivity, not causal preferences, memorization alone or held-out generalization. "
                        "Host validation is unchanged; no four-category macro is fabricated."
                    ),
                }
                return {"metrics": metrics, "metric_definitions": definitions}
        finally:
            for module, mode in modes:
                module.training = mode
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

    ### <<< Self_EvolveRec-BLOCK-END
