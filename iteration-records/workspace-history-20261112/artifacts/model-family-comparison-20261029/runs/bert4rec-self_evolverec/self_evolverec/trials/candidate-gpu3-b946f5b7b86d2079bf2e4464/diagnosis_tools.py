import torch
import numpy as np
import torch.nn.functional as F
import json
import random
from collections import Counter

### >>> Self_EvolveRec-BLOCK-START: Bounded canonical-score catalog audit with exact sampling reference
import random
import time
from collections import Counter

import numpy as np
import torch
from scipy.stats import hypergeom


def run_catalog_audit(probe, *, serving_parity=None):
    """Frozen training-prefix measurement; never selects or updates a model."""
    net, device = probe.net, probe.device
    modes = [(module, module.training) for module in net.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    settings = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    synchronize()
    started = time.perf_counter()
    result = {
        "available": False,
        "complete": False,
        "status": "preparing",
        "dataset_name": probe.category,
        "records": [],
        "selected_examples": [],
        "counts": Counter(
            {
                key: 0
                for key in (
                    "scanned_users",
                    "invalid_users",
                    "eligible_endpoints",
                    "repeated_target_endpoints_excluded",
                    "no_endpoint_users",
                    "insufficient_pool_users",
                    "selected_users",
                    "completed_users",
                    "scored_chunks",
                    "raw_nonfinite_scores",
                    "invalid_queries",
                )
            }
        ),
        "summary": {},
        "reason": None,
        "selection_complete": False,
        "protocol": {
            "maximum_users": 32,
            "minimum_prefix": 4,
            "seconds_cap": 30.0,
            "user_seed": 42,
            "endpoint_seed": 1042,
            "candidate_seed": 42,
            "selection": "Lexical user IDs, seeded permutation, first 32 eligible users; uniform eligible endpoint per user",
            "endpoint": "Real target absent from complete prefix; no suffix exclusions",
            "universe": "All real catalog IDs minus complete prefix, including target",
            "sample": "99 unique competitors uniformly without replacement; per-selected-example SeedSequence([42,index])",
            "order": "Descending computed score, then ascending item ID; no epsilon ties",
            "chunk_size": 4096,
            "user_batch_size": 1,
            "score_cache": "One float32 CPU score per real catalog ID per query; no second pass",
            "frequency_bins": ["0", "1-4", "5-16", "17-64", ">=65"],
            "visible_history_bins": ["4-7", "8-15", ">=16"],
            "numerics": "Float32 parameters/features/scores; autocast and TF32 disabled; deterministic algorithms required",
            "torch_version": str(torch.__version__),
            "numpy_version": np.__version__,
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "hardware": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
            "training_objective": "Not changed or inferred by this diagnostic; consult the trial training manifest",
            "aggregation": "Equal category weights only after all four declared protocols complete",
        },
        "metric_definitions": {
            "full_rank": "One plus eligible competitors preceding target under exhaustive computed-score ordering",
            "sampled_rank": "One plus sampled competitors preceding target using the same cached canonical scores",
            "expected_sampled_hit10": "Hypergeometric CDF(9; competitor_count, full_rank-1, 99), conditional on the frozen ordering",
            "expected_sampled_rank": "1 + 99*(full_rank-1)/competitor_count",
            "margin": "Target score minus tenth-highest competitor score; zero requires rank-based tie resolution",
            "boundary_gap": "Tenth minus eleventh competitor score; excludes the target",
            "summary": "Equal-user descriptive means; partial coverage is explicitly marked. No bootstrap or significance claim",
            "strata": "Training-corpus occurrence frequency and encoder-visible prefix length; descriptive, not causal exposure",
            "elapsed_seconds": "CUDA-synchronized wall seconds including preparation and summaries; all GPU work consumes search budget",
        },
        "limitations": "Training prefixes can favor memorization. Unseen competitors are unlabeled. This universe need not match host evaluation. Mathematical sampling references are not model gains. No training-seed or candidate-sampling uncertainty interval is estimated.",
    }

    def boundary():
        synchronize()
        if time.perf_counter() - started >= result["protocol"]["seconds_cap"]:
            raise TimeoutError("Catalog audit cap reached at a synchronized boundary")

    def summarize(rows):
        keys = (
            "full_rank",
            "sampled_rank",
            "full_hit10",
            "sampled_hit10",
            "expected_sampled_hit10",
            "expected_sampled_rank",
            "sampled_only_hit",
            "full_only_hit",
            "observed_minus_expected_hit",
            "full_margin",
            "sampled_margin",
            "full_boundary_gap",
            "sampled_boundary_gap",
        )
        return {
            "users": len(rows),
            "means": {
                key: float(np.mean([row[key] for row in rows])) if rows else None
                for key in keys
            },
        }

    try:
        if not probe._contract.get("passed", False):
            raise ValueError("Sampler contract must pass")
        if not isinstance(serving_parity, dict) or not all(
            serving_parity.get(key, False)
            for key in (
                "available",
                "passed",
                "checkpoint_serialization_verified",
                "coverage_complete",
            )
        ):
            raise ValueError("Complete serialized-serving parity must pass")
        if probe.category not in {
            "Movies_and_TV",
            "Industrial_and_Scientific",
            "Electronics",
            "CDs_and_Vinyl",
        }:
            raise ValueError("Actual routed category is required")
        if net.maxlen < 5 or net.item_emb.weight.dtype != torch.float32:
            raise ValueError("Require float32 BERT4Rec and at least four visible slots")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        net.eval()
        histories = probe._histories()
        frequencies = Counter()
        for history in histories.values():
            boundary()
            frequencies.update(
                int(item) for item in history if 1 <= int(item) <= net.itemnum
            )
        users = sorted(histories, key=lambda user: str(user))
        permutation = np.random.default_rng(42).permutation(len(users))
        endpoint_rng = random.Random(1042)
        selected = []
        for index in permutation:
            boundary()
            user = users[int(index)]
            result["counts"]["scanned_users"] += 1
            try:
                history = [int(item) for item in histories[user]]
            except (TypeError, ValueError):
                result["counts"]["invalid_users"] += 1
                continue
            if any(item < 1 or item > net.itemnum for item in history):
                result["counts"]["invalid_users"] += 1
                continue
            seen, endpoints = set(), []
            for endpoint, target in enumerate(history):
                if endpoint >= 4:
                    if target in seen:
                        result["counts"]["repeated_target_endpoints_excluded"] += 1
                    else:
                        endpoints.append(endpoint)
                seen.add(target)
            result["counts"]["eligible_endpoints"] += len(endpoints)
            if not endpoints:
                result["counts"]["no_endpoint_users"] += 1
                continue
            endpoint = endpoint_rng.choice(endpoints)
            prefix, target = history[:endpoint], history[endpoint]
            pool = np.setdiff1d(
                np.arange(1, net.itemnum + 1), list(set(prefix) | {target})
            )
            if len(pool) < 99:
                result["counts"]["insufficient_pool_users"] += 1
                continue
            candidate_rng = np.random.default_rng(
                np.random.SeedSequence([42, len(selected)])
            )
            sampled = candidate_rng.choice(pool, 99, replace=False)
            selected.append((str(user), endpoint, prefix, target, pool, sampled))
            result["selected_examples"].append(
                {
                    "user": str(user),
                    "endpoint": endpoint,
                    "target": target,
                    "sampled_competitors": sampled.tolist(),
                    "competitor_count": len(pool),
                }
            )
            if len(selected) == 32:
                break
        result["counts"]["selected_users"] = len(selected)
        result["counts"]["requested_users"] = 32
        result["counts"]["training_users"] = len(users)
        result["selection_complete"] = True
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, enabled=False
        ):
            for user, endpoint, prefix, target, pool, sampled in selected:
                boundary()
                try:
                    _, query, _ = probe._serving_tensors(prefix)
                    if int(query[0, -1]) != net.mask_token:
                        raise ValueError("Final nonpadding query position must be MASK")
                    features = probe.view.log2feats(query.cpu().numpy())
                    if features.shape != (1, net.maxlen, net.hidden_units):
                        raise ValueError("Unexpected ModelView feature shape")
                    feature = features[:, -1:, :]
                    if feature.dtype != torch.float32 or not bool(
                        torch.isfinite(feature).all()
                    ):
                        raise ValueError("Nonfinite or non-float32 query feature")
                    # Cache every raw real-item score before exclusions. This gives
                    # sampled and exhaustive ranks identical canonical numerics.
                    cached = np.empty(net.itemnum, dtype=np.float32)
                    for start in range(1, net.itemnum + 1, 4096):
                        boundary()
                        ids = torch.arange(
                            start, min(start + 4096, net.itemnum + 1), device=device
                        )
                        scores = probe.view.predict_score(
                            feature, ids.reshape(1, 1, -1)
                        )
                        if (
                            scores.shape != (1, 1, len(ids))
                            or scores.dtype != torch.float32
                        ):
                            raise ValueError(
                                "Expected float32 B x L x C scoring contract"
                            )
                        result["counts"]["scored_chunks"] += 1
                        result["counts"]["raw_nonfinite_scores"] += int(
                            (~torch.isfinite(scores)).sum()
                        )
                        if not bool(torch.isfinite(scores).all()):
                            raise ValueError("Raw catalog contains nonfinite scores")
                        cached[start - 1 : start - 1 + len(ids)] = (
                            scores.reshape(-1).cpu().numpy()
                        )
                        boundary()
                    target_score = float(cached[target - 1])

                    def measure(ids):
                        boundary()
                        values = cached[ids - 1]
                        ahead = (values > target_score) | (
                            (values == target_score) & (ids < target)
                        )
                        rank = 1 + int(ahead.sum())
                        # Only boundary score values are needed here. Exhaustive
                        # rank already resolves ties by ascending item identity.
                        # Partition avoids sorting the entire catalog per query.
                        top = np.partition(values, len(values) - 11)[-11:]
                        top.sort()
                        margin = target_score - float(top[1])
                        gap = float(top[1]) - float(top[0])
                        boundary()
                        return rank, margin, gap

                    full_rank, full_margin, full_gap = measure(pool)
                    sampled_rank, sampled_margin, sampled_gap = measure(sampled)
                    n = len(pool) + 1
                    if not (n >= 100 and 1 <= sampled_rank <= full_rank <= n):
                        raise ValueError("Nested-rank invariant failed")
                    full_hit, sampled_hit = int(full_rank <= 10), int(
                        sampled_rank <= 10
                    )
                    if full_hit and not sampled_hit:
                        raise ValueError("Full hit must imply sampled hit")
                    expected = float(hypergeom.cdf(9, n - 1, full_rank - 1, 99))
                    if not np.isfinite(expected) or not 0.0 <= expected <= 1.0:
                        raise ValueError("Invalid hypergeometric reference")
                    frequency, visible = frequencies[target], min(
                        len(prefix), net.maxlen - 1
                    )
                    row = {
                        "user": user,
                        "endpoint": endpoint,
                        "target": target,
                        "N": n,
                        "full_rank": full_rank,
                        "sampled_rank": sampled_rank,
                        "full_hit10": full_hit,
                        "sampled_hit10": sampled_hit,
                        "expected_sampled_hit10": expected,
                        "expected_sampled_rank": 1.0 + 99.0 * (full_rank - 1) / (n - 1),
                        "sampled_only_hit": int(sampled_hit and not full_hit),
                        "full_only_hit": int(full_hit and not sampled_hit),
                        "observed_minus_expected_hit": sampled_hit - expected,
                        "full_margin": full_margin,
                        "sampled_margin": sampled_margin,
                        "full_boundary_gap": full_gap,
                        "sampled_boundary_gap": sampled_gap,
                        "target_frequency": frequency,
                        "visible_history": visible,
                        "frequency_bin": (
                            "0"
                            if frequency == 0
                            else (
                                "1-4"
                                if frequency <= 4
                                else (
                                    "5-16"
                                    if frequency <= 16
                                    else "17-64" if frequency <= 64 else ">=65"
                                )
                            )
                        ),
                        "history_bin": (
                            "4-7"
                            if visible <= 7
                            else "8-15" if visible <= 15 else ">=16"
                        ),
                    }
                    boundary()
                    result["records"].append(row)
                except TimeoutError:
                    raise
                except Exception as exc:
                    result["counts"]["invalid_queries"] += 1
                    result.setdefault("query_errors", []).append(
                        {"user": user, "reason": f"{type(exc).__name__}: {exc}"}
                    )
        result["complete"] = len(selected) == 32 and len(result["records"]) == 32
        result["status"] = "complete" if result["complete"] else "incomplete_coverage"
    except TimeoutError as exc:
        result["status"], result["reason"] = "timed_out", str(exc)
    except Exception as exc:
        result["status"], result["reason"] = (
            "unavailable",
            f"{type(exc).__name__}: {exc}",
        )
    finally:
        try:
            result["available"] = bool(result["records"])
            result["counts"]["completed_users"] = len(result["records"])
            result["summary"] = summarize(result["records"])
            result["strata"] = {
                field: {
                    label: summarize(
                        [row for row in result["records"] if row[field] == label]
                    )
                    for label in labels
                }
                for field, labels in (
                    ("frequency_bin", result["protocol"]["frequency_bins"]),
                    ("history_bin", result["protocol"]["visible_history_bins"]),
                )
            }
        finally:
            for module, mode in modes:
                module.training = mode
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
            torch.backends.cuda.matmul.allow_tf32 = settings[0]
            torch.backends.cudnn.allow_tf32 = settings[1]
            torch.use_deterministic_algorithms(settings[2], warn_only=settings[3])
            synchronize()
            result["elapsed_seconds"] = time.perf_counter() - started
            result["cap_overshoot_seconds"] = max(0.0, result["elapsed_seconds"] - 30.0)
            result["counts"] = dict(result["counts"])
    if not result["complete"]:
        print(
            "BERT4Rec catalog audit: "
            f"status={result['status']} "
            f"completed={result['counts']['completed_users']}/32 "
            f"reason={result['reason'] or 'insufficient coverage; inspect query_errors and exclusion counts'}",
            flush=True,
        )
    return result


### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Evidence-linked interpretation including canonical catalog boundaries
def diagnosis_interpreter_prompt(raw_diagnosis):
    return f"""
Analyze these measured BERT4Rec diagnostics and their definitions:
{json.dumps(raw_diagnosis, indent=2, allow_nan=False)}
Return JSON with status, core_findings, key_implications, and evidence.
Use at most three findings and three implications. Cite metric names, values,
counts, availability flags, and uncertainty intervals for each substantive claim.
Distinguish serving defects, measured history sensitivity, missing information,
and hypotheses requiring independent validation. If serving parity or sampler
checks fail or are unavailable, withhold substantive sensitivity conclusions.
Null values mean unavailable, never zero or successful checks. Missing category
coverage prevents a four-category macro conclusion. These training-prefix
candidate results are not host validation HR@10. Stability, sensitivity, and
embedding alignment do not establish recommendation quality or validation gains.
A mapped embedding cosine of 0.5 means approximately zero mean cosine, not proven
healthy representation capacity. Do not infer intent, simulator traits, ratings,
review content, item categories, prices, stores, or ownership. Unknown identity
and abstention are not negative labels. Intervals cover sampled-user uncertainty
only, not training seeds or candidate sampling. Repeated validation selection is
optimistic. No recommendation-score fusion or additional predictors are allowed.
Use INSUFFICIENT_EVIDENCE when the available measurements do not support a status.
For residual_patching, report its cap, elapsed time, completed interventions,
partition denominators, all attempted sites, missing donors, and control results.
Separate exploration from held-out-user confirmation. Interpret effects only
conditional on the specified corruption and patch; natural activation donors can
produce out-of-distribution hybrid computations. Frequency and recency matching
provide no semantic compatibility evidence. Never describe a local raw p-value
or the conservative Bonferroni bound as a four-category Holm-adjusted result.
No Holm discovery is established without the complete cross-category family.
Identity, structural-zero, and MASK-restoration controls test implementation;
nonzero unrelated or random control effects require discussion, not concealment.
Missing trajectories leave optimization/convergence unassessed. This frozen-model
pilot neither demonstrates a repair nor establishes validation improvements.
For catalog_boundary_audit, distinguish observed sampled/full-catalog outcomes
from conditional hypergeometric sampling references. Report completion status,
selected/completed users, exclusions, nonfinite counts, cap, elapsed time and
overshoot. Partial coverage is descriptive and cannot represent the declared
32-user panel or a four-category macro. A full hit must imply a sampled hit;
sampled rank cannot exceed exhaustive rank when canonical scores are shared.
Competitor boundary gaps exclude the target. A zero target margin requires the
recorded score-then-ID rank to resolve ties. Frequency strata describe permitted
training-corpus counts, not causal exposure, shortcut learning or user intent.
One sampled set per query cannot establish sampling calibration. No validation
gain, objective advantage or model-comparison reversal follows from this audit.
The diagnostic does not establish that training used a fixed 0.5 mixture;
require the actual training manifest before making that claim.
"""


### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Auditable budgeted residual patching with verified hooks and controls
@torch.no_grad()
def _residual_patching_pilot(probe, examples, seconds=30.0):
    import time
    from scipy.stats import binomtest

    net = probe.net
    started = time.perf_counter()
    rng = np.random.default_rng(1842)
    result = {
        "available": False,
        "reason": None,
        "records": [],
        "summary": {},
        "protocol": {
            "seed": 1842,
            "seconds_cap": float(seconds),
            "maximum_users": 64,
            "minimum_users_per_partition": 8,
            "partition": "Seeded permutation; first half exploration, remainder confirmation.",
            "rules": ["matched_donor_A", "matched_donor_B"],
            "replacement_rules": {
                "matched_donor_A": "Uniform distinct item from the matched donor pool.",
                "matched_donor_B": "Closest training frequency to the original among remaining matched items; ascending item ID breaks ties.",
            },
            "frequency_edges": [1, 4, 16, 64, 256],
            "recency_bands": 4,
            "primary": "target minus strongest clean negative, identity fixed before intervention",
            "directions": ["restore", "disrupt"],
            "sites": ["input_residual", "enter_second_attention"],
            "selection": "Largest absolute exploration mean per rule and direction; lock sign and site.",
            "test": "One-sided exact sign test of locked-direction confirmation effects; zeros excluded and reported.",
            "family_size": 16,
            "multiplicity": "Four categories x two rules x two directions. Global Holm requires all category p-values; no Holm rejection is declared here.",
            "magnitude_edges": None,
            "magnitude_matching": "Match intervention norm directly; descriptive quartile boundaries fitted on exploration only and frozen before confirmation.",
            "random_control": "One seeded random history site in the same recency band and layer, with an isotropic residual change of exactly the intervention norm.",
            "tolerances": {"atol": 1e-5, "rtol": 1e-4},
        },
        "training_trajectories": {
            "available": False,
            "reason": "The supplied ModelView exposes no loss or validation trajectory; convergence is unassessed.",
        },
        "confirmation": [],
        "counts": {},
        "timed_out": False,
    }
    hooks = []
    locked = {}

    def checkpoint(synchronize=False):
        if synchronize and probe.device.type == "cuda":
            torch.cuda.synchronize(probe.device)
        if time.perf_counter() - started >= seconds:
            raise TimeoutError("Diagnostic cap reached at a checked boundary.")

    try:
        layers = net.encoder.layers
        if len(layers) != 2 or not all(layer.norm_first for layer in layers):
            raise ValueError(
                "Pilot supports the supplied two-layer pre-norm BERT4Rec only."
            )
        modules = {
            "input_residual": net.dropout,
            "enter_second_attention": layers[0],
            "final_history_output": layers[-1],
            "final_mask_feature": net.norm,
        }
        result["hook_locations"] = {
            "input_residual": "net.dropout output; before first attention",
            "enter_second_attention": "net.encoder.layers[0] output; before second attention",
            "final_history_output": "net.encoder.layers[1] output; no later cross-position operation",
            "final_mask_feature": "net.norm output at appended MASK; positive control",
        }
        frequencies, occurrences = Counter(), {}
        for user, history in probe._histories().items():
            checkpoint()
            items = [int(item) for item in history]
            for item in items:
                if 1 <= item <= net.itemnum:
                    frequencies[item] += 1
            occurrences[str(user)] = items
        bins = lambda item: int(
            np.searchsorted([1, 4, 16, 64, 256], frequencies[item], side="right")
        )
        donors = {}
        for user, history in occurrences.items():
            checkpoint()
            visible = history[-(net.maxlen - 1) :]
            for index, item in enumerate(visible):
                if 1 <= item <= net.itemnum:
                    band = min(3, 4 * index // max(1, len(visible)))
                    donors.setdefault((band, bins(item)), []).append((user, item))
        permutation = rng.permutation(len(examples))
        ordered = [examples[int(index)] for index in permutation]
        midpoint = len(ordered) // 2
        if midpoint < 8 or len(ordered) - midpoint < 8:
            raise ValueError("Fewer than eight eligible users in each partition.")
        result["counts"].update(
            eligible_users=len(ordered),
            exploration_users=midpoint,
            confirmation_users=len(ordered) - midpoint,
        )
        result["locked_sites"] = []
        donor_histories = sorted(occurrences.items())

        def forward(sequence, positions, candidates, patch=None, capture=False):
            checkpoint(synchronize=True)
            cache, visits = {}, Counter()
            try:
                for name, module in modules.items():

                    def hook(_module, _args, output, name=name):
                        visits[name] += 1
                        if not isinstance(output, torch.Tensor) or output.ndim != 3:
                            raise ValueError(f"Unsupported residual output at {name}")
                        if not bool(torch.isfinite(output).all()):
                            raise ValueError(f"Nonfinite residual output at {name}")
                        if capture:
                            cache[name] = output.detach().clone()
                        if patch is not None and name == patch[0]:
                            if output[0, patch[1]].shape != patch[2].shape:
                                raise ValueError(f"Patch shape mismatch at {name}")
                            output = output.clone()
                            output[0, patch[1]] = patch[2]
                        return output

                    hooks.append(module.register_forward_hook(hook))
                scores = probe._score(sequence, positions, candidates)
                if any(visits[name] != 1 for name in modules):
                    raise ValueError(
                        f"Unsupported hook execution counts: {dict(visits)}"
                    )
                result["counts"]["completed_forwards"] = (
                    result["counts"].get("completed_forwards", 0) + 1
                )
                checkpoint(synchronize=True)
                return scores, cache
            finally:
                while hooks:
                    hooks.pop().remove()

        def measures(scores, candidates, competitor, baseline_order):
            order = probe._ranking(scores, candidates)
            rank = order.index(int(candidates[0])) + 1
            negatives = scores[1:]
            return {
                "fixed_contrast": float(scores[0] - scores[competitor]),
                "switching_margin": float(scores[0] - negatives.max()),
                "tenth_negative_margin": float(
                    scores[0] - negatives.sort(descending=True).values[9]
                ),
                "target_rank": float(rank),
                "reciprocal_rank": 1.0 / rank,
                "hit10": float(rank <= 10),
                "top10_overlap": len(set(order[:10]) & set(baseline_order[:10])) / 10.0,
                "ordered_agreement": float(order == baseline_order),
            }

        for partition, subset in (
            ("exploration", ordered[:midpoint]),
            ("confirmation", ordered[midpoint:]),
        ):
            if partition == "confirmation":
                norms = [row["change_norm"] for row in result["records"]]
                result["protocol"]["magnitude_edges"] = (
                    np.unique(np.quantile(norms, [0.25, 0.5, 0.75])).tolist()
                    if norms
                    else []
                )
                for rule in result["protocol"]["rules"]:
                    for direction in ("restore", "disrupt"):
                        choices = []
                        for site in result["protocol"]["sites"]:
                            rows = [
                                r
                                for r in result["records"]
                                if r["partition"] == "exploration"
                                and r["rule"] == rule
                                and r["direction"] == direction
                                and r["site"] == site
                            ]
                            if len(rows) >= 8 and all(
                                r["controls_passed"] for r in rows
                            ):
                                effect = float(
                                    np.mean(
                                        [r["delta"]["fixed_contrast"] for r in rows]
                                    )
                                )
                                if effect != 0.0:
                                    choices.append(
                                        (abs(effect), site, 1 if effect > 0 else -1)
                                    )
                        if choices:
                            _, site, sign = max(choices)
                            locked[(rule, direction)] = (site, sign)
                            result["locked_sites"].append(
                                {
                                    "rule": rule,
                                    "direction": direction,
                                    "site": site,
                                    "sign": sign,
                                }
                            )
                if not locked:
                    raise ValueError(
                        "No nonzero exploration effect with sufficient users and passing controls."
                    )
            for user, prefix, target, ids in subset:
                checkpoint()
                _, clean, positions = probe._serving_tensors(prefix)
                visible = torch.nonzero(clean[0, :-1].ne(0)).flatten().tolist()
                position = visible[-1]
                band = min(3, 4 * visible.index(position) // len(visible))
                original = int(clean[0, position])
                eligible = sorted(
                    set(
                        item
                        for owner, item in donors.get((band, bins(original)), [])
                        if owner != user and item != original and item not in ids
                    )
                )
                if len(eligible) < 2:
                    result["counts"]["unavailable_matched_donors"] = (
                        result["counts"].get("unavailable_matched_donors", 0) + 1
                    )
                    continue
                clean_scores, clean_cache = forward(
                    clean,
                    positions,
                    torch.tensor(ids, device=probe.device),
                    capture=True,
                )
                candidates = torch.tensor(ids, dtype=torch.long, device=probe.device)
                clean_order = probe._ranking(clean_scores, candidates)
                competitor_id = next(item for item in clean_order if item != target)
                competitor = ids.index(competitor_id)
                first = int(rng.choice(eligible))
                second = min(
                    (item for item in eligible if item != first),
                    key=lambda item: (
                        abs(frequencies[item] - frequencies[original]),
                        item,
                    ),
                )
                chosen = (first, second)
                unrelated = next(
                    (
                        history
                        for owner, history in donor_histories
                        if owner != user
                        and len(history) >= len(visible)
                        and not (set(history[-len(visible) :]) & set(ids))
                        and all(
                            1 <= item <= net.itemnum
                            for item in history[-len(visible) :]
                        )
                    ),
                    None,
                )
                unrelated_cache = None
                if unrelated is not None:
                    _, donor_sequence, donor_positions = probe._serving_tensors(
                        unrelated[-len(visible) :]
                    )
                    _, unrelated_cache = forward(
                        donor_sequence, donor_positions, candidates, capture=True
                    )
                for rule, donor in zip(result["protocol"]["rules"], chosen):
                    corrupt = clean.clone()
                    corrupt[0, position] = int(donor)
                    corrupt_scores, corrupt_cache = forward(
                        corrupt, positions, candidates, capture=True
                    )
                    mask_scores, _ = forward(
                        corrupt,
                        positions,
                        candidates,
                        (
                            "final_mask_feature",
                            net.maxlen - 1,
                            clean_cache["final_mask_feature"][0, -1],
                        ),
                    )
                    zero_scores, _ = forward(
                        corrupt,
                        positions,
                        candidates,
                        (
                            "final_history_output",
                            position,
                            clean_cache["final_history_output"][0, position],
                        ),
                    )
                    positive_ok = bool(
                        torch.allclose(mask_scores, clean_scores, atol=1e-5, rtol=1e-4)
                    )
                    structural_ok = bool(
                        torch.allclose(
                            zero_scores, corrupt_scores, atol=1e-5, rtol=1e-4
                        )
                    )
                    for direction in ("restore", "disrupt"):
                        base_sequence, base_scores, source, destination = (
                            (corrupt, corrupt_scores, clean_cache, corrupt_cache)
                            if direction == "restore"
                            else (clean, clean_scores, corrupt_cache, clean_cache)
                        )
                        sites = (
                            result["protocol"]["sites"]
                            if partition == "exploration"
                            else (
                                [locked[(rule, direction)][0]]
                                if (rule, direction) in locked
                                else []
                            )
                        )
                        for site in sites:
                            base = measures(
                                base_scores, candidates, competitor, clean_order
                            )
                            vector = source[site][0, position]
                            scores, _ = forward(
                                base_sequence,
                                positions,
                                candidates,
                                (site, position, vector),
                            )
                            identity, _ = forward(
                                base_sequence,
                                positions,
                                candidates,
                                (site, position, destination[site][0, position]),
                            )
                            norm = float(
                                (vector - destination[site][0, position]).norm()
                            )
                            same_band = [
                                p
                                for i, p in enumerate(visible)
                                if min(3, 4 * i // len(visible)) == band
                            ]
                            random_position = int(rng.choice(same_band))
                            noise = torch.tensor(
                                rng.normal(size=vector.numel()),
                                dtype=vector.dtype,
                                device=vector.device,
                            )
                            noise = F.normalize(noise, dim=0) * norm
                            random_vector = (
                                destination[site][0, random_position] + noise
                            )
                            achieved_norm = float(
                                (
                                    random_vector
                                    - destination[site][0, random_position]
                                ).norm()
                            )
                            magnitude_ok = bool(
                                np.isclose(achieved_norm, norm, atol=1e-5, rtol=1e-4)
                            )
                            random_scores, _ = forward(
                                base_sequence,
                                positions,
                                candidates,
                                (site, random_position, random_vector),
                            )
                            unrelated_scores = None
                            if unrelated_cache is not None:
                                unrelated_scores, _ = forward(
                                    base_sequence,
                                    positions,
                                    candidates,
                                    (
                                        site,
                                        position,
                                        unrelated_cache[site][0, position],
                                    ),
                                )
                            values = measures(
                                scores, candidates, competitor, clean_order
                            )
                            identity_ok = bool(
                                torch.allclose(
                                    identity, base_scores, atol=1e-5, rtol=1e-4
                                )
                            )
                            result["records"].append(
                                {
                                    "user": user,
                                    "partition": partition,
                                    "rule": rule,
                                    "direction": direction,
                                    "site": site,
                                    "position": position,
                                    "donor": int(donor),
                                    "competitor": competitor_id,
                                    "baseline": base,
                                    "clean": measures(
                                        clean_scores,
                                        candidates,
                                        competitor,
                                        clean_order,
                                    ),
                                    "corrupt": measures(
                                        corrupt_scores,
                                        candidates,
                                        competitor,
                                        clean_order,
                                    ),
                                    "delta": {
                                        key: values[key] - base[key] for key in base
                                    },
                                    "random_control": measures(
                                        random_scores,
                                        candidates,
                                        competitor,
                                        clean_order,
                                    ),
                                    "unrelated_control": (
                                        measures(
                                            unrelated_scores,
                                            candidates,
                                            competitor,
                                            clean_order,
                                        )
                                        if unrelated_scores is not None
                                        else None
                                    ),
                                    "unrelated_control_reason": (
                                        None
                                        if unrelated_scores is not None
                                        else "No candidate-excluded unrelated training history of matching length."
                                    ),
                                    "change_norm": norm,
                                    "random_change_norm": achieved_norm,
                                    "random_position": random_position,
                                    "random_magnitude_passed": magnitude_ok,
                                    "magnitude_bin": None,
                                    "identity_passed": identity_ok,
                                    "mask_restoration_passed": positive_ok,
                                    "structural_zero_passed": structural_ok,
                                    "controls_passed": identity_ok
                                    and positive_ok
                                    and structural_ok
                                    and magnitude_ok
                                    and unrelated_scores is not None,
                                }
                            )
        result["available"] = bool(result["records"])
        if not result["available"]:
            result["reason"] = (
                "No completed interventions with available matched donors."
            )
    except TimeoutError as exc:
        result["timed_out"] = True
        result["reason"] = str(exc)
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        while hooks:
            hooks.pop().remove()
        if probe.device.type == "cuda":
            torch.cuda.synchronize(probe.device)
        result["forward_phase_seconds"] = time.perf_counter() - started
    # Partial runs remain descriptive; confirmation requires a completed pilot.
    groups = {}
    edges = result["protocol"]["magnitude_edges"]
    for row in result["records"]:
        if edges is not None:
            row["magnitude_bin"] = int(
                np.searchsorted(edges, row["change_norm"], side="right")
            )
        key = (row["partition"], row["rule"], row["direction"], row["site"])
        groups.setdefault(key, []).append(row)
    for key, rows in groups.items():
        values = np.asarray(
            [[row["delta"][name] for name in row["delta"]] for row in rows]
        )
        indices = np.random.default_rng(42).integers(0, len(rows), (500, len(rows)))
        intervals = np.percentile(values[indices].mean(1), [2.5, 97.5], axis=0)
        result["summary"]["/".join(key)] = {
            "users": len(rows),
            "controls_passed": all(row["controls_passed"] for row in rows),
            "effects": {
                name: {
                    "mean": float(values[:, i].mean()),
                    "ci95": intervals[:, i].tolist(),
                    "zero_effect_users": int(np.count_nonzero(values[:, i] == 0)),
                }
                for i, name in enumerate(rows[0]["delta"])
            },
        }
        if key[0] == "confirmation":
            sign = locked[(key[1], key[2])][1]
            effects = values[:, 0] * sign
            nonzero = effects[effects != 0]
            valid = (
                result["available"]
                and not result["timed_out"]
                and result["reason"] is None
                and len(rows) >= 8
                and all(row["controls_passed"] for row in rows)
            )
            p = (
                float(
                    binomtest(
                        int((nonzero > 0).sum()),
                        len(nonzero),
                        0.5,
                        alternative="greater",
                    ).pvalue
                )
                if valid and len(nonzero)
                else None
            )
            result["confirmation"].append(
                {
                    "rule": key[1],
                    "direction": key[2],
                    "site": key[3],
                    "locked_sign": sign,
                    "users": len(rows),
                    "nonzero_users": len(nonzero),
                    "raw_p": p,
                    "holm_adjusted_p": None,
                    "holm_status": "Requires all four category results; no confirmed discovery declared.",
                    "conservative_family_bound_p": (
                        min(1.0, 16 * p) if p is not None else None
                    ),
                }
            )
    result["counts"]["completed_interventions"] = len(result["records"])
    result["counts"]["completed_users_by_partition"] = {
        partition: len(
            {r["user"] for r in result["records"] if r["partition"] == partition}
        )
        for partition in ("exploration", "confirmation")
    }
    result["limitations"] = (
        "Frozen-checkpoint intervention-conditional pilot. Natural donor hybrids may be out of distribution. Frequency/recency matching does not establish semantic compatibility. Random controls directly match the measured residual-change norm; descriptive bins are learned on exploration only. No unique mechanism, intent, validation gain, training-seed uncertainty, or repair comparison is established. The wall-time cap is checked around synchronized forwards and during preparation; an in-flight operation and CPU summaries can overrun it."
    )
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


### <<< Self_EvolveRec-BLOCK-END


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Propagate explicit category identity from the trained network
    def __init__(self, view, device):
        self.model = view
        self.view = view
        self.net = view.model
        self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train
        self.category = getattr(view, "dataset_name", None)
        if self.category is None:
            self.category = getattr(self.net, "dataset_name", None)
        self._contract = {}
        self._embedding_count = 0

    def _get_categories(self, item_id: int):
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Seeded real-item embedding alignment with explicit missingness
    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.view.item_emb.weight.detach()[1 : self.net.itemnum + 1]
            k = min(2048, sample_size, emb.size(0))
            self._embedding_count = k
            if k < 2:
                return None
            generator = torch.Generator(device="cpu").manual_seed(seed)
            indices = torch.randperm(emb.size(0), generator=generator)[:k].to(
                emb.device
            )
            x = F.normalize(emb.index_select(0, indices).float(), dim=-1)
            if not torch.isfinite(x).all():
                return None
            mean_cos = (x.sum(0).square().sum() - x.square().sum()) / (k * (k - 1))
            return float((mean_cos.clamp(-1, 1).item() + 1.0) / 2.0)

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Check supplied all-MASK sampler before in-sample margin measurement
    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(
        self,
        dataloader,
        num_batches: int = 10,
        eps: float = 0.5,
        topk_categories: int = 20,
    ):
        from itertools import islice

        margins = []
        checks = {
            "batches": 0,
            "valid_batches": 0,
            "invalid_batches": 0,
            "supervised_positions": 0,
            "errors": [],
        }
        reserved_ok = (
            self.net.mask_token == self.net.itemnum + 1
            and self.view.item_emb.num_embeddings == self.net.itemnum + 2
            and self.view.item_emb.padding_idx == 0
        )
        checks["reserved_tokens_valid"] = bool(reserved_ok)
        checks["policy"] = (
            "Supplied terminal/interior sampler: all selected targets replaced by MASK."
        )
        previous_state = self.net.training
        try:
            self.net.eval()
            for batch in islice(dataloader, min(10, num_batches)):
                checks["batches"] += 1
                try:
                    if not isinstance(batch, (tuple, list)) or len(batch) != 4:
                        raise ValueError(
                            "expected (None, sequence, positive, negative)"
                        )
                    seq, pos, neg = [
                        torch.as_tensor(x, device=self.device, dtype=torch.long)
                        for x in batch[1:4]
                    ]
                    if (
                        seq.ndim != 2
                        or seq.shape != pos.shape
                        or seq.shape != neg.shape
                    ):
                        raise ValueError("expected matching B x L tensors")
                    if seq.shape[1] != self.net.maxlen or seq.shape[0] == 0:
                        raise ValueError("unexpected sequence length or empty batch")
                    valid = pos.ne(0)
                    ok = (
                        reserved_ok
                        and bool(((seq >= 0) & (seq <= self.net.mask_token)).all())
                        and bool((valid.sum(1) > 0).all())
                        and bool((seq[valid] == self.net.mask_token).all())
                        and bool(
                            ((pos[valid] >= 1) & (pos[valid] <= self.net.itemnum)).all()
                        )
                        and bool(
                            ((neg[valid] >= 1) & (neg[valid] <= self.net.itemnum)).all()
                        )
                        and bool((pos[valid] != neg[valid]).all())
                        and bool((neg[~valid] == 0).all())
                    )
                    if not ok:
                        raise ValueError(
                            "reserved-token, all-MASK supervision, or real-item contract failed"
                        )
                    features = self.view.log2feats(seq.detach().cpu().numpy())
                    logits = self.view.predict_score(
                        features, torch.stack((pos, neg), -1)
                    )
                    margin = (logits[..., 0] - logits[..., 1])[valid].float()
                    if not torch.isfinite(margin).all():
                        raise ValueError("nonfinite supervised logits")
                    margins.append(margin.detach().cpu())
                    checks["valid_batches"] += 1
                    checks["supervised_positions"] += int(valid.sum().item())
                except (ValueError, RuntimeError, TypeError, IndexError) as exc:
                    checks["invalid_batches"] += 1
                    checks["errors"].append(str(exc))
        finally:
            self.net.train(previous_state)
        checks["passed"] = bool(
            reserved_ok and checks["valid_batches"] and not checks["invalid_batches"]
        )
        checks["provenance_limit"] = (
            "Checks the supplied all-MASK policy; loader tensors alone cannot establish "
            "prefix provenance or verify uniform negative sampling."
        )
        self._contract = checks
        result = dict.fromkeys(
            ("mean", "p10", "neg_beats_pos_rate", "small_margin_rate")
        )
        result["count"] = 0
        low = wrong = 0
        if margins:
            values = torch.cat(margins)
            result.update(
                count=int(values.numel()),
                mean=float(values.mean()),
                p10=float(
                    values.kthvalue(max(1, int(np.ceil(0.1 * values.numel())))).values
                ),
                neg_beats_pos_rate=float((values < 0).float().mean()),
                small_margin_rate=float((values < eps).float().mean()),
            )
            low, wrong = int((values < eps).sum()), int((values < 0).sum())
        breakdown = {
            "available": False,
            "reason": "Item categories are unavailable in this benchmark.",
            "eps": float(eps),
            "low_margin_total_positions": low,
            "wrong_rank_total_positions": wrong,
            "low_margin_top_categories": {"pos_item": [], "neg_item": []},
            "wrong_rank_top_categories": {"pos_item": [], "neg_item": []},
        }
        return result, breakdown

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Independently gated sensitivity, residual and canonical catalog audits
    def _histories(self):
        if "History" in self.user_train:
            return self.user_train["History"]
        return self.user_train

    def _serving_tensors(self, history):
        length = self.net.maxlen
        raw = torch.zeros((1, length), dtype=torch.long, device=self.device)
        retained = list(history[-length:])
        if retained:
            raw[0, -len(retained) :] = torch.tensor(retained, device=self.device)
        serving = torch.zeros_like(raw)
        serving[:, :-1] = raw[:, 1:]
        serving[:, -1] = self.net.mask_token
        positions = torch.arange(1, length + 1, device=self.device)[None, :]
        positions = positions * serving.ne(0)
        return raw, serving, positions

    def _explicit_forward(self, sequence, positions):
        # This is the supplied BERT4Rec encoder, with fixed absolute positions.
        # Interior PAD holes are excluded from attention keys in every layer.
        padding = sequence.eq(0)
        if bool(padding.all(1).any()):
            raise ValueError("diagnostic inputs must contain an attention-visible MASK")
        x = self.net.item_emb(sequence) * (self.net.hidden_units**0.5)
        x = self.net.dropout(x + self.net.pos_emb(positions))
        return self.net.norm(self.net.encoder(x, src_key_padding_mask=padding))

    @staticmethod
    def _ranking(scores, candidates):
        scores = torch.as_tensor(scores).detach().float().cpu().reshape(-1)
        candidates = torch.as_tensor(candidates).detach().long().cpu().reshape(-1)
        if scores.numel() != candidates.numel() or not torch.isfinite(scores).all():
            raise ValueError("candidate scores must have matching length and be finite")
        by_id = torch.argsort(candidates, stable=True)
        order = by_id[torch.argsort(scores[by_id], descending=True, stable=True)]
        return candidates[order].tolist()

    def _feature_scores(self, feature, candidates):
        # Preserve the documented B x L x C scoring contract even for one MASK.
        feature = feature.reshape(1, 1, -1)
        items = candidates.reshape(1, 1, -1)
        scores = self.view.predict_score(feature, items)
        if scores.shape != items.shape or not bool(torch.isfinite(scores).all()):
            raise ValueError("ModelView must return finite B x L x C logits")
        return scores.reshape(-1)

    def _score(self, sequence, positions, candidates):
        features = self._explicit_forward(sequence, positions)
        # Read the actual appended MASK slot, including after deletion holes.
        if int(sequence[0, -1]) != self.net.mask_token:
            raise ValueError("serving readout must contain the reserved MASK")
        return self._feature_scores(features[0, -1], candidates)

    def _examples(self):
        rng = random.Random(42)
        candidate_rng = np.random.default_rng(42)
        histories = self._histories()
        users = sorted(histories, key=lambda user: str(user))
        selected = rng.sample(users, min(64, len(users)))
        counts = {
            "training_users": len(users),
            "selected_users": len(selected),
            "invalid_identity": 0,
            "no_eligible_endpoint": 0,
            "repeated_target_endpoints_excluded": 0,
            "insufficient_candidate_pool": 0,
            "insufficient_visible_history": 0,
        }
        examples = []
        for user in selected:
            try:
                history = [int(item) for item in histories[user]]
            except (TypeError, ValueError):
                counts["invalid_identity"] += 1
                continue
            if any(item < 1 or item > self.net.itemnum for item in history):
                counts["invalid_identity"] += 1
                continue
            endpoints, seen = [], set()
            for endpoint, target in enumerate(history):
                if endpoint >= 4:
                    if target not in seen:
                        endpoints.append(endpoint)
                    else:
                        counts["repeated_target_endpoints_excluded"] += 1
                seen.add(target)
            if not endpoints:
                counts["no_eligible_endpoint"] += 1
                continue
            endpoint = rng.choice(endpoints)
            prefix, target = history[:endpoint], history[endpoint]
            excluded = set(prefix) | {target}
            pool = np.setdiff1d(
                np.arange(1, self.net.itemnum + 1, dtype=np.int64),
                np.fromiter(excluded, dtype=np.int64),
                assume_unique=False,
            )
            if pool.size < 99:
                counts["insufficient_candidate_pool"] += 1
                continue
            candidates = [target] + candidate_rng.choice(
                pool, 99, replace=False
            ).tolist()
            examples.append((str(user), prefix, target, candidates))
        counts["candidate_examples"] = len(examples)
        return examples, counts

    def _exported_predictor(self):
        import importlib.util
        import os
        import sys

        # Inside the existing diagnostic block: avoid dynamic-class introspection.
        path = getattr(self.view, "predictor_path", None)
        if path is None:
            path = getattr(self.net, "predictor_path", None)
        if path is None:
            # diagnosis_tools.py belongs to this trial; never search other trials.
            path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "working", "predict.py"
            )
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "Current trial's exported predict.py is not available"
            )
        spec = importlib.util.spec_from_file_location(
            "_bert4rec_diagnostic_export", path
        )
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load the current trial's exported predictor")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if not all(
            hasattr(module, name)
            for name in ("predict", "_models", "_metas", "_device")
        ):
            raise ValueError(
                "exported predictor does not expose the supplied baseline interface"
            )
        required_routes = (
            "Movies_and_TV",
            "Industrial_and_Scientific",
            "Electronics",
            "CDs_and_Vinyl",
        )
        route = self.category
        if route not in required_routes:
            # Some host adapters rebuild ModelView from a checkpoint and omit
            # Python attributes. Resolve only by exact current-trial state identity.
            import pickle

            directory = os.path.dirname(os.path.abspath(path))
            matches = []
            actual = self.net.state_dict()
            for candidate in required_routes:
                checkpoint = os.path.join(directory, f"{candidate}_model.pt")
                metadata = os.path.join(directory, f"{candidate}_meta.pkl")
                if not os.path.isfile(checkpoint) or not os.path.isfile(metadata):
                    continue
                with open(metadata, "rb") as stream:
                    config = pickle.load(stream)
                if int(config["itemnum"]) != self.net.itemnum:
                    continue
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                identical = actual.keys() == state.keys() and all(
                    actual[name].shape == state[name].shape
                    and actual[name].dtype == state[name].dtype
                    and torch.equal(actual[name].detach().cpu(), state[name])
                    for name in actual
                )
                del state
                if identical:
                    matches.append(candidate)
            if len(matches) != 1:
                raise ValueError(
                    "Category identity unavailable: require one exact current-trial checkpoint match"
                )
            route = self.category = matches[0]
        if not callable(getattr(module, "_ensure_loaded", None)):
            raise ValueError("exported predictor has no checkpoint loading entry point")
        # Exercise the real writer-generated checkpoint loader for this route.
        # This reloads the same fitted checkpoint; it never fits or fuses models.
        module.DATASETS = [route]
        module._ensure_loaded()
        if set(module._models) != {route} or route not in module._metas:
            raise ValueError("exported loader must load exactly the requested category")
        exported = module._models[route].to(self.device)
        module._device = self.device
        exported.eval()
        if int(module._metas[route]["maxlen"]) != self.net.maxlen:
            raise ValueError("exported metadata disagrees with the diagnostic model")
        actual_state = self.net.state_dict()
        loaded_state = exported.state_dict()
        if actual_state.keys() != loaded_state.keys():
            raise ValueError(
                "checkpoint architecture differs from the diagnostic model"
            )
        for name, actual in actual_state.items():
            loaded = loaded_state[name]
            if (
                actual.shape != loaded.shape
                or actual.dtype != loaded.dtype
                or not torch.equal(actual, loaded)
            ):
                raise ValueError(
                    f"checkpoint differs at {name}; diagnose the selected checkpoint"
                )
        return module, route

    def _parity(self, examples):
        result = {
            "passed": False,
            "available": False,
            "atol": 1e-5,
            "rtol": 1e-4,
            "max_absolute_score_error": None,
            "ordered_ranking_agreement": None,
            "top10_agreement": None,
            "cases": 0,
            "coverage": {
                "short": False,
                "full_length": False,
                "truncated": False,
                "batched_vs_individual": False,
            },
            "checkpoint_serialization_verified": False,
            "weight_identity": "Unverified until the exported loader reloads the current category checkpoint and every state tensor matches exactly.",
            "external_label_invariance_tested": False,
        }
        try:
            if not examples:
                raise ValueError("no eligible training-prefix examples")
            module, route = self._exported_predictor()
            result["available"] = True
            result["checkpoint_serialization_verified"] = True
            result["weight_identity"] = (
                "All state tensors from the exported category checkpoint exactly "
                "match the actual diagnostic model, including dtype and shape."
            )
            cases = [(prefix, candidates) for _, prefix, _, candidates in examples]
            # Add a short case, then real full/truncated training-prefix cases.
            cases.append((examples[0][1][:2], examples[0][3]))
            for required in (self.net.maxlen - 1, self.net.maxlen + 1):
                for history in self._histories().values():
                    if len(history) >= required and required > 0:
                        prefix = [int(item) for item in history[:required]]
                        if all(1 <= item <= self.net.itemnum for item in prefix):
                            cases.append((prefix, examples[0][3]))
                            break
            errors, ordered, overlaps, passed = [], [], [], True
            paired_inputs, paired_positions, paired_scores = [], [], []
            for history, ids in cases:
                raw, sequence, positions = self._serving_tensors(history)
                candidates = torch.tensor(ids, dtype=torch.long, device=self.device)
                direct_features = self._explicit_forward(sequence, positions)
                view_features = self.view.log2feats(sequence.detach().cpu().numpy())
                if direct_features.shape != view_features.shape:
                    raise ValueError("ModelView returned an unexpected feature shape")
                visible = sequence.ne(0)
                passed &= bool(
                    torch.allclose(
                        direct_features[visible],
                        view_features[visible],
                        atol=1e-5,
                        rtol=1e-4,
                    )
                )
                # Compare the actual ModelView readout scores as well as features.
                direct = self._feature_scores(direct_features[0, -1], candidates)
                view_scores = self._feature_scores(view_features[0, -1], candidates)
                internal = self.net.score_candidates(raw, candidates).reshape(-1)
                external = torch.tensor(
                    module.predict(0, history, ids, dataset_name=route),
                    dtype=direct.dtype,
                    device=self.device,
                ).reshape(-1)
                reference_rank = self._ranking(direct, candidates)
                for other in (view_scores, internal, external):
                    finite = bool(torch.isfinite(other).all())
                    passed &= finite and bool(
                        torch.allclose(direct, other, atol=1e-5, rtol=1e-4)
                    )
                    if not finite:
                        raise ValueError("nonfinite parity scores")
                    errors.append(float((direct - other).abs().max()))
                    rank = self._ranking(other, candidates)
                    ordered.append(float(rank == reference_rank))
                    overlaps.append(
                        len(set(rank[:10]) & set(reference_rank[:10])) / 10.0
                    )
                result["coverage"]["short"] |= len(history) < self.net.maxlen - 1
                result["coverage"]["full_length"] |= len(history) == self.net.maxlen - 1
                result["coverage"]["truncated"] |= len(history) > self.net.maxlen - 1
                if len(paired_inputs) < 8:
                    paired_inputs.append(sequence)
                    paired_positions.append(positions)
                    paired_scores.append((direct, candidates))
            batch_features = self._explicit_forward(
                torch.cat(paired_inputs), torch.cat(paired_positions)
            )
            for index, (single, candidates) in enumerate(paired_scores):
                batched = self._feature_scores(batch_features[index, -1], candidates)
                if not torch.isfinite(batched).all():
                    raise ValueError("nonfinite batched parity scores")
                passed &= bool(torch.allclose(single, batched, atol=1e-5, rtol=1e-4))
                errors.append(float((single - batched).abs().max()))
                single_rank = self._ranking(single, candidates)
                batch_rank = self._ranking(batched, candidates)
                ordered.append(float(single_rank == batch_rank))
                overlaps.append(
                    len(set(single_rank[:10]) & set(batch_rank[:10])) / 10.0
                )
            result["coverage"]["batched_vs_individual"] = len(paired_inputs) > 1
            coverage_complete = all(result["coverage"].values())
            result.update(
                passed=bool(
                    passed
                    and coverage_complete
                    and result["checkpoint_serialization_verified"]
                    and all(value == 1.0 for value in ordered)
                ),
                coverage_complete=coverage_complete,
                cases=len(cases),
                max_absolute_score_error=max(errors),
                ordered_ranking_agreement=float(np.mean(ordered)),
                top10_agreement=float(np.mean(overlaps)),
            )
            if not coverage_complete:
                result["error"] = (
                    "Required parity coverage is incomplete; sensitivity is withheld."
                )
            elif not result["passed"]:
                result["error"] = (
                    "Feature, score, or ordered-ranking parity failed at fixed tolerances."
                )
        except Exception as exc:
            result["passed"] = False
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            # Release the checkpoint reload after verification; it is never fused.
            if "module" in locals():
                module._models.clear()
        return result

    def _paired_sensitivity(self, examples, counts):
        rows = []
        for user, prefix, target, ids in examples:
            _, original, positions = self._serving_tensors(prefix)
            visible = torch.nonzero(original[0, :-1].ne(0), as_tuple=False).flatten()
            k = min(2, (visible.numel() - 1) // 2)
            if k < 1:
                counts["insufficient_visible_history"] += 1
                continue
            recent, older, swapped = (
                original.clone(),
                original.clone(),
                original.clone(),
            )
            recent[0, visible[-k:]] = 0
            older[0, visible[:k]] = 0
            left, right = int(visible[-2]), int(visible[-1])
            swapped[0, left], swapped[0, right] = original[0, right], original[0, left]
            candidates = torch.tensor(ids, dtype=torch.long, device=self.device)
            measurements = {}
            for name, sequence in (
                ("original", original),
                ("recent_deleted", recent),
                ("older_deleted", older),
                ("last_two_swapped", swapped),
            ):
                scores = self._score(sequence, positions, candidates)
                ranking = self._ranking(scores, candidates)
                rank = ranking.index(target) + 1
                measurements[name] = {
                    "target_score": float(scores[0]),
                    "target_rank": float(rank),
                    "reciprocal_rank": 1.0 / rank,
                    "diagnostic_hit10": float(rank <= 10),
                    "ranking": ranking,
                }
            baseline = measurements["original"]
            row = {
                "user": user,
                "visible_history": int(visible.numel()),
                "deleted_per_variant": k,
            }
            for name, values in measurements.items():
                if name == "original":
                    row[name] = {
                        key: value for key, value in values.items() if key != "ranking"
                    }
                    continue
                row[name] = {
                    key + "_delta": values[key] - baseline[key]
                    for key in (
                        "target_score",
                        "target_rank",
                        "reciprocal_rank",
                        "diagnostic_hit10",
                    )
                }
                row[name].update(
                    top10_overlap=len(
                        set(values["ranking"][:10]) & set(baseline["ranking"][:10])
                    )
                    / 10.0,
                    ordered_ranking_agreement=float(
                        values["ranking"] == baseline["ranking"]
                    ),
                )
            rows.append(row)
        counts["paired_users"] = len(rows)
        rng = np.random.default_rng(42)
        # Reuse paired-user resamples across variants for comparable intervals.
        indices = rng.integers(0, len(rows), size=(500, len(rows))) if rows else None
        summary = {}
        for variant in ("recent_deleted", "older_deleted", "last_two_swapped"):
            keys = (
                "target_score_delta",
                "target_rank_delta",
                "reciprocal_rank_delta",
                "diagnostic_hit10_delta",
                "top10_overlap",
                "ordered_ranking_agreement",
            )
            if not rows:
                summary[variant] = {
                    key: {"mean": None, "ci95": None, "users": 0} for key in keys
                }
                continue
            values = np.asarray(
                [[row[variant][key] for key in keys] for row in rows], dtype=np.float64
            )
            boot = values[indices].mean(axis=1)
            summary[variant] = {
                key: {
                    "mean": float(values[:, index].mean()),
                    "ci95": np.percentile(boot[:, index], [2.5, 97.5]).tolist(),
                    "users": len(rows),
                }
                for index, key in enumerate(keys)
            }
        return {
            "per_category": [
                {
                    "dataset_name": self.category,
                    "counts": counts,
                    "summary": summary,
                    "paired_user_outcomes": rows,
                }
            ],
            "four_category_macro": None,
            "four_category_macro_available": False,
            "macro_unavailable_reason": "This ModelView contains one category; four category results are required.",
        }

    def run_full_diagnosis(self, train_loader):
        import time

        previous_modes = [(module, module.training) for module in self.net.modules()]
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        cpu_state = torch.random.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        metrics = {
            "status": "diagnostic_failed",
            "history_sensitivity": None,
            "catalog_boundary_audit": {
                "available": False,
                "complete": False,
                "reason": "Sampler and complete exported-serving parity must pass first.",
            },
            "residual_patching": {
                "available": False,
                "reason": "Sampler and complete exported-serving parity must pass first.",
            },
        }
        definitions = {
            "embedding_collapse_score": "(Mean ordered off-diagonal cosine + 1)/2 over at most 2048 real item embeddings sampled with seed 42. PAD/MASK excluded. 0.5 means approximately zero mean cosine; capacity or quality is not established. Null if fewer than two or nonfinite observations.",
            "embedding_sample_count": "Number of sampled real item embeddings, not number of pairs.",
            "pos_neg_margin_mean": "In-sample mean positive-minus-negative logit over verified masked supervised positions. Higher means stronger sampled-pair separation.",
            "pos_neg_margin_p10": "In-sample nearest-rank 10th percentile of those margins, in logit units.",
            "pos_neg_neg_beats_pos_rate": "Fraction of verified supervised positions with negative-minus-positive logit > 0; lower is better.",
            "pos_neg_small_margin_rate": "Fraction of verified supervised positions with positive-minus-negative logit < 0.5.",
            "pos_neg_margin_count": "Number of verified supervised positions in at most ten loader batches; zero observations yield null statistics.",
            "pos_neg_margin_category_breakdown": "Unavailable: item categories are not supplied. Category lists remain empty.",
            "sampler_contract": "Checks PAD=0, MASK=itemnum+1, real supervised IDs, and the supplied all-MASK corruption policy. Failure blocks sensitivity. Does not establish sampler provenance from tensors alone.",
            "serving_parity": "Evaluation mode, float32, atol=1e-5/rtol=1e-4. The exported loader reloads exactly the current category checkpoint; every state tensor must exactly match the actual diagnostic model. Compare explicit fixed-position and ModelView features at visible positions, their MASK logits, internal appended-MASK scoring, exported predict, and batched/individual scoring. Scores must be finite and all ordered rankings identical. Rankings use descending score then ascending item ID; top10 agreement is intersection size/10. Short, full-length, truncated, and batched coverage are all required before sensitivity. External-label mutation remains explicitly untested; no hidden labels are accessed.",
            "history_sensitivity": "Training-prefix diagnostic, not host validation. Uniformly select up to 64 users before inspecting outcomes, one endpoint/user uniformly among endpoints with >=4 preceding interactions and a target absent from the complete prefix. Negatives are 99 unique real items absent from that complete prefix and distinct from target; later events never determine eligibility. Seed 42. Complete-prefix exclusions and candidates remain fixed across all variants. Serving truncates to maxlen-1 and appends MASK at index maxlen-1. Delete earliest/latest k=min(2,floor((visible_count-1)/2)) visible items into attention-masked PAD holes; preserve surviving absolute positions and MASK index, without backfill. Swap final two visible items. Repeated historical IDs are permitted. Unknown IDs are excluded and counted, not negative labels.",
            "paired_statistics": "Variant minus original: target_score_delta in logits; target_rank_delta in one-based positions (positive means deterioration); reciprocal_rank_delta uses 1/rank; diagnostic_hit10_delta uses indicator(rank<=10). top10_overlap is intersection/10; ordered_ranking_agreement compares complete ordered candidate rankings. Equal-user category means and 500 CPU paired-user bootstrap replicates, seed 42, percentile 95% intervals. Intervals cover sampled-user uncertainty only. Raw paired outcomes are retained for a four-category aggregation; that macro is unavailable in a single-category ModelView, never computed over fewer categories.",
            "elapsed_diagnostic_seconds": "Wall seconds including warmup, loading, probes, and CPU bootstrap, CUDA synchronized at both boundaries; all diagnostic GPU work consumes search budget.",
            "availability": "Null means unavailable; counts/exclusions and parity coverage must accompany interpretation. No ratings, review text, item categories/prices/stores, or simulator traits are measured.",
        }
        try:
            self.net.eval()
            with torch.no_grad():
                metrics["embedding_collapse_score"] = (
                    self.analyze_embeddings_collapse_cosine()
                )
                metrics["embedding_sample_count"] = self._embedding_count
                margin, breakdown = self.analyze_pos_neg_margin_with_category_breakdown(
                    train_loader
                )
                metrics.update(
                    {
                        "pos_neg_margin_mean": margin["mean"],
                        "pos_neg_margin_p10": margin["p10"],
                        "pos_neg_neg_beats_pos_rate": margin["neg_beats_pos_rate"],
                        "pos_neg_small_margin_rate": margin["small_margin_rate"],
                        "pos_neg_margin_count": margin["count"],
                        "pos_neg_margin_category_breakdown": breakdown,
                        "sampler_contract": self._contract,
                    }
                )
                examples, counts = self._examples()
                metrics["prefix_counts"] = counts
                if not self._contract["passed"]:
                    metrics["status"] = "sampler_contract_failed"
                else:
                    # Explicitly require float32 instead of silently widening tolerances.
                    if self.net.item_emb.weight.dtype != torch.float32:
                        raise ValueError(
                            "serving parity requires the predeclared float32 precision"
                        )
                    parity = self._parity(examples)
                    metrics["serving_parity"] = parity
                    if not parity["passed"]:
                        metrics["status"] = (
                            "serving_parity_failed"
                            if parity["available"]
                            else "serving_parity_unavailable"
                        )
                    else:
                        # Independent failures preserve already completed probes.
                        # The catalog audit selects its own score-independent panel
                        # and never advances the diagnostic training loader.
                        try:
                            # The implementation is defined in this module.
                            metrics["catalog_boundary_audit"] = run_catalog_audit(
                                self, serving_parity=parity
                            )
                        except Exception as exc:
                            metrics["catalog_boundary_audit"] = {
                                "available": False,
                                "complete": False,
                                "reason": f"{type(exc).__name__}: {exc}",
                            }
                        try:
                            metrics["history_sensitivity"] = self._paired_sensitivity(
                                examples, counts
                            )
                        except Exception as exc:
                            metrics["history_sensitivity"] = {
                                "available": False,
                                "reason": f"{type(exc).__name__}: {exc}",
                            }
                        metrics["status"] = (
                            "measured"
                            if counts.get("paired_users", 0)
                            else "insufficient_history"
                        )
                        try:
                            metrics["residual_patching"] = _residual_patching_pilot(
                                self, examples, seconds=30.0
                            )
                        except Exception as exc:
                            metrics["residual_patching"] = {
                                "available": False,
                                "reason": f"{type(exc).__name__}: {exc}",
                            }
        except Exception as exc:
            metrics["status"] = "diagnostic_failed"
            metrics["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            for module, mode in previous_modes:
                module.training = mode
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            metrics["elapsed_diagnostic_seconds"] = time.perf_counter() - start
        definitions["catalog_boundary_audit"] = (
            "Frozen training-prefix diagnostic gated by sampler and serialized-serving "
            "parity. Independent seed-42 user permutation; up to 32 eligible users, "
            "uniform unseen-target endpoint with at least four preceding interactions. "
            "Exactly 99 unique prefix-excluded competitors, uniformly without replacement. "
            "One cached canonical float32 catalog traversal supplies sampled and exhaustive "
            "ranks using descending score then ascending ID. The nested metric_definitions "
            "define hypergeometric references, competitor-only margins and gaps, strata, "
            "counts and timing. A complete panel requires 32 valid completed users. "
            "Partial panels remain descriptive; no four-category macro is produced here. "
            "No optimizer steps, checkpoint selection, fitted auxiliary scorer or score "
            "fusion occur. Existing training and inference are unchanged; a fixed 0.5 "
            "training objective is not asserted by this measurement-only adapter."
        )
        definitions["residual_patching"] = (
            "Frozen two-layer pre-norm BERT4Rec intervention pilot, gated by sampler "
            "and serialized-checkpoint serving parity. Public training prefixes only; "
            "fixed 100 candidates, absolute positions, padding mask, and appended MASK. "
            "Two frequency/recency-matched non-PAD replacement rules at the most "
            "recent history position: uniform donor, then closest-frequency distinct "
            "donor. Exploration selects one site and sign per rule "
            "and direction; confirmation uses disjoint users. Delta means intervention "
            "minus its unpatched destination baseline. Primary fixed_contrast holds "
            "the strongest clean negative identity constant; switching_margin permits "
            "competitor changes. tenth_negative_margin measures the diagnostic HR@10 "
            "boundary. Rank is one-based; overlap/agreement reference the clean ranking. "
            "Identity, final-history structural-zero, and final-MASK restoration checks "
            "use atol=1e-5/rtol=1e-4. Missing unrelated donors block confirmation. "
            "Random-site controls match layer, recency band, component count, and "
            "activation-change norm, verified after floating-point addition. Descriptive "
            "magnitude quartiles use exploration only and are frozen for confirmation. "
            "Every registered residual hook must execute exactly once per forward. "
            "Raw control outcomes are retained. Equal-user "
            "means and 500 paired-user bootstrap percentile intervals are descriptive. "
            "Confirmatory sign tests exclude and count exact zeros. Global Holm "
            "adjustment remains unavailable until all four categories are aggregated; "
            "the reported 16*p bound is conservative Bonferroni, not Holm. Partial "
            "or timed-out runs never yield confirmatory p-values. No repairs are "
            "trained, fused, or compared by this diagnostic."
        )
        return {"metrics": metrics, "metric_definitions": definitions}

    ### <<< Self_EvolveRec-BLOCK-END
