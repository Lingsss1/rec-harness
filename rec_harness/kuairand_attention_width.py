"""CPU-only frozen HSTU dynamic-width experiment, separate from campaigns.

Uses only AlignedDataset(split='train'). Query-bank donors, calibration donors,
test donors, and task-panel users have disjoint identities. The task labels were
seen during original model training: this is a mechanism/compression diagnostic,
not a new validation score or a claim about retrained model capacity.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import torch
from torch.nn import functional as F

from .attention_width import (balanced_groups, merged_output, merged_parameters, nmse,
                              paired_interval, response_features, responses, spectrum)
from .core import ContractError, canonical, file_hash, read_json, write_json


@torch.inference_mode()
def capture(model, ids, times, query_tail=64):
    """Native full-prefix forward plus independent last-query replay caches."""
    encoder, cfg = model.encoder, model.cfg
    if cfg.name != "hstu":
        raise ContractError("this adapter implements dense HSTU only; sparse/Top-K adapters need separate checks")
    x = encoder.initial(ids)
    initial = x[:, -1].clone()
    valid = ids.ne(0).unsqueeze(-1)
    batch, length = ids.shape
    snapshots = [[] for _ in range(batch)]
    bank = []
    for layer in encoder.layers:
        projected = F.silu(F.layer_norm(x, (cfg.dim,), eps=1e-6) @ layer.uvqk)
        _, value, query, key = projected.split(
            [cfg.heads * cfg.dv, cfg.heads * cfg.dv, cfg.heads * cfg.dqk, cfg.heads * cfg.dqk], -1)
        query = query.reshape(batch, length, cfg.heads, cfg.dqk)
        key = key.reshape(batch, length, cfg.heads, cfg.dqk)
        value = value.reshape(batch, length, cfg.heads, cfg.dv)
        bank.append(query[:, -query_tail:].reshape(-1, cfg.heads, cfg.dqk).clone())
        bias = layer.bias(times, torch.tensor([length - 1]), torch.arange(length))[:, 0]
        for b in range(batch):
            snapshots[b].append(dict(key=key[b].clone(), value=value[b].clone(),
                                     query=query[b, -1].clone(), bias=bias[b].clone()))
        # The original implementation remains the reference forward path.
        x = layer(x, times) * valid
    result = F.normalize(x[:, -1], dim=-1, eps=1e-6)
    records = [dict(initial=initial[b], layers=snapshots[b]) for b in range(batch)]
    return records, result, bank


@torch.inference_mode()
def replay_last(model, record, groups, only_layer=None, visible_length=None):
    """Change only the final query at selected layers; prefix rows remain native.

    Causality makes cached prefix rows exact. The final token's Q/K/V are
    recomputed after every intervention, so changes propagate through depth.
    This does not pretend to compress every historical query or training cost.
    """
    cfg = model.cfg
    x = record["initial"].clone()
    for li, layer in enumerate(model.encoder.layers):
        projected = F.silu(F.layer_norm(x, (cfg.dim,), eps=1e-6) @ layer.uvqk)
        u, vl, q, kl = projected.split(
            [cfg.heads * cfg.dv, cfg.heads * cfg.dv, cfg.heads * cfg.dqk, cfg.heads * cfg.dqk], -1)
        q, kl, vl = q.reshape(cfg.heads, cfg.dqk), kl.reshape(cfg.heads, cfg.dqk), vl.reshape(cfg.heads, cfg.dv)
        saved = record["layers"][li]
        keys, values = saved["key"].clone(), saved["value"].clone()
        keys[-1], values[-1] = kl, vl
        start = 0 if visible_length is None else max(0, len(keys) - visible_length)
        pieces = []
        for hi in range(cfg.heads):
            k, v, b = keys[start:, hi], values[start:, hi], saved["bias"][start:]
            selected = groups.get((li, hi)) if groups is not None and (only_layer is None or only_layer == li) else None
            if selected is not None:
                k, v, b = merged_parameters(k, v, b, selected)
            weight = F.silu(k @ q[hi] + b) / layer.bias.padded_length
            pieces.append(weight @ v)
        attended = torch.cat(pieces)
        gated = u * F.layer_norm(attended, (cfg.heads * cfg.dv,), eps=1e-6)
        x = x + layer.output(gated)
    return F.normalize(x, dim=-1, eps=1e-6)


def groups_for(keys, calibration, budgets, seed):
    """Only calibration responses enter assignments; all controls size-matched."""
    n = len(keys)
    response = balanced_groups(response_features(calibration, seed), budgets)
    key = balanced_groups(F.normalize(keys, dim=-1, eps=1e-12), budgets)
    permutation = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    return {"response": response, "key": key,
            "random": {m: permutation.reshape(m, n // m) for m in budgets},
            "recent": {m: torch.arange(n - m, n)[:, None] for m in budgets}}


def analyze_snapshot(saved, calibration_queries, test_queries, lengths, budgets,
                     query_counts, seed, scale, measure):
    geometry, compressions, full_groups = [], [], None
    for length in lengths:
        keys, values, bias = saved["key"][-length:], saved["value"][-length:], saved["bias"][-length:]
        calibration = responses(calibration_queries, keys, bias, scale=scale)
        test = responses(test_queries, keys, bias, scale=scale)
        selected_budgets = sorted(set([m for m in budgets if m <= length] + [length]))
        assignments = groups_for(keys, calibration, selected_budgets, seed + length)
        if length == len(saved["key"]):
            full_groups = assignments
        if not measure:
            continue
        # Interleave the two query pools for stable sample-count convergence.
        both = torch.stack((calibration, test), 1).reshape(-1, length)
        for count in query_counts:
            if count > len(both):
                continue
            h = both[:count]
            mean_energy = float(h.mean(0).square().sum() * len(h) / h.square().sum().clamp_min(1e-24))
            for centered in (False, True):
                geometry.append(dict(length=length, queries=count, centered=centered,
                                     mean_energy_fraction=mean_energy, **spectrum(h, centered)))
        reference_test, reference_calibration = test @ values, calibration @ values
        natural_query = saved["query"][None]
        reference_natural = responses(natural_query, keys, bias, scale=scale) @ values
        for method, choices in assignments.items():
            for m, group in choices.items():
                predicted_test = merged_output(test_queries, keys, values, bias, group, scale=scale)
                predicted_cal = merged_output(calibration_queries, keys, values, bias, group, scale=scale)
                predicted_natural = merged_output(natural_query, keys, values, bias, group, scale=scale)
                compressions.append(dict(length=length, budget=m, method=method,
                    test_nmse=nmse(predicted_test, reference_test),
                    calibration_nmse=nmse(predicted_cal, reference_calibration),
                    natural_query_nmse=nmse(predicted_natural, reference_natural),
                    scale_preserved=scale, assignment_uses_test_queries=False))
    return geometry, compressions, full_groups


def append_rows(path, rows):
    with path.open("a") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


@torch.inference_mode()
def task_scores(contexts, items, ids, target, negatives, temperature, chunk=8192):
    """Full-catalog ranks with a shared seen filter; paired sampled training loss."""
    catalog = len(items) - 1
    positive = contexts @ items[target]
    seen = torch.zeros(catalog + 1, dtype=torch.bool)
    seen[ids] = True
    repeated = bool(seen[target])
    better = torch.zeros(len(contexts), dtype=torch.long)
    for start in range(1, catalog + 1, chunk):
        stop = min(start + chunk, catalog + 1)
        candidates = torch.arange(start, stop)
        logits = contexts @ items[start:stop].T
        eligible = ~seen[start:stop] & candidates.ne(target)
        ahead = (logits > positive[:, None]) | ((logits == positive[:, None]) & (candidates < target))
        better += (ahead & eligible).sum(-1)
    ranks = better + 1
    reference_ranks = torch.where((ranks > 2500) | repeated, 2501, ranks)
    hits = reference_ranks <= 10
    ndcg = hits.float() / torch.log2(reference_ranks.float() + 1)
    candidate_ids = torch.cat((torch.tensor([target]), negatives))
    sampled = contexts @ items[candidate_ids].T / temperature
    sampled[:, 1:] = sampled[:, 1:].masked_fill(negatives.eq(target)[None], -5e4)
    loss = -sampled.log_softmax(-1)[:, 0]
    return [dict(ndcg_at_10=float(ndcg[i]), hr_at_10=float(hits[i]),
                 rank=int(reference_ranks[i]), sampled_loss=float(loss[i]), repeated_target=repeated)
            for i in range(len(contexts))]


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(output):
    from collections import defaultdict
    import statistics
    spectra, merges, tasks = (read_rows(output / name) for name in ("spectra.jsonl", "merges.jsonl", "tasks.jsonl"))
    grouped = defaultdict(list)
    for row in spectra:
        grouped[(row["layer"], row["head"], row["length"], row["queries"], row["centered"])].append(row)
    spectrum_rows = []
    for (layer, head, length, queries, centered), rows in sorted(grouped.items()):
        spectrum_rows.append(dict(layer=layer, head=head, length=length, queries=queries, centered=centered,
            users=len(rows), effective_rank_median=statistics.median(r["effective_rank"] for r in rows),
            rank99_median=statistics.median(r["energy_rank_99"] for r in rows),
            mean_energy_fraction=statistics.mean(r["mean_energy_fraction"] for r in rows),
            sample_ceiling_reached=sum(r["ceiling_reached"] for r in rows)))
    grouped = defaultdict(list)
    for row in merges:
        grouped[(row["length"], row["budget"], row["method"])].append(row)
    merge_rows = []
    for (length, budget, method), rows in sorted(grouped.items()):
        values = torch.tensor([r["test_nmse"] for r in rows], dtype=torch.float64)
        merge_rows.append(dict(length=length, budget=budget, method=method, head_user_snapshots=len(rows),
            test_nmse_mean=float(values.mean()), test_nmse_median=float(values.median()),
            test_nmse_p90=float(values.quantile(.9)), fraction_under_one_percent=float((values <= .01).double().mean()),
            natural_query_nmse_median=statistics.median(r["natural_query_nmse"] for r in rows)))
    references = {r["user_index"]: r for r in tasks if r["method"] == "baseline"}
    grouped = defaultdict(list)
    for row in tasks:
        if row["method"] != "baseline":
            grouped[(row["method"], row["budget"], row["scope"])].append(row)
    task_rows = []
    for (method, budget, scope), rows in sorted(grouped.items()):
        ndcg_drop = [references[r["user_index"]]["ndcg_at_10"] - r["ndcg_at_10"] for r in rows]
        relative_loss = [(r["sampled_loss"] - references[r["user_index"]]["sampled_loss"]) /
                         max(references[r["user_index"]]["sampled_loss"], 1e-12) for r in rows]
        ndcg_ci, loss_ci = paired_interval(ndcg_drop), paired_interval(relative_loss)
        task_rows.append(dict(method=method, budget=budget, scope=scope, users=len(rows),
            ndcg_at_10=statistics.mean(r["ndcg_at_10"] for r in rows),
            ndcg_drop=ndcg_ci["mean"], ndcg_drop_ci_low=ndcg_ci["low"], ndcg_drop_ci_high=ndcg_ci["high"],
            relative_loss_increase=loss_ci["mean"], loss_ci_low=loss_ci["low"], loss_ci_high=loss_ci["high"],
            context_nmse_mean=statistics.mean(r["context_nmse"] for r in rows),
            task_tolerances_met=ndcg_ci["high"] <= .001 and loss_ci["high"] <= .01))
    write_csv(output / "spectra-summary.csv", spectrum_rows)
    write_csv(output / "compression-summary.csv", merge_rows)
    write_csv(output / "task-summary.csv", task_rows)
    report = dict(kind="frozen_hstu_dynamic_width_diagnostic", formal_result=False,
        research_context_eligible=False, final_holdout_used=False, training_partition_only=True,
        retrained_capacity_measured=False, speedup_measured=False,
        compression_scope="only last-query aggregation at each selected layer; causal prefix rows are native",
        task_users=len(references),
        baseline_ndcg_at_10=statistics.mean(r["ndcg_at_10"] for r in references.values()) if references else None,
        spectrum_summary=spectrum_rows, compression_summary=merge_rows, task_summary=task_rows,
        qualifications=["Labels were seen by the original baseline training; task changes are diagnostic, not new validation.",
            "Pooled donor queries probe a standardized conditional function, not the per-context natural query distribution.",
            "Rank is a proxy. Successful merging provides a constructive compression result, not a capacity lower bound.",
            "No end-to-end speedup claim: calibration, clustering and cached prefix computation are not amortized here."])
    write_json(output / "report.json", report)
    return report


def integer_list(value):
    return sorted(set(int(v) for v in value.split(",")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmark", default="artifacts/kuairand-a37797a/benchmarks/kuairand-longseq")
    parser.add_argument("--baseline", default="artifacts/kuairand-research/baseline/hstu")
    parser.add_argument("--cache", default="artifacts/kuairand-research/data/cache-l1024")
    parser.add_argument("--users", type=int, default=512)
    parser.add_argument("--geometry-users", type=int, default=64)
    parser.add_argument("--bank-users", type=int, default=32)
    parser.add_argument("--queries", type=int, default=512, help="per calibration/test donor pool")
    parser.add_argument("--lengths", type=integer_list, default=integer_list("128,256,512,1024"))
    parser.add_argument("--budgets", type=integer_list, default=integer_list("32,64,128,256,512"))
    parser.add_argument("--spectrum-queries", type=integer_list, default=integer_list("256,512,1024"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if args.summarize_only:
        summarize(output)
        return
    if args.bank_users < 4 or args.bank_users % 2 or args.users <= 0 or not 0 <= args.geometry_users <= args.users:
        parser.error("positive user count, even bank-users >=4, and geometry-users <= users required")
    if args.queries > args.bank_users // 2 * 64:
        parser.error("not enough distinct donor query vectors")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    output.mkdir(parents=True, exist_ok=False)
    start = time.time()
    state = dict(status="running", stage="load_and_verify", pid=os.getpid(), started_at=start,
                 completed_task_users=0, completed_geometry_users=0,
                 planned_task_users=args.users, planned_geometry_users=args.geometry_users)
    def status(**updates):
        state.update(updates, updated_at=time.time(), wall_seconds=time.time() - start)
        write_json(output / "status.json", state)
        print(canonical(state), flush=True)
    status()
    try:
        sys.path.insert(0, str(Path(args.benchmark).resolve()))
        import numpy as np
        from longseq_bench.models import BenchmarkModel, ModelConfig
        from longseq_bench.data import AlignedDataset, verify_cache
        manifest = read_json(Path(args.baseline) / "run.json")
        if manifest["model"]["name"] != "hstu":
            raise ContractError("dense HSTU checkpoint required")
        cfg = ModelConfig(**manifest["model"])
        if max(args.lengths) != cfg.max_history or cfg.max_history not in args.lengths:
            raise ContractError("length scan must include the full trained memory")
        for length in args.lengths:
            if length <= 0 or length & (length - 1):
                raise ContractError("memory lengths must be powers of two")
        for name, expected in manifest["code_files"].items():
            if file_hash(Path(args.benchmark) / "longseq_bench" / name) != expected:
                raise ContractError("pinned benchmark source differs: " + name)
        cache_meta = verify_cache(args.cache)
        if cache_meta["files"] != manifest["cache_files"]:
            raise ContractError("checkpoint/cache identity mismatch")
        checkpoint = Path(args.baseline) / "last.pt"
        checkpoint_hash = file_hash(checkpoint)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["fingerprint"] != manifest["fingerprint"] or saved["next_epoch"] != 101:
            raise ContractError("expected completed fixed-final-epoch HSTU checkpoint")
        tc = manifest["training"]
        model = BenchmarkModel(cfg, tc["negatives"], tc["temperature"], tc["loss_chunk"]).eval()
        model.load_state_dict(saved["model"])
        del saved
        dataset = AlignedDataset(args.cache, "train")
        eligible = np.flatnonzero(dataset.lengths >= cfg.max_history + 2)
        generator = np.random.default_rng(args.seed)
        selected = generator.permutation(eligible)[:args.bank_users + args.users].tolist()
        if len(selected) != args.bank_users + args.users:
            raise ContractError("insufficient users with complete 1024-event training histories")
        donors, panel = selected[:args.bank_users], selected[args.bank_users:]
        half = args.bank_users // 2
        contract = dict(kind="frozen_hstu_dynamic_width_v1", formal_result=False,
            research_context_eligible=False, partition="original_baseline_train",
            final_holdout_used=False, device="cpu", args=vars(args),
            checkpoint_sha256=checkpoint_hash, checkpoint_fingerprint=manifest["fingerprint"],
            source_hashes={name: file_hash(Path(__file__).with_name(name))
                           for name in ("kuairand_attention_width.py", "attention_width.py")},
            calibration_donor_users=donors[:half], test_donor_users=donors[half:], panel_users=panel,
            query_probe="empirical same-layer/head donor queries; anchor position/time biases fixed per snapshot",
            full_query_count=2 * args.queries, original_model=manifest["model"],
            thresholds=dict(branch_nmse=.01, relative_sampled_loss_increase=.01, ndcg_absolute_drop=.001),
            runtime=dict(torch=torch.__version__, threads=args.threads, python=sys.version,
                         affinity_cpus=len(os.sched_getaffinity(0)), cuda_used=False),
            intervention="nonlinear key/bias means, value sums, fixed original SiLU denominator; final query only")
        write_json(output / "contract.json", contract)
        banks = [[], []]
        for pool, user_indices in enumerate((donors[:half], donors[half:])):
            for left in range(0, len(user_indices), args.batch_size):
                indices = user_indices[left:left + args.batch_size]
                batch = [dataset[index] for index in indices]
                ids = torch.stack([b["ids"] for b in batch])
                times = torch.stack([b["times"] for b in batch])
                _, _, vectors = capture(model, ids, times)
                banks[pool].append(torch.stack(vectors))
                status(stage="query_bank", query_pool=pool, completed_donors=left + len(indices))
        query_pools = []
        for pool, parts in enumerate(banks):
            vectors = torch.cat(parts, dim=1)
            order = torch.randperm(vectors.shape[1], generator=torch.Generator().manual_seed(args.seed + pool))
            query_pools.append(vectors[:, order[:args.queries]].clone())
        calibration_bank, test_bank = query_pools
        torch.save(dict(calibration=calibration_bank, test=test_bank), output / "query-bank.pt")
        del banks, query_pools
        items = F.normalize(model.encoder.item_embedding.weight.detach(), dim=-1, eps=1e-6)
        with torch.inference_mode():
            for left in range(0, len(panel), args.batch_size):
                indices = panel[left:left + args.batch_size]
                batch = [dataset[index] for index in indices]
                ids = torch.stack([b["ids"] for b in batch])
                times = torch.stack([b["times"] for b in batch])
                records, original, _ = capture(model, ids, times)
                for offset, (user, record, example) in enumerate(zip(indices, records, batch)):
                    ordinal = left + offset
                    status(stage="geometry_and_compression" if ordinal < args.geometry_users else "task_panel",
                           active_user=ordinal, completed_task_users=ordinal)
                    replay = replay_last(model, record, None)
                    error = nmse(replay, original[offset])
                    if error > 1e-9:
                        raise ContractError("native/replay identity check failed: " + str(error))
                    task_groups = {}
                    for li, layer in enumerate(model.encoder.layers):
                        saved_layer = record["layers"][li]
                        for hi in range(cfg.heads):
                            snapshot = dict(key=saved_layer["key"][:, hi], value=saved_layer["value"][:, hi],
                                bias=saved_layer["bias"], query=saved_layer["query"][hi])
                            geometry, merges, assignments = analyze_snapshot(snapshot,
                                calibration_bank[li, :, hi], test_bank[li, :, hi],
                                args.lengths if ordinal < args.geometry_users else [cfg.max_history],
                                args.budgets, args.spectrum_queries, args.seed + user * 100 + li * 4 + hi,
                                layer.bias.padded_length, ordinal < args.geometry_users)
                            common = dict(user_index=user, layer=li, head=hi)
                            append_rows(output / "spectra.jsonl", [{**common, **row} for row in geometry])
                            append_rows(output / "merges.jsonl", [{**common, **row} for row in merges])
                            for method, choices in assignments.items():
                                for budget, group in choices.items():
                                    task_groups.setdefault((method, budget), {})[(li, hi)] = group
                    contexts = [original[offset]]
                    labels = [("baseline", cfg.max_history, "native")]
                    for (method, budget), assignments in sorted(task_groups.items()):
                        if budget == cfg.max_history:
                            continue
                        contexts.append(replay_last(model, record, assignments))
                        labels.append((method, budget, "all_layers_last_query"))
                    # A layer-local control distinguishes depth interactions from local effects.
                    layer_budget = min(args.budgets, key=lambda m: abs(m - 128))
                    assignments = task_groups.get(("response", layer_budget))
                    if assignments is not None:
                        for li in range(cfg.blocks):
                            contexts.append(replay_last(model, record, assignments, only_layer=li))
                            labels.append(("response", layer_budget, "layer_%d_last_query" % li))
                    # Whole-network truncation explicitly masks old history at every layer.
                    # It retains physical positions and original length normalization.
                    for length in args.lengths:
                        if length == cfg.max_history:
                            continue
                        shortened = example["ids"].clone()
                        shortened[:-length] = 0
                        context = model.encoder.encode_all(shortened[None], example["times"][None])[0, -1]
                        contexts.append(context)
                        labels.append(("history_truncation", length, "all_queries_recomputed"))
                    contexts = torch.stack(contexts)
                    negatives = torch.randint(1, cfg.num_items + 1, (tc["negatives"],),
                                              generator=torch.Generator().manual_seed(args.seed + user))
                    scores = task_scores(contexts, items, example["ids"], int(example["target"]),
                                         negatives, tc["temperature"])
                    task_rows = [dict(user_index=user, method=method, budget=budget, scope=scope,
                                      context_nmse=nmse(contexts[i], contexts[0]), replay_identity_nmse=error,
                                      **scores[i])
                                 for i, (method, budget, scope) in enumerate(labels)]
                    append_rows(output / "tasks.jsonl", task_rows)
                    status(completed_task_users=ordinal + 1,
                           completed_geometry_users=min(ordinal + 1, args.geometry_users))
                    if (ordinal + 1) in {1, 4, 16, 64, 128, 256, args.users}:
                        summarize(output)
        status(stage="summarize")
        report = summarize(output)
        if file_hash(checkpoint) != checkpoint_hash:
            raise ContractError("checkpoint changed during measurement")
        status(status="completed", stage="complete", report="report.json")
        print(canonical(dict(task_users=report["task_users"], baseline_ndcg_at_10=report["baseline_ndcg_at_10"])), flush=True)
    except BaseException as error:
        status(status="failed", error=str(error))
        (output / "error.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
