"""A800 latency/memory tradeoffs for the frozen HSTU response-merging probe.

Single-request tests include native prefix encoding, online calibration/grouping,
packing, last-query replay and scoring. Cached-read microbenchmarks explicitly
fix history, query anchor and time/position biases. They are not a streaming KV
cache, and cached final-user-vector scoring is reported as a separate baseline.
"""
from __future__ import annotations

import argparse
import csv
import gc
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import torch
from torch.nn import functional as F

from .attention_width import balanced_groups, nmse, response_features, responses
from .core import ContractError, canonical, file_hash, read_json, write_json


def tensor_bytes(value):
    """Count actual live tensor storages once, including views and metadata."""
    storages = {}
    def visit(x):
        if isinstance(x, torch.Tensor):
            storage = x.untyped_storage()
            storages[(str(x.device), storage.data_ptr())] = storage.nbytes()
        elif isinstance(x, dict):
            for child in x.values(): visit(child)
        elif isinstance(x, (tuple, list)):
            for child in x: visit(child)
    visit(value)
    return sum(storages.values())


@torch.inference_mode()
def native_memory(model, ids, times, retain_cache=True):
    """Original dense HSTU equations; project Q/K/V only once per layer.

    Prefix encoding remains full length and unmodified. Caching adds materialized
    K/V copies, but does not re-run the native encoder or duplicate projections.
    """
    cfg, encoder = model.cfg, model.encoder
    if cfg.name != "hstu" or ids.shape[0] != 1:
        raise ContractError("dense HSTU, batch one, full histories required")
    x = encoder.initial(ids)
    initial = x[0, -1].clone()
    length = ids.shape[1]
    positions = torch.arange(length, device=ids.device)
    layers = []
    for layer in encoder.layers:
        p = F.silu(F.layer_norm(x, (cfg.dim,), eps=1e-6) @ layer.uvqk)
        u, v, q, k = p.split([cfg.heads * cfg.dv, cfg.heads * cfg.dv,
                              cfg.heads * cfg.dqk, cfg.heads * cfg.dqk], -1)
        q = q.reshape(1, length, cfg.heads, cfg.dqk)
        k = k.reshape(1, length, cfg.heads, cfg.dqk)
        v = v.reshape(1, length, cfg.heads, cfg.dv)
        pieces = []
        for left in range(0, length, cfg.query_chunk):
            right = min(left + cfg.query_chunk, length)
            qp = positions[left:right]
            allowed = positions[None] <= qp[:, None]
            # Match the original tile's union of visible keys (including scale).
            end = right
            logits = torch.einsum("bqhd,bkhd->bhqk", q[:, left:right], k[:, :end])
            logits = logits + layer.bias(times, qp, positions[:end]).unsqueeze(1)
            weights = F.silu(logits) / layer.bias.padded_length
            weights = weights * allowed[:, :end][None, None]
            attended = torch.einsum("bhqk,bkhd->bqhd", weights, v[:, :end])
            pieces.append(attended.reshape(1, right - left, cfg.heads * cfg.dv))
        attended = torch.cat(pieces, 1)
        gated = u * F.layer_norm(attended, (cfg.heads * cfg.dv,), eps=1e-6)
        x = x + layer.output(gated)
        if retain_cache:
            key = k[0].transpose(0, 1).contiguous()
            value = v[0].transpose(0, 1).contiguous()
            bias = layer.bias(times, positions[-1:], positions)[0, 0].clone()
            layers.append(dict(key=key, value=value, bias=bias[None],
                old_key=key[:, -1], old_value=value[:, -1],
                self_group=torch.full((cfg.heads,), length - 1, dtype=torch.long, device=ids.device),
                group_size=1, scale=layer.bias.padded_length))
    return dict(initial=initial, layers=layers,
                native_context=F.normalize(x[0, -1], dim=-1, eps=1e-6))


@torch.inference_mode()
def compress_memory(memory, bank, budget, user_seed):
    """Charge all response probing, feature projection, grouping and packing."""
    packed = []
    for li, saved in enumerate(memory["layers"]):
        keys, values, offsets, self_groups = [], [], [], []
        length = saved["key"].shape[1]
        for hi in range(saved["key"].shape[0]):
            key, value, bias = saved["key"][hi], saved["value"][hi], saved["bias"][0]
            h = responses(bank[li, :, hi], key, bias, scale=saved["scale"])
            seed = user_seed + li * 4 + hi + length
            groups = balanced_groups(response_features(h, seed), [budget])[budget]
            keys.append(key[groups].mean(1))
            values.append(value[groups].sum(1))
            offsets.append(bias[groups].mean(1))
            self_groups.append((groups == length - 1).any(-1).long().argmax())
        packed.append(dict(key=torch.stack(keys), value=torch.stack(values), bias=torch.stack(offsets),
            old_key=saved["old_key"].clone(), old_value=saved["old_value"].clone(),
            self_group=torch.stack(self_groups), group_size=length // budget, scale=saved["scale"]))
    return dict(initial=memory["initial"].clone(), layers=packed,
                native_context=memory["native_context"].clone())


@torch.inference_mode()
def read_memory(model, memory, initial=None):
    """Batched-head cached read with exact correction of the evolving last token.

    Only the self token changes through replay. Correct its group's logit and
    weighted value algebraically, without cloning the full K/V cache per read.
    """
    cfg = model.cfg
    x = memory["initial"] if initial is None else initial
    for layer, saved in zip(model.encoder.layers, memory["layers"]):
        p = F.silu(F.layer_norm(x, (cfg.dim,), eps=1e-6) @ layer.uvqk)
        u, value, query, key = p.split([cfg.heads * cfg.dv, cfg.heads * cfg.dv,
                                      cfg.heads * cfg.dqk, cfg.heads * cfg.dqk], -1)
        query = query.reshape(cfg.heads, cfg.dqk)
        key = key.reshape(cfg.heads, cfg.dqk)
        value = value.reshape(cfg.heads, cfg.dv)
        logits = torch.bmm(saved["key"], query[:, :, None]).squeeze(-1) + saved["bias"]
        correction = (query * (key - saved["old_key"])).sum(-1) / saved["group_size"]
        group = saved["self_group"][:, None]
        logits.scatter_add_(1, group, correction[:, None])
        weights = F.silu(logits) / saved["scale"]
        attended = torch.bmm(weights[:, None], saved["value"]).squeeze(1)
        attended = attended + weights.gather(1, group) * (value - saved["old_value"])
        attended = attended.flatten()
        gated = u * F.layer_norm(attended, (cfg.heads * cfg.dv,), eps=1e-6)
        x = x + layer.output(gated)
    return F.normalize(x, dim=-1, eps=1e-6)


@torch.inference_mode()
def top_items(context, items, candidates, seen):
    # The only full-catalog candidate set passed here is arange(1, num_items+1).
    # Use a view, avoiding an unnecessary copy of the entire embedding table.
    if candidates.numel() == len(items) - 1:
        scores = items[1:] @ context
        scores = scores.masked_fill(seen[1:], -torch.inf)
    else:
        scores = items[candidates] @ context
        scores = scores.masked_fill(seen[candidates], -torch.inf)
    return scores.topk(min(10, len(scores))).indices


def sample_time(function, warmup=3, repeats=10):
    """Synchronized wall time includes Python, CPU RNG, copies and CUDA work."""
    for _ in range(warmup):
        result = function()
        del result
    torch.cuda.synchronize()
    rows = []
    for _ in range(repeats):
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        result = function()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        rows.append(dict(milliseconds=elapsed * 1000,
                         extra_peak_allocated_bytes=torch.cuda.max_memory_allocated() - before,
                         absolute_peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        del result
    return rows


def stats(rows):
    values = torch.tensor([r["milliseconds"] for r in rows], dtype=torch.float64)
    return dict(samples=len(rows), p50_ms=float(values.quantile(.5)), p95_ms=float(values.quantile(.95)),
                mean_ms=float(values.mean()), serial_batch_one_requests_per_second=1000 / float(values.mean()),
                peak_extra_mib=max(r["extra_peak_allocated_bytes"] for r in rows) / 2**20,
                peak_absolute_mib=max(r["absolute_peak_allocated_bytes"] for r in rows) / 2**20)


def append_rows(path, rows):
    with path.open("a") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def summarize(output, rows, cache_rows, checks):
    from collections import defaultdict
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["scenario"], row["budget"], row["candidates"])].append(row)
    summary = [dict(scenario=key[0], budget=key[1], candidates=key[2], **stats(value))
               for key, value in sorted(grouped.items())]
    read_rows = {r["budget"]: r for r in summary if r["scenario"] == "cached_read_only"}
    merge_rows = {r["budget"]: r for r in summary if r["scenario"] == "merge_existing_kv"}
    raw_budget = max(read_rows) if read_rows else None
    amortization = []
    if raw_budget is not None:
        for budget, merged in sorted(merge_rows.items()):
            saved_ms = read_rows[raw_budget]["p50_ms"] - read_rows[budget]["p50_ms"]
            amortization.append(dict(budget=budget, read_time_saved_ms=saved_ms,
                merge_existing_kv_p50_ms=merged["p50_ms"],
                estimated_break_even_reads=math.floor(merged["p50_ms"] / saved_ms) + 1 if saved_ms > 0 else None))
    report = dict(kind="frozen_hstu_latency_memory_tradeoff", formal_result=False,
        training_partition_only=True, final_holdout_used=False, training_modified=False,
        measured_device="single A800, FP32, batch one", measured_timing="synchronized wall time",
        rows=summary, resident_cache=cache_rows, correctness_checks=checks,
        amortization_estimates=amortization,
        qualifications=["Warm model weights, calibration library, normalized catalog and device inputs are resident before all timing.",
            "Single-request merged timing includes full prefix encoding, calibration responses, random projections, grouping, packing, replay and top-10 scoring.",
            "Cached-read timing fixes prefix, query position and time bias. Streaming/sliding-window cache updates are not implemented.",
            "Cached final user vector is a valid alternative when history and query are unchanged; report it separately.",
            "The equivalent-prefix baseline uses the same dense prefix equations as the merged path without caching or grouping; native output equivalence is checked.",
            "Throughput is serial batch-one throughput, not a concurrency or maximum batch throughput test.",
            "The implementation is portable PyTorch eager code, without custom fused kernels."])
    write_json(output / "report.json", report)
    if summary:
        with (output / "timing-summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader(); writer.writerows(summary)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", default="artifacts/kuairand-research/attention-width/hstu-20261029-01")
    parser.add_argument("--output", required=True)
    parser.add_argument("--users", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--read-repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--budgets", default="128,256,512")
    args = parser.parse_args()
    budgets = [int(v) for v in args.budgets.split(",")]
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    output, source = Path(args.output).resolve(), Path(args.source_run).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    state = dict(status="running", stage="verify", pid=os.getpid(), started_at=started, completed_users=0)
    def status(**updates):
        state.update(updates, updated_at=time.time(), wall_seconds=time.time() - started)
        write_json(output / "status.json", state)
        print(canonical(state), flush=True)
    status()
    try:
        if not torch.cuda.is_available():
            raise ContractError("CUDA device required for this performance test")
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        contract = read_json(source / "contract.json")
        if read_json(source / "status.json")["status"] != "completed":
            raise ContractError("source run is incomplete")
        for name, expected in contract["source_hashes"].items():
            if file_hash(Path(__file__).with_name(name)) != expected:
                raise ContractError("source implementation changed: " + name)
        old_args = contract["args"]
        benchmark = Path(old_args["benchmark"]).resolve()
        sys.path.insert(0, str(benchmark))
        from longseq_bench.models import BenchmarkModel, ModelConfig
        from longseq_bench.data import AlignedDataset, verify_cache
        baseline = Path(old_args["baseline"]).resolve()
        manifest = read_json(baseline / "run.json")
        if verify_cache(old_args["cache"])["files"] != manifest["cache_files"]:
            raise ContractError("cache identity differs")
        for name, expected in manifest["code_files"].items():
            if file_hash(benchmark / "longseq_bench" / name) != expected:
                raise ContractError("pinned benchmark source changed: " + name)
        checkpoint = baseline / "last.pt"
        if file_hash(checkpoint) != contract["checkpoint_sha256"]:
            raise ContractError("checkpoint identity differs")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg, tc = ModelConfig(**manifest["model"]), manifest["training"]
        model = BenchmarkModel(cfg, tc["negatives"], tc["temperature"], tc["loss_chunk"]).eval()
        model.load_state_dict(saved["model"]); del saved
        cpu_model = model
        import copy
        model = copy.deepcopy(cpu_model).cuda().eval()
        items = F.normalize(model.encoder.item_embedding.weight.detach(), dim=-1, eps=1e-6)
        bank_cpu = torch.load(source / "query-bank.pt", weights_only=True)["calibration"]
        bank = bank_cpu.cuda()
        dataset = AlignedDataset(old_args["cache"], "train")
        panel = contract["panel_users"][:args.users]
        gpu_snapshot = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used", "--format=csv,noheader"], text=True)
        write_json(output / "contract.json", dict(source_run=str(source), source_contract_sha256=file_hash(source / "contract.json"),
            checkpoint_sha256=contract["checkpoint_sha256"], query_bank_sha256=file_hash(source / "query-bank.pt"),
            implementation_sha256=file_hash(__file__), args=vars(args), users=panel,
            gpu_name=torch.cuda.get_device_name(0), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            gpu_snapshot=gpu_snapshot, torch=torch.__version__, tf32=False, batch_size=1,
            timing_excludes="disk/model loading, donor library creation, input transfer, static item normalization",
            timing_includes="per-user random projection CPU RNG/copy, all calibration responses, clustering, packing and scoring"))
        rows, cache_rows, checks = [], [], []
        for ordinal, user in enumerate(panel):
            status(stage="measure", active_user=ordinal)
            example = dataset[user]
            ids, times = example["ids"][None].cuda(), example["times"][None].cuda()
            lengths = torch.tensor([cfg.max_history], device="cuda")
            seen = torch.zeros(cfg.num_items + 1, dtype=torch.bool, device="cuda")
            seen[ids] = True
            candidate_sets = {128: torch.randint(1, cfg.num_items + 1, (128,), device="cuda",
                            generator=torch.Generator(device="cuda").manual_seed(user)),
                              cfg.num_items: torch.arange(1, cfg.num_items + 1, device="cuda")}
            user_seed = old_args["seed"] + user * 100
            with torch.inference_mode():
                reference = model.encoder.encode_current(ids, times, lengths)[0]
                raw = native_memory(model, ids, times)
                check = dict(user_index=user, native_adapter_nmse=nmse(raw["native_context"], reference),
                             uncompressed_read_nmse=nmse(read_memory(model, raw), reference))
                if max(check["native_adapter_nmse"], check["uncompressed_read_nmse"]) > 1e-9:
                    raise ContractError("native adapter or cached replay mismatch")
                # Confirm prototype against the earlier CPU intervention on real users.
                cpu_records = None
                if ordinal < 4:
                    from .kuairand_attention_width import capture, replay_last
                    cpu_records, _, _ = capture(cpu_model, example["ids"][None], example["times"][None])
                del raw
                gc.collect(); torch.cuda.synchronize()
                for budget in [cfg.max_history] + budgets:
                    resident_before = torch.cuda.memory_allocated()
                    raw = native_memory(model, ids, times)
                    if budget == cfg.max_history:
                        memory = raw
                    else:
                        timed = sample_time(lambda: compress_memory(raw, bank, budget, user_seed), args.warmup, args.repeats)
                        common = dict(user_index=user, scenario="merge_existing_kv", budget=budget, candidates=0)
                        rows.extend([{**common, **r} for r in timed])
                        memory = compress_memory(raw, bank, budget, user_seed)
                        if cpu_records is not None:
                            assignments = {}
                            for li, layer in enumerate(cpu_model.encoder.layers):
                                layer_record = cpu_records[0]["layers"][li]
                                for hi in range(cfg.heads):
                                    h = responses(bank_cpu[li, :, hi], layer_record["key"][:, hi], layer_record["bias"],
                                                  scale=layer.bias.padded_length)
                                    seed = user_seed + li * 4 + hi + cfg.max_history
                                    groups = balanced_groups(response_features(h, seed), [budget])[budget]
                                    assignments[(li, hi)] = groups
                            cpu_output = replay_last(cpu_model, cpu_records[0], assignments)
                            check["compressed_%d_vs_cpu_nmse" % budget] = nmse(read_memory(model, memory).cpu(), cpu_output)
                    del raw
                    gc.collect(); torch.cuda.synchronize()
                    cache_rows.append(dict(user_index=user, budget=budget, storage_bytes=tensor_bytes(memory),
                        storage_mib=tensor_bytes(memory) / 2**20,
                        resident_cuda_mib=(torch.cuda.memory_allocated() - resident_before) / 2**20,
                        final_user_vector_bytes=reference.numel() * reference.element_size()))
                    # Warm cache + repeated reads: exclude one-time preparation explicitly.
                    timed = sample_time(lambda: read_memory(model, memory), args.warmup, args.read_repeats)
                    common = dict(user_index=user, scenario="cached_read_only", budget=budget, candidates=0)
                    rows.extend([{**common, **r} for r in timed])
                    for count, candidates in candidate_sets.items():
                        timed = sample_time(lambda: top_items(read_memory(model, memory), items, candidates, seen),
                                            args.warmup, args.repeats)
                        common = dict(user_index=user, scenario="cached_read_and_score", budget=budget, candidates=count)
                        rows.extend([{**common, **r} for r in timed])
                    del memory
                    gc.collect(); torch.cuda.synchronize()
                gc.collect(); torch.cuda.synchronize()
                # No raw or compressed per-user cache remains during cold-request timing.
                # All paths start from the same resident model, input, bank and item table.
                for count, candidates in candidate_sets.items():
                    timed = sample_time(lambda: top_items(model.encoder.encode_current(ids, times, lengths)[0],
                                                         items, candidates, seen), args.warmup, args.repeats)
                    common = dict(user_index=user, scenario="single_request", budget=cfg.max_history, candidates=count)
                    rows.extend([{**common, **r} for r in timed])
                    timed = sample_time(lambda: top_items(native_memory(model, ids, times, retain_cache=False)["native_context"],
                                                         items, candidates, seen), args.warmup, args.repeats)
                    common = dict(user_index=user, scenario="equivalent_prefix_base", budget=cfg.max_history, candidates=count)
                    rows.extend([{**common, **r} for r in timed])
                    timed = sample_time(lambda: top_items(reference, items, candidates, seen), args.warmup, args.read_repeats)
                    common = dict(user_index=user, scenario="cached_user_vector_and_score", budget=0, candidates=count)
                    rows.extend([{**common, **r} for r in timed])
                    for budget in budgets:
                        def request():
                            original_memory = native_memory(model, ids, times)
                            packed = compress_memory(original_memory, bank, budget, user_seed)
                            del original_memory
                            return top_items(read_memory(model, packed), items, candidates, seen)
                        timed = sample_time(request, args.warmup, args.repeats)
                        common = dict(user_index=user, scenario="single_request", budget=budget, candidates=count)
                        rows.extend([{**common, **r} for r in timed])
            checks.append(check)
            append_rows(output / "checks.jsonl", [check])
            append_rows(output / "timings.jsonl", [r for r in rows if r["user_index"] == user])
            summarize(output, rows, cache_rows, checks)
            status(completed_users=ordinal + 1)
        if file_hash(checkpoint) != contract["checkpoint_sha256"]:
            raise ContractError("checkpoint changed during measurement")
        status(status="completed", stage="complete", report="report.json")
    except BaseException as error:
        status(status="failed", error=str(error))
        (output / "error.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
