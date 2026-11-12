"""Linux/A800 training entry point; --smoke uses small synthetic CPU/CUDA data."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .data import AlignedDataset, make_cache, sha256, verify_cache
from .checkpoints import recover_metrics
from .download import atomic_json
from .evaluation import evaluate
from .models import BenchmarkModel, ModelConfig
from .registry import MODEL_NAMES, MODEL_LABELS


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def smoke_input(output, seed=42):
    source = output / "synthetic.csv"
    rng = random.Random(seed)
    # Cover the entire tiny vocabulary while allowing repeat-item holdouts.
    with source.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id", "sequence_item_ids", "sequence_ratings", "sequence_timestamps"])
        for uid in range(12):
            values = [(uid + i) % 24 for i in range(8)] + [rng.randrange(24) for _ in range(10)]
            writer.writerow([uid, ",".join(map(str, values)), ",".join(["4"] * len(values)),
                             ",".join(str(1000 * i) for i in range(len(values)))])
    cache = output / "synthetic-cache"
    make_cache(source, cache, max_history=16)
    return cache


def prepare_smoke_cache(output, resume, rank, world):
    cache = output / "synthetic-cache"
    if not resume and rank == 0:
        smoke_input(output)
    if world > 1:
        dist.barrier()  # Other ranks must not write or read a partial shared cache.
    return cache


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def atomic_save(value, path):
    temporary = path.with_suffix(".temporary")
    torch.save(value, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/benchmark.json")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--precision", choices=["fp32", "bf16"])
    parser.add_argument("--allow-data-variant", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.smoke and world > 1 and args.device == "cpu":
        parser.error("Distributed synthetic smoke uses CUDA/NCCL; CPU smoke uses one process")
    if args.device == "cpu" and not args.smoke:
        parser.error("Real training requires CUDA; CPU is reserved for --smoke")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            parser.error("NVIDIA CUDA unavailable; use --device cpu --smoke for synthetic checks")
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank) if args.device == "cuda" else torch.device("cpu")
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=24))
    cfg_doc = json.loads(args.config.read_text())
    train_cfg = dict(cfg_doc["training"])
    model_cfg = ModelConfig(name=args.model, **cfg_doc["model"])
    precision = args.precision or train_cfg["precision"]
    if precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        parser.error("BF16 training requires a supporting CUDA device")
    args.output = args.output.resolve()
    if args.resume and args.resume.resolve().parent != args.output:
        parser.error("Resume into the original run's output directory")
    if rank == 0:
        if args.output.exists() and not args.resume:
            parser.error("Output directory exists; use a new directory or --resume")
        args.output.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    if args.smoke:
        model_cfg = ModelConfig(name=args.model, num_items=24, max_history=16, dim=8,
                                blocks=4, heads=2, dqk=4, dv=8, input_dropout=0,
                                linear_dropout=0, query_chunk=4, sla_local=4,
                                sla_global=2, truncate_keep=8, truncate_after=3,
                                dare_top_k=3, checkpoint_blocks=True,
                                cc_group_size=4, cc_channel_size=2)
        train_cfg.update(epochs=1, local_batch_size=3, global_batch_size=6,
                         negatives=4, loss_chunk=8)
        args.cache = prepare_smoke_cache(args.output, args.resume, rank, world)
        torch.set_num_threads(1)
    if args.cache is None:
        parser.error("--cache is required for real training")
    cache_meta = verify_cache(args.cache)
    if cache_meta["max_history"] != model_cfg.max_history:
        parser.error("Cache history length differs from model configuration")
    if not args.smoke and not cache_meta["paper_statistics_match"] and not args.allow_data_variant:
        parser.error("Data statistics differ from the paper; resolve them or explicitly label with --allow-data-variant")
    if cache_meta["statistics"]["observed_items"] > model_cfg.num_items:
        parser.error("Observed item IDs exceed configured catalog")
    denominator = world * train_cfg["local_batch_size"]
    if train_cfg["global_batch_size"] % denominator:
        parser.error("Global batch must be divisible by world size times local batch")
    accumulation = train_cfg["global_batch_size"] // denominator
    if accumulation < 1:
        parser.error("More ranks than supported by the frozen global batch")
    seed_everything(train_cfg["seed"])
    model = BenchmarkModel(model_cfg, train_cfg["negatives"], train_cfg["temperature"], train_cfg["loss_chunk"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"],
                                  betas=tuple(train_cfg["betas"]), weight_decay=train_cfg["weight_decay"])
    run = {
        "run_kind": "synthetic_smoke" if args.smoke else "kuairand27k_benchmark",
        "model_label": MODEL_LABELS[args.model],
        "model": asdict(model_cfg), "training": train_cfg, "precision": precision,
        "world_size": world, "gradient_accumulation": accumulation,
        "source_csv_sha256": cache_meta["source_csv_sha256"],
        "cache_files": cache_meta["files"], "paper_statistics_match": cache_meta["paper_statistics_match"],
        "data_statistics": cache_meta["statistics"],
        "preprocessing_manifest": cache_meta["preprocessing_manifest"],
        "code_files": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        "evaluation": cfg_doc["evaluation"], "adaptations": cfg_doc.get("adaptations", {}),
        "parameters": sum(p.numel() for p in model.parameters()),
        "architecture": {"kind": "target_conditioned_dare", "hidden": [200, 80], "top_k": model_cfg.dare_top_k}
            if args.model == "dare" else {"kind": "ccformer_causal_id_next_item", "blocks": model_cfg.blocks,
                "heads": model_cfg.heads, "user_tokens": ["prefix_mean", "latest_history"],
                "group_size": model_cfg.cc_group_size, "channel_size": model_cfg.cc_channel_size,
                "mixing_expansion": model_cfg.cc_mixing_expansion, "kernel": model_cfg.cc_kernel_size,
                "stride": model_cfg.cc_stride, "time_unit_ms": model_cfg.cc_time_unit_ms,
                "causal_mixing": True, "causal_compression": True, "candidate_conditioned": True}
            if args.model == "ccformer" else {"kind": args.model, "blocks": model_cfg.blocks,
                                        "heads": model_cfg.heads, "dqk": model_cfg.dqk, "dv": model_cfg.dv},
        "non_embedding_parameters": sum(p.numel() for name, p in model.named_parameters() if "embedding" not in name),
        "runtime": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
                    "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"},
    }
    run["runtime"]["cuda_matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
    run["runtime"]["cudnn_allow_tf32"] = torch.backends.cudnn.allow_tf32
    fingerprint = hashlib.sha256(json.dumps(run, sort_keys=True).encode()).hexdigest()
    start_epoch = 0
    resumed_rng = None
    resumed_metrics = None
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved["fingerprint"] != fingerprint:
            raise ValueError("Resume config/data/code/runtime differs from saved run")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = saved["next_epoch"]
        resumed_rng = saved["rng_states"][rank]
        resumed_metrics = saved.get("last_metrics")
    if rank == 0:
        manifest = args.output / "run.json"
        if manifest.exists() and json.loads(manifest.read_text())["fingerprint"] != fingerprint:
            raise ValueError("Output directory belongs to a different run")
        atomic_json(manifest, {**run, "fingerprint": fingerprint})
        if args.resume:
            recover_metrics(args.output, fingerprint, start_epoch, resumed_metrics)
    wrapped = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False) if world > 1 else model
    train_data, eval_data = AlignedDataset(args.cache, "train"), AlignedDataset(args.cache, "eval")
    sampler = DistributedSampler(train_data, num_replicas=world, rank=rank, shuffle=True, seed=0, drop_last=world > 1)
    loader = DataLoader(train_data, batch_size=train_cfg["local_batch_size"], sampler=sampler, num_workers=0)
    eval_loader = DataLoader(eval_data, batch_size=cfg_doc["evaluation"]["batch_size"], shuffle=False, num_workers=0)
    if resumed_rng is not None:
        restore_rng(resumed_rng)
    completed = None
    for epoch in range(start_epoch, train_cfg["epochs"]):
        sampler.set_epoch(epoch)
        wrapped.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_started = time.perf_counter()
        loss_sum, batches = 0.0, 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for i, cpu_batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in cpu_batch.items()}
            group_start = (i // accumulation) * accumulation
            group_size = min(accumulation, len(loader) - group_start)
            synchronize = (i + 1) % accumulation == 0 or i + 1 == len(loader)
            communication = wrapped.no_sync() if world > 1 and not synchronize else nullcontext()
            with communication:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
                    loss = wrapped(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                (loss / group_size).backward()
            loss_sum += float(loss.detach())
            batches += 1
            if synchronize:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - epoch_started
        local_cost = {"rank": rank, "seconds": elapsed,
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                      "mean_batch_loss": loss_sum / batches}
        costs = [None] * world
        if world > 1:
            dist.all_gather_object(costs, local_cost)
        else:
            costs = [local_cost]
        if rank == 0:
            eval_started = time.perf_counter()
            metrics = evaluate(model, eval_loader, device, model_cfg.num_items,
                               cfg_doc["evaluation"]["candidate_chunk"])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            completed = {"epoch": epoch, "metrics": metrics, "rank_training_costs": costs,
                         "evaluation_seconds": time.perf_counter() - eval_started,
                         "run_kind": run["run_kind"], "fingerprint": fingerprint}
            print(json.dumps(completed), flush=True)
        # Evaluation is deterministic but DataLoader consumes a CPU RNG seed.
        # Save each rank's actual state only after evaluation, at an epoch boundary.
        states = [None] * world
        if world > 1:
            dist.all_gather_object(states, rng_state())
        else:
            states = [rng_state()]
        if rank == 0:
            atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "rng_states": states, "next_epoch": epoch + 1,
                         "last_metrics": completed, "fingerprint": fingerprint}, args.output / "last.pt")
            with (args.output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(completed) + "\n")
            atomic_json(args.output / "latest.json", completed)
        if world > 1:
            dist.barrier()
    if rank == 0:
        # A completed resume does not train/evaluate again or append duplicate metrics.
        if completed is None:
            completed = json.loads((args.output / "latest.json").read_text())
        atomic_json(args.output / "final.json", {**completed, "selection": "fixed_final_epoch",
                                               "training_complete": True})
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
