"""Reproduce a completed width panel and add tie-aware AUC without retraining.

Reports pooled ROC AUC and mean within-user AUC on the original fixed sampled
negatives, plus mean user AUC over the unseen catalog. Original run artifacts
are read-only. All uncertainty resamples users jointly across model variants.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
import time
import traceback

import torch
from torch.nn import functional as F

from .attention_width import balanced_groups, nmse, paired_interval, response_features, responses
from .core import ContractError, canonical, file_hash, read_json, write_json
from .kuairand_attention_width import capture, read_rows, replay_last


def within_user_auc(positive, negative, valid):
    """P(positive > negative) + 0.5 P(tie), for every variant."""
    if valid.sum() == 0:
        raise ValueError("at least one valid negative required")
    wins = (positive[:, None] > negative).double() + .5 * (positive[:, None] == negative).double()
    return (wins * valid).sum(-1) / valid.sum()


def roc_auc(positives, negatives):
    """Exact empirical binary ROC AUC, including ties; no sklearn dependency."""
    positives, negatives = positives.double().flatten(), negatives.double().flatten()
    if not len(positives) or not len(negatives):
        raise ValueError("both classes required")
    values = torch.cat((positives, negatives))
    labels = torch.cat((torch.ones(len(positives)), torch.zeros(len(negatives)))).double()
    if not torch.isfinite(values).all():
        raise ValueError("nonfinite score")
    order = values.argsort()
    _, inverse, counts = torch.unique_consecutive(values[order], return_inverse=True, return_counts=True)
    positive_count = torch.bincount(inverse, weights=labels[order], minlength=len(counts))
    negative_count = counts.double() - positive_count
    below = negative_count.cumsum(0) - negative_count
    wins = (positive_count * (below + .5 * negative_count)).sum()
    return float(wins / (len(positives) * len(negatives)))


def user_pair_wins(positives, negatives, valid, chunk=64):
    """G[u,v] sums wins of positive u against valid negatives belonging to v."""
    pieces = []
    for start in range(0, len(positives), chunk):
        p = positives[start:start + chunk, None, None]
        wins = (p > negatives[None]).double() + .5 * (p == negatives[None]).double()
        pieces.append((wins * valid[None]).sum(-1))
    return torch.cat(pieces)


def bootstrap_multiplicities(users, repeats=2000, seed=42):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(users, (repeats, users), generator=generator)
    counts = torch.zeros(repeats, users, dtype=torch.float64)
    return counts.scatter_add_(1, indices, torch.ones_like(indices, dtype=torch.float64))


def pooled_bootstrap(wins, valid_counts, counts):
    # Resampling users repeats their positive and their entire negative cluster.
    numerator = ((counts @ wins) * counts).sum(-1)
    denominator = counts.sum(-1) * (counts @ valid_counts.double())
    return numerator / denominator


@torch.inference_mode()
def score_catalog(contexts, items, ids, target, chunk=8192):
    positive = contexts @ items[target]
    seen = torch.zeros(len(items), dtype=torch.bool)
    seen[ids] = True
    repeated = bool(seen[target])
    greater = torch.zeros(len(contexts), dtype=torch.long)
    ties = torch.zeros_like(greater)
    ahead = torch.zeros_like(greater)
    count = 0
    for start in range(1, len(items), chunk):
        stop = min(start + chunk, len(items))
        candidates = torch.arange(start, stop)
        negative = contexts @ items[start:stop].T
        eligible = ~seen[start:stop] & candidates.ne(target)
        is_greater = negative > positive[:, None]
        is_tie = negative == positive[:, None]
        greater += (is_greater & eligible).sum(-1)
        ties += (is_tie & eligible).sum(-1)
        ahead += ((is_greater | (is_tie & (candidates < target))) & eligible).sum(-1)
        count += int(eligible.sum())
    auc = 1. - (greater.double() + .5 * ties.double()) / count
    ranks = ahead + 1
    ranks = torch.where((ranks > 2500) | repeated, 2501, ranks)
    return auc, ranks, repeated, count


def interval(values):
    return dict(low=float(values.quantile(.025)), high=float(values.quantile(.975)))


def summarize(output, scores, rows, budgets, seed):
    positive, negative, valid = scores["positive"], scores["negative"], scores["valid"]
    counts = bootstrap_multiplicities(len(valid), seed=seed)
    baseline_bootstrap = None
    baseline_pooled = None
    baseline_user = None
    baseline_catalog = None
    summary = []
    for i, budget in enumerate(budgets):
        pooled = roc_auc(positive[i], negative[i][valid])
        wins = user_pair_wins(positive[i], negative[i], valid)
        if abs(pooled - float(wins.sum() / (len(valid) * valid.sum()))) > 1e-12:
            raise ContractError("pooled AUC rank/count implementations disagree")
        boot = pooled_bootstrap(wins, valid.sum(-1), counts)
        user_auc = ((positive[i, :, None] > negative[i]).double() +
                    .5 * (positive[i, :, None] == negative[i]).double())
        user_auc = (user_auc * valid).sum(-1) / valid.sum(-1)
        catalog = torch.tensor([r["catalog_user_auc"] for r in rows
                                if r["budget"] == budget and r["catalog_user_auc"] is not None], dtype=torch.float64)
        if i == 0:
            baseline_bootstrap, baseline_pooled = boot, pooled
            baseline_user, baseline_catalog = user_auc, catalog
        delta_boot = boot - baseline_bootstrap
        summary.append(dict(budget=budget, method="baseline" if i == 0 else "response",
            pooled_sampled_auc=pooled, pooled_auc_delta=pooled - baseline_pooled,
            pooled_delta_ci95=interval(delta_boot),
            mean_user_sampled_auc=float(user_auc.mean()),
            mean_user_sampled_auc_delta=paired_interval((user_auc - baseline_user).tolist(), seed=seed),
            mean_user_catalog_auc=float(catalog.mean()),
            mean_user_catalog_auc_delta=paired_interval((catalog - baseline_catalog).tolist(), seed=seed),
            sampled_users=len(user_auc), catalog_users=len(catalog)))
    report = dict(kind="frozen_hstu_auc_followup", source_run=scores["source_run"],
        formal_result=False, training_partition_only=True, final_holdout_used=False,
        checkpoint_changed=False, users=len(valid), seed=seed, bootstrap_repeats=2000,
        tie_credit=.5, sampled_negative_policy="original seeded 128 uniform draws; positive collisions excluded",
        pooled_auc_policy="pool all positive and sampled-negative scores across users",
        mean_user_auc_policy="equal-user mean of within-user AUC; no cross-user score comparisons",
        catalog_policy="unseen catalog, original full-history filter; repeated-target users excluded from all variants",
        qualifications=["Synthetic next-item negatives, not observed nonclick/exposure labels or production CTR AUC.",
            "Diagnostic labels participated in baseline training; these are not new validation results.",
            "Intervention changes final-query aggregation only, with native causal prefix rows."],
        summary=summary)
    write_json(output / "report.json", report)
    csv_rows = [dict(budget=r["budget"], method=r["method"],
        pooled_auc=r["pooled_sampled_auc"], pooled_auc_delta=r["pooled_auc_delta"],
        delta_ci95_low=r["pooled_delta_ci95"]["low"], delta_ci95_high=r["pooled_delta_ci95"]["high"],
        mean_user_sampled_auc=r["mean_user_sampled_auc"],
        mean_user_catalog_auc=r["mean_user_catalog_auc"],
        sampled_users=r["sampled_users"], catalog_users=r["catalog_users"]) for r in summary]
    with (output / "auc-summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", default="artifacts/kuairand-research/attention-width/hstu-20261029-01")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    source, output = Path(args.source_run).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    state = dict(status="running", stage="verify", pid=os.getpid(), started_at=started, completed_users=0)
    def status(**updates):
        state.update(updates, updated_at=time.time(), wall_seconds=time.time() - started)
        write_json(output / "status.json", state)
        print(canonical(state), flush=True)
    status()
    try:
        contract = read_json(source / "contract.json")
        if read_json(source / "status.json")["status"] != "completed":
            raise ContractError("a completed source experiment is required")
        for name, expected in contract["source_hashes"].items():
            if file_hash(Path(__file__).with_name(name)) != expected:
                raise ContractError("source experiment implementation changed: " + name)
        original_args = contract["args"]
        benchmark = Path(original_args["benchmark"]).resolve()
        sys.path.insert(0, str(benchmark))
        from longseq_bench.models import BenchmarkModel, ModelConfig
        from longseq_bench.data import AlignedDataset, verify_cache
        baseline = Path(original_args["baseline"]).resolve()
        manifest = read_json(baseline / "run.json")
        for name, expected in manifest["code_files"].items():
            if file_hash(benchmark / "longseq_bench" / name) != expected:
                raise ContractError("pinned benchmark source changed: " + name)
        if verify_cache(original_args["cache"])["files"] != manifest["cache_files"]:
            raise ContractError("source cache identity differs")
        checkpoint = baseline / "last.pt"
        if file_hash(checkpoint) != contract["checkpoint_sha256"]:
            raise ContractError("source checkpoint identity differs")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["fingerprint"] != contract["checkpoint_fingerprint"]:
            raise ContractError("checkpoint fingerprint differs")
        cfg = ModelConfig(**manifest["model"])
        tc = manifest["training"]
        model = BenchmarkModel(cfg, tc["negatives"], tc["temperature"], tc["loss_chunk"]).eval()
        model.load_state_dict(saved["model"]); del saved
        dataset = AlignedDataset(original_args["cache"], "train")
        bank = torch.load(source / "query-bank.pt", map_location="cpu", weights_only=True)["calibration"]
        reference_rows = {(r["user_index"], r["budget"]): r for r in read_rows(source / "tasks.jsonl")
            if r["method"] == "baseline" or (r["method"] == "response" and r["scope"] == "all_layers_last_query")}
        panel = contract["panel_users"]
        budgets = [cfg.max_history] + original_args["budgets"]
        seed = original_args["seed"]
        batch_size = original_args["batch_size"]
        state["planned_users"] = len(panel)
        write_json(output / "contract.json", dict(source_run=str(source),
            source_contract_sha256=file_hash(source / "contract.json"),
            source_query_bank_sha256=file_hash(source / "query-bank.pt"),
            checkpoint_sha256=contract["checkpoint_sha256"],
            new_source_sha256=file_hash(__file__), threads=args.threads, device="cpu",
            budgets=budgets, sampled_negatives=tc["negatives"], seed=seed,
            bootstrap_seed=42, bootstrap_repeats=2000, tie_credit=.5,
            training_partition_only=True, final_holdout_used=False))
        items = F.normalize(model.encoder.item_embedding.weight.detach(), dim=-1, eps=1e-6)
        all_positive, all_negative, all_valid, rows = [], [], [], []
        max_loss_difference = 0.
        with torch.inference_mode():
            for left in range(0, len(panel), batch_size):
                indices = panel[left:left + batch_size]
                examples = [dataset[user] for user in indices]
                records, native, _ = capture(model, torch.stack([e["ids"] for e in examples]),
                                              torch.stack([e["times"] for e in examples]))
                for offset, (user, record, example) in enumerate(zip(indices, records, examples)):
                    assignments = {budget: {} for budget in original_args["budgets"]}
                    for li, layer in enumerate(model.encoder.layers):
                        snapshot = record["layers"][li]
                        for hi in range(cfg.heads):
                            key = snapshot["key"][:, hi]
                            h = responses(bank[li, :, hi], key, snapshot["bias"], scale=layer.bias.padded_length)
                            local_seed = seed + user * 100 + li * 4 + hi + cfg.max_history
                            groups = balanced_groups(response_features(h, local_seed), original_args["budgets"])
                            for budget, group in groups.items():
                                assignments[budget][(li, hi)] = group
                    contexts = torch.stack([native[offset]] + [replay_last(model, record, assignments[b])
                                                               for b in original_args["budgets"]])
                    target = int(example["target"])
                    negatives = torch.randint(1, cfg.num_items + 1, (tc["negatives"],),
                                              generator=torch.Generator().manual_seed(seed + user))
                    valid = negatives.ne(target)
                    positive_scores = contexts @ items[target]
                    negative_scores = contexts @ items[negatives].T
                    user_auc = within_user_auc(positive_scores, negative_scores, valid)
                    logits = torch.cat((positive_scores[:, None], negative_scores), -1) / tc["temperature"]
                    logits[:, 1:] = logits[:, 1:].masked_fill(~valid[None], -5e4)
                    losses = -logits.log_softmax(-1)[:, 0]
                    catalog_auc, ranks, repeated, count = score_catalog(contexts, items, example["ids"], target)
                    for i, budget in enumerate(budgets):
                        old = reference_rows[(user, budget)]
                        difference = abs(float(losses[i]) - old["sampled_loss"])
                        max_loss_difference = max(max_loss_difference, difference)
                        if difference > 1e-4 or int(ranks[i]) != old["rank"]:
                            raise ContractError("source replay mismatch at user %s budget %s: loss %s, rank %s/%s" %
                                                (user, budget, difference, int(ranks[i]), old["rank"]))
                        if abs(nmse(contexts[i], contexts[0]) - old["context_nmse"]) > 1e-6:
                            raise ContractError("source context mismatch")
                        rows.append(dict(user_index=user, budget=budget,
                            method="baseline" if i == 0 else "response", sampled_user_auc=float(user_auc[i]),
                            sampled_valid_negatives=int(valid.sum()),
                            catalog_user_auc=None if repeated else float(catalog_auc[i]),
                            catalog_valid_negatives=count, repeated_target=repeated,
                            reproduced_rank=int(ranks[i]), reproduced_sampled_loss=float(losses[i])))
                    all_positive.append(positive_scores.clone())
                    all_negative.append(negative_scores.clone())
                    all_valid.append(valid.clone())
                with (output / "users.jsonl").open("a") as handle:
                    for row in rows[-len(indices) * len(budgets):]:
                        handle.write(canonical(row) + "\n")
                status(stage="replay_and_score", completed_users=left + len(indices),
                       max_reproduced_loss_error=max_loss_difference)
        scores = dict(positive=torch.stack(all_positive, 1), negative=torch.stack(all_negative, 1),
                      valid=torch.stack(all_valid), budgets=budgets, users=panel, source_run=str(source))
        torch.save(scores, output / "sample-scores.pt")
        status(stage="paired_user_bootstrap")
        report = summarize(output, scores, rows, budgets, 42)
        if file_hash(checkpoint) != contract["checkpoint_sha256"]:
            raise ContractError("checkpoint changed during evaluation")
        status(status="completed", stage="complete", report="report.json")
        print(canonical(report["summary"]), flush=True)
    except BaseException as error:
        status(status="failed", error=str(error))
        (output / "error.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
