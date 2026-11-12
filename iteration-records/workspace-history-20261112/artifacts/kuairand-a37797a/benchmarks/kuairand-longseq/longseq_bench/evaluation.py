"""Exhaustive catalog evaluation, including the reference's MRR convention."""
import torch


@torch.inference_mode()
def evaluate(model, loader, device, num_items, candidate_chunk=256):
    model.eval()
    encoder = model.encoder
    totals = {k: 0.0 for k in ("ndcg@10", "ndcg@50", "hr@10", "hr@50", "mrr", "mrr_exact")}
    count, repeated = 0, 0
    for cpu_batch in loader:
        batch = {k: v.to(device) for k, v in cpu_batch.items()}
        ids, times, lengths, targets = batch["ids"], batch["times"], batch["length"], batch["target"]
        b = len(ids)
        rows = torch.arange(b, device=device)
        if model.cfg.name in {"dare", "ccformer"}:
            context = encoder.encode_all(ids, times)
            def score(candidates):
                return encoder.score_queries(context, ids, rows, lengths - 1, candidates)
            positive_scores = score(targets[:, None]).squeeze(-1).float()
        else:
            context = encoder.encode_current(ids, times, lengths)
            def score(candidates):
                items = torch.nn.functional.normalize(encoder.item_embedding(candidates), dim=-1, eps=1e-6)
                return torch.einsum("bd,bcd->bc", context, items)
            positive_scores = score(targets[:, None]).squeeze(-1).float()
        seen = torch.zeros(b, num_items + 1, dtype=torch.bool, device=device)
        seen.scatter_(1, ids, True)
        target_seen = seen[rows, targets]
        repeated += int(target_seen.sum())
        better = torch.zeros(b, dtype=torch.long, device=device)
        for start in range(1, num_items + 1, candidate_chunk):
            candidates = torch.arange(start, min(start + candidate_chunk, num_items + 1), device=device)
            matrix = candidates[None].expand(b, -1)
            logits = score(matrix).float()
            eligible = ~seen[:, candidates]
            # Stable ID order for exact ties; record this distinction from the
            # reference's backend-dependent torch.topk tie behavior.
            ahead = (logits > positive_scores[:, None]) | ((logits == positive_scores[:, None]) & (matrix < targets[:, None]))
            better += (ahead & eligible & matrix.ne(targets[:, None])).sum(dim=1)
        exact_ranks = better + 1
        ranks = torch.where(target_seen | (exact_ranks > 2500), 2501, exact_ranks)
        for k in (10, 50):
            hit = ranks <= k
            totals["hr@%d" % k] += float(hit.float().sum())
            totals["ndcg@%d" % k] += float((hit / torch.log2(ranks.float() + 1)).sum())
        totals["mrr"] += float((1.0 / ranks.float()).sum())
        totals["mrr_exact"] += float(torch.where(target_seen, 0.0, 1.0 / exact_ranks.float()).sum())
        count += b
    if not count:
        raise ValueError("No evaluation users")
    return {**{k: v / count for k, v in totals.items()}, "users": count,
            "repeated_target_filtered_users": repeated, "candidate_catalog_size": num_items,
            "evaluation": "exhaustive_after_seen_filter", "tie_policy": "ascending_item_id"}
