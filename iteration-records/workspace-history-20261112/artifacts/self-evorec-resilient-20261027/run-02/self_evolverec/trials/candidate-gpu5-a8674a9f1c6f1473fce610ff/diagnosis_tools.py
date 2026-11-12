import json
import random
from collections import defaultdict
import numpy as np
import torch

class DiagnosisProbe:
    def __init__(self, view, device):
        self.view = view
        self.model = view.model
        self.device = torch.device(device)

    def _full_catalog_recommendations(self):
        # Five actual training users per category; never substitute pseudo-users.
        histories = self.view.user_train
        histories = histories.get("History", histories)
        users = sorted(
            (user for user, history in histories.items() if len(history)),
            key=lambda user: str(user),
        )[:5]
        model = self.model
        recommendations = []
        for user in users:
            history = [int(item) for item in histories[user]]
            if any(item < 1 or item > model.itemnum for item in history):
                raise ValueError("Recommendation history contains an invalid item ID")
            seen = set(history)
            sequence = torch.zeros(
                (1, model.maxlen), dtype=torch.long, device=self.device
            )
            suffix = history[-model.maxlen :]
            sequence[0, -len(suffix) :] = torch.tensor(suffix, device=self.device)
            features = model.final_features(sequence)
            best = []
            for start in range(1, model.itemnum + 1, 1024):
                ids = [
                    item
                    for item in range(start, min(start + 1024, model.itemnum + 1))
                    if item not in seen
                ]
                if not ids:
                    continue
                candidates = torch.tensor([ids], dtype=torch.long, device=self.device)
                scores = model.predict_score(features, candidates)[0]
                if not bool(torch.isfinite(scores).all()):
                    raise ValueError("Nonfinite full-catalog recommendation scores")
                best.extend(zip(scores.float().cpu().tolist(), ids))
                # Deterministic tie handling, also at chunk boundaries.
                best = sorted(best, key=lambda pair: (-pair[0], pair[1]))[:10]
            recommendations.append(
                {
                    "user": str(user),
                    "items": [item for _, item in best],
                }
            )
        return recommendations

    def run_full_diagnosis(self, train_loader):
        import itertools

        model = self.model
        flags = [(module, module.training) for module in model.modules()]
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        cpu_rng = torch.get_rng_state()
        cuda_rng = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        )
        samples = defaultdict(list)
        definitions = {
            "margin_mean": "Mean complete positive-minus-negative score on sampled training positions.",
            "margin_p10": "Linear-interpolated tenth percentile of complete training margins.",
            "negative_beats_positive_rate": "Fraction of sampled margins strictly below zero.",
            "small_margin_rate": "Fraction of sampled margins below 0.5; score-scale dependent.",
            "gate": "Mean recent-branch scalar mixture weight.",
            "full_raw_norm": "Mean L2 norm of unnormalized full branch outputs.",
            "recent_raw_norm": "Mean L2 norm of unnormalized recent branch outputs.",
            "full_contribution_norm": "Mean L2 norm of (1-g) times normalized full features; raw features for baseline mode.",
            "recent_contribution_norm": "Mean L2 norm of g times normalized recent features.",
            "branch_cosine": "Mean full/recent cosine similarity; does not establish intent or compatibility.",
            "embedding_cosine_mean": "Mean cosine of seed-42 sampled distinct nonpadding item pairs; alone not a collapse measure.",
            "sampled_positions": "Measured positions from at most four batches and 256 endpoints per batch.",
        }
        try:
            model.eval()
            with torch.no_grad():
                for _, sequence, positive, negative in itertools.islice(
                    train_loader, 4
                ):
                    sequence, positive, negative = [
                        torch.as_tensor(x, dtype=torch.long, device=self.device)
                        for x in (sequence, positive, negative)
                    ]
                    endpoints = (
                        sequence.ne(0) & positive.ne(0) & negative.ne(0)
                    ).nonzero()
                    if endpoints.shape[0] > 256:
                        indices = torch.linspace(
                            0, endpoints.shape[0] - 1, 256, device=self.device
                        ).long()
                        endpoints = endpoints[indices]
                    if not endpoints.numel():
                        continue
                    b, t = endpoints.unbind(1)
                    full = model._encode(sequence)[b, t]
                    windows, lengths = model.recent_windows(sequence, endpoints)
                    recent = model._encode(windows)[:, -1]
                    fused, gate = model.fuse(full, recent, lengths)
                    margin = model.predict_score(
                        fused, positive[b, t]
                    ) - model.predict_score(fused, negative[b, t])
                    z_full = (
                        full
                        if model.fusion_mode == "baseline"
                        else model._normalize(full)
                    )
                    values = {
                        "margins": margin,
                        "gate": gate.flatten(),
                        "full_raw_norm": full.norm(dim=-1),
                        "recent_raw_norm": recent.norm(dim=-1),
                        "full_contribution_norm": ((1 - gate) * z_full).norm(dim=-1),
                        "recent_contribution_norm": (
                            gate * model._normalize(recent)
                        ).norm(dim=-1),
                        "branch_cosine": torch.nn.functional.cosine_similarity(
                            full, recent, dim=-1
                        ),
                    }
                    for name, value in values.items():
                        samples[name].append(value.detach().cpu())
                    for name, mask in {
                        "prefix_1_to_10": lengths <= 10,
                        "prefix_11_to_30": (lengths > 10) & (lengths <= 30),
                        "prefix_over_30": lengths > 30,
                    }.items():
                        samples[name].append(margin[mask].detach().cpu())
                metrics = {}
                unavailable = {
                    "item_categories": "Category metadata is unavailable; no item-category breakdown is inferred.",
                    "review_preferences": "Review ratings/text are unavailable.",
                    "simulator_traits": "Diversity and conformity traits are unobserved.",
                }
                for name, parts in samples.items():
                    values = torch.cat(parts).float()
                    if name == "margins":
                        metrics["sampled_positions"] = values.numel()
                        if values.numel():
                            metrics.update(
                                {
                                    "margin_mean": float(values.mean()),
                                    "margin_p10": float(torch.quantile(values, 0.1)),
                                    "negative_beats_positive_rate": float(
                                        (values < 0).float().mean()
                                    ),
                                    "small_margin_rate": float(
                                        (values < 0.5).float().mean()
                                    ),
                                }
                            )
                        continue
                    definitions[name] = definitions.get(
                        name,
                        "Mean complete training margin in the named observed-prefix-length stratum.",
                    )
                    definitions[name + "_count"] = (
                        "Number of measured observations for " + name
                    )
                    metrics[name + "_count"] = values.numel()
                    if values.numel():
                        metrics[name] = float(values.mean())
                    else:
                        unavailable[name] = "No sampled positions in this stratum."
                metrics.setdefault("sampled_positions", 0)
                if not metrics["sampled_positions"]:
                    unavailable["margins"] = (
                        "No eligible training positions were sampled."
                    )
                embedding = model.item_emb.weight[1:]
                if len(embedding) >= 2:
                    generator = torch.Generator().manual_seed(42)
                    pairs = torch.randint(
                        len(embedding), (2048, 2), generator=generator
                    )
                    pairs = pairs[pairs[:, 0] != pairs[:, 1]].to(embedding.device)
                    metrics["embedding_cosine_mean"] = float(
                        torch.nn.functional.cosine_similarity(
                            embedding[pairs[:, 0]], embedding[pairs[:, 1]], dim=-1
                        ).mean()
                    )
                else:
                    unavailable["embedding_cosine_mean"] = "Fewer than two items."
                recommendations = self._full_catalog_recommendations()
                metrics["full_catalog_user_count"] = len(recommendations)
                definitions["full_catalog_user_count"] = (
                    "Number of real training users receiving full-catalog unseen-item "
                    "top-ten lists in this category view; five per view are required "
                    "for twenty users across the four categories."
                )
                if len(recommendations) < 5:
                    unavailable["twenty_user_report"] = (
                        "This category has fewer than five nonempty training histories."
                    )
                return {
                    "metrics": metrics,
                    "metric_definitions": definitions,
                    "full_catalog_recommendations": recommendations,
                    "dataset_name": getattr(self.view, "dataset_name", None),
                    "unavailable": unavailable,
                    "limitations": [
                        "Training diagnostics do not establish validation improvement.",
                        "Branch normalization changes score scale; compare matched normalized controls.",
                        "Gate variation does not establish intent; branch cancellation remains possible.",
                        "No test labels or simulator traits are observed.",
                    ],
                }
        finally:
            for module, training in flags:
                module.training = training
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)


def diagnosis_interpreter_prompt(raw_diagnosis):
    return (
        "Interpret only these measured metrics, definitions, sample counts and missing-data "
        "limitations. Training margins do not establish validation gains. A mapped cosine "
        "near 0.5 does not establish collapse. Discuss branch scale, cancellation and gate "
        "variation without inferring intent, unavailable metadata, or simulator traits.\n\n"
        + json.dumps(raw_diagnosis, indent=2, ensure_ascii=False, allow_nan=False)
    )
