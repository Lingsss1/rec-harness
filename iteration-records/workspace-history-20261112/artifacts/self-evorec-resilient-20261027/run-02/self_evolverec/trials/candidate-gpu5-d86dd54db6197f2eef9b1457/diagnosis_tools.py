from __future__ import annotations

import itertools
import json
import torch
import torch.nn.functional as F

class DiagnosisProbe:
    def __init__(self, view, device):
        self.view = view
        self.model = view.model
        self.device = torch.device(device)

    def _sequence_folds(self):
        existing = getattr(self.model, "training_sequence_folds", {})
        if existing:
            return existing
        histories = self.view.user_train
        histories = histories.get("History", histories)
        histories = {
            int(u): [int(i) for i in history] for u, history in histories.items()
        }
        import numpy as np

        users = sorted(histories)
        permutation = np.random.RandomState(42).permutation(len(users))
        folds = {users[int(index)]: rank % 5 for rank, index in enumerate(permutation)}
        result = {}
        for user, history in histories.items():
            if len(history) < 2:
                continue
            key = tuple(history[:-1][-self.model.maxlen :])
            fold = folds[user]
            previous = result.get(key, fold)
            result[key] = fold if previous == fold else -1
        return result

    def run_full_diagnosis(self, train_loader):
        model = self.model
        required = ("score", "transition_evidence", "recent_contexts", "log2feats")
        if any(not callable(getattr(model, name, None)) for name in required):
            raise TypeError(
                "Diagnosis requires the trained transition-aware SASRec model"
            )
        was_training = model.training
        model.eval()
        metrics = {}
        definitions = {}
        collected = {
            "raw_dot_margin": [],
            "complete_score_margin": [],
            "context_coverage": [],
            "positive_edge_coverage": [],
            "absolute_residual": [],
            "deployment_training_residual_MAE": [],
            "singleton_user_support_margin": [],
            "multiple_user_support_margin": [],
        }
        sequence_folds = self._sequence_folds()
        unresolved_rows = 0
        sampled_positions = 0
        try:
            with torch.no_grad():
                for _, sequence, positive, negative in itertools.islice(
                    train_loader, 4
                ):
                    sequence = torch.as_tensor(
                        sequence, dtype=torch.long, device=self.device
                    )
                    positive = torch.as_tensor(
                        positive, dtype=torch.long, device=self.device
                    )
                    negative = torch.as_tensor(
                        negative, dtype=torch.long, device=self.device
                    )
                    fold_list = []
                    for row in sequence.detach().cpu().tolist():
                        key = tuple(int(i) for i in row if i != 0)
                        fold_list.append(sequence_folds.get(key, -1))
                    folds = torch.tensor(
                        fold_list, dtype=torch.long, device=self.device
                    )
                    unresolved_rows += int((folds < 0).sum())
                    valid = (positive != 0) & (folds[:, None] >= 0)
                    locations = valid.nonzero(as_tuple=False)
                    if not locations.numel():
                        continue
                    if locations.shape[0] > 512:
                        indices = torch.linspace(
                            0, locations.shape[0] - 1, 512, device=self.device
                        ).long()
                        locations = locations[indices]
                    b, t = locations.unbind(dim=1)
                    features = model.log2feats(sequence)[b, t]
                    contexts = model.recent_contexts(sequence)[b, t]
                    candidates = torch.stack((positive[b, t], negative[b, t]), dim=1)
                    selectors = folds[b]
                    raw = (model.item_emb(candidates) * features.unsqueeze(1)).sum(-1)
                    complete = model.score(features, contexts, candidates, selectors)
                    residual, coverage, support = model.transition_evidence(
                        contexts, candidates, selectors
                    )
                    deployment_residual, _, _ = model.transition_evidence(
                        contexts, candidates
                    )
                    margin = complete[:, 0] - complete[:, 1]
                    collected["raw_dot_margin"].append(raw[:, 0] - raw[:, 1])
                    collected["complete_score_margin"].append(margin)
                    collected["context_coverage"].append(coverage)
                    collected["positive_edge_coverage"].append(
                        (support[:, 0] > 0).float()
                    )
                    collected["absolute_residual"].append(residual.abs().reshape(-1))
                    collected["deployment_training_residual_MAE"].append(
                        (deployment_residual - residual).abs().reshape(-1)
                    )
                    collected["singleton_user_support_margin"].append(
                        margin[support[:, 0] == 1]
                    )
                    collected["multiple_user_support_margin"].append(
                        margin[support[:, 0] >= 2]
                    )
                    sampled_positions += int(locations.shape[0])
                descriptions = {
                    "raw_dot_margin": "Mean positive-minus-negative embedding dot-product score before transition evidence.",
                    "complete_score_margin": "Mean positive-minus-negative complete score using each training user's excluded-fold graph.",
                    "context_coverage": "Mean recency-weighted share of context sources with retained outgoing edges in the user-excluding graph.",
                    "positive_edge_coverage": "Fraction of sampled positive targets with a retained edge from at least one recent context item.",
                    "absolute_residual": "Mean absolute pooled transition residual before multiplication by the learned gate, across positive and negative candidates.",
                    "deployment_training_residual_MAE": "Mean absolute difference between deployment-average residual and user-excluding residual on sampled training positions; this is a training-side distribution-shift diagnostic.",
                    "singleton_user_support_margin": "Mean complete margin where the maximum distinct-other-user support among retained positive-target context edges is exactly one.",
                    "multiple_user_support_margin": "Mean complete margin where at least one retained positive-target context edge has support from two or more distinct other training users.",
                }
                unavailable = {}
                for name, parts in collected.items():
                    parts = [part.reshape(-1) for part in parts if part.numel()]
                    definitions[name] = descriptions[name]
                    if parts:
                        values = torch.cat(parts).float()
                        metrics[name] = float(values.mean())
                        metrics[name + "_count"] = int(values.numel())
                        definitions[name + "_count"] = (
                            "Number of scalar observations used for " + name + "."
                        )
                    else:
                        unavailable[name] = (
                            "No eligible observations in the bounded diagnostic sample."
                        )
                embedding = model.item_emb.weight[1:]
                if embedding.shape[0] >= 2:
                    generator = torch.Generator(device="cpu").manual_seed(42)
                    pairs = torch.randint(
                        0, embedding.shape[0], (1024, 2), generator=generator
                    )
                    pairs = pairs[pairs[:, 0] != pairs[:, 1]].to(embedding.device)
                    if pairs.numel():
                        cosine = F.cosine_similarity(
                            embedding[pairs[:, 0]], embedding[pairs[:, 1]], dim=-1
                        )
                        metrics["embedding_cosine_mean"] = float(cosine.mean())
                        metrics["embedding_cosine_std"] = float(
                            cosine.std(unbiased=False)
                        )
                        definitions["embedding_cosine_mean"] = (
                            "Mean cosine similarity of seed-42 sampled distinct nonpadding item pairs; alone this does not establish embedding collapse."
                        )
                        definitions["embedding_cosine_std"] = (
                            "Population standard deviation of the sampled pair cosine similarities."
                        )
                metrics["sampled_positions"] = sampled_positions
                metrics["unresolved_training_rows"] = unresolved_rows
                definitions["sampled_positions"] = (
                    "Number of supervised positions measured from at most four batches and 512 positions per batch."
                )
                definitions["unresolved_training_rows"] = (
                    "Rows excluded because their exact sampled training history could not be assigned unambiguously to a user fold."
                )
                unavailable.update(
                    {
                        "category_breakdown": "Item category metadata is unavailable in this matched task.",
                        "review_preferences": "Review ratings and text are unavailable.",
                        "simulator_traits": "Diversity and conformity traits are unobserved.",
                    }
                )
                return {
                    "metrics": metrics,
                    "metric_definitions": definitions,
                    "unavailable": unavailable,
                    "limitations": [
                        "Only public training histories contribute to transition graphs.",
                        "Diagnostics use a bounded training sample and do not establish validation improvement.",
                        "Distinct-user support is per directed edge, not a union of users across context edges.",
                        "Validation JSON records constituent-fold versus deployment score shift and HR@10 on fixed validation candidates.",
                        "The host independently computes the benchmark selection metric.",
                    ],
                }
        finally:
            model.train(was_training)


def diagnosis_interpreter_prompt(raw_diagnosis):
    return (
        "Interpret only the measured diagnostics and their supplied definitions below. "
        "Separate raw embedding margins from complete transition-aware margins. "
        "Discuss coverage, residual magnitude, user-support strata, and measured graph distribution shift. "
        "Do not infer benchmark improvement from training margins or diagnose severe collapse from "
        "a cosine mean alone. Respect unavailable fields and sampling limitations. "
        "Do not invent review preferences, category metadata, simulator traits, or test results. "
        "Keep uncertain recommendations separate from explicit implementation mismatches.\n\n"
        + json.dumps(raw_diagnosis, indent=2, ensure_ascii=False, allow_nan=False)
    )
