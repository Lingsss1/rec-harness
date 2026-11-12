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
    ### >>> Self_EvolveRec-BLOCK-START: Interpret missing evidence and training diagnostics conservatively
    - Every statement must be grounded in the provided metric values/definitions.
    - Null metrics and unavailable probes are missing evidence, never zero values or healthy results.
    - If essential measurements are unavailable, use NEEDS_IMPROVEMENT and explicitly describe the diagnostic limitation; do not infer model failure.
    - Training-position margins do not establish validation HR@10, generalization, recommendation diversity, or simulator acceptance.
    - Embedding alignment alone does not establish representation collapse or representation rank.
    - Do not attribute improvements to residual gating without measured comparisons against matched controls.
    - Use STABLE only when available measurements support that limited conclusion, and state their scope.
    ### <<< Self_EvolveRec-BLOCK-END
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the diagnostic view and actual trained module
    def __init__(self, view, device):
        self.model = view
        self.module = view.model
        if not isinstance(self.module, torch.nn.Module):
            raise TypeError("view.model must be the actual trained nn.Module")
        self.device = torch.device(device)
        self.meta_data = getattr(view, "meta_data", {})
        self.review_data = getattr(view, "review_data", {})
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        # Item categories are unavailable in the shared RecHarness task.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Measure embedding alignment without materializing a Gram matrix
    @torch.no_grad()
    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        emb = self.model.item_emb.weight.detach()[1:]
        k = min(int(sample_size), emb.size(0))
        if k < 2:
            return None
        generator = torch.Generator(device=emb.device)
        generator.manual_seed(seed)
        indices = torch.randperm(emb.size(0), generator=generator, device=emb.device)[
            :k
        ]
        x = emb.index_select(0, indices).float()
        if not torch.isfinite(x).all():
            raise ValueError("Sampled item embeddings contain nonfinite values")
        if x.norm(dim=-1).eq(0).any():
            raise ValueError(
                "Cosine alignment is undefined for zero-norm sampled embeddings"
            )
        x = F.normalize(x, p=2, dim=-1).double()
        # Sum of all off-diagonal dot products, using O(k*d) work and storage.
        offdiag_sum = x.sum(dim=0).square().sum() - x.square().sum()
        mean_cos = (offdiag_sum / (k * (k - 1))).clamp(-1, 1)
        return float(((mean_cos + 1.0) * 0.5).item())

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Probe mixed tensor features at valid training positions
    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(
        self,
        dataloader,
        num_batches: int = 10,
        eps: float = 0.5,
        topk_categories: int = 20,
    ):
        modes = [(module, module.training) for module in self.module.modules()]
        margins_all = []
        batches = 0
        try:
            self.module.eval()
            iterator = iter(dataloader)
            for _ in range(num_batches):
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                if not isinstance(batch, (list, tuple)) or len(batch) != 4:
                    raise ValueError("Expected (None, sequence, positive, negative)")
                seq, pos, neg = [
                    torch.as_tensor(value, device=self.device, dtype=torch.long)
                    for value in batch[1:]
                ]
                if seq.ndim != 2 or pos.shape != seq.shape or neg.shape != seq.shape:
                    raise ValueError(
                        "Sequence, positive and negative must all have shape B x L"
                    )
                batches += 1
                valid = seq.ne(0) & pos.ne(0) & neg.ne(0)
                if (valid & pos.eq(neg)).any():
                    raise ValueError("A sampled negative equals its positive target")
                if not valid.any():
                    continue
                features = self.model.log2feats(seq)
                if (
                    not isinstance(features, torch.Tensor)
                    or features.ndim != 3
                    or features.shape[:2] != seq.shape
                ):
                    raise ValueError("log2feats must return B x L x D tensor features")
                if not torch.isfinite(features[valid]).all():
                    raise ValueError("Valid-position features contain nonfinite values")
                items = torch.stack((pos, neg), dim=-1)
                logits = self.model.predict_score(features, items)
                if not isinstance(logits, torch.Tensor) or logits.shape != items.shape:
                    raise ValueError(
                        "predict_score must return B x L x 2 tensor logits"
                    )
                margins = (logits[..., 0].float() - logits[..., 1].float())[valid]
                if not torch.isfinite(margins).all():
                    raise ValueError("Valid-position margins contain nonfinite values")
                margins_all.append(margins.cpu())
        finally:
            for module, training in modes:
                module.training = training

        breakdown = {
            "status": "unavailable",
            "reason": "Item categories are not available in this shared task.",
            "eps": float(eps),
            "batches_examined": batches,
            "valid_positions": sum(m.numel() for m in margins_all),
        }
        if not margins_all:
            breakdown["margin_status"] = (
                "unavailable: no valid sampled training positions"
            )
            return (
                dict.fromkeys(
                    ("mean", "p10", "neg_beats_pos_rate", "small_margin_rate"), None
                ),
                breakdown,
            )

        margins = torch.cat(margins_all)
        rank = max(1, int(np.ceil(0.10 * margins.numel())))
        breakdown["low_margin_total_positions"] = int((margins < eps).sum().item())
        breakdown["wrong_rank_total_positions"] = int((margins < 0).sum().item())
        return {
            "mean": float(margins.mean().item()),
            "p10": float(torch.kthvalue(margins, rank).values.item()),
            "neg_beats_pos_rate": float((margins < 0).float().mean().item()),
            "small_margin_rate": float((margins < eps).float().mean().item()),
        }, breakdown

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Return measured diagnostics while restoring module modes and global random streams
    def run_full_diagnosis(self, train_loader):
        modes = [(module, module.training) for module in self.module.modules()]
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.get_rng_state()
        embedding_device = self.model.item_emb.weight.device
        cuda_rng = (
            torch.cuda.get_rng_state(embedding_device)
            if embedding_device.type == "cuda"
            else None
        )
        unavailable = {
            "item_category_breakdown": "Item categories are unavailable in the shared task.",
            "review_and_rating_probes": "Review text, review ratings and item ratings are unavailable.",
            "simulator_traits": "Diversity and conformity traits are unobserved.",
            "validation_hr10": "Training-position diagnostics do not evaluate pinned validation rankings.",
        }
        collapse_score = None
        margin_metrics = dict.fromkeys(
            ("mean", "p10", "neg_beats_pos_rate", "small_margin_rate"), None
        )
        breakdown = {
            "status": "unavailable",
            "reason": "Margin probe did not complete.",
        }
        try:
            self.module.eval()
            try:
                collapse_score = self.analyze_embeddings_collapse_cosine(
                    sample_size=2048
                )
                if collapse_score is None:
                    unavailable["embedding_collapse_score"] = (
                        "Fewer than two catalog embeddings."
                    )
            except Exception as error:
                unavailable["embedding_collapse_score"] = (
                    f"{type(error).__name__}: {error}"
                )
            try:
                margin_metrics, breakdown = (
                    self.analyze_pos_neg_margin_with_category_breakdown(
                        train_loader, num_batches=10, eps=0.5, topk_categories=20
                    )
                )
                if margin_metrics["mean"] is None:
                    unavailable["training_margins"] = (
                        "No valid sampled training positions."
                    )
            except Exception as error:
                reason = f"{type(error).__name__}: {error}"
                unavailable["training_margins"] = reason
                breakdown = {"status": "unavailable", "reason": reason}
        finally:
            for module, training in modes:
                module.training = training
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(torch_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, embedding_device)

        return {
            "metrics": {
                "embedding_collapse_score": collapse_score,
                "pos_neg_margin_mean": margin_metrics["mean"],
                "pos_neg_margin_p10": margin_metrics["p10"],
                "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                "pos_neg_margin_category_breakdown": breakdown,
                "unavailable_probes": unavailable,
            },
            "metric_definitions": {
                "embedding_collapse_score": (
                    "(Mean off-diagonal cosine + 1)/2 for up to 2048 nonpadding item embeddings, "
                    "sampled without replacement with seed 42. About 0.5 means near-zero mean cosine; "
                    "higher means greater directional alignment. This alone does not measure ranking "
                    "quality, recommendation diversity or representation rank. Null means unavailable."
                ),
                "pos_neg_margin_mean": (
                    "Mean positive-minus-negative score using mixed log2feats features in eval mode, "
                    "over positions with nonzero sequence, positive and negative IDs in up to ten "
                    "training batches. Uses the loader's negative distribution; not final-position HR@10."
                ),
                "pos_neg_margin_p10": (
                    "Nearest-rank 10th percentile, rank ceil(0.1*N), of the same training-position margins."
                ),
                "pos_neg_neg_beats_pos_rate": (
                    "Fraction of the same training positions with margin < 0; ties are excluded."
                ),
                "pos_neg_small_margin_rate": (
                    "Fraction of the same training positions with margin < 0.5; depends on score scale."
                ),
                "pos_neg_margin_category_breakdown": (
                    "Explicit category unavailability plus measured batch/position counts and margin "
                    "counts when available. No item categories or dataset macro scores are inferred."
                ),
                "unavailable_probes": (
                    "Reasons for unavailable probes, including explicit errors. Null numeric metrics "
                    "indicate missing measurements, not zero values. No validation improvement or "
                    "subjective acceptance is estimated. Module modes and global Python, NumPy, "
                    "Torch CPU and model-device CUDA random streams are restored. Loader iteration "
                    "and loader-owned generators are consumed; use a separate diagnostic loader "
                    "to avoid changing subsequent training batches."
                ),
            },
        }

    ### <<< Self_EvolveRec-BLOCK-END
