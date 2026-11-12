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
    ### >>> Self_EvolveRec-BLOCK-START: Interpret only observed diagnostics and distinguish training margins from benchmark results
    - Every statement must be grounded in the provided metric values/definitions.
    - Null values are unavailable measurements, never zero or evidence of healthy behavior.
    - Report nonfinite margins or insufficient observations before interpreting finite-only summaries.
    - If observations are missing, state that model health cannot be established; do not describe missing evidence as healthy.
    - These probes do not test causality, receptive-field size, recent-prefix crop equivalence, gate usefulness, or train/predict consistency. Do not claim those properties were verified.
    - Training-negative margins do not measure validation HR@10 or establish an improvement.
    - Embedding alignment does not establish recommendation diversity, user conformity, or simulator traits.
    - Ratings, review text, item categories, prices and stores are unavailable; do not infer findings about them.
    ### <<< Self_EvolveRec-BLOCK-END
    - If metrics look healthy, status should remain STABLE and implications should focus on "what is already working" plus one potential risk to monitor.
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the NextItNet ModelView contract and exclude unavailable metadata
    def __init__(self, view, device):
        self.model = view
        self.network = view.model
        if not isinstance(self.network, torch.nn.Module):
            raise TypeError("view.model must be the actual trained nn.Module")
        self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train
        self.embedding_sample_count = 0

    def _get_categories(self, item_id: int):
        # Item categories are unavailable in this matched benchmark.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Measure embedding alignment without a quadratic Gram matrix or global RNG changes
    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()
            emb = emb[1 : self.network.itemnum + 1]
            n = emb.size(0)
            k = min(max(0, int(sample_size)), n)
            self.embedding_sample_count = k
            if k < 2:
                return None

            # CPU-local sampling leaves training random streams untouched.
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            idx = torch.randperm(n, generator=generator)[:k].to(emb.device)
            x = emb.index_select(0, idx).float()
            if not torch.isfinite(x).all():
                return None
            if (x.norm(dim=-1) == 0).any():
                return None
            x = F.normalize(x, p=2, dim=-1)
            # Sum over ordered distinct pairs, algebraically equal to the
            # off-diagonal sum of x @ x.T, using O(kD) memory and arithmetic.
            offdiag_sum = x.sum(dim=0).square().sum() - x.square().sum()
            mean_cos = (offdiag_sum / (k * (k - 1))).clamp(-1, 1).item()
            return round(float((mean_cos + 1.0) * 0.5), 4)

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Diagnose actual causal sequence features with training-equivalent masks and explicit missingness
    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(
        self,
        dataloader,
        num_batches: int = 10,
        eps: float = 0.5,
        topk_categories: int = 20,
    ):
        # Keep this signature for existing diagnostic callers. Categories are
        # unavailable, so no category-based interpretation is computed.
        modes = [(module, module.training) for module in self.network.modules()]
        margins_all = []
        valid_count = 0
        nonfinite_count = 0
        batch_count = 0
        skipped_batch_count = 0
        self.network.eval()
        try:
            iterator = iter(dataloader)
            for _ in range(max(0, int(num_batches))):
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                if not (isinstance(batch, (list, tuple)) and len(batch) == 4):
                    raise ValueError(
                        "Diagnosis loader must yield (None, sequence, positive, negative)"
                    )
                seq_t, pos_t, neg_t = [
                    torch.as_tensor(value, device=self.device, dtype=torch.long)
                    for value in batch[1:4]
                ]
                if (
                    seq_t.ndim != 2
                    or pos_t.shape != seq_t.shape
                    or neg_t.shape != seq_t.shape
                ):
                    raise ValueError(
                        "Diagnosis expects sequence, positive and negative tensors of shape B x L"
                    )
                for name, tensor in (
                    ("sequence", seq_t),
                    ("positive", pos_t),
                    ("negative", neg_t),
                ):
                    if ((tensor < 0) | (tensor > self.network.itemnum)).any():
                        raise ValueError(
                            f"Diagnosis {name} contains IDs outside the item vocabulary"
                        )
                batch_count += 1
                valid = (seq_t != 0) & (pos_t != 0) & (neg_t != 0)
                if not valid.any():
                    continue

                # Retain B x L x D features; never use count-1 on left padding.
                features = self.model.log2feats(seq_t)
                if features.ndim != 3 or features.shape[:2] != seq_t.shape:
                    raise ValueError("log2feats must return B x L x D")
                items = torch.stack([pos_t, neg_t], dim=-1)
                logits = self.model.predict_score(features, items)
                if logits.shape != items.shape:
                    raise ValueError(
                        "predict_score must return B x L x 2 for paired items"
                    )
                margins = (logits[..., 0].float() - logits[..., 1].float())[valid]
                finite = torch.isfinite(margins)
                valid_count += margins.numel()
                nonfinite_count += int((~finite).sum().item())
                margins_all.append(margins[finite].cpu())
        finally:
            # Restore even if a probe raises, including mixed submodule modes.
            for module, training in modes:
                module.training = training

        m = torch.cat(margins_all) if margins_all else torch.empty(0)
        count = m.numel()
        margin_metrics = {
            "mean": None,
            "p10": None,
            "neg_beats_pos_rate": None,
            "small_margin_rate": None,
            "counts": {
                "batches": batch_count,
                "skipped_batches": skipped_batch_count,
                "valid_positions": valid_count,
                "finite_positions": count,
                "nonfinite_positions": nonfinite_count,
                "category_observations": 0,
            },
            "eps": float(eps),
        }
        if count:
            k = max(1, int(np.ceil(0.10 * count)))
            margin_metrics.update(
                {
                    "mean": round(float(m.mean().item()), 4),
                    "p10": round(float(torch.kthvalue(m, k).values.item()), 4),
                    "neg_beats_pos_rate": round(
                        float((m < 0).float().mean().item()), 4
                    ),
                    "small_margin_rate": round(
                        float((m < eps).float().mean().item()), 4
                    ),
                }
            )
        return margin_metrics, None

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Preserve training random streams while consuming diagnostic batches
    def run_full_diagnosis(self, train_loader):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        cuda_devices = sorted(
            {
                tensor.device.index
                for tensor in (
                    list(self.network.parameters()) + list(self.network.buffers())
                )
                if tensor.device.type == "cuda"
            }
        )
        try:
            # A loader may invoke sample_batch, which consumes Python and NumPy
            # randomness. Its iterator may also consume the default Torch RNG.
            # Loader-owned generators and worker state remain caller-owned.
            with torch.random.fork_rng(devices=cuda_devices):
                return self._run_full_diagnosis(train_loader)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    def _run_full_diagnosis(self, train_loader):
        print(">>> Running NextItNet training-margin and embedding diagnosis...")
        ### <<< Self_EvolveRec-BLOCK-END

        collapse_score = self.analyze_embeddings_collapse_cosine(sample_size=2048)
        eps = 0.5
        margin_metrics, breakdown = self.analyze_pos_neg_margin_with_category_breakdown(
            train_loader, num_batches=10, eps=eps, topk_categories=20
        )
        return {
            "metrics": {
                ### >>> Self_EvolveRec-BLOCK-START: Expose diagnostic sample counts and the measured margin threshold
                "embedding_collapse_score": collapse_score,
                "embedding_sample_count": self.embedding_sample_count,
                "pos_neg_margin_counts": margin_metrics["counts"],
                "pos_neg_margin_eps": margin_metrics["eps"],
                "pos_neg_margin_mean": margin_metrics["mean"],
                ### <<< Self_EvolveRec-BLOCK-END
                "pos_neg_margin_p10": margin_metrics["p10"],
                "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                "pos_neg_margin_category_breakdown": breakdown,
            },
            "metric_definitions": {
                ### >>> Self_EvolveRec-BLOCK-START: Define measured quantities without unsupported diversity or metadata claims
                "embedding_collapse_score": (
                    "Mean cosine similarity over distinct sampled item-embedding pairs, mapped to [0,1]. "
                    "Padding is excluded. A value near 0.5 corresponds to mean cosine near zero; "
                    "higher values indicate greater directional alignment, not measured recommendation diversity. "
                    "Null means fewer than two samples, a zero-norm sample, or nonfinite embeddings."
                ),
                "embedding_sample_count": "Number of item embeddings sampled using a local seed-42 generator.",
                "pos_neg_margin_counts": (
                    "Counts of processed batches, malformed skipped batches, valid positions, finite margins, "
                    "nonfinite margins and observed item categories. Valid requires nonzero sequence, positive "
                    "and negative IDs. Category observations are zero because categories are unavailable. "
                    "Margin summaries exclude nonfinite margins; inspect their count before interpreting summaries."
                ),
                "pos_neg_margin_eps": "Measured small-margin threshold used by this probe, in logit units.",
                "pos_neg_margin_mean": (
                    "Mean positive-minus-negative logit over finite valid training positions in evaluation mode. "
                    "Uses supplied training negatives, not validation candidates. Null means no finite observations."
                ),
                "pos_neg_margin_p10": (
                    "Nearest-rank lower 10th percentile of finite valid training margins; null if none exist."
                ),
                "pos_neg_neg_beats_pos_rate": (
                    "Fraction of finite valid training margins below zero; null if none exist."
                ),
                "pos_neg_small_margin_rate": (
                    "Fraction of finite valid training margins below pos_neg_margin_eps; null if none exist."
                ),
                "pos_neg_margin_category_breakdown": (
                    "Null: item categories are unavailable in this benchmark. No category breakdown is measured."
                ),
                ### <<< Self_EvolveRec-BLOCK-END
            },
        }
