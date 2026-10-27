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
    - Every statement must be grounded in the provided metric values/definitions.
    - If metrics look healthy, status should remain STABLE and implications should focus on "what is already working" plus one potential risk to monitor.
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the view contract and respect unavailable metadata
    def __init__(self, view, device):
        self.model = view
        self.actual_model = view.model
        try:
            self.device = next(self.actual_model.parameters()).device
        except StopIteration:
            self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        # Item categories are unavailable in this matched task.
        return []

    def _snapshot_modes(self):
        return [(module, module.training) for module in self.actual_model.modules()]

    @staticmethod
    def _restore_modes(modes):
        for module, training in modes:
            module.training = training

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()
            emb = emb[1:]

            ### >>> Self_EvolveRec-BLOCK-START: Return null when cosine collapse cannot be observed
            n = emb.size(0)
            k = min(max(int(sample_size), 0), n)
            if k < 2:
                return None

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            ### <<< Self_EvolveRec-BLOCK-END
            ### >>> Self_EvolveRec-BLOCK-START: Compute equivalent cosine collapse in linear memory with finite checks
            x = emb.index_select(0, idx).float()
            if not torch.isfinite(x).all().item():
                return None
            x = F.normalize(x, p=2, dim=-1)

            # Sum of off-diagonal dot products without allocating a k x k Gram matrix.
            # Accumulate in float64 to reduce cancellation near orthogonal embeddings.
            x = x.double()
            summed = x.sum(dim=0)
            offdiag_sum = summed.square().sum() - x.square().sum()
            offdiag_cnt = k * (k - 1)
            mean_cos = (offdiag_sum / offdiag_cnt).clamp(-1, 1).item()
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Keep nonfinite collapse observations JSON compatible
            collapse_score = (mean_cos + 1.0) * 0.5
            return (
                round(float(collapse_score), 4) if np.isfinite(collapse_score) else None
            )
            ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Diagnose the actual trained module and restore mode on every exit
    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(
        self,
        dataloader,
        num_batches: int = 10,
        eps: float = 0.5,
        topk_categories: int = 20,
    ):
        modes = self._snapshot_modes()
        try:
            self.actual_model.eval()
            return self._analyze_training_margins(
                dataloader, num_batches, eps, topk_categories
            )
        finally:
            self._restore_modes(modes)

    @torch.no_grad()
    def _analyze_training_margins(self, dataloader, num_batches, eps, topk_categories):
        from itertools import islice

        observed_positions = 0
        nonfinite_positions = 0
        batches_used = 0
        margins_all = []
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Validate diagnostic batches explicitly and aggregate training margins on CPU
        if not np.isfinite(eps):
            raise ValueError("Diagnosis margin threshold must be finite")
        for batch_index, batch in enumerate(
            islice(iter(dataloader), max(0, int(num_batches)))
        ):
            if not (isinstance(batch, (list, tuple)) and len(batch) == 4):
                raise ValueError(
                    f"Diagnosis batch {batch_index} must contain "
                    "(None, sequence, positive, negative)"
                )
            _, seq, pos, neg = batch
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
                or seq_t.shape[1] == 0
            ):
                raise ValueError(
                    "Diagnosis expects sequence, positive and negative of shape B x L, L > 0"
                )

            batches_used += 1
            valid = seq_t.ne(0) & pos_t.ne(0) & neg_t.ne(0)
            if not valid.any().item():
                continue

            # Reuse this bounded diagnostic pass; inspect one batch only for
            # baseline and the centered-normalized contender.
            if (
                getattr(self, "_history_observations", None) is None
                and getattr(self.actual_model, "history_mode", "none")
                in {"none", "centered"}
                and hasattr(self.actual_model, "history_components")
            ):
                raw, branch = self.actual_model.history_components(seq_t)
                raw = raw[seq_t.ne(0)].float()
                branch = branch[seq_t.ne(0)].float()
                finite = torch.isfinite(raw).all(-1) & torch.isfinite(branch).all(-1)
                self._history_observations = {
                    "observed_positions": int(finite.numel()),
                    "finite_positions": int(finite.sum().item()),
                    "nonfinite_positions": int((~finite).sum().item()),
                    "raw_delta_rms": (
                        float(raw[finite].square().mean().sqrt().cpu())
                        if finite.any().item()
                        else None
                    ),
                    "gated_branch_rms": (
                        float(branch[finite].square().mean().sqrt().cpu())
                        if finite.any().item()
                        else None
                    ),
                }
            log_feats = self.model.log2feats(seq_t)
            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
            if not isinstance(all_logits, torch.Tensor):
                raise TypeError("predict_score must return a torch.Tensor")
            if all_logits.shape != all_items.shape:
                raise ValueError("predict_score must return shape B x L x 2")
            if all_logits.device != valid.device:
                raise ValueError("predict_score must return logits on the model device")

            # Transfer only supervised logits; count nonfinite observations explicitly.
            logits = all_logits[valid].detach().float().cpu()
            margin = logits[:, 0] - logits[:, 1]
            finite = torch.isfinite(logits).all(dim=-1) & torch.isfinite(margin)
            observed_positions += margin.numel()
            nonfinite_positions += int((~finite).sum().item())
            m = margin[finite]
            if m.numel():
                margins_all.append(m)
                low_total += int((m < float(eps)).sum().item())
                wrong_total += int((m < 0.0).sum().item())
        ### <<< Self_EvolveRec-BLOCK-END

        ### >>> Self_EvolveRec-BLOCK-START: Distinguish absent observations from measured zero margins
        if len(margins_all) == 0:
            margin_metrics = {
                "mean": None,
                "p10": None,
                "neg_beats_pos_rate": None,
                "small_margin_rate": None,
            }
        ### <<< Self_EvolveRec-BLOCK-END
        else:
            m = torch.cat(margins_all, dim=0)
            N = m.numel()

            k = max(1, int(np.ceil(0.10 * N)))
            p10 = float(torch.kthvalue(m, k).values.item())

            margin_metrics = {
                "mean": round(float(m.mean().item()), 4),
                "p10": round(p10, 4),
                "neg_beats_pos_rate": round(float((m < 0.0).float().mean().item()), 4),
                "small_margin_rate": round(
                    float((m < float(eps)).float().mean().item()), 4
                ),
            }

        def _topk(counter: Counter, k: int):
            return [{"category": c, "count": int(n)} for c, n in counter.most_common(k)]

        ### >>> Self_EvolveRec-BLOCK-START: Explicit training-only scope and unavailable category breakdowns
        breakdown = {
            "scope": "training_only",
            "categories_available": False,
            "unavailable_reason": "Item categories are not provided in the matched task",
            "observed_positions": observed_positions,
            "finite_positions": observed_positions - nonfinite_positions,
            "nonfinite_positions": nonfinite_positions,
            "batches_used": batches_used,
            "eps": float(eps),
            "low_margin_total_positions": int(low_total) if margins_all else None,
            "wrong_rank_total_positions": int(wrong_total) if margins_all else None,
            ### <<< Self_EvolveRec-BLOCK-END
            "low_margin_top_categories": {
                "pos_item": _topk(low_pos_cat, topk_categories),
                "neg_item": _topk(low_neg_cat, topk_categories),
            },
            "wrong_rank_top_categories": {
                "pos_item": _topk(wrong_pos_cat, topk_categories),
                "neg_item": _topk(wrong_neg_cat, topk_categories),
            },
        }

        return margin_metrics, breakdown

    ### >>> Self_EvolveRec-BLOCK-START: Bound diagnostic autograd overhead and preserve model modes on failures
    @torch.no_grad()
    def run_full_diagnosis(self, train_loader):
        import time

        modes = self._snapshot_modes()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        result = None
        try:
            self.actual_model.eval()
            result = self._run_full_diagnosis(train_loader)
            return result
        finally:
            self._restore_modes(modes)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - started
            if result is not None:
                result["metrics"]["diagnosis_elapsed_seconds"] = elapsed
                result["metric_definitions"][
                    "diagnosis_elapsed_seconds"
                ] = "Measured wall time for this bounded diagnostic pass, including device synchronization."

    def _run_full_diagnosis(self, train_loader):
        print(">>> Running bounded training-only diagnosis...")
        ### <<< Self_EvolveRec-BLOCK-END

        ### >>> Self_EvolveRec-BLOCK-START: Diagnose actual sigmoid gate and learned scalar retention
        history_parameters = {
            "mode": getattr(self.actual_model, "history_mode", "none"),
            "gate": None,
            "retention": None,
            "infinite_history_horizon": None,
            "pool_eps": getattr(self.actual_model, "pool_eps", None),
        }
        if hasattr(self.actual_model, "gate_logit"):
            gate = float(self.actual_model.gate_logit.detach().sigmoid().cpu())
            history_parameters["gate"] = gate if np.isfinite(gate) else None
        if hasattr(self.actual_model, "retention_logit"):
            retention = float(
                self.actual_model.retention_logit.detach().sigmoid().cpu()
            )
            history_parameters["retention"] = (
                retention if np.isfinite(retention) else None
            )
            if np.isfinite(retention) and 0.0 <= retention < 1.0:
                history_parameters["infinite_history_horizon"] = 1.0 / (1.0 - retention)
        self._history_observations = None
        collapse_score = self.analyze_embeddings_collapse_cosine(sample_size=2048)
        ### <<< Self_EvolveRec-BLOCK-END
        eps = 0.5
        margin_metrics, breakdown = self.analyze_pos_neg_margin_with_category_breakdown(
            train_loader, num_batches=10, eps=eps, topk_categories=20
        )
        return {
            "metrics": {
                ### >>> Self_EvolveRec-BLOCK-START: Include trained history parameters and bounded branch observations
                "history_residual_parameters": history_parameters,
                "history_branch_observations": self._history_observations,
                "embedding_collapse_score": collapse_score,
                ### <<< Self_EvolveRec-BLOCK-END
                "pos_neg_margin_mean": margin_metrics["mean"],
                "pos_neg_margin_p10": margin_metrics["p10"],
                "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                "pos_neg_margin_category_breakdown": breakdown,
            },
            "metric_definitions": {
                ### >>> Self_EvolveRec-BLOCK-START: Define gate retention and branch magnitudes without ranking claims
                "history_residual_parameters": (
                    "Actual trained pooling mode, sigmoid gate, sigmoid retention when learned, "
                    "and normalization epsilon. Horizon 1/(1-retention) describes infinite-history "
                    "weighting, not effective context under truncation. Null means absent or "
                    "nonfinite; horizon is also null at retention one. These quantities do not "
                    "establish ranking benefit, disentangled interests, or simulator traits."
                ),
                "history_branch_observations": (
                    "One existing diagnostic batch for baseline or centered-normalized mode. "
                    "Counts cover nonpadding sequence positions; RMS values average squared "
                    "components over positions where both raw delta and gated branch are finite. "
                    "Raw delta precedes LayerNorm and gating; gated branch follows both. "
                    "Baseline has zero branch. Null means no eligible observations or a control "
                    "outside this expanded diagnostic schedule. Not a selection metric."
                ),
                ### <<< Self_EvolveRec-BLOCK-END
                ### >>> Self_EvolveRec-BLOCK-START: Interpret measured representations and training margins without unsupported claims
                "embedding_collapse_score": (
                    "Mean off-diagonal cosine similarity of up to 2048 seeded sampled nonpadding "
                    "item embeddings, mapped from [-1,1] to [0,1]. Higher indicates more aligned "
                    "embedding directions; this does not directly measure recommendation diversity. "
                    "Null means fewer than two sampled items or a nonfinite sampled embedding."
                ),
                "pos_neg_margin_mean": (
                    "Training-only mean positive-minus-negative logit over nonpadding positions "
                    "with finite margins. Null means no finite observations; not validation performance."
                ),
                "pos_neg_margin_p10": (
                    "Training-only nearest-rank 10th percentile of finite positive-minus-negative "
                    "logits at nonpadding positions. Null means no finite observations."
                ),
                "pos_neg_neg_beats_pos_rate": (
                    "Training-only fraction of finite valid margins below zero. "
                    "Null means no finite observations."
                ),
                "pos_neg_small_margin_rate": (
                    "Training-only fraction of finite valid margins below eps=0.5. "
                    "Null means no finite observations."
                ),
                "pos_neg_margin_category_breakdown": (
                    "Training-only margin counts, finite/nonfinite observation counts and batches used. "
                    "Item categories are unavailable, so category lists remain empty with an explicit "
                    "availability flag. Counts of low or negative margins are null without finite observations. "
                    "Simulator diversity and conformity traits are unobserved. "
                    "Malformed loader batches raise explicit errors."
                ),
                ### <<< Self_EvolveRec-BLOCK-END
            },
        }
