"""Bounded public-training diagnostics; no validation or test selection."""
import random
from itertools import islice

import numpy as np
import torch
import torch.nn.functional as F


class DiagnosisProbe:
    def __init__(self, view, device):
        self.view = view
        self.model = view.model
        self.device = torch.device(device)
        self.item_emb = view.item_emb
        self.user_train = view.user_train
        self.meta_data = view.meta_data
        self.review_data = view.review_data

    @torch.no_grad()
    def run_full_diagnosis(self, train_loader):
        modes = [(module, module.training) for module in self.model.modules()]
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        devices = []
        if self.device.type == "cuda":
            devices = [self.device.index if self.device.index is not None
                       else torch.cuda.current_device()]
        try:
            with torch.random.fork_rng(devices=devices):
                random.seed(1042)
                np.random.seed(1042)
                torch.random.default_generator.manual_seed(1042)
                if devices:
                    with torch.cuda.device(devices[0]):
                        torch.cuda.manual_seed(1042)
                self.model.eval()
                return self._measure(train_loader)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            for module, training in modes:
                module.training = training

    def _measure(self, train_loader):
        metrics = {
            "base_item_mean_cosine": None,
            "pos_neg_margin": None,
            "margin_valid_positions": 0,
            "pos_neg_margin_category_breakdown": {"low_margin": [], "wrong_rank": []},
            "attention_entropy": None,
            "effective_attended_history_count": None,
            "residual_scores": None,
            "residual_to_base_rms_ratio": None,
            "raw_branch_table_norms": {},
            "last_retained_branch_gradient_norms": {},
        }
        cfg = getattr(self.model, "residual_config", None)
        attention = cfg is not None and cfg.get("mode") == "candidate_attention"
        metrics["attention_status"] = "no_valid_positions" if attention else "unavailable"
        embedding = self.item_emb.weight.detach()
        count = min(2048, embedding.shape[0] - 1)
        if count >= 2:
            generator = torch.Generator(device=embedding.device).manual_seed(1042)
            ids = torch.randperm(embedding.shape[0] - 1, device=embedding.device,
                                 generator=generator)[:count] + 1
            vectors = F.normalize(embedding[ids].float(), dim=-1)
            metrics["base_item_mean_cosine"] = float(
                ((vectors.sum(0).square().sum() - vectors.square().sum())
                 / (count * (count - 1))).clamp(-1, 1).item())
        for name in ("value_emb", "target_emb", "attention_emb"):
            table = getattr(self.model, name, None)
            if table is None:
                continue
            norms = table.weight.detach()[1:2049].float().norm(dim=-1)
            if norms.numel():
                metrics["raw_branch_table_norms"][name] = {
                    "mean": float(norms.mean().item()),
                    "min": float(norms.min().item()),
                    "max": float(norms.max().item()),
                }
            gradient = table.weight.grad
            metrics["last_retained_branch_gradient_norms"][name] = (
                None if gradient is None else float(gradient.detach().float().norm().item()))
        margins, residuals, bases, entropies = [], [], [], []
        for batch in islice(train_loader, 2):
            if not isinstance(batch, (tuple, list)) or len(batch) != 4:
                raise ValueError("Expected (None, sequence, positive, negative)")
            _, sequence, positive, negative = batch
            sequence, positive, negative = [
                torch.as_tensor(x, dtype=torch.long, device=self.device)[:32]
                for x in (sequence, positive, negative)
            ]
            if sequence.ndim != 2 or positive.shape != sequence.shape or negative.shape != sequence.shape:
                raise ValueError("Diagnostic tensors must have matching B x L shapes")
            features = self.view.log2feats(sequence)
            pos = self.view.predict_score(features, positive)
            neg = self.view.predict_score(features, negative)
            if pos.shape != sequence.shape or neg.shape != sequence.shape:
                raise ValueError("Complete aligned scorer must return B x L scores")
            valid = sequence.ne(0) & positive.ne(0) & negative.ne(0)
            if not valid.any():
                continue
            scores = torch.stack((pos[valid], neg[valid]), dim=-1).float()
            if not torch.isfinite(scores).all():
                raise FloatingPointError("Nonfinite complete diagnostic scores")
            margins.append((scores[:, 0] - scores[:, 1]).cpu())
            if cfg is None:
                continue
            if not hasattr(features, "h") or not hasattr(features, "recent_ids"):
                raise TypeError("Candidate residual diagnostics require prefix feature bundles")
            candidates = torch.stack((positive[valid], negative[valid]), dim=-1)
            base = (self.item_emb(candidates) * features.h[valid, None, :]).sum(-1).float()
            residuals.append((scores - base).cpu())
            bases.append(base.cpu())
            if attention:
                recent = features.recent_ids[valid]
                keys = self.model._normalized(self.model.attention_emb, recent)
                queries = self.model._normalized(self.model.attention_emb, candidates)
                logits = torch.bmm(queries, keys.transpose(1, 2)) / cfg["tau"]
                lag = torch.arange(cfg["K"], device=self.device, dtype=logits.dtype)
                logits = logits + lag * np.log(cfg["decay"])
                logits = logits.masked_fill(~recent.ne(0).unsqueeze(1), -torch.inf)
                weights = logits.softmax(-1).float()
                entropy = -(weights * weights.clamp_min(cfg["eps"]).log()).sum(-1)
                entropies.append(entropy.cpu())
        if margins:
            margin = torch.cat(margins)
            metrics["margin_valid_positions"] = margin.numel()
            metrics["pos_neg_margin"] = {
                "mean": float(margin.mean()),
                "p10": float(torch.quantile(margin, 0.1)),
                "neg_beats_pos_rate": float((margin < 0).float().mean()),
                "small_margin_rate": float((margin < 1).float().mean()),
            }
        if residuals:
            residual, base = torch.cat(residuals), torch.cat(bases)
            metrics["residual_scores"] = {
                "mean": float(residual.mean()),
                "std": float(residual.std(unbiased=False)),
                "min": float(residual.min()),
                "max": float(residual.max()),
            }
            metrics["residual_to_base_rms_ratio"] = float(
                residual.square().mean().sqrt() / base.square().mean().sqrt().clamp_min(1e-6))
        if entropies:
            entropy = torch.cat(entropies)
            metrics["attention_status"] = "measured"
            metrics["attention_entropy"] = float(entropy.mean())
            metrics["effective_attended_history_count"] = float(entropy.exp().mean())
        definitions = {
            "base_item_mean_cosine": "Off-diagonal mean cosine of at most 2048 base-table rows; directional alignment, not whole-model collapse.",
            "pos_neg_margin": "Complete positive-minus-negative scores on at most two batches of 32 public training users; small means margin below 1; not validation accuracy.",
            "margin_valid_positions": "Number of nonpadding positions summarized.",
            "pos_neg_margin_category_breakdown": "Empty: product categories are unavailable.",
            "attention_entropy": "Mean candidate-conditioned attention entropy in natural units on sampled positive and negative targets.",
            "effective_attended_history_count": "Mean exp(entropy), using the same target-conditioned weights.",
            "attention_status": "Measured, no_valid_positions, or unavailable for baseline/recency-only models.",
            "residual_scores": "Complete score minus base dot product on the same sampled targets; includes fixed alpha.",
            "residual_to_base_rms_ratio": "Residual RMS divided by base-score RMS, with denominator floor 1e-6.",
            "raw_branch_table_norms": "Raw row L2 norm summaries for the first at most 2048 nonpadding rows of each branch table.",
            "last_retained_branch_gradient_norms": "L2 norms of retained gradients from the last training backward pass; null if unavailable; no diagnostic backward pass.",
        }
        return {"metrics": metrics, "metric_definitions": definitions}
