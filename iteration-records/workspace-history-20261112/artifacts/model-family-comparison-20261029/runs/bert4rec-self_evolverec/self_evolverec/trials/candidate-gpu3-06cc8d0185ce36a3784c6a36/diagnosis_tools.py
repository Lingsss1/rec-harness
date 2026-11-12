import torch
import numpy as np
import torch.nn.functional as F
import json
import random
from collections import Counter

### >>> Self_EvolveRec-BLOCK-START: Interpret diagnostic availability and finite-suite mutation evidence


def diagnosis_interpreter_prompt(raw_diagnosis):
    return f"""
Interpret only these measured diagnostics and their definitions:
{json.dumps(raw_diagnosis, indent=2, allow_nan=False)}
Return JSON with status, core_findings, key_implications and evidence.
Use at most three findings and three implications. Cite observation counts,
statuses, denominators, tolerances and reasons. Distinguish measured FAIL from
execution ERROR and missing-evidence UNAVAILABLE. Use INSUFFICIENT_EVIDENCE when
contracts remain untested. Mutation coverage describes this finite fixture suite,
not general diagnostic accuracy. Repeated controls are dependent executions;
Wilson intervals are descriptive only. Surviving mutants are not equivalent.
Fixture results do not establish correctness of untested production paths.
Training margins and perturbation sensitivity are not host validation HR@10.
Embedding scores near 0.5 indicate approximately zero mean cosine, not collapse
or proven representation quality. Do not infer convergence, validation gains,
intent, ownership, compatibility, simulator traits or unavailable metadata.
No extra fitted predictors, score fusion or checkpoint averaging are allowed.
"""


# Keep the previous prompt implementation private for source compatibility.
def _legacy_diagnosis_interpreter_prompt(raw_diagnosis):
    ### <<< Self_EvolveRec-BLOCK-END
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
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the public ModelView constructor contract
    def __init__(self, view, device):
        self.model = view
        self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    ### <<< Self_EvolveRec-BLOCK-END

    def _get_categories(self, item_id: int):
        key = str(int(item_id))
        info = self.meta_data.get(key, None)
        if not isinstance(info, dict):
            return []
        cats = info.get("categories", [])
        if isinstance(cats, list):
            return [c for c in cats if isinstance(c, str)]
        return []

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()
            emb = emb[
                1 : self.model.model.itemnum + 1
            ]  # exclude padding and reserved mask tokens

            n = emb.size(0)
            if n <= 2:
                return 0.0

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx)
            x = F.normalize(x, p=2, dim=-1)

            gram = x @ x.t()
            offdiag_sum = gram.sum() - gram.diag().sum()
            offdiag_cnt = k * (k - 1)
            mean_cos = (offdiag_sum / offdiag_cnt).clamp(-1, 1).item()

            collapse_score = (mean_cos + 1.0) * 0.5
            return round(float(collapse_score), 4)

    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(
        self,
        dataloader,
        num_batches: int = 10,
        eps: float = 0.5,
        topk_categories: int = 20,
    ):
        self.model.eval()

        margins_all = []
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        for step, batch in enumerate(dataloader):
            if step >= num_batches:
                break
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 4):
                continue

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            seq_np = seq.cpu().numpy() if isinstance(seq, torch.Tensor) else seq
            log_feats = self.model.log2feats(seq_np)

            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)

            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
            margin = all_logits[..., 0] - all_logits[..., 1]

            valid = pos_t != 0
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            valid_cpu = pos_cpu != 0

            low_mask = valid_cpu & (margin_cpu < float(eps))
            wrong_mask = valid_cpu & (margin_cpu < 0.0)

            for b, l in np.argwhere(low_mask):
                low_total += 1
                pid = int(pos_cpu[b, l])
                nid = int(neg_cpu[b, l])
                for c in self._get_categories(pid):
                    low_pos_cat[c] += 1
                for c in self._get_categories(nid):
                    low_neg_cat[c] += 1

            for b, l in np.argwhere(wrong_mask):
                wrong_total += 1
                pid = int(pos_cpu[b, l])
                nid = int(neg_cpu[b, l])
                for c in self._get_categories(pid):
                    wrong_pos_cat[c] += 1
                for c in self._get_categories(nid):
                    wrong_neg_cat[c] += 1

        if len(margins_all) == 0:
            margin_metrics = {
                "mean": 0.0,
                "p10": 0.0,
                "neg_beats_pos_rate": 0.0,
                "small_margin_rate": 0.0,
            }
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

        breakdown = {
            "eps": float(eps),
            "low_margin_total_positions": int(low_total),
            "wrong_rank_total_positions": int(wrong_total),
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

    ### >>> Self_EvolveRec-BLOCK-START: Isolated dependency-light diagnostic execution and locked mutation fixtures
    def run_full_diagnosis(self, train_loader):
        return _DiagnosticAdapter(self.model, self.device).run(train_loader)

    ### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Read-only replay, bounded mutation fixtures and fault-specific completion reporting
class _DiagnosticAdapter:
    VERSION = "bert4rec-contract-fixtures-v1"
    ROUTES = (
        "Movies_and_TV",
        "Industrial_and_Scientific",
        "Electronics",
        "CDs_and_Vinyl",
    )
    REPEATS = 10
    CAP_SECONDS = 30.0

    def __init__(self, view, device):
        self.view = view
        self.net = view.model
        self.device = torch.device(device)
        self.metrics = {}
        self.definitions = {}
        self.started = None

    def result(self, status, values=None, count=0, reason=None, **extra):
        return dict(
            status=status,
            values=values,
            observations=int(count),
            reason=reason,
            **extra,
        )

    def measure(self, name, definition, function, dependencies=()):
        import time

        self.definitions[name] = definition
        blockers = {
            key: self.metrics[key]
            for key in dependencies
            if self.metrics[key]["status"] != "PASS"
        }
        if time.perf_counter() - self.started >= self.CAP_SECONDS:
            value = self.result("UNAVAILABLE", reason="Declared diagnostic cap reached")
        elif blockers:
            value = self.result(
                "UNAVAILABLE", reason="Prerequisites did not pass", blockers=blockers
            )
        else:
            try:
                value = function()
            except Exception as exc:
                value = self.result("ERROR", reason=f"{type(exc).__name__}: {exc}")
        value["dependencies"] = list(dependencies)
        self.metrics[name] = value
        return value

    @staticmethod
    def contract(batch, itemnum, maxlen):
        if not isinstance(batch, (tuple, list)) or len(batch) != 4:
            return False
        seq, pos, neg = [torch.as_tensor(x).detach().cpu() for x in batch[1:]]
        if (
            seq.ndim != 2
            or seq.shape != pos.shape
            or seq.shape != neg.shape
            or seq.shape[0] == 0
            or seq.shape[1] != maxlen
        ):
            return False
        if any(x.dtype != torch.long for x in (seq, pos, neg)):
            return False
        valid = pos.ne(0)
        return bool(
            ((seq >= 0) & (seq <= itemnum + 1)).all()
            and valid.any(1).all()
            and seq[valid].eq(itemnum + 1).all()
            and ((pos[valid] >= 1) & (pos[valid] <= itemnum)).all()
            and ((neg[valid] >= 1) & (neg[valid] <= itemnum)).all()
            and pos[valid].ne(neg[valid]).all()
            and neg[~valid].eq(0).all()
        )

    @staticmethod
    def ranking(scores, ids):
        return sorted(range(len(ids)), key=lambda i: (-float(scores[i]), int(ids[i])))

    @staticmethod
    def discrepancies(reference, actual):
        if reference.shape != actual.shape:
            return False, {"shape_mismatch": True}
        finite = bool(torch.isfinite(reference).all() and torch.isfinite(actual).all())
        difference = (reference - actual).abs()
        return finite and torch.allclose(reference, actual, atol=1e-5, rtol=1e-4), {
            "max_absolute_error": float(difference.max()) if finite else None,
            "max_relative_error": (
                float((difference / reference.abs().clamp_min(1e-8)).max())
                if finite
                else None
            ),
        }

    def replay(self, loader):
        # Never call iter() on an unknown loader: parent RNG restoration cannot
        # restore worker queues, a custom sampler cursor or dataset side effects.
        source = (
            loader
            if isinstance(loader, (tuple, list))
            else getattr(loader, "diagnostic_replay_batches", None)
        )
        if not isinstance(source, (tuple, list)):
            return self.result(
                "UNAVAILABLE",
                reason=(
                    "No explicit read-only replay cache. Supply a list/tuple of already "
                    "consumed training batches or diagnostic_replay_batches; live loader "
                    "iteration is intentionally unavailable."
                ),
            )
        self.batches = []
        for batch in source[:10]:
            if not isinstance(batch, (tuple, list)) or len(batch) != 4:
                return self.result("FAIL", reason="Malformed replay tuple")
            self.batches.append(
                (None, *[torch.as_tensor(x).detach().cpu().clone() for x in batch[1:]])
            )
        return self.result(
            "PASS" if self.batches else "UNAVAILABLE",
            count=len(self.batches),
            reason=None if self.batches else "Empty replay",
        )

    def sampler(self):
        verdicts = [
            self.contract(b, self.net.itemnum, self.net.maxlen) for b in self.batches
        ]
        return self.result(
            "PASS" if all(verdicts) else "FAIL",
            {
                "valid_batches": sum(verdicts),
                "invalid_batches": len(verdicts) - sum(verdicts),
                "supervised_positions": sum(
                    int(b[2].ne(0).sum()) for b in self.batches
                ),
            },
            len(verdicts),
        )

    def routing(self):
        # Category identity must be supplied explicitly by the host ModelView.
        category = getattr(self.view, "dataset_name", None)
        if category not in self.ROUTES:
            return self.result(
                "UNAVAILABLE",
                reason="Host ModelView.dataset_name is missing or invalid",
            )
        net_category = getattr(self.net, "dataset_name", None)
        if net_category is not None and net_category != category:
            return self.result("FAIL", reason="Host route disagrees with model route")
        self.category = category
        return self.result(
            "PASS",
            {"category": category, "source": "explicit host ModelView.dataset_name"},
            1,
        )

    def embeddings(self):
        weight = self.net.item_emb.weight.detach()[1 : self.net.itemnum + 1].float()
        count = min(2048, weight.shape[0])
        if count < 2:
            return self.result("UNAVAILABLE", reason="Fewer than two real items")
        generator = torch.Generator().manual_seed(42)
        ids = torch.randperm(weight.shape[0], generator=generator)[:count].to(
            weight.device
        )
        vectors = F.normalize(weight[ids], dim=-1)
        if not bool(torch.isfinite(vectors).all()):
            return self.result("FAIL", reason="Nonfinite real item embeddings")
        cosine = (vectors.sum(0).square().sum() - vectors.square().sum()) / (
            count * (count - 1)
        )
        return self.result(
            "PASS", {"mapped_mean_cosine": float((cosine.clamp(-1, 1) + 1) / 2)}, count
        )

    def margins(self):
        values = []
        for batch in self.batches:
            seq, pos, neg = [x.to(self.device) for x in batch[1:]]
            features = self.view.log2feats(seq.cpu().numpy().copy())
            scores = self.view.predict_score(features, torch.stack((pos, neg), -1))
            if scores.shape != (*pos.shape, 2):
                return self.result("FAIL", reason="ModelView scoring shape mismatch")
            margin = (scores[..., 0] - scores[..., 1])[pos.ne(0)].float().cpu()
            if not bool(torch.isfinite(margin).all()):
                return self.result("FAIL", reason="Nonfinite supervised logits")
            values.append(margin)
        values = torch.cat(values)
        return self.result(
            "PASS",
            {
                "mean": float(values.mean()),
                "p10": float(
                    values.kthvalue(max(1, math.ceil(0.1 * values.numel()))).values
                ),
                "negative_beats_positive_rate": float((values < 0).float().mean()),
                "small_margin_rate": float((values < 0.5).float().mean()),
            },
            values.numel(),
        )

    def fixtures(self):
        import copy
        import time
        from unittest.mock import patch

        # Fixtures execute disposable, never-trained copies. No deployed file or
        # live model method is mutated, and no fixture score reaches prediction.
        # Copy tensors directly to CPU, avoiding a transient second GPU model.
        memo = {}
        for module in self.net.modules():
            for value in (*module._parameters.values(), *module._buffers.values()):
                if value is not None and id(value) not in memo:
                    cloned = value.detach().cpu().clone()
                    memo[id(value)] = (
                        torch.nn.Parameter(cloned, requires_grad=value.requires_grad)
                        if isinstance(value, torch.nn.Parameter)
                        else cloned
                    )
        prototype = copy.deepcopy(self.net, memo).cpu().float().eval()
        if prototype.itemnum < 2 or prototype.maxlen < 4:
            return self.result(
                "UNAVAILABLE",
                reason="Discriminating left-padding fixture needs two items and maxlen >= 4",
            )
        # Copied hooks can retain references to live external state. Do not run
        # such copies without a separately audited isolation mechanism.
        if any(
            module._forward_hooks or module._forward_pre_hooks
            for module in prototype.modules()
        ):
            return self.result(
                "UNAVAILABLE",
                reason="Pre-existing forward hooks prevent fixture isolation",
            )
        fixture_seed = 73142
        generator = torch.Generator().manual_seed(fixture_seed)
        cases = [
            "count_minus_one",
            "reserved_mask",
            "visible_target",
            "checkpoint_consistency",
            "disabled_padding",
            "readout_and_padding",
        ]
        records = []

        def attention(disable=False):
            # Independent arithmetic oracle: zero Q/K means uniform attention;
            # identity V/output projections expose the contribution of PAD keys.
            module = torch.nn.MultiheadAttention(
                2, 1, dropout=0.0, batch_first=True
            ).eval()
            with torch.no_grad():
                module.in_proj_weight.zero_()
                module.in_proj_bias.zero_()
                module.in_proj_weight[4:6].copy_(torch.eye(2))
                module.out_proj.weight.copy_(torch.eye(2))
                module.out_proj.bias.zero_()
            inputs = torch.tensor([[[13.0, -7.0], [2.0, 4.0], [6.0, 8.0]]])
            inputs[:, 0] += torch.rand((1, 2), generator=generator) * 0.1
            mask = torch.tensor([[True, False, False]])
            actual = module(
                inputs,
                inputs,
                inputs,
                key_padding_mask=None if disable else mask,
                need_weights=False,
            )[0][:, 1:]
            expected = torch.tensor([[[4.0, 6.0], [4.0, 6.0]]])
            return bool(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5))

        def readout(fault=None):
            length = prototype.maxlen
            raw = torch.zeros((1, length), dtype=torch.long)
            raw[0, -2:] = torch.tensor([1, 2])
            # The oracle uses explicitly enumerated position codes, independent
            # of production readout/truncation/position-ID helpers.
            # A deterministic nonzero direction makes discrimination independent
            # of the trained embedding's magnitude or degeneracy.
            direction = torch.zeros(prototype.hidden_units)
            direction[0] = 1.0
            features = (
                torch.arange(1, length + 1).float()[None, :, None]
                * direction[None, None, :]
            )
            captured = []

            def encode(sequence):
                captured.append(sequence.clone())
                return features

            # Reuse only the disposable copy; context managers restore every
            # patched attribute even if the production scorer raises.
            with patch.object(
                prototype, "encode_sequence", side_effect=encode
            ), patch.object(
                prototype,
                "mask_token",
                0 if fault == "reserved_mask" else prototype.itemnum + 1,
            ):
                with torch.no_grad():
                    saved_item = prototype.item_emb.weight[1].clone()
                    prototype.item_emb.weight[1].copy_(direction)
                try:
                    if fault == "count_minus_one":
                        appended = torch.zeros_like(raw)
                        appended[:, :-1] = raw[:, 1:]
                        appended[:, -1] = prototype.mask_token
                        output = prototype.encode_sequence(appended)
                        wrong_index = appended.ne(0).sum(1) - 1
                        actual = (
                            (output[torch.arange(1), wrong_index] * direction)
                            .sum()
                            .reshape(1)
                        )
                    else:
                        actual = prototype.score_candidates(
                            raw, torch.tensor([1])
                        ).reshape(1)
                finally:
                    with torch.no_grad():
                        prototype.item_emb.weight[1].copy_(saved_item)
            expected = (features[0, length - 1] * direction).sum().reshape(1)
            expected_input = torch.zeros_like(raw)
            expected_input[0, -3:] = torch.tensor([1, 2, prototype.itemnum + 1])
            return bool(
                torch.equal(captured[0], expected_input)
                and torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
            )

        def check(name, mutated):
            if name == "count_minus_one":
                return readout("count_minus_one" if mutated else None)
            if name == "reserved_mask":
                return readout("reserved_mask" if mutated else None)
            if name == "disabled_padding":
                return attention(mutated)
            if name == "readout_and_padding":
                # Execute both component checks; do not short-circuit coverage.
                a, b = readout("count_minus_one" if mutated else None), attention(
                    mutated
                )
                return a and b
            if name == "visible_target":
                seq = torch.tensor([[0, 1, prototype.itemnum + 1]])
                pos, neg = torch.tensor([[0, 0, 2]]), torch.tensor([[0, 0, 1]])
                if mutated:
                    seq[0, -1] = 2
                return self.contract((None, seq, pos, neg), prototype.itemnum, 3)
            # This fixture exercises tensor identity, not serialized loading.
            # A shallow state mapping plus one cloned changed tensor suffices.
            original = prototype.state_dict()
            candidate = dict(original)
            if mutated:
                candidate["item_emb.weight"] = original["item_emb.weight"].clone()
                candidate["item_emb.weight"][1, 0].add_(1.0)
            return original.keys() == candidate.keys() and all(
                original[k].shape == candidate[k].shape
                and original[k].dtype == candidate[k].dtype
                and torch.equal(original[k], candidate[k])
                for k in original
            )

        for name in cases:
            if time.perf_counter() - self.started >= self.CAP_SECONDS:
                records.append(
                    {"fault": name, "outcome": "UNAVAILABLE", "reason": "Cap reached"}
                )
                continue
            row = {
                "fault": name,
                "controls_completed": 0,
                "false_alarms": 0,
                "nonequivalence": "not_established",
                "outcome": "ERROR",
            }
            try:
                for _ in range(self.REPEATS):
                    if time.perf_counter() - self.started >= self.CAP_SECONDS:
                        raise TimeoutError("Cap reached during unchanged controls")
                    accepted = check(name, False)
                    row["controls_completed"] += 1
                    row["false_alarms"] += int(not accepted)
                if row["false_alarms"]:
                    row["outcome"] = "CONTROL_FAILED"
                else:
                    # Structural witnesses do not depend on the detector verdict:
                    # wrong slot has a distinct code; PAD values change the mean;
                    # MASK/visible-label/weight mutations change exact identities.
                    if time.perf_counter() - self.started >= self.CAP_SECONDS:
                        raise TimeoutError("Cap reached before mutation execution")
                    row["nonequivalence"] = "structural_witness"
                    try:
                        accepted_mutant = check(name, True)
                    finally:
                        row["restoration_passed"] = check(name, False)
                    if not row["restoration_passed"]:
                        raise RuntimeError("Post-mutation unchanged control failed")
                    row["outcome"] = "SURVIVING" if accepted_mutant else "DETECTED"
            except TimeoutError as exc:
                row["outcome"] = "UNAVAILABLE"
                row["reason"] = str(exc)
            except Exception as exc:
                row["outcome"] = "ERROR"
                row["reason"] = f"{type(exc).__name__}: {exc}"
            records.append(row)
        completed = sum(r.get("controls_completed", 0) for r in records)
        alarms = sum(r.get("false_alarms", 0) for r in records)
        eligible = sum(r["outcome"] in {"DETECTED", "SURVIVING"} for r in records)
        detected = sum(r["outcome"] == "DETECTED" for r in records)
        interval = None
        if completed:
            z, rate = 1.959963984540054, alarms / completed
            denominator = 1 + z * z / completed
            center = (rate + z * z / (2 * completed)) / denominator
            radius = (
                z
                * math.sqrt(rate * (1 - rate) / completed + z * z / (4 * completed**2))
                / denominator
            )
            interval = [max(0.0, center - radius), min(1.0, center + radius)]
        outcomes = {row["outcome"] for row in records}
        suite_status = (
            "ERROR"
            if "ERROR" in outcomes
            else (
                "FAIL"
                if outcomes & {"CONTROL_FAILED", "SURVIVING"}
                else "UNAVAILABLE" if "UNAVAILABLE" in outcomes else "PASS"
            )
        )
        return self.result(
            suite_status,
            {
                "records": records,
                "constructed": len(cases),
                "eligible_nonequivalent": eligible,
                "detected": detected,
                "surviving": eligible - detected,
                "demonstrably_equivalent": 0,
                "unresolved": sum(r["outcome"] == "CONTROL_FAILED" for r in records),
                "unavailable": sum(r["outcome"] == "UNAVAILABLE" for r in records),
                "errored": sum(r["outcome"] == "ERROR" for r in records),
                "detection_coverage": detected / eligible if eligible else None,
                "control_false_positive_numerator": alarms,
                "control_false_positive_denominator": completed,
                "control_false_positive_rate": (
                    alarms / completed if completed else None
                ),
                "descriptive_wilson95": interval,
                "fixture_seed": fixture_seed,
                "scope": "Readout uses a copied production scorer with synthetic encoder output; attention and checkpoint checks are standalone adapter fixtures. No claim of production attention or exported-loader mutation sensitivity.",
            },
            len(records),
        )

    def run(self, loader):
        import math as math_module
        import time

        # Existing module has no math dependency; make it available only inside
        # adapter method globals without importing SciPy or other optional tools.
        global math
        math = math_module
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        modes = [(module, module.training) for module in self.net.modules()]
        self.started = time.perf_counter()
        try:
            self.metrics["protocol"] = self.result(
                "PASS",
                {
                    "version": self.VERSION,
                    "repeat_count": self.REPEATS,
                    "cap_seconds": self.CAP_SECONDS,
                    "torch_version": str(torch.__version__),
                    "device": str(self.device),
                    "precision": "float32; autocast disabled",
                    "serving_tolerances": {"atol": 1e-5, "rtol": 1e-4},
                    "fixture_tolerances": {"atol": 1e-6, "rtol": 1e-5},
                    "development_calibration": "Not performed; thresholds preregistered, not calibrated",
                    "checkpoint_hash": None,
                    "production_state_policy": "Live network forwards withheld; fixtures use disposable copies",
                },
                1,
            )
            self.definitions["protocol"] = (
                "Versioned finite diagnostic protocol; no runtime or validation improvement is presumed."
            )
            with torch.no_grad(), torch.autocast(
                device_type=self.device.type, enabled=False
            ):
                self.measure(
                    "replay",
                    "At most ten independently cloned cached batches; never advances a live loader.",
                    lambda: self.replay(loader),
                )
                self.measure(
                    "routing",
                    "Explicit host category identity checked against the model attribute when supplied.",
                    self.routing,
                )
                self.measure(
                    "sampler_contract",
                    "Reserved MASK, real labels, B x L int64 arrays and all-MASK supervision; does not verify provenance or negative distribution.",
                    self.sampler,
                    ("replay",),
                )
                self.measure(
                    "embedding_alignment",
                    "Mapped ordered off-diagonal mean cosine over up to 2048 real embeddings, seed 42. 0.5 means approximately zero cosine.",
                    self.embeddings,
                )
                self.measure(
                    "mutation_suite",
                    "Ten dependent unchanged controls per fixture, then one isolated mutant and restoration control. Coverage excludes execution errors and missing evidence; Wilson interval is descriptive.",
                    self.fixtures,
                )
                for name, reason in {
                    "serving_parity": "Serialized export, authoritative vocabulary manifest and isolated ModelView forward state are not supplied to this adapter; numerical parity is unmeasured.",
                    "training_margins": "Live ModelView forwards may execute arbitrary hooks or mutate state; no audited disposable ModelView factory was supplied.",
                    "prefix_provenance": "Replay tensors alone cannot establish source-window provenance.",
                    "negative_distribution": "Replay tensors alone cannot establish proposal probabilities.",
                    "history_sensitivity": "Complete serialized-serving parity has not passed.",
                    "residual_patching": "Core serialized-serving and state-isolation prerequisites have not passed.",
                    "item_category_breakdown": "Item categories are unavailable in this benchmark.",
                }.items():
                    self.definitions[name] = reason
                    self.metrics[name] = self.result("UNAVAILABLE", reason=reason)
        finally:
            # No production forwards, hooks, mode changes or buffers were mutated.
            # Restore each flag individually and all controlled default RNGs.
            for module, training in modes:
                module.training = training
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.metrics["diagnostic_cost"] = self.result(
                "PASS",
                {
                    "elapsed_seconds": time.perf_counter() - self.started,
                    "cap_seconds": self.CAP_SECONDS,
                    "cap_is_soft": True,
                    "limitation": "Checked between probes and fixture cases; in-flight operations can overrun.",
                },
                1,
            )
            self.definitions["diagnostic_cost"] = (
                "Wall time including copying, fixtures and synchronization; no optimizer step or training is performed."
            )
        self.metrics["status"] = "INSUFFICIENT_EVIDENCE"
        return {"metrics": self.metrics, "metric_definitions": self.definitions}


### <<< Self_EvolveRec-BLOCK-END
