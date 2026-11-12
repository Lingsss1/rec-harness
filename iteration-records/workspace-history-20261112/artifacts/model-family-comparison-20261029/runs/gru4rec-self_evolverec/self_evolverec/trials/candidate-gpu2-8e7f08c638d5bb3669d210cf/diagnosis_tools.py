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
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the host ModelView contract and unavailable metadata boundary
    def __init__(self, view, device):
        self.view = view
        self.model = view.model
        self.device = device
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            ### >>> Self_EvolveRec-BLOCK-START: Measure real item embeddings with explicit unavailable results
            emb = self.view.item_emb.weight.detach()
            emb = emb[1 : self.model.itemnum + 1]
            if not torch.isfinite(emb).all().item():
                raise FloatingPointError("Nonfinite diagnostic item embeddings")
            n = emb.size(0)
            if min(int(sample_size), n) < 2:
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx)
            x = F.normalize(x, p=2, dim=-1)

            ### >>> Self_EvolveRec-BLOCK-START: Exact mean cosine without a quadratic Gram allocation
            x = x.float()
            offdiag_sum = x.sum(dim=0).square().sum() - x.square().sum()
            offdiag_cnt = k * (k - 1)
            mean_cos = (offdiag_sum / offdiag_cnt).clamp(-1, 1).item()
            ### <<< Self_EvolveRec-BLOCK-END

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
        ### >>> Self_EvolveRec-BLOCK-START: Restore the actual module mode after margin measurements
        was_training = self.model.training
        self.model.eval()
        try:
            return self._measure_margins(dataloader, num_batches, eps, topk_categories)
        finally:
            self.model.train(was_training)

    def _measure_margins(self, dataloader, num_batches, eps, topk_categories):
        margins_all = []
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        for step, batch in enumerate(dataloader):
            if step >= num_batches:
                break
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 4):
                continue

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            ### >>> Self_EvolveRec-BLOCK-START: Family-specific aligned features and complete diagnostic validity masks
            seq_t, pos_t, neg_t = [
                torch.as_tensor(value, device=self.device, dtype=torch.long)
                for value in (seq, pos, neg)
            ]
            if seq_t.ndim != 2 or any(
                value.shape != seq_t.shape for value in (pos_t, neg_t)
            ):
                raise ValueError("Diagnostic inputs must be aligned B x L tensors")
            if any(
                ((value < 0) | (value > self.model.itemnum)).any().item()
                for value in (seq_t, pos_t, neg_t)
            ):
                raise ValueError("Diagnostic IDs exceed the trained vocabulary")
            log_feats = self.view.log2feats(seq_t)
            if log_feats.ndim != 3 or log_feats.shape[:2] != seq_t.shape:
                raise ValueError("log2feats must return aligned B x L x H features")
            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.view.predict_score(log_feats, all_items)
            if all_logits.shape != all_items.shape:
                raise ValueError("predict_score must return aligned B x L x 2 scores")
            margin = all_logits[..., 0] - all_logits[..., 1]
            valid = seq_t.ne(0) & pos_t.ne(0) & neg_t.ne(0)
            if valid.any():
                measured = margin[valid].detach().float()
                if not torch.isfinite(measured).all().item():
                    raise FloatingPointError("Nonfinite valid diagnostic margins")
                margins_all.append(measured.cpu())
            ### <<< Self_EvolveRec-BLOCK-END

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Use the same valid positions for counts and margins
            valid_cpu = valid.detach().cpu().numpy()
            ### <<< Self_EvolveRec-BLOCK-END

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
            ### >>> Self_EvolveRec-BLOCK-START: Missing measurements are unavailable rather than healthy zeros
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

        ### >>> Self_EvolveRec-BLOCK-START: Report measured counts without unavailable category attributes
        breakdown = {
            "eps": float(eps),
            "valid_positions": sum(int(m.numel()) for m in margins_all),
            "low_margin_total_positions": int(low_total),
            "wrong_rank_total_positions": int(wrong_total),
            "category_metadata_available": False,
            "low_margin_top_categories": None,
            "wrong_rank_top_categories": None,
        }
        ### <<< Self_EvolveRec-BLOCK-END

        return margin_metrics, breakdown

    ### >>> Self_EvolveRec-BLOCK-START: Localize padding numerics with independent float64 recurrence and gradient references
    def analyze_float64_reference(self):
        # Unfitted copies of identical parameters, used only for diagnosis.
        import copy

        if self.model.gru.bidirectional or not self.model.gru.batch_first:
            raise ValueError(
                "Reference requires the supplied unidirectional batch-first GRU"
            )
        with torch.inference_mode(False), torch.enable_grad():
            gru = copy.deepcopy(self.model.gru).cpu().double().eval()
            cells = []
            # GRUCell construction initializes parameters before we overwrite
            # them. Preserve the caller's CPU RNG stream during construction.
            with torch.random.fork_rng(devices=[]):
                for layer in range(gru.num_layers):
                    cell = torch.nn.GRUCell(
                        gru.input_size if layer == 0 else gru.hidden_size,
                        gru.hidden_size,
                        bias=gru.bias,
                        dtype=torch.float64,
                        device="cpu",
                    )
                    with torch.no_grad():
                        for name, parameter in cell.named_parameters():
                            parameter.copy_(getattr(gru, f"{name}_l{layer}"))
                    cells.append(cell)

            valid = torch.tensor(
                [
                    [False, False, True, True, True],
                    [False, False, False, False, True],
                    [True, True, True, True, True],
                    [False, True, False, True, True],
                ]
            )
            batch, width = valid.shape
            values = torch.arange(batch * width * gru.input_size, dtype=torch.float64)
            values = torch.sin(values.reshape(batch, width, gru.input_size) * 0.017)
            initial = (
                torch.cos(
                    torch.arange(
                        gru.num_layers * batch * gru.hidden_size, dtype=torch.float64
                    ).reshape(gru.num_layers, batch, gru.hidden_size)
                    * 0.013
                )
                * 0.1
            )
            x = values.clone().requires_grad_(True)
            y = values.clone().requires_grad_(True)
            h = initial.clone().requires_grad_(True)
            h_reference = initial.clone().requires_grad_(True)
            rows, columns = valid.nonzero(as_tuple=True)
            compact_columns = valid.long().cumsum(dim=1)[rows, columns] - 1
            compact = torch.zeros_like(x)
            compact[rows, compact_columns] = x[rows, columns]
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                compact, valid.sum(dim=1), batch_first=True, enforce_sorted=False
            )
            packed_output, final = gru(packed, h)
            unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True, total_length=width
            )
            aligned = torch.zeros_like(unpacked)
            aligned[rows, columns] = unpacked[rows, compact_columns]

            states = list(h_reference.unbind(dim=0))
            outputs = []
            for time in range(width):
                layer_input = y[:, time]
                for layer, cell in enumerate(cells):
                    proposed = cell(layer_input, states[layer])
                    states[layer] = torch.where(
                        valid[:, time, None], proposed, states[layer]
                    )
                    layer_input = states[layer]
                outputs.append(
                    torch.where(
                        valid[:, time, None], layer_input, torch.zeros_like(layer_input)
                    )
                )
            reference = torch.stack(outputs, dim=1)
            reference_final = torch.stack(states, dim=0)
            loss = aligned.square().sum() + final.square().sum()
            reference_loss = reference.square().sum() + reference_final.square().sum()
            parameters = list(gru.named_parameters())
            reference_parameters = {
                f"{name}_l{layer}": parameter
                for layer, cell in enumerate(cells)
                for name, parameter in cell.named_parameters()
            }
            gradients = torch.autograd.grad(
                loss, [x, h] + [parameter for _, parameter in parameters]
            )
            reference_gradients = torch.autograd.grad(
                reference_loss,
                [y, h_reference]
                + [reference_parameters[name] for name, _ in parameters],
            )
            comparisons = {}

            def record(name, actual, expected, atol=1e-9, rtol=1e-7):
                actual, expected = actual.detach(), expected.detach()
                difference = (actual - expected).abs()
                finite = torch.isfinite(actual) & torch.isfinite(expected)
                failed = (~finite) | (difference > atol + rtol * expected.abs())
                comparisons[name] = {
                    "elements": actual.numel(),
                    "nonfinite_elements": int((~finite).sum().item()),
                    "max_abs_error": (
                        float(difference[finite].max().item()) if finite.any() else None
                    ),
                    "failures": int(failed.sum().item()),
                }

            record("aligned_outputs", aligned, reference)
            record("every_layer_final_state", final, reference_final)
            record("scalar_loss", loss, reference_loss)
            names = ["continuous_input_gradient", "initial_state_gradient"]
            names.extend(f"parameter_gradient:{name}" for name, _ in parameters)
            for name, actual, expected in zip(names, gradients, reference_gradients):
                record(name, actual, expected)
            for name, gradient in (
                ("packed", gradients[0]),
                ("masked", reference_gradients[0]),
            ):
                record(
                    f"{name}_padding_input_gradient",
                    gradient[~valid],
                    torch.zeros_like(gradient[~valid]),
                    rtol=0.0,
                )
            return {
                "atol": 1e-9,
                "rtol": 1e-7,
                "comparisons": comparisons,
                "passed": all(entry["failures"] == 0 for entry in comparisons.values()),
                "all_empty_gradient_path": "excluded: packing requires nonempty rows",
                "scope": "fixed synthetic continuous inputs; identical unfitted GRU weights",
            }

    @torch.no_grad()
    def analyze_padding_localization(self, batches, max_rows=32):
        # The supplied production encoder already compacts, packs and scatters
        # valid tokens correctly. This probe changes no training or inference code.
        if not isinstance(max_rows, int) or max_rows < 1:
            raise ValueError("max_rows must be a positive integer")
        absolute_tolerance = 1e-5
        relative_tolerance = 1e-4
        records = {}
        counts = {"rows": 0, "empty_rows": 0, "valid_positions": 0}
        failures = 0
        nonfinite = 0

        def compare(
            name, actual, reference, atol=absolute_tolerance, rtol=relative_tolerance
        ):
            nonlocal failures, nonfinite
            a = actual.detach().double()
            b = reference.detach().double()
            if a.shape != b.shape:
                raise ValueError(f"Shape mismatch in {name}: {a.shape}, {b.shape}")
            if not a.numel():
                return
            finite = torch.isfinite(a) & torch.isfinite(b)
            bad = int((~finite).sum().item())
            nonfinite += bad
            difference = (a - b).abs()
            entry = records.setdefault(
                name,
                {
                    "elements": 0,
                    "nonfinite_elements": 0,
                    "max_abs_error": None,
                    "max_relative_error": None,
                    "absolute_threshold_failures": 0,
                    "scale_aware_failures": 0,
                },
            )
            entry["elements"] += a.numel()
            entry["nonfinite_elements"] += bad
            absolute_failed = (~finite) | (difference > atol)
            scaled_failed = (~finite) | (difference > atol + rtol * b.abs())
            entry["absolute_threshold_failures"] += int(absolute_failed.sum().item())
            entry["scale_aware_failures"] += int(scaled_failed.sum().item())
            failures += int(absolute_failed.sum().item())
            if finite.any():
                maximum = float(difference[finite].max().item())
                relative = float(
                    (difference[finite] / b[finite].abs().clamp_min(1e-12)).max().item()
                )
                old = entry["max_abs_error"]
                entry["max_abs_error"] = maximum if old is None else max(old, maximum)
                old = entry["max_relative_error"]
                entry["max_relative_error"] = (
                    relative if old is None else max(old, relative)
                )

        was_training = self.model.training
        self.model.eval()
        try:
            # Fixed candidates are real catalog IDs, used only for numerical
            # comparisons. They are not an unseen-item ranking evaluation.
            candidates = torch.arange(
                1, min(int(self.model.itemnum), 32) + 1, device=self.device
            )
            for batch in batches:
                remaining = max_rows - counts["rows"]
                if remaining <= 0:
                    break
                sequence = torch.as_tensor(
                    batch[1][:remaining], device=self.device, dtype=torch.long
                )
                if sequence.size(0) == 0 or sequence.size(1) == 0:
                    continue
                features = self.view.log2feats(sequence)
                valid = sequence.ne(0)
                counts["rows"] += sequence.size(0)
                counts["empty_rows"] += int((~valid.any(dim=1)).sum().item())
                counts["valid_positions"] += int(valid.sum().item())
                compare(
                    "padding_outputs",
                    features[~valid],
                    torch.zeros_like(features[~valid]),
                    atol=0.0,
                    rtol=0.0,
                )
                padded = F.pad(sequence, (3, 0))
                compare(
                    "extra_leading_padding",
                    self.view.log2feats(padded)[:, 3:],
                    features,
                )
                permutation = torch.arange(
                    sequence.size(0) - 1, -1, -1, device=self.device
                )
                compare(
                    "batch_permutation",
                    self.view.log2feats(sequence[permutation]),
                    features[permutation],
                )
                compare("repeat_execution", self.view.log2feats(sequence), features)
                empty = torch.zeros_like(sequence)
                compare(
                    "all_empty_outputs",
                    self.view.log2feats(empty),
                    torch.zeros_like(features),
                    atol=0.0,
                    rtol=0.0,
                )
                compare(
                    "all_empty_readout",
                    self.model.encode(empty),
                    torch.zeros_like(features[:, 0]),
                    atol=0.0,
                    rtol=0.0,
                )
                for row in range(sequence.size(0)):
                    positions = valid[row].nonzero(as_tuple=True)[0]
                    if positions.numel() == 0:
                        continue
                    tokens = sequence[row, positions].unsqueeze(0)
                    compact = self.view.log2feats(tokens)[0]
                    compare(
                        "individual_unpadded_valid_features",
                        features[row, positions],
                        compact,
                    )
                    compare(
                        "final_nonpadding_readout",
                        self.model.encode(sequence[row : row + 1])[0],
                        compact[-1],
                    )
                    # Direct dense unpadded GRU is independent of the production
                    # compaction/scattering implementation. Dropout is disabled.
                    direct, _ = self.model.gru(self.model.item_emb(tokens))
                    compare("direct_unpadded_recurrence", compact, direct[0])
                    original_scores = self.view.predict_score(
                        features[row, positions[-1]].unsqueeze(0),
                        candidates.unsqueeze(0),
                    )
                    compact_scores = self.view.predict_score(
                        compact[-1].unsqueeze(0), candidates.unsqueeze(0)
                    )
                    compare("candidate_scores", original_scores, compact_scores)
                    compare(
                        "production_candidate_scores",
                        self.model.score_candidates(
                            sequence[row : row + 1], candidates
                        ),
                        compact_scores.squeeze(0),
                    )
                    if tokens.size(1) > 1:
                        prefix = tokens[:, :-1]
                        compare(
                            "causal_prefix_features",
                            self.view.log2feats(prefix)[0],
                            compact[:-1],
                        )
        finally:
            self.model.train(was_training)
        return {
            "counts": counts,
            "comparisons": records,
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
            "relative_denominator_floor": 1e-12,
            "nonfinite_elements": nonfinite,
            "padding_check_passed": (
                failures == 0 if counts["valid_positions"] > 0 else None
            ),
            "device": str(self.device),
            "float64_reference_and_gradient_checks": self.analyze_float64_reference(),
            "checkpoint_and_generated_predictor_checks": None,
            "validation_ranking_effect": None,
            "production_encoding_changed": False,
        }

    @torch.no_grad()
    def run_full_diagnosis(self, train_loader):
        print(">>> Running GRU4Rec margin and padding-localization diagnosis...")
        was_training = self.model.training
        self.model.eval()
        try:
            batches = []
            iterator = iter(train_loader)
            for _ in range(10):
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                if not isinstance(batch, (tuple, list)) or len(batch) != 4:
                    raise ValueError("Expected (None, sequence, positive, negative)")
                values = tuple(
                    torch.as_tensor(value)
                    .detach()
                    .to(device="cpu", dtype=torch.long)
                    .clone()
                    for value in batch[1:4]
                )
                if values[0].ndim != 2 or any(
                    value.shape != values[0].shape for value in values[1:]
                ):
                    raise ValueError("Diagnostic inputs must be aligned B x L tensors")
                if any(
                    ((value < 0) | (value > self.model.itemnum)).any().item()
                    for value in values
                ):
                    raise ValueError("Diagnostic IDs exceed the trained vocabulary")
                batches.append((None, *values))
            collapse_score = self.analyze_embeddings_collapse_cosine(sample_size=2048)
            eps = 0.5
            margin_metrics, breakdown = (
                self.analyze_pos_neg_margin_with_category_breakdown(
                    batches, num_batches=10, eps=eps, topk_categories=20
                )
            )
            padding_metrics = self.analyze_padding_localization(batches)
        finally:
            self.model.train(was_training)
        ### <<< Self_EvolveRec-BLOCK-END
        return {
            "metrics": {
                "embedding_collapse_score": collapse_score,
                "pos_neg_margin_mean": margin_metrics["mean"],
                "pos_neg_margin_p10": margin_metrics["p10"],
                "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                ### >>> Self_EvolveRec-BLOCK-START: Return measured localization alongside sampled margins
                "pos_neg_margin_category_breakdown": breakdown,
                "padding_localization": padding_metrics,
                ### <<< Self_EvolveRec-BLOCK-END
            },
            "metric_definitions": {
                "embedding_collapse_score": (
                    ### >>> Self_EvolveRec-BLOCK-START: Distinguish embedding alignment from recommendation diversity
                    "Mean off-diagonal cosine of at most 2048 real item embeddings, mapped "
                    "from [-1,1] to [0,1]. About 0.5 means zero mean cosine; larger values "
                    "indicate directional alignment. This does not measure recommendation "
                    "diversity or unobserved simulator traits. Null means insufficient pairs."
                    ### <<< Self_EvolveRec-BLOCK-END
                ),
                "pos_neg_margin_mean": "Mean of (pos_logit - neg_logit) over valid positions; higher means clearer separation.",
                "pos_neg_margin_p10": "10th percentile of (pos_logit - neg_logit); lower suggests many hard/confusing cases.",
                "pos_neg_neg_beats_pos_rate": "Fraction where neg_logit > pos_logit (margin < 0); lower is better.",
                "pos_neg_small_margin_rate": "Fraction where (pos_logit - neg_logit) < eps; higher means weak separation.",
                ### >>> Self_EvolveRec-BLOCK-START: Define diagnostic scope and separate numerical findings from semantic defects
                "pos_neg_margin_category_breakdown": (
                    "Counts valid, low-margin and wrong-order positions. Validity requires "
                    "nonzero input, positive and negative IDs. Category metadata is unavailable, "
                    "so category breakdowns are null. Margin statistics are null without valid "
                    "samples. These are conditional sampled training diagnostics, not HR@10 "
                    "or full-catalog ranking measurements."
                ),
                "padding_localization": (
                    "Eval-mode numerical comparisons on at most the first 32 cached rows, "
                    "with dropout disabled and original module mode restored. No sampling RNG "
                    "is consumed. All aligned valid positions are compared with individual "
                    "unpadded execution and direct unpadded native GRU recurrence. Additional "
                    "comparisons cover extra leading padding, reversed batch order, repeated "
                    "execution, exact zero padding outputs, all-empty outputs/readouts, "
                    "maximum-nonpadding-index readout, production candidate scores and "
                    "causal prefixes. Candidate IDs are the first at most 32 real catalog IDs; "
                    "they are numerical controls, not an unseen recommendation evaluation. "
                    "Each comparison reports element counts, nonfinite counts, maximum absolute "
                    "and relative errors, failures of the absolute threshold and failures of "
                    "abs(a-b)<=atol+rtol*abs(b). Absolute tolerance is 1e-5 and separately "
                    "declared relative tolerance is 1e-4; exact-zero controls use zero "
                    "tolerances. Relative error divides by max(abs(reference),1e-12). "
                    "padding_check_passed requires every absolute comparison to pass and no "
                    "nonfinite elements; no real input positions yields null. These finite-sample "
                    "checks localize discrepancies but do not prove recurrence correctness. "
                    "A CUDA-only discrepancy does not establish a semantic encoding defect. "
                    "The separate float64 CPU reference uses identical unfitted GRU and GRUCell "
                    "parameters, disabled dropout, fixed continuous inputs, nonzero initial "
                    "states, unsorted/tied lengths, singleton/full histories and internal gaps. "
                    "It compares aligned outputs, every layer's final state, a common scalar "
                    "loss, continuous-input gradients, initial-state gradients and every GRU "
                    "parameter gradient using atol=1e-9 and rtol=1e-7. Padding input gradients "
                    "are checked against zero with atol=1e-9. All-empty packing gradients are "
                    "explicitly excluded. This reference does not test the production scatter "
                    "implementation's gradients or stochastic training equivalence. Checkpoint/"
                    "export comparisons and validation ranking effects remain unmeasured and null. The "
                    "supplied canonical production encoder, predictor writer, sampler, loss, "
                    "optimizer and checkpoint selection are retained. No correction arm or "
                    "validation improvement is claimed, and no predictors are combined."
                    # Review summary: checked supplied signatures, tensor alignment,
                    # reference parameter mapping, gradient execution under no_grad,
                    # RNG isolation, empty-data reporting and workflow invocation.
                    # Production encoding is already canonical and remains unchanged.
                    # No execution or validation measurements were performed here.
                ),
                ### <<< Self_EvolveRec-BLOCK-END
            },
        }
