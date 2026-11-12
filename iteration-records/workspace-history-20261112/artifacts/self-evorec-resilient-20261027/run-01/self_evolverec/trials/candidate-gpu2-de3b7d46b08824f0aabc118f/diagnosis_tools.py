import torch
import numpy as np
import torch.nn.functional as F
import json
import random
from collections import Counter


def diagnosis_interpreter_prompt(raw_diagnosis):
    ### >>> Self_EvolveRec-BLOCK-START: Include diagnostic scope and missing evidence in interpretation
    context = json.dumps(
        {
            "observation_counts": raw_diagnosis.get("observation_counts"),
            "measurement_context": raw_diagnosis.get("measurement_context"),
        },
        indent=2,
        allow_nan=False,
    )
    prefix = (
        f"Measurement context and observation counts:\n{context}\n"
        "Null means unavailable evidence, not measured zero. "
        "Check observation counts and nonfinite exclusions before interpreting averages. "
        "Training pairwise margins do not establish validation HR@10 improvements. "
        "Scaled margins cannot be compared directly with raw neural margins. "
        "Unavailable deployed scores do not establish that transition corrections were implemented. "
        "Do not infer user diversity/conformity or recommendation relevance from embedding alignment.\n"
    )
    ### <<< Self_EvolveRec-BLOCK-END
    return prefix + f"""
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


### >>> Self_EvolveRec-BLOCK-START: Validate diagnostic inputs and compare neural and deployed margins on matched observations
class DiagnosisProbe:
    def __init__(self, view, device):
        self.view = view
        self.model = view.model
        self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train
        self.deployed_score = getattr(view, "score", None)
        if not callable(self.deployed_score):
            self.deployed_score = getattr(self.model, "score", None)
        if not callable(self.deployed_score):
            self.deployed_score = None
        # Source support counts adjacent occurrences, including repeated pairs.
        # Never join histories or remove padding before constructing adjacency.
        if self.user_train is not None and not isinstance(self.user_train, dict):
            raise TypeError("view.user_train must be a dictionary or None")
        histories = (
            self.user_train.get("History")
            if isinstance(self.user_train, dict)
            else None
        )
        if histories is not None and not isinstance(histories, dict):
            raise TypeError("user_train['History'] must be a dictionary")
        self.source_support = None
        self.training_transition_occurrences = None
        if isinstance(histories, dict):
            self.source_support = Counter()
            self.training_transition_occurrences = 0
            itemnum = int(self.view.item_emb.weight.shape[0]) - 1
            for history in histories.values():
                for source, destination in zip(history, history[1:]):
                    source, destination = int(source), int(destination)
                    if 1 <= source <= itemnum and 1 <= destination <= itemnum:
                        self.source_support[source] += 1
                        self.training_transition_occurrences += 1

    @torch.no_grad()
    def analyze_embeddings_collapse_cosine(self, sample_size=2048, seed=42):
        # Local RNG: diagnostic sampling must not change training randomness.
        emb = self.view.item_emb.weight.detach()[1:]
        count = min(max(int(sample_size), 0), emb.shape[0])
        self.embedding_observation_counts = {
            "catalog_nonpadding_embeddings": int(emb.shape[0]),
            "sampled_embeddings": int(count),
            "usable_embeddings": 0,
            "excluded_embeddings": 0,
        }
        if count == 0:
            return None
        rng = np.random.default_rng(seed)
        indices = torch.as_tensor(
            rng.choice(emb.shape[0], count, replace=False),
            dtype=torch.long,
            device=emb.device,
        )
        # Convert before computing norms: float32 norms can overflow or underflow.
        sampled = emb.index_select(0, indices).double()
        norms = sampled.norm(dim=-1)
        usable = torch.isfinite(sampled).all(-1) & torch.isfinite(norms) & norms.gt(0)
        self.embedding_observation_counts = {
            "catalog_nonpadding_embeddings": int(emb.shape[0]),
            "sampled_embeddings": int(count),
            "usable_embeddings": int(usable.sum().item()),
            "excluded_embeddings": int((~usable).sum().item()),
        }
        sampled = sampled[usable]
        count = sampled.shape[0]
        if count < 2:
            return None
        x = sampled / norms[usable].unsqueeze(-1)
        # Sum of off-diagonal dot products without a quadratic Gram matrix.
        total = x.sum(0).square().sum() - x.square().sum()
        mean_cosine = (total / (count * (count - 1))).clamp(-1, 1)
        return float(((mean_cosine + 1) / 2).item())

    @staticmethod
    def _summarize(chunks, eps):
        names = ("mean", "p10", "neg_beats_pos_rate", "small_margin_rate", "tie_rate")
        if not chunks:
            return dict.fromkeys(names, None)
        margins = torch.cat(chunks)
        k = max(1, int(np.ceil(0.10 * margins.numel())))
        return {
            "mean": float(margins.mean().item()),
            "p10": float(margins.kthvalue(k).values.item()),
            "neg_beats_pos_rate": float((margins < 0).double().mean().item()),
            "small_margin_rate": float((margins < eps).double().mean().item()),
            "tie_rate": float((margins == 0).double().mean().item()),
        }

    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(
        self, dataloader, num_batches=10, eps=0.5, topk_categories=20
    ):
        from itertools import islice

        if not np.isfinite(float(eps)):
            raise ValueError("Diagnostic margin threshold must be finite")
        # Iterating a training loader can consume global RNG state even in eval mode.
        # Restore the RNGs used by this single-device workflow on success or failure.
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.random.get_rng_state()
        cuda_rng = (
            torch.cuda.get_rng_state(self.device)
            if self.device.type == "cuda"
            else None
        )
        chunks = {"neural": [], "deployed": []}
        paired_chunks = []
        itemnum = int(self.view.item_emb.weight.shape[0]) - 1
        support_names = ("n_a=0", "n_a=1-4", "n_a>=5")
        support_chunks = {
            name: {"neural": [], "deployed": []} for name in support_names
        }
        support_counts = dict.fromkeys(support_names, 0)
        counts = {
            "batches": 0,
            "valid_positions": 0,
            "neural_finite_positions": 0,
            "deployed_finite_positions": 0,
            "neural_nonfinite_positions": 0,
            "deployed_nonfinite_positions": 0,
        }
        # Preserve mixed submodule modes too, including when scoring raises.
        modes = [(module, module.training) for module in self.model.modules()]
        try:
            self.model.eval()
            for batch in islice(dataloader, max(int(num_batches), 0)):
                if not isinstance(batch, (list, tuple)) or len(batch) != 4:
                    raise ValueError("Expected (None, sequence, positive, negative)")
                seq, pos, neg = [
                    torch.as_tensor(value, dtype=torch.long, device=self.device)
                    for value in batch[1:]
                ]
                if seq.ndim != 2 or pos.shape != seq.shape or neg.shape != seq.shape:
                    raise ValueError(
                        "Diagnostic sequence, positive and negative must be B x L"
                    )
                for name, values in (
                    ("sequence", seq),
                    ("positive", pos),
                    ("negative", neg),
                ):
                    if bool(((values < 0) | (values > itemnum)).any().item()):
                        raise ValueError(
                            f"Diagnostic {name} item IDs must be in [0, {itemnum}]"
                        )
                counts["batches"] += 1
                valid = seq.ne(0) & pos.ne(0) & neg.ne(0)
                counts["valid_positions"] += int(valid.sum().item())
                if not bool(valid.any().item()):
                    continue
                support_masks = {}
                if self.source_support is not None:
                    # Support belongs to the source at each causal position.
                    # Building only the unique-source lookup avoids a dense catalog table.
                    sources, inverse = torch.unique(seq, return_inverse=True)
                    lookup = torch.tensor(
                        [self.source_support[int(s)] for s in sources.cpu().tolist()],
                        dtype=torch.long,
                        device=seq.device,
                    )
                    support = lookup[inverse.reshape(-1)].reshape_as(seq)
                    support_masks = {
                        "n_a=0": valid & support.eq(0),
                        "n_a=1-4": valid & support.ge(1) & support.le(4),
                        "n_a>=5": valid & support.ge(5),
                    }
                    for name, mask in support_masks.items():
                        support_counts[name] += int(mask.sum().item())
                features = self.view.log2feats(seq)
                if (
                    not isinstance(features, torch.Tensor)
                    or features.ndim != 3
                    or features.shape[:2] != seq.shape
                ):
                    raise ValueError("view.log2feats must return B x L x D features")
                items = torch.stack((pos, neg), dim=-1)
                batch_margins = {}
                batch_finite = {}
                scorers = [("neural", self.view.predict_score)]
                if self.deployed_score is not None:
                    scorers.append(("deployed", self.deployed_score))
                for scope, scorer in scorers:
                    if scope == "deployed":
                        # At t the causal feature predicts positive[:, t] from seq[:, t].
                        logits = scorer(features, items, seq)
                    else:
                        logits = scorer(features, items)
                    if (
                        not isinstance(logits, torch.Tensor)
                        or logits.shape != items.shape
                        or not logits.is_floating_point()
                        or logits.device != items.device
                    ):
                        raise ValueError(
                            f"{scope} scorer must return a floating B x L x 2 tensor "
                            "on the input device"
                        )
                    # Convert before subtracting to avoid float32 overflow.
                    logits64 = logits.detach().double()
                    margins = logits64[..., 0] - logits64[..., 1]
                    finite = torch.isfinite(logits64).all(dim=-1) & torch.isfinite(
                        margins
                    )
                    batch_margins[scope] = margins
                    batch_finite[scope] = finite
                    usable = valid & finite
                    counts[f"{scope}_finite_positions"] += int(usable.sum().item())
                    counts[f"{scope}_nonfinite_positions"] += int(
                        (valid & ~finite).sum().item()
                    )
                    if bool(usable.any().item()):
                        chunks[scope].append(margins[usable].cpu())
                    for name, mask in support_masks.items():
                        selected = margins[mask & finite]
                        if selected.numel():
                            support_chunks[name][scope].append(selected.cpu())
                if "deployed" in batch_margins:
                    common = valid & batch_finite["neural"] & batch_finite["deployed"]
                    if bool(common.any().item()):
                        paired_chunks.append(
                            torch.stack(
                                (
                                    batch_margins["neural"][common],
                                    batch_margins["deployed"][common],
                                ),
                                dim=-1,
                            ).cpu()
                        )
        finally:
            for module, training in modes:
                module.training = training
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.random.set_rng_state(torch_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, self.device)

        neural = self._summarize(chunks["neural"], float(eps))
        paired_comparison = {
            "finite_positions": 0,
            "neural_strict_win_rate": None,
            "deployed_strict_win_rate": None,
            "neural_loss_to_deployed_win_rate": None,
            "neural_win_to_deployed_loss_rate": None,
            "either_tied_rate": None,
        }
        if paired_chunks:
            paired = torch.cat(paired_chunks)
            neural_margin, deployed_margin = paired.unbind(dim=-1)
            paired_comparison.update(
                {
                    "finite_positions": int(paired.shape[0]),
                    "neural_strict_win_rate": float(
                        (neural_margin > 0).double().mean().item()
                    ),
                    "deployed_strict_win_rate": float(
                        (deployed_margin > 0).double().mean().item()
                    ),
                    "neural_loss_to_deployed_win_rate": float(
                        ((neural_margin < 0) & (deployed_margin > 0))
                        .double()
                        .mean()
                        .item()
                    ),
                    "neural_win_to_deployed_loss_rate": float(
                        ((neural_margin > 0) & (deployed_margin < 0))
                        .double()
                        .mean()
                        .item()
                    ),
                    "either_tied_rate": float(
                        ((neural_margin == 0) | (deployed_margin == 0))
                        .double()
                        .mean()
                        .item()
                    ),
                }
            )
        breakdown = {
            "paired_comparison": paired_comparison,
            "eps": float(eps),
            "observation_counts": counts,
            "deployed_available": self.deployed_score is not None,
            "deployed_margins": self._summarize(chunks["deployed"], float(eps)),
            "source_support_slices": (
                {
                    name: {
                        "valid_positions": support_counts[name],
                        **{
                            scope: {
                                "finite_positions": sum(
                                    part.numel() for part in support_chunks[name][scope]
                                ),
                                "margins": self._summarize(
                                    support_chunks[name][scope], float(eps)
                                ),
                            }
                            for scope in ("neural", "deployed")
                        },
                    }
                    for name in support_names
                }
                if self.source_support is not None
                else None
            ),
            "category_breakdown": None,
            "category_breakdown_reason": "Item categories are unavailable in this task.",
        }
        return neural, breakdown

    @torch.no_grad()
    def run_full_diagnosis(self, train_loader):
        neural, detail = self.analyze_pos_neg_margin_with_category_breakdown(
            train_loader
        )
        metrics = {
            "embedding_collapse_score": self.analyze_embeddings_collapse_cosine(),
            "pos_neg_margin_mean": neural["mean"],
            "pos_neg_margin_p10": neural["p10"],
            "pos_neg_neg_beats_pos_rate": neural["neg_beats_pos_rate"],
            "pos_neg_small_margin_rate": neural["small_margin_rate"],
            "pos_neg_tie_rate": neural["tie_rate"],
            "deployed_pos_neg_margins": detail["deployed_margins"],
            "paired_neural_deployed_comparison": detail["paired_comparison"],
            "training_source_support_slices": detail["source_support_slices"],
            "occurrence_excluded_deployed_margins": None,
            "pos_neg_margin_category_breakdown": None,
        }
        definitions = {
            "embedding_collapse_score": (
                "(Mean off-diagonal cosine + 1)/2 among up to 2048 seed-42 sampled "
                "nonpadding embeddings, excluding zero or nonfinite vectors. "
                "Isotropic embeddings are near 0.5. This measures directional alignment, "
                "not recommendation diversity, relevance, or user traits. Null if fewer than two usable vectors."
            ),
            "pos_neg_margin_mean": (
                "Mean positive-minus-negative score from view.predict_score: the unscaled "
                "neural component only, excluding transition corrections. Valid positions "
                "require nonpadding source, positive and negative; nonfinite score pairs are excluded."
            ),
            "pos_neg_margin_p10": "Nearest-rank 10th percentile of the same neural margins.",
            "pos_neg_neg_beats_pos_rate": (
                "Fraction of finite neural pairs with negative strictly above positive. "
                "Pairwise training evidence, not validation HR@10."
            ),
            "pos_neg_small_margin_rate": (
                "Fraction of finite neural margins below 0.5 in raw neural-score units; "
                "scale-dependent and not comparable to normalized deployed margins."
            ),
            "pos_neg_tie_rate": "Fraction of finite neural pairs with exactly equal scores; ties are reported separately.",
            "deployed_pos_neg_margins": (
                "Same mean, nearest-rank p10, strict negative-win, margin<0.5 and exact-tie "
                "statistics using score(features, items, source_items), with source_items=sequence "
                "at every B x L position. Uses the scorer's configured normalization, correction "
                "and alpha. Null values if no shared scorer or finite observations exist. "
                "Magnitudes depend on deployed scaling; these are not candidate-set ranks or HR@10. "
                "If the scorer uses transitions fitted on these training histories, positive "
                "occurrences contribute to their own counts and normalization. These inclusive "
                "margins are not occurrence-excluded or held-out evidence."
            ),
            "paired_neural_deployed_comparison": (
                "Comparison restricted to the identical valid training positions with finite "
                "scores and margins from both scorers. Reports observation count, strict "
                "positive-win rates, strict loss-to-win and win-to-loss fractions, and the "
                "fraction tied under either scorer. All fractions use the full matched count "
                "as denominator; ties are excluded from strict wins and reversals. Null rates "
                "mean no matched evidence. Compares ordering without subtracting differently "
                "scaled margins. Inclusive transition self-count influence still applies; "
                "these measurements do not establish validation HR@10 improvements."
            ),
            "training_source_support_slices": (
                "Training diagnostic positions grouped by n_a=0, 1-4, or >=5, where n_a "
                "counts adjacent training occurrences leaving the source item. Repeats and "
                "self-transitions count separately; padding and cross-user transitions do not. "
                "Each slice reports valid positions and finite neural/deployed margin counts "
                "and summaries using the same definitions as the overall margins. Support "
                "uses sequence[:, t] at each position. These are loader-sampled training "
                "slices, not validation coverage or validation HR@10. Null if the original "
                "user_train['History'] mapping is unavailable."
            ),
            "occurrence_excluded_deployed_margins": (
                "Unavailable: batches contain no user/prefix/occurrence identifiers. Item "
                "pairs alone cannot identify the evaluated occurrence or its membership in "
                "normalization prefixes. Correct exclusion requires decrementing that one "
                "occurrence from c_ab, n_a, d_b and N, recomputing q and affected reference "
                "normalization, and constructing nulls from remaining destination occurrences. "
                "No pair-level approximation or neural retraining is substituted."
            ),
            "pos_neg_margin_category_breakdown": (
                "Unavailable: item categories, ratings, reviews, prices and stores are not provided. "
                "No diversity or conformity traits are observed or inferred."
            ),
        }
        return {
            "metrics": metrics,
            "metric_definitions": definitions,
            "observation_counts": {
                **detail["observation_counts"],
                "embedding_sample": self.embedding_observation_counts,
                "training_transition_occurrences": self.training_transition_occurrences,
            },
            "measurement_context": {
                "split": "training",
                "maximum_batches": 10,
                "deployed_scorer_available": detail["deployed_available"],
                "transition_count_scope": (
                    "Adjacent occurrences within original training histories only; "
                    "inclusive counts, repeated occurrences retained."
                    if self.source_support is not None
                    else "Unavailable: original user_train['History'] mapping missing."
                ),
                "occurrence_excluded_probe": {
                    "available": False,
                    "evaluated_occurrences": 0,
                    "reason": (
                        "The loader supplies (None, sequence, positive, negative), without "
                        "occurrence provenance or reference-prefix membership."
                    ),
                },
                "validation_attribution": (
                    "Unavailable in this training-loader probe. Validation coverage, support "
                    "slices, paired bootstrap intervals and shuffled-control spread require "
                    "the pinned evaluator's matched per-user validation outcomes."
                ),
                "missing_values": "Null denotes unavailable or empty evidence, never measured zero.",
                "selection_metric": "External pinned four-category macro validation HR@10 only.",
                "rng_restoration": (
                    "Python, NumPy global, Torch CPU and the selected CUDA device RNGs "
                    "are restored. Loader-owned generators and worker state are not rewound; "
                    "use a separate diagnostic loader when those are present."
                ),
                "research_scope": (
                    "These probes measure the supplied model and optional shared scorer. "
                    "They do not implement transition fitting, normalization, shared-alpha "
                    "selection, predictor serialization, or validation attribution controls."
                ),
            },
        }


### <<< Self_EvolveRec-BLOCK-END
