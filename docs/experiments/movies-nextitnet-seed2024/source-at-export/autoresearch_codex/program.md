# Autoresearch comparison: Movies_and_TV / NextItNet

Apply Karpathy autoresearch's edit, run, evaluate, keep-or-discard loop to
NextItNet. This task contract overrides the nanochat-specific instructions in
the upstream program.md.

The user requests the same training-call count as the completed MLE-STAR run:
four trajectories, eleven candidate attempts per trajectory (44 total), and one
fresh shared baseline training. Stop at that limit or the GPU budget limit.

For each trajectory, propose one concrete experiment based on its current
incumbent source and previous keep/discard/crash results. You may modify the
entire permitted model file; there is no preselected component, forced ablation,
retrieval initialization, or ensemble stage. Prefer coherent, useful changes
and simple implementations. Preserve the NextItNet causal convolution backbone.
Each candidate is trained from scratch with training seed 2024. The host keeps
a candidate only when validation HR@10 strictly improves over its own parent;
otherwise the next attempt starts from the unchanged parent. A crash consumes
an experiment slot; use its concrete error to repair the next proposal if useful.

The host supplies four trajectory items in one Codex request, as in the previous
run's batched proposal generation. Each item has its own incumbent and history.
Never use another trajectory's checkpoint or validation history as its parent.

Data, evaluation, Adam, BPR loss, gradient clipping, maximum 200 epochs,
200 batches per epoch, evaluation every five epochs, patience twenty epochs,
and 200 validation users are frozen to the preceding experiment. Evaluation
samples 99 unique negatives with seed 42. The original five-minute nanochat
training budget is replaced by this shared protocol for comparability.

The editable file contains exactly the three provided imports, eight bounded
numeric model constants, and the two existing nn.Module classes. You may edit
any of these model components and hyperparameters. Do not add dependencies,
change the host trainer, change data or scoring, access files/network, load
checkpoints, or read held-out test examples/results. No shell tools or web tools
are needed: respond with the requested complete model sources as structured JSON.

Shared pre-run code-audit fact: histories are right aligned, but the raw baseline
scorer indexes count_nonzero-1. Review this mismatch; distinguish correctness
fixes from architectural changes in your hypothesis. This fact was also supplied
to MLE-STAR. No MLE-STAR candidate source, hypothesis, validation score, selected
architecture, or test result is supplied to this search.

After all attempts, choose the single model with the highest validation HR@10
across the four final incumbents. Freeze that choice, then evaluate it once on
held-out test data. Record every attempt and keep/discard/crash decision in TSV
and JSON; immutable source hashes serve as revisions instead of git reset.
