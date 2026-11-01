# Movies_and_TV / NextItNet / MLE-STAR via Codex CLI

This task-specific adaptation follows the initialization, ablation-guided outer
loop, block refinement inner loop and ensemble stages in the official Google
implementation at `google/adk-recipes` commit
`8c0862c103e36e1bb503a87bc1b134ddf6319d62`, under
`python/agents/machine-learning-engineering/` (Apache-2.0), and
[the MLE-STAR paper](https://arxiv.org/abs/2506.15692).
It uses `codex exec` instead of ADK/Gemini. The upstream recipe was removed
from the current main branch in commit `63bfb33c4df5d00b6ef9e2e0e864fbcb01e37645`.

The target task and cold start come from `6lyc/RecHarness` commit
`74a927fefaf7d120c52b6e3a0809958bd33984f0`. This runner does not use its Thompson
router. The four GPUs run four solution trajectories; subsequent inner-loop
plans see earlier validation results. Model search stays within NextItNet as
requested. Host code freezes BPR, Adam, clipping, training limits and evaluation.
Only model classes and bounded model hyperparameters can be generated. After an
outer-loop scope selection, edits outside that class/constant block are rejected.
Invalid/crashed proposals get at most two Codex debug attempts. Timeout failures
are retained, charged and never promoted.

Training/preprocessing seed is **2024**. Candidate sampling uses the upstream
benchmark seed **42**, 99 unique negatives and up to 200 validation users.
The fixed candidate lists are reused across checkpoints and candidates. Compared
with template-local evaluation, this separates training RNG from evaluation RNG.
Final evaluation uses up to 10,000 users, training plus validation history, and
the held-out test target. Validation/test row generation is checked against the
upstream benchmark's raw-data fallback. Only vocabulary size from preprocessing
is shared with training; no test examples or scores appear in Codex prompts.
This is workflow isolation, not a claim of OS-level hidden-data isolation.

Four independent trajectories each evaluate two search-informed initial
candidates, attempt a model merge, and run two outer iterations with two genuine
retraining ablations and two sequential block refinements. Budget exhaustion
ends search with the best completed models. Two feedback rounds propose bounded
score ensembles. Selection is saved before any test inference. The single seed
and this NextItNet-constrained search are not a reproduction of MLE-bench results.

The default budget is **43,200 allocated GPU seconds**, summed over concurrent
workers, including failed jobs and final inference; 1,800 seconds are reserved for
final inference. Codex calls and data preparation are outside GPU accounting.
Full trials time out at 1,800 seconds. An interrupted unreceipted worker is
conservatively charged its entire timeout; an active orphan stops recovery.
Code/config/data hashes are frozen at startup. Never edit them in a running run.

The upstream cold start's `score_candidates` uses `count_nonzero - 1` with
right-aligned input, which can read padding. The baseline is preserved exactly
(its two classes and numeric hyperparameters); host scoring calls its public
scorer and does not silently substitute tail readout. This known defect is
included in retrieval context; candidates may repair it. Numerical checks record
whether tail readout matches, but do not reject the original baseline for it.

From the repository root:

```bash
artifacts/RecHarness-6lyc/.venv/bin/python -m mle_star_codex.prepare \
  --repo artifacts/RecHarness-6lyc --root artifacts/mle-star-movies-2024/data --seed 2024
artifacts/RecHarness-6lyc/.venv/bin/python -m pytest -q mle_star_codex/test_runner.py
artifacts/RecHarness-6lyc/.venv/bin/python -m mle_star_codex.smoke \
  --config artifacts/mle-star-movies-2024/config.json
artifacts/RecHarness-6lyc/.venv/bin/python -u -m mle_star_codex.run \
  --config artifacts/mle-star-movies-2024/config.json
```

Run on the GPU host with network access for Codex. The same run command resumes
completed training and Codex calls using their original receipts. A started but
unfinished Codex call requires review before retry; remove its `started.json`
only after verifying the call failed and preserving its original trace.

To leave the verified controller running after disconnecting, use
`python -m mle_star_codex.launch --config artifacts/mle-star-movies-2024/config.json`.
The launcher writes `launcher.json` and `controller.log` and uses a separate
process session. The four-GPU smoke check has its own receipts and GPU costs
under `smoke/`, outside the main search budget.

Artifacts: `calls/` contains every prompt, schema, CLI event, stderr and response;
`models/` contains generated source and hypotheses; `trials/` holds worker logs,
curves, code/data hashes, checkpoints, score matrices and allocation receipts;
`outer-*.json` records ablation/selection/refinement decisions; `selection.json`
freezes the chosen model/ensemble; `summary.json` reports final metrics.

Read live status with:

```bash
python3 -m mle_star_codex.status artifacts/mle-star-movies-2024
```
