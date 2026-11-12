# Karpathy autoresearch vs MLE-STAR: matched NextItNet experiment calls

The user requested a comparison with the same iteration count as the completed
MLE-STAR run. The reference is
[karpathy/autoresearch](https://github.com/karpathy/autoresearch), commit
`228791fb499afffb54b46200aca536f79142f117`, especially `program.md` (MIT license).
The upstream repository trains nanochat; its policy is applied here to the
existing Movies_and_TV / NextItNet task through Codex CLI.

The MLE-STAR run made **44 candidate training calls**, plus one baseline and
four final test-inference calls. Its 44 calls comprise 8 initial candidates,
4 model-code merges, 16 retraining ablations, and 16 targeted refinements.
This comparison instead uses **11 greedy attempts on each of 4 trajectories**:
edit the current incumbent, train from scratch, keep only a validation HR@10
improvement, otherwise start the next attempt from the unchanged incumbent.
There is no forced ablation, initial web retrieval, nested block selection, or
ensemble stage. Four proposals are generated in one Codex call, matching the
previous arm's batched four-proposal pattern. Greedy parent chains are separate;
the request contains all four items, so this is not a claim of isolated LLM contexts.

Both arms share the **same unchanged worker and admission code**, raw baseline
model, data bytes, model-source bounds, Adam/BPR, epochs/batches, validation users,
early stopping, GPU IDs 0–3, training seed 2024 and evaluation seed 42. The baseline
is freshly trained; it is not substituted with a prior outcome. The shared static
right-aligned-readout audit is supplied to both arms, but this search receives no
previous candidate source, score, chosen architecture, or test results.

The original autoresearch five-minute nanochat training budget and unrestricted
optimizer/training-loop changes are replaced by the common RecHarness protocol
to isolate the search-policy comparison. Immutable model source snapshots and
hashes implement keep/revert; no git commits or resets are manufactured. The
user's finite experiment count overrides the upstream infinite-loop instruction.

The GPU allowance remains 43,200 allocated seconds. This is a **matched maximum
attempt count**, not matched actual GPU seconds or LLM tokens. A training crash
consumes one candidate slot. An AST-rejected attempt is recorded as invalid and
consumes a logical slot without a training launch; exact actual training-call
equality is reported explicitly. No hidden extra GPU debugging runs are allowed.

The final best validation single model is frozen before test inference. The
primary comparison is against MLE-STAR's best validation single model and its
already-recorded test result. MLE-STAR's ensemble is supplementary and clearly
labelled. Old test outcomes are loaded by the host only after this run's search
and final selection, and are never sent to the search agent.

Run from the workspace root on the GPU host:

```bash
artifacts/RecHarness-6lyc/.venv/bin/python -m pytest -q autoresearch_codex/test_runner.py
artifacts/RecHarness-6lyc/.venv/bin/python -m autoresearch_codex.launch \
  --config artifacts/autoresearch-movies-2024/config.json
python3 -m autoresearch_codex.status artifacts/autoresearch-movies-2024
```

`results.tsv` and `rounds/iteration-*.json` retain every keep/discard/crash decision,
parent and incumbent. `calls/` archives the full prompt, upstream and adapted
instructions, schema, CLI events and response. `trials/` retains training logs,
curves, checkpoints, candidate scores and GPU receipts. `selection.json` freezes
the chosen single model; `summary.json` and `comparison.json` contain final results.
Repeat the controller command to recover completed work with request/hash checks;
a started but unfinished Codex call stops for review before an explicit retry.
