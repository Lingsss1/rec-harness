# MLEvolve / Movies-and-TV comparison

This adapter runs the actual upstream MLEvolve `AgentSearch`, SearchNode, UCT/Top-K selection, backpropagation, branch stagnation detection, draft/debug/improve/evolution/fusion/aggregation agents, memory-enhanced diff planner, and code reviewer. Upstream repository: https://github.com/InternScience/MLEvolve, pinned at `f4a37b96c948bfed6564e62f3229e0b38e908d33` (Apache-2.0).

The task boundary uses the same frozen RecHarness NextItNet trainer and data as the earlier MLE-STAR experiment. Seed 2024; GPUs 0–3; exactly **44 candidate attempt slots**, including debugging/fusion/invalid proposals. A fresh baseline training call and final baseline/selected-checkpoint tests are separate. Selection uses validation HR@10; final output is one model. Candidate code must pass the common AST and causal scoring checks. The original baseline readout is retained for comparison; the agent may discover and fix it. The test split has already appeared in earlier studies.

LLM calls use Codex CLI, `gpt-6.1-sol`, explicit `high`, with raw requests, CLI events, usage, and responses retained. Earlier MLE-STAR reasoning settings are not retroactively changed: this is candidate-budget alignment, not proof of equal LLM compute. Training time and token usage should be compared separately.

Task adaptations:

- Replace Gemini/OpenAI API transport with native Codex completions. Keep upstream role prompts, full model-code drafts, SEARCH/REPLACE refinement, memory planning, and review.
- The trainer owns data, BPR loss, Adam, negative sampling, seed, maximum 200 epochs × 200 batches, validation sampling, early stopping, and final test. LLM interpretation cannot override host metrics. CSV submission checks are replaced by worker correctness and checkpoint checks.
- Disable pretrained cold-start recommendations and stepwise generation of data/training pipelines because the search space is restricted to NextItNet model classes and eight numeric hyperparameters. Cross-branch fusion merges code insights into one model.
- Keep upstream BM25 + FAISS hybrid experience retrieval and its default `BAAI/bge-base-en-v1.5` model. Pin embedding revision, hash weights, run embeddings on CPU, and require successful memory initialization. Do not silently continue without memory.
- Map the upstream 12-hour exploration/fusion schedule to `completed_candidate_slots / 44`, preserving its phase fractions while enforcing the requested call budget. Other search thresholds retain upstream defaults. Four initial drafts fill four GPU slots; subsequent branches follow native search decisions.
- Invalid source consumes a candidate slot and is reported separately from actual training calls. Transport or unknown execution failures stop for inspection; restarting never silently repeats work.

The isolated adapter environment inherits the existing PyTorch installation by a `.pth` file, installs search/memory dependencies separately, and records `pip freeze`. It does not update the existing training or staged-autoresearch environments. Local process execution is a trusted execution setup, not an OS filesystem jail.

```sh
artifacts/MLEvolve/.venv/bin/python -m mlevolve_codex.prepare
artifacts/MLEvolve/.venv/bin/python -m unittest mlevolve_codex.test_runner
artifacts/MLEvolve/.venv/bin/python -m mlevolve_codex.smoke
artifacts/MLEvolve/.venv/bin/python -m mlevolve_codex.launch
```

`artifacts/mlevolve-native-smoke` contains separate interface acceptance calls and **no training**. Formal artifacts are in `artifacts/mlevolve-movies-2024`: `config.json`, `frozen.json`, `controller.log`, `progress.json`, `calls/`, `trials/`, `native/global_memory/`, `selection.json`, and final `summary.json`. The native upstream CLI prompt/role pipeline is used; this is a task-specific adaptation rather than the unrestricted MLE-bench evaluation from the paper.
