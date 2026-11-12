# Movies / NextItNet full study

This profile runs the native staged-autoresearch engine on Movies_and_TV with seed 2024. All four arms use gpt-6.1-sol with high reasoning effort: A is the ordinary iterative baseline, B adds staged planning and reviews, C resets native sessions at reviews, and D also enables diagnostic trials. The engine retains the original arm definitions.

The corrected comparison has four arms, one replicate, and 44 candidate training calls per arm: 176 candidate calls plus four final selected-checkpoint tests. This matches the earlier MLE-STAR candidate budget per method. Diagnostics consume the same 44 slots. Stages 1–4 have ten calls each; stage 5 stops after four calls and receives a final review. GPUs 4–7 run the four arms in parallel. Each arm selects one model without ensembling. All arms use high effort; earlier studies retain their original effort settings.

The original 2,000-call launch was stopped and preserved at `artifacts/staged-movies-2024`. Corrected results are written to `artifacts/staged-movies-2024-aligned44`, starting from the original baseline without importing candidate results. Baseline calibration assets are reused. Extra calls from the stopped launch remain separate overhead.

The profile shares the frozen RecHarness trainer, sampled dataset, seed, validation protocol, and original NextItNet baseline with those runs. The baseline validation HR@10 is 0.395. Candidate predictions are independently rescored outside the candidate process. The test dataset was already used in earlier studies and is not a new untouched holdout.

Codex sessions are real native CLI threads. A/B resume their threads; C/D create fresh native threads when required by the engine. Execution uses the local process backend, not an operating-system filesystem jail. Credentials and native conversation files reside in ignored private run directories and must not be published.

From this directory:

```sh
python3 -m staged_research.movies launch --prefix ../artifacts/staged-movies-2024-aligned44
python3 -m staged_research.movies status --prefix ../artifacts/staged-movies-2024-aligned44
```

Preparation performs fresh baseline calibration once, records the environment, and validates the draft. Launch freezes code and configuration before starting detached workers. Do not edit registered runtime code after launch. Configuration, scheduling, controller logs, trajectory evidence, and final reports are retained under the run prefix. Unknown interrupted agent or evaluation outcomes are retained for inspection instead of silently repeated.
