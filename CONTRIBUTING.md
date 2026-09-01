# Contributing to rec-harness

`rec-harness` 只接受框架无关、可公开复现的 Codex Skills 与诊断工具。训练平台适配、业务模型实现和内部实验材料必须保留在下游私有 workspace。

## Repository boundaries

- `model-diagnostic-toolkit/`: stable, framework-independent diagnostics.
- `model-optimization-harness-updater/`: persistent experiment memory and research-strategy consolidation.
- `model-diagnostic-tool-evolver/`: isolated discovery and validation of new diagnostic methods.
- `app/` and `public/lessons/`: public documentation only; never copy internal code or reconstructed model topology here.

Do not couple the Skills to KML, TensorBoard, BlobStore, a specific training platform, or a private model repository.

## Change contract

1. State the information gap or failure mode addressed by the change.
2. Keep evaluation data, seeds, budgets and acceptance gates fixed during comparison.
3. Add a synthetic correctness check or documented validation procedure.
4. Preserve append-only evidence; do not rewrite prior experiment outcomes.
5. Keep datasets, checkpoints, logs, credentials, internal paths and business model code out of Git.

## Validation

Run from the repository root:

```bash
bash scripts/validate-core.sh
```

For documentation-site changes, also run:

```bash
npm ci
npm run lint
npm test
```

## Pull requests

A focused pull request should explain the research problem, evidence used for acceptance, affected Skill boundary, rollback plan and remaining uncertainty.

New diagnostic tools must pass the validation ladder in `model-diagnostic-tool-evolver/references/research-validation-protocol.md`.
