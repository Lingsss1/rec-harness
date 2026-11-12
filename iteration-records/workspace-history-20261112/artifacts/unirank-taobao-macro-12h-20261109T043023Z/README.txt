Taobao macro-AUC12h research

Start:2026-11-09 04:30:24 UTC
Deadline:2026-11-09 16:30:24 UTC (Beijing:2026-11-10 00:30:24)
Budget:43200s, including calibration, baseline, every research call, failed fit, control, training and validation after recorded start.
Start point:fresh DIN; no previous model or experimental trajectory imported as an incumbent.
Dataset:exact pinned UniRank Taobao_Action train/valid split and official_offline feedback policy.
Selection:(is_click AUC+cart AUC+fav AUC+buy AUC)/4 on validation only. Ties:mean Logloss, parameters,trial ID.
The same rule governs checkpointing, parent/incumbent choice, topic gains, merge eligibility and final selection.
Per-task AUC/Logloss and costs remain visible; no undeclared task veto.

GPU0:统一表示与 token 构造
GPU1:序列与字段的统一交互结构
GPU2:统一结构中的历史压缩
GPU3:任务条件表示与统一学习

Original auto context inside each topic; compact decisions every6completed rounds can continue, switch or merge.
Whole-model replacement and coherent structural mechanisms are allowed; DIN is an empirical starting point.
LLM:gpt-6.1-sol/high. Generic agent instructions unchanged. Personal knowledge disabled.
Industry knowledge:15UniRank comparison methods, inspected code and source provenance; previous local trajectories and test outcomes are not supplied.

Artifacts:
  budget.json / protocol.json / launch-contract-check.json
  baselines/din/                         fresh reference and checkpoint
  topics/gpu-N/rounds/                  agent trajectories and prospective studies
  topics/gpu-N/trials/                  every fit, failure, config, model and checkpoint
  topic-board.json                     committed gains, costs and task-level trade-offs
  dashboard/                           read-only live snapshots

The fixed official test split has been assessed in a previous campaign. It remains unavailable to this search; any later test reuse is supplementary.
Original completed campaigns and original verification receipts are preserved. Parquet is shared; mutable feature metadata is copied into this campaign.

Attach:
  /usr/bin/env -u LD_LIBRARY_PATH -u LD_PRELOAD /usr/bin/tmux -L ctr-research attach -t ctr-research
The old dashboard session was archived; the current session also includes a live cross-dataset test window.

Stop this campaign:
  touch artifacts/unirank-taobao-macro-12h-latest/STOP

User update:after research closes, automatically freeze and assess the validation-macro-AUC top3models, with DIN as an additional reference. Deduplicate identical code/config/checkpoint artifacts; validation replay must pass before test access. No retraining or test-based reselection; post-search cost reported separately. See final-assessment-top3/automatic-plan.json and eventual report.json / report.txt. A STOP file cancels the assessment.
