Validation rank-1 cross-dataset architecture transfer

Source: g3-r0018-pair_occupancy_task_private_linear, retained best for topic g3-t003.
Selected by four-task validation macro AUC, not source test ranking.
Taobao validation macro AUC 0.832413742; observed test macro AUC 0.832093660.

Native datasets: QK-Video, KuaiRand, TAAC-25, MerRec on GPUs 4,5,6,7.
Fixed recipe: 1 epoch, batch 8192, history 100, seed 20262027, BF16,
AdamW dense 1e-4 + Adagrad embeddings 0.05, equal sum of every native-task BCE.
No architecture search or target-test-based tuning. Each fit starts from scratch.
The original Taobao checkpoint is frozen for replay only; it is not loaded into target fits.
Schema changes parameterize native task/feedback count and native item-field/pair count.
Five temporal bins, pair occupancy, task-private scorer readout, and open gradients are preserved.
CPU and BF16 CUDA exact initialization/output/loss/gradient/buffer parity precede training.
The adapted model must replay the full original Taobao validation split before target fits.

Each dataset first executes 20 calibration steps without validation or test, then
resets to the original seed for the full fit. Calibration and failures remain logged.
Validation macro AUC selects checkpoints; one final target test follows checkpoint/hash freeze.
Per-task AUC and Logloss, macro metrics, parameter count, time and GPU memory are saved.
Historical DIN and g3-r0006 controls reuse the same pinned data, features and training recipe.
Their original metrics and source identity are retained and labeled reused.

All runs are single seed. Source and target test sets have been inspected for previous models;
this is an exploratory transfer evaluation, not a new untouched-holdout claim.
MerRec has Like as its native first task, and no click label.

Progress: supervisor-status.json, report.json, runs/*/candidate/*/progress.json.
Tmux: tmux -L ctr-research attach -t ctr-research, window 第一名跨数据集.
Stop this campaign only: touch artifacts/unirank-rank1-generalization-latest/STOP
