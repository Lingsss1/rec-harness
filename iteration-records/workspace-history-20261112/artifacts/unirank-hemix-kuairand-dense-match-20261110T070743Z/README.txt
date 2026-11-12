KuaiRand HeMix dense-capacity comparison

Requested: keep total and dense parameter budgets near our final rank-1 architecture.
Run only reduced HeMix; original width-256 HeMix is NOT trained.
Only architectural change from pinned UniRank HeMix: token_dim 256 -> 80.
Preserve embedding16, three layers, expansion4, two global and two recent NS tokens,
two mixing heads, low-rank16, recent20/global80 history route, tower128/64.
The host adapter supplies native six-task loss/optimizer constructor settings;
CPU and BF16 CUDA initialization/prediction/loss/gradient parity passed at width80.

Our KuaiRand model: total515,498,493; embedding513,095,952; dense2,402,541.
Reduced HeMix: total515,449,862; embedding513,095,952; dense2,353,910.
Total difference -0.0094%; dense difference -2.024%. No dormant padding parameters.
The width was frozen using parameter counts, without target AUC screening.

Same pinned KuaiRand data revision2163073060fb510f41e2db284393ee478a80d9c1,
native feature map and split as the reference; every cached file rehashed.
One epoch, batch8192, history100, seed20262027, BF16, TF32 off, no compile,
AdamW dense1e-4 + Adagrad embedding0.05, dense gradient clip10, official_offline.
Training loss is the equal sum of all six true-label BCE terms.
Validation and checkpoint selection use equal mean AUC over all six tasks;
ties use mean Logloss. Freeze checkpoint hash before one full final test.
The sole host code change adds actual embedding/dense counts to calibration metadata;
see host-metadata-only.patch. Training and scoring arithmetic are preserved.

Workflow: 20 calibration steps without validation/test, then a fresh process
resets the original seed for full training, validation, checkpoint freeze and test.
Calibration and full fit costs are separate. Ours reuses the same-dataset fit
from the ongoing final-rank-1 cross-dataset campaign; no new reference fit is added.
Report all six task AUC/Logloss, macro metrics, parameter splits, memory and time.
Single seed and previously observed benchmark test sets limit conclusions.

Source: source/; frozen protocol and data manifests: datasets/KuaiRand/.
Progress: status.json, runs/hemix_token80/*/progress.json.
Results: report.json, report.txt; final raw result and selection under fit-01/.
Tmux: tmux -L ctr-research attach -t ctr-research; window HeMix参数对齐.
Stop this fit only: touch artifacts/unirank-hemix-kuairand-latest/STOP
