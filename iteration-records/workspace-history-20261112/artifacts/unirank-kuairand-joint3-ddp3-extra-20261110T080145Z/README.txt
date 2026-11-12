Additional KuaiRand three-layer DDP speed/control run

Keep all three original experiments running. No migration, no switch, no restart.
Only add our g3-r0018 native-field model at embedding16/token256/joint_layers3.
Use GPUs3,4,5, DDP3 + SyncBatchNorm, global batch8192, unchanged optimizers/LR,
seed20262027 and the exact original global batch stream and optimizer step count.
Each rank forwards its contiguous portion; loss weighting preserves the global mean.
Training data/config/model are resolved through the frozen base_campaign in execution-freeze.json.
Its protocol hash plus this execution freeze and executor SHA describe the combined run.
The old single-GPU train.py/scorer and all three old PIDs/configs remain untouched.

FP32 and BF16 numerical parity passed on a global batch97 split32/32/33.
FP32 gradient relative error about1e-6; BF16 gradient relative error about1.3%.
This is semantic parity within declared numeric tolerance, not bitwise equivalence.
Measured30-step calibration on the same GPUs:about0.298s/global step versus0.543s single GPU;
estimated training2.59h, plus validation/test. Calibration is archived as reused preflight, not a new search.
The full fit restarts the declared seed; no partial old optimizer state is loaded.

The extra ranks release their GPUs after training. Validation and test retain the
original single-GPU batch8192 score implementation on GPU3, including full six-task
AUC/Logloss, history slices and sample checks. Save checkpoint/validation selection
before opening the test split. Single seed; target tests are supplementary/exploratory.
Training GPU seconds and per-rank training peaks are recorded separately from wall time.
Raw results:fit-01/result.json; complete report:report.json.

Tmux KuaiRand对照: all original three rows plus this additional fourth row.
Stop only this extra fit:touch artifacts/unirank-kuairand-joint3-ddp-extra-latest/STOP
