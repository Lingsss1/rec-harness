KuaiRand three-experiment comparison

Existing HeMix: embedding16/token80, three released HeteroMixer layers, GPU0.
New ours joint1: embedding16/token256, one original JointBlock, GPU1.
New ours joint3: embedding16/token256, three independent JointBlocks, GPU2.
Original width-256 HeMix is not trained. Existing HeMix and TAAC jobs are untouched.

Both new runs use the frozen g3-r0018-pair_occupancy_task_private_linear native-field
architecture, including feedback memory, pair occupancy and task-private readout.
Only token width changes in joint1. joint3 additionally replaces the original main
joint block with a stack containing that exact block followed by two new blocks.
History memory and task readout remain outside the stack and are retained.
Added layers use the same Linear Xavier initialization style as the inherited model.
A CPU RNG fork preserves all shared initial parameters and the host sampling stream
between the one- and three-layer versions. Layers are not tied or copied weights.

Parameter counts:
HeMix80: total515,449,862; embedding513,095,952; dense2,353,910.
Ours256/1: total528,536,637; embedding513,095,952; dense15,440,685.
Ours256/3: total529,590,845; embedding513,095,952; dense16,494,893.
Historical ours32/1: total515,498,493; dense2,402,541; test macro AUC0.865792594.
Dimension and depth settings are fixed user choices, not selected by target AUC.
Equal dimension does not imply equal parameter budget or equal block operation type.

Same native KuaiRand data revision2163073060fb510f41e2db284393ee478a80d9c1,
feature map, train/valid/test splits, official_offline feedback, recent100 history,
one epoch, batch8192, seed20262027, BF16, TF32 off, num_workers0, no compile.
AdamW dense1e-4 + Adagrad embedding0.05; dense gradient clip10.
Loss is the equal sum of six native true-label BCEs. Select by validation equal mean
AUC across all six native tasks, tie by mean Logloss; freeze checkpoint/hash before test.
Every task AUC/Logloss, parameter count and measured train/inference/memory cost is kept.

Each new run performs20 calibration training steps without validation/test, then
resets the original seed in a fresh full-fit process. Calibration and failures are logged.
CPU and BF16 CUDA finite forward/backward checks, exact one-layer base parity,
common initial-state/RNG parity and active gradients in all3blocks passed.
No original HeMix training, no new autoresearch or AUC-based hyperparameter search.
Single seed and previously observed target tests limit the conclusions.

Source/config: source/, datasets/KuaiRand/{joint1,joint3}-config.json and protocol.json.
Progress: status.json and runs/{joint1,joint3}/*/progress.json.
Results: report.json/report.txt plus full raw results and checkpoint-selection receipts.
The report also reads the existing HeMix outcome; its weights and jobs are not copied/restarted.
The host adaptation changes kind/config routing and adds config-hash checks; see host-adaptation.patch.

Tmux: tmux -L ctr-research attach -t ctr-research; window KuaiRand三模型.
Stop only the two new runs: touch artifacts/unirank-kuairand-three-models-latest/STOP
Existing HeMix retains its separate stop control.
