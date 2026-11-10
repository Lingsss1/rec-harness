Current validation-macro best fixed cross-dataset transfer

Model:g3-r0006-linear_private
Source Taobao test:macro AUC0.829080534, click AUC0.669157914; validation macro AUC0.829441908.
The source was selected by the current validation mean, then assessed at the user request. The source test has been observed before this follow-up. This is exploratory architecture transfer with retraining, not zero-shot transfer or a new untouched-holdout claim.

New fits:QK-Video,KuaiRand,TAAC-25,MerRec on GPUs4..7. GPUs0..3 continue the12h autoresearch.
Fixed architecture and hyperparameters,1epoch,batch8192,history100,seed20262027,BF16,AdamW1e-4+Adagrad0.05.
Native item fields form global+field-match relations; five temporal slots and linear rank moments stay fixed. Native task-head count and primary scorer-input stop role are parameterized.
No custom loss hook; joint sum of native true-label BCE is preserved. CPU/BF16 GPU forward,loss,gradient,state parity and full Taobao validation checkpoint replay passed before starting.

DIN and the previous g3-r0029 macro model are reused as exact same-recipe references; their results are labeled historical/reused. The old g1-r0088 click-best campaign was cancelled; completed old test records were retained.

Each new model first runs20cost-calibration steps, then resets to the original seed for a full from-scratch fit and one test after validation checkpoint freeze. Full data coverage and every native task are reported, including regressions. MerRec has Like as its primary task and no click label.

Artifacts:source-selection.json,interface-adaptation.patch,interface-checks.*,cached-release-checks.json,taobao-validation-replay/,datasets/*/protocol.json,runs/*/candidate/,reference-runs/,report.json,report.txt.

Watch:tmux -L ctr-research attach -t ctr-research; select 当前最佳泛化.
Stop only this transfer campaign:touch artifacts/unirank-current-best-generalization-latest/STOP
