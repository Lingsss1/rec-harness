# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round13/r13_self_between_zero/perturbations.tsv`
- Rows: `1197`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `extra_attention_off` | `off` | 256 | `20262027` | `logloss` | 57 | 0.001450323 | [0.0009170766, 0.002050372] | 0.7719298 |
| `inference_placement_steps1` | `steps1` | 256 | `20262027` | `logloss` | 57 | 0.04952818 | [0.04751804, 0.05152504] | 1 |
| `inference_placement_steps3` | `steps3` | 256 | `20262027` | `logloss` | 57 | 0.02921367 | [0.02739212, 0.03114665] | 1 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004435389 | [0.003815391, 0.00500252] | 0.9824561 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.003104808 | [0.002637683, 0.003604036] | 0.9298246 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.001563152 | [0.001219659, 0.001912669] | 0.8245614 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0007893542 | [0.0003571509, 0.0011938] | 0.7192982 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.006999862 | [0.006154979, 0.007849046] | 0.9824561 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.004436112 | [0.003562216, 0.005237433] | 0.9298246 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | 0.0003957558 | [3.42355e-05, 0.0007466942] | 0.6315789 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004461963 | [0.003984453, 0.004920956] | 0.9649123 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002713141 | [0.002017447, 0.003402832] | 0.8596491 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004886075 | [0.004295749, 0.005435503] | 0.9649123 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002244533 | [0.001468979, 0.003024507] | 0.8421053 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004064775 | [0.003392314, 0.004774548] | 0.9824561 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001353113 | [0.0009065925, 0.001772525] | 0.7719298 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004321841 | [0.003720143, 0.004912176] | 0.9824561 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.005517014 | [0.004593641, 0.006446256] | 0.9298246 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.00174945 | [0.001447039, 0.00203691] | 0.9649123 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001501997 | [0.00124602, 0.001768463] | 0.8947368 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
