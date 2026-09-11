# Round 3: change structure and recurrence placement

User steering: prioritize architecture and placement changes, beyond round-2 scalar state mixing. Cross8 remains the primary baseline; Cross11 remains an optimized reference. Preserve all earlier results.

All configurations use all queries during training and validation, unchanged data hashes and kuairec-loop-v1 protocol, one epoch, batch 8192, 1240 updates. No test evaluation.

N0/N1 denote distinct field QFormer blocks; S0/S1/S2 denote distinct history QFormer blocks. Repeated symbols share parameters. Both input streams remain immutable. Ns final norm is applied at each transition from field blocks to sequence blocks; sequence final norm is applied once before the unchanged query route/tower.

| Experiment | Block order | Calls | Parameters |
|---|---|---:|---:|
| Cross11 | N0 N1 S0 S1 S2 | 5 | 11.408M |
| middle2 | N0 N1 S0 S1 S1 S2 | 6 | 11.408M |
| place_field2 | N0 N1 N1 S0 S1 S2 | 6 | 11.408M |
| place_front2 | N0 N1 S0 S0 S1 S2 | 6 | 11.408M |
| place_tail2 | N0 N1 S0 S1 S2 S2 | 6 | 11.408M |
| place_feedback2 | N0 N1 S0 N1 S1 S2 | 6 | 11.408M |
| place_alternating2 | N0 N1 S0 N1 S0 S1 S2 | 7 | 11.408M |

The first four placement candidates match middle2's block-call count. Field and history sources have different lengths, so this is not exact FLOP or wall-clock equality. The alternating candidate explores a larger cross-source recurrent core and is reported separately for cost.

The feedback and alternating schedules cross the field/sequence interface twice, so they apply the shared field final norm twice. Field-only recurrence applies it once after the repeated field blocks. This is an interface/placement architecture comparison, not a causal isolation of ordering with all normalization calls held fixed.

Initial screen: five new placements at seed 20262027. Replicate the top two by full-validation AUC at seeds 20262028/29; break exact AUC ties by lower logloss, then lower measured latency. Do not promote from one seed. This stage has a maximum of nine new full training runs.

Primary quality metric: full-validation global AUC. Report Logloss, UAUC, gAUC, NDCG@10, parameter count, wall time and same-GPU inference latency. Compare seed-paired results to Cross8 original, matched Cross8, Cross11 all-query and the existing middle2 all-query/damped candidates. Near-baseline models remain optimization candidates, per user instruction. Stable improvement requires stronger paired evidence than candidate retention.

Implementation acceptance: strict state-dict compatibility with Cross11, exact single-pass output and gradient equivalence in training/eval, expected shared-block order, gradient flow through all calls, no source mutation, padded and empty-history invariance, optimizer parameter coverage. Existing rec-harness hook/geometry/paired-intervention tools cover the new tensor contracts through adapter changes; no new diagnostic method is needed.
