# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round15/r15_w1536_history/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q6` | 256 | 0.000993972 | 1840.232 | 0.001208538 |
| 2 | `q1` | 256 | 0.0009829728 | 1819.868 | 0.001246109 |
| 3 | `q7` | 256 | 0.0009754718 | 1805.981 | 0.001212097 |
| 4 | `q3` | 256 | 0.0009405125 | 1741.257 | 0.00119167 |
| 5 | `q4` | 256 | 0.0009345287 | 1730.179 | 0.001166326 |
| 6 | `q0` | 256 | 0.0009199588 | 1703.204 | 0.001205684 |
| 7 | `q2` | 256 | 0.0009185804 | 1700.652 | 0.001149847 |
| 8 | `q5` | 256 | 0.0008737134 | 1617.586 | 0.001146023 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q6', 'q1'], Random=['q0', 'q3'], Bottom=['q2', 'q5']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
