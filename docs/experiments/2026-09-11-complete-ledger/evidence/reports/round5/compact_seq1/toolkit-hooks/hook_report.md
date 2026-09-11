# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round5/compact_seq1/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q5` | 256 | 0.001085506 | 2009.697 | 0.001413811 |
| 2 | `q4` | 256 | 0.001046141 | 1936.817 | 0.001356957 |
| 3 | `q1` | 256 | 0.001012062 | 1873.723 | 0.001406506 |
| 4 | `q6` | 256 | 0.000994637 | 1841.463 | 0.001349433 |
| 5 | `q0` | 256 | 0.0009856847 | 1824.889 | 0.001303056 |
| 6 | `q2` | 256 | 0.0009756024 | 1806.223 | 0.001339468 |
| 7 | `q3` | 256 | 0.0009460427 | 1751.496 | 0.001284147 |
| 8 | `q7` | 256 | 0.0009127754 | 1689.905 | 0.001257367 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q5', 'q4'], Random=['q2', 'q6'], Bottom=['q3', 'q7']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
