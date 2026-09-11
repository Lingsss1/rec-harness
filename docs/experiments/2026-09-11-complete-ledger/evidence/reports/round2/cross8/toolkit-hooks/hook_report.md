# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round2/cross8/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q1` | 256 | 0.0009310907 | 1723.814 | 0.001197125 |
| 2 | `q0` | 256 | 0.0009165988 | 1696.984 | 0.001155309 |
| 3 | `q5` | 256 | 0.0009093434 | 1683.551 | 0.00116953 |
| 4 | `q6` | 256 | 0.0008888842 | 1645.673 | 0.001132179 |
| 5 | `q2` | 256 | 0.0008710559 | 1612.666 | 0.001104304 |
| 6 | `q4` | 256 | 0.0008596608 | 1591.569 | 0.00104816 |
| 7 | `q3` | 256 | 0.0008551575 | 1583.232 | 0.001079906 |
| 8 | `q7` | 256 | 0.0008352737 | 1546.419 | 0.001059713 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q1', 'q0'], Random=['q4', 'q6'], Bottom=['q3', 'q7']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
