# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round13/r13_self_between_zero/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q2` | 256 | 0.0009425369 | 1745.005 | 0.001238217 |
| 2 | `q6` | 256 | 0.000930199 | 1722.163 | 0.001174789 |
| 3 | `q3` | 256 | 0.0009116245 | 1687.774 | 0.001166023 |
| 4 | `q1` | 256 | 0.0009046077 | 1674.783 | 0.001108149 |
| 5 | `q4` | 256 | 0.0008713974 | 1613.298 | 0.001085968 |
| 6 | `q0` | 256 | 0.0008598704 | 1591.957 | 0.001105223 |
| 7 | `q7` | 256 | 0.000846293 | 1566.82 | 0.001119516 |
| 8 | `q5` | 256 | 0.0008442085 | 1562.961 | 0.001136585 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q2', 'q6'], Random=['q0', 'q1'], Bottom=['q7', 'q5']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
