# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round18/r18_field512_norm_phase132/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q1` | 256 | 0.0007562545 | 1400.124 | 0.0009709138 |
| 2 | `q4` | 256 | 0.0007456786 | 1380.543 | 0.0009246213 |
| 3 | `q7` | 256 | 0.0007293004 | 1350.221 | 0.0009234632 |
| 4 | `q5` | 256 | 0.0007135614 | 1321.082 | 0.0009178157 |
| 5 | `q0` | 256 | 0.0007117195 | 1317.672 | 0.0009141845 |
| 6 | `q6` | 256 | 0.0006925146 | 1282.116 | 0.0009129549 |
| 7 | `q2` | 256 | 0.0006902824 | 1277.983 | 0.000894675 |
| 8 | `q3` | 256 | 0.0006827114 | 1263.966 | 0.0009179992 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q1', 'q4'], Random=['q6', 'q5'], Bottom=['q2', 'q3']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
