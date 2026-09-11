# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round7/r7_small_attention_first/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q2` | 256 | 0.001027639 | 1902.563 | 0.001273666 |
| 2 | `q3` | 256 | 0.0009511024 | 1760.863 | 0.001170794 |
| 3 | `q7` | 256 | 0.0009276167 | 1717.382 | 0.001176891 |
| 4 | `q6` | 256 | 0.000919442 | 1702.247 | 0.001177461 |
| 5 | `q5` | 256 | 0.0009034849 | 1672.705 | 0.001161572 |
| 6 | `q1` | 256 | 0.0008948063 | 1656.637 | 0.001128599 |
| 7 | `q0` | 256 | 0.0008691372 | 1609.114 | 0.001149203 |
| 8 | `q4` | 256 | 0.0008591537 | 1590.63 | 0.001107661 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q2', 'q3'], Random=['q1', 'q6'], Bottom=['q0', 'q4']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
