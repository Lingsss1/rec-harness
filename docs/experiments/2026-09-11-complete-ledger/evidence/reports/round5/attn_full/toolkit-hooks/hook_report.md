# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round5/attn_full/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q5` | 256 | 0.0009764667 | 1807.823 | 0.001195544 |
| 2 | `q2` | 256 | 0.0009541526 | 1766.511 | 0.001176745 |
| 3 | `q4` | 256 | 0.0009368609 | 1734.497 | 0.001182018 |
| 4 | `q6` | 256 | 0.000907968 | 1681.005 | 0.001237721 |
| 5 | `q3` | 256 | 0.0008928681 | 1653.049 | 0.001149346 |
| 6 | `q1` | 256 | 0.000888473 | 1644.912 | 0.001201924 |
| 7 | `q0` | 256 | 0.0008545879 | 1582.177 | 0.00111352 |
| 8 | `q7` | 256 | 0.0008172175 | 1512.99 | 0.001074844 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q5', 'q2'], Random=['q1', 'q6'], Bottom=['q0', 'q7']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
