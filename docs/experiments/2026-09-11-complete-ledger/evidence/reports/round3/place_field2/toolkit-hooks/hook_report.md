# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round3/place_field2/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q5` | 256 | 0.0008691052 | 1609.054 | 0.001089313 |
| 2 | `q2` | 256 | 0.0008502043 | 1574.061 | 0.001016832 |
| 3 | `q1` | 256 | 0.0008313806 | 1539.211 | 0.001009822 |
| 4 | `q4` | 256 | 0.000829181 | 1535.139 | 0.001089007 |
| 5 | `q0` | 256 | 0.0008202339 | 1518.574 | 0.001039749 |
| 6 | `q6` | 256 | 0.00081755 | 1513.605 | 0.001069525 |
| 7 | `q7` | 256 | 0.0008057305 | 1491.723 | 0.001062399 |
| 8 | `q3` | 256 | 0.0007961881 | 1474.056 | 0.001052441 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q5', 'q2'], Random=['q6', 'q4'], Bottom=['q7', 'q3']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
