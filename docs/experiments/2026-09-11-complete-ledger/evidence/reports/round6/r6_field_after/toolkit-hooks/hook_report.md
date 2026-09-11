# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round6/r6_field_after/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q7` | 256 | 0.001043988 | 1932.831 | 0.001318835 |
| 2 | `q5` | 256 | 0.0009878076 | 1828.819 | 0.001257197 |
| 3 | `q4` | 256 | 0.0009870287 | 1827.377 | 0.001283932 |
| 4 | `q3` | 256 | 0.0009684519 | 1792.984 | 0.00125123 |
| 5 | `q1` | 256 | 0.000968217 | 1792.549 | 0.001209464 |
| 6 | `q0` | 256 | 0.0009428247 | 1745.538 | 0.001272017 |
| 7 | `q6` | 256 | 0.0009162334 | 1696.307 | 0.001190579 |
| 8 | `q2` | 256 | 0.0008843886 | 1637.35 | 0.001138878 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q7', 'q5'], Random=['q0', 'q3'], Bottom=['q6', 'q2']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
