# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round9/r9_untied_clone/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q1` | 256 | 0.001029608 | 1906.208 | 0.00132799 |
| 2 | `q6` | 256 | 0.001025956 | 1899.448 | 0.001360547 |
| 3 | `q2` | 256 | 0.001022384 | 1892.833 | 0.001337196 |
| 4 | `q7` | 256 | 0.001018646 | 1885.914 | 0.001312283 |
| 5 | `q4` | 256 | 0.001004632 | 1859.968 | 0.001331195 |
| 6 | `q0` | 256 | 0.0009937515 | 1839.824 | 0.001254229 |
| 7 | `q5` | 256 | 0.0009743022 | 1803.815 | 0.001296801 |
| 8 | `q3` | 256 | 0.0009731123 | 1801.612 | 0.001269666 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q1', 'q6'], Random=['q0', 'q7'], Bottom=['q5', 'q3']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
