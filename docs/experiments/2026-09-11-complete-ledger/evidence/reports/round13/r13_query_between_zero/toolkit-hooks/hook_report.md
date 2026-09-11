# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round13/r13_query_between_zero/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q2` | 256 | 0.001079652 | 1998.859 | 0.001421111 |
| 2 | `q3` | 256 | 0.00103251 | 1911.581 | 0.001292153 |
| 3 | `q5` | 256 | 0.001017507 | 1883.805 | 0.001336624 |
| 4 | `q6` | 256 | 0.001011024 | 1871.801 | 0.001299063 |
| 5 | `q1` | 256 | 0.001009101 | 1868.241 | 0.001313907 |
| 6 | `q0` | 256 | 0.0009932289 | 1838.856 | 0.001259631 |
| 7 | `q4` | 256 | 0.0009834727 | 1820.793 | 0.001310258 |
| 8 | `q7` | 256 | 0.0009613949 | 1779.919 | 0.001255592 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q2', 'q3'], Random=['q0', 'q6'], Bottom=['q4', 'q7']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
