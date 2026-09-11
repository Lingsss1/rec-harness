# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round2/middle2-v2/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q2` | 256 | 0.001275227 | 2360.945 | 0.001636689 |
| 2 | `q5` | 256 | 0.001234542 | 2285.621 | 0.0016615 |
| 3 | `q0` | 256 | 0.001217631 | 2254.313 | 0.001582339 |
| 4 | `q1` | 256 | 0.001175948 | 2177.141 | 0.001555602 |
| 5 | `q4` | 256 | 0.001175561 | 2176.424 | 0.001612604 |
| 6 | `q6` | 256 | 0.001148624 | 2126.553 | 0.001490511 |
| 7 | `q7` | 256 | 0.001087983 | 2014.283 | 0.001470357 |
| 8 | `q3` | 256 | 0 | 0 | 0 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q2', 'q5'], Random=['q6', 'q1'], Bottom=['q7', 'q3']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
