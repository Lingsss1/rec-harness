# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round3/place_feedback2/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q1` | 256 | 0.0009870284 | 1827.376 | 0.001252416 |
| 2 | `q3` | 256 | 0.0009190017 | 1701.432 | 0.001242562 |
| 3 | `q4` | 256 | 0.0009133207 | 1690.915 | 0.00120307 |
| 4 | `q6` | 256 | 0.0008943536 | 1655.799 | 0.001187334 |
| 5 | `q0` | 256 | 0.0008936368 | 1654.472 | 0.001160276 |
| 6 | `q7` | 256 | 0.0008851794 | 1638.814 | 0.001188501 |
| 7 | `q5` | 256 | 0.0008613304 | 1594.66 | 0.001164822 |
| 8 | `q2` | 256 | 0.0008316749 | 1539.756 | 0.001168763 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q1', 'q3'], Random=['q7', 'q6'], Bottom=['q5', 'q2']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
