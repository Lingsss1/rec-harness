# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round18/r18_base_norm_phase132/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q1` | 256 | 0.0008426845 | 1560.139 | 0.001103856 |
| 2 | `q5` | 256 | 0.0008402439 | 1555.621 | 0.001061372 |
| 3 | `q0` | 256 | 0.0008242663 | 1526.04 | 0.001110823 |
| 4 | `q3` | 256 | 0.0008225108 | 1522.79 | 0.001115717 |
| 5 | `q6` | 256 | 0.0008195781 | 1517.36 | 0.001080787 |
| 6 | `q7` | 256 | 0.0008190257 | 1516.338 | 0.001096795 |
| 7 | `q2` | 256 | 0.0008116016 | 1502.593 | 0.001068703 |
| 8 | `q4` | 256 | 0.0007897155 | 1462.073 | 0.00102793 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q1', 'q5'], Random=['q7', 'q3'], Bottom=['q2', 'q4']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
