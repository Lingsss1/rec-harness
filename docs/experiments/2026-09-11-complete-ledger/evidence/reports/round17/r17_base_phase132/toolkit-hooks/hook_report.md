# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round17/r17_base_phase132/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q1` | 256 | 0.0007959562 | 1473.627 | 0.001007655 |
| 2 | `q7` | 256 | 0.0007900891 | 1462.765 | 0.0009915978 |
| 3 | `q6` | 256 | 0.0007867409 | 1456.566 | 0.0009999903 |
| 4 | `q2` | 256 | 0.0007660598 | 1418.277 | 0.0009498612 |
| 5 | `q3` | 256 | 0.0007547975 | 1397.426 | 0.0009642234 |
| 6 | `q4` | 256 | 0.0007509721 | 1390.344 | 0.0009485367 |
| 7 | `q5` | 256 | 0.0007429435 | 1375.48 | 0.0009530274 |
| 8 | `q0` | 256 | 0.0007302828 | 1352.04 | 0.0009490303 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q1', 'q7'], Random=['q4', 'q2'], Bottom=['q5', 'q0']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
