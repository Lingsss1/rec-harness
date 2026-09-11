# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round8/r8_first_loop1/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q0` | 256 | 0.0009331317 | 1727.593 | 0.001199872 |
| 2 | `q2` | 256 | 0.0009172883 | 1698.26 | 0.001192956 |
| 3 | `q5` | 256 | 0.0008970231 | 1660.741 | 0.001151263 |
| 4 | `q3` | 256 | 0.0008965429 | 1659.852 | 0.001152396 |
| 5 | `q1` | 256 | 0.0008932379 | 1653.733 | 0.001185568 |
| 6 | `q4` | 256 | 0.0008594242 | 1591.131 | 0.001138206 |
| 7 | `q7` | 256 | 0.0008427935 | 1560.341 | 0.001128456 |
| 8 | `q6` | 256 | 0.0008277778 | 1532.541 | 0.001072475 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q0', 'q2'], Random=['q4', 'q3'], Bottom=['q7', 'q6']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
