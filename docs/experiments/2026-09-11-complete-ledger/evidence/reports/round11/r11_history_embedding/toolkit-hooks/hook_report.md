# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round11/r11_history_embedding/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q3` | 256 | 0.0009758488 | 1806.679 | 0.001220988 |
| 2 | `q2` | 256 | 0.0009616871 | 1780.46 | 0.001282763 |
| 3 | `q6` | 256 | 0.0009394626 | 1739.314 | 0.001168419 |
| 4 | `q1` | 256 | 0.0009348842 | 1730.837 | 0.001171701 |
| 5 | `q7` | 256 | 0.0009081047 | 1681.258 | 0.001165156 |
| 6 | `q5` | 256 | 0.0008791942 | 1627.733 | 0.00119094 |
| 7 | `q4` | 256 | 0.0008677906 | 1606.621 | 0.001134071 |
| 8 | `q0` | 256 | 0.0008630339 | 1597.814 | 0.00113713 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q3', 'q2'], Random=['q5', 'q1'], Bottom=['q4', 'q0']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
