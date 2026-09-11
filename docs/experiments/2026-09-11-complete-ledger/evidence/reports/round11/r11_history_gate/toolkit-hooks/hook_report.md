# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round11/r11_history_gate/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q3` | 256 | 0.001039188 | 1923.944 | 0.00127532 |
| 2 | `q2` | 256 | 0.001013659 | 1876.68 | 0.001321469 |
| 3 | `q6` | 256 | 0.001006156 | 1862.789 | 0.001296869 |
| 4 | `q1` | 256 | 0.0009978508 | 1847.413 | 0.001339025 |
| 5 | `q5` | 256 | 0.0009809485 | 1816.12 | 0.001324891 |
| 6 | `q4` | 256 | 0.000950281 | 1759.343 | 0.001237433 |
| 7 | `q0` | 256 | 0.0009443869 | 1748.43 | 0.001242644 |
| 8 | `q7` | 256 | 0.0009353135 | 1731.632 | 0.001263047 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q3', 'q2'], Random=['q4', 'q1'], Bottom=['q0', 'q7']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
