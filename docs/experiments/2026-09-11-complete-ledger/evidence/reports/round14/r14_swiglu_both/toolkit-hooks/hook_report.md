# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round14/r14_swiglu_both/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q1` | 256 | 0.001029109 | 1905.285 | 0.001263601 |
| 2 | `q0` | 256 | 0.001007042 | 1864.43 | 0.001314058 |
| 3 | `q2` | 256 | 0.0009804963 | 1815.283 | 0.001269647 |
| 4 | `q6` | 256 | 0.0009739415 | 1803.148 | 0.001266967 |
| 5 | `q4` | 256 | 0.0009649365 | 1786.476 | 0.00121775 |
| 6 | `q3` | 256 | 0.0009591375 | 1775.739 | 0.001249432 |
| 7 | `q5` | 256 | 0.0009513877 | 1761.392 | 0.001227735 |
| 8 | `q7` | 256 | 0.0009188352 | 1701.124 | 0.001177699 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q1', 'q0'], Random=['q3', 'q6'], Bottom=['q5', 'q7']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
