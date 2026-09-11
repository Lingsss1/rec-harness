# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round10/r10_history_cross_ffn/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q7` | 256 | 0.0009268248 | 1715.916 | 0.001195573 |
| 2 | `q4` | 256 | 0.0008777857 | 1625.125 | 0.001167774 |
| 3 | `q3` | 256 | 0.0008658695 | 1603.064 | 0.001178208 |
| 4 | `q6` | 256 | 0.00086267 | 1597.14 | 0.001100084 |
| 5 | `q2` | 256 | 0.0008592632 | 1590.833 | 0.001155395 |
| 6 | `q5` | 256 | 0.0008486379 | 1571.161 | 0.001133621 |
| 7 | `q1` | 256 | 0.0008462514 | 1566.743 | 0.001124002 |
| 8 | `q0` | 256 | 0.0008220107 | 1521.864 | 0.001126156 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q7', 'q4'], Random=['q5', 'q6'], Bottom=['q1', 'q0']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
