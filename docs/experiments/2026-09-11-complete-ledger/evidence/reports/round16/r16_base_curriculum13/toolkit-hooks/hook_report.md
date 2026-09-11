# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round16/r16_base_curriculum13/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q6` | 256 | 0.000767993 | 1421.856 | 0.0009393758 |
| 2 | `q1` | 256 | 0.0007422158 | 1374.132 | 0.0009425141 |
| 3 | `q7` | 256 | 0.0007348888 | 1360.567 | 0.0009367131 |
| 4 | `q3` | 256 | 0.000724979 | 1342.22 | 0.0009148836 |
| 5 | `q4` | 256 | 0.0007163725 | 1326.286 | 0.0009016927 |
| 6 | `q5` | 256 | 0.0007090947 | 1312.812 | 0.0008964172 |
| 7 | `q2` | 256 | 0.0007089002 | 1312.452 | 0.0009097693 |
| 8 | `q0` | 256 | 0.0007050116 | 1305.253 | 0.00093236 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q6', 'q1'], Random=['q5', 'q3'], Bottom=['q2', 'q0']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
