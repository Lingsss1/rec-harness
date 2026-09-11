# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round15/r15_w512_field/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q1` | 256 | 0.0009646246 | 1785.898 | 0.001208551 |
| 2 | `q2` | 256 | 0.0009141922 | 1692.528 | 0.001143467 |
| 3 | `q0` | 256 | 0.0009118169 | 1688.131 | 0.0011683 |
| 4 | `q7` | 256 | 0.0008997934 | 1665.87 | 0.001198326 |
| 5 | `q4` | 256 | 0.0008875826 | 1643.263 | 0.001176997 |
| 6 | `q3` | 256 | 0.0008799144 | 1629.067 | 0.001132847 |
| 7 | `q6` | 256 | 0.0008554146 | 1583.708 | 0.001073488 |
| 8 | `q5` | 256 | 0.0008490261 | 1571.88 | 0.001122856 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q1', 'q2'], Random=['q3', 'q7'], Bottom=['q6', 'q5']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
