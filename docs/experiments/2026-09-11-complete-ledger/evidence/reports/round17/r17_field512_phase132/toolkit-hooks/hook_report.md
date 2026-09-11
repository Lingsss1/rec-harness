# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round17/r17_field512_phase132/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q2` | 256 | 0.000820535 | 1519.132 | 0.001012777 |
| 2 | `q3` | 256 | 0.000818751 | 1515.829 | 0.001040785 |
| 3 | `q1` | 256 | 0.0008095375 | 1498.771 | 0.0009788445 |
| 4 | `q4` | 256 | 0.0007968365 | 1475.257 | 0.0009997525 |
| 5 | `q7` | 256 | 0.0007921394 | 1466.561 | 0.001003676 |
| 6 | `q5` | 256 | 0.0007634807 | 1413.502 | 0.0009679897 |
| 7 | `q0` | 256 | 0.0007544055 | 1396.7 | 0.0009375334 |
| 8 | `q6` | 256 | 0.0007523083 | 1392.818 | 0.000993074 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q2', 'q3'], Random=['q5', 'q4'], Bottom=['q0', 'q6']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
