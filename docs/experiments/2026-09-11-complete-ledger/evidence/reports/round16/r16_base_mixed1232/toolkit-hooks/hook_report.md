# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round16/r16_base_mixed1232/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q0` | 256 | 0.0007034122 | 1302.292 | 0.0008840005 |
| 2 | `q6` | 256 | 0.0006913254 | 1279.914 | 0.0008512557 |
| 3 | `q1` | 256 | 0.0006792417 | 1257.543 | 0.0008645804 |
| 4 | `q2` | 256 | 0.0006792221 | 1257.506 | 0.0008426361 |
| 5 | `q5` | 256 | 0.0006762288 | 1251.965 | 0.0008733939 |
| 6 | `q7` | 256 | 0.0006707756 | 1241.869 | 0.000833762 |
| 7 | `q3` | 256 | 0.0006440159 | 1192.326 | 0.0008155963 |
| 8 | `q4` | 256 | 0.0006421863 | 1188.939 | 0.0008166108 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q0', 'q6'], Random=['q7', 'q2'], Bottom=['q3', 'q4']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
