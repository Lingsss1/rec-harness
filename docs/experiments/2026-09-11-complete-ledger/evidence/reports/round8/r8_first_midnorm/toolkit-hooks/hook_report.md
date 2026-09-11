# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round8/r8_first_midnorm/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q2` | 256 | 0.001023356 | 1894.633 | 0.001331295 |
| 2 | `q3` | 256 | 0.001015388 | 1879.881 | 0.001290591 |
| 3 | `q7` | 256 | 0.001009804 | 1869.543 | 0.001284194 |
| 4 | `q6` | 256 | 0.001003804 | 1858.434 | 0.001267434 |
| 5 | `q1` | 256 | 0.0009928329 | 1838.123 | 0.001276568 |
| 6 | `q4` | 256 | 0.0009781617 | 1810.961 | 0.001213478 |
| 7 | `q0` | 256 | 0.0009538575 | 1765.964 | 0.001239491 |
| 8 | `q5` | 256 | 0.0009368642 | 1734.503 | 0.001235563 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q2', 'q3'], Random=['q4', 'q6'], Bottom=['q0', 'q5']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
