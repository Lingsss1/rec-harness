# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round7/r7_small_midnorm/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q2` | 256 | 0.00107593 | 1991.968 | 0.001393198 |
| 2 | `q0` | 256 | 0.00101526 | 1879.644 | 0.001313173 |
| 3 | `q3` | 256 | 0.001012373 | 1874.3 | 0.001300308 |
| 4 | `q6` | 256 | 0.0009995386 | 1850.538 | 0.001334256 |
| 5 | `q1` | 256 | 0.0009732625 | 1801.89 | 0.001303283 |
| 6 | `q7` | 256 | 0.0009657667 | 1788.013 | 0.001286874 |
| 7 | `q5` | 256 | 0.0009519302 | 1762.396 | 0.001294205 |
| 8 | `q4` | 256 | 0.0009224877 | 1707.886 | 0.001268791 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q2', 'q0'], Random=['q7', 'q6'], Bottom=['q5', 'q4']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
