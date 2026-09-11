# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round12/r12_both_caf/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q0` | 256 | 0.0009623874 | 1781.756 | 0.001210026 |
| 2 | `q1` | 256 | 0.0009583164 | 1774.219 | 0.001203666 |
| 3 | `q7` | 256 | 0.0009393366 | 1739.08 | 0.001198609 |
| 4 | `q2` | 256 | 0.0009281987 | 1718.46 | 0.001188928 |
| 5 | `q3` | 256 | 0.0009161064 | 1696.072 | 0.001175805 |
| 6 | `q5` | 256 | 0.0009023723 | 1670.645 | 0.001186746 |
| 7 | `q6` | 256 | 0.0008928288 | 1652.976 | 0.001141393 |
| 8 | `q4` | 256 | 0.0008744632 | 1618.974 | 0.001151805 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q0', 'q1'], Random=['q5', 'q2'], Bottom=['q6', 'q4']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
