# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round9/r9_field_splitnorm/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q7` | 256 | 0.0009998262 | 1851.07 | 0.001277049 |
| 2 | `q1` | 256 | 0.000998952 | 1849.452 | 0.001237835 |
| 3 | `q3` | 256 | 0.0009889898 | 1831.008 | 0.00122789 |
| 4 | `q2` | 256 | 0.0009821726 | 1818.386 | 0.001264438 |
| 5 | `q6` | 256 | 0.000970745 | 1797.229 | 0.001255471 |
| 6 | `q4` | 256 | 0.0009380166 | 1736.636 | 0.001191333 |
| 7 | `q5` | 256 | 0.0009274401 | 1717.055 | 0.001227494 |
| 8 | `q0` | 256 | 0.0008982084 | 1662.936 | 0.001153066 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q7', 'q1'], Random=['q4', 'q2'], Bottom=['q5', 'q0']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
