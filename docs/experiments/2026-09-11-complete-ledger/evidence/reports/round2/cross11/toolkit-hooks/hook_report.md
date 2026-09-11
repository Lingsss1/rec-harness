# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round2/cross11/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q0` | 256 | 0.001269497 | 2350.337 | 0.001601697 |
| 2 | `q6` | 256 | 0.001207728 | 2235.979 | 0.001684208 |
| 3 | `q5` | 256 | 0.001197242 | 2216.564 | 0.001530172 |
| 4 | `q1` | 256 | 0.001179531 | 2183.774 | 0.001569256 |
| 5 | `q4` | 256 | 0.001177008 | 2179.104 | 0.001571742 |
| 6 | `q7` | 256 | 0.001121715 | 2076.735 | 0.001504628 |
| 7 | `q2` | 256 | 0.001116158 | 2066.445 | 0.001538164 |
| 8 | `q3` | 256 | 0 | 0 | 0 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q0', 'q6'], Random=['q7', 'q1'], Bottom=['q2', 'q3']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
