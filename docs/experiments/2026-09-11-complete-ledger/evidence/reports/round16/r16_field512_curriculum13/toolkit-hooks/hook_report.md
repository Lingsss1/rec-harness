# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round16/r16_field512_curriculum13/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q1` | 256 | 0.0009625647 | 1782.085 | 0.001212085 |
| 2 | `q6` | 256 | 0.0009465493 | 1752.434 | 0.001218 |
| 3 | `q5` | 256 | 0.000928711 | 1719.408 | 0.001146836 |
| 4 | `q3` | 256 | 0.0009138726 | 1691.936 | 0.001150633 |
| 5 | `q4` | 256 | 0.0009007991 | 1667.732 | 0.001170305 |
| 6 | `q0` | 256 | 0.0008963402 | 1659.477 | 0.0011694 |
| 7 | `q7` | 256 | 0.0008792877 | 1627.906 | 0.001145074 |
| 8 | `q2` | 256 | 0.0008141305 | 1507.275 | 0.001076501 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q1', 'q6'], Random=['q0', 'q3'], Bottom=['q7', 'q2']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
