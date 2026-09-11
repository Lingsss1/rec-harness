# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round12/r12_history_afc/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q7` | 256 | 0.0009624439 | 1781.861 | 0.00119785 |
| 2 | `q4` | 256 | 0.0009384825 | 1737.499 | 0.001140203 |
| 3 | `q6` | 256 | 0.0009278853 | 1717.879 | 0.001188833 |
| 4 | `q2` | 256 | 0.0009196599 | 1702.651 | 0.001171304 |
| 5 | `q0` | 256 | 0.000912606 | 1689.591 | 0.001146116 |
| 6 | `q3` | 256 | 0.0009046321 | 1674.829 | 0.001120147 |
| 7 | `q5` | 256 | 0.0008778749 | 1625.291 | 0.001117063 |
| 8 | `q1` | 256 | 0.0008680695 | 1607.137 | 0.001128295 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q7', 'q4'], Random=['q3', 'q2'], Bottom=['q5', 'q1']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
