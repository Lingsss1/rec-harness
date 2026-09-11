# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round14/r14_silu_both/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q1` | 256 | 0.001194695 | 2211.848 | 0.001489083 |
| 2 | `q3` | 256 | 0.001143198 | 2116.507 | 0.001404089 |
| 3 | `q6` | 256 | 0.001142371 | 2114.976 | 0.001466722 |
| 4 | `q0` | 256 | 0.001115414 | 2065.069 | 0.001464791 |
| 5 | `q7` | 256 | 0.001109943 | 2054.94 | 0.001420536 |
| 6 | `q4` | 256 | 0.001037465 | 1920.755 | 0.001321438 |
| 7 | `q2` | 256 | 0.001018767 | 1886.137 | 0.001368393 |
| 8 | `q5` | 256 | 0.001001112 | 1853.45 | 0.001324038 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q1', 'q3'], Random=['q4', 'q0'], Bottom=['q2', 'q5']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
