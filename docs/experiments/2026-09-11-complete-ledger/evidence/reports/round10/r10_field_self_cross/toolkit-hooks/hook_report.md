# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round10/r10_field_self_cross/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q1` | 256 | 0.0009560672 | 1770.055 | 0.001170982 |
| 2 | `q0` | 256 | 0.0009393172 | 1739.044 | 0.001220892 |
| 3 | `q2` | 256 | 0.000924164 | 1710.99 | 0.001158824 |
| 4 | `q4` | 256 | 0.0009104668 | 1685.631 | 0.001199591 |
| 5 | `q6` | 256 | 0.0009093793 | 1683.618 | 0.00119274 |
| 6 | `q5` | 256 | 0.0008676788 | 1606.414 | 0.001116035 |
| 7 | `q3` | 256 | 0.0008611649 | 1594.354 | 0.001086327 |
| 8 | `q7` | 256 | 0.0008520484 | 1577.476 | 0.001088573 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q1', 'q0'], Random=['q5', 'q4'], Bottom=['q3', 'q7']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
