# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round4/feedback_context/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q3` | 256 | 0.001042229 | 1929.574 | 0.001367416 |
| 2 | `q7` | 256 | 0.001039347 | 1924.24 | 0.001285657 |
| 3 | `q6` | 256 | 0.001032733 | 1911.993 | 0.001322168 |
| 4 | `q1` | 256 | 0.001018814 | 1886.224 | 0.00132991 |
| 5 | `q4` | 256 | 0.00101558 | 1880.237 | 0.00134334 |
| 6 | `q2` | 256 | 0.0009777157 | 1810.135 | 0.001278293 |
| 7 | `q0` | 256 | 0.0009584781 | 1774.519 | 0.001220307 |
| 8 | `q5` | 256 | 0.0009472056 | 1753.649 | 0.001188941 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q3', 'q7'], Random=['q2', 'q1'], Bottom=['q0', 'q5']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
