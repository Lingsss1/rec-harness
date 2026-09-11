# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round6/r6_small_attention_after/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q0` | 256 | 0.001117955 | 2069.772 | 0.001422037 |
| 2 | `q2` | 256 | 0.001077606 | 1995.072 | 0.001376777 |
| 3 | `q5` | 256 | 0.001041278 | 1927.815 | 0.001384319 |
| 4 | `q3` | 256 | 0.001036429 | 1918.837 | 0.001365909 |
| 5 | `q4` | 256 | 0.001035907 | 1917.869 | 0.001383721 |
| 6 | `q1` | 256 | 0.001017886 | 1884.507 | 0.001398871 |
| 7 | `q6` | 256 | 0.001008557 | 1867.234 | 0.001341993 |
| 8 | `q7` | 256 | 0.000983185 | 1820.261 | 0.00135803 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q0', 'q2'], Random=['q1', 'q3'], Bottom=['q6', 'q7']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
