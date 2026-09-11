# Hook Activation and Gradient Diagnostic

- Input: `workspace/qformer-loop/reports/round2/middle2-damped50/hooks.npz`
- Hooks: `8`

## `logloss_sum` / `final_query_native_route`

| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |
|---:|---|---:|---:|---:|---:|
| 1 | `q4` | 256 | 0.001049548 | 1943.125 | 0.001294446 |
| 2 | `q2` | 256 | 0.000992123 | 1836.809 | 0.001314 |
| 3 | `q5` | 256 | 0.0009799871 | 1814.34 | 0.001266346 |
| 4 | `q0` | 256 | 0.0009700631 | 1795.967 | 0.001323307 |
| 5 | `q6` | 256 | 0.0009692817 | 1794.52 | 0.001257997 |
| 6 | `q7` | 256 | 0.0009457367 | 1750.929 | 0.001253194 |
| 7 | `q1` | 256 | 0.0009379513 | 1736.516 | 0.001335179 |
| 8 | `q3` | 256 | 0.000925943 | 1714.283 | 0.00120661 |

## Equal-width candidate suggestions

- `logloss_sum` / `final_query_native_route` / 256d: Top=['q4', 'q2'], Random=['q7', 'q0'], Bottom=['q1', 'q3']

## Caveats

- Activation×gradient is a local first-order screening proxy, not causal importance.
- Ranks are comparable only within the same objective and configured Hook group.
- A present gradient array proves export, not that every loss component reaches the Hook.
- Confirm candidates with paired interventions and retrained validation outcomes.
