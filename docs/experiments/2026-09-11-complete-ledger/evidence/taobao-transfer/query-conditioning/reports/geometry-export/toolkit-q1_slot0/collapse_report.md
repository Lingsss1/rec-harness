# Representation Collapse Diagnostic

Overall heuristic risk: **MEDIUM**

## Data

- Shape: `4096 × 384`
- Removed non-finite rows: `0`
- Random pairs sampled: `20000`
- Rows used for spectrum: `4096`

## Core metrics

| Metric | Value |
|---|---:|
| Centered/raw energy ratio | 0.235905 |
| Dead-dimension ratio | 0 |
| Random-pair cosine mean | 0.764766 |
| Random-pair cosine p05 / median / p95 | 0.60992 / 0.77421 / 0.889564 |
| Effective rank / available rank | 56.139 / 384 |
| Effective-rank ratio | 0.146195 |
| Top-1 variance ratio | 0.0993177 |
| Top-5 variance ratio | 0.365907 |

## Findings

- **dimensional_concentration (medium)**: The singular-value spectrum uses less than 15% of the available rank.

## Label-aware metrics

- Unique Labels: `2`
- Sampled Same Label Pairs: `18254`
- Sampled Different Label Pairs: `1746`
- Within Label Cosine Mean: `0.765414`
- Between Label Cosine Mean: `0.757982`
- Within Between Cosine Gap: `0.00743241`

## Caveats

- Heuristic flags are not proof; compare against healthy baselines and downstream performance.
- High cosine similarity can reflect a common mean or anisotropy rather than complete collapse.
- Low effective rank can be natural when data lies on a low-dimensional manifold.
