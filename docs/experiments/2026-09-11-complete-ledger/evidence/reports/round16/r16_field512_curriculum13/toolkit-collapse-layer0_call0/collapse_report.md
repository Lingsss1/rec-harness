# Representation Collapse Diagnostic

Overall heuristic risk: **MEDIUM**

## Data

- Shape: `7232 × 256`
- Removed non-finite rows: `0`
- Random pairs sampled: `20000`
- Rows used for spectrum: `7232`

## Core metrics

| Metric | Value |
|---|---:|
| Centered/raw energy ratio | 0.338106 |
| Dead-dimension ratio | 0 |
| Random-pair cosine mean | 0.660708 |
| Random-pair cosine p05 / median / p95 | 0.184557 / 0.72286 / 0.92531 |
| Effective rank / available rank | 10.6906 / 256 |
| Effective-rank ratio | 0.0417602 |
| Top-1 variance ratio | 0.392389 |
| Top-5 variance ratio | 0.78251 |

## Findings

- **dimensional_concentration (medium)**: The singular-value spectrum uses less than 5% of the available rank.

## Label-aware metrics

- Unique Labels: `2`
- Sampled Same Label Pairs: `17241`
- Sampled Different Label Pairs: `2759`
- Within Label Cosine Mean: `0.68457`
- Between Label Cosine Mean: `0.5116`
- Within Between Cosine Gap: `0.17297`

## Caveats

- Heuristic flags are not proof; compare against healthy baselines and downstream performance.
- High cosine similarity can reflect a common mean or anisotropy rather than complete collapse.
- Low effective rank can be natural when data lies on a low-dimensional manifold.
