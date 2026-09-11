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
| Centered/raw energy ratio | 0.272987 |
| Dead-dimension ratio | 0 |
| Random-pair cosine mean | 0.726623 |
| Random-pair cosine p05 / median / p95 | 0.375387 / 0.77056 / 0.935836 |
| Effective rank / available rank | 11.4395 / 256 |
| Effective-rank ratio | 0.0446856 |
| Top-1 variance ratio | 0.387752 |
| Top-5 variance ratio | 0.758913 |

## Findings

- **dimensional_concentration (medium)**: The singular-value spectrum uses less than 5% of the available rank.

## Label-aware metrics

- Unique Labels: `2`
- Sampled Same Label Pairs: `17241`
- Sampled Different Label Pairs: `2759`
- Within Label Cosine Mean: `0.743977`
- Between Label Cosine Mean: `0.618177`
- Within Between Cosine Gap: `0.125801`

## Caveats

- Heuristic flags are not proof; compare against healthy baselines and downstream performance.
- High cosine similarity can reflect a common mean or anisotropy rather than complete collapse.
- Low effective rank can be natural when data lies on a low-dimensional manifold.
