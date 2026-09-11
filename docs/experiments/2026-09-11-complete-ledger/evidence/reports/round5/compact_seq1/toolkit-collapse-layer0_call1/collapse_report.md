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
| Centered/raw energy ratio | 0.210406 |
| Dead-dimension ratio | 0 |
| Random-pair cosine mean | 0.788854 |
| Random-pair cosine p05 / median / p95 | 0.521463 / 0.819089 / 0.955666 |
| Effective rank / available rank | 9.86576 / 256 |
| Effective-rank ratio | 0.0385381 |
| Top-1 variance ratio | 0.440677 |
| Top-5 variance ratio | 0.794017 |

## Findings

- **dimensional_concentration (medium)**: The singular-value spectrum uses less than 5% of the available rank.

## Label-aware metrics

- Unique Labels: `2`
- Sampled Same Label Pairs: `17241`
- Sampled Different Label Pairs: `2759`
- Within Label Cosine Mean: `0.798791`
- Between Label Cosine Mean: `0.726757`
- Within Between Cosine Gap: `0.0720345`

## Caveats

- Heuristic flags are not proof; compare against healthy baselines and downstream performance.
- High cosine similarity can reflect a common mean or anisotropy rather than complete collapse.
- Low effective rank can be natural when data lies on a low-dimensional manifold.
