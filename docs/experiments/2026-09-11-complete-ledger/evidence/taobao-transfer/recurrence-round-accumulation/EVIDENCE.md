# Round accumulation: complete numerical evidence

Fixed seed 20262027; original full validation 2,871,767 rows / 268,201 users. All intervals exploratory paired user-cluster 95%, 500 draws; no training-seed or multiplicity correction.

| Model | macro AUC | Parameters | CUDA mean ms | Ratio to mature_r2 |
|---|---:|---:|---:|---:|
| mature_r2 | 0.767094951 | 51724260 | 14.157492 | 1.000000 |
| cross8_mature | 0.759406420 | 81616356 | 14.867966 | 1.050184 |
| shared_carry | 0.760394022 | 51724644 | 14.416477 | 1.018293 |
| distinct_carry | 0.760117432 | 51725028 | 14.382460 | 1.015890 |
| shared_no_carry | 0.763738068 | 51724644 | 14.319995 | 1.011478 |
| distinct_no_carry | 0.756551314 | 51725028 | 14.342538 | 1.013071 |

Model-only latency: same GPU and same 1,024-row native batch, 10 warmups + 40 CUDA event timings/model; excludes IO/loading.

| Full-validation effect | macro AUC delta [95% CI] |
|---|---|
| shared_carry__minus__mature_r2 | -0.00670093 [-0.01015154, -0.00337088] |
| shared_carry__minus__cross8_mature | +0.00098760 [-0.00177564, +0.00419064] |
| distinct_carry__minus__mature_r2 | -0.00697752 [-0.00998185, -0.00430688] |
| distinct_carry__minus__cross8_mature | +0.00071101 [-0.00236579, +0.00382375] |
| shared_no_carry__minus__mature_r2 | -0.00335688 [-0.00692807, +0.00011476] |
| shared_no_carry__minus__cross8_mature | +0.00433165 [+0.00129442, +0.00735100] |
| distinct_no_carry__minus__mature_r2 | -0.01054364 [-0.01407058, -0.00729323] |
| distinct_no_carry__minus__cross8_mature | -0.00285511 [-0.00562545, +0.00038330] |
| shared_carry__minus__shared_no_carry | -0.00334405 [-0.00637649, +0.00015664] |
| distinct_carry__minus__distinct_no_carry | +0.00356612 [+0.00034196, +0.00639850] |
| distinct_carry__minus__shared_carry | -0.00027659 [-0.00365872, +0.00292287] |
| distinct_no_carry__minus__shared_no_carry | -0.00718675 [-0.01008356, -0.00424934] |
| round_distinction_by_carry_interaction | +0.00691016 [+0.00223523, +0.01116231] |

## Fixed-checkpoint diagnostic

Confirmation: 91,850 rows / 8,353 users; discovery: 91,470 rows / 8,528 users. No user overlap. BCE harm = perturbed loss minus native loss (positive favors native route); trained gain = reference loss minus distinct_carry loss. These signs differ from the full-validation BCE difference CSV.

Groups rows: [91850, 1663, 7810, 31483, 50894]; users: [8353, 222, 1876, 5239, 1991] in order all,0–1,2–10,11–50,51–100 valid history.
Discovery-only selected pairs: {'mature_r2': [0, 3], 'shared_carry': [0, 3], 'distinct_carry': [2, 3]}. Slot indices have no established shared semantics across checkpoints.

| Effect | BCE estimate [95% CI] |
|---|---|
| all/mature_r2/zero_all/is_click | +0.00030194 [+0.00020457, +0.00039429] |
| all/mature_r2/zero_all/cart | -0.00001692 [-0.00006642, +0.00002567] |
| all/mature_r2/zero_all/fav | +0.00004536 [-0.00003710, +0.00013261] |
| all/mature_r2/zero_all/buy | -0.00000452 [-0.00006703, +0.00007937] |
| all/mature_r2/shuffle_all/is_click | +0.00016880 [+0.00002120, +0.00032661] |
| all/mature_r2/shuffle_all/cart | -0.00007199 [-0.00012465, -0.00001926] |
| all/mature_r2/shuffle_all/fav | +0.00000500 [-0.00017167, +0.00026707] |
| all/mature_r2/shuffle_all/buy | -0.00006882 [-0.00012819, +0.00000098] |
| all/shared_carry/zero_all/is_click | +0.00038887 [+0.00029514, +0.00047625] |
| all/shared_carry/zero_all/cart | +0.00000728 [-0.00003992, +0.00004767] |
| all/shared_carry/zero_all/fav | +0.00001693 [-0.00009017, +0.00012633] |
| all/shared_carry/zero_all/buy | -0.00008070 [-0.00014696, -0.00001961] |
| all/shared_carry/shuffle_all/is_click | +0.00016613 [+0.00003731, +0.00029153] |
| all/shared_carry/shuffle_all/cart | -0.00006904 [-0.00012585, -0.00000695] |
| all/shared_carry/shuffle_all/fav | +0.00003244 [-0.00008621, +0.00018278] |
| all/shared_carry/shuffle_all/buy | -0.00007330 [-0.00013224, -0.00001196] |
| all/distinct_carry/zero_all/is_click | +0.00032780 [+0.00023100, +0.00041068] |
| all/distinct_carry/zero_all/cart | +0.00002052 [-0.00002044, +0.00006053] |
| all/distinct_carry/zero_all/fav | +0.00003074 [-0.00006413, +0.00014532] |
| all/distinct_carry/zero_all/buy | -0.00006122 [-0.00010661, -0.00000070] |
| all/distinct_carry/shuffle_all/is_click | +0.00013569 [+0.00001322, +0.00026797] |
| all/distinct_carry/shuffle_all/cart | -0.00000089 [-0.00006962, +0.00007478] |
| all/distinct_carry/shuffle_all/fav | +0.00012787 [-0.00002233, +0.00032483] |
| all/distinct_carry/shuffle_all/buy | -0.00011060 [-0.00022354, -0.00002495] |
| all/distinct_carry/trained_gain_vs_mature_r2/is_click | +0.00083226 [+0.00044214, +0.00122991] |
| all/distinct_carry/trained_gain_vs_mature_r2/cart | +0.00000752 [-0.00037659, +0.00038984] |
| all/distinct_carry/trained_gain_vs_mature_r2/fav | -0.00009654 [-0.00069164, +0.00039170] |
| all/distinct_carry/trained_gain_vs_mature_r2/buy | -0.00048510 [-0.00125663, +0.00014427] |
| all/distinct_carry/trained_gain_vs_shared_carry/is_click | +0.00042177 [+0.00001997, +0.00078473] |
| all/distinct_carry/trained_gain_vs_shared_carry/cart | +0.00022938 [-0.00016430, +0.00058644] |
| all/distinct_carry/trained_gain_vs_shared_carry/fav | +0.00009691 [-0.00051829, +0.00064132] |
| all/distinct_carry/trained_gain_vs_shared_carry/buy | -0.00054008 [-0.00141279, +0.00016917] |
| all/distinct_carry/round_swap/is_click | +0.00001485 [+0.00000179, +0.00002789] |
| all/distinct_carry/round_swap/cart | +0.00000367 [-0.00000192, +0.00000849] |
| all/distinct_carry/round_swap/fav | +0.00000005 [-0.00000350, +0.00000407] |
| all/distinct_carry/round_swap/buy | +0.00000261 [-0.00000230, +0.00000823] |

All 942 effects retained in diagnostic-effects.csv. 18 direct task contrasts were added after reviewing reference point estimates; tagged exploratory. Group, slot and selected-pair contrasts are in the full CSV, including null/negative results. No inference that a null group has zero effect. AUC subgroup values below are descriptive only, without intervals.

| History group | Task | Rows | Users | Positive | Negative | AUC eligibility |
|---|---|---:|---:|---:|---:|---|
| all | is_click | 91850 | 8353 | 4185 | 87665 | descriptive_point_only |
| all | cart | 91850 | 8353 | 295 | 91555 | descriptive_point_only |
| all | fav | 91850 | 8353 | 228 | 91622 | descriptive_point_only |
| all | buy | 91850 | 8353 | 146 | 91704 | descriptive_point_only |
| empty_singleton | is_click | 1663 | 222 | 103 | 1560 | descriptive_point_only |
| empty_singleton | cart | 1663 | 222 | 14 | 1649 | insufficient |
| empty_singleton | fav | 1663 | 222 | 4 | 1659 | insufficient |
| empty_singleton | buy | 1663 | 222 | 4 | 1659 | insufficient |
| short | is_click | 7810 | 1876 | 488 | 7322 | descriptive_point_only |
| short | cart | 7810 | 1876 | 32 | 7778 | insufficient |
| short | fav | 7810 | 1876 | 17 | 7793 | insufficient |
| short | buy | 7810 | 1876 | 15 | 7795 | insufficient |
| medium | is_click | 31483 | 5239 | 1788 | 29695 | descriptive_point_only |
| medium | cart | 31483 | 5239 | 127 | 31356 | descriptive_point_only |
| medium | fav | 31483 | 5239 | 81 | 31402 | insufficient |
| medium | buy | 31483 | 5239 | 57 | 31426 | insufficient |
| long | is_click | 50894 | 1991 | 1806 | 49088 | descriptive_point_only |
| long | cart | 50894 | 1991 | 122 | 50772 | descriptive_point_only |
| long | fav | 50894 | 1991 | 126 | 50768 | descriptive_point_only |
| long | buy | 50894 | 1991 | 70 | 50824 | insufficient |

Observational per-row positive fractions are recorded in diagnostic-analysis/result.json; they are not identified individual causal benefit fractions. Q2 interventions alter the entire second work-block input (addressing, interaction and residual paths), while keeping y1. Round-row swap changes both work/prediction paths and is reported separately. Distinct updates add 384 effective parameters relative to shared updates; the factorial interaction alone does not isolate pure round identity.

Execution record: all four formal workers completed cleanly. Four smoke runners succeeded, but outer smoke verification initially failed and was recovered from saved artifacts without replay. Initial postprocess dispatcher failed before launching any child due to str/Path mismatch; recovery retained all numerical/runtime sources. No formal-training restart or new training seed.

The full-validation result contains stale prose saying twenty-nine comparisons; authoritative effect_order, coefficient matrix, tests and arrays contain exactly 13 effects / 273 intervals. This renderer documents the wording error without mutating the completed statistic artifact.
