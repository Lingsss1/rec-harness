## 168. taobao-query-mixing-q4-channel-gate-diagnostics-5514

- 假设：Test sample-to-gate correspondence using matched offset shuffle while preserving each receiving sample entry/work states
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-09T04:53:45.708275+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-query-mixing-q4-channel-gate-diagnostics-5514",
  "parent_experiment_ids": [
    "mix_q4_channel_joint-s20262027-20260909T035059317050Z",
    "mix_q4_channel_constant-s20262027-20260909T035109726499Z"
  ],
  "family": "fixed_checkpoint_query_gate_probe",
  "code_id": "28c859a07f5b9f04f452626bee4c4e52f0b6eedc57b0a86d0be361734ec88f02",
  "hypothesis": "Test sample-to-gate correspondence using matched offset shuffle while preserving each receiving sample entry/work states",
  "change": "Native identity, zero-offset and gate-offset shuffle; fixed joint/constant q4 channel pair and shared donors",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "schema_version": 1,
    "state": "both_fixed_gate_diagnostics_completed",
    "created_at_utc": "2026-09-09T04:35:36.929579+00:00",
    "training_updates": 0,
    "new_training_runs": 0,
    "new_training_seeds": [],
    "test_evaluated": false,
    "selected_pair": [
      "mix_q4_channel_joint",
      "mix_q4_channel_constant"
    ],
    "selection_rule": "Predeclared q4 channel joint/constant pair; no result-dependent selection or extra checkpoints",
    "evidence_verification": {
      "pair_completion": true,
      "family_receipts": true,
      "all_six_export_artifact_hashes_each": true,
      "both_tool_output_hashes_each": true,
      "selected_checkpoints_match_quality": true,
      "matched_subset_rows_ids_labels_weights": true,
      "matched_donor_rows_exact": true,
      "matched_donor_row_bytes_sha256": "c00236aa7f217f428dfe2a5f1072bdc62306ab610f97e5f2cd25911bec23268f",
      "actual_validation_parquet_rehashed_in_this_interpretation": false,
      "input_scope": "Completed exporters verified all validation/metadata hashes before and after CPU inference; this interpretation rechecks output receipts, source identities and selected checkpoints, without rereading the full validation parquet"
    },
    "protocol": {
      "samples": 5514,
      "users": 4647,
      "population": 2871767,
      "modes": [
        "identity",
        "zero",
        "shuffle"
      ],
      "precision": "CPU float32",
      "torch_threads": 4,
      "worker_cpu_environment": {
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "NUMEXPR_NUM_THREADS": "4",
        "VECLIB_MAXIMUM_THREADS": "4",
        "PYTHONDONTWRITEBYTECODE": "1"
      },
      "minibatch": 128,
      "original_payload_batch": 8192,
      "observed_selected_minibatches": 353,
      "paired_tsv_rows_per_checkpoint": 55764,
      "bootstrap_iterations": 500,
      "bootstrap_block_size": 1,
      "bootstrap_seed": 20262027,
      "bootstrap_unit": "Selected user contribution within each intervention/task; stable toolkit resampling",
      "contribution_definition": "U/N * sum_i_in_user (N_stratum/n_stratum)*BCE_i; mean across selected users equals IPW population BCE estimate",
      "sign": "Perturbed minus original BCE; negative improves this frozen checkpoint/subset loss"
    },
    "families": {
      "mix_q4_channel_joint": {
        "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/runs/mix_q4_channel_joint-s20262027-20260909T035059317050Z",
        "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/runs/mix_q4_channel_joint-s20262027-20260909T035059317050Z/Taobao_Action/mix_q4_channel_joint-s20262027-20260909T035059317050Z.model",
        "checkpoint_sha256": "4eff77172c8791dc64df0bfc123e9980b0015a95262504d6504c20a62b2a743d",
        "protocol": {
          "state": "completed",
          "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/runs/mix_q4_channel_joint-s20262027-20260909T035059317050Z",
          "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/runs/mix_q4_channel_joint-s20262027-20260909T035059317050Z/Taobao_Action/mix_q4_channel_joint-s20262027-20260909T035059317050Z.model",
          "modes": [
            "identity",
            "zero",
            "shuffle"
          ],
          "seed": 20262027,
          "labels": [
            "is_click",
            "cart",
            "fav",
            "buy"
          ],
          "samples": 5514,
          "mixing_conditioning": "joint",
          "mixing_granularity": "channel",
          "mixing_rank": 16,
          "mixing_bound": 0.25,
          "num_queries": 4,
          "token_dim": 384,
          "users": 4647,
          "positive_counts": {
            "is_click": 2497,
            "cart": 2173,
            "fav": 1772,
            "buy": 1998
          },
          "validation_population": 2871767,
          "precision": "CPU float32",
          "torch_threads": 4,
          "minibatch_size": 128,
          "original_payload_batch_size": 8192,
          "minibatch_boundaries": "selected rows chunked at 128 within each original 8192-row payload",
          "shuffle_seed": "20262027 + count of previously processed selected rows",
          "shuffle_definition": "base_i + offset_permutation(i)*(work_i-entry_i); shuffle gate offsets while retaining each sample state pair",
          "training_updates": 0,
          "test_evaluated": false,
          "candidate_training_updates": 2193,
          "weighting": "IPW w_i=N_stratum/n_stratum; per-user contribution U/N * sum_i_in_user w_i*BCE_i",
          "sign": "perturbed BCE minus original BCE; positive means harmful",
          "limitations": [
            "Frozen stratified subset and one completed checkpoint; no training-seed uncertainty",
            "Tool bootstrap over selected users is conditional on subset; not sampling-design uncertainty",
            "Within-minibatch shuffle includes same-user donors and possible fixed points",
            "CPU FP32 diagnostics need not match original GPU BF16 predictions",
            "Branch interventions remove correction only and may move activations off distribution",
            "No AUC gain is inferred; necessity at fixed weights does not establish retraining benefit",
            "Zero uses the same checkpoint with mean fusion, not a retrained original model",
            "The whole-input gate affects retrieval, interaction and residual; no isolated retrieval attribution"
          ]
        },
        "perturbations": {
          "zero": {
            "is_click": {
              "original_ipw_bce": 0.18057706583378152,
              "perturbed_ipw_bce": 0.18058727059927848,
              "delta_ipw_bce": 1.0204765496962045e-05,
              "ci95": [
                -5.030836741134513e-06,
                2.4357800563652016e-05
              ],
              "interval_position": "crosses_zero",
              "users": 4647
            },
            "cart": {
              "original_ipw_bce": 0.017926234716385537,
              "perturbed_ipw_bce": 0.017925918869498632,
              "delta_ipw_bce": -3.15846886906102e-07,
              "ci95": [
                -1.7139434328638722e-06,
                1.4950080380488442e-06
              ],
              "interval_position": "crosses_zero",
              "users": 4647
            },
            "fav": {
              "original_ipw_bce": 0.013385327378269333,
              "perturbed_ipw_bce": 0.013382321502242674,
              "delta_ipw_bce": -3.0058760266606403e-06,
              "ci95": [
                -5.669880008535475e-06,
                -6.058644986344504e-07
              ],
              "interval_position": "negative",
              "users": 4647
            },
            "buy": {
              "original_ipw_bce": 0.006743103446140139,
              "perturbed_ipw_bce": 0.006742655654385224,
              "delta_ipw_bce": -4.4779175491530086e-07,
              "ci95": [
                -8.410827941407023e-07,
                4.0299075236706245e-08
              ],
              "interval_position": "crosses_zero",
              "users": 4647
            }
          },
          "shuffle": {
            "is_click": {
              "original_ipw_bce": 0.18057706583378152,
              "perturbed_ipw_bce": 0.18058423412489377,
              "delta_ipw_bce": 7.168291112259214e-06,
              "ci95": [
                -3.744813346687321e-06,
                1.9066880322491946e-05
              ],
              "interval_position": "crosses_zero",
              "users": 4647
            },
            "cart": {
              "original_ipw_bce": 0.017926234716385537,
              "perturbed_ipw_bce": 0.01792735132863872,
              "delta_ipw_bce": 1.116612253182249e-06,
              "ci95": [
                -1.0287024671596873e-06,
                3.6165133320320552e-06
              ],
              "interval_position": "crosses_zero",
              "users": 4647
            },
            "fav": {
              "original_ipw_bce": 0.013385327378269333,
              "perturbed_ipw_bce": 0.013381925219922163,
              "delta_ipw_bce": -3.4021583471712413e-06,
              "ci95": [
                -5.685328977264781e-06,
                -1.3059295141313788e-06
              ],
              "interval_position": "negative",
              "users": 4647
            },
            "buy": {
              "original_ipw_bce": 0.006743103446140139,
              "perturbed_ipw_bce": 0.006743161277632983,
              "delta_ipw_bce": 5.783149284375293e-08,
              "ci95": [
                -2.355855225839231e-07,
                3.7450404959052537e-07
              ],
              "interval_position": "crosses_zero",
              "users": 4647
            }
          }
        },
        "gate_statistics": {
          "shape": [
            5514,
            4,
            384
          ],
          "minimum": 0.43539464473724365,
          "maximum": 0.5725645273923874,
          "ipw_mean_gate": 0.4994308131008234,
          "ipw_mean_coordinate_variance": 3.4956581623400675e-05,
          "scope": "Gate variance across examples, separately per slot/channel; not a proof of useful conditioning or collapse"
        },
        "gate_maximum_absolute_difference_from_first_example": 0.07843221817165613,
        "delta_query_geometry": {
          "subset_unweighted": {
            "mean_coordinate_variance": 0.00023758786946701733,
            "total_centered_variance": 0.36493496750133864,
            "mean_correction_squared_norm": 0.5345606649075095,
            "centered_energy_fraction": 0.6826820442624228,
            "mean_delta_to_base_l2_ratio": 0.007789672406726995,
            "mean_delta_l2": 0.6728264032162928,
            "mean_base_l2": 86.6941317818739
          },
          "population_ipw": {
            "mean_coordinate_variance": 0.0002213574211582185,
            "total_centered_variance": 0.34000499889902364,
            "mean_correction_squared_norm": 0.6951809679490057,
            "centered_energy_fraction": 0.4890884741884423,
            "mean_delta_to_base_l2_ratio": 0.008893584399711461,
            "mean_delta_l2": 0.7694040423960876,
            "mean_base_l2": 86.98631554633411
          },
          "shape": [
            5514,
            4,
            384
          ],
          "all_finite": true,
          "variance_axis": "examples, independently per query slot/channel; population denominator",
          "zero_base_count": 0
        },
        "parameter_statistics": {
          "A": {
            "shape": [
              16,
              1152
            ],
            "finite": true,
            "frobenius_norm": 2.872248190162489,
            "max_abs": 0.06710272282361984,
            "nonzero_elements": 18432,
            "initial_frobenius_norm": 2.3100137794380755,
            "change_from_initial_frobenius_norm": 1.6916355340968345,
            "relative_change_from_initial": 0.7323053867273185,
            "exactly_equal_to_initial": false,
            "initial_sha256": "a2c15db5c041d6738a5c6afdcd2f1ea8b1f000166929a860358bb02e98c45a29",
            "checkpoint_sha256": "625c28b1bfcb915107ef7a77dbf1db5aec897b19a6f0f02dad803814f1877359"
          },
          "B": {
            "shape": [
              384,
              16
            ],
            "finite": true,
            "frobenius_norm": 1.0589213983282983,
            "max_abs": 0.057618334889411926,
            "nonzero_elements": 6144,
            "initial_frobenius_norm": 0.0,
            "change_from_initial_frobenius_norm": 1.0589213983282983,
            "relative_change_from_initial": null,
            "exactly_equal_to_initial": false,
            "initial_sha256": "3439ca09d1f1b6b344d40417ab9c28c0c50a87c61a03511c19ab89a8d080d5c6",
            "checkpoint_sha256": "e2c279a038a790d2e7c5f4ba97cbe7ebd769d5b4bc217a345b652a9d599f5064"
          },
          "initialization_replay": "same training seed and constructor, clone before checkpoint load; CPU init"
        },
        "donor_statistics": {
          "samples": 5514,
          "different_user_count": 4927,
          "same_user_count_including_fixed_points": 587,
          "fixed_point_count": 379,
          "different_user_fraction": 0.8935437069278201,
          "same_user_fraction_including_fixed_points": 0.10645629307217991,
          "fixed_point_fraction": 0.06873413130214001
        },
        "identity_exact": true,
        "constant_shuffle_exact": false,
        "constant_shuffle_applicable": false,
        "export_seconds": 44.021969608962536,
        "export_audit_path": "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/mix_q4_channel_joint/export/audit.json",
        "tool_report_path": "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/mix_q4_channel_joint/toolkit/perturbation_report.json"
      },
      "mix_q4_channel_constant": {
        "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/runs/mix_q4_channel_constant-s20262027-20260909T035109726499Z",
        "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/runs/mix_q4_channel_constant-s20262027-20260909T035109726499Z/Taobao_Action/mix_q4_channel_constant-s20262027-20260909T035109726499Z.model",
        "checkpoint_sha256": "eb5488f7bed2c48e4cc39aae29dde5ead4f21e96d98de35ffca20b6cb3c80f73",
        "protocol": {
          "state": "completed",
          "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/runs/mix_q4_channel_constant-s20262027-20260909T035109726499Z",
          "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/runs/mix_q4_channel_constant-s20262027-20260909T035109726499Z/Taobao_Action/mix_q4_channel_constant-s20262027-20260909T035109726499Z.model",
          "modes": [
            "identity",
            "zero",
            "shuffle"
          ],
          "seed": 20262027,
          "labels": [
            "is_click",
            "cart",
            "fav",
            "buy"
          ],
          "samples": 5514,
          "mixing_conditioning": "constant",
          "mixing_granularity": "channel",
          "mixing_rank": 16,
          "mixing_bound": 0.25,
          "num_queries": 4,
          "token_dim": 384,
          "users": 4647,
          "positive_counts": {
            "is_click": 2497,
            "cart": 2173,
            "fav": 1772,
            "buy": 1998
          },
          "validation_population": 2871767,
          "precision": "CPU float32",
          "torch_threads": 4,
          "minibatch_size": 128,
          "original_payload_batch_size": 8192,
          "minibatch_boundaries": "selected rows chunked at 128 within each original 8192-row payload",
          "shuffle_seed": "20262027 + count of previously processed selected rows",
          "shuffle_definition": "base_i + offset_permutation(i)*(work_i-entry_i); shuffle gate offsets while retaining each sample state pair",
          "training_updates": 0,
          "test_evaluated": false,
          "candidate_training_updates": 2193,
          "weighting": "IPW w_i=N_stratum/n_stratum; per-user contribution U/N * sum_i_in_user w_i*BCE_i",
          "sign": "perturbed BCE minus original BCE; positive means harmful",
          "limitations": [
            "Frozen stratified subset and one completed checkpoint; no training-seed uncertainty",
            "Tool bootstrap over selected users is conditional on subset; not sampling-design uncertainty",
            "Within-minibatch shuffle includes same-user donors and possible fixed points",
            "CPU FP32 diagnostics need not match original GPU BF16 predictions",
            "Branch interventions remove correction only and may move activations off distribution",
            "No AUC gain is inferred; necessity at fixed weights does not establish retraining benefit",
            "Zero uses the same checkpoint with mean fusion, not a retrained original model",
            "The whole-input gate affects retrieval, interaction and residual; no isolated retrieval attribution"
          ]
        },
        "perturbations": {
          "zero": {
            "is_click": {
              "original_ipw_bce": 0.18170596162670669,
              "perturbed_ipw_bce": 0.18170616448025279,
              "delta_ipw_bce": 2.0285354610037612e-07,
              "ci95": [
                5.164492231017958e-09,
                4.337338648627793e-07
              ],
              "interval_position": "positive",
              "users": 4647
            },
            "cart": {
              "original_ipw_bce": 0.019322289655011957,
              "perturbed_ipw_bce": 0.019322494706861894,
              "delta_ipw_bce": 2.0505184993905237e-07,
              "ci95": [
                -9.721513012866394e-09,
                6.071386973686327e-07
              ],
              "interval_position": "crosses_zero",
              "users": 4647
            },
            "fav": {
              "original_ipw_bce": 0.013259621660404867,
              "perturbed_ipw_bce": 0.013259634386427209,
              "delta_ipw_bce": 1.2726022342258106e-08,
              "ci95": [
                -1.701479670132352e-08,
                3.984729750412884e-08
              ],
              "interval_position": "crosses_zero",
              "users": 4647
            },
            "buy": {
              "original_ipw_bce": 0.006646721098703435,
              "perturbed_ipw_bce": 0.006646712927162072,
              "delta_ipw_bce": -8.171541362462717e-09,
              "ci95": [
                -1.7505650694848714e-08,
                5.546847299896314e-10
              ],
              "interval_position": "crosses_zero",
              "users": 4647
            }
          },
          "shuffle": {
            "is_click": {
              "original_ipw_bce": 0.18170596162670669,
              "perturbed_ipw_bce": 0.18170596162670669,
              "delta_ipw_bce": 0.0,
              "ci95": [
                0.0,
                0.0
              ],
              "interval_position": "exact_zero_implementation_control",
              "users": 4647
            },
            "cart": {
              "original_ipw_bce": 0.019322289655011957,
              "perturbed_ipw_bce": 0.019322289655011957,
              "delta_ipw_bce": 0.0,
              "ci95": [
                0.0,
                0.0
              ],
              "interval_position": "exact_zero_implementation_control",
              "users": 4647
            },
            "fav": {
              "original_ipw_bce": 0.013259621660404867,
              "perturbed_ipw_bce": 0.013259621660404867,
              "delta_ipw_bce": 0.0,
              "ci95": [
                0.0,
                0.0
              ],
              "interval_position": "exact_zero_implementation_control",
              "users": 4647
            },
            "buy": {
              "original_ipw_bce": 0.006646721098703435,
              "perturbed_ipw_bce": 0.006646721098703435,
              "delta_ipw_bce": 0.0,
              "ci95": [
                0.0,
                0.0
              ],
              "interval_position": "exact_zero_implementation_control",
              "users": 4647
            }
          }
        },
        "gate_statistics": {
          "shape": [
            5514,
            4,
            384
          ],
          "minimum": 0.4988792116055265,
          "maximum": 0.5009774029022083,
          "ipw_mean_gate": 0.4999334321039129,
          "ipw_mean_coordinate_variance": 2.1286829415063223e-28,
          "scope": "Gate variance across examples, separately per slot/channel; not a proof of useful conditioning or collapse"
        },
        "gate_maximum_absolute_difference_from_first_example": 0.0,
        "delta_query_geometry": {
          "subset_unweighted": {
            "mean_coordinate_variance": 1.8609946662058192e-07,
            "total_centered_variance": 0.0002858487807292138,
            "mean_correction_squared_norm": 0.0006303942045024302,
            "centered_energy_fraction": 0.4534444934417411,
            "mean_delta_to_base_l2_ratio": 0.0002774441075108076,
            "mean_delta_l2": 0.024643447225798296,
            "mean_base_l2": 88.1080416658009
          },
          "population_ipw": {
            "mean_coordinate_variance": 1.129726301556168e-07,
            "total_centered_variance": 0.0001735259599190274,
            "mean_correction_squared_norm": 0.0005742520097469398,
            "centered_energy_fraction": 0.30217736633694403,
            "mean_delta_to_base_l2_ratio": 0.00026829406804860395,
            "mean_delta_l2": 0.023677318100724126,
            "mean_base_l2": 87.77171490396262
          },
          "shape": [
            5514,
            4,
            384
          ],
          "all_finite": true,
          "variance_axis": "examples, independently per query slot/channel; population denominator",
          "zero_base_count": 0
        },
        "parameter_statistics": {
          "A": {
            "shape": [
              16,
              1152
            ],
            "finite": true,
            "frobenius_norm": 2.3116963634756327,
            "max_abs": 0.03416569158434868,
            "nonzero_elements": 18432,
            "initial_frobenius_norm": 2.3100137794380755,
            "change_from_initial_frobenius_norm": 0.08874210029880052,
            "relative_change_from_initial": 0.03841626447803595,
            "exactly_equal_to_initial": false,
            "initial_sha256": "a2c15db5c041d6738a5c6afdcd2f1ea8b1f000166929a860358bb02e98c45a29",
            "checkpoint_sha256": "29a39e14ffaaac02bfcc099fce3c913147fc08618e6e7740a08bddf39313d89c"
          },
          "B": {
            "shape": [
              384,
              16
            ],
            "finite": true,
            "frobenius_norm": 0.5876281880811561,
            "max_abs": 0.03346674516797066,
            "nonzero_elements": 6144,
            "initial_frobenius_norm": 0.0,
            "change_from_initial_frobenius_norm": 0.5876281880811561,
            "relative_change_from_initial": null,
            "exactly_equal_to_initial": false,
            "initial_sha256": "3439ca09d1f1b6b344d40417ab9c28c0c50a87c61a03511c19ab89a8d080d5c6",
            "checkpoint_sha256": "6b021cb614e15062e68e8cefd9123b72ea31c47225b9ea0ce9127f23e5d7077d"
          },
          "initialization_replay": "same training seed and constructor, clone before checkpoint load; CPU init"
        },
        "donor_statistics": {
          "samples": 5514,
          "different_user_count": 4927,
          "same_user_count_including_fixed_points": 587,
          "fixed_point_count": 379,
          "different_user_fraction": 0.8935437069278201,
          "same_user_fraction_including_fixed_points": 0.10645629307217991,
          "fixed_point_fraction": 0.06873413130214001
        },
        "identity_exact": true,
        "constant_shuffle_exact": true,
        "constant_shuffle_applicable": true,
        "export_seconds": 39.322688706219196,
        "export_audit_path": "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/mix_q4_channel_constant/export/audit.json",
        "tool_report_path": "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/mix_q4_channel_constant/toolkit/perturbation_report.json"
      }
    },
    "findings": [
      {
        "id": "joint_fav",
        "claim": "Joint fav has negative zero and shuffle intervals. In this fixed checkpoint/subset, removing or misassigning the gate lowers fav BCE; this does not support useful sample-to-gate correspondence for fav.",
        "zero": {
          "original_ipw_bce": 0.013385327378269333,
          "perturbed_ipw_bce": 0.013382321502242674,
          "delta_ipw_bce": -3.0058760266606403e-06,
          "ci95": [
            -5.669880008535475e-06,
            -6.058644986344504e-07
          ],
          "interval_position": "negative",
          "users": 4647
        },
        "shuffle": {
          "original_ipw_bce": 0.013385327378269333,
          "perturbed_ipw_bce": 0.013381925219922163,
          "delta_ipw_bce": -3.4021583471712413e-06,
          "ci95": [
            -5.685328977264781e-06,
            -1.3059295141313788e-06
          ],
          "interval_position": "negative",
          "users": 4647
        }
      },
      {
        "id": "joint_other_tasks",
        "claim": "Joint is_click/cart/buy zero and shuffle intervals cross zero; effect direction is unresolved, not equivalence or evidence that inputs are unused."
      },
      {
        "id": "constant_zero_click",
        "claim": "Constant zero/is_click has a strictly positive but very small interval: the learned static mixture has a tiny conditional BCE advantage over zero-offset mean fusion on this subset. This is not evidence for useful input-conditioned mixing.",
        "evidence": {
          "original_ipw_bce": 0.18170596162670669,
          "perturbed_ipw_bce": 0.18170616448025279,
          "delta_ipw_bce": 2.0285354610037612e-07,
          "ci95": [
            5.164492231017958e-09,
            4.337338648627793e-07
          ],
          "interval_position": "positive",
          "users": 4647
        }
      },
      {
        "id": "constant_other_zero",
        "claim": "Constant zero cart/fav/buy intervals cross zero."
      },
      {
        "id": "constant_shuffle_control",
        "claim": "Constant shuffle is exactly invariant in probabilities and BCE for all four tasks. This confirms the expected implementation negative control, not that user/item/history data have no effect."
      },
      {
        "id": "gate_vs_delta",
        "claim": "Gate coefficient variance and delta=(gate-.5)*(work-entry) variance are different quantities. Constant gate is exactly identical across examples; its reported ~2e-28 variance is accumulation residue, while delta still varies because work-entry varies. Joint nonzero gate variance alone does not establish useful personalization."
      }
    ],
    "training_context": {
      "quality_summary_path": "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/reports/quality-summary.json",
      "quality_sha256": "d6d77b983ba2d5790d3fe26aa89e3fd919bbfa120fd273482bbd0cd89a06a08f",
      "macro_auc": {
        "mix_q4_channel_joint": 0.8282286909051509,
        "mix_q4_channel_constant": 0.8276819214156276
      },
      "delta_macro_auc_to_same_q_original": {
        "mix_q4_channel_joint": -0.0002438745210926463,
        "mix_q4_channel_constant": -0.0007906440106160151,
        "mix_q4_query_joint": -0.00018370257401822965,
        "mix_q4_query_constant": -0.002888358984481587,
        "mix_q8_channel_joint": -0.004995945974868965,
        "mix_q8_channel_constant": -0.0024559520635528687,
        "mix_q8_query_joint": -0.0029945913700639704,
        "mix_q8_query_constant": -0.0014206449336429117
      },
      "all_eight_candidates_below_original_at_point_estimate": true,
      "selected_joint_minus_constant_macro_auc": 0.0005467694895233688,
      "full_training_paired_bootstrap_read": false,
      "postprocess_state_observed": "paired_bootstrap",
      "statement": "Full released validation AUC and fixed stratified-subset IPW BCE measure different outcomes. The joint checkpoint beating its constant control in point AUC does not compensate for both remaining below the original recurrence. This diagnostic cannot rescue the training outcome; pending full-training uncertainty is handled separately."
    },
    "limitations": [
      "Conditional on two fixed checkpoints and one fixed stratified subset; no training-seed or sampling-design uncertainty.",
      "Stable-tool task intervals are not corrected for multiple interventions/tasks or model selection; do not synthesize an uncomputed macro interval.",
      "Within-minibatch donors include 379 fixed points and 587 same-user assignments; identical donor rows do not remove those limitations.",
      "Zero uses the trained checkpoint with mean fusion, not a retrained mean-fusion baseline; shuffle changes the complete second work query, not an isolated retrieval projection.",
      "No test data, new optimizer updates, new seed, or new model run was used. No memory was consolidated by this artifact."
    ],
    "input_sha256": {
      "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/completion.json": "7d7012e703f7d6e4693cf0fe6c7e61f31404931b2cadce8dcae53041cc904baf",
      "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/reports/quality-summary.json": "d6d77b983ba2d5790d3fe26aa89e3fd919bbfa120fd273482bbd0cd89a06a08f",
      "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/analysis-plan.json": "507b304b810966db43e2776e634ad4215741950b9dba84dd8908ea3615667388",
      "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/diagnostic-identity.json": "74e9723f50dea6a15fd0a44c5a534994f56c6afa12974cedfc096deab6da6ee8",
      "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/run_when_ready.py": "28c859a07f5b9f04f452626bee4c4e52f0b6eedc57b0a86d0be361734ec88f02",
      "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/gate_probe.py": "2c995f54af16de23db1c8253f1dd807ac012d8e6320e19e4717c065f6fd0c6a8",
      "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/test_gate_probe.py": "bf903a8f7f2025478dba0425ffb350a939fbbb4e2548acd666c4fc2ac917ba96",
      "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/export_gates.py": "4e4aca816a8154c8deb96adb6fe0e7e89c4755e3e827e5d1e3bab00385851bb5",
      "workspace/qformer-loop/taobao-transfer/query-conditioning/reports/paired-query-parts/subset.npz": "a7986436bc968fb6fbe1379df4ed662648d8fe16d2ae2bda1b67475395f8ae0f",
      "workspace/qformer-loop/taobao-transfer/query-conditioning/reports/paired-query-parts/protocol.json": "3da5e80709e9104e06793daaec1c701122e920b72d9cf0a8b74c72cc7b6a276d",
      "workspace/rec-harness/model-diagnostic-toolkit/scripts/analyze_perturbations.py": "ca02990e4be3c2a1c640f97be75b5784ea4bbd06c432992fdc7cfedea52125ed",
      "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/mix_q4_channel_joint/completion.json": "8baca4b77015479dc532e1ccabf5cc7c33abaa73ab0af2b1d58be962ab9b0b14",
      "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/mix_q4_channel_joint/export-exit.json": "a72b5662f33b999a3dab8d4667fc35348677977afc809970c6eeab36ebcbc6ff",
      "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/mix_q4_channel_joint/tool-exit.json": "3126565c606dc20c3faf9122511a849f2cb7184c2c8df485fe33c30ec3e6c8c4",
      "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/mix_q4_channel_joint/export/completion.json": "420f059d88605a2d5f0b3160cae2f9237268c7c00d6d55c6239ade36861eaa98",
      "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/mix_q4_channel_joint/export/losses.npz": "6f931b37509f749632d5c88efc7d22868defd251114bbe87b982d584df479c2a",
      "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/mix_q4_channel_joint/export/paired-user-index.npz": "914b8ba92a0b16c3122f71acf2c5be7caf175b85c88b15695dce7f93d13a4da3",
      "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/mix_q4_channel_joint/export/paired.tsv": "e35120c3b3b1b08eafd0bcb36112fb37815a4d171854372f8689d65d7c5d84ac",
      "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/mix_q4_channel_joint/export/audit.json": "e42cd15de1922e99369426acefdc19e06c62940a7bb515a349df2092d410ec2e",
      "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/mix_q4_channel_joint/export/identities.json": "380bd69b9493aa1094fadf753d459449e030fbce936ce79a28264ba7265f80df",
      "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/mix_q4_channel_joint/export/protocol.json": "be9b96d235eccf9c96d1d003e1ff5c0f230e4cf6e459ca030c39c448c4ed0d39",
      "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/mix_q4_channel_joint/toolkit/perturbation_report.json": "9a681fc27639c5f7bf827baa81312e4bd14892bf0e08a2b230512adbfd399eff",
      "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/mix_q4_channel_joint/toolkit/perturbation_report.md": "83ca1658d3036be83985eb172f5b37f81ebecdbaebfa02ef42e4fba77b7c12d5",
      "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/runs/mix_q4_channel_joint-s20262027-20260909T035059317050Z/result.json": "08e7cd5b020000c77bc09261a76fcb709d7728dcdc6e8ac4644031fadb6b3e95",
      "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/runs/mix_q4_channel_joint-s20262027-20260909T035059317050Z/manifest.json": "b65a28cac3d8453e46313afe7030e5c081751daad60b96dfbfe5641c00913f02",
      "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/runs/mix_q4_channel_joint-s20262027-20260909T035059317050Z/Taobao_Action/mix_q4_channel_joint-s20262027-20260909T035059317050Z.model": "4eff77172c8791dc64df0bfc123e9980b0015a95262504d6504c20a62b2a743d",
      "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/mix_q4_channel_constant/completion.json": "c4e3af7b733486d6dd1340bd6772475fc36fbd93425193a395f76d292f1be16d",
      "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/mix_q4_channel_constant/export-exit.json": "5f8a61702cc95bb96c27625abc653e07d3ace5ccafc09682cf77a7c7a343b350",
      "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/mix_q4_channel_constant/tool-exit.json": "fae873c569cb5b4a766ac97d92fd9925f16ddcc9f3d3eaf1b8babd02976dbefa",
      "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/mix_q4_channel_constant/export/completion.json": "9f2a48f2e5cebb89d4c196b55b75bdaf998865fb48f813199b5f2efd76ca9a41",
      "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/mix_q4_channel_constant/export/losses.npz": "259717534d652b4a752d7253b9bae05a9dfa785d2fd5f95cb69fc737140da665",
      "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/mix_q4_channel_constant/export/paired-user-index.npz": "914b8ba92a0b16c3122f71acf2c5be7caf175b85c88b15695dce7f93d13a4da3",
      "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/mix_q4_channel_constant/export/paired.tsv": "8c45e6881b258228ef872955c8c7d619460f70184301910d40d1129c6e12cb1c",
      "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/mix_q4_channel_constant/export/audit.json": "9b259d66c5e01dac50c78da290acfe378051944d69c20c60f141e5d9d30d5f23",
      "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/mix_q4_channel_constant/export/identities.json": "6ba24c31b1be92c3c23678ee7079b32818a8d3dc3abb49d0734e1c02a08a4d8a",
      "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/mix_q4_channel_constant/export/protocol.json": "735f4a58bde5ccf1a45fa3e8c826016f19bc2670dd29348cc03a23600bace916",
      "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/mix_q4_channel_constant/toolkit/perturbation_report.json": "72913052a7d9be1738b30428c9b23d27c4a621470d87973651bcc2078f827237",
      "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/mix_q4_channel_constant/toolkit/perturbation_report.md": "daa04931cb52916b6f7b7d73b08bc8942c33493059ff1064935c73f0eb1ade8d",
      "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/runs/mix_q4_channel_constant-s20262027-20260909T035109726499Z/result.json": "7caf1a409660d38072e717c4a7404bf15aec55a74728f2af12a2cdfbe6b93c25",
      "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/runs/mix_q4_channel_constant-s20262027-20260909T035109726499Z/manifest.json": "f7182d070b064495d5113513712b015f34864152e7cf9e48d5466c2f7ab6b717",
      "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/runs/mix_q4_channel_constant-s20262027-20260909T035109726499Z/Taobao_Action/mix_q4_channel_constant-s20262027-20260909T035109726499Z.model": "eb5488f7bed2c48e4cc39aae29dde5ead4f21e96d98de35ffca20b6cb3c80f73"
    },
    "memory_consolidated_at_analysis_creation": false,
    "memory_consolidation_receipt": "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/memory-consolidation-status.json"
  },
  "decision": "provisional",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/interpretation.json",
    "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/RESULTS.md",
    "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/memory-consolidation-status.json"
  ]
}
```

</details>
