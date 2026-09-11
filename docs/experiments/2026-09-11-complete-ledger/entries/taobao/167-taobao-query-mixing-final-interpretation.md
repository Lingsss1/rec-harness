## 167. taobao-query-mixing-final-interpretation

- 假设：Joint bounded mixing should improve matched static controls and original recurrence if conditioning adds useful training benefit
- 判定：discard
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-09T04:51:29.374586+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-query-mixing-final-interpretation",
  "family": "round_interpretation",
  "code_id": "5ce023190b426cdf552c1406b338376cc2fad8f1eb0bb2c75df1daf4fe93f601",
  "hypothesis": "Joint bounded mixing should improve matched static controls and original recurrence if conditioning adds useful training benefit",
  "change": "Adjudicate eight completed training episodes under the prospective quality/compute policy; preserve historical rows and pending diagnostic evidence",
  "supersedes_episode_ids": [
    "mix_q4_channel_joint-s20262027-20260909T035059317050Z",
    "mix_q4_channel_constant-s20262027-20260909T035109726499Z",
    "mix_q4_query_joint-s20262027-20260909T035110579796Z",
    "mix_q4_query_constant-s20262027-20260909T035110802164Z",
    "mix_q8_channel_joint-s20262027-20260909T035053715005Z",
    "mix_q8_channel_constant-s20262027-20260909T035059340196Z",
    "mix_q8_query_joint-s20262027-20260909T035104953096Z",
    "mix_q8_query_constant-s20262027-20260909T035108993695Z"
  ],
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "state": "completed_training_adjudication_diagnostics_separate",
    "new_training_runs": 0,
    "adjudicated_training_runs": 8,
    "training_count": 109,
    "decisions": {
      "mix_q4_channel_joint": {
        "decision": "discard",
        "retained_by_queue_policy": false,
        "supersedes_training_episode": "mix_q4_channel_joint-s20262027-20260909T035059317050Z",
        "macro_auc": 0.8282286909051509,
        "parameter_count": 51748804,
        "per_task": {
          "is_click": {
            "logloss": 0.18197870182049147,
            "AUC": 0.6670218349990896
          },
          "cart": {
            "logloss": 0.017785875712606477,
            "AUC": 0.8661792510424196
          },
          "fav": {
            "logloss": 0.013340235487606105,
            "AUC": 0.88254085953893
          },
          "buy": {
            "logloss": 0.006763304771558457,
            "AUC": 0.8971728180401648
          }
        },
        "point_task_tradeoffs": {
          "cart": {
            "auc_delta": -0.0005357268052302855,
            "logloss_delta": 2.419264972502652e-05
          },
          "fav": {
            "auc_delta": -0.0001330516824029626,
            "logloss_delta": -2.5782306015541268e-05
          },
          "buy": {
            "auc_delta": -0.0013545075722954936,
            "logloss_delta": 4.6650580009272025e-05
          }
        },
        "versus_original": {
          "reference": "ctx_q4",
          "candidate": "mix_q4_channel_joint",
          "reference_role": "original_recurrence",
          "delta_macro_auc": -0.0002438745210926463,
          "delta_auc": {
            "is_click": 0.0010477879755581565,
            "cart": -0.0005357268052302855,
            "fav": -0.0001330516824029626,
            "buy": -0.0013545075722954936
          },
          "delta_logloss": {
            "is_click": -0.0003349434426095921,
            "cart": 2.419264972502652e-05,
            "fav": -2.5782306015541268e-05,
            "buy": 4.6650580009272025e-05
          },
          "delta_macro_logloss": -7.247062972270957e-05,
          "parameter_ratio": 1.0004751351726313
        },
        "original_bootstrap": {
          "reference": "ctx_q4",
          "candidate": "mix_q4_channel_joint",
          "users": 268201,
          "repetitions": 500,
          "bootstrap_seed": 20260906,
          "training_updates": 0,
          "new_training_seeds": [],
          "reference_role": "original_recurrence",
          "mandatory": true,
          "mandatory_joint_constant_contrast": false,
          "mandatory_original_recurrence_contrast": true,
          "delta_auc": {
            "is_click": 0.0010477879755581565,
            "cart": -0.0005357268052302855,
            "fav": -0.0001330516824029626,
            "buy": -0.0013545075722954936
          },
          "ci95_auc": {
            "is_click": [
              0.0004861313039931425,
              0.0015792725353308583
            ],
            "cart": [
              -0.002235592051036747,
              0.0010961163527792054
            ],
            "fav": [
              -0.0016314016624072946,
              0.001227642832564915
            ],
            "buy": [
              -0.0036294354134415283,
              0.0006782916088449606
            ]
          },
          "delta_macro_auc": -0.0002438745210926463,
          "ci95_macro_auc": [
            -0.001100299388456618,
            0.000557029552632559
          ],
          "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
        },
        "original_evidence": "inconclusive_crosses_zero",
        "versus_best_single": {
          "reference": "psgd_precond_every_step",
          "candidate": "mix_q4_channel_joint",
          "reference_role": "best_single_pass",
          "delta_macro_auc": -0.00039484161452374433,
          "delta_auc": {
            "is_click": -0.0007679433923523105,
            "cart": -0.0008397755986456756,
            "fav": 0.00017963065154313895,
            "buy": -0.0001512781186401302
          },
          "delta_logloss": {
            "is_click": -2.5283058038783057e-05,
            "cart": 0.00015084450868941765,
            "fav": 6.991349026806183e-05,
            "buy": 0.00014142499913227542
          },
          "delta_macro_logloss": 8.422498501273884e-05,
          "parameter_ratio": 0.6340496785912583
        },
        "optional_single_bootstrap": null,
        "forward_mean_ms": 14.495008063316345,
        "forward_ratio_to_original": 1.0074550684083148,
        "near_reference_policy_passed": false,
        "conditioning_evidence": {
          "point_case": "joint_better_than_constant_but_not_above_original",
          "comparison": {
            "reference": "mix_q4_channel_constant",
            "candidate": "mix_q4_channel_joint",
            "reference_role": "conditioning_control",
            "delta_macro_auc": 0.0005467694895233688,
            "delta_auc": {
              "is_click": 0.0009396760089108724,
              "cart": -0.00030779292300375793,
              "fav": 0.0003297028138634639,
              "buy": 0.0012254920583231188
            },
            "delta_logloss": {
              "is_click": -0.00019831593619590304,
              "cart": -2.8927485681600612e-05,
              "fav": -3.953922521408686e-05,
              "buy": -3.7646928037400954e-05
            },
            "delta_macro_logloss": -7.610739378224873e-05,
            "parameter_ratio": 1.0
          },
          "bootstrap": {
            "reference": "mix_q4_channel_constant",
            "candidate": "mix_q4_channel_joint",
            "users": 268201,
            "repetitions": 500,
            "bootstrap_seed": 20260906,
            "training_updates": 0,
            "new_training_seeds": [],
            "reference_role": "conditioning_control",
            "mandatory": true,
            "mandatory_joint_constant_contrast": true,
            "mandatory_original_recurrence_contrast": false,
            "delta_auc": {
              "is_click": 0.0009396760089108724,
              "cart": -0.00030779292300375793,
              "fav": 0.0003297028138634639,
              "buy": 0.0012254920583231188
            },
            "ci95_auc": {
              "is_click": [
                0.0003943137650293805,
                0.0014970511368180475
              ],
              "cart": [
                -0.0018891050093504669,
                0.0012896822093320906
              ],
              "fav": [
                -0.0011219161088202034,
                0.0016688241167196354
              ],
              "buy": [
                -0.0011389509503691025,
                0.0036465499866002904
              ]
            },
            "delta_macro_auc": 0.0005467694895233688,
            "ci95_macro_auc": [
              -0.00034580346755074193,
              0.0014149764367677583
            ],
            "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
          },
          "joint_constant_evidence": "inconclusive_crosses_zero",
          "original_evidence": "inconclusive_crosses_zero",
          "both_conditional_intervals_positive": false,
          "scope": "Input-conditioned versus static parameterization under this fixed seed/budget; not isolated retrieval or training-seed robustness"
        }
      },
      "mix_q4_channel_constant": {
        "decision": "discard",
        "retained_by_queue_policy": false,
        "supersedes_training_episode": "mix_q4_channel_constant-s20262027-20260909T035109726499Z",
        "macro_auc": 0.8276819214156276,
        "parameter_count": 51748804,
        "per_task": {
          "is_click": {
            "logloss": 0.18217701775668738,
            "AUC": 0.6660821589901788
          },
          "cart": {
            "logloss": 0.017814803198288078,
            "AUC": 0.8664870439654233
          },
          "fav": {
            "logloss": 0.013379774712820192,
            "AUC": 0.8822111567250666
          },
          "buy": {
            "logloss": 0.006800951699595858,
            "AUC": 0.8959473259818417
          }
        },
        "point_task_tradeoffs": {
          "cart": {
            "auc_delta": -0.00022793388222652755,
            "logloss_delta": 5.3120135406627134e-05
          },
          "fav": {
            "auc_delta": -0.0004627544962664265,
            "logloss_delta": 1.3756919198545589e-05
          },
          "buy": {
            "auc_delta": -0.0025799996306186124,
            "logloss_delta": 8.429750804667298e-05
          }
        },
        "versus_original": {
          "reference": "ctx_q4",
          "candidate": "mix_q4_channel_constant",
          "reference_role": "original_recurrence",
          "delta_macro_auc": -0.0007906440106160151,
          "delta_auc": {
            "is_click": 0.00010811196664728406,
            "cart": -0.00022793388222652755,
            "fav": -0.0004627544962664265,
            "buy": -0.0025799996306186124
          },
          "delta_logloss": {
            "is_click": -0.00013662750641368904,
            "cart": 5.3120135406627134e-05,
            "fav": 1.3756919198545589e-05,
            "buy": 8.429750804667298e-05
          },
          "delta_macro_logloss": 3.636764059539166e-06,
          "parameter_ratio": 1.0004751351726313
        },
        "original_bootstrap": {
          "reference": "ctx_q4",
          "candidate": "mix_q4_channel_constant",
          "users": 268201,
          "repetitions": 500,
          "bootstrap_seed": 20260906,
          "training_updates": 0,
          "new_training_seeds": [],
          "reference_role": "original_recurrence",
          "mandatory": true,
          "mandatory_joint_constant_contrast": false,
          "mandatory_original_recurrence_contrast": true,
          "delta_auc": {
            "is_click": 0.00010811196664728406,
            "cart": -0.00022793388222652755,
            "fav": -0.0004627544962664265,
            "buy": -0.0025799996306186124
          },
          "ci95_auc": {
            "is_click": [
              -0.00040503714291564575,
              0.0006541029600254017
            ],
            "cart": [
              -0.001821141295435813,
              0.0014881386366081685
            ],
            "fav": [
              -0.002042880667650912,
              0.0009823466517281938
            ],
            "buy": [
              -0.004687336377400162,
              -0.0004541224120995038
            ]
          },
          "delta_macro_auc": -0.0007906440106160151,
          "ci95_macro_auc": [
            -0.0015777067009283365,
            6.777985181484609e-05
          ],
          "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
        },
        "original_evidence": "inconclusive_crosses_zero",
        "versus_best_single": {
          "reference": "psgd_precond_every_step",
          "candidate": "mix_q4_channel_constant",
          "reference_role": "best_single_pass",
          "delta_macro_auc": -0.0009416111040471131,
          "delta_auc": {
            "is_click": -0.001707619401263183,
            "cart": -0.0005319826756419177,
            "fav": -0.00015007216232032494,
            "buy": -0.001376770176963249
          },
          "delta_logloss": {
            "is_click": 0.00017303287815711998,
            "cart": 0.00017977199437101826,
            "fav": 0.00010945271548214869,
            "buy": 0.00017907192716967637
          },
          "delta_macro_logloss": 0.00016033237879498757,
          "parameter_ratio": 0.6340496785912583
        },
        "optional_single_bootstrap": null,
        "forward_mean_ms": 14.351776790618896,
        "forward_ratio_to_original": 0.997499981042837,
        "near_reference_policy_passed": false,
        "conditioning_evidence": "static_control_not_a_conditioning_claim"
      },
      "mix_q4_query_joint": {
        "decision": "discard",
        "retained_by_queue_policy": false,
        "supersedes_training_episode": "mix_q4_query_joint-s20262027-20260909T035110579796Z",
        "macro_auc": 0.8282888628522254,
        "parameter_count": 51742676,
        "per_task": {
          "is_click": {
            "logloss": 0.1818974011556235,
            "AUC": 0.6671756738576081
          },
          "cart": {
            "logloss": 0.017773827164852922,
            "AUC": 0.865483602204592
          },
          "fav": {
            "logloss": 0.013359253409815778,
            "AUC": 0.8837114132519582
          },
          "buy": {
            "logloss": 0.006730089127050176,
            "AUC": 0.896784762094743
          }
        },
        "point_task_tradeoffs": {
          "cart": {
            "auc_delta": -0.0012313756430578637,
            "logloss_delta": 1.2144101971471588e-05
          },
          "buy": {
            "auc_delta": -0.0017425635177172882,
            "logloss_delta": 1.3434935500990577e-05
          }
        },
        "versus_original": {
          "reference": "ctx_q4",
          "candidate": "mix_q4_query_joint",
          "reference_role": "original_recurrence",
          "delta_macro_auc": -0.00018370257401822965,
          "delta_auc": {
            "is_click": 0.0012016268340766523,
            "cart": -0.0012313756430578637,
            "fav": 0.001037502030625248,
            "buy": -0.0017425635177172882
          },
          "delta_logloss": {
            "is_click": -0.0004162441074775647,
            "cart": 1.2144101971471588e-05,
            "fav": -6.7643838058684125e-06,
            "buy": 1.3434935500990577e-05
          },
          "delta_macro_logloss": -9.935736345274143e-05,
          "parameter_ratio": 1.0003566607122683
        },
        "original_bootstrap": {
          "reference": "ctx_q4",
          "candidate": "mix_q4_query_joint",
          "users": 268201,
          "repetitions": 500,
          "bootstrap_seed": 20260906,
          "training_updates": 0,
          "new_training_seeds": [],
          "reference_role": "original_recurrence",
          "mandatory": true,
          "mandatory_joint_constant_contrast": false,
          "mandatory_original_recurrence_contrast": true,
          "delta_auc": {
            "is_click": 0.0012016268340766523,
            "cart": -0.0012313756430578637,
            "fav": 0.001037502030625248,
            "buy": -0.0017425635177172882
          },
          "ci95_auc": {
            "is_click": [
              0.0006715907249102065,
              0.0017842724149109783
            ],
            "cart": [
              -0.002744995310557466,
              0.00033933726732431605
            ],
            "fav": [
              -0.0005161908320194086,
              0.0024694815205208915
            ],
            "buy": [
              -0.003806432781218372,
              0.00020590958116525608
            ]
          },
          "delta_macro_auc": -0.00018370257401822965,
          "ci95_macro_auc": [
            -0.0009206318506348954,
            0.0005333416620977589
          ],
          "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
        },
        "original_evidence": "inconclusive_crosses_zero",
        "versus_best_single": {
          "reference": "psgd_precond_every_step",
          "candidate": "mix_q4_query_joint",
          "reference_role": "best_single_pass",
          "delta_macro_auc": -0.0003346696674493277,
          "delta_auc": {
            "is_click": -0.0006141045338338147,
            "cart": -0.0015354244364732539,
            "fav": 0.0013501843645713496,
            "buy": -0.0005393340640619249
          },
          "delta_logloss": {
            "is_click": -0.00010658372290675566,
            "cart": 0.00013879596093586272,
            "fav": 8.893141247773469e-05,
            "buy": 0.00010820935462399397
          },
          "delta_macro_logloss": 5.7338251282706976e-05,
          "parameter_ratio": 0.6339745955723269
        },
        "optional_single_bootstrap": null,
        "forward_mean_ms": 14.45967357158661,
        "forward_ratio_to_original": 1.0049991944531396,
        "near_reference_policy_passed": false,
        "conditioning_evidence": {
          "point_case": "joint_better_than_constant_but_not_above_original",
          "comparison": {
            "reference": "mix_q4_query_constant",
            "candidate": "mix_q4_query_joint",
            "reference_role": "conditioning_control",
            "delta_macro_auc": 0.0027046564104633575,
            "delta_auc": {
              "is_click": 0.004026301924688225,
              "cart": 0.00033814925358510717,
              "fav": 0.0037764315173665075,
              "buy": 0.0026777429462135904
            },
            "delta_logloss": {
              "is_click": -0.0007442234062083053,
              "cart": 5.574872390758101e-06,
              "fav": -4.072181049226409e-05,
              "buy": -4.3329811548059594e-05
            },
            "delta_macro_logloss": -0.00020567503896445904,
            "parameter_ratio": 1.0
          },
          "bootstrap": {
            "reference": "mix_q4_query_constant",
            "candidate": "mix_q4_query_joint",
            "users": 268201,
            "repetitions": 500,
            "bootstrap_seed": 20260906,
            "training_updates": 0,
            "new_training_seeds": [],
            "reference_role": "conditioning_control",
            "mandatory": true,
            "mandatory_joint_constant_contrast": true,
            "mandatory_original_recurrence_contrast": false,
            "delta_auc": {
              "is_click": 0.004026301924688225,
              "cart": 0.00033814925358510717,
              "fav": 0.0037764315173665075,
              "buy": 0.0026777429462135904
            },
            "ci95_auc": {
              "is_click": [
                0.0033928000699526588,
                0.00464941256216283
              ],
              "cart": [
                -0.0012317892293798455,
                0.002091886799480741
              ],
              "fav": [
                0.002033056015334997,
                0.005640834004541323
              ],
              "buy": [
                0.0005798320885039538,
                0.004631683856751228
              ]
            },
            "delta_macro_auc": 0.0027046564104633575,
            "ci95_macro_auc": [
              0.0018877894099423621,
              0.0034858632233484375
            ],
            "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
          },
          "joint_constant_evidence": "positive_conditional_interval",
          "original_evidence": "inconclusive_crosses_zero",
          "both_conditional_intervals_positive": false,
          "scope": "Input-conditioned versus static parameterization under this fixed seed/budget; not isolated retrieval or training-seed robustness"
        }
      },
      "mix_q4_query_constant": {
        "decision": "discard",
        "retained_by_queue_policy": false,
        "supersedes_training_episode": "mix_q4_query_constant-s20262027-20260909T035110802164Z",
        "macro_auc": 0.825584206441762,
        "parameter_count": 51742676,
        "per_task": {
          "is_click": {
            "logloss": 0.1826416245618318,
            "AUC": 0.6631493719329199
          },
          "cart": {
            "logloss": 0.017768252292462164,
            "AUC": 0.8651454529510069
          },
          "fav": {
            "logloss": 0.013399975220308042,
            "AUC": 0.8799349817345917
          },
          "buy": {
            "logloss": 0.006773418938598235,
            "AUC": 0.8941070191485294
          }
        },
        "point_task_tradeoffs": {
          "is_click": {
            "auc_delta": -0.0028246750906115725,
            "logloss_delta": 0.0003279792987307406
          },
          "cart": {
            "auc_delta": -0.001569524896642971,
            "logloss_delta": 6.569229580713487e-06
          },
          "fav": {
            "auc_delta": -0.0027389294867412595,
            "logloss_delta": 3.3957426686395675e-05
          },
          "buy": {
            "auc_delta": -0.004420306463930879,
            "logloss_delta": 5.676474704905017e-05
          }
        },
        "versus_original": {
          "reference": "ctx_q4",
          "candidate": "mix_q4_query_constant",
          "reference_role": "original_recurrence",
          "delta_macro_auc": -0.002888358984481587,
          "delta_auc": {
            "is_click": -0.0028246750906115725,
            "cart": -0.001569524896642971,
            "fav": -0.0027389294867412595,
            "buy": -0.004420306463930879
          },
          "delta_logloss": {
            "is_click": 0.0003279792987307406,
            "cart": 6.569229580713487e-06,
            "fav": 3.3957426686395675e-05,
            "buy": 5.676474704905017e-05
          },
          "delta_macro_logloss": 0.00010631767551171761,
          "parameter_ratio": 1.0003566607122683
        },
        "original_bootstrap": {
          "reference": "ctx_q4",
          "candidate": "mix_q4_query_constant",
          "users": 268201,
          "repetitions": 500,
          "bootstrap_seed": 20260906,
          "training_updates": 0,
          "new_training_seeds": [],
          "reference_role": "original_recurrence",
          "mandatory": true,
          "mandatory_joint_constant_contrast": false,
          "mandatory_original_recurrence_contrast": true,
          "delta_auc": {
            "is_click": -0.0028246750906115725,
            "cart": -0.001569524896642971,
            "fav": -0.0027389294867412595,
            "buy": -0.004420306463930879
          },
          "ci95_auc": {
            "is_click": [
              -0.0034464809343528803,
              -0.0022335174474880544
            ],
            "cart": [
              -0.003158353599223049,
              4.170809789978412e-05
            ],
            "fav": [
              -0.004562041673711182,
              -0.0012425652951645967
            ],
            "buy": [
              -0.006733434564550341,
              -0.002332306765200038
            ]
          },
          "delta_macro_auc": -0.002888358984481587,
          "ci95_macro_auc": [
            -0.0037251126323888406,
            -0.0020567260238820934
          ],
          "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
        },
        "original_evidence": "negative_conditional_interval",
        "versus_best_single": {
          "reference": "psgd_precond_every_step",
          "candidate": "mix_q4_query_constant",
          "reference_role": "best_single_pass",
          "delta_macro_auc": -0.003039326077912685,
          "delta_auc": {
            "is_click": -0.0046404064585220395,
            "cart": -0.001873573690058361,
            "fav": -0.002426247152795158,
            "buy": -0.0032170770102755153
          },
          "delta_logloss": {
            "is_click": 0.0006376396833015496,
            "cart": 0.00013322108854510462,
            "fav": 0.00012965322296999877,
            "buy": 0.00015153916617205356
          },
          "delta_macro_logloss": 0.000263013290247166,
          "parameter_ratio": 0.6339745955723269
        },
        "optional_single_bootstrap": null,
        "forward_mean_ms": 14.328207206726074,
        "forward_ratio_to_original": 0.9958618103947506,
        "near_reference_policy_passed": false,
        "conditioning_evidence": "static_control_not_a_conditioning_claim"
      },
      "mix_q8_channel_joint": {
        "decision": "discard",
        "retained_by_queue_policy": false,
        "supersedes_training_episode": "mix_q8_channel_joint-s20262027-20260909T035053715005Z",
        "macro_auc": 0.8235252673993565,
        "parameter_count": 81640900,
        "per_task": {
          "is_click": {
            "logloss": 0.18272091678328747,
            "AUC": 0.6622034202233482
          },
          "cart": {
            "logloss": 0.01791688324904791,
            "AUC": 0.8608140654299984
          },
          "fav": {
            "logloss": 0.01345851636820282,
            "AUC": 0.8782044405252997
          },
          "buy": {
            "logloss": 0.006847528837483302,
            "AUC": 0.8928791434187797
          }
        },
        "point_task_tradeoffs": {
          "is_click": {
            "auc_delta": -0.004666465801792641,
            "logloss_delta": 0.0006768189396294255
          },
          "cart": {
            "auc_delta": -0.006682770277031458,
            "logloss_delta": 0.00028893116846214276
          },
          "fav": {
            "auc_delta": -0.0038061049515943912,
            "logloss_delta": 0.0001820443870290487
          },
          "buy": {
            "auc_delta": -0.004828442869057148,
            "logloss_delta": 0.0002434760786514693
          }
        },
        "versus_original": {
          "reference": "explore_dual_no_feedback",
          "candidate": "mix_q8_channel_joint",
          "reference_role": "original_recurrence",
          "delta_macro_auc": -0.004995945974868965,
          "delta_auc": {
            "is_click": -0.004666465801792641,
            "cart": -0.006682770277031458,
            "fav": -0.0038061049515943912,
            "buy": -0.004828442869057148
          },
          "delta_logloss": {
            "is_click": 0.0006768189396294255,
            "cart": 0.00028893116846214276,
            "fav": 0.0001820443870290487,
            "buy": 0.0002434760786514693
          },
          "delta_macro_logloss": 0.0003478176434430283,
          "parameter_ratio": 1.0003011162325812
        },
        "original_bootstrap": {
          "reference": "explore_dual_no_feedback",
          "candidate": "mix_q8_channel_joint",
          "users": 268201,
          "repetitions": 500,
          "bootstrap_seed": 20260906,
          "training_updates": 0,
          "new_training_seeds": [],
          "reference_role": "original_recurrence",
          "mandatory": true,
          "mandatory_joint_constant_contrast": false,
          "mandatory_original_recurrence_contrast": true,
          "delta_auc": {
            "is_click": -0.004666465801792641,
            "cart": -0.006682770277031458,
            "fav": -0.0038061049515943912,
            "buy": -0.004828442869057148
          },
          "ci95_auc": {
            "is_click": [
              -0.005303926320941235,
              -0.004020554237367991
            ],
            "cart": [
              -0.008424294821025326,
              -0.004947287797701411
            ],
            "fav": [
              -0.005325637926526164,
              -0.0021637805524471645
            ],
            "buy": [
              -0.007317971514083537,
              -0.002309674782114167
            ]
          },
          "delta_macro_auc": -0.004995945974868965,
          "ci95_macro_auc": [
            -0.005948306359540133,
            -0.004113655213096361
          ],
          "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
        },
        "original_evidence": "negative_conditional_interval",
        "versus_best_single": {
          "reference": "psgd_precond_every_step",
          "candidate": "mix_q8_channel_joint",
          "reference_role": "best_single_pass",
          "delta_macro_auc": -0.005098265120318191,
          "delta_auc": {
            "is_click": -0.005586358168093719,
            "cart": -0.0062049612110668395,
            "fav": -0.004156788362087216,
            "buy": -0.00444495274002521
          },
          "delta_logloss": {
            "is_click": 0.0007169319047572176,
            "cart": 0.00028185204513084955,
            "fav": 0.00018819437086477556,
            "buy": 0.00022564906505711992
          },
          "delta_macro_logloss": 0.00035315684645249174,
          "parameter_ratio": 1.0003011162325812
        },
        "optional_single_bootstrap": null,
        "forward_mean_ms": 16.10703749656677,
        "forward_ratio_to_original": 1.020528206073035,
        "near_reference_policy_passed": false,
        "conditioning_evidence": {
          "point_case": "no_positive_joint_conditioning_point_advantage",
          "comparison": {
            "reference": "mix_q8_channel_constant",
            "candidate": "mix_q8_channel_joint",
            "reference_role": "conditioning_control",
            "delta_macro_auc": -0.0025399939113160963,
            "delta_auc": {
              "is_click": -0.003176234324373972,
              "cart": -0.003630568236864251,
              "fav": -0.001617365217203015,
              "buy": -0.001735807866823147
            },
            "delta_logloss": {
              "is_click": 0.00024175437788784282,
              "cart": 0.00024508298709384826,
              "fav": 0.00019735666414383637,
              "buy": 0.00018102639325117374
            },
            "delta_macro_logloss": 0.00021630510559418614,
            "parameter_ratio": 1.0
          },
          "bootstrap": {
            "reference": "mix_q8_channel_constant",
            "candidate": "mix_q8_channel_joint",
            "users": 268201,
            "repetitions": 500,
            "bootstrap_seed": 20260906,
            "training_updates": 0,
            "new_training_seeds": [],
            "reference_role": "conditioning_control",
            "mandatory": true,
            "mandatory_joint_constant_contrast": true,
            "mandatory_original_recurrence_contrast": false,
            "delta_auc": {
              "is_click": -0.003176234324373972,
              "cart": -0.003630568236864251,
              "fav": -0.001617365217203015,
              "buy": -0.001735807866823147
            },
            "ci95_auc": {
              "is_click": [
                -0.003709854482947536,
                -0.002597738623642038
              ],
              "cart": [
                -0.005280279111062139,
                -0.002204917418873953
              ],
              "fav": [
                -0.0032190323571760714,
                9.241166552236895e-06
              ],
              "buy": [
                -0.004123537443484607,
                0.0008153443603363585
              ]
            },
            "delta_macro_auc": -0.0025399939113160963,
            "ci95_macro_auc": [
              -0.00339650489045982,
              -0.0017128938910583942
            ],
            "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
          },
          "joint_constant_evidence": "negative_conditional_interval",
          "original_evidence": "negative_conditional_interval",
          "both_conditional_intervals_positive": false,
          "scope": "Input-conditioned versus static parameterization under this fixed seed/budget; not isolated retrieval or training-seed robustness"
        }
      },
      "mix_q8_channel_constant": {
        "decision": "discard",
        "retained_by_queue_policy": false,
        "supersedes_training_episode": "mix_q8_channel_constant-s20262027-20260909T035059340196Z",
        "macro_auc": 0.8260652613106726,
        "parameter_count": 81640900,
        "per_task": {
          "is_click": {
            "logloss": 0.18247916240539963,
            "AUC": 0.6653796545477222
          },
          "cart": {
            "logloss": 0.01767180026195406,
            "AUC": 0.8644446336668626
          },
          "fav": {
            "logloss": 0.013261159704058983,
            "AUC": 0.8798218057425027
          },
          "buy": {
            "logloss": 0.006666502444232128,
            "AUC": 0.8946149512856029
          }
        },
        "point_task_tradeoffs": {
          "is_click": {
            "auc_delta": -0.0014902314774186687,
            "logloss_delta": 0.0004350645617415827
          },
          "cart": {
            "auc_delta": -0.0030522020401672068,
            "logloss_delta": 4.384818136829449e-05
          },
          "fav": {
            "auc_delta": -0.002188739734391376,
            "logloss_delta": -1.531227711478768e-05
          },
          "buy": {
            "auc_delta": -0.003092635002234001,
            "logloss_delta": 6.244968540029556e-05
          }
        },
        "versus_original": {
          "reference": "explore_dual_no_feedback",
          "candidate": "mix_q8_channel_constant",
          "reference_role": "original_recurrence",
          "delta_macro_auc": -0.0024559520635528687,
          "delta_auc": {
            "is_click": -0.0014902314774186687,
            "cart": -0.0030522020401672068,
            "fav": -0.002188739734391376,
            "buy": -0.003092635002234001
          },
          "delta_logloss": {
            "is_click": 0.0004350645617415827,
            "cart": 4.384818136829449e-05,
            "fav": -1.531227711478768e-05,
            "buy": 6.244968540029556e-05
          },
          "delta_macro_logloss": 0.00013151253784884215,
          "parameter_ratio": 1.0003011162325812
        },
        "original_bootstrap": {
          "reference": "explore_dual_no_feedback",
          "candidate": "mix_q8_channel_constant",
          "users": 268201,
          "repetitions": 500,
          "bootstrap_seed": 20260906,
          "training_updates": 0,
          "new_training_seeds": [],
          "reference_role": "original_recurrence",
          "mandatory": true,
          "mandatory_joint_constant_contrast": false,
          "mandatory_original_recurrence_contrast": true,
          "delta_auc": {
            "is_click": -0.0014902314774186687,
            "cart": -0.0030522020401672068,
            "fav": -0.002188739734391376,
            "buy": -0.003092635002234001
          },
          "ci95_auc": {
            "is_click": [
              -0.0021199715335199405,
              -0.00097348431138746
            ],
            "cart": [
              -0.0046886865729858646,
              -0.0015056325840211028
            ],
            "fav": [
              -0.0037316955363647596,
              -0.0005708480873126682
            ],
            "buy": [
              -0.0052659927506580025,
              -0.0010059651883430447
            ]
          },
          "delta_macro_auc": -0.0024559520635528687,
          "ci95_macro_auc": [
            -0.0032940635403033297,
            -0.0016133641263571176
          ],
          "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
        },
        "original_evidence": "negative_conditional_interval",
        "versus_best_single": {
          "reference": "psgd_precond_every_step",
          "candidate": "mix_q8_channel_constant",
          "reference_role": "best_single_pass",
          "delta_macro_auc": -0.0025582712090020943,
          "delta_auc": {
            "is_click": -0.002410123843719747,
            "cart": -0.0025743929742025884,
            "fav": -0.002539423144884201,
            "buy": -0.002709144873202063
          },
          "delta_logloss": {
            "is_click": 0.00047517752686937476,
            "cart": 3.6769058037001284e-05,
            "fav": -9.162293279060818e-06,
            "buy": 4.462267180594618e-05
          },
          "delta_macro_logloss": 0.0001368517408583056,
          "parameter_ratio": 1.0003011162325812
        },
        "optional_single_bootstrap": null,
        "forward_mean_ms": 15.946015238761902,
        "forward_ratio_to_original": 1.010325972674717,
        "near_reference_policy_passed": false,
        "conditioning_evidence": "static_control_not_a_conditioning_claim"
      },
      "mix_q8_query_joint": {
        "decision": "discard",
        "retained_by_queue_policy": false,
        "supersedes_training_episode": "mix_q8_query_joint-s20262027-20260909T035104953096Z",
        "macro_auc": 0.8255266220041615,
        "parameter_count": 81634772,
        "per_task": {
          "is_click": {
            "logloss": 0.1826896649205254,
            "AUC": 0.6636820818896882
          },
          "cart": {
            "logloss": 0.017753992803167504,
            "AUC": 0.8645688445103342
          },
          "fav": {
            "logloss": 0.013331503253219825,
            "AUC": 0.8786972377457489
          },
          "buy": {
            "logloss": 0.006667643838479266,
            "AUC": 0.8951583238708746
          }
        },
        "point_task_tradeoffs": {
          "is_click": {
            "auc_delta": -0.00318780413545261,
            "logloss_delta": 0.0006455670768673594
          },
          "cart": {
            "auc_delta": -0.002927991196695623,
            "logloss_delta": 0.00012604072258173837
          },
          "fav": {
            "auc_delta": -0.003313307731145132,
            "logloss_delta": 5.503127204605461e-05
          },
          "buy": {
            "auc_delta": -0.0025492624169622946,
            "logloss_delta": 6.359107964743361e-05
          }
        },
        "versus_original": {
          "reference": "explore_dual_no_feedback",
          "candidate": "mix_q8_query_joint",
          "reference_role": "original_recurrence",
          "delta_macro_auc": -0.0029945913700639704,
          "delta_auc": {
            "is_click": -0.00318780413545261,
            "cart": -0.002927991196695623,
            "fav": -0.003313307731145132,
            "buy": -0.0025492624169622946
          },
          "delta_logloss": {
            "is_click": 0.0006455670768673594,
            "cart": 0.00012604072258173837,
            "fav": 5.503127204605461e-05,
            "buy": 6.359107964743361e-05
          },
          "delta_macro_logloss": 0.00022255753778564757,
          "parameter_ratio": 1.0002260332136497
        },
        "original_bootstrap": {
          "reference": "explore_dual_no_feedback",
          "candidate": "mix_q8_query_joint",
          "users": 268201,
          "repetitions": 500,
          "bootstrap_seed": 20260906,
          "training_updates": 0,
          "new_training_seeds": [],
          "reference_role": "original_recurrence",
          "mandatory": true,
          "mandatory_joint_constant_contrast": false,
          "mandatory_original_recurrence_contrast": true,
          "delta_auc": {
            "is_click": -0.00318780413545261,
            "cart": -0.002927991196695623,
            "fav": -0.003313307731145132,
            "buy": -0.0025492624169622946
          },
          "ci95_auc": {
            "is_click": [
              -0.00377688877041345,
              -0.0026360426485845497
            ],
            "cart": [
              -0.004382965118201504,
              -0.0011609163136749258
            ],
            "fav": [
              -0.005025863481237708,
              -0.0016432432616736634
            ],
            "buy": [
              -0.0045612975915463125,
              -0.0002295001348551552
            ]
          },
          "delta_macro_auc": -0.0029945913700639704,
          "ci95_macro_auc": [
            -0.003824687661344195,
            -0.0021077088514912207
          ],
          "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
        },
        "original_evidence": "negative_conditional_interval",
        "versus_best_single": {
          "reference": "psgd_precond_every_step",
          "candidate": "mix_q8_query_joint",
          "reference_role": "best_single_pass",
          "delta_macro_auc": -0.003096910515513196,
          "delta_auc": {
            "is_click": -0.004107696501753688,
            "cart": -0.0024501821307310045,
            "fav": -0.0036639911416379567,
            "buy": -0.0021657722879303565
          },
          "delta_logloss": {
            "is_click": 0.0006856800419951514,
            "cart": 0.00011896159925044517,
            "fav": 6.118125588178147e-05,
            "buy": 4.576406605308423e-05
          },
          "delta_macro_logloss": 0.00022789674079511102,
          "parameter_ratio": 1.0002260332136497
        },
        "optional_single_bootstrap": null,
        "forward_mean_ms": 16.05049443244934,
        "forward_ratio_to_original": 1.0169456855876922,
        "near_reference_policy_passed": false,
        "conditioning_evidence": {
          "point_case": "no_positive_joint_conditioning_point_advantage",
          "comparison": {
            "reference": "mix_q8_query_constant",
            "candidate": "mix_q8_query_joint",
            "reference_role": "conditioning_control",
            "delta_macro_auc": -0.0015739464364210587,
            "delta_auc": {
              "is_click": -0.001051994572609516,
              "cart": -0.0013688491357850552,
              "fav": -0.0019518679182036758,
              "buy": -0.001923074119085988
            },
            "delta_logloss": {
              "is_click": 0.0002646316550688754,
              "cart": 0.00011754365246798015,
              "fav": 4.604147602435722e-05,
              "buy": 8.756649582195887e-05
            },
            "delta_macro_logloss": 0.00012894581984578685,
            "parameter_ratio": 1.0
          },
          "bootstrap": {
            "reference": "mix_q8_query_constant",
            "candidate": "mix_q8_query_joint",
            "users": 268201,
            "repetitions": 500,
            "bootstrap_seed": 20260906,
            "training_updates": 0,
            "new_training_seeds": [],
            "reference_role": "conditioning_control",
            "mandatory": true,
            "mandatory_joint_constant_contrast": true,
            "mandatory_original_recurrence_contrast": false,
            "delta_auc": {
              "is_click": -0.001051994572609516,
              "cart": -0.0013688491357850552,
              "fav": -0.0019518679182036758,
              "buy": -0.001923074119085988
            },
            "ci95_auc": {
              "is_click": [
                -0.0016327013381994908,
                -0.0004522785970629773
              ],
              "cart": [
                -0.002993602416902938,
                0.00022663232321096734
              ],
              "fav": [
                -0.0034549692956826,
                -0.0005346780531825391
              ],
              "buy": [
                -0.0040314029894937455,
                0.00020800352740136724
              ]
            },
            "delta_macro_auc": -0.0015739464364210587,
            "ci95_macro_auc": [
              -0.002349156794962986,
              -0.0007747138269309635
            ],
            "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
          },
          "joint_constant_evidence": "negative_conditional_interval",
          "original_evidence": "negative_conditional_interval",
          "both_conditional_intervals_positive": false,
          "scope": "Input-conditioned versus static parameterization under this fixed seed/budget; not isolated retrieval or training-seed robustness"
        }
      },
      "mix_q8_query_constant": {
        "decision": "discard",
        "retained_by_queue_policy": false,
        "supersedes_training_episode": "mix_q8_query_constant-s20262027-20260909T035108993695Z",
        "macro_auc": 0.8271005684405826,
        "parameter_count": 81634772,
        "per_task": {
          "is_click": {
            "logloss": 0.18242503326545653,
            "AUC": 0.6647340764622978
          },
          "cart": {
            "logloss": 0.017636449150699524,
            "AUC": 0.8659376936461193
          },
          "fav": {
            "logloss": 0.013285461777195468,
            "AUC": 0.8806491056639526
          },
          "buy": {
            "logloss": 0.006580077342657307,
            "AUC": 0.8970813979899606
          }
        },
        "point_task_tradeoffs": {
          "is_click": {
            "auc_delta": -0.002135809562843094,
            "logloss_delta": 0.00038093542179848394
          },
          "cart": {
            "auc_delta": -0.0015591420609105677,
            "logloss_delta": 8.49707011375822e-06
          },
          "fav": {
            "auc_delta": -0.0013614398129414562,
            "logloss_delta": 8.989796021697388e-06
          },
          "buy": {
            "auc_delta": -0.0006261882978763067,
            "logloss_delta": -2.397541617452526e-05
          }
        },
        "versus_original": {
          "reference": "explore_dual_no_feedback",
          "candidate": "mix_q8_query_constant",
          "reference_role": "original_recurrence",
          "delta_macro_auc": -0.0014206449336429117,
          "delta_auc": {
            "is_click": -0.002135809562843094,
            "cart": -0.0015591420609105677,
            "fav": -0.0013614398129414562,
            "buy": -0.0006261882978763067
          },
          "delta_logloss": {
            "is_click": 0.00038093542179848394,
            "cart": 8.49707011375822e-06,
            "fav": 8.989796021697388e-06,
            "buy": -2.397541617452526e-05
          },
          "delta_macro_logloss": 9.361171793986073e-05,
          "parameter_ratio": 1.0002260332136497
        },
        "original_bootstrap": {
          "reference": "explore_dual_no_feedback",
          "candidate": "mix_q8_query_constant",
          "users": 268201,
          "repetitions": 500,
          "bootstrap_seed": 20260906,
          "training_updates": 0,
          "new_training_seeds": [],
          "reference_role": "original_recurrence",
          "mandatory": true,
          "mandatory_joint_constant_contrast": false,
          "mandatory_original_recurrence_contrast": true,
          "delta_auc": {
            "is_click": -0.002135809562843094,
            "cart": -0.0015591420609105677,
            "fav": -0.0013614398129414562,
            "buy": -0.0006261882978763067
          },
          "ci95_auc": {
            "is_click": [
              -0.002732984634094346,
              -0.0015873871743291219
            ],
            "cart": [
              -0.002929317925476585,
              -0.0001227159747359377
            ],
            "fav": [
              -0.0029469159526564954,
              0.00013214074761747896
            ],
            "buy": [
              -0.002721334952174195,
              0.0014771292637797944
            ]
          },
          "delta_macro_auc": -0.0014206449336429117,
          "ci95_macro_auc": [
            -0.0021973074694157215,
            -0.0006391588014134531
          ],
          "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
        },
        "original_evidence": "negative_conditional_interval",
        "versus_best_single": {
          "reference": "psgd_precond_every_step",
          "candidate": "mix_q8_query_constant",
          "reference_role": "best_single_pass",
          "delta_macro_auc": -0.0015229640790921373,
          "delta_auc": {
            "is_click": -0.0030557019291441723,
            "cart": -0.0010813329949459494,
            "fav": -0.001712123223434281,
            "buy": -0.00024269816884436857
          },
          "delta_logloss": {
            "is_click": 0.000421048386926276,
            "cart": 1.4179467824650127e-06,
            "fav": 1.5139779857424249e-05,
            "buy": -4.180242976887464e-05
          },
          "delta_macro_logloss": 9.895092094932417e-05,
          "parameter_ratio": 1.0002260332136497
        },
        "optional_single_bootstrap": null,
        "forward_mean_ms": 15.948627185821532,
        "forward_ratio_to_original": 1.0104914634204665,
        "near_reference_policy_passed": false,
        "conditioning_evidence": "static_control_not_a_conditioning_claim"
      }
    },
    "conditioning_comparisons": {
      "mix_q4_channel_joint": {
        "point_case": "joint_better_than_constant_but_not_above_original",
        "comparison": {
          "reference": "mix_q4_channel_constant",
          "candidate": "mix_q4_channel_joint",
          "reference_role": "conditioning_control",
          "delta_macro_auc": 0.0005467694895233688,
          "delta_auc": {
            "is_click": 0.0009396760089108724,
            "cart": -0.00030779292300375793,
            "fav": 0.0003297028138634639,
            "buy": 0.0012254920583231188
          },
          "delta_logloss": {
            "is_click": -0.00019831593619590304,
            "cart": -2.8927485681600612e-05,
            "fav": -3.953922521408686e-05,
            "buy": -3.7646928037400954e-05
          },
          "delta_macro_logloss": -7.610739378224873e-05,
          "parameter_ratio": 1.0
        },
        "bootstrap": {
          "reference": "mix_q4_channel_constant",
          "candidate": "mix_q4_channel_joint",
          "users": 268201,
          "repetitions": 500,
          "bootstrap_seed": 20260906,
          "training_updates": 0,
          "new_training_seeds": [],
          "reference_role": "conditioning_control",
          "mandatory": true,
          "mandatory_joint_constant_contrast": true,
          "mandatory_original_recurrence_contrast": false,
          "delta_auc": {
            "is_click": 0.0009396760089108724,
            "cart": -0.00030779292300375793,
            "fav": 0.0003297028138634639,
            "buy": 0.0012254920583231188
          },
          "ci95_auc": {
            "is_click": [
              0.0003943137650293805,
              0.0014970511368180475
            ],
            "cart": [
              -0.0018891050093504669,
              0.0012896822093320906
            ],
            "fav": [
              -0.0011219161088202034,
              0.0016688241167196354
            ],
            "buy": [
              -0.0011389509503691025,
              0.0036465499866002904
            ]
          },
          "delta_macro_auc": 0.0005467694895233688,
          "ci95_macro_auc": [
            -0.00034580346755074193,
            0.0014149764367677583
          ],
          "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
        },
        "joint_constant_evidence": "inconclusive_crosses_zero",
        "original_evidence": "inconclusive_crosses_zero",
        "both_conditional_intervals_positive": false,
        "scope": "Input-conditioned versus static parameterization under this fixed seed/budget; not isolated retrieval or training-seed robustness"
      },
      "mix_q4_query_joint": {
        "point_case": "joint_better_than_constant_but_not_above_original",
        "comparison": {
          "reference": "mix_q4_query_constant",
          "candidate": "mix_q4_query_joint",
          "reference_role": "conditioning_control",
          "delta_macro_auc": 0.0027046564104633575,
          "delta_auc": {
            "is_click": 0.004026301924688225,
            "cart": 0.00033814925358510717,
            "fav": 0.0037764315173665075,
            "buy": 0.0026777429462135904
          },
          "delta_logloss": {
            "is_click": -0.0007442234062083053,
            "cart": 5.574872390758101e-06,
            "fav": -4.072181049226409e-05,
            "buy": -4.3329811548059594e-05
          },
          "delta_macro_logloss": -0.00020567503896445904,
          "parameter_ratio": 1.0
        },
        "bootstrap": {
          "reference": "mix_q4_query_constant",
          "candidate": "mix_q4_query_joint",
          "users": 268201,
          "repetitions": 500,
          "bootstrap_seed": 20260906,
          "training_updates": 0,
          "new_training_seeds": [],
          "reference_role": "conditioning_control",
          "mandatory": true,
          "mandatory_joint_constant_contrast": true,
          "mandatory_original_recurrence_contrast": false,
          "delta_auc": {
            "is_click": 0.004026301924688225,
            "cart": 0.00033814925358510717,
            "fav": 0.0037764315173665075,
            "buy": 0.0026777429462135904
          },
          "ci95_auc": {
            "is_click": [
              0.0033928000699526588,
              0.00464941256216283
            ],
            "cart": [
              -0.0012317892293798455,
              0.002091886799480741
            ],
            "fav": [
              0.002033056015334997,
              0.005640834004541323
            ],
            "buy": [
              0.0005798320885039538,
              0.004631683856751228
            ]
          },
          "delta_macro_auc": 0.0027046564104633575,
          "ci95_macro_auc": [
            0.0018877894099423621,
            0.0034858632233484375
          ],
          "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
        },
        "joint_constant_evidence": "positive_conditional_interval",
        "original_evidence": "inconclusive_crosses_zero",
        "both_conditional_intervals_positive": false,
        "scope": "Input-conditioned versus static parameterization under this fixed seed/budget; not isolated retrieval or training-seed robustness"
      },
      "mix_q8_channel_joint": {
        "point_case": "no_positive_joint_conditioning_point_advantage",
        "comparison": {
          "reference": "mix_q8_channel_constant",
          "candidate": "mix_q8_channel_joint",
          "reference_role": "conditioning_control",
          "delta_macro_auc": -0.0025399939113160963,
          "delta_auc": {
            "is_click": -0.003176234324373972,
            "cart": -0.003630568236864251,
            "fav": -0.001617365217203015,
            "buy": -0.001735807866823147
          },
          "delta_logloss": {
            "is_click": 0.00024175437788784282,
            "cart": 0.00024508298709384826,
            "fav": 0.00019735666414383637,
            "buy": 0.00018102639325117374
          },
          "delta_macro_logloss": 0.00021630510559418614,
          "parameter_ratio": 1.0
        },
        "bootstrap": {
          "reference": "mix_q8_channel_constant",
          "candidate": "mix_q8_channel_joint",
          "users": 268201,
          "repetitions": 500,
          "bootstrap_seed": 20260906,
          "training_updates": 0,
          "new_training_seeds": [],
          "reference_role": "conditioning_control",
          "mandatory": true,
          "mandatory_joint_constant_contrast": true,
          "mandatory_original_recurrence_contrast": false,
          "delta_auc": {
            "is_click": -0.003176234324373972,
            "cart": -0.003630568236864251,
            "fav": -0.001617365217203015,
            "buy": -0.001735807866823147
          },
          "ci95_auc": {
            "is_click": [
              -0.003709854482947536,
              -0.002597738623642038
            ],
            "cart": [
              -0.005280279111062139,
              -0.002204917418873953
            ],
            "fav": [
              -0.0032190323571760714,
              9.241166552236895e-06
            ],
            "buy": [
              -0.004123537443484607,
              0.0008153443603363585
            ]
          },
          "delta_macro_auc": -0.0025399939113160963,
          "ci95_macro_auc": [
            -0.00339650489045982,
            -0.0017128938910583942
          ],
          "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
        },
        "joint_constant_evidence": "negative_conditional_interval",
        "original_evidence": "negative_conditional_interval",
        "both_conditional_intervals_positive": false,
        "scope": "Input-conditioned versus static parameterization under this fixed seed/budget; not isolated retrieval or training-seed robustness"
      },
      "mix_q8_query_joint": {
        "point_case": "no_positive_joint_conditioning_point_advantage",
        "comparison": {
          "reference": "mix_q8_query_constant",
          "candidate": "mix_q8_query_joint",
          "reference_role": "conditioning_control",
          "delta_macro_auc": -0.0015739464364210587,
          "delta_auc": {
            "is_click": -0.001051994572609516,
            "cart": -0.0013688491357850552,
            "fav": -0.0019518679182036758,
            "buy": -0.001923074119085988
          },
          "delta_logloss": {
            "is_click": 0.0002646316550688754,
            "cart": 0.00011754365246798015,
            "fav": 4.604147602435722e-05,
            "buy": 8.756649582195887e-05
          },
          "delta_macro_logloss": 0.00012894581984578685,
          "parameter_ratio": 1.0
        },
        "bootstrap": {
          "reference": "mix_q8_query_constant",
          "candidate": "mix_q8_query_joint",
          "users": 268201,
          "repetitions": 500,
          "bootstrap_seed": 20260906,
          "training_updates": 0,
          "new_training_seeds": [],
          "reference_role": "conditioning_control",
          "mandatory": true,
          "mandatory_joint_constant_contrast": true,
          "mandatory_original_recurrence_contrast": false,
          "delta_auc": {
            "is_click": -0.001051994572609516,
            "cart": -0.0013688491357850552,
            "fav": -0.0019518679182036758,
            "buy": -0.001923074119085988
          },
          "ci95_auc": {
            "is_click": [
              -0.0016327013381994908,
              -0.0004522785970629773
            ],
            "cart": [
              -0.002993602416902938,
              0.00022663232321096734
            ],
            "fav": [
              -0.0034549692956826,
              -0.0005346780531825391
            ],
            "buy": [
              -0.0040314029894937455,
              0.00020800352740136724
            ]
          },
          "delta_macro_auc": -0.0015739464364210587,
          "ci95_macro_auc": [
            -0.002349156794962986,
            -0.0007747138269309635
          ],
          "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
        },
        "joint_constant_evidence": "negative_conditional_interval",
        "original_evidence": "negative_conditional_interval",
        "both_conditional_intervals_positive": false,
        "scope": "Input-conditioned versus static parameterization under this fixed seed/budget; not isolated retrieval or training-seed robustness"
      }
    },
    "kept_new_candidates": [],
    "quality_compute_tradeoff_candidates": [],
    "all_below_original_and_outside_queue_rule": true,
    "previous_best": {
      "family": "psgd_precond_every_step",
      "macro_auc": 0.8286235325196747,
      "per_task": {
        "is_click": {
          "logloss": 0.18200398487853026,
          "AUC": 0.6677897783914419
        },
        "cart": {
          "logloss": 0.01763503120391706,
          "AUC": 0.8670190266410652
        },
        "fav": {
          "logloss": 0.013270321997338044,
          "AUC": 0.8823612288873869
        },
        "buy": {
          "logloss": 0.006621879772426182,
          "AUC": 0.8973240961588049
        }
      },
      "seed": 20262027,
      "confidence": "single_seed_selection_uncorrected"
    },
    "current_best": {
      "family": "psgd_precond_every_step",
      "macro_auc": 0.8286235325196747,
      "per_task": {
        "is_click": {
          "logloss": 0.18200398487853026,
          "AUC": 0.6677897783914419
        },
        "cart": {
          "logloss": 0.01763503120391706,
          "AUC": 0.8670190266410652
        },
        "fav": {
          "logloss": 0.013270321997338044,
          "AUC": 0.8823612288873869
        },
        "buy": {
          "logloss": 0.006621879772426182,
          "AUC": 0.8973240961588049
        }
      },
      "seed": 20262027,
      "confidence": "single_seed_selection_uncorrected"
    },
    "global_best_changed_during_interpretation": false,
    "round_exceeds_plan_incumbent": false,
    "diagnostic_integration": {
      "state": "pending_separate_zero_training_evidence_adjudication",
      "fixed_pair": [
        "mix_q4_channel_joint",
        "mix_q4_channel_constant"
      ],
      "diagnostic_queue_preserved": true,
      "note": "No diagnostic metric is consumed here; the gate diagnostic may be running or awaiting its separate evidence episode"
    },
    "retrieval_mechanism_claimed": false,
    "confidence": [
      "One training seed20262027; no additional training seeds",
      "500 paired validation-user resamples; intervals conditional on checkpoints and released validation protocol",
      "Selection and multiple comparisons uncorrected; do not infer seed stability or statistical equivalence",
      "Diagnostic subset bootstrap conditions on frozen subset and ignores stratified sampling-design uncertainty",
      "Channel and query gates have different parameter counts; q4 and q8 interpreted separately",
      "Forward latency is one repeated real batch on matched GPU, not end-to-end serving latency; tiny mean differences are not reliable speedup claims"
    ],
    "near_reference_rule": {
      "status": "Optimization queue policy; not an equivalence or noninferiority margin",
      "macro_auc_delta_to_same_q_original_minimum": -0.0001,
      "forward_mean_ratio_to_same_q_original_maximum": 1.05,
      "source": "workspace/qformer-loop/taobao-transfer/recurrence-query-routes/reports/analysis-plan.json",
      "requirements": [
        "Complete verified 2193-step training, all mandatory paired comparisons, and all 11 matched latency measurements",
        "Show all four task AUC/logloss and parameter/compute differences",
        "Any task harm remains an explicit tradeoff; never promote solely because it falls inside this queue threshold"
      ],
      "threshold_fixed_before_results": true
    },
    "input_sha256": {
      "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/reports/interpretation-plan.json": "b4b6b540abe70b204c6304d2912eee92f573cf185f4e331465f6e1a00f2d7481",
      "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/reports/interpretation-plan-identity.json": "76804a6c683aca5109cef3b44d4716ced4f94ddeb032a96dd2a0a1a7c4451513",
      "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/reports/finish-identity.json": "347acf0e887e66f81eb6e42a38b3a289e0ef29c10ab398ef9fab34ccacad0dd5",
      "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/reports/supplemental-identity.json": "24f74666e2649c9c1b389c491477d65ed43f5025c1d66d9e8051e625cdf8e986",
      "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/reports/consolidation-status.json": "67a9640830cdde2441dd85ab14aeffab54cb1e0e47450ff81e693603739767f6",
      "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/reports/consolidation-plan.json": "3c411d47385cf9d072f0200a498c2635bc59f83c27f0f739c1fdd26ac2e6c83c",
      "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/verify_consolidation_inputs.py": "f8cf83b68bb861c357fd1b4a78bdad431401ab4b8b24552d902bb1b003c04d9f",
      "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/test_verify_consolidation_inputs.py": "d28c3c81d397b5fa59648262d5504d80591b5342e3290ebbf55464821fa83ff8",
      "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/finish_interpretation.py": "5ce023190b426cdf552c1406b338376cc2fad8f1eb0bb2c75df1daf4fe93f601",
      "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/test_finish_interpretation.py": "25236f5b0d75c0e66f832abc744a5b2ae282f5bd616a6b0967f25d17876ef443",
      "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/PLAN.md": "bcf6f73b114fea3ecf63e31d7099bde35f74a4d771d1c1f9a1d5b3bc16890a7e",
      "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/references.json": "f78250053ea193446739c743fb026f7d856a82727b09c93dea2e860c85f90d81",
      "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/reports/frozen-identity.json": "ea793b1fb2a95d51b0c1deb04a10087bdd813844fe4d32c45b64cb0b5df9e610",
      "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/reports/postprocess-identity.json": "33ad5d886de8e6d390bca57bda2b03c5db7470178580a63426b5388f8de6dc84",
      "workspace/qformer-loop/taobao-transfer/recurrence-query-routes/reports/analysis-plan.json": "3e50c4491cfb0058d5cb8f648e5583c4eda60c50f5dbcf65592d8ef9b648081d",
      "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/analysis-plan.json": "507b304b810966db43e2776e634ad4215741950b9dba84dd8908ea3615667388",
      "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/reports/summary.json": "859243590be64ffb92c155f6a0377e97dda07e3ecb6185d1c6b2fe3b5ccf6731",
      "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/reports/quality-summary.json": "d6d77b983ba2d5790d3fe26aa89e3fd919bbfa120fd273482bbd0cd89a06a08f",
      "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/reports/final-audit.json": "2987d7a3f9050da8ebf8877fb0017dc6200f42fe60a499746a73626698d09fb0",
      "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/reports/paired-bootstrap.json": "b29562a740504f331dea5b20dd339db0af49a91c7edec4f8f80a2385a3cf6aee",
      "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/reports/paired-bootstrap-identities.json": "38fd4c4d15e6e39b9b4aaca1f3dadad4ee224d0de7f7659a689cd55cc9ef0930"
    },
    "training_updates": 0,
    "new_training_seeds": [],
    "test_evaluated": false
  },
  "decision": "discard",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/reports/interpretation.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/reports/interpretation-plan.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/INTERPRETATION.md"
  ]
}
```

</details>
