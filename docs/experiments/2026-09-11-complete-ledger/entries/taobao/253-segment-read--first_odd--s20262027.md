## 253. segment-read--first_odd--s20262027

- 假设：Equal-count first_odd stored-history support may change useful two-round retrieval conditions; no pure Q2 mediation claim.
- 判定：keep
- 原因：Macro delta +0.001237919,CI95 [-0.0013729652000838489, 0.003909334895993671],latency ratio 1.011287. Research candidate under frozen point/cost policy only; no confirmed parent improvement, click regression, no incumbent promotion.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "segment-read--first_odd--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-10T14:13:03.864962+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "first_odd",
  "code_id": "cf7b0840f7d3f28bb5451ac3cc4d9491bfcf60c5a00b32fc75f78b7e407fc0de",
  "hypothesis": "Equal-count first_odd stored-history support may change useful two-round retrieval conditions; no pure Q2 mediation claim.",
  "change": "{\"model\": \"QFormerCross8SegmentRead\", \"maturity_mode\": \"mature\", \"segment_placement\": \"first\", \"segment_selection\": \"odd\", \"work_steps\": 2, \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8SegmentRead",
    "maturity_mode": "mature",
    "segment_placement": "first",
    "segment_selection": "odd",
    "work_steps": 2,
    "num_queries": 4
  },
  "input_protocol_id": "strict-past-native-last100-action19-mature",
  "budget": {
    "formal_training_runs": 1,
    "epochs": 1,
    "seeds": [
      20262027
    ],
    "new_training_seeds": [],
    "optimizer_steps": 2193,
    "dense_optimizer": "HeavyBall.PSGDKron",
    "preconditioner_update_probability": 1.0,
    "dense_lr": 0.0001,
    "sparse_optimizer": "Adagrad",
    "sparse_lr": 0.05,
    "clip": 1,
    "batch_size": 8192,
    "precision": "BF16"
  },
  "metrics": {
    "primary": {
      "name": "macro_AUC",
      "value": 0.768332870730035,
      "delta": 0.0012379194151744821
    },
    "per_task": {
      "is_click": {
        "logloss": 0.18464359438170927,
        "AUC": 0.6490607458023349
      },
      "cart": {
        "logloss": 0.021799999943337957,
        "AUC": 0.7938518684843248
      },
      "fav": {
        "logloss": 0.016783694676536898,
        "AUC": 0.8057001081833611
      },
      "buy": {
        "logloss": 0.009644000779846729,
        "AUC": 0.8247187604501192
      }
    },
    "macro_source_logloss": 0.05821782244535771,
    "source_metric_comparison": {
      "is_click": {
        "source_AUC": 0.6490607458023349,
        "computed_AUC": 0.6490607458023349,
        "auc_difference": 0.0,
        "source_logloss": 0.18464359438170927,
        "computed_bce": 0.18464359438171415,
        "bce_difference": 4.884981308350689e-15
      },
      "cart": {
        "source_AUC": 0.7938518684843248,
        "computed_AUC": 0.7938518684843248,
        "auc_difference": 0.0,
        "source_logloss": 0.021799999943337957,
        "computed_bce": 0.0217999425717414,
        "bce_difference": -5.7371596556832216e-08
      },
      "fav": {
        "source_AUC": 0.8057001081833611,
        "computed_AUC": 0.8057001081833611,
        "auc_difference": 0.0,
        "source_logloss": 0.016783694676536898,
        "computed_bce": 0.016783699052286154,
        "bce_difference": 4.37574925599149e-09
      },
      "buy": {
        "source_AUC": 0.8247187604501192,
        "computed_AUC": 0.8247187604501192,
        "auc_difference": 0.0,
        "source_logloss": 0.009644000779846729,
        "computed_bce": 0.00964400696492858,
        "bce_difference": 6.18508185119826e-09
      }
    },
    "matched_parent_effect": {
      "coefficients": {
        "mature_r2": -1.0,
        "cross8_mature": 0.0,
        "first_prefix": 0.0,
        "first_suffix": 0.0,
        "first_even": 0.0,
        "first_odd": 1.0,
        "second_prefix": 0.0,
        "second_suffix": 0.0,
        "second_even": 0.0,
        "second_odd": 0.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": 0.0012379194151744821,
          "ci95": [
            -0.0013729652000838489,
            0.003909334895993671
          ]
        },
        "auc.is_click": {
          "estimate": -0.0008755473986545859,
          "ci95": [
            -0.001735897019821689,
            -1.898452239201001e-05
          ]
        },
        "auc.cart": {
          "estimate": 0.0014175981269810567,
          "ci95": [
            -0.002649596980623231,
            0.0051472820182798746
          ]
        },
        "auc.fav": {
          "estimate": 0.0032366178083130803,
          "ci95": [
            -0.003049520080541829,
            0.010133129168680968
          ]
        },
        "auc.buy": {
          "estimate": 0.0011730091240583773,
          "ci95": [
            -0.006949131391057253,
            0.008919267877519912
          ]
        },
        "bce.is_click": {
          "estimate": 0.0002519481139274804,
          "ci95": [
            0.00017571511306379363,
            0.0003309376349701888
          ]
        },
        "bce.cart": {
          "estimate": -2.8011797692632096e-05,
          "ci95": [
            -0.00012342658639499115,
            6.0305015870905053e-05
          ]
        },
        "bce.fav": {
          "estimate": -1.4458642066179916e-05,
          "ci95": [
            -0.00018827486007913323,
            0.0001616139228817837
          ]
        },
        "bce.buy": {
          "estimate": 2.0502820287718326e-07,
          "ci95": [
            -8.428041619821694e-05,
            7.996215822000901e-05
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": -0.0022931455256356426,
          "ci95": [
            -0.006193375343455726,
            0.0019885295121167858
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": -0.004112165206967666,
          "ci95": [
            -0.011075535282577941,
            0.0023228165249348886
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": -0.002048556522712963,
          "ci95": [
            -0.01002105739834997,
            0.00605849817765574
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": -0.0018190196813320236,
          "ci95": [
            -0.010913920253027265,
            0.005948331667839157
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": 0.0002445890029226794,
          "ci95": [
            -0.008713162130597585,
            0.008756127161364058
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": 0.002063608684254703,
          "ci95": [
            -0.008168873458302944,
            0.012578543935367743
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": 0.0002799599116201125,
          "ci95": [
            0.0001555449799536969,
            0.0004010559510699867
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": 0.0002664067559936603,
          "ci95": [
            7.946213163289246e-05,
            0.0004538441574383541
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": 0.0002517430857246032,
          "ci95": [
            0.00013438502155488133,
            0.0003677990708430517
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": -1.3553155626452179e-05,
          "ci95": [
            -0.00024088211568186279,
            0.0001801739026985624
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": -2.821682589550928e-05,
          "ci95": [
            -0.00014318161867608176,
            9.623502499510572e-05
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": -1.46636702690571e-05,
          "ci95": [
            -0.00020671279340111394,
            0.0001772972460641397
          ]
        }
      }
    },
    "all_prespecified_effects_involving_candidate": {
      "first_odd__minus__mature_r2": {
        "coefficients": {
          "mature_r2": -1.0,
          "cross8_mature": 0.0,
          "first_prefix": 0.0,
          "first_suffix": 0.0,
          "first_even": 0.0,
          "first_odd": 1.0,
          "second_prefix": 0.0,
          "second_suffix": 0.0,
          "second_even": 0.0,
          "second_odd": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.0012379194151744821,
            "ci95": [
              -0.0013729652000838489,
              0.003909334895993671
            ]
          },
          "auc.is_click": {
            "estimate": -0.0008755473986545859,
            "ci95": [
              -0.001735897019821689,
              -1.898452239201001e-05
            ]
          },
          "auc.cart": {
            "estimate": 0.0014175981269810567,
            "ci95": [
              -0.002649596980623231,
              0.0051472820182798746
            ]
          },
          "auc.fav": {
            "estimate": 0.0032366178083130803,
            "ci95": [
              -0.003049520080541829,
              0.010133129168680968
            ]
          },
          "auc.buy": {
            "estimate": 0.0011730091240583773,
            "ci95": [
              -0.006949131391057253,
              0.008919267877519912
            ]
          },
          "bce.is_click": {
            "estimate": 0.0002519481139274804,
            "ci95": [
              0.00017571511306379363,
              0.0003309376349701888
            ]
          },
          "bce.cart": {
            "estimate": -2.8011797692632096e-05,
            "ci95": [
              -0.00012342658639499115,
              6.0305015870905053e-05
            ]
          },
          "bce.fav": {
            "estimate": -1.4458642066179916e-05,
            "ci95": [
              -0.00018827486007913323,
              0.0001616139228817837
            ]
          },
          "bce.buy": {
            "estimate": 2.0502820287718326e-07,
            "ci95": [
              -8.428041619821694e-05,
              7.996215822000901e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.0022931455256356426,
            "ci95": [
              -0.006193375343455726,
              0.0019885295121167858
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.004112165206967666,
            "ci95": [
              -0.011075535282577941,
              0.0023228165249348886
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.002048556522712963,
            "ci95": [
              -0.01002105739834997,
              0.00605849817765574
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.0018190196813320236,
            "ci95": [
              -0.010913920253027265,
              0.005948331667839157
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.0002445890029226794,
            "ci95": [
              -0.008713162130597585,
              0.008756127161364058
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.002063608684254703,
            "ci95": [
              -0.008168873458302944,
              0.012578543935367743
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 0.0002799599116201125,
            "ci95": [
              0.0001555449799536969,
              0.0004010559510699867
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.0002664067559936603,
            "ci95": [
              7.946213163289246e-05,
              0.0004538441574383541
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.0002517430857246032,
            "ci95": [
              0.00013438502155488133,
              0.0003677990708430517
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -1.3553155626452179e-05,
            "ci95": [
              -0.00024088211568186279,
              0.0001801739026985624
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -2.821682589550928e-05,
            "ci95": [
              -0.00014318161867608176,
              9.623502499510572e-05
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -1.46636702690571e-05,
            "ci95": [
              -0.00020671279340111394,
              0.0001772972460641397
            ]
          }
        }
      },
      "first_odd__minus__cross8_mature": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": -1.0,
          "first_prefix": 0.0,
          "first_suffix": 0.0,
          "first_even": 0.0,
          "first_odd": 1.0,
          "second_prefix": 0.0,
          "second_suffix": 0.0,
          "second_even": 0.0,
          "second_odd": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.008926450634575145,
            "ci95": [
              0.006004513438290563,
              0.012429005603353194
            ]
          },
          "auc.is_click": {
            "estimate": -0.0008760243406308321,
            "ci95": [
              -0.0017153218175521956,
              -0.0001228945978719118
            ]
          },
          "auc.cart": {
            "estimate": 0.004645133113208799,
            "ci95": [
              1.9171847827922983e-05,
              0.009351600355131787
            ]
          },
          "auc.fav": {
            "estimate": 0.009198097155911578,
            "ci95": [
              0.004505805729228216,
              0.014864223207247517
            ]
          },
          "auc.buy": {
            "estimate": 0.022738596609811035,
            "ci95": [
              0.013000076649500735,
              0.03418797642853242
            ]
          },
          "bce.is_click": {
            "estimate": 0.0003256565331228034,
            "ci95": [
              0.00025278964465221335,
              0.00040485118726659716
            ]
          },
          "bce.cart": {
            "estimate": -3.446906533762992e-05,
            "ci95": [
              -0.00012578782071631005,
              5.102430690584797e-05
            ]
          },
          "bce.fav": {
            "estimate": -0.00018968012916689356,
            "ci95": [
              -0.0002920573430859777,
              -9.478900664959872e-05
            ]
          },
          "bce.buy": {
            "estimate": -0.00036338336350954896,
            "ci95": [
              -0.0004866657169586066,
              -0.00026045737387610343
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.005521157453839631,
            "ci95": [
              -0.010179473829300434,
              -0.0006764911876287851
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.01007412149654241,
            "ci95": [
              -0.015651741713670543,
              -0.005010716341527419
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.023614620950441867,
            "ci95": [
              -0.03552229191773355,
              -0.014007496663762393
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.0045529640427027784,
            "ci95": [
              -0.012019360018712738,
              0.0018561095145809472
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.018093463496602236,
            "ci95": [
              -0.030844526278477315,
              -0.00748806683000453
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.013540499453899457,
            "ci95": [
              -0.02557539802441226,
              -0.002267659982628601
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 0.0003601255984604333,
            "ci95": [
              0.0002542541935224579,
              0.00047958413116606745
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.000515336662289697,
            "ci95": [
              0.0003991105618741856,
              0.0006447839166821004
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.0006890398966323524,
            "ci95": [
              0.0005533628167439737,
              0.0008470814237649787
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.00015521106382926364,
            "ci95": [
              2.6914540555340178e-05,
              0.00028775679232946713
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00032891429817191904,
            "ci95": [
              0.00020145116364052215,
              0.0004905410916931556
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.0001737032343426554,
            "ci95": [
              2.4455422224746233e-05,
              0.000323648921668227
            ]
          }
        }
      },
      "first_odd__minus__second_odd": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "first_prefix": 0.0,
          "first_suffix": 0.0,
          "first_even": 0.0,
          "first_odd": 1.0,
          "second_prefix": 0.0,
          "second_suffix": 0.0,
          "second_even": 0.0,
          "second_odd": -1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.005011362607507591,
            "ci95": [
              0.0022504757905253397,
              0.007689341673881042
            ]
          },
          "auc.is_click": {
            "estimate": -0.0002136480196670565,
            "ci95": [
              -0.0010802529552935286,
              0.0005769539543299835
            ]
          },
          "auc.cart": {
            "estimate": 0.0028110687165837644,
            "ci95": [
              -0.0020431251221843996,
              0.007441123320431644
            ]
          },
          "auc.fav": {
            "estimate": 0.019410865088255846,
            "ci95": [
              0.012676362747471675,
              0.02626915312952085
            ]
          },
          "auc.buy": {
            "estimate": -0.0019628353551421895,
            "ci95": [
              -0.008114414874609847,
              0.005130645962316543
            ]
          },
          "bce.is_click": {
            "estimate": 0.00033241187326812915,
            "ci95": [
              0.0002566854879711866,
              0.0004095274945963137
            ]
          },
          "bce.cart": {
            "estimate": -6.996304109863072e-05,
            "ci95": [
              -0.0001719369828161921,
              3.208862592394134e-05
            ]
          },
          "bce.fav": {
            "estimate": -0.00035391673103677204,
            "ci95": [
              -0.0004987328752848679,
              -0.00021834400090067855
            ]
          },
          "bce.buy": {
            "estimate": -0.00012831755254026286,
            "ci95": [
              -0.0002210577761585968,
              -4.528430774963086e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.003024716736250821,
            "ci95": [
              -0.007685132240673587,
              0.0017132941403039963
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.019624513107922903,
            "ci95": [
              -0.026441606508265342,
              -0.012673672064113215
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.001749187335475133,
            "ci95": [
              -0.005268662283482312,
              0.008011775479600152
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.016599796371672082,
            "ci95": [
              -0.02511424753526213,
              -0.00865912145590286
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.004773904071725954,
            "ci95": [
              -0.00445754305074853,
              0.013400685547289442
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.021373700443398036,
            "ci95": [
              0.01201611285808123,
              0.0310424852255057
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 0.00040237491436675987,
            "ci95": [
              0.00027656321754305715,
              0.0005216841127036046
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.0006863286043049012,
            "ci95": [
              0.0005262266418380246,
              0.0008431507129856094
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.000460729425808392,
            "ci95": [
              0.00034903333374768314,
              0.0005796734938542781
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.0002839536899381413,
            "ci95": [
              0.00011997168731827752,
              0.00046961231662786846
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 5.8354511441632145e-05,
            "ci95": [
              -6.659440194903129e-05,
              0.00021021200040213691
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.00022559917849650918,
            "ci95": [
              -0.0004030798402166392,
              -6.65017928758426e-05
            ]
          }
        }
      },
      "first_prefix__minus__first_odd": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "first_prefix": 1.0,
          "first_suffix": 0.0,
          "first_even": 0.0,
          "first_odd": -1.0,
          "second_prefix": 0.0,
          "second_suffix": 0.0,
          "second_even": 0.0,
          "second_odd": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.0045307481582674,
            "ci95": [
              -0.0073558261815326546,
              -0.0018547853348243565
            ]
          },
          "auc.is_click": {
            "estimate": -0.001433920405015665,
            "ci95": [
              -0.002220212015517234,
              -0.0005531872698258362
            ]
          },
          "auc.cart": {
            "estimate": 0.005388394071429881,
            "ci95": [
              0.0011137780658515952,
              0.010532601806640849
            ]
          },
          "auc.fav": {
            "estimate": -0.014896909363185107,
            "ci95": [
              -0.02084373538098501,
              -0.009106839258884848
            ]
          },
          "auc.buy": {
            "estimate": -0.00718055693629871,
            "ci95": [
              -0.015888739566109516,
              0.0011950349481032044
            ]
          },
          "bce.is_click": {
            "estimate": -0.00045000316556687014,
            "ci95": [
              -0.0005355326605249635,
              -0.00036710342789683805
            ]
          },
          "bce.cart": {
            "estimate": -2.0896850022756458e-05,
            "ci95": [
              -0.00013071111842834164,
              7.555937196265141e-05
            ]
          },
          "bce.fav": {
            "estimate": 0.00022257562492642574,
            "ci95": [
              0.00011815754244116879,
              0.0003423874320130903
            ]
          },
          "bce.buy": {
            "estimate": 0.0001726047048248372,
            "ci95": [
              9.198752751529696e-05,
              0.00026686646009730177
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.006822314476445546,
            "ci95": [
              -0.012369589987454074,
              -0.0025266497518053764
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.013462988958169442,
            "ci95": [
              0.007622814927078903,
              0.01946734812289649
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.005746636531283045,
            "ci95": [
              -0.0023965817898023906,
              0.014680591255211907
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.020285303434614987,
            "ci95": [
              0.013616655683202967,
              0.02810479961471802
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.01256895100772859,
            "ci95": [
              0.0035588567540706054,
              0.023293390594991286
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.007716352426886397,
            "ci95": [
              -0.01794101410202834,
              0.002324073992372374
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.0004291063155441137,
            "ci95": [
              -0.0005607422930388866,
              -0.0002965851084805981
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0006725787904932959,
            "ci95": [
              -0.0008122561282801847,
              -0.0005486038739688242
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0006226078703917073,
            "ci95": [
              -0.0007543424952450147,
              -0.0005060447342366622
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.0002434724749491822,
            "ci95": [
              -0.00040034683654843734,
              -0.00011291578789069221
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -0.00019350155484759365,
            "ci95": [
              -0.0003391946966480564,
              -6.933077881379449e-05
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 4.9970920101588545e-05,
            "ci95": [
              -8.105805115687227e-05,
              0.00020297520620677687
            ]
          }
        }
      },
      "first_suffix__minus__first_odd": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "first_prefix": 0.0,
          "first_suffix": 1.0,
          "first_even": 0.0,
          "first_odd": -1.0,
          "second_prefix": 0.0,
          "second_suffix": 0.0,
          "second_even": 0.0,
          "second_odd": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.006109761747662729,
            "ci95": [
              -0.009019188728277084,
              -0.0035315702830934657
            ]
          },
          "auc.is_click": {
            "estimate": -0.0012574508124980621,
            "ci95": [
              -0.0020830507050595415,
              -0.00046568744862094687
            ]
          },
          "auc.cart": {
            "estimate": -0.0022469015743855625,
            "ci95": [
              -0.006610549386567815,
              0.002486132090509292
            ]
          },
          "auc.fav": {
            "estimate": -0.014430457714335376,
            "ci95": [
              -0.020865620618596037,
              -0.007780583471790843
            ]
          },
          "auc.buy": {
            "estimate": -0.006504236889431914,
            "ci95": [
              -0.014616912392191072,
              0.0007220260193942539
            ]
          },
          "bce.is_click": {
            "estimate": -0.00018205039321714245,
            "ci95": [
              -0.00026184841810394027,
              -0.00011068183754537956
            ]
          },
          "bce.cart": {
            "estimate": 9.133320997639638e-06,
            "ci95": [
              -8.934368427796976e-05,
              0.00011029533736547596
            ]
          },
          "bce.fav": {
            "estimate": 0.00025140495102754165,
            "ci95": [
              0.000141354021652254,
              0.0003712023497266197
            ]
          },
          "bce.buy": {
            "estimate": 9.774494629154196e-05,
            "ci95": [
              2.147166339233251e-05,
              0.00018875959980949736
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.0009894507618875004,
            "ci95": [
              -0.0039010035761656374,
              0.005457627726279637
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.013173006901837314,
            "ci95": [
              0.006481312088216837,
              0.02002196445498243
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.0052467860769338515,
            "ci95": [
              -0.0021360535449441776,
              0.013670951501122184
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.012183556139949814,
            "ci95": [
              0.004117205769275656,
              0.020664059344402925
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.004257335315046351,
            "ci95": [
              -0.004007236214954887,
              0.013624425210822508
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.007926220824903463,
            "ci95": [
              -0.018247216073240678,
              0.0019602505298964056
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00019118371421478209,
            "ci95": [
              -0.0003160448601342524,
              -7.155404946442542e-05
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0004334553442446841,
            "ci95": [
              -0.0005723517575590418,
              -0.00029914992133860195
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0002797953395086844,
            "ci95": [
              -0.0003942476039269809,
              -0.00016815009448189314
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.000242271630029902,
            "ci95": [
              -0.0003946309209093298,
              -8.752263650036714e-05
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -8.861162529390232e-05,
            "ci95": [
              -0.00021248343026283346,
              3.4849928750768564e-05
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.0001536600047359997,
            "ci95": [
              1.2798332815017365e-05,
              0.0002913969442236105
            ]
          }
        }
      }
    },
    "parameter_count": 51724260,
    "initializer_sha256": "fd5f866c35c5e29b37841aeed070f6ef5259c773de173df7549d149f87a8c09e",
    "optimizer_layout_sha256": "c5fcaffa9fb82b5bde4678464b0ed4b24cc3f5571f3a724307b965282b1b10db",
    "formal_worker_cost": {
      "wall_seconds": 2528.382029980421,
      "cpu_seconds": 3044.418955067,
      "peak_rss_bytes": 17071022080,
      "gpu_peak_bytes": 28371320832,
      "samples": 2317
    },
    "smoke_worker_cost": {
      "wall_seconds": 152.04752030968666,
      "cpu_seconds": 166.190266503,
      "peak_rss_bytes": 16907530240,
      "gpu_peak_bytes": 29570891776,
      "samples": 140
    },
    "complete_recorded_costs": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2528.382029980421,
        "cpu_seconds": 3044.418955067,
        "peak_rss_bytes": 17071022080,
        "gpu_peak_bytes": 28371320832,
        "samples": 2317
      },
      "formal_runner": {
        "wall_seconds": 2524.141234844923,
        "cpu_seconds": 2927.1683170829997,
        "peak_process_rss_bytes": 5244911616
      },
      "fit_including_validation": {
        "wall_seconds": 2397.915154516697,
        "cpu_seconds": 2747.219195707,
        "peak_process_rss_bytes": 5244911616
      },
      "prediction_export": {
        "wall_seconds": 120.8867464736104,
        "cpu_seconds": 174.2454546419999,
        "peak_process_rss_bytes": 5244911616
      },
      "smoke_worker": {
        "wall_seconds": 152.04752030968666,
        "cpu_seconds": 166.190266503,
        "peak_rss_bytes": 16907530240,
        "gpu_peak_bytes": 29570891776,
        "samples": 140
      },
      "smoke_runner": {
        "wall_seconds": 148.8715279698372,
        "cpu_seconds": 145.14428338000002,
        "peak_process_rss_bytes": 3207770112
      },
      "latency_constructor": {
        "wall_seconds": 0.7723697051405907,
        "cpu_seconds": 0.8444687860000002,
        "peak_rss_bytes": 2092183552
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.12406931072473526,
        "cpu_seconds": 0.1233411509999982,
        "peak_rss_bytes": 2092183552
      },
      "latency_peak_cuda_bytes": 1281155072,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "forward_latency": {
      "mean_ms": 14.31399438381195,
      "p50_ms": 14.303679943084717,
      "p95_ms": 14.376165056228638,
      "batch_size": 1024,
      "warmups": 10,
      "iterations": 40
    },
    "latency_ratio_to_parent": 1.0112867988159484,
    "quality_candidate": true,
    "near_candidate": true
  },
  "validity": {
    "valid": true,
    "violations": [],
    "state": "passed",
    "quality_summary": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/quality-summary.json",
    "no_prediction_to_work_feedback": true,
    "test_evaluated": false,
    "infrastructure_limitations": [
      "Source metrics are identity/range/arithmetic checks, not recomputed AUC/BCE.",
      "Actual saved-array pairing is deferred to the paired analysis; row fingerprints cover original payload user/item/labels.",
      "Production cost receipts are sampled evidence, not new GPU measurements or hard resource guarantees.",
      "Maturity timestamp inference and click-availability assumptions remain unchanged.",
      "Launcher terminal/aggregate cleanup and cost missing; independent real worker terminals accepted via explicit recovery. Original pipeline absent before any measured phase."
    ],
    "recovered_terminal_evidence": {
      "completion_path": null,
      "completion": null,
      "launcher_exit_code": null,
      "launcher_aggregate_cost": null,
      "worker_costs": {
        "first_prefix": {
          "wall_seconds": 2515.3734920471907,
          "cpu_seconds": 3027.6236540930004,
          "peak_rss_bytes": 17074065408,
          "gpu_peak_bytes": 28369223680,
          "samples": 2307
        },
        "first_suffix": {
          "wall_seconds": 2515.9312758892775,
          "cpu_seconds": 3025.838640644,
          "peak_rss_bytes": 16970461184,
          "gpu_peak_bytes": 28369223680,
          "samples": 2308
        },
        "first_even": {
          "wall_seconds": 2515.3534044995904,
          "cpu_seconds": 3025.922370799,
          "peak_rss_bytes": 16983592960,
          "gpu_peak_bytes": 28371320832,
          "samples": 2306
        },
        "first_odd": {
          "wall_seconds": 2528.382029980421,
          "cpu_seconds": 3044.418955067,
          "peak_rss_bytes": 17071022080,
          "gpu_peak_bytes": 28371320832,
          "samples": 2317
        },
        "second_prefix": {
          "wall_seconds": 2519.057643711567,
          "cpu_seconds": 3030.651342093,
          "peak_rss_bytes": 17107640320,
          "gpu_peak_bytes": 28373417984,
          "samples": 2309
        },
        "second_suffix": {
          "wall_seconds": 2511.607097297907,
          "cpu_seconds": 3031.608810618,
          "peak_rss_bytes": 17054855168,
          "gpu_peak_bytes": 28369223680,
          "samples": 2301
        },
        "second_even": {
          "wall_seconds": 2531.232098005712,
          "cpu_seconds": 3039.49522237,
          "peak_rss_bytes": 17119875072,
          "gpu_peak_bytes": 28371320832,
          "samples": 2322
        },
        "second_odd": {
          "wall_seconds": 2538.3111224472523,
          "cpu_seconds": 3053.287257262,
          "peak_rss_bytes": 17063968768,
          "gpu_peak_bytes": 28369223680,
          "samples": 2329
        }
      },
      "recovery_path": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/recovered-worker-terminal.json",
      "recovery_sha256": "e8b859ea8595b1388ca0aecbcef90aaff14e8d86da5a6d28d71fead9dec21285",
      "scope": "All eight real worker/runner terminals verified independently. Launcher terminal and aggregate cleanup/cost are unavailable, not inferred successful; recorded workers absent or zombie. No model retry."
    },
    "source_and_input_identity_scope": "Original N full-input content SHA256 values are inherited through completed C quality and E authorization. Production runners checked input stats before/after their traversals and two original metadata hashes before them. This collector now checks 122 current stats and both original metadata hashes; it does not rehash Parquet or open checkpoint/prediction arrays. New E worker/smoke/effective-optimizer gates are evaluated from actual small receipts. Two references retain exact C quality records and producer completion hashes, with small source/artifact identity checks. Stat continuity cannot independently exclude content changes that preserve stat fields."
  },
  "decision": "keep",
  "decision_reason": "Macro delta +0.001237919,CI95 [-0.0013729652000838489, 0.003909334895993671],latency ratio 1.011287. Research candidate under frozen point/cost policy only; no confirmed parent improvement, click regression, no incumbent promotion.",
  "decision_scope": "Root queue disposition only; no released-incumbent promotion or supported lesson.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_odd",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_odd/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_odd/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_odd/Taobao_Action/first_odd.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/paired-analysis/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/latency/completion.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_odd",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_odd/result.json",
    "result_sha256": "9ca8426ec0ef9fc2f65fd48208f7869c55048dc044b94c3ce6c7d01e9c83c3de",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_odd/manifest.json",
    "manifest_sha256": "62896430d504d095bd7c3bd08c54ce9da5935d8a5aa0e2b589f9709d2a4984ee",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_odd/Taobao_Action/first_odd.model",
    "checkpoint_sha256": "e2123b38b543c10bba28b4ebbd45f558185c3701314463e91e03abcb3197dd1f",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/paired-analysis/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
