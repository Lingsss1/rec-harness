## 246. moment-query--weighted_variance--s20262027

- 假设：First-read weighted_variance projected into second-work w_q before q_norm may improve addressing under fixed two-cycle mature protocol.
- 判定：discard
- 原因：Macro delta -0.002997810, CI95 [-0.005467256714010241, -0.0008875458971996674]; forward ratio 1.164332. Fails fixed positive/near quality-cost policy.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "moment-query--weighted_variance--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-10T07:45:38.318827+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "weighted_variance",
  "code_id": "6503505e6c1699b76ec8cce57b783280806e825c1df3cd8278fa7c9bd751c711",
  "hypothesis": "First-read weighted_variance projected into second-work w_q before q_norm may improve addressing under fixed two-cycle mature protocol.",
  "change": "{\"model\": \"QFormerCross8MomentQuery\", \"maturity_mode\": \"mature\", \"moment_source\": \"weighted_variance\", \"moment_rank\": 16, \"work_steps\": 2, \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8MomentQuery",
    "maturity_mode": "mature",
    "moment_source": "weighted_variance",
    "moment_rank": 16,
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
      "value": 0.7640971416612319,
      "delta": -0.002997809653628558
    },
    "per_task": {
      "is_click": {
        "logloss": 0.18416897702266552,
        "AUC": 0.649038378421717
      },
      "cart": {
        "logloss": 0.021807050961507955,
        "AUC": 0.795304539035351
      },
      "fav": {
        "logloss": 0.01708747710357211,
        "AUC": 0.7954342835053327
      },
      "buy": {
        "logloss": 0.009695444320134111,
        "AUC": 0.816611365682527
      }
    },
    "macro_source_logloss": 0.058189737351969924,
    "source_metric_comparison": {
      "is_click": {
        "source_AUC": 0.649038378421717,
        "computed_AUC": 0.649038378421717,
        "auc_difference": 0.0,
        "source_logloss": 0.18416897702266552,
        "computed_bce": 0.18416897702266932,
        "bce_difference": 3.802513859341161e-15
      },
      "cart": {
        "source_AUC": 0.795304539035351,
        "computed_AUC": 0.795304539035351,
        "auc_difference": 0.0,
        "source_logloss": 0.021807050961507955,
        "computed_bce": 0.021807050985555163,
        "bce_difference": 2.4047208668775966e-11
      },
      "fav": {
        "source_AUC": 0.7954342835053327,
        "computed_AUC": 0.7954342835053327,
        "auc_difference": 0.0,
        "source_logloss": 0.01708747710357211,
        "computed_bce": 0.017087477104170218,
        "bce_difference": 5.981083683881394e-13
      },
      "buy": {
        "source_AUC": 0.816611365682527,
        "computed_AUC": 0.816611365682527,
        "auc_difference": 0.0,
        "source_logloss": 0.009695444320134111,
        "computed_bce": 0.009695444320136542,
        "bce_difference": 2.4303475898435067e-15
      }
    },
    "matched_parent_effect": {
      "coefficients": {
        "mature_r2": -1.0,
        "cross8_mature": 0.0,
        "weighted_variance": 1.0,
        "weighted_mean": 0.0,
        "uniform_variance": 0.0,
        "constant": 0.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": -0.002997809653628558,
          "ci95": [
            -0.005467256714010241,
            -0.0008875458971996674
          ]
        },
        "auc.is_click": {
          "estimate": -0.0008979147792724795,
          "ci95": [
            -0.0016217944355051916,
            -0.0002026190017774904
          ]
        },
        "auc.cart": {
          "estimate": 0.002870268678007326,
          "ci95": [
            -0.001588573175929178,
            0.007094282213311745
          ]
        },
        "auc.fav": {
          "estimate": -0.00702920686971531,
          "ci95": [
            -0.013162474117770606,
            -0.0006596854114584613
          ]
        },
        "auc.buy": {
          "estimate": -0.006934385643533769,
          "ci95": [
            -0.013476621194298266,
            -0.0010259079770417112
          ]
        },
        "bce.is_click": {
          "estimate": -0.00022266924511735176,
          "ci95": [
            -0.0002989554820804159,
            -0.00015890786643934856
          ]
        },
        "bce.cart": {
          "estimate": -2.0903383878868864e-05,
          "ci95": [
            -0.0001184609821208985,
            6.927121010458326e-05
          ]
        },
        "bce.fav": {
          "estimate": 0.0002893194098178842,
          "ci95": [
            0.00017988390234977507,
            0.0004025218576513921
          ]
        },
        "bce.buy": {
          "estimate": 5.164238341083885e-05,
          "ci95": [
            -2.1159508802860624e-05,
            0.0001257481447093608
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": -0.0037681834572798056,
          "ci95": [
            -0.008180642509287525,
            0.0005682088913600906
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": 0.00613129209044283,
          "ci95": [
            -5.944827322835173e-05,
            0.012199586289878541
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": 0.00603647086426129,
          "ci95": [
            -2.087121513610621e-05,
            0.012463884121407759
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": 0.009899475547722636,
          "ci95": [
            0.0017553241424431154,
            0.017433041832032002
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": 0.009804654321541095,
          "ci95": [
            0.0028487200362385086,
            0.01784137216282123
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": -9.482122618154065e-05,
          "ci95": [
            -0.00855915373312191,
            0.009161971303951076
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": -0.0002017658612384829,
          "ci95": [
            -0.000320379919712184,
            -8.252533920682113e-05
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": -0.0005119886549352359,
          "ci95": [
            -0.0006412290636320357,
            -0.00036344476666425285
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": -0.0002743116285281906,
          "ci95": [
            -0.00037515490535737744,
            -0.0001765502973415274
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": -0.00031022279369675304,
          "ci95": [
            -0.0004587860504916886,
            -0.0001513524685944177
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": -7.254576728970771e-05,
          "ci95": [
            -0.0001870373078817585,
            3.4531989038685286e-05
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": 0.00023767702640704533,
          "ci95": [
            0.0001013015345765974,
            0.00037767529922131684
          ]
        }
      }
    },
    "all_prespecified_effects_involving_candidate": {
      "weighted_variance__minus__mature_r2": {
        "coefficients": {
          "mature_r2": -1.0,
          "cross8_mature": 0.0,
          "weighted_variance": 1.0,
          "weighted_mean": 0.0,
          "uniform_variance": 0.0,
          "constant": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.002997809653628558,
            "ci95": [
              -0.005467256714010241,
              -0.0008875458971996674
            ]
          },
          "auc.is_click": {
            "estimate": -0.0008979147792724795,
            "ci95": [
              -0.0016217944355051916,
              -0.0002026190017774904
            ]
          },
          "auc.cart": {
            "estimate": 0.002870268678007326,
            "ci95": [
              -0.001588573175929178,
              0.007094282213311745
            ]
          },
          "auc.fav": {
            "estimate": -0.00702920686971531,
            "ci95": [
              -0.013162474117770606,
              -0.0006596854114584613
            ]
          },
          "auc.buy": {
            "estimate": -0.006934385643533769,
            "ci95": [
              -0.013476621194298266,
              -0.0010259079770417112
            ]
          },
          "bce.is_click": {
            "estimate": -0.00022266924511735176,
            "ci95": [
              -0.0002989554820804159,
              -0.00015890786643934856
            ]
          },
          "bce.cart": {
            "estimate": -2.0903383878868864e-05,
            "ci95": [
              -0.0001184609821208985,
              6.927121010458326e-05
            ]
          },
          "bce.fav": {
            "estimate": 0.0002893194098178842,
            "ci95": [
              0.00017988390234977507,
              0.0004025218576513921
            ]
          },
          "bce.buy": {
            "estimate": 5.164238341083885e-05,
            "ci95": [
              -2.1159508802860624e-05,
              0.0001257481447093608
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.0037681834572798056,
            "ci95": [
              -0.008180642509287525,
              0.0005682088913600906
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.00613129209044283,
            "ci95": [
              -5.944827322835173e-05,
              0.012199586289878541
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.00603647086426129,
            "ci95": [
              -2.087121513610621e-05,
              0.012463884121407759
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.009899475547722636,
            "ci95": [
              0.0017553241424431154,
              0.017433041832032002
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.009804654321541095,
            "ci95": [
              0.0028487200362385086,
              0.01784137216282123
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -9.482122618154065e-05,
            "ci95": [
              -0.00855915373312191,
              0.009161971303951076
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.0002017658612384829,
            "ci95": [
              -0.000320379919712184,
              -8.252533920682113e-05
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0005119886549352359,
            "ci95": [
              -0.0006412290636320357,
              -0.00036344476666425285
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0002743116285281906,
            "ci95": [
              -0.00037515490535737744,
              -0.0001765502973415274
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.00031022279369675304,
            "ci95": [
              -0.0004587860504916886,
              -0.0001513524685944177
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -7.254576728970771e-05,
            "ci95": [
              -0.0001870373078817585,
              3.4531989038685286e-05
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.00023767702640704533,
            "ci95": [
              0.0001013015345765974,
              0.00037767529922131684
            ]
          }
        }
      },
      "weighted_variance__minus__cross8_mature": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": -1.0,
          "weighted_variance": 1.0,
          "weighted_mean": 0.0,
          "uniform_variance": 0.0,
          "constant": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.004690721565772105,
            "ci95": [
              0.0021007615184228745,
              0.007722091482036069
            ]
          },
          "auc.is_click": {
            "estimate": -0.0008983917212487258,
            "ci95": [
              -0.0016967376012269081,
              -0.00016408953303775905
            ]
          },
          "auc.cart": {
            "estimate": 0.006097803664235069,
            "ci95": [
              0.0017191639346535116,
              0.010179314712989922
            ]
          },
          "auc.fav": {
            "estimate": -0.0010677275221168125,
            "ci95": [
              -0.0063923368986875725,
              0.0042861615870950324
            ]
          },
          "auc.buy": {
            "estimate": 0.014631201842218888,
            "ci95": [
              0.006730669250565028,
              0.02483000141343032
            ]
          },
          "bce.is_click": {
            "estimate": -0.00014896082592202875,
            "ci95": [
              -0.00022258766479504242,
              -7.491005292653634e-05
            ]
          },
          "bce.cart": {
            "estimate": -2.7360651523866686e-05,
            "ci95": [
              -0.00012220002763555746,
              8.217905991476448e-05
            ]
          },
          "bce.fav": {
            "estimate": 0.00011409792271717054,
            "ci95": [
              -7.503456525029859e-06,
              0.00022883150189914324
            ]
          },
          "bce.buy": {
            "estimate": -0.0003119460083015873,
            "ci95": [
              -0.0004222007052146032,
              -0.00022113087215740757
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.006996195385483794,
            "ci95": [
              -0.011163362782186362,
              -0.002489428199208194
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.00016933580086808675,
            "ci95": [
              -0.005306463898482469,
              0.00546491111464256
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.015529593563467614,
            "ci95": [
              -0.02579410831887959,
              -0.007373283118005141
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.007165531186351881,
            "ci95": [
              -0.00024910364524085,
              0.014374917985784637
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.00853339817798382,
            "ci95": [
              -0.018566812493968327,
              0.0008660642512761986
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.0156989293643357,
            "ci95": [
              -0.027691480630839198,
              -0.005954708852439973
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00012160017439816206,
            "ci95": [
              -0.0002436458669910711,
              -2.3794843354414535e-06
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0002630587486391993,
            "ci95": [
              -0.0003927069288448324,
              -0.0001217203673296636
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.00016298518237955854,
            "ci95": [
              5.439536535596723e-05,
              0.000303457468306504
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.00014145857424103722,
            "ci95": [
              -0.0002901311582244509,
              2.8788747964745542e-05
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.0002845853567777206,
            "ci95": [
              0.00015236564320528697,
              0.00044154709576423013
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.00042604393101875783,
            "ci95": [
              0.0002783024858356399,
              0.0005949294213600468
            ]
          }
        }
      },
      "weighted_variance__minus__weighted_mean": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "weighted_variance": 1.0,
          "weighted_mean": -1.0,
          "uniform_variance": 0.0,
          "constant": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.0023036146041760797,
            "ci95": [
              -0.0004429570498071285,
              0.0048404831683537
            ]
          },
          "auc.is_click": {
            "estimate": -0.0034692032512519955,
            "ci95": [
              -0.0042099210311445255,
              -0.002734907013497859
            ]
          },
          "auc.cart": {
            "estimate": 0.01411341872191707,
            "ci95": [
              0.009317102525011584,
              0.019059107992267173
            ]
          },
          "auc.fav": {
            "estimate": -0.0077425590093122,
            "ci95": [
              -0.01367159045648055,
              -0.0027438616966513956
            ]
          },
          "auc.buy": {
            "estimate": 0.006312801955351444,
            "ci95": [
              -0.0013911740661540631,
              0.013448899226253964
            ]
          },
          "bce.is_click": {
            "estimate": 0.0004369692999866004,
            "ci95": [
              0.0003638927365315785,
              0.0005070077072059767
            ]
          },
          "bce.cart": {
            "estimate": -0.0002633237012521529,
            "ci95": [
              -0.000353929271513768,
              -0.00016377326243316077
            ]
          },
          "bce.fav": {
            "estimate": 0.00021095386264838659,
            "ci95": [
              9.730977006989225e-05,
              0.0003461506670391807
            ]
          },
          "bce.buy": {
            "estimate": -0.00029253054590009167,
            "ci95": [
              -0.00037832145400506785,
              -0.0002107467897930234
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.017582621973169066,
            "ci95": [
              -0.02246165978622653,
              -0.012950341331327931
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.004273355758060204,
            "ci95": [
              -0.0007873012386292315,
              0.00998734591031036
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.009782005206603439,
            "ci95": [
              -0.017100468864611643,
              -0.0020482422989873744
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.02185597773122927,
            "ci95": [
              0.01502369422673729,
              0.029470760346938584
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.007800616766565627,
            "ci95": [
              -0.00037241663395438387,
              0.016663965084684543
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.014055360964663643,
            "ci95": [
              -0.0232680643810132,
              -0.005418260291564084
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 0.0007002930012387533,
            "ci95": [
              0.0005751403762502239,
              0.0008290950764614996
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.0002260154373382138,
            "ci95": [
              8.134245661605621e-05,
              0.0003576479919595265
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.0007294998458866921,
            "ci95": [
              0.0006257469304969351,
              0.0008514099858420474
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.0004742775639005395,
            "ci95": [
              -0.0006422393225277434,
              -0.0003197592857339179
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 2.9206844647938768e-05,
            "ci95": [
              -9.447780391591323e-05,
              0.00015528664410633964
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.0005034844085484783,
            "ci95": [
              0.00035791386472333115,
              0.0006562708647581595
            ]
          }
        }
      },
      "weighted_variance__minus__uniform_variance": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "weighted_variance": 1.0,
          "weighted_mean": 0.0,
          "uniform_variance": -1.0,
          "constant": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.003018002147557569,
            "ci95": [
              9.167537280184164e-05,
              0.005520583184496593
            ]
          },
          "auc.is_click": {
            "estimate": 0.000951775930532639,
            "ci95": [
              0.00014297745677526735,
              0.0017385178220825272
            ]
          },
          "auc.cart": {
            "estimate": 0.007285048028443986,
            "ci95": [
              0.0026277258414744127,
              0.011710599131552047
            ]
          },
          "auc.fav": {
            "estimate": 0.0016125458520793723,
            "ci95": [
              -0.00502695067212002,
              0.007959299223811148
            ]
          },
          "auc.buy": {
            "estimate": 0.0022226387791742797,
            "ci95": [
              -0.004852794634018748,
              0.008783675743769799
            ]
          },
          "bce.is_click": {
            "estimate": -0.0002984928244045004,
            "ci95": [
              -0.0003740195136190699,
              -0.0002254925075857367
            ]
          },
          "bce.cart": {
            "estimate": -0.00015642330547279565,
            "ci95": [
              -0.00025880888923498455,
              -5.5484922238189366e-05
            ]
          },
          "bce.fav": {
            "estimate": -5.6952749974237965e-05,
            "ci95": [
              -0.0001852582940801142,
              7.365153581451671e-05
            ]
          },
          "bce.buy": {
            "estimate": -0.00010859226464286255,
            "ci95": [
              -0.00018362069272142827,
              -3.564987618490615e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.006333272097911347,
            "ci95": [
              -0.011014441589988497,
              -0.001602167910228042
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.0006607699215467333,
            "ci95": [
              -0.00721806463893093,
              0.005826817334460169
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.0012708628486416407,
            "ci95": [
              -0.007727414134466465,
              0.005764080254292491
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.005672502176364613,
            "ci95": [
              -0.00213364854916806,
              0.014256181731033773
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.005062409249269706,
            "ci95": [
              -0.0029616352562864628,
              0.013403864271763958
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.0006100929270949074,
            "ci95": [
              -0.009734634627143548,
              0.009032846396008539
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00014206951893170477,
            "ci95": [
              -0.0002625617421785112,
              -1.612607009093067e-05
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.00024154007443026246,
            "ci95": [
              -0.0003946144150725248,
              -9.574988574552302e-05
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.00018990055976163787,
            "ci95": [
              -0.00030322606489840173,
              -8.883451437438966e-05
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -9.947055549855768e-05,
            "ci95": [
              -0.0002764136467444895,
              7.332735129048814e-05
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -4.78310408299331e-05,
            "ci95": [
              -0.00017103267286881622,
              8.579658837159913e-05
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 5.1639514668624584e-05,
            "ci95": [
              -0.00010030087937810011,
              0.00020800788996259617
            ]
          }
        }
      },
      "weighted_variance__minus__constant": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "weighted_variance": 1.0,
          "weighted_mean": 0.0,
          "uniform_variance": 0.0,
          "constant": -1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.0010076147409633918,
            "ci95": [
              -0.001312000048539222,
              0.003495990945661
            ]
          },
          "auc.is_click": {
            "estimate": -0.0028478833736630405,
            "ci95": [
              -0.00361373678771999,
              -0.0020723054500859373
            ]
          },
          "auc.cart": {
            "estimate": 0.0028015730033721375,
            "ci95": [
              -0.001035873870430906,
              0.006743344175486587
            ]
          },
          "auc.fav": {
            "estimate": 0.0013715418070762153,
            "ci95": [
              -0.004169516632175555,
              0.006571537200249285
            ]
          },
          "auc.buy": {
            "estimate": 0.0027052275270682546,
            "ci95": [
              -0.004679063429016292,
              0.010305595557222045
            ]
          },
          "bce.is_click": {
            "estimate": 0.00033260436741466837,
            "ci95": [
              0.00025981228773980377,
              0.0004007325433934082
            ]
          },
          "bce.cart": {
            "estimate": -4.571943715880877e-05,
            "ci95": [
              -0.00013342733540122116,
              4.8798669873091224e-05
            ]
          },
          "bce.fav": {
            "estimate": 0.00013312273266448404,
            "ci95": [
              8.51187137205908e-06,
              0.0002681496698649932
            ]
          },
          "bce.buy": {
            "estimate": -0.00012026760662950733,
            "ci95": [
              -0.00020263939652770006,
              -4.557491454568131e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.005649456377035178,
            "ci95": [
              -0.009457557955567047,
              -0.0018518483206230916
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.004219425180739256,
            "ci95": [
              -0.009602103771511935,
              0.001025710431043123
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.005553110900731295,
            "ci95": [
              -0.013454521797720834,
              0.0019785285599988223
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.0014300311962959222,
            "ci95": [
              -0.005373383107667031,
              0.007991635766073773
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 9.63454763038829e-05,
            "ci95": [
              -0.007776640167332302,
              0.008966430777498629
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.0013336857199920393,
            "ci95": [
              -0.011092980704187527,
              0.00781149511559419
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 0.00037832380457347714,
            "ci95": [
              0.00025899680818555526,
              0.0004842273109754739
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.00019948163475018432,
            "ci95": [
              6.111490177526718e-05,
              0.00033755102478208633
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.0004528719740441757,
            "ci95": [
              0.00035974802496465953,
              0.0005651185493939918
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.00017884216982329282,
            "ci95": [
              -0.0003368528277927165,
              -2.6262331534257255e-05
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 7.454816947069856e-05,
            "ci95": [
              -4.124682649963448e-05,
              0.00019852555365949218
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.0002533903392939914,
            "ci95": [
              0.0001100210252623472,
              0.0004226471333333151
            ]
          }
        }
      }
    },
    "parameter_count": 51736548,
    "initializer_sha256": "796cec890828fda2f18f4ecf0c052e81999552b8179857b95f1c309d637d6ccb",
    "optimizer_layout_sha256": "8643b4ea874ae9f7144e436250213c25b98bd6419c31113f4b0a2f04e9aeed1d",
    "formal_worker_cost": {
      "wall_seconds": 2648.5044471919537,
      "cpu_seconds": 3150.032396146,
      "peak_rss_bytes": 17106907136,
      "gpu_peak_bytes": 32622247936,
      "samples": 2426
    },
    "smoke_worker_cost": {
      "wall_seconds": 164.33842162787914,
      "cpu_seconds": 221.305303578,
      "peak_rss_bytes": 19607048192,
      "gpu_peak_bytes": 33826013184,
      "samples": 151
    },
    "complete_recorded_costs": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2648.5044471919537,
        "cpu_seconds": 3150.032396146,
        "peak_rss_bytes": 17106907136,
        "gpu_peak_bytes": 32622247936,
        "samples": 2426
      },
      "formal_runner": {
        "wall_seconds": 2644.3102850094438,
        "cpu_seconds": 3032.661496179,
        "peak_process_rss_bytes": 5248606208
      },
      "fit_including_validation": {
        "wall_seconds": 2514.1464553028345,
        "cpu_seconds": 2855.036611121,
        "peak_process_rss_bytes": 5248606208
      },
      "prediction_export": {
        "wall_seconds": 125.24761318415403,
        "cpu_seconds": 172.05325436300018,
        "peak_process_rss_bytes": 5248606208
      },
      "smoke_worker": {
        "wall_seconds": 164.33842162787914,
        "cpu_seconds": 221.305303578,
        "peak_rss_bytes": 19607048192,
        "gpu_peak_bytes": 33826013184,
        "samples": 151
      },
      "smoke_runner": {
        "wall_seconds": 159.66691584140062,
        "cpu_seconds": 152.100370713,
        "peak_process_rss_bytes": 3300425728
      },
      "latency_constructor": {
        "wall_seconds": 0.7710365206003189,
        "cpu_seconds": 0.840356689,
        "peak_rss_bytes": 2091470848
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.12224019318819046,
        "cpu_seconds": 0.12112652300000093,
        "peak_rss_bytes": 2091470848
      },
      "latency_peak_cuda_bytes": 1369385472,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "forward_latency": {
      "mean_ms": 16.722143125534057,
      "p50_ms": 16.66982364654541,
      "p95_ms": 16.864068698883056,
      "batch_size": 1024,
      "warmups": 10,
      "iterations": 40
    },
    "latency_ratio_to_parent": 1.164332289034566,
    "quality_candidate": false,
    "near_candidate": false
  },
  "validity": {
    "valid": true,
    "violations": [],
    "state": "passed",
    "quality_summary": "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/quality-summary.json",
    "no_prediction_to_work_feedback": true,
    "test_evaluated": false,
    "source_and_input_identity_scope": "Original N full-input content SHA256 values are inherited through completed C quality and E authorization. Production runners checked input stats before/after their traversals and two original metadata hashes before them. This collector now checks 122 current stats and both original metadata hashes; it does not rehash Parquet or open checkpoint/prediction arrays. New E worker/smoke/effective-optimizer gates are evaluated from actual small receipts. Two references retain exact C quality records and producer completion hashes, with small source/artifact identity checks. Stat continuity cannot independently exclude content changes that preserve stat fields."
  },
  "decision": "discard",
  "decision_reason": "Macro delta -0.002997810, CI95 [-0.005467256714010241, -0.0008875458971996674]; forward ratio 1.164332. Fails fixed positive/near quality-cost policy.",
  "decision_scope": "Root queue disposition only; no released-incumbent promotion or supported lesson.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/weighted_variance",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/weighted_variance/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/weighted_variance/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/weighted_variance/Taobao_Action/weighted_variance.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/paired-analysis/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/latency/completion.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/weighted_variance",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/weighted_variance/result.json",
    "result_sha256": "3fe81cc4deb4139c08d48c09c015a6e357c11749c29e6bab58fc3ee108e2ad61",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/weighted_variance/manifest.json",
    "manifest_sha256": "c0ec969e86c38efaf31a5a34e15d930c6059b64ef19743babe7c2dc8060f73af",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/weighted_variance/Taobao_Action/weighted_variance.model",
    "checkpoint_sha256": "d2a5e9275bb323d892216d1108d0059c1e280ffaeee930997fdc92724f64f016",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/paired-analysis/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
