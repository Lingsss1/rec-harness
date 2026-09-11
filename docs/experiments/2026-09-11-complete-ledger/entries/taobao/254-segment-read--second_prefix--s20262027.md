## 254. segment-read--second_prefix--s20262027

- 假设：Equal-count second_prefix stored-history support may change useful two-round retrieval conditions; no pure Q2 mediation claim.
- 判定：discard
- 原因：Macro delta -0.012856043,CI95 [-0.017204751230466074, -0.00938755160395674],latency ratio 1.006201. Fails positive and near-candidate quality policy; preserve negative evidence.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "segment-read--second_prefix--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-10T14:13:03.864962+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "second_prefix",
  "code_id": "cf7b0840f7d3f28bb5451ac3cc4d9491bfcf60c5a00b32fc75f78b7e407fc0de",
  "hypothesis": "Equal-count second_prefix stored-history support may change useful two-round retrieval conditions; no pure Q2 mediation claim.",
  "change": "{\"model\": \"QFormerCross8SegmentRead\", \"maturity_mode\": \"mature\", \"segment_placement\": \"second\", \"segment_selection\": \"prefix\", \"work_steps\": 2, \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8SegmentRead",
    "maturity_mode": "mature",
    "segment_placement": "second",
    "segment_selection": "prefix",
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
      "value": 0.7542389083026836,
      "delta": -0.012856043012176799
    },
    "per_task": {
      "is_click": {
        "logloss": 0.18438424595934844,
        "AUC": 0.6495391421153702
      },
      "cart": {
        "logloss": 0.02217250524845255,
        "AUC": 0.7869076647085901
      },
      "fav": {
        "logloss": 0.01724822753660957,
        "AUC": 0.7842745302113829
      },
      "buy": {
        "logloss": 0.010402233071131192,
        "AUC": 0.7962342961753915
      }
    },
    "macro_source_logloss": 0.058551802953885436,
    "source_metric_comparison": {
      "is_click": {
        "source_AUC": 0.6495391421153702,
        "computed_AUC": 0.6495391421153702,
        "auc_difference": 0.0,
        "source_logloss": 0.18438424595934844,
        "computed_bce": 0.184384245959348,
        "bce_difference": -4.440892098500626e-16
      },
      "cart": {
        "source_AUC": 0.7869076647085901,
        "computed_AUC": 0.7869076647085901,
        "auc_difference": 0.0,
        "source_logloss": 0.02217250524845255,
        "computed_bce": 0.022172505250173353,
        "bce_difference": 1.7208040548055692e-12
      },
      "fav": {
        "source_AUC": 0.7842745302113829,
        "computed_AUC": 0.7842745302113829,
        "auc_difference": 0.0,
        "source_logloss": 0.01724822753660957,
        "computed_bce": 0.01724822756509016,
        "bce_difference": 2.8480589414625612e-11
      },
      "buy": {
        "source_AUC": 0.7962342961753915,
        "computed_AUC": 0.7962342961753915,
        "auc_difference": 0.0,
        "source_logloss": 0.010402233071131192,
        "computed_bce": 0.010402234719522535,
        "bce_difference": 1.6483913435616504e-09
      }
    },
    "matched_parent_effect": {
      "coefficients": {
        "mature_r2": -1.0,
        "cross8_mature": 0.0,
        "first_prefix": 0.0,
        "first_suffix": 0.0,
        "first_even": 0.0,
        "first_odd": 0.0,
        "second_prefix": 1.0,
        "second_suffix": 0.0,
        "second_even": 0.0,
        "second_odd": 0.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": -0.012856043012176799,
          "ci95": [
            -0.017204751230466074,
            -0.00938755160395674
          ]
        },
        "auc.is_click": {
          "estimate": -0.0003971510856192184,
          "ci95": [
            -0.001106794023088814,
            0.00040417785523328136
          ]
        },
        "auc.cart": {
          "estimate": -0.005526605648753624,
          "ci95": [
            -0.010413570538736283,
            0.0002624083696411795
          ]
        },
        "auc.fav": {
          "estimate": -0.018188960163665113,
          "ci95": [
            -0.024614289840437304,
            -0.012881221637827569
          ]
        },
        "auc.buy": {
          "estimate": -0.02731145515066924,
          "ci95": [
            -0.03995402574488956,
            -0.016597849666017243
          ]
        },
        "bce.is_click": {
          "estimate": -7.400308438682002e-06,
          "ci95": [
            -8.266586161347899e-05,
            6.50399030491403e-05
          ]
        },
        "bce.cart": {
          "estimate": 0.0003445508807393208,
          "ci95": [
            0.00021829887202845313,
            0.0004454590754602181
          ]
        },
        "bce.fav": {
          "estimate": 0.00045006987073782645,
          "ci95": [
            0.0003229326397830515,
            0.0005925733403505732
          ]
        },
        "bce.buy": {
          "estimate": 0.0007584327827968324,
          "ci95": [
            0.0006490853115403459,
            0.0008904521298836857
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": 0.005129454563134406,
          "ci95": [
            -0.00044615420456618185,
            0.010311450581938443
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": 0.017791809078045895,
          "ci95": [
            0.012535746193351382,
            0.02425564856411638
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": 0.02691430406505002,
          "ci95": [
            0.0164601796125095,
            0.03932146664937408
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": 0.012662354514911489,
          "ci95": [
            0.004942017819198827,
            0.021274208235521955
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": 0.021784849501915615,
          "ci95": [
            0.011063038055416098,
            0.03517804945589498
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": 0.009122494987004126,
          "ci95": [
            -0.00331234978380589,
            0.021730817911621476
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": -0.0003519511891780028,
          "ci95": [
            -0.0004899187716339627,
            -0.00020979236101787807
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": -0.00045747017917650845,
          "ci95": [
            -0.0006169037246248787,
            -0.00031405758849797086
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": -0.0007658330912355144,
          "ci95": [
            -0.0009167285233602198,
            -0.000645678150485077
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": -0.00010551898999850567,
          "ci95": [
            -0.00029832851775603455,
            5.744926513422173e-05
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": -0.00041388190205751164,
          "ci95": [
            -0.000599613389884752,
            -0.0002719153383675574
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": -0.00030836291205900597,
          "ci95": [
            -0.0004820470528112894,
            -0.00013348509001399523
          ]
        }
      }
    },
    "all_prespecified_effects_involving_candidate": {
      "second_prefix__minus__mature_r2": {
        "coefficients": {
          "mature_r2": -1.0,
          "cross8_mature": 0.0,
          "first_prefix": 0.0,
          "first_suffix": 0.0,
          "first_even": 0.0,
          "first_odd": 0.0,
          "second_prefix": 1.0,
          "second_suffix": 0.0,
          "second_even": 0.0,
          "second_odd": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.012856043012176799,
            "ci95": [
              -0.017204751230466074,
              -0.00938755160395674
            ]
          },
          "auc.is_click": {
            "estimate": -0.0003971510856192184,
            "ci95": [
              -0.001106794023088814,
              0.00040417785523328136
            ]
          },
          "auc.cart": {
            "estimate": -0.005526605648753624,
            "ci95": [
              -0.010413570538736283,
              0.0002624083696411795
            ]
          },
          "auc.fav": {
            "estimate": -0.018188960163665113,
            "ci95": [
              -0.024614289840437304,
              -0.012881221637827569
            ]
          },
          "auc.buy": {
            "estimate": -0.02731145515066924,
            "ci95": [
              -0.03995402574488956,
              -0.016597849666017243
            ]
          },
          "bce.is_click": {
            "estimate": -7.400308438682002e-06,
            "ci95": [
              -8.266586161347899e-05,
              6.50399030491403e-05
            ]
          },
          "bce.cart": {
            "estimate": 0.0003445508807393208,
            "ci95": [
              0.00021829887202845313,
              0.0004454590754602181
            ]
          },
          "bce.fav": {
            "estimate": 0.00045006987073782645,
            "ci95": [
              0.0003229326397830515,
              0.0005925733403505732
            ]
          },
          "bce.buy": {
            "estimate": 0.0007584327827968324,
            "ci95": [
              0.0006490853115403459,
              0.0008904521298836857
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.005129454563134406,
            "ci95": [
              -0.00044615420456618185,
              0.010311450581938443
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.017791809078045895,
            "ci95": [
              0.012535746193351382,
              0.02425564856411638
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.02691430406505002,
            "ci95": [
              0.0164601796125095,
              0.03932146664937408
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.012662354514911489,
            "ci95": [
              0.004942017819198827,
              0.021274208235521955
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.021784849501915615,
            "ci95": [
              0.011063038055416098,
              0.03517804945589498
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.009122494987004126,
            "ci95": [
              -0.00331234978380589,
              0.021730817911621476
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.0003519511891780028,
            "ci95": [
              -0.0004899187716339627,
              -0.00020979236101787807
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.00045747017917650845,
            "ci95": [
              -0.0006169037246248787,
              -0.00031405758849797086
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0007658330912355144,
            "ci95": [
              -0.0009167285233602198,
              -0.000645678150485077
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.00010551898999850567,
            "ci95": [
              -0.00029832851775603455,
              5.744926513422173e-05
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -0.00041388190205751164,
            "ci95": [
              -0.000599613389884752,
              -0.0002719153383675574
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.00030836291205900597,
            "ci95": [
              -0.0004820470528112894,
              -0.00013348509001399523
            ]
          }
        }
      },
      "second_prefix__minus__cross8_mature": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": -1.0,
          "first_prefix": 0.0,
          "first_suffix": 0.0,
          "first_even": 0.0,
          "first_odd": 0.0,
          "second_prefix": 1.0,
          "second_suffix": 0.0,
          "second_even": 0.0,
          "second_odd": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.005167511792776136,
            "ci95": [
              -0.008388430236713419,
              -0.002434450735359026
            ]
          },
          "auc.is_click": {
            "estimate": -0.00039762802759546467,
            "ci95": [
              -0.00121344629945542,
              0.00040962641042941963
            ]
          },
          "auc.cart": {
            "estimate": -0.0022990706625258817,
            "ci95": [
              -0.006772765751770193,
              0.0025456301550262754
            ]
          },
          "auc.fav": {
            "estimate": -0.012227480816066616,
            "ci95": [
              -0.018787970929453703,
              -0.006534769684544927
            ]
          },
          "auc.buy": {
            "estimate": -0.005745867664916582,
            "ci95": [
              -0.014258767544567697,
              0.0028455549754504415
            ]
          },
          "bce.is_click": {
            "estimate": 6.6308110756641e-05,
            "ci95": [
              -5.857067716415892e-06,
              0.0001360926338690719
            ]
          },
          "bce.cart": {
            "estimate": 0.00033809361309432295,
            "ci95": [
              0.00022885842779100956,
              0.0004384787727043346
            ]
          },
          "bce.fav": {
            "estimate": 0.0002748483836371128,
            "ci95": [
              0.00016986895968348765,
              0.00039317068683925396
            ]
          },
          "bce.buy": {
            "estimate": 0.0003948443910844063,
            "ci95": [
              0.0003108512325265589,
              0.0004894365432539157
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.001901442634930417,
            "ci95": [
              -0.002997783907049722,
              0.006624265346440896
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.011829852788471151,
            "ci95": [
              0.006078429872115399,
              0.018514582531333677
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.005348239637321117,
            "ci95": [
              -0.0031015874073714892,
              0.013830912400962993
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.009928410153540734,
            "ci95": [
              0.002531489953002042,
              0.01804646748615099
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.0034467970023907,
            "ci95": [
              -0.005876866139100472,
              0.012544472656608855
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.006481613151150034,
            "ci95": [
              -0.017228436568072967,
              0.0034534252249790354
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00027178550233768195,
            "ci95": [
              -0.0003988215099348665,
              -0.00014975499213574375
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0002085402728804718,
            "ci95": [
              -0.00035389321237711024,
              -8.38118806065956e-05
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0003285362803277653,
            "ci95": [
              -0.00045167869428677196,
              -0.0002087861873532572
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 6.324522945721014e-05,
            "ci95": [
              -0.0001023897204448847,
              0.00021402384487455428
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -5.675077799008332e-05,
            "ci95": [
              -0.00019406351832189912,
              7.974622654313788e-05
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.00011999600744729347,
            "ci95": [
              -0.00025691532494320107,
              3.1804789138802e-05
            ]
          }
        }
      },
      "first_prefix__minus__second_prefix": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "first_prefix": 1.0,
          "first_suffix": 0.0,
          "first_even": 0.0,
          "first_odd": 0.0,
          "second_prefix": -1.0,
          "second_suffix": 0.0,
          "second_even": 0.0,
          "second_odd": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.00956321426908388,
            "ci95": [
              0.006487903188882395,
              0.012791765389221813
            ]
          },
          "auc.is_click": {
            "estimate": -0.0019123167180510325,
            "ci95": [
              -0.002708069862342108,
              -0.0011151527778077294
            ]
          },
          "auc.cart": {
            "estimate": 0.012332597847164561,
            "ci95": [
              0.007606437189914636,
              0.01660450814823223
            ]
          },
          "auc.fav": {
            "estimate": 0.006528668608793087,
            "ci95": [
              0.0010715384832973964,
              0.012545626221149587
            ]
          },
          "auc.buy": {
            "estimate": 0.021303907338428907,
            "ci95": [
              0.012200701657887175,
              0.03214753013351208
            ]
          },
          "bce.is_click": {
            "estimate": -0.00019065474320070774,
            "ci95": [
              -0.00027084877460938,
              -0.00011017924410523737
            ]
          },
          "bce.cart": {
            "estimate": -0.00039345952845470933,
            "ci95": [
              -0.0004903578109213673,
              -0.0002933538165091549
            ]
          },
          "bce.fav": {
            "estimate": -0.00024195288787758062,
            "ci95": [
              -0.00034029287645004173,
              -0.00014933617192225467
            ]
          },
          "bce.buy": {
            "estimate": -0.000585623049769118,
            "ci95": [
              -0.0007029480767538493,
              -0.000483544153222687
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.014244914565215594,
            "ci95": [
              -0.018630049405753184,
              -0.009421810655348447
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.00844098532684412,
            "ci95": [
              -0.014357684309107083,
              -0.0027693685741523935
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.02321622405647994,
            "ci95": [
              -0.034329669382129634,
              -0.01424002321765235
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.005803929238371475,
            "ci95": [
              -0.0018282509660889012,
              0.013570892680277743
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.008971309491264345,
            "ci95": [
              -0.019244146609258772,
              0.0009694893886259763
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.01477523872963582,
            "ci95": [
              -0.025897040095752544,
              -0.004118652077734051
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 0.00020280478525400158,
            "ci95": [
              7.431393662499161e-05,
              0.0003268132899766669
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 5.129814467687288e-05,
            "ci95": [
              -6.723080860852022e-05,
              0.0001802313375675941
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.0003949683065684103,
            "ci95": [
              0.00027440218552851625,
              0.0005361226342804539
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.0001515066405771287,
            "ci95": [
              -0.0002989522537804487,
              -1.3265204008866264e-05
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.0001921635213144087,
            "ci95": [
              6.064099717223554e-05,
              0.0003335769715833229
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.0003436701618915374,
            "ci95": [
              0.00020444228592947727,
              0.0004881098895368556
            ]
          }
        }
      },
      "second_prefix__minus__second_even": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "first_prefix": 0.0,
          "first_suffix": 0.0,
          "first_even": 0.0,
          "first_odd": 0.0,
          "second_prefix": 1.0,
          "second_suffix": 0.0,
          "second_even": -1.0,
          "second_odd": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.010748280348868294,
            "ci95": [
              -0.014002541293038531,
              -0.008163577204027885
            ]
          },
          "auc.is_click": {
            "estimate": -0.0012379447803100296,
            "ci95": [
              -0.002007532069612564,
              -0.0004273447689777057
            ]
          },
          "auc.cart": {
            "estimate": -0.009463697506211788,
            "ci95": [
              -0.013903724132838182,
              -0.005082059476513132
            ]
          },
          "auc.fav": {
            "estimate": -0.009535812322085113,
            "ci95": [
              -0.01646187740483889,
              -0.003467790541667004
            ]
          },
          "auc.buy": {
            "estimate": -0.022755666786866247,
            "ci95": [
              -0.03214472622178517,
              -0.015054924219023179
            ]
          },
          "bce.is_click": {
            "estimate": 1.4263451797724702e-06,
            "ci95": [
              -6.829935612439331e-05,
              6.657686870108757e-05
            ]
          },
          "bce.cart": {
            "estimate": 0.00021803607013800783,
            "ci95": [
              0.00011729644140085205,
              0.00031508176270616395
            ]
          },
          "bce.fav": {
            "estimate": 0.00016760891532698052,
            "ci95": [
              6.717638796101032e-05,
              0.00028257243083204265
            ]
          },
          "bce.buy": {
            "estimate": 0.0006077629156135706,
            "ci95": [
              0.0005202937317137855,
              0.0007131471574461109
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.008225752725901758,
            "ci95": [
              0.003979407138466057,
              0.012702854625247792
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.008297867541775084,
            "ci95": [
              0.002085428339112977,
              0.015362085715778595
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.021517722006556217,
            "ci95": [
              0.01364816710562455,
              0.03102794294873698
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 7.211481587332536e-05,
            "ci95": [
              -0.007252193567807782,
              0.008151288787962292
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.013291969280654459,
            "ci95": [
              0.0040845612873496836,
              0.024335140035365587
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.013219854464781133,
            "ci95": [
              0.0023080696571488816,
              0.02400088998149202
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00021660972495823536,
            "ci95": [
              -0.00034610193182770557,
              -9.613482105627532e-05
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.00016618257014720805,
            "ci95": [
              -0.0003014458998537552,
              -5.082871294254164e-05
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0006063365704337981,
            "ci95": [
              -0.0007286984793172682,
              -0.0004968806329081164
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 5.04271548110273e-05,
            "ci95": [
              -0.00011502132391167816,
              0.00019110450890513488
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -0.0003897268454755628,
            "ci95": [
              -0.0005392487366485203,
              -0.0002624742778312523
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.0004401540002865901,
            "ci95": [
              -0.00058105265074601,
              -0.0002973536573848461
            ]
          }
        }
      },
      "second_prefix__minus__second_odd": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "first_prefix": 0.0,
          "first_suffix": 0.0,
          "first_even": 0.0,
          "first_odd": 0.0,
          "second_prefix": 1.0,
          "second_suffix": 0.0,
          "second_even": 0.0,
          "second_odd": -1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.00908259981984369,
            "ci95": [
              -0.012232931472353294,
              -0.00643034544077727
            ]
          },
          "auc.is_click": {
            "estimate": 0.00026474829336831096,
            "ci95": [
              -0.0004657737478307583,
              0.001145597285320407
            ]
          },
          "auc.cart": {
            "estimate": -0.004133135059150916,
            "ci95": [
              -0.008956875887183868,
              0.0009577371131199566
            ]
          },
          "auc.fav": {
            "estimate": -0.002014712883722347,
            "ci95": [
              -0.008196437015917848,
              0.0033336262494544217
            ]
          },
          "auc.buy": {
            "estimate": -0.030447299629869806,
            "ci95": [
              -0.040148956948136506,
              -0.022647722537596525
            ]
          },
          "bce.is_click": {
            "estimate": 7.306345090196675e-05,
            "ci95": [
              2.158899767025492e-06,
              0.00013906114469521488
            ]
          },
          "bce.cart": {
            "estimate": 0.00030259963733332215,
            "ci95": [
              0.00018108541423140077,
              0.00042062076446789634
            ]
          },
          "bce.fav": {
            "estimate": 0.00011061178176723432,
            "ci95": [
              5.7355031410271345e-06,
              0.00022091037010628812
            ]
          },
          "bce.buy": {
            "estimate": 0.0006299102020536924,
            "ci95": [
              0.0005468867520025655,
              0.0007320647210362568
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.004397883352519227,
            "ci95": [
              -0.0008124980306285972,
              0.009664757153056125
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.002279461177090658,
            "ci95": [
              -0.002922908998581908,
              0.008582842622854681
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.030712047923238117,
            "ci95": [
              0.022698796418675875,
              0.040426901915632064
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.0021184221754285693,
            "ci95": [
              -0.009889719772763175,
              0.006014983510805009
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.02631416457071889,
            "ci95": [
              0.017048015677836877,
              0.037731410123387474
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.02843258674614746,
            "ci95": [
              0.018363492579814043,
              0.03966508847386031
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.0002295361864313554,
            "ci95": [
              -0.00038138587730040553,
              -8.739572821558602e-05
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -3.754833086526757e-05,
            "ci95": [
              -0.00017150915303453172,
              7.430463883173476e-05
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0005568467511517256,
            "ci95": [
              -0.0006795917358869694,
              -0.0004430113692166181
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.00019198785556608783,
            "ci95": [
              1.827642133567856e-05,
              0.00035441908907253263
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -0.0003273105647203702,
            "ci95": [
              -0.000493742283858282,
              -0.00018329783938343192
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.000519298420286458,
            "ci95": [
              -0.0006692872270253576,
              -0.0003827369278309978
            ]
          }
        }
      }
    },
    "parameter_count": 51724260,
    "initializer_sha256": "fd5f866c35c5e29b37841aeed070f6ef5259c773de173df7549d149f87a8c09e",
    "optimizer_layout_sha256": "c5e0357e309b56fe9a32ea136bb10053e17ee829010e8c16a98cedffc42669a0",
    "formal_worker_cost": {
      "wall_seconds": 2519.057643711567,
      "cpu_seconds": 3030.651342093,
      "peak_rss_bytes": 17107640320,
      "gpu_peak_bytes": 28373417984,
      "samples": 2309
    },
    "smoke_worker_cost": {
      "wall_seconds": 146.30966036766768,
      "cpu_seconds": 160.27972168399998,
      "peak_rss_bytes": 16942546944,
      "gpu_peak_bytes": 29572988928,
      "samples": 134
    },
    "complete_recorded_costs": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2519.057643711567,
        "cpu_seconds": 3030.651342093,
        "peak_rss_bytes": 17107640320,
        "gpu_peak_bytes": 28373417984,
        "samples": 2309
      },
      "formal_runner": {
        "wall_seconds": 2514.669528260827,
        "cpu_seconds": 2913.995312215,
        "peak_process_rss_bytes": 5292318720
      },
      "fit_including_validation": {
        "wall_seconds": 2389.772987000644,
        "cpu_seconds": 2740.770105793,
        "peak_process_rss_bytes": 5292318720
      },
      "prediction_export": {
        "wall_seconds": 119.00944044440985,
        "cpu_seconds": 167.03438005099997,
        "peak_process_rss_bytes": 5292318720
      },
      "smoke_worker": {
        "wall_seconds": 146.30966036766768,
        "cpu_seconds": 160.27972168399998,
        "peak_rss_bytes": 16942546944,
        "gpu_peak_bytes": 29572988928,
        "samples": 134
      },
      "smoke_runner": {
        "wall_seconds": 142.10140262544155,
        "cpu_seconds": 139.846790512,
        "peak_process_rss_bytes": 3240267776
      },
      "latency_constructor": {
        "wall_seconds": 0.7278218269348145,
        "cpu_seconds": 0.8003103280000019,
        "peak_rss_bytes": 2092183552
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.12073562294244766,
        "cpu_seconds": 0.11998605400000173,
        "peak_rss_bytes": 2092183552
      },
      "latency_peak_cuda_bytes": 1281155072,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "forward_latency": {
      "mean_ms": 14.242003202438354,
      "p50_ms": 14.230496406555176,
      "p95_ms": 14.282278728485107,
      "batch_size": 1024,
      "warmups": 10,
      "iterations": 40
    },
    "latency_ratio_to_parent": 1.0062006062828133,
    "quality_candidate": false,
    "near_candidate": false
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
  "decision": "discard",
  "decision_reason": "Macro delta -0.012856043,CI95 [-0.017204751230466074, -0.00938755160395674],latency ratio 1.006201. Fails positive and near-candidate quality policy; preserve negative evidence.",
  "decision_scope": "Root queue disposition only; no released-incumbent promotion or supported lesson.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_prefix",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_prefix/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_prefix/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_prefix/Taobao_Action/second_prefix.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/paired-analysis/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/latency/completion.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_prefix",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_prefix/result.json",
    "result_sha256": "cb5ac16cae01ec500e55e43976530414b4b8fe4863f39046eda353ac8c92fcfc",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_prefix/manifest.json",
    "manifest_sha256": "e8875b5e68c8e6c46ac40a236a466f8b4a7a845967d895a85ed17edce7cbcce8",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_prefix/Taobao_Action/second_prefix.model",
    "checkpoint_sha256": "bc9f00db12641808798208dcf5a9533c74183c5243ba7ce4796bc5f2af2903f7",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/paired-analysis/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
