## 250. segment-read--first_prefix--s20262027

- 假设：Equal-count first_prefix stored-history support may change useful two-round retrieval conditions; no pure Q2 mediation claim.
- 判定：discard
- 原因：Macro delta -0.003292829,CI95 [-0.006288188901790596, -0.0005409899799124383],latency ratio 1.005509. Fails positive and near-candidate quality policy; preserve negative evidence.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "segment-read--first_prefix--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-10T14:13:03.864962+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "first_prefix",
  "code_id": "cf7b0840f7d3f28bb5451ac3cc4d9491bfcf60c5a00b32fc75f78b7e407fc0de",
  "hypothesis": "Equal-count first_prefix stored-history support may change useful two-round retrieval conditions; no pure Q2 mediation claim.",
  "change": "{\"model\": \"QFormerCross8SegmentRead\", \"maturity_mode\": \"mature\", \"segment_placement\": \"first\", \"segment_selection\": \"prefix\", \"work_steps\": 2, \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8SegmentRead",
    "maturity_mode": "mature",
    "segment_placement": "first",
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
      "value": 0.7638021225717676,
      "delta": -0.003292828743092918
    },
    "per_task": {
      "is_click": {
        "logloss": 0.18419359121614262,
        "AUC": 0.6476268253973192
      },
      "cart": {
        "logloss": 0.021779045700050428,
        "AUC": 0.7992402625557546
      },
      "fav": {
        "logloss": 0.01700627460090522,
        "AUC": 0.790803198820176
      },
      "buy": {
        "logloss": 0.009816611669298652,
        "AUC": 0.8175382035138204
      }
    },
    "macro_source_logloss": 0.05819888079659923,
    "source_metric_comparison": {
      "is_click": {
        "source_AUC": 0.6476268253973192,
        "computed_AUC": 0.6476268253973192,
        "auc_difference": 0.0,
        "source_logloss": 0.18419359121614262,
        "computed_bce": 0.18419359121614728,
        "bce_difference": 4.6629367034256575e-15
      },
      "cart": {
        "source_AUC": 0.7992402625557546,
        "computed_AUC": 0.7992402625557546,
        "auc_difference": 0.0,
        "source_logloss": 0.021779045700050428,
        "computed_bce": 0.021779045721718644,
        "bce_difference": 2.1668215832715276e-11
      },
      "fav": {
        "source_AUC": 0.790803198820176,
        "computed_AUC": 0.790803198820176,
        "auc_difference": 0.0,
        "source_logloss": 0.01700627460090522,
        "computed_bce": 0.01700627467721258,
        "bce_difference": 7.630735973651603e-11
      },
      "buy": {
        "source_AUC": 0.8175382035138204,
        "computed_AUC": 0.8175382035138204,
        "auc_difference": 0.0,
        "source_logloss": 0.009816611669298652,
        "computed_bce": 0.009816611669753417,
        "bce_difference": 4.547646981212239e-13
      }
    },
    "matched_parent_effect": {
      "coefficients": {
        "mature_r2": -1.0,
        "cross8_mature": 0.0,
        "first_prefix": 1.0,
        "first_suffix": 0.0,
        "first_even": 0.0,
        "first_odd": 0.0,
        "second_prefix": 0.0,
        "second_suffix": 0.0,
        "second_even": 0.0,
        "second_odd": 0.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": -0.003292828743092918,
          "ci95": [
            -0.006288188901790596,
            -0.0005409899799124383
          ]
        },
        "auc.is_click": {
          "estimate": -0.002309467803670251,
          "ci95": [
            -0.003059841847618297,
            -0.0015223544009450086
          ]
        },
        "auc.cart": {
          "estimate": 0.006805992198410937,
          "ci95": [
            0.0017913172376513048,
            0.012303067344969902
          ]
        },
        "auc.fav": {
          "estimate": -0.011660291554872027,
          "ci95": [
            -0.017397424382386194,
            -0.006553139037782311
          ]
        },
        "auc.buy": {
          "estimate": -0.0060075478122403325,
          "ci95": [
            -0.015360740095926658,
            0.0016833545673150695
          ]
        },
        "bce.is_click": {
          "estimate": -0.00019805505163938975,
          "ci95": [
            -0.0002715809245434603,
            -0.000124807341399584
          ]
        },
        "bce.cart": {
          "estimate": -4.8908647715388553e-05,
          "ci95": [
            -0.00016548906680430905,
            6.336739247648778e-05
          ]
        },
        "bce.fav": {
          "estimate": 0.00020811698286024582,
          "ci95": [
            8.111822378528274e-05,
            0.00034519226737104384
          ]
        },
        "bce.buy": {
          "estimate": 0.00017280973302771438,
          "ci95": [
            8.994916474817909e-05,
            0.00026630732004686073
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": -0.009115460002081188,
          "ci95": [
            -0.014475947194097932,
            -0.0039040703955928887
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": 0.009350823751201776,
          "ci95": [
            0.0043600914271064,
            0.015102640211801728
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": 0.0036980800085700816,
          "ci95": [
            -0.004153893811178918,
            0.013012543538332082
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": 0.018466283753282964,
          "ci95": [
            0.011274894932288846,
            0.02685328050977635
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": 0.01281354001065127,
          "ci95": [
            0.003945256196879657,
            0.02333092889705349
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": -0.005652743742631694,
          "ci95": [
            -0.016013411590192008,
            0.004766407077576324
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": -0.0001491464039240012,
          "ci95": [
            -0.0002779469864404592,
            -4.503183835386911e-06
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": -0.00040617203449963557,
          "ci95": [
            -0.0005817519130803801,
            -0.00025791113945333207
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": -0.0003708647846671041,
          "ci95": [
            -0.00049935292763144,
            -0.0002566965192314924
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": -0.0002570256305756344,
          "ci95": [
            -0.00044499200047906423,
            -0.00010344764748612655
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": -0.00022171838074310293,
          "ci95": [
            -0.000369258833436694,
            -0.0001058221245301681
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": 3.5307249832531445e-05,
          "ci95": [
            -0.0001257884179505024,
            0.0002079927015158586
          ]
        }
      }
    },
    "all_prespecified_effects_involving_candidate": {
      "first_prefix__minus__mature_r2": {
        "coefficients": {
          "mature_r2": -1.0,
          "cross8_mature": 0.0,
          "first_prefix": 1.0,
          "first_suffix": 0.0,
          "first_even": 0.0,
          "first_odd": 0.0,
          "second_prefix": 0.0,
          "second_suffix": 0.0,
          "second_even": 0.0,
          "second_odd": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.003292828743092918,
            "ci95": [
              -0.006288188901790596,
              -0.0005409899799124383
            ]
          },
          "auc.is_click": {
            "estimate": -0.002309467803670251,
            "ci95": [
              -0.003059841847618297,
              -0.0015223544009450086
            ]
          },
          "auc.cart": {
            "estimate": 0.006805992198410937,
            "ci95": [
              0.0017913172376513048,
              0.012303067344969902
            ]
          },
          "auc.fav": {
            "estimate": -0.011660291554872027,
            "ci95": [
              -0.017397424382386194,
              -0.006553139037782311
            ]
          },
          "auc.buy": {
            "estimate": -0.0060075478122403325,
            "ci95": [
              -0.015360740095926658,
              0.0016833545673150695
            ]
          },
          "bce.is_click": {
            "estimate": -0.00019805505163938975,
            "ci95": [
              -0.0002715809245434603,
              -0.000124807341399584
            ]
          },
          "bce.cart": {
            "estimate": -4.8908647715388553e-05,
            "ci95": [
              -0.00016548906680430905,
              6.336739247648778e-05
            ]
          },
          "bce.fav": {
            "estimate": 0.00020811698286024582,
            "ci95": [
              8.111822378528274e-05,
              0.00034519226737104384
            ]
          },
          "bce.buy": {
            "estimate": 0.00017280973302771438,
            "ci95": [
              8.994916474817909e-05,
              0.00026630732004686073
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.009115460002081188,
            "ci95": [
              -0.014475947194097932,
              -0.0039040703955928887
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.009350823751201776,
            "ci95": [
              0.0043600914271064,
              0.015102640211801728
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.0036980800085700816,
            "ci95": [
              -0.004153893811178918,
              0.013012543538332082
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.018466283753282964,
            "ci95": [
              0.011274894932288846,
              0.02685328050977635
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.01281354001065127,
            "ci95": [
              0.003945256196879657,
              0.02333092889705349
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.005652743742631694,
            "ci95": [
              -0.016013411590192008,
              0.004766407077576324
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.0001491464039240012,
            "ci95": [
              -0.0002779469864404592,
              -4.503183835386911e-06
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.00040617203449963557,
            "ci95": [
              -0.0005817519130803801,
              -0.00025791113945333207
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0003708647846671041,
            "ci95": [
              -0.00049935292763144,
              -0.0002566965192314924
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.0002570256305756344,
            "ci95": [
              -0.00044499200047906423,
              -0.00010344764748612655
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -0.00022171838074310293,
            "ci95": [
              -0.000369258833436694,
              -0.0001058221245301681
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 3.5307249832531445e-05,
            "ci95": [
              -0.0001257884179505024,
              0.0002079927015158586
            ]
          }
        }
      },
      "first_prefix__minus__cross8_mature": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": -1.0,
          "first_prefix": 1.0,
          "first_suffix": 0.0,
          "first_even": 0.0,
          "first_odd": 0.0,
          "second_prefix": 0.0,
          "second_suffix": 0.0,
          "second_even": 0.0,
          "second_odd": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.004395702476307745,
            "ci95": [
              0.001024663269957546,
              0.007560991693365318
            ]
          },
          "auc.is_click": {
            "estimate": -0.002309944745646497,
            "ci95": [
              -0.0031409031276668164,
              -0.001478312202263929
            ]
          },
          "auc.cart": {
            "estimate": 0.01003352718463868,
            "ci95": [
              0.005006464038326627,
              0.015103586937381907
            ]
          },
          "auc.fav": {
            "estimate": -0.005698812207273529,
            "ci95": [
              -0.011041453990210726,
              -0.0004575422000380807
            ]
          },
          "auc.buy": {
            "estimate": 0.015558039673512325,
            "ci95": [
              0.006899631521434246,
              0.025133730900191737
            ]
          },
          "bce.is_click": {
            "estimate": -0.00012434663244406674,
            "ci95": [
              -0.00019991364033312975,
              -4.8857889424591706e-05
            ]
          },
          "bce.cart": {
            "estimate": -5.5365915360386375e-05,
            "ci95": [
              -0.00016571278311256098,
              5.8983628244852606e-05
            ]
          },
          "bce.fav": {
            "estimate": 3.2895495759532184e-05,
            "ci95": [
              -5.961956897176197e-05,
              0.0001271321705674657
            ]
          },
          "bce.buy": {
            "estimate": -0.00019077865868471176,
            "ci95": [
              -0.00029560341161680886,
              -9.822306691055793e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.012343471930285177,
            "ci95": [
              -0.017452152089407577,
              -0.007358408491752488
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.003388867461627032,
            "ci95": [
              -0.0018057704451814815,
              0.008788646925870051
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.017867984419158822,
            "ci95": [
              -0.027281207746926904,
              -0.008502001639138935
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.01573233939191221,
            "ci95": [
              0.008892542598626822,
              0.02333192879506812
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.005524512488873645,
            "ci95": [
              -0.016086602662199113,
              0.003920277832796369
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.021256851880785854,
            "ci95": [
              -0.0319440932118826,
              -0.011767162065289077
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -6.898071708368037e-05,
            "ci95": [
              -0.0001930997552335982,
              7.454472401410253e-05
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.00015724212820359892,
            "ci95": [
              -0.00028925304889780107,
              -3.39002214648672e-05
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 6.643202624064502e-05,
            "ci95": [
              -5.401626816306801e-05,
              0.00019629215352320216
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -8.826141111991856e-05,
            "ci95": [
              -0.00023392715479030496,
              4.883284711517082e-05
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.0001354127433243254,
            "ci95": [
              3.290490032062301e-06,
              0.00027175576936435986
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.00022367415444424395,
            "ci95": [
              0.00010352109006766681,
              0.00036070131953711536
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
      "first_prefix__minus__first_even": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "first_prefix": 1.0,
          "first_suffix": 0.0,
          "first_even": -1.0,
          "first_odd": 0.0,
          "second_prefix": 0.0,
          "second_suffix": 0.0,
          "second_even": 0.0,
          "second_odd": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.0013822623458151495,
            "ci95": [
              -0.0010283945236185838,
              0.003997605283987673
            ]
          },
          "auc.is_click": {
            "estimate": -0.0017479043475070455,
            "ci95": [
              -0.00256566909093599,
              -0.0008644326793436242
            ]
          },
          "auc.cart": {
            "estimate": 0.007003520182462308,
            "ci95": [
              0.0028191709875383273,
              0.011335071017800237
            ]
          },
          "auc.fav": {
            "estimate": -0.0035692710281214346,
            "ci95": [
              -0.009662081239672604,
              0.0020834717958607054
            ]
          },
          "auc.buy": {
            "estimate": 0.0038427045764267698,
            "ci95": [
              -0.002827381639495929,
              0.010673463120340631
            ]
          },
          "bce.is_click": {
            "estimate": -4.99271805518009e-05,
            "ci95": [
              -0.00013259607652832823,
              2.9649935138398974e-05
            ]
          },
          "bce.cart": {
            "estimate": -1.0928639867662654e-05,
            "ci95": [
              -0.00011772698432013983,
              9.093548772327336e-05
            ]
          },
          "bce.fav": {
            "estimate": -0.0002379585856152125,
            "ci95": [
              -0.00034187819148026756,
              -0.0001320826371594573
            ]
          },
          "bce.buy": {
            "estimate": -6.336487689531309e-05,
            "ci95": [
              -0.0001465699395798336,
              1.149515464712843e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.008751424529969354,
            "ci95": [
              -0.0132105001120119,
              -0.004222838493186424
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.0018213666806143891,
            "ci95": [
              -0.003909512384191618,
              0.00760967071819509
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.005590608923933815,
            "ci95": [
              -0.012414234754643103,
              0.001114778742709392
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.010572791210583743,
            "ci95": [
              0.0039674105826150085,
              0.01783438964009611
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.0031608156060355386,
            "ci95": [
              -0.0050249505742612334,
              0.011815204260044388
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.007411975604548204,
            "ci95": [
              -0.01748981763375493,
              0.001254402077236959
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -3.8998540684138244e-05,
            "ci95": [
              -0.00017371277034653292,
              9.055465161728377e-05
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.0001880314050634116,
            "ci95": [
              6.687671032229023e-05,
              0.0003093504562228586
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 1.343769634351219e-05,
            "ci95": [
              -9.457056910358004e-05,
              0.0001262084114413736
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.00022702994574754984,
            "ci95": [
              8.357630888133914e-05,
              0.0003623325910906981
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 5.2436237027650434e-05,
            "ci95": [
              -7.91173377282572e-05,
              0.00019044152583163186
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.0001745937087198994,
            "ci95": [
              -0.00029851893950953604,
              -2.4242125954031756e-05
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
      }
    },
    "parameter_count": 51724260,
    "initializer_sha256": "fd5f866c35c5e29b37841aeed070f6ef5259c773de173df7549d149f87a8c09e",
    "optimizer_layout_sha256": "cf6764857f1fdb70b39086008f5c8c5f85c5c9921f6a2422a0f5edc0470877c3",
    "formal_worker_cost": {
      "wall_seconds": 2515.3734920471907,
      "cpu_seconds": 3027.6236540930004,
      "peak_rss_bytes": 17074065408,
      "gpu_peak_bytes": 28369223680,
      "samples": 2307
    },
    "smoke_worker_cost": {
      "wall_seconds": 140.26956152170897,
      "cpu_seconds": 153.914924593,
      "peak_rss_bytes": 16939466752,
      "gpu_peak_bytes": 29570891776,
      "samples": 129
    },
    "complete_recorded_costs": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2515.3734920471907,
        "cpu_seconds": 3027.6236540930004,
        "peak_rss_bytes": 17074065408,
        "gpu_peak_bytes": 28369223680,
        "samples": 2307
      },
      "formal_runner": {
        "wall_seconds": 2511.4031151235104,
        "cpu_seconds": 2913.355100181,
        "peak_process_rss_bytes": 5250736128
      },
      "fit_including_validation": {
        "wall_seconds": 2386.5013393461704,
        "cpu_seconds": 2735.4487980989998,
        "peak_process_rss_bytes": 5250736128
      },
      "prediction_export": {
        "wall_seconds": 119.57211645692587,
        "cpu_seconds": 172.42904857799977,
        "peak_process_rss_bytes": 5250736128
      },
      "smoke_worker": {
        "wall_seconds": 140.26956152170897,
        "cpu_seconds": 153.914924593,
        "peak_rss_bytes": 16939466752,
        "gpu_peak_bytes": 29570891776,
        "samples": 129
      },
      "smoke_runner": {
        "wall_seconds": 135.68229084461927,
        "cpu_seconds": 134.544787276,
        "peak_process_rss_bytes": 3234557952
      },
      "latency_constructor": {
        "wall_seconds": 0.7381048575043678,
        "cpu_seconds": 0.8098489739999994,
        "peak_rss_bytes": 2092183552
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.13329923152923584,
        "cpu_seconds": 0.1324712789999989,
        "peak_rss_bytes": 2092183552
      },
      "latency_peak_cuda_bytes": 1281155072,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "forward_latency": {
      "mean_ms": 14.232216095924377,
      "p50_ms": 14.228447914123535,
      "p95_ms": 14.2561616897583,
      "batch_size": 1024,
      "warmups": 10,
      "iterations": 40
    },
    "latency_ratio_to_parent": 1.0055091450910034,
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
  "decision_reason": "Macro delta -0.003292829,CI95 [-0.006288188901790596, -0.0005409899799124383],latency ratio 1.005509. Fails positive and near-candidate quality policy; preserve negative evidence.",
  "decision_scope": "Root queue disposition only; no released-incumbent promotion or supported lesson.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_prefix",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_prefix/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_prefix/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_prefix/Taobao_Action/first_prefix.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/paired-analysis/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/latency/completion.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_prefix",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_prefix/result.json",
    "result_sha256": "233d9271226e854abc905c37ba8ab23eae8c21001e35a6124a1e0e3ef01be8d3",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_prefix/manifest.json",
    "manifest_sha256": "8ccd149a5dfe675c91f264895f0e3432a328cf0062ba2cf2ae0f326cbbe32a83",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_prefix/Taobao_Action/first_prefix.model",
    "checkpoint_sha256": "31c7b560cae02dd8468bb9229319db2a1925f1607b20269fb420608de73bbd04",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/paired-analysis/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
