## 242. value-content--event_constant--s20262027

- 假设：Second-work value supplementation; event_constant conditional nonlinear content before/after original weighted aggregation.
- 判定：discard
- 原因：Macro delta -0.004475306, CI95 [-0.007605460367732112, -0.0014428959967006203], forward ratio 1.114353; misses both frozen quality and near policy.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "value-content--event_constant--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-10T06:16:19.753374+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "event_constant",
  "code_id": "24e06fb09c6658ef53f419e5fa410cfa79eb699cc16354bfd31a299d62180a70",
  "hypothesis": "Second-work value supplementation; event_constant conditional nonlinear content before/after original weighted aggregation.",
  "change": "{\"model\": \"QFormerCross8ValueContent\", \"maturity_mode\": \"mature\", \"value_position\": \"event\", \"value_condition\": \"constant\", \"value_rank\": 16, \"work_steps\": 2, \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8ValueContent",
    "maturity_mode": "mature",
    "value_position": "event",
    "value_condition": "constant",
    "value_rank": 16,
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
      "value": 0.7626196453641493,
      "delta": -0.0044753059507111614
    },
    "per_task": {
      "is_click": {
        "logloss": 0.18427831550718302,
        "AUC": 0.6486161106391781
      },
      "cart": {
        "logloss": 0.021924896866166724,
        "AUC": 0.7876104270993859
      },
      "fav": {
        "logloss": 0.017015814707750758,
        "AUC": 0.7968399772809308
      },
      "buy": {
        "logloss": 0.009912036724534695,
        "AUC": 0.8174120664371025
      }
    },
    "macro_source_logloss": 0.0582827659514088,
    "source_metric_comparison": {
      "is_click": {
        "source_AUC": 0.6486161106391781,
        "computed_AUC": 0.6486161106391781,
        "auc_difference": 0.0,
        "source_logloss": 0.18427831550718302,
        "computed_bce": 0.18427831550718066,
        "bce_difference": -2.3592239273284576e-15
      },
      "cart": {
        "source_AUC": 0.7876104270993859,
        "computed_AUC": 0.7876104270993859,
        "auc_difference": 0.0,
        "source_logloss": 0.021924896866166724,
        "computed_bce": 0.021924896984976606,
        "bce_difference": 1.1880988207146892e-10
      },
      "fav": {
        "source_AUC": 0.7968399772809308,
        "computed_AUC": 0.7968399772809308,
        "auc_difference": 0.0,
        "source_logloss": 0.017015814707750758,
        "computed_bce": 0.017015814721846805,
        "bce_difference": 1.4096047246114907e-11
      },
      "buy": {
        "source_AUC": 0.8174120664371025,
        "computed_AUC": 0.8174120664371025,
        "auc_difference": 0.0,
        "source_logloss": 0.009912036724534695,
        "computed_bce": 0.009912037037840007,
        "bce_difference": 3.1330531224949e-10
      }
    },
    "matched_parent_effect": {
      "coefficients": {
        "mature_r2": -1.0,
        "cross8_mature": 0.0,
        "event_entry": 0.0,
        "event_work": 0.0,
        "event_constant": 1.0,
        "pooled_entry": 0.0,
        "pooled_work": 0.0,
        "pooled_constant": 0.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": -0.0044753059507111614,
          "ci95": [
            -0.007605460367732112,
            -0.0014428959967006203
          ]
        },
        "auc.is_click": {
          "estimate": -0.0013201825618113228,
          "ci95": [
            -0.0020545155679223244,
            -0.0006175207990985675
          ]
        },
        "auc.cart": {
          "estimate": -0.004823843257957816,
          "ci95": [
            -0.00931023886102195,
            0.00010201467214459857
          ]
        },
        "auc.fav": {
          "estimate": -0.005623513094117238,
          "ci95": [
            -0.011743941968709148,
            -0.0005126842484289411
          ]
        },
        "auc.buy": {
          "estimate": -0.0061336848889582685,
          "ci95": [
            -0.016132336140732343,
            0.0028630790449508683
          ]
        },
        "bce.is_click": {
          "estimate": -0.00011333076060601743,
          "ci95": [
            -0.00019041790206978938,
            -4.135961959938345e-05
          ]
        },
        "bce.cart": {
          "estimate": 9.694261554257441e-05,
          "ci95": [
            -1.2816631641411522e-05,
            0.00020779475620695455
          ]
        },
        "bce.fav": {
          "estimate": 0.00021765702749447155,
          "ci95": [
            0.00010678258242030466,
            0.00034244757912540064
          ]
        },
        "bce.buy": {
          "estimate": 0.0002682351011143048,
          "ci95": [
            0.00017183325703172748,
            0.0003620755818348181
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": 0.0035036606961464933,
          "ci95": [
            -0.0016600978566061186,
            0.008085776302840315
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": 0.004303330532305916,
          "ci95": [
            -0.00105394565849809,
            0.010204381354914343
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": 0.004813502327146946,
          "ci95": [
            -0.004114306466146946,
            0.015125141498387088
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": 0.0007996698361594223,
          "ci95": [
            -0.006864415942347956,
            0.009080515317980467
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": 0.0013098416310004524,
          "ci95": [
            -0.007415927087978887,
            0.011314352862775394
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": 0.00051017179484103,
          "ci95": [
            -0.009915068167383518,
            0.010840559161511592
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": -0.00021027337614859185,
          "ci95": [
            -0.00034548155358328183,
            -6.53170934635899e-05
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": -0.000330987788100489,
          "ci95": [
            -0.0004690171327903286,
            -0.00019420203993046807
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": -0.0003815658617203222,
          "ci95": [
            -0.000511681542677419,
            -0.0002695217556726128
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": -0.00012071441195189714,
          "ci95": [
            -0.000293191381594338,
            3.5242688222724015e-05
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": -0.00017129248557173037,
          "ci95": [
            -0.000313777606608151,
            -4.575288866936761e-05
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": -5.057807361983323e-05,
          "ci95": [
            -0.0001992204040240466,
            9.280370998433077e-05
          ]
        }
      }
    },
    "all_prespecified_effects_involving_candidate": {
      "event_constant__minus__mature_r2": {
        "coefficients": {
          "mature_r2": -1.0,
          "cross8_mature": 0.0,
          "event_entry": 0.0,
          "event_work": 0.0,
          "event_constant": 1.0,
          "pooled_entry": 0.0,
          "pooled_work": 0.0,
          "pooled_constant": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.0044753059507111614,
            "ci95": [
              -0.007605460367732112,
              -0.0014428959967006203
            ]
          },
          "auc.is_click": {
            "estimate": -0.0013201825618113228,
            "ci95": [
              -0.0020545155679223244,
              -0.0006175207990985675
            ]
          },
          "auc.cart": {
            "estimate": -0.004823843257957816,
            "ci95": [
              -0.00931023886102195,
              0.00010201467214459857
            ]
          },
          "auc.fav": {
            "estimate": -0.005623513094117238,
            "ci95": [
              -0.011743941968709148,
              -0.0005126842484289411
            ]
          },
          "auc.buy": {
            "estimate": -0.0061336848889582685,
            "ci95": [
              -0.016132336140732343,
              0.0028630790449508683
            ]
          },
          "bce.is_click": {
            "estimate": -0.00011333076060601743,
            "ci95": [
              -0.00019041790206978938,
              -4.135961959938345e-05
            ]
          },
          "bce.cart": {
            "estimate": 9.694261554257441e-05,
            "ci95": [
              -1.2816631641411522e-05,
              0.00020779475620695455
            ]
          },
          "bce.fav": {
            "estimate": 0.00021765702749447155,
            "ci95": [
              0.00010678258242030466,
              0.00034244757912540064
            ]
          },
          "bce.buy": {
            "estimate": 0.0002682351011143048,
            "ci95": [
              0.00017183325703172748,
              0.0003620755818348181
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.0035036606961464933,
            "ci95": [
              -0.0016600978566061186,
              0.008085776302840315
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.004303330532305916,
            "ci95": [
              -0.00105394565849809,
              0.010204381354914343
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.004813502327146946,
            "ci95": [
              -0.004114306466146946,
              0.015125141498387088
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.0007996698361594223,
            "ci95": [
              -0.006864415942347956,
              0.009080515317980467
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.0013098416310004524,
            "ci95": [
              -0.007415927087978887,
              0.011314352862775394
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.00051017179484103,
            "ci95": [
              -0.009915068167383518,
              0.010840559161511592
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00021027337614859185,
            "ci95": [
              -0.00034548155358328183,
              -6.53170934635899e-05
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.000330987788100489,
            "ci95": [
              -0.0004690171327903286,
              -0.00019420203993046807
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0003815658617203222,
            "ci95": [
              -0.000511681542677419,
              -0.0002695217556726128
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.00012071441195189714,
            "ci95": [
              -0.000293191381594338,
              3.5242688222724015e-05
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -0.00017129248557173037,
            "ci95": [
              -0.000313777606608151,
              -4.575288866936761e-05
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -5.057807361983323e-05,
            "ci95": [
              -0.0001992204040240466,
              9.280370998433077e-05
            ]
          }
        }
      },
      "event_constant__minus__cross8_mature": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": -1.0,
          "event_entry": 0.0,
          "event_work": 0.0,
          "event_constant": 1.0,
          "pooled_entry": 0.0,
          "pooled_work": 0.0,
          "pooled_constant": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.0032132252686895013,
            "ci95": [
              7.870933662974407e-05,
              0.006063728247578188
            ]
          },
          "auc.is_click": {
            "estimate": -0.001320659503787569,
            "ci95": [
              -0.002180700933262808,
              -0.0005715642396867979
            ]
          },
          "auc.cart": {
            "estimate": -0.0015963082717300736,
            "ci95": [
              -0.00623050332463852,
              0.002834586291552365
            ]
          },
          "auc.fav": {
            "estimate": 0.00033796625348125886,
            "ci95": [
              -0.005269625939863923,
              0.006394461104477199
            ]
          },
          "auc.buy": {
            "estimate": 0.015431902596794389,
            "ci95": [
              0.006968075646771744,
              0.023385519986924605
            ]
          },
          "bce.is_click": {
            "estimate": -3.962234141069443e-05,
            "ci95": [
              -0.00010944495033844564,
              4.1734975154869564e-05
            ]
          },
          "bce.cart": {
            "estimate": 9.048534789757659e-05,
            "ci95": [
              -3.76664211900345e-05,
              0.000240024436780808
            ]
          },
          "bce.fav": {
            "estimate": 4.243554039375791e-05,
            "ci95": [
              -6.09977066228385e-05,
              0.0001468517223459078
            ]
          },
          "bce.buy": {
            "estimate": -9.535329059812137e-05,
            "ci95": [
              -0.00019245083668864658,
              2.9834295611003167e-06
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.0002756487679425046,
            "ci95": [
              -0.0043468882552379655,
              0.005070176496231246
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.0016586257572688279,
            "ci95": [
              -0.007861177725987139,
              0.003849633943600393
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.016752562100581958,
            "ci95": [
              -0.025175759960618837,
              -0.008328107460166151
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.0019342745252113325,
            "ci95": [
              -0.008716261923322834,
              0.005477667217123055
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.017028210868524463,
            "ci95": [
              -0.0260055296175188,
              -0.007801776086490496
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.01509393634331313,
            "ci95": [
              -0.024518361019991018,
              -0.005622794505105432
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00013010768930827102,
            "ci95": [
              -0.0002909535124873357,
              1.7598730157320222e-05
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -8.205788180445234e-05,
            "ci95": [
              -0.0002100383713697935,
              5.125105998813613e-05
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 5.573094918742694e-05,
            "ci95": [
              -5.7921055588124046e-05,
              0.0001881606474610231
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 4.804980750381868e-05,
            "ci95": [
              -0.00011840605479749482,
              0.00021760439028210648
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00018583863849569796,
            "ci95": [
              3.0651899589006485e-05,
              0.00036317680598217775
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.00013778883099187927,
            "ci95": [
              -1.3813684782643304e-07,
              0.0002790051636124355
            ]
          }
        }
      },
      "event_work__minus__event_constant": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "event_entry": 0.0,
          "event_work": 1.0,
          "event_constant": -1.0,
          "pooled_entry": 0.0,
          "pooled_work": 0.0,
          "pooled_constant": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.0015700970212505772,
            "ci95": [
              -0.0012037464265058318,
              0.004583983959190252
            ]
          },
          "auc.is_click": {
            "estimate": 0.0007573467187041993,
            "ci95": [
              -6.28665615653584e-05,
              0.0015928189010475902
            ]
          },
          "auc.cart": {
            "estimate": 0.0029019072810178015,
            "ci95": [
              -0.002514743047051299,
              0.008309245230365296
            ]
          },
          "auc.fav": {
            "estimate": 0.0076995324086515105,
            "ci95": [
              0.0023605254489027145,
              0.01357021278613473
            ]
          },
          "auc.buy": {
            "estimate": -0.005078398323371203,
            "ci95": [
              -0.013385526487487864,
              0.003557524633008627
            ]
          },
          "bce.is_click": {
            "estimate": 2.2340652098357117e-05,
            "ci95": [
              -5.727493681433535e-05,
              9.746491189857942e-05
            ]
          },
          "bce.cart": {
            "estimate": -2.1991507738280958e-05,
            "ci95": [
              -0.00014723621106387026,
              8.922799681134494e-05
            ]
          },
          "bce.fav": {
            "estimate": -0.00036615260385347917,
            "ci95": [
              -0.0004817332066619281,
              -0.00026530931132757844
            ]
          },
          "bce.buy": {
            "estimate": -0.00012461515106612642,
            "ci95": [
              -0.00022337110553145387,
              -3.326556713173118e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.002144560562313602,
            "ci95": [
              -0.0074993970581742455,
              0.003154012746867645
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.006942185689947311,
            "ci95": [
              -0.012765195693829197,
              -0.0015175104914806838
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.005835745042075402,
            "ci95": [
              -0.0024683428661737366,
              0.014207408473965493
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.004797625127633709,
            "ci95": [
              -0.012909817912614053,
              0.002626915298669454
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.007980305604389004,
            "ci95": [
              -0.0018996430549236688,
              0.017098681276080673
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.012777930732022713,
            "ci95": [
              0.0030170254842271428,
              0.021772797337066867
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 4.4332159836638074e-05,
            "ci95": [
              -9.815583392772803e-05,
              0.00017672989864692218
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.0003884932559518363,
            "ci95": [
              0.00026067949422588086,
              0.0005177605257428507
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.00014695580316448353,
            "ci95": [
              2.7091115892173772e-05,
              0.0002648742976510261
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.0003441610961151982,
            "ci95": [
              0.00018299192397680988,
              0.0005107145297403943
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00010262364332784546,
            "ci95": [
              -4.756996251425011e-05,
              0.0002560096115763925
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.00024153745278735275,
            "ci95": [
              -0.0003689590487784947,
              -0.00011406820898857355
            ]
          }
        }
      },
      "event_entry__minus__event_constant": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "event_entry": 1.0,
          "event_work": 0.0,
          "event_constant": -1.0,
          "pooled_entry": 0.0,
          "pooled_work": 0.0,
          "pooled_constant": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.0001359460844422844,
            "ci95": [
              -0.0031499835696204397,
              0.003429056158559577
            ]
          },
          "auc.is_click": {
            "estimate": 0.0017491599501973099,
            "ci95": [
              0.0009028538689436677,
              0.002564257196811317
            ]
          },
          "auc.cart": {
            "estimate": -0.0034824112184441702,
            "ci95": [
              -0.009495343923423393,
              0.0037158903630673144
            ]
          },
          "auc.fav": {
            "estimate": 0.00667505440947036,
            "ci95": [
              0.0018352892037827306,
              0.01222210798613629
            ]
          },
          "auc.buy": {
            "estimate": -0.004398018803454362,
            "ci95": [
              -0.013999154663998084,
              0.00562398795940428
            ]
          },
          "bce.is_click": {
            "estimate": -0.00011470660157048007,
            "ci95": [
              -0.00019352321188956313,
              -3.424355043041423e-05
            ]
          },
          "bce.cart": {
            "estimate": 0.00013962889310246626,
            "ci95": [
              -2.2800212042932758e-05,
              0.00026850958420151757
            ]
          },
          "bce.fav": {
            "estimate": -0.0002306023472838821,
            "ci95": [
              -0.0003291264013082384,
              -0.00013339491360410833
            ]
          },
          "bce.buy": {
            "estimate": -0.0002350721550273549,
            "ci95": [
              -0.0003462595177033317,
              -0.00012443562248458923
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.00523157116864148,
            "ci95": [
              -0.0019439986729692436,
              0.01114805982287453
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.00492589445927305,
            "ci95": [
              -0.010372527779704318,
              0.00027790290594555844
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.006147178753651672,
            "ci95": [
              -0.0037189913923527263,
              0.015896698012413647
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.01015746562791453,
            "ci95": [
              -0.018661395823175938,
              -0.0020106507571443035
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.0009156075850101919,
            "ci95": [
              -0.010385155519854968,
              0.012236035416783263
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.011073073212924722,
            "ci95": [
              0.0011008577495754516,
              0.0212542276397325
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00025433549467294633,
            "ci95": [
              -0.0003973313866089146,
              -8.007446378345972e-05
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.00011589574571340203,
            "ci95": [
              -7.295629429033331e-06,
              0.00023755840082066566
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.00012036555345687483,
            "ci95": [
              -1.270943024106524e-05,
              0.0002656638393092303
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.00037023124038634836,
            "ci95": [
              0.00018972738989587413,
              0.0005464159811993922
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00037470104812982116,
            "ci95": [
              0.00019711536383447208,
              0.0005558048002388228
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 4.469807743472803e-06,
            "ci95": [
              -0.00014097977129642019,
              0.00014458234368330825
            ]
          }
        }
      },
      "event_constant__minus__pooled_constant": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "event_entry": 0.0,
          "event_work": 0.0,
          "event_constant": 1.0,
          "pooled_entry": 0.0,
          "pooled_work": 0.0,
          "pooled_constant": -1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.00025180421395237795,
            "ci95": [
              -0.0029954776395498214,
              0.0024113440355742212
            ]
          },
          "auc.is_click": {
            "estimate": -0.00036261017189942457,
            "ci95": [
              -0.001204460403404023,
              0.0004541129405169484
            ]
          },
          "auc.cart": {
            "estimate": -0.0026232904168230764,
            "ci95": [
              -0.007784982009576738,
              0.0022129506811554896
            ]
          },
          "auc.fav": {
            "estimate": -0.007176534468365969,
            "ci95": [
              -0.01290605647249353,
              -0.001198030453358779
            ]
          },
          "auc.buy": {
            "estimate": 0.009155218201278958,
            "ci95": [
              0.003360643318409079,
              0.015074213415986403
            ]
          },
          "bce.is_click": {
            "estimate": -8.25878002652658e-06,
            "ci95": [
              -8.219178077217281e-05,
              7.14883278681029e-05
            ]
          },
          "bce.cart": {
            "estimate": -8.399051095163423e-05,
            "ci95": [
              -0.00020023333117741613,
              4.4975476812745966e-05
            ]
          },
          "bce.fav": {
            "estimate": 0.00017158834138939622,
            "ci95": [
              6.18915433318607e-05,
              0.0002737970716519975
            ]
          },
          "bce.buy": {
            "estimate": 6.463249922088535e-05,
            "ci95": [
              -8.694135214013832e-06,
              0.0001419208854116593
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.002260680244923652,
            "ci95": [
              -0.002632639009839413,
              0.007713893652079347
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.006813924296466545,
            "ci95": [
              0.0007364360215334962,
              0.01242633393337315
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.009517828373178383,
            "ci95": [
              -0.015539235638789792,
              -0.00393216843347357
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.004553244051542893,
            "ci95": [
              -0.0029590849402455627,
              0.01168784207975763
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.011778508618102035,
            "ci95": [
              -0.019022092863581027,
              -0.0034682293413243162
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.016331752669644928,
            "ci95": [
              -0.023399269415326458,
              -0.00864582247176971
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 7.573173092510765e-05,
            "ci95": [
              -8.344323253162326e-05,
              0.00021545722574128852
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0001798471214159228,
            "ci95": [
              -0.0003158866489188897,
              -3.773717156397079e-05
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -7.289127924741193e-05,
            "ci95": [
              -0.00017999178847246303,
              3.718250157276669e-05
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.00025557885234103045,
            "ci95": [
              -0.00041011730140790795,
              -7.841781496916532e-05
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -0.00014862301017251958,
            "ci95": [
              -0.00029319008539428434,
              -4.2401175398655987e-07
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.00010695584216851087,
            "ci95": [
              -2.181020963889883e-05,
              0.00023079173973156376
            ]
          }
        }
      }
    },
    "parameter_count": 51742692,
    "initializer_sha256": "aa3b2ab7b9014c0f0183713dca1e9fcc087775a3c1683e2efaa4346181bd573d",
    "optimizer_layout_sha256": "be964b342c7b3ca1b2588d7dd6a250d0f6dd626ae2da9218d03c81836a20cd75",
    "formal_worker_cost": {
      "wall_seconds": 2627.9050848707557,
      "cpu_seconds": 3127.0364961640003,
      "peak_rss_bytes": 17082970112,
      "gpu_peak_bytes": 30260854784,
      "samples": 2412
    },
    "smoke_worker_cost": {
      "wall_seconds": 196.88931387662888,
      "cpu_seconds": 252.435959932,
      "peak_rss_bytes": 19725701120,
      "gpu_peak_bytes": 32792117248,
      "samples": 180
    },
    "complete_recorded_costs": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2627.9050848707557,
        "cpu_seconds": 3127.0364961640003,
        "peak_rss_bytes": 17082970112,
        "gpu_peak_bytes": 30260854784,
        "samples": 2412
      },
      "formal_runner": {
        "wall_seconds": 2624.0310389623046,
        "cpu_seconds": 3009.6277336079997,
        "peak_process_rss_bytes": 5293260800
      },
      "fit_including_validation": {
        "wall_seconds": 2495.8989638984203,
        "cpu_seconds": 2834.030955709,
        "peak_process_rss_bytes": 5293260800
      },
      "prediction_export": {
        "wall_seconds": 123.08756574243307,
        "cpu_seconds": 170.1786524529998,
        "peak_process_rss_bytes": 5293260800
      },
      "smoke_worker": {
        "wall_seconds": 196.88931387662888,
        "cpu_seconds": 252.435959932,
        "peak_rss_bytes": 19725701120,
        "gpu_peak_bytes": 32792117248,
        "samples": 180
      },
      "smoke_runner": {
        "wall_seconds": 193.41917557269335,
        "cpu_seconds": 176.10555503199998,
        "peak_process_rss_bytes": 3283312640
      },
      "latency_constructor": {
        "wall_seconds": 0.7622329965233803,
        "cpu_seconds": 0.8374763460000008,
        "peak_rss_bytes": 2134814720
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.1455247551202774,
        "cpu_seconds": 0.14504931600000148,
        "peak_rss_bytes": 2134814720
      },
      "latency_peak_cuda_bytes": 1284208128,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "forward_latency": {
      "mean_ms": 15.909326362609864,
      "p50_ms": 15.892608165740967,
      "p95_ms": 15.959024047851562,
      "batch_size": 1024,
      "warmups": 10,
      "iterations": 40
    },
    "latency_ratio_to_parent": 1.1143532575688595,
    "quality_candidate": false,
    "near_candidate": false
  },
  "validity": {
    "valid": true,
    "violations": [],
    "state": "passed",
    "quality_summary": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/quality-summary.json",
    "no_prediction_to_work_feedback": true,
    "test_evaluated": false,
    "source_and_input_identity_scope": "Original N full-input content SHA256 values are inherited through completed C quality and E authorization. Production runners checked input stats before/after their traversals and two original metadata hashes before them. This collector now checks 122 current stats and both original metadata hashes; it does not rehash Parquet or open checkpoint/prediction arrays. New E worker/smoke/effective-optimizer gates are evaluated from actual small receipts. Three references retain exact C quality records and producer completion hashes, with small source/artifact identity checks. Stat continuity cannot independently exclude content changes that preserve stat fields."
  },
  "decision": "discard",
  "decision_reason": "Macro delta -0.004475306, CI95 [-0.007605460367732112, -0.0014428959967006203], forward ratio 1.114353; misses both frozen quality and near policy.",
  "decision_scope": "Root queue disposition only; no released-incumbent promotion or supported lesson.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/event_constant",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/event_constant/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/event_constant/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/event_constant/Taobao_Action/event_constant.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/paired-analysis/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/latency/completion.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/event_constant",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/event_constant/result.json",
    "result_sha256": "00600836e75b40c5bd45a68aa2a868a7b161e656f8902dc1c7c1a4e913ef4c83",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/event_constant/manifest.json",
    "manifest_sha256": "a5a88e37afc2a87bb1a20dfd2425cfb7f03d5ea06a51de88951d4c1fb0ca5568",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/event_constant/Taobao_Action/event_constant.model",
    "checkpoint_sha256": "b200a40f26f33f54d54d60eb064ffb92245b6141a49714ff199674753993e719",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/paired-analysis/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
