## 243. value-content--pooled_entry--s20262027

- 假设：Second-work value supplementation; pooled_entry conditional nonlinear content before/after original weighted aggregation.
- 判定：discard
- 原因：Macro delta -0.003351628, CI95 [-0.00633145636213481, -0.0003471266979163905], forward ratio 0.999264; misses both frozen quality and near policy.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "value-content--pooled_entry--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-10T06:16:19.753374+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "pooled_entry",
  "code_id": "24e06fb09c6658ef53f419e5fa410cfa79eb699cc16354bfd31a299d62180a70",
  "hypothesis": "Second-work value supplementation; pooled_entry conditional nonlinear content before/after original weighted aggregation.",
  "change": "{\"model\": \"QFormerCross8ValueContent\", \"maturity_mode\": \"mature\", \"value_position\": \"pooled\", \"value_condition\": \"entry\", \"value_rank\": 16, \"work_steps\": 2, \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8ValueContent",
    "maturity_mode": "mature",
    "value_position": "pooled",
    "value_condition": "entry",
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
      "value": 0.763743323786078,
      "delta": -0.003351627528782458
    },
    "per_task": {
      "is_click": {
        "logloss": 0.1842901101028526,
        "AUC": 0.6506573597832078
      },
      "cart": {
        "logloss": 0.021888156488502356,
        "AUC": 0.7914855486394777
      },
      "fav": {
        "logloss": 0.017092205863364237,
        "AUC": 0.7836411566524256
      },
      "buy": {
        "logloss": 0.00965514973163172,
        "AUC": 0.829189230069201
      }
    },
    "macro_source_logloss": 0.05823140554658773,
    "source_metric_comparison": {
      "is_click": {
        "source_AUC": 0.6506573597832078,
        "computed_AUC": 0.6506573597832078,
        "auc_difference": 0.0,
        "source_logloss": 0.1842901101028526,
        "computed_bce": 0.1842901101028564,
        "bce_difference": 3.802513859341161e-15
      },
      "cart": {
        "source_AUC": 0.7914855486394777,
        "computed_AUC": 0.7914855486394777,
        "auc_difference": 0.0,
        "source_logloss": 0.021888156488502356,
        "computed_bce": 0.02188815653914788,
        "bce_difference": 5.064552596745209e-11
      },
      "fav": {
        "source_AUC": 0.7836411566524256,
        "computed_AUC": 0.7836411566524256,
        "auc_difference": 0.0,
        "source_logloss": 0.017092205863364237,
        "computed_bce": 0.01709220595547122,
        "bce_difference": 9.210698176387311e-11
      },
      "buy": {
        "source_AUC": 0.829189230069201,
        "computed_AUC": 0.829189230069201,
        "auc_difference": 0.0,
        "source_logloss": 0.00965514973163172,
        "computed_bce": 0.00965514974599314,
        "bce_difference": 1.436142003929941e-11
      }
    },
    "matched_parent_effect": {
      "coefficients": {
        "mature_r2": -1.0,
        "cross8_mature": 0.0,
        "event_entry": 0.0,
        "event_work": 0.0,
        "event_constant": 0.0,
        "pooled_entry": 1.0,
        "pooled_work": 0.0,
        "pooled_constant": 0.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": -0.003351627528782458,
          "ci95": [
            -0.00633145636213481,
            -0.0003471266979163905
          ]
        },
        "auc.is_click": {
          "estimate": 0.000721066582218377,
          "ci95": [
            5.109020975223866e-05,
            0.0014588073040384948
          ]
        },
        "auc.cart": {
          "estimate": -0.0009487217178659746,
          "ci95": [
            -0.005139922031891004,
            0.0042049899840784135
          ]
        },
        "auc.fav": {
          "estimate": -0.01882233372262243,
          "ci95": [
            -0.02632636091621869,
            -0.012117723933052988
          ]
        },
        "auc.buy": {
          "estimate": 0.005643478743140196,
          "ci95": [
            -0.002289177690026836,
            0.013200495681185746
          ]
        },
        "bce.is_click": {
          "estimate": -0.00010153616493027706,
          "ci95": [
            -0.000170336260195713,
            -3.7911647649774315e-05
          ]
        },
        "bce.cart": {
          "estimate": 6.0202169713849396e-05,
          "ci95": [
            -5.9563595393488566e-05,
            0.0001598750073741966
          ]
        },
        "bce.fav": {
          "estimate": 0.0002940482611188851,
          "ci95": [
            0.00014956750388538456,
            0.0004569091792165916
          ]
        },
        "bce.buy": {
          "estimate": 1.1347809267437878e-05,
          "ci95": [
            -8.540321812001488e-05,
            0.00011794848666654985
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": 0.0016697883000843516,
          "ci95": [
            -0.0035117086762521234,
            0.006184232003831778
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": 0.019543400304840808,
          "ci95": [
            0.012871300853817255,
            0.027393015877771935
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": -0.0049224121609218185,
          "ci95": [
            -0.012531230817464722,
            0.003137840268406988
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": 0.017873612004756456,
          "ci95": [
            0.00980161403047417,
            0.02659594604405051
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": -0.00659220046100617,
          "ci95": [
            -0.015148107057738242,
            0.0026525589072220147
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": -0.024465812465762626,
          "ci95": [
            -0.035248031102740635,
            -0.014243677753847887
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": -0.00016173833464412646,
          "ci95": [
            -0.00028798436678460624,
            -3.343119040393796e-05
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": -0.00039558442604916214,
          "ci95": [
            -0.0005638519449314757,
            -0.00023801565249220247
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": -0.00011288397419771494,
          "ci95": [
            -0.0002272334826066733,
            3.1189534364235328e-06
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": -0.00023384609140503568,
          "ci95": [
            -0.0004259139062935842,
            -6.254857004072305e-05
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": 4.885436044641152e-05,
          "ci95": [
            -0.0001090973443140767,
            0.000178544295199401
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": 0.0002827004518514472,
          "ci95": [
            0.00011705926195334253,
            0.0004676862401687823
          ]
        }
      }
    },
    "all_prespecified_effects_involving_candidate": {
      "pooled_entry__minus__mature_r2": {
        "coefficients": {
          "mature_r2": -1.0,
          "cross8_mature": 0.0,
          "event_entry": 0.0,
          "event_work": 0.0,
          "event_constant": 0.0,
          "pooled_entry": 1.0,
          "pooled_work": 0.0,
          "pooled_constant": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.003351627528782458,
            "ci95": [
              -0.00633145636213481,
              -0.0003471266979163905
            ]
          },
          "auc.is_click": {
            "estimate": 0.000721066582218377,
            "ci95": [
              5.109020975223866e-05,
              0.0014588073040384948
            ]
          },
          "auc.cart": {
            "estimate": -0.0009487217178659746,
            "ci95": [
              -0.005139922031891004,
              0.0042049899840784135
            ]
          },
          "auc.fav": {
            "estimate": -0.01882233372262243,
            "ci95": [
              -0.02632636091621869,
              -0.012117723933052988
            ]
          },
          "auc.buy": {
            "estimate": 0.005643478743140196,
            "ci95": [
              -0.002289177690026836,
              0.013200495681185746
            ]
          },
          "bce.is_click": {
            "estimate": -0.00010153616493027706,
            "ci95": [
              -0.000170336260195713,
              -3.7911647649774315e-05
            ]
          },
          "bce.cart": {
            "estimate": 6.0202169713849396e-05,
            "ci95": [
              -5.9563595393488566e-05,
              0.0001598750073741966
            ]
          },
          "bce.fav": {
            "estimate": 0.0002940482611188851,
            "ci95": [
              0.00014956750388538456,
              0.0004569091792165916
            ]
          },
          "bce.buy": {
            "estimate": 1.1347809267437878e-05,
            "ci95": [
              -8.540321812001488e-05,
              0.00011794848666654985
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.0016697883000843516,
            "ci95": [
              -0.0035117086762521234,
              0.006184232003831778
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.019543400304840808,
            "ci95": [
              0.012871300853817255,
              0.027393015877771935
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.0049224121609218185,
            "ci95": [
              -0.012531230817464722,
              0.003137840268406988
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.017873612004756456,
            "ci95": [
              0.00980161403047417,
              0.02659594604405051
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.00659220046100617,
            "ci95": [
              -0.015148107057738242,
              0.0026525589072220147
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.024465812465762626,
            "ci95": [
              -0.035248031102740635,
              -0.014243677753847887
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00016173833464412646,
            "ci95": [
              -0.00028798436678460624,
              -3.343119040393796e-05
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.00039558442604916214,
            "ci95": [
              -0.0005638519449314757,
              -0.00023801565249220247
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.00011288397419771494,
            "ci95": [
              -0.0002272334826066733,
              3.1189534364235328e-06
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.00023384609140503568,
            "ci95": [
              -0.0004259139062935842,
              -6.254857004072305e-05
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 4.885436044641152e-05,
            "ci95": [
              -0.0001090973443140767,
              0.000178544295199401
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.0002827004518514472,
            "ci95": [
              0.00011705926195334253,
              0.0004676862401687823
            ]
          }
        }
      },
      "pooled_entry__minus__cross8_mature": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": -1.0,
          "event_entry": 0.0,
          "event_work": 0.0,
          "event_constant": 0.0,
          "pooled_entry": 1.0,
          "pooled_work": 0.0,
          "pooled_constant": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.004336903690618205,
            "ci95": [
              0.0012580335331483553,
              0.007782453669999475
            ]
          },
          "auc.is_click": {
            "estimate": 0.0007205896402421308,
            "ci95": [
              -5.802558799951618e-05,
              0.0013952382098605554
            ]
          },
          "auc.cart": {
            "estimate": 0.002278813268361768,
            "ci95": [
              -0.0016180329748819028,
              0.006439198292715352
            ]
          },
          "auc.fav": {
            "estimate": -0.012860854375023933,
            "ci95": [
              -0.02015615154092133,
              -0.005492352552031893
            ]
          },
          "auc.buy": {
            "estimate": 0.027209066228892853,
            "ci95": [
              0.01795269170572262,
              0.037889441228172636
            ]
          },
          "bce.is_click": {
            "estimate": -2.7827745734954057e-05,
            "ci95": [
              -9.090017183610646e-05,
              4.172190454757859e-05
            ]
          },
          "bce.cart": {
            "estimate": 5.3744902068851574e-05,
            "ci95": [
              -2.6639310588234385e-05,
              0.00013861173274374393
            ]
          },
          "bce.fav": {
            "estimate": 0.00011882677401817143,
            "ci95": [
              -1.0395048222625133e-05,
              0.00024781481353907455
            ]
          },
          "bce.buy": {
            "estimate": -0.00035224058244498826,
            "ci95": [
              -0.00046762663857538366,
              -0.00024216834310260166
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.001558223628119637,
            "ci95": [
              -0.0058135677269042,
              0.0022078585882238234
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.013581444015266064,
            "ci95": [
              0.006028382017492365,
              0.02073345338131549
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.026488476588650722,
            "ci95": [
              -0.037311626366121994,
              -0.017077743296846314
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.015139667643385701,
            "ci95": [
              0.00720963489807116,
              0.0234311044850108
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.024930252960531085,
            "ci95": [
              -0.036500024964371304,
              -0.014299718730019482
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.04006992060391679,
            "ci95": [
              -0.053443459041624064,
              -0.02869554306493801
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -8.157264780380563e-05,
            "ci95": [
              -0.0001815764945086975,
              2.9880719147724947e-05
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0001466545197531255,
            "ci95": [
              -0.00029920812879908546,
              6.264042692139199e-07
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.0003244128367100342,
            "ci95": [
              0.00020441311924345799,
              0.000467066557138585
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -6.508187194931986e-05,
            "ci95": [
              -0.00021880421196710615,
              8.987369249997763e-05
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00040598548451383984,
            "ci95": [
              0.0002793312503093396,
              0.0005601149563422373
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.0004710673564631597,
            "ci95": [
              0.0002947109898508662,
              0.0006552668136883812
            ]
          }
        }
      },
      "pooled_work__minus__pooled_entry": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "event_entry": 0.0,
          "event_work": 0.0,
          "event_constant": 0.0,
          "pooled_entry": -1.0,
          "pooled_work": 1.0,
          "pooled_constant": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.00386556188181042,
            "ci95": [
              -0.006989559898956577,
              -0.0008467517321487585
            ]
          },
          "auc.is_click": {
            "estimate": -0.003468739326984327,
            "ci95": [
              -0.004281777450623603,
              -0.002756124101644869
            ]
          },
          "auc.cart": {
            "estimate": -0.001540574908289627,
            "ci95": [
              -0.0072056957063961116,
              0.005018617784339542
            ]
          },
          "auc.fav": {
            "estimate": 0.008370189933013306,
            "ci95": [
              0.0017383458269289085,
              0.015190857701649104
            ]
          },
          "auc.buy": {
            "estimate": -0.01882312322498103,
            "ci95": [
              -0.026850979587958256,
              -0.010905905847206047
            ]
          },
          "bce.is_click": {
            "estimate": 0.0002752153904525545,
            "ci95": [
              0.00020894379090919332,
              0.0003448228221252826
            ]
          },
          "bce.cart": {
            "estimate": 5.8161762999803246e-05,
            "ci95": [
              -6.529737403775936e-05,
              0.00018738467753308318
            ]
          },
          "bce.fav": {
            "estimate": -0.00017260103768727483,
            "ci95": [
              -0.00028274423578069903,
              -5.075273682823007e-05
            ]
          },
          "bce.buy": {
            "estimate": 0.00019929114107548537,
            "ci95": [
              9.441247418264537e-05,
              0.000297522768416555
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.0019281644186947,
            "ci95": [
              -0.008762390991848645,
              0.003703825566812497
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.011838929259997633,
            "ci95": [
              -0.018700724414216337,
              -0.00502284902050254
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.015354383897996704,
            "ci95": [
              0.007466763676911276,
              0.023303692383187698
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.009910764841302933,
            "ci95": [
              -0.018982147265086163,
              -0.001041274482056235
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.017282548316691404,
            "ci95": [
              0.007243285993760398,
              0.027462316040593394
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.027193313157994337,
            "ci95": [
              0.017419590249516552,
              0.03848161913637629
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 0.00021705362745275125,
            "ci95": [
              7.413895825939179e-05,
              0.00036527100979703354
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.00044781642813982933,
            "ci95": [
              0.00030335328832673604,
              0.0005816932448391962
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 7.592424937706913e-05,
            "ci95": [
              -5.611117525414484e-05,
              0.00020122713284594852
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.00023076280068707808,
            "ci95": [
              7.106123862116436e-05,
              0.00039889714308601027
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -0.00014112937807568213,
            "ci95": [
              -0.000298689886617078,
              2.6226328085481484e-05
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.0003718921787627602,
            "ci95": [
              -0.0005326125152031365,
              -0.0002170338747814381
            ]
          }
        }
      },
      "pooled_entry__minus__pooled_constant": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "event_entry": 0.0,
          "event_work": 0.0,
          "event_constant": 0.0,
          "pooled_entry": 1.0,
          "pooled_work": 0.0,
          "pooled_constant": -1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.0008718742079763253,
            "ci95": [
              -0.002541345630540502,
              0.00416568162071926
            ]
          },
          "auc.is_click": {
            "estimate": 0.0016786389721302752,
            "ci95": [
              0.000908394469842605,
              0.002342168470819914
            ]
          },
          "auc.cart": {
            "estimate": 0.0012518311232687651,
            "ci95": [
              -0.0035416219653413125,
              0.005590917853007678
            ]
          },
          "auc.fav": {
            "estimate": -0.02037535509687116,
            "ci95": [
              -0.028706715681554477,
              -0.013348930726415676
            ]
          },
          "auc.buy": {
            "estimate": 0.020932381833377423,
            "ci95": [
              0.01075953199622393,
              0.031046952684248872
            ]
          },
          "bce.is_click": {
            "estimate": 3.5358156492137915e-06,
            "ci95": [
              -5.7924685324634014e-05,
              7.138304705624553e-05
            ]
          },
          "bce.cart": {
            "estimate": -0.00012073095678035925,
            "ci95": [
              -0.0002293966279230117,
              -2.4042088648557063e-06
            ]
          },
          "bce.fav": {
            "estimate": 0.00024797957501380974,
            "ci95": [
              0.00012569326090932642,
              0.00037819111541030756
            ]
          },
          "bce.buy": {
            "estimate": -0.00019225479262598155,
            "ci95": [
              -0.00030314726578801316,
              -8.068776257810484e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.00042680784886151013,
            "ci95": [
              -0.004003171167310795,
              0.005316424296402355
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.022053994069001437,
            "ci95": [
              0.014645672366693003,
              0.030641689043125902
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.019253742861247147,
            "ci95": [
              -0.029249193236655553,
              -0.009201582352405424
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.021627186220139927,
            "ci95": [
              0.013687604123135433,
              0.03132150735028872
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.019680550710108657,
            "ci95": [
              -0.030531640271696102,
              -0.008492204363971297
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.041307736930248584,
            "ci95": [
              -0.0551530823990585,
              -0.03000889559932831
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 0.00012426677242957304,
            "ci95": [
              1.8085409608919012e-06,
              0.00024621528182420843
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.00024444375936459595,
            "ci95": [
              -0.00040586271570650957,
              -9.940155481536201e-05
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.00019579060827519534,
            "ci95": [
              7.549487064049896e-05,
              0.0003171994250660267
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.000368710531794169,
            "ci95": [
              -0.0005482073244132364,
              -0.00020748858533717633
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 7.15238358456223e-05,
            "ci95": [
              -8.297364084205926e-05,
              0.00023741015416631165
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.0004402343676397913,
            "ci95": [
              0.00027058208645828525,
              0.0006191311508135866
            ]
          }
        }
      },
      "event_entry__minus__pooled_entry": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "event_entry": 1.0,
          "event_work": 0.0,
          "event_constant": 0.0,
          "pooled_entry": -1.0,
          "pooled_work": 0.0,
          "pooled_constant": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.0009877323374864189,
            "ci95": [
              -0.004773053872229927,
              0.002503981295255304
            ]
          },
          "auc.is_click": {
            "estimate": -0.00029208919383238996,
            "ci95": [
              -0.0009620216643889318,
              0.000409636075419964
            ]
          },
          "auc.cart": {
            "estimate": -0.007357532758536012,
            "ci95": [
              -0.012494611714045053,
              -0.0020126412779741423
            ]
          },
          "auc.fav": {
            "estimate": 0.019873875037975552,
            "ci95": [
              0.01301890810835942,
              0.02685511233880901
            ]
          },
          "auc.buy": {
            "estimate": -0.016175182435552826,
            "ci95": [
              -0.02816054788465566,
              -0.005763677187238844
            ]
          },
          "bce.is_click": {
            "estimate": -0.00012650119724622044,
            "ci95": [
              -0.00019570871860319984,
              -5.9802535855739475e-05
            ]
          },
          "bce.cart": {
            "estimate": 0.00017636933893119128,
            "ci95": [
              7.492147203376887e-05,
              0.00028896149500761185
            ]
          },
          "bce.fav": {
            "estimate": -0.0003069935809082956,
            "ci95": [
              -0.00042635524716436185,
              -0.0001997665181986138
            ]
          },
          "bce.buy": {
            "estimate": 2.1815136819512e-05,
            "ci95": [
              -9.32339752974446e-05,
              0.00015384150215306985
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.007065443564703622,
            "ci95": [
              0.0019293127398789412,
              0.012138008208263842
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.020165964231807942,
            "ci95": [
              -0.02723591756979243,
              -0.013237967778805659
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.015883093241720436,
            "ci95": [
              0.005488984182877768,
              0.02797269103818055
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.027231407796511564,
            "ci95": [
              -0.035924381697605215,
              -0.019084978417158886
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.008817649677016814,
            "ci95": [
              -0.0032689378750491546,
              0.021599940676113625
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.03604905747352838,
            "ci95": [
              0.02299210376211544,
              0.050440193634552266
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.0003028705361774117,
            "ci95": [
              -0.00043177577932001213,
              -0.0001775565008561202
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.00018049238366207518,
            "ci95": [
              4.484558241326366e-05,
              0.00030691695417367303
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.00014831633406573244,
            "ci95": [
              -0.0002943219311189305,
              -7.744970183356876e-06
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.0004833629198394869,
            "ci95": [
              0.0003279209637716405,
              0.0006352154836461961
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00015455420211167928,
            "ci95": [
              -1.8470972802825732e-05,
              0.0003214507040136172
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.0003288087177278076,
            "ci95": [
              -0.0005070744255646415,
              -0.00015189494997674163
            ]
          }
        }
      }
    },
    "parameter_count": 51742692,
    "initializer_sha256": "aa3b2ab7b9014c0f0183713dca1e9fcc087775a3c1683e2efaa4346181bd573d",
    "optimizer_layout_sha256": "2cff1cc82fbdd9c46bc4882a0ee33547b317bc85eaf8cc1c1829237fa6068b1a",
    "formal_worker_cost": {
      "wall_seconds": 2543.965931020677,
      "cpu_seconds": 3047.307200656,
      "peak_rss_bytes": 17067089920,
      "gpu_peak_bytes": 28375515136,
      "samples": 2334
    },
    "smoke_worker_cost": {
      "wall_seconds": 209.93510429561138,
      "cpu_seconds": 269.716068123,
      "peak_rss_bytes": 19876069376,
      "gpu_peak_bytes": 30000807936,
      "samples": 192
    },
    "complete_recorded_costs": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2543.965931020677,
        "cpu_seconds": 3047.307200656,
        "peak_rss_bytes": 17067089920,
        "gpu_peak_bytes": 28375515136,
        "samples": 2334
      },
      "formal_runner": {
        "wall_seconds": 2539.445438310504,
        "cpu_seconds": 2930.838562725,
        "peak_process_rss_bytes": 5303574528
      },
      "fit_including_validation": {
        "wall_seconds": 2415.562258735299,
        "cpu_seconds": 2758.594179099,
        "peak_process_rss_bytes": 5303574528
      },
      "prediction_export": {
        "wall_seconds": 118.96000053733587,
        "cpu_seconds": 166.62952875300016,
        "peak_process_rss_bytes": 5303574528
      },
      "smoke_worker": {
        "wall_seconds": 209.93510429561138,
        "cpu_seconds": 269.716068123,
        "peak_rss_bytes": 19876069376,
        "gpu_peak_bytes": 30000807936,
        "samples": 192
      },
      "smoke_runner": {
        "wall_seconds": 205.9832237586379,
        "cpu_seconds": 186.734289816,
        "peak_process_rss_bytes": 3371372544
      },
      "latency_constructor": {
        "wall_seconds": 0.7465212643146515,
        "cpu_seconds": 0.823265211999999,
        "peak_rss_bytes": 2134851584
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.14822092652320862,
        "cpu_seconds": 0.14773312300000185,
        "peak_rss_bytes": 2134851584
      },
      "latency_peak_cuda_bytes": 1284208128,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "forward_latency": {
      "mean_ms": 14.26623044013977,
      "p50_ms": 14.264368057250977,
      "p95_ms": 14.29220118522644,
      "batch_size": 1024,
      "warmups": 10,
      "iterations": 40
    },
    "latency_ratio_to_parent": 0.9992642052752404,
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
  "decision_reason": "Macro delta -0.003351628, CI95 [-0.00633145636213481, -0.0003471266979163905], forward ratio 0.999264; misses both frozen quality and near policy.",
  "decision_scope": "Root queue disposition only; no released-incumbent promotion or supported lesson.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/pooled_entry",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/pooled_entry/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/pooled_entry/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/pooled_entry/Taobao_Action/pooled_entry.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/paired-analysis/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/latency/completion.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/pooled_entry",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/pooled_entry/result.json",
    "result_sha256": "2a8756d1983d4d9de3291952463837a2dd0fdcc06b44b76516ac1bb721cf9b73",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/pooled_entry/manifest.json",
    "manifest_sha256": "d6c274f63292488dd3d1550b38956a6837c8ad71fee589e800d378a7e7ea4ebd",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/pooled_entry/Taobao_Action/pooled_entry.model",
    "checkpoint_sha256": "7e72c8a7d7934e479c1845b96c10858266bd627d62d7f5736722d158c74a04a2",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/paired-analysis/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
