## 244. value-content--pooled_work--s20262027

- 假设：Second-work value supplementation; pooled_work conditional nonlinear content before/after original weighted aggregation.
- 判定：discard
- 原因：Macro delta -0.007217189, CI95 [-0.010118841791676636, -0.0042218699572851605], forward ratio 1.002274; misses both frozen quality and near policy.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "value-content--pooled_work--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-10T06:16:19.753374+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "pooled_work",
  "code_id": "24e06fb09c6658ef53f419e5fa410cfa79eb699cc16354bfd31a299d62180a70",
  "hypothesis": "Second-work value supplementation; pooled_work conditional nonlinear content before/after original weighted aggregation.",
  "change": "{\"model\": \"QFormerCross8ValueContent\", \"maturity_mode\": \"mature\", \"value_position\": \"pooled\", \"value_condition\": \"work\", \"value_rank\": 16, \"work_steps\": 2, \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8ValueContent",
    "maturity_mode": "mature",
    "value_position": "pooled",
    "value_condition": "work",
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
      "value": 0.7598777619042676,
      "delta": -0.007217189410592878
    },
    "per_task": {
      "is_click": {
        "logloss": 0.18456532549330895,
        "AUC": 0.6471886204562235
      },
      "cart": {
        "logloss": 0.02194631713023504,
        "AUC": 0.7899449737311881
      },
      "fav": {
        "logloss": 0.016919604907625795,
        "AUC": 0.7920113465854389
      },
      "buy": {
        "logloss": 0.009854440804347941,
        "AUC": 0.81036610684422
      }
    },
    "macro_source_logloss": 0.05832142208387943,
    "source_metric_comparison": {
      "is_click": {
        "source_AUC": 0.6471886204562235,
        "computed_AUC": 0.6471886204562235,
        "auc_difference": 0.0,
        "source_logloss": 0.18456532549330895,
        "computed_bce": 0.18456532549330895,
        "bce_difference": 0.0
      },
      "cart": {
        "source_AUC": 0.7899449737311881,
        "computed_AUC": 0.7899449737311881,
        "auc_difference": 0.0,
        "source_logloss": 0.02194631713023504,
        "computed_bce": 0.021946318302147685,
        "bce_difference": 1.1719126433162863e-09
      },
      "fav": {
        "source_AUC": 0.7920113465854389,
        "computed_AUC": 0.7920113465854389,
        "auc_difference": 0.0,
        "source_logloss": 0.016919604907625795,
        "computed_bce": 0.016919604917783944,
        "bce_difference": 1.0158148627814612e-11
      },
      "buy": {
        "source_AUC": 0.81036610684422,
        "computed_AUC": 0.81036610684422,
        "auc_difference": 0.0,
        "source_logloss": 0.009854440804347941,
        "computed_bce": 0.009854440887068626,
        "bce_difference": 8.272068446890657e-11
      }
    },
    "matched_parent_effect": {
      "coefficients": {
        "mature_r2": -1.0,
        "cross8_mature": 0.0,
        "event_entry": 0.0,
        "event_work": 0.0,
        "event_constant": 0.0,
        "pooled_entry": 0.0,
        "pooled_work": 1.0,
        "pooled_constant": 0.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": -0.007217189410592878,
          "ci95": [
            -0.010118841791676636,
            -0.0042218699572851605
          ]
        },
        "auc.is_click": {
          "estimate": -0.00274767274476595,
          "ci95": [
            -0.0035756351445204978,
            -0.0019933995781585227
          ]
        },
        "auc.cart": {
          "estimate": -0.0024892966261556015,
          "ci95": [
            -0.008398146296724673,
            0.005287715062274468
          ]
        },
        "auc.fav": {
          "estimate": -0.010452143789609125,
          "ci95": [
            -0.016786443014469833,
            -0.005347341219821262
          ]
        },
        "auc.buy": {
          "estimate": -0.013179644481840835,
          "ci95": [
            -0.021758934759766986,
            -0.005481312177090706
          ]
        },
        "bce.is_click": {
          "estimate": 0.00017367922552227744,
          "ci95": [
            0.0001109141942652929,
            0.00024628976541890904
          ]
        },
        "bce.cart": {
          "estimate": 0.00011836393271365264,
          "ci95": [
            -1.8926758507659905e-05,
            0.00023646872173988467
          ]
        },
        "bce.fav": {
          "estimate": 0.00012144722343161024,
          "ci95": [
            -1.5134416015668335e-05,
            0.0002698835004877098
          ]
        },
        "bce.buy": {
          "estimate": 0.00021063895034292325,
          "ci95": [
            0.00012791595101299143,
            0.00030521342192011157
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": -0.00025837611861034837,
          "ci95": [
            -0.007809725225446848,
            0.005822668342361834
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": 0.007704471044843175,
          "ci95": [
            0.0025949487000703347,
            0.014032416860461408
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": 0.010431971737074885,
          "ci95": [
            0.0028346524906857017,
            0.019115963262447036
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": 0.007962847163453524,
          "ci95": [
            0.00010015497249425174,
            0.016881215880595914
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": 0.010690347855685234,
          "ci95": [
            0.0004292967998090703,
            0.02211643041012474
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": 0.00272750069223171,
          "ci95": [
            -0.007320958742502683,
            0.013033743478753605
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": 5.5315292808624794e-05,
          "ci95": [
            -8.440305717384889e-05,
            0.00021331483252767192
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": 5.2232002090667196e-05,
          "ci95": [
            -0.00010649849828524519,
            0.0002090265080330827
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": -3.6959724820645815e-05,
          "ci95": [
            -0.00015277685296339038,
            7.451737603928866e-05
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": -3.0832907179575986e-06,
          "ci95": [
            -0.00019585194871046168,
            0.000193732968593531
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": -9.227501762927061e-05,
          "ci95": [
            -0.0002608611699623994,
            6.129401679111598e-05
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": -8.919172691131301e-05,
          "ci95": [
            -0.0002527565470004411,
            7.885359480192615e-05
          ]
        }
      }
    },
    "all_prespecified_effects_involving_candidate": {
      "pooled_work__minus__mature_r2": {
        "coefficients": {
          "mature_r2": -1.0,
          "cross8_mature": 0.0,
          "event_entry": 0.0,
          "event_work": 0.0,
          "event_constant": 0.0,
          "pooled_entry": 0.0,
          "pooled_work": 1.0,
          "pooled_constant": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.007217189410592878,
            "ci95": [
              -0.010118841791676636,
              -0.0042218699572851605
            ]
          },
          "auc.is_click": {
            "estimate": -0.00274767274476595,
            "ci95": [
              -0.0035756351445204978,
              -0.0019933995781585227
            ]
          },
          "auc.cart": {
            "estimate": -0.0024892966261556015,
            "ci95": [
              -0.008398146296724673,
              0.005287715062274468
            ]
          },
          "auc.fav": {
            "estimate": -0.010452143789609125,
            "ci95": [
              -0.016786443014469833,
              -0.005347341219821262
            ]
          },
          "auc.buy": {
            "estimate": -0.013179644481840835,
            "ci95": [
              -0.021758934759766986,
              -0.005481312177090706
            ]
          },
          "bce.is_click": {
            "estimate": 0.00017367922552227744,
            "ci95": [
              0.0001109141942652929,
              0.00024628976541890904
            ]
          },
          "bce.cart": {
            "estimate": 0.00011836393271365264,
            "ci95": [
              -1.8926758507659905e-05,
              0.00023646872173988467
            ]
          },
          "bce.fav": {
            "estimate": 0.00012144722343161024,
            "ci95": [
              -1.5134416015668335e-05,
              0.0002698835004877098
            ]
          },
          "bce.buy": {
            "estimate": 0.00021063895034292325,
            "ci95": [
              0.00012791595101299143,
              0.00030521342192011157
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.00025837611861034837,
            "ci95": [
              -0.007809725225446848,
              0.005822668342361834
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.007704471044843175,
            "ci95": [
              0.0025949487000703347,
              0.014032416860461408
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.010431971737074885,
            "ci95": [
              0.0028346524906857017,
              0.019115963262447036
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.007962847163453524,
            "ci95": [
              0.00010015497249425174,
              0.016881215880595914
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.010690347855685234,
            "ci95": [
              0.0004292967998090703,
              0.02211643041012474
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.00272750069223171,
            "ci95": [
              -0.007320958742502683,
              0.013033743478753605
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 5.5315292808624794e-05,
            "ci95": [
              -8.440305717384889e-05,
              0.00021331483252767192
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 5.2232002090667196e-05,
            "ci95": [
              -0.00010649849828524519,
              0.0002090265080330827
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -3.6959724820645815e-05,
            "ci95": [
              -0.00015277685296339038,
              7.451737603928866e-05
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -3.0832907179575986e-06,
            "ci95": [
              -0.00019585194871046168,
              0.000193732968593531
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -9.227501762927061e-05,
            "ci95": [
              -0.0002608611699623994,
              6.129401679111598e-05
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -8.919172691131301e-05,
            "ci95": [
              -0.0002527565470004411,
              7.885359480192615e-05
            ]
          }
        }
      },
      "pooled_work__minus__cross8_mature": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": -1.0,
          "event_entry": 0.0,
          "event_work": 0.0,
          "event_constant": 0.0,
          "pooled_entry": 0.0,
          "pooled_work": 1.0,
          "pooled_constant": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.00047134180880778476,
            "ci95": [
              -0.002503317264701223,
              0.0036486390863802232
            ]
          },
          "auc.is_click": {
            "estimate": -0.002748149686742196,
            "ci95": [
              -0.003546468872521802,
              -0.002034453747777252
            ]
          },
          "auc.cart": {
            "estimate": 0.0007382383600721409,
            "ci95": [
              -0.005009765552930989,
              0.006076888209916283
            ]
          },
          "auc.fav": {
            "estimate": -0.004490664442010628,
            "ci95": [
              -0.011072750895249788,
              0.0017499058597179454
            ]
          },
          "auc.buy": {
            "estimate": 0.008385943003911822,
            "ci95": [
              0.00030465702537827734,
              0.01682975615363887
            ]
          },
          "bce.is_click": {
            "estimate": 0.00024738764471760044,
            "ci95": [
              0.00018151360570295427,
              0.0003199845567850495
            ]
          },
          "bce.cart": {
            "estimate": 0.00011190666506865482,
            "ci95": [
              5.60115921414261e-06,
              0.00022506558701465543
            ]
          },
          "bce.fav": {
            "estimate": -5.37742636691034e-05,
            "ci95": [
              -0.00016611909400928324,
              7.039386869885002e-05
            ]
          },
          "bce.buy": {
            "estimate": -0.0001529494413695029,
            "ci95": [
              -0.0002478883841032143,
              -5.9362666680941096e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.003486388046814337,
            "ci95": [
              -0.009227694355536577,
              0.0022628896940698343
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.0017425147552684317,
            "ci95": [
              -0.004622619711365378,
              0.008295314473904343
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.011134092690654018,
            "ci95": [
              -0.01985976787392127,
              -0.0033713031091692227
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.005228902802082769,
            "ci95": [
              -0.0031336549937418245,
              0.013247355191935744
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.007647704643839681,
            "ci95": [
              -0.017806143818071978,
              0.002497116740420495
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.01287660744592245,
            "ci95": [
              -0.02323076139022687,
              -0.0036162246008326015
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 0.00013548097964894562,
            "ci95": [
              7.551082433961568e-06,
              0.0002801660271836314
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.00030116190838670384,
            "ci95": [
              0.00016629617716175323,
              0.0004353491282650795
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.00040033708608710333,
            "ci95": [
              0.0002937100191830864,
              0.0005200393316789238
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.00016568092873775822,
            "ci95": [
              1.539164238505411e-05,
              0.0003318757079338313
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.0002648561064381577,
            "ci95": [
              0.0001373251432376514,
              0.000409089517332965
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 9.917517770039949e-05,
            "ci95": [
              -3.8703521841171924e-05,
              0.0002297536224066981
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
      "pooled_work__minus__pooled_constant": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "event_entry": 0.0,
          "event_work": 0.0,
          "event_constant": 0.0,
          "pooled_entry": 0.0,
          "pooled_work": 1.0,
          "pooled_constant": -1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.0029936876738340945,
            "ci95": [
              -0.006090106449406745,
              -0.0002223709344236027
            ]
          },
          "auc.is_click": {
            "estimate": -0.0017901003548540517,
            "ci95": [
              -0.002615026337983084,
              -0.0009935371232632747
            ]
          },
          "auc.cart": {
            "estimate": -0.00028874378502086184,
            "ci95": [
              -0.005308251772275463,
              0.004641629475435562
            ]
          },
          "auc.fav": {
            "estimate": -0.012005165163857856,
            "ci95": [
              -0.01903828609967125,
              -0.0047812673367427765
            ]
          },
          "auc.buy": {
            "estimate": 0.0021092586083963916,
            "ci95": [
              -0.004618041416292032,
              0.009798563295707151
            ]
          },
          "bce.is_click": {
            "estimate": 0.0002787512061017683,
            "ci95": [
              0.00021006672238330082,
              0.00035575983444029104
            ]
          },
          "bce.cart": {
            "estimate": -6.2569193780556e-05,
            "ci95": [
              -0.00015899139505085947,
              4.32870558539637e-05
            ]
          },
          "bce.fav": {
            "estimate": 7.53785373265349e-05,
            "ci95": [
              -4.828690034879993e-05,
              0.00019811660211104106
            ]
          },
          "bce.buy": {
            "estimate": 7.036348449503821e-06,
            "ci95": [
              -7.67923101569562e-05,
              8.225657449540733e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.0015013565698331899,
            "ci95": [
              -0.006626870334370765,
              0.00341274066106135
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.010215064809003804,
            "ci95": [
              0.003044517156223884,
              0.016837666126490847
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.0038993589632504433,
            "ci95": [
              -0.01163210632904547,
              0.0031765206147340843
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.011716421378836994,
            "ci95": [
              0.002466178307395134,
              0.020288203882456692
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.0023980023934172534,
            "ci95": [
              -0.010839111438292451,
              0.005536250171018036
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.014114423772254248,
            "ci95": [
              -0.02474374274058776,
              -0.004305986566643614
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 0.0003413203998823243,
            "ci95": [
              0.00021966939583556322,
              0.00047346331099792677
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.00020337266877523338,
            "ci95": [
              5.642418607666082e-05,
              0.0003386463586889399
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.00027171485765226447,
            "ci95": [
              0.00017664218828703084,
              0.00038189959008054936
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.0001379477311070909,
            "ci95": [
              -0.0002915241884792028,
              1.9530571318271142e-05
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -6.960554223005982e-05,
            "ci95": [
              -0.00020032626248529784,
              5.648234965069387e-05
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 6.834218887703108e-05,
            "ci95": [
              -7.526913683789117e-05,
              0.00021281142028830089
            ]
          }
        }
      },
      "event_work__minus__pooled_work": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "event_entry": 0.0,
          "event_work": 1.0,
          "event_constant": 0.0,
          "pooled_entry": 0.0,
          "pooled_work": -1.0,
          "pooled_constant": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.004311980481132294,
            "ci95": [
              0.001515423766723957,
              0.007267848372803551
            ]
          },
          "auc.is_click": {
            "estimate": 0.0021848369016588265,
            "ci95": [
              0.0014675651841233235,
              0.00291508251035587
            ]
          },
          "auc.cart": {
            "estimate": 0.0005673606492155869,
            "ci95": [
              -0.006376927203733609,
              0.006009401801088072
            ]
          },
          "auc.fav": {
            "estimate": 0.012528163104143397,
            "ci95": [
              0.00652357652024807,
              0.018695638983228607
            ]
          },
          "auc.buy": {
            "estimate": 0.0019675612695113642,
            "ci95": [
              -0.005183876834256479,
              0.010037010790273039
            ]
          },
          "bce.is_click": {
            "estimate": -0.00026466933402993775,
            "ci95": [
              -0.00033369185522380757,
              -0.0002032292703560285
            ]
          },
          "bce.cart": {
            "estimate": -4.341282490935919e-05,
            "ci95": [
              -0.00016164586938129848,
              8.981976497942059e-05
            ]
          },
          "bce.fav": {
            "estimate": -0.00026994279979061786,
            "ci95": [
              -0.00037534429673040945,
              -0.00016373704490553964
            ]
          },
          "bce.buy": {
            "estimate": -6.701900029474489e-05,
            "ci95": [
              -0.00016316195705913118,
              2.0518416618203602e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.0016174762524432396,
            "ci95": [
              -0.003976672234703974,
              0.008472990663394247
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.01034332620248457,
            "ci95": [
              -0.016757761852712376,
              -0.004341781868182303
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.00021727563214746226,
            "ci95": [
              -0.007852444350619445,
              0.0076634940932099015
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.01196080245492781,
            "ci95": [
              -0.021829286182300393,
              -0.0041215727487615315
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.0014002006202957773,
            "ci95": [
              -0.012052583107692272,
              0.008407508213752824
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.010560601834632033,
            "ci95": [
              0.0002627286721500478,
              0.02033334207382318
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00022125650912057857,
            "ci95": [
              -0.0003737287537697747,
              -8.729364661155636e-05
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 5.273465760680107e-06,
            "ci95": [
              -0.00012455954139126336,
              0.00012588067883235253
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.00019765033373519286,
            "ci95": [
              -0.0003114116747729416,
              -7.418478162508132e-05
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.00022652997488125867,
            "ci95": [
              7.724953957003803e-05,
              0.0003983375782262043
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 2.3606175385385703e-05,
            "ci95": [
              -0.00012625479036756086,
              0.00018817277275768698
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.00020292379949587297,
            "ci95": [
              -0.0003490647648064012,
              -6.543350142011114e-05
            ]
          }
        }
      }
    },
    "parameter_count": 51742692,
    "initializer_sha256": "aa3b2ab7b9014c0f0183713dca1e9fcc087775a3c1683e2efaa4346181bd573d",
    "optimizer_layout_sha256": "e263a5725bbf3dde6f110b7c886171863238c00bff6b297ae9108e78ffc9bf2f",
    "formal_worker_cost": {
      "wall_seconds": 2530.5203538760543,
      "cpu_seconds": 3032.308274989,
      "peak_rss_bytes": 17086054400,
      "gpu_peak_bytes": 28371320832,
      "samples": 2322
    },
    "smoke_worker_cost": {
      "wall_seconds": 201.25274242460728,
      "cpu_seconds": 258.87551321999996,
      "peak_rss_bytes": 19847925760,
      "gpu_peak_bytes": 30000807936,
      "samples": 184
    },
    "complete_recorded_costs": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2530.5203538760543,
        "cpu_seconds": 3032.308274989,
        "peak_rss_bytes": 17086054400,
        "gpu_peak_bytes": 28371320832,
        "samples": 2322
      },
      "formal_runner": {
        "wall_seconds": 2526.054277524352,
        "cpu_seconds": 2917.193958029,
        "peak_process_rss_bytes": 5221584896
      },
      "fit_including_validation": {
        "wall_seconds": 2403.0723568201065,
        "cpu_seconds": 2746.774595737,
        "peak_process_rss_bytes": 5221584896
      },
      "prediction_export": {
        "wall_seconds": 117.90648832172155,
        "cpu_seconds": 164.90178650299958,
        "peak_process_rss_bytes": 5221584896
      },
      "smoke_worker": {
        "wall_seconds": 201.25274242460728,
        "cpu_seconds": 258.87551321999996,
        "peak_rss_bytes": 19847925760,
        "gpu_peak_bytes": 30000807936,
        "samples": 184
      },
      "smoke_runner": {
        "wall_seconds": 197.00315845757723,
        "cpu_seconds": 179.777079433,
        "peak_process_rss_bytes": 3362832384
      },
      "latency_constructor": {
        "wall_seconds": 0.7421007975935936,
        "cpu_seconds": 0.8177351739999992,
        "peak_rss_bytes": 2134851584
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.13428709656000137,
        "cpu_seconds": 0.13383748300000065,
        "peak_rss_bytes": 2134851584
      },
      "latency_peak_cuda_bytes": 1284208128,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "forward_latency": {
      "mean_ms": 14.309194302558899,
      "p50_ms": 14.262975692749023,
      "p95_ms": 14.31193404197693,
      "batch_size": 1024,
      "warmups": 10,
      "iterations": 40
    },
    "latency_ratio_to_parent": 1.0022735671396759,
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
  "decision_reason": "Macro delta -0.007217189, CI95 [-0.010118841791676636, -0.0042218699572851605], forward ratio 1.002274; misses both frozen quality and near policy.",
  "decision_scope": "Root queue disposition only; no released-incumbent promotion or supported lesson.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/pooled_work",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/pooled_work/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/pooled_work/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/pooled_work/Taobao_Action/pooled_work.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/paired-analysis/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/latency/completion.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/pooled_work",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/pooled_work/result.json",
    "result_sha256": "c88b7a68e53c64c355bc1a09b75174a04ae1cf63ecc889d3ccf7aed2ef5c14db",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/pooled_work/manifest.json",
    "manifest_sha256": "127a4953dd524c52352ee47bf9accc3c2aa56640deabd13c26517f26c78c7b14",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/pooled_work/Taobao_Action/pooled_work.model",
    "checkpoint_sha256": "60c32da55bd63b69948849a1ba6a8dc90abd07603437bbc455a79a44867f6a60",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/paired-analysis/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
