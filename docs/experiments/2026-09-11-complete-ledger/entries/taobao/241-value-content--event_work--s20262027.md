## 241. value-content--event_work--s20262027

- 假设：Second-work value supplementation; event_work conditional nonlinear content before/after original weighted aggregation.
- 判定：discard
- 原因：Macro delta -0.002905209, CI95 [-0.005597721227477079, -0.00040624993050394914], forward ratio 1.116838; misses both frozen quality and near policy.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "value-content--event_work--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-10T06:16:19.753374+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "event_work",
  "code_id": "24e06fb09c6658ef53f419e5fa410cfa79eb699cc16354bfd31a299d62180a70",
  "hypothesis": "Second-work value supplementation; event_work conditional nonlinear content before/after original weighted aggregation.",
  "change": "{\"model\": \"QFormerCross8ValueContent\", \"maturity_mode\": \"mature\", \"value_position\": \"event\", \"value_condition\": \"work\", \"value_rank\": 16, \"work_steps\": 2, \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8ValueContent",
    "maturity_mode": "mature",
    "value_position": "event",
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
      "value": 0.7641897423853998,
      "delta": -0.0029052089294605843
    },
    "per_task": {
      "is_click": {
        "logloss": 0.18430065615927715,
        "AUC": 0.6493734573578823
      },
      "cart": {
        "logloss": 0.02190290516305314,
        "AUC": 0.7905123343804037
      },
      "fav": {
        "logloss": 0.01664966211620482,
        "AUC": 0.8045395096895823
      },
      "buy": {
        "logloss": 0.009787421886456631,
        "AUC": 0.8123336681137313
      }
    },
    "macro_source_logloss": 0.05816016133124794,
    "source_metric_comparison": {
      "is_click": {
        "source_AUC": 0.6493734573578823,
        "computed_AUC": 0.6493734573578823,
        "auc_difference": 0.0,
        "source_logloss": 0.18430065615927715,
        "computed_bce": 0.184300656159279,
        "bce_difference": 1.8596235662471372e-15
      },
      "cart": {
        "source_AUC": 0.7905123343804037,
        "computed_AUC": 0.7905123343804037,
        "auc_difference": 0.0,
        "source_logloss": 0.02190290516305314,
        "computed_bce": 0.021902905477238326,
        "bce_difference": 3.1418518481318713e-10
      },
      "fav": {
        "source_AUC": 0.8045395096895823,
        "computed_AUC": 0.8045395096895823,
        "auc_difference": 0.0,
        "source_logloss": 0.01664966211620482,
        "computed_bce": 0.016649662117993326,
        "bce_difference": 1.788506842625992e-12
      },
      "buy": {
        "source_AUC": 0.8123336681137313,
        "computed_AUC": 0.8123336681137313,
        "auc_difference": 0.0,
        "source_logloss": 0.009787421886456631,
        "computed_bce": 0.009787421886773881,
        "bce_difference": 3.1724969873359044e-13
      }
    },
    "matched_parent_effect": {
      "coefficients": {
        "mature_r2": -1.0,
        "cross8_mature": 0.0,
        "event_entry": 0.0,
        "event_work": 1.0,
        "event_constant": 0.0,
        "pooled_entry": 0.0,
        "pooled_work": 0.0,
        "pooled_constant": 0.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": -0.0029052089294605843,
          "ci95": [
            -0.005597721227477079,
            -0.00040624993050394914
          ]
        },
        "auc.is_click": {
          "estimate": -0.0005628358431071234,
          "ci95": [
            -0.0012663796860318654,
            0.00015624531752636003
          ]
        },
        "auc.cart": {
          "estimate": -0.0019219359769400146,
          "ci95": [
            -0.0061832140235080265,
            0.002943207981091517
          ]
        },
        "auc.fav": {
          "estimate": 0.002076019314534272,
          "ci95": [
            -0.0034129792876028712,
            0.00804171015658191
          ]
        },
        "auc.buy": {
          "estimate": -0.011212083212329471,
          "ci95": [
            -0.01945391555159274,
            -0.004238184493719355
          ]
        },
        "bce.is_click": {
          "estimate": -9.099010850766032e-05,
          "ci95": [
            -0.00016086393727701085,
            -2.566611390617542e-05
          ]
        },
        "bce.cart": {
          "estimate": 7.495110780429345e-05,
          "ci95": [
            -9.93798848636061e-06,
            0.00016927814679767197
          ]
        },
        "bce.fav": {
          "estimate": -0.00014849557635900762,
          "ci95": [
            -0.00028092678702515276,
            -1.83262888967059e-05
          ]
        },
        "bce.buy": {
          "estimate": 0.00014361995004817836,
          "ci95": [
            6.30145991880127e-05,
            0.00023770537766530458
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": 0.0013591001338328912,
          "ci95": [
            -0.0034695802852028916,
            0.00561271215682331
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": -0.0026388551576413954,
          "ci95": [
            -0.00855039057361486,
            0.0030218844449050996
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": 0.010649247369222348,
          "ci95": [
            0.003772587504769015,
            0.018819074135703984
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": -0.003997955291474287,
          "ci95": [
            -0.01152970978998499,
            0.003460166396713447
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": 0.009290147235389457,
          "ci95": [
            0.0004437762527925744,
            0.01758839787079302
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": 0.013288102526863743,
          "ci95": [
            0.0048419209228614925,
            0.023384299463265177
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": -0.00016594121631195377,
          "ci95": [
            -0.00028040091405683135,
            -6.0705903368447114e-05
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": 5.75054678513473e-05,
          "ci95": [
            -8.684381655792326e-05,
            0.00020931449459517232
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": -0.00023461005855583868,
          "ci95": [
            -0.00035461540390313497,
            -0.00014174297223986738
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": 0.00022344668416330107,
          "ci95": [
            5.996677312126702e-05,
            0.0003979246178896241
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": -6.86688422438849e-05,
          "ci95": [
            -0.00018208363523877693,
            5.1169124157368486e-05
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": -0.000292115526407186,
          "ci95": [
            -0.0004454940269171774,
            -0.00013227657905839685
          ]
        }
      }
    },
    "all_prespecified_effects_involving_candidate": {
      "event_work__minus__mature_r2": {
        "coefficients": {
          "mature_r2": -1.0,
          "cross8_mature": 0.0,
          "event_entry": 0.0,
          "event_work": 1.0,
          "event_constant": 0.0,
          "pooled_entry": 0.0,
          "pooled_work": 0.0,
          "pooled_constant": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.0029052089294605843,
            "ci95": [
              -0.005597721227477079,
              -0.00040624993050394914
            ]
          },
          "auc.is_click": {
            "estimate": -0.0005628358431071234,
            "ci95": [
              -0.0012663796860318654,
              0.00015624531752636003
            ]
          },
          "auc.cart": {
            "estimate": -0.0019219359769400146,
            "ci95": [
              -0.0061832140235080265,
              0.002943207981091517
            ]
          },
          "auc.fav": {
            "estimate": 0.002076019314534272,
            "ci95": [
              -0.0034129792876028712,
              0.00804171015658191
            ]
          },
          "auc.buy": {
            "estimate": -0.011212083212329471,
            "ci95": [
              -0.01945391555159274,
              -0.004238184493719355
            ]
          },
          "bce.is_click": {
            "estimate": -9.099010850766032e-05,
            "ci95": [
              -0.00016086393727701085,
              -2.566611390617542e-05
            ]
          },
          "bce.cart": {
            "estimate": 7.495110780429345e-05,
            "ci95": [
              -9.93798848636061e-06,
              0.00016927814679767197
            ]
          },
          "bce.fav": {
            "estimate": -0.00014849557635900762,
            "ci95": [
              -0.00028092678702515276,
              -1.83262888967059e-05
            ]
          },
          "bce.buy": {
            "estimate": 0.00014361995004817836,
            "ci95": [
              6.30145991880127e-05,
              0.00023770537766530458
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.0013591001338328912,
            "ci95": [
              -0.0034695802852028916,
              0.00561271215682331
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.0026388551576413954,
            "ci95": [
              -0.00855039057361486,
              0.0030218844449050996
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.010649247369222348,
            "ci95": [
              0.003772587504769015,
              0.018819074135703984
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.003997955291474287,
            "ci95": [
              -0.01152970978998499,
              0.003460166396713447
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.009290147235389457,
            "ci95": [
              0.0004437762527925744,
              0.01758839787079302
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.013288102526863743,
            "ci95": [
              0.0048419209228614925,
              0.023384299463265177
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00016594121631195377,
            "ci95": [
              -0.00028040091405683135,
              -6.0705903368447114e-05
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 5.75054678513473e-05,
            "ci95": [
              -8.684381655792326e-05,
              0.00020931449459517232
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.00023461005855583868,
            "ci95": [
              -0.00035461540390313497,
              -0.00014174297223986738
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.00022344668416330107,
            "ci95": [
              5.996677312126702e-05,
              0.0003979246178896241
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -6.86688422438849e-05,
            "ci95": [
              -0.00018208363523877693,
              5.1169124157368486e-05
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.000292115526407186,
            "ci95": [
              -0.0004454940269171774,
              -0.00013227657905839685
            ]
          }
        }
      },
      "event_work__minus__cross8_mature": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": -1.0,
          "event_entry": 0.0,
          "event_work": 1.0,
          "event_constant": 0.0,
          "pooled_entry": 0.0,
          "pooled_work": 0.0,
          "pooled_constant": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.0047833222899400785,
            "ci95": [
              0.0019252730358451549,
              0.007803504812156308
            ]
          },
          "auc.is_click": {
            "estimate": -0.0005633127850833697,
            "ci95": [
              -0.0013990903569954358,
              0.00024385165373580456
            ]
          },
          "auc.cart": {
            "estimate": 0.0013055990092877279,
            "ci95": [
              -0.003017971199630162,
              0.0055380525032247266
            ]
          },
          "auc.fav": {
            "estimate": 0.00803749866213277,
            "ci95": [
              0.002205313330009942,
              0.014471215666444893
            ]
          },
          "auc.buy": {
            "estimate": 0.010353504273423186,
            "ci95": [
              0.0019039822143931547,
              0.02034363511492562
            ]
          },
          "bce.is_click": {
            "estimate": -1.728168931233731e-05,
            "ci95": [
              -8.84910808461481e-05,
              5.692961167585095e-05
            ]
          },
          "bce.cart": {
            "estimate": 6.849384015929563e-05,
            "ci95": [
              -2.1719157834153983e-05,
              0.0001654616837189133
            ]
          },
          "bce.fav": {
            "estimate": -0.00032371706345972126,
            "ci95": [
              -0.0004302629334924436,
              -0.00021272418701584624
            ]
          },
          "bce.buy": {
            "estimate": -0.00021996844166424778,
            "ci95": [
              -0.0003191010739051633,
              -0.00013283728677105573
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.0018689117943710976,
            "ci95": [
              -0.006050936430711073,
              0.0024615486707592338
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.008600811447216139,
            "ci95": [
              -0.01480052038178491,
              -0.0028425419812904257
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.010916817058506556,
            "ci95": [
              -0.020908719662449154,
              -0.0023261070677934982
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.0067318996528450414,
            "ci95": [
              -0.014667818846349686,
              0.0005877879828284103
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.009047905264135458,
            "ci95": [
              -0.02015168028161992,
              0.0010239679398301622
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.002316005611290417,
            "ci95": [
              -0.014395854060014706,
              0.008595024455705529
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -8.577552947163294e-05,
            "ci95": [
              -0.00021367679249568354,
              2.468934255104841e-05
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.00030643537414738395,
            "ci95": [
              0.00017365335955836424,
              0.00044141609980552325
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.00020268675235191047,
            "ci95": [
              6.923948541581746e-05,
              0.0003268432796804478
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.0003922109036190169,
            "ci95": [
              0.0002495289083807773,
              0.0005488292182045356
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.0002884622818235434,
            "ci95": [
              0.00016158639165875997,
              0.00043182115212085095
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.00010374862179547348,
            "ci95": [
              -0.00024024286583997399,
              4.526287420031798e-05
            ]
          }
        }
      },
      "event_work__minus__event_entry": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "event_entry": -1.0,
          "event_work": 1.0,
          "event_constant": 0.0,
          "pooled_entry": 0.0,
          "pooled_work": 0.0,
          "pooled_constant": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.0014341509368082928,
            "ci95": [
              -0.0017163203768568059,
              0.004531709042086111
            ]
          },
          "auc.is_click": {
            "estimate": -0.0009918132314931105,
            "ci95": [
              -0.001796855787764673,
              -0.00021736486391852607
            ]
          },
          "auc.cart": {
            "estimate": 0.006384318499461972,
            "ci95": [
              0.0010672809494749942,
              0.01149141844889579
            ]
          },
          "auc.fav": {
            "estimate": 0.0010244779991811503,
            "ci95": [
              -0.004149702189453669,
              0.006568218463497515
            ]
          },
          "auc.buy": {
            "estimate": -0.0006803795199168405,
            "ci95": [
              -0.01022382263940432,
              0.009401658201121689
            ]
          },
          "bce.is_click": {
            "estimate": 0.0001370472536688372,
            "ci95": [
              6.989874316040492e-05,
              0.00021522358106064592
            ]
          },
          "bce.cart": {
            "estimate": -0.00016162040084074722,
            "ci95": [
              -0.0002708018047951085,
              -4.554179270663893e-05
            ]
          },
          "bce.fav": {
            "estimate": -0.00013555025656959707,
            "ci95": [
              -0.00025101815108504914,
              -2.3596905116636885e-05
            ]
          },
          "bce.buy": {
            "estimate": 0.00011045700396122848,
            "ci95": [
              7.923393214493955e-06,
              0.0002140520540930401
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.007376131730955082,
            "ci95": [
              -0.012599883122280692,
              -0.0018633625520478874
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.002016291230674261,
            "ci95": [
              -0.00789841030258073,
              0.003050277726039498
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.00031143371157627,
            "ci95": [
              -0.010476721918166102,
              0.0092390695300444
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.005359840500280821,
            "ci95": [
              -0.0033867799176251177,
              0.01244185434369685
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.007064698019378812,
            "ci95": [
              -0.0041240470957167345,
              0.01694943884043986
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.0017048575190979909,
            "ci95": [
              -0.01000081093433857,
              0.012604641942101773
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 0.0002986676545095844,
            "ci95": [
              0.00016664902245829257,
              0.0004219909339024411
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.00027259751023843426,
            "ci95": [
              0.0001497504332460653,
              0.0004103539420385615
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 2.6590249707608704e-05,
            "ci95": [
              -0.00010419171911748171,
              0.00015005908141207085
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -2.6070144271150147e-05,
            "ci95": [
              -0.00017863714780998297,
              0.00015892189440639293
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -0.0002720774048019757,
            "ci95": [
              -0.00042188138568269534,
              -0.00012473228242583124
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.00024600726053082556,
            "ci95": [
              -0.0003852682853085855,
              -9.67294424387923e-05
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
    "optimizer_layout_sha256": "50213db587720c4428e8d744b8d35ee4fbc0eea410838291f1e68436f484f3b2",
    "formal_worker_cost": {
      "wall_seconds": 2633.404682211578,
      "cpu_seconds": 3135.145446079,
      "peak_rss_bytes": 17063841792,
      "gpu_peak_bytes": 30262951936,
      "samples": 2416
    },
    "smoke_worker_cost": {
      "wall_seconds": 200.153367318213,
      "cpu_seconds": 254.368102502,
      "peak_rss_bytes": 19768623104,
      "gpu_peak_bytes": 32792117248,
      "samples": 183
    },
    "complete_recorded_costs": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2633.404682211578,
        "cpu_seconds": 3135.145446079,
        "peak_rss_bytes": 17063841792,
        "gpu_peak_bytes": 30262951936,
        "samples": 2416
      },
      "formal_runner": {
        "wall_seconds": 2629.4034451395273,
        "cpu_seconds": 3017.2109455540003,
        "peak_process_rss_bytes": 5271363584
      },
      "fit_including_validation": {
        "wall_seconds": 2500.870648562908,
        "cpu_seconds": 2841.480256417,
        "peak_process_rss_bytes": 5271363584
      },
      "prediction_export": {
        "wall_seconds": 123.4742039591074,
        "cpu_seconds": 170.3087260780003,
        "peak_process_rss_bytes": 5271363584
      },
      "smoke_worker": {
        "wall_seconds": 200.153367318213,
        "cpu_seconds": 254.368102502,
        "peak_rss_bytes": 19768623104,
        "gpu_peak_bytes": 32792117248,
        "samples": 183
      },
      "smoke_runner": {
        "wall_seconds": 195.74498146772385,
        "cpu_seconds": 178.15051898000002,
        "peak_process_rss_bytes": 3311620096
      },
      "latency_constructor": {
        "wall_seconds": 0.7709200605750084,
        "cpu_seconds": 0.8482239109999998,
        "peak_rss_bytes": 2134388736
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.14211653172969818,
        "cpu_seconds": 0.14165325799999984,
        "peak_rss_bytes": 2134388736
      },
      "latency_peak_cuda_bytes": 1284208128,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "forward_latency": {
      "mean_ms": 15.944793558120727,
      "p50_ms": 15.881551742553711,
      "p95_ms": 15.994388532638549,
      "batch_size": 1024,
      "warmups": 10,
      "iterations": 40
    },
    "latency_ratio_to_parent": 1.116837522707027,
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
  "decision_reason": "Macro delta -0.002905209, CI95 [-0.005597721227477079, -0.00040624993050394914], forward ratio 1.116838; misses both frozen quality and near policy.",
  "decision_scope": "Root queue disposition only; no released-incumbent promotion or supported lesson.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/event_work",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/event_work/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/event_work/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/event_work/Taobao_Action/event_work.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/paired-analysis/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/latency/completion.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/event_work",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/event_work/result.json",
    "result_sha256": "65b206717b57488d8bc1b60dde847b5e5688d51f877e2aec753ef444724c33e8",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/event_work/manifest.json",
    "manifest_sha256": "7b1b6da6d2d924f7504024cd77896b5c19b74f08c5e1e2f93c9957584153ce71",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/event_work/Taobao_Action/event_work.model",
    "checkpoint_sha256": "c6c74705e7d34184c1d08ec9a729cac61f0eb3e1eb942f1e5260f1254bedcc59",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/paired-analysis/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
