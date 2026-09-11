## 258. endpoint-read--even_last--s20262027

- 假设：Separate first-work stored-endpoint retention from interleaving phase at fixed support count, two work rounds and no prediction feedback.
- 判定：keep
- 原因：Macro delta +0.001435345,CI95 [-0.001703624913421238, 0.004117847779228911];latency ratio1.006071. Research candidate only, macroCIcrosszero andbuyBCEregression;noincumbentpromotion.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "endpoint-read--even_last--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-10T15:23:24.099568+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "even_last",
  "code_id": "b16543b20d537e062ccfd9a027ec0a882a6c396851eddf999162c82a0a9c9bc7",
  "hypothesis": "Separate first-work stored-endpoint retention from interleaving phase at fixed support count, two work rounds and no prediction feedback.",
  "change": "{\"model\": \"QFormerCross8EndpointRead\", \"maturity_mode\": \"mature\", \"segment_placement\": \"first\", \"segment_selection\": \"even_last\", \"work_steps\": 2, \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8EndpointRead",
    "maturity_mode": "mature",
    "segment_placement": "first",
    "segment_selection": "even_last",
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
      "value": 0.7685302966601931,
      "delta": 0.0014353453453326004
    },
    "per_task": {
      "is_click": {
        "logloss": 0.18401725498556862,
        "AUC": 0.6509680242653215
      },
      "cart": {
        "logloss": 0.02173933901026899,
        "AUC": 0.7968610981688303
      },
      "fav": {
        "logloss": 0.016816804092953264,
        "AUC": 0.8040051435503058
      },
      "buy": {
        "logloss": 0.009848382737518827,
        "AUC": 0.8222869206563147
      }
    },
    "macro_source_logloss": 0.05810544520657743,
    "source_metric_comparison": {
      "is_click": {
        "source_AUC": 0.6509680242653215,
        "computed_AUC": 0.6509680242653215,
        "auc_difference": 0.0,
        "source_logloss": 0.18401725498556862,
        "computed_bce": 0.18401725498557284,
        "bce_difference": 4.218847493575595e-15
      },
      "cart": {
        "source_AUC": 0.7968610981688303,
        "computed_AUC": 0.7968610981688303,
        "auc_difference": 0.0,
        "source_logloss": 0.02173933901026899,
        "computed_bce": 0.021739343339065353,
        "bce_difference": 4.328796363028964e-09
      },
      "fav": {
        "source_AUC": 0.8040051435503058,
        "computed_AUC": 0.8040051435503058,
        "auc_difference": 0.0,
        "source_logloss": 0.016816804092953264,
        "computed_bce": 0.01681680438757486,
        "bce_difference": 2.946215944821251e-10
      },
      "buy": {
        "source_AUC": 0.8222869206563147,
        "computed_AUC": 0.8222869206563147,
        "auc_difference": 0.0,
        "source_logloss": 0.009848382737518827,
        "computed_bce": 0.009848383296242746,
        "bce_difference": 5.587239195609106e-10
      }
    },
    "matched_parent_effect": {
      "coefficients": {
        "mature_r2": -1.0,
        "cross8_mature": 0.0,
        "first_even": 0.0,
        "first_odd": 0.0,
        "even_last": 1.0,
        "odd_last": 0.0,
        "odd_no_last": 0.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": 0.0014353453453326004,
          "ci95": [
            -0.001703624913421238,
            0.004117847779228911
          ]
        },
        "auc.is_click": {
          "estimate": 0.0010317310643320754,
          "ci95": [
            0.0003148554993234437,
            0.0017706980814470696
          ]
        },
        "auc.cart": {
          "estimate": 0.004426827811486622,
          "ci95": [
            -0.0005926162291382015,
            0.009870272819447879
          ]
        },
        "auc.fav": {
          "estimate": 0.0015416531752577844,
          "ci95": [
            -0.0047605299483127975,
            0.007133679958159557
          ]
        },
        "auc.buy": {
          "estimate": -0.0012588306697460805,
          "ci95": [
            -0.009628048012538684,
            0.005487416772213954
          ]
        },
        "bce.is_click": {
          "estimate": -0.00037439128221383733,
          "ci95": [
            -0.0004426732198811956,
            -0.0003064136640966643
          ]
        },
        "bce.cart": {
          "estimate": -8.861103036867865e-05,
          "ci95": [
            -0.0002110536632238795,
            2.3942383716619777e-05
          ]
        },
        "bce.fav": {
          "estimate": 1.8646693222525157e-05,
          "ci95": [
            -8.194506409512758e-05,
            0.000127431443817435
          ]
        },
        "bce.buy": {
          "estimate": 0.00020458135951704343,
          "ci95": [
            0.00012088873461454693,
            0.0002969891608543945
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": -0.003395096747154547,
          "ci95": [
            -0.008990991845390877,
            0.0017335508478190707
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": -0.000509922110925709,
          "ci95": [
            -0.006278749623387553,
            0.005755052025813661
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": 0.002290561734078156,
          "ci95": [
            -0.0047806668032554815,
            0.01061024131020708
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": 0.0028851746362288377,
          "ci95": [
            -0.004617238807906904,
            0.010779392788896955
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": 0.005685658481232703,
          "ci95": [
            -0.002244022212162802,
            0.015467314245448872
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": 0.002800483845003865,
          "ci95": [
            -0.007178757466491364,
            0.013040858752491972
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": -0.0002857802518451587,
          "ci95": [
            -0.0004064378208334811,
            -0.00014941628114956785
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": -0.0003930379754363625,
          "ci95": [
            -0.0005255039249715172,
            -0.0002673431017648119
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": -0.0005789726417308808,
          "ci95": [
            -0.0007153699126294038,
            -0.00047026941959690474
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": -0.00010725772359120381,
          "ci95": [
            -0.00025166268388644215,
            3.340450701558669e-05
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": -0.0002931923898857221,
          "ci95": [
            -0.00045774647259687605,
            -0.0001618138200127195
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": -0.00018593466629451827,
          "ci95": [
            -0.00032410291246025947,
            -4.639347544431791e-05
          ]
        }
      }
    },
    "all_prespecified_effects_involving_candidate": {
      "even_last__minus__mature_r2": {
        "coefficients": {
          "mature_r2": -1.0,
          "cross8_mature": 0.0,
          "first_even": 0.0,
          "first_odd": 0.0,
          "even_last": 1.0,
          "odd_last": 0.0,
          "odd_no_last": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.0014353453453326004,
            "ci95": [
              -0.001703624913421238,
              0.004117847779228911
            ]
          },
          "auc.is_click": {
            "estimate": 0.0010317310643320754,
            "ci95": [
              0.0003148554993234437,
              0.0017706980814470696
            ]
          },
          "auc.cart": {
            "estimate": 0.004426827811486622,
            "ci95": [
              -0.0005926162291382015,
              0.009870272819447879
            ]
          },
          "auc.fav": {
            "estimate": 0.0015416531752577844,
            "ci95": [
              -0.0047605299483127975,
              0.007133679958159557
            ]
          },
          "auc.buy": {
            "estimate": -0.0012588306697460805,
            "ci95": [
              -0.009628048012538684,
              0.005487416772213954
            ]
          },
          "bce.is_click": {
            "estimate": -0.00037439128221383733,
            "ci95": [
              -0.0004426732198811956,
              -0.0003064136640966643
            ]
          },
          "bce.cart": {
            "estimate": -8.861103036867865e-05,
            "ci95": [
              -0.0002110536632238795,
              2.3942383716619777e-05
            ]
          },
          "bce.fav": {
            "estimate": 1.8646693222525157e-05,
            "ci95": [
              -8.194506409512758e-05,
              0.000127431443817435
            ]
          },
          "bce.buy": {
            "estimate": 0.00020458135951704343,
            "ci95": [
              0.00012088873461454693,
              0.0002969891608543945
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.003395096747154547,
            "ci95": [
              -0.008990991845390877,
              0.0017335508478190707
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.000509922110925709,
            "ci95": [
              -0.006278749623387553,
              0.005755052025813661
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.002290561734078156,
            "ci95": [
              -0.0047806668032554815,
              0.01061024131020708
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.0028851746362288377,
            "ci95": [
              -0.004617238807906904,
              0.010779392788896955
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.005685658481232703,
            "ci95": [
              -0.002244022212162802,
              0.015467314245448872
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.002800483845003865,
            "ci95": [
              -0.007178757466491364,
              0.013040858752491972
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.0002857802518451587,
            "ci95": [
              -0.0004064378208334811,
              -0.00014941628114956785
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0003930379754363625,
            "ci95": [
              -0.0005255039249715172,
              -0.0002673431017648119
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0005789726417308808,
            "ci95": [
              -0.0007153699126294038,
              -0.00047026941959690474
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.00010725772359120381,
            "ci95": [
              -0.00025166268388644215,
              3.340450701558669e-05
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -0.0002931923898857221,
            "ci95": [
              -0.00045774647259687605,
              -0.0001618138200127195
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.00018593466629451827,
            "ci95": [
              -0.00032410291246025947,
              -4.639347544431791e-05
            ]
          }
        }
      },
      "even_last__minus__cross8_mature": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": -1.0,
          "first_even": 0.0,
          "first_odd": 0.0,
          "even_last": 1.0,
          "odd_last": 0.0,
          "odd_no_last": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.009123876564733263,
            "ci95": [
              0.006325664326927619,
              0.011991797300099279
            ]
          },
          "auc.is_click": {
            "estimate": 0.0010312541223558291,
            "ci95": [
              0.00021837889354039966,
              0.0017283640491449754
            ]
          },
          "auc.cart": {
            "estimate": 0.007654362797714365,
            "ci95": [
              0.0019612781952095815,
              0.012517436491852878
            ]
          },
          "auc.fav": {
            "estimate": 0.007503132522856282,
            "ci95": [
              0.0025910105804005158,
              0.012694628265062003
            ]
          },
          "auc.buy": {
            "estimate": 0.020306756816006577,
            "ci95": [
              0.012597367420640205,
              0.02909366903988896
            ]
          },
          "bce.is_click": {
            "estimate": -0.0003006828630185143,
            "ci95": [
              -0.00036560237039302,
              -0.00022469308942287341
            ]
          },
          "bce.cart": {
            "estimate": -9.506829801367647e-05,
            "ci95": [
              -0.0001875451462376475,
              1.4820792583591066e-05
            ]
          },
          "bce.fav": {
            "estimate": -0.00015657479387818848,
            "ci95": [
              -0.00027876890563499954,
              -2.3494801827922163e-05
            ]
          },
          "bce.buy": {
            "estimate": -0.00015900703219538272,
            "ci95": [
              -0.0002627709680272844,
              -6.927797194670761e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.0066231086753585355,
            "ci95": [
              -0.011533695283923126,
              -0.0011353336279964087
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.006471878400500453,
            "ci95": [
              -0.011905208017680921,
              -0.0016158808604171914
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.019275502693650748,
            "ci95": [
              -0.028212049972687375,
              -0.01127489370124524
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.00015123027485808294,
            "ci95": [
              -0.007923207077528121,
              0.0075499568423159835
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.012652394018292212,
            "ci95": [
              -0.021658082847423314,
              -0.004091622511119859
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.012803624293150295,
            "ci95": [
              -0.023494868305987883,
              -0.003074575160038605
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00020561456500483785,
            "ci95": [
              -0.0003275226175816069,
              -8.25975591188886e-05
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.00014410806914032584,
            "ci95": [
              -0.00029007771344604223,
              -1.0609476578977135e-05
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0001416758308231316,
            "ci95": [
              -0.0002514287152237968,
              -8.458487531080583e-06
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 6.150649586451201e-05,
            "ci95": [
              -9.927985125615153e-05,
              0.00022422387208212994
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 6.393873418170624e-05,
            "ci95": [
              -5.434447660740494e-05,
              0.00019175779436931406
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 2.4322383171942324e-06,
            "ci95": [
              -0.00014501811988607482,
              0.0001689574200345717
            ]
          }
        }
      },
      "even_last__minus__first_odd": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "first_even": 0.0,
          "first_odd": -1.0,
          "even_last": 1.0,
          "odd_last": 0.0,
          "odd_no_last": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.00019742593015811827,
            "ci95": [
              -0.00291896933166691,
              0.002967311915196923
            ]
          },
          "auc.is_click": {
            "estimate": 0.0019072784629866613,
            "ci95": [
              0.0011534841584818684,
              0.002744562744140996
            ]
          },
          "auc.cart": {
            "estimate": 0.0030092296845055655,
            "ci95": [
              -0.0015420489272802483,
              0.008176627484727441
            ]
          },
          "auc.fav": {
            "estimate": -0.0016949646330552959,
            "ci95": [
              -0.008060684353730233,
              0.004910971391264884
            ]
          },
          "auc.buy": {
            "estimate": -0.002431839793804458,
            "ci95": [
              -0.01084644162151463,
              0.0063265318004571405
            ]
          },
          "bce.is_click": {
            "estimate": -0.0006263393961413177,
            "ci95": [
              -0.0007012003145674649,
              -0.0005566372030752241
            ]
          },
          "bce.cart": {
            "estimate": -6.0599232676046555e-05,
            "ci95": [
              -0.00016703761769294194,
              3.744087043746033e-05
            ]
          },
          "bce.fav": {
            "estimate": 3.310533528870507e-05,
            "ci95": [
              -0.00010579293519783839,
              0.00020149752318511876
            ]
          },
          "bce.buy": {
            "estimate": 0.00020437633131416624,
            "ci95": [
              0.00010775245490309547,
              0.0003021976229631235
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.0011019512215189042,
            "ci95": [
              -0.006471995430289429,
              0.0034524857512280094
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.003602243096041957,
            "ci95": [
              -0.002794724295105286,
              0.010023987828254135
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.004339118256791119,
            "ci95": [
              -0.004703693927290422,
              0.012866205877300625
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.004704194317560861,
            "ci95": [
              -0.0028216205225555235,
              0.01219909592072245
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.005441069478310023,
            "ci95": [
              -0.0035462768337756355,
              0.015468588343463752
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.0007368751607491619,
            "ci95": [
              -0.010670700188193686,
              0.010985402526113346
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.0005657401634652712,
            "ci95": [
              -0.0006919567381202721,
              -0.0004426778696292173
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0006594447314300228,
            "ci95": [
              -0.0008438239668893093,
              -0.0005104081336903469
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.000830715727455484,
            "ci95": [
              -0.0009608399568234089,
              -0.0007137596670186466
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -9.370456796475163e-05,
            "ci95": [
              -0.00028255134185556,
              7.389880660293777e-05
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -0.0002649755639902128,
            "ci95": [
              -0.00040995058381899913,
              -0.00013504179010733557
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.00017127099602546117,
            "ci95": [
              -0.00035054512972459685,
              1.5082800490343688e-05
            ]
          }
        }
      },
      "even_last__minus__first_even": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "first_even": -1.0,
          "first_odd": 0.0,
          "even_last": 1.0,
          "odd_last": 0.0,
          "odd_no_last": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.006110436434240668,
            "ci95": [
              0.0030009711184742992,
              0.008898163943809555
            ]
          },
          "auc.is_click": {
            "estimate": 0.0015932945204952809,
            "ci95": [
              0.0007805681713853208,
              0.0023433264021248067
            ]
          },
          "auc.cart": {
            "estimate": 0.004624355795537993,
            "ci95": [
              0.00022657083021415957,
              0.009534227629930978
            ]
          },
          "auc.fav": {
            "estimate": 0.009632673702008376,
            "ci95": [
              0.003527729874413302,
              0.015253232266437534
            ]
          },
          "auc.buy": {
            "estimate": 0.008591421718921022,
            "ci95": [
              -0.000123666636549163,
              0.01785934019705183
            ]
          },
          "bce.is_click": {
            "estimate": -0.00022626341112624848,
            "ci95": [
              -0.00028936276048256574,
              -0.00015830778726652064
            ]
          },
          "bce.cart": {
            "estimate": -5.063102252095275e-05,
            "ci95": [
              -0.00015618610047910586,
              4.933020196008201e-05
            ]
          },
          "bce.fav": {
            "estimate": -0.00042742887525293316,
            "ci95": [
              -0.0005739653550609939,
              -0.000258325480579078
            ]
          },
          "bce.buy": {
            "estimate": -3.159325040598404e-05,
            "ci95": [
              -0.00013088107890427443,
              6.262379944687744e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.0030310612750427124,
            "ci95": [
              -0.007940159240977306,
              0.0016657800098574007
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.008039379181513095,
            "ci95": [
              -0.013570485529107798,
              -0.001990146563117324
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.006998127198425741,
            "ci95": [
              -0.016288870343030795,
              0.0014967671928087295
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.005008317906470383,
            "ci95": [
              -0.012056143673880915,
              0.0017386134235203526
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.0039670659233830285,
            "ci95": [
              -0.015589264422357895,
              0.005277688908229845
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.0010412519830873546,
            "ci95": [
              -0.010289028430425184,
              0.011170525167487146
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00017563238860529573,
            "ci95": [
              -0.00030574931646270437,
              -5.282474492366662e-05
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.00020116546412668468,
            "ci95": [
              2.863146876667832e-05,
              0.0003501222846257417
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.00019467016072026444,
            "ci95": [
              -0.000306928329246957,
              -7.515029036603575e-05
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.0003767978527319804,
            "ci95": [
              0.00020728316820134248,
              0.0005559127837352972
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -1.903777211496871e-05,
            "ci95": [
              -0.00015211425482028004,
              0.00011836046428041656
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.0003958356248469491,
            "ci95": [
              -0.0005783275611959145,
              -0.00021573388341951676
            ]
          }
        }
      },
      "odd_last__minus__even_last": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "first_even": 0.0,
          "first_odd": 0.0,
          "even_last": -1.0,
          "odd_last": 1.0,
          "odd_no_last": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.006220056195469698,
            "ci95": [
              -0.009240939085681788,
              -0.0031071687483180525
            ]
          },
          "auc.is_click": {
            "estimate": -0.0032958955647868216,
            "ci95": [
              -0.00406758554726667,
              -0.0024570990797073977
            ]
          },
          "auc.cart": {
            "estimate": -0.011756235384419389,
            "ci95": [
              -0.017160798941470608,
              -0.006478827673708816
            ]
          },
          "auc.fav": {
            "estimate": -0.005594096877898913,
            "ci95": [
              -0.011064314241648235,
              5.327240608569964e-05
            ]
          },
          "auc.buy": {
            "estimate": -0.004233996954773667,
            "ci95": [
              -0.011774306105869891,
              0.003167522045830747
            ]
          },
          "bce.is_click": {
            "estimate": 0.00018439309311732432,
            "ci95": [
              0.00010652105047450692,
              0.0002497649178231032
            ]
          },
          "bce.cart": {
            "estimate": 0.00022290789978268027,
            "ci95": [
              0.00010641322557621023,
              0.0003478381819178856
            ]
          },
          "bce.fav": {
            "estimate": 0.000175269799324404,
            "ci95": [
              5.797801335367678e-05,
              0.0002964311634848277
            ]
          },
          "bce.buy": {
            "estimate": -8.436236818961176e-05,
            "ci95": [
              -0.00017666996656672218,
              2.4055018028015115e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.008460339819632567,
            "ci95": [
              0.0031347724061119935,
              0.013931951289391557
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.0022982013131120915,
            "ci95": [
              -0.0036093566253435653,
              0.008049095647509339
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.0009381013899868451,
            "ci95": [
              -0.0065143818624461836,
              0.008384298527251938
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.006162138506520476,
            "ci95": [
              -0.013464313466846527,
              0.0011678356411724106
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.007522238429645722,
            "ci95": [
              -0.0156777024332632,
              0.0006383764533144631
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.0013600999231252464,
            "ci95": [
              -0.010588708201506147,
              0.007500131262991794
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -3.851480666535595e-05,
            "ci95": [
              -0.00018568517776988737,
              0.00010034540563336986
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 9.123293792920317e-06,
            "ci95": [
              -0.00012178303075282586,
              0.0001357669993991026
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.0002687554613069361,
            "ci95": [
              0.00014167360889683094,
              0.0003801107486412279
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 4.763810045827627e-05,
            "ci95": [
              -0.00010879009958681494,
              0.0002021626702406259
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00030727026797229204,
            "ci95": [
              0.0001617854032375056,
              0.0004525094165045366
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.00025963216751401577,
            "ci95": [
              0.00010924886257970158,
              0.0004071506972521106
            ]
          }
        }
      },
      "endpoint_effect_odd_minus_even": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "first_even": 1.0,
          "first_odd": 0.0,
          "even_last": -1.0,
          "odd_last": 1.0,
          "odd_no_last": -1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.003683270221179119,
            "ci95": [
              -0.008154072547029424,
              0.0007703183683111161
            ]
          },
          "auc.is_click": {
            "estimate": -0.004076195972098073,
            "ci95": [
              -0.005138171365454964,
              -0.0029189230819194416
            ]
          },
          "auc.cart": {
            "estimate": -0.0049367021002625355,
            "ci95": [
              -0.011334951614394375,
              0.0014201582743916606
            ]
          },
          "auc.fav": {
            "estimate": 4.225693741344294e-05,
            "ci95": [
              -0.008565787371336517,
              0.008056617857407656
            ]
          },
          "auc.buy": {
            "estimate": -0.00576243974976931,
            "ci95": [
              -0.01966413203340297,
              0.007041952743169709
            ]
          },
          "bce.is_click": {
            "estimate": 0.00021533378919871837,
            "ci95": [
              0.00011463617265766863,
              0.00031803043488899105
            ]
          },
          "bce.cart": {
            "estimate": 0.00016970668512293965,
            "ci95": [
              2.7375774076672784e-05,
              0.00031731075563044646
            ]
          },
          "bce.fav": {
            "estimate": 0.0005001475323210165,
            "ci95": [
              0.0003155640213035646,
              0.0006885420341292617
            ]
          },
          "bce.buy": {
            "estimate": -3.1977866183099946e-05,
            "ci95": [
              -0.00016131817773513756,
              0.00011875046826143662
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.0008605061281644621,
            "ci95": [
              -0.005591605800147853,
              0.00727056389696085
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.004118452909511516,
            "ci95": [
              -0.012181535372450713,
              0.004783261492760941
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.001686243777671237,
            "ci95": [
              -0.011044793199981728,
              0.015362051870597757
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.0049789590376759785,
            "ci95": [
              -0.015042396996486393,
              0.005805849650137093
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.0008257376495067748,
            "ci95": [
              -0.012498625221194664,
              0.016245006850071447
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.005804696687182753,
            "ci95": [
              -0.010390286105382578,
              0.022268230642195297
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 4.562710407577872e-05,
            "ci95": [
              -0.00012962927552175916,
              0.000211161259525985
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.00028481374312229815,
            "ci95": [
              -0.0004877340199523501,
              -7.288225907590442e-05
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.0002473116553818183,
            "ci95": [
              7.964917619699437e-05,
              0.0004079337019393981
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.00033044084719807687,
            "ci95": [
              -0.000571024754572906,
              -0.00010143940421444398
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.0002016845513060396,
            "ci95": [
              1.5228246296311554e-05,
              0.0003954909589620846
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.0005321253985041165,
            "ci95": [
              0.0002920697624310177,
              0.0007621072960968108
            ]
          }
        }
      }
    },
    "parameter_count": 51724260,
    "initializer_sha256": "fd5f866c35c5e29b37841aeed070f6ef5259c773de173df7549d149f87a8c09e",
    "optimizer_layout_sha256": "cf6764857f1fdb70b39086008f5c8c5f85c5c9921f6a2422a0f5edc0470877c3",
    "formal_worker_cost": {
      "wall_seconds": 2504.268677316606,
      "cpu_seconds": 3004.7624286259997,
      "peak_rss_bytes": 17043628032,
      "gpu_peak_bytes": 28369223680,
      "samples": 2296
    },
    "smoke_worker_cost": {
      "wall_seconds": 142.17819248884916,
      "cpu_seconds": 155.180190424,
      "peak_rss_bytes": 16914710528,
      "gpu_peak_bytes": 29572988928,
      "samples": 131
    },
    "complete_recorded_costs": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2504.268677316606,
        "cpu_seconds": 3004.7624286259997,
        "peak_rss_bytes": 17043628032,
        "gpu_peak_bytes": 28369223680,
        "samples": 2296
      },
      "formal_runner": {
        "wall_seconds": 2500.447254151106,
        "cpu_seconds": 2895.7986620089996,
        "peak_process_rss_bytes": 5262491648
      },
      "fit_including_validation": {
        "wall_seconds": 2376.759330958128,
        "cpu_seconds": 2722.683524909,
        "peak_process_rss_bytes": 5262491648
      },
      "prediction_export": {
        "wall_seconds": 118.8226051479578,
        "cpu_seconds": 167.5417705169998,
        "peak_process_rss_bytes": 5262491648
      },
      "smoke_worker": {
        "wall_seconds": 142.17819248884916,
        "cpu_seconds": 155.180190424,
        "peak_rss_bytes": 16914710528,
        "gpu_peak_bytes": 29572988928,
        "samples": 131
      },
      "smoke_runner": {
        "wall_seconds": 138.36269782483578,
        "cpu_seconds": 137.21933768600002,
        "peak_process_rss_bytes": 3236663296
      },
      "latency_constructor": {
        "wall_seconds": 0.7623036354780197,
        "cpu_seconds": 0.8318260120000005,
        "peak_rss_bytes": 1530429440
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.16031786799430847,
        "cpu_seconds": 0.1597225790000003,
        "peak_rss_bytes": 1530429440
      },
      "latency_peak_cuda_bytes": 1280827392,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "forward_latency": {
      "mean_ms": 14.499318432807922,
      "p50_ms": 14.477983951568604,
      "p95_ms": 14.614841413497924,
      "batch_size": 1024,
      "warmups": 10,
      "iterations": 40
    },
    "latency_ratio_to_parent": 1.0060710164746387,
    "quality_candidate": true,
    "near_candidate": true
  },
  "validity": {
    "valid": true,
    "violations": [],
    "state": "passed",
    "quality_summary": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/quality-summary.json",
    "no_prediction_to_work_feedback": true,
    "test_evaluated": false,
    "infrastructure_limitations": [
      "Source metrics are identity/range/arithmetic checks, not recomputed AUC/BCE.",
      "Actual saved-array pairing is deferred to the paired analysis; row fingerprints cover original payload user/item/labels.",
      "Production cost receipts are sampled evidence, not new GPU measurements or hard resource guarantees.",
      "Maturity timestamp inference and click-availability assumptions remain unchanged.",
      "Two segment references have independently recovered real-worker terminals; original aggregate launcher terminal/cost unavailable.",
      "Same GPU and exact input; references and new models measured in separate execution windows after import failure. Original failure and timings retained.",
      "Original paired process terminated by failed latency pipeline; completed draws unknown. Recovered500-draw result preserves same protocol."
    ],
    "measurement_recovery_evidence": {
      "pipeline_failure": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/measurement-pipeline/failure.json",
      "paired_recovery": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/paired-recovery-review.json",
      "prior_latency_failure": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/latency/failure.json"
    },
    "new_terminal_evidence": {
      "completion_path": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/launch-formal/completion.json",
      "completion": {
        "state": "completed",
        "phase": "formal",
        "launcher_exit_code": 0,
        "worker_exit_codes": {
          "even_last": 0,
          "odd_last": 0,
          "odd_no_last": 0
        },
        "authorization_sha256": "23b18e68185190f0e7f9130a18b60cb4c750eb004ce3da407cdd7ca662c5bbf4",
        "smoke_review_sha256": "08540d3f96b3311ad2e13ec1691b525d8fe5feb3682777c7fd3a802b9316d334",
        "worker_receipt_paths": {
          "even_last": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/even_last/formal/exit.json",
          "odd_last": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/odd_last/formal/exit.json",
          "odd_no_last": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/odd_no_last/formal/exit.json"
        },
        "worker_receipt_sha256": {
          "even_last": "42ec66c3cf5be09d64964b46f48b417146a9c546991339f3f9b37229cdcc152d",
          "odd_last": "681b95cf3e65aa1b591cb92612c06ec9a1c0ec5a2e864918f425d5ac0d7e4fb0",
          "odd_no_last": "6143947e9693d9060ac0be95b7a7984bca8a7af90bf2b07ecf8d9c858da8c47b"
        },
        "error": null,
        "receipt_errors": [],
        "deadline_exceeded": false,
        "signals": [],
        "cleanup": {
          "completed": true,
          "signals": [],
          "reaped_adoptees": [],
          "remaining": [],
          "wall_seconds": 0.015955939888954163,
          "subreaper": true,
          "pidfd_identity_checked": false,
          "signalling_methods": [],
          "fallback_scope": "ENOSYS only: fresh owned PPID chain and start-time checks before os.kill; these checks and kill are not atomic like pidfd."
        },
        "wall_seconds": 2507.1962953284383,
        "launcher_wall_limit_seconds": 7320,
        "cpu_seconds_including_reaped_children": 9020.509701064,
        "launcher_peak_rss_bytes": 132202496,
        "worker_costs": {
          "even_last": {
            "wall_seconds": 2504.268677316606,
            "cpu_seconds": 3004.7624286259997,
            "peak_rss_bytes": 17043628032,
            "gpu_peak_bytes": 28369223680,
            "samples": 2296
          },
          "odd_last": {
            "wall_seconds": 2506.589051552117,
            "cpu_seconds": 3006.224509484,
            "peak_rss_bytes": 17033830400,
            "gpu_peak_bytes": 28369223680,
            "samples": 2298
          },
          "odd_no_last": {
            "wall_seconds": 2506.5823511853814,
            "cpu_seconds": 3007.4703000850004,
            "peak_rss_bytes": 17018593280,
            "gpu_peak_bytes": 28369223680,
            "samples": 2297
          }
        },
        "automatic_retry": false,
        "partial_preserved": true,
        "missing_workers": []
      },
      "worker_costs": {
        "even_last": {
          "wall_seconds": 2504.268677316606,
          "cpu_seconds": 3004.7624286259997,
          "peak_rss_bytes": 17043628032,
          "gpu_peak_bytes": 28369223680,
          "samples": 2296
        },
        "odd_last": {
          "wall_seconds": 2506.589051552117,
          "cpu_seconds": 3006.224509484,
          "peak_rss_bytes": 17033830400,
          "gpu_peak_bytes": 28369223680,
          "samples": 2298
        },
        "odd_no_last": {
          "wall_seconds": 2506.5823511853814,
          "cpu_seconds": 3007.4703000850004,
          "peak_rss_bytes": 17018593280,
          "gpu_peak_bytes": 28369223680,
          "samples": 2297
        }
      }
    },
    "inherited_reference_provenance": {
      "mature_r2": {
        "quality_path": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/quality-summary.json",
        "quality_sha256": "8012e1ea78f95e8d9e1fae8703bb4293ceccdb501b4b3995808698433a18abbd",
        "producer_small_artifact_sha256": {
          "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2/result.json": "15c18a7aee32a1dd3015c9a8c9613fe532887416cd49e5a063943f3eee4d246b",
          "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2/manifest.json": "a8439ee9ab331449eb1625f5822deb9a2ad15bf4ad5be8fefa451e78a17949de",
          "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/reports/mature_r2/formal/exit.json": "0a814029b8cedcea62429041dacff3eda1fa875e93151ed300e2151e3456cae7",
          "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2/initial-identity.json": "e1ecc656ac3fb59e413147bc94c59b8d3bfeb4a4808e7de257b5d6c8cf20500c",
          "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2/optimizer-layout.json": "2dfcb20341c7d6da0ff6080dd8a97d83dadfbead33e9f227c95a28054bf1edb6",
          "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2/optimizer-effective.json": "03048a8e046c10cbd16657171c94ab7fa0e79fe8367501548df3db312445f163",
          "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2/loader-preparation.json": "665d88436606d0df274dcc7eb96377720251ad9bdc7cc3ced6d5c207f2d84cd8",
          "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2/loader-audit.json": "d1a196f4a9f56af32e8256ca3231361d5871dd273cdcb7186f053e2f270a9bbf",
          "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2/derived-feature-map.json": "b0e310901463881c1bb3f6f0fad3fc0aa9648108effe45b1d9255f86584944e0",
          "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2/derived-metadata.json": "fd3fe51915b980646675e32f400514bec7e38313a74968bac7b4b079865efe8f",
          "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2/completion.json": "f3d6eedc37c7de8e4f26383a4b26f07881c6fd96e03a8d64a264fef012c8ca9e",
          "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/smoke-runs/mature_r2/smoke.json": "d034bd61e75061a211157047d5c4eac9d94559470ef60f348bb94f09223f81f9",
          "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/reports/mature_r2/smoke/exit.json": "dcae789a6a0c64dd982291e62eaab85cfce32f5758cb74fb0fabce223047cbdc"
        },
        "recorded_large_artifact_sha256": {
          "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2/Taobao_Action/mature_r2.model": "c514fb78a24c2bbf4460c65d5b6d97e642e5d9a2359be3f9b4dd43558f872ecd",
          "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2/validation_predictions.npz": "22ad8962837c30ed96704b979c8a7f983c422d9c3626b59b60a701b427553647"
        },
        "smoke_optimizer_initialization_scope": "Inherited successful C quality gates with immutable producer/source identities; original smoke/model execution is not repeated."
      },
      "cross8_mature": {
        "quality_path": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/quality-summary.json",
        "quality_sha256": "8012e1ea78f95e8d9e1fae8703bb4293ceccdb501b4b3995808698433a18abbd",
        "producer_small_artifact_sha256": {
          "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature/result.json": "e301e2959cd9d8aa5443bdc8ee9bdee02576bc62946241c34996435bf9ef6103",
          "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature/manifest.json": "f1a54a6ebd39c772cb1b858b69cb5a132d3b2773da264276c402eb689567e50f",
          "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/cross8_mature/formal/exit.json": "e726a6438fa2c9958595b167f608b369ab559c027705a2e8709f01d3e1b17fc7",
          "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature/initial-identity.json": "6d3e1eb30c86a01f3ce07c1f964738a166bb612a830d0e9d1baa26155b7662c3",
          "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature/optimizer-layout.json": "d1d5b6df938178a71a234b289de7f2c44ad75849abfedb8b8f43769923d56e43",
          "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature/optimizer-effective.json": "0672e2adadbdab668ce94f4bc8ec2d77484ab85d46315766defdafc6825feeb9",
          "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature/loader-preparation.json": "723475dbce92652a4e229e78aa543d55d30d300e077c18891514ef8942ccd474",
          "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature/loader-audit.json": "2c9df0264864bbd1b7e261738b364915c0779be40248abbaa808f55e1f1a2ee3",
          "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature/derived-feature-map.json": "b0e310901463881c1bb3f6f0fad3fc0aa9648108effe45b1d9255f86584944e0",
          "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature/derived-metadata.json": "fd3fe51915b980646675e32f400514bec7e38313a74968bac7b4b079865efe8f",
          "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature/completion.json": "eff96b7015a8a2dd28d68d4471e56e1ff6f8c464dcca63212c3aeadb9eb08cac",
          "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/smoke-runs/cross8_mature/smoke.json": "b63ff5e265f171d54a6865ee40a926cd63435dd7aff5e971c89e0be591434ba3",
          "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/cross8_mature/smoke/exit.json": "e2b92d7e21d9f0f2faacae0ddfb40ae2031ecf4981e15c147b0a172497b7b468"
        },
        "recorded_large_artifact_sha256": {
          "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature/Taobao_Action/cross8_mature.model": "e685bda693b75c6e838a36bfa48f7f74d0fdd6fcd50cb44c81c6858b578056ae",
          "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature/validation_predictions.npz": "7a7b7973ba25b8c52142cb6cd52c2b0683b8a16484b7bb9768c63fcd4b6e6790"
        },
        "smoke_optimizer_initialization_scope": "Inherited successful C quality gates with immutable producer/source identities; original smoke/model execution is not repeated."
      },
      "first_even": {
        "quality_path": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/quality-summary.json",
        "quality_sha256": "8012e1ea78f95e8d9e1fae8703bb4293ceccdb501b4b3995808698433a18abbd",
        "producer_small_artifact_sha256": {
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_even/result.json": "a6955c66720f6698a85c6b1b48224c385fb161428fdc39d5158b6393ed185145",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_even/manifest.json": "4a7e620352c18b79d89cf49cec6363400a000943f290d698a8af60cb812ec59a",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/first_even/formal/exit.json": "875ea534b3fdc2cd7e99939c6672bd2c96e4d7050b12b6ce9725e9d281d49038",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_even/initial-identity.json": "052c134c6b636ee83bfef2c6234e010b85064652155f3e90b157a67e8af0f86f",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_even/optimizer-layout.json": "96b185f1feaea60c6e693d7982052a79184d4ba57105acb11e45042625aaf67c",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_even/optimizer-effective.json": "03048a8e046c10cbd16657171c94ab7fa0e79fe8367501548df3db312445f163",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_even/loader-preparation.json": "cb0e949658ea71aa027eb0a1dc026034eab46bcb415e250b01d0c911f8ceb12f",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_even/loader-audit.json": "9d53916f0cb1983b952931c9dd61d1ab1090f8532274bddcde6f4a974479e514",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_even/derived-feature-map.json": "b0e310901463881c1bb3f6f0fad3fc0aa9648108effe45b1d9255f86584944e0",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_even/derived-metadata.json": "fd3fe51915b980646675e32f400514bec7e38313a74968bac7b4b079865efe8f",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_even/completion.json": "f0f1b7ffd576bdaf72664e492860f53f6984baab4876ac426f7d6596b00e61a0",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/smoke-runs/first_even/smoke.json": "e456922f23eed006b4ffa0d17be5e4c44c8cb8b275a899cec44933df4f4474f4",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/first_even/smoke/exit.json": "f1ff58f5be4b9f5eec70861ca8d7d43720bcf8f505c785f34f0fd24ec516a229"
        },
        "recorded_large_artifact_sha256": {
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_even/Taobao_Action/first_even.model": "a08f350cdda9524b5b64fa1ab257c999c89108bf79faa4003fcad72fef2bd760",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_even/validation_predictions.npz": "ba55d162846e963126fc5065052ebb72c7ee973ed17591110cb1f92315d9c27d"
        },
        "smoke_optimizer_initialization_scope": "Inherited successful C quality gates with immutable producer/source identities; original smoke/model execution is not repeated."
      },
      "first_odd": {
        "quality_path": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/quality-summary.json",
        "quality_sha256": "8012e1ea78f95e8d9e1fae8703bb4293ceccdb501b4b3995808698433a18abbd",
        "producer_small_artifact_sha256": {
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_odd/result.json": "9ca8426ec0ef9fc2f65fd48208f7869c55048dc044b94c3ce6c7d01e9c83c3de",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_odd/manifest.json": "62896430d504d095bd7c3bd08c54ce9da5935d8a5aa0e2b589f9709d2a4984ee",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/first_odd/formal/exit.json": "2830a5f1142efad9bf7d2870ec6ad384ae3468104a15e0f50dc0eada0146d589",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_odd/initial-identity.json": "2a73b409dbaee134ab3b7a734927528aa9aa7ce94765ae91763348c13b6ce229",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_odd/optimizer-layout.json": "f92b61b437f3ea0e8610574e1546bf4af731ed57ab9ce96872c4cb87f94763f1",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_odd/optimizer-effective.json": "03048a8e046c10cbd16657171c94ab7fa0e79fe8367501548df3db312445f163",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_odd/loader-preparation.json": "9244f43791a5c92b7d35dd9bb823f8eb5fd32ed5f699f7d71e7583f49e70bdbe",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_odd/loader-audit.json": "5b532fca1c299eed13f345aff211674c029fdfb2ef39005608dc01184cf15572",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_odd/derived-feature-map.json": "b0e310901463881c1bb3f6f0fad3fc0aa9648108effe45b1d9255f86584944e0",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_odd/derived-metadata.json": "fd3fe51915b980646675e32f400514bec7e38313a74968bac7b4b079865efe8f",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_odd/completion.json": "c81c54df3a4e54751d80a18b95eec38e273e22536e9cf1edc75b0c181d5a01ce",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/smoke-runs/first_odd/smoke.json": "4388338f9a9faab0035065a911a6a6ee5ecf8f5d85d6dd629d133929509bc8a5",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/first_odd/smoke/exit.json": "cd8760337a84b26c3958e18e48cc8cdc0d34cf93643b4166741ff28428c3ac17"
        },
        "recorded_large_artifact_sha256": {
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_odd/Taobao_Action/first_odd.model": "e2123b38b543c10bba28b4ebbd45f558185c3701314463e91e03abcb3197dd1f",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_odd/validation_predictions.npz": "fead70989033318cb4ad5a6c5fda322eeda531662d49a0c244b86b5a59c330a5"
        },
        "smoke_optimizer_initialization_scope": "Inherited successful C quality gates with immutable producer/source identities; original smoke/model execution is not repeated."
      }
    },
    "source_and_input_identity_scope": "Original N full-input content SHA256 values are inherited through completed C quality and E authorization. Production runners checked input stats before/after their traversals and two original metadata hashes before them. This collector now checks 122 current stats and both original metadata hashes; it does not rehash Parquet or open checkpoint/prediction arrays. New E worker/smoke/effective-optimizer gates are evaluated from actual small receipts. Two references retain exact C quality records and producer completion hashes, with small source/artifact identity checks. Stat continuity cannot independently exclude content changes that preserve stat fields."
  },
  "decision": "keep",
  "decision_reason": "Macro delta +0.001435345,CI95 [-0.001703624913421238, 0.004117847779228911];latency ratio1.006071. Research candidate only, macroCIcrosszero andbuyBCEregression;noincumbentpromotion.",
  "decision_scope": "Root queue disposition only; no released-incumbent promotion or supported lesson.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/runs/even_last",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/runs/even_last/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/runs/even_last/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/runs/even_last/Taobao_Action/even_last.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/paired-analysis-recovery/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/latency-recovery/completion.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/runs/even_last",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/runs/even_last/result.json",
    "result_sha256": "369dd225e41698d50a0ec225357d01f4c330497aae2a1917c89f2d7192ed3d69",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/runs/even_last/manifest.json",
    "manifest_sha256": "7825c5671ac2a0e9b3371633aa68b11b62861d19e95f4e8ab6399e80d432f672",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/runs/even_last/Taobao_Action/even_last.model",
    "checkpoint_sha256": "04f96743908689a6d49c91f76b64cd4dfe9697bc15bcec48ffd9afa890732e85",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/paired-analysis-recovery/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
