## 252. segment-read--first_even--s20262027

- 假设：Equal-count first_even stored-history support may change useful two-round retrieval conditions; no pure Q2 mediation claim.
- 判定：discard
- 原因：Macro delta -0.004675091,CI95 [-0.007934792701201314, -0.0017019098487541366],latency ratio 1.013763. Fails positive and near-candidate quality policy; preserve negative evidence.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "segment-read--first_even--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-10T14:13:03.864962+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "first_even",
  "code_id": "cf7b0840f7d3f28bb5451ac3cc4d9491bfcf60c5a00b32fc75f78b7e407fc0de",
  "hypothesis": "Equal-count first_even stored-history support may change useful two-round retrieval conditions; no pure Q2 mediation claim.",
  "change": "{\"model\": \"QFormerCross8SegmentRead\", \"maturity_mode\": \"mature\", \"segment_placement\": \"first\", \"segment_selection\": \"even\", \"work_steps\": 2, \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8SegmentRead",
    "maturity_mode": "mature",
    "segment_placement": "first",
    "segment_selection": "even",
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
      "value": 0.7624198602259524,
      "delta": -0.004675091088908068
    },
    "per_task": {
      "is_click": {
        "logloss": 0.184243518396697,
        "AUC": 0.6493747297448262
      },
      "cart": {
        "logloss": 0.02178997429391363,
        "AUC": 0.7922367423732923
      },
      "fav": {
        "logloss": 0.017244233262799377,
        "AUC": 0.7943724698482975
      },
      "buy": {
        "logloss": 0.009879976545092569,
        "AUC": 0.8136954989373937
      }
    },
    "macro_source_logloss": 0.05828942562462564,
    "source_metric_comparison": {
      "is_click": {
        "source_AUC": 0.6493747297448262,
        "computed_AUC": 0.6493747297448262,
        "auc_difference": 0.0,
        "source_logloss": 0.184243518396697,
        "computed_bce": 0.18424351839669909,
        "bce_difference": 2.0816681711721685e-15
      },
      "cart": {
        "source_AUC": 0.7922367423732923,
        "computed_AUC": 0.7922367423732923,
        "auc_difference": 0.0,
        "source_logloss": 0.02178997429391363,
        "computed_bce": 0.021789974361586306,
        "bce_difference": 6.767267649032682e-11
      },
      "fav": {
        "source_AUC": 0.7943724698482975,
        "computed_AUC": 0.7943724698482975,
        "auc_difference": 0.0,
        "source_logloss": 0.017244233262799377,
        "computed_bce": 0.017244233262827792,
        "bce_difference": 2.84147705365001e-14
      },
      "buy": {
        "source_AUC": 0.8136954989373937,
        "computed_AUC": 0.8136954989373937,
        "auc_difference": 0.0,
        "source_logloss": 0.009879976545092569,
        "computed_bce": 0.00987997654664873,
        "bce_difference": 1.5561614497006104e-12
      }
    },
    "matched_parent_effect": {
      "coefficients": {
        "mature_r2": -1.0,
        "cross8_mature": 0.0,
        "first_prefix": 0.0,
        "first_suffix": 0.0,
        "first_even": 1.0,
        "first_odd": 0.0,
        "second_prefix": 0.0,
        "second_suffix": 0.0,
        "second_even": 0.0,
        "second_odd": 0.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": -0.004675091088908068,
          "ci95": [
            -0.007934792701201314,
            -0.0017019098487541366
          ]
        },
        "auc.is_click": {
          "estimate": -0.0005615634561632055,
          "ci95": [
            -0.0012522365692116955,
            0.00022492579408634467
          ]
        },
        "auc.cart": {
          "estimate": -0.00019752798405137106,
          "ci95": [
            -0.004400183926438555,
            0.003803017558635711
          ]
        },
        "auc.fav": {
          "estimate": -0.008091020526750592,
          "ci95": [
            -0.01508165035763837,
            -0.0009864809615284993
          ]
        },
        "auc.buy": {
          "estimate": -0.009850252388667102,
          "ci95": [
            -0.020311750167241557,
            -0.0013515262862966036
          ]
        },
        "bce.is_click": {
          "estimate": -0.00014812787108758885,
          "ci95": [
            -0.00022340861043707375,
            -8.095711239884397e-05
          ]
        },
        "bce.cart": {
          "estimate": -3.79800078477259e-05,
          "ci95": [
            -0.0001473750153537655,
            6.28343541408326e-05
          ]
        },
        "bce.fav": {
          "estimate": 0.0004460755684754583,
          "ci95": [
            0.000281911068520524,
            0.0006086841064344137
          ]
        },
        "bce.buy": {
          "estimate": 0.00023617460992302747,
          "ci95": [
            0.00014668406673696248,
            0.0003360615986810999
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": -0.0003640354721118344,
          "ci95": [
            -0.004581472685856744,
            0.0039148577824244385
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": 0.0075294570705873864,
          "ci95": [
            0.00057020325326726,
            0.014420758216418467
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": 0.009288688932503897,
          "ci95": [
            0.0008069593852173235,
            0.019930475236818724
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": 0.00789349254269922,
          "ci95": [
            0.00034881756228553076,
            0.01597497218946204
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": 0.009652724404615731,
          "ci95": [
            0.0003903077988263739,
            0.021153848899094302
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": 0.0017592318619165104,
          "ci95": [
            -0.009260131656547863,
            0.014973712891287209
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": -0.00011014786323986295,
          "ci95": [
            -0.00024204515564769146,
            2.6737966081772118e-05
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": -0.0005942034395630472,
          "ci95": [
            -0.0007727907063133683,
            -0.0004174581379296865
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": -0.0003843024810106163,
          "ci95": [
            -0.000527792081909896,
            -0.00027934547228452455
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": -0.0004840555763231842,
          "ci95": [
            -0.0006740775220812002,
            -0.00030894771878067724
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": -0.00027415461777075337,
          "ci95": [
            -0.00042762912817144655,
            -0.00014330038261839843
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": 0.00020990095855243085,
          "ci95": [
            1.9725755015333367e-05,
            0.00040585872340236946
          ]
        }
      }
    },
    "all_prespecified_effects_involving_candidate": {
      "first_even__minus__mature_r2": {
        "coefficients": {
          "mature_r2": -1.0,
          "cross8_mature": 0.0,
          "first_prefix": 0.0,
          "first_suffix": 0.0,
          "first_even": 1.0,
          "first_odd": 0.0,
          "second_prefix": 0.0,
          "second_suffix": 0.0,
          "second_even": 0.0,
          "second_odd": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.004675091088908068,
            "ci95": [
              -0.007934792701201314,
              -0.0017019098487541366
            ]
          },
          "auc.is_click": {
            "estimate": -0.0005615634561632055,
            "ci95": [
              -0.0012522365692116955,
              0.00022492579408634467
            ]
          },
          "auc.cart": {
            "estimate": -0.00019752798405137106,
            "ci95": [
              -0.004400183926438555,
              0.003803017558635711
            ]
          },
          "auc.fav": {
            "estimate": -0.008091020526750592,
            "ci95": [
              -0.01508165035763837,
              -0.0009864809615284993
            ]
          },
          "auc.buy": {
            "estimate": -0.009850252388667102,
            "ci95": [
              -0.020311750167241557,
              -0.0013515262862966036
            ]
          },
          "bce.is_click": {
            "estimate": -0.00014812787108758885,
            "ci95": [
              -0.00022340861043707375,
              -8.095711239884397e-05
            ]
          },
          "bce.cart": {
            "estimate": -3.79800078477259e-05,
            "ci95": [
              -0.0001473750153537655,
              6.28343541408326e-05
            ]
          },
          "bce.fav": {
            "estimate": 0.0004460755684754583,
            "ci95": [
              0.000281911068520524,
              0.0006086841064344137
            ]
          },
          "bce.buy": {
            "estimate": 0.00023617460992302747,
            "ci95": [
              0.00014668406673696248,
              0.0003360615986810999
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.0003640354721118344,
            "ci95": [
              -0.004581472685856744,
              0.0039148577824244385
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.0075294570705873864,
            "ci95": [
              0.00057020325326726,
              0.014420758216418467
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.009288688932503897,
            "ci95": [
              0.0008069593852173235,
              0.019930475236818724
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.00789349254269922,
            "ci95": [
              0.00034881756228553076,
              0.01597497218946204
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.009652724404615731,
            "ci95": [
              0.0003903077988263739,
              0.021153848899094302
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.0017592318619165104,
            "ci95": [
              -0.009260131656547863,
              0.014973712891287209
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00011014786323986295,
            "ci95": [
              -0.00024204515564769146,
              2.6737966081772118e-05
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0005942034395630472,
            "ci95": [
              -0.0007727907063133683,
              -0.0004174581379296865
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0003843024810106163,
            "ci95": [
              -0.000527792081909896,
              -0.00027934547228452455
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.0004840555763231842,
            "ci95": [
              -0.0006740775220812002,
              -0.00030894771878067724
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -0.00027415461777075337,
            "ci95": [
              -0.00042762912817144655,
              -0.00014330038261839843
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.00020990095855243085,
            "ci95": [
              1.9725755015333367e-05,
              0.00040585872340236946
            ]
          }
        }
      },
      "first_even__minus__cross8_mature": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": -1.0,
          "first_prefix": 0.0,
          "first_suffix": 0.0,
          "first_even": 1.0,
          "first_odd": 0.0,
          "second_prefix": 0.0,
          "second_suffix": 0.0,
          "second_even": 0.0,
          "second_odd": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.003013440130492595,
            "ci95": [
              -0.00012574522496085116,
              0.006000488575488294
            ]
          },
          "auc.is_click": {
            "estimate": -0.0005620403981394517,
            "ci95": [
              -0.0013886967646070695,
              0.000229221095966106
            ]
          },
          "auc.cart": {
            "estimate": 0.0030300070021763714,
            "ci95": [
              -0.0012143641527789933,
              0.0077682888822951995
            ]
          },
          "auc.fav": {
            "estimate": -0.0021295411791520946,
            "ci95": [
              -0.007612616360471896,
              0.004247756735170358
            ]
          },
          "auc.buy": {
            "estimate": 0.011715335097085555,
            "ci95": [
              0.0026911608505887806,
              0.019809256990079054
            ]
          },
          "bce.is_click": {
            "estimate": -7.441945189226584e-05,
            "ci95": [
              -0.0001521687334203721,
              -3.6197772321805794e-06
            ]
          },
          "bce.cart": {
            "estimate": -4.443727549272372e-05,
            "ci95": [
              -0.00014075050806372277,
              4.939060862054347e-05
            ]
          },
          "bce.fav": {
            "estimate": 0.0002708540813747447,
            "ci95": [
              0.0001570380464535316,
              0.00038275870296018043
            ]
          },
          "bce.buy": {
            "estimate": -0.00012741378178939868,
            "ci95": [
              -0.0002127583008865524,
              -3.891422529346811e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.003592047400315823,
            "ci95": [
              -0.008334168910961389,
              0.00042816256501075427
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.001567500781012643,
            "ci95": [
              -0.0050707865459233,
              0.007111395380299359
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.012277375495225007,
            "ci95": [
              -0.02043065223973481,
              -0.003057629409317287
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.005159548181328466,
            "ci95": [
              -0.0020324824697783017,
              0.011956078076513655
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.008685328094909184,
            "ci95": [
              -0.018722155428350006,
              0.0008839926840402988
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.01384487627623765,
            "ci95": [
              -0.023766081860319966,
              -0.004074882715996832
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -2.998217639954212e-05,
            "ci95": [
              -0.00014071938110278275,
              9.464603318634842e-05
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0003452735332670105,
            "ci95": [
              -0.00046896379933174915,
              -0.0002218827428164402
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 5.299432989713283e-05,
            "ci95": [
              -6.702977455471343e-05,
              0.00017916341149141088
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.0003152913568674684,
            "ci95": [
              -0.0004520540326435903,
              -0.00017991964452730323
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 8.297650629667495e-05,
            "ci95": [
              -4.5753326174067424e-05,
              0.0002118919461998303
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.00039826786316414335,
            "ci95": [
              0.00026652055217331815,
              0.0005447160207937965
            ]
          }
        }
      },
      "first_even__minus__second_even": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "first_prefix": 0.0,
          "first_suffix": 0.0,
          "first_even": 1.0,
          "first_odd": 0.0,
          "second_prefix": 0.0,
          "second_suffix": 0.0,
          "second_even": -1.0,
          "second_odd": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.002567328425599563,
            "ci95": [
              -0.005659813985604775,
              0.0002909665130584482
            ]
          },
          "auc.is_click": {
            "estimate": -0.0014023571508540167,
            "ci95": [
              -0.0021374674162870865,
              -0.00068201330595403
            ]
          },
          "auc.cart": {
            "estimate": -0.004134619841509535,
            "ci95": [
              -0.009278762599012192,
              0.0007151527356667125
            ]
          },
          "auc.fav": {
            "estimate": 0.0005621273148294081,
            "ci95": [
              -0.005156703877462368,
              0.005918813556508139
            ]
          },
          "auc.buy": {
            "estimate": -0.0052944640248641095,
            "ci95": [
              -0.014458751808165707,
              0.002426139184592045
            ]
          },
          "bce.is_click": {
            "estimate": -0.00013930121746913438,
            "ci95": [
              -0.00020565291854803268,
              -7.07663032622083e-05
            ]
          },
          "bce.cart": {
            "estimate": -0.00016449481844903885,
            "ci95": [
              -0.0002888328805695906,
              -4.776001094618777e-05
            ]
          },
          "bce.fav": {
            "estimate": 0.0001636146130646124,
            "ci95": [
              6.084866939648224e-05,
              0.0002785809234801293
            ]
          },
          "bce.buy": {
            "estimate": 8.550474273976566e-05,
            "ci95": [
              -2.455409506829089e-06,
              0.00018165475654635007
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.002732262690655518,
            "ci95": [
              -0.0022063377661752175,
              0.0076030610060478735
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.001964484465683425,
            "ci95": [
              -0.007448378907638822,
              0.003717684097976286
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.003892106874010093,
            "ci95": [
              -0.003952115107100853,
              0.012875309531047345
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.004696747156338943,
            "ci95": [
              -0.012164965191136018,
              0.0024164597494997246
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.0011598441833545747,
            "ci95": [
              -0.007964288727586538,
              0.010380379604470615
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.005856591339693518,
            "ci95": [
              -0.0044646313506372095,
              0.01630803005746252
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 2.5193600979904474e-05,
            "ci95": [
              -0.00010504964281544378,
              0.0001600891181610679
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.00030291583053374677,
            "ci95": [
              -0.0004290305370822516,
              -0.00017803790079869942
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.00022480596020890004,
            "ci95": [
              -0.00034000348236008315,
              -0.00011165519158080348
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.00032810943151365124,
            "ci95": [
              -0.0004884613914593895,
              -0.00017169771589222103
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -0.0002499995611888045,
            "ci95": [
              -0.00040067745170644243,
              -0.00010216579380930187
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 7.810987032484673e-05,
            "ci95": [
              -6.306535544937696e-05,
              0.00022691384513529495
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
      "first_suffix__minus__first_even": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "first_prefix": 0.0,
          "first_suffix": 1.0,
          "first_even": -1.0,
          "first_odd": 0.0,
          "second_prefix": 0.0,
          "second_suffix": 0.0,
          "second_even": 0.0,
          "second_odd": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.00019675124358017881,
            "ci95": [
              -0.0029292809844941895,
              0.0025546791977236467
            ]
          },
          "auc.is_click": {
            "estimate": -0.0015714347549894425,
            "ci95": [
              -0.002408178604628064,
              -0.0007017020817533617
            ]
          },
          "auc.cart": {
            "estimate": -0.0006317754633531347,
            "ci95": [
              -0.005537828207562245,
              0.003772482041222841
            ]
          },
          "auc.fav": {
            "estimate": -0.003102819379271704,
            "ci95": [
              -0.009237177815508296,
              0.003098142178340654
            ]
          },
          "auc.buy": {
            "estimate": 0.004519024623293566,
            "ci95": [
              -0.002831710334785703,
              0.012343905082454625
            ]
          },
          "bce.is_click": {
            "estimate": 0.0002180255917979268,
            "ci95": [
              0.0001405986079854711,
              0.0002938985381602241
            ]
          },
          "bce.cart": {
            "estimate": 1.9101531152733442e-05,
            "ci95": [
              -9.366075631918845e-05,
              0.00014452755765200593
            ]
          },
          "bce.fav": {
            "estimate": -0.00020912925951409658,
            "ci95": [
              -0.0003351301943428814,
              -9.554368373702368e-05
            ]
          },
          "bce.buy": {
            "estimate": -0.00013822463542860833,
            "ci95": [
              -0.0002182011666706269,
              -5.591616723384871e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.0009396592916363078,
            "ci95": [
              -0.005447703300950937,
              0.003890827040897159
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.0015313846242822615,
            "ci95": [
              -0.004655132749669025,
              0.007915675402744262
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.0060904593782830085,
            "ci95": [
              -0.01401816922884008,
              0.0013137105501119866
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.0024710439159185693,
            "ci95": [
              -0.005247232767551099,
              0.010319281003640714
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.005150800086646701,
            "ci95": [
              -0.013985201628388207,
              0.003243617792355638
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.00762184400256527,
            "ci95": [
              -0.016945131311230897,
              0.0017535523649972711
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 0.00019892406064519336,
            "ci95": [
              4.6214535783670016e-05,
              0.0003399432030028673
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.0004271548513120234,
            "ci95": [
              0.0002861712109388825,
              0.0005642563360330962
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.0003562502272265351,
            "ci95": [
              0.00024025262269697172,
              0.00047049015820761476
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.00022823079066683002,
            "ci95": [
              5.906426143105449e-05,
              0.0003846694866169834
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00015732616658134177,
            "ci95": [
              1.8968118020833888e-05,
              0.0003012581403540774
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -7.090462408548825e-05,
            "ci95": [
              -0.00021456863064672145,
              6.924705232512517e-05
            ]
          }
        }
      }
    },
    "parameter_count": 51724260,
    "initializer_sha256": "fd5f866c35c5e29b37841aeed070f6ef5259c773de173df7549d149f87a8c09e",
    "optimizer_layout_sha256": "a007450b61e908e8c8832065c25f3161edcc388544c307ba2bfaa426c20502f3",
    "formal_worker_cost": {
      "wall_seconds": 2515.3534044995904,
      "cpu_seconds": 3025.922370799,
      "peak_rss_bytes": 16983592960,
      "gpu_peak_bytes": 28371320832,
      "samples": 2306
    },
    "smoke_worker_cost": {
      "wall_seconds": 147.4100959300995,
      "cpu_seconds": 161.93760790800002,
      "peak_rss_bytes": 16945926144,
      "gpu_peak_bytes": 29570891776,
      "samples": 135
    },
    "complete_recorded_costs": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2515.3534044995904,
        "cpu_seconds": 3025.922370799,
        "peak_rss_bytes": 16983592960,
        "gpu_peak_bytes": 28371320832,
        "samples": 2306
      },
      "formal_runner": {
        "wall_seconds": 2510.978076197207,
        "cpu_seconds": 2911.186048419,
        "peak_process_rss_bytes": 5240745984
      },
      "fit_including_validation": {
        "wall_seconds": 2387.611212775111,
        "cpu_seconds": 2739.738395661,
        "peak_process_rss_bytes": 5240745984
      },
      "prediction_export": {
        "wall_seconds": 117.95157391577959,
        "cpu_seconds": 165.77488881099998,
        "peak_process_rss_bytes": 5240745984
      },
      "smoke_worker": {
        "wall_seconds": 147.4100959300995,
        "cpu_seconds": 161.93760790800002,
        "peak_rss_bytes": 16945926144,
        "gpu_peak_bytes": 29570891776,
        "samples": 135
      },
      "smoke_runner": {
        "wall_seconds": 141.91200194507837,
        "cpu_seconds": 139.593388292,
        "peak_process_rss_bytes": 3239772160
      },
      "latency_constructor": {
        "wall_seconds": 0.7686734199523926,
        "cpu_seconds": 0.8429877159999997,
        "peak_rss_bytes": 2092183552
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.1590116173028946,
        "cpu_seconds": 0.15740078299999993,
        "peak_rss_bytes": 2092183552
      },
      "latency_peak_cuda_bytes": 1281155072,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "forward_latency": {
      "mean_ms": 14.349047255516052,
      "p50_ms": 14.283056259155273,
      "p95_ms": 14.38448805809021,
      "batch_size": 1024,
      "warmups": 10,
      "iterations": 40
    },
    "latency_ratio_to_parent": 1.0137632917825123,
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
  "decision_reason": "Macro delta -0.004675091,CI95 [-0.007934792701201314, -0.0017019098487541366],latency ratio 1.013763. Fails positive and near-candidate quality policy; preserve negative evidence.",
  "decision_scope": "Root queue disposition only; no released-incumbent promotion or supported lesson.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_even",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_even/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_even/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_even/Taobao_Action/first_even.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/paired-analysis/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/latency/completion.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_even",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_even/result.json",
    "result_sha256": "a6955c66720f6698a85c6b1b48224c385fb161428fdc39d5158b6393ed185145",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_even/manifest.json",
    "manifest_sha256": "4a7e620352c18b79d89cf49cec6363400a000943f290d698a8af60cb812ec59a",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_even/Taobao_Action/first_even.model",
    "checkpoint_sha256": "a08f350cdda9524b5b64fa1ab257c999c89108bf79faa4003fcad72fef2bd760",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/paired-analysis/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
