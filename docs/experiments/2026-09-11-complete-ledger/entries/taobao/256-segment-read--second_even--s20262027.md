## 256. segment-read--second_even--s20262027

- 假设：Equal-count second_even stored-history support may change useful two-round retrieval conditions; no pure Q2 mediation claim.
- 判定：discard
- 原因：Macro delta -0.002107763,CI95 [-0.005397456379620315, 0.0012215223930179182],latency ratio 1.012147. Fails positive and near-candidate quality policy; preserve negative evidence.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "segment-read--second_even--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-10T14:13:03.864962+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "second_even",
  "code_id": "cf7b0840f7d3f28bb5451ac3cc4d9491bfcf60c5a00b32fc75f78b7e407fc0de",
  "hypothesis": "Equal-count second_even stored-history support may change useful two-round retrieval conditions; no pure Q2 mediation claim.",
  "change": "{\"model\": \"QFormerCross8SegmentRead\", \"maturity_mode\": \"mature\", \"segment_placement\": \"second\", \"segment_selection\": \"even\", \"work_steps\": 2, \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8SegmentRead",
    "maturity_mode": "mature",
    "segment_placement": "second",
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
      "value": 0.7649871886515519,
      "delta": -0.0021077626633085045
    },
    "per_task": {
      "is_click": {
        "logloss": 0.18438281961416356,
        "AUC": 0.6507770868956803
      },
      "cart": {
        "logloss": 0.021954468677202972,
        "AUC": 0.7963713622148019
      },
      "fav": {
        "logloss": 0.01708061803345837,
        "AUC": 0.793810342533468
      },
      "buy": {
        "logloss": 0.009794471419520745,
        "AUC": 0.8189899629622578
      }
    },
    "macro_source_logloss": 0.05830309443608641,
    "source_metric_comparison": {
      "is_click": {
        "source_AUC": 0.6507770868956803,
        "computed_AUC": 0.6507770868956803,
        "auc_difference": 0.0,
        "source_logloss": 0.18438281961416356,
        "computed_bce": 0.18438281961416822,
        "bce_difference": 4.6629367034256575e-15
      },
      "cart": {
        "source_AUC": 0.7963713622148019,
        "computed_AUC": 0.7963713622148019,
        "auc_difference": 0.0,
        "source_logloss": 0.021954468677202972,
        "computed_bce": 0.021954469180035345,
        "bce_difference": 5.028323729616702e-10
      },
      "fav": {
        "source_AUC": 0.793810342533468,
        "computed_AUC": 0.793810342533468,
        "auc_difference": 0.0,
        "source_logloss": 0.01708061803345837,
        "computed_bce": 0.01708061864976318,
        "bce_difference": 6.163048092699963e-10
      },
      "buy": {
        "source_AUC": 0.8189899629622578,
        "computed_AUC": 0.8189899629622578,
        "auc_difference": 0.0,
        "source_logloss": 0.009794471419520745,
        "computed_bce": 0.009794471803908964,
        "bce_difference": 3.843882191711945e-10
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
        "second_prefix": 0.0,
        "second_suffix": 0.0,
        "second_even": 1.0,
        "second_odd": 0.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": -0.0021077626633085045,
          "ci95": [
            -0.005397456379620315,
            0.0012215223930179182
          ]
        },
        "auc.is_click": {
          "estimate": 0.0008407936946908112,
          "ci95": [
            6.680304673929163e-05,
            0.0016182676727580058
          ]
        },
        "auc.cart": {
          "estimate": 0.003937091857458164,
          "ci95": [
            -0.000945748627436968,
            0.009354599792643364
          ]
        },
        "auc.fav": {
          "estimate": -0.00865314784158,
          "ci95": [
            -0.015206793192426401,
            -0.002410846816058491
          ]
        },
        "auc.buy": {
          "estimate": -0.004555788363802993,
          "ci95": [
            -0.013882104033168063,
            0.005050972248599247
          ]
        },
        "bce.is_click": {
          "estimate": -8.826653618454472e-06,
          "ci95": [
            -8.182229078013206e-05,
            6.517683119417194e-05
          ]
        },
        "bce.cart": {
          "estimate": 0.00012651481060131295,
          "ci95": [
            4.487210289554505e-06,
            0.00023468333549172475
          ]
        },
        "bce.fav": {
          "estimate": 0.0002824609554108459,
          "ci95": [
            0.00015335429069040916,
            0.00043963739571797937
          ]
        },
        "bce.buy": {
          "estimate": 0.0001506698671832618,
          "ci95": [
            5.669417688413882e-05,
            0.0002476153599423328
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": -0.0030962981627673525,
          "ci95": [
            -0.008676217716140052,
            0.0018115602982373578
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": 0.009493941536270811,
          "ci95": [
            0.003416380103601538,
            0.01623722620908797
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": 0.005396582058493804,
          "ci95": [
            -0.004374680738778741,
            0.014508679803139888
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": 0.012590239699038164,
          "ci95": [
            0.00406866249701329,
            0.020890669875071374
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": 0.008492880221261156,
          "ci95": [
            -0.0017363742211666281,
            0.01792587799146484
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": -0.004097359477777007,
          "ci95": [
            -0.014416377961025809,
            0.007611144367200942
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": -0.00013534146421976742,
          "ci95": [
            -0.0002735177559204705,
            2.3776481124769108e-07
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": -0.0002912876090293004,
          "ci95": [
            -0.00046942915671667067,
            -0.00015075764319547847
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": -0.00015949652080171628,
          "ci95": [
            -0.0002849780028881335,
            -5.346356803800269e-05
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": -0.00015594614480953298,
          "ci95": [
            -0.0003401707202052022,
            1.9710811686112474e-05
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": -2.4155056581948858e-05,
          "ci95": [
            -0.00017801023704519794,
            0.00010137117660623632
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": 0.00013179108822758412,
          "ci95": [
            -3.6261292450239864e-05,
            0.0002930776107752442
          ]
        }
      }
    },
    "all_prespecified_effects_involving_candidate": {
      "second_even__minus__mature_r2": {
        "coefficients": {
          "mature_r2": -1.0,
          "cross8_mature": 0.0,
          "first_prefix": 0.0,
          "first_suffix": 0.0,
          "first_even": 0.0,
          "first_odd": 0.0,
          "second_prefix": 0.0,
          "second_suffix": 0.0,
          "second_even": 1.0,
          "second_odd": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.0021077626633085045,
            "ci95": [
              -0.005397456379620315,
              0.0012215223930179182
            ]
          },
          "auc.is_click": {
            "estimate": 0.0008407936946908112,
            "ci95": [
              6.680304673929163e-05,
              0.0016182676727580058
            ]
          },
          "auc.cart": {
            "estimate": 0.003937091857458164,
            "ci95": [
              -0.000945748627436968,
              0.009354599792643364
            ]
          },
          "auc.fav": {
            "estimate": -0.00865314784158,
            "ci95": [
              -0.015206793192426401,
              -0.002410846816058491
            ]
          },
          "auc.buy": {
            "estimate": -0.004555788363802993,
            "ci95": [
              -0.013882104033168063,
              0.005050972248599247
            ]
          },
          "bce.is_click": {
            "estimate": -8.826653618454472e-06,
            "ci95": [
              -8.182229078013206e-05,
              6.517683119417194e-05
            ]
          },
          "bce.cart": {
            "estimate": 0.00012651481060131295,
            "ci95": [
              4.487210289554505e-06,
              0.00023468333549172475
            ]
          },
          "bce.fav": {
            "estimate": 0.0002824609554108459,
            "ci95": [
              0.00015335429069040916,
              0.00043963739571797937
            ]
          },
          "bce.buy": {
            "estimate": 0.0001506698671832618,
            "ci95": [
              5.669417688413882e-05,
              0.0002476153599423328
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.0030962981627673525,
            "ci95": [
              -0.008676217716140052,
              0.0018115602982373578
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.009493941536270811,
            "ci95": [
              0.003416380103601538,
              0.01623722620908797
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.005396582058493804,
            "ci95": [
              -0.004374680738778741,
              0.014508679803139888
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.012590239699038164,
            "ci95": [
              0.00406866249701329,
              0.020890669875071374
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.008492880221261156,
            "ci95": [
              -0.0017363742211666281,
              0.01792587799146484
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.004097359477777007,
            "ci95": [
              -0.014416377961025809,
              0.007611144367200942
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00013534146421976742,
            "ci95": [
              -0.0002735177559204705,
              2.3776481124769108e-07
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0002912876090293004,
            "ci95": [
              -0.00046942915671667067,
              -0.00015075764319547847
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.00015949652080171628,
            "ci95": [
              -0.0002849780028881335,
              -5.346356803800269e-05
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.00015594614480953298,
            "ci95": [
              -0.0003401707202052022,
              1.9710811686112474e-05
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -2.4155056581948858e-05,
            "ci95": [
              -0.00017801023704519794,
              0.00010137117660623632
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.00013179108822758412,
            "ci95": [
              -3.6261292450239864e-05,
              0.0002930776107752442
            ]
          }
        }
      },
      "second_even__minus__cross8_mature": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": -1.0,
          "first_prefix": 0.0,
          "first_suffix": 0.0,
          "first_even": 0.0,
          "first_odd": 0.0,
          "second_prefix": 0.0,
          "second_suffix": 0.0,
          "second_even": 1.0,
          "second_odd": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.005580768556092158,
            "ci95": [
              0.0027951218077896605,
              0.009058572060600572
            ]
          },
          "auc.is_click": {
            "estimate": 0.000840316752714565,
            "ci95": [
              3.2995168427849065e-05,
              0.0016353251267188771
            ]
          },
          "auc.cart": {
            "estimate": 0.007164626843685906,
            "ci95": [
              0.002486104702353678,
              0.0116659817153663
            ]
          },
          "auc.fav": {
            "estimate": -0.0026916684939815028,
            "ci95": [
              -0.007945959085083198,
              0.002758071335269056
            ]
          },
          "auc.buy": {
            "estimate": 0.017009799121949665,
            "ci95": [
              0.007838431983770319,
              0.026915200911441783
            ]
          },
          "bce.is_click": {
            "estimate": 6.488176557686853e-05,
            "ci95": [
              -7.146918298210625e-06,
              0.00013964867000802665
            ]
          },
          "bce.cart": {
            "estimate": 0.00012005754295631513,
            "ci95": [
              1.692435981314267e-05,
              0.0002164662954513483
            ]
          },
          "bce.fav": {
            "estimate": 0.00010723946831013229,
            "ci95": [
              1.367442550064815e-05,
              0.0001951074338654943
            ]
          },
          "bce.buy": {
            "estimate": -0.00021291852452916434,
            "ci95": [
              -0.0003072073041798096,
              -0.00012086450829558761
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.006324310090971341,
            "ci95": [
              -0.011021198310546824,
              -0.0015519394048234855
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.0035319852466960677,
            "ci95": [
              -0.002131193562388192,
              0.008625668299630724
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.0161694823692351,
            "ci95": [
              -0.026106077808395956,
              -0.007172647597074318
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.009856295337667409,
            "ci95": [
              0.002601226679087407,
              0.01675358936560852
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.009845172278263759,
            "ci95": [
              -0.019858122612391966,
              -0.0006657936446731863
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.019701467615931167,
            "ci95": [
              -0.030520910513699095,
              -0.009164421591311731
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -5.5175777379446594e-05,
            "ci95": [
              -0.0001758730490387126,
              7.690237600216555e-05
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -4.235770273326375e-05,
            "ci95": [
              -0.00015570190121628693,
              8.553573698311137e-05
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.00027780029010603287,
            "ci95": [
              0.00016361940022713048,
              0.00041053539469345794
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 1.2818074646182842e-05,
            "ci95": [
              -0.00011951619919754688,
              0.00014535809145585984
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00033297606748547946,
            "ci95": [
              0.00020371612653564925,
              0.00047883345296116897
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.0003201579928392966,
            "ci95": [
              0.0001871246028886061,
              0.0004460703313256611
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
      "second_suffix__minus__second_even": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "first_prefix": 0.0,
          "first_suffix": 0.0,
          "first_even": 0.0,
          "first_odd": 0.0,
          "second_prefix": 0.0,
          "second_suffix": 1.0,
          "second_even": -1.0,
          "second_odd": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.00010083648693942626,
            "ci95": [
              -0.003079744334956494,
              0.002707492664425175
            ]
          },
          "auc.is_click": {
            "estimate": 0.0006988540871134541,
            "ci95": [
              -0.00015809582822501057,
              0.0013822795880535142
            ]
          },
          "auc.cart": {
            "estimate": -0.009766338566557287,
            "ci95": [
              -0.015685506930360537,
              -0.00416531129267691
            ]
          },
          "auc.fav": {
            "estimate": 0.009602274718439774,
            "ci95": [
              0.0028667519291044412,
              0.015304290590478305
            ]
          },
          "auc.buy": {
            "estimate": -0.0009381361867536464,
            "ci95": [
              -0.009202163893434456,
              0.0068948194985855825
            ]
          },
          "bce.is_click": {
            "estimate": -0.0002456671973943836,
            "ci95": [
              -0.0003038216120097081,
              -0.0001683420999026723
            ]
          },
          "bce.cart": {
            "estimate": 4.396430488151984e-05,
            "ci95": [
              -7.043950611951107e-05,
              0.00017785676426593198
            ]
          },
          "bce.fav": {
            "estimate": -0.0003244391305631533,
            "ci95": [
              -0.0004233656177677557,
              -0.00020088949187925705
            ]
          },
          "bce.buy": {
            "estimate": -0.00012175927965227783,
            "ci95": [
              -0.0002107982944347192,
              -3.131779337126365e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.01046519265367074,
            "ci95": [
              0.0046836095714160786,
              0.01619425555474997
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.00890342063132632,
            "ci95": [
              -0.014539257024247326,
              -0.0020237967061377403
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.0016369902738671005,
            "ci95": [
              -0.006338484943959984,
              0.0099791988453838
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.01936861328499706,
            "ci95": [
              -0.02791100756412916,
              -0.010683963654653472
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.00882820237980364,
            "ci95": [
              -0.019039218454404407,
              0.00171851297903729
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.01054041090519342,
            "ci95": [
              0.00038487193414386844,
              0.020201215804786808
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.0002896315022759034,
            "ci95": [
              -0.00043397735813942875,
              -0.00015125596704772105
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 7.877193316876974e-05,
            "ci95": [
              -4.131618452792229e-05,
              0.00019628749096262988
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.00012390791774210576,
            "ci95": [
              -0.00023496035482059802,
              -1.837742467798043e-05
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.00036840343544467316,
            "ci95": [
              0.00019673639229432686,
              0.0005343344931928816
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00016572358453379767,
            "ci95": [
              1.3585052007230368e-05,
              0.00032273652681112804
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.0002026798509108755,
            "ci95": [
              -0.00033266066937213186,
              -5.460191130449551e-05
            ]
          }
        }
      }
    },
    "parameter_count": 51724260,
    "initializer_sha256": "fd5f866c35c5e29b37841aeed070f6ef5259c773de173df7549d149f87a8c09e",
    "optimizer_layout_sha256": "a1b4f30adffd4efaa4eace1891cd3441b42625451169cc27b1a6d87c98c1d3c6",
    "formal_worker_cost": {
      "wall_seconds": 2531.232098005712,
      "cpu_seconds": 3039.49522237,
      "peak_rss_bytes": 17119875072,
      "gpu_peak_bytes": 28371320832,
      "samples": 2322
    },
    "smoke_worker_cost": {
      "wall_seconds": 157.3881392478943,
      "cpu_seconds": 173.41980454999998,
      "peak_rss_bytes": 17164509184,
      "gpu_peak_bytes": 29996613632,
      "samples": 144
    },
    "complete_recorded_costs": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2531.232098005712,
        "cpu_seconds": 3039.49522237,
        "peak_rss_bytes": 17119875072,
        "gpu_peak_bytes": 28371320832,
        "samples": 2322
      },
      "formal_runner": {
        "wall_seconds": 2527.1809554323554,
        "cpu_seconds": 2924.2699800670002,
        "peak_process_rss_bytes": 5262262272
      },
      "fit_including_validation": {
        "wall_seconds": 2401.580759651959,
        "cpu_seconds": 2750.4913705550002,
        "peak_process_rss_bytes": 5262262272
      },
      "prediction_export": {
        "wall_seconds": 120.27337439358234,
        "cpu_seconds": 168.22248308000007,
        "peak_process_rss_bytes": 5262262272
      },
      "smoke_worker": {
        "wall_seconds": 157.3881392478943,
        "cpu_seconds": 173.41980454999998,
        "peak_rss_bytes": 17164509184,
        "gpu_peak_bytes": 29996613632,
        "samples": 144
      },
      "smoke_runner": {
        "wall_seconds": 153.03456237912178,
        "cpu_seconds": 149.510633044,
        "peak_process_rss_bytes": 3293913088
      },
      "latency_constructor": {
        "wall_seconds": 0.7560282275080681,
        "cpu_seconds": 0.8299144679999984,
        "peak_rss_bytes": 2092183552
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.14342691749334335,
        "cpu_seconds": 0.142560256000003,
        "peak_rss_bytes": 2092183552
      },
      "latency_peak_cuda_bytes": 1281155072,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "forward_latency": {
      "mean_ms": 14.326175975799561,
      "p50_ms": 14.267744064331055,
      "p95_ms": 14.342430114746094,
      "batch_size": 1024,
      "warmups": 10,
      "iterations": 40
    },
    "latency_ratio_to_parent": 1.0121474309243113,
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
  "decision_reason": "Macro delta -0.002107763,CI95 [-0.005397456379620315, 0.0012215223930179182],latency ratio 1.012147. Fails positive and near-candidate quality policy; preserve negative evidence.",
  "decision_scope": "Root queue disposition only; no released-incumbent promotion or supported lesson.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_even",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_even/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_even/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_even/Taobao_Action/second_even.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/paired-analysis/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/latency/completion.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_even",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_even/result.json",
    "result_sha256": "75935c3862faaede9ac4bc6ae383589d520c0bbb6563b940b7e72959bd662e05",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_even/manifest.json",
    "manifest_sha256": "6437b589340aca8da768e42b7850194dc11e164259ad5ec86943bafe90c9a85e",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_even/Taobao_Action/second_even.model",
    "checkpoint_sha256": "573fd3508d2145ebb1a63a49caa7bac0ddb316d8d4b5c5f446c86661ffd022a4",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/paired-analysis/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
