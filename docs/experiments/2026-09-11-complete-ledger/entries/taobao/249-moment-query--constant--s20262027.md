## 249. moment-query--constant--s20262027

- 假设：First-read constant projected into second-work w_q before q_norm may improve addressing under fixed two-cycle mature protocol.
- 判定：discard
- 原因：Macro delta -0.004005424, CI95 [-0.0072340066829167556, -0.0012263822228989322]; forward ratio 1.161950. Fails fixed positive/near quality-cost policy.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "moment-query--constant--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-10T07:45:38.318827+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "constant",
  "code_id": "6503505e6c1699b76ec8cce57b783280806e825c1df3cd8278fa7c9bd751c711",
  "hypothesis": "First-read constant projected into second-work w_q before q_norm may improve addressing under fixed two-cycle mature protocol.",
  "change": "{\"model\": \"QFormerCross8MomentQuery\", \"maturity_mode\": \"mature\", \"moment_source\": \"constant\", \"moment_rank\": 16, \"work_steps\": 2, \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8MomentQuery",
    "maturity_mode": "mature",
    "moment_source": "constant",
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
      "value": 0.7630895269202685,
      "delta": -0.00400542439459195
    },
    "per_task": {
      "is_click": {
        "logloss": 0.18383637265525102,
        "AUC": 0.65188626179538
      },
      "cart": {
        "logloss": 0.021852770418513048,
        "AUC": 0.7925029660319789
      },
      "fav": {
        "logloss": 0.016954354359317896,
        "AUC": 0.7940627416982565
      },
      "buy": {
        "logloss": 0.009815711925954582,
        "AUC": 0.8139061381554588
      }
    },
    "macro_source_logloss": 0.05811480233975913,
    "source_metric_comparison": {
      "is_click": {
        "source_AUC": 0.65188626179538,
        "computed_AUC": 0.65188626179538,
        "auc_difference": 0.0,
        "source_logloss": 0.18383637265525102,
        "computed_bce": 0.18383637265525465,
        "bce_difference": 3.635980405647388e-15
      },
      "cart": {
        "source_AUC": 0.7925029660319789,
        "computed_AUC": 0.7925029660319789,
        "auc_difference": 0.0,
        "source_logloss": 0.021852770418513048,
        "computed_bce": 0.021852770422713972,
        "bce_difference": 4.2009243306218025e-12
      },
      "fav": {
        "source_AUC": 0.7940627416982565,
        "computed_AUC": 0.7940627416982565,
        "auc_difference": 0.0,
        "source_logloss": 0.016954354359317896,
        "computed_bce": 0.016954354371505734,
        "bce_difference": 1.2187837544752611e-11
      },
      "buy": {
        "source_AUC": 0.8139061381554588,
        "computed_AUC": 0.8139061381554588,
        "auc_difference": 0.0,
        "source_logloss": 0.009815711925954582,
        "computed_bce": 0.009815711926766049,
        "bce_difference": 8.114672128689548e-13
      }
    },
    "matched_parent_effect": {
      "coefficients": {
        "mature_r2": -1.0,
        "cross8_mature": 0.0,
        "weighted_variance": 0.0,
        "weighted_mean": 0.0,
        "uniform_variance": 0.0,
        "constant": 1.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": -0.00400542439459195,
          "ci95": [
            -0.0072340066829167556,
            -0.0012263822228989322
          ]
        },
        "auc.is_click": {
          "estimate": 0.001949968594390561,
          "ci95": [
            0.0012872351469446375,
            0.0026750964699652153
          ]
        },
        "auc.cart": {
          "estimate": 6.869567463518855e-05,
          "ci95": [
            -0.0038783994140952293,
            0.004401651424195706
          ]
        },
        "auc.fav": {
          "estimate": -0.008400748676791525,
          "ci95": [
            -0.015056964225587726,
            -0.0018698315642118166
          ]
        },
        "auc.buy": {
          "estimate": -0.009639613170602024,
          "ci95": [
            -0.019449799244930514,
            -0.001037846615851193
          ]
        },
        "bce.is_click": {
          "estimate": -0.0005552736125320201,
          "ci95": [
            -0.0006284918051088224,
            -0.0004930947489716162
          ]
        },
        "bce.cart": {
          "estimate": 2.481605327993991e-05,
          "ci95": [
            -6.365250559969715e-05,
            0.00011070582674910112
          ]
        },
        "bce.fav": {
          "estimate": 0.00015619667715340013,
          "ci95": [
            -1.3588205968324132e-05,
            0.00030556832704850204
          ]
        },
        "bce.buy": {
          "estimate": 0.00017190999004034618,
          "ci95": [
            9.284592540748119e-05,
            0.0002661005001833113
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": 0.0018812729197553724,
          "ci95": [
            -0.0024411376497472685,
            0.006089000775638813
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": 0.010350717271182086,
          "ci95": [
            0.003978377965450977,
            0.01708901341699549
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": 0.011589581764992585,
          "ci95": [
            0.002882881128045134,
            0.021491442928036335
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": 0.008469444351426714,
          "ci95": [
            0.0010216343927380162,
            0.016783005026867075
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": 0.009708308845237212,
          "ci95": [
            4.5776875203368494e-05,
            0.01929026017619823
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": 0.0012388644938104987,
          "ci95": [
            -0.008810721702067776,
            0.012534524426594336
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": -0.00058008966581196,
          "ci95": [
            -0.0006919239232408317,
            -0.0004514509677601492
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": -0.0007114702896854203,
          "ci95": [
            -0.0008798840424978748,
            -0.000536843373136186
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": -0.0007271836025723663,
          "ci95": [
            -0.0008494659755067384,
            -0.0006257201228801862
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": -0.00013138062387346022,
          "ci95": [
            -0.00031216391160757774,
            5.713774530406531e-05
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": -0.00014709393676040627,
          "ci95": [
            -0.00027202907813450025,
            -3.082302090175803e-05
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": -1.571331288694605e-05,
          "ci95": [
            -0.00021003149001317862,
            0.00014974895321428707
          ]
        }
      }
    },
    "all_prespecified_effects_involving_candidate": {
      "constant__minus__mature_r2": {
        "coefficients": {
          "mature_r2": -1.0,
          "cross8_mature": 0.0,
          "weighted_variance": 0.0,
          "weighted_mean": 0.0,
          "uniform_variance": 0.0,
          "constant": 1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.00400542439459195,
            "ci95": [
              -0.0072340066829167556,
              -0.0012263822228989322
            ]
          },
          "auc.is_click": {
            "estimate": 0.001949968594390561,
            "ci95": [
              0.0012872351469446375,
              0.0026750964699652153
            ]
          },
          "auc.cart": {
            "estimate": 6.869567463518855e-05,
            "ci95": [
              -0.0038783994140952293,
              0.004401651424195706
            ]
          },
          "auc.fav": {
            "estimate": -0.008400748676791525,
            "ci95": [
              -0.015056964225587726,
              -0.0018698315642118166
            ]
          },
          "auc.buy": {
            "estimate": -0.009639613170602024,
            "ci95": [
              -0.019449799244930514,
              -0.001037846615851193
            ]
          },
          "bce.is_click": {
            "estimate": -0.0005552736125320201,
            "ci95": [
              -0.0006284918051088224,
              -0.0004930947489716162
            ]
          },
          "bce.cart": {
            "estimate": 2.481605327993991e-05,
            "ci95": [
              -6.365250559969715e-05,
              0.00011070582674910112
            ]
          },
          "bce.fav": {
            "estimate": 0.00015619667715340013,
            "ci95": [
              -1.3588205968324132e-05,
              0.00030556832704850204
            ]
          },
          "bce.buy": {
            "estimate": 0.00017190999004034618,
            "ci95": [
              9.284592540748119e-05,
              0.0002661005001833113
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.0018812729197553724,
            "ci95": [
              -0.0024411376497472685,
              0.006089000775638813
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.010350717271182086,
            "ci95": [
              0.003978377965450977,
              0.01708901341699549
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.011589581764992585,
            "ci95": [
              0.002882881128045134,
              0.021491442928036335
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.008469444351426714,
            "ci95": [
              0.0010216343927380162,
              0.016783005026867075
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.009708308845237212,
            "ci95": [
              4.5776875203368494e-05,
              0.01929026017619823
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.0012388644938104987,
            "ci95": [
              -0.008810721702067776,
              0.012534524426594336
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00058008966581196,
            "ci95": [
              -0.0006919239232408317,
              -0.0004514509677601492
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0007114702896854203,
            "ci95": [
              -0.0008798840424978748,
              -0.000536843373136186
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0007271836025723663,
            "ci95": [
              -0.0008494659755067384,
              -0.0006257201228801862
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.00013138062387346022,
            "ci95": [
              -0.00031216391160757774,
              5.713774530406531e-05
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -0.00014709393676040627,
            "ci95": [
              -0.00027202907813450025,
              -3.082302090175803e-05
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -1.571331288694605e-05,
            "ci95": [
              -0.00021003149001317862,
              0.00014974895321428707
            ]
          }
        }
      },
      "constant__minus__cross8_mature": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": -1.0,
          "weighted_variance": 0.0,
          "weighted_mean": 0.0,
          "uniform_variance": 0.0,
          "constant": 1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.003683106824808713,
            "ci95": [
              0.0009564965195302034,
              0.006603019450163549
            ]
          },
          "auc.is_click": {
            "estimate": 0.0019494916524143147,
            "ci95": [
              0.0011415763529251527,
              0.0026890057734532515
            ]
          },
          "auc.cart": {
            "estimate": 0.003296230660862931,
            "ci95": [
              -0.00043024166825387513,
              0.007064734433715343
            ]
          },
          "auc.fav": {
            "estimate": -0.002439269329193028,
            "ci95": [
              -0.007064317571383242,
              0.0026539430526632997
            ]
          },
          "auc.buy": {
            "estimate": 0.011925974315150634,
            "ci95": [
              0.003224661123311759,
              0.020725619216686652
            ]
          },
          "bce.is_click": {
            "estimate": -0.0004815651933366971,
            "ci95": [
              -0.0005506001406419194,
              -0.0004098713851792613
            ]
          },
          "bce.cart": {
            "estimate": 1.8358785634942087e-05,
            "ci95": [
              -8.055810921275708e-05,
              0.00012075298250543428
            ]
          },
          "bce.fav": {
            "estimate": -1.902480994731351e-05,
            "ci95": [
              -0.000144022830609564,
              9.331792668919276e-05
            ]
          },
          "bce.buy": {
            "estimate": -0.00019167840167207996,
            "ci95": [
              -0.00028303377360855864,
              -0.00010675535957012516
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.0013467390084486164,
            "ci95": [
              -0.005341794443101211,
              0.002353437149055496
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.0043887609816073425,
            "ci95": [
              -0.0008393892799656882,
              0.009099227966562111
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.009976482662736319,
            "ci95": [
              -0.01887029950596949,
              -0.0012805433532877305
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.005735499990055959,
            "ci95": [
              -0.000671897165292179,
              0.012108551485127043
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.008629743654287703,
            "ci95": [
              -0.018077690731638892,
              0.0012275081725084845
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.014365243644343662,
            "ci95": [
              -0.024716599292941285,
              -0.004647615650386046
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.0004999239789716392,
            "ci95": [
              -0.0006113842788459836,
              -0.00038401118150278727
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0004625403833893836,
            "ci95": [
              -0.0005907823965205696,
              -0.00032358626336673765
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.00028988679166461716,
            "ci95": [
              -0.00040310112657031705,
              -0.00017582101014145687
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 3.7383595582255597e-05,
            "ci95": [
              -0.00011045389946947775,
              0.0001927135713518001
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00021003718730702205,
            "ci95": [
              9.59399388523041e-05,
              0.00034244950100114236
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.00017265359172476645,
            "ci95": [
              3.200501529266021e-05,
              0.00032147701933536363
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
    "optimizer_layout_sha256": "f73ee384562c3194dbb9a53ba8addf534e6428b2f027249ca8bf84618076124d",
    "formal_worker_cost": {
      "wall_seconds": 2573.4254642799497,
      "cpu_seconds": 3082.337381397,
      "peak_rss_bytes": 17102426112,
      "gpu_peak_bytes": 31370248192,
      "samples": 2357
    },
    "smoke_worker_cost": {
      "wall_seconds": 168.6941207125783,
      "cpu_seconds": 228.416749538,
      "peak_rss_bytes": 19714613248,
      "gpu_peak_bytes": 33830207488,
      "samples": 155
    },
    "complete_recorded_costs": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2573.4254642799497,
        "cpu_seconds": 3082.337381397,
        "peak_rss_bytes": 17102426112,
        "gpu_peak_bytes": 31370248192,
        "samples": 2357
      },
      "formal_runner": {
        "wall_seconds": 2569.155454210937,
        "cpu_seconds": 2966.0550361769997,
        "peak_process_rss_bytes": 5273583616
      },
      "fit_including_validation": {
        "wall_seconds": 2441.0259727463126,
        "cpu_seconds": 2790.806207023,
        "peak_process_rss_bytes": 5273583616
      },
      "prediction_export": {
        "wall_seconds": 123.1439737379551,
        "cpu_seconds": 169.75532905,
        "peak_process_rss_bytes": 5273583616
      },
      "smoke_worker": {
        "wall_seconds": 168.6941207125783,
        "cpu_seconds": 228.416749538,
        "peak_rss_bytes": 19714613248,
        "gpu_peak_bytes": 33830207488,
        "samples": 155
      },
      "smoke_runner": {
        "wall_seconds": 164.32935041934252,
        "cpu_seconds": 156.152255692,
        "peak_process_rss_bytes": 3274907648
      },
      "latency_constructor": {
        "wall_seconds": 0.7705128565430641,
        "cpu_seconds": 0.8377152730000006,
        "peak_rss_bytes": 2276122624
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.15250489860773087,
        "cpu_seconds": 0.15127142099999702,
        "peak_rss_bytes": 2276122624
      },
      "latency_peak_cuda_bytes": 1369385472,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "forward_latency": {
      "mean_ms": 16.68792552947998,
      "p50_ms": 16.68609619140625,
      "p95_ms": 16.785590648651123,
      "batch_size": 1024,
      "warmups": 10,
      "iterations": 40
    },
    "latency_ratio_to_parent": 1.161949780307077,
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
  "decision_reason": "Macro delta -0.004005424, CI95 [-0.0072340066829167556, -0.0012263822228989322]; forward ratio 1.161950. Fails fixed positive/near quality-cost policy.",
  "decision_scope": "Root queue disposition only; no released-incumbent promotion or supported lesson.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/constant",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/constant/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/constant/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/constant/Taobao_Action/constant.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/paired-analysis/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/latency/completion.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/constant",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/constant/result.json",
    "result_sha256": "9271a9d486b88c82c70e0c3ef6ee9b93f186993217947104cc14843318d7bd43",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/constant/manifest.json",
    "manifest_sha256": "f1ddfbaf8e3afcd4aedd33bf569bded0f40cfdf1e920f7fc028a85b89b6d7e71",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/constant/Taobao_Action/constant.model",
    "checkpoint_sha256": "655ce61512cbb812b455c8fc6ef0e67eab5f244af1eb3f9da26e0c1dc51cff0c",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/paired-analysis/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
