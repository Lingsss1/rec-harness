## 247. moment-query--weighted_mean--s20262027

- 假设：First-read weighted_mean projected into second-work w_q before q_norm may improve addressing under fixed two-cycle mature protocol.
- 判定：discard
- 原因：Macro delta -0.005301424, CI95 [-0.008202795852167826, -0.0027167669438501324]; forward ratio 1.161172. Fails fixed positive/near quality-cost policy.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "moment-query--weighted_mean--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-10T07:45:38.318827+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "weighted_mean",
  "code_id": "6503505e6c1699b76ec8cce57b783280806e825c1df3cd8278fa7c9bd751c711",
  "hypothesis": "First-read weighted_mean projected into second-work w_q before q_norm may improve addressing under fixed two-cycle mature protocol.",
  "change": "{\"model\": \"QFormerCross8MomentQuery\", \"maturity_mode\": \"mature\", \"moment_source\": \"weighted_mean\", \"moment_rank\": 16, \"work_steps\": 2, \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8MomentQuery",
    "maturity_mode": "mature",
    "moment_source": "weighted_mean",
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
      "value": 0.7617935270570558,
      "delta": -0.005301424257804638
    },
    "per_task": {
      "is_click": {
        "logloss": 0.1837320077226782,
        "AUC": 0.652507581672969
      },
      "cart": {
        "logloss": 0.022070374657243527,
        "AUC": 0.781191120313434
      },
      "fav": {
        "logloss": 0.016876523193958083,
        "AUC": 0.8031768425146449
      },
      "buy": {
        "logloss": 0.009987974864959212,
        "AUC": 0.8102985637271756
      }
    },
    "macro_source_logloss": 0.058166720109709746,
    "source_metric_comparison": {
      "is_click": {
        "source_AUC": 0.652507581672969,
        "computed_AUC": 0.652507581672969,
        "auc_difference": 0.0,
        "source_logloss": 0.1837320077226782,
        "computed_bce": 0.18373200772268272,
        "bce_difference": 4.524158825347513e-15
      },
      "cart": {
        "source_AUC": 0.781191120313434,
        "computed_AUC": 0.781191120313434,
        "auc_difference": 0.0,
        "source_logloss": 0.022070374657243527,
        "computed_bce": 0.022070374686807316,
        "bce_difference": 2.9563788916942e-11
      },
      "fav": {
        "source_AUC": 0.8031768425146449,
        "computed_AUC": 0.8031768425146449,
        "auc_difference": 0.0,
        "source_logloss": 0.016876523193958083,
        "computed_bce": 0.01687652324152183,
        "bce_difference": 4.7563748079015866e-11
      },
      "buy": {
        "source_AUC": 0.8102985637271756,
        "computed_AUC": 0.8102985637271756,
        "auc_difference": 0.0,
        "source_logloss": 0.009987974864959212,
        "computed_bce": 0.009987974866036633,
        "bce_difference": 1.077421138417911e-12
      }
    },
    "matched_parent_effect": {
      "coefficients": {
        "mature_r2": -1.0,
        "cross8_mature": 0.0,
        "weighted_variance": 0.0,
        "weighted_mean": 1.0,
        "uniform_variance": 0.0,
        "constant": 0.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": -0.005301424257804638,
          "ci95": [
            -0.008202795852167826,
            -0.0027167669438501324
          ]
        },
        "auc.is_click": {
          "estimate": 0.002571288471979516,
          "ci95": [
            0.0019143064942110238,
            0.0033136573985826532
          ]
        },
        "auc.cart": {
          "estimate": -0.011243150043909744,
          "ci95": [
            -0.015505171539886745,
            -0.007178875602194882
          ]
        },
        "auc.fav": {
          "estimate": 0.00071335213959689,
          "ci95": [
            -0.005069627561337267,
            0.006878188825064882
          ]
        },
        "auc.buy": {
          "estimate": -0.013247187598885213,
          "ci95": [
            -0.02014371693120932,
            -0.005956517305753832
          ]
        },
        "bce.is_click": {
          "estimate": -0.0006596385451039521,
          "ci95": [
            -0.0007306614427721354,
            -0.0005936256245044334
          ]
        },
        "bce.cart": {
          "estimate": 0.00024242031737328404,
          "ci95": [
            0.00014731053655376295,
            0.0003347800941628312
          ]
        },
        "bce.fav": {
          "estimate": 7.836554716949759e-05,
          "ci95": [
            -6.343905697142232e-05,
            0.00020811711746048964
          ]
        },
        "bce.buy": {
          "estimate": 0.0003441729293109305,
          "ci95": [
            0.00026286652557077847,
            0.00042236901799606153
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": 0.01381443851588926,
          "ci95": [
            0.009856569231734147,
            0.018304514207269616
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": 0.001857936332382626,
          "ci95": [
            -0.004380285889246091,
            0.007639351473761077
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": 0.01581847607086473,
          "ci95": [
            0.008430521413444858,
            0.02271353347790173
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": -0.011956502183506634,
          "ci95": [
            -0.01928970860895498,
            -0.004940513121028187
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": 0.0020040375549754685,
          "ci95": [
            -0.004954484934379854,
            0.008950561204258157
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": 0.013960539738482103,
          "ci95": [
            0.00527619463852243,
            0.02339968736338722
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": -0.0009020588624772362,
          "ci95": [
            -0.0010239497678218529,
            -0.0007905090020964267
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": -0.0007380040922734497,
          "ci95": [
            -0.0008809617413961673,
            -0.0005884102160251305
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": -0.0010038114744148827,
          "ci95": [
            -0.0011106162263336513,
            -0.0009020334255278066
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": 0.00016405477020378645,
          "ci95": [
            1.3788295461948913e-05,
            0.00032915796668587757
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": -0.00010175261193764648,
          "ci95": [
            -0.00022054477223274523,
            4.992380910350405e-06
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": -0.00026580738214143293,
          "ci95": [
            -0.0004362458687073356,
            -0.00011987743320783658
          ]
        }
      }
    },
    "all_prespecified_effects_involving_candidate": {
      "weighted_mean__minus__mature_r2": {
        "coefficients": {
          "mature_r2": -1.0,
          "cross8_mature": 0.0,
          "weighted_variance": 0.0,
          "weighted_mean": 1.0,
          "uniform_variance": 0.0,
          "constant": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.005301424257804638,
            "ci95": [
              -0.008202795852167826,
              -0.0027167669438501324
            ]
          },
          "auc.is_click": {
            "estimate": 0.002571288471979516,
            "ci95": [
              0.0019143064942110238,
              0.0033136573985826532
            ]
          },
          "auc.cart": {
            "estimate": -0.011243150043909744,
            "ci95": [
              -0.015505171539886745,
              -0.007178875602194882
            ]
          },
          "auc.fav": {
            "estimate": 0.00071335213959689,
            "ci95": [
              -0.005069627561337267,
              0.006878188825064882
            ]
          },
          "auc.buy": {
            "estimate": -0.013247187598885213,
            "ci95": [
              -0.02014371693120932,
              -0.005956517305753832
            ]
          },
          "bce.is_click": {
            "estimate": -0.0006596385451039521,
            "ci95": [
              -0.0007306614427721354,
              -0.0005936256245044334
            ]
          },
          "bce.cart": {
            "estimate": 0.00024242031737328404,
            "ci95": [
              0.00014731053655376295,
              0.0003347800941628312
            ]
          },
          "bce.fav": {
            "estimate": 7.836554716949759e-05,
            "ci95": [
              -6.343905697142232e-05,
              0.00020811711746048964
            ]
          },
          "bce.buy": {
            "estimate": 0.0003441729293109305,
            "ci95": [
              0.00026286652557077847,
              0.00042236901799606153
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.01381443851588926,
            "ci95": [
              0.009856569231734147,
              0.018304514207269616
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.001857936332382626,
            "ci95": [
              -0.004380285889246091,
              0.007639351473761077
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.01581847607086473,
            "ci95": [
              0.008430521413444858,
              0.02271353347790173
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.011956502183506634,
            "ci95": [
              -0.01928970860895498,
              -0.004940513121028187
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.0020040375549754685,
            "ci95": [
              -0.004954484934379854,
              0.008950561204258157
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.013960539738482103,
            "ci95": [
              0.00527619463852243,
              0.02339968736338722
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.0009020588624772362,
            "ci95": [
              -0.0010239497678218529,
              -0.0007905090020964267
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0007380040922734497,
            "ci95": [
              -0.0008809617413961673,
              -0.0005884102160251305
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0010038114744148827,
            "ci95": [
              -0.0011106162263336513,
              -0.0009020334255278066
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.00016405477020378645,
            "ci95": [
              1.3788295461948913e-05,
              0.00032915796668587757
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -0.00010175261193764648,
            "ci95": [
              -0.00022054477223274523,
              4.992380910350405e-06
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.00026580738214143293,
            "ci95": [
              -0.0004362458687073356,
              -0.00011987743320783658
            ]
          }
        }
      },
      "weighted_mean__minus__cross8_mature": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": -1.0,
          "weighted_variance": 0.0,
          "weighted_mean": 1.0,
          "uniform_variance": 0.0,
          "constant": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.002387106961596025,
            "ci95": [
              -0.00012144941345763804,
              0.00551680614691667
            ]
          },
          "auc.is_click": {
            "estimate": 0.0025708115300032697,
            "ci95": [
              0.001813090973083542,
              0.003238199323522209
            ]
          },
          "auc.cart": {
            "estimate": -0.008015615057682002,
            "ci95": [
              -0.012712280960987008,
              -0.003359919802119624
            ]
          },
          "auc.fav": {
            "estimate": 0.006674831487195387,
            "ci95": [
              0.0017236190165517225,
              0.012304432222608486
            ]
          },
          "auc.buy": {
            "estimate": 0.008318399886867445,
            "ci95": [
              -0.0004175678252580056,
              0.018379480706609337
            ]
          },
          "bce.is_click": {
            "estimate": -0.0005859301259086291,
            "ci95": [
              -0.0006515285942664886,
              -0.000511526232294747
            ]
          },
          "bce.cart": {
            "estimate": 0.00023596304972828622,
            "ci95": [
              0.0001387349708223921,
              0.00033261571037938636
            ]
          },
          "bce.fav": {
            "estimate": -9.685593993121605e-05,
            "ci95": [
              -0.00019964949632647922,
              7.244551010570543e-08
            ]
          },
          "bce.buy": {
            "estimate": -1.9415462401495623e-05,
            "ci95": [
              -0.00012327215537755962,
              6.943548989261469e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.010586426587685271,
            "ci95": [
              0.005925635300747958,
              0.015243132773146692
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.004104019957192118,
            "ci95": [
              -0.009688685757336923,
              0.0006838350461848959
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.005747588356864175,
            "ci95": [
              -0.01597603551667092,
              0.0031416723794232054
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.014690446544877389,
            "ci95": [
              -0.022311976413270783,
              -0.007854501650003957
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.016334014944549446,
            "ci95": [
              -0.02659697175366609,
              -0.007213020004324308
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.0016435683996720574,
            "ci95": [
              -0.012663013836518366,
              0.008763120291588813
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.0008218931756369154,
            "ci95": [
              -0.0009355799267757062,
              -0.000690239578241994
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0004890741859774131,
            "ci95": [
              -0.0006023405936998596,
              -0.0003579567975855798
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0005665146635071335,
            "ci95": [
              -0.00068158349201925,
              -0.0004388560477396499
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.00033281898965950227,
            "ci95": [
              0.00019470040305279512,
              0.0004917802672090883
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00025537851212978184,
            "ci95": [
              0.0001339192451524654,
              0.00039985022167954765
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -7.744047752972043e-05,
            "ci95": [
              -0.00022732629387304078,
              6.0963739858134887e-05
            ]
          }
        }
      },
      "weighted_variance__minus__weighted_mean": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "weighted_variance": 1.0,
          "weighted_mean": -1.0,
          "uniform_variance": 0.0,
          "constant": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.0023036146041760797,
            "ci95": [
              -0.0004429570498071285,
              0.0048404831683537
            ]
          },
          "auc.is_click": {
            "estimate": -0.0034692032512519955,
            "ci95": [
              -0.0042099210311445255,
              -0.002734907013497859
            ]
          },
          "auc.cart": {
            "estimate": 0.01411341872191707,
            "ci95": [
              0.009317102525011584,
              0.019059107992267173
            ]
          },
          "auc.fav": {
            "estimate": -0.0077425590093122,
            "ci95": [
              -0.01367159045648055,
              -0.0027438616966513956
            ]
          },
          "auc.buy": {
            "estimate": 0.006312801955351444,
            "ci95": [
              -0.0013911740661540631,
              0.013448899226253964
            ]
          },
          "bce.is_click": {
            "estimate": 0.0004369692999866004,
            "ci95": [
              0.0003638927365315785,
              0.0005070077072059767
            ]
          },
          "bce.cart": {
            "estimate": -0.0002633237012521529,
            "ci95": [
              -0.000353929271513768,
              -0.00016377326243316077
            ]
          },
          "bce.fav": {
            "estimate": 0.00021095386264838659,
            "ci95": [
              9.730977006989225e-05,
              0.0003461506670391807
            ]
          },
          "bce.buy": {
            "estimate": -0.00029253054590009167,
            "ci95": [
              -0.00037832145400506785,
              -0.0002107467897930234
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.017582621973169066,
            "ci95": [
              -0.02246165978622653,
              -0.012950341331327931
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.004273355758060204,
            "ci95": [
              -0.0007873012386292315,
              0.00998734591031036
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.009782005206603439,
            "ci95": [
              -0.017100468864611643,
              -0.0020482422989873744
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.02185597773122927,
            "ci95": [
              0.01502369422673729,
              0.029470760346938584
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.007800616766565627,
            "ci95": [
              -0.00037241663395438387,
              0.016663965084684543
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.014055360964663643,
            "ci95": [
              -0.0232680643810132,
              -0.005418260291564084
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 0.0007002930012387533,
            "ci95": [
              0.0005751403762502239,
              0.0008290950764614996
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.0002260154373382138,
            "ci95": [
              8.134245661605621e-05,
              0.0003576479919595265
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.0007294998458866921,
            "ci95": [
              0.0006257469304969351,
              0.0008514099858420474
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.0004742775639005395,
            "ci95": [
              -0.0006422393225277434,
              -0.0003197592857339179
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 2.9206844647938768e-05,
            "ci95": [
              -9.447780391591323e-05,
              0.00015528664410633964
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.0005034844085484783,
            "ci95": [
              0.00035791386472333115,
              0.0006562708647581595
            ]
          }
        }
      }
    },
    "parameter_count": 51736548,
    "initializer_sha256": "796cec890828fda2f18f4ecf0c052e81999552b8179857b95f1c309d637d6ccb",
    "optimizer_layout_sha256": "7d25e9f94c33b50178f17e1d08514861096b16d19604bcf2b28218a7da4b3833",
    "formal_worker_cost": {
      "wall_seconds": 2613.6563271060586,
      "cpu_seconds": 3119.7526819329996,
      "peak_rss_bytes": 17078816768,
      "gpu_peak_bytes": 31366053888,
      "samples": 2394
    },
    "smoke_worker_cost": {
      "wall_seconds": 162.1670394986868,
      "cpu_seconds": 220.020367819,
      "peak_rss_bytes": 19715153920,
      "gpu_peak_bytes": 33826013184,
      "samples": 149
    },
    "complete_recorded_costs": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2613.6563271060586,
        "cpu_seconds": 3119.7526819329996,
        "peak_rss_bytes": 17078816768,
        "gpu_peak_bytes": 31366053888,
        "samples": 2394
      },
      "formal_runner": {
        "wall_seconds": 2609.928161472082,
        "cpu_seconds": 3003.2489274480004,
        "peak_process_rss_bytes": 5271633920
      },
      "fit_including_validation": {
        "wall_seconds": 2480.7927304655313,
        "cpu_seconds": 2826.817824467,
        "peak_process_rss_bytes": 5271633920
      },
      "prediction_export": {
        "wall_seconds": 124.16637706756592,
        "cpu_seconds": 170.98404432200005,
        "peak_process_rss_bytes": 5271633920
      },
      "smoke_worker": {
        "wall_seconds": 162.1670394986868,
        "cpu_seconds": 220.020367819,
        "peak_rss_bytes": 19715153920,
        "gpu_peak_bytes": 33826013184,
        "samples": 149
      },
      "smoke_runner": {
        "wall_seconds": 158.17813896387815,
        "cpu_seconds": 150.808072362,
        "peak_process_rss_bytes": 3284299776
      },
      "latency_constructor": {
        "wall_seconds": 0.7936275526881218,
        "cpu_seconds": 0.8601520479999998,
        "peak_rss_bytes": 2275430400
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.1169351115822792,
        "cpu_seconds": 0.11588948600000037,
        "peak_rss_bytes": 2275430400
      },
      "latency_peak_cuda_bytes": 1369385472,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "forward_latency": {
      "mean_ms": 16.67675848007202,
      "p50_ms": 16.64668846130371,
      "p95_ms": 16.773075580596924,
      "batch_size": 1024,
      "warmups": 10,
      "iterations": 40
    },
    "latency_ratio_to_parent": 1.1611722390492776,
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
  "decision_reason": "Macro delta -0.005301424, CI95 [-0.008202795852167826, -0.0027167669438501324]; forward ratio 1.161172. Fails fixed positive/near quality-cost policy.",
  "decision_scope": "Root queue disposition only; no released-incumbent promotion or supported lesson.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/weighted_mean",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/weighted_mean/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/weighted_mean/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/weighted_mean/Taobao_Action/weighted_mean.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/paired-analysis/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/latency/completion.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/weighted_mean",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/weighted_mean/result.json",
    "result_sha256": "e8ff5d9fd7eead51e82bb1fc25a80fe75e5fa63b9bf4ce8badd2bfefe939d399",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/weighted_mean/manifest.json",
    "manifest_sha256": "159d62f8dd0f111d55563d67ae6367b0381d867bf72e2f54fe87e557a4804491",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/weighted_mean/Taobao_Action/weighted_mean.model",
    "checkpoint_sha256": "5a3797622335503e3cc7ab6cfa485577ab96a56e1f3a28c63dafe341f350bd6d",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/paired-analysis/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
