## 248. moment-query--uniform_variance--s20262027

- 假设：First-read uniform_variance projected into second-work w_q before q_norm may improve addressing under fixed two-cycle mature protocol.
- 判定：discard
- 原因：Macro delta -0.006015812, CI95 [-0.008745644480710763, -0.003450261280446835]; forward ratio 1.162326. Fails fixed positive/near quality-cost policy.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "moment-query--uniform_variance--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-10T07:45:38.318827+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "uniform_variance",
  "code_id": "6503505e6c1699b76ec8cce57b783280806e825c1df3cd8278fa7c9bd751c711",
  "hypothesis": "First-read uniform_variance projected into second-work w_q before q_norm may improve addressing under fixed two-cycle mature protocol.",
  "change": "{\"model\": \"QFormerCross8MomentQuery\", \"maturity_mode\": \"mature\", \"moment_source\": \"uniform_variance\", \"moment_rank\": 16, \"work_steps\": 2, \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8MomentQuery",
    "maturity_mode": "mature",
    "moment_source": "uniform_variance",
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
      "value": 0.7610791395136743,
      "delta": -0.006015811801186127
    },
    "per_task": {
      "is_click": {
        "logloss": 0.18446746984707063,
        "AUC": 0.6480866024911843
      },
      "cart": {
        "logloss": 0.021963474273989366,
        "AUC": 0.788019491006907
      },
      "fav": {
        "logloss": 0.017144429842815355,
        "AUC": 0.7938217376532534
      },
      "buy": {
        "logloss": 0.009804036561829769,
        "AUC": 0.8143887269033527
      }
    },
    "macro_source_logloss": 0.05834485263142628,
    "source_metric_comparison": {
      "is_click": {
        "source_AUC": 0.6480866024911843,
        "computed_AUC": 0.6480866024911843,
        "auc_difference": 0.0,
        "source_logloss": 0.18446746984707063,
        "computed_bce": 0.18446746984707382,
        "bce_difference": 3.191891195797325e-15
      },
      "cart": {
        "source_AUC": 0.788019491006907,
        "computed_AUC": 0.788019491006907,
        "auc_difference": 0.0,
        "source_logloss": 0.021963474273989366,
        "computed_bce": 0.02196347429102796,
        "bce_difference": 1.7038592758922277e-11
      },
      "fav": {
        "source_AUC": 0.7938217376532534,
        "computed_AUC": 0.7938217376532534,
        "auc_difference": 0.0,
        "source_logloss": 0.017144429842815355,
        "computed_bce": 0.017144429854144456,
        "bce_difference": 1.1329100851886764e-11
      },
      "buy": {
        "source_AUC": 0.8143887269033527,
        "computed_AUC": 0.8143887269033527,
        "auc_difference": 0.0,
        "source_logloss": 0.009804036561829769,
        "computed_bce": 0.009804036584779404,
        "bce_difference": 2.2949635247737632e-11
      }
    },
    "matched_parent_effect": {
      "coefficients": {
        "mature_r2": -1.0,
        "cross8_mature": 0.0,
        "weighted_variance": 0.0,
        "weighted_mean": 0.0,
        "uniform_variance": 1.0,
        "constant": 0.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": -0.006015811801186127,
          "ci95": [
            -0.008745644480710763,
            -0.003450261280446835
          ]
        },
        "auc.is_click": {
          "estimate": -0.0018496907098051185,
          "ci95": [
            -0.0026029382142647244,
            -0.0010084121257545073
          ]
        },
        "auc.cart": {
          "estimate": -0.0044147793504366595,
          "ci95": [
            -0.008484981834563591,
            -0.00018648607032111872
          ]
        },
        "auc.fav": {
          "estimate": -0.008641752721794682,
          "ci95": [
            -0.01496238916691654,
            -0.0025986061664790153
          ]
        },
        "auc.buy": {
          "estimate": -0.009157024422708049,
          "ci95": [
            -0.01639650773325519,
            -0.002267768462051856
          ]
        },
        "bce.is_click": {
          "estimate": 7.582357928714867e-05,
          "ci95": [
            -3.558883421218816e-07,
            0.00015059702456162515
          ]
        },
        "bce.cart": {
          "estimate": 0.00013551992159392678,
          "ci95": [
            2.9280868849633235e-05,
            0.00022803356100289026
          ]
        },
        "bce.fav": {
          "estimate": 0.00034627215979212214,
          "ci95": [
            0.00019529364032450734,
            0.0004935716550600397
          ]
        },
        "bce.buy": {
          "estimate": 0.0001602346480537014,
          "ci95": [
            8.105350415934044e-05,
            0.00024313056859646208
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": 0.002565088640631541,
          "ci95": [
            -0.0016788773557803087,
            0.006682105002330101
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": 0.006792062011989564,
          "ci95": [
            0.0008532913798328743,
            0.01323465027588386
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": 0.00730733371290293,
          "ci95": [
            0.0005270046216613867,
            0.01436404617803112
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": 0.004226973371358023,
          "ci95": [
            -0.003531100928550862,
            0.01168959392209506
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": 0.004742245072271389,
          "ci95": [
            -0.002102999104315134,
            0.013045372592576527
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": 0.0005152717009133667,
          "ci95": [
            -0.0090857201308096,
            0.010062210260836622
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": -5.9696342306778116e-05,
          "ci95": [
            -0.00017900082828584796,
            5.2476566331227435e-05
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": -0.0002704485805049735,
          "ci95": [
            -0.0004163877249418255,
            -0.00011224869292783273
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": -8.441106876655273e-05,
          "ci95": [
            -0.0001955471030304098,
            1.5011886406890622e-05
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": -0.00021075223819819536,
          "ci95": [
            -0.00037945780711575675,
            -2.4611502869950777e-05
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": -2.4714726459774614e-05,
          "ci95": [
            -0.00013918280514928991,
            7.783464970242999e-05
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": 0.00018603751173842074,
          "ci95": [
            1.819497465199824e-05,
            0.000359390579232102
          ]
        }
      }
    },
    "all_prespecified_effects_involving_candidate": {
      "uniform_variance__minus__mature_r2": {
        "coefficients": {
          "mature_r2": -1.0,
          "cross8_mature": 0.0,
          "weighted_variance": 0.0,
          "weighted_mean": 0.0,
          "uniform_variance": 1.0,
          "constant": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.006015811801186127,
            "ci95": [
              -0.008745644480710763,
              -0.003450261280446835
            ]
          },
          "auc.is_click": {
            "estimate": -0.0018496907098051185,
            "ci95": [
              -0.0026029382142647244,
              -0.0010084121257545073
            ]
          },
          "auc.cart": {
            "estimate": -0.0044147793504366595,
            "ci95": [
              -0.008484981834563591,
              -0.00018648607032111872
            ]
          },
          "auc.fav": {
            "estimate": -0.008641752721794682,
            "ci95": [
              -0.01496238916691654,
              -0.0025986061664790153
            ]
          },
          "auc.buy": {
            "estimate": -0.009157024422708049,
            "ci95": [
              -0.01639650773325519,
              -0.002267768462051856
            ]
          },
          "bce.is_click": {
            "estimate": 7.582357928714867e-05,
            "ci95": [
              -3.558883421218816e-07,
              0.00015059702456162515
            ]
          },
          "bce.cart": {
            "estimate": 0.00013551992159392678,
            "ci95": [
              2.9280868849633235e-05,
              0.00022803356100289026
            ]
          },
          "bce.fav": {
            "estimate": 0.00034627215979212214,
            "ci95": [
              0.00019529364032450734,
              0.0004935716550600397
            ]
          },
          "bce.buy": {
            "estimate": 0.0001602346480537014,
            "ci95": [
              8.105350415934044e-05,
              0.00024313056859646208
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.002565088640631541,
            "ci95": [
              -0.0016788773557803087,
              0.006682105002330101
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.006792062011989564,
            "ci95": [
              0.0008532913798328743,
              0.01323465027588386
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.00730733371290293,
            "ci95": [
              0.0005270046216613867,
              0.01436404617803112
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.004226973371358023,
            "ci95": [
              -0.003531100928550862,
              0.01168959392209506
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.004742245072271389,
            "ci95": [
              -0.002102999104315134,
              0.013045372592576527
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.0005152717009133667,
            "ci95": [
              -0.0090857201308096,
              0.010062210260836622
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -5.9696342306778116e-05,
            "ci95": [
              -0.00017900082828584796,
              5.2476566331227435e-05
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0002704485805049735,
            "ci95": [
              -0.0004163877249418255,
              -0.00011224869292783273
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -8.441106876655273e-05,
            "ci95": [
              -0.0001955471030304098,
              1.5011886406890622e-05
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.00021075223819819536,
            "ci95": [
              -0.00037945780711575675,
              -2.4611502869950777e-05
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -2.4714726459774614e-05,
            "ci95": [
              -0.00013918280514928991,
              7.783464970242999e-05
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.00018603751173842074,
            "ci95": [
              1.819497465199824e-05,
              0.000359390579232102
            ]
          }
        }
      },
      "uniform_variance__minus__cross8_mature": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": -1.0,
          "weighted_variance": 0.0,
          "weighted_mean": 0.0,
          "uniform_variance": 1.0,
          "constant": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.0016727194182145355,
            "ci95": [
              -0.0014210356992132992,
              0.00511525888900908
            ]
          },
          "auc.is_click": {
            "estimate": -0.0018501676517813648,
            "ci95": [
              -0.0027356346671221802,
              -0.0010761071309595427
            ]
          },
          "auc.cart": {
            "estimate": -0.001187244364208917,
            "ci95": [
              -0.005610039649860549,
              0.0033452787067954756
            ]
          },
          "auc.fav": {
            "estimate": -0.002680273374196185,
            "ci95": [
              -0.008656089952790541,
              0.00356999360475606
            ]
          },
          "auc.buy": {
            "estimate": 0.012408563063044609,
            "ci95": [
              0.002835496892896805,
              0.024255898010031165
            ]
          },
          "bce.is_click": {
            "estimate": 0.00014953199848247167,
            "ci95": [
              7.301581320674835e-05,
              0.00022899718073292604
            ]
          },
          "bce.cart": {
            "estimate": 0.00012906265394892896,
            "ci95": [
              2.3633891790617903e-05,
              0.00022520653430104317
            ]
          },
          "bce.fav": {
            "estimate": 0.0001710506726914085,
            "ci95": [
              6.759785840962513e-05,
              0.0002719699777539775
            ]
          },
          "bce.buy": {
            "estimate": -0.00020335374365872474,
            "ci95": [
              -0.00031789426670017935,
              -0.00010028280359878062
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.0006629232875724478,
            "ci95": [
              -0.005062222360212485,
              0.0035690046618762792
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.0008301057224148201,
            "ci95": [
              -0.005246743948442156,
              0.006755402204929395
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.014258730714825973,
            "ci95": [
              -0.02578660865672537,
              -0.004610445263398228
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.0014930290099872678,
            "ci95": [
              -0.005790968785069456,
              0.009051146668588868
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.013595807427253526,
            "ci95": [
              -0.025972339980213754,
              -0.0027543143646250384
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.015088836437240793,
            "ci95": [
              -0.0274172227264203,
              -0.0031475057288820373
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 2.046934453354271e-05,
            "ci95": [
              -9.639911080116647e-05,
              0.00014476350608520178
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -2.1518674208936828e-05,
            "ci95": [
              -0.0001373558334959219,
              0.00010973219174707709
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.0003528857421411964,
            "ci95": [
              0.00023522366254250573,
              0.0005091085226577358
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -4.198801874247954e-05,
            "ci95": [
              -0.00019243973525477328,
              0.00010536321026677323
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.0003324163976076537,
            "ci95": [
              0.00018935670562411188,
              0.0004918607754898966
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.00037440441635013325,
            "ci95": [
              0.00022209226062758344,
              0.0005196165436489993
            ]
          }
        }
      },
      "weighted_variance__minus__uniform_variance": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "weighted_variance": 1.0,
          "weighted_mean": 0.0,
          "uniform_variance": -1.0,
          "constant": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.003018002147557569,
            "ci95": [
              9.167537280184164e-05,
              0.005520583184496593
            ]
          },
          "auc.is_click": {
            "estimate": 0.000951775930532639,
            "ci95": [
              0.00014297745677526735,
              0.0017385178220825272
            ]
          },
          "auc.cart": {
            "estimate": 0.007285048028443986,
            "ci95": [
              0.0026277258414744127,
              0.011710599131552047
            ]
          },
          "auc.fav": {
            "estimate": 0.0016125458520793723,
            "ci95": [
              -0.00502695067212002,
              0.007959299223811148
            ]
          },
          "auc.buy": {
            "estimate": 0.0022226387791742797,
            "ci95": [
              -0.004852794634018748,
              0.008783675743769799
            ]
          },
          "bce.is_click": {
            "estimate": -0.0002984928244045004,
            "ci95": [
              -0.0003740195136190699,
              -0.0002254925075857367
            ]
          },
          "bce.cart": {
            "estimate": -0.00015642330547279565,
            "ci95": [
              -0.00025880888923498455,
              -5.5484922238189366e-05
            ]
          },
          "bce.fav": {
            "estimate": -5.6952749974237965e-05,
            "ci95": [
              -0.0001852582940801142,
              7.365153581451671e-05
            ]
          },
          "bce.buy": {
            "estimate": -0.00010859226464286255,
            "ci95": [
              -0.00018362069272142827,
              -3.564987618490615e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.006333272097911347,
            "ci95": [
              -0.011014441589988497,
              -0.001602167910228042
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.0006607699215467333,
            "ci95": [
              -0.00721806463893093,
              0.005826817334460169
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.0012708628486416407,
            "ci95": [
              -0.007727414134466465,
              0.005764080254292491
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.005672502176364613,
            "ci95": [
              -0.00213364854916806,
              0.014256181731033773
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.005062409249269706,
            "ci95": [
              -0.0029616352562864628,
              0.013403864271763958
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.0006100929270949074,
            "ci95": [
              -0.009734634627143548,
              0.009032846396008539
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00014206951893170477,
            "ci95": [
              -0.0002625617421785112,
              -1.612607009093067e-05
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.00024154007443026246,
            "ci95": [
              -0.0003946144150725248,
              -9.574988574552302e-05
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.00018990055976163787,
            "ci95": [
              -0.00030322606489840173,
              -8.883451437438966e-05
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -9.947055549855768e-05,
            "ci95": [
              -0.0002764136467444895,
              7.332735129048814e-05
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -4.78310408299331e-05,
            "ci95": [
              -0.00017103267286881622,
              8.579658837159913e-05
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 5.1639514668624584e-05,
            "ci95": [
              -0.00010030087937810011,
              0.00020800788996259617
            ]
          }
        }
      }
    },
    "parameter_count": 51736548,
    "initializer_sha256": "796cec890828fda2f18f4ecf0c052e81999552b8179857b95f1c309d637d6ccb",
    "optimizer_layout_sha256": "74c4e7b2c162fcc08a81818ce75528430233d1a0d32c8b1a35bf602ae01dfb77",
    "formal_worker_cost": {
      "wall_seconds": 2609.309548191726,
      "cpu_seconds": 3118.178352815,
      "peak_rss_bytes": 17156976640,
      "gpu_peak_bytes": 31359762432,
      "samples": 2390
    },
    "smoke_worker_cost": {
      "wall_seconds": 167.59183104336262,
      "cpu_seconds": 225.642715105,
      "peak_rss_bytes": 19711717376,
      "gpu_peak_bytes": 33832304640,
      "samples": 154
    },
    "complete_recorded_costs": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2609.309548191726,
        "cpu_seconds": 3118.178352815,
        "peak_rss_bytes": 17156976640,
        "gpu_peak_bytes": 31359762432,
        "samples": 2390
      },
      "formal_runner": {
        "wall_seconds": 2605.421052493155,
        "cpu_seconds": 3001.9935053020004,
        "peak_process_rss_bytes": 5270626304
      },
      "fit_including_validation": {
        "wall_seconds": 2475.459663592279,
        "cpu_seconds": 2823.857328107,
        "peak_process_rss_bytes": 5270626304
      },
      "prediction_export": {
        "wall_seconds": 124.97570534050465,
        "cpu_seconds": 172.6486653840002,
        "peak_process_rss_bytes": 5270626304
      },
      "smoke_worker": {
        "wall_seconds": 167.59183104336262,
        "cpu_seconds": 225.642715105,
        "peak_rss_bytes": 19711717376,
        "gpu_peak_bytes": 33832304640,
        "samples": 154
      },
      "smoke_runner": {
        "wall_seconds": 163.55575590580702,
        "cpu_seconds": 156.311133073,
        "peak_process_rss_bytes": 3294953472
      },
      "latency_constructor": {
        "wall_seconds": 0.8055965378880501,
        "cpu_seconds": 0.8719005580000001,
        "peak_rss_bytes": 2276122624
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.1287439838051796,
        "cpu_seconds": 0.1276145660000001,
        "peak_rss_bytes": 2276122624
      },
      "latency_peak_cuda_bytes": 1369385472,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "forward_latency": {
      "mean_ms": 16.69333186149597,
      "p50_ms": 16.675535202026367,
      "p95_ms": 16.83497257232666,
      "batch_size": 1024,
      "warmups": 10,
      "iterations": 40
    },
    "latency_ratio_to_parent": 1.162326213332689,
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
  "decision_reason": "Macro delta -0.006015812, CI95 [-0.008745644480710763, -0.003450261280446835]; forward ratio 1.162326. Fails fixed positive/near quality-cost policy.",
  "decision_scope": "Root queue disposition only; no released-incumbent promotion or supported lesson.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/uniform_variance",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/uniform_variance/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/uniform_variance/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/uniform_variance/Taobao_Action/uniform_variance.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/paired-analysis/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/latency/completion.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/uniform_variance",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/uniform_variance/result.json",
    "result_sha256": "d8e5acff92f0bef90a529d03d8770029595c375a16fde9c04cdf0ec82b6d4e61",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/uniform_variance/manifest.json",
    "manifest_sha256": "2a3234f31faa4fc0478d69143b8c9c8a2fbedd034f839705cf987ef37108e6d5",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/uniform_variance/Taobao_Action/uniform_variance.model",
    "checkpoint_sha256": "47222896f0e2ca3b9f48504b95de586f9e9072df150acedc2574c9ebc15c14b4",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/paired-analysis/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
