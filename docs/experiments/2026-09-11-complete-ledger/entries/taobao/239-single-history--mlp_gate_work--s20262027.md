## 239. single-history--mlp_gate_work--s20262027

- 假设：First-work-conditioned modulation can improve second-read history relative to entry conditioning.
- 判定：discard
- 原因：Macro delta versus A -0.00590641, exploratory CI [-0.00892655,-0.00283662], forward latency ratio 1.330854. Fails both predeclared quality/near-candidate rules; retain all artifacts as negative research evidence, with no claim that every interval establishes harm.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "single-history--mlp_gate_work--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-10T03:17:15.913140+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "mlp_gate_work",
  "code_id": "b973d173c1edbfcedffee2c6ee09f14ebe9b735d9ca52d41c7448e0433b6ab06",
  "hypothesis": "First-work-conditioned modulation can improve second-read history relative to entry conditioning.",
  "change": "{\"model\": \"QFormerCross8SingleHistory\", \"maturity_mode\": \"mature\", \"history_encoder_kind\": \"mlp\", \"history_read_mode\": \"original_encoded\", \"history_condition\": \"work\", \"history_gate_rank\": 32, \"history_bottleneck\": 128, \"work_steps\": 2, \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8SingleHistory",
    "maturity_mode": "mature",
    "history_encoder_kind": "mlp",
    "history_read_mode": "original_encoded",
    "history_condition": "work",
    "history_gate_rank": 32,
    "history_bottleneck": 128,
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
      "value": 0.7611885433598942,
      "delta": -0.005906407954966358
    },
    "per_task": {
      "is_click": {
        "logloss": 0.18422545099562573,
        "AUC": 0.6493754819709477
      },
      "cart": {
        "logloss": 0.022005031105704953,
        "AUC": 0.7890267184047406
      },
      "fav": {
        "logloss": 0.017271596141309055,
        "AUC": 0.7902138356246206
      },
      "buy": {
        "logloss": 0.009854838003640419,
        "AUC": 0.8161381374392677
      }
    },
    "macro_source_logloss": 0.05833922906157004,
    "source_metric_comparison": {
      "is_click": {
        "source_AUC": 0.6493754819709477,
        "computed_AUC": 0.6493754819709477,
        "auc_difference": 0.0,
        "source_logloss": 0.18422545099562573,
        "computed_bce": 0.1842254509956306,
        "bce_difference": 4.85722573273506e-15
      },
      "cart": {
        "source_AUC": 0.7890267184047406,
        "computed_AUC": 0.7890267184047406,
        "auc_difference": 0.0,
        "source_logloss": 0.022005031105704953,
        "computed_bce": 0.022005031106041562,
        "bce_difference": 3.366092127254916e-13
      },
      "fav": {
        "source_AUC": 0.7902138356246206,
        "computed_AUC": 0.7902138356246206,
        "auc_difference": 0.0,
        "source_logloss": 0.017271596141309055,
        "computed_bce": 0.01727159615534347,
        "bce_difference": 1.4034415990460403e-11
      },
      "buy": {
        "source_AUC": 0.8161381374392677,
        "computed_AUC": 0.8161381374392677,
        "auc_difference": 0.0,
        "source_logloss": 0.009854838003640419,
        "computed_bce": 0.0098548380044063,
        "bce_difference": 7.658804146437603e-13
      }
    },
    "matched_parent_effect": {
      "coefficients": {
        "mature_r2": -1.0,
        "cross8_mature": 0.0,
        "hist_shallow_deep": 0.0,
        "sa_01": 0.0,
        "sa_11": 0.0,
        "sa_10": 0.0,
        "mlp_01": 0.0,
        "mlp_11": 0.0,
        "mlp_10": 0.0,
        "mlp_gate_entry": 0.0,
        "mlp_gate_work": 1.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": -0.005906407954966358,
          "ci95": [
            -0.008926550947056182,
            -0.002836623341080538
          ]
        },
        "auc.is_click": {
          "estimate": -0.0005608112300417023,
          "ci95": [
            -0.0013041488564865089,
            0.00016783069631016987
          ]
        },
        "auc.cart": {
          "estimate": -0.003407551952603116,
          "ci95": [
            -0.008434587316880654,
            0.0023032119225543128
          ]
        },
        "auc.fav": {
          "estimate": -0.012249654750427497,
          "ci95": [
            -0.018155084130121887,
            -0.00647701847232499
          ]
        },
        "auc.buy": {
          "estimate": -0.0074076138867931185,
          "ci95": [
            -0.015772159005420193,
            0.00038285887836039513
          ]
        },
        "bce.is_click": {
          "estimate": -0.00016619527215608332,
          "ci95": [
            -0.00023912322930395163,
            -9.983225175646053e-05
          ]
        },
        "bce.cart": {
          "estimate": 0.00017707673660753015,
          "ci95": [
            6.021144517660264e-05,
            0.00029313133787934985
          ]
        },
        "bce.fav": {
          "estimate": 0.00047343846099113726,
          "ci95": [
            0.00034005936148786794,
            0.0005961645500109118
          ]
        },
        "bce.buy": {
          "estimate": 0.00021103606768059675,
          "ci95": [
            0.00013107397063064817,
            0.00031060616667205506
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": 0.002846740722561414,
          "ci95": [
            -0.0029418589792972486,
            0.007951648157679712
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": 0.011688843520385794,
          "ci95": [
            0.005973179667536671,
            0.017704514275167886
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": 0.006846802656751416,
          "ci95": [
            -0.0007777698587707436,
            0.015056510399488864
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": 0.00884210279782438,
          "ci95": [
            0.0009251378726186698,
            0.017162464204249578
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": 0.004000061934190002,
          "ci95": [
            -0.005565744341290892,
            0.01381435700631944
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": -0.004842040863634378,
          "ci95": [
            -0.014600462316143259,
            0.004748601368585449
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": -0.00034327200876361347,
          "ci95": [
            -0.0004895585594977893,
            -0.00020628686956702445
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": -0.0006396337331472206,
          "ci95": [
            -0.0008041971313198162,
            -0.00048720595795629897
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": -0.00037723133983668007,
          "ci95": [
            -0.0005010448787658879,
            -0.00026416733323093253
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": -0.0002963617243836071,
          "ci95": [
            -0.00047548049413348697,
            -0.00012057725365275472
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": -3.3959331073066604e-05,
          "ci95": [
            -0.0001893138499106777,
            0.00011160297832694957
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": 0.0002624023933105405,
          "ci95": [
            0.00010077530749105867,
            0.0004114123405594629
          ]
        }
      }
    },
    "all_prespecified_effects_involving_candidate": {
      "mlp_gate_work__minus__mature_r2": {
        "coefficients": {
          "mature_r2": -1.0,
          "cross8_mature": 0.0,
          "hist_shallow_deep": 0.0,
          "sa_01": 0.0,
          "sa_11": 0.0,
          "sa_10": 0.0,
          "mlp_01": 0.0,
          "mlp_11": 0.0,
          "mlp_10": 0.0,
          "mlp_gate_entry": 0.0,
          "mlp_gate_work": 1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.005906407954966358,
            "ci95": [
              -0.008926550947056182,
              -0.002836623341080538
            ]
          },
          "auc.is_click": {
            "estimate": -0.0005608112300417023,
            "ci95": [
              -0.0013041488564865089,
              0.00016783069631016987
            ]
          },
          "auc.cart": {
            "estimate": -0.003407551952603116,
            "ci95": [
              -0.008434587316880654,
              0.0023032119225543128
            ]
          },
          "auc.fav": {
            "estimate": -0.012249654750427497,
            "ci95": [
              -0.018155084130121887,
              -0.00647701847232499
            ]
          },
          "auc.buy": {
            "estimate": -0.0074076138867931185,
            "ci95": [
              -0.015772159005420193,
              0.00038285887836039513
            ]
          },
          "bce.is_click": {
            "estimate": -0.00016619527215608332,
            "ci95": [
              -0.00023912322930395163,
              -9.983225175646053e-05
            ]
          },
          "bce.cart": {
            "estimate": 0.00017707673660753015,
            "ci95": [
              6.021144517660264e-05,
              0.00029313133787934985
            ]
          },
          "bce.fav": {
            "estimate": 0.00047343846099113726,
            "ci95": [
              0.00034005936148786794,
              0.0005961645500109118
            ]
          },
          "bce.buy": {
            "estimate": 0.00021103606768059675,
            "ci95": [
              0.00013107397063064817,
              0.00031060616667205506
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.002846740722561414,
            "ci95": [
              -0.0029418589792972486,
              0.007951648157679712
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.011688843520385794,
            "ci95": [
              0.005973179667536671,
              0.017704514275167886
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.006846802656751416,
            "ci95": [
              -0.0007777698587707436,
              0.015056510399488864
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.00884210279782438,
            "ci95": [
              0.0009251378726186698,
              0.017162464204249578
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.004000061934190002,
            "ci95": [
              -0.005565744341290892,
              0.01381435700631944
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.004842040863634378,
            "ci95": [
              -0.014600462316143259,
              0.004748601368585449
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00034327200876361347,
            "ci95": [
              -0.0004895585594977893,
              -0.00020628686956702445
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0006396337331472206,
            "ci95": [
              -0.0008041971313198162,
              -0.00048720595795629897
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.00037723133983668007,
            "ci95": [
              -0.0005010448787658879,
              -0.00026416733323093253
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.0002963617243836071,
            "ci95": [
              -0.00047548049413348697,
              -0.00012057725365275472
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -3.3959331073066604e-05,
            "ci95": [
              -0.0001893138499106777,
              0.00011160297832694957
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.0002624023933105405,
            "ci95": [
              0.00010077530749105867,
              0.0004114123405594629
            ]
          }
        }
      },
      "mlp_gate_work__minus__cross8_mature": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": -1.0,
          "hist_shallow_deep": 0.0,
          "sa_01": 0.0,
          "sa_11": 0.0,
          "sa_10": 0.0,
          "mlp_01": 0.0,
          "mlp_11": 0.0,
          "mlp_10": 0.0,
          "mlp_gate_entry": 0.0,
          "mlp_gate_work": 1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.0017821232644343044,
            "ci95": [
              -0.001222296320262163,
              0.00494046816459492
            ]
          },
          "auc.is_click": {
            "estimate": -0.0005612881720179486,
            "ci95": [
              -0.001309177433087552,
              0.00016943341156899723
            ]
          },
          "auc.cart": {
            "estimate": -0.00018001696637537368,
            "ci95": [
              -0.005423091760698547,
              0.0047815579897359396
            ]
          },
          "auc.fav": {
            "estimate": -0.006288175402828999,
            "ci95": [
              -0.012291257664044774,
              6.21029931862514e-05
            ]
          },
          "auc.buy": {
            "estimate": 0.014157973598959539,
            "ci95": [
              0.005737482230542224,
              0.023793319588124716
            ]
          },
          "bce.is_click": {
            "estimate": -9.248685296076031e-05,
            "ci95": [
              -0.00016155572216355814,
              -2.258054351225725e-05
            ]
          },
          "bce.cart": {
            "estimate": 0.00017061946896253233,
            "ci95": [
              4.424686996287669e-05,
              0.00031374107839731254
            ]
          },
          "bce.fav": {
            "estimate": 0.0002982169738904236,
            "ci95": [
              0.00018675097477631916,
              0.0004079918818824496
            ]
          },
          "bce.buy": {
            "estimate": -0.0001525523240318294,
            "ci95": [
              -0.0002522440937146307,
              -5.7506355244158195e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.0003812712056425749,
            "ci95": [
              -0.00549003444160959,
              0.004804884840571521
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.005726887230811051,
            "ci95": [
              -0.0007234518539801033,
              0.011767839206959799
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.014719261770977488,
            "ci95": [
              -0.024300996649553753,
              -0.006329031263881272
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.006108158436453626,
            "ci95": [
              -0.0012457920665212615,
              0.013603930346601494
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.014337990565334913,
            "ci95": [
              -0.025063315684030273,
              -0.0048181456065686246
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.02044614900178854,
            "ci95": [
              -0.03192988163623248,
              -0.010172098692113015
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00026310632192329264,
            "ci95": [
              -0.00040758576230722783,
              -0.00012580570187654494
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.00039070382685118393,
            "ci95": [
              -0.0005156168470803195,
              -0.00024894823998805187
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 6.0065471071069076e-05,
            "ci95": [
              -5.639666264636903e-05,
              0.00018457568191763556
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.0001275975049278913,
            "ci95": [
              -0.00028313352431441055,
              4.523927115514644e-05
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.0003231717929943617,
            "ci95": [
              0.00017264790594164796,
              0.0005014115967757373
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.000450769297922253,
            "ci95": [
              0.0003206328725040907,
              0.0005991712163567468
            ]
          }
        }
      },
      "mlp_gate_work__minus__mlp_gate_entry": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "hist_shallow_deep": 0.0,
          "sa_01": 0.0,
          "sa_11": 0.0,
          "sa_10": 0.0,
          "mlp_01": 0.0,
          "mlp_11": 0.0,
          "mlp_10": 0.0,
          "mlp_gate_entry": -1.0,
          "mlp_gate_work": 1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.002127208468936548,
            "ci95": [
              -0.005072207957567672,
              0.0011542932041979845
            ]
          },
          "auc.is_click": {
            "estimate": -0.0010454761543259306,
            "ci95": [
              -0.0018139909724279647,
              -0.00034020650884539785
            ]
          },
          "auc.cart": {
            "estimate": -0.00745778273105524,
            "ci95": [
              -0.014325757530057192,
              -0.0014465264149976904
            ]
          },
          "auc.fav": {
            "estimate": -0.006317987716684015,
            "ci95": [
              -0.012455671767016055,
              -0.00024634535900410794
            ]
          },
          "auc.buy": {
            "estimate": 0.006312412726318994,
            "ci95": [
              -0.001903349655278799,
              0.013180331259956626
            ]
          },
          "bce.is_click": {
            "estimate": 0.00019336094575256313,
            "ci95": [
              0.00012721325991166683,
              0.0002647007605074997
            ]
          },
          "bce.cart": {
            "estimate": 0.00023109539752538474,
            "ci95": [
              9.366045388694098e-05,
              0.0004019964623918928
            ]
          },
          "bce.fav": {
            "estimate": 0.00032502929416583504,
            "ci95": [
              0.00020544581195525662,
              0.0004548822101946318
            ]
          },
          "bce.buy": {
            "estimate": -0.00013161377774672904,
            "ci95": [
              -0.00021871458264048944,
              -4.131246031010253e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.00641230657672931,
            "ci95": [
              0.000502332888985191,
              0.013097635028445055
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.005272511562358084,
            "ci95": [
              -0.000772800854717695,
              0.01138723698040189
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.007357888880644925,
            "ci95": [
              -0.014399468543034996,
              0.0006135023219770877
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.0011397950143712254,
            "ci95": [
              -0.009963557002050469,
              0.007446844600116035
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.013770195457374235,
            "ci95": [
              -0.02398221306680787,
              -0.0028821915341564553
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.01263040044300301,
            "ci95": [
              -0.02250146716294579,
              -0.002382966534247469
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -3.7734451772821614e-05,
            "ci95": [
              -0.00020815948427316033,
              0.00011754370491832235
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0001316683484132719,
            "ci95": [
              -0.000276861623783861,
              7.767607907882762e-06
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.00032497472349929217,
            "ci95": [
              0.0002054804998713382,
              0.00044324944418863157
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -9.393389664045029e-05,
            "ci95": [
              -0.00027275499309744835,
              0.00010973274265881777
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.0003627091752721138,
            "ci95": [
              0.0001921036587103768,
              0.0005533895950905797
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.0004566430719125641,
            "ci95": [
              0.00031256315615580104,
              0.0006137994719244544
            ]
          }
        }
      },
      "mlp_gate_work__minus__mlp_01": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "hist_shallow_deep": 0.0,
          "sa_01": 0.0,
          "sa_11": 0.0,
          "sa_10": 0.0,
          "mlp_01": -1.0,
          "mlp_11": 0.0,
          "mlp_10": 0.0,
          "mlp_gate_entry": 0.0,
          "mlp_gate_work": 1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.0019250074403004136,
            "ci95": [
              -0.0012370754958358375,
              0.00524804025922324
            ]
          },
          "auc.is_click": {
            "estimate": -0.0010784789003018824,
            "ci95": [
              -0.0018718385733048932,
              -0.00038972147176257187
            ]
          },
          "auc.cart": {
            "estimate": 0.0002519478900415706,
            "ci95": [
              -0.005060783274691133,
              0.004857054084705159
            ]
          },
          "auc.fav": {
            "estimate": 0.0008899842671031832,
            "ci95": [
              -0.0065861443449619335,
              0.009830443801320664
            ]
          },
          "auc.buy": {
            "estimate": 0.007636576504358783,
            "ci95": [
              0.0005685056806052609,
              0.015909893601144764
            ]
          },
          "bce.is_click": {
            "estimate": -3.308968408038959e-05,
            "ci95": [
              -0.00010407910390486802,
              4.069390615068565e-05
            ]
          },
          "bce.cart": {
            "estimate": 0.00010660792108367623,
            "ci95": [
              -4.8453309157466464e-06,
              0.00023670618763274392
            ]
          },
          "bce.fav": {
            "estimate": 0.00021997778685431055,
            "ci95": [
              5.6679258954021526e-05,
              0.00036236849850688097
            ]
          },
          "bce.buy": {
            "estimate": -0.00013096033604371657,
            "ci95": [
              -0.00021026878899138663,
              -5.089181576571834e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.001330426790343453,
            "ci95": [
              -0.005915236704084925,
              0.0039643738929958415
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.0019684631674050657,
            "ci95": [
              -0.011036276772738714,
              0.005451550850257887
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.008715055404660665,
            "ci95": [
              -0.017105156573826007,
              -0.0016359556668069913
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.0006380363770616126,
            "ci95": [
              -0.010864406419774356,
              0.007788258717458095
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.007384628614317212,
            "ci95": [
              -0.015947820820012622,
              0.0012722301203228425
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.0067465922372556,
            "ci95": [
              -0.01855864126394519,
              0.004716532797378014
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00013969760516406582,
            "ci95": [
              -0.00028931131458042204,
              -7.124598915900677e-06
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.00025306747093470014,
            "ci95": [
              -0.0004082511768556785,
              -7.948064002957249e-05
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 9.787065196332698e-05,
            "ci95": [
              -1.240453739062429e-05,
              0.00020535278708107182
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.00011336986577063432,
            "ci95": [
              -0.0002931352692425379,
              8.7006847055046e-05
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.0002375682571273928,
            "ci95": [
              9.80944530225201e-05,
              0.00040408047543961684
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.0003509381228980271,
            "ci95": [
              0.00017234284724844273,
              0.0005246494693430548
            ]
          }
        }
      }
    },
    "parameter_count": 51915364,
    "initializer_sha256": "2bfd0a790078c5fd31b60394cea2e93f84831741fcd59502d78c5129020d074c",
    "optimizer_layout_sha256": "6190ac32ea9602fe0376b85636d1980ba8cdd06969af3c1f2c41daa3cd949a04",
    "formal_worker_cost": {
      "wall_seconds": 2795.627824872732,
      "cpu_seconds": 3308.538712535,
      "peak_rss_bytes": 17096470528,
      "gpu_peak_bytes": 35356934144,
      "samples": 2565
    },
    "smoke_worker_cost": {
      "wall_seconds": 372.1879269257188,
      "cpu_seconds": 748.0175769309999,
      "peak_rss_bytes": 21358002176,
      "gpu_peak_bytes": 38303432704,
      "samples": 340
    },
    "complete_recorded_costs": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2795.627824872732,
        "cpu_seconds": 3308.538712535,
        "peak_rss_bytes": 17096470528,
        "gpu_peak_bytes": 35356934144,
        "samples": 2565
      },
      "formal_runner": {
        "wall_seconds": 2791.9022495672107,
        "cpu_seconds": 3179.3194538790003,
        "peak_process_rss_bytes": 5362212864
      },
      "fit_including_validation": {
        "wall_seconds": 2655.282621279359,
        "cpu_seconds": 2996.456239095,
        "peak_process_rss_bytes": 5362212864
      },
      "prediction_export": {
        "wall_seconds": 131.2333941012621,
        "cpu_seconds": 177.27610152299985,
        "peak_process_rss_bytes": 5362212864
      },
      "smoke_worker": {
        "wall_seconds": 372.1879269257188,
        "cpu_seconds": 748.0175769309999,
        "peak_rss_bytes": 21358002176,
        "gpu_peak_bytes": 38303432704,
        "samples": 340
      },
      "smoke_runner": {
        "wall_seconds": 366.3057048469782,
        "cpu_seconds": 302.939835425,
        "peak_process_rss_bytes": 4047511552
      },
      "latency_constructor": {
        "wall_seconds": 0.7362759038805962,
        "cpu_seconds": 0.8097608479999998,
        "peak_rss_bytes": 2165723136
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.11422152817249298,
        "cpu_seconds": 0.11374941200000066,
        "peak_rss_bytes": 2165723136
      },
      "latency_peak_cuda_bytes": 1675310080,
      "latency_peak_cuda_reserved_bytes": 2042626048
    },
    "forward_latency": {
      "mean_ms": 18.90602402687073,
      "p50_ms": 18.882640838623047,
      "p95_ms": 18.972338294982908,
      "batch_size": 1024,
      "warmups": 10,
      "iterations": 40
    },
    "latency_ratio_to_parent": 1.3308537388599342,
    "quality_candidate": false,
    "near_candidate": false
  },
  "validity": {
    "valid": true,
    "violations": [],
    "state": "passed",
    "quality_summary": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/quality-summary.json",
    "no_prediction_to_work_feedback": true,
    "test_evaluated": false,
    "source_and_input_identity_scope": "Original N full-input content SHA256 values are inherited through completed C quality and E authorization. Production runners checked input stats before/after their traversals and two original metadata hashes before them. This collector now checks 122 current stats and both original metadata hashes; it does not rehash Parquet or open checkpoint/prediction arrays. New E worker/smoke/effective-optimizer gates are evaluated from actual small receipts. Three references retain exact C quality records and producer completion hashes, with small source/artifact identity checks. Stat continuity cannot independently exclude content changes that preserve stat fields."
  },
  "decision": "discard",
  "decision_reason": "Macro delta versus A -0.00590641, exploratory CI [-0.00892655,-0.00283662], forward latency ratio 1.330854. Fails both predeclared quality/near-candidate rules; retain all artifacts as negative research evidence, with no claim that every interval establishes harm.",
  "decision_scope": "Root queue disposition only; no released-incumbent promotion or supported lesson.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_gate_work",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_gate_work/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_gate_work/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_gate_work/Taobao_Action/mlp_gate_work.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/paired-analysis/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/latency/completion.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_gate_work",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_gate_work/result.json",
    "result_sha256": "420968af1fe679dc2ca9c66216f5237d165985fd12e4c7f490bd0d9960dff356",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_gate_work/manifest.json",
    "manifest_sha256": "5f093ceb3227bd5c3e908fb67359dd94243d8bb5c9cf1e702a622ca4911e5e5d",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_gate_work/Taobao_Action/mlp_gate_work.model",
    "checkpoint_sha256": "4e5e2ec5090783fa281220fc09317a1a122c58057dbae4dd87ccc57594a20860",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/paired-analysis/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
