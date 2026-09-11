## 238. single-history--mlp_gate_entry--s20262027

- 假设：Entry-conditioned modulation can make the second-read history increment useful.
- 判定：discard
- 原因：Macro delta versus A -0.00377920, exploratory CI [-0.00735817,-0.00072761], forward latency ratio 1.329651. Fails both predeclared quality/near-candidate rules; retain all artifacts as negative research evidence, with no claim that every interval establishes harm.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "single-history--mlp_gate_entry--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-10T03:17:15.913140+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "mlp_gate_entry",
  "code_id": "b973d173c1edbfcedffee2c6ee09f14ebe9b735d9ca52d41c7448e0433b6ab06",
  "hypothesis": "Entry-conditioned modulation can make the second-read history increment useful.",
  "change": "{\"model\": \"QFormerCross8SingleHistory\", \"maturity_mode\": \"mature\", \"history_encoder_kind\": \"mlp\", \"history_read_mode\": \"original_encoded\", \"history_condition\": \"entry\", \"history_gate_rank\": 32, \"history_bottleneck\": 128, \"work_steps\": 2, \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8SingleHistory",
    "maturity_mode": "mature",
    "history_encoder_kind": "mlp",
    "history_read_mode": "original_encoded",
    "history_condition": "entry",
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
      "value": 0.7633157518288307,
      "delta": -0.0037791994860298106
    },
    "per_task": {
      "is_click": {
        "logloss": 0.18403209004987728,
        "AUC": 0.6504209581252737
      },
      "cart": {
        "logloss": 0.02177393559769394,
        "AUC": 0.7964845011357958
      },
      "fav": {
        "logloss": 0.016946566826470836,
        "AUC": 0.7965318233413046
      },
      "buy": {
        "logloss": 0.00998645178009348,
        "AUC": 0.8098257247129487
      }
    },
    "macro_source_logloss": 0.05818476106353388,
    "source_metric_comparison": {
      "is_click": {
        "source_AUC": 0.6504209581252737,
        "computed_AUC": 0.6504209581252737,
        "auc_difference": 0.0,
        "source_logloss": 0.18403209004987728,
        "computed_bce": 0.18403209004987803,
        "bce_difference": 7.494005416219807e-16
      },
      "cart": {
        "source_AUC": 0.7964845011357958,
        "computed_AUC": 0.7964845011357958,
        "auc_difference": 0.0,
        "source_logloss": 0.02177393559769394,
        "computed_bce": 0.021773935708516177,
        "bce_difference": 1.1082223680403125e-10
      },
      "fav": {
        "source_AUC": 0.7965318233413046,
        "computed_AUC": 0.7965318233413046,
        "auc_difference": 0.0,
        "source_logloss": 0.016946566826470836,
        "computed_bce": 0.016946566861177636,
        "bce_difference": 3.4706799934003385e-11
      },
      "buy": {
        "source_AUC": 0.8098257247129487,
        "computed_AUC": 0.8098257247129487,
        "auc_difference": 0.0,
        "source_logloss": 0.00998645178009348,
        "computed_bce": 0.009986451782153028,
        "bce_difference": 2.0595487121299882e-12
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
        "mlp_gate_entry": 1.0,
        "mlp_gate_work": 0.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": -0.0037791994860298106,
          "ci95": [
            -0.0073581702327787345,
            -0.0007276072941647469
          ]
        },
        "auc.is_click": {
          "estimate": 0.00048466492428422825,
          "ci95": [
            -0.0002075751624487726,
            0.0012085007058210693
          ]
        },
        "auc.cart": {
          "estimate": 0.004050230778452124,
          "ci95": [
            -0.0005795864540707462,
            0.009821503345962978
          ]
        },
        "auc.fav": {
          "estimate": -0.005931667033743482,
          "ci95": [
            -0.012263216258403705,
            -0.0006620800197152558
          ]
        },
        "auc.buy": {
          "estimate": -0.013720026613112113,
          "ci95": [
            -0.024123943911395683,
            -0.0047299303490265275
          ]
        },
        "bce.is_click": {
          "estimate": -0.00035955621790864645,
          "ci95": [
            -0.00042772635698331283,
            -0.00028487060316082364
          ]
        },
        "bce.cart": {
          "estimate": -5.4018660917854594e-05,
          "ci95": [
            -0.00017729178844296212,
            5.411624618844354e-05
          ]
        },
        "bce.fav": {
          "estimate": 0.00014840916682530222,
          "ci95": [
            1.3901589558121268e-05,
            0.00027898122616416946
          ]
        },
        "bce.buy": {
          "estimate": 0.0003426498454273258,
          "ci95": [
            0.0002636217483870769,
            0.0004388116557163171
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": -0.003565565854167896,
          "ci95": [
            -0.0092669123356429,
            0.0013873353294481354
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": 0.00641633195802771,
          "ci95": [
            0.0011414803783896894,
            0.012925263258312449
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": 0.014204691537396341,
          "ci95": [
            0.005139522883745407,
            0.02443211809577098
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": 0.009981897812195606,
          "ci95": [
            0.0029996658177419895,
            0.01786696409897753
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": 0.017770257391564237,
          "ci95": [
            0.007474386302430696,
            0.02862759523606165
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": 0.007788359579368631,
          "ci95": [
            -0.0028375658604843333,
            0.018569308893204954
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": -0.00030553755699079185,
          "ci95": [
            -0.0004386141165761092,
            -0.00017068371840431082
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": -0.0005079653847339487,
          "ci95": [
            -0.0006556605983974097,
            -0.00035757055048283383
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": -0.0007022060633359722,
          "ci95": [
            -0.0008181688313564567,
            -0.0005905082803061728
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": -0.00020242782774315682,
          "ci95": [
            -0.0003671094518502627,
            -3.566147874991054e-05
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": -0.0003966685063451804,
          "ci95": [
            -0.0005414665447339834,
            -0.0002583175553558326
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": -0.00019424067860202357,
          "ci95": [
            -0.00035479824123189755,
            -2.3284578282599133e-05
          ]
        }
      }
    },
    "all_prespecified_effects_involving_candidate": {
      "mlp_gate_entry__minus__mature_r2": {
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
          "mlp_gate_entry": 1.0,
          "mlp_gate_work": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.0037791994860298106,
            "ci95": [
              -0.0073581702327787345,
              -0.0007276072941647469
            ]
          },
          "auc.is_click": {
            "estimate": 0.00048466492428422825,
            "ci95": [
              -0.0002075751624487726,
              0.0012085007058210693
            ]
          },
          "auc.cart": {
            "estimate": 0.004050230778452124,
            "ci95": [
              -0.0005795864540707462,
              0.009821503345962978
            ]
          },
          "auc.fav": {
            "estimate": -0.005931667033743482,
            "ci95": [
              -0.012263216258403705,
              -0.0006620800197152558
            ]
          },
          "auc.buy": {
            "estimate": -0.013720026613112113,
            "ci95": [
              -0.024123943911395683,
              -0.0047299303490265275
            ]
          },
          "bce.is_click": {
            "estimate": -0.00035955621790864645,
            "ci95": [
              -0.00042772635698331283,
              -0.00028487060316082364
            ]
          },
          "bce.cart": {
            "estimate": -5.4018660917854594e-05,
            "ci95": [
              -0.00017729178844296212,
              5.411624618844354e-05
            ]
          },
          "bce.fav": {
            "estimate": 0.00014840916682530222,
            "ci95": [
              1.3901589558121268e-05,
              0.00027898122616416946
            ]
          },
          "bce.buy": {
            "estimate": 0.0003426498454273258,
            "ci95": [
              0.0002636217483870769,
              0.0004388116557163171
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.003565565854167896,
            "ci95": [
              -0.0092669123356429,
              0.0013873353294481354
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.00641633195802771,
            "ci95": [
              0.0011414803783896894,
              0.012925263258312449
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.014204691537396341,
            "ci95": [
              0.005139522883745407,
              0.02443211809577098
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.009981897812195606,
            "ci95": [
              0.0029996658177419895,
              0.01786696409897753
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.017770257391564237,
            "ci95": [
              0.007474386302430696,
              0.02862759523606165
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.007788359579368631,
            "ci95": [
              -0.0028375658604843333,
              0.018569308893204954
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00030553755699079185,
            "ci95": [
              -0.0004386141165761092,
              -0.00017068371840431082
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0005079653847339487,
            "ci95": [
              -0.0006556605983974097,
              -0.00035757055048283383
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0007022060633359722,
            "ci95": [
              -0.0008181688313564567,
              -0.0005905082803061728
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.00020242782774315682,
            "ci95": [
              -0.0003671094518502627,
              -3.566147874991054e-05
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -0.0003966685063451804,
            "ci95": [
              -0.0005414665447339834,
              -0.0002583175553558326
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.00019424067860202357,
            "ci95": [
              -0.00035479824123189755,
              -2.3284578282599133e-05
            ]
          }
        }
      },
      "mlp_gate_entry__minus__cross8_mature": {
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
          "mlp_gate_entry": 1.0,
          "mlp_gate_work": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.003909331733370852,
            "ci95": [
              0.001321724393199849,
              0.007100190070306134
            ]
          },
          "auc.is_click": {
            "estimate": 0.000484187982307982,
            "ci95": [
              -0.0002739458360468211,
              0.0012427797367778567
            ]
          },
          "auc.cart": {
            "estimate": 0.007277765764679867,
            "ci95": [
              0.002810459443030683,
              0.011495729897170865
            ]
          },
          "auc.fav": {
            "estimate": 2.9812313855015482e-05,
            "ci95": [
              -0.006123401175935997,
              0.0056519035040879925
            ]
          },
          "auc.buy": {
            "estimate": 0.007845560872640545,
            "ci95": [
              -0.0012449192196145788,
              0.01821261900818114
            ]
          },
          "bce.is_click": {
            "estimate": -0.00028584779871332344,
            "ci95": [
              -0.0003584289628391989,
              -0.0002142341560910033
            ]
          },
          "bce.cart": {
            "estimate": -6.0475928562852416e-05,
            "ci95": [
              -0.0001544496556344168,
              3.2301232965495556e-05
            ]
          },
          "bce.fav": {
            "estimate": -2.6812320275411416e-05,
            "ci95": [
              -0.00014386433768005924,
              9.91990424307934e-05
            ]
          },
          "bce.buy": {
            "estimate": -2.093854628510035e-05,
            "ci95": [
              -0.00011658657899580788,
              6.455735873961504e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.006793577782371885,
            "ci95": [
              -0.011168340340284732,
              -0.002423471958521474
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.0004543756684529665,
            "ci95": [
              -0.005580362433061053,
              0.006603502857615048
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.007361372890332563,
            "ci95": [
              -0.01806366639336474,
              0.0015889679804023803
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.007247953450824851,
            "ci95": [
              9.722688820948405e-05,
              0.014860110325825848
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.000567795107960678,
            "ci95": [
              -0.011357444276167849,
              0.009954422974514918
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.007815748558785529,
            "ci95": [
              -0.020375225690162625,
              0.0020427689919258075
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00022537187015047103,
            "ci95": [
              -0.00033768062343963925,
              -0.00010832361798481927
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.000259035478437912,
            "ci95": [
              -0.0004069947759697674,
              -0.00011139232696190045
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0002649092524282231,
            "ci95": [
              -0.0003904612852428628,
              -0.00014157898197120385
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -3.3663608287441e-05,
            "ci95": [
              -0.00019286304434112267,
              0.00011335252122833497
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -3.953738227775207e-05,
            "ci95": [
              -0.000164515935020326,
              0.00010657771111641069
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -5.873773990311068e-06,
            "ci95": [
              -0.000136861074637542,
              0.00016097557959105006
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
      "mlp_gate_entry__minus__mlp_01": {
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
          "mlp_gate_entry": 1.0,
          "mlp_gate_work": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.004052215909236961,
            "ci95": [
              0.0007062625259981548,
              0.0076638590995939965
            ]
          },
          "auc.is_click": {
            "estimate": -3.3002745975951875e-05,
            "ci95": [
              -0.0008090299400911244,
              0.0006519848158818675
            ]
          },
          "auc.cart": {
            "estimate": 0.007709730621096811,
            "ci95": [
              0.0027880711126056254,
              0.013032483456138126
            ]
          },
          "auc.fav": {
            "estimate": 0.007207971983787198,
            "ci95": [
              0.00032522309599748175,
              0.015099337248151682
            ]
          },
          "auc.buy": {
            "estimate": 0.0013241637780397886,
            "ci95": [
              -0.007241235378438421,
              0.011162937206992523
            ]
          },
          "bce.is_click": {
            "estimate": -0.00022645062983295272,
            "ci95": [
              -0.00029140801993736903,
              -0.0001566171617238044
            ]
          },
          "bce.cart": {
            "estimate": -0.00012448747644170852,
            "ci95": [
              -0.0002507860713331921,
              -2.1462622010975714e-05
            ]
          },
          "bce.fav": {
            "estimate": -0.00010505150731152449,
            "ci95": [
              -0.00023433025971135403,
              1.7551838314588458e-05
            ]
          },
          "bce.buy": {
            "estimate": 6.534417030124723e-07,
            "ci95": [
              -9.930726179171e-05,
              8.148397229172072e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.007742733367072763,
            "ci95": [
              -0.013001155597540707,
              -0.0028675658954604136
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.00724097472976315,
            "ci95": [
              -0.015007146771730654,
              -0.00017305031951892735
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.0013571665240157404,
            "ci95": [
              -0.011358017524946123,
              0.007475607395242659
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.0005017586373096128,
            "ci95": [
              -0.008365752230925718,
              0.008990123212228454
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.006385566843057022,
            "ci95": [
              -0.005336145466454356,
              0.01678774725675616
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.0058838082057474095,
            "ci95": [
              -0.005984997764941795,
              0.017399936348887837
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.0001019631533912442,
            "ci95": [
              -0.0002317249253686181,
              3.292876354823838e-05
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.00012139912252142823,
            "ci95": [
              -0.00026517815505543873,
              3.5131993777831774e-05
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0002271040715359652,
            "ci95": [
              -0.00034188656794460103,
              -0.00011051063276028206
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -1.9435969130184028e-05,
            "ci95": [
              -0.00018401937883901105,
              0.000137762476662453
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -0.000125140918144721,
            "ci95": [
              -0.00026460538891912505,
              2.256097475491743e-05
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.00010570494901453696,
            "ci95": [
              -0.00025076908109786705,
              6.169585030194832e-05
            ]
          }
        }
      }
    },
    "parameter_count": 51915364,
    "initializer_sha256": "2bfd0a790078c5fd31b60394cea2e93f84831741fcd59502d78c5129020d074c",
    "optimizer_layout_sha256": "470cb13ce88d6cec8b021e8a6388f0a6d7a740e0afca674391f36a84c3518574",
    "formal_worker_cost": {
      "wall_seconds": 2797.814953006804,
      "cpu_seconds": 3322.488007626,
      "peak_rss_bytes": 17079107584,
      "gpu_peak_bytes": 35361128448,
      "samples": 2567
    },
    "smoke_worker_cost": {
      "wall_seconds": 363.3127840310335,
      "cpu_seconds": 732.6299671910001,
      "peak_rss_bytes": 21325135872,
      "gpu_peak_bytes": 38303432704,
      "samples": 333
    },
    "complete_recorded_costs": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2797.814953006804,
        "cpu_seconds": 3322.488007626,
        "peak_rss_bytes": 17079107584,
        "gpu_peak_bytes": 35361128448,
        "samples": 2567
      },
      "formal_runner": {
        "wall_seconds": 2793.009986616671,
        "cpu_seconds": 3192.404120614,
        "peak_process_rss_bytes": 5424553984
      },
      "fit_including_validation": {
        "wall_seconds": 2655.057724826038,
        "cpu_seconds": 3006.824734512,
        "peak_process_rss_bytes": 5424553984
      },
      "prediction_export": {
        "wall_seconds": 132.64384725689888,
        "cpu_seconds": 180.05422202399996,
        "peak_process_rss_bytes": 5424553984
      },
      "smoke_worker": {
        "wall_seconds": 363.3127840310335,
        "cpu_seconds": 732.6299671910001,
        "peak_rss_bytes": 21325135872,
        "gpu_peak_bytes": 38303432704,
        "samples": 333
      },
      "smoke_runner": {
        "wall_seconds": 358.2261445298791,
        "cpu_seconds": 297.617862696,
        "peak_process_rss_bytes": 4024729600
      },
      "latency_constructor": {
        "wall_seconds": 0.7524328827857971,
        "cpu_seconds": 0.8231612879999979,
        "peak_rss_bytes": 2127785984
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.1393095999956131,
        "cpu_seconds": 0.1387498100000002,
        "peak_rss_bytes": 2127785984
      },
      "latency_peak_cuda_bytes": 1675310080,
      "latency_peak_cuda_reserved_bytes": 2042626048
    },
    "forward_latency": {
      "mean_ms": 18.888937520980836,
      "p50_ms": 18.887807846069336,
      "p95_ms": 18.917617416381837,
      "batch_size": 1024,
      "warmups": 10,
      "iterations": 40
    },
    "latency_ratio_to_parent": 1.3296509666527638,
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
  "decision_reason": "Macro delta versus A -0.00377920, exploratory CI [-0.00735817,-0.00072761], forward latency ratio 1.329651. Fails both predeclared quality/near-candidate rules; retain all artifacts as negative research evidence, with no claim that every interval establishes harm.",
  "decision_scope": "Root queue disposition only; no released-incumbent promotion or supported lesson.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_gate_entry",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_gate_entry/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_gate_entry/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_gate_entry/Taobao_Action/mlp_gate_entry.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/paired-analysis/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/latency/completion.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_gate_entry",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_gate_entry/result.json",
    "result_sha256": "4c6a94218554a83da08adbe0de768bafadff9b85104f41664a0521caaabede81",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_gate_entry/manifest.json",
    "manifest_sha256": "b47f41677a605c2bcee8fc864aa9e6fc0345ced3bf70cbbafb4985f6e759f8de",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_gate_entry/Taobao_Action/mlp_gate_entry.model",
    "checkpoint_sha256": "5750361a8c67a6a7e5a29e9aea79310742f655edc51bed6068e6e708dde8a297",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/paired-analysis/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
