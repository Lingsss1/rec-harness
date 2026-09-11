## 264. round-accumulation--distinct_no_carry--s20262027

- 假设：检验轮次区分的残差更新与首轮状态携带的交互，固定两轮、优化器和seed。
- 判定：discard
- 原因：相对原两轮macro AUC下降0.010544且区间在零以下，轮次更新脱离携带路径没有收益。

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "round-accumulation--distinct_no_carry--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-10T15:44:40.685443+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "round_accumulation_factorial",
  "code_id": "2511cf353e09f9b83e8b18b3e701b1d57b7471c2ebaeb2986b93add858312e94",
  "hypothesis": "检验轮次区分的残差更新与首轮状态携带的交互，固定两轮、优化器和seed。",
  "change": "{\"model\": \"QFormerCross8RoundAccumulation\", \"maturity_mode\": \"mature\", \"work_steps\": 2, \"num_queries\": 4, \"accumulation_distinct\": true, \"carry_first\": false}",
  "config_change": {
    "model": "QFormerCross8RoundAccumulation",
    "maturity_mode": "mature",
    "work_steps": 2,
    "num_queries": 4,
    "accumulation_distinct": true,
    "carry_first": false
  },
  "input_protocol_id": "strict-past-native-last100-action19-mature",
  "budget": {
    "formal_training_runs": 1,
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
      "value": 0.7565513137319784,
      "delta": -0.010543637582882126
    },
    "per_task": {
      "is_click": {
        "logloss": 0.18418887722284913,
        "AUC": 0.6474833190539738
      },
      "cart": {
        "logloss": 0.022037927736995255,
        "AUC": 0.7872136947814017
      },
      "fav": {
        "logloss": 0.01752105707139308,
        "AUC": 0.7823513932960032
      },
      "buy": {
        "logloss": 0.009929819743412987,
        "AUC": 0.8091568477965347
      }
    },
    "macro_source_logloss": 0.05841942044366261,
    "matched_parent_effect": {
      "coefficients": {
        "mature_r2": -1.0,
        "cross8_mature": 0.0,
        "shared_carry": 0.0,
        "distinct_carry": 0.0,
        "shared_no_carry": 0.0,
        "distinct_no_carry": 1.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": -0.010543637582882126,
          "ci95": [
            -0.01407057532398757,
            -0.007293229142440938
          ]
        },
        "auc.is_click": {
          "estimate": -0.0024529741470156186,
          "ci95": [
            -0.0032268243839037062,
            -0.001560778721852696
          ]
        },
        "auc.cart": {
          "estimate": -0.005220575575942021,
          "ci95": [
            -0.009627117060569948,
            -0.0005682010393333446
          ]
        },
        "auc.fav": {
          "estimate": -0.02011209707904482,
          "ci95": [
            -0.026770000091700892,
            -0.013305988068935884
          ]
        },
        "auc.buy": {
          "estimate": -0.014388903529526043,
          "ci95": [
            -0.025899706845661473,
            -0.005594036290646536
          ]
        },
        "bce.is_click": {
          "estimate": -0.00020276904493479742,
          "ci95": [
            -0.00028517884500233945,
            -0.00013502752303341665
          ]
        },
        "bce.cart": {
          "estimate": 0.00020997340766624323,
          "ci95": [
            0.00011109337144234603,
            0.000310226239397632
          ]
        },
        "bce.fav": {
          "estimate": 0.000722899434348865,
          "ci95": [
            0.0005704541059987241,
            0.0008730807232427345
          ]
        },
        "bce.buy": {
          "estimate": 0.00028601780668766584,
          "ci95": [
            0.000199676194606925,
            0.00040458709549040246
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": 0.002767601428926403,
          "ci95": [
            -0.0019173953796686877,
            0.0075168969513803125
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": 0.017659122932029203,
          "ci95": [
            0.010620477385969367,
            0.024450221291210522
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": 0.011935929382510424,
          "ci95": [
            0.0032232012260614705,
            0.023699758013590918
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": 0.0148915215031028,
          "ci95": [
            0.006600749161173039,
            0.023341259725197096
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": 0.009168327953584021,
          "ci95": [
            -0.0007358060325970999,
            0.020423703656806288
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": -0.005723193549518779,
          "ci95": [
            -0.01675043358803584,
            0.008381705122369467
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": -0.00041274245260104064,
          "ci95": [
            -0.0005282253571518584,
            -0.00029926688683937273
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": -0.0009256684792836624,
          "ci95": [
            -0.0010917182777215439,
            -0.0007706946378611847
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": -0.0004887868516224633,
          "ci95": [
            -0.0006346735884363555,
            -0.0003798090225507728
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": -0.0005129260266826217,
          "ci95": [
            -0.0006827432597746784,
            -0.000335593885602991
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": -7.604439902142261e-05,
          "ci95": [
            -0.00022319523049098535,
            6.215172371483995e-05
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": 0.00043688162766119913,
          "ci95": [
            0.00023338732096742714,
            0.0006216196975610466
          ]
        }
      }
    },
    "all_prespecified_effects_involving_candidate": {
      "distinct_no_carry__minus__mature_r2": {
        "coefficients": {
          "mature_r2": -1.0,
          "cross8_mature": 0.0,
          "shared_carry": 0.0,
          "distinct_carry": 0.0,
          "shared_no_carry": 0.0,
          "distinct_no_carry": 1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.010543637582882126,
            "ci95": [
              -0.01407057532398757,
              -0.007293229142440938
            ]
          },
          "auc.is_click": {
            "estimate": -0.0024529741470156186,
            "ci95": [
              -0.0032268243839037062,
              -0.001560778721852696
            ]
          },
          "auc.cart": {
            "estimate": -0.005220575575942021,
            "ci95": [
              -0.009627117060569948,
              -0.0005682010393333446
            ]
          },
          "auc.fav": {
            "estimate": -0.02011209707904482,
            "ci95": [
              -0.026770000091700892,
              -0.013305988068935884
            ]
          },
          "auc.buy": {
            "estimate": -0.014388903529526043,
            "ci95": [
              -0.025899706845661473,
              -0.005594036290646536
            ]
          },
          "bce.is_click": {
            "estimate": -0.00020276904493479742,
            "ci95": [
              -0.00028517884500233945,
              -0.00013502752303341665
            ]
          },
          "bce.cart": {
            "estimate": 0.00020997340766624323,
            "ci95": [
              0.00011109337144234603,
              0.000310226239397632
            ]
          },
          "bce.fav": {
            "estimate": 0.000722899434348865,
            "ci95": [
              0.0005704541059987241,
              0.0008730807232427345
            ]
          },
          "bce.buy": {
            "estimate": 0.00028601780668766584,
            "ci95": [
              0.000199676194606925,
              0.00040458709549040246
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.002767601428926403,
            "ci95": [
              -0.0019173953796686877,
              0.0075168969513803125
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.017659122932029203,
            "ci95": [
              0.010620477385969367,
              0.024450221291210522
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.011935929382510424,
            "ci95": [
              0.0032232012260614705,
              0.023699758013590918
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.0148915215031028,
            "ci95": [
              0.006600749161173039,
              0.023341259725197096
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.009168327953584021,
            "ci95": [
              -0.0007358060325970999,
              0.020423703656806288
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.005723193549518779,
            "ci95": [
              -0.01675043358803584,
              0.008381705122369467
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00041274245260104064,
            "ci95": [
              -0.0005282253571518584,
              -0.00029926688683937273
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0009256684792836624,
            "ci95": [
              -0.0010917182777215439,
              -0.0007706946378611847
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0004887868516224633,
            "ci95": [
              -0.0006346735884363555,
              -0.0003798090225507728
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.0005129260266826217,
            "ci95": [
              -0.0006827432597746784,
              -0.000335593885602991
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -7.604439902142261e-05,
            "ci95": [
              -0.00022319523049098535,
              6.215172371483995e-05
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.00043688162766119913,
            "ci95": [
              0.00023338732096742714,
              0.0006216196975610466
            ]
          }
        }
      },
      "distinct_no_carry__minus__cross8_mature": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": -1.0,
          "shared_carry": 0.0,
          "distinct_carry": 0.0,
          "shared_no_carry": 0.0,
          "distinct_no_carry": 1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.0028551063634814633,
            "ci95": [
              -0.005625445654037076,
              0.0003832982175522685
            ]
          },
          "auc.is_click": {
            "estimate": -0.002453451088991865,
            "ci95": [
              -0.0032275874396188453,
              -0.001621236318245251
            ]
          },
          "auc.cart": {
            "estimate": -0.001993040589714279,
            "ci95": [
              -0.006985288283847401,
              0.0026634697109721296
            ]
          },
          "auc.fav": {
            "estimate": -0.014150617731446324,
            "ci95": [
              -0.020604113157297172,
              -0.007776417199124745
            ]
          },
          "auc.buy": {
            "estimate": 0.007176683956226615,
            "ci95": [
              -0.0009010744019496526,
              0.014842788510329668
            ]
          },
          "bce.is_click": {
            "estimate": -0.0001290606257394744,
            "ci95": [
              -0.00020691938384408622,
              -5.170603455688997e-05
            ]
          },
          "bce.cart": {
            "estimate": 0.0002035161400212454,
            "ci95": [
              8.499752786020036e-05,
              0.00032703285161908514
            ]
          },
          "bce.fav": {
            "estimate": 0.0005476779472481513,
            "ci95": [
              0.0004376605316332689,
              0.0006578121693908777
            ]
          },
          "bce.buy": {
            "estimate": -7.75705850247603e-05,
            "ci95": [
              -0.00015492864801353875,
              1.549148319717705e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.00046041049927758593,
            "ci95": [
              -0.00523051938857873,
              0.004465949242805691
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.01169716664245446,
            "ci95": [
              0.005221124964883659,
              0.018412801501032156
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.00963013504521848,
            "ci95": [
              -0.017413054562580372,
              -0.0017783172381111432
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.012157577141732046,
            "ci95": [
              0.0032409295028319493,
              0.020086824785700997
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.009169724545940894,
            "ci95": [
              -0.01785059902529603,
              0.0003921610613299522
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.02132730168767294,
            "ci95": [
              -0.030904028524183426,
              -0.011399484721057474
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.0003325767657607198,
            "ci95": [
              -0.00048811827585720286,
              -0.0001953344855015323
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0006767385729876257,
            "ci95": [
              -0.0008071941741493954,
              -0.0005503149266830699
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -5.149004071471411e-05,
            "ci95": [
              -0.0001838317126249374,
              5.4631117395742554e-05
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.0003441618072269059,
            "ci95": [
              -0.0004953500412222713,
              -0.0001666251416804345
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.0002810867250460057,
            "ci95": [
              0.00014032017260171757,
              0.00042086876536982055
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.0006252485322729116,
            "ci95": [
              0.0004892700518903424,
              0.0007701871265133103
            ]
          }
        }
      },
      "distinct_carry__minus__distinct_no_carry": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "shared_carry": 0.0,
          "distinct_carry": 1.0,
          "shared_no_carry": 0.0,
          "distinct_no_carry": -1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.0035661181866422365,
            "ci95": [
              0.00034195770098013477,
              0.006398495271376823
            ]
          },
          "auc.is_click": {
            "estimate": 0.004372459629348957,
            "ci95": [
              0.003537092991350557,
              0.005113739225658397
            ]
          },
          "auc.cart": {
            "estimate": -0.000966664918710225,
            "ci95": [
              -0.005621579412087563,
              0.0036952384439763917
            ]
          },
          "auc.fav": {
            "estimate": 0.007608878398335461,
            "ci95": [
              -2.094998828992813e-05,
              0.013464273245248383
            ]
          },
          "auc.buy": {
            "estimate": 0.0032497996375947524,
            "ci95": [
              -0.004822870797389708,
              0.012346205459308235
            ]
          },
          "bce.is_click": {
            "estimate": -0.0003846817133170699,
            "ci95": [
              -0.000453958452557577,
              -0.0003080230536352875
            ]
          },
          "bce.cart": {
            "estimate": -2.5236365335783856e-05,
            "ci95": [
              -0.00013112824065938104,
              7.610249155972132e-05
            ]
          },
          "bce.fav": {
            "estimate": -0.0005039201100854943,
            "ci95": [
              -0.0006033261667914751,
              -0.0003771107232527732
            ]
          },
          "bce.buy": {
            "estimate": -0.00015258877396000063,
            "ci95": [
              -0.00024194588947537235,
              -6.906912748298139e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.005339124548059182,
            "ci95": [
              0.0004362758427549743,
              0.009982325252757916
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.0032364187689865043,
            "ci95": [
              -0.009014882560071625,
              0.004239310303026602
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.0011226599917542046,
            "ci95": [
              -0.00760394405818528,
              0.00917809215412246
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.008575543317045686,
            "ci95": [
              -0.017196205786438275,
              0.0005370175772643757
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.004216464556304977,
            "ci95": [
              -0.013683735022799778,
              0.004641904406152943
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.004359078760740709,
            "ci95": [
              -0.006478116017201948,
              0.013682457409487649
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00035944534798128605,
            "ci95": [
              -0.0004753264432389344,
              -0.00022979716611108578
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.00011923839676842435,
            "ci95": [
              -1.9913421094049262e-05,
              0.00024925223224023783
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.00023209293935706928,
            "ci95": [
              -0.00034106271637938994,
              -0.000121236060299655
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.0004786837447497104,
            "ci95": [
              0.00031022920669601374,
              0.0006423907586203428
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00012735240862421678,
            "ci95": [
              -4.991689986766657e-06,
              0.00025734715075950774
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.0003513313361254936,
            "ci95": [
              -0.0004697654626217248,
              -0.00018728820823342164
            ]
          }
        }
      },
      "distinct_no_carry__minus__shared_no_carry": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "shared_carry": 0.0,
          "distinct_carry": 0.0,
          "shared_no_carry": -1.0,
          "distinct_no_carry": 1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.007186754477775087,
            "ci95": [
              -0.010083559105976032,
              -0.004249344130655758
            ]
          },
          "auc.is_click": {
            "estimate": -0.0023960042016302197,
            "ci95": [
              -0.003165299258050361,
              -0.0015644436360017777
            ]
          },
          "auc.cart": {
            "estimate": -0.0059825199142453656,
            "ci95": [
              -0.010010312265218058,
              -0.001862213345982224
            ]
          },
          "auc.fav": {
            "estimate": -0.014905679660279114,
            "ci95": [
              -0.02215640482617924,
              -0.006679753252738359
            ]
          },
          "auc.buy": {
            "estimate": -0.005462814134945648,
            "ci95": [
              -0.012370543883630883,
              0.0006935980580361209
            ]
          },
          "bce.is_click": {
            "estimate": 4.86530201288804e-05,
            "ci95": [
              -2.7644084702555664e-05,
              0.00012237551686419332
            ]
          },
          "bce.cart": {
            "estimate": 4.633831039102185e-05,
            "ci95": [
              -5.511281776053676e-05,
              0.0001494135724063071
            ]
          },
          "bce.fav": {
            "estimate": 0.0004826130817920868,
            "ci95": [
              0.0003329215456236344,
              0.0006133039161779682
            ]
          },
          "bce.buy": {
            "estimate": 7.221448358865276e-05,
            "ci95": [
              -9.656017174221057e-06,
              0.00014692129754645178
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.003586515712615146,
            "ci95": [
              -0.0007966440429349435,
              0.007599799207503163
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.012509675458648895,
            "ci95": [
              0.004680512699795151,
              0.020036463824323576
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.0030668099333154286,
            "ci95": [
              -0.003154607149949973,
              0.010101892352504055
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.008923159746033749,
            "ci95": [
              -0.0005489662000010081,
              0.018149388353402095
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.0005197057792997173,
            "ci95": [
              -0.00851325885242643,
              0.007095269785597115
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.009442865525333466,
            "ci95": [
              -0.018742073170072628,
              0.0016307262045322725
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 2.31470973785855e-06,
            "ci95": [
              -0.0001275006086020291,
              0.00013293093894440223
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0004339600616632064,
            "ci95": [
              -0.0005820974914070014,
              -0.0002704946305654998
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -2.3561463459772364e-05,
            "ci95": [
              -0.0001274436088782588,
              9.193366516081619e-05
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.00043627477140106494,
            "ci95": [
              -0.000598116943445139,
              -0.0002685774298334735
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -2.5876173197630914e-05,
            "ci95": [
              -0.00015213274333012299,
              0.00010053494037292997
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.000410398598203434,
            "ci95": [
              0.0002466743098457573,
              0.0005657434269671629
            ]
          }
        }
      },
      "round_distinction_by_carry_interaction": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "shared_carry": -1.0,
          "distinct_carry": 1.0,
          "shared_no_carry": 1.0,
          "distinct_no_carry": -1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.006910164870632368,
            "ci95": [
              0.0022352313157072714,
              0.011162313956113803
            ]
          },
          "auc.is_click": {
            "estimate": 0.006444741265639764,
            "ci95": [
              0.005154744914931855,
              0.007596694513746946
            ]
          },
          "auc.cart": {
            "estimate": -0.0011860184627686676,
            "ci95": [
              -0.008176642809233026,
              0.005500630457601698
            ]
          },
          "auc.fav": {
            "estimate": 0.020417326854844875,
            "ci95": [
              0.008025400625413301,
              0.031888149773463076
            ]
          },
          "auc.buy": {
            "estimate": 0.0019646098248135013,
            "ci95": [
              -0.008677049764914463,
              0.012794482018612096
            ]
          },
          "bce.is_click": {
            "estimate": -0.0005324274127773421,
            "ci95": [
              -0.0006370337752105131,
              -0.00040978825354679593
            ]
          },
          "bce.cart": {
            "estimate": 0.0001989097962389573,
            "ci95": [
              5.359455861695031e-05,
              0.0003321151186739102
            ]
          },
          "bce.fav": {
            "estimate": -0.000651063001921956,
            "ci95": [
              -0.000841971056642634,
              -0.0004189358921363577
            ]
          },
          "bce.buy": {
            "estimate": -7.855632657234775e-05,
            "ci95": [
              -0.00018403598138465004,
              3.322503984070643e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.007630759728408432,
            "ci95": [
              0.000791261436428506,
              0.01432892253692511
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.013972585589205111,
            "ci95": [
              -0.025595530831840484,
              -0.0014082995552591945
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.004480131440826263,
            "ci95": [
              -0.006685101899936088,
              0.015405324239041222
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.021603345317613543,
            "ci95": [
              -0.03534470719036087,
              -0.007480968279894292
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.003150628287582169,
            "ci95": [
              -0.014806643479490346,
              0.00892429510617557
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.018452717030031374,
            "ci95": [
              0.001975279502433969,
              0.03472398069432884
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.0007313372090162994,
            "ci95": [
              -0.0009117680432500046,
              -0.0005438868036526819
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.00011863558914461392,
            "ci95": [
              -0.00012263562586974243,
              0.0003362809277361666
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0004538710862049943,
            "ci95": [
              -0.0006165163844854629,
              -0.0002955998573458529
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.0008499727981609133,
            "ci95": [
              0.0005821141022723191,
              0.0010809855570858287
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00027746612281130505,
            "ci95": [
              0.00010254690636004424,
              0.00045208388795231594
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.0005725066753496082,
            "ci95": [
              -0.0007981551168739626,
              -0.00033525745657378284
            ]
          }
        }
      }
    },
    "parameter_count": 51725028,
    "initializer_sha256": "c8d3ec7c5ee23537abfba615031f8f972b4a12e5e3412ac9cd3ed4b9c9df2d0f",
    "formal_worker_cost": {
      "wall_seconds": 2532.6367180794477,
      "cpu_seconds": 3033.1375620989998,
      "peak_rss_bytes": 17066680320,
      "gpu_peak_bytes": 28849471488,
      "samples": 2317
    },
    "smoke_worker_cost": {
      "wall_seconds": 218.20204113423824,
      "cpu_seconds": 411.807170161,
      "peak_rss_bytes": 20333928448,
      "gpu_peak_bytes": 30137122816,
      "samples": 199
    },
    "forward_latency": {
      "mean_ms": 14.342538380622864,
      "p50_ms": 14.283360004425049,
      "p95_ms": 14.3924925327301,
      "batch_size": 1024,
      "warmups": 10,
      "iterations": 40
    },
    "latency_ratio_to_parent": 1.0130705678137681,
    "quality_candidate": false,
    "near_candidate": false
  },
  "validity": {
    "valid": true,
    "state": "passed",
    "violations": [],
    "no_prediction_to_work_feedback": true,
    "test_evaluated": false,
    "quality_summary": "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/reports/quality-summary.json",
    "infrastructure_limitations": [
      "Four successful smoke runners had outer verifier errors; recovered saved checks and fixed formal worker accepted without replay. Original failed terminals preserved.",
      "One training seed; no training-seed uncertainty.",
      "Postprocess dispatcher failed before any child; str/Path adapter fixed with original numerical/runtime sources unchanged."
    ],
    "source_and_input_identity_scope": "Inherited original input content hashes plus frozen run stat continuity and original row-stream fingerprints; actual arrays revalidated in statistical execution."
  },
  "decision": "discard",
  "decision_reason": "相对原两轮macro AUC下降0.010544且区间在零以下，轮次更新脱离携带路径没有收益。",
  "decision_scope": "Research queue only; no incumbent promotion or new supported lesson.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/runs/distinct_no_carry",
    "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/RESULTS.md",
    "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/EVIDENCE.md",
    "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/reports/diagnostic-analysis/result.json"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/runs/distinct_no_carry",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/runs/distinct_no_carry/result.json",
    "result_sha256": "b8a3b7bd005738a339c90a4c084a0646d1d7ab5fb4a2f6450af03f298b4c10f3",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/runs/distinct_no_carry/manifest.json",
    "manifest_sha256": "6e032aa7bf1f3b1207383600c5b7883b1ea093b3fcd8a9915e58d03c807d6ced",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/runs/distinct_no_carry/Taobao_Action/distinct_no_carry.model",
    "checkpoint_sha256": "fa92ab994f89384525ed9765937e362c95fa9846a6c29d883a977a80f956a88e",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/reports/paired-analysis/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
