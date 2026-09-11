## 262. round-accumulation--distinct_carry--s20262027

- 假设：检验轮次区分的残差更新与首轮状态携带的交互，固定两轮、优化器和seed。
- 判定：keep
- 原因：仅作为点击/状态路由研究候选：条件携带收益和交互为正，点击改善；整体显著低于原两轮，不晋升。

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "round-accumulation--distinct_carry--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-10T15:44:40.685443+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "round_accumulation_factorial",
  "code_id": "2511cf353e09f9b83e8b18b3e701b1d57b7471c2ebaeb2986b93add858312e94",
  "hypothesis": "检验轮次区分的残差更新与首轮状态携带的交互，固定两轮、优化器和seed。",
  "change": "{\"model\": \"QFormerCross8RoundAccumulation\", \"maturity_mode\": \"mature\", \"work_steps\": 2, \"num_queries\": 4, \"accumulation_distinct\": true, \"carry_first\": true}",
  "config_change": {
    "model": "QFormerCross8RoundAccumulation",
    "maturity_mode": "mature",
    "work_steps": 2,
    "num_queries": 4,
    "accumulation_distinct": true,
    "carry_first": true
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
      "value": 0.7601174319186207,
      "delta": -0.00697751939623989
    },
    "per_task": {
      "is_click": {
        "logloss": 0.18380419550953314,
        "AUC": 0.6518557786833228
      },
      "cart": {
        "logloss": 0.022012691361966853,
        "AUC": 0.7862470298626915
      },
      "fav": {
        "logloss": 0.017017136961121847,
        "AUC": 0.7899602716943387
      },
      "buy": {
        "logloss": 0.009777230956151219,
        "AUC": 0.8124066474341295
      }
    },
    "macro_source_logloss": 0.05815281369719326,
    "matched_parent_effect": {
      "coefficients": {
        "mature_r2": -1.0,
        "cross8_mature": 0.0,
        "shared_carry": 0.0,
        "distinct_carry": 1.0,
        "shared_no_carry": 0.0,
        "distinct_no_carry": 0.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": -0.00697751939623989,
          "ci95": [
            -0.00998184893683949,
            -0.00430688024340647
          ]
        },
        "auc.is_click": {
          "estimate": 0.0019194854823333385,
          "ci95": [
            0.0012706854504446418,
            0.0026207203413582147
          ]
        },
        "auc.cart": {
          "estimate": -0.006187240494652246,
          "ci95": [
            -0.010201656474414239,
            -0.0017764890134049252
          ]
        },
        "auc.fav": {
          "estimate": -0.01250321868070936,
          "ci95": [
            -0.020528665533766508,
            -0.0059721817737200394
          ]
        },
        "auc.buy": {
          "estimate": -0.01113910389193129,
          "ci95": [
            -0.020989872774135238,
            -0.003562584798874078
          ]
        },
        "bce.is_click": {
          "estimate": -0.0005874507582518673,
          "ci95": [
            -0.0006571430444417437,
            -0.0005273307386174618
          ]
        },
        "bce.cart": {
          "estimate": 0.00018473704233045937,
          "ci95": [
            9.297722733483862e-05,
            0.00027613000619812825
          ]
        },
        "bce.fav": {
          "estimate": 0.00021897932426337072,
          "ci95": [
            8.60463300577109e-05,
            0.0003785366860628937
          ]
        },
        "bce.buy": {
          "estimate": 0.0001334290327276652,
          "ci95": [
            5.875421776122434e-05,
            0.00022498790279677254
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": 0.008106725976985585,
          "ci95": [
            0.0036769345998549753,
            0.012307517713557068
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": 0.014422704163042699,
          "ci95": [
            0.008013485969849249,
            0.022308440703947795
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": 0.013058589374264629,
          "ci95": [
            0.0052767843656237635,
            0.022898919370182184
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": 0.006315978186057114,
          "ci95": [
            -0.0014204519363899381,
            0.015547946834198636
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": 0.004951863397279044,
          "ci95": [
            -0.003412358038465604,
            0.014751344721212574
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": -0.0013641147887780702,
          "ci95": [
            -0.012823658296192952,
            0.011198264532337621
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": -0.0007721878005823267,
          "ci95": [
            -0.0008803897401806645,
            -0.0006596433364972351
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": -0.000806430082515238,
          "ci95": [
            -0.0009932827039134302,
            -0.0006536819550030813
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": -0.0007208797909795325,
          "ci95": [
            -0.0008351530876383393,
            -0.0006266123750143895
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": -3.4242281932911345e-05,
          "ci95": [
            -0.00021487340156316776,
            0.00012424554972552347
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": 5.1308009602794163e-05,
          "ci95": [
            -7.381586455234447e-05,
            0.00016620373004610863
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": 8.555029153570551e-05,
          "ci95": [
            -8.845896721324645e-05,
            0.0002679915952776009
          ]
        }
      }
    },
    "all_prespecified_effects_involving_candidate": {
      "distinct_carry__minus__mature_r2": {
        "coefficients": {
          "mature_r2": -1.0,
          "cross8_mature": 0.0,
          "shared_carry": 0.0,
          "distinct_carry": 1.0,
          "shared_no_carry": 0.0,
          "distinct_no_carry": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.00697751939623989,
            "ci95": [
              -0.00998184893683949,
              -0.00430688024340647
            ]
          },
          "auc.is_click": {
            "estimate": 0.0019194854823333385,
            "ci95": [
              0.0012706854504446418,
              0.0026207203413582147
            ]
          },
          "auc.cart": {
            "estimate": -0.006187240494652246,
            "ci95": [
              -0.010201656474414239,
              -0.0017764890134049252
            ]
          },
          "auc.fav": {
            "estimate": -0.01250321868070936,
            "ci95": [
              -0.020528665533766508,
              -0.0059721817737200394
            ]
          },
          "auc.buy": {
            "estimate": -0.01113910389193129,
            "ci95": [
              -0.020989872774135238,
              -0.003562584798874078
            ]
          },
          "bce.is_click": {
            "estimate": -0.0005874507582518673,
            "ci95": [
              -0.0006571430444417437,
              -0.0005273307386174618
            ]
          },
          "bce.cart": {
            "estimate": 0.00018473704233045937,
            "ci95": [
              9.297722733483862e-05,
              0.00027613000619812825
            ]
          },
          "bce.fav": {
            "estimate": 0.00021897932426337072,
            "ci95": [
              8.60463300577109e-05,
              0.0003785366860628937
            ]
          },
          "bce.buy": {
            "estimate": 0.0001334290327276652,
            "ci95": [
              5.875421776122434e-05,
              0.00022498790279677254
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.008106725976985585,
            "ci95": [
              0.0036769345998549753,
              0.012307517713557068
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.014422704163042699,
            "ci95": [
              0.008013485969849249,
              0.022308440703947795
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.013058589374264629,
            "ci95": [
              0.0052767843656237635,
              0.022898919370182184
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.006315978186057114,
            "ci95": [
              -0.0014204519363899381,
              0.015547946834198636
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.004951863397279044,
            "ci95": [
              -0.003412358038465604,
              0.014751344721212574
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.0013641147887780702,
            "ci95": [
              -0.012823658296192952,
              0.011198264532337621
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.0007721878005823267,
            "ci95": [
              -0.0008803897401806645,
              -0.0006596433364972351
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.000806430082515238,
            "ci95": [
              -0.0009932827039134302,
              -0.0006536819550030813
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0007208797909795325,
            "ci95": [
              -0.0008351530876383393,
              -0.0006266123750143895
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -3.4242281932911345e-05,
            "ci95": [
              -0.00021487340156316776,
              0.00012424554972552347
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 5.1308009602794163e-05,
            "ci95": [
              -7.381586455234447e-05,
              0.00016620373004610863
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 8.555029153570551e-05,
            "ci95": [
              -8.845896721324645e-05,
              0.0002679915952776009
            ]
          }
        }
      },
      "distinct_carry__minus__cross8_mature": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": -1.0,
          "shared_carry": 0.0,
          "distinct_carry": 1.0,
          "shared_no_carry": 0.0,
          "distinct_no_carry": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.0007110118231607732,
            "ci95": [
              -0.0023657879639048473,
              0.003823749682934647
            ]
          },
          "auc.is_click": {
            "estimate": 0.0019190085403570922,
            "ci95": [
              0.0010719561521876865,
              0.0026100352523199728
            ]
          },
          "auc.cart": {
            "estimate": -0.002959705508424504,
            "ci95": [
              -0.007126273089276306,
              0.001616991670352774
            ]
          },
          "auc.fav": {
            "estimate": -0.006541739333110863,
            "ci95": [
              -0.01277446596940438,
              -0.0005728979311379149
            ]
          },
          "auc.buy": {
            "estimate": 0.010426483593821367,
            "ci95": [
              0.0018227237153449955,
              0.019686324339660513
            ]
          },
          "bce.is_click": {
            "estimate": -0.0005137423390565443,
            "ci95": [
              -0.0005813587234332146,
              -0.0004389728929138914
            ]
          },
          "bce.cart": {
            "estimate": 0.00017827977468546155,
            "ci95": [
              6.814068551649977e-05,
              0.00028768535735011454
            ]
          },
          "bce.fav": {
            "estimate": 4.3757837162657076e-05,
            "ci95": [
              -7.504466522329483e-05,
              0.00016550659857194732
            ]
          },
          "bce.buy": {
            "estimate": -0.00023015935898476093,
            "ci95": [
              -0.00032297905505768404,
              -0.00013697677047849394
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.004878714048781596,
            "ci95": [
              0.00029710669487840476,
              0.00906406539341402
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.008460747873467955,
            "ci95": [
              0.002234872838541663,
              0.0148905254242083
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.008507475053464275,
            "ci95": [
              -0.017845830463454428,
              0.00034581417480642525
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.0035820338246863592,
            "ci95": [
              -0.0038315579032968112,
              0.010800456994118305
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.013386189102245871,
            "ci95": [
              -0.02300367720867944,
              -0.0036275110236745384
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.01696822292693223,
            "ci95": [
              -0.028234658399747606,
              -0.005981309138379148
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.0006920221137420059,
            "ci95": [
              -0.0008170267206872882,
              -0.0005631368346168251
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0005575001762192014,
            "ci95": [
              -0.0007052267292727316,
              -0.00042289489636973646
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0002835829800717834,
            "ci95": [
              -0.00039761497882389126,
              -0.000150018874199412
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.00013452193752280447,
            "ci95": [
              -1.0538040755986037e-05,
              0.0003074597752676526
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.0004084391336702225,
            "ci95": [
              0.0002796221722401617,
              0.000547163144860488
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.000273917196147418,
            "ci95": [
              0.00012919155534884953,
              0.0004256316409351557
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
      "distinct_carry__minus__shared_carry": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "shared_carry": -1.0,
          "distinct_carry": 1.0,
          "shared_no_carry": 0.0,
          "distinct_no_carry": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.00027658960714271874,
            "ci95": [
              -0.0036587174143853053,
              0.002922870940870184
            ]
          },
          "auc.is_click": {
            "estimate": 0.004048737064009544,
            "ci95": [
              0.0032030964111110062,
              0.004820854242960365
            ]
          },
          "auc.cart": {
            "estimate": -0.007168538377014033,
            "ci95": [
              -0.0125136648327246,
              -0.0011832114869971962
            ]
          },
          "auc.fav": {
            "estimate": 0.005511647194565761,
            "ci95": [
              -0.0017768662329467637,
              0.01206243185400375
            ]
          },
          "auc.buy": {
            "estimate": -0.003498204310132147,
            "ci95": [
              -0.011611980229133687,
              0.00472865845148192
            ]
          },
          "bce.is_click": {
            "estimate": -0.00048377439264846167,
            "ci95": [
              -0.0005579723537022102,
              -0.00040798034901303604
            ]
          },
          "bce.cart": {
            "estimate": 0.00024524810662997915,
            "ci95": [
              0.00013388655041395676,
              0.00035509123630977115
            ]
          },
          "bce.fav": {
            "estimate": -0.0001684499201298692,
            "ci95": [
              -0.00026930606894545695,
              -3.6590339313623043e-05
            ]
          },
          "bce.buy": {
            "estimate": -6.341842983694987e-06,
            "ci95": [
              -8.287031589837077e-05,
              8.656582924966757e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.011217275441023578,
            "ci95": [
              0.005268199188632463,
              0.016678136250069014
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.0014629101305562164,
            "ci95": [
              -0.007755418385008827,
              0.005708569465124399
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.007546941374141691,
            "ci95": [
              -0.001481260483024388,
              0.015813185781261622
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.012680185571579794,
            "ci95": [
              -0.022013792782011033,
              -0.0030988248337413516
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.0036703340668818862,
            "ci95": [
              -0.01160078705826605,
              0.005087733390641487
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.009009851504697908,
            "ci95": [
              -0.0022304047477526593,
              0.020235792132454144
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.0007290224992784408,
            "ci95": [
              -0.0008717345334294164,
              -0.0005958566844204607
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.00031532447251859247,
            "ci95": [
              -0.0004587189797000202,
              -0.0001878446546086529
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0004774325496647667,
            "ci95": [
              -0.000598105396464037,
              -0.0003682730263395025
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.00041369802675984835,
            "ci95": [
              0.0002442574988004851,
              0.0005546237320333615
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00025158994961367413,
            "ci95": [
              0.00013114875674953208,
              0.0003679610620303196
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.00016210807714617421,
            "ci95": [
              -0.0003107008030427006,
              -1.6025225751227682e-05
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
      "wall_seconds": 2550.3793687671423,
      "cpu_seconds": 3056.10163182,
      "peak_rss_bytes": 17034981376,
      "gpu_peak_bytes": 27591180288,
      "samples": 2333
    },
    "smoke_worker_cost": {
      "wall_seconds": 215.90129647403955,
      "cpu_seconds": 404.568051318,
      "peak_rss_bytes": 20337840128,
      "gpu_peak_bytes": 30137122816,
      "samples": 197
    },
    "forward_latency": {
      "mean_ms": 14.382460021972657,
      "p50_ms": 14.321392059326172,
      "p95_ms": 14.534981060028075,
      "batch_size": 1024,
      "warmups": 10,
      "iterations": 40
    },
    "latency_ratio_to_parent": 1.0158903922267837,
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
  "decision": "keep",
  "decision_reason": "仅作为点击/状态路由研究候选：条件携带收益和交互为正，点击改善；整体显著低于原两轮，不晋升。",
  "decision_scope": "Research queue only; no incumbent promotion or new supported lesson.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/runs/distinct_carry",
    "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/RESULTS.md",
    "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/EVIDENCE.md",
    "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/reports/diagnostic-analysis/result.json"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/runs/distinct_carry",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/runs/distinct_carry/result.json",
    "result_sha256": "1e57a26fc0a7f1c129510511789a7e82bb2cbc1be2eac2b04e02b1fe922a8137",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/runs/distinct_carry/manifest.json",
    "manifest_sha256": "5fb83fd1498413625182e318b8c2ee0ae28c1213b46d0a671cdefe977151d763",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/runs/distinct_carry/Taobao_Action/distinct_carry.model",
    "checkpoint_sha256": "981ec077b538b6c9c01a03a329a849e3ef8f51d071adf2f936c5d5d2e5c51473",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/reports/paired-analysis/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
