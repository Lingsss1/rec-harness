## 263. round-accumulation--shared_no_carry--s20262027

- 假设：检验轮次区分的残差更新与首轮状态携带的交互，固定两轮、优化器和seed。
- 判定：keep
- 原因：作为对照候选：较同协议Cross8 macro AUC提高0.004332且区间为正，但低于原两轮点估计，不晋升。

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "round-accumulation--shared_no_carry--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-10T15:44:40.685443+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "round_accumulation_factorial",
  "code_id": "2511cf353e09f9b83e8b18b3e701b1d57b7471c2ebaeb2986b93add858312e94",
  "hypothesis": "检验轮次区分的残差更新与首轮状态携带的交互，固定两轮、优化器和seed。",
  "change": "{\"model\": \"QFormerCross8RoundAccumulation\", \"maturity_mode\": \"mature\", \"work_steps\": 2, \"num_queries\": 4, \"accumulation_distinct\": false, \"carry_first\": false}",
  "config_change": {
    "model": "QFormerCross8RoundAccumulation",
    "maturity_mode": "mature",
    "work_steps": 2,
    "num_queries": 4,
    "accumulation_distinct": false,
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
      "value": 0.7637380682097534,
      "delta": -0.003356883105107039
    },
    "per_task": {
      "is_click": {
        "logloss": 0.1841402242027199,
        "AUC": 0.649879323255604
      },
      "cart": {
        "logloss": 0.02199158945170008,
        "AUC": 0.793196214695647
      },
      "fav": {
        "logloss": 0.017038443909657395,
        "AUC": 0.7972570729562823
      },
      "buy": {
        "logloss": 0.00985760525982242,
        "AUC": 0.8146196619314804
      }
    },
    "macro_source_logloss": 0.058256965705974946,
    "matched_parent_effect": {
      "coefficients": {
        "mature_r2": -1.0,
        "cross8_mature": 0.0,
        "shared_carry": 0.0,
        "distinct_carry": 0.0,
        "shared_no_carry": 1.0,
        "distinct_no_carry": 0.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": -0.003356883105107039,
          "ci95": [
            -0.0069280735074044725,
            0.00011475696920316965
          ]
        },
        "auc.is_click": {
          "estimate": -5.69699453853989e-05,
          "ci95": [
            -0.0007600126236936594,
            0.0007548909474123426
          ]
        },
        "auc.cart": {
          "estimate": 0.0007619443383033442,
          "ci95": [
            -0.003419172246336696,
            0.00579233875088207
          ]
        },
        "auc.fav": {
          "estimate": -0.0052064174187657075,
          "ci95": [
            -0.01185015285117757,
            0.0009907675189765029
          ]
        },
        "auc.buy": {
          "estimate": -0.008926089394580394,
          "ci95": [
            -0.020128111590652902,
            0.0005175247223021807
          ]
        },
        "bce.is_click": {
          "estimate": -0.0002514220650636778,
          "ci95": [
            -0.0003292032974827511,
            -0.00018292325215583594
          ]
        },
        "bce.cart": {
          "estimate": 0.00016363509727522138,
          "ci95": [
            5.6543802039350174e-05,
            0.0002632760000562408
          ]
        },
        "bce.fav": {
          "estimate": 0.0002402863525567782,
          "ci95": [
            0.00011879278694269715,
            0.00035065567171784704
          ]
        },
        "bce.buy": {
          "estimate": 0.00021380332309901308,
          "ci95": [
            0.00012133649146449741,
            0.0003314628976246686
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": -0.0008189142836887431,
          "ci95": [
            -0.005809329330085189,
            0.00344821062335783
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": 0.005149447473380309,
          "ci95": [
            -0.0011048759815032447,
            0.011727452254128698
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": 0.008869119449194995,
          "ci95": [
            -0.0010287709239182041,
            0.020052864687363003
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": 0.005968361757069052,
          "ci95": [
            -0.0019704852592495187,
            0.014556631825841954
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": 0.009688033732883738,
          "ci95": [
            -0.0005609536360472564,
            0.02090649972560873
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": 0.0037196719758146868,
          "ci95": [
            -0.007238341557659655,
            0.015521231626084386
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": -0.0004150571623388992,
          "ci95": [
            -0.0005325054277209231,
            -0.00028436208964666176
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": -0.000491708417620456,
          "ci95": [
            -0.0006283880838619271,
            -0.00036361173638639605
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": -0.0004652253881626909,
          "ci95": [
            -0.0005997236126092597,
            -0.000350550141861477
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": -7.665125528155681e-05,
          "ci95": [
            -0.0002430940099782914,
            7.486862964774098e-05
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": -5.01682258237917e-05,
          "ci95": [
            -0.0001893054404628153,
            7.528942144124599e-05
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": 2.648302945776511e-05,
          "ci95": [
            -0.0001432038922448646,
            0.00017493886511262504
          ]
        }
      }
    },
    "all_prespecified_effects_involving_candidate": {
      "shared_no_carry__minus__mature_r2": {
        "coefficients": {
          "mature_r2": -1.0,
          "cross8_mature": 0.0,
          "shared_carry": 0.0,
          "distinct_carry": 0.0,
          "shared_no_carry": 1.0,
          "distinct_no_carry": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.003356883105107039,
            "ci95": [
              -0.0069280735074044725,
              0.00011475696920316965
            ]
          },
          "auc.is_click": {
            "estimate": -5.69699453853989e-05,
            "ci95": [
              -0.0007600126236936594,
              0.0007548909474123426
            ]
          },
          "auc.cart": {
            "estimate": 0.0007619443383033442,
            "ci95": [
              -0.003419172246336696,
              0.00579233875088207
            ]
          },
          "auc.fav": {
            "estimate": -0.0052064174187657075,
            "ci95": [
              -0.01185015285117757,
              0.0009907675189765029
            ]
          },
          "auc.buy": {
            "estimate": -0.008926089394580394,
            "ci95": [
              -0.020128111590652902,
              0.0005175247223021807
            ]
          },
          "bce.is_click": {
            "estimate": -0.0002514220650636778,
            "ci95": [
              -0.0003292032974827511,
              -0.00018292325215583594
            ]
          },
          "bce.cart": {
            "estimate": 0.00016363509727522138,
            "ci95": [
              5.6543802039350174e-05,
              0.0002632760000562408
            ]
          },
          "bce.fav": {
            "estimate": 0.0002402863525567782,
            "ci95": [
              0.00011879278694269715,
              0.00035065567171784704
            ]
          },
          "bce.buy": {
            "estimate": 0.00021380332309901308,
            "ci95": [
              0.00012133649146449741,
              0.0003314628976246686
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.0008189142836887431,
            "ci95": [
              -0.005809329330085189,
              0.00344821062335783
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.005149447473380309,
            "ci95": [
              -0.0011048759815032447,
              0.011727452254128698
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.008869119449194995,
            "ci95": [
              -0.0010287709239182041,
              0.020052864687363003
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.005968361757069052,
            "ci95": [
              -0.0019704852592495187,
              0.014556631825841954
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.009688033732883738,
            "ci95": [
              -0.0005609536360472564,
              0.02090649972560873
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.0037196719758146868,
            "ci95": [
              -0.007238341557659655,
              0.015521231626084386
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.0004150571623388992,
            "ci95": [
              -0.0005325054277209231,
              -0.00028436208964666176
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.000491708417620456,
            "ci95": [
              -0.0006283880838619271,
              -0.00036361173638639605
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0004652253881626909,
            "ci95": [
              -0.0005997236126092597,
              -0.000350550141861477
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -7.665125528155681e-05,
            "ci95": [
              -0.0002430940099782914,
              7.486862964774098e-05
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -5.01682258237917e-05,
            "ci95": [
              -0.0001893054404628153,
              7.528942144124599e-05
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 2.648302945776511e-05,
            "ci95": [
              -0.0001432038922448646,
              0.00017493886511262504
            ]
          }
        }
      },
      "shared_no_carry__minus__cross8_mature": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": -1.0,
          "shared_carry": 0.0,
          "distinct_carry": 0.0,
          "shared_no_carry": 1.0,
          "distinct_no_carry": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.004331648114293624,
            "ci95": [
              0.0012944240248963298,
              0.007350999327078884
            ]
          },
          "auc.is_click": {
            "estimate": -5.744688736164516e-05,
            "ci95": [
              -0.00084507305787003,
              0.0007145258240047603
            ]
          },
          "auc.cart": {
            "estimate": 0.003989479324531087,
            "ci95": [
              -0.0005278989318238919,
              0.00812456273212933
            ]
          },
          "auc.fav": {
            "estimate": 0.0007550619288327898,
            "ci95": [
              -0.005129456191078593,
              0.006654007088671433
            ]
          },
          "auc.buy": {
            "estimate": 0.012639498091172263,
            "ci95": [
              0.0037895695533818026,
              0.021784114771997817
            ]
          },
          "bce.is_click": {
            "estimate": -0.0001777136458683548,
            "ci95": [
              -0.0002553126775863369,
              -0.00010009901991930513
            ]
          },
          "bce.cart": {
            "estimate": 0.00015717782963022356,
            "ci95": [
              5.526443140182937e-05,
              0.00026251046273652675
            ]
          },
          "bce.fav": {
            "estimate": 6.506486545606455e-05,
            "ci95": [
              -5.544750120454146e-05,
              0.0001838766737007377
            ]
          },
          "bce.buy": {
            "estimate": -0.00014978506861341306,
            "ci95": [
              -0.0002511442701576274,
              -4.9193071812962344e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.004046926211892732,
            "ci95": [
              -0.008374856185832726,
              0.0003861202491830323
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.0008125088161944349,
            "ci95": [
              -0.007086691582171278,
              0.005284924354884663
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.012696944978533908,
            "ci95": [
              -0.02215726465696097,
              -0.003675482417900589
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.003234417395698297,
            "ci95": [
              -0.0038884929445733712,
              0.010671611758600876
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.008650018766641177,
            "ci95": [
              -0.01813104943914552,
              0.0007332711695032743
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.011884436162339473,
            "ci95": [
              -0.023427675804050817,
              -0.0016238583247383366
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00033489147549857837,
            "ci95": [
              -0.0004571689254193203,
              -0.00019418086840374907
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.00024277851132441935,
            "ci95": [
              -0.00038628850987802564,
              -0.000113271263114554
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -2.7928577254941744e-05,
            "ci95": [
              -0.0001502481049022705,
              0.00010044279373977945
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 9.211296417415901e-05,
            "ci95": [
              -7.34819109372866e-05,
              0.00023992574730219394
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.0003069628982436366,
            "ci95": [
              0.00018135238960298715,
              0.00044122623111112993
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.0002148499340694776,
            "ci95": [
              4.981915660347616e-05,
              0.00037167880038532346
            ]
          }
        }
      },
      "shared_carry__minus__shared_no_carry": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "shared_carry": 1.0,
          "distinct_carry": 0.0,
          "shared_no_carry": -1.0,
          "distinct_no_carry": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.0033440466839901317,
            "ci95": [
              -0.006376487384915059,
              0.00015664353969591156
            ]
          },
          "auc.is_click": {
            "estimate": -0.002072281636290807,
            "ci95": [
              -0.002901096968433389,
              -0.0011639239559934013
            ]
          },
          "auc.cart": {
            "estimate": 0.00021935354405844265,
            "ci95": [
              -0.004305264194594546,
              0.004890747657216152
            ]
          },
          "auc.fav": {
            "estimate": -0.012808448456509414,
            "ci95": [
              -0.021733348971723385,
              -0.004491329405541582
            ]
          },
          "auc.buy": {
            "estimate": 0.001285189812781251,
            "ci95": [
              -0.00495456079413055,
              0.008615104682170898
            ]
          },
          "bce.is_click": {
            "estimate": 0.00014774569946027216,
            "ci95": [
              6.0164530574972895e-05,
              0.00022652019982420736
            ]
          },
          "bce.cart": {
            "estimate": -0.00022414616157474115,
            "ci95": [
              -0.00032763821359605637,
              -0.00011971785678424513
            ]
          },
          "bce.fav": {
            "estimate": 0.00014714289183646173,
            "ci95": [
              -7.227064764650601e-06,
              0.00031581028144960985
            ]
          },
          "bce.buy": {
            "estimate": -7.403244738765288e-05,
            "ci95": [
              -0.00015280062481891243,
              -2.205276588299728e-06
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.0022916351803492496,
            "ci95": [
              -0.007126217124866013,
              0.0022561261163572366
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.010736166820218607,
            "ci95": [
              0.002084561749798164,
              0.019686211238046394
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.003357471449072058,
            "ci95": [
              -0.010795503200718603,
              0.0030647043502525463
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.013027802000567856,
            "ci95": [
              0.004362509946404797,
              0.022625827576296244
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.0010658362687228085,
            "ci95": [
              -0.009579827260529328,
              0.006350080770627593
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.014093638269290665,
            "ci95": [
              -0.026201450372216238,
              -0.004426104412547037
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 0.0003718918610350133,
            "ci95": [
              0.00023479592273935974,
              0.00050731260700262
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 6.028076238104307e-07,
            "ci95": [
              -0.00017866752060327444,
              0.00017367404141172557
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.00022177814684792505,
            "ci95": [
              0.00011240389502498757,
              0.00033905931978755495
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.0003712890534112029,
            "ci95": [
              -0.0005545705387859595,
              -0.0002081441762627038
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -0.00015011371418708827,
            "ci95": [
              -0.0002751484681312537,
              -2.1687400934500214e-05
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.00022117533922411461,
            "ci95": [
              7.008263278629225e-05,
              0.0004020452896097417
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
    "parameter_count": 51724644,
    "initializer_sha256": "4f81bf632c885decdc491249310fd82cb27eacf53dd17bb959577399dfcf1c87",
    "formal_worker_cost": {
      "wall_seconds": 2536.380836136639,
      "cpu_seconds": 3032.135403029,
      "peak_rss_bytes": 17131966464,
      "gpu_peak_bytes": 28845277184,
      "samples": 2319
    },
    "smoke_worker_cost": {
      "wall_seconds": 213.45612256228924,
      "cpu_seconds": 370.121589433,
      "peak_rss_bytes": 20166889472,
      "gpu_peak_bytes": 30135025664,
      "samples": 195
    },
    "forward_latency": {
      "mean_ms": 14.319995164871216,
      "p50_ms": 14.303823947906494,
      "p95_ms": 14.550932455062867,
      "batch_size": 1024,
      "warmups": 10,
      "iterations": 40
    },
    "latency_ratio_to_parent": 1.0114782507652933,
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
  "decision_reason": "作为对照候选：较同协议Cross8 macro AUC提高0.004332且区间为正，但低于原两轮点估计，不晋升。",
  "decision_scope": "Research queue only; no incumbent promotion or new supported lesson.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/runs/shared_no_carry",
    "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/RESULTS.md",
    "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/EVIDENCE.md",
    "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/reports/diagnostic-analysis/result.json"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/runs/shared_no_carry",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/runs/shared_no_carry/result.json",
    "result_sha256": "a3e4dab8a8185aed56388e131dc651b0db33eaf6215c83b45c31b83ce2c45542",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/runs/shared_no_carry/manifest.json",
    "manifest_sha256": "4c0c29bc014a537687320d2df88a4a7bc24deca13eea5706a632060f41705815",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/runs/shared_no_carry/Taobao_Action/shared_no_carry.model",
    "checkpoint_sha256": "52d91d8be6a59823c6566caf1be9eb3fdae403c270a2c887ba327fee2b3cbb3c",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/reports/paired-analysis/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
