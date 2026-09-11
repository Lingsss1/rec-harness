## 259. endpoint-read--odd_last--s20262027

- 假设：Separate first-work stored-endpoint retention from interleaving phase at fixed support count, two work rounds and no prediction feedback.
- 判定：discard
- 原因：Macro delta -0.004784711,CI95 [-0.007743902623144495, -0.00213613676149459];latency ratio1.005589. Below matched original recurrence; preserve negative outcome.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "endpoint-read--odd_last--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-10T15:23:24.099568+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "odd_last",
  "code_id": "b16543b20d537e062ccfd9a027ec0a882a6c396851eddf999162c82a0a9c9bc7",
  "hypothesis": "Separate first-work stored-endpoint retention from interleaving phase at fixed support count, two work rounds and no prediction feedback.",
  "change": "{\"model\": \"QFormerCross8EndpointRead\", \"maturity_mode\": \"mature\", \"segment_placement\": \"first\", \"segment_selection\": \"odd_last\", \"work_steps\": 2, \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8EndpointRead",
    "maturity_mode": "mature",
    "segment_placement": "first",
    "segment_selection": "odd_last",
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
      "value": 0.7623102404647235,
      "delta": -0.004784710850137097
    },
    "per_task": {
      "is_click": {
        "logloss": 0.1842016480786893,
        "AUC": 0.6476721287005347
      },
      "cart": {
        "logloss": 0.02196225123846316,
        "AUC": 0.7851048627844109
      },
      "fav": {
        "logloss": 0.016992074185814575,
        "AUC": 0.7984110466724069
      },
      "buy": {
        "logloss": 0.009764020926814356,
        "AUC": 0.818052923701541
      }
    },
    "macro_source_logloss": 0.05822999860744535,
    "source_metric_comparison": {
      "is_click": {
        "source_AUC": 0.6476721287005347,
        "computed_AUC": 0.6476721287005347,
        "auc_difference": 0.0,
        "source_logloss": 0.1842016480786893,
        "computed_bce": 0.18420164807869016,
        "bce_difference": 8.604228440844963e-16
      },
      "cart": {
        "source_AUC": 0.7851048627844109,
        "computed_AUC": 0.7851048627844109,
        "auc_difference": 0.0,
        "source_logloss": 0.02196225123846316,
        "computed_bce": 0.021962251238848034,
        "bce_difference": 3.8487268927411833e-13
      },
      "fav": {
        "source_AUC": 0.7984110466724069,
        "computed_AUC": 0.7984110466724069,
        "auc_difference": 0.0,
        "source_logloss": 0.016992074185814575,
        "computed_bce": 0.016992074186899263,
        "bce_difference": 1.084687895058778e-12
      },
      "buy": {
        "source_AUC": 0.818052923701541,
        "computed_AUC": 0.818052923701541,
        "auc_difference": 0.0,
        "source_logloss": 0.009764020926814356,
        "computed_bce": 0.009764020928053134,
        "bce_difference": 1.2387781772593698e-12
      }
    },
    "matched_parent_effect": {
      "coefficients": {
        "mature_r2": -1.0,
        "cross8_mature": 0.0,
        "first_even": 0.0,
        "first_odd": 0.0,
        "even_last": 0.0,
        "odd_last": 1.0,
        "odd_no_last": 0.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": -0.004784710850137097,
          "ci95": [
            -0.007743902623144495,
            -0.00213613676149459
          ]
        },
        "auc.is_click": {
          "estimate": -0.002264164500454746,
          "ci95": [
            -0.0030171066495972483,
            -0.0014737632863397272
          ]
        },
        "auc.cart": {
          "estimate": -0.007329407572932767,
          "ci95": [
            -0.011991476040016703,
            -0.0025584436643536624
          ]
        },
        "auc.fav": {
          "estimate": -0.004052443702641129,
          "ci95": [
            -0.009664284942519025,
            0.001376705599930468
          ]
        },
        "auc.buy": {
          "estimate": -0.005492827624519747,
          "ci95": [
            -0.013615979996231115,
            0.0023330259598308718
          ]
        },
        "bce.is_click": {
          "estimate": -0.000189998189096513,
          "ci95": [
            -0.0002677734187685278,
            -0.00011875611798954286
          ]
        },
        "bce.cart": {
          "estimate": 0.00013429686941400162,
          "ci95": [
            3.183209576204625e-05,
            0.00023190895462877806
          ]
        },
        "bce.fav": {
          "estimate": 0.00019391649254692916,
          "ci95": [
            9.033667642315444e-05,
            0.00030200105928368786
          ]
        },
        "bce.buy": {
          "estimate": 0.00012021899132743166,
          "ci95": [
            3.1172829776338066e-05,
            0.00022182620213904847
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": 0.005065243072478021,
          "ci95": [
            0.0002686963401646093,
            0.009774334463720589
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": 0.0017882792021863825,
          "ci95": [
            -0.0037281551335958613,
            0.007427233810155665
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": 0.003228663124065001,
          "ci95": [
            -0.004993563818240235,
            0.01146135082730251
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": -0.003276963870291638,
          "ci95": [
            -0.010474898283970163,
            0.0040052521079998755
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": -0.0018365799484130196,
          "ci95": [
            -0.011289695639414562,
            0.007320242171931528
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": 0.0014403839218786185,
          "ci95": [
            -0.008960927712068415,
            0.012334732867911261
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": -0.00032429505851051463,
          "ci95": [
            -0.00045321865472852717,
            -0.00019397874924410882
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": -0.00038391468164344217,
          "ci95": [
            -0.0005134573925961281,
            -0.0002680073257604147
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": -0.00031021718042394467,
          "ci95": [
            -0.00043736468466243914,
            -0.00020089822076077297
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": -5.961962313292754e-05,
          "ci95": [
            -0.00020835563325873467,
            8.691336768064617e-05
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": 1.4077878086569962e-05,
          "ci95": [
            -0.00012997135417541317,
            0.00013781589794625629
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": 7.36975012194975e-05,
          "ci95": [
            -8.283371068935323e-05,
            0.00021603322241040853
          ]
        }
      }
    },
    "all_prespecified_effects_involving_candidate": {
      "odd_last__minus__mature_r2": {
        "coefficients": {
          "mature_r2": -1.0,
          "cross8_mature": 0.0,
          "first_even": 0.0,
          "first_odd": 0.0,
          "even_last": 0.0,
          "odd_last": 1.0,
          "odd_no_last": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.004784710850137097,
            "ci95": [
              -0.007743902623144495,
              -0.00213613676149459
            ]
          },
          "auc.is_click": {
            "estimate": -0.002264164500454746,
            "ci95": [
              -0.0030171066495972483,
              -0.0014737632863397272
            ]
          },
          "auc.cart": {
            "estimate": -0.007329407572932767,
            "ci95": [
              -0.011991476040016703,
              -0.0025584436643536624
            ]
          },
          "auc.fav": {
            "estimate": -0.004052443702641129,
            "ci95": [
              -0.009664284942519025,
              0.001376705599930468
            ]
          },
          "auc.buy": {
            "estimate": -0.005492827624519747,
            "ci95": [
              -0.013615979996231115,
              0.0023330259598308718
            ]
          },
          "bce.is_click": {
            "estimate": -0.000189998189096513,
            "ci95": [
              -0.0002677734187685278,
              -0.00011875611798954286
            ]
          },
          "bce.cart": {
            "estimate": 0.00013429686941400162,
            "ci95": [
              3.183209576204625e-05,
              0.00023190895462877806
            ]
          },
          "bce.fav": {
            "estimate": 0.00019391649254692916,
            "ci95": [
              9.033667642315444e-05,
              0.00030200105928368786
            ]
          },
          "bce.buy": {
            "estimate": 0.00012021899132743166,
            "ci95": [
              3.1172829776338066e-05,
              0.00022182620213904847
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.005065243072478021,
            "ci95": [
              0.0002686963401646093,
              0.009774334463720589
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.0017882792021863825,
            "ci95": [
              -0.0037281551335958613,
              0.007427233810155665
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.003228663124065001,
            "ci95": [
              -0.004993563818240235,
              0.01146135082730251
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.003276963870291638,
            "ci95": [
              -0.010474898283970163,
              0.0040052521079998755
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.0018365799484130196,
            "ci95": [
              -0.011289695639414562,
              0.007320242171931528
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.0014403839218786185,
            "ci95": [
              -0.008960927712068415,
              0.012334732867911261
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00032429505851051463,
            "ci95": [
              -0.00045321865472852717,
              -0.00019397874924410882
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.00038391468164344217,
            "ci95": [
              -0.0005134573925961281,
              -0.0002680073257604147
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.00031021718042394467,
            "ci95": [
              -0.00043736468466243914,
              -0.00020089822076077297
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -5.961962313292754e-05,
            "ci95": [
              -0.00020835563325873467,
              8.691336768064617e-05
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 1.4077878086569962e-05,
            "ci95": [
              -0.00012997135417541317,
              0.00013781589794625629
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 7.36975012194975e-05,
            "ci95": [
              -8.283371068935323e-05,
              0.00021603322241040853
            ]
          }
        }
      },
      "odd_last__minus__cross8_mature": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": -1.0,
          "first_even": 0.0,
          "first_odd": 0.0,
          "even_last": 0.0,
          "odd_last": 1.0,
          "odd_no_last": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.0029038203692635656,
            "ci95": [
              3.991417333172186e-05,
              0.005648346734759868
            ]
          },
          "auc.is_click": {
            "estimate": -0.0022646414424309924,
            "ci95": [
              -0.0030575679324747777,
              -0.0015802338842444647
            ]
          },
          "auc.cart": {
            "estimate": -0.004101872586705024,
            "ci95": [
              -0.00894590618966223,
              0.0006144821346604761
            ]
          },
          "auc.fav": {
            "estimate": 0.0019090356449573687,
            "ci95": [
              -0.0038905445514752193,
              0.0074793144700898645
            ]
          },
          "auc.buy": {
            "estimate": 0.01607275986123291,
            "ci95": [
              0.008579456794095417,
              0.02498141404816162
            ]
          },
          "bce.is_click": {
            "estimate": -0.00011628976990119,
            "ci95": [
              -0.00018136325772092344,
              -4.639661488721886e-05
            ]
          },
          "bce.cart": {
            "estimate": 0.0001278396017690038,
            "ci95": [
              2.460013313861148e-05,
              0.00024440788026211584
            ]
          },
          "bce.fav": {
            "estimate": 1.8695005446215524e-05,
            "ci95": [
              -9.733567453107419e-05,
              0.00013848027840483276
            ]
          },
          "bce.buy": {
            "estimate": -0.00024336940038499448,
            "ci95": [
              -0.00033021342410174585,
              -0.00015467330909126344
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.0018372311442740319,
            "ci95": [
              -0.0029994015536184415,
              0.006668204498082285
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.004173677087388361,
            "ci95": [
              -0.010071493254271712,
              0.0015320040710068716
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.018337401303663903,
            "ci95": [
              -0.0274439881237083,
              -0.01060608462943486
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.006010908231662393,
            "ci95": [
              -0.013369174640790548,
              0.001376116027623531
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.020174632447937935,
            "ci95": [
              -0.029545456760271795,
              -0.01205470037755232
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.014163724216275542,
            "ci95": [
              -0.023960168040928303,
              -0.005540906361518705
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.0002441293716701938,
            "ci95": [
              -0.0003746998077367476,
              -0.00012501371282320212
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.00013498477534740552,
            "ci95": [
              -0.0002771824041785661,
              -4.3687192100968445e-07
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.00012707963048380448,
            "ci95": [
              2.163974226935908e-05,
              0.0002389418029011614
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.00010914459632278828,
            "ci95": [
              -4.681830387040994e-05,
              0.0002751002325207939
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.0003712090021539983,
            "ci95": [
              0.00023112402370331023,
              0.0005153675406708224
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.00026206440583121,
            "ci95": [
              0.00011458072039746289,
              0.00040362689630411033
            ]
          }
        }
      },
      "odd_last__minus__first_odd": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "first_even": 0.0,
          "first_odd": -1.0,
          "even_last": 0.0,
          "odd_last": 1.0,
          "odd_no_last": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.006022630265311579,
            "ci95": [
              -0.00889742853069146,
              -0.0033833486818336195
            ]
          },
          "auc.is_click": {
            "estimate": -0.0013886171018001603,
            "ci95": [
              -0.002223332072081685,
              -0.0006234038255095069
            ]
          },
          "auc.cart": {
            "estimate": -0.008747005699913823,
            "ci95": [
              -0.01299800548444927,
              -0.004486577411031159
            ]
          },
          "auc.fav": {
            "estimate": -0.007289061510954209,
            "ci95": [
              -0.013631030487405401,
              -0.0014890661337168987
            ]
          },
          "auc.buy": {
            "estimate": -0.0066658367485781245,
            "ci95": [
              -0.014309558152168991,
              0.0006489206775979055
            ]
          },
          "bce.is_click": {
            "estimate": -0.0004419463030239934,
            "ci95": [
              -0.0005097215264894199,
              -0.0003628507667751025
            ]
          },
          "bce.cart": {
            "estimate": 0.00016230866710663372,
            "ci95": [
              7.305040112271328e-05,
              0.0002621229408487235
            ]
          },
          "bce.fav": {
            "estimate": 0.00020837513461310908,
            "ci95": [
              6.67317362092256e-05,
              0.00036138380429062814
            ]
          },
          "bce.buy": {
            "estimate": 0.00012001396312455448,
            "ci95": [
              2.9504850392185476e-05,
              0.00022021907861039945
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.007358388598113663,
            "ci95": [
              0.002671677360639399,
              0.01161422762181923
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.005900444409154049,
            "ci95": [
              2.780676255200758e-05,
              0.012358618206662973
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.005277219646777964,
            "ci95": [
              -0.0019483624703746953,
              0.013000922899513616
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.0014579441889596145,
            "ci95": [
              -0.008532910671673648,
              0.006688107170783924
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.002081168951335699,
            "ci95": [
              -0.010723256834566186,
              0.006785357618575447
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.0006232247623760845,
            "ci95": [
              -0.010373061124465404,
              0.009783179775936427
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.0006042549701306271,
            "ci95": [
              -0.0007223115542366902,
              -0.00048131625096399717
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0006503214376371025,
            "ci95": [
              -0.0008195256551238142,
              -0.0004998386006884387
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0005619602661485479,
            "ci95": [
              -0.000686128890170033,
              -0.00044472906195626335
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -4.606646750647536e-05,
            "ci95": [
              -0.0002297053386931743,
              0.0001202152370409929
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 4.229470398207924e-05,
            "ci95": [
              -8.742750352097139e-05,
              0.00017430875296269285
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 8.83611714885546e-05,
            "ci95": [
              -9.354635295420978e-05,
              0.0002544018378100326
            ]
          }
        }
      },
      "odd_last__minus__odd_no_last": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "first_even": 0.0,
          "first_odd": 0.0,
          "even_last": 0.0,
          "odd_last": 1.0,
          "odd_no_last": -1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.002427166213061549,
            "ci95": [
              -0.0008317778533958271,
              0.005225739664836559
            ]
          },
          "auc.is_click": {
            "estimate": -0.0024829014516027925,
            "ci95": [
              -0.0032381058664437525,
              -0.0016659768147832155
            ]
          },
          "auc.cart": {
            "estimate": -0.0003123463047245423,
            "ci95": [
              -0.00473346793710189,
              0.004259046633167263
            ]
          },
          "auc.fav": {
            "estimate": 0.00967493063942182,
            "ci95": [
              0.0032608001605550635,
              0.017041403962247043
            ]
          },
          "auc.buy": {
            "estimate": 0.0028289819691517115,
            "ci95": [
              -0.005037999621762043,
              0.01000304283946957
            ]
          },
          "bce.is_click": {
            "estimate": -1.0929621927530109e-05,
            "ci95": [
              -8.583117184316272e-05,
              5.3111262788485546e-05
            ]
          },
          "bce.cart": {
            "estimate": 0.0001190756626019869,
            "ci95": [
              2.7397692129382817e-05,
              0.00022743820762101763
            ]
          },
          "bce.fav": {
            "estimate": 7.271865706808336e-05,
            "ci95": [
              -7.046496478449547e-05,
              0.00021825909959942712
            ]
          },
          "bce.buy": {
            "estimate": -6.357111658908399e-05,
            "ci95": [
              -0.00014218264984896352,
              2.5835039936762613e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.0021705551468782502,
            "ci95": [
              -0.0067383827106936856,
              0.0024183412490149715
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.012157832091024612,
            "ci95": [
              -0.019695780568510908,
              -0.005561222201181666
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.005311883420754504,
            "ci95": [
              -0.012585007694333641,
              0.0025039744552457894
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.009987276944146362,
            "ci95": [
              -0.018999963075901634,
              -0.0014012666850168375
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.003141328273876254,
            "ci95": [
              -0.01172881846118518,
              0.005000532677397409
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.006845948670270108,
            "ci95": [
              -0.003386344152991938,
              0.01724255312937661
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.000130005284529517,
            "ci95": [
              -0.00025298024081321505,
              -1.646229091235018e-05
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -8.364827899561347e-05,
            "ci95": [
              -0.00024494186205828813,
              7.954122783866439e-05
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 5.264149466155388e-05,
            "ci95": [
              -5.890498379138853e-05,
              0.00015329772773442667
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 4.635700553390354e-05,
            "ci95": [
              -0.00013322694590658978,
              0.0002321708098695787
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.0001826467791910709,
            "ci95": [
              6.262723022783012e-05,
              0.00030807965697478
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.00013628977365716735,
            "ci95": [
              -3.717365142913466e-05,
              0.000301006900109077
            ]
          }
        }
      },
      "odd_last__minus__even_last": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "first_even": 0.0,
          "first_odd": 0.0,
          "even_last": -1.0,
          "odd_last": 1.0,
          "odd_no_last": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.006220056195469698,
            "ci95": [
              -0.009240939085681788,
              -0.0031071687483180525
            ]
          },
          "auc.is_click": {
            "estimate": -0.0032958955647868216,
            "ci95": [
              -0.00406758554726667,
              -0.0024570990797073977
            ]
          },
          "auc.cart": {
            "estimate": -0.011756235384419389,
            "ci95": [
              -0.017160798941470608,
              -0.006478827673708816
            ]
          },
          "auc.fav": {
            "estimate": -0.005594096877898913,
            "ci95": [
              -0.011064314241648235,
              5.327240608569964e-05
            ]
          },
          "auc.buy": {
            "estimate": -0.004233996954773667,
            "ci95": [
              -0.011774306105869891,
              0.003167522045830747
            ]
          },
          "bce.is_click": {
            "estimate": 0.00018439309311732432,
            "ci95": [
              0.00010652105047450692,
              0.0002497649178231032
            ]
          },
          "bce.cart": {
            "estimate": 0.00022290789978268027,
            "ci95": [
              0.00010641322557621023,
              0.0003478381819178856
            ]
          },
          "bce.fav": {
            "estimate": 0.000175269799324404,
            "ci95": [
              5.797801335367678e-05,
              0.0002964311634848277
            ]
          },
          "bce.buy": {
            "estimate": -8.436236818961176e-05,
            "ci95": [
              -0.00017666996656672218,
              2.4055018028015115e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.008460339819632567,
            "ci95": [
              0.0031347724061119935,
              0.013931951289391557
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.0022982013131120915,
            "ci95": [
              -0.0036093566253435653,
              0.008049095647509339
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.0009381013899868451,
            "ci95": [
              -0.0065143818624461836,
              0.008384298527251938
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.006162138506520476,
            "ci95": [
              -0.013464313466846527,
              0.0011678356411724106
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.007522238429645722,
            "ci95": [
              -0.0156777024332632,
              0.0006383764533144631
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.0013600999231252464,
            "ci95": [
              -0.010588708201506147,
              0.007500131262991794
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -3.851480666535595e-05,
            "ci95": [
              -0.00018568517776988737,
              0.00010034540563336986
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 9.123293792920317e-06,
            "ci95": [
              -0.00012178303075282586,
              0.0001357669993991026
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.0002687554613069361,
            "ci95": [
              0.00014167360889683094,
              0.0003801107486412279
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 4.763810045827627e-05,
            "ci95": [
              -0.00010879009958681494,
              0.0002021626702406259
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00030727026797229204,
            "ci95": [
              0.0001617854032375056,
              0.0004525094165045366
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.00025963216751401577,
            "ci95": [
              0.00010924886257970158,
              0.0004071506972521106
            ]
          }
        }
      },
      "endpoint_effect_odd_minus_even": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "first_even": 1.0,
          "first_odd": 0.0,
          "even_last": -1.0,
          "odd_last": 1.0,
          "odd_no_last": -1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.003683270221179119,
            "ci95": [
              -0.008154072547029424,
              0.0007703183683111161
            ]
          },
          "auc.is_click": {
            "estimate": -0.004076195972098073,
            "ci95": [
              -0.005138171365454964,
              -0.0029189230819194416
            ]
          },
          "auc.cart": {
            "estimate": -0.0049367021002625355,
            "ci95": [
              -0.011334951614394375,
              0.0014201582743916606
            ]
          },
          "auc.fav": {
            "estimate": 4.225693741344294e-05,
            "ci95": [
              -0.008565787371336517,
              0.008056617857407656
            ]
          },
          "auc.buy": {
            "estimate": -0.00576243974976931,
            "ci95": [
              -0.01966413203340297,
              0.007041952743169709
            ]
          },
          "bce.is_click": {
            "estimate": 0.00021533378919871837,
            "ci95": [
              0.00011463617265766863,
              0.00031803043488899105
            ]
          },
          "bce.cart": {
            "estimate": 0.00016970668512293965,
            "ci95": [
              2.7375774076672784e-05,
              0.00031731075563044646
            ]
          },
          "bce.fav": {
            "estimate": 0.0005001475323210165,
            "ci95": [
              0.0003155640213035646,
              0.0006885420341292617
            ]
          },
          "bce.buy": {
            "estimate": -3.1977866183099946e-05,
            "ci95": [
              -0.00016131817773513756,
              0.00011875046826143662
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.0008605061281644621,
            "ci95": [
              -0.005591605800147853,
              0.00727056389696085
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.004118452909511516,
            "ci95": [
              -0.012181535372450713,
              0.004783261492760941
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.001686243777671237,
            "ci95": [
              -0.011044793199981728,
              0.015362051870597757
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.0049789590376759785,
            "ci95": [
              -0.015042396996486393,
              0.005805849650137093
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.0008257376495067748,
            "ci95": [
              -0.012498625221194664,
              0.016245006850071447
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.005804696687182753,
            "ci95": [
              -0.010390286105382578,
              0.022268230642195297
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 4.562710407577872e-05,
            "ci95": [
              -0.00012962927552175916,
              0.000211161259525985
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.00028481374312229815,
            "ci95": [
              -0.0004877340199523501,
              -7.288225907590442e-05
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.0002473116553818183,
            "ci95": [
              7.964917619699437e-05,
              0.0004079337019393981
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.00033044084719807687,
            "ci95": [
              -0.000571024754572906,
              -0.00010143940421444398
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.0002016845513060396,
            "ci95": [
              1.5228246296311554e-05,
              0.0003954909589620846
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.0005321253985041165,
            "ci95": [
              0.0002920697624310177,
              0.0007621072960968108
            ]
          }
        }
      }
    },
    "parameter_count": 51724260,
    "initializer_sha256": "fd5f866c35c5e29b37841aeed070f6ef5259c773de173df7549d149f87a8c09e",
    "optimizer_layout_sha256": "3546519bfd702e680acf2f149bc2b97d52851a6e7e6a228d7e063b78f77c181f",
    "formal_worker_cost": {
      "wall_seconds": 2506.589051552117,
      "cpu_seconds": 3006.224509484,
      "peak_rss_bytes": 17033830400,
      "gpu_peak_bytes": 28369223680,
      "samples": 2298
    },
    "smoke_worker_cost": {
      "wall_seconds": 143.2635766044259,
      "cpu_seconds": 156.45439083600002,
      "peak_rss_bytes": 16941613056,
      "gpu_peak_bytes": 29570891776,
      "samples": 132
    },
    "complete_recorded_costs": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2506.589051552117,
        "cpu_seconds": 3006.224509484,
        "peak_rss_bytes": 17033830400,
        "gpu_peak_bytes": 28369223680,
        "samples": 2298
      },
      "formal_runner": {
        "wall_seconds": 2501.779577061534,
        "cpu_seconds": 2896.588143339,
        "peak_process_rss_bytes": 5275492352
      },
      "fit_including_validation": {
        "wall_seconds": 2376.681172825396,
        "cpu_seconds": 2717.655542041,
        "peak_process_rss_bytes": 5275492352
      },
      "prediction_export": {
        "wall_seconds": 120.26605863124132,
        "cpu_seconds": 173.62243787099987,
        "peak_process_rss_bytes": 5275492352
      },
      "smoke_worker": {
        "wall_seconds": 143.2635766044259,
        "cpu_seconds": 156.45439083600002,
        "peak_rss_bytes": 16941613056,
        "gpu_peak_bytes": 29570891776,
        "samples": 132
      },
      "smoke_runner": {
        "wall_seconds": 138.54176331311464,
        "cpu_seconds": 137.051349051,
        "peak_process_rss_bytes": 3235897344
      },
      "latency_constructor": {
        "wall_seconds": 0.7668727189302444,
        "cpu_seconds": 0.8368182060000002,
        "peak_rss_bytes": 2098741248
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.16916535794734955,
        "cpu_seconds": 0.16852593200000143,
        "peak_rss_bytes": 2098741248
      },
      "latency_peak_cuda_bytes": 1281155072,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "forward_latency": {
      "mean_ms": 14.492368817329407,
      "p50_ms": 14.459216117858887,
      "p95_ms": 14.629506874084473,
      "batch_size": 1024,
      "warmups": 10,
      "iterations": 40
    },
    "latency_ratio_to_parent": 1.0055888002421325,
    "quality_candidate": false,
    "near_candidate": false
  },
  "validity": {
    "valid": true,
    "violations": [],
    "state": "passed",
    "quality_summary": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/quality-summary.json",
    "no_prediction_to_work_feedback": true,
    "test_evaluated": false,
    "infrastructure_limitations": [
      "Source metrics are identity/range/arithmetic checks, not recomputed AUC/BCE.",
      "Actual saved-array pairing is deferred to the paired analysis; row fingerprints cover original payload user/item/labels.",
      "Production cost receipts are sampled evidence, not new GPU measurements or hard resource guarantees.",
      "Maturity timestamp inference and click-availability assumptions remain unchanged.",
      "Two segment references have independently recovered real-worker terminals; original aggregate launcher terminal/cost unavailable.",
      "Same GPU and exact input; references and new models measured in separate execution windows after import failure. Original failure and timings retained.",
      "Original paired process terminated by failed latency pipeline; completed draws unknown. Recovered500-draw result preserves same protocol."
    ],
    "measurement_recovery_evidence": {
      "pipeline_failure": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/measurement-pipeline/failure.json",
      "paired_recovery": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/paired-recovery-review.json",
      "prior_latency_failure": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/latency/failure.json"
    },
    "new_terminal_evidence": {
      "completion_path": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/launch-formal/completion.json",
      "completion": {
        "state": "completed",
        "phase": "formal",
        "launcher_exit_code": 0,
        "worker_exit_codes": {
          "even_last": 0,
          "odd_last": 0,
          "odd_no_last": 0
        },
        "authorization_sha256": "23b18e68185190f0e7f9130a18b60cb4c750eb004ce3da407cdd7ca662c5bbf4",
        "smoke_review_sha256": "08540d3f96b3311ad2e13ec1691b525d8fe5feb3682777c7fd3a802b9316d334",
        "worker_receipt_paths": {
          "even_last": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/even_last/formal/exit.json",
          "odd_last": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/odd_last/formal/exit.json",
          "odd_no_last": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/odd_no_last/formal/exit.json"
        },
        "worker_receipt_sha256": {
          "even_last": "42ec66c3cf5be09d64964b46f48b417146a9c546991339f3f9b37229cdcc152d",
          "odd_last": "681b95cf3e65aa1b591cb92612c06ec9a1c0ec5a2e864918f425d5ac0d7e4fb0",
          "odd_no_last": "6143947e9693d9060ac0be95b7a7984bca8a7af90bf2b07ecf8d9c858da8c47b"
        },
        "error": null,
        "receipt_errors": [],
        "deadline_exceeded": false,
        "signals": [],
        "cleanup": {
          "completed": true,
          "signals": [],
          "reaped_adoptees": [],
          "remaining": [],
          "wall_seconds": 0.015955939888954163,
          "subreaper": true,
          "pidfd_identity_checked": false,
          "signalling_methods": [],
          "fallback_scope": "ENOSYS only: fresh owned PPID chain and start-time checks before os.kill; these checks and kill are not atomic like pidfd."
        },
        "wall_seconds": 2507.1962953284383,
        "launcher_wall_limit_seconds": 7320,
        "cpu_seconds_including_reaped_children": 9020.509701064,
        "launcher_peak_rss_bytes": 132202496,
        "worker_costs": {
          "even_last": {
            "wall_seconds": 2504.268677316606,
            "cpu_seconds": 3004.7624286259997,
            "peak_rss_bytes": 17043628032,
            "gpu_peak_bytes": 28369223680,
            "samples": 2296
          },
          "odd_last": {
            "wall_seconds": 2506.589051552117,
            "cpu_seconds": 3006.224509484,
            "peak_rss_bytes": 17033830400,
            "gpu_peak_bytes": 28369223680,
            "samples": 2298
          },
          "odd_no_last": {
            "wall_seconds": 2506.5823511853814,
            "cpu_seconds": 3007.4703000850004,
            "peak_rss_bytes": 17018593280,
            "gpu_peak_bytes": 28369223680,
            "samples": 2297
          }
        },
        "automatic_retry": false,
        "partial_preserved": true,
        "missing_workers": []
      },
      "worker_costs": {
        "even_last": {
          "wall_seconds": 2504.268677316606,
          "cpu_seconds": 3004.7624286259997,
          "peak_rss_bytes": 17043628032,
          "gpu_peak_bytes": 28369223680,
          "samples": 2296
        },
        "odd_last": {
          "wall_seconds": 2506.589051552117,
          "cpu_seconds": 3006.224509484,
          "peak_rss_bytes": 17033830400,
          "gpu_peak_bytes": 28369223680,
          "samples": 2298
        },
        "odd_no_last": {
          "wall_seconds": 2506.5823511853814,
          "cpu_seconds": 3007.4703000850004,
          "peak_rss_bytes": 17018593280,
          "gpu_peak_bytes": 28369223680,
          "samples": 2297
        }
      }
    },
    "inherited_reference_provenance": {
      "mature_r2": {
        "quality_path": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/quality-summary.json",
        "quality_sha256": "8012e1ea78f95e8d9e1fae8703bb4293ceccdb501b4b3995808698433a18abbd",
        "producer_small_artifact_sha256": {
          "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2/result.json": "15c18a7aee32a1dd3015c9a8c9613fe532887416cd49e5a063943f3eee4d246b",
          "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2/manifest.json": "a8439ee9ab331449eb1625f5822deb9a2ad15bf4ad5be8fefa451e78a17949de",
          "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/reports/mature_r2/formal/exit.json": "0a814029b8cedcea62429041dacff3eda1fa875e93151ed300e2151e3456cae7",
          "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2/initial-identity.json": "e1ecc656ac3fb59e413147bc94c59b8d3bfeb4a4808e7de257b5d6c8cf20500c",
          "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2/optimizer-layout.json": "2dfcb20341c7d6da0ff6080dd8a97d83dadfbead33e9f227c95a28054bf1edb6",
          "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2/optimizer-effective.json": "03048a8e046c10cbd16657171c94ab7fa0e79fe8367501548df3db312445f163",
          "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2/loader-preparation.json": "665d88436606d0df274dcc7eb96377720251ad9bdc7cc3ced6d5c207f2d84cd8",
          "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2/loader-audit.json": "d1a196f4a9f56af32e8256ca3231361d5871dd273cdcb7186f053e2f270a9bbf",
          "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2/derived-feature-map.json": "b0e310901463881c1bb3f6f0fad3fc0aa9648108effe45b1d9255f86584944e0",
          "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2/derived-metadata.json": "fd3fe51915b980646675e32f400514bec7e38313a74968bac7b4b079865efe8f",
          "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2/completion.json": "f3d6eedc37c7de8e4f26383a4b26f07881c6fd96e03a8d64a264fef012c8ca9e",
          "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/smoke-runs/mature_r2/smoke.json": "d034bd61e75061a211157047d5c4eac9d94559470ef60f348bb94f09223f81f9",
          "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/reports/mature_r2/smoke/exit.json": "dcae789a6a0c64dd982291e62eaab85cfce32f5758cb74fb0fabce223047cbdc"
        },
        "recorded_large_artifact_sha256": {
          "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2/Taobao_Action/mature_r2.model": "c514fb78a24c2bbf4460c65d5b6d97e642e5d9a2359be3f9b4dd43558f872ecd",
          "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2/validation_predictions.npz": "22ad8962837c30ed96704b979c8a7f983c422d9c3626b59b60a701b427553647"
        },
        "smoke_optimizer_initialization_scope": "Inherited successful C quality gates with immutable producer/source identities; original smoke/model execution is not repeated."
      },
      "cross8_mature": {
        "quality_path": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/quality-summary.json",
        "quality_sha256": "8012e1ea78f95e8d9e1fae8703bb4293ceccdb501b4b3995808698433a18abbd",
        "producer_small_artifact_sha256": {
          "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature/result.json": "e301e2959cd9d8aa5443bdc8ee9bdee02576bc62946241c34996435bf9ef6103",
          "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature/manifest.json": "f1a54a6ebd39c772cb1b858b69cb5a132d3b2773da264276c402eb689567e50f",
          "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/cross8_mature/formal/exit.json": "e726a6438fa2c9958595b167f608b369ab559c027705a2e8709f01d3e1b17fc7",
          "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature/initial-identity.json": "6d3e1eb30c86a01f3ce07c1f964738a166bb612a830d0e9d1baa26155b7662c3",
          "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature/optimizer-layout.json": "d1d5b6df938178a71a234b289de7f2c44ad75849abfedb8b8f43769923d56e43",
          "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature/optimizer-effective.json": "0672e2adadbdab668ce94f4bc8ec2d77484ab85d46315766defdafc6825feeb9",
          "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature/loader-preparation.json": "723475dbce92652a4e229e78aa543d55d30d300e077c18891514ef8942ccd474",
          "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature/loader-audit.json": "2c9df0264864bbd1b7e261738b364915c0779be40248abbaa808f55e1f1a2ee3",
          "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature/derived-feature-map.json": "b0e310901463881c1bb3f6f0fad3fc0aa9648108effe45b1d9255f86584944e0",
          "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature/derived-metadata.json": "fd3fe51915b980646675e32f400514bec7e38313a74968bac7b4b079865efe8f",
          "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature/completion.json": "eff96b7015a8a2dd28d68d4471e56e1ff6f8c464dcca63212c3aeadb9eb08cac",
          "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/smoke-runs/cross8_mature/smoke.json": "b63ff5e265f171d54a6865ee40a926cd63435dd7aff5e971c89e0be591434ba3",
          "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/cross8_mature/smoke/exit.json": "e2b92d7e21d9f0f2faacae0ddfb40ae2031ecf4981e15c147b0a172497b7b468"
        },
        "recorded_large_artifact_sha256": {
          "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature/Taobao_Action/cross8_mature.model": "e685bda693b75c6e838a36bfa48f7f74d0fdd6fcd50cb44c81c6858b578056ae",
          "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature/validation_predictions.npz": "7a7b7973ba25b8c52142cb6cd52c2b0683b8a16484b7bb9768c63fcd4b6e6790"
        },
        "smoke_optimizer_initialization_scope": "Inherited successful C quality gates with immutable producer/source identities; original smoke/model execution is not repeated."
      },
      "first_even": {
        "quality_path": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/quality-summary.json",
        "quality_sha256": "8012e1ea78f95e8d9e1fae8703bb4293ceccdb501b4b3995808698433a18abbd",
        "producer_small_artifact_sha256": {
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_even/result.json": "a6955c66720f6698a85c6b1b48224c385fb161428fdc39d5158b6393ed185145",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_even/manifest.json": "4a7e620352c18b79d89cf49cec6363400a000943f290d698a8af60cb812ec59a",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/first_even/formal/exit.json": "875ea534b3fdc2cd7e99939c6672bd2c96e4d7050b12b6ce9725e9d281d49038",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_even/initial-identity.json": "052c134c6b636ee83bfef2c6234e010b85064652155f3e90b157a67e8af0f86f",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_even/optimizer-layout.json": "96b185f1feaea60c6e693d7982052a79184d4ba57105acb11e45042625aaf67c",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_even/optimizer-effective.json": "03048a8e046c10cbd16657171c94ab7fa0e79fe8367501548df3db312445f163",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_even/loader-preparation.json": "cb0e949658ea71aa027eb0a1dc026034eab46bcb415e250b01d0c911f8ceb12f",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_even/loader-audit.json": "9d53916f0cb1983b952931c9dd61d1ab1090f8532274bddcde6f4a974479e514",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_even/derived-feature-map.json": "b0e310901463881c1bb3f6f0fad3fc0aa9648108effe45b1d9255f86584944e0",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_even/derived-metadata.json": "fd3fe51915b980646675e32f400514bec7e38313a74968bac7b4b079865efe8f",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_even/completion.json": "f0f1b7ffd576bdaf72664e492860f53f6984baab4876ac426f7d6596b00e61a0",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/smoke-runs/first_even/smoke.json": "e456922f23eed006b4ffa0d17be5e4c44c8cb8b275a899cec44933df4f4474f4",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/first_even/smoke/exit.json": "f1ff58f5be4b9f5eec70861ca8d7d43720bcf8f505c785f34f0fd24ec516a229"
        },
        "recorded_large_artifact_sha256": {
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_even/Taobao_Action/first_even.model": "a08f350cdda9524b5b64fa1ab257c999c89108bf79faa4003fcad72fef2bd760",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_even/validation_predictions.npz": "ba55d162846e963126fc5065052ebb72c7ee973ed17591110cb1f92315d9c27d"
        },
        "smoke_optimizer_initialization_scope": "Inherited successful C quality gates with immutable producer/source identities; original smoke/model execution is not repeated."
      },
      "first_odd": {
        "quality_path": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/quality-summary.json",
        "quality_sha256": "8012e1ea78f95e8d9e1fae8703bb4293ceccdb501b4b3995808698433a18abbd",
        "producer_small_artifact_sha256": {
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_odd/result.json": "9ca8426ec0ef9fc2f65fd48208f7869c55048dc044b94c3ce6c7d01e9c83c3de",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_odd/manifest.json": "62896430d504d095bd7c3bd08c54ce9da5935d8a5aa0e2b589f9709d2a4984ee",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/first_odd/formal/exit.json": "2830a5f1142efad9bf7d2870ec6ad384ae3468104a15e0f50dc0eada0146d589",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_odd/initial-identity.json": "2a73b409dbaee134ab3b7a734927528aa9aa7ce94765ae91763348c13b6ce229",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_odd/optimizer-layout.json": "f92b61b437f3ea0e8610574e1546bf4af731ed57ab9ce96872c4cb87f94763f1",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_odd/optimizer-effective.json": "03048a8e046c10cbd16657171c94ab7fa0e79fe8367501548df3db312445f163",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_odd/loader-preparation.json": "9244f43791a5c92b7d35dd9bb823f8eb5fd32ed5f699f7d71e7583f49e70bdbe",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_odd/loader-audit.json": "5b532fca1c299eed13f345aff211674c029fdfb2ef39005608dc01184cf15572",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_odd/derived-feature-map.json": "b0e310901463881c1bb3f6f0fad3fc0aa9648108effe45b1d9255f86584944e0",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_odd/derived-metadata.json": "fd3fe51915b980646675e32f400514bec7e38313a74968bac7b4b079865efe8f",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_odd/completion.json": "c81c54df3a4e54751d80a18b95eec38e273e22536e9cf1edc75b0c181d5a01ce",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/smoke-runs/first_odd/smoke.json": "4388338f9a9faab0035065a911a6a6ee5ecf8f5d85d6dd629d133929509bc8a5",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/first_odd/smoke/exit.json": "cd8760337a84b26c3958e18e48cc8cdc0d34cf93643b4166741ff28428c3ac17"
        },
        "recorded_large_artifact_sha256": {
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_odd/Taobao_Action/first_odd.model": "e2123b38b543c10bba28b4ebbd45f558185c3701314463e91e03abcb3197dd1f",
          "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_odd/validation_predictions.npz": "fead70989033318cb4ad5a6c5fda322eeda531662d49a0c244b86b5a59c330a5"
        },
        "smoke_optimizer_initialization_scope": "Inherited successful C quality gates with immutable producer/source identities; original smoke/model execution is not repeated."
      }
    },
    "source_and_input_identity_scope": "Original N full-input content SHA256 values are inherited through completed C quality and E authorization. Production runners checked input stats before/after their traversals and two original metadata hashes before them. This collector now checks 122 current stats and both original metadata hashes; it does not rehash Parquet or open checkpoint/prediction arrays. New E worker/smoke/effective-optimizer gates are evaluated from actual small receipts. Two references retain exact C quality records and producer completion hashes, with small source/artifact identity checks. Stat continuity cannot independently exclude content changes that preserve stat fields."
  },
  "decision": "discard",
  "decision_reason": "Macro delta -0.004784711,CI95 [-0.007743902623144495, -0.00213613676149459];latency ratio1.005589. Below matched original recurrence; preserve negative outcome.",
  "decision_scope": "Root queue disposition only; no released-incumbent promotion or supported lesson.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/runs/odd_last",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/runs/odd_last/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/runs/odd_last/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/runs/odd_last/Taobao_Action/odd_last.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/paired-analysis-recovery/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/latency-recovery/completion.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/runs/odd_last",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/runs/odd_last/result.json",
    "result_sha256": "6fe5a68fce701f7226f1b96d66c16bc4b1627b61d29303d2d48f758bb2e3c7fa",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/runs/odd_last/manifest.json",
    "manifest_sha256": "742c70a8f5dfc1af89e24bc1ff4153854be43facb60e286e3ea0d7d8538b85a6",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/runs/odd_last/Taobao_Action/odd_last.model",
    "checkpoint_sha256": "1347bd8c2881f7de250b331bde7f39b29bddb5cc8b67ba4271867ed04f2f47d3",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/paired-analysis-recovery/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
