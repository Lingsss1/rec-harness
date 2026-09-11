## 260. endpoint-read--odd_no_last--s20262027

- 假设：Separate first-work stored-endpoint retention from interleaving phase at fixed support count, two work rounds and no prediction feedback.
- 判定：discard
- 原因：Macro delta -0.007211877,CI95 [-0.010450704717793483, -0.004335497843915446];latency ratio1.008237. Below matched original recurrence; preserve negative outcome.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "endpoint-read--odd_no_last--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-10T15:23:24.099568+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "odd_no_last",
  "code_id": "b16543b20d537e062ccfd9a027ec0a882a6c396851eddf999162c82a0a9c9bc7",
  "hypothesis": "Separate first-work stored-endpoint retention from interleaving phase at fixed support count, two work rounds and no prediction feedback.",
  "change": "{\"model\": \"QFormerCross8EndpointRead\", \"maturity_mode\": \"mature\", \"segment_placement\": \"first\", \"segment_selection\": \"odd_no_last\", \"work_steps\": 2, \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8EndpointRead",
    "maturity_mode": "mature",
    "segment_placement": "first",
    "segment_selection": "odd_no_last",
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
      "value": 0.7598830742516619,
      "delta": -0.007211877063198646
    },
    "per_task": {
      "is_click": {
        "logloss": 0.18421257770061603,
        "AUC": 0.6501550301521375
      },
      "cart": {
        "logloss": 0.02184317557286285,
        "AUC": 0.7854172090891355
      },
      "fav": {
        "logloss": 0.016919355523988308,
        "AUC": 0.7887361160329851
      },
      "buy": {
        "logloss": 0.00982759204459624,
        "AUC": 0.8152239417323893
      }
    },
    "macro_source_logloss": 0.058200675210515855,
    "source_metric_comparison": {
      "is_click": {
        "source_AUC": 0.6501550301521375,
        "computed_AUC": 0.6501550301521375,
        "auc_difference": 0.0,
        "source_logloss": 0.18421257770061603,
        "computed_bce": 0.1842125777006177,
        "bce_difference": 1.6653345369377348e-15
      },
      "cart": {
        "source_AUC": 0.7854172090891355,
        "computed_AUC": 0.7854172090891355,
        "auc_difference": 0.0,
        "source_logloss": 0.02184317557286285,
        "computed_bce": 0.021843175576246047,
        "bce_difference": 3.3831965007280473e-12
      },
      "fav": {
        "source_AUC": 0.7887361160329851,
        "computed_AUC": 0.7887361160329851,
        "auc_difference": 0.0,
        "source_logloss": 0.016919355523988308,
        "computed_bce": 0.01691935552983118,
        "bce_difference": 5.842871325656418e-12
      },
      "buy": {
        "source_AUC": 0.8152239417323893,
        "computed_AUC": 0.8152239417323893,
        "auc_difference": 0.0,
        "source_logloss": 0.00982759204459624,
        "computed_bce": 0.009827592044642218,
        "bce_difference": 4.597884573076527e-14
      }
    },
    "matched_parent_effect": {
      "coefficients": {
        "mature_r2": -1.0,
        "cross8_mature": 0.0,
        "first_even": 0.0,
        "first_odd": 0.0,
        "even_last": 0.0,
        "odd_last": 0.0,
        "odd_no_last": 1.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": -0.007211877063198646,
          "ci95": [
            -0.010450704717793483,
            -0.004335497843915446
          ]
        },
        "auc.is_click": {
          "estimate": 0.00021873695114804637,
          "ci95": [
            -0.0006215488156432468,
            0.0010375658790260514
          ]
        },
        "auc.cart": {
          "estimate": -0.0070170612682082245,
          "ci95": [
            -0.011528383421199296,
            -0.0020213935962564355
          ]
        },
        "auc.fav": {
          "estimate": -0.013727374342062948,
          "ci95": [
            -0.021030738254513924,
            -0.007090775369115237
          ]
        },
        "auc.buy": {
          "estimate": -0.008321809593671459,
          "ci95": [
            -0.0162681313668192,
            -0.0013407905682458346
          ]
        },
        "bce.is_click": {
          "estimate": -0.0001790685671689829,
          "ci95": [
            -0.00025137098129006315,
            -0.00010501265191124667
          ]
        },
        "bce.cart": {
          "estimate": 1.522120681201472e-05,
          "ci95": [
            -8.489571233192524e-05,
            0.00011502263293947195
          ]
        },
        "bce.fav": {
          "estimate": 0.0001211978354788458,
          "ci95": [
            -2.646358327328792e-05,
            0.0002805527624010523
          ]
        },
        "bce.buy": {
          "estimate": 0.00018379010791651565,
          "ci95": [
            9.836047300364756e-05,
            0.00028138560332595695
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": 0.007235798219356271,
          "ci95": [
            0.002033912932313634,
            0.012083418519346531
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": 0.013946111293210994,
          "ci95": [
            0.007295671669147242,
            0.02110973764568168
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": 0.008540546544819505,
          "ci95": [
            0.0015765715726061365,
            0.01632236520739133
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": 0.006710313073854723,
          "ci95": [
            -0.0018732538770476227,
            0.015150721416350621
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": 0.0013047483254632342,
          "ci95": [
            -0.0077558352158897505,
            0.010384902320044066
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": -0.005405564748391489,
          "ci95": [
            -0.016291355620291263,
            0.00505180173635093
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": -0.00019428977398099762,
          "ci95": [
            -0.00032661627439225344,
            -6.865640805768442e-05
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": -0.0003002664026478287,
          "ci95": [
            -0.0004678067841240623,
            -0.0001431035054842087
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": -0.00036285867508549854,
          "ci95": [
            -0.00047506621075379284,
            -0.00025179612713963816
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": -0.00010597662866683108,
          "ci95": [
            -0.0003008787230147368,
            7.451376945698994e-05
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": -0.00016856890110450093,
          "ci95": [
            -0.00029976674963725844,
            -3.770891122710903e-05
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": -6.259227243766985e-05,
          "ci95": [
            -0.000233814467480291,
            0.00012552559875253922
          ]
        }
      }
    },
    "all_prespecified_effects_involving_candidate": {
      "odd_no_last__minus__mature_r2": {
        "coefficients": {
          "mature_r2": -1.0,
          "cross8_mature": 0.0,
          "first_even": 0.0,
          "first_odd": 0.0,
          "even_last": 0.0,
          "odd_last": 0.0,
          "odd_no_last": 1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.007211877063198646,
            "ci95": [
              -0.010450704717793483,
              -0.004335497843915446
            ]
          },
          "auc.is_click": {
            "estimate": 0.00021873695114804637,
            "ci95": [
              -0.0006215488156432468,
              0.0010375658790260514
            ]
          },
          "auc.cart": {
            "estimate": -0.0070170612682082245,
            "ci95": [
              -0.011528383421199296,
              -0.0020213935962564355
            ]
          },
          "auc.fav": {
            "estimate": -0.013727374342062948,
            "ci95": [
              -0.021030738254513924,
              -0.007090775369115237
            ]
          },
          "auc.buy": {
            "estimate": -0.008321809593671459,
            "ci95": [
              -0.0162681313668192,
              -0.0013407905682458346
            ]
          },
          "bce.is_click": {
            "estimate": -0.0001790685671689829,
            "ci95": [
              -0.00025137098129006315,
              -0.00010501265191124667
            ]
          },
          "bce.cart": {
            "estimate": 1.522120681201472e-05,
            "ci95": [
              -8.489571233192524e-05,
              0.00011502263293947195
            ]
          },
          "bce.fav": {
            "estimate": 0.0001211978354788458,
            "ci95": [
              -2.646358327328792e-05,
              0.0002805527624010523
            ]
          },
          "bce.buy": {
            "estimate": 0.00018379010791651565,
            "ci95": [
              9.836047300364756e-05,
              0.00028138560332595695
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.007235798219356271,
            "ci95": [
              0.002033912932313634,
              0.012083418519346531
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.013946111293210994,
            "ci95": [
              0.007295671669147242,
              0.02110973764568168
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.008540546544819505,
            "ci95": [
              0.0015765715726061365,
              0.01632236520739133
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.006710313073854723,
            "ci95": [
              -0.0018732538770476227,
              0.015150721416350621
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.0013047483254632342,
            "ci95": [
              -0.0077558352158897505,
              0.010384902320044066
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.005405564748391489,
            "ci95": [
              -0.016291355620291263,
              0.00505180173635093
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00019428977398099762,
            "ci95": [
              -0.00032661627439225344,
              -6.865640805768442e-05
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0003002664026478287,
            "ci95": [
              -0.0004678067841240623,
              -0.0001431035054842087
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.00036285867508549854,
            "ci95": [
              -0.00047506621075379284,
              -0.00025179612713963816
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.00010597662866683108,
            "ci95": [
              -0.0003008787230147368,
              7.451376945698994e-05
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -0.00016856890110450093,
            "ci95": [
              -0.00029976674963725844,
              -3.770891122710903e-05
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -6.259227243766985e-05,
            "ci95": [
              -0.000233814467480291,
              0.00012552559875253922
            ]
          }
        }
      },
      "odd_no_last__minus__cross8_mature": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": -1.0,
          "first_even": 0.0,
          "first_odd": 0.0,
          "even_last": 0.0,
          "odd_last": 0.0,
          "odd_no_last": 1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.00047665415620201657,
            "ci95": [
              -0.002152493483931351,
              0.0036434595326973736
            ]
          },
          "auc.is_click": {
            "estimate": 0.0002182600091718001,
            "ci95": [
              -0.0005639049195782758,
              0.0008632253667535644
            ]
          },
          "auc.cart": {
            "estimate": -0.003789526281980482,
            "ci95": [
              -0.007927144900679046,
              0.000105431817310876
            ]
          },
          "auc.fav": {
            "estimate": -0.007765894994464451,
            "ci95": [
              -0.0140960104954836,
              -0.0017512228961798421
            ]
          },
          "auc.buy": {
            "estimate": 0.013243777892081199,
            "ci95": [
              0.004828506265797803,
              0.02287253863460059
            ]
          },
          "bce.is_click": {
            "estimate": -0.00010536014797365989,
            "ci95": [
              -0.00016599264762869325,
              -3.210905534687324e-05
            ]
          },
          "bce.cart": {
            "estimate": 8.763939167016899e-06,
            "ci95": [
              -9.236595628907526e-05,
              9.967500571189138e-05
            ]
          },
          "bce.fav": {
            "estimate": -5.402365162186784e-05,
            "ci95": [
              -0.00017275899528893272,
              6.177010917126002e-05
            ]
          },
          "bce.buy": {
            "estimate": -0.0001797982837959105,
            "ci95": [
              -0.00027721992894584114,
              -9.610986962096519e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.004007786291152282,
            "ci95": [
              -0.00015466176793047225,
              0.007963821863863274
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.00798415500363625,
            "ci95": [
              0.002073935278449815,
              0.014519211776766171
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.013025517882909399,
            "ci95": [
              -0.022686511624666613,
              -0.004762395212030254
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.003976368712483969,
            "ci95": [
              -0.003475459760737895,
              0.01138039040318486
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.01703330417406168,
            "ci95": [
              -0.0278463466019292,
              -0.008159081130240555
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.02100967288654565,
            "ci95": [
              -0.03183949412444964,
              -0.011783802450770516
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00011412408714067679,
            "ci95": [
              -0.00021578080348572793,
              2.331028293415562e-06
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -5.133649635179205e-05,
            "ci95": [
              -0.00017818131809963726,
              8.916072647709187e-05
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 7.44381358222506e-05,
            "ci95": [
              -2.9078195951138255e-05,
              0.00019403799093657963
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 6.278759078888474e-05,
            "ci95": [
              -8.06478791227771e-05,
              0.0002132517255605096
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.0001885622229629274,
            "ci95": [
              6.517806646542976e-05,
              0.00033439475355145936
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.00012577463217404265,
            "ci95": [
              -2.0263531129138767e-05,
              0.00028060336548768887
            ]
          }
        }
      },
      "odd_no_last__minus__first_odd": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "first_even": 0.0,
          "first_odd": -1.0,
          "even_last": 0.0,
          "odd_last": 0.0,
          "odd_no_last": 1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.008449796478373128,
            "ci95": [
              -0.011282326392177782,
              -0.005765063190823054
            ]
          },
          "auc.is_click": {
            "estimate": 0.0010942843498026322,
            "ci95": [
              0.00030920697017778616,
              0.0018991155698810935
            ]
          },
          "auc.cart": {
            "estimate": -0.008434659395189281,
            "ci95": [
              -0.013047169806225311,
              -0.0035658191249826124
            ]
          },
          "auc.fav": {
            "estimate": -0.016963992150376028,
            "ci95": [
              -0.02342923733529972,
              -0.011401724966266746
            ]
          },
          "auc.buy": {
            "estimate": -0.009494818717729836,
            "ci95": [
              -0.018557489490479275,
              -0.0007250863475600196
            ]
          },
          "bce.is_click": {
            "estimate": -0.0004310166810964633,
            "ci95": [
              -0.0004985596866731445,
              -0.0003527996707364818
            ]
          },
          "bce.cart": {
            "estimate": 4.3233004504646816e-05,
            "ci95": [
              -4.9073016906035676e-05,
              0.0001367515818899792
            ]
          },
          "bce.fav": {
            "estimate": 0.00013565647754502572,
            "ci95": [
              3.521505916500596e-05,
              0.00025933356876875756
            ]
          },
          "bce.buy": {
            "estimate": 0.00018358507971363847,
            "ci95": [
              8.996976811541575e-05,
              0.0002835010699767706
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.009528943744991913,
            "ci95": [
              0.004597220245976816,
              0.013909892482105009
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.01805827650017866,
            "ci95": [
              0.012231207381951981,
              0.024267739190841225
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.010589103067532468,
            "ci95": [
              0.001808245651068502,
              0.019592906932906207
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.008529332755186747,
            "ci95": [
              0.001130894784179787,
              0.01641892441165282
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.0010601593225405548,
            "ci95": [
              -0.009739036511094411,
              0.011913033308645321
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.007469173432646192,
            "ci95": [
              -0.01840331737250304,
              0.0036183251607094504
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.0004742496856011101,
            "ci95": [
              -0.0005848445151080129,
              -0.0003549348177189802
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.000566673158641489,
            "ci95": [
              -0.0007064725373786984,
              -0.0004431116028394951
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0006146017608101018,
            "ci95": [
              -0.0007321102646315692,
              -0.0004891615999272127
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -9.24234730403789e-05,
            "ci95": [
              -0.00024304541498223683,
              4.308928434168716e-05
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -0.00014035207520899165,
            "ci95": [
              -0.00026963858033555207,
              -6.925418515164307e-06
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -4.792860216861275e-05,
            "ci95": [
              -0.00019293568477870825,
              0.00011074776093323925
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
      "odd_no_last__minus__first_even": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "first_even": -1.0,
          "first_odd": 0.0,
          "even_last": 0.0,
          "odd_last": 0.0,
          "odd_no_last": 1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.0025367859742905785,
            "ci95": [
              -0.005345358582909904,
              0.0006210780299661046
            ]
          },
          "auc.is_click": {
            "estimate": 0.0007803004073112518,
            "ci95": [
              -3.9978046241356795e-05,
              0.0015850239626314458
            ]
          },
          "auc.cart": {
            "estimate": -0.0068195332841568534,
            "ci95": [
              -0.011788933112107724,
              -0.001830848027816643
            ]
          },
          "auc.fav": {
            "estimate": -0.005636353815312356,
            "ci95": [
              -0.01209980172494099,
              0.0007165924514521692
            ]
          },
          "auc.buy": {
            "estimate": 0.0015284427949956436,
            "ci95": [
              -0.0072116015247703415,
              0.012166517522881823
            ]
          },
          "bce.is_click": {
            "estimate": -3.094069608139405e-05,
            "ci95": [
              -0.00010739564189859927,
              3.9771756786253336e-05
            ]
          },
          "bce.cart": {
            "estimate": 5.320121465974062e-05,
            "ci95": [
              -5.6523547793049366e-05,
              0.00016310913420684902
            ]
          },
          "bce.fav": {
            "estimate": -0.0003248777329966125,
            "ci95": [
              -0.00044669449278977955,
              -0.0002031193379967989
            ]
          },
          "bce.buy": {
            "estimate": -5.238450200651182e-05,
            "ci95": [
              -0.00015568432784731812,
              3.284417210266649e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.007599833691468105,
            "ci95": [
              0.002579294740104346,
              0.01267710954336558
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.006416654222623608,
            "ci95": [
              -4.2663423753028206e-05,
              0.012859445789142625
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.0007481423876843918,
            "ci95": [
              -0.011557381762436422,
              0.00815701930306548
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.0011831794688444974,
            "ci95": [
              -0.009107626515245132,
              0.006221275885496133
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.008347976079152497,
            "ci95": [
              -0.020102656548899495,
              0.0021639142932250786
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.007164796610308,
            "ci95": [
              -0.020398217563614804,
              0.00487718548732949
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -8.414191074113467e-05,
            "ci95": [
              -0.00021837812673965718,
              4.732810304790696e-05
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.00029393703691521847,
            "ci95": [
              0.0001430534700142895,
              0.000427084151334619
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 2.144380592511777e-05,
            "ci95": [
              -8.261925152371688e-05,
              0.00015138532247860415
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.00037807894765635314,
            "ci95": [
              0.0002232238420875366,
              0.0005426003874247117
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00010558571666625244,
            "ci95": [
              -4.084643013366625e-05,
              0.0002580287619509442
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.0002724932309901007,
            "ci95": [
              -0.0004397770236600994,
              -0.00011394552837202286
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
    "optimizer_layout_sha256": "a007450b61e908e8c8832065c25f3161edcc388544c307ba2bfaa426c20502f3",
    "formal_worker_cost": {
      "wall_seconds": 2506.5823511853814,
      "cpu_seconds": 3007.4703000850004,
      "peak_rss_bytes": 17018593280,
      "gpu_peak_bytes": 28369223680,
      "samples": 2297
    },
    "smoke_worker_cost": {
      "wall_seconds": 139.8507951721549,
      "cpu_seconds": 153.768312719,
      "peak_rss_bytes": 16946548736,
      "gpu_peak_bytes": 29570891776,
      "samples": 129
    },
    "complete_recorded_costs": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2506.5823511853814,
        "cpu_seconds": 3007.4703000850004,
        "peak_rss_bytes": 17018593280,
        "gpu_peak_bytes": 28369223680,
        "samples": 2297
      },
      "formal_runner": {
        "wall_seconds": 2501.7509671002626,
        "cpu_seconds": 2896.83181013,
        "peak_process_rss_bytes": 5272592384
      },
      "fit_including_validation": {
        "wall_seconds": 2378.765804104507,
        "cpu_seconds": 2723.707955877,
        "peak_process_rss_bytes": 5272592384
      },
      "prediction_export": {
        "wall_seconds": 118.09135127812624,
        "cpu_seconds": 167.66163491099996,
        "peak_process_rss_bytes": 5272592384
      },
      "smoke_worker": {
        "wall_seconds": 139.8507951721549,
        "cpu_seconds": 153.768312719,
        "peak_rss_bytes": 16946548736,
        "gpu_peak_bytes": 29570891776,
        "samples": 129
      },
      "smoke_runner": {
        "wall_seconds": 136.00064123421907,
        "cpu_seconds": 135.380392657,
        "peak_process_rss_bytes": 3245948928
      },
      "latency_constructor": {
        "wall_seconds": 0.7393206655979156,
        "cpu_seconds": 0.8097832609999998,
        "peak_rss_bytes": 2099417088
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.1104494109749794,
        "cpu_seconds": 0.11000088800000007,
        "peak_rss_bytes": 2099417088
      },
      "latency_peak_cuda_bytes": 1281155072,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "forward_latency": {
      "mean_ms": 14.530531239509582,
      "p50_ms": 14.507599830627441,
      "p95_ms": 14.672894859313965,
      "batch_size": 1024,
      "warmups": 10,
      "iterations": 40
    },
    "latency_ratio_to_parent": 1.008236794149699,
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
  "decision_reason": "Macro delta -0.007211877,CI95 [-0.010450704717793483, -0.004335497843915446];latency ratio1.008237. Below matched original recurrence; preserve negative outcome.",
  "decision_scope": "Root queue disposition only; no released-incumbent promotion or supported lesson.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/runs/odd_no_last",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/runs/odd_no_last/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/runs/odd_no_last/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/runs/odd_no_last/Taobao_Action/odd_no_last.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/paired-analysis-recovery/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/latency-recovery/completion.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/runs/odd_no_last",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/runs/odd_no_last/result.json",
    "result_sha256": "b5490a2193cb61d68bc9d67b4a26fffc4d788e03763efd2da15c325d54b0282c",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/runs/odd_no_last/manifest.json",
    "manifest_sha256": "0f417cfd48fef3f15ba8a6e6a6797722c4ea344fd61b11c1af1ade87612fb2b9",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/runs/odd_no_last/Taobao_Action/odd_no_last.model",
    "checkpoint_sha256": "e4ecae11a1dd6596264ed6e48c7323bc89ca829a02a16e2643157143105f1e2c",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/paired-analysis-recovery/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
