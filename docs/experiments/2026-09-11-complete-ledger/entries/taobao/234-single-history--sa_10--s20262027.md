## 234. single-history--sa_10--s20262027

- 假设：One attention encoding used by the first work read while the second returns to H0 can improve retrieval.
- 判定：discard
- 原因：Macro delta versus A -0.00194309, exploratory CI [-0.00499568,+0.00060710], forward latency ratio 1.240588. Fails both predeclared quality/near-candidate rules; retain all artifacts as negative research evidence, with no claim that every interval establishes harm.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "single-history--sa_10--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-10T03:17:15.913140+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "sa_10",
  "code_id": "b973d173c1edbfcedffee2c6ee09f14ebe9b735d9ca52d41c7448e0433b6ab06",
  "hypothesis": "One attention encoding used by the first work read while the second returns to H0 can improve retrieval.",
  "change": "{\"model\": \"QFormerCross8SingleHistory\", \"maturity_mode\": \"mature\", \"history_encoder_kind\": \"attention\", \"history_read_mode\": \"encoded_original\", \"history_condition\": \"none\", \"history_gate_rank\": 32, \"history_bottleneck\": 128, \"work_steps\": 2, \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8SingleHistory",
    "maturity_mode": "mature",
    "history_encoder_kind": "attention",
    "history_read_mode": "encoded_original",
    "history_condition": "none",
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
      "value": 0.7651518652500107,
      "delta": -0.001943086064849775
    },
    "per_task": {
      "is_click": {
        "logloss": 0.1838755378935845,
        "AUC": 0.6515550442894256
      },
      "cart": {
        "logloss": 0.02179183840216676,
        "AUC": 0.7937294149276779
      },
      "fav": {
        "logloss": 0.017171148369273644,
        "AUC": 0.7977689177245938
      },
      "buy": {
        "logloss": 0.009979895541855867,
        "AUC": 0.8175540840583456
      }
    },
    "macro_source_logloss": 0.05820460505172018,
    "source_metric_comparison": {
      "is_click": {
        "source_AUC": 0.6515550442894256,
        "computed_AUC": 0.6515550442894256,
        "auc_difference": 0.0,
        "source_logloss": 0.1838755378935845,
        "computed_bce": 0.18387553789358654,
        "bce_difference": 2.0539125955565396e-15
      },
      "cart": {
        "source_AUC": 0.7937294149276779,
        "computed_AUC": 0.7937294149276779,
        "auc_difference": 0.0,
        "source_logloss": 0.02179183840216676,
        "computed_bce": 0.02179183840218186,
        "bce_difference": 1.5102502581854083e-14
      },
      "fav": {
        "source_AUC": 0.7977689177245938,
        "computed_AUC": 0.7977689177245938,
        "auc_difference": 0.0,
        "source_logloss": 0.017171148369273644,
        "computed_bce": 0.01717114836927353,
        "bce_difference": -1.1449174941446927e-16
      },
      "buy": {
        "source_AUC": 0.8175540840583456,
        "computed_AUC": 0.8175540840583456,
        "auc_difference": 0.0,
        "source_logloss": 0.009979895541855867,
        "computed_bce": 0.009979895587980661,
        "bce_difference": 4.6124794955693105e-11
      }
    },
    "matched_parent_effect": {
      "coefficients": {
        "mature_r2": -1.0,
        "cross8_mature": 0.0,
        "hist_shallow_deep": 0.0,
        "sa_01": 0.0,
        "sa_11": 0.0,
        "sa_10": 1.0,
        "mlp_01": 0.0,
        "mlp_11": 0.0,
        "mlp_10": 0.0,
        "mlp_gate_entry": 0.0,
        "mlp_gate_work": 0.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": -0.001943086064849775,
          "ci95": [
            -0.004995679734683112,
            0.0006071034222691932
          ]
        },
        "auc.is_click": {
          "estimate": 0.0016187510884361211,
          "ci95": [
            0.000928619919922602,
            0.002393087192228682
          ]
        },
        "auc.cart": {
          "estimate": 0.0012951445703341946,
          "ci95": [
            -0.0025723794121350783,
            0.005006301190660905
          ]
        },
        "auc.fav": {
          "estimate": -0.004694572650454254,
          "ci95": [
            -0.011403247152654977,
            0.0011231050358106007
          ]
        },
        "auc.buy": {
          "estimate": -0.005991667267715162,
          "ci95": [
            -0.014350682611751842,
            0.0010857838633049535
          ]
        },
        "bce.is_click": {
          "estimate": -0.0005161083742001304,
          "ci95": [
            -0.0005928819429813894,
            -0.00044728226411067736
          ]
        },
        "bce.cart": {
          "estimate": -3.611596725217048e-05,
          "ci95": [
            -0.00012671992936686528,
            5.6313507448375225e-05
          ]
        },
        "bce.fav": {
          "estimate": 0.00037299067492119545,
          "ci95": [
            0.000245365063584956,
            0.000493529196286127
          ]
        },
        "bce.buy": {
          "estimate": 0.0003360936512549588,
          "ci95": [
            0.0002508564015641738,
            0.00043200248985228115
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": 0.00032360651810192653,
          "ci95": [
            -0.0033568029192969583,
            0.004232153847349371
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": 0.006313323738890375,
          "ci95": [
            0.0003725830345761517,
            0.01307875164899804
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": 0.007610418356151283,
          "ci95": [
            0.0004928078075918164,
            0.01625839103312894
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": 0.005989717220788449,
          "ci95": [
            -0.001044625190786641,
            0.013058753707384666
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": 0.007286811838049356,
          "ci95": [
            -0.0004764853500597742,
            0.015626084770791814
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": 0.0012970946172609077,
          "ci95": [
            -0.008132322690465505,
            0.011354480086816185
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": -0.00047999240694795997,
          "ci95": [
            -0.0005987089987289594,
            -0.0003715359219633379
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": -0.0008890990491213259,
          "ci95": [
            -0.0010231754091820193,
            -0.000739709222674721
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": -0.0008522020254550892,
          "ci95": [
            -0.0009701030662137549,
            -0.0007495581544498777
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": -0.0004091066421733659,
          "ci95": [
            -0.0005670803987777523,
            -0.00026918777182945424
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": -0.0003722096185071293,
          "ci95": [
            -0.0004904962747618105,
            -0.00025937283770712004
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": 3.689702366623665e-05,
          "ci95": [
            -0.00010517416750066748,
            0.00017773926742426532
          ]
        }
      }
    },
    "all_prespecified_effects_involving_candidate": {
      "sa_10__minus__mature_r2": {
        "coefficients": {
          "mature_r2": -1.0,
          "cross8_mature": 0.0,
          "hist_shallow_deep": 0.0,
          "sa_01": 0.0,
          "sa_11": 0.0,
          "sa_10": 1.0,
          "mlp_01": 0.0,
          "mlp_11": 0.0,
          "mlp_10": 0.0,
          "mlp_gate_entry": 0.0,
          "mlp_gate_work": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.001943086064849775,
            "ci95": [
              -0.004995679734683112,
              0.0006071034222691932
            ]
          },
          "auc.is_click": {
            "estimate": 0.0016187510884361211,
            "ci95": [
              0.000928619919922602,
              0.002393087192228682
            ]
          },
          "auc.cart": {
            "estimate": 0.0012951445703341946,
            "ci95": [
              -0.0025723794121350783,
              0.005006301190660905
            ]
          },
          "auc.fav": {
            "estimate": -0.004694572650454254,
            "ci95": [
              -0.011403247152654977,
              0.0011231050358106007
            ]
          },
          "auc.buy": {
            "estimate": -0.005991667267715162,
            "ci95": [
              -0.014350682611751842,
              0.0010857838633049535
            ]
          },
          "bce.is_click": {
            "estimate": -0.0005161083742001304,
            "ci95": [
              -0.0005928819429813894,
              -0.00044728226411067736
            ]
          },
          "bce.cart": {
            "estimate": -3.611596725217048e-05,
            "ci95": [
              -0.00012671992936686528,
              5.6313507448375225e-05
            ]
          },
          "bce.fav": {
            "estimate": 0.00037299067492119545,
            "ci95": [
              0.000245365063584956,
              0.000493529196286127
            ]
          },
          "bce.buy": {
            "estimate": 0.0003360936512549588,
            "ci95": [
              0.0002508564015641738,
              0.00043200248985228115
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.00032360651810192653,
            "ci95": [
              -0.0033568029192969583,
              0.004232153847349371
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.006313323738890375,
            "ci95": [
              0.0003725830345761517,
              0.01307875164899804
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.007610418356151283,
            "ci95": [
              0.0004928078075918164,
              0.01625839103312894
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.005989717220788449,
            "ci95": [
              -0.001044625190786641,
              0.013058753707384666
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.007286811838049356,
            "ci95": [
              -0.0004764853500597742,
              0.015626084770791814
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.0012970946172609077,
            "ci95": [
              -0.008132322690465505,
              0.011354480086816185
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00047999240694795997,
            "ci95": [
              -0.0005987089987289594,
              -0.0003715359219633379
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0008890990491213259,
            "ci95": [
              -0.0010231754091820193,
              -0.000739709222674721
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0008522020254550892,
            "ci95": [
              -0.0009701030662137549,
              -0.0007495581544498777
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.0004091066421733659,
            "ci95": [
              -0.0005670803987777523,
              -0.00026918777182945424
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -0.0003722096185071293,
            "ci95": [
              -0.0004904962747618105,
              -0.00025937283770712004
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 3.689702366623665e-05,
            "ci95": [
              -0.00010517416750066748,
              0.00017773926742426532
            ]
          }
        }
      },
      "sa_10__minus__cross8_mature": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": -1.0,
          "hist_shallow_deep": 0.0,
          "sa_01": 0.0,
          "sa_11": 0.0,
          "sa_10": 1.0,
          "mlp_01": 0.0,
          "mlp_11": 0.0,
          "mlp_10": 0.0,
          "mlp_gate_entry": 0.0,
          "mlp_gate_work": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.005745445154550888,
            "ci95": [
              0.0028001120184552044,
              0.008918758244778483
            ]
          },
          "auc.is_click": {
            "estimate": 0.0016182741464598749,
            "ci95": [
              0.0008519099160027854,
              0.002308196172782234
            ]
          },
          "auc.cart": {
            "estimate": 0.004522679556561937,
            "ci95": [
              0.0008357624617591071,
              0.008807704520678706
            ]
          },
          "auc.fav": {
            "estimate": 0.0012669066971442433,
            "ci95": [
              -0.004631319962957475,
              0.007067073534710935
            ]
          },
          "auc.buy": {
            "estimate": 0.015573920218037496,
            "ci95": [
              0.0063672502320017985,
              0.02639618368874815
            ]
          },
          "bce.is_click": {
            "estimate": -0.00044239995500480744,
            "ci95": [
              -0.0005095600319479602,
              -0.00036536204649969547
            ]
          },
          "bce.cart": {
            "estimate": -4.25732348971683e-05,
            "ci95": [
              -0.00013526993257374176,
              5.7536957044846875e-05
            ]
          },
          "bce.fav": {
            "estimate": 0.0001977691878204818,
            "ci95": [
              9.543925984350153e-05,
              0.0003239245934576721
            ]
          },
          "bce.buy": {
            "estimate": -2.7494740457467345e-05,
            "ci95": [
              -0.00013856868930385606,
              6.234718832071836e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.002904405410102062,
            "ci95": [
              -0.007220286356409792,
              0.0008710894973587759
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.0003513674493156316,
            "ci95": [
              -0.005601803721357812,
              0.00627376685842505
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.013955646071577621,
            "ci95": [
              -0.024995822014366734,
              -0.004743911188715877
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.003255772859417694,
            "ci95": [
              -0.0030016621107337854,
              0.01065703408875961
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.011051240661475559,
            "ci95": [
              -0.022405287432091504,
              -0.0012210097818344643
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.014307013520893253,
            "ci95": [
              -0.026620007605814778,
              -0.00400857609192371
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00039982672010763914,
            "ci95": [
              -0.0005199631558280248,
              -0.0002815396407995831
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0006401691428252893,
            "ci95": [
              -0.0007676114685835026,
              -0.000518774304284981
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0004149052145473401,
            "ci95": [
              -0.000526191759601888,
              -0.00029349551495281927
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.0002403424227176501,
            "ci95": [
              -0.000394632730372503,
              -0.00010405634555672479
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -1.5078494439700954e-05,
            "ci95": [
              -0.00013920914262427678,
              0.0001289973597877638
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.00022526392827794915,
            "ci95": [
              8.575037139423249e-05,
              0.0003874737518115513
            ]
          }
        }
      },
      "sa_01__minus__sa_10": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "hist_shallow_deep": 0.0,
          "sa_01": 1.0,
          "sa_11": 0.0,
          "sa_10": -1.0,
          "mlp_01": 0.0,
          "mlp_11": 0.0,
          "mlp_10": 0.0,
          "mlp_gate_entry": 0.0,
          "mlp_gate_work": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.0005125881192454551,
            "ci95": [
              -0.003604111017406096,
              0.0025060848111952004
            ]
          },
          "auc.is_click": {
            "estimate": -0.0008290958831126405,
            "ci95": [
              -0.0015702551078220412,
              -0.00014090677059970275
            ]
          },
          "auc.cart": {
            "estimate": 0.0001755223351561641,
            "ci95": [
              -0.0045672242037502805,
              0.004821957567228409
            ]
          },
          "auc.fav": {
            "estimate": 0.008364499374789336,
            "ci95": [
              0.0027564265626459335,
              0.014497874454277006
            ]
          },
          "auc.buy": {
            "estimate": -0.00976127830381468,
            "ci95": [
              -0.01862627049532307,
              -0.0012984121860973507
            ]
          },
          "bce.is_click": {
            "estimate": 0.0001621990304898624,
            "ci95": [
              9.502414298944022e-05,
              0.00023193026748016541
            ]
          },
          "bce.cart": {
            "estimate": 4.4553949715316066e-05,
            "ci95": [
              -5.495323070699252e-05,
              0.00015611299212814385
            ]
          },
          "bce.fav": {
            "estimate": -0.00039185793051185747,
            "ci95": [
              -0.0005299976560351365,
              -0.00028217134130264983
            ]
          },
          "bce.buy": {
            "estimate": -0.00013724871627542046,
            "ci95": [
              -0.00022012580892395473,
              -3.903199115165755e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.0010046182182688046,
            "ci95": [
              -0.005627399641748776,
              0.003939149708477088
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.009193595257901976,
            "ci95": [
              -0.015210126536078606,
              -0.003623256734849439
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.00893218242070204,
            "ci95": [
              0.0002124224131615814,
              0.017833655461825394
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.008188977039633172,
            "ci95": [
              -0.015726192846235713,
              -0.0016590942066759603
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.009936800638970844,
            "ci95": [
              0.0006905643821897988,
              0.019502539993886323
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.018125777678604016,
            "ci95": [
              0.007995877042752435,
              0.028687614875462505
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 0.00011764508077454633,
            "ci95": [
              -1.5788962541150427e-05,
              0.00023832819676180986
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.0005540569610017199,
            "ci95": [
              0.00042038783615004437,
              0.0007186967975997108
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.00029944774676528285,
            "ci95": [
              0.00017578495779591876,
              0.00041285748686089446
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.00043641188022717353,
            "ci95": [
              0.00029008499016879485,
              0.0005989872724444439
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00018180266599073652,
            "ci95": [
              5.570168746564489e-05,
              0.00031350783946435147
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.000254609214236437,
            "ci95": [
              -0.00042674401815586684,
              -0.00011786042839606038
            ]
          }
        }
      },
      "sa_11__minus__sa_10": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "hist_shallow_deep": 0.0,
          "sa_01": 0.0,
          "sa_11": 1.0,
          "sa_10": -1.0,
          "mlp_01": 0.0,
          "mlp_11": 0.0,
          "mlp_10": 0.0,
          "mlp_gate_entry": 0.0,
          "mlp_gate_work": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.000198050222240842,
            "ci95": [
              -0.0029956517743005943,
              0.0024809379811390996
            ]
          },
          "auc.is_click": {
            "estimate": -8.578880686016532e-05,
            "ci95": [
              -0.0007876682914566419,
              0.0007474609400853121
            ]
          },
          "auc.cart": {
            "estimate": -0.0043286965603528405,
            "ci95": [
              -0.008581181783373174,
              0.00018258175215026182
            ]
          },
          "auc.fav": {
            "estimate": 0.001162999285691746,
            "ci95": [
              -0.005063141048481412,
              0.007117561699828165
            ]
          },
          "auc.buy": {
            "estimate": 0.002459285192557892,
            "ci95": [
              -0.00574268595179212,
              0.011240263078520153
            ]
          },
          "bce.is_click": {
            "estimate": 2.5524744303950797e-05,
            "ci95": [
              -5.179838102218823e-05,
              9.286912812353355e-05
            ]
          },
          "bce.cart": {
            "estimate": 0.00010917113127398986,
            "ci95": [
              -1.6872413774781168e-06,
              0.00020972162290787806
            ]
          },
          "bce.fav": {
            "estimate": -0.00017898591515345258,
            "ci95": [
              -0.000300293914460981,
              -6.611790772009216e-05
            ]
          },
          "bce.buy": {
            "estimate": -8.452308860927571e-05,
            "ci95": [
              -0.0001845191907543664,
              6.028396800653679e-06
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.004242907753492675,
            "ci95": [
              -0.0004607469854314733,
              0.008670685448211556
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.0012487880925519113,
            "ci95": [
              -0.007131978548188383,
              0.004872776363489715
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.002545073999418057,
            "ci95": [
              -0.01119210000014448,
              0.005576918179976334
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.005491695846044586,
            "ci95": [
              -0.011781997670442131,
              0.002417525221124149
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.006787981752910732,
            "ci95": [
              -0.016014830569126323,
              0.0021688488725017272
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.0012962859068661459,
            "ci95": [
              -0.011760613050505821,
              0.008984619086398749
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -8.364638697003907e-05,
            "ci95": [
              -0.00022734673598112748,
              5.4228925219781137e-05
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.00020451065945740338,
            "ci95": [
              6.562695219353276e-05,
              0.0003533732770660981
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.0001100478329132265,
            "ci95": [
              -3.796959933421084e-06,
              0.00021120828832719152
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.00028815704642744244,
            "ci95": [
              0.00012507592551720039,
              0.00044809951799703294
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00019369421988326557,
            "ci95": [
              5.640163520245715e-05,
              0.0003349957015014367
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -9.446282654417687e-05,
            "ci95": [
              -0.00024568281299814786,
              6.391797176066276e-05
            ]
          }
        }
      },
      "mlp_10__minus__sa_10": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "hist_shallow_deep": 0.0,
          "sa_01": 0.0,
          "sa_11": 0.0,
          "sa_10": -1.0,
          "mlp_01": 0.0,
          "mlp_11": 0.0,
          "mlp_10": 1.0,
          "mlp_gate_entry": 0.0,
          "mlp_gate_work": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.0010297654268190959,
            "ci95": [
              -0.0018140477107920112,
              0.0042790264788722075
            ]
          },
          "auc.is_click": {
            "estimate": -0.002121381664167421,
            "ci95": [
              -0.0028949244624254635,
              -0.0013811892201099577
            ]
          },
          "auc.cart": {
            "estimate": -0.0034918132257787393,
            "ci95": [
              -0.007693514562901999,
              0.0007795241701385342
            ]
          },
          "auc.fav": {
            "estimate": 0.010557857382757607,
            "ci95": [
              0.00583017585678099,
              0.016342146876464975
            ]
          },
          "auc.buy": {
            "estimate": -0.0008256007855350633,
            "ci95": [
              -0.009545580161446202,
              0.007783607666076363
            ]
          },
          "bce.is_click": {
            "estimate": 0.00021308950470660948,
            "ci95": [
              0.00014374501682504066,
              0.00028628403843885956
            ]
          },
          "bce.cart": {
            "estimate": 9.649675952410133e-05,
            "ci95": [
              1.2779071442642706e-06,
              0.00019074792305277938
            ]
          },
          "bce.fav": {
            "estimate": -0.000521036287303038,
            "ci95": [
              -0.0006344857597280202,
              -0.00042693204314159635
            ]
          },
          "bce.buy": {
            "estimate": -0.0001021052022265774,
            "ci95": [
              -0.00020253730212415762,
              -7.658266496029082e-06
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.0013704315616113183,
            "ci95": [
              -0.003151679302951901,
              0.005667663489700816
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.012679239046925028,
            "ci95": [
              -0.01844912741716582,
              -0.007942205764418717
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.0012957808786323577,
            "ci95": [
              -0.009993344502919882,
              0.007550768619570463
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.014049670608536347,
            "ci95": [
              -0.020793475771046347,
              -0.008115683399938452
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.002666212440243676,
            "ci95": [
              -0.011301636417867144,
              0.005661339213130485
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.01138345816829267,
            "ci95": [
              0.0014819204591394186,
              0.021482394447183886
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 0.00011659274518250815,
            "ci95": [
              7.710458894753545e-06,
              0.0002228046339840836
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.0007341257920096475,
            "ci95": [
              0.0006190032039934035,
              0.0008627828135641467
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.0003151947069331869,
            "ci95": [
              0.00020568156493680668,
              0.0004264566056315602
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.0006175330468271394,
            "ci95": [
              0.00048610057424747455,
              0.0007529251001861125
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00019860196175067873,
            "ci95": [
              8.492862678602564e-05,
              0.00031565374437932646
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.0004189310850764606,
            "ci95": [
              -0.0005635620629170467,
              -0.0002910512101097789
            ]
          }
        }
      }
    },
    "parameter_count": 51955556,
    "initializer_sha256": "914ccddd39668aa2cd2e294f413889bc9659d29ba06f8e62a8ea750f46e1e357",
    "optimizer_layout_sha256": "149116671d54002d0f9504ac58802d48fc6ba81ae0917d013337dbf6c14529b1",
    "formal_worker_cost": {
      "wall_seconds": 2742.3903021663427,
      "cpu_seconds": 3266.386917536,
      "peak_rss_bytes": 17142169600,
      "gpu_peak_bytes": 32842448896,
      "samples": 2515
    },
    "smoke_worker_cost": {
      "wall_seconds": 220.2592827528715,
      "cpu_seconds": 248.913234207,
      "peak_rss_bytes": 18462494720,
      "gpu_peak_bytes": 34518073344,
      "samples": 201
    },
    "complete_recorded_costs": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2742.3903021663427,
        "cpu_seconds": 3266.386917536,
        "peak_rss_bytes": 17142169600,
        "gpu_peak_bytes": 32842448896,
        "samples": 2515
      },
      "formal_runner": {
        "wall_seconds": 2737.9571212530136,
        "cpu_seconds": 3139.25531185,
        "peak_process_rss_bytes": 5309968384
      },
      "fit_including_validation": {
        "wall_seconds": 2604.688973426819,
        "cpu_seconds": 2958.6417768439997,
        "peak_process_rss_bytes": 5309968384
      },
      "prediction_export": {
        "wall_seconds": 128.02782264351845,
        "cpu_seconds": 174.90762048299985,
        "peak_process_rss_bytes": 5309968384
      },
      "smoke_worker": {
        "wall_seconds": 220.2592827528715,
        "cpu_seconds": 248.913234207,
        "peak_rss_bytes": 18462494720,
        "gpu_peak_bytes": 34518073344,
        "samples": 201
      },
      "smoke_runner": {
        "wall_seconds": 215.4159775301814,
        "cpu_seconds": 206.778589205,
        "peak_process_rss_bytes": 3581857792
      },
      "latency_constructor": {
        "wall_seconds": 0.7632900476455688,
        "cpu_seconds": 0.8349890309999992,
        "peak_rss_bytes": 2127785984
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.16074050217866898,
        "cpu_seconds": 0.16013383599999997,
        "peak_rss_bytes": 2127785984
      },
      "latency_peak_cuda_bytes": 1439624704,
      "latency_peak_cuda_reserved_bytes": 1728053248
    },
    "forward_latency": {
      "mean_ms": 17.62371997833252,
      "p50_ms": 17.529744148254395,
      "p95_ms": 17.621412849426267,
      "batch_size": 1024,
      "warmups": 10,
      "iterations": 40
    },
    "latency_ratio_to_parent": 1.240588375030564,
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
  "decision_reason": "Macro delta versus A -0.00194309, exploratory CI [-0.00499568,+0.00060710], forward latency ratio 1.240588. Fails both predeclared quality/near-candidate rules; retain all artifacts as negative research evidence, with no claim that every interval establishes harm.",
  "decision_scope": "Root queue disposition only; no released-incumbent promotion or supported lesson.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/sa_10",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/sa_10/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/sa_10/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/sa_10/Taobao_Action/sa_10.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/paired-analysis/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/latency/completion.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/sa_10",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/sa_10/result.json",
    "result_sha256": "be7453154da0c7b3dc778ed637739891b1a5e18831a925470f00b57a2ec3768b",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/sa_10/manifest.json",
    "manifest_sha256": "4060ba58315c4887ee8a02ec6101bee2e6f35c02535c23b7d71c53a3c64fde80",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/sa_10/Taobao_Action/sa_10.model",
    "checkpoint_sha256": "ca995647efb5073565050c42ce1c05906a8cc1b488367fd7621c7a6172646e3e",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/paired-analysis/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
