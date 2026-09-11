## 232. single-history--sa_01--s20262027

- 假设：One attention encoding used only by the second work read can improve retrieval.
- 判定：discard
- 原因：Macro delta versus A -0.00245567, exploratory CI [-0.00498069,-0.00000191], forward latency ratio 1.238432. Fails both predeclared quality/near-candidate rules; retain all artifacts as negative research evidence, with no claim that every interval establishes harm.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "single-history--sa_01--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-10T03:17:15.913140+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "sa_01",
  "code_id": "b973d173c1edbfcedffee2c6ee09f14ebe9b735d9ca52d41c7448e0433b6ab06",
  "hypothesis": "One attention encoding used only by the second work read can improve retrieval.",
  "change": "{\"model\": \"QFormerCross8SingleHistory\", \"maturity_mode\": \"mature\", \"history_encoder_kind\": \"attention\", \"history_read_mode\": \"original_encoded\", \"history_condition\": \"none\", \"history_gate_rank\": 32, \"history_bottleneck\": 128, \"work_steps\": 2, \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8SingleHistory",
    "maturity_mode": "mature",
    "history_encoder_kind": "attention",
    "history_read_mode": "original_encoded",
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
      "value": 0.7646392771307653,
      "delta": -0.00245567418409523
    },
    "per_task": {
      "is_click": {
        "logloss": 0.1840377369240722,
        "AUC": 0.6507259484063129
      },
      "cart": {
        "logloss": 0.021836390999966523,
        "AUC": 0.7939049372628341
      },
      "fav": {
        "logloss": 0.01677929043870635,
        "AUC": 0.8061334170993831
      },
      "buy": {
        "logloss": 0.009842646804300898,
        "AUC": 0.8077928057545309
      }
    },
    "macro_source_logloss": 0.058124016291761484,
    "source_metric_comparison": {
      "is_click": {
        "source_AUC": 0.6507259484063129,
        "computed_AUC": 0.6507259484063129,
        "auc_difference": 0.0,
        "source_logloss": 0.1840377369240722,
        "computed_bce": 0.1840377369240764,
        "bce_difference": 4.218847493575595e-15
      },
      "cart": {
        "source_AUC": 0.7939049372628341,
        "computed_AUC": 0.7939049372628341,
        "auc_difference": 0.0,
        "source_logloss": 0.021836390999966523,
        "computed_bce": 0.021836392351897178,
        "bce_difference": 1.3519306542308929e-09
      },
      "fav": {
        "source_AUC": 0.8061334170993831,
        "computed_AUC": 0.8061334170993831,
        "auc_difference": 0.0,
        "source_logloss": 0.01677929043870635,
        "computed_bce": 0.01677929043876167,
        "bce_difference": 5.532033164890038e-14
      },
      "buy": {
        "source_AUC": 0.8077928057545309,
        "computed_AUC": 0.8077928057545309,
        "auc_difference": 0.0,
        "source_logloss": 0.009842646804300898,
        "computed_bce": 0.009842646871705241,
        "bce_difference": 6.740434252416883e-11
      }
    },
    "matched_parent_effect": {
      "coefficients": {
        "mature_r2": -1.0,
        "cross8_mature": 0.0,
        "hist_shallow_deep": 0.0,
        "sa_01": 1.0,
        "sa_11": 0.0,
        "sa_10": 0.0,
        "mlp_01": 0.0,
        "mlp_11": 0.0,
        "mlp_10": 0.0,
        "mlp_gate_entry": 0.0,
        "mlp_gate_work": 0.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": -0.00245567418409523,
          "ci95": [
            -0.0049806936028239795,
            -1.911620399861819e-06
          ]
        },
        "auc.is_click": {
          "estimate": 0.0007896552053234807,
          "ci95": [
            5.444390671902333e-05,
            0.00144342566026219
          ]
        },
        "auc.cart": {
          "estimate": 0.0014706669054903587,
          "ci95": [
            -0.0029737962776800993,
            0.006327690269823269
          ]
        },
        "auc.fav": {
          "estimate": 0.003669926724335082,
          "ci95": [
            -0.001803054180037372,
            0.009104761215315475
          ]
        },
        "auc.buy": {
          "estimate": -0.01575294557152984,
          "ci95": [
            -0.023729607205891567,
            -0.009112692168921096
          ]
        },
        "bce.is_click": {
          "estimate": -0.00035390934371026805,
          "ci95": [
            -0.0004208096813372815,
            -0.0002840558482685246
          ]
        },
        "bce.cart": {
          "estimate": 8.437982463145588e-06,
          "ci95": [
            -9.731121930040164e-05,
            0.00010990909256489258
          ]
        },
        "bce.fav": {
          "estimate": -1.886725559066202e-05,
          "ci95": [
            -0.00014344297417161452,
            0.00010939741113082136
          ]
        },
        "bce.buy": {
          "estimate": 0.00019884493497953834,
          "ci95": [
            0.00013022877578609324,
            0.00028380873275480104
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": -0.0006810117001668781,
          "ci95": [
            -0.005546920984138636,
            0.0036352450081920317
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": -0.0028802715190116013,
          "ci95": [
            -0.008410684357351433,
            0.0025586413184736003
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": 0.016542600776853322,
          "ci95": [
            0.009792644168326156,
            0.024323426947989803
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": -0.002199259818844723,
          "ci95": [
            -0.009514841509452355,
            0.005112200825267006
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": 0.0172236124770202,
          "ci95": [
            0.008922930121789832,
            0.02677544846250207
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": 0.019422872295864924,
          "ci95": [
            0.01042276936094666,
            0.02933138408145856
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": -0.00036234732617341364,
          "ci95": [
            -0.00048243667487911455,
            -0.00023645646475163004
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": -0.00033504208811960604,
          "ci95": [
            -0.0004940211666343469,
            -0.00017899052548818903
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": -0.0005527542786898064,
          "ci95": [
            -0.0006661733074789005,
            -0.00045701726524581424
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": 2.7305238053807607e-05,
          "ci95": [
            -0.00014124066455410337,
            0.0001990391982358343
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": -0.00019040695251639275,
          "ci95": [
            -0.00032992054607775185,
            -7.106480402962186e-05
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": -0.00021771219057020036,
          "ci95": [
            -0.00036741147783479934,
            -7.483430611033055e-05
          ]
        }
      }
    },
    "all_prespecified_effects_involving_candidate": {
      "sa_01__minus__mature_r2": {
        "coefficients": {
          "mature_r2": -1.0,
          "cross8_mature": 0.0,
          "hist_shallow_deep": 0.0,
          "sa_01": 1.0,
          "sa_11": 0.0,
          "sa_10": 0.0,
          "mlp_01": 0.0,
          "mlp_11": 0.0,
          "mlp_10": 0.0,
          "mlp_gate_entry": 0.0,
          "mlp_gate_work": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.00245567418409523,
            "ci95": [
              -0.0049806936028239795,
              -1.911620399861819e-06
            ]
          },
          "auc.is_click": {
            "estimate": 0.0007896552053234807,
            "ci95": [
              5.444390671902333e-05,
              0.00144342566026219
            ]
          },
          "auc.cart": {
            "estimate": 0.0014706669054903587,
            "ci95": [
              -0.0029737962776800993,
              0.006327690269823269
            ]
          },
          "auc.fav": {
            "estimate": 0.003669926724335082,
            "ci95": [
              -0.001803054180037372,
              0.009104761215315475
            ]
          },
          "auc.buy": {
            "estimate": -0.01575294557152984,
            "ci95": [
              -0.023729607205891567,
              -0.009112692168921096
            ]
          },
          "bce.is_click": {
            "estimate": -0.00035390934371026805,
            "ci95": [
              -0.0004208096813372815,
              -0.0002840558482685246
            ]
          },
          "bce.cart": {
            "estimate": 8.437982463145588e-06,
            "ci95": [
              -9.731121930040164e-05,
              0.00010990909256489258
            ]
          },
          "bce.fav": {
            "estimate": -1.886725559066202e-05,
            "ci95": [
              -0.00014344297417161452,
              0.00010939741113082136
            ]
          },
          "bce.buy": {
            "estimate": 0.00019884493497953834,
            "ci95": [
              0.00013022877578609324,
              0.00028380873275480104
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.0006810117001668781,
            "ci95": [
              -0.005546920984138636,
              0.0036352450081920317
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.0028802715190116013,
            "ci95": [
              -0.008410684357351433,
              0.0025586413184736003
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.016542600776853322,
            "ci95": [
              0.009792644168326156,
              0.024323426947989803
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.002199259818844723,
            "ci95": [
              -0.009514841509452355,
              0.005112200825267006
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.0172236124770202,
            "ci95": [
              0.008922930121789832,
              0.02677544846250207
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.019422872295864924,
            "ci95": [
              0.01042276936094666,
              0.02933138408145856
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00036234732617341364,
            "ci95": [
              -0.00048243667487911455,
              -0.00023645646475163004
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.00033504208811960604,
            "ci95": [
              -0.0004940211666343469,
              -0.00017899052548818903
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0005527542786898064,
            "ci95": [
              -0.0006661733074789005,
              -0.00045701726524581424
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 2.7305238053807607e-05,
            "ci95": [
              -0.00014124066455410337,
              0.0001990391982358343
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -0.00019040695251639275,
            "ci95": [
              -0.00032992054607775185,
              -7.106480402962186e-05
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.00021771219057020036,
            "ci95": [
              -0.00036741147783479934,
              -7.483430611033055e-05
            ]
          }
        }
      },
      "sa_01__minus__cross8_mature": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": -1.0,
          "hist_shallow_deep": 0.0,
          "sa_01": 1.0,
          "sa_11": 0.0,
          "sa_10": 0.0,
          "mlp_01": 0.0,
          "mlp_11": 0.0,
          "mlp_10": 0.0,
          "mlp_gate_entry": 0.0,
          "mlp_gate_work": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.005232857035305433,
            "ci95": [
              0.0022710007569899334,
              0.008424031276797932
            ]
          },
          "auc.is_click": {
            "estimate": 0.0007891782633472344,
            "ci95": [
              1.5212726100513636e-05,
              0.0015357526744236183
            ]
          },
          "auc.cart": {
            "estimate": 0.004698201891718101,
            "ci95": [
              -7.931115566539574e-06,
              0.009051384227350058
            ]
          },
          "auc.fav": {
            "estimate": 0.00963140607193358,
            "ci95": [
              0.0045927848256950765,
              0.015114444470709944
            ]
          },
          "auc.buy": {
            "estimate": 0.005812641914222816,
            "ci95": [
              -0.003967204678009808,
              0.01707908968089488
            ]
          },
          "bce.is_click": {
            "estimate": -0.00028020092451494505,
            "ci95": [
              -0.0003420587040066203,
              -0.0002096071307166138
            ]
          },
          "bce.cart": {
            "estimate": 1.9807148181477663e-06,
            "ci95": [
              -9.297504474880631e-05,
              0.00012069081701324596
            ]
          },
          "bce.fav": {
            "estimate": -0.00019408874269137566,
            "ci95": [
              -0.00029658595074646477,
              -0.00010531415185148167
            ]
          },
          "bce.buy": {
            "estimate": -0.0001647434567328878,
            "ci95": [
              -0.0002667075550311902,
              -6.762666104054555e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.003909023628370867,
            "ci95": [
              -0.008376173042345178,
              0.0007810799778342864
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.008842227808586345,
            "ci95": [
              -0.014474300476760588,
              -0.0038179245126169583
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.0050234636508755814,
            "ci95": [
              -0.016150915576933145,
              0.004945529772350665
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.004933204180215478,
            "ci95": [
              -0.01175265072382673,
              0.001717056486077141
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.0011144400225047146,
            "ci95": [
              -0.012971822731984873,
              0.009526404437321483
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.0038187641577107634,
            "ci95": [
              -0.007808064630099523,
              0.01563106917727362
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.0002821816393330928,
            "ci95": [
              -0.00040763871087652457,
              -0.00014749500831196994
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -8.611218182356939e-05,
            "ci95": [
              -0.0001885763818911556,
              4.485642689799148e-05
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.00011545746778205725,
            "ci95": [
              -0.0002193648868775686,
              1.3050888546708032e-05
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.00019606945750952343,
            "ci95": [
              7.678647512816794e-05,
              0.0003462884874738064
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00016672417155103557,
            "ci95": [
              3.875705883732328e-05,
              0.00032541122990609113
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -2.9345285958487857e-05,
            "ci95": [
              -0.00017325850224680196,
              0.00010759403392450787
            ]
          }
        }
      },
      "sa_01__minus__sa_11": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "hist_shallow_deep": 0.0,
          "sa_01": 1.0,
          "sa_11": -1.0,
          "sa_10": 0.0,
          "mlp_01": 0.0,
          "mlp_11": 0.0,
          "mlp_10": 0.0,
          "mlp_gate_entry": 0.0,
          "mlp_gate_work": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.0003145378970046131,
            "ci95": [
              -0.003233751383252831,
              0.002299064015915359
            ]
          },
          "auc.is_click": {
            "estimate": -0.0007433070762524752,
            "ci95": [
              -0.0015382913742521486,
              -1.6226391335890842e-06
            ]
          },
          "auc.cart": {
            "estimate": 0.0045042188955090046,
            "ci95": [
              -0.0009515905644680716,
              0.009986261106489532
            ]
          },
          "auc.fav": {
            "estimate": 0.00720150008909759,
            "ci95": [
              0.002237398227389914,
              0.012967000698662693
            ]
          },
          "auc.buy": {
            "estimate": -0.012220563496372572,
            "ci95": [
              -0.01986744870577025,
              -0.005547608962381292
            ]
          },
          "bce.is_click": {
            "estimate": 0.0001366742861859116,
            "ci95": [
              6.812486332483172e-05,
              0.00021122016828976908
            ]
          },
          "bce.cart": {
            "estimate": -6.46171815586738e-05,
            "ci95": [
              -0.00020141211109768532,
              6.839309716476551e-05
            ]
          },
          "bce.fav": {
            "estimate": -0.0002128720153584049,
            "ci95": [
              -0.0003146718941688778,
              -0.00012063428645657534
            ]
          },
          "bce.buy": {
            "estimate": -5.272562766614475e-05,
            "ci95": [
              -0.00012534409847333284,
              3.000157132161114e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.00524752597176148,
            "ci95": [
              -0.010887301493673512,
              0.00024221114246300465
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.007944807165350065,
            "ci95": [
              -0.013482030951261015,
              -0.0029399641684413366
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.011477256420120097,
            "ci95": [
              0.0047028054373784685,
              0.019136014635569066
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.0026972811935885854,
            "ci95": [
              -0.009762497629031308,
              0.0043342938568148226
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.016724782391881576,
            "ci95": [
              0.008111033991166344,
              0.02502803627863774
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.019422063585470162,
            "ci95": [
              0.01127091540515538,
              0.02907719961561301
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 0.0002012914677445854,
            "ci95": [
              5.663201538563734e-05,
              0.0003608149114766954
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.0003495463015443165,
            "ci95": [
              0.00023992381003729892,
              0.0004721978612948812
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.00018939991385205635,
            "ci95": [
              8.161797212260999e-05,
              0.0002896485839453875
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.0001482548337997311,
            "ci95": [
              -5.5049233151687785e-06,
              0.0003201720265128377
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -1.189155389252905e-05,
            "ci95": [
              -0.00015352742590120445,
              0.00013960192665324206
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.00016014638769226014,
            "ci95": [
              -0.0002907864814867724,
              -4.509395499812391e-05
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
      "mlp_01__minus__sa_01": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "hist_shallow_deep": 0.0,
          "sa_01": -1.0,
          "sa_11": 0.0,
          "sa_10": 0.0,
          "mlp_01": 1.0,
          "mlp_11": 0.0,
          "mlp_10": 0.0,
          "mlp_gate_entry": 0.0,
          "mlp_gate_work": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.005375741211171542,
            "ci95": [
              -0.008784315141514936,
              -0.0018884602720180641
            ]
          },
          "auc.is_click": {
            "estimate": -0.00027198753506330053,
            "ci95": [
              -0.0010001653482913987,
              0.00048317237006431874
            ]
          },
          "auc.cart": {
            "estimate": -0.0051301667481350455,
            "ci95": [
              -0.010944658599629961,
              0.0006019167470808187
            ]
          },
          "auc.fav": {
            "estimate": -0.016809565741865762,
            "ci95": [
              -0.024833654674131247,
              -0.01054556030164451
            ]
          },
          "auc.buy": {
            "estimate": 0.0007087551803779402,
            "ci95": [
              -0.007652396861067256,
              0.008377591247461907
            ]
          },
          "bce.is_click": {
            "estimate": 0.00022080375563457433,
            "ci95": [
              0.00015081350308480748,
              0.0002908186957663296
            ]
          },
          "bce.cart": {
            "estimate": 6.203083306070833e-05,
            "ci95": [
              -6.93634138343533e-05,
              0.00018512044471420585
            ]
          },
          "bce.fav": {
            "estimate": 0.00027232792972748873,
            "ci95": [
              0.00014847357402145088,
              0.000420031726807469
            ]
          },
          "bce.buy": {
            "estimate": 0.00014315146874477498,
            "ci95": [
              6.049641908111672e-05,
              0.00024258802502128535
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.004858179213071745,
            "ci95": [
              -0.0009801254079504632,
              0.010940547705204133
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.01653757820680246,
            "ci95": [
              0.010234718590511269,
              0.024642137671527373
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.0009807427154412407,
            "ci95": [
              -0.008531308106824861,
              0.007431014624691453
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.011679398993730716,
            "ci95": [
              0.003055930771937715,
              0.021870149956704426
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.005838921928512986,
            "ci95": [
              -0.015355972394066861,
              0.004274135114725462
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.017518320922243702,
            "ci95": [
              -0.028243647668222953,
              -0.005816576449181102
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 0.000158772922573866,
            "ci95": [
              7.890229541159066e-06,
              0.000303554348139498
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -5.1524174092914404e-05,
            "ci95": [
              -0.0002089495603553478,
              7.791111841653823e-05
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 7.765228688979935e-05,
            "ci95": [
              -3.302729237441085e-05,
              0.0001905842982760167
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.0002102970966667804,
            "ci95": [
              -0.0004092425491520199,
              -3.643496421784123e-05
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -8.112063568406665e-05,
            "ci95": [
              -0.0002312801253718388,
              5.9297139596260195e-05
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.00012917646098271375,
            "ci95": [
              -1.3176010963237942e-05,
              0.0002914138690242915
            ]
          }
        }
      },
      "sa_01__minus__hist_shallow_deep": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "hist_shallow_deep": -1.0,
          "sa_01": 1.0,
          "sa_11": 0.0,
          "sa_10": 0.0,
          "mlp_01": 0.0,
          "mlp_11": 0.0,
          "mlp_10": 0.0,
          "mlp_gate_entry": 0.0,
          "mlp_gate_work": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.002744966765059559,
            "ci95": [
              -0.005826776097188106,
              0.0004329421294960064
            ]
          },
          "auc.is_click": {
            "estimate": -0.0025353699089891713,
            "ci95": [
              -0.0032991721664230394,
              -0.0017925481800146416
            ]
          },
          "auc.cart": {
            "estimate": 0.003246986529956919,
            "ci95": [
              -0.001686185737410567,
              0.008306315390076986
            ]
          },
          "auc.fav": {
            "estimate": -0.0005230916653703455,
            "ci95": [
              -0.0063266455513957985,
              0.005435101099286513
            ]
          },
          "auc.buy": {
            "estimate": -0.011168392015835638,
            "ci95": [
              -0.019568065410881276,
              -0.0027441661449611435
            ]
          },
          "bce.is_click": {
            "estimate": 0.0003992871547746357,
            "ci95": [
              0.00033226984359599406,
              0.00047008340083998647
            ]
          },
          "bce.cart": {
            "estimate": 2.4570368373211943e-05,
            "ci95": [
              -6.500162753074641e-05,
              0.00012197783281037225
            ]
          },
          "bce.fav": {
            "estimate": -4.4525175801151534e-05,
            "ci95": [
              -0.00015881856810415353,
              5.387990276144328e-05
            ]
          },
          "bce.buy": {
            "estimate": 0.00012814092243895042,
            "ci95": [
              4.013096529442185e-05,
              0.00021457557626833304
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.00578235643894609,
            "ci95": [
              -0.011270723400563057,
              -0.0009427162977253782
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.002012278243618826,
            "ci95": [
              -0.00781697432420191,
              0.0037013233842085333
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.008633022106846466,
            "ci95": [
              0.0004906470602521445,
              0.01709101506015191
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.0037700781953272644,
            "ci95": [
              -0.004190592438842844,
              0.011104224292728912
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.014415378545792557,
            "ci95": [
              0.0050733685655853684,
              0.023539519213432746
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.010645300350465292,
            "ci95": [
              0.00026773449483427647,
              0.02070928205533722
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 0.0003747167864014238,
            "ci95": [
              0.000257467702731829,
              0.0005064079779942319
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.00044381233057578726,
            "ci95": [
              0.00033135199280604096,
              0.0005881657663126081
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.0002711462323356853,
            "ci95": [
              0.00016656921757098926,
              0.00038259871455982786
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 6.909554417436348e-05,
            "ci95": [
              -5.883546068106539e-05,
              0.000215952868391603
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -0.00010357055406573848,
            "ci95": [
              -0.000221678540581818,
              1.5090670362438584e-05
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.00017266609824010196,
            "ci95": [
              -0.0003115325484879755,
              -4.642986389933274e-05
            ]
          }
        }
      }
    },
    "parameter_count": 51955556,
    "initializer_sha256": "914ccddd39668aa2cd2e294f413889bc9659d29ba06f8e62a8ea750f46e1e357",
    "optimizer_layout_sha256": "ca403e9b939a935015544275cd3c919cecf68e540400b0bc0c8f6f3b5f0a2c87",
    "formal_worker_cost": {
      "wall_seconds": 2735.8964571282268,
      "cpu_seconds": 3266.0868918600004,
      "peak_rss_bytes": 17174323200,
      "gpu_peak_bytes": 32846643200,
      "samples": 2508
    },
    "smoke_worker_cost": {
      "wall_seconds": 214.01883608847857,
      "cpu_seconds": 242.118427847,
      "peak_rss_bytes": 18256908288,
      "gpu_peak_bytes": 34518073344,
      "samples": 195
    },
    "complete_recorded_costs": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2735.8964571282268,
        "cpu_seconds": 3266.0868918600004,
        "peak_rss_bytes": 17174323200,
        "gpu_peak_bytes": 32846643200,
        "samples": 2508
      },
      "formal_runner": {
        "wall_seconds": 2731.286681301892,
        "cpu_seconds": 3137.908126878,
        "peak_process_rss_bytes": 5363322880
      },
      "fit_including_validation": {
        "wall_seconds": 2598.997734941542,
        "cpu_seconds": 2958.509525001,
        "peak_process_rss_bytes": 5363322880
      },
      "prediction_export": {
        "wall_seconds": 126.9584673345089,
        "cpu_seconds": 173.91830803700032,
        "peak_process_rss_bytes": 5363322880
      },
      "smoke_worker": {
        "wall_seconds": 214.01883608847857,
        "cpu_seconds": 242.118427847,
        "peak_rss_bytes": 18256908288,
        "gpu_peak_bytes": 34518073344,
        "samples": 195
      },
      "smoke_runner": {
        "wall_seconds": 209.05336381495,
        "cpu_seconds": 201.525273956,
        "peak_process_rss_bytes": 3588407296
      },
      "latency_constructor": {
        "wall_seconds": 0.7607376798987389,
        "cpu_seconds": 0.8288122829999995,
        "peak_rss_bytes": 2127491072
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.1378684788942337,
        "cpu_seconds": 0.13734789999999997,
        "peak_rss_bytes": 2127491072
      },
      "latency_peak_cuda_bytes": 1439624704,
      "latency_peak_cuda_reserved_bytes": 1728053248
    },
    "forward_latency": {
      "mean_ms": 17.593087100982665,
      "p50_ms": 17.514464378356934,
      "p95_ms": 17.570224285125732,
      "batch_size": 1024,
      "warmups": 10,
      "iterations": 40
    },
    "latency_ratio_to_parent": 1.238432031671688,
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
  "decision_reason": "Macro delta versus A -0.00245567, exploratory CI [-0.00498069,-0.00000191], forward latency ratio 1.238432. Fails both predeclared quality/near-candidate rules; retain all artifacts as negative research evidence, with no claim that every interval establishes harm.",
  "decision_scope": "Root queue disposition only; no released-incumbent promotion or supported lesson.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/sa_01",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/sa_01/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/sa_01/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/sa_01/Taobao_Action/sa_01.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/paired-analysis/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/latency/completion.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/sa_01",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/sa_01/result.json",
    "result_sha256": "db332d367e9e986085b519d592cc1b26b97003e2f043a0952504d90959300095",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/sa_01/manifest.json",
    "manifest_sha256": "e2c740a9395a36c098f6b4e3bc918d2ab42f27c33ac7cbbb268cf7f21fcb5472",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/sa_01/Taobao_Action/sa_01.model",
    "checkpoint_sha256": "eb0b4a16e4c9e93b88c321c43d696b8afb9a492d60242bc9339f55547eeb5f85",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/paired-analysis/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
