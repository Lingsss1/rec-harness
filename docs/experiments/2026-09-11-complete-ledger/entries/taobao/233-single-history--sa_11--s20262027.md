## 233. single-history--sa_11--s20262027

- 假设：One attention encoding reused by both work reads can improve retrieval.
- 判定：discard
- 原因：Macro delta versus A -0.00214114, exploratory CI [-0.00474877,+0.00039578], forward latency ratio 1.233557. Fails both predeclared quality/near-candidate rules; retain all artifacts as negative research evidence, with no claim that every interval establishes harm.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "single-history--sa_11--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-10T03:17:15.913140+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "sa_11",
  "code_id": "b973d173c1edbfcedffee2c6ee09f14ebe9b735d9ca52d41c7448e0433b6ab06",
  "hypothesis": "One attention encoding reused by both work reads can improve retrieval.",
  "change": "{\"model\": \"QFormerCross8SingleHistory\", \"maturity_mode\": \"mature\", \"history_encoder_kind\": \"attention\", \"history_read_mode\": \"encoded_encoded\", \"history_condition\": \"none\", \"history_gate_rank\": 32, \"history_bottleneck\": 128, \"work_steps\": 2, \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8SingleHistory",
    "maturity_mode": "mature",
    "history_encoder_kind": "attention",
    "history_read_mode": "encoded_encoded",
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
      "value": 0.7649538150277698,
      "delta": -0.002141136287090617
    },
    "per_task": {
      "is_click": {
        "logloss": 0.18390106263788591,
        "AUC": 0.6514692554825654
      },
      "cart": {
        "logloss": 0.021901009515363046,
        "AUC": 0.7894007183673251
      },
      "fav": {
        "logloss": 0.016992162442040375,
        "AUC": 0.7989319170102855
      },
      "buy": {
        "logloss": 0.009895372498689726,
        "AUC": 0.8200133692509035
      }
    },
    "macro_source_logloss": 0.05817240177349477,
    "source_metric_comparison": {
      "is_click": {
        "source_AUC": 0.6514692554825654,
        "computed_AUC": 0.6514692554825654,
        "auc_difference": 0.0,
        "source_logloss": 0.18390106263788591,
        "computed_bce": 0.1839010626378905,
        "bce_difference": 4.579669976578771e-15
      },
      "cart": {
        "source_AUC": 0.7894007183673251,
        "computed_AUC": 0.7894007183673251,
        "auc_difference": 0.0,
        "source_logloss": 0.021901009515363046,
        "computed_bce": 0.02190100953345585,
        "bce_difference": 1.8092805031955095e-11
      },
      "fav": {
        "source_AUC": 0.7989319170102855,
        "computed_AUC": 0.7989319170102855,
        "auc_difference": 0.0,
        "source_logloss": 0.016992162442040375,
        "computed_bce": 0.016992162454120077,
        "bce_difference": 1.207970182215412e-11
      },
      "buy": {
        "source_AUC": 0.8200133692509035,
        "computed_AUC": 0.8200133692509035,
        "auc_difference": 0.0,
        "source_logloss": 0.009895372498689726,
        "computed_bce": 0.009895372499371386,
        "bce_difference": 6.816595898850863e-13
      }
    },
    "matched_parent_effect": {
      "coefficients": {
        "mature_r2": -1.0,
        "cross8_mature": 0.0,
        "hist_shallow_deep": 0.0,
        "sa_01": 0.0,
        "sa_11": 1.0,
        "sa_10": 0.0,
        "mlp_01": 0.0,
        "mlp_11": 0.0,
        "mlp_10": 0.0,
        "mlp_gate_entry": 0.0,
        "mlp_gate_work": 0.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": -0.002141136287090617,
          "ci95": [
            -0.004748774418284264,
            0.0003957834683623955
          ]
        },
        "auc.is_click": {
          "estimate": 0.0015329622815759558,
          "ci95": [
            0.0007856161390927985,
            0.002369683691692895
          ]
        },
        "auc.cart": {
          "estimate": -0.003033551990018646,
          "ci95": [
            -0.007414883872256001,
            0.001411689685284139
          ]
        },
        "auc.fav": {
          "estimate": -0.003531573364762508,
          "ci95": [
            -0.009779864157262255,
            0.0014511111725892328
          ]
        },
        "auc.buy": {
          "estimate": -0.00353238207515727,
          "ci95": [
            -0.011199533058628156,
            0.0035297284955575926
          ]
        },
        "bce.is_click": {
          "estimate": -0.0004905836298961797,
          "ci95": [
            -0.0005694154729602864,
            -0.00041683560702148874
          ]
        },
        "bce.cart": {
          "estimate": 7.305516402181939e-05,
          "ci95": [
            -4.610998781394643e-05,
            0.0001778893734852349
          ]
        },
        "bce.fav": {
          "estimate": 0.00019400475976774287,
          "ci95": [
            5.4713731293158824e-05,
            0.00033244063665461426
          ]
        },
        "bce.buy": {
          "estimate": 0.0002515705626456831,
          "ci95": [
            0.0001719203561297815,
            0.00033457577693545373
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": 0.004566514271594602,
          "ci95": [
            1.1505228586719695e-05,
            0.009083055348150503
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": 0.005064535646338464,
          "ci95": [
            7.390361036865308e-05,
            0.011644403600313245
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": 0.005065344356733226,
          "ci95": [
            -0.0020743049808024435,
            0.012359028362426671
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": 0.0004980213747438622,
          "ci95": [
            -0.005903051179200169,
            0.008612639416644401
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": 0.000498830085138624,
          "ci95": [
            -0.006909420982979538,
            0.00861738487232233
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": 8.087103947618246e-07,
          "ci95": [
            -0.008864324243575994,
            0.009053236964539874
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": -0.000563638793917999,
          "ci95": [
            -0.0007040388325204121,
            -0.00042658701907719607
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": -0.0006845883896639225,
          "ci95": [
            -0.0008408038114085355,
            -0.0005312950102602086
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": -0.0007421541925418627,
          "ci95": [
            -0.0008534382070641104,
            -0.0006395429703614549
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": -0.00012094959574592348,
          "ci95": [
            -0.00029369228485426,
            4.9947664326608665e-05
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": -0.0001785153986238637,
          "ci95": [
            -0.00032109381797190624,
            -5.215956386088802e-05
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": -5.756580287794022e-05,
          "ci95": [
            -0.00021948209609185792,
            0.00010315677271721825
          ]
        }
      }
    },
    "all_prespecified_effects_involving_candidate": {
      "sa_11__minus__mature_r2": {
        "coefficients": {
          "mature_r2": -1.0,
          "cross8_mature": 0.0,
          "hist_shallow_deep": 0.0,
          "sa_01": 0.0,
          "sa_11": 1.0,
          "sa_10": 0.0,
          "mlp_01": 0.0,
          "mlp_11": 0.0,
          "mlp_10": 0.0,
          "mlp_gate_entry": 0.0,
          "mlp_gate_work": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.002141136287090617,
            "ci95": [
              -0.004748774418284264,
              0.0003957834683623955
            ]
          },
          "auc.is_click": {
            "estimate": 0.0015329622815759558,
            "ci95": [
              0.0007856161390927985,
              0.002369683691692895
            ]
          },
          "auc.cart": {
            "estimate": -0.003033551990018646,
            "ci95": [
              -0.007414883872256001,
              0.001411689685284139
            ]
          },
          "auc.fav": {
            "estimate": -0.003531573364762508,
            "ci95": [
              -0.009779864157262255,
              0.0014511111725892328
            ]
          },
          "auc.buy": {
            "estimate": -0.00353238207515727,
            "ci95": [
              -0.011199533058628156,
              0.0035297284955575926
            ]
          },
          "bce.is_click": {
            "estimate": -0.0004905836298961797,
            "ci95": [
              -0.0005694154729602864,
              -0.00041683560702148874
            ]
          },
          "bce.cart": {
            "estimate": 7.305516402181939e-05,
            "ci95": [
              -4.610998781394643e-05,
              0.0001778893734852349
            ]
          },
          "bce.fav": {
            "estimate": 0.00019400475976774287,
            "ci95": [
              5.4713731293158824e-05,
              0.00033244063665461426
            ]
          },
          "bce.buy": {
            "estimate": 0.0002515705626456831,
            "ci95": [
              0.0001719203561297815,
              0.00033457577693545373
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.004566514271594602,
            "ci95": [
              1.1505228586719695e-05,
              0.009083055348150503
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.005064535646338464,
            "ci95": [
              7.390361036865308e-05,
              0.011644403600313245
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.005065344356733226,
            "ci95": [
              -0.0020743049808024435,
              0.012359028362426671
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.0004980213747438622,
            "ci95": [
              -0.005903051179200169,
              0.008612639416644401
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.000498830085138624,
            "ci95": [
              -0.006909420982979538,
              0.00861738487232233
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 8.087103947618246e-07,
            "ci95": [
              -0.008864324243575994,
              0.009053236964539874
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.000563638793917999,
            "ci95": [
              -0.0007040388325204121,
              -0.00042658701907719607
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0006845883896639225,
            "ci95": [
              -0.0008408038114085355,
              -0.0005312950102602086
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0007421541925418627,
            "ci95": [
              -0.0008534382070641104,
              -0.0006395429703614549
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.00012094959574592348,
            "ci95": [
              -0.00029369228485426,
              4.9947664326608665e-05
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -0.0001785153986238637,
            "ci95": [
              -0.00032109381797190624,
              -5.215956386088802e-05
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -5.756580287794022e-05,
            "ci95": [
              -0.00021948209609185792,
              0.00010315677271721825
            ]
          }
        }
      },
      "sa_11__minus__cross8_mature": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": -1.0,
          "hist_shallow_deep": 0.0,
          "sa_01": 0.0,
          "sa_11": 1.0,
          "sa_10": 0.0,
          "mlp_01": 0.0,
          "mlp_11": 0.0,
          "mlp_10": 0.0,
          "mlp_gate_entry": 0.0,
          "mlp_gate_work": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.005547394932310046,
            "ci95": [
              0.0024601535110210363,
              0.008815920472683607
            ]
          },
          "auc.is_click": {
            "estimate": 0.0015324853395997096,
            "ci95": [
              0.0007615957649366678,
              0.0022751805650235313
            ]
          },
          "auc.cart": {
            "estimate": 0.00019398299620909665,
            "ci95": [
              -0.0047671248984405504,
              0.005318454868301517
            ]
          },
          "auc.fav": {
            "estimate": 0.0024299059828359892,
            "ci95": [
              -0.0036149752294927034,
              0.007425642466019578
            ]
          },
          "auc.buy": {
            "estimate": 0.018033205410595388,
            "ci95": [
              0.0078758949354002,
              0.028898364701433425
            ]
          },
          "bce.is_click": {
            "estimate": -0.00041687521070085665,
            "ci95": [
              -0.000485479720272005,
              -0.0003488862187697964
            ]
          },
          "bce.cart": {
            "estimate": 6.659789637682156e-05,
            "ci95": [
              -5.6811538744854814e-05,
              0.00018768383111814792
            ]
          },
          "bce.fav": {
            "estimate": 1.8783272667029227e-05,
            "ci95": [
              -7.671411862272435e-05,
              0.00012135939289263537
            ]
          },
          "bce.buy": {
            "estimate": -0.00011201782906674305,
            "ci95": [
              -0.00023137317214289818,
              -1.0168604401371599e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.001338502343390613,
            "ci95": [
              -0.003929216858993853,
              0.006364776325438497
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.0008974206432362797,
            "ci95": [
              -0.006062611814882263,
              0.0050629840719497825
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.016500720070995678,
            "ci95": [
              -0.027787980470560435,
              -0.006473594966432478
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.0022359229866268926,
            "ci95": [
              -0.009285006395556544,
              0.005509712998382559
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.01783922241438629,
            "ci95": [
              -0.030424910731689254,
              -0.007240733775280701
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.015603299427759398,
            "ci95": [
              -0.028492574220894254,
              -0.003830262771841433
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.0004834731070776782,
            "ci95": [
              -0.0006209948138591519,
              -0.0003521900981351213
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0004356584833678859,
            "ci95": [
              -0.0005499042843758058,
              -0.0003104030261550811
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0003048573816341136,
            "ci95": [
              -0.00043069283717151624,
              -0.00016739723949350035
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 4.7814623709792337e-05,
            "ci95": [
              -0.0001262074978405251,
              0.00021149457313088341
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00017861572544356462,
            "ci95": [
              3.1279987170778705e-05,
              0.0003453274902166583
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.00013080110173377228,
            "ci95": [
              -1.2965937011737971e-05,
              0.0002771665454468236
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
      "mlp_11__minus__sa_11": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "hist_shallow_deep": 0.0,
          "sa_01": 0.0,
          "sa_11": -1.0,
          "sa_10": 0.0,
          "mlp_01": 0.0,
          "mlp_11": 1.0,
          "mlp_10": 0.0,
          "mlp_gate_entry": 0.0,
          "mlp_gate_work": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.0035978447772007927,
            "ci95": [
              -0.007051491063451549,
              -0.0004772533767752726
            ]
          },
          "auc.is_click": {
            "estimate": -0.002418330769243915,
            "ci95": [
              -0.0032764072400742,
              -0.0016784018968553666
            ]
          },
          "auc.cart": {
            "estimate": -0.001477188529559359,
            "ci95": [
              -0.007719818752913754,
              0.006447994996723896
            ]
          },
          "auc.fav": {
            "estimate": -0.0016826966008586242,
            "ci95": [
              -0.006571215296611907,
              0.0036846152522629527
            ]
          },
          "auc.buy": {
            "estimate": -0.008813163209141273,
            "ci95": [
              -0.018797869252858177,
              9.401203905996156e-05
            ]
          },
          "bce.is_click": {
            "estimate": 0.0007711476778981263,
            "ci95": [
              0.0006980151998741857,
              0.0008521867109647161
            ]
          },
          "bce.cart": {
            "estimate": 5.518719911001488e-05,
            "ci95": [
              -0.00014106099832231558,
              0.00021468395707505093
            ]
          },
          "bce.fav": {
            "estimate": -2.4546104378509598e-05,
            "ci95": [
              -0.0001237235100414554,
              7.65058178728975e-05
            ]
          },
          "bce.buy": {
            "estimate": -5.5299378008598354e-05,
            "ci95": [
              -0.00015719265969270612,
              6.002653858371422e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.0009411422396845559,
            "ci95": [
              -0.00921741608365791,
              0.005069488512182729
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.0007356341683852907,
            "ci95": [
              -0.006373218391023735,
              0.004366526351170969
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.006394832439897358,
            "ci95": [
              -0.002272906294228979,
              0.01669662187114619
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.00020550807129926518,
            "ci95": [
              -0.008416890072741106,
              0.010395764770273255
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.007335974679581914,
            "ci95": [
              -0.003414734642438563,
              0.020035283018242166
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.0071304666082826484,
            "ci95": [
              -0.0033032774130745284,
              0.018132723195167913
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 0.0007159604787881114,
            "ci95": [
              0.0005377891152610657,
              0.0009247227205686557
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.0007956937822766359,
            "ci95": [
              0.000667904764456865,
              0.0009198827885790898
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.0008264470559067247,
            "ci95": [
              0.0006869218665191455,
              0.0009472563685083985
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 7.973330348852448e-05,
            "ci95": [
              -0.0001702479466914668,
              0.00027043537672280824
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00011048657711861323,
            "ci95": [
              -0.00012758553694330955,
              0.00030131501740280405
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 3.0753273630088757e-05,
            "ci95": [
              -0.00011096851591588441,
              0.0001843843125587606
            ]
          }
        }
      }
    },
    "parameter_count": 51955556,
    "initializer_sha256": "914ccddd39668aa2cd2e294f413889bc9659d29ba06f8e62a8ea750f46e1e357",
    "optimizer_layout_sha256": "1fc58ffd5118989bf46475c8a81a4ad16487bf3cbaa2a8e4b3c6413f953a1be3",
    "formal_worker_cost": {
      "wall_seconds": 2729.3299817740917,
      "cpu_seconds": 3254.001422899,
      "peak_rss_bytes": 17137303552,
      "gpu_peak_bytes": 32838254592,
      "samples": 2501
    },
    "smoke_worker_cost": {
      "wall_seconds": 220.34590608626604,
      "cpu_seconds": 247.291189484,
      "peak_rss_bytes": 18178170880,
      "gpu_peak_bytes": 34518073344,
      "samples": 201
    },
    "complete_recorded_costs": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2729.3299817740917,
        "cpu_seconds": 3254.001422899,
        "peak_rss_bytes": 17137303552,
        "gpu_peak_bytes": 32838254592,
        "samples": 2501
      },
      "formal_runner": {
        "wall_seconds": 2725.120545618236,
        "cpu_seconds": 3126.140330797,
        "peak_process_rss_bytes": 5348847616
      },
      "fit_including_validation": {
        "wall_seconds": 2593.0873567461967,
        "cpu_seconds": 2946.3063217139998,
        "peak_process_rss_bytes": 5348847616
      },
      "prediction_export": {
        "wall_seconds": 126.71776077896357,
        "cpu_seconds": 174.3501731309998,
        "peak_process_rss_bytes": 5348847616
      },
      "smoke_worker": {
        "wall_seconds": 220.34590608626604,
        "cpu_seconds": 247.291189484,
        "peak_rss_bytes": 18178170880,
        "gpu_peak_bytes": 34518073344,
        "samples": 201
      },
      "smoke_runner": {
        "wall_seconds": 215.0680452734232,
        "cpu_seconds": 206.74445259200002,
        "peak_process_rss_bytes": 3583287296
      },
      "latency_constructor": {
        "wall_seconds": 0.7746501937508583,
        "cpu_seconds": 0.845911490999999,
        "peak_rss_bytes": 2127491072
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.1796056553721428,
        "cpu_seconds": 0.17891683599999908,
        "peak_rss_bytes": 2127491072
      },
      "latency_peak_cuda_bytes": 1439624704,
      "latency_peak_cuda_reserved_bytes": 1728053248
    },
    "forward_latency": {
      "mean_ms": 17.523835945129395,
      "p50_ms": 17.507344245910645,
      "p95_ms": 17.59444055557251,
      "batch_size": 1024,
      "warmups": 10,
      "iterations": 40
    },
    "latency_ratio_to_parent": 1.2335572277702063,
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
  "decision_reason": "Macro delta versus A -0.00214114, exploratory CI [-0.00474877,+0.00039578], forward latency ratio 1.233557. Fails both predeclared quality/near-candidate rules; retain all artifacts as negative research evidence, with no claim that every interval establishes harm.",
  "decision_scope": "Root queue disposition only; no released-incumbent promotion or supported lesson.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/sa_11",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/sa_11/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/sa_11/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/sa_11/Taobao_Action/sa_11.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/paired-analysis/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/latency/completion.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/sa_11",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/sa_11/result.json",
    "result_sha256": "0b1781fe1ff9f34d32330c04c4056ac27af276a68bc61f7c305417aaadb20761",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/sa_11/manifest.json",
    "manifest_sha256": "6a32be1ebc5f489454f2bd2365208092978805263f7220f734117eb2b9948849",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/sa_11/Taobao_Action/sa_11.model",
    "checkpoint_sha256": "ca1a9c1ba8417eeb654b7c1d9e6b455c4f53b211ae9db100b3c6e4791499f409",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/paired-analysis/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
