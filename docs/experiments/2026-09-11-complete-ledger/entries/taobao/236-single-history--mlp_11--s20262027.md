## 236. single-history--mlp_11--s20262027

- 假设：Both work reads can benefit from one tokenwise history encoding.
- 判定：discard
- 原因：Macro delta versus A -0.00573898, exploratory CI [-0.00935142,-0.00235989], forward latency ratio 1.173186. Fails both predeclared quality/near-candidate rules; retain all artifacts as negative research evidence, with no claim that every interval establishes harm.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "single-history--mlp_11--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-10T03:17:15.913140+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "mlp_11",
  "code_id": "b973d173c1edbfcedffee2c6ee09f14ebe9b735d9ca52d41c7448e0433b6ab06",
  "hypothesis": "Both work reads can benefit from one tokenwise history encoding.",
  "change": "{\"model\": \"QFormerCross8SingleHistory\", \"maturity_mode\": \"mature\", \"history_encoder_kind\": \"mlp\", \"history_read_mode\": \"encoded_encoded\", \"history_condition\": \"none\", \"history_gate_rank\": 32, \"history_bottleneck\": 128, \"work_steps\": 2, \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8SingleHistory",
    "maturity_mode": "mature",
    "history_encoder_kind": "mlp",
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
      "value": 0.7613559702505691,
      "delta": -0.00573898106429141
    },
    "per_task": {
      "is_click": {
        "logloss": 0.18467221031578515,
        "AUC": 0.6490509247133215
      },
      "cart": {
        "logloss": 0.02195619673049702,
        "AUC": 0.7879235298377657
      },
      "fav": {
        "logloss": 0.016967616349056018,
        "AUC": 0.7972492204094269
      },
      "buy": {
        "logloss": 0.009840073114780462,
        "AUC": 0.8112002060417622
      }
    },
    "macro_source_logloss": 0.05835902412752966,
    "source_metric_comparison": {
      "is_click": {
        "source_AUC": 0.6490509247133215,
        "computed_AUC": 0.6490509247133215,
        "auc_difference": 0.0,
        "source_logloss": 0.18467221031578515,
        "computed_bce": 0.18467221031578862,
        "bce_difference": 3.469446951953614e-15
      },
      "cart": {
        "source_AUC": 0.7879235298377657,
        "computed_AUC": 0.7879235298377657,
        "auc_difference": 0.0,
        "source_logloss": 0.02195619673049702,
        "computed_bce": 0.021956196732565866,
        "bce_difference": 2.068845095237748e-12
      },
      "fav": {
        "source_AUC": 0.7972492204094269,
        "computed_AUC": 0.7972492204094269,
        "auc_difference": 0.0,
        "source_logloss": 0.016967616349056018,
        "computed_bce": 0.016967616349741567,
        "bce_difference": 6.855488399182263e-13
      },
      "buy": {
        "source_AUC": 0.8112002060417622,
        "computed_AUC": 0.8112002060417622,
        "auc_difference": 0.0,
        "source_logloss": 0.009840073114780462,
        "computed_bce": 0.009840073121362787,
        "bce_difference": 6.582324962867148e-12
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
        "mlp_11": 1.0,
        "mlp_10": 0.0,
        "mlp_gate_entry": 0.0,
        "mlp_gate_work": 0.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": -0.00573898106429141,
          "ci95": [
            -0.009351415282375233,
            -0.00235989370915753
          ]
        },
        "auc.is_click": {
          "estimate": -0.0008853684876679591,
          "ci95": [
            -0.0015560968159453264,
            -0.00017716388165066053
          ]
        },
        "auc.cart": {
          "estimate": -0.004510740519578005,
          "ci95": [
            -0.010803629312245407,
            0.004618767840415348
          ]
        },
        "auc.fav": {
          "estimate": -0.005214269965621132,
          "ci95": [
            -0.010234587126603542,
            -0.00038276261850238335
          ]
        },
        "auc.buy": {
          "estimate": -0.012345545284298542,
          "ci95": [
            -0.022556193588327916,
            -0.00407966023932215
          ]
        },
        "bce.is_click": {
          "estimate": 0.0002805640480019467,
          "ci95": [
            0.00020984211112292582,
            0.00034429062241918273
          ]
        },
        "bce.cart": {
          "estimate": 0.00012824236313183426,
          "ci95": [
            -5.7852550123781416e-05,
            0.00025783029176996606
          ]
        },
        "bce.fav": {
          "estimate": 0.00016945865538923327,
          "ci95": [
            5.945110990691876e-05,
            0.00027847071205931533
          ]
        },
        "bce.buy": {
          "estimate": 0.00019627118463708473,
          "ci95": [
            0.00010171679310164439,
            0.00031097519715330465
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": 0.0036253720319100458,
          "ci95": [
            -0.00540682067667598,
            0.009787615268707795
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": 0.004328901477953173,
          "ci95": [
            -0.00046800837012776205,
            0.009543397570437524
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": 0.011460176796630583,
          "ci95": [
            0.0032704446972284308,
            0.021446688573589525
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": 0.0007035294460431274,
          "ci95": [
            -0.007018721285604959,
            0.010150320792396453
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": 0.007834804764720538,
          "ci95": [
            -0.0027068625428454834,
            0.021087819766063713
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": 0.00713127531867741,
          "ci95": [
            -0.0025064734171619473,
            0.018018137228644013
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": 0.0001523216848701124,
          "ci95": [
            1.1381663510778019e-06,
            0.0003516668729911426
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": 0.0001111053926127134,
          "ci95": [
            -2.7998614309254505e-05,
            0.00024026335199484131
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": 8.429286336486194e-05,
          "ci95": [
            -5.094391869921253e-05,
            0.00020232943558402296
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": -4.1216292257399006e-05,
          "ci95": [
            -0.00025829578962417764,
            0.00013059662839866754
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": -6.802882150525047e-05,
          "ci95": [
            -0.0002972372122071984,
            9.777197426070836e-05
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": -2.6812529247851463e-05,
          "ci95": [
            -0.00019365438173712523,
            0.00011402394055856058
          ]
        }
      }
    },
    "all_prespecified_effects_involving_candidate": {
      "mlp_11__minus__mature_r2": {
        "coefficients": {
          "mature_r2": -1.0,
          "cross8_mature": 0.0,
          "hist_shallow_deep": 0.0,
          "sa_01": 0.0,
          "sa_11": 0.0,
          "sa_10": 0.0,
          "mlp_01": 0.0,
          "mlp_11": 1.0,
          "mlp_10": 0.0,
          "mlp_gate_entry": 0.0,
          "mlp_gate_work": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.00573898106429141,
            "ci95": [
              -0.009351415282375233,
              -0.00235989370915753
            ]
          },
          "auc.is_click": {
            "estimate": -0.0008853684876679591,
            "ci95": [
              -0.0015560968159453264,
              -0.00017716388165066053
            ]
          },
          "auc.cart": {
            "estimate": -0.004510740519578005,
            "ci95": [
              -0.010803629312245407,
              0.004618767840415348
            ]
          },
          "auc.fav": {
            "estimate": -0.005214269965621132,
            "ci95": [
              -0.010234587126603542,
              -0.00038276261850238335
            ]
          },
          "auc.buy": {
            "estimate": -0.012345545284298542,
            "ci95": [
              -0.022556193588327916,
              -0.00407966023932215
            ]
          },
          "bce.is_click": {
            "estimate": 0.0002805640480019467,
            "ci95": [
              0.00020984211112292582,
              0.00034429062241918273
            ]
          },
          "bce.cart": {
            "estimate": 0.00012824236313183426,
            "ci95": [
              -5.7852550123781416e-05,
              0.00025783029176996606
            ]
          },
          "bce.fav": {
            "estimate": 0.00016945865538923327,
            "ci95": [
              5.945110990691876e-05,
              0.00027847071205931533
            ]
          },
          "bce.buy": {
            "estimate": 0.00019627118463708473,
            "ci95": [
              0.00010171679310164439,
              0.00031097519715330465
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.0036253720319100458,
            "ci95": [
              -0.00540682067667598,
              0.009787615268707795
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.004328901477953173,
            "ci95": [
              -0.00046800837012776205,
              0.009543397570437524
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.011460176796630583,
            "ci95": [
              0.0032704446972284308,
              0.021446688573589525
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.0007035294460431274,
            "ci95": [
              -0.007018721285604959,
              0.010150320792396453
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.007834804764720538,
            "ci95": [
              -0.0027068625428454834,
              0.021087819766063713
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.00713127531867741,
            "ci95": [
              -0.0025064734171619473,
              0.018018137228644013
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 0.0001523216848701124,
            "ci95": [
              1.1381663510778019e-06,
              0.0003516668729911426
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.0001111053926127134,
            "ci95": [
              -2.7998614309254505e-05,
              0.00024026335199484131
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 8.429286336486194e-05,
            "ci95": [
              -5.094391869921253e-05,
              0.00020232943558402296
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -4.1216292257399006e-05,
            "ci95": [
              -0.00025829578962417764,
              0.00013059662839866754
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -6.802882150525047e-05,
            "ci95": [
              -0.0002972372122071984,
              9.777197426070836e-05
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -2.6812529247851463e-05,
            "ci95": [
              -0.00019365438173712523,
              0.00011402394055856058
            ]
          }
        }
      },
      "mlp_11__minus__cross8_mature": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": -1.0,
          "hist_shallow_deep": 0.0,
          "sa_01": 0.0,
          "sa_11": 0.0,
          "sa_10": 0.0,
          "mlp_01": 0.0,
          "mlp_11": 1.0,
          "mlp_10": 0.0,
          "mlp_gate_entry": 0.0,
          "mlp_gate_work": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.0019495501551092531,
            "ci95": [
              -0.0008373022862368988,
              0.005171741249464526
            ]
          },
          "auc.is_click": {
            "estimate": -0.0008858454296442053,
            "ci95": [
              -0.00161425710556925,
              -0.00017229084872483248
            ]
          },
          "auc.cart": {
            "estimate": -0.0012832055333502623,
            "ci95": [
              -0.006939233556494264,
              0.0053677690456436404
            ]
          },
          "auc.fav": {
            "estimate": 0.0007472093819773651,
            "ci95": [
              -0.004730421347717087,
              0.006369631516560384
            ]
          },
          "auc.buy": {
            "estimate": 0.009220042201454115,
            "ci95": [
              0.000905124635678781,
              0.01671360684356826
            ]
          },
          "bce.is_click": {
            "estimate": 0.0003542724671972697,
            "ci95": [
              0.0002912322081865232,
              0.0004244963397385455
            ]
          },
          "bce.cart": {
            "estimate": 0.00012178509548683644,
            "ci95": [
              -2.1668915400862174e-05,
              0.00023896998121697148
            ]
          },
          "bce.fav": {
            "estimate": -5.762831711480371e-06,
            "ci95": [
              -9.877245029200144e-05,
              9.113192395316407e-05
            ]
          },
          "bce.buy": {
            "estimate": -0.0001673172070753414,
            "ci95": [
              -0.00026122697824534056,
              -6.45693769388415e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.000397360103706057,
            "ci95": [
              -0.006519286300875526,
              0.006076506103714382
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.0016330548116215704,
            "ci95": [
              -0.007293970933509544,
              0.0035859236807418452
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.01010588763109832,
            "ci95": [
              -0.01774282116483928,
              -0.0016674099571820154
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.0020304149153276274,
            "ci95": [
              -0.009683641996512971,
              0.00610217154509628
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.010503247734804377,
            "ci95": [
              -0.019499303743714703,
              -0.0008004523177488348
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.00847283281947675,
            "ci95": [
              -0.017470753621089453,
              0.0007790421892359098
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 0.00023248737171043324,
            "ci95": [
              0.00010100971581896478,
              0.0004021865039812813
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.00036003529890875005,
            "ci95": [
              0.00024039473226112945,
              0.00048282894678822153
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.0005215896742726111,
            "ci95": [
              0.0004049209651050582,
              0.0006392689466643915
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.0001275479271983168,
            "ci95": [
              -3.36265191918708e-05,
              0.0002785584967018111
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00028910230256217785,
            "ci95": [
              0.0001309057197892278,
              0.00043018322506291595
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.00016155437536386104,
            "ci95": [
              2.7954794004966248e-05,
              0.00030504095209282637
            ]
          }
        }
      },
      "mlp_01__minus__mlp_11": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "hist_shallow_deep": 0.0,
          "sa_01": 0.0,
          "sa_11": 0.0,
          "sa_10": 0.0,
          "mlp_01": 1.0,
          "mlp_11": -1.0,
          "mlp_10": 0.0,
          "mlp_gate_entry": 0.0,
          "mlp_gate_work": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.0020924343309753624,
            "ci95": [
              -0.004915025330534822,
              0.0009517862544734652
            ]
          },
          "auc.is_click": {
            "estimate": 0.0014030361579281392,
            "ci95": [
              0.0007082867415154154,
              0.0021466377535018836
            ]
          },
          "auc.cart": {
            "estimate": 0.0008512406769333181,
            "ci95": [
              -0.005908737261633418,
              0.005938112575855043
            ]
          },
          "auc.fav": {
            "estimate": -0.007925369051909548,
            "ci95": [
              -0.0162629339363369,
              -0.0013855967259000213
            ]
          },
          "auc.buy": {
            "estimate": -0.002698645106853359,
            "ci95": [
              -0.009472032064439176,
              0.004570724869015737
            ]
          },
          "bce.is_click": {
            "estimate": -0.0004136696360776404,
            "ci95": [
              -0.00048102539533239686,
              -0.0003468383925282599
            ]
          },
          "bce.cart": {
            "estimate": -5.777354760798034e-05,
            "ci95": [
              -0.0001761127334036842,
              9.548998847274617e-05
            ]
          },
          "bce.fav": {
            "estimate": 8.400201874759344e-05,
            "ci95": [
              -3.601760592264269e-05,
              0.00022166356752871425
            ]
          },
          "bce.buy": {
            "estimate": 0.0001457252190872286,
            "ci95": [
              5.762270561459806e-05,
              0.00023613846097909923
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.0005517954809948211,
            "ci95": [
              -0.004724194563851125,
              0.00743528881576625
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.009328405209837687,
            "ci95": [
              0.0028363353846223376,
              0.018044276235494396
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.004101681264781498,
            "ci95": [
              -0.0030613201130107485,
              0.010771266439695856
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.008776609728842866,
            "ci95": [
              -0.0007143255978902694,
              0.017752199747423005
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.003549885783786677,
            "ci95": [
              -0.006337782851438508,
              0.0121993358434633
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.005226723945056189,
            "ci95": [
              -0.015671152470908525,
              0.0048337759168773095
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00035589608846966006,
            "ci95": [
              -0.0005299529274476037,
              -0.00022591858936693088
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0004976716548252338,
            "ci95": [
              -0.0006511141429239638,
              -0.0003587770393942843
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.000559394855164869,
            "ci95": [
              -0.0006696587497717459,
              -0.00045274194710230613
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.00014177556635557378,
            "ci95": [
              -0.00031547583036323586,
              6.323074670958115e-05
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -0.00020349876669520893,
            "ci95": [
              -0.0003434531636595136,
              -2.5056776167304882e-05
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -6.172320033963515e-05,
            "ci95": [
              -0.00020797796916228946,
              0.00011891248955842315
            ]
          }
        }
      },
      "mlp_11__minus__mlp_10": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "hist_shallow_deep": 0.0,
          "sa_01": 0.0,
          "sa_11": 0.0,
          "sa_10": 0.0,
          "mlp_01": 0.0,
          "mlp_11": 1.0,
          "mlp_10": -1.0,
          "mlp_gate_entry": 0.0,
          "mlp_gate_work": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.0048256604262607306,
            "ci95": [
              -0.007839924084461242,
              -0.002106650357904917
            ]
          },
          "auc.is_click": {
            "estimate": -0.0003827379119366592,
            "ci95": [
              -0.001136849570193754,
              0.0003951986865801416
            ]
          },
          "auc.cart": {
            "estimate": -0.00231407186413346,
            "ci95": [
              -0.00810841598204494,
              0.004115737333532977
            ]
          },
          "auc.fav": {
            "estimate": -0.011077554697924485,
            "ci95": [
              -0.016571954951406075,
              -0.0061102708521373825
            ]
          },
          "auc.buy": {
            "estimate": -0.0055282772310483175,
            "ci95": [
              -0.013130794902894732,
              0.0016502126741922305
            ]
          },
          "bce.is_click": {
            "estimate": 0.0005835829174954676,
            "ci95": [
              0.0005056017394035606,
              0.0006538186972995782
            ]
          },
          "bce.cart": {
            "estimate": 6.786157085990341e-05,
            "ci95": [
              -0.00010387668528140017,
              0.0001963514533168259
            ]
          },
          "bce.fav": {
            "estimate": 0.00031750426777107585,
            "ci95": [
              0.00022094410873972408,
              0.00041420444668984635
            ]
          },
          "bce.buy": {
            "estimate": -3.771726439129666e-05,
            "ci95": [
              -0.0001203676721977998,
              5.7056409249723815e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.001931333952196801,
            "ci95": [
              -0.004687646139970522,
              0.007957319434776075
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.010694816785987826,
            "ci95": [
              0.005744285260121889,
              0.01623567249915322
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.005145539319111658,
            "ci95": [
              -0.001967552682076984,
              0.012843394272513164
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.008763482833791025,
            "ci95": [
              0.0014032160910614794,
              0.017536072150783368
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.0032142053669148574,
            "ci95": [
              -0.006494953658041347,
              0.013130125722304433
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.005549277466876168,
            "ci95": [
              -0.014031014421850849,
              0.0032394642547407117
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 0.0005157213466355642,
            "ci95": [
              0.0003591499835684861,
              0.0006935386830282656
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.0002660786497243918,
            "ci95": [
              0.0001466757792351301,
              0.0003797008349025667
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.0006213001818867643,
            "ci95": [
              0.0004922040887278058,
              0.0007384400907175275
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.00024964269691117244,
            "ci95": [
              -0.0004222332670796464,
              -9.171436980228181e-05
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00010557883525120007,
            "ci95": [
              -8.43042583667596e-05,
              0.0002528942722370478
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.0003552215321623725,
            "ci95": [
              0.00022940274110223418,
              0.0004762356176280529
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
    "parameter_count": 51889252,
    "initializer_sha256": "cf9e6537a9f480ff9c67b09844c15127bea4f97481dc4376de42d84fe1a66e2d",
    "optimizer_layout_sha256": "93e9fe870123ef3c133b1b8a5786d21db31cd1fb788b5b2fe03f3ae1f40d08a7",
    "formal_worker_cost": {
      "wall_seconds": 2688.214631251991,
      "cpu_seconds": 3214.7056538939996,
      "peak_rss_bytes": 17146290176,
      "gpu_peak_bytes": 31579963392,
      "samples": 2463
    },
    "smoke_worker_cost": {
      "wall_seconds": 330.0408606007695,
      "cpu_seconds": 648.563692039,
      "peak_rss_bytes": 21133541376,
      "gpu_peak_bytes": 34520170496,
      "samples": 302
    },
    "complete_recorded_costs": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2688.214631251991,
        "cpu_seconds": 3214.7056538939996,
        "peak_rss_bytes": 17146290176,
        "gpu_peak_bytes": 31579963392,
        "samples": 2463
      },
      "formal_runner": {
        "wall_seconds": 2683.2708602547646,
        "cpu_seconds": 3086.7875545469997,
        "peak_process_rss_bytes": 5415366656
      },
      "fit_including_validation": {
        "wall_seconds": 2552.6138797700405,
        "cpu_seconds": 2907.9745994659997,
        "peak_process_rss_bytes": 5415366656
      },
      "prediction_export": {
        "wall_seconds": 125.24789342284203,
        "cpu_seconds": 173.1782290279998,
        "peak_process_rss_bytes": 5415366656
      },
      "smoke_worker": {
        "wall_seconds": 330.0408606007695,
        "cpu_seconds": 648.563692039,
        "peak_rss_bytes": 21133541376,
        "gpu_peak_bytes": 34520170496,
        "samples": 302
      },
      "smoke_runner": {
        "wall_seconds": 324.9010729715228,
        "cpu_seconds": 273.638816288,
        "peak_process_rss_bytes": 3967148032
      },
      "latency_constructor": {
        "wall_seconds": 0.757634311914444,
        "cpu_seconds": 0.828613777000001,
        "peak_rss_bytes": 2127785984
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.15426567941904068,
        "cpu_seconds": 0.1536313800000002,
        "peak_rss_bytes": 2127785984
      },
      "latency_peak_cuda_bytes": 1517870080,
      "latency_peak_cuda_reserved_bytes": 1885339648
    },
    "forward_latency": {
      "mean_ms": 16.666205644607544,
      "p50_ms": 16.654288291931152,
      "p95_ms": 16.695143508911134,
      "batch_size": 1024,
      "warmups": 10,
      "iterations": 40
    },
    "latency_ratio_to_parent": 1.1731859677746166,
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
  "decision_reason": "Macro delta versus A -0.00573898, exploratory CI [-0.00935142,-0.00235989], forward latency ratio 1.173186. Fails both predeclared quality/near-candidate rules; retain all artifacts as negative research evidence, with no claim that every interval establishes harm.",
  "decision_scope": "Root queue disposition only; no released-incumbent promotion or supported lesson.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_11",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_11/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_11/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_11/Taobao_Action/mlp_11.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/paired-analysis/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/latency/completion.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_11",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_11/result.json",
    "result_sha256": "f7051c61f2a06393d1a619eb07a33c98b2901831245ed6c254aafbdfa3f0b8fc",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_11/manifest.json",
    "manifest_sha256": "17ba6cfcb9e6357e819a6325cefd99324660fbd4ff8433c27e05b4f826e053fe",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_11/Taobao_Action/mlp_11.model",
    "checkpoint_sha256": "289c1aff2e74f2ccade2ce1f36e20cd6444177d3ccf80b37b7c428b22f8d7390",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/paired-analysis/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
