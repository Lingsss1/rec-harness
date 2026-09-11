## 237. single-history--mlp_10--s20262027

- 假设：Encoding the first work history only can improve the representation used to form the second query.
- 判定：discard
- 原因：Macro delta versus A -0.00091332, exploratory CI [-0.00361977,+0.00169206], forward latency ratio 1.172513. Fails both predeclared quality/near-candidate rules; retain all artifacts as negative research evidence, with no claim that every interval establishes harm.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "single-history--mlp_10--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-10T03:17:15.913140+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "mlp_10",
  "code_id": "b973d173c1edbfcedffee2c6ee09f14ebe9b735d9ca52d41c7448e0433b6ab06",
  "hypothesis": "Encoding the first work history only can improve the representation used to form the second query.",
  "change": "{\"model\": \"QFormerCross8SingleHistory\", \"maturity_mode\": \"mature\", \"history_encoder_kind\": \"mlp\", \"history_read_mode\": \"encoded_original\", \"history_condition\": \"none\", \"history_gate_rank\": 32, \"history_bottleneck\": 128, \"work_steps\": 2, \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8SingleHistory",
    "maturity_mode": "mature",
    "history_encoder_kind": "mlp",
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
      "value": 0.7661816306768298,
      "delta": -0.0009133206380306791
    },
    "per_task": {
      "is_click": {
        "logloss": 0.1840886273982876,
        "AUC": 0.6494336626252581
      },
      "cart": {
        "logloss": 0.021888526906206466,
        "AUC": 0.7902376017018992
      },
      "fav": {
        "logloss": 0.016650111613982593,
        "AUC": 0.8083267751073514
      },
      "buy": {
        "logloss": 0.009877790336358,
        "AUC": 0.8167284832728106
      }
    },
    "macro_source_logloss": 0.05812626406370866,
    "source_metric_comparison": {
      "is_click": {
        "source_AUC": 0.6494336626252581,
        "computed_AUC": 0.6494336626252581,
        "auc_difference": 0.0,
        "source_logloss": 0.1840886273982876,
        "computed_bce": 0.18408862739829315,
        "bce_difference": 5.551115123125783e-15
      },
      "cart": {
        "source_AUC": 0.7902376017018992,
        "computed_AUC": 0.7902376017018992,
        "auc_difference": 0.0,
        "source_logloss": 0.021888526906206466,
        "computed_bce": 0.021888335161705963,
        "bce_difference": -1.9174450050299385e-07
      },
      "fav": {
        "source_AUC": 0.8083267751073514,
        "computed_AUC": 0.8083267751073514,
        "auc_difference": 0.0,
        "source_logloss": 0.016650111613982593,
        "computed_bce": 0.01665011208197049,
        "bce_difference": 4.679878984337194e-10
      },
      "buy": {
        "source_AUC": 0.8167284832728106,
        "computed_AUC": 0.8167284832728106,
        "auc_difference": 0.0,
        "source_logloss": 0.009877790336358,
        "computed_bce": 0.009877790385754084,
        "bce_difference": 4.939608444498589e-11
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
        "mlp_10": 1.0,
        "mlp_gate_entry": 0.0,
        "mlp_gate_work": 0.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": -0.0009133206380306791,
          "ci95": [
            -0.0036197733930555588,
            0.0016920571023525668
          ]
        },
        "auc.is_click": {
          "estimate": -0.0005026305757312999,
          "ci95": [
            -0.0012518809588639656,
            0.00017620866639518604
          ]
        },
        "auc.cart": {
          "estimate": -0.0021966686554445447,
          "ci95": [
            -0.006581459784431756,
            0.0026646904940161656
          ]
        },
        "auc.fav": {
          "estimate": 0.005863284732303353,
          "ci95": [
            0.00024996284172323934,
            0.011934704277209711
          ]
        },
        "auc.buy": {
          "estimate": -0.006817268053250225,
          "ci95": [
            -0.01430453946392737,
            0.00012075945081626602
          ]
        },
        "bce.is_click": {
          "estimate": -0.00030301886949352097,
          "ci95": [
            -0.0003787169094019979,
            -0.00023166113266649834
          ]
        },
        "bce.cart": {
          "estimate": 6.0380792271930855e-05,
          "ci95": [
            -5.008759751197328e-05,
            0.00015865421556008374
          ]
        },
        "bce.fav": {
          "estimate": -0.00014804561238184258,
          "ci95": [
            -0.00028174357022508513,
            -2.600949504118394e-05
          ]
        },
        "bce.buy": {
          "estimate": 0.0002339884490283814,
          "ci95": [
            0.00015411887124063372,
            0.00031095068423781034
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": 0.0016940380797132448,
          "ci95": [
            -0.003177204158008748,
            0.006154769718194983
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": -0.006365915308034653,
          "ci95": [
            -0.012519108579088048,
            -0.0006463001391672703
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": 0.006314637477518925,
          "ci95": [
            -0.0004608885613443691,
            0.013577781462701177
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": -0.008059953387747898,
          "ci95": [
            -0.014832407487532956,
            -0.0007717257092207105
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": 0.00462059939780568,
          "ci95": [
            -0.0034619143435197478,
            0.013428769213476622
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": 0.012680552785553578,
          "ci95": [
            0.004230473980356489,
            0.0226623906826638
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": -0.0003633996617654518,
          "ci95": [
            -0.00048291097510985546,
            -0.0002338322274563547
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": -0.0001549732571116784,
          "ci95": [
            -0.00030162309398520124,
            1.1296103337427584e-05
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": -0.0005370073185219024,
          "ci95": [
            -0.0006619664133343759,
            -0.00042953015360806416
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": 0.00020842640465377343,
          "ci95": [
            5.6205691751185564e-05,
            0.0003541574330990235
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": -0.00017360765675645054,
          "ci95": [
            -0.0003029940569458594,
            -5.456563767083345e-05
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": -0.000382034061410224,
          "ci95": [
            -0.0005437853989233209,
            -0.00024116841922187989
          ]
        }
      }
    },
    "all_prespecified_effects_involving_candidate": {
      "mlp_10__minus__mature_r2": {
        "coefficients": {
          "mature_r2": -1.0,
          "cross8_mature": 0.0,
          "hist_shallow_deep": 0.0,
          "sa_01": 0.0,
          "sa_11": 0.0,
          "sa_10": 0.0,
          "mlp_01": 0.0,
          "mlp_11": 0.0,
          "mlp_10": 1.0,
          "mlp_gate_entry": 0.0,
          "mlp_gate_work": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.0009133206380306791,
            "ci95": [
              -0.0036197733930555588,
              0.0016920571023525668
            ]
          },
          "auc.is_click": {
            "estimate": -0.0005026305757312999,
            "ci95": [
              -0.0012518809588639656,
              0.00017620866639518604
            ]
          },
          "auc.cart": {
            "estimate": -0.0021966686554445447,
            "ci95": [
              -0.006581459784431756,
              0.0026646904940161656
            ]
          },
          "auc.fav": {
            "estimate": 0.005863284732303353,
            "ci95": [
              0.00024996284172323934,
              0.011934704277209711
            ]
          },
          "auc.buy": {
            "estimate": -0.006817268053250225,
            "ci95": [
              -0.01430453946392737,
              0.00012075945081626602
            ]
          },
          "bce.is_click": {
            "estimate": -0.00030301886949352097,
            "ci95": [
              -0.0003787169094019979,
              -0.00023166113266649834
            ]
          },
          "bce.cart": {
            "estimate": 6.0380792271930855e-05,
            "ci95": [
              -5.008759751197328e-05,
              0.00015865421556008374
            ]
          },
          "bce.fav": {
            "estimate": -0.00014804561238184258,
            "ci95": [
              -0.00028174357022508513,
              -2.600949504118394e-05
            ]
          },
          "bce.buy": {
            "estimate": 0.0002339884490283814,
            "ci95": [
              0.00015411887124063372,
              0.00031095068423781034
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.0016940380797132448,
            "ci95": [
              -0.003177204158008748,
              0.006154769718194983
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.006365915308034653,
            "ci95": [
              -0.012519108579088048,
              -0.0006463001391672703
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.006314637477518925,
            "ci95": [
              -0.0004608885613443691,
              0.013577781462701177
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.008059953387747898,
            "ci95": [
              -0.014832407487532956,
              -0.0007717257092207105
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.00462059939780568,
            "ci95": [
              -0.0034619143435197478,
              0.013428769213476622
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.012680552785553578,
            "ci95": [
              0.004230473980356489,
              0.0226623906826638
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.0003633996617654518,
            "ci95": [
              -0.00048291097510985546,
              -0.0002338322274563547
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0001549732571116784,
            "ci95": [
              -0.00030162309398520124,
              1.1296103337427584e-05
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0005370073185219024,
            "ci95": [
              -0.0006619664133343759,
              -0.00042953015360806416
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.00020842640465377343,
            "ci95": [
              5.6205691751185564e-05,
              0.0003541574330990235
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -0.00017360765675645054,
            "ci95": [
              -0.0003029940569458594,
              -5.456563767083345e-05
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.000382034061410224,
            "ci95": [
              -0.0005437853989233209,
              -0.00024116841922187989
            ]
          }
        }
      },
      "mlp_10__minus__cross8_mature": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": -1.0,
          "hist_shallow_deep": 0.0,
          "sa_01": 0.0,
          "sa_11": 0.0,
          "sa_10": 0.0,
          "mlp_01": 0.0,
          "mlp_11": 0.0,
          "mlp_10": 1.0,
          "mlp_gate_entry": 0.0,
          "mlp_gate_work": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.006775210581369984,
            "ci95": [
              0.004113021397935619,
              0.00965495564262214
            ]
          },
          "auc.is_click": {
            "estimate": -0.0005031075177075461,
            "ci95": [
              -0.0013303651868290327,
              0.00021290662189753188
            ]
          },
          "auc.cart": {
            "estimate": 0.0010308663307831978,
            "ci95": [
              -0.0027752125861386923,
              0.005165981351090069
            ]
          },
          "auc.fav": {
            "estimate": 0.01182476407990185,
            "ci95": [
              0.007191687865690785,
              0.017560759613660777
            ]
          },
          "auc.buy": {
            "estimate": 0.014748319432502432,
            "ci95": [
              0.006648319762328536,
              0.023302584206819913
            ]
          },
          "bce.is_click": {
            "estimate": -0.00022931045029819797,
            "ci95": [
              -0.0003086662608786761,
              -0.0001487633434391463
            ]
          },
          "bce.cart": {
            "estimate": 5.3923524626933034e-05,
            "ci95": [
              -3.4953479667709004e-05,
              0.0001436864813053402
            ]
          },
          "bce.fav": {
            "estimate": -0.0003232670994825562,
            "ci95": [
              -0.0004164885261003284,
              -0.00023703053375500322
            ]
          },
          "bce.buy": {
            "estimate": -0.00012959994268404475,
            "ci95": [
              -0.00022858294355799475,
              -4.867648019682269e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.001533973848490744,
            "ci95": [
              -0.0058165769472309755,
              0.0022912185990042228
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.012327871597609397,
            "ci95": [
              -0.018274873264074985,
              -0.007723874986526169
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.015251426950209979,
            "ci95": [
              -0.023838052417616044,
              -0.007218088449527521
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.010793897749118653,
            "ci95": [
              -0.017704228272428887,
              -0.005043992749903203
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.013717453101719235,
            "ci95": [
              -0.02383383540911172,
              -0.004703233229035933
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.002923555352600582,
            "ci95": [
              -0.014096632416341072,
              0.0058647792552925645
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.000283233974925131,
            "ci95": [
              -0.00039847981403755934,
              -0.00016035590899857803
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 9.395664918435825e-05,
            "ci95": [
              -1.675937870841562e-05,
              0.00020818723358865726
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -9.971050761415322e-05,
            "ci95": [
              -0.00021284047584269674,
              2.415925953411991e-05
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.00037719062410948925,
            "ci95": [
              0.0002579897891343496,
              0.0005022412272275963
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00018352346731097778,
            "ci95": [
              6.799402677822685e-05,
              0.00031669839824345077
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.00019366715679851147,
            "ci95": [
              -0.0003246237543504631,
              -5.464756628466723e-05
            ]
          }
        }
      },
      "mlp_01__minus__mlp_10": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "hist_shallow_deep": 0.0,
          "sa_01": 0.0,
          "sa_11": 0.0,
          "sa_10": 0.0,
          "mlp_01": 1.0,
          "mlp_11": 0.0,
          "mlp_10": -1.0,
          "mlp_gate_entry": 0.0,
          "mlp_gate_work": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.006918094757236093,
            "ci95": [
              -0.010399855220696287,
              -0.0037754379272514453
            ]
          },
          "auc.is_click": {
            "estimate": 0.00102029824599148,
            "ci95": [
              0.00030572448854299664,
              0.0017853362398295074
            ]
          },
          "auc.cart": {
            "estimate": -0.001462831187200142,
            "ci95": [
              -0.007005124997039572,
              0.004122946665746142
            ]
          },
          "auc.fav": {
            "estimate": -0.019002923749834033,
            "ci95": [
              -0.027081609062925983,
              -0.012262203869184272
            ]
          },
          "auc.buy": {
            "estimate": -0.008226922337901676,
            "ci95": [
              -0.01679541347276941,
              -0.0002226376389596533
            ]
          },
          "bce.is_click": {
            "estimate": 0.00016991328141782724,
            "ci95": [
              8.758848567468892e-05,
              0.0002377730684070281
            ]
          },
          "bce.cart": {
            "estimate": 1.0088023251923067e-05,
            "ci95": [
              -0.00010778435827082454,
              0.000126293042995277
            ]
          },
          "bce.fav": {
            "estimate": 0.0004015062865186693,
            "ci95": [
              0.0002772108384360942,
              0.0005461757571506764
            ]
          },
          "bce.buy": {
            "estimate": 0.00010800795469593193,
            "ci95": [
              1.133703535508941e-05,
              0.00020986589486299414
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.002483129433191622,
            "ci95": [
              -0.003160312042324481,
              0.008016982947216002
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.020023221995825513,
            "ci95": [
              0.013105365390266096,
              0.028130563937554226
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.009247220583893156,
            "ci95": [
              0.0013864463982830454,
              0.018194673848964595
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.01754009256263389,
            "ci95": [
              0.009205680810293188,
              0.02771897779811186
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.0067640911507015344,
            "ci95": [
              -0.002398377785561603,
              0.016409605561336218
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.010776001411932357,
            "ci95": [
              -0.021588146897715755,
              0.00012974093377309415
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 0.00015982525816590418,
            "ci95": [
              1.0082622139010772e-05,
              0.0002931151451037398
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.00023159300510084205,
            "ci95": [
              -0.00039820707313950355,
              -8.930762368686675e-05
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 6.190532672189532e-05,
            "ci95": [
              -6.994735265678123e-05,
              0.00017780328690139086
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.0003914182632667462,
            "ci95": [
              -0.0005683625427467943,
              -0.00022695678219243924
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -9.791993144400886e-05,
            "ci95": [
              -0.00024591605599027095,
              4.565167221723697e-05
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.00029349833182273737,
            "ci95": [
              0.0001445912427513088,
              0.00045095399373996437
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
    "parameter_count": 51889252,
    "initializer_sha256": "cf9e6537a9f480ff9c67b09844c15127bea4f97481dc4376de42d84fe1a66e2d",
    "optimizer_layout_sha256": "9be815f0b608107fca4efcd2bc26679635c815701fdcdc91677e76bbccd87089",
    "formal_worker_cost": {
      "wall_seconds": 2686.6765558496118,
      "cpu_seconds": 3200.8075648920003,
      "peak_rss_bytes": 17117454336,
      "gpu_peak_bytes": 31582060544,
      "samples": 2465
    },
    "smoke_worker_cost": {
      "wall_seconds": 331.70812125504017,
      "cpu_seconds": 640.181904754,
      "peak_rss_bytes": 21112852480,
      "gpu_peak_bytes": 34520170496,
      "samples": 303
    },
    "complete_recorded_costs": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2686.6765558496118,
        "cpu_seconds": 3200.8075648920003,
        "peak_rss_bytes": 17117454336,
        "gpu_peak_bytes": 31582060544,
        "samples": 2465
      },
      "formal_runner": {
        "wall_seconds": 2681.9731072708964,
        "cpu_seconds": 3074.678935057,
        "peak_process_rss_bytes": 5262561280
      },
      "fit_including_validation": {
        "wall_seconds": 2550.924990721047,
        "cpu_seconds": 2895.549258471,
        "peak_process_rss_bytes": 5262561280
      },
      "prediction_export": {
        "wall_seconds": 125.67101988941431,
        "cpu_seconds": 173.53620460400043,
        "peak_process_rss_bytes": 5262561280
      },
      "smoke_worker": {
        "wall_seconds": 331.70812125504017,
        "cpu_seconds": 640.181904754,
        "peak_rss_bytes": 21112852480,
        "gpu_peak_bytes": 34520170496,
        "samples": 303
      },
      "smoke_runner": {
        "wall_seconds": 326.35462345182896,
        "cpu_seconds": 275.634385866,
        "peak_process_rss_bytes": 3954733056
      },
      "latency_constructor": {
        "wall_seconds": 0.7541157677769661,
        "cpu_seconds": 0.8249897829999995,
        "peak_rss_bytes": 2127785984
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.14292560517787933,
        "cpu_seconds": 0.14234471199999987,
        "peak_rss_bytes": 2127785984
      },
      "latency_peak_cuda_bytes": 1517870080,
      "latency_peak_cuda_reserved_bytes": 1885339648
    },
    "forward_latency": {
      "mean_ms": 16.656649589538574,
      "p50_ms": 16.649503707885742,
      "p95_ms": 16.70556526184082,
      "batch_size": 1024,
      "warmups": 10,
      "iterations": 40
    },
    "latency_ratio_to_parent": 1.172513287384535,
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
  "decision_reason": "Macro delta versus A -0.00091332, exploratory CI [-0.00361977,+0.00169206], forward latency ratio 1.172513. Fails both predeclared quality/near-candidate rules; retain all artifacts as negative research evidence, with no claim that every interval establishes harm.",
  "decision_scope": "Root queue disposition only; no released-incumbent promotion or supported lesson.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_10",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_10/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_10/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_10/Taobao_Action/mlp_10.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/paired-analysis/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/latency/completion.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_10",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_10/result.json",
    "result_sha256": "864fbf969dd257c882ad1b0077dc28da62fec41ff6f1dc362a1edb659c0b91c9",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_10/manifest.json",
    "manifest_sha256": "0f21c1acb407f2364d5537753a3deb5993f60b4e826465119e81ace6a5bfb191",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_10/Taobao_Action/mlp_10.model",
    "checkpoint_sha256": "ac2a39f94f0152dfb5b45e0644645acadb5ac8af72e1f4008ebd37e6da62666d",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/paired-analysis/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
