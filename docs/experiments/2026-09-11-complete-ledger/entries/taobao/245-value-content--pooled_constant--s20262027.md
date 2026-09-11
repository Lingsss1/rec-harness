## 245. value-content--pooled_constant--s20262027

- 假设：Second-work value supplementation; pooled_constant conditional nonlinear content before/after original weighted aggregation.
- 判定：discard
- 原因：Macro delta -0.004223502, CI95 [-0.007324589522790518, -0.000923203284268388], forward ratio 0.999204; misses both frozen quality and near policy.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "value-content--pooled_constant--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-10T06:16:19.753374+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "pooled_constant",
  "code_id": "24e06fb09c6658ef53f419e5fa410cfa79eb699cc16354bfd31a299d62180a70",
  "hypothesis": "Second-work value supplementation; pooled_constant conditional nonlinear content before/after original weighted aggregation.",
  "change": "{\"model\": \"QFormerCross8ValueContent\", \"maturity_mode\": \"mature\", \"value_position\": \"pooled\", \"value_condition\": \"constant\", \"value_rank\": 16, \"work_steps\": 2, \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8ValueContent",
    "maturity_mode": "mature",
    "value_position": "pooled",
    "value_condition": "constant",
    "value_rank": 16,
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
      "value": 0.7628714495781016,
      "delta": -0.0042235017367587835
    },
    "per_task": {
      "is_click": {
        "logloss": 0.18428657428720363,
        "AUC": 0.6489787208110775
      },
      "cart": {
        "logloss": 0.022008886261591337,
        "AUC": 0.790233717516209
      },
      "fav": {
        "logloss": 0.016844224273651093,
        "AUC": 0.8040165117492968
      },
      "buy": {
        "logloss": 0.009847404142656805,
        "AUC": 0.8082568482358236
      }
    },
    "macro_source_logloss": 0.05824677224127572,
    "source_metric_comparison": {
      "is_click": {
        "source_AUC": 0.6489787208110775,
        "computed_AUC": 0.6489787208110775,
        "auc_difference": 0.0,
        "source_logloss": 0.18428657428720363,
        "computed_bce": 0.18428657428720718,
        "bce_difference": 3.552713678800501e-15
      },
      "cart": {
        "source_AUC": 0.790233717516209,
        "computed_AUC": 0.790233717516209,
        "auc_difference": 0.0,
        "source_logloss": 0.022008886261591337,
        "computed_bce": 0.02200888749592824,
        "bce_difference": 1.2343369035217044e-09
      },
      "fav": {
        "source_AUC": 0.8040165117492968,
        "computed_AUC": 0.8040165117492968,
        "auc_difference": 0.0,
        "source_logloss": 0.016844224273651093,
        "computed_bce": 0.01684422638045741,
        "bce_difference": 2.1068063162099637e-09
      },
      "buy": {
        "source_AUC": 0.8082568482358236,
        "computed_AUC": 0.8082568482358236,
        "auc_difference": 0.0,
        "source_logloss": 0.009847404142656805,
        "computed_bce": 0.009847404538619122,
        "bce_difference": 3.959623167543169e-10
      }
    },
    "matched_parent_effect": {
      "coefficients": {
        "mature_r2": -1.0,
        "cross8_mature": 0.0,
        "event_entry": 0.0,
        "event_work": 0.0,
        "event_constant": 0.0,
        "pooled_entry": 0.0,
        "pooled_work": 0.0,
        "pooled_constant": 1.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": -0.0042235017367587835,
          "ci95": [
            -0.007324589522790518,
            -0.000923203284268388
          ]
        },
        "auc.is_click": {
          "estimate": -0.0009575723899118982,
          "ci95": [
            -0.0016996682132955603,
            -5.824352269378363e-05
          ]
        },
        "auc.cart": {
          "estimate": -0.0022005528411347397,
          "ci95": [
            -0.0067203211105046155,
            0.0031123360640100666
          ]
        },
        "auc.fav": {
          "estimate": 0.0015530213742487309,
          "ci95": [
            -0.004372308328031699,
            0.008007065189874141
          ]
        },
        "auc.buy": {
          "estimate": -0.015288903090237227,
          "ci95": [
            -0.025911671875335555,
            -0.006000429044110378
          ]
        },
        "bce.is_click": {
          "estimate": -0.00010507198057949085,
          "ci95": [
            -0.00018305323023582246,
            -4.152960773948364e-05
          ]
        },
        "bce.cart": {
          "estimate": 0.00018093312649420865,
          "ci95": [
            5.975293180209207e-05,
            0.0002869935987907534
          ]
        },
        "bce.fav": {
          "estimate": 4.6068686105075335e-05,
          "ci95": [
            -8.56122682329357e-05,
            0.00018332917948443063
          ]
        },
        "bce.buy": {
          "estimate": 0.00020360260189341943,
          "ci95": [
            0.00011711356401445366,
            0.0003014812962696167
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": 0.0012429804512228415,
          "ci95": [
            -0.004130188242823102,
            0.005894418522516549
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": -0.002510593764160629,
          "ci95": [
            -0.008910900097575692,
            0.0034872381423807866
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": 0.014331330700325329,
          "ci95": [
            0.005114553947637776,
            0.025138900143680608
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": -0.0037535742153834706,
          "ci95": [
            -0.011641050886091334,
            0.004609549484416847
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": 0.013088350249102487,
          "ci95": [
            0.0030524125045211102,
            0.024522003815249275
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": 0.016841924464485958,
          "ci95": [
            0.006543398782456611,
            0.02968006775047121
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": -0.0002860051070736995,
          "ci95": [
            -0.0004116382941994999,
            -0.00014828909508157292
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": -0.0001511406666845662,
          "ci95": [
            -0.00029471208304860557,
            -1.2544725429385528e-05
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": -0.0003086745824729103,
          "ci95": [
            -0.0004261106880529097,
            -0.0001953610625933579
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": 0.0001348644403891333,
          "ci95": [
            -4.321205258817854e-05,
            0.00029638856996648874
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": -2.2669475399210784e-05,
          "ci95": [
            -0.00017882805535707956,
            0.00011578138409379756
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": -0.0001575339157883441,
          "ci95": [
            -0.0003215143324184003,
            1.4990568334317126e-05
          ]
        }
      }
    },
    "all_prespecified_effects_involving_candidate": {
      "pooled_constant__minus__mature_r2": {
        "coefficients": {
          "mature_r2": -1.0,
          "cross8_mature": 0.0,
          "event_entry": 0.0,
          "event_work": 0.0,
          "event_constant": 0.0,
          "pooled_entry": 0.0,
          "pooled_work": 0.0,
          "pooled_constant": 1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.0042235017367587835,
            "ci95": [
              -0.007324589522790518,
              -0.000923203284268388
            ]
          },
          "auc.is_click": {
            "estimate": -0.0009575723899118982,
            "ci95": [
              -0.0016996682132955603,
              -5.824352269378363e-05
            ]
          },
          "auc.cart": {
            "estimate": -0.0022005528411347397,
            "ci95": [
              -0.0067203211105046155,
              0.0031123360640100666
            ]
          },
          "auc.fav": {
            "estimate": 0.0015530213742487309,
            "ci95": [
              -0.004372308328031699,
              0.008007065189874141
            ]
          },
          "auc.buy": {
            "estimate": -0.015288903090237227,
            "ci95": [
              -0.025911671875335555,
              -0.006000429044110378
            ]
          },
          "bce.is_click": {
            "estimate": -0.00010507198057949085,
            "ci95": [
              -0.00018305323023582246,
              -4.152960773948364e-05
            ]
          },
          "bce.cart": {
            "estimate": 0.00018093312649420865,
            "ci95": [
              5.975293180209207e-05,
              0.0002869935987907534
            ]
          },
          "bce.fav": {
            "estimate": 4.6068686105075335e-05,
            "ci95": [
              -8.56122682329357e-05,
              0.00018332917948443063
            ]
          },
          "bce.buy": {
            "estimate": 0.00020360260189341943,
            "ci95": [
              0.00011711356401445366,
              0.0003014812962696167
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.0012429804512228415,
            "ci95": [
              -0.004130188242823102,
              0.005894418522516549
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.002510593764160629,
            "ci95": [
              -0.008910900097575692,
              0.0034872381423807866
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.014331330700325329,
            "ci95": [
              0.005114553947637776,
              0.025138900143680608
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.0037535742153834706,
            "ci95": [
              -0.011641050886091334,
              0.004609549484416847
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.013088350249102487,
            "ci95": [
              0.0030524125045211102,
              0.024522003815249275
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.016841924464485958,
            "ci95": [
              0.006543398782456611,
              0.02968006775047121
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.0002860051070736995,
            "ci95": [
              -0.0004116382941994999,
              -0.00014828909508157292
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0001511406666845662,
            "ci95": [
              -0.00029471208304860557,
              -1.2544725429385528e-05
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0003086745824729103,
            "ci95": [
              -0.0004261106880529097,
              -0.0001953610625933579
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.0001348644403891333,
            "ci95": [
              -4.321205258817854e-05,
              0.00029638856996648874
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -2.2669475399210784e-05,
            "ci95": [
              -0.00017882805535707956,
              0.00011578138409379756
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.0001575339157883441,
            "ci95": [
              -0.0003215143324184003,
              1.4990568334317126e-05
            ]
          }
        }
      },
      "pooled_constant__minus__cross8_mature": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": -1.0,
          "event_entry": 0.0,
          "event_work": 0.0,
          "event_constant": 0.0,
          "pooled_entry": 0.0,
          "pooled_work": 0.0,
          "pooled_constant": 1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.0034650294826418793,
            "ci95": [
              0.0004733433606766717,
              0.005931692134220944
            ]
          },
          "auc.is_click": {
            "estimate": -0.0009580493318881445,
            "ci95": [
              -0.0017432201014719816,
              -0.00011889041363613616
            ]
          },
          "auc.cart": {
            "estimate": 0.0010269821450930028,
            "ci95": [
              -0.003727843008447676,
              0.005572464961256141
            ]
          },
          "auc.fav": {
            "estimate": 0.007514500721847228,
            "ci95": [
              0.0018048854406276739,
              0.013055937105805169
            ]
          },
          "auc.buy": {
            "estimate": 0.0062766843955154306,
            "ci95": [
              -0.0023800859361454902,
              0.014268026457489194
            ]
          },
          "bce.is_click": {
            "estimate": -3.136356138416785e-05,
            "ci95": [
              -0.00010582881741865654,
              3.811199061465452e-05
            ]
          },
          "bce.cart": {
            "estimate": 0.00017447585884921082,
            "ci95": [
              7.679990105035843e-05,
              0.0002794732744834971
            ]
          },
          "bce.fav": {
            "estimate": -0.0001291528009956383,
            "ci95": [
              -0.00023196354758187512,
              -1.3285609004081869e-05
            ]
          },
          "bce.buy": {
            "estimate": -0.0001599857898190067,
            "ci95": [
              -0.000244656721088648,
              -7.998838970187612e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.0019850314769811472,
            "ci95": [
              -0.006471781775358423,
              0.002650436814717245
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.008472550053735373,
            "ci95": [
              -0.014010176847994372,
              -0.0027272422840017396
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.007234733727403575,
            "ci95": [
              -0.015656236181929524,
              0.0015164243346660064
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.006487518576754225,
            "ci95": [
              -0.013288975570383884,
              0.0010145117179694806
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.005249702250422428,
            "ci95": [
              -0.013781277246797388,
              0.0036911852731920797
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.0012378163263317976,
            "ci95": [
              -0.008495993638711475,
              0.011146363148487243
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00020583942023337867,
            "ci95": [
              -0.0003327676209704415,
              -8.678860569962014e-05
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 9.778923961147046e-05,
            "ci95": [
              -3.593822454922334e-05,
              0.00022708148176584991
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.00012862222843483886,
            "ci95": [
              1.4601672783178679e-05,
              0.00024642330721598974
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.00030362865984484913,
            "ci95": [
              0.00014690313894618264,
              0.00044859017717195296
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00033446164866821754,
            "ci95": [
              0.00020749185523950484,
              0.0004778182077949101
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 3.083298882336841e-05,
            "ci95": [
              -0.00011120854647392259,
              0.00018795202566636185
            ]
          }
        }
      },
      "pooled_work__minus__pooled_constant": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "event_entry": 0.0,
          "event_work": 0.0,
          "event_constant": 0.0,
          "pooled_entry": 0.0,
          "pooled_work": 1.0,
          "pooled_constant": -1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.0029936876738340945,
            "ci95": [
              -0.006090106449406745,
              -0.0002223709344236027
            ]
          },
          "auc.is_click": {
            "estimate": -0.0017901003548540517,
            "ci95": [
              -0.002615026337983084,
              -0.0009935371232632747
            ]
          },
          "auc.cart": {
            "estimate": -0.00028874378502086184,
            "ci95": [
              -0.005308251772275463,
              0.004641629475435562
            ]
          },
          "auc.fav": {
            "estimate": -0.012005165163857856,
            "ci95": [
              -0.01903828609967125,
              -0.0047812673367427765
            ]
          },
          "auc.buy": {
            "estimate": 0.0021092586083963916,
            "ci95": [
              -0.004618041416292032,
              0.009798563295707151
            ]
          },
          "bce.is_click": {
            "estimate": 0.0002787512061017683,
            "ci95": [
              0.00021006672238330082,
              0.00035575983444029104
            ]
          },
          "bce.cart": {
            "estimate": -6.2569193780556e-05,
            "ci95": [
              -0.00015899139505085947,
              4.32870558539637e-05
            ]
          },
          "bce.fav": {
            "estimate": 7.53785373265349e-05,
            "ci95": [
              -4.828690034879993e-05,
              0.00019811660211104106
            ]
          },
          "bce.buy": {
            "estimate": 7.036348449503821e-06,
            "ci95": [
              -7.67923101569562e-05,
              8.225657449540733e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.0015013565698331899,
            "ci95": [
              -0.006626870334370765,
              0.00341274066106135
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.010215064809003804,
            "ci95": [
              0.003044517156223884,
              0.016837666126490847
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.0038993589632504433,
            "ci95": [
              -0.01163210632904547,
              0.0031765206147340843
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.011716421378836994,
            "ci95": [
              0.002466178307395134,
              0.020288203882456692
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.0023980023934172534,
            "ci95": [
              -0.010839111438292451,
              0.005536250171018036
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.014114423772254248,
            "ci95": [
              -0.02474374274058776,
              -0.004305986566643614
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 0.0003413203998823243,
            "ci95": [
              0.00021966939583556322,
              0.00047346331099792677
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.00020337266877523338,
            "ci95": [
              5.642418607666082e-05,
              0.0003386463586889399
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.00027171485765226447,
            "ci95": [
              0.00017664218828703084,
              0.00038189959008054936
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.0001379477311070909,
            "ci95": [
              -0.0002915241884792028,
              1.9530571318271142e-05
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -6.960554223005982e-05,
            "ci95": [
              -0.00020032626248529784,
              5.648234965069387e-05
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 6.834218887703108e-05,
            "ci95": [
              -7.526913683789117e-05,
              0.00021281142028830089
            ]
          }
        }
      },
      "pooled_entry__minus__pooled_constant": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "event_entry": 0.0,
          "event_work": 0.0,
          "event_constant": 0.0,
          "pooled_entry": 1.0,
          "pooled_work": 0.0,
          "pooled_constant": -1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.0008718742079763253,
            "ci95": [
              -0.002541345630540502,
              0.00416568162071926
            ]
          },
          "auc.is_click": {
            "estimate": 0.0016786389721302752,
            "ci95": [
              0.000908394469842605,
              0.002342168470819914
            ]
          },
          "auc.cart": {
            "estimate": 0.0012518311232687651,
            "ci95": [
              -0.0035416219653413125,
              0.005590917853007678
            ]
          },
          "auc.fav": {
            "estimate": -0.02037535509687116,
            "ci95": [
              -0.028706715681554477,
              -0.013348930726415676
            ]
          },
          "auc.buy": {
            "estimate": 0.020932381833377423,
            "ci95": [
              0.01075953199622393,
              0.031046952684248872
            ]
          },
          "bce.is_click": {
            "estimate": 3.5358156492137915e-06,
            "ci95": [
              -5.7924685324634014e-05,
              7.138304705624553e-05
            ]
          },
          "bce.cart": {
            "estimate": -0.00012073095678035925,
            "ci95": [
              -0.0002293966279230117,
              -2.4042088648557063e-06
            ]
          },
          "bce.fav": {
            "estimate": 0.00024797957501380974,
            "ci95": [
              0.00012569326090932642,
              0.00037819111541030756
            ]
          },
          "bce.buy": {
            "estimate": -0.00019225479262598155,
            "ci95": [
              -0.00030314726578801316,
              -8.068776257810484e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.00042680784886151013,
            "ci95": [
              -0.004003171167310795,
              0.005316424296402355
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.022053994069001437,
            "ci95": [
              0.014645672366693003,
              0.030641689043125902
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.019253742861247147,
            "ci95": [
              -0.029249193236655553,
              -0.009201582352405424
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.021627186220139927,
            "ci95": [
              0.013687604123135433,
              0.03132150735028872
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.019680550710108657,
            "ci95": [
              -0.030531640271696102,
              -0.008492204363971297
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.041307736930248584,
            "ci95": [
              -0.0551530823990585,
              -0.03000889559932831
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 0.00012426677242957304,
            "ci95": [
              1.8085409608919012e-06,
              0.00024621528182420843
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.00024444375936459595,
            "ci95": [
              -0.00040586271570650957,
              -9.940155481536201e-05
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.00019579060827519534,
            "ci95": [
              7.549487064049896e-05,
              0.0003171994250660267
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.000368710531794169,
            "ci95": [
              -0.0005482073244132364,
              -0.00020748858533717633
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 7.15238358456223e-05,
            "ci95": [
              -8.297364084205926e-05,
              0.00023741015416631165
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.0004402343676397913,
            "ci95": [
              0.00027058208645828525,
              0.0006191311508135866
            ]
          }
        }
      },
      "event_constant__minus__pooled_constant": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "event_entry": 0.0,
          "event_work": 0.0,
          "event_constant": 1.0,
          "pooled_entry": 0.0,
          "pooled_work": 0.0,
          "pooled_constant": -1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.00025180421395237795,
            "ci95": [
              -0.0029954776395498214,
              0.0024113440355742212
            ]
          },
          "auc.is_click": {
            "estimate": -0.00036261017189942457,
            "ci95": [
              -0.001204460403404023,
              0.0004541129405169484
            ]
          },
          "auc.cart": {
            "estimate": -0.0026232904168230764,
            "ci95": [
              -0.007784982009576738,
              0.0022129506811554896
            ]
          },
          "auc.fav": {
            "estimate": -0.007176534468365969,
            "ci95": [
              -0.01290605647249353,
              -0.001198030453358779
            ]
          },
          "auc.buy": {
            "estimate": 0.009155218201278958,
            "ci95": [
              0.003360643318409079,
              0.015074213415986403
            ]
          },
          "bce.is_click": {
            "estimate": -8.25878002652658e-06,
            "ci95": [
              -8.219178077217281e-05,
              7.14883278681029e-05
            ]
          },
          "bce.cart": {
            "estimate": -8.399051095163423e-05,
            "ci95": [
              -0.00020023333117741613,
              4.4975476812745966e-05
            ]
          },
          "bce.fav": {
            "estimate": 0.00017158834138939622,
            "ci95": [
              6.18915433318607e-05,
              0.0002737970716519975
            ]
          },
          "bce.buy": {
            "estimate": 6.463249922088535e-05,
            "ci95": [
              -8.694135214013832e-06,
              0.0001419208854116593
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.002260680244923652,
            "ci95": [
              -0.002632639009839413,
              0.007713893652079347
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.006813924296466545,
            "ci95": [
              0.0007364360215334962,
              0.01242633393337315
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.009517828373178383,
            "ci95": [
              -0.015539235638789792,
              -0.00393216843347357
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.004553244051542893,
            "ci95": [
              -0.0029590849402455627,
              0.01168784207975763
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.011778508618102035,
            "ci95": [
              -0.019022092863581027,
              -0.0034682293413243162
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.016331752669644928,
            "ci95": [
              -0.023399269415326458,
              -0.00864582247176971
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 7.573173092510765e-05,
            "ci95": [
              -8.344323253162326e-05,
              0.00021545722574128852
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0001798471214159228,
            "ci95": [
              -0.0003158866489188897,
              -3.773717156397079e-05
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -7.289127924741193e-05,
            "ci95": [
              -0.00017999178847246303,
              3.718250157276669e-05
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.00025557885234103045,
            "ci95": [
              -0.00041011730140790795,
              -7.841781496916532e-05
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -0.00014862301017251958,
            "ci95": [
              -0.00029319008539428434,
              -4.2401175398655987e-07
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.00010695584216851087,
            "ci95": [
              -2.181020963889883e-05,
              0.00023079173973156376
            ]
          }
        }
      }
    },
    "parameter_count": 51742692,
    "initializer_sha256": "aa3b2ab7b9014c0f0183713dca1e9fcc087775a3c1683e2efaa4346181bd573d",
    "optimizer_layout_sha256": "1c0d89f2031d378230cea4f873579c283bea42f7c3db6882cfc8a50ea03fae63",
    "formal_worker_cost": {
      "wall_seconds": 2552.4282142072916,
      "cpu_seconds": 3065.293133013,
      "peak_rss_bytes": 17054146560,
      "gpu_peak_bytes": 28369223680,
      "samples": 2343
    },
    "smoke_worker_cost": {
      "wall_seconds": 204.48706128448248,
      "cpu_seconds": 261.991356354,
      "peak_rss_bytes": 19870900224,
      "gpu_peak_bytes": 30002905088,
      "samples": 187
    },
    "complete_recorded_costs": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2552.4282142072916,
        "cpu_seconds": 3065.293133013,
        "peak_rss_bytes": 17054146560,
        "gpu_peak_bytes": 28369223680,
        "samples": 2343
      },
      "formal_runner": {
        "wall_seconds": 2548.5179096236825,
        "cpu_seconds": 2950.633148782,
        "peak_process_rss_bytes": 5300428800
      },
      "fit_including_validation": {
        "wall_seconds": 2424.5024347752333,
        "cpu_seconds": 2778.4093952010003,
        "peak_process_rss_bytes": 5300428800
      },
      "prediction_export": {
        "wall_seconds": 118.95606475323439,
        "cpu_seconds": 166.78306442300027,
        "peak_process_rss_bytes": 5300428800
      },
      "smoke_worker": {
        "wall_seconds": 204.48706128448248,
        "cpu_seconds": 261.991356354,
        "peak_rss_bytes": 19870900224,
        "gpu_peak_bytes": 30002905088,
        "samples": 187
      },
      "smoke_runner": {
        "wall_seconds": 200.56655617058277,
        "cpu_seconds": 182.56273697100002,
        "peak_process_rss_bytes": 3390656512
      },
      "latency_constructor": {
        "wall_seconds": 0.7515730932354927,
        "cpu_seconds": 0.8265921199999973,
        "peak_rss_bytes": 2135166976
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.13166256248950958,
        "cpu_seconds": 0.13121063000000177,
        "peak_rss_bytes": 2135166976
      },
      "latency_peak_cuda_bytes": 1284208128,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "forward_latency": {
      "mean_ms": 14.265369629859924,
      "p50_ms": 14.26470422744751,
      "p95_ms": 14.280579042434692,
      "batch_size": 1024,
      "warmups": 10,
      "iterations": 40
    },
    "latency_ratio_to_parent": 0.9992039106582571,
    "quality_candidate": false,
    "near_candidate": false
  },
  "validity": {
    "valid": true,
    "violations": [],
    "state": "passed",
    "quality_summary": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/quality-summary.json",
    "no_prediction_to_work_feedback": true,
    "test_evaluated": false,
    "source_and_input_identity_scope": "Original N full-input content SHA256 values are inherited through completed C quality and E authorization. Production runners checked input stats before/after their traversals and two original metadata hashes before them. This collector now checks 122 current stats and both original metadata hashes; it does not rehash Parquet or open checkpoint/prediction arrays. New E worker/smoke/effective-optimizer gates are evaluated from actual small receipts. Three references retain exact C quality records and producer completion hashes, with small source/artifact identity checks. Stat continuity cannot independently exclude content changes that preserve stat fields."
  },
  "decision": "discard",
  "decision_reason": "Macro delta -0.004223502, CI95 [-0.007324589522790518, -0.000923203284268388], forward ratio 0.999204; misses both frozen quality and near policy.",
  "decision_scope": "Root queue disposition only; no released-incumbent promotion or supported lesson.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/pooled_constant",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/pooled_constant/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/pooled_constant/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/pooled_constant/Taobao_Action/pooled_constant.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/paired-analysis/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/latency/completion.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/pooled_constant",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/pooled_constant/result.json",
    "result_sha256": "b37f43f7bec6a68d26cd3662f64543df5c1d9992d6b42031457ea3dec21a6d3d",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/pooled_constant/manifest.json",
    "manifest_sha256": "db42b383fc3e34fbf9544d875707834809a335190773f47433d3f061b4f8ded0",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/pooled_constant/Taobao_Action/pooled_constant.model",
    "checkpoint_sha256": "fc5f2f20e735a6565e50afbc8ce3b4cf3df69c1611337147ec21f7426f5e3f29",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/paired-analysis/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
