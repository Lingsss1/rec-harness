## 146. taobao-query-routes-final-interpretation

- 假设：Route-restricted query correction should improve matching all-route/original recurrence if the diagnostic-guided architecture hypothesis transfers to retraining
- 判定：discard
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-09T03:32:14.315612+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-query-routes-final-interpretation",
  "family": "round_interpretation",
  "hypothesis": "Route-restricted query correction should improve matching all-route/original recurrence if the diagnostic-guided architecture hypothesis transfers to retraining",
  "change": "Adjudicate the eight previously provisional training episodes against the prospective interpretation and latency policy; preserve original rows",
  "supersedes_episode_ids": [
    "route_q4_retrieval-s20262027-20260908T190038665636Z",
    "route_q4_interaction-s20262027-20260908T190043679512Z",
    "route_q4_residual-s20262027-20260908T190045767031Z",
    "route_q4_attention-s20262027-20260908T190044491955Z",
    "route_q8_retrieval-s20262027-20260908T190022268468Z",
    "route_q8_interaction-s20262027-20260908T190028413046Z",
    "route_q8_residual-s20262027-20260908T194602060181Z",
    "route_q8_attention-s20262027-20260908T190022893149Z"
  ],
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "state": "completed_interpreted_and_consolidated",
    "new_training_runs": 8,
    "decisions": {
      "route_q4_retrieval": {
        "decision": "discard",
        "supersedes_training_episode": "route_q4_retrieval-s20262027-20260908T190038665636Z",
        "macro_auc": 0.8280084442398983,
        "versus_all_route": {
          "reference": "query_q4_work",
          "candidate": "route_q4_retrieval",
          "reference_role": "all_route",
          "delta_macro_auc": -0.00045733235159151064,
          "delta_auc": {
            "is_click": -0.0004622748162819157,
            "cart": -0.0015662881059558265,
            "fav": -0.0006671693585835747,
            "buy": 0.0008664028744558294
          },
          "delta_logloss": {
            "is_click": 0.00027941097742589793,
            "cart": 5.246466991745344e-05,
            "fav": 4.470603191234282e-06,
            "buy": 8.236684429840935e-06
          },
          "delta_macro_logloss": 8.614573374110535e-05,
          "parameter_ratio": 1.0
        },
        "versus_original": {
          "reference": "ctx_q4",
          "candidate": "route_q4_retrieval",
          "reference_role": "original_recurrence",
          "delta_macro_auc": -0.0004641211863453121,
          "delta_auc": {
            "is_click": 0.0006404787039047122,
            "cart": -0.0013599407265042496,
            "fav": -0.0007813297242216244,
            "buy": -0.0003556929985600865
          },
          "delta_logloss": {
            "is_click": -8.398026472433706e-05,
            "cart": 2.0307071089069317e-05,
            "fav": -1.9640814721409716e-05,
            "buy": 1.5729630233846797e-05
          },
          "delta_macro_logloss": -1.6896094530705064e-05,
          "parameter_ratio": 1.0002375675863155
        },
        "versus_best_single": {
          "reference": "psgd_precond_every_step",
          "candidate": "route_q4_retrieval",
          "reference_role": "best_single_pass",
          "delta_macro_auc": -0.0006150882797764101,
          "delta_auc": {
            "is_click": -0.0011752526640057548,
            "cart": -0.0016639895199196397,
            "fav": -0.0004686473902755228,
            "buy": 0.0008475364550952769
          },
          "delta_logloss": {
            "is_click": 0.00022568011984647196,
            "cart": 0.00014695893005346045,
            "fav": 7.605498156219338e-05,
            "buy": 0.00011050404935685019
          },
          "delta_macro_logloss": 0.00013979952020474334,
          "parameter_ratio": 0.6338991204749678
        },
        "all_route_ci95": [
          -0.0012380908214365501,
          0.00038367189431029393
        ],
        "original_ci95": null,
        "original_ci_note": "Not estimated by the frozen positive-candidate-only supplementary bootstrap policy",
        "matched_forward_mean_ms": 14.240333604812623,
        "forward_ratio_to_original": 1.0016971043170797,
        "near_reference_policy_passed": false,
        "per_task": {
          "is_click": {
            "logloss": 0.18222966499837673,
            "AUC": 0.6666145257274362
          },
          "cart": {
            "logloss": 0.01778199013397052,
            "AUC": 0.8653550371211456
          },
          "fav": {
            "logloss": 0.013346376978900237,
            "AUC": 0.8818925814971114
          },
          "buy": {
            "logloss": 0.006732383821783032,
            "AUC": 0.8981716326139002
          }
        }
      },
      "route_q4_interaction": {
        "decision": "discard",
        "supersedes_training_episode": "route_q4_interaction-s20262027-20260908T190043679512Z",
        "macro_auc": 0.827423329017671,
        "versus_all_route": {
          "reference": "query_q4_work",
          "candidate": "route_q4_interaction",
          "reference_role": "all_route",
          "delta_macro_auc": -0.0010424475738187722,
          "delta_auc": {
            "is_click": -0.0006969844377778811,
            "cart": -0.0018230978597447711,
            "fav": -0.0005237597308971331,
            "buy": -0.0011259482668550813
          },
          "delta_logloss": {
            "is_click": 7.277905832420162e-05,
            "cart": 0.00016482728789110612,
            "fav": -5.573409923976744e-06,
            "buy": 3.114771478229979e-05
          },
          "delta_macro_logloss": 6.579516276840336e-05,
          "parameter_ratio": 1.0
        },
        "versus_original": {
          "reference": "ctx_q4",
          "candidate": "route_q4_interaction",
          "reference_role": "original_recurrence",
          "delta_macro_auc": -0.0010492364085725736,
          "delta_auc": {
            "is_click": 0.00040576908240874676,
            "cart": -0.0016167504802931942,
            "fav": -0.0006379200965351828,
            "buy": -0.0023480441398709972
          },
          "delta_logloss": {
            "is_click": -0.0002906121838260334,
            "cart": 0.000132669689062722,
            "fav": -2.9684827836620742e-05,
            "buy": 3.864066058630565e-05
          },
          "delta_macro_logloss": -3.724666550340705e-05,
          "parameter_ratio": 1.0002375675863155
        },
        "versus_best_single": {
          "reference": "psgd_precond_every_step",
          "candidate": "route_q4_interaction",
          "reference_role": "best_single_pass",
          "delta_macro_auc": -0.0012002035020036717,
          "delta_auc": {
            "is_click": -0.0014099622855017202,
            "cart": -0.0019207992737085844,
            "fav": -0.0003252377625890812,
            "buy": -0.0011448146862156339
          },
          "delta_logloss": {
            "is_click": 1.9048200744775645e-05,
            "cart": 0.00025932154802711313,
            "fav": 6.601096844698236e-05,
            "buy": 0.00013341507970930904
          },
          "delta_macro_logloss": 0.00011944894923204136,
          "parameter_ratio": 0.6338991204749678
        },
        "all_route_ci95": [
          -0.0018051809894652972,
          -0.0003267512035853428
        ],
        "original_ci95": null,
        "original_ci_note": "Not estimated by the frozen positive-candidate-only supplementary bootstrap policy",
        "matched_forward_mean_ms": 14.237878370285035,
        "forward_ratio_to_original": 1.0015243975965102,
        "near_reference_policy_passed": false,
        "per_task": {
          "is_click": {
            "logloss": 0.18202303307927503,
            "AUC": 0.6663798161059402
          },
          "cart": {
            "logloss": 0.017894352751944172,
            "AUC": 0.8650982273673566
          },
          "fav": {
            "logloss": 0.013336332965785026,
            "AUC": 0.8820359911247978
          },
          "buy": {
            "logloss": 0.006755294852135491,
            "AUC": 0.8961792814725893
          }
        }
      },
      "route_q4_residual": {
        "decision": "discard",
        "supersedes_training_episode": "route_q4_residual-s20262027-20260908T190045767031Z",
        "macro_auc": 0.8259837513740176,
        "versus_all_route": {
          "reference": "query_q4_work",
          "candidate": "route_q4_residual",
          "reference_role": "all_route",
          "delta_macro_auc": -0.0024820252174722057,
          "delta_auc": {
            "is_click": -0.00018398864549418192,
            "cart": -0.0032051886648571948,
            "fav": -0.003109561726142962,
            "buy": -0.003429361833394151
          },
          "delta_logloss": {
            "is_click": 4.2031655257041e-05,
            "cart": 8.275492321007175e-05,
            "fav": 7.540220265096591e-05,
            "buy": 5.76575288457953e-05
          },
          "delta_macro_logloss": 6.446157749096654e-05,
          "parameter_ratio": 1.0
        },
        "versus_original": {
          "reference": "ctx_q4",
          "candidate": "route_q4_residual",
          "reference_role": "original_recurrence",
          "delta_macro_auc": -0.002488814052226007,
          "delta_auc": {
            "is_click": 0.0009187648746924459,
            "cart": -0.002998841285405618,
            "fav": -0.0032237220917810117,
            "buy": -0.004651457706410067
          },
          "delta_logloss": {
            "is_click": -0.000321359586893194,
            "cart": 5.0597324381687625e-05,
            "fav": 5.129078473832191e-05,
            "buy": 6.515047464980116e-05
          },
          "delta_macro_logloss": -3.858025078084387e-05,
          "parameter_ratio": 1.0002375675863155
        },
        "versus_best_single": {
          "reference": "psgd_precond_every_step",
          "candidate": "route_q4_residual",
          "reference_role": "best_single_pass",
          "delta_macro_auc": -0.002639781145657105,
          "delta_auc": {
            "is_click": -0.000896966493218021,
            "cart": -0.003302890078821008,
            "fav": -0.00291103975783491,
            "buy": -0.0034482282527547037
          },
          "delta_logloss": {
            "is_click": -1.1699202322384972e-05,
            "cart": 0.00017724918334607875,
            "fav": 0.000146986581021925,
            "buy": 0.00015992489377280455
          },
          "delta_macro_logloss": 0.00011811536395460454,
          "parameter_ratio": 0.6338991204749678
        },
        "all_route_ci95": [
          -0.003322204222685762,
          -0.0016184830848244827
        ],
        "original_ci95": null,
        "original_ci_note": "Not estimated by the frozen positive-candidate-only supplementary bootstrap policy",
        "matched_forward_mean_ms": 14.2431800365448,
        "forward_ratio_to_original": 1.0018973287291535,
        "near_reference_policy_passed": false,
        "per_task": {
          "is_click": {
            "logloss": 0.18199228567620787,
            "AUC": 0.6668928118982239
          },
          "cart": {
            "logloss": 0.017812280387263138,
            "AUC": 0.8637161365622442
          },
          "fav": {
            "logloss": 0.013417308578359969,
            "AUC": 0.879450189129552
          },
          "buy": {
            "logloss": 0.006781804666198986,
            "AUC": 0.8938758679060502
          }
        }
      },
      "route_q4_attention": {
        "decision": "discard",
        "supersedes_training_episode": "route_q4_attention-s20262027-20260908T190044491955Z",
        "macro_auc": 0.8247006223642905,
        "versus_all_route": {
          "reference": "query_q4_work",
          "candidate": "route_q4_attention",
          "reference_role": "all_route",
          "delta_macro_auc": -0.003765154227199252,
          "delta_auc": {
            "is_click": -0.00615657540659631,
            "cart": -0.001842093358151109,
            "fav": -0.002520994028748391,
            "buy": -0.004540954115300755
          },
          "delta_logloss": {
            "is_click": 0.000765041921418963,
            "cart": 3.4881500075391314e-05,
            "fav": 5.7949506469704565e-05,
            "buy": 6.015536888660626e-05
          },
          "delta_macro_logloss": 0.00022950707421266064,
          "parameter_ratio": 1.0
        },
        "versus_original": {
          "reference": "ctx_q4",
          "candidate": "route_q4_attention",
          "reference_role": "original_recurrence",
          "delta_macro_auc": -0.0037719430619530536,
          "delta_auc": {
            "is_click": -0.005053821886409682,
            "cart": -0.001635745978699532,
            "fav": -0.002635154394386441,
            "buy": -0.005763049988316671
          },
          "delta_logloss": {
            "is_click": 0.000401650679268728,
            "cart": 2.7239012470071933e-06,
            "fav": 3.383808855706057e-05,
            "buy": 6.764831469061212e-05
          },
          "delta_macro_logloss": 0.00012646524594085023,
          "parameter_ratio": 1.0002375675863155
        },
        "versus_best_single": {
          "reference": "psgd_precond_every_step",
          "candidate": "route_q4_attention",
          "reference_role": "best_single_pass",
          "delta_macro_auc": -0.003922910155384152,
          "delta_auc": {
            "is_click": -0.006869553254320149,
            "cart": -0.0019397947721149222,
            "fav": -0.0023224720604403393,
            "buy": -0.0045598205346613074
          },
          "delta_logloss": {
            "is_click": 0.000711311063839537,
            "cart": 0.00012937576021139832,
            "fav": 0.00012953388484066367,
            "buy": 0.00016242273381361551
          },
          "delta_macro_logloss": 0.00028316086067629864,
          "parameter_ratio": 0.6338991204749678
        },
        "all_route_ci95": [
          -0.004567699385545,
          -0.0029791525489986643
        ],
        "original_ci95": null,
        "original_ci_note": "Not estimated by the frozen positive-candidate-only supplementary bootstrap policy",
        "matched_forward_mean_ms": 14.2286936044693,
        "forward_ratio_to_original": 1.0008783205047247,
        "near_reference_policy_passed": false,
        "per_task": {
          "is_click": {
            "logloss": 0.1827152959423698,
            "AUC": 0.6609202251371218
          },
          "cart": {
            "logloss": 0.017764406964128458,
            "AUC": 0.8650792318689503
          },
          "fav": {
            "logloss": 0.013399855882178707,
            "AUC": 0.8800387568269465
          },
          "buy": {
            "logloss": 0.006784302506239797,
            "AUC": 0.8927642756241436
          }
        }
      },
      "route_q8_retrieval": {
        "decision": "discard",
        "supersedes_training_episode": "route_q8_retrieval-s20262027-20260908T190022268468Z",
        "macro_auc": 0.824528857028706,
        "versus_all_route": {
          "reference": "query_q8_entry",
          "candidate": "route_q8_retrieval",
          "reference_role": "all_route",
          "delta_macro_auc": -0.003692149549567758,
          "delta_auc": {
            "is_click": -0.0035101699785703433,
            "cart": -0.005462728705010189,
            "fav": -0.0038275792846406143,
            "buy": -0.0019681202300498857
          },
          "delta_logloss": {
            "is_click": 0.00030734771919382897,
            "cart": 0.00013038274612635,
            "fav": 5.489089244513183e-05,
            "buy": 2.8857770976401832e-05
          },
          "delta_macro_logloss": 0.00013036978218543965,
          "parameter_ratio": 1.0
        },
        "versus_original": {
          "reference": "explore_dual_no_feedback",
          "candidate": "route_q8_retrieval",
          "reference_role": "original_recurrence",
          "delta_macro_auc": -0.003992356345519421,
          "delta_auc": {
            "is_click": -0.003137245267883837,
            "cart": -0.004596845898274582,
            "fav": -0.004267134164282704,
            "buy": -0.00396820005163645
          },
          "delta_logloss": {
            "is_click": 0.0002702847100328476,
            "cart": 5.115641324855702e-05,
            "fav": 3.024882700084809e-05,
            "buy": 4.944041674073436e-05
          },
          "delta_macro_logloss": 0.00010028259175575566,
          "parameter_ratio": 1.0001505581162906
        },
        "versus_best_single": {
          "reference": "psgd_precond_every_step",
          "candidate": "route_q8_retrieval",
          "reference_role": "best_single_pass",
          "delta_macro_auc": -0.004094675490968647,
          "delta_auc": {
            "is_click": -0.004057137634184915,
            "cart": -0.0041190368323099635,
            "fav": -0.004617817574775529,
            "buy": -0.003584709922604512
          },
          "delta_logloss": {
            "is_click": 0.00031039767516063965,
            "cart": 4.407728991726381e-05,
            "fav": 3.639881083657495e-05,
            "buy": 3.161340314638498e-05
          },
          "delta_macro_logloss": 0.0001056217947652191,
          "parameter_ratio": 1.0001505581162906
        },
        "all_route_ci95": [
          -0.004526139306515168,
          -0.002840970183414894
        ],
        "original_ci95": null,
        "original_ci_note": "Not estimated by the frozen positive-candidate-only supplementary bootstrap policy",
        "matched_forward_mean_ms": 15.793389582633973,
        "forward_ratio_to_original": 1.0027334521872724,
        "near_reference_policy_passed": false,
        "per_task": {
          "is_click": {
            "logloss": 0.1823143825536909,
            "AUC": 0.663732640757257
          },
          "cart": {
            "logloss": 0.017679108493834323,
            "AUC": 0.8628999898087553
          },
          "fav": {
            "logloss": 0.013306720808174619,
            "AUC": 0.8777434113126114
          },
          "buy": {
            "logloss": 0.006653493175572567,
            "AUC": 0.8937393862362004
          }
        }
      },
      "route_q8_interaction": {
        "decision": "discard",
        "supersedes_training_episode": "route_q8_interaction-s20262027-20260908T190028413046Z",
        "macro_auc": 0.8258797824447256,
        "versus_all_route": {
          "reference": "query_q8_entry",
          "candidate": "route_q8_interaction",
          "reference_role": "all_route",
          "delta_macro_auc": -0.002341224133548203,
          "delta_auc": {
            "is_click": -0.0023880215844530595,
            "cart": -0.004532707761128685,
            "fav": -0.0020752318926116065,
            "buy": -0.00036893529599957287
          },
          "delta_logloss": {
            "is_click": 0.00027528031977058776,
            "cart": 0.0002593836636216422,
            "fav": 6.512467555479413e-05,
            "buy": 9.219007816382505e-05
          },
          "delta_macro_logloss": 0.00017299468427772052,
          "parameter_ratio": 1.0
        },
        "versus_original": {
          "reference": "explore_dual_no_feedback",
          "candidate": "route_q8_interaction",
          "reference_role": "original_recurrence",
          "delta_macro_auc": -0.002641430929499866,
          "delta_auc": {
            "is_click": -0.002015096873766553,
            "cart": -0.0036668249543930775,
            "fav": -0.0025147867722536965,
            "buy": -0.0023690151175861374
          },
          "delta_logloss": {
            "is_click": 0.00023821731060960638,
            "cart": 0.0001801573307438492,
            "fav": 4.0482610110510395e-05,
            "buy": 0.00011277272392815757
          },
          "delta_macro_logloss": 0.00014290749384803653,
          "parameter_ratio": 1.0001505581162906
        },
        "versus_best_single": {
          "reference": "psgd_precond_every_step",
          "candidate": "route_q8_interaction",
          "reference_role": "best_single_pass",
          "delta_macro_auc": -0.0027437500749490917,
          "delta_auc": {
            "is_click": -0.0029349892400676314,
            "cart": -0.003189015888428459,
            "fav": -0.002865470182746521,
            "buy": -0.0019855249885541992
          },
          "delta_logloss": {
            "is_click": 0.00027833027573739844,
            "cart": 0.000173078207412556,
            "fav": 4.6632593946237255e-05,
            "buy": 9.494571033380819e-05
          },
          "delta_macro_logloss": 0.00014824669685749997,
          "parameter_ratio": 1.0001505581162906
        },
        "all_route_ci95": [
          -0.003260880697020183,
          -0.00151798204853904
        ],
        "original_ci95": null,
        "original_ci_note": "Not estimated by the frozen positive-candidate-only supplementary bootstrap policy",
        "matched_forward_mean_ms": 15.794897556304932,
        "forward_ratio_to_original": 1.0028291945000278,
        "near_reference_policy_passed": false,
        "per_task": {
          "is_click": {
            "logloss": 0.18228231515426765,
            "AUC": 0.6648547891513743
          },
          "cart": {
            "logloss": 0.017808109411329615,
            "AUC": 0.8638300107526368
          },
          "fav": {
            "logloss": 0.01331695459128428,
            "AUC": 0.8794957587046404
          },
          "buy": {
            "logloss": 0.00671682548275999,
            "AUC": 0.8953385711702507
          }
        }
      },
      "route_q8_residual": {
        "decision": "discard",
        "supersedes_training_episode": "route_q8_residual-s20262027-20260908T194602060181Z",
        "macro_auc": 0.8236724059767455,
        "versus_all_route": {
          "reference": "query_q8_entry",
          "candidate": "route_q8_residual",
          "reference_role": "all_route",
          "delta_macro_auc": -0.004548600601528285,
          "delta_auc": {
            "is_click": -0.005057084673297241,
            "cart": -0.004419185956446081,
            "fav": -0.004329441596192973,
            "buy": -0.004388690180176735
          },
          "delta_logloss": {
            "is_click": 0.0008874024232513322,
            "cart": 0.00016580497403182598,
            "fav": 0.0001756554790395739,
            "buy": 7.720180104665072e-05
          },
          "delta_macro_logloss": 0.00032651616934235156,
          "parameter_ratio": 1.0
        },
        "versus_original": {
          "reference": "explore_dual_no_feedback",
          "candidate": "route_q8_residual",
          "reference_role": "original_recurrence",
          "delta_macro_auc": -0.004848807397479948,
          "delta_auc": {
            "is_click": -0.004684159962610734,
            "cart": -0.003553303149710474,
            "fav": -0.004768996475835063,
            "buy": -0.0063887700017633
          },
          "delta_logloss": {
            "is_click": 0.0008503394140903509,
            "cart": 8.657864115403299e-05,
            "fav": 0.00015101341359529015,
            "buy": 9.778444681098324e-05
          },
          "delta_macro_logloss": 0.00029642897891266756,
          "parameter_ratio": 1.0001505581162906
        },
        "versus_best_single": {
          "reference": "psgd_precond_every_step",
          "candidate": "route_q8_residual",
          "reference_role": "best_single_pass",
          "delta_macro_auc": -0.004951126542929174,
          "delta_auc": {
            "is_click": -0.005604052328911813,
            "cart": -0.0030754940837458555,
            "fav": -0.005119679886327888,
            "buy": -0.006005279872731362
          },
          "delta_logloss": {
            "is_click": 0.0008904523792181429,
            "cart": 7.949951782273978e-05,
            "fav": 0.000157163397431017,
            "buy": 7.995743321663386e-05
          },
          "delta_macro_logloss": 0.000301768181922131,
          "parameter_ratio": 1.0001505581162906
        },
        "all_route_ci95": [
          -0.005454212685787415,
          -0.003618478271830941
        ],
        "original_ci95": null,
        "original_ci_note": "Not estimated by the frozen positive-candidate-only supplementary bootstrap policy",
        "matched_forward_mean_ms": 15.849574375152589,
        "forward_ratio_to_original": 1.0063006643216812,
        "near_reference_policy_passed": false,
        "per_task": {
          "is_click": {
            "logloss": 0.1828944372577484,
            "AUC": 0.6621857260625301
          },
          "cart": {
            "logloss": 0.0177145307217398,
            "AUC": 0.8639435325573194
          },
          "fav": {
            "logloss": 0.01342748539476906,
            "AUC": 0.877241549001059
          },
          "buy": {
            "logloss": 0.006701837205642816,
            "AUC": 0.8913188162860736
          }
        }
      },
      "route_q8_attention": {
        "decision": "discard",
        "supersedes_training_episode": "route_q8_attention-s20262027-20260908T190022893149Z",
        "macro_auc": 0.8256134511464231,
        "versus_all_route": {
          "reference": "query_q8_entry",
          "candidate": "route_q8_attention",
          "reference_role": "all_route",
          "delta_macro_auc": -0.00260755543185065,
          "delta_auc": {
            "is_click": -0.0019389878450730702,
            "cart": -0.004795425811231047,
            "fav": -0.002276787449856821,
            "buy": -0.0014190206212415513
          },
          "delta_logloss": {
            "is_click": 0.0002349278183414738,
            "cart": 0.00019302357416589241,
            "fav": 0.00012550140423868254,
            "buy": 3.857641061030641e-05
          },
          "delta_macro_logloss": 0.000148007301839094,
          "parameter_ratio": 1.0
        },
        "versus_original": {
          "reference": "explore_dual_no_feedback",
          "candidate": "route_q8_attention",
          "reference_role": "original_recurrence",
          "delta_macro_auc": -0.002907762227802313,
          "delta_auc": {
            "is_click": -0.0015660631343865639,
            "cart": -0.003929543004495439,
            "fav": -0.002716342329498911,
            "buy": -0.0034191004428281158
          },
          "delta_logloss": {
            "is_click": 0.00019786480918049243,
            "cart": 0.00011379724128809943,
            "fav": 0.0001008593387943988,
            "buy": 5.915905637463893e-05
          },
          "delta_macro_logloss": 0.00011792011140941,
          "parameter_ratio": 1.0001505581162906
        },
        "versus_best_single": {
          "reference": "psgd_precond_every_step",
          "candidate": "route_q8_attention",
          "reference_role": "best_single_pass",
          "delta_macro_auc": -0.0030100813732515386,
          "delta_auc": {
            "is_click": -0.002485955500687642,
            "cart": -0.003451733938530821,
            "fav": -0.0030670257399917356,
            "buy": -0.0030356103137961776
          },
          "delta_logloss": {
            "is_click": 0.00023797777430828448,
            "cart": 0.00010671811795680622,
            "fav": 0.00010700932263012566,
            "buy": 4.1332042780289555e-05
          },
          "delta_macro_logloss": 0.00012325931441887344,
          "parameter_ratio": 1.0001505581162906
        },
        "all_route_ci95": [
          -0.003517084885309231,
          -0.0017464702277697282
        ],
        "original_ci95": null,
        "original_ci_note": "Not estimated by the frozen positive-candidate-only supplementary bootstrap policy",
        "matched_forward_mean_ms": 15.837009596824647,
        "forward_ratio_to_original": 1.005502917676933,
        "near_reference_policy_passed": false,
        "per_task": {
          "is_click": {
            "logloss": 0.18224196265283854,
            "AUC": 0.6653038228907543
          },
          "cart": {
            "logloss": 0.017741749321873865,
            "AUC": 0.8635672927025344
          },
          "fav": {
            "logloss": 0.01337733131996817,
            "AUC": 0.8792942031473951
          },
          "buy": {
            "logloss": 0.006663211815206471,
            "AUC": 0.8942884858450088
          }
        }
      }
    },
    "kept_new_candidates": [],
    "global_best_changed": false,
    "best_new_family": "route_q4_retrieval",
    "findings": [
      "All eight route-restricted retrainings have lower macroAUC point estimates than both matching all-route and original recurrence.",
      "Seven all-route conditional intervals are entirely negative; q4 retrieval-only crosses zero but misses the predeclared near-reference queue criterion. This is not proof of zero effect.",
      "Fixed-checkpoint removal of residual correction did not predict a retraining gain: attention-only retraining regressed in both selected source/query-count conditions.",
      "A second cycle remains supported by the earlier matched q4 R1/R2 result; these failures constrain rank16 additive correction sources and routes, not recurrence itself."
    ],
    "next_hypothesis": "Joint bounded mixing of entry and first-work states versus a parameter-matched sample-independent gate; channel and per-query granularity in q4/q8",
    "confidence": "One training seed; checkpoint-conditional user intervals, selection uncorrected; no equivalence or seed-robustness claim",
    "test_evaluated": false,
    "artifact_sha256": {
      "workspace/qformer-loop/taobao-transfer/recurrence-query-routes/reports/summary.json": "ea917954b3ed50ddce14b54defe120f5c679f89c95c497f500791d8ef0563e26",
      "workspace/qformer-loop/taobao-transfer/recurrence-query-routes/reports/paired-bootstrap.json": "790a177fb101845d98981dff8543f6b4106473521386a68c933ca23e1b4c8ff9",
      "workspace/qformer-loop/taobao-transfer/recurrence-query-routes/reports/final-audit.json": "6d34f910d60b01baffcc964cd6256bdb4c7a5623b39208b49fcc1ca35c131a88",
      "workspace/qformer-loop/taobao-transfer/recurrence-query-routes/reports/analysis-plan.json": "3e50c4491cfb0058d5cb8f648e5583c4eda60c50f5dbcf65592d8ef9b648081d"
    }
  },
  "decision": "discard",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-query-routes/reports/interpretation.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-query-routes/reports/analysis-plan-review.json"
  ]
}
```

</details>
