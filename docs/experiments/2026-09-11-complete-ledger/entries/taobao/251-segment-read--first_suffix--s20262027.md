## 251. segment-read--first_suffix--s20262027

- 假设：Equal-count first_suffix stored-history support may change useful two-round retrieval conditions; no pure Q2 mediation claim.
- 判定：discard
- 原因：Macro delta -0.004871842,CI95 [-0.007814593774694572, -0.0022479055763103472],latency ratio 1.006671. Fails positive and near-candidate quality policy; preserve negative evidence.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "segment-read--first_suffix--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-10T14:13:03.864962+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "first_suffix",
  "code_id": "cf7b0840f7d3f28bb5451ac3cc4d9491bfcf60c5a00b32fc75f78b7e407fc0de",
  "hypothesis": "Equal-count first_suffix stored-history support may change useful two-round retrieval conditions; no pure Q2 mediation claim.",
  "change": "{\"model\": \"QFormerCross8SegmentRead\", \"maturity_mode\": \"mature\", \"segment_placement\": \"first\", \"segment_selection\": \"suffix\", \"work_steps\": 2, \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8SegmentRead",
    "maturity_mode": "mature",
    "segment_placement": "first",
    "segment_selection": "suffix",
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
      "value": 0.7622231089823723,
      "delta": -0.0048718423324882465
    },
    "per_task": {
      "is_click": {
        "logloss": 0.1844615439884963,
        "AUC": 0.6478032949898368
      },
      "cart": {
        "logloss": 0.02180907582680886,
        "AUC": 0.7916049669099392
      },
      "fav": {
        "logloss": 0.017035103980372057,
        "AUC": 0.7912696504690258
      },
      "buy": {
        "logloss": 0.00974175191026024,
        "AUC": 0.8182145235606872
      }
    },
    "macro_source_logloss": 0.05826186892648436,
    "source_metric_comparison": {
      "is_click": {
        "source_AUC": 0.6478032949898368,
        "computed_AUC": 0.6478032949898368,
        "auc_difference": 0.0,
        "source_logloss": 0.1844615439884963,
        "computed_bce": 0.184461543988497,
        "bce_difference": 7.216449660063518e-16
      },
      "cart": {
        "source_AUC": 0.7916049669099392,
        "computed_AUC": 0.7916049669099392,
        "auc_difference": 0.0,
        "source_logloss": 0.02180907582680886,
        "computed_bce": 0.02180907589273904,
        "bce_difference": 6.593017798373069e-11
      },
      "fav": {
        "source_AUC": 0.7912696504690258,
        "computed_AUC": 0.7912696504690258,
        "auc_difference": 0.0,
        "source_logloss": 0.017035103980372057,
        "computed_bce": 0.017035104003313695,
        "bce_difference": 2.294163817251338e-11
      },
      "buy": {
        "source_AUC": 0.8182145235606872,
        "computed_AUC": 0.8182145235606872,
        "auc_difference": 0.0,
        "source_logloss": 0.00974175191026024,
        "computed_bce": 0.009741751911220122,
        "bce_difference": 9.598814798561506e-13
      }
    },
    "matched_parent_effect": {
      "coefficients": {
        "mature_r2": -1.0,
        "cross8_mature": 0.0,
        "first_prefix": 0.0,
        "first_suffix": 1.0,
        "first_even": 0.0,
        "first_odd": 0.0,
        "second_prefix": 0.0,
        "second_suffix": 0.0,
        "second_even": 0.0,
        "second_odd": 0.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": -0.0048718423324882465,
          "ci95": [
            -0.007814593774694572,
            -0.0022479055763103472
          ]
        },
        "auc.is_click": {
          "estimate": -0.002132998211152648,
          "ci95": [
            -0.0028474367739627195,
            -0.0013668034674844622
          ]
        },
        "auc.cart": {
          "estimate": -0.0008293034474045058,
          "ci95": [
            -0.005527889882315129,
            0.004551494339934897
          ]
        },
        "auc.fav": {
          "estimate": -0.011193839906022296,
          "ci95": [
            -0.016665694895974238,
            -0.005605718100028959
          ]
        },
        "auc.buy": {
          "estimate": -0.005331227765373536,
          "ci95": [
            -0.014008184216330752,
            0.002125705300489889
          ]
        },
        "bce.is_click": {
          "estimate": 6.989772071033795e-05,
          "ci95": [
            -4.0976609426215915e-06,
            0.0001338409392648135
          ]
        },
        "bce.cart": {
          "estimate": -1.8878476694992458e-05,
          "ci95": [
            -0.00012901116626690064,
            8.265717667631414e-05
          ]
        },
        "bce.fav": {
          "estimate": 0.00023694630896136173,
          "ci95": [
            0.00010757288496807933,
            0.00036574142438543483
          ]
        },
        "bce.buy": {
          "estimate": 9.794997449441914e-05,
          "ci95": [
            1.9401607871777883e-05,
            0.00018610299109863502
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": -0.0013036947637481422,
          "ci95": [
            -0.006675898432860297,
            0.0032839673024846437
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": 0.009060841694869648,
          "ci95": [
            0.0036496411905315913,
            0.014410388923289672
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": 0.0031982295542208883,
          "ci95": [
            -0.004312648739359972,
            0.01208399864841736
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": 0.01036453645861779,
          "ci95": [
            0.003395185613108923,
            0.018785884120675063
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": 0.0045019243179690305,
          "ci95": [
            -0.00392631732102385,
            0.014330076692209811
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": -0.00586261214064876,
          "ci95": [
            -0.015740466745824654,
            0.004112195324700389
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": 8.877619740533041e-05,
          "ci95": [
            -3.925529659145974e-05,
            0.00020884609028288655
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": -0.00016704858825102378,
          "ci95": [
            -0.0003130869910012571,
            -2.9052735653935642e-05
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": -2.8052253784081188e-05,
          "ci95": [
            -0.00015098763562273343,
            7.470498096013713e-05
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": -0.0002558247856563542,
          "ci95": [
            -0.0004107505001797582,
            -0.00010741957396214692
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": -0.0001168284511894116,
          "ci95": [
            -0.0002482021689887572,
            7.827178616171609e-06
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": 0.0001389963344669426,
          "ci95": [
            -2.1813038097825668e-05,
            0.0002934386024873066
          ]
        }
      }
    },
    "all_prespecified_effects_involving_candidate": {
      "first_suffix__minus__mature_r2": {
        "coefficients": {
          "mature_r2": -1.0,
          "cross8_mature": 0.0,
          "first_prefix": 0.0,
          "first_suffix": 1.0,
          "first_even": 0.0,
          "first_odd": 0.0,
          "second_prefix": 0.0,
          "second_suffix": 0.0,
          "second_even": 0.0,
          "second_odd": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.0048718423324882465,
            "ci95": [
              -0.007814593774694572,
              -0.0022479055763103472
            ]
          },
          "auc.is_click": {
            "estimate": -0.002132998211152648,
            "ci95": [
              -0.0028474367739627195,
              -0.0013668034674844622
            ]
          },
          "auc.cart": {
            "estimate": -0.0008293034474045058,
            "ci95": [
              -0.005527889882315129,
              0.004551494339934897
            ]
          },
          "auc.fav": {
            "estimate": -0.011193839906022296,
            "ci95": [
              -0.016665694895974238,
              -0.005605718100028959
            ]
          },
          "auc.buy": {
            "estimate": -0.005331227765373536,
            "ci95": [
              -0.014008184216330752,
              0.002125705300489889
            ]
          },
          "bce.is_click": {
            "estimate": 6.989772071033795e-05,
            "ci95": [
              -4.0976609426215915e-06,
              0.0001338409392648135
            ]
          },
          "bce.cart": {
            "estimate": -1.8878476694992458e-05,
            "ci95": [
              -0.00012901116626690064,
              8.265717667631414e-05
            ]
          },
          "bce.fav": {
            "estimate": 0.00023694630896136173,
            "ci95": [
              0.00010757288496807933,
              0.00036574142438543483
            ]
          },
          "bce.buy": {
            "estimate": 9.794997449441914e-05,
            "ci95": [
              1.9401607871777883e-05,
              0.00018610299109863502
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.0013036947637481422,
            "ci95": [
              -0.006675898432860297,
              0.0032839673024846437
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.009060841694869648,
            "ci95": [
              0.0036496411905315913,
              0.014410388923289672
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.0031982295542208883,
            "ci95": [
              -0.004312648739359972,
              0.01208399864841736
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.01036453645861779,
            "ci95": [
              0.003395185613108923,
              0.018785884120675063
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.0045019243179690305,
            "ci95": [
              -0.00392631732102385,
              0.014330076692209811
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.00586261214064876,
            "ci95": [
              -0.015740466745824654,
              0.004112195324700389
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 8.877619740533041e-05,
            "ci95": [
              -3.925529659145974e-05,
              0.00020884609028288655
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.00016704858825102378,
            "ci95": [
              -0.0003130869910012571,
              -2.9052735653935642e-05
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -2.8052253784081188e-05,
            "ci95": [
              -0.00015098763562273343,
              7.470498096013713e-05
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.0002558247856563542,
            "ci95": [
              -0.0004107505001797582,
              -0.00010741957396214692
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -0.0001168284511894116,
            "ci95": [
              -0.0002482021689887572,
              7.827178616171609e-06
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.0001389963344669426,
            "ci95": [
              -2.1813038097825668e-05,
              0.0002934386024873066
            ]
          }
        }
      },
      "first_suffix__minus__cross8_mature": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": -1.0,
          "first_prefix": 0.0,
          "first_suffix": 1.0,
          "first_even": 0.0,
          "first_odd": 0.0,
          "second_prefix": 0.0,
          "second_suffix": 0.0,
          "second_even": 0.0,
          "second_odd": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.0028166888869124163,
            "ci95": [
              -8.375742450305935e-05,
              0.0056259086933696375
            ]
          },
          "auc.is_click": {
            "estimate": -0.0021334751531288942,
            "ci95": [
              -0.0029332160893469406,
              -0.0014255414193129368
            ]
          },
          "auc.cart": {
            "estimate": 0.0023982315388232367,
            "ci95": [
              -0.0016852360550182965,
              0.006072499208322361
            ]
          },
          "auc.fav": {
            "estimate": -0.005232360558423799,
            "ci95": [
              -0.010631954381456263,
              0.0006541073696937277
            ]
          },
          "auc.buy": {
            "estimate": 0.01623435972037912,
            "ci95": [
              0.00802808472640564,
              0.02425337292136036
            ]
          },
          "bce.is_click": {
            "estimate": 0.00014360613990566096,
            "ci95": [
              7.400985226368312e-05,
              0.0002130504806963043
            ]
          },
          "bce.cart": {
            "estimate": -2.533574433999028e-05,
            "ci95": [
              -0.00012001238515524587,
              6.964817065335646e-05
            ]
          },
          "bce.fav": {
            "estimate": 6.17248218606481e-05,
            "ci95": [
              -3.3498673873164074e-05,
              0.00015325715373871345
            ]
          },
          "bce.buy": {
            "estimate": -0.000265638417218007,
            "ci95": [
              -0.0003654336355307994,
              -0.0001819370038361067
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.004531706691952131,
            "ci95": [
              -0.008396779712033467,
              -0.0003516765749718891
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.0030988854052949044,
            "ci95": [
              -0.0026955287799984213,
              0.00852965624781638
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.018367834873508015,
            "ci95": [
              -0.02658562090210643,
              -0.009902727526343191
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.007630592097247035,
            "ci95": [
              0.0007970847595662275,
              0.014382973056535736
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.013836128181555885,
            "ci95": [
              -0.02347730024489409,
              -0.004878867390108186
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.02146672027880292,
            "ci95": [
              -0.031966025361687564,
              -0.011369575421271266
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 0.00016894188424565124,
            "ci95": [
              3.815116136768483e-05,
              0.00029268884323028875
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 8.188131804501286e-05,
            "ci95": [
              -3.9735625169724614e-05,
              0.00020061493368642757
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.00040924455712366796,
            "ci95": [
              0.00029119305053022635,
              0.0005203872675414034
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -8.706056620063837e-05,
            "ci95": [
              -0.00021110317571330326,
              5.787937845747741e-05
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00024030267287801672,
            "ci95": [
              0.00011651863208385319,
              0.0003826935706028147
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.0003273632390786551,
            "ci95": [
              0.0002091205937856896,
              0.00046662361446858533
            ]
          }
        }
      },
      "first_suffix__minus__second_suffix": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "first_prefix": 0.0,
          "first_suffix": 1.0,
          "first_even": 0.0,
          "first_odd": 0.0,
          "second_prefix": 0.0,
          "second_suffix": -1.0,
          "second_even": 0.0,
          "second_odd": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.0026632431822403158,
            "ci95": [
              -0.005277334359706552,
              8.642408780872287e-07
            ]
          },
          "auc.is_click": {
            "estimate": -0.0036726459929569133,
            "ci95": [
              -0.0044081200905215475,
              -0.002868550113723004
            ]
          },
          "auc.cart": {
            "estimate": 0.004999943261694617,
            "ci95": [
              0.00024071694700915645,
              0.01064317990483749
            ]
          },
          "auc.fav": {
            "estimate": -0.01214296678288207,
            "ci95": [
              -0.017443579367643965,
              -0.006303308865232137
            ]
          },
          "auc.buy": {
            "estimate": 0.00016269678518310293,
            "ci95": [
              -0.0071238491130952404,
              0.007933699241043788
            ]
          },
          "bce.is_click": {
            "estimate": 0.000324391571723176,
            "ci95": [
              0.0002560322487920895,
              0.00039215218478610005
            ]
          },
          "bce.cart": {
            "estimate": -0.00018935759217782525,
            "ci95": [
              -0.0003103563253193041,
              -7.581559297401725e-05
            ]
          },
          "bce.fav": {
            "estimate": 0.00027892448411366913,
            "ci95": [
              0.00017475110055412656,
              0.00037470906705869424
            ]
          },
          "bce.buy": {
            "estimate": 6.903938696343516e-05,
            "ci95": [
              -9.873159819262683e-06,
              0.00014956385551457828
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.00867258925465153,
            "ci95": [
              -0.014305540893364154,
              -0.003802595197881987
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.008470320789925156,
            "ci95": [
              0.0029330917432632527,
              0.013838706911615244
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.0038353427781400162,
            "ci95": [
              -0.011851274426234229,
              0.003626766917287412
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.017142910044576687,
            "ci95": [
              0.009605065842654003,
              0.024069612934105547
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.004837246476511514,
            "ci95": [
              -0.004672800804628852,
              0.014568127722054326
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.012305663568065173,
            "ci95": [
              -0.021484032037400975,
              -0.003645891856007714
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 0.0005137491639010013,
            "ci95": [
              0.0003834453106137465,
              0.0006456624578790471
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 4.546708760950688e-05,
            "ci95": [
              -7.414694927076417e-05,
              0.00015963018009307377
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.00025535218475974085,
            "ci95": [
              0.00014450067790837255,
              0.00036303497194906
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.0004682820762914944,
            "ci95": [
              -0.0006138340127828462,
              -0.0003174580666847408
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -0.0002583969791412604,
            "ci95": [
              -0.000411325127149297,
              -0.00011237377012508888
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.00020988509715023397,
            "ci95": [
              7.762433315015018e-05,
              0.00033623905912696564
            ]
          }
        }
      },
      "first_suffix__minus__first_even": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "first_prefix": 0.0,
          "first_suffix": 1.0,
          "first_even": -1.0,
          "first_odd": 0.0,
          "second_prefix": 0.0,
          "second_suffix": 0.0,
          "second_even": 0.0,
          "second_odd": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.00019675124358017881,
            "ci95": [
              -0.0029292809844941895,
              0.0025546791977236467
            ]
          },
          "auc.is_click": {
            "estimate": -0.0015714347549894425,
            "ci95": [
              -0.002408178604628064,
              -0.0007017020817533617
            ]
          },
          "auc.cart": {
            "estimate": -0.0006317754633531347,
            "ci95": [
              -0.005537828207562245,
              0.003772482041222841
            ]
          },
          "auc.fav": {
            "estimate": -0.003102819379271704,
            "ci95": [
              -0.009237177815508296,
              0.003098142178340654
            ]
          },
          "auc.buy": {
            "estimate": 0.004519024623293566,
            "ci95": [
              -0.002831710334785703,
              0.012343905082454625
            ]
          },
          "bce.is_click": {
            "estimate": 0.0002180255917979268,
            "ci95": [
              0.0001405986079854711,
              0.0002938985381602241
            ]
          },
          "bce.cart": {
            "estimate": 1.9101531152733442e-05,
            "ci95": [
              -9.366075631918845e-05,
              0.00014452755765200593
            ]
          },
          "bce.fav": {
            "estimate": -0.00020912925951409658,
            "ci95": [
              -0.0003351301943428814,
              -9.554368373702368e-05
            ]
          },
          "bce.buy": {
            "estimate": -0.00013822463542860833,
            "ci95": [
              -0.0002182011666706269,
              -5.591616723384871e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.0009396592916363078,
            "ci95": [
              -0.005447703300950937,
              0.003890827040897159
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.0015313846242822615,
            "ci95": [
              -0.004655132749669025,
              0.007915675402744262
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.0060904593782830085,
            "ci95": [
              -0.01401816922884008,
              0.0013137105501119866
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.0024710439159185693,
            "ci95": [
              -0.005247232767551099,
              0.010319281003640714
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.005150800086646701,
            "ci95": [
              -0.013985201628388207,
              0.003243617792355638
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.00762184400256527,
            "ci95": [
              -0.016945131311230897,
              0.0017535523649972711
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 0.00019892406064519336,
            "ci95": [
              4.6214535783670016e-05,
              0.0003399432030028673
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.0004271548513120234,
            "ci95": [
              0.0002861712109388825,
              0.0005642563360330962
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.0003562502272265351,
            "ci95": [
              0.00024025262269697172,
              0.00047049015820761476
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.00022823079066683002,
            "ci95": [
              5.906426143105449e-05,
              0.0003846694866169834
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00015732616658134177,
            "ci95": [
              1.8968118020833888e-05,
              0.0003012581403540774
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -7.090462408548825e-05,
            "ci95": [
              -0.00021456863064672145,
              6.924705232512517e-05
            ]
          }
        }
      },
      "first_suffix__minus__first_odd": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "first_prefix": 0.0,
          "first_suffix": 1.0,
          "first_even": 0.0,
          "first_odd": -1.0,
          "second_prefix": 0.0,
          "second_suffix": 0.0,
          "second_even": 0.0,
          "second_odd": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.006109761747662729,
            "ci95": [
              -0.009019188728277084,
              -0.0035315702830934657
            ]
          },
          "auc.is_click": {
            "estimate": -0.0012574508124980621,
            "ci95": [
              -0.0020830507050595415,
              -0.00046568744862094687
            ]
          },
          "auc.cart": {
            "estimate": -0.0022469015743855625,
            "ci95": [
              -0.006610549386567815,
              0.002486132090509292
            ]
          },
          "auc.fav": {
            "estimate": -0.014430457714335376,
            "ci95": [
              -0.020865620618596037,
              -0.007780583471790843
            ]
          },
          "auc.buy": {
            "estimate": -0.006504236889431914,
            "ci95": [
              -0.014616912392191072,
              0.0007220260193942539
            ]
          },
          "bce.is_click": {
            "estimate": -0.00018205039321714245,
            "ci95": [
              -0.00026184841810394027,
              -0.00011068183754537956
            ]
          },
          "bce.cart": {
            "estimate": 9.133320997639638e-06,
            "ci95": [
              -8.934368427796976e-05,
              0.00011029533736547596
            ]
          },
          "bce.fav": {
            "estimate": 0.00025140495102754165,
            "ci95": [
              0.000141354021652254,
              0.0003712023497266197
            ]
          },
          "bce.buy": {
            "estimate": 9.774494629154196e-05,
            "ci95": [
              2.147166339233251e-05,
              0.00018875959980949736
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.0009894507618875004,
            "ci95": [
              -0.0039010035761656374,
              0.005457627726279637
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.013173006901837314,
            "ci95": [
              0.006481312088216837,
              0.02002196445498243
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.0052467860769338515,
            "ci95": [
              -0.0021360535449441776,
              0.013670951501122184
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.012183556139949814,
            "ci95": [
              0.004117205769275656,
              0.020664059344402925
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.004257335315046351,
            "ci95": [
              -0.004007236214954887,
              0.013624425210822508
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.007926220824903463,
            "ci95": [
              -0.018247216073240678,
              0.0019602505298964056
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00019118371421478209,
            "ci95": [
              -0.0003160448601342524,
              -7.155404946442542e-05
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0004334553442446841,
            "ci95": [
              -0.0005723517575590418,
              -0.00029914992133860195
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0002797953395086844,
            "ci95": [
              -0.0003942476039269809,
              -0.00016815009448189314
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.000242271630029902,
            "ci95": [
              -0.0003946309209093298,
              -8.752263650036714e-05
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -8.861162529390232e-05,
            "ci95": [
              -0.00021248343026283346,
              3.4849928750768564e-05
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.0001536600047359997,
            "ci95": [
              1.2798332815017365e-05,
              0.0002913969442236105
            ]
          }
        }
      }
    },
    "parameter_count": 51724260,
    "initializer_sha256": "fd5f866c35c5e29b37841aeed070f6ef5259c773de173df7549d149f87a8c09e",
    "optimizer_layout_sha256": "3546519bfd702e680acf2f149bc2b97d52851a6e7e6a228d7e063b78f77c181f",
    "formal_worker_cost": {
      "wall_seconds": 2515.9312758892775,
      "cpu_seconds": 3025.838640644,
      "peak_rss_bytes": 16970461184,
      "gpu_peak_bytes": 28369223680,
      "samples": 2308
    },
    "smoke_worker_cost": {
      "wall_seconds": 142.06238438934088,
      "cpu_seconds": 157.28573032900002,
      "peak_rss_bytes": 16923885568,
      "gpu_peak_bytes": 29572988928,
      "samples": 130
    },
    "complete_recorded_costs": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2515.9312758892775,
        "cpu_seconds": 3025.838640644,
        "peak_rss_bytes": 16970461184,
        "gpu_peak_bytes": 28369223680,
        "samples": 2308
      },
      "formal_runner": {
        "wall_seconds": 2512.0273484811187,
        "cpu_seconds": 2911.2831492200003,
        "peak_process_rss_bytes": 5224308736
      },
      "fit_including_validation": {
        "wall_seconds": 2386.3059087842703,
        "cpu_seconds": 2737.348164384,
        "peak_process_rss_bytes": 5224308736
      },
      "prediction_export": {
        "wall_seconds": 120.38198284059763,
        "cpu_seconds": 168.29038888500008,
        "peak_process_rss_bytes": 5224308736
      },
      "smoke_worker": {
        "wall_seconds": 142.06238438934088,
        "cpu_seconds": 157.28573032900002,
        "peak_rss_bytes": 16923885568,
        "gpu_peak_bytes": 29572988928,
        "samples": 130
      },
      "smoke_runner": {
        "wall_seconds": 138.9994521290064,
        "cpu_seconds": 137.70408842400002,
        "peak_process_rss_bytes": 3218051072
      },
      "latency_constructor": {
        "wall_seconds": 0.7834699973464012,
        "cpu_seconds": 0.8559188819999992,
        "peak_rss_bytes": 2092183552
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.156565360724926,
        "cpu_seconds": 0.1553112210000016,
        "peak_rss_bytes": 2092183552
      },
      "latency_peak_cuda_bytes": 1281155072,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "forward_latency": {
      "mean_ms": 14.248657608032227,
      "p50_ms": 14.234943866729736,
      "p95_ms": 14.26779227256775,
      "batch_size": 1024,
      "warmups": 10,
      "iterations": 40
    },
    "latency_ratio_to_parent": 1.0066707414771279,
    "quality_candidate": false,
    "near_candidate": false
  },
  "validity": {
    "valid": true,
    "violations": [],
    "state": "passed",
    "quality_summary": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/quality-summary.json",
    "no_prediction_to_work_feedback": true,
    "test_evaluated": false,
    "infrastructure_limitations": [
      "Source metrics are identity/range/arithmetic checks, not recomputed AUC/BCE.",
      "Actual saved-array pairing is deferred to the paired analysis; row fingerprints cover original payload user/item/labels.",
      "Production cost receipts are sampled evidence, not new GPU measurements or hard resource guarantees.",
      "Maturity timestamp inference and click-availability assumptions remain unchanged.",
      "Launcher terminal/aggregate cleanup and cost missing; independent real worker terminals accepted via explicit recovery. Original pipeline absent before any measured phase."
    ],
    "recovered_terminal_evidence": {
      "completion_path": null,
      "completion": null,
      "launcher_exit_code": null,
      "launcher_aggregate_cost": null,
      "worker_costs": {
        "first_prefix": {
          "wall_seconds": 2515.3734920471907,
          "cpu_seconds": 3027.6236540930004,
          "peak_rss_bytes": 17074065408,
          "gpu_peak_bytes": 28369223680,
          "samples": 2307
        },
        "first_suffix": {
          "wall_seconds": 2515.9312758892775,
          "cpu_seconds": 3025.838640644,
          "peak_rss_bytes": 16970461184,
          "gpu_peak_bytes": 28369223680,
          "samples": 2308
        },
        "first_even": {
          "wall_seconds": 2515.3534044995904,
          "cpu_seconds": 3025.922370799,
          "peak_rss_bytes": 16983592960,
          "gpu_peak_bytes": 28371320832,
          "samples": 2306
        },
        "first_odd": {
          "wall_seconds": 2528.382029980421,
          "cpu_seconds": 3044.418955067,
          "peak_rss_bytes": 17071022080,
          "gpu_peak_bytes": 28371320832,
          "samples": 2317
        },
        "second_prefix": {
          "wall_seconds": 2519.057643711567,
          "cpu_seconds": 3030.651342093,
          "peak_rss_bytes": 17107640320,
          "gpu_peak_bytes": 28373417984,
          "samples": 2309
        },
        "second_suffix": {
          "wall_seconds": 2511.607097297907,
          "cpu_seconds": 3031.608810618,
          "peak_rss_bytes": 17054855168,
          "gpu_peak_bytes": 28369223680,
          "samples": 2301
        },
        "second_even": {
          "wall_seconds": 2531.232098005712,
          "cpu_seconds": 3039.49522237,
          "peak_rss_bytes": 17119875072,
          "gpu_peak_bytes": 28371320832,
          "samples": 2322
        },
        "second_odd": {
          "wall_seconds": 2538.3111224472523,
          "cpu_seconds": 3053.287257262,
          "peak_rss_bytes": 17063968768,
          "gpu_peak_bytes": 28369223680,
          "samples": 2329
        }
      },
      "recovery_path": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/recovered-worker-terminal.json",
      "recovery_sha256": "e8b859ea8595b1388ca0aecbcef90aaff14e8d86da5a6d28d71fead9dec21285",
      "scope": "All eight real worker/runner terminals verified independently. Launcher terminal and aggregate cleanup/cost are unavailable, not inferred successful; recorded workers absent or zombie. No model retry."
    },
    "source_and_input_identity_scope": "Original N full-input content SHA256 values are inherited through completed C quality and E authorization. Production runners checked input stats before/after their traversals and two original metadata hashes before them. This collector now checks 122 current stats and both original metadata hashes; it does not rehash Parquet or open checkpoint/prediction arrays. New E worker/smoke/effective-optimizer gates are evaluated from actual small receipts. Two references retain exact C quality records and producer completion hashes, with small source/artifact identity checks. Stat continuity cannot independently exclude content changes that preserve stat fields."
  },
  "decision": "discard",
  "decision_reason": "Macro delta -0.004871842,CI95 [-0.007814593774694572, -0.0022479055763103472],latency ratio 1.006671. Fails positive and near-candidate quality policy; preserve negative evidence.",
  "decision_scope": "Root queue disposition only; no released-incumbent promotion or supported lesson.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_suffix",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_suffix/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_suffix/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_suffix/Taobao_Action/first_suffix.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/paired-analysis/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/latency/completion.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_suffix",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_suffix/result.json",
    "result_sha256": "b1449afb1b9b5ae124a3336bb03477ff93c7e815d83d0dc954f45f5eafcdda9c",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_suffix/manifest.json",
    "manifest_sha256": "1c1d7a1b1b50f064b4434b81fa61cdca12c75c2c08d9bb73df2e748af91d174e",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_suffix/Taobao_Action/first_suffix.model",
    "checkpoint_sha256": "37b5f97a25815f0f8851d26d32b532576bc9a9683bbe6b410a3e09c344fd4dd2",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/paired-analysis/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
