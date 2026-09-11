## 255. segment-read--second_suffix--s20262027

- 假设：Equal-count second_suffix stored-history support may change useful two-round retrieval conditions; no pure Q2 mediation claim.
- 判定：discard
- 原因：Macro delta -0.002208599,CI95 [-0.005286403531037203, 0.0005428426213684278],latency ratio 1.007043. Fails positive and near-candidate quality policy; preserve negative evidence.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "segment-read--second_suffix--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-10T14:13:03.864962+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "second_suffix",
  "code_id": "cf7b0840f7d3f28bb5451ac3cc4d9491bfcf60c5a00b32fc75f78b7e407fc0de",
  "hypothesis": "Equal-count second_suffix stored-history support may change useful two-round retrieval conditions; no pure Q2 mediation claim.",
  "change": "{\"model\": \"QFormerCross8SegmentRead\", \"maturity_mode\": \"mature\", \"segment_placement\": \"second\", \"segment_selection\": \"suffix\", \"work_steps\": 2, \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8SegmentRead",
    "maturity_mode": "mature",
    "segment_placement": "second",
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
      "value": 0.7648863521646126,
      "delta": -0.0022085991502479307
    },
    "per_task": {
      "is_click": {
        "logloss": 0.1841371524167702,
        "AUC": 0.6514759409827937
      },
      "cart": {
        "logloss": 0.021998538623062405,
        "AUC": 0.7866050236482446
      },
      "fav": {
        "logloss": 0.016756179518760447,
        "AUC": 0.8034126172519078
      },
      "buy": {
        "logloss": 0.009672712515046058,
        "AUC": 0.8180518267755041
      }
    },
    "macro_source_logloss": 0.05814114576840978,
    "source_metric_comparison": {
      "is_click": {
        "source_AUC": 0.6514759409827937,
        "computed_AUC": 0.6514759409827937,
        "auc_difference": 0.0,
        "source_logloss": 0.1841371524167702,
        "computed_bce": 0.18413715241677384,
        "bce_difference": 3.635980405647388e-15
      },
      "cart": {
        "source_AUC": 0.7866050236482446,
        "computed_AUC": 0.7866050236482446,
        "auc_difference": 0.0,
        "source_logloss": 0.021998538623062405,
        "computed_bce": 0.021998433484916865,
        "bce_difference": -1.0513814554038992e-07
      },
      "fav": {
        "source_AUC": 0.8034126172519078,
        "computed_AUC": 0.8034126172519078,
        "auc_difference": 0.0,
        "source_logloss": 0.016756179518760447,
        "computed_bce": 0.016756179519200026,
        "bce_difference": 4.395789288125229e-13
      },
      "buy": {
        "source_AUC": 0.8180518267755041,
        "computed_AUC": 0.8180518267755041,
        "auc_difference": 0.0,
        "source_logloss": 0.009672712515046058,
        "computed_bce": 0.009672712524256687,
        "bce_difference": 9.210628787448272e-12
      }
    },
    "matched_parent_effect": {
      "coefficients": {
        "mature_r2": -1.0,
        "cross8_mature": 0.0,
        "first_prefix": 0.0,
        "first_suffix": 0.0,
        "first_even": 0.0,
        "first_odd": 0.0,
        "second_prefix": 0.0,
        "second_suffix": 1.0,
        "second_even": 0.0,
        "second_odd": 0.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": -0.0022085991502479307,
          "ci95": [
            -0.005286403531037203,
            0.0005428426213684278
          ]
        },
        "auc.is_click": {
          "estimate": 0.0015396477818042653,
          "ci95": [
            0.0008366518894125236,
            0.0022916074277259495
          ]
        },
        "auc.cart": {
          "estimate": -0.005829246709099123,
          "ci95": [
            -0.010553606111170639,
            -0.00036642701286202314
          ]
        },
        "auc.fav": {
          "estimate": 0.0009491268768597738,
          "ci95": [
            -0.0052449832741405635,
            0.0066461098154351715
          ]
        },
        "auc.buy": {
          "estimate": -0.005493924550556639,
          "ci95": [
            -0.014597903717881455,
            0.00213849977876496
          ]
        },
        "bce.is_click": {
          "estimate": -0.00025449385101283806,
          "ci95": [
            -0.00032493764965856807,
            -0.00018714078947623599
          ]
        },
        "bce.cart": {
          "estimate": 0.0001704791154828328,
          "ci95": [
            6.222400218317036e-05,
            0.00027175168408430394
          ]
        },
        "bce.fav": {
          "estimate": -4.1978175152307395e-05,
          "ci95": [
            -0.00016902866774246364,
            0.00012254757017017467
          ]
        },
        "bce.buy": {
          "estimate": 2.891058753098398e-05,
          "ci95": [
            -4.561350452713655e-05,
            0.0001274661146635064
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": 0.007368894490903388,
          "ci95": [
            0.0017905161068967252,
            0.012034148343672646
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": 0.0005905209049444915,
          "ci95": [
            -0.005033489789023879,
            0.006961648975959505
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": 0.0070335723323609045,
          "ci95": [
            -0.0009497098705209814,
            0.016264298787758147
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": -0.006778373585958897,
          "ci95": [
            -0.014214458084744693,
            0.0013657477775693727
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": -0.0003353221585424837,
          "ci95": [
            -0.009261573371109383,
            0.00999654889295813
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": 0.006443051427416413,
          "ci95": [
            -0.0028611864164546754,
            0.01737707957431701
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": -0.00042497296649567085,
          "ci95": [
            -0.0005446030733780674,
            -0.0003067831302930007
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": -0.00021251567586053066,
          "ci95": [
            -0.000401468332899602,
            -6.527279874381963e-05
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": -0.00028340443854382204,
          "ci95": [
            -0.00040535213237859235,
            -0.00017110659779893129
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": 0.00021245729063514018,
          "ci95": [
            3.772112974335107e-05,
            0.0003667029778199858
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": 0.0001415685279518488,
          "ci95": [
            -2.7332710969360306e-06,
            0.00026007234744314156
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": -7.088876268329138e-05,
          "ci95": [
            -0.00023881878646076713,
            0.00010253269396835481
          ]
        }
      }
    },
    "all_prespecified_effects_involving_candidate": {
      "second_suffix__minus__mature_r2": {
        "coefficients": {
          "mature_r2": -1.0,
          "cross8_mature": 0.0,
          "first_prefix": 0.0,
          "first_suffix": 0.0,
          "first_even": 0.0,
          "first_odd": 0.0,
          "second_prefix": 0.0,
          "second_suffix": 1.0,
          "second_even": 0.0,
          "second_odd": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.0022085991502479307,
            "ci95": [
              -0.005286403531037203,
              0.0005428426213684278
            ]
          },
          "auc.is_click": {
            "estimate": 0.0015396477818042653,
            "ci95": [
              0.0008366518894125236,
              0.0022916074277259495
            ]
          },
          "auc.cart": {
            "estimate": -0.005829246709099123,
            "ci95": [
              -0.010553606111170639,
              -0.00036642701286202314
            ]
          },
          "auc.fav": {
            "estimate": 0.0009491268768597738,
            "ci95": [
              -0.0052449832741405635,
              0.0066461098154351715
            ]
          },
          "auc.buy": {
            "estimate": -0.005493924550556639,
            "ci95": [
              -0.014597903717881455,
              0.00213849977876496
            ]
          },
          "bce.is_click": {
            "estimate": -0.00025449385101283806,
            "ci95": [
              -0.00032493764965856807,
              -0.00018714078947623599
            ]
          },
          "bce.cart": {
            "estimate": 0.0001704791154828328,
            "ci95": [
              6.222400218317036e-05,
              0.00027175168408430394
            ]
          },
          "bce.fav": {
            "estimate": -4.1978175152307395e-05,
            "ci95": [
              -0.00016902866774246364,
              0.00012254757017017467
            ]
          },
          "bce.buy": {
            "estimate": 2.891058753098398e-05,
            "ci95": [
              -4.561350452713655e-05,
              0.0001274661146635064
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.007368894490903388,
            "ci95": [
              0.0017905161068967252,
              0.012034148343672646
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.0005905209049444915,
            "ci95": [
              -0.005033489789023879,
              0.006961648975959505
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.0070335723323609045,
            "ci95": [
              -0.0009497098705209814,
              0.016264298787758147
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.006778373585958897,
            "ci95": [
              -0.014214458084744693,
              0.0013657477775693727
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.0003353221585424837,
            "ci95": [
              -0.009261573371109383,
              0.00999654889295813
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.006443051427416413,
            "ci95": [
              -0.0028611864164546754,
              0.01737707957431701
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00042497296649567085,
            "ci95": [
              -0.0005446030733780674,
              -0.0003067831302930007
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.00021251567586053066,
            "ci95": [
              -0.000401468332899602,
              -6.527279874381963e-05
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.00028340443854382204,
            "ci95": [
              -0.00040535213237859235,
              -0.00017110659779893129
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.00021245729063514018,
            "ci95": [
              3.772112974335107e-05,
              0.0003667029778199858
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.0001415685279518488,
            "ci95": [
              -2.7332710969360306e-06,
              0.00026007234744314156
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -7.088876268329138e-05,
            "ci95": [
              -0.00023881878646076713,
              0.00010253269396835481
            ]
          }
        }
      },
      "second_suffix__minus__cross8_mature": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": -1.0,
          "first_prefix": 0.0,
          "first_suffix": 0.0,
          "first_even": 0.0,
          "first_odd": 0.0,
          "second_prefix": 0.0,
          "second_suffix": 1.0,
          "second_even": 0.0,
          "second_odd": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.005479932069152732,
            "ci95": [
              0.00267475260652567,
              0.008257294156149446
            ]
          },
          "auc.is_click": {
            "estimate": 0.001539170839828019,
            "ci95": [
              0.0006503205111799304,
              0.0023168230620326047
            ]
          },
          "auc.cart": {
            "estimate": -0.0026017117228713804,
            "ci95": [
              -0.007311275542800266,
              0.002088678564051751
            ]
          },
          "auc.fav": {
            "estimate": 0.006910606224458271,
            "ci95": [
              0.0009959388864489939,
              0.01232576243748743
            ]
          },
          "auc.buy": {
            "estimate": 0.01607166293519602,
            "ci95": [
              0.008177942919785856,
              0.024213641005266195
            ]
          },
          "bce.is_click": {
            "estimate": -0.00018078543181751505,
            "ci95": [
              -0.00024655527814788913,
              -0.00010657424783043005
            ]
          },
          "bce.cart": {
            "estimate": 0.00016402184783783497,
            "ci95": [
              5.3278159445999034e-05,
              0.0002739659256098778
            ]
          },
          "bce.fav": {
            "estimate": -0.00021719966225302104,
            "ci95": [
              -0.00032122322984738774,
              -0.00010564737661806879
            ]
          },
          "bce.buy": {
            "estimate": -0.00033467780418144216,
            "ci95": [
              -0.00042309829680753824,
              -0.0002516456846187612
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.0041408825626993995,
            "ci95": [
              -0.0006431830670680511,
              0.008910810230111837
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.005371435384630252,
            "ci95": [
              -0.010813771768040053,
              0.0003583804987383014
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.014532492095368,
            "ci95": [
              -0.02251583284221669,
              -0.006608619678593228
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.009512317947329652,
            "ci95": [
              -0.01667581591679861,
              -0.002008325523849481
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.0186733746580674,
            "ci95": [
              -0.027370542785557107,
              -0.009791379772903388
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.009161056710737747,
            "ci95": [
              -0.01889476193386073,
              0.00024399380632200094
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00034480727965535,
            "ci95": [
              -0.00047705089267895834,
              -0.00020948785907691354
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 3.6414230435505984e-05,
            "ci95": [
              -8.772293581686086e-05,
              0.00015457371062155653
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.0001538923723639271,
            "ci95": [
              4.1169404892971214e-05,
              0.0002650399025953157
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.000381221510090856,
            "ci95": [
              0.00022355953337737871,
              0.0005254428214064803
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.0004986996520192771,
            "ci95": [
              0.0003620704436214579,
              0.0006431935446327684
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.00011747814192842113,
            "ci95": [
              -8.407357604949461e-06,
              0.00026757350185967676
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
      "second_suffix__minus__second_even": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "first_prefix": 0.0,
          "first_suffix": 0.0,
          "first_even": 0.0,
          "first_odd": 0.0,
          "second_prefix": 0.0,
          "second_suffix": 1.0,
          "second_even": -1.0,
          "second_odd": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.00010083648693942626,
            "ci95": [
              -0.003079744334956494,
              0.002707492664425175
            ]
          },
          "auc.is_click": {
            "estimate": 0.0006988540871134541,
            "ci95": [
              -0.00015809582822501057,
              0.0013822795880535142
            ]
          },
          "auc.cart": {
            "estimate": -0.009766338566557287,
            "ci95": [
              -0.015685506930360537,
              -0.00416531129267691
            ]
          },
          "auc.fav": {
            "estimate": 0.009602274718439774,
            "ci95": [
              0.0028667519291044412,
              0.015304290590478305
            ]
          },
          "auc.buy": {
            "estimate": -0.0009381361867536464,
            "ci95": [
              -0.009202163893434456,
              0.0068948194985855825
            ]
          },
          "bce.is_click": {
            "estimate": -0.0002456671973943836,
            "ci95": [
              -0.0003038216120097081,
              -0.0001683420999026723
            ]
          },
          "bce.cart": {
            "estimate": 4.396430488151984e-05,
            "ci95": [
              -7.043950611951107e-05,
              0.00017785676426593198
            ]
          },
          "bce.fav": {
            "estimate": -0.0003244391305631533,
            "ci95": [
              -0.0004233656177677557,
              -0.00020088949187925705
            ]
          },
          "bce.buy": {
            "estimate": -0.00012175927965227783,
            "ci95": [
              -0.0002107982944347192,
              -3.131779337126365e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.01046519265367074,
            "ci95": [
              0.0046836095714160786,
              0.01619425555474997
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.00890342063132632,
            "ci95": [
              -0.014539257024247326,
              -0.0020237967061377403
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.0016369902738671005,
            "ci95": [
              -0.006338484943959984,
              0.0099791988453838
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.01936861328499706,
            "ci95": [
              -0.02791100756412916,
              -0.010683963654653472
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.00882820237980364,
            "ci95": [
              -0.019039218454404407,
              0.00171851297903729
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.01054041090519342,
            "ci95": [
              0.00038487193414386844,
              0.020201215804786808
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.0002896315022759034,
            "ci95": [
              -0.00043397735813942875,
              -0.00015125596704772105
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 7.877193316876974e-05,
            "ci95": [
              -4.131618452792229e-05,
              0.00019628749096262988
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.00012390791774210576,
            "ci95": [
              -0.00023496035482059802,
              -1.837742467798043e-05
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.00036840343544467316,
            "ci95": [
              0.00019673639229432686,
              0.0005343344931928816
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00016572358453379767,
            "ci95": [
              1.3585052007230368e-05,
              0.00032273652681112804
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.0002026798509108755,
            "ci95": [
              -0.00033266066937213186,
              -5.460191130449551e-05
            ]
          }
        }
      },
      "second_suffix__minus__second_odd": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "first_prefix": 0.0,
          "first_suffix": 0.0,
          "first_even": 0.0,
          "first_odd": 0.0,
          "second_prefix": 0.0,
          "second_suffix": 1.0,
          "second_even": 0.0,
          "second_odd": -1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.0015648440420851784,
            "ci95": [
              -0.0009105712924647728,
              0.004029780782590069
            ]
          },
          "auc.is_click": {
            "estimate": 0.0022015471607917947,
            "ci95": [
              0.0014452504905629632,
              0.002916381786628638
            ]
          },
          "auc.cart": {
            "estimate": -0.004435776119496415,
            "ci95": [
              -0.00886425620223635,
              2.9811838997075268e-05
            ]
          },
          "auc.fav": {
            "estimate": 0.01712337415680254,
            "ci95": [
              0.011009549369418906,
              0.02255243059113718
            ]
          },
          "auc.buy": {
            "estimate": -0.008629769029757206,
            "ci95": [
              -0.015176990883151945,
              -0.0022374498225030967
            ]
          },
          "bce.is_click": {
            "estimate": -0.0001740300916721893,
            "ci95": [
              -0.00024151517820210355,
              -0.00010347299743723035
            ]
          },
          "bce.cart": {
            "estimate": 0.00012852787207683417,
            "ci95": [
              2.445468023747075e-05,
              0.00023301226349560142
            ]
          },
          "bce.fav": {
            "estimate": -0.0003814362641228995,
            "ci95": [
              -0.00048243483668792577,
              -0.00026102215073620067
            ]
          },
          "bce.buy": {
            "estimate": -9.961199321215607e-05,
            "ci95": [
              -0.00016748760222993204,
              -2.5230126214515795e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.00663732328028821,
            "ci95": [
              0.0020641429810101617,
              0.010990234724450095
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.014921826996010745,
            "ci95": [
              -0.020298932888257726,
              -0.008947165611920159
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.010831316190549,
            "ci95": [
              0.0046088813241187635,
              0.017591444482179235
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.021559150276298955,
            "ci95": [
              -0.028474628116762928,
              -0.013799254524319567
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.004193992910260791,
            "ci95": [
              -0.0034276378931024807,
              0.012123707185929977
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.025753143186559746,
            "ci95": [
              0.016844253600845564,
              0.034523696491557565
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00030255796374902347,
            "ci95": [
              -0.00042460602697535967,
              -0.00017719864791988194
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.00020740617245071022,
            "ci95": [
              5.439212576462223e-05,
              0.000333280175165767
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -7.441809846003324e-05,
            "ci95": [
              -0.00017816364627465676,
              2.674234843920809e-05
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.0005099641361997337,
            "ci95": [
              0.0003396337836032525,
              0.0006565794949955708
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00022813986528899023,
            "ci95": [
              0.00010005404145255839,
              0.00034537937319775334
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.00028182427091074345,
            "ci95": [
              -0.0004153633393670445,
              -0.00013818384140454405
            ]
          }
        }
      }
    },
    "parameter_count": 51724260,
    "initializer_sha256": "fd5f866c35c5e29b37841aeed070f6ef5259c773de173df7549d149f87a8c09e",
    "optimizer_layout_sha256": "9e95235cbaf06044f94be5035bd1a3b1477ffed5b1c46d0b584fc98438db86c7",
    "formal_worker_cost": {
      "wall_seconds": 2511.607097297907,
      "cpu_seconds": 3031.608810618,
      "peak_rss_bytes": 17054855168,
      "gpu_peak_bytes": 28369223680,
      "samples": 2301
    },
    "smoke_worker_cost": {
      "wall_seconds": 146.33005591481924,
      "cpu_seconds": 160.821220856,
      "peak_rss_bytes": 16945524736,
      "gpu_peak_bytes": 29572988928,
      "samples": 134
    },
    "complete_recorded_costs": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2511.607097297907,
        "cpu_seconds": 3031.608810618,
        "peak_rss_bytes": 17054855168,
        "gpu_peak_bytes": 28369223680,
        "samples": 2301
      },
      "formal_runner": {
        "wall_seconds": 2507.0435846969485,
        "cpu_seconds": 2914.699161456,
        "peak_process_rss_bytes": 5207805952
      },
      "fit_including_validation": {
        "wall_seconds": 2383.804154254496,
        "cpu_seconds": 2742.398085055,
        "peak_process_rss_bytes": 5207805952
      },
      "prediction_export": {
        "wall_seconds": 118.1188790872693,
        "cpu_seconds": 166.57568426100033,
        "peak_process_rss_bytes": 5207805952
      },
      "smoke_worker": {
        "wall_seconds": 146.33005591481924,
        "cpu_seconds": 160.821220856,
        "peak_rss_bytes": 16945524736,
        "gpu_peak_bytes": 29572988928,
        "samples": 134
      },
      "smoke_runner": {
        "wall_seconds": 141.8813523054123,
        "cpu_seconds": 140.110500597,
        "peak_process_rss_bytes": 3242672128
      },
      "latency_constructor": {
        "wall_seconds": 0.7430275455117226,
        "cpu_seconds": 0.8157588590000024,
        "peak_rss_bytes": 2092183552
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.10594730824232101,
        "cpu_seconds": 0.10527239299999991,
        "peak_rss_bytes": 2092183552
      },
      "latency_peak_cuda_bytes": 1281155072,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "forward_latency": {
      "mean_ms": 14.253929615020752,
      "p50_ms": 14.242447853088379,
      "p95_ms": 14.319464063644409,
      "batch_size": 1024,
      "warmups": 10,
      "iterations": 40
    },
    "latency_ratio_to_parent": 1.0070432099110116,
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
  "decision_reason": "Macro delta -0.002208599,CI95 [-0.005286403531037203, 0.0005428426213684278],latency ratio 1.007043. Fails positive and near-candidate quality policy; preserve negative evidence.",
  "decision_scope": "Root queue disposition only; no released-incumbent promotion or supported lesson.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_suffix",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_suffix/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_suffix/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_suffix/Taobao_Action/second_suffix.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/paired-analysis/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/latency/completion.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_suffix",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_suffix/result.json",
    "result_sha256": "2784289062f3197dc82eaa446a39c5db355939b25b78d62a4021de5430901b8a",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_suffix/manifest.json",
    "manifest_sha256": "8892ecd991f995a539b1cf1a214f36c125166622f5d5f2fcd5695443d980db47",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_suffix/Taobao_Action/second_suffix.model",
    "checkpoint_sha256": "096245302aa18d32f3a02550058be29f0572dc2da74baca68d879c3cbcd8f2ad",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/paired-analysis/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
