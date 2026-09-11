## 257. segment-read--second_odd--s20262027

- 假设：Equal-count second_odd stored-history support may change useful two-round retrieval conditions; no pure Q2 mediation claim.
- 判定：discard
- 原因：Macro delta -0.003773443,CI95 [-0.0063889316344838635, -0.0012820873871544369],latency ratio 1.013405. Fails positive and near-candidate quality policy; preserve negative evidence.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "segment-read--second_odd--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-10T14:13:03.864962+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "second_odd",
  "code_id": "cf7b0840f7d3f28bb5451ac3cc4d9491bfcf60c5a00b32fc75f78b7e407fc0de",
  "hypothesis": "Equal-count second_odd stored-history support may change useful two-round retrieval conditions; no pure Q2 mediation claim.",
  "change": "{\"model\": \"QFormerCross8SegmentRead\", \"maturity_mode\": \"mature\", \"segment_placement\": \"second\", \"segment_selection\": \"odd\", \"work_steps\": 2, \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8SegmentRead",
    "maturity_mode": "mature",
    "segment_placement": "second",
    "segment_selection": "odd",
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
      "value": 0.7633215081225274,
      "delta": -0.003773443192333109
    },
    "per_task": {
      "is_click": {
        "logloss": 0.18431118250844458,
        "AUC": 0.6492743938220019
      },
      "cart": {
        "logloss": 0.021869903937152445,
        "AUC": 0.791040799767741
      },
      "fav": {
        "logloss": 0.017137615748316272,
        "AUC": 0.7862892430951053
      },
      "buy": {
        "logloss": 0.00977232403246592,
        "AUC": 0.8266815958052613
      }
    },
    "macro_source_logloss": 0.0582727565565948,
    "source_metric_comparison": {
      "is_click": {
        "source_AUC": 0.6492743938220019,
        "computed_AUC": 0.6492743938220019,
        "auc_difference": 0.0,
        "source_logloss": 0.18431118250844458,
        "computed_bce": 0.18431118250844603,
        "bce_difference": 1.4432899320127035e-15
      },
      "cart": {
        "source_AUC": 0.791040799767741,
        "computed_AUC": 0.791040799767741,
        "auc_difference": 0.0,
        "source_logloss": 0.021869903937152445,
        "computed_bce": 0.02186990561284003,
        "bce_difference": 1.6756875852175224e-09
      },
      "fav": {
        "source_AUC": 0.7862892430951053,
        "computed_AUC": 0.7862892430951053,
        "auc_difference": 0.0,
        "source_logloss": 0.017137615748316272,
        "computed_bce": 0.017137615783322926,
        "bce_difference": 3.500665382571988e-11
      },
      "buy": {
        "source_AUC": 0.8266815958052613,
        "computed_AUC": 0.8266815958052613,
        "auc_difference": 0.0,
        "source_logloss": 0.00977232403246592,
        "computed_bce": 0.009772324517468843,
        "bce_difference": 4.85002923239497e-10
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
        "second_suffix": 0.0,
        "second_even": 0.0,
        "second_odd": 1.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": -0.003773443192333109,
          "ci95": [
            -0.0063889316344838635,
            -0.0012820873871544369
          ]
        },
        "auc.is_click": {
          "estimate": -0.0006618993789875294,
          "ci95": [
            -0.0013660031151723538,
            5.1333597500476726e-05
          ]
        },
        "auc.cart": {
          "estimate": -0.0013934705896027078,
          "ci95": [
            -0.005721746407395101,
            0.0029978054096043718
          ]
        },
        "auc.fav": {
          "estimate": -0.016174247279942766,
          "ci95": [
            -0.021762524579914008,
            -0.01108491749911817
          ]
        },
        "auc.buy": {
          "estimate": 0.003135844479200567,
          "ci95": [
            -0.004123732077358014,
            0.010864292814145584
          ]
        },
        "bce.is_click": {
          "estimate": -8.046375934064876e-05,
          "ci95": [
            -0.00014922172147448717,
            -1.691137998043533e-05
          ]
        },
        "bce.cart": {
          "estimate": 4.195124340599862e-05,
          "ci95": [
            -5.732787749968809e-05,
            0.000131284164672416
          ]
        },
        "bce.fav": {
          "estimate": 0.0003394580889705921,
          "ci95": [
            0.00024131859528932895,
            0.0004516590212372471
          ]
        },
        "bce.buy": {
          "estimate": 0.00012852258074314005,
          "ci95": [
            4.965654626649238e-05,
            0.00022370545832757992
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": 0.0007315712106151784,
          "ci95": [
            -0.003513954219859422,
            0.0052588893915831315
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": 0.015512347900955237,
          "ci95": [
            0.010297758168578289,
            0.021074202390259086
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": -0.0037977438581880962,
          "ci95": [
            -0.01162531114489225,
            0.0037502353683198934
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": 0.014780776690340058,
          "ci95": [
            0.008419900039478057,
            0.02188514926605143
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": -0.004529315068803275,
          "ci95": [
            -0.013005519615261663,
            0.0042897753071802925
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": -0.019310091759143333,
          "ci95": [
            -0.028312240355180436,
            -0.010593242852005398
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": -0.00012241500274664738,
          "ci95": [
            -0.000238034402829501,
            -1.2934735834172374e-05
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": -0.0004199218483112409,
          "ci95": [
            -0.000548852465195847,
            -0.00030790586088929147
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": -0.0002089863400837888,
          "ci95": [
            -0.00032614846528046987,
            -9.201649474770966e-05
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": -0.0002975068455645935,
          "ci95": [
            -0.00044417654324722147,
            -0.00016449700036376058
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": -8.657133733714142e-05,
          "ci95": [
            -0.0002308351712623679,
            3.428959177810543e-05
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": 0.00021093550822745208,
          "ci95": [
            7.547237656407857e-05,
            0.000350301151463136
          ]
        }
      }
    },
    "all_prespecified_effects_involving_candidate": {
      "second_odd__minus__mature_r2": {
        "coefficients": {
          "mature_r2": -1.0,
          "cross8_mature": 0.0,
          "first_prefix": 0.0,
          "first_suffix": 0.0,
          "first_even": 0.0,
          "first_odd": 0.0,
          "second_prefix": 0.0,
          "second_suffix": 0.0,
          "second_even": 0.0,
          "second_odd": 1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.003773443192333109,
            "ci95": [
              -0.0063889316344838635,
              -0.0012820873871544369
            ]
          },
          "auc.is_click": {
            "estimate": -0.0006618993789875294,
            "ci95": [
              -0.0013660031151723538,
              5.1333597500476726e-05
            ]
          },
          "auc.cart": {
            "estimate": -0.0013934705896027078,
            "ci95": [
              -0.005721746407395101,
              0.0029978054096043718
            ]
          },
          "auc.fav": {
            "estimate": -0.016174247279942766,
            "ci95": [
              -0.021762524579914008,
              -0.01108491749911817
            ]
          },
          "auc.buy": {
            "estimate": 0.003135844479200567,
            "ci95": [
              -0.004123732077358014,
              0.010864292814145584
            ]
          },
          "bce.is_click": {
            "estimate": -8.046375934064876e-05,
            "ci95": [
              -0.00014922172147448717,
              -1.691137998043533e-05
            ]
          },
          "bce.cart": {
            "estimate": 4.195124340599862e-05,
            "ci95": [
              -5.732787749968809e-05,
              0.000131284164672416
            ]
          },
          "bce.fav": {
            "estimate": 0.0003394580889705921,
            "ci95": [
              0.00024131859528932895,
              0.0004516590212372471
            ]
          },
          "bce.buy": {
            "estimate": 0.00012852258074314005,
            "ci95": [
              4.965654626649238e-05,
              0.00022370545832757992
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.0007315712106151784,
            "ci95": [
              -0.003513954219859422,
              0.0052588893915831315
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.015512347900955237,
            "ci95": [
              0.010297758168578289,
              0.021074202390259086
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.0037977438581880962,
            "ci95": [
              -0.01162531114489225,
              0.0037502353683198934
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.014780776690340058,
            "ci95": [
              0.008419900039478057,
              0.02188514926605143
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.004529315068803275,
            "ci95": [
              -0.013005519615261663,
              0.0042897753071802925
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.019310091759143333,
            "ci95": [
              -0.028312240355180436,
              -0.010593242852005398
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00012241500274664738,
            "ci95": [
              -0.000238034402829501,
              -1.2934735834172374e-05
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0004199218483112409,
            "ci95": [
              -0.000548852465195847,
              -0.00030790586088929147
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0002089863400837888,
            "ci95": [
              -0.00032614846528046987,
              -9.201649474770966e-05
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.0002975068455645935,
            "ci95": [
              -0.00044417654324722147,
              -0.00016449700036376058
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -8.657133733714142e-05,
            "ci95": [
              -0.0002308351712623679,
              3.428959177810543e-05
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.00021093550822745208,
            "ci95": [
              7.547237656407857e-05,
              0.000350301151463136
            ]
          }
        }
      },
      "second_odd__minus__cross8_mature": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": -1.0,
          "first_prefix": 0.0,
          "first_suffix": 0.0,
          "first_even": 0.0,
          "first_odd": 0.0,
          "second_prefix": 0.0,
          "second_suffix": 0.0,
          "second_even": 0.0,
          "second_odd": 1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.003915088027067554,
            "ci95": [
              0.0013812840607195912,
              0.006997224457180522
            ]
          },
          "auc.is_click": {
            "estimate": -0.0006623763209637756,
            "ci95": [
              -0.0014892520693262994,
              0.0001140931627196999
            ]
          },
          "auc.cart": {
            "estimate": 0.0018340643966250347,
            "ci95": [
              -0.0020566218142722553,
              0.006017108011112361
            ]
          },
          "auc.fav": {
            "estimate": -0.010212767932344269,
            "ci95": [
              -0.015885464989981175,
              -0.003708231545017127
            ]
          },
          "auc.buy": {
            "estimate": 0.024701431964953224,
            "ci95": [
              0.016551983453822104,
              0.034203605602339496
            ]
          },
          "bce.is_click": {
            "estimate": -6.75534014532575e-06,
            "ci95": [
              -8.075252442838535e-05,
              6.691515423975432e-05
            ]
          },
          "bce.cart": {
            "estimate": 3.54939757610008e-05,
            "ci95": [
              -5.845874264067711e-05,
              0.00013421599540101715
            ]
          },
          "bce.fav": {
            "estimate": 0.00016423660186987848,
            "ci95": [
              5.927909000579511e-05,
              0.00027088129740752775
            ]
          },
          "bce.buy": {
            "estimate": -0.0002350658109692861,
            "ci95": [
              -0.000323471254880616,
              -0.00015494942984343072
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.0024964407175888104,
            "ci95": [
              -0.006593850980360414,
              0.0014172794761964295
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.009550391611380493,
            "ci95": [
              0.002911093345702559,
              0.015237622911495314
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.025363808285917,
            "ci95": [
              -0.0349971779295965,
              -0.01689613810888378
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.012046832328969304,
            "ci95": [
              0.0051903273037243345,
              0.019115207536762716
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.02286736756832819,
            "ci95": [
              -0.032967153970995124,
              -0.013390178585391418
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.03491419989729749,
            "ci95": [
              -0.04665626073465308,
              -0.02534496428834765
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -4.224931590632655e-05,
            "ci95": [
              -0.00015304071672307405,
              8.37201734596987e-05
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.00017099194201520423,
            "ci95": [
              -0.0002949731415280028,
              -4.3899892208844485e-05
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.00022831047082396035,
            "ci95": [
              0.00010654523691439596,
              0.00035084341552905527
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.00012874262610887768,
            "ci95": [
              -0.0002877420570433642,
              1.467396946943991e-05
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.0002705597867302869,
            "ci95": [
              0.0001321635383537847,
              0.0004327416704879721
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.0003993024128391646,
            "ci95": [
              0.0002656161317944358,
              0.0005481786491270487
            ]
          }
        }
      },
      "first_odd__minus__second_odd": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "first_prefix": 0.0,
          "first_suffix": 0.0,
          "first_even": 0.0,
          "first_odd": 1.0,
          "second_prefix": 0.0,
          "second_suffix": 0.0,
          "second_even": 0.0,
          "second_odd": -1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.005011362607507591,
            "ci95": [
              0.0022504757905253397,
              0.007689341673881042
            ]
          },
          "auc.is_click": {
            "estimate": -0.0002136480196670565,
            "ci95": [
              -0.0010802529552935286,
              0.0005769539543299835
            ]
          },
          "auc.cart": {
            "estimate": 0.0028110687165837644,
            "ci95": [
              -0.0020431251221843996,
              0.007441123320431644
            ]
          },
          "auc.fav": {
            "estimate": 0.019410865088255846,
            "ci95": [
              0.012676362747471675,
              0.02626915312952085
            ]
          },
          "auc.buy": {
            "estimate": -0.0019628353551421895,
            "ci95": [
              -0.008114414874609847,
              0.005130645962316543
            ]
          },
          "bce.is_click": {
            "estimate": 0.00033241187326812915,
            "ci95": [
              0.0002566854879711866,
              0.0004095274945963137
            ]
          },
          "bce.cart": {
            "estimate": -6.996304109863072e-05,
            "ci95": [
              -0.0001719369828161921,
              3.208862592394134e-05
            ]
          },
          "bce.fav": {
            "estimate": -0.00035391673103677204,
            "ci95": [
              -0.0004987328752848679,
              -0.00021834400090067855
            ]
          },
          "bce.buy": {
            "estimate": -0.00012831755254026286,
            "ci95": [
              -0.0002210577761585968,
              -4.528430774963086e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.003024716736250821,
            "ci95": [
              -0.007685132240673587,
              0.0017132941403039963
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.019624513107922903,
            "ci95": [
              -0.026441606508265342,
              -0.012673672064113215
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.001749187335475133,
            "ci95": [
              -0.005268662283482312,
              0.008011775479600152
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.016599796371672082,
            "ci95": [
              -0.02511424753526213,
              -0.00865912145590286
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.004773904071725954,
            "ci95": [
              -0.00445754305074853,
              0.013400685547289442
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.021373700443398036,
            "ci95": [
              0.01201611285808123,
              0.0310424852255057
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 0.00040237491436675987,
            "ci95": [
              0.00027656321754305715,
              0.0005216841127036046
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.0006863286043049012,
            "ci95": [
              0.0005262266418380246,
              0.0008431507129856094
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.000460729425808392,
            "ci95": [
              0.00034903333374768314,
              0.0005796734938542781
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.0002839536899381413,
            "ci95": [
              0.00011997168731827752,
              0.00046961231662786846
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 5.8354511441632145e-05,
            "ci95": [
              -6.659440194903129e-05,
              0.00021021200040213691
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.00022559917849650918,
            "ci95": [
              -0.0004030798402166392,
              -6.65017928758426e-05
            ]
          }
        }
      },
      "second_prefix__minus__second_odd": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "first_prefix": 0.0,
          "first_suffix": 0.0,
          "first_even": 0.0,
          "first_odd": 0.0,
          "second_prefix": 1.0,
          "second_suffix": 0.0,
          "second_even": 0.0,
          "second_odd": -1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.00908259981984369,
            "ci95": [
              -0.012232931472353294,
              -0.00643034544077727
            ]
          },
          "auc.is_click": {
            "estimate": 0.00026474829336831096,
            "ci95": [
              -0.0004657737478307583,
              0.001145597285320407
            ]
          },
          "auc.cart": {
            "estimate": -0.004133135059150916,
            "ci95": [
              -0.008956875887183868,
              0.0009577371131199566
            ]
          },
          "auc.fav": {
            "estimate": -0.002014712883722347,
            "ci95": [
              -0.008196437015917848,
              0.0033336262494544217
            ]
          },
          "auc.buy": {
            "estimate": -0.030447299629869806,
            "ci95": [
              -0.040148956948136506,
              -0.022647722537596525
            ]
          },
          "bce.is_click": {
            "estimate": 7.306345090196675e-05,
            "ci95": [
              2.158899767025492e-06,
              0.00013906114469521488
            ]
          },
          "bce.cart": {
            "estimate": 0.00030259963733332215,
            "ci95": [
              0.00018108541423140077,
              0.00042062076446789634
            ]
          },
          "bce.fav": {
            "estimate": 0.00011061178176723432,
            "ci95": [
              5.7355031410271345e-06,
              0.00022091037010628812
            ]
          },
          "bce.buy": {
            "estimate": 0.0006299102020536924,
            "ci95": [
              0.0005468867520025655,
              0.0007320647210362568
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.004397883352519227,
            "ci95": [
              -0.0008124980306285972,
              0.009664757153056125
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.002279461177090658,
            "ci95": [
              -0.002922908998581908,
              0.008582842622854681
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.030712047923238117,
            "ci95": [
              0.022698796418675875,
              0.040426901915632064
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.0021184221754285693,
            "ci95": [
              -0.009889719772763175,
              0.006014983510805009
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.02631416457071889,
            "ci95": [
              0.017048015677836877,
              0.037731410123387474
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.02843258674614746,
            "ci95": [
              0.018363492579814043,
              0.03966508847386031
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.0002295361864313554,
            "ci95": [
              -0.00038138587730040553,
              -8.739572821558602e-05
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -3.754833086526757e-05,
            "ci95": [
              -0.00017150915303453172,
              7.430463883173476e-05
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0005568467511517256,
            "ci95": [
              -0.0006795917358869694,
              -0.0004430113692166181
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.00019198785556608783,
            "ci95": [
              1.827642133567856e-05,
              0.00035441908907253263
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -0.0003273105647203702,
            "ci95": [
              -0.000493742283858282,
              -0.00018329783938343192
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.000519298420286458,
            "ci95": [
              -0.0006692872270253576,
              -0.0003827369278309978
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
    "optimizer_layout_sha256": "084b196854130b4105bc6794f0a6ae12d7d90dbb9084950d330cff3ae52a136c",
    "formal_worker_cost": {
      "wall_seconds": 2538.3111224472523,
      "cpu_seconds": 3053.287257262,
      "peak_rss_bytes": 17063968768,
      "gpu_peak_bytes": 28369223680,
      "samples": 2329
    },
    "smoke_worker_cost": {
      "wall_seconds": 153.95577307790518,
      "cpu_seconds": 169.960421068,
      "peak_rss_bytes": 17000247296,
      "gpu_peak_bytes": 29996613632,
      "samples": 141
    },
    "complete_recorded_costs": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2538.3111224472523,
        "cpu_seconds": 3053.287257262,
        "peak_rss_bytes": 17063968768,
        "gpu_peak_bytes": 28369223680,
        "samples": 2329
      },
      "formal_runner": {
        "wall_seconds": 2533.129072897136,
        "cpu_seconds": 2936.433795101,
        "peak_process_rss_bytes": 5301084160
      },
      "fit_including_validation": {
        "wall_seconds": 2409.0147029161453,
        "cpu_seconds": 2763.090627665,
        "peak_process_rss_bytes": 5301084160
      },
      "prediction_export": {
        "wall_seconds": 118.78991805016994,
        "cpu_seconds": 167.7172589500001,
        "peak_process_rss_bytes": 5301084160
      },
      "smoke_worker": {
        "wall_seconds": 153.95577307790518,
        "cpu_seconds": 169.960421068,
        "peak_rss_bytes": 17000247296,
        "gpu_peak_bytes": 29996613632,
        "samples": 141
      },
      "smoke_runner": {
        "wall_seconds": 150.0005791336298,
        "cpu_seconds": 147.146033156,
        "peak_process_rss_bytes": 3291389952
      },
      "latency_constructor": {
        "wall_seconds": 0.7516213431954384,
        "cpu_seconds": 0.8240607020000006,
        "peak_rss_bytes": 2092183552
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.13237854093313217,
        "cpu_seconds": 0.1315488120000019,
        "peak_rss_bytes": 2092183552
      },
      "latency_peak_cuda_bytes": 1281155072,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "forward_latency": {
      "mean_ms": 14.343976759910584,
      "p50_ms": 14.288623809814453,
      "p95_ms": 14.347693967819215,
      "batch_size": 1024,
      "warmups": 10,
      "iterations": 40
    },
    "latency_ratio_to_parent": 1.013405060171421,
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
  "decision_reason": "Macro delta -0.003773443,CI95 [-0.0063889316344838635, -0.0012820873871544369],latency ratio 1.013405. Fails positive and near-candidate quality policy; preserve negative evidence.",
  "decision_scope": "Root queue disposition only; no released-incumbent promotion or supported lesson.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_odd",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_odd/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_odd/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_odd/Taobao_Action/second_odd.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/paired-analysis/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/latency/completion.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_odd",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_odd/result.json",
    "result_sha256": "2dd97782b3d5fd821f0da8c9b86c8667be4398361c818a8e1dfb5e37e6265f53",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_odd/manifest.json",
    "manifest_sha256": "f68199c4064e85d6fa7b93d5594b5aa3f2057d50f52c0d7b5d07898ed4b58de5",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_odd/Taobao_Action/second_odd.model",
    "checkpoint_sha256": "201cfd7e1462e47a03e2a5255278ada4ede93c760ce2d30c6e76b9beb01f21d9",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/paired-analysis/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
