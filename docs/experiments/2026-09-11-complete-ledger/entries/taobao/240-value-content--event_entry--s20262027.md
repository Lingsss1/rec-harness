## 240. value-content--event_entry--s20262027

- 假设：Second-work value supplementation; event_entry conditional nonlinear content before/after original weighted aggregation.
- 判定：discard
- 原因：Macro delta -0.004339360, CI95 [-0.00789869835970372, -0.0008016800889363466], forward ratio 1.113860; misses both frozen quality and near policy.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "value-content--event_entry--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-10T06:16:19.753374+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "event_entry",
  "code_id": "24e06fb09c6658ef53f419e5fa410cfa79eb699cc16354bfd31a299d62180a70",
  "hypothesis": "Second-work value supplementation; event_entry conditional nonlinear content before/after original weighted aggregation.",
  "change": "{\"model\": \"QFormerCross8ValueContent\", \"maturity_mode\": \"mature\", \"value_position\": \"event\", \"value_condition\": \"entry\", \"value_rank\": 16, \"work_steps\": 2, \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8ValueContent",
    "maturity_mode": "mature",
    "value_position": "event",
    "value_condition": "entry",
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
      "value": 0.7627555914485915,
      "delta": -0.004339359866268877
    },
    "per_task": {
      "is_click": {
        "logloss": 0.18416360890560712,
        "AUC": 0.6503652705893754
      },
      "cart": {
        "logloss": 0.022064524100784407,
        "AUC": 0.7841280158809417
      },
      "fav": {
        "logloss": 0.016785212090061695,
        "AUC": 0.8035150316904012
      },
      "buy": {
        "logloss": 0.009676964841924714,
        "AUC": 0.8130140476336482
      }
    },
    "macro_source_logloss": 0.058172577484594486,
    "source_metric_comparison": {
      "is_click": {
        "source_AUC": 0.6503652705893754,
        "computed_AUC": 0.6503652705893754,
        "auc_difference": 0.0,
        "source_logloss": 0.18416360890560712,
        "computed_bce": 0.18416360890561018,
        "bce_difference": 3.0531133177191805e-15
      },
      "cart": {
        "source_AUC": 0.7841280158809417,
        "computed_AUC": 0.7841280158809417,
        "auc_difference": 0.0,
        "source_logloss": 0.022064524100784407,
        "computed_bce": 0.022064525878079073,
        "bce_difference": 1.777294665605833e-09
      },
      "fav": {
        "source_AUC": 0.8035150316904012,
        "computed_AUC": 0.8035150316904012,
        "auc_difference": 0.0,
        "source_logloss": 0.016785212090061695,
        "computed_bce": 0.016785212374562923,
        "bce_difference": 2.8450122813161727e-10
      },
      "buy": {
        "source_AUC": 0.8130140476336482,
        "computed_AUC": 0.8130140476336482,
        "auc_difference": 0.0,
        "source_logloss": 0.009676964841924714,
        "computed_bce": 0.009676964882812653,
        "bce_difference": 4.088793886802833e-11
      }
    },
    "matched_parent_effect": {
      "coefficients": {
        "mature_r2": -1.0,
        "cross8_mature": 0.0,
        "event_entry": 1.0,
        "event_work": 0.0,
        "event_constant": 0.0,
        "pooled_entry": 0.0,
        "pooled_work": 0.0,
        "pooled_constant": 0.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": -0.004339359866268877,
          "ci95": [
            -0.00789869835970372,
            -0.0008016800889363466
          ]
        },
        "auc.is_click": {
          "estimate": 0.0004289773883859871,
          "ci95": [
            -0.00032545863121373475,
            0.0012312873888201124
          ]
        },
        "auc.cart": {
          "estimate": -0.008306254476401986,
          "ci95": [
            -0.014083630638524516,
            -0.0017035647225935558
          ]
        },
        "auc.fav": {
          "estimate": 0.0010515413153531217,
          "ci95": [
            -0.004354940846647004,
            0.005788675183350388
          ]
        },
        "auc.buy": {
          "estimate": -0.01053170369241263,
          "ci95": [
            -0.022479472161014986,
            -0.0009431220836329749
          ]
        },
        "bce.is_click": {
          "estimate": -0.0002280373621764975,
          "ci95": [
            -0.0002997307249833235,
            -0.00015599150979847196
          ]
        },
        "bce.cart": {
          "estimate": 0.00023657150864504067,
          "ci95": [
            0.00011092625400607277,
            0.00035097926714166926
          ]
        },
        "bce.fav": {
          "estimate": -1.2945319789410548e-05,
          "ci95": [
            -0.00012803026195623663,
            0.00011404342128708277
          ]
        },
        "bce.buy": {
          "estimate": 3.3162946086949877e-05,
          "ci95": [
            -7.169886112674905e-05,
            0.00015066420607880415
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": 0.008735231864787973,
          "ci95": [
            0.0020508338309380733,
            0.014574225997272476
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": -0.0006225639269671346,
          "ci95": [
            -0.00584828852233823,
            0.004720261227594938
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": 0.010960681080798618,
          "ci95": [
            0.0014738246172585178,
            0.02280505290368534
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": -0.009357795791755108,
          "ci95": [
            -0.016724996667932687,
            -0.0011801353514736908
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": 0.0022254492160106443,
          "ci95": [
            -0.008803450690586261,
            0.014587186807789
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": 0.011583245007765752,
          "ci95": [
            0.000880487574055,
            0.023589881511313426
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": -0.0004646088708215382,
          "ci95": [
            -0.000610400895844576,
            -0.0003104299809038276
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": -0.00021509204238708696,
          "ci95": [
            -0.0003605671397840048,
            -7.892921057996993e-05
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": -0.0002612003082634474,
          "ci95": [
            -0.00039785131869749397,
            -0.000143717660987282
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": 0.0002495168284344512,
          "ci95": [
            7.378712584122242e-05,
            0.00041529554722494
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": 0.0002034085625580908,
          "ci95": [
            4.167899728107364e-05,
            0.00035052712118362325
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": -4.6108265876360424e-05,
          "ci95": [
            -0.00020151585744935738,
            0.00010479941525919393
          ]
        }
      }
    },
    "all_prespecified_effects_involving_candidate": {
      "event_entry__minus__mature_r2": {
        "coefficients": {
          "mature_r2": -1.0,
          "cross8_mature": 0.0,
          "event_entry": 1.0,
          "event_work": 0.0,
          "event_constant": 0.0,
          "pooled_entry": 0.0,
          "pooled_work": 0.0,
          "pooled_constant": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.004339359866268877,
            "ci95": [
              -0.00789869835970372,
              -0.0008016800889363466
            ]
          },
          "auc.is_click": {
            "estimate": 0.0004289773883859871,
            "ci95": [
              -0.00032545863121373475,
              0.0012312873888201124
            ]
          },
          "auc.cart": {
            "estimate": -0.008306254476401986,
            "ci95": [
              -0.014083630638524516,
              -0.0017035647225935558
            ]
          },
          "auc.fav": {
            "estimate": 0.0010515413153531217,
            "ci95": [
              -0.004354940846647004,
              0.005788675183350388
            ]
          },
          "auc.buy": {
            "estimate": -0.01053170369241263,
            "ci95": [
              -0.022479472161014986,
              -0.0009431220836329749
            ]
          },
          "bce.is_click": {
            "estimate": -0.0002280373621764975,
            "ci95": [
              -0.0002997307249833235,
              -0.00015599150979847196
            ]
          },
          "bce.cart": {
            "estimate": 0.00023657150864504067,
            "ci95": [
              0.00011092625400607277,
              0.00035097926714166926
            ]
          },
          "bce.fav": {
            "estimate": -1.2945319789410548e-05,
            "ci95": [
              -0.00012803026195623663,
              0.00011404342128708277
            ]
          },
          "bce.buy": {
            "estimate": 3.3162946086949877e-05,
            "ci95": [
              -7.169886112674905e-05,
              0.00015066420607880415
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.008735231864787973,
            "ci95": [
              0.0020508338309380733,
              0.014574225997272476
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.0006225639269671346,
            "ci95": [
              -0.00584828852233823,
              0.004720261227594938
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.010960681080798618,
            "ci95": [
              0.0014738246172585178,
              0.02280505290368534
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.009357795791755108,
            "ci95": [
              -0.016724996667932687,
              -0.0011801353514736908
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.0022254492160106443,
            "ci95": [
              -0.008803450690586261,
              0.014587186807789
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.011583245007765752,
            "ci95": [
              0.000880487574055,
              0.023589881511313426
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.0004646088708215382,
            "ci95": [
              -0.000610400895844576,
              -0.0003104299809038276
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.00021509204238708696,
            "ci95": [
              -0.0003605671397840048,
              -7.892921057996993e-05
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0002612003082634474,
            "ci95": [
              -0.00039785131869749397,
              -0.000143717660987282
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.0002495168284344512,
            "ci95": [
              7.378712584122242e-05,
              0.00041529554722494
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.0002034085625580908,
            "ci95": [
              4.167899728107364e-05,
              0.00035052712118362325
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -4.6108265876360424e-05,
            "ci95": [
              -0.00020151585744935738,
              0.00010479941525919393
            ]
          }
        }
      },
      "event_entry__minus__cross8_mature": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": -1.0,
          "event_entry": 1.0,
          "event_work": 0.0,
          "event_constant": 0.0,
          "pooled_entry": 0.0,
          "pooled_work": 0.0,
          "pooled_constant": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.0033491713531317857,
            "ci95": [
              2.1065200761098056e-07,
              0.006480661449644977
            ]
          },
          "auc.is_click": {
            "estimate": 0.00042850044640974083,
            "ci95": [
              -0.00043042976243736755,
              0.0012181438561917526
            ]
          },
          "auc.cart": {
            "estimate": -0.005078719490174244,
            "ci95": [
              -0.01062210632558148,
              5.9824414226074436e-05
            ]
          },
          "auc.fav": {
            "estimate": 0.007013020662951619,
            "ci95": [
              0.0014949254224416065,
              0.011723389360087226
            ]
          },
          "auc.buy": {
            "estimate": 0.011033883793340027,
            "ci95": [
              0.00022589864620056242,
              0.02111800095321182
            ]
          },
          "bce.is_click": {
            "estimate": -0.0001543289429811745,
            "ci95": [
              -0.00022910533714773915,
              -8.127415591171475e-05
            ]
          },
          "bce.cart": {
            "estimate": 0.00023011424100004285,
            "ci95": [
              0.00012453267012744505,
              0.0003425039161254996
            ]
          },
          "bce.fav": {
            "estimate": -0.0001881668068901242,
            "ci95": [
              -0.0002915129566104118,
              -7.808119610832932e-05
            ]
          },
          "bce.buy": {
            "estimate": -0.00033042544562547627,
            "ci95": [
              -0.00043830192845882137,
              -0.00021628889540164093
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.005507219936583985,
            "ci95": [
              0.0003587946183274016,
              0.011143966058763447
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.006584520216541878,
            "ci95": [
              -0.011377497797934682,
              -0.001306322171512909
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.010605383346930286,
            "ci95": [
              -0.020771791716567633,
              0.00015338341826642976
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.012091740153125863,
            "ci95": [
              -0.01979416662332793,
              -0.005003324357314839
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.01611260328351427,
            "ci95": [
              -0.02866548327323531,
              -0.004898531101611138
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.004020863130388408,
            "ci95": [
              -0.016566388424487502,
              0.006801590829330684
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00038444318398121735,
            "ci95": [
              -0.0005202430141622225,
              -0.0002467485241994127
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 3.383786390894969e-05,
            "ci95": [
              -0.00011256185284013939,
              0.0001606058122608578
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.00017609650264430177,
            "ci95": [
              4.927107157502741e-05,
              0.0003146103879113199
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.00041828104789016704,
            "ci95": [
              0.00027219102552327866,
              0.0005785402991801456
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.0005605396866255191,
            "ci95": [
              0.00041078526699365656,
              0.0007330412597952678
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.00014225863873535208,
            "ci95": [
              -3.6429975026951635e-06,
              0.00029200648609716494
            ]
          }
        }
      },
      "event_work__minus__event_entry": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "event_entry": -1.0,
          "event_work": 1.0,
          "event_constant": 0.0,
          "pooled_entry": 0.0,
          "pooled_work": 0.0,
          "pooled_constant": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.0014341509368082928,
            "ci95": [
              -0.0017163203768568059,
              0.004531709042086111
            ]
          },
          "auc.is_click": {
            "estimate": -0.0009918132314931105,
            "ci95": [
              -0.001796855787764673,
              -0.00021736486391852607
            ]
          },
          "auc.cart": {
            "estimate": 0.006384318499461972,
            "ci95": [
              0.0010672809494749942,
              0.01149141844889579
            ]
          },
          "auc.fav": {
            "estimate": 0.0010244779991811503,
            "ci95": [
              -0.004149702189453669,
              0.006568218463497515
            ]
          },
          "auc.buy": {
            "estimate": -0.0006803795199168405,
            "ci95": [
              -0.01022382263940432,
              0.009401658201121689
            ]
          },
          "bce.is_click": {
            "estimate": 0.0001370472536688372,
            "ci95": [
              6.989874316040492e-05,
              0.00021522358106064592
            ]
          },
          "bce.cart": {
            "estimate": -0.00016162040084074722,
            "ci95": [
              -0.0002708018047951085,
              -4.554179270663893e-05
            ]
          },
          "bce.fav": {
            "estimate": -0.00013555025656959707,
            "ci95": [
              -0.00025101815108504914,
              -2.3596905116636885e-05
            ]
          },
          "bce.buy": {
            "estimate": 0.00011045700396122848,
            "ci95": [
              7.923393214493955e-06,
              0.0002140520540930401
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.007376131730955082,
            "ci95": [
              -0.012599883122280692,
              -0.0018633625520478874
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.002016291230674261,
            "ci95": [
              -0.00789841030258073,
              0.003050277726039498
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.00031143371157627,
            "ci95": [
              -0.010476721918166102,
              0.0092390695300444
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.005359840500280821,
            "ci95": [
              -0.0033867799176251177,
              0.01244185434369685
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.007064698019378812,
            "ci95": [
              -0.0041240470957167345,
              0.01694943884043986
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.0017048575190979909,
            "ci95": [
              -0.01000081093433857,
              0.012604641942101773
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 0.0002986676545095844,
            "ci95": [
              0.00016664902245829257,
              0.0004219909339024411
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.00027259751023843426,
            "ci95": [
              0.0001497504332460653,
              0.0004103539420385615
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 2.6590249707608704e-05,
            "ci95": [
              -0.00010419171911748171,
              0.00015005908141207085
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -2.6070144271150147e-05,
            "ci95": [
              -0.00017863714780998297,
              0.00015892189440639293
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -0.0002720774048019757,
            "ci95": [
              -0.00042188138568269534,
              -0.00012473228242583124
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.00024600726053082556,
            "ci95": [
              -0.0003852682853085855,
              -9.67294424387923e-05
            ]
          }
        }
      },
      "event_entry__minus__event_constant": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "event_entry": 1.0,
          "event_work": 0.0,
          "event_constant": -1.0,
          "pooled_entry": 0.0,
          "pooled_work": 0.0,
          "pooled_constant": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.0001359460844422844,
            "ci95": [
              -0.0031499835696204397,
              0.003429056158559577
            ]
          },
          "auc.is_click": {
            "estimate": 0.0017491599501973099,
            "ci95": [
              0.0009028538689436677,
              0.002564257196811317
            ]
          },
          "auc.cart": {
            "estimate": -0.0034824112184441702,
            "ci95": [
              -0.009495343923423393,
              0.0037158903630673144
            ]
          },
          "auc.fav": {
            "estimate": 0.00667505440947036,
            "ci95": [
              0.0018352892037827306,
              0.01222210798613629
            ]
          },
          "auc.buy": {
            "estimate": -0.004398018803454362,
            "ci95": [
              -0.013999154663998084,
              0.00562398795940428
            ]
          },
          "bce.is_click": {
            "estimate": -0.00011470660157048007,
            "ci95": [
              -0.00019352321188956313,
              -3.424355043041423e-05
            ]
          },
          "bce.cart": {
            "estimate": 0.00013962889310246626,
            "ci95": [
              -2.2800212042932758e-05,
              0.00026850958420151757
            ]
          },
          "bce.fav": {
            "estimate": -0.0002306023472838821,
            "ci95": [
              -0.0003291264013082384,
              -0.00013339491360410833
            ]
          },
          "bce.buy": {
            "estimate": -0.0002350721550273549,
            "ci95": [
              -0.0003462595177033317,
              -0.00012443562248458923
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.00523157116864148,
            "ci95": [
              -0.0019439986729692436,
              0.01114805982287453
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.00492589445927305,
            "ci95": [
              -0.010372527779704318,
              0.00027790290594555844
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.006147178753651672,
            "ci95": [
              -0.0037189913923527263,
              0.015896698012413647
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.01015746562791453,
            "ci95": [
              -0.018661395823175938,
              -0.0020106507571443035
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.0009156075850101919,
            "ci95": [
              -0.010385155519854968,
              0.012236035416783263
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.011073073212924722,
            "ci95": [
              0.0011008577495754516,
              0.0212542276397325
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.00025433549467294633,
            "ci95": [
              -0.0003973313866089146,
              -8.007446378345972e-05
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.00011589574571340203,
            "ci95": [
              -7.295629429033331e-06,
              0.00023755840082066566
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.00012036555345687483,
            "ci95": [
              -1.270943024106524e-05,
              0.0002656638393092303
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.00037023124038634836,
            "ci95": [
              0.00018972738989587413,
              0.0005464159811993922
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00037470104812982116,
            "ci95": [
              0.00019711536383447208,
              0.0005558048002388228
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 4.469807743472803e-06,
            "ci95": [
              -0.00014097977129642019,
              0.00014458234368330825
            ]
          }
        }
      },
      "event_entry__minus__pooled_entry": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "event_entry": 1.0,
          "event_work": 0.0,
          "event_constant": 0.0,
          "pooled_entry": -1.0,
          "pooled_work": 0.0,
          "pooled_constant": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.0009877323374864189,
            "ci95": [
              -0.004773053872229927,
              0.002503981295255304
            ]
          },
          "auc.is_click": {
            "estimate": -0.00029208919383238996,
            "ci95": [
              -0.0009620216643889318,
              0.000409636075419964
            ]
          },
          "auc.cart": {
            "estimate": -0.007357532758536012,
            "ci95": [
              -0.012494611714045053,
              -0.0020126412779741423
            ]
          },
          "auc.fav": {
            "estimate": 0.019873875037975552,
            "ci95": [
              0.01301890810835942,
              0.02685511233880901
            ]
          },
          "auc.buy": {
            "estimate": -0.016175182435552826,
            "ci95": [
              -0.02816054788465566,
              -0.005763677187238844
            ]
          },
          "bce.is_click": {
            "estimate": -0.00012650119724622044,
            "ci95": [
              -0.00019570871860319984,
              -5.9802535855739475e-05
            ]
          },
          "bce.cart": {
            "estimate": 0.00017636933893119128,
            "ci95": [
              7.492147203376887e-05,
              0.00028896149500761185
            ]
          },
          "bce.fav": {
            "estimate": -0.0003069935809082956,
            "ci95": [
              -0.00042635524716436185,
              -0.0001997665181986138
            ]
          },
          "bce.buy": {
            "estimate": 2.1815136819512e-05,
            "ci95": [
              -9.32339752974446e-05,
              0.00015384150215306985
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.007065443564703622,
            "ci95": [
              0.0019293127398789412,
              0.012138008208263842
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.020165964231807942,
            "ci95": [
              -0.02723591756979243,
              -0.013237967778805659
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.015883093241720436,
            "ci95": [
              0.005488984182877768,
              0.02797269103818055
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.027231407796511564,
            "ci95": [
              -0.035924381697605215,
              -0.019084978417158886
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.008817649677016814,
            "ci95": [
              -0.0032689378750491546,
              0.021599940676113625
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.03604905747352838,
            "ci95": [
              0.02299210376211544,
              0.050440193634552266
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.0003028705361774117,
            "ci95": [
              -0.00043177577932001213,
              -0.0001775565008561202
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.00018049238366207518,
            "ci95": [
              4.484558241326366e-05,
              0.00030691695417367303
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.00014831633406573244,
            "ci95": [
              -0.0002943219311189305,
              -7.744970183356876e-06
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.0004833629198394869,
            "ci95": [
              0.0003279209637716405,
              0.0006352154836461961
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00015455420211167928,
            "ci95": [
              -1.8470972802825732e-05,
              0.0003214507040136172
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.0003288087177278076,
            "ci95": [
              -0.0005070744255646415,
              -0.00015189494997674163
            ]
          }
        }
      }
    },
    "parameter_count": 51742692,
    "initializer_sha256": "aa3b2ab7b9014c0f0183713dca1e9fcc087775a3c1683e2efaa4346181bd573d",
    "optimizer_layout_sha256": "0442eee3ecfa8bcbed06d4da5c7378a7b63827345d1479a1da14226e77940f1b",
    "formal_worker_cost": {
      "wall_seconds": 2625.7551608383656,
      "cpu_seconds": 3125.975532019,
      "peak_rss_bytes": 17061019648,
      "gpu_peak_bytes": 30267146240,
      "samples": 2410
    },
    "smoke_worker_cost": {
      "wall_seconds": 202.33171118050814,
      "cpu_seconds": 257.755041561,
      "peak_rss_bytes": 19793399808,
      "gpu_peak_bytes": 32792117248,
      "samples": 185
    },
    "complete_recorded_costs": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2625.7551608383656,
        "cpu_seconds": 3125.975532019,
        "peak_rss_bytes": 17061019648,
        "gpu_peak_bytes": 30267146240,
        "samples": 2410
      },
      "formal_runner": {
        "wall_seconds": 2622.1908311396837,
        "cpu_seconds": 3009.048346127,
        "peak_process_rss_bytes": 5226352640
      },
      "fit_including_validation": {
        "wall_seconds": 2496.0205204561353,
        "cpu_seconds": 2836.12697659,
        "peak_process_rss_bytes": 5226352640
      },
      "prediction_export": {
        "wall_seconds": 121.10354164987803,
        "cpu_seconds": 167.40220941500002,
        "peak_process_rss_bytes": 5226352640
      },
      "smoke_worker": {
        "wall_seconds": 202.33171118050814,
        "cpu_seconds": 257.755041561,
        "peak_rss_bytes": 19793399808,
        "gpu_peak_bytes": 32792117248,
        "samples": 185
      },
      "smoke_runner": {
        "wall_seconds": 198.37353361397982,
        "cpu_seconds": 180.623610195,
        "peak_process_rss_bytes": 3304587264
      },
      "latency_constructor": {
        "wall_seconds": 0.7557265013456345,
        "cpu_seconds": 0.8347829929999993,
        "peak_rss_bytes": 2080583680
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.12367325276136398,
        "cpu_seconds": 0.1232786569999984,
        "peak_rss_bytes": 2080583680
      },
      "latency_peak_cuda_bytes": 1284208128,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "forward_latency": {
      "mean_ms": 15.90228886604309,
      "p50_ms": 15.8886399269104,
      "p95_ms": 15.913143920898438,
      "batch_size": 1024,
      "warmups": 10,
      "iterations": 40
    },
    "latency_ratio_to_parent": 1.1138603229815884,
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
  "decision_reason": "Macro delta -0.004339360, CI95 [-0.00789869835970372, -0.0008016800889363466], forward ratio 1.113860; misses both frozen quality and near policy.",
  "decision_scope": "Root queue disposition only; no released-incumbent promotion or supported lesson.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/event_entry",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/event_entry/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/event_entry/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/event_entry/Taobao_Action/event_entry.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/paired-analysis/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/latency/completion.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/event_entry",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/event_entry/result.json",
    "result_sha256": "ac13577abdfedb68d2fcf706a127c80c40f3b92ffa7045fd9d973c2ba7914ace",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/event_entry/manifest.json",
    "manifest_sha256": "df317fb4aa68b7a502e133629b1a5d793381947c293ef1267167206a21368b47",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/event_entry/Taobao_Action/event_entry.model",
    "checkpoint_sha256": "242af1491d1b9711d6ab67fdd5074b12d550ac822b39f02108d891b9cd35dbee",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/paired-analysis/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
