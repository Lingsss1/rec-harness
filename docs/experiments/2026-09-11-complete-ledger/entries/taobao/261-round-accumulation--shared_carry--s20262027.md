## 261. round-accumulation--shared_carry--s20262027

- 假设：检验轮次区分的残差更新与首轮状态携带的交互，固定两轮、优化器和seed。
- 判定：discard
- 原因：相对原两轮macro AUC下降0.006701且区间在零以下；未提供整体优化收益。

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "round-accumulation--shared_carry--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-10T15:44:40.685443+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "round_accumulation_factorial",
  "code_id": "2511cf353e09f9b83e8b18b3e701b1d57b7471c2ebaeb2986b93add858312e94",
  "hypothesis": "检验轮次区分的残差更新与首轮状态携带的交互，固定两轮、优化器和seed。",
  "change": "{\"model\": \"QFormerCross8RoundAccumulation\", \"maturity_mode\": \"mature\", \"work_steps\": 2, \"num_queries\": 4, \"accumulation_distinct\": false, \"carry_first\": true}",
  "config_change": {
    "model": "QFormerCross8RoundAccumulation",
    "maturity_mode": "mature",
    "work_steps": 2,
    "num_queries": 4,
    "accumulation_distinct": false,
    "carry_first": true
  },
  "input_protocol_id": "strict-past-native-last100-action19-mature",
  "budget": {
    "formal_training_runs": 1,
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
      "value": 0.7603940215257633,
      "delta": -0.006700929789097171
    },
    "per_task": {
      "is_click": {
        "logloss": 0.18428796990218144,
        "AUC": 0.6478070416193132
      },
      "cart": {
        "logloss": 0.021767443305021072,
        "AUC": 0.7934155682397055
      },
      "fav": {
        "logloss": 0.017185586934913417,
        "AUC": 0.7844486244997729
      },
      "buy": {
        "logloss": 0.009783572802540512,
        "AUC": 0.8159048517442616
      }
    },
    "macro_source_logloss": 0.05825614323616411,
    "matched_parent_effect": {
      "coefficients": {
        "mature_r2": -1.0,
        "cross8_mature": 0.0,
        "shared_carry": 1.0,
        "distinct_carry": 0.0,
        "shared_no_carry": 0.0,
        "distinct_no_carry": 0.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": -0.006700929789097171,
          "ci95": [
            -0.010151543204758619,
            -0.0033708810668955274
          ]
        },
        "auc.is_click": {
          "estimate": -0.002129251581676206,
          "ci95": [
            -0.0028825373773933767,
            -0.0013817793251067942
          ]
        },
        "auc.cart": {
          "estimate": 0.0009812978823617868,
          "ci95": [
            -0.0032018771268514127,
            0.0054418849006120295
          ]
        },
        "auc.fav": {
          "estimate": -0.01801486587527512,
          "ci95": [
            -0.026717092654065976,
            -0.010488174051285588
          ]
        },
        "auc.buy": {
          "estimate": -0.007640899581799143,
          "ci95": [
            -0.01740038172234689,
            0.0010010179173101435
          ]
        },
        "bce.is_click": {
          "estimate": -0.00010367636560340565,
          "ci95": [
            -0.00017728944959024341,
            -2.898710175194349e-05
          ]
        },
        "bce.cart": {
          "estimate": -6.0511064299519773e-05,
          "ci95": [
            -0.0001634880662591453,
            3.9325999206971315e-05
          ]
        },
        "bce.fav": {
          "estimate": 0.0003874292443932399,
          "ci95": [
            0.00024294493559346632,
            0.0005614850203903269
          ]
        },
        "bce.buy": {
          "estimate": 0.0001397708757113602,
          "ci95": [
            5.580975495774955e-05,
            0.00022930209515159336
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": -0.0031105494640379927,
          "ci95": [
            -0.007632102653148784,
            0.0010754965490268969
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": 0.015885614293598915,
          "ci95": [
            0.008166849665769096,
            0.024770638375625976
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": 0.005511648000122937,
          "ci95": [
            -0.0033724411030413566,
            0.015262900292081294
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": 0.018996163757636908,
          "ci95": [
            0.010674716738461163,
            0.02898252003877601
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": 0.00862219746416093,
          "ci95": [
            -0.000615610998096899,
            0.018319208112298525
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": -0.010373966293475978,
          "ci95": [
            -0.022443295081796856,
            0.0020770360991994637
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": -4.316530130388588e-05,
          "ci95": [
            -0.00017078223686198477,
            8.371055693701597e-05
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": -0.0004911056099966456,
          "ci95": [
            -0.0006771910965623288,
            -0.0003186462039088713
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": -0.00024344724131476585,
          "ci95": [
            -0.0003605276366924978,
            -0.00013811487581205195
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": -0.0004479403086927597,
          "ci95": [
            -0.0006590094108431628,
            -0.0002778411588207804
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": -0.00020028194001087997,
          "ci95": [
            -0.00032219380727469547,
            -7.788784081568554e-05
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": 0.0002476583686818797,
          "ci95": [
            6.747866843335254e-05,
            0.0004515495748165237
          ]
        }
      }
    },
    "all_prespecified_effects_involving_candidate": {
      "shared_carry__minus__mature_r2": {
        "coefficients": {
          "mature_r2": -1.0,
          "cross8_mature": 0.0,
          "shared_carry": 1.0,
          "distinct_carry": 0.0,
          "shared_no_carry": 0.0,
          "distinct_no_carry": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.006700929789097171,
            "ci95": [
              -0.010151543204758619,
              -0.0033708810668955274
            ]
          },
          "auc.is_click": {
            "estimate": -0.002129251581676206,
            "ci95": [
              -0.0028825373773933767,
              -0.0013817793251067942
            ]
          },
          "auc.cart": {
            "estimate": 0.0009812978823617868,
            "ci95": [
              -0.0032018771268514127,
              0.0054418849006120295
            ]
          },
          "auc.fav": {
            "estimate": -0.01801486587527512,
            "ci95": [
              -0.026717092654065976,
              -0.010488174051285588
            ]
          },
          "auc.buy": {
            "estimate": -0.007640899581799143,
            "ci95": [
              -0.01740038172234689,
              0.0010010179173101435
            ]
          },
          "bce.is_click": {
            "estimate": -0.00010367636560340565,
            "ci95": [
              -0.00017728944959024341,
              -2.898710175194349e-05
            ]
          },
          "bce.cart": {
            "estimate": -6.0511064299519773e-05,
            "ci95": [
              -0.0001634880662591453,
              3.9325999206971315e-05
            ]
          },
          "bce.fav": {
            "estimate": 0.0003874292443932399,
            "ci95": [
              0.00024294493559346632,
              0.0005614850203903269
            ]
          },
          "bce.buy": {
            "estimate": 0.0001397708757113602,
            "ci95": [
              5.580975495774955e-05,
              0.00022930209515159336
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.0031105494640379927,
            "ci95": [
              -0.007632102653148784,
              0.0010754965490268969
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.015885614293598915,
            "ci95": [
              0.008166849665769096,
              0.024770638375625976
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.005511648000122937,
            "ci95": [
              -0.0033724411030413566,
              0.015262900292081294
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.018996163757636908,
            "ci95": [
              0.010674716738461163,
              0.02898252003877601
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": 0.00862219746416093,
            "ci95": [
              -0.000615610998096899,
              0.018319208112298525
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.010373966293475978,
            "ci95": [
              -0.022443295081796856,
              0.0020770360991994637
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -4.316530130388588e-05,
            "ci95": [
              -0.00017078223686198477,
              8.371055693701597e-05
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.0004911056099966456,
            "ci95": [
              -0.0006771910965623288,
              -0.0003186462039088713
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.00024344724131476585,
            "ci95": [
              -0.0003605276366924978,
              -0.00013811487581205195
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.0004479403086927597,
            "ci95": [
              -0.0006590094108431628,
              -0.0002778411588207804
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -0.00020028194001087997,
            "ci95": [
              -0.00032219380727469547,
              -7.788784081568554e-05
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.0002476583686818797,
            "ci95": [
              6.747866843335254e-05,
              0.0004515495748165237
            ]
          }
        }
      },
      "shared_carry__minus__cross8_mature": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": -1.0,
          "shared_carry": 1.0,
          "distinct_carry": 0.0,
          "shared_no_carry": 0.0,
          "distinct_no_carry": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.000987601430303492,
            "ci95": [
              -0.00177564216408084,
              0.004190641695538917
            ]
          },
          "auc.is_click": {
            "estimate": -0.002129728523652452,
            "ci95": [
              -0.002861422601189714,
              -0.0014352382830959494
            ]
          },
          "auc.cart": {
            "estimate": 0.004208832868589529,
            "ci95": [
              -0.0006463562502875719,
              0.00911317191723642
            ]
          },
          "auc.fav": {
            "estimate": -0.012053386527676624,
            "ci95": [
              -0.018927723207890986,
              -0.005046534116929271
            ]
          },
          "auc.buy": {
            "estimate": 0.013924687903953514,
            "ci95": [
              0.005317948000972891,
              0.023395920173641303
            ]
          },
          "bce.is_click": {
            "estimate": -2.9967946408082646e-05,
            "ci95": [
              -9.534138312975101e-05,
              4.29098987754297e-05
            ]
          },
          "bce.cart": {
            "estimate": -6.69683319445176e-05,
            "ci95": [
              -0.0001683408630067644,
              2.9596307177461286e-05
            ]
          },
          "bce.fav": {
            "estimate": 0.00021220775729252628,
            "ci95": [
              9.463988392753438e-05,
              0.0003289595298873952
            ]
          },
          "bce.buy": {
            "estimate": -0.00022381751600106595,
            "ci95": [
              -0.0003161368433021579,
              -0.0001363585603669986
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.0063385613922419815,
            "ci95": [
              -0.011069853277680213,
              -0.0015048612747253147
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.009923658004024172,
            "ci95": [
              0.002894517207754108,
              0.016745325595207017
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.016054416427605966,
            "ci95": [
              -0.025686235485242567,
              -0.007246777362604465
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.016262219396266153,
            "ci95": [
              0.00810868858853633,
              0.026209547924774437
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.009715855035363985,
            "ci95": [
              -0.020155986710859615,
              -0.0010426701570207554
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.025978074431630138,
            "ci95": [
              -0.037412839520565885,
              -0.014313194296132347
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 3.700038553643495e-05,
            "ci95": [
              -8.654724264363324e-05,
              0.00016317153781091433
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.00024217570370060892,
            "ci95": [
              -0.00037967470984586436,
              -0.00010057626011157848
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.0001938495695929833,
            "ci95": [
              7.628415060421607e-05,
              0.00032665032541237825
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.0002791760892370439,
            "ci95": [
              -0.00044494669219834697,
              -0.00013188974062264857
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00015684918405654835,
            "ci95": [
              3.812537974794573e-05,
              0.0002998251522430182
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.0004360252732935922,
            "ci95": [
              0.00027653149721000846,
              0.0005801877880830676
            ]
          }
        }
      },
      "shared_carry__minus__shared_no_carry": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "shared_carry": 1.0,
          "distinct_carry": 0.0,
          "shared_no_carry": -1.0,
          "distinct_no_carry": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.0033440466839901317,
            "ci95": [
              -0.006376487384915059,
              0.00015664353969591156
            ]
          },
          "auc.is_click": {
            "estimate": -0.002072281636290807,
            "ci95": [
              -0.002901096968433389,
              -0.0011639239559934013
            ]
          },
          "auc.cart": {
            "estimate": 0.00021935354405844265,
            "ci95": [
              -0.004305264194594546,
              0.004890747657216152
            ]
          },
          "auc.fav": {
            "estimate": -0.012808448456509414,
            "ci95": [
              -0.021733348971723385,
              -0.004491329405541582
            ]
          },
          "auc.buy": {
            "estimate": 0.001285189812781251,
            "ci95": [
              -0.00495456079413055,
              0.008615104682170898
            ]
          },
          "bce.is_click": {
            "estimate": 0.00014774569946027216,
            "ci95": [
              6.0164530574972895e-05,
              0.00022652019982420736
            ]
          },
          "bce.cart": {
            "estimate": -0.00022414616157474115,
            "ci95": [
              -0.00032763821359605637,
              -0.00011971785678424513
            ]
          },
          "bce.fav": {
            "estimate": 0.00014714289183646173,
            "ci95": [
              -7.227064764650601e-06,
              0.00031581028144960985
            ]
          },
          "bce.buy": {
            "estimate": -7.403244738765288e-05,
            "ci95": [
              -0.00015280062481891243,
              -2.205276588299728e-06
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": -0.0022916351803492496,
            "ci95": [
              -0.007126217124866013,
              0.0022561261163572366
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": 0.010736166820218607,
            "ci95": [
              0.002084561749798164,
              0.019686211238046394
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": -0.003357471449072058,
            "ci95": [
              -0.010795503200718603,
              0.0030647043502525463
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": 0.013027802000567856,
            "ci95": [
              0.004362509946404797,
              0.022625827576296244
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.0010658362687228085,
            "ci95": [
              -0.009579827260529328,
              0.006350080770627593
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": -0.014093638269290665,
            "ci95": [
              -0.026201450372216238,
              -0.004426104412547037
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": 0.0003718918610350133,
            "ci95": [
              0.00023479592273935974,
              0.00050731260700262
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 6.028076238104307e-07,
            "ci95": [
              -0.00017866752060327444,
              0.00017367404141172557
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": 0.00022177814684792505,
            "ci95": [
              0.00011240389502498757,
              0.00033905931978755495
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": -0.0003712890534112029,
            "ci95": [
              -0.0005545705387859595,
              -0.0002081441762627038
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": -0.00015011371418708827,
            "ci95": [
              -0.0002751484681312537,
              -2.1687400934500214e-05
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": 0.00022117533922411461,
            "ci95": [
              7.008263278629225e-05,
              0.0004020452896097417
            ]
          }
        }
      },
      "distinct_carry__minus__shared_carry": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "shared_carry": -1.0,
          "distinct_carry": 1.0,
          "shared_no_carry": 0.0,
          "distinct_no_carry": 0.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": -0.00027658960714271874,
            "ci95": [
              -0.0036587174143853053,
              0.002922870940870184
            ]
          },
          "auc.is_click": {
            "estimate": 0.004048737064009544,
            "ci95": [
              0.0032030964111110062,
              0.004820854242960365
            ]
          },
          "auc.cart": {
            "estimate": -0.007168538377014033,
            "ci95": [
              -0.0125136648327246,
              -0.0011832114869971962
            ]
          },
          "auc.fav": {
            "estimate": 0.005511647194565761,
            "ci95": [
              -0.0017768662329467637,
              0.01206243185400375
            ]
          },
          "auc.buy": {
            "estimate": -0.003498204310132147,
            "ci95": [
              -0.011611980229133687,
              0.00472865845148192
            ]
          },
          "bce.is_click": {
            "estimate": -0.00048377439264846167,
            "ci95": [
              -0.0005579723537022102,
              -0.00040798034901303604
            ]
          },
          "bce.cart": {
            "estimate": 0.00024524810662997915,
            "ci95": [
              0.00013388655041395676,
              0.00035509123630977115
            ]
          },
          "bce.fav": {
            "estimate": -0.0001684499201298692,
            "ci95": [
              -0.00026930606894545695,
              -3.6590339313623043e-05
            ]
          },
          "bce.buy": {
            "estimate": -6.341842983694987e-06,
            "ci95": [
              -8.287031589837077e-05,
              8.656582924966757e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.011217275441023578,
            "ci95": [
              0.005268199188632463,
              0.016678136250069014
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.0014629101305562164,
            "ci95": [
              -0.007755418385008827,
              0.005708569465124399
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.007546941374141691,
            "ci95": [
              -0.001481260483024388,
              0.015813185781261622
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.012680185571579794,
            "ci95": [
              -0.022013792782011033,
              -0.0030988248337413516
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.0036703340668818862,
            "ci95": [
              -0.01160078705826605,
              0.005087733390641487
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.009009851504697908,
            "ci95": [
              -0.0022304047477526593,
              0.020235792132454144
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.0007290224992784408,
            "ci95": [
              -0.0008717345334294164,
              -0.0005958566844204607
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": -0.00031532447251859247,
            "ci95": [
              -0.0004587189797000202,
              -0.0001878446546086529
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0004774325496647667,
            "ci95": [
              -0.000598105396464037,
              -0.0003682730263395025
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.00041369802675984835,
            "ci95": [
              0.0002442574988004851,
              0.0005546237320333615
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00025158994961367413,
            "ci95": [
              0.00013114875674953208,
              0.0003679610620303196
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.00016210807714617421,
            "ci95": [
              -0.0003107008030427006,
              -1.6025225751227682e-05
            ]
          }
        }
      },
      "round_distinction_by_carry_interaction": {
        "coefficients": {
          "mature_r2": 0.0,
          "cross8_mature": 0.0,
          "shared_carry": -1.0,
          "distinct_carry": 1.0,
          "shared_no_carry": 1.0,
          "distinct_no_carry": -1.0
        },
        "metrics": {
          "macro_auc": {
            "estimate": 0.006910164870632368,
            "ci95": [
              0.0022352313157072714,
              0.011162313956113803
            ]
          },
          "auc.is_click": {
            "estimate": 0.006444741265639764,
            "ci95": [
              0.005154744914931855,
              0.007596694513746946
            ]
          },
          "auc.cart": {
            "estimate": -0.0011860184627686676,
            "ci95": [
              -0.008176642809233026,
              0.005500630457601698
            ]
          },
          "auc.fav": {
            "estimate": 0.020417326854844875,
            "ci95": [
              0.008025400625413301,
              0.031888149773463076
            ]
          },
          "auc.buy": {
            "estimate": 0.0019646098248135013,
            "ci95": [
              -0.008677049764914463,
              0.012794482018612096
            ]
          },
          "bce.is_click": {
            "estimate": -0.0005324274127773421,
            "ci95": [
              -0.0006370337752105131,
              -0.00040978825354679593
            ]
          },
          "bce.cart": {
            "estimate": 0.0001989097962389573,
            "ci95": [
              5.359455861695031e-05,
              0.0003321151186739102
            ]
          },
          "bce.fav": {
            "estimate": -0.000651063001921956,
            "ci95": [
              -0.000841971056642634,
              -0.0004189358921363577
            ]
          },
          "bce.buy": {
            "estimate": -7.855632657234775e-05,
            "ci95": [
              -0.00018403598138465004,
              3.322503984070643e-05
            ]
          },
          "auc.is_click_minus_cart": {
            "estimate": 0.007630759728408432,
            "ci95": [
              0.000791261436428506,
              0.01432892253692511
            ]
          },
          "auc.is_click_minus_fav": {
            "estimate": -0.013972585589205111,
            "ci95": [
              -0.025595530831840484,
              -0.0014082995552591945
            ]
          },
          "auc.is_click_minus_buy": {
            "estimate": 0.004480131440826263,
            "ci95": [
              -0.006685101899936088,
              0.015405324239041222
            ]
          },
          "auc.cart_minus_fav": {
            "estimate": -0.021603345317613543,
            "ci95": [
              -0.03534470719036087,
              -0.007480968279894292
            ]
          },
          "auc.cart_minus_buy": {
            "estimate": -0.003150628287582169,
            "ci95": [
              -0.014806643479490346,
              0.00892429510617557
            ]
          },
          "auc.fav_minus_buy": {
            "estimate": 0.018452717030031374,
            "ci95": [
              0.001975279502433969,
              0.03472398069432884
            ]
          },
          "bce.is_click_minus_cart": {
            "estimate": -0.0007313372090162994,
            "ci95": [
              -0.0009117680432500046,
              -0.0005438868036526819
            ]
          },
          "bce.is_click_minus_fav": {
            "estimate": 0.00011863558914461392,
            "ci95": [
              -0.00012263562586974243,
              0.0003362809277361666
            ]
          },
          "bce.is_click_minus_buy": {
            "estimate": -0.0004538710862049943,
            "ci95": [
              -0.0006165163844854629,
              -0.0002955998573458529
            ]
          },
          "bce.cart_minus_fav": {
            "estimate": 0.0008499727981609133,
            "ci95": [
              0.0005821141022723191,
              0.0010809855570858287
            ]
          },
          "bce.cart_minus_buy": {
            "estimate": 0.00027746612281130505,
            "ci95": [
              0.00010254690636004424,
              0.00045208388795231594
            ]
          },
          "bce.fav_minus_buy": {
            "estimate": -0.0005725066753496082,
            "ci95": [
              -0.0007981551168739626,
              -0.00033525745657378284
            ]
          }
        }
      }
    },
    "parameter_count": 51724644,
    "initializer_sha256": "4f81bf632c885decdc491249310fd82cb27eacf53dd17bb959577399dfcf1c87",
    "formal_worker_cost": {
      "wall_seconds": 2516.3608919605613,
      "cpu_seconds": 3013.0740411909997,
      "peak_rss_bytes": 17103974400,
      "gpu_peak_bytes": 27584888832,
      "samples": 2301
    },
    "smoke_worker_cost": {
      "wall_seconds": 147.18908870220184,
      "cpu_seconds": 163.05105121100001,
      "peak_rss_bytes": 17162715136,
      "gpu_peak_bytes": 30049042432,
      "samples": 134
    },
    "forward_latency": {
      "mean_ms": 14.416476821899414,
      "p50_ms": 14.209216117858887,
      "p95_ms": 14.655354833602896,
      "batch_size": 1024,
      "warmups": 10,
      "iterations": 40
    },
    "latency_ratio_to_parent": 1.0182931341893615,
    "quality_candidate": false,
    "near_candidate": false
  },
  "validity": {
    "valid": true,
    "state": "passed",
    "violations": [],
    "no_prediction_to_work_feedback": true,
    "test_evaluated": false,
    "quality_summary": "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/reports/quality-summary.json",
    "infrastructure_limitations": [
      "Four successful smoke runners had outer verifier errors; recovered saved checks and fixed formal worker accepted without replay. Original failed terminals preserved.",
      "One training seed; no training-seed uncertainty.",
      "Postprocess dispatcher failed before any child; str/Path adapter fixed with original numerical/runtime sources unchanged."
    ],
    "source_and_input_identity_scope": "Inherited original input content hashes plus frozen run stat continuity and original row-stream fingerprints; actual arrays revalidated in statistical execution."
  },
  "decision": "discard",
  "decision_reason": "相对原两轮macro AUC下降0.006701且区间在零以下；未提供整体优化收益。",
  "decision_scope": "Research queue only; no incumbent promotion or new supported lesson.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/runs/shared_carry",
    "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/RESULTS.md",
    "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/EVIDENCE.md",
    "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/reports/diagnostic-analysis/result.json"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/runs/shared_carry",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/runs/shared_carry/result.json",
    "result_sha256": "282159711538efa675a0cb1b02d86457e70ed54410310ed6bfe118e9f7ab99e7",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/runs/shared_carry/manifest.json",
    "manifest_sha256": "10ae7418ed8995735a3c51da8ae8e26fa0baed9336e54f553239320250299807",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/runs/shared_carry/Taobao_Action/shared_carry.model",
    "checkpoint_sha256": "e32534929540a58f9488e5fe4c9be3904106ec748e49cad0e6d92ffb60d0c2a5",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/reports/paired-analysis/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
