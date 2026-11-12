# 淘宝最新轮：配对、读出诊断与测量记录

快照时间：2026-09-09T07:36:23.974768+00:00。返回[总览](README.md)或[淘宝全部实验](TAOBAO_EXPERIMENTS.md)。

本页补充正在归档中的报告。下方P/D/E编号只是在本文按证据路径建立的索引，不是主线程新建的experiment_id；完整8次训练只计入淘宝训练明细一次。未出现完整报告的项目保留待完成状态。

## P01–P10：预定的全部配对

**方法**：候选−参考；完整验证2,871,767行、268,201用户；500次用户簇配对bootstrap，统计seed20260906；四任务共享用户抽样权重。这里没有增加训练seed。AUC越高越好，BCE越低越好。区间仅条件于固定checkpoint，未校正连续筛选、多重比较或训练随机性。

| 索引 | 候选−参考 | 已保存状态 | ΔmacroAUC | 描述性95%CI |
|---|---|---|---:|---|
| [P01](#p01) | `role_q4_read_shared` − `ctx_q4` | completed | -0.002101239 | [-0.0029530018665683195,-0.0013466269325520333] |
| [P02](#p02) | `role_q4_read_task` − `ctx_q4` | completed | -0.002279866 | [-0.0030728592080578295,-0.001484301160999619] |
| [P03](#p03) | `role_q4_query_norm` − `ctx_q4` | completed | -0.001111250 | [-0.0019764427161915025,-0.0002361357190031594] |
| [P04](#p04) | `role_q4_query_projection` − `ctx_q4` | completed | -0.002169216 | [-0.0029324797408686034,-0.0012914051468726267] |
| [P05](#p05) | `role_q4_read_task` − `role_q4_read_shared` | completed | -0.000178626 | [-0.001016508869217094,0.0006256667651079935] |
| [P06](#p06) | `role_q8_read_shared` − `explore_dual_no_feedback` | completed | -0.007014073 | [-0.007866283972609859,-0.006060648655048171] |
| [P07](#p07) | `role_q8_read_task` − `explore_dual_no_feedback` | completed | -0.001858568 | [-0.002595481985899975,-0.0010398460633241318] |
| [P08](#p08) | `role_q8_query_norm` − `explore_dual_no_feedback` | completed | -0.003596423 | [-0.004485080957732017,-0.0027734499118867327] |
| [P09](#p09) | `role_q8_query_projection` − `explore_dual_no_feedback` | completed | -0.001941715 | [-0.002739531904242352,-0.001115766118959524] |
| [P10](#p10) | `role_q8_read_task` − `role_q8_read_shared` | completed | +0.005155505 | [0.004281966516499934,0.005986110310820656] |

<a id="p01"></a>
### P01 · `role_q4_read_shared` − `ctx_q4`

- **方法**：同q、从头独立训练的新结构对原循环；按上方固定协议做用户配对。
- **结果**：ΔmacroAUC -0.002101239；95%CI [-0.0029530018665683195,-0.0013466269325520333]。

| 任务 | ΔAUC | AUC 95%CI | ΔBCE | BCE 95%CI |
|---|---:|---|---:|---|
| is_click | +0.000505696 | [2.6248184558341778e-05,0.0010414479864392229] | -0.000290185 | [-0.0003466689446297367,-0.00023269097220463784] |
| cart | -0.004470052 | [-0.0062594183399964755,-0.0026745057917026516] | +0.000087209 | [3.4002677987146775e-05,0.00013486529450706375] |
| fav | -0.000531789 | [-0.0020479240817787153,0.00086325200463083] | -0.000005708 | [-4.6989830599412505e-05,3.576621762562861e-05] |
| buy | -0.003908813 | [-0.00602430685923093,-0.0018575527705035875] | +0.000071182 | [3.9769045848846304e-05,0.0001061410936073094] |

- **归因**：区间完全低于0，支持这个固定checkpoint对照中的macroAUC退化。结构和新增参数的优化轨迹同时变化，不能只归于某一机制。
- **后续假设**：把本对照与原循环质量、四任务取舍和实际延迟联合判断；不依据单个任务区间自动增加结构或seed。

<details><summary>六个任务间效应差（全量保留）</summary>

```json
[
  {
    "first_task": "is_click",
    "second_task": "cart",
    "sign": "first effect minus second effect",
    "delta_auc": 0.004975748025552562,
    "ci95_auc": [
      0.0032384201783790035,
      0.006794393140624485
    ],
    "delta_bce": -0.00037739336827720036,
    "ci95_bce": [
      -0.00044813403678623507,
      -0.00030489238889442217
    ]
  },
  {
    "first_task": "is_click",
    "second_task": "fav",
    "sign": "first effect minus second effect",
    "delta_auc": 0.0010374851334351565,
    "ci95_auc": [
      -0.00042773436244108727,
      0.0026380308978828614
    ],
    "delta_bce": -0.0002844769399318444,
    "ci95_bce": [
      -0.00035140106509320057,
      -0.00021578325176714198
    ]
  },
  {
    "first_task": "is_click",
    "second_task": "buy",
    "sign": "first effect minus second effect",
    "delta_auc": 0.00441450993232928,
    "ci95_auc": [
      0.002389567207572402,
      0.006553766668221826
    ],
    "delta_bce": -0.0003613667233740718,
    "ci95_bce": [
      -0.00042440567198954874,
      -0.0002956827866523001
    ]
  },
  {
    "first_task": "cart",
    "second_task": "fav",
    "sign": "first effect minus second effect",
    "delta_auc": -0.003938262892117406,
    "ci95_auc": [
      -0.006303911466350953,
      -0.0017363861821390993
    ],
    "delta_bce": 9.291642834535593e-05,
    "ci95_bce": [
      3.206176427239104e-05,
      0.00015038381322392814
    ]
  },
  {
    "first_task": "cart",
    "second_task": "buy",
    "sign": "first effect minus second effect",
    "delta_auc": -0.0005612380932232819,
    "ci95_auc": [
      -0.0032861288004701504,
      0.0022586522018049767
    ],
    "delta_bce": 1.602664490312858e-05,
    "ci95_bce": [
      -4.452926318084869e-05,
      6.762707816942432e-05
    ]
  },
  {
    "first_task": "fav",
    "second_task": "buy",
    "sign": "first effect minus second effect",
    "delta_auc": 0.0033770247988941238,
    "ci95_auc": [
      0.0007670387635121152,
      0.005791043600020251
    ],
    "delta_bce": -7.688978344222735e-05,
    "ci95_bce": [
      -0.0001296740863324521,
      -2.7711928691913352e-05
    ]
  }
]
```

</details>

- **证据**：`qformer-loop/taobao-transfer/recurrence-readout-retrieval/reports/paired-analysis/pairs/ctx_q4--role_q4_read_shared/result.json`（SHA256前缀 `278717782719`）。

<a id="p02"></a>
### P02 · `role_q4_read_task` − `ctx_q4`

- **方法**：同q、从头独立训练的新结构对原循环；按上方固定协议做用户配对。
- **结果**：ΔmacroAUC -0.002279866；95%CI [-0.0030728592080578295,-0.001484301160999619]。

| 任务 | ΔAUC | AUC 95%CI | ΔBCE | BCE 95%CI |
|---|---:|---|---:|---|
| is_click | -0.001755066 | [-0.0022645828573958776,-0.0012068514180407167] | +0.000145163 | [9.446455090790458e-05,0.00019969636485890336] |
| cart | -0.003455411 | [-0.005024766507445802,-0.00173376331746572] | +0.000112805 | [6.543215025687304e-05,0.0001587361546518006] |
| fav | -0.001959202 | [-0.0034012493757074242,-0.0005388332822342918] | -0.000008315 | [-4.874401631846147e-05,3.388936166507473e-05] |
| buy | -0.001949783 | [-0.004069568363010076,0.0002221980713810669] | +0.000046615 | [1.1343145117855405e-05,9.039529301997691e-05] |

- **归因**：区间完全低于0，支持这个固定checkpoint对照中的macroAUC退化。结构和新增参数的优化轨迹同时变化，不能只归于某一机制。
- **后续假设**：把本对照与原循环质量、四任务取舍和实际延迟联合判断；不依据单个任务区间自动增加结构或seed。

<details><summary>六个任务间效应差（全量保留）</summary>

```json
[
  {
    "first_task": "is_click",
    "second_task": "cart",
    "sign": "first effect minus second effect",
    "delta_auc": 0.0017003445506895698,
    "ci95_auc": [
      3.81001728169017e-05,
      0.0033979571368489337
    ],
    "delta_bce": 3.2357639551731644e-05,
    "ci95_bce": [
      -3.786173110068328e-05,
      9.866959380399084e-05
    ]
  },
  {
    "first_task": "is_click",
    "second_task": "fav",
    "sign": "first effect minus second effect",
    "delta_auc": 0.00020413629505444675,
    "ci95_auc": [
      -0.0013151876142090512,
      0.0018371519677980105
    ],
    "delta_bce": 0.00015347765851761263,
    "ci95_bce": [
      8.670247386771045e-05,
      0.00021989053240365627
    ]
  },
  {
    "first_task": "is_click",
    "second_task": "buy",
    "sign": "first effect minus second effect",
    "delta_auc": 0.0001947172584462198,
    "ci95_auc": [
      -0.0021908201821930705,
      0.002435364735739112
    ],
    "delta_bce": 9.854835103086885e-05,
    "ci95_bce": [
      3.091196663894873e-05,
      0.00016339010003600549
    ]
  },
  {
    "first_task": "cart",
    "second_task": "fav",
    "sign": "first effect minus second effect",
    "delta_auc": -0.001496208255635123,
    "ci95_auc": [
      -0.003530715888607772,
      0.0007079691220497088
    ],
    "delta_bce": 0.00012112001896588098,
    "ci95_bce": [
      5.692558224684193e-05,
      0.00017731023145136818
    ]
  },
  {
    "first_task": "cart",
    "second_task": "buy",
    "sign": "first effect minus second effect",
    "delta_auc": -0.00150562729224335,
    "ci95_auc": [
      -0.004138049061299651,
      0.001180262443742138
    ],
    "delta_bce": 6.61907114791372e-05,
    "ci95_bce": [
      8.367029189788871e-06,
      0.00012270032045360308
    ]
  },
  {
    "first_task": "fav",
    "second_task": "buy",
    "sign": "first effect minus second effect",
    "delta_auc": -9.41903660822696e-06,
    "ci95_auc": [
      -0.00257580995161015,
      0.0024171550716872697
    ],
    "delta_bce": -5.492930748674378e-05,
    "ci95_bce": [
      -0.00010809557785520344,
      -2.564072306486682e-06
    ]
  }
]
```

</details>

- **证据**：`qformer-loop/taobao-transfer/recurrence-readout-retrieval/reports/paired-analysis/pairs/ctx_q4--role_q4_read_task/result.json`（SHA256前缀 `7ae137eac9ec`）。

<a id="p03"></a>
### P03 · `role_q4_query_norm` − `ctx_q4`

- **方法**：同q、从头独立训练的新结构对原循环；按上方固定协议做用户配对。
- **结果**：ΔmacroAUC -0.001111250；95%CI [-0.0019764427161915025,-0.0002361357190031594]。

| 任务 | ΔAUC | AUC 95%CI | ΔBCE | BCE 95%CI |
|---|---:|---|---:|---|
| is_click | -0.000165529 | [-0.0006981293725845661,0.00039186260857095295] | -0.000225773 | [-0.0002800904868394131,-0.00017429315703489762] |
| cart | -0.000955084 | [-0.002673379171875043,0.0006423016689403287] | -0.000092915 | [-0.00016188543134427279,-2.6958016596004195e-05] |
| fav | -0.001859281 | [-0.00368511532236378,-0.0001316658659895056] | -0.000013641 | [-5.5948647823800463e-05,3.431410616305717e-05] |
| buy | -0.001465107 | [-0.0035275598685302945,0.00041772208665680194] | -0.000024751 | [-6.079051383304471e-05,1.4843433268658395e-05] |

- **归因**：区间完全低于0，支持这个固定checkpoint对照中的macroAUC退化。结构和新增参数的优化轨迹同时变化，不能只归于某一机制。
- **后续假设**：把本对照与原循环质量、四任务取舍和实际延迟联合判断；不依据单个任务区间自动增加结构或seed。

<details><summary>六个任务间效应差（全量保留）</summary>

```json
[
  {
    "first_task": "is_click",
    "second_task": "cart",
    "sign": "first effect minus second effect",
    "delta_auc": 0.0007895546847572765,
    "ci95_auc": [
      -0.0008522777461061809,
      0.002472585362156848
    ],
    "delta_bce": -0.00013285883401372965,
    "ci95_bce": [
      -0.00021475573386950112,
      -4.014651762385972e-05
    ]
  },
  {
    "first_task": "is_click",
    "second_task": "fav",
    "sign": "first effect minus second effect",
    "delta_auc": 0.0016937520515726279,
    "ci95_auc": [
      -9.285761283570336e-05,
      0.0037215348844962205
    ],
    "delta_bce": -0.00021213193571858766,
    "ci95_bce": [
      -0.0002882496463583654,
      -0.0001408814997276727
    ]
  },
  {
    "first_task": "is_click",
    "second_task": "buy",
    "sign": "first effect minus second effect",
    "delta_auc": 0.0012995778148569004,
    "ci95_auc": [
      -0.0005255546531495069,
      0.003515234184644056
    ],
    "delta_bce": -0.00020102191509060338,
    "ci95_bce": [
      -0.0002656464917559338,
      -0.0001382799437071037
    ]
  },
  {
    "first_task": "cart",
    "second_task": "fav",
    "sign": "first effect minus second effect",
    "delta_auc": 0.0009041973668153513,
    "ci95_auc": [
      -0.0012415619642550501,
      0.0031924929739276958
    ],
    "delta_bce": -7.927310170485802e-05,
    "ci95_bce": [
      -0.00016184392956492384,
      -3.365605516039298e-06
    ]
  },
  {
    "first_task": "cart",
    "second_task": "buy",
    "sign": "first effect minus second effect",
    "delta_auc": 0.0005100231300996239,
    "ci95_auc": [
      -0.0019921598758707248,
      0.0031121412330717855
    ],
    "delta_bce": -6.816308107687372e-05,
    "ci95_bce": [
      -0.00014726829013349645,
      8.122096072210616e-06
    ]
  },
  {
    "first_task": "fav",
    "second_task": "buy",
    "sign": "first effect minus second effect",
    "delta_auc": -0.0003941742367157275,
    "ci95_auc": [
      -0.0027719262342183358,
      0.0024025198330385558
    ],
    "delta_bce": 1.1110020627984296e-05,
    "ci95_bce": [
      -4.589664145699596e-05,
      6.750946227744821e-05
    ]
  }
]
```

</details>

- **证据**：`qformer-loop/taobao-transfer/recurrence-readout-retrieval/reports/paired-analysis/pairs/ctx_q4--role_q4_query_norm/result.json`（SHA256前缀 `9a753b7e11cc`）。

<a id="p04"></a>
### P04 · `role_q4_query_projection` − `ctx_q4`

- **方法**：同q、从头独立训练的新结构对原循环；按上方固定协议做用户配对。
- **结果**：ΔmacroAUC -0.002169216；95%CI [-0.0029324797408686034,-0.0012914051468726267]。

| 任务 | ΔAUC | AUC 95%CI | ΔBCE | BCE 95%CI |
|---|---:|---|---:|---|
| is_click | -0.001521986 | [-0.0020922064558857507,-0.0010188591191837599] | +0.000016527 | [-3.727190261799292e-05,7.417231377856729e-05] |
| cart | -0.002020899 | [-0.0037376585880127627,-0.0004065728474060675] | -0.000080725 | [-0.00014263711418364204,-1.7750019350736018e-05] |
| fav | -0.002402198 | [-0.003949590509445783,-0.0008240066584425915] | -0.000043403 | [-8.996265355630389e-05,7.563396094379238e-06] |
| buy | -0.002731780 | [-0.004669979468751595,-0.0008366045813912127] | -0.000028573 | [-6.695988530452798e-05,1.1816748380863092e-05] |

- **归因**：区间完全低于0，支持这个固定checkpoint对照中的macroAUC退化。结构和新增参数的优化轨迹同时变化，不能只归于某一机制。
- **后续假设**：把本对照与原循环质量、四任务取舍和实际延迟联合判断；不依据单个任务区间自动增加结构或seed。

<details><summary>六个任务间效应差（全量保留）</summary>

```json
[
  {
    "first_task": "is_click",
    "second_task": "cart",
    "sign": "first effect minus second effect",
    "delta_auc": 0.0004989131109661704,
    "ci95_auc": [
      -0.0010129744359287957,
      0.0022348876628489997
    ],
    "delta_bce": 9.725139925779789e-05,
    "ci95_bce": [
      1.5460056572118868e-05,
      0.00018019551094714152
    ]
  },
  {
    "first_task": "is_click",
    "second_task": "fav",
    "sign": "first effect minus second effect",
    "delta_auc": 0.0008802124865772143,
    "ci95_auc": [
      -0.0007369462508144575,
      0.002577319773108957
    ],
    "delta_bce": 5.992904383322522e-05,
    "ci95_bce": [
      -1.6089525330729717e-05,
      0.0001281816033092473
    ]
  },
  {
    "first_task": "is_click",
    "second_task": "buy",
    "sign": "first effect minus second effect",
    "delta_auc": 0.0012097940546057506,
    "ci95_auc": [
      -0.0008826770452571963,
      0.003276471345477655
    ],
    "delta_bce": 4.5099618054960234e-05,
    "ci95_bce": [
      -2.2902042364935e-05,
      0.00011875410985025478
    ]
  },
  {
    "first_task": "cart",
    "second_task": "fav",
    "sign": "first effect minus second effect",
    "delta_auc": 0.0003812993756110439,
    "ci95_auc": [
      -0.0020089484163589243,
      0.002463573145999315
    ],
    "delta_bce": -3.7322355424572673e-05,
    "ci95_bce": [
      -0.0001137567923895494,
      4.250970498126278e-05
    ]
  },
  {
    "first_task": "cart",
    "second_task": "buy",
    "sign": "first effect minus second effect",
    "delta_auc": 0.0007108809436395802,
    "ci95_auc": [
      -0.0019870667222289693,
      0.0031052115349400884
    ],
    "delta_bce": -5.215178120283766e-05,
    "ci95_bce": [
      -0.00011809685976429324,
      2.0873215937055602e-05
    ]
  },
  {
    "first_task": "fav",
    "second_task": "buy",
    "sign": "first effect minus second effect",
    "delta_auc": 0.00032958156802853633,
    "ci95_auc": [
      -0.002330611290964557,
      0.002798873617844393
    ],
    "delta_bce": -1.4829425778264984e-05,
    "ci95_bce": [
      -7.338052646718588e-05,
      4.9337028973150904e-05
    ]
  }
]
```

</details>

- **证据**：`qformer-loop/taobao-transfer/recurrence-readout-retrieval/reports/paired-analysis/pairs/ctx_q4--role_q4_query_projection/result.json`（SHA256前缀 `7146576e91bc`）。

<a id="p05"></a>
### P05 · `role_q4_read_task` − `role_q4_read_shared`

- **方法**：同q、任务独立读出对共享读出的独立训练控制；按上方固定协议做用户配对。
- **结果**：ΔmacroAUC -0.000178626；95%CI [-0.001016508869217094,0.0006256667651079935]。

| 任务 | ΔAUC | AUC 95%CI | ΔBCE | BCE 95%CI |
|---|---:|---|---:|---|
| is_click | -0.002260763 | [-0.0028072817651096397,-0.0016954582990925816] | +0.000435347 | [0.0003719817000894537,0.0004985949166242193] |
| cart | +0.001014641 | [-0.0005985236244905906,0.0027599731670048507] | +0.000025596 | [-3.2757962831027185e-05,7.65980643078643e-05] |
| fav | -0.001427414 | [-0.002761644312183953,5.54994615482106e-06] | -0.000002607 | [-4.076334897382632e-05,4.0867602671004166e-05] |
| buy | +0.001959030 | [-0.00017576240688215168,0.004190024725234492] | -0.000024568 | [-5.981169918578172e-05,1.4298632225633873e-05] |

- **归因**：区间跨0，本比较仍不确定，不能认定等价。task胜过shared也须高于原循环，不能把战胜较弱控制当作结构成功。
- **后续假设**：把本对照与原循环质量、四任务取舍和实际延迟联合判断；不依据单个任务区间自动增加结构或seed。

<details><summary>六个任务间效应差（全量保留）</summary>

```json
[
  {
    "first_task": "is_click",
    "second_task": "cart",
    "sign": "first effect minus second effect",
    "delta_auc": -0.0032754034748629923,
    "ci95_auc": [
      -0.005086046122448912,
      -0.0015187584332668263
    ],
    "delta_bce": 0.0004097510078289382,
    "ci95_bce": [
      0.00033506689778615517,
      0.00048588563597128516
    ]
  },
  {
    "first_task": "is_click",
    "second_task": "fav",
    "sign": "first effect minus second effect",
    "delta_auc": -0.0008333488383807097,
    "ci95_auc": [
      -0.002425676189076123,
      0.0005922521042954789
    ],
    "delta_bce": 0.0004379545984494627,
    "ci95_bce": [
      0.00036972642419421704,
      0.0005099044084960997
    ]
  },
  {
    "first_task": "is_click",
    "second_task": "buy",
    "sign": "first effect minus second effect",
    "delta_auc": -0.0042197926738830605,
    "ci95_auc": [
      -0.006707569264535898,
      -0.0020338135854085005
    ],
    "delta_bce": 0.0004599150744049445,
    "ci95_bce": [
      0.0003868029808563391,
      0.0005290314906236062
    ]
  },
  {
    "first_task": "cart",
    "second_task": "fav",
    "sign": "first effect minus second effect",
    "delta_auc": 0.0024420546364822826,
    "ci95_auc": [
      0.0004184748022359104,
      0.0046814112320468825
    ],
    "delta_bce": 2.8203590620524456e-05,
    "ci95_bce": [
      -3.616319226659449e-05,
      9.324281564806992e-05
    ]
  },
  {
    "first_task": "cart",
    "second_task": "buy",
    "sign": "first effect minus second effect",
    "delta_auc": -0.0009443891990200681,
    "ci95_auc": [
      -0.0035197699608861626,
      0.0016640264328121866
    ],
    "delta_bce": 5.016406657600629e-05,
    "ci95_bce": [
      -1.2159110347048261e-05,
      0.00011189765255286948
    ]
  },
  {
    "first_task": "fav",
    "second_task": "buy",
    "sign": "first effect minus second effect",
    "delta_auc": -0.0033864438355023507,
    "ci95_auc": [
      -0.006101679890393874,
      -0.0007902215844005049
    ],
    "delta_bce": 2.196047595548183e-05,
    "ci95_bce": [
      -2.6772081180213854e-05,
      7.884677328703541e-05
    ]
  }
]
```

</details>

- **证据**：`qformer-loop/taobao-transfer/recurrence-readout-retrieval/reports/paired-analysis/pairs/role_q4_read_shared--role_q4_read_task/result.json`（SHA256前缀 `83682268eddd`）。

<a id="p06"></a>
### P06 · `role_q8_read_shared` − `explore_dual_no_feedback`

- **方法**：同q、从头独立训练的新结构对原循环；按上方固定协议做用户配对。
- **结果**：ΔmacroAUC -0.007014073；95%CI [-0.007866283972609859,-0.006060648655048171]。

| 任务 | ΔAUC | AUC 95%CI | ΔBCE | BCE 95%CI |
|---|---:|---|---:|---|
| is_click | -0.009882998 | [-0.010549254320915968,-0.009191986525701617] | +0.001484733 | [0.0014193701598001554,0.0015532871449918165] |
| cart | -0.006264227 | [-0.00787204014607729,-0.004614500782170739] | +0.000065792 | [2.1343847017903175e-05,0.00011147901257007701] |
| fav | -0.006453114 | [-0.008007351482761303,-0.004717493157013139] | +0.000059921 | [1.1703305253908227e-05,9.911772290737534e-05] |
| buy | -0.005455954 | [-0.007701085050745307,-0.0030370570883265255] | +0.000041051 | [8.665197328042631e-06,7.453796144832787e-05] |

- **归因**：区间完全低于0，支持这个固定checkpoint对照中的macroAUC退化。结构和新增参数的优化轨迹同时变化，不能只归于某一机制。
- **后续假设**：把本对照与原循环质量、四任务取舍和实际延迟联合判断；不依据单个任务区间自动增加结构或seed。

<details><summary>六个任务间效应差（全量保留）</summary>

```json
[
  {
    "first_task": "is_click",
    "second_task": "cart",
    "sign": "first effect minus second effect",
    "delta_auc": -0.0036187707592248985,
    "ci95_auc": [
      -0.005396586463850795,
      -0.001928873776518303
    ],
    "delta_bce": 0.001418940623538664,
    "ci95_bce": [
      0.0013407337900281655,
      0.001497719308460673
    ]
  },
  {
    "first_task": "is_click",
    "second_task": "fav",
    "sign": "first effect minus second effect",
    "delta_auc": -0.0034298832264173518,
    "ci95_auc": [
      -0.0051765771312181795,
      -0.0016935220737793522
    ],
    "delta_bce": 0.0014248120129268845,
    "ci95_bce": [
      0.0013512720387774343,
      0.0015025932828289749
    ]
  },
  {
    "first_task": "is_click",
    "second_task": "buy",
    "sign": "first effect minus second effect",
    "delta_auc": -0.004427043841934042,
    "ci95_auc": [
      -0.006755857009567981,
      -0.002230311415101644
    ],
    "delta_bce": 0.0014436820738484358,
    "ci95_bce": [
      0.0013688006859780132,
      0.0015205607346313618
    ]
  },
  {
    "first_task": "cart",
    "second_task": "fav",
    "sign": "first effect minus second effect",
    "delta_auc": 0.00018888753280754678,
    "ci95_auc": [
      -0.0020295846167553006,
      0.002368913036629976
    ],
    "delta_bce": 5.871389388220383e-06,
    "ci95_bce": [
      -4.982626698521562e-05,
      7.140965242416e-05
    ]
  },
  {
    "first_task": "cart",
    "second_task": "buy",
    "sign": "first effect minus second effect",
    "delta_auc": -0.0008082730827091433,
    "ci95_auc": [
      -0.0034504889159078316,
      0.0021275239288314866
    ],
    "delta_bce": 2.474145030977175e-05,
    "ci95_bce": [
      -2.4200376054948573e-05,
      8.232116787809162e-05
    ]
  },
  {
    "first_task": "fav",
    "second_task": "buy",
    "sign": "first effect minus second effect",
    "delta_auc": -0.00099716061551669,
    "ci95_auc": [
      -0.0036925979716626975,
      0.0015933049731317371
    ],
    "delta_bce": 1.8870060921551367e-05,
    "ci95_bce": [
      -3.603743584350878e-05,
      6.618153668200061e-05
    ]
  }
]
```

</details>

- **证据**：`qformer-loop/taobao-transfer/recurrence-readout-retrieval/reports/paired-analysis/pairs/explore_dual_no_feedback--role_q8_read_shared/result.json`（SHA256前缀 `d389abd55bf7`）。

<a id="p07"></a>
### P07 · `role_q8_read_task` − `explore_dual_no_feedback`

- **方法**：同q、从头独立训练的新结构对原循环；按上方固定协议做用户配对。
- **结果**：ΔmacroAUC -0.001858568；95%CI [-0.002595481985899975,-0.0010398460633241318]。

| 任务 | ΔAUC | AUC 95%CI | ΔBCE | BCE 95%CI |
|---|---:|---|---:|---|
| is_click | -0.002735685 | [-0.003308252707205933,-0.002225603768067416] | +0.000432762 | [0.00037994891044707517,0.000491489213929126] |
| cart | -0.001830733 | [-0.003516831779483076,-0.00025259247451090723] | +0.000089643 | [3.761504329184786e-05,0.0001420995763580841] |
| fav | -0.001088610 | [-0.0027661912998773797,0.0006474454132731298] | +0.000035850 | [-8.108563708216627e-06,7.688560712765615e-05] |
| buy | -0.001779245 | [-0.003803054230408062,0.0003553716031241313] | +0.000077569 | [4.1924622676257595e-05,0.00012180661368185748] |

- **归因**：区间完全低于0，支持这个固定checkpoint对照中的macroAUC退化。结构和新增参数的优化轨迹同时变化，不能只归于某一机制。
- **后续假设**：把本对照与原循环质量、四任务取舍和实际延迟联合判断；不依据单个任务区间自动增加结构或seed。

<details><summary>六个任务间效应差（全量保留）</summary>

```json
[
  {
    "first_task": "is_click",
    "second_task": "cart",
    "sign": "first effect minus second effect",
    "delta_auc": -0.0009049517118556727,
    "ci95_auc": [
      -0.0025624982819400788,
      0.0008119466186016942
    ],
    "delta_bce": 0.0003431194091757479,
    "ci95_bce": [
      0.0002735817495423033,
      0.00041907266220989975
    ]
  },
  {
    "first_task": "is_click",
    "second_task": "fav",
    "sign": "first effect minus second effect",
    "delta_auc": -0.0016470755332980147,
    "ci95_auc": [
      -0.0034956982885989084,
      0.00023246769117068116
    ],
    "delta_bce": 0.0003969127052484311,
    "ci95_bce": [
      0.00032959235929753766,
      0.0004701858272163019
    ]
  },
  {
    "first_task": "is_click",
    "second_task": "buy",
    "sign": "first effect minus second effect",
    "delta_auc": -0.0009564404293752604,
    "ci95_auc": [
      -0.0032408956274536074,
      0.0012504049520099584
    ],
    "delta_bce": 0.00035519336263886486,
    "ci95_bce": [
      0.0002856046755176476,
      0.0004231275464780389
    ]
  },
  {
    "first_task": "cart",
    "second_task": "fav",
    "sign": "first effect minus second effect",
    "delta_auc": -0.000742123821442342,
    "ci95_auc": [
      -0.0030973392561120055,
      0.0015812773172613752
    ],
    "delta_bce": 5.3793296072683194e-05,
    "ci95_bce": [
      -9.34233842015094e-06,
      0.00012511972723157675
    ]
  },
  {
    "first_task": "cart",
    "second_task": "buy",
    "sign": "first effect minus second effect",
    "delta_auc": -5.148871751958772e-05,
    "ci95_auc": [
      -0.0029060934924578117,
      0.002547474306858055
    ],
    "delta_bce": 1.2073953463116936e-05,
    "ci95_bce": [
      -5.349119456202834e-05,
      6.966311236802401e-05
    ]
  },
  {
    "first_task": "fav",
    "second_task": "buy",
    "sign": "first effect minus second effect",
    "delta_auc": 0.0006906351039227543,
    "ci95_auc": [
      -0.002084108837731699,
      0.003220081444155847
    ],
    "delta_bce": -4.171934260956626e-05,
    "ci95_bce": [
      -9.844640357601588e-05,
      1.4539140501980428e-05
    ]
  }
]
```

</details>

- **证据**：`qformer-loop/taobao-transfer/recurrence-readout-retrieval/reports/paired-analysis/pairs/explore_dual_no_feedback--role_q8_read_task/result.json`（SHA256前缀 `441202f213a1`）。

<a id="p08"></a>
### P08 · `role_q8_query_norm` − `explore_dual_no_feedback`

- **方法**：同q、从头独立训练的新结构对原循环；按上方固定协议做用户配对。
- **结果**：ΔmacroAUC -0.003596423；95%CI [-0.004485080957732017,-0.0027734499118867327]。

| 任务 | ΔAUC | AUC 95%CI | ΔBCE | BCE 95%CI |
|---|---:|---|---:|---|
| is_click | -0.005592164 | [-0.006230944807287692,-0.005026187433487261] | +0.000459603 | [0.0004035957744516398,0.0005216236218604024] |
| cart | -0.002395549 | [-0.0038497648316073605,-0.0007340829478956563] | -0.000024130 | [-7.31760841075072e-05,2.2310326045248087e-05] |
| fav | -0.003817083 | [-0.005554487896866892,-0.0020741944600991937] | +0.000053872 | [5.676044889223532e-06,9.740316526626105e-05] |
| buy | -0.002580897 | [-0.004706007517552319,-0.000621299650615788] | +0.000040686 | [4.598512763968304e-06,7.462072565566766e-05] |

- **归因**：区间完全低于0，支持这个固定checkpoint对照中的macroAUC退化。结构和新增参数的优化轨迹同时变化，不能只归于某一机制。
- **后续假设**：把本对照与原循环质量、四任务取舍和实际延迟联合判断；不依据单个任务区间自动增加结构或seed。

<details><summary>六个任务间效应差（全量保留）</summary>

```json
[
  {
    "first_task": "is_click",
    "second_task": "cart",
    "sign": "first effect minus second effect",
    "delta_auc": -0.003196615830186378,
    "ci95_auc": [
      -0.004911089642052327,
      -0.0016613542739376268
    ],
    "delta_bce": 0.0004837330040048907,
    "ci95_bce": [
      0.0004154601135549586,
      0.000558163825723989
    ]
  },
  {
    "first_task": "is_click",
    "second_task": "fav",
    "sign": "first effect minus second effect",
    "delta_auc": -0.0017750814717290009,
    "ci95_auc": [
      -0.0035178312567417684,
      5.397799626506158e-05
    ],
    "delta_bce": 0.00040573148272076277,
    "ci95_bce": [
      0.00033443625808539027,
      0.0004848809138394621
    ]
  },
  {
    "first_task": "is_click",
    "second_task": "buy",
    "sign": "first effect minus second effect",
    "delta_auc": -0.003011267650398386,
    "ci95_auc": [
      -0.005082040979511019,
      -0.000911837087893275
    ],
    "delta_bce": 0.00041891680570918,
    "ci95_bce": [
      0.00035254611589070683,
      0.0004886225351356372
    ]
  },
  {
    "first_task": "cart",
    "second_task": "fav",
    "sign": "first effect minus second effect",
    "delta_auc": 0.001421534358457377,
    "ci95_auc": [
      -0.0007060963217292799,
      0.003918939395283212
    ],
    "delta_bce": -7.800152128412791e-05,
    "ci95_bce": [
      -0.00014046754123771905,
      -1.5931846508570038e-05
    ]
  },
  {
    "first_task": "cart",
    "second_task": "buy",
    "sign": "first effect minus second effect",
    "delta_auc": 0.0001853481797879919,
    "ci95_auc": [
      -0.0022035512408553892,
      0.002910895402157318
    ],
    "delta_bce": -6.481619829571065e-05,
    "ci95_bce": [
      -0.00012318306404597644,
      -9.558568713952668e-06
    ]
  },
  {
    "first_task": "fav",
    "second_task": "buy",
    "sign": "first effect minus second effect",
    "delta_auc": -0.001236186178669385,
    "ci95_auc": [
      -0.0035550887347631896,
      0.0012677198857852844
    ],
    "delta_bce": 1.3185322988417272e-05,
    "ci95_bce": [
      -4.531773509142044e-05,
      6.40095803752832e-05
    ]
  }
]
```

</details>

- **证据**：`qformer-loop/taobao-transfer/recurrence-readout-retrieval/reports/paired-analysis/pairs/explore_dual_no_feedback--role_q8_query_norm/result.json`（SHA256前缀 `5cda5b98db99`）。

<a id="p09"></a>
### P09 · `role_q8_query_projection` − `explore_dual_no_feedback`

- **方法**：同q、从头独立训练的新结构对原循环；按上方固定协议做用户配对。
- **结果**：ΔmacroAUC -0.001941715；95%CI [-0.002739531904242352,-0.001115766118959524]。

| 任务 | ΔAUC | AUC 95%CI | ΔBCE | BCE 95%CI |
|---|---:|---|---:|---|
| is_click | -0.000471282 | [-0.0010388135814327903,0.00010594787382749698] | +0.000063379 | [8.60353566614712e-06,0.00012143346978408215] |
| cart | -0.002565459 | [-0.003952716897640068,-0.0010001796050720213] | +0.000114242 | [5.649530972308769e-05,0.0001690501255871668] |
| fav | -0.001952044 | [-0.003561744957726087,-0.00022105288305731088] | +0.000087427 | [4.516808760938133e-05,0.0001273356626423657] |
| buy | -0.002778074 | [-0.004898975447640447,-0.00052554723528866] | +0.000090139 | [5.524601925654527e-05,0.00012587612780076484] |

- **归因**：区间完全低于0，支持这个固定checkpoint对照中的macroAUC退化。结构和新增参数的优化轨迹同时变化，不能只归于某一机制。
- **后续假设**：把本对照与原循环质量、四任务取舍和实际延迟联合判断；不依据单个任务区间自动增加结构或seed。

<details><summary>六个任务间效应差（全量保留）</summary>

```json
[
  {
    "first_task": "is_click",
    "second_task": "cart",
    "sign": "first effect minus second effect",
    "delta_auc": 0.0020941776820260216,
    "ci95_auc": [
      0.0004223813826097267,
      0.003579501456863472
    ],
    "delta_bce": -5.086304680862447e-05,
    "ci95_bce": [
      -0.0001282987218908433,
      2.7353435745247983e-05
    ]
  },
  {
    "first_task": "is_click",
    "second_task": "fav",
    "sign": "first effect minus second effect",
    "delta_auc": 0.0014807624037542766,
    "ci95_auc": [
      -0.0003427417520034419,
      0.0031368322450758346
    ],
    "delta_bce": -2.404868197211536e-05,
    "ci95_bce": [
      -9.24732717524514e-05,
      4.523332970127901e-05
    ]
  },
  {
    "first_task": "is_click",
    "second_task": "buy",
    "sign": "first effect minus second effect",
    "delta_auc": 0.002306792834405491,
    "ci95_auc": [
      -2.026531477601894e-05,
      0.00455617234420998
    ],
    "delta_bce": -2.6760291848925162e-05,
    "ci95_bce": [
      -9.405664276990762e-05,
      4.315298066088868e-05
    ]
  },
  {
    "first_task": "cart",
    "second_task": "fav",
    "sign": "first effect minus second effect",
    "delta_auc": -0.000613415278271745,
    "ci95_auc": [
      -0.002961996595054686,
      0.0015128932235315385
    ],
    "delta_bce": 2.6814364836509112e-05,
    "ci95_bce": [
      -4.059561765328485e-05,
      9.151772332021291e-05
    ]
  },
  {
    "first_task": "cart",
    "second_task": "buy",
    "sign": "first effect minus second effect",
    "delta_auc": 0.00021261515237946949,
    "ci95_auc": [
      -0.0023509619299884738,
      0.002947156835542572
    ],
    "delta_bce": 2.410275495969931e-05,
    "ci95_bce": [
      -4.0619865827564544e-05,
      8.825072369828603e-05
    ]
  },
  {
    "first_task": "fav",
    "second_task": "buy",
    "sign": "first effect minus second effect",
    "delta_auc": 0.0008260304306512145,
    "ci95_auc": [
      -0.001989308029648101,
      0.003317646022883177
    ],
    "delta_bce": -2.711609876809802e-06,
    "ci95_bce": [
      -6.0001347456551686e-05,
      4.924973859745096e-05
    ]
  }
]
```

</details>

- **证据**：`qformer-loop/taobao-transfer/recurrence-readout-retrieval/reports/paired-analysis/pairs/explore_dual_no_feedback--role_q8_query_projection/result.json`（SHA256前缀 `d6d5c7c0d79d`）。

<a id="p10"></a>
### P10 · `role_q8_read_task` − `role_q8_read_shared`

- **方法**：同q、任务独立读出对共享读出的独立训练控制；按上方固定协议做用户配对。
- **结果**：ΔmacroAUC +0.005155505；95%CI [0.004281966516499934,0.005986110310820656]。

| 任务 | ΔAUC | AUC 95%CI | ΔBCE | BCE 95%CI |
|---|---:|---|---:|---|
| is_click | +0.007147312 | [0.006498537828491224,0.0077868547812544545] | -0.001051971 | [-0.0011139193477760676,-0.0009899834273145382] |
| cart | +0.004433493 | [0.002705166521533353,0.006178719572561234] | +0.000023851 | [-2.4594863292999446e-05,8.00858140477505e-05] |
| fav | +0.005364505 | [0.003433991930323116,0.007022933846086725] | -0.000024071 | [-6.542605379414234e-05,1.9462558625296016e-05] |
| buy | +0.003676709 | [0.0014326412301266967,0.005834068269436105] | +0.000036518 | [-4.723646057172006e-06,8.336474970411064e-05] |

- **归因**：区间完全高于0，支持这个固定checkpoint对照中的macroAUC增量。task胜过shared也须高于原循环，不能把战胜较弱控制当作结构成功。
- **后续假设**：把本对照与原循环质量、四任务取舍和实际延迟联合判断；不依据单个任务区间自动增加结构或seed。

<details><summary>六个任务间效应差（全量保留）</summary>

```json
[
  {
    "first_task": "is_click",
    "second_task": "cart",
    "sign": "first effect minus second effect",
    "delta_auc": 0.002713819047369226,
    "ci95_auc": [
      0.0008705717605803021,
      0.004534596688807726
    ],
    "delta_bce": -0.0010758212143629348,
    "ci95_bce": [
      -0.0011537630678978281,
      -0.0009997684482114207
    ]
  },
  {
    "first_task": "is_click",
    "second_task": "fav",
    "sign": "first effect minus second effect",
    "delta_auc": 0.001782807693119337,
    "ci95_auc": [
      -0.00010540090630715147,
      0.0037537054712834523
    ],
    "delta_bce": -0.00102789930767847,
    "ci95_bce": [
      -0.0011028631731164718,
      -0.0009556519324094073
    ]
  },
  {
    "first_task": "is_click",
    "second_task": "buy",
    "sign": "first effect minus second effect",
    "delta_auc": 0.0034706034125587815,
    "ci95_auc": [
      0.0011091466946680146,
      0.005703292358196165
    ],
    "delta_bce": -0.0010884887112095865,
    "ci95_bce": [
      -0.0011705052012899557,
      -0.0010206052464181692
    ]
  },
  {
    "first_task": "cart",
    "second_task": "fav",
    "sign": "first effect minus second effect",
    "delta_auc": -0.0009310113542498888,
    "ci95_auc": [
      -0.0033258296171743947,
      0.0014904347242673138
    ],
    "delta_bce": 4.7921906684464844e-05,
    "ci95_bce": [
      -1.6223942223097928e-05,
      0.00011776063271382733
    ]
  },
  {
    "first_task": "cart",
    "second_task": "buy",
    "sign": "first effect minus second effect",
    "delta_auc": 0.0007567843651895556,
    "ci95_auc": [
      -0.001881438342355926,
      0.0036752226954344367
    ],
    "delta_bce": -1.266749684665169e-05,
    "ci95_bce": [
      -7.541263876946217e-05,
      5.2550014715668954e-05
    ]
  },
  {
    "first_task": "fav",
    "second_task": "buy",
    "sign": "first effect minus second effect",
    "delta_auc": 0.0016877957194394444,
    "ci95_auc": [
      -0.0014554260397450277,
      0.004541543832218425
    ],
    "delta_bce": -6.0589403531116534e-05,
    "ci95_bce": [
      -0.00012125009408297095,
      1.1108295861749409e-06
    ]
  }
]
```

</details>

- **证据**：`qformer-loop/taobao-transfer/recurrence-readout-retrieval/reports/paired-analysis/pairs/role_q8_read_shared--role_q8_read_task/result.json`（SHA256前缀 `ed3c506cfc88`）。

## D01：固定权重的任务读出扰动

- **方法**：预选q4/q8两个task读出checkpoint；原生系数native、全部系数替换成均值mean、全部归零zero。相同子集CPU FP32输出；固定原验证人口的IPW BCE估计，全部16列共享10000次用户bootstrap。

- **结果**：5514行、4647用户；16个任务×扰动效应均保留。训练更新0；GPU秒0。

| 模型 | 扰动−native | 任务 | ΔIPW BCE | 探索性95%CI |
|---|---|---|---:|---|
| role_q4_read_task | mean_minus_native | is_click | +9.27254778969e-06 | [7.079242937049017e-07,1.77657778956119e-05] |
| role_q4_read_task | mean_minus_native | cart | -2.642426061e-07 | [-7.823587678982792e-07,6.438157811288345e-08] |
| role_q4_read_task | mean_minus_native | fav | -1.0472039413e-07 | [-6.224345006872112e-07,3.6594628620867614e-07] |
| role_q4_read_task | mean_minus_native | buy | +3.33826517147e-07 | [-5.1116982538029e-07,1.2188318881409335e-06] |
| role_q4_read_task | zero_minus_native | is_click | -0.000103232242684 | [-0.0002060328477919399,7.678263762828285e-07] |
| role_q4_read_task | zero_minus_native | cart | +3.97614581298e-05 | [-6.837551738833088e-06,0.00011302299007379865] |
| role_q4_read_task | zero_minus_native | fav | +4.72295545261e-06 | [-8.991736576321023e-06,1.9885349760817633e-05] |
| role_q4_read_task | zero_minus_native | buy | +4.79701364029e-06 | [-3.3910912374851757e-06,1.3418727330439994e-05] |
| role_q8_read_task | mean_minus_native | is_click | -3.289511198e-06 | [-9.627870568186343e-06,3.118512368463815e-06] |
| role_q8_read_task | mean_minus_native | cart | -6.44722424238e-06 | [-2.63509509878373e-05,4.542014457418477e-06] |
| role_q8_read_task | mean_minus_native | fav | -7.51624536647e-07 | [-2.366304951511593e-06,1.6734494464580737e-06] |
| role_q8_read_task | mean_minus_native | buy | -4.99502575939e-08 | [-4.361366025807976e-07,3.0517262223880697e-07] |
| role_q8_read_task | zero_minus_native | is_click | -3.82354183716e-05 | [-0.0001215971666196725,4.547905821225699e-05] |
| role_q8_read_task | zero_minus_native | cart | +4.68007169544e-05 | [-2.9398030607420494e-05,0.00018528581492702927] |
| role_q8_read_task | zero_minus_native | fav | -9.22890164483e-06 | [-2.9265434446091327e-05,2.1450234972230498e-05] |
| role_q8_read_task | zero_minus_native | buy | +2.82300405302e-06 | [-5.915895837819814e-06,1.2351125960681868e-05] |

- **归因**：16个探索性95%区间中，q4点击mean−native的区间为正，其余15个跨0。该结果提示这个checkpoint使用了任务系数差异；未经多重校正，且q4 task完整训练低于原循环，不能宣称任务读出改善了训练、只有点击需要循环或第二次检索更好。
- **后续假设**：任务读出是否形成可用增量，必须与独立训练shared及原循环比较；native→zero只检验此checkpoint对早期读出路径的依赖，不等同于原循环或删第二轮。
- **完整范围说明**：["Native versus mean tests whether the already trained checkpoint uses learned task-specific coefficients; it is not the independently trained shared-readout control.","Native versus zero tests reliance on its early-state readout pathway; it is not the original recurrence checkpoint or an independently trained last-round-only model.","A harmful same-weight perturbation does not establish that the readout structure improves training, nor that the second retrieval became better. Use parent mandatory trained comparisons for those performance claims.","Do not rank intrinsic task depth needs by coefficient order, compare raw BCE effect magnitude as information content, or claim equivalence from a crossing-zero interval.","CPU FP32 native outputs are not assumed bitwise equal to saved GPU BF16 full-validation predictions. The three diagnostic conditions share one CPU kernel and subset; do not substitute their BCE/AUC for full-validation training results.","All four tasks, both contrasts and both preselected models are retained regardless of sign; no test data, new training seed, new sample selection or structural edit follows automatically."]
- **证据**：`qformer-loop/taobao-transfer/recurrence-readout-retrieval/reports/readout-diagnostics/aggregate/aggregate.json`（SHA256前缀 `05a120336060`）；`qformer-loop/taobao-transfer/recurrence-readout-retrieval/reports/readout-diagnostics/aggregate/completion.json`（SHA256前缀 `4cf4605a9c97`）；`qformer-loop/taobao-transfer/recurrence-readout-retrieval/reports/readout-diagnostics/aggregate/toolkit/perturbation_report.json`（SHA256前缀 `9f600409e2db`）。

## E01：11模型同卡延迟测量

- **方法**：8个新模型与q4、q8、最佳单轮3个对照，在同GPU上测量真实验证batch前向，按预定报告保留全部结果。
- **结果**：`failed`，原因 `AttributeError("'list' object has no attribute 'detach'")`；保留部分产物，没有完整延迟结论。
- **归因**：这是测量程序接口错误；不否定已完成训练的AUC，也不能用不完整测量声称提速。
- **后续假设**：主线程若修复测量并得到完整同卡结果，再判断质量/成本取舍；本文整理没有触发重试。
- **证据**：`qformer-loop/taobao-transfer/recurrence-readout-retrieval/reports/latency/failure.json`（SHA256前缀 `4dfda3458e81`）；`qformer-loop/taobao-transfer/recurrence-readout-retrieval/reports/latency/plan.json`（SHA256前缀 `76801700e7a2`）。

**本轮设计证据**：`qformer-loop/taobao-transfer/recurrence-readout-retrieval/protocol.json`（SHA256前缀 `a50f0b55fc33`）。
