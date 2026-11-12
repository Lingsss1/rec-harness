# 淘宝：全部迭代实验明细

快照时间：2026-09-09T07:36:23.974768+00:00。返回[总览](README.md)。

每条保留原始实验ID；历史decision只作记录，不等于当前推荐。原始历史假设按当时语境保留；阶段后续方向是研究链条，后续已完成的实验不重新列为待执行任务。

历史episode 170条；本次补充发现的未归档运行 8条。完整训练结果文件 117份。诊断、统计、启动失败不增加完整训练计数。

结果来源是保存的JSON/TSV/Markdown，本次未训练、未加载模型或预测数组、未重新评估数据。证据路径为工作区逻辑相对路径，并非Git附件；大数据、权重和日志不随文档上传。行内为SHA256前12位，完整值见[证据索引](EVIDENCE_INDEX.md)。

最新轮尚未归档的配对、读出诊断和延迟测量状态另见[最新轮补充记录](LATEST_READOUT_RETRIEVAL.md)，不把这些分析重复计作训练。

<a id="phase-initial"></a>
## 阶段 initial：淘宝迁移与裁剪基线

**方法**：Cross8、Cross8 clip1、Cross11及KuaiRec r7结构从头训练四任务。

**归因与边界**：r7直接迁移低于淘宝Cross8；clip10→1对Cross8的macroAUC贡献+0.001979455，不能归于循环。

**历史后续假设**：增宽与共享循环分开做因子对照；后见round2。

<a id="t001"></a>
### T001 · `taobao_cross8-s20262027-20260907T084102636174Z`

- **方法**：taobao_cross8；设置 {"model":"QFormerCross8","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":10}
- **原假设**：Transfer the KuaiRec best architecture and compare four-task Taobao controls
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"epochs":1,"optimizer_steps":2193}。
- **结果**：macroAUC=0.821069672；macroLogloss=0.056061816；is_click AUC/Logloss=0.660002863/0.182594919；cart AUC/Logloss=0.860224981/0.019115957；fav AUC/Logloss=0.875395523/0.014385520；buy AUC/Logloss=0.888655320/0.008150866。
- **开销**：参数 54763844；训练含验证 1499.6513188630342 秒；峰值CUDA 16366309376 bytes；更新 2193。
- **原记录差值**：+0.000000000；对照 `null`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":11.878864288330078,"mean_ms":11.892812037467957,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计持平0.000000000，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 initial](#phase-initial)。
- **后续假设**：增宽与共享循环分开做因子对照；后见round2。
- **证据**：`qformer-loop/taobao-transfer/runs/taobao_cross8-s20262027-20260907T084102636174Z/manifest.json`（SHA256前缀 `b19cd4d9adc6`）；`qformer-loop/taobao-transfer/runs/taobao_cross8-s20262027-20260907T084102636174Z/result.json`（SHA256前缀 `d1f254d7a14c`）；`qformer-loop/taobao-transfer/reports/final-audit.json`（SHA256前缀 `9aca2585eafb`）

<a id="t002"></a>
### T002 · `taobao_cross8_clip1-s20262027-20260907T084102654002Z`

- **方法**：taobao_cross8_clip1；设置 {"model":"QFormerCross8","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Transfer the KuaiRec best architecture and compare four-task Taobao controls
- **状态／预算**：原记录 `keep`；{"seed":20262027,"epochs":1,"optimizer_steps":2193}。
- **结果**：macroAUC=0.823049127；macroLogloss=0.055583807；is_click AUC/Logloss=0.661437520/0.182632232；cart AUC/Logloss=0.861455904/0.018404539；fav AUC/Logloss=0.878362020/0.013938001；buy AUC/Logloss=0.890941062/0.007360454。
- **开销**：参数 54763844；训练含验证 1499.6627897769213 秒；峰值CUDA 16367532032 bytes；更新 2193。
- **原记录差值**：+0.001979455；对照 `taobao_cross8-s20262027-20260907T084102636174Z`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":11.877568244934082,"mean_ms":11.9293128490448,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.001979455，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 initial](#phase-initial)。
- **后续假设**：增宽与共享循环分开做因子对照；后见round2。
- **证据**：`qformer-loop/taobao-transfer/runs/taobao_cross8_clip1-s20262027-20260907T084102654002Z/manifest.json`（SHA256前缀 `731f818f1ba2`）；`qformer-loop/taobao-transfer/runs/taobao_cross8_clip1-s20262027-20260907T084102654002Z/result.json`（SHA256前缀 `c82e9103bf1a`）；`qformer-loop/taobao-transfer/reports/final-audit.json`（SHA256前缀 `9aca2585eafb`）

<a id="t003"></a>
### T003 · `taobao_cross11-s20262027-20260907T084102626054Z`

- **方法**：taobao_cross11；设置 {"model":"QFormerCross11","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Transfer the KuaiRec best architecture and compare four-task Taobao controls
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"epochs":1,"optimizer_steps":2193}。
- **结果**：macroAUC=0.818314208；macroLogloss=0.056220550；is_click AUC/Logloss=0.657694300/0.183179867；cart AUC/Logloss=0.855577763/0.019251487；fav AUC/Logloss=0.875769217/0.014457664；buy AUC/Logloss=0.884215552/0.007993180。
- **开销**：参数 56940356；训练含验证 1593.2617662623525 秒；峰值CUDA 19230560256 bytes；更新 2193。
- **原记录差值**：-0.002755464；对照 `taobao_cross8-s20262027-20260907T084102636174Z`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":13.592015743255615,"mean_ms":13.606524729728699,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.002755464，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 initial](#phase-initial)。
- **后续假设**：增宽与共享循环分开做因子对照；后见round2。
- **证据**：`qformer-loop/taobao-transfer/runs/taobao_cross11-s20262027-20260907T084102626054Z/manifest.json`（SHA256前缀 `7b8168a8baf1`）；`qformer-loop/taobao-transfer/runs/taobao_cross11-s20262027-20260907T084102626054Z/result.json`（SHA256前缀 `878c48ddf55f`）；`qformer-loop/taobao-transfer/reports/final-audit.json`（SHA256前缀 `9aca2585eafb`）

<a id="t004"></a>
### T004 · `taobao_r7_loop2-s20262027-20260907T084102641511Z`

- **方法**：taobao_r7_loop2；设置 {"model":"QFormerCross11Memory","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Transfer the KuaiRec best architecture and compare four-task Taobao controls
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"epochs":1,"optimizer_steps":2193}。
- **结果**：macroAUC=0.817924507；macroLogloss=0.056134306；is_click AUC/Logloss=0.659762782/0.182788785；cart AUC/Logloss=0.855817150/0.019142379；fav AUC/Logloss=0.871620279/0.014507713；buy AUC/Logloss=0.884497818/0.008098346。
- **开销**：参数 53647684；训练含验证 1490.269450545311 秒；峰值CUDA 15876542976 bytes；更新 2193。
- **原记录差值**：-0.003145164；对照 `taobao_cross8-s20262027-20260907T084102636174Z`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":11.016672134399414,"mean_ms":11.026762461662292,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.003145164，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 initial](#phase-initial)。
- **后续假设**：增宽与共享循环分开做因子对照；后见round2。
- **证据**：`qformer-loop/taobao-transfer/runs/taobao_r7_loop2-s20262027-20260907T084102641511Z/manifest.json`（SHA256前缀 `d044a10d0372`）；`qformer-loop/taobao-transfer/runs/taobao_r7_loop2-s20262027-20260907T084102641511Z/result.json`（SHA256前缀 `7b604a8346dc`）；`qformer-loop/taobao-transfer/reports/final-audit.json`（SHA256前缀 `9aca2585eafb`）

<a id="t005"></a>
### T005 · `taobao-transfer-paired-bootstrap`

- **方法**：500 paired user resamples with shared user weights across four tasks
- **原假设**：Quantify r7 transfer deltas conditional on frozen checkpoints
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"bootstrap_seed":20260906,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
[
  {
    "reference": "taobao_cross8",
    "candidate": "taobao_r7_loop2",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "delta_auc": {
      "is_click": -0.0002400806568211289,
      "cart": -0.00440783142495238,
      "fav": -0.003775244129356925,
      "buy": -0.004157501397183627
    },
    "ci95_auc": {
      "is_click": [
        -0.0009786180430587626,
        0.0004570052888857504
      ],
      "cart": [
        -0.006414036584354584,
        -0.0024636172252152757
      ],
      "fav": [
        -0.005945695244795404,
        -0.0018237185121322533
      ],
      "buy": [
        -0.0072995715432280115,
        -0.0011806396413112758
      ]
    },
    "delta_macro_auc": -0.0031451644020784597,
    "ci95_macro_auc": [
      -0.004290393382710493,
      -0.002026580021838932
    ],
    "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance; correlated heads share bootstrap user weights"
  },
  {
    "reference": "taobao_cross8_clip1",
    "candidate": "taobao_r7_loop2",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "delta_auc": {
      "is_click": -0.0016747379628531123,
      "cart": -0.0056387547595274645,
      "fav": -0.006741741605768903,
      "buy": -0.006443243274453447
    },
    "ci95_auc": {
      "is_click": [
        -0.0022897146023902583,
        -0.0010732111514503652
      ],
      "cart": [
        -0.007434445466007361,
        -0.0038172461079368342
      ],
      "fav": [
        -0.008648324565023468,
        -0.004906211081217849
      ],
      "buy": [
        -0.009704139426471128,
        -0.0034952704190272356
      ]
    },
    "delta_macro_auc": -0.005124619400650676,
    "ci95_macro_auc": [
      -0.006203125802890035,
      -0.004111887938053171
    ],
    "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance; correlated heads share bootstrap user weights"
  },
  {
    "reference": "taobao_cross11",
    "candidate": "taobao_r7_loop2",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "delta_auc": {
      "is_click": 0.002068482821352191,
      "cart": 0.00023938629212294504,
      "fav": -0.00414893776076819,
      "buy": 0.0002822663887721566
    },
    "ci95_auc": {
      "is_click": [
        0.0013994566003808523,
        0.0026179615713444737
      ],
      "cart": [
        -0.0016416957373312862,
        0.0021065846698046925
      ],
      "fav": [
        -0.0060140750278366295,
        -0.002220977869553354
      ],
      "buy": [
        -0.002360787179727211,
        0.002910642485796455
      ]
    },
    "delta_macro_auc": -0.0003897005646301688,
    "ci95_macro_auc": [
      -0.0013557787272563673,
      0.0006798263023157943
    ],
    "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance; correlated heads share bootstrap user weights"
  }
]
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 initial](#phase-initial)。
- **后续假设**：增宽与共享循环分开做因子对照；后见round2。
- **证据**：`qformer-loop/taobao-transfer/reports/paired-bootstrap.json`（SHA256前缀 `b1cc24e3c519`）

<a id="t125"></a>
### T125 · `taobao-q4-difference-near-zero-psgd-reproduction`

- **方法**：Same seed and original two-step smoke with read-only gradient/scale instrumentation
- **原假设**：Distinguish actual zero gradient from an approximate-zero rejection
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"smoke_steps_completed":1,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "grad_elements_nonzero": 6144,
  "clipped_a_gradient_max_abs": 8.77190586834331e-09,
  "finite_default_automatic_scale": 140.19190910151133,
  "fp32_fourth_moment_underflow": false
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 initial](#phase-initial)。
- **后续假设**：增宽与共享循环分开做因子对照；后见round2。
- **证据**：`qformer-loop/taobao-transfer/query-correction-diagnostics/psgd-failure-q4-difference/attempt-20260908T181924351492Z/result.json`（SHA256前缀 `2e93938e24ed`）；`qformer-loop/taobao-transfer/query-correction-diagnostics/psgd-failure-q4-difference/attempt-20260908T181924351492Z/observations.jsonl`（SHA256前缀 `679d878b532d`）

<a id="phase-round2"></a>
## 阶段 round2：增宽与整段循环

**方法**：token宽度256/384/512；单轮/整段循环；增宽r7。

**归因与边界**：384单轮优先，整段循环在三个宽度下均降低macroAUC；增益主要来自点击，且有延迟代价。

**历史后续假设**：把循环移到局部并减弱重复更新；见round3。

<a id="t006"></a>
### T006 · `t2_cross8_w256_loop2-s20262027-20260907T092620181179Z`

- **方法**：t2_cross8_w256_loop2；设置 {"model":"QFormerCross8CapacityLoop","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Increase width and repeat shared history stage
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.819862307；macroLogloss=0.055933498；is_click AUC/Logloss=0.661707121/0.182717034；cart AUC/Logloss=0.857515407/0.018997233；fav AUC/Logloss=0.875344614/0.014221539；buy AUC/Logloss=0.884882084/0.007798186。
- **开销**：参数 54763844；训练含验证 1804.6303344070911 秒；峰值CUDA 25841173504 bytes；更新 2193。
- **原记录差值**：-0.003186820；对照 `taobao_cross8_clip1-s20262027-20260907T084102654002Z`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":18.377951622009277,"mean_ms":18.387437486648558,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.003186820，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 round2](#phase-round2)。
- **后续假设**：把循环移到局部并减弱重复更新；见round3。
- **证据**：`qformer-loop/taobao-transfer/round2/runs/t2_cross8_w256_loop2-s20262027-20260907T092620181179Z/manifest.json`（SHA256前缀 `85075c53bc9f`）；`qformer-loop/taobao-transfer/round2/runs/t2_cross8_w256_loop2-s20262027-20260907T092620181179Z/result.json`（SHA256前缀 `877789bb1c3a`）；`qformer-loop/taobao-transfer/round2/reports/final-audit.json`（SHA256前缀 `e66e196e01d4`）

<a id="t007"></a>
### T007 · `t2_cross8_w384_loop1-s20262027-20260907T092620130333Z`

- **方法**：t2_cross8_w384_loop1；设置 {"model":"QFormerCross8CapacityLoop","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Increase width and repeat shared history stage
- **状态／预算**：原记录 `keep`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.824054887；macroLogloss=0.055368034；is_click AUC/Logloss=0.664513532/0.182371092；cart AUC/Logloss=0.861906218/0.018303893；fav AUC/Logloss=0.878320666/0.013553188；buy AUC/Logloss=0.891479133/0.007243964。
- **开销**：参数 81616324；训练含验证 1713.6810667142272 秒；峰值CUDA 24103977984 bytes；更新 2193。
- **原记录差值**：+0.001005761；对照 `taobao_cross8_clip1-s20262027-20260907T084102654002Z`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":14.675648212432861,"mean_ms":14.765788817405701,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.001005761，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 round2](#phase-round2)。
- **后续假设**：把循环移到局部并减弱重复更新；见round3。
- **证据**：`qformer-loop/taobao-transfer/round2/runs/t2_cross8_w384_loop1-s20262027-20260907T092620130333Z/manifest.json`（SHA256前缀 `c0097b8b0aac`）；`qformer-loop/taobao-transfer/round2/runs/t2_cross8_w384_loop1-s20262027-20260907T092620130333Z/result.json`（SHA256前缀 `06a3f33702d4`）；`qformer-loop/taobao-transfer/round2/reports/final-audit.json`（SHA256前缀 `e66e196e01d4`）

<a id="t008"></a>
### T008 · `t2_cross8_w384_loop2-s20262027-20260907T092619887297Z`

- **方法**：t2_cross8_w384_loop2；设置 {"model":"QFormerCross8CapacityLoop","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Increase width and repeat shared history stage
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.820489779；macroLogloss=0.055679647；is_click AUC/Logloss=0.660869315/0.182905547；cart AUC/Logloss=0.857413030/0.018457255；fav AUC/Logloss=0.877185743/0.014031650；buy AUC/Logloss=0.886491028/0.007324136。
- **开销**：参数 81616324；训练含验证 2149.4165845811367 秒；峰值CUDA 38263728128 bytes；更新 2193。
- **原记录差值**：-0.002559348；对照 `taobao_cross8_clip1-s20262027-20260907T084102654002Z`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":22.838303565979004,"mean_ms":22.84225516319275,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.002559348，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 round2](#phase-round2)。
- **后续假设**：把循环移到局部并减弱重复更新；见round3。
- **证据**：`qformer-loop/taobao-transfer/round2/runs/t2_cross8_w384_loop2-s20262027-20260907T092619887297Z/manifest.json`（SHA256前缀 `a6c05f1f6ca9`）；`qformer-loop/taobao-transfer/round2/runs/t2_cross8_w384_loop2-s20262027-20260907T092619887297Z/result.json`（SHA256前缀 `1736957a092e`）；`qformer-loop/taobao-transfer/round2/reports/final-audit.json`（SHA256前缀 `e66e196e01d4`）

<a id="t009"></a>
### T009 · `t2_cross8_w512_loop1-s20262027-20260907T092620048171Z`

- **方法**：t2_cross8_w512_loop1；设置 {"model":"QFormerCross8CapacityLoop","token_dim":512,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Increase width and repeat shared history stage
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.822974901；macroLogloss=0.055455145；is_click AUC/Logloss=0.661965157/0.182751883；cart AUC/Logloss=0.862523620/0.018170298；fav AUC/Logloss=0.876944008/0.013731605；buy AUC/Logloss=0.890466820/0.007166792。
- **开销**：参数 118757956；训练含验证 1877.9927652999759 秒；峰值CUDA 31972221440 bytes；更新 2193。
- **原记录差值**：-0.000074225；对照 `taobao_cross8_clip1-s20262027-20260907T084102654002Z`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":17.851231575012207,"mean_ms":17.86896071434021,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000074225，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 round2](#phase-round2)。
- **后续假设**：把循环移到局部并减弱重复更新；见round3。
- **证据**：`qformer-loop/taobao-transfer/round2/runs/t2_cross8_w512_loop1-s20262027-20260907T092620048171Z/manifest.json`（SHA256前缀 `8219a0ed07bc`）；`qformer-loop/taobao-transfer/round2/runs/t2_cross8_w512_loop1-s20262027-20260907T092620048171Z/result.json`（SHA256前缀 `7894134cff4d`）；`qformer-loop/taobao-transfer/round2/reports/final-audit.json`（SHA256前缀 `e66e196e01d4`）

<a id="t010"></a>
### T010 · `t2_cross8_w512_loop2-s20262027-20260907T092620129939Z`

- **方法**：t2_cross8_w512_loop2；设置 {"model":"QFormerCross8CapacityLoop","token_dim":512,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Increase width and repeat shared history stage
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.822121038；macroLogloss=0.055478832；is_click AUC/Logloss=0.661558268/0.182626464；cart AUC/Logloss=0.861190203/0.018263253；fav AUC/Logloss=0.877789708/0.013733421；buy AUC/Logloss=0.887945975/0.007292191。
- **开销**：参数 118757956；训练含验证 2407.142288401723 秒；峰值CUDA 50740002304 bytes；更新 2193。
- **原记录差值**：-0.000928088；对照 `taobao_cross8_clip1-s20262027-20260907T084102654002Z`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":27.94126319885254,"mean_ms":27.9591423034668,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000928088，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 round2](#phase-round2)。
- **后续假设**：把循环移到局部并减弱重复更新；见round3。
- **证据**：`qformer-loop/taobao-transfer/round2/runs/t2_cross8_w512_loop2-s20262027-20260907T092620129939Z/manifest.json`（SHA256前缀 `8f5acd860bc4`）；`qformer-loop/taobao-transfer/round2/runs/t2_cross8_w512_loop2-s20262027-20260907T092620129939Z/result.json`（SHA256前缀 `f0687a967479`）；`qformer-loop/taobao-transfer/round2/reports/final-audit.json`（SHA256前缀 `e66e196e01d4`）

<a id="t011"></a>
### T011 · `t2_r7_w384_loop2-s20262027-20260907T092620135016Z`

- **方法**：t2_r7_w384_loop2；设置 {"model":"QFormerCross11Memory","token_dim":384,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Increase width and repeat shared history stage
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.821451680；macroLogloss=0.055617192；is_click AUC/Logloss=0.660834784/0.182614974；cart AUC/Logloss=0.860849999/0.018446951；fav AUC/Logloss=0.877735075/0.013804738；buy AUC/Logloss=0.886386861/0.007602105。
- **开销**：参数 79106500；训练含验证 1660.260270677507 秒；峰值CUDA 23369321984 bytes；更新 2193。
- **原记录差值**：-0.001597447；对照 `taobao_cross8_clip1-s20262027-20260907T084102654002Z`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":13.37230396270752,"mean_ms":13.384310388565064,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001597447，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 round2](#phase-round2)。
- **后续假设**：把循环移到局部并减弱重复更新；见round3。
- **证据**：`qformer-loop/taobao-transfer/round2/runs/t2_r7_w384_loop2-s20262027-20260907T092620135016Z/manifest.json`（SHA256前缀 `cc66f3fed09e`）；`qformer-loop/taobao-transfer/round2/runs/t2_r7_w384_loop2-s20262027-20260907T092620135016Z/result.json`（SHA256前缀 `41a9fdc26f2f`）；`qformer-loop/taobao-transfer/round2/reports/final-audit.json`（SHA256前缀 `e66e196e01d4`）

<a id="t012"></a>
### T012 · `taobao-round2-paired-bootstrap`

- **方法**：500 paired user resamples
- **原假设**：Quantify best widened checkpoint gain
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"bootstrap_seed":20260906,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
[
  {
    "reference": "taobao_cross8_clip1",
    "candidate": "t2_cross8_w384_loop1",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "delta_auc": {
      "is_click": 0.0030760114073425937,
      "cart": 0.0004503138479481361,
      "fav": -4.135420852569016e-05,
      "buy": 0.0005380717264861223
    },
    "ci95_auc": {
      "is_click": [
        0.002496308301459793,
        0.003650137893312366
      ],
      "cart": [
        -0.001142837605769717,
        0.0019925460543049798
      ],
      "fav": [
        -0.0017138880355355712,
        0.0016832693467095032
      ],
      "buy": [
        -0.00209024873864595,
        0.0029586721706443436
      ]
    },
    "delta_macro_auc": 0.0010057606933127072,
    "ci95_macro_auc": [
      6.5303811687803076e-06,
      0.0017900831365844948
    ],
    "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance; correlated heads share bootstrap user weights"
  }
]
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 round2](#phase-round2)。
- **后续假设**：把循环移到局部并减弱重复更新；见round3。
- **证据**：`qformer-loop/taobao-transfer/round2/reports/paired-bootstrap.json`（SHA256前缀 `fc2a48325e6d`）

<a id="phase-round3"></a>
## 阶段 round3：局部阻尼循环

**方法**：整段、尾部和post等位置及半幅更新。

**归因与边界**：tail-half在AdamW下提高四任务AUC，macro差+0.001386；它比全栈重复更有价值，但增加计算。

**历史后续假设**：控制尾部宽度、幅度和轮数；见round4。

<a id="t013"></a>
### T013 · `t3_cross_full-s20262027-20260907T110200877980Z`

- **方法**：t3_cross_full；设置 {"model":"QFormerCross8SelectiveLoop","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Selective shared recurrence at increased width384
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.821762855；macroLogloss=0.055445083；is_click AUC/Logloss=0.662612260/0.182563347；cart AUC/Logloss=0.857562364/0.018251710；fav AUC/Logloss=0.876983718/0.013716990；buy AUC/Logloss=0.889893079/0.007248284。
- **开销**：参数 81616324；训练含验证 2107.3548786640167 秒；峰值CUDA 36994745344 bytes；更新 2193。
- **原记录差值**：-0.002292032；对照 `t2_cross8_w384_loop1-s20262027-20260907T092620130333Z`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":22.13219165802002,"mean_ms":22.145223951339723,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.002292032，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 round3](#phase-round3)。
- **后续假设**：控制尾部宽度、幅度和轮数；见round4。
- **证据**：`qformer-loop/taobao-transfer/round3/runs/t3_cross_full-s20262027-20260907T110200877980Z/manifest.json`（SHA256前缀 `633688dc9033`）；`qformer-loop/taobao-transfer/round3/runs/t3_cross_full-s20262027-20260907T110200877980Z/result.json`（SHA256前缀 `4e18672aaf0e`）；`qformer-loop/taobao-transfer/round3/reports/final-audit.json`（SHA256前缀 `8138dc8ba582`）

<a id="t014"></a>
### T014 · `t3_cross_half-s20262027-20260907T110201098892Z`

- **方法**：t3_cross_half；设置 {"model":"QFormerCross8SelectiveLoop","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Selective shared recurrence at increased width384
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.822582944；macroLogloss=0.055533024；is_click AUC/Logloss=0.660427674/0.182721193；cart AUC/Logloss=0.861678333/0.018346262；fav AUC/Logloss=0.877727089/0.013836072；buy AUC/Logloss=0.890498681/0.007228571。
- **开销**：参数 81616324；训练含验证 2116.13457801193 秒；峰值CUDA 36994745344 bytes；更新 2193。
- **原记录差值**：-0.001471943；对照 `t2_cross8_w384_loop1-s20262027-20260907T092620130333Z`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":22.15507221221924,"mean_ms":22.219432020187377,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001471943，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 round3](#phase-round3)。
- **后续假设**：控制尾部宽度、幅度和轮数；见round4。
- **证据**：`qformer-loop/taobao-transfer/round3/runs/t3_cross_half-s20262027-20260907T110201098892Z/manifest.json`（SHA256前缀 `ec242f52f081`）；`qformer-loop/taobao-transfer/round3/runs/t3_cross_half-s20262027-20260907T110201098892Z/result.json`（SHA256前缀 `1a433b0e2a3c`）；`qformer-loop/taobao-transfer/round3/reports/final-audit.json`（SHA256前缀 `8138dc8ba582`）

<a id="t015"></a>
### T015 · `t3_tail_full-s20262027-20260907T110201091912Z`

- **方法**：t3_tail_full；设置 {"model":"QFormerCross8SelectiveLoop","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Selective shared recurrence at increased width384
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.824722721；macroLogloss=0.055429355；is_click AUC/Logloss=0.663486144/0.182504786；cart AUC/Logloss=0.863882798/0.018134724；fav AUC/Logloss=0.877194191/0.013840348；buy AUC/Logloss=0.894327751/0.007237563。
- **开销**：参数 81616324；训练含验证 1853.3924357742071 秒；峰值CUDA 28367455232 bytes；更新 2193。
- **原记录差值**：+0.000667834；对照 `t2_cross8_w384_loop1-s20262027-20260907T092620130333Z`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":17.162495613098145,"mean_ms":17.247754430770875,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.000667834，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 round3](#phase-round3)。
- **后续假设**：控制尾部宽度、幅度和轮数；见round4。
- **证据**：`qformer-loop/taobao-transfer/round3/runs/t3_tail_full-s20262027-20260907T110201091912Z/manifest.json`（SHA256前缀 `57340b08e79b`）；`qformer-loop/taobao-transfer/round3/runs/t3_tail_full-s20262027-20260907T110201091912Z/result.json`（SHA256前缀 `fe2273744179`）；`qformer-loop/taobao-transfer/round3/reports/final-audit.json`（SHA256前缀 `8138dc8ba582`）

<a id="t016"></a>
### T016 · `t3_tail_half-s20262027-20260907T110200904859Z`

- **方法**：t3_tail_half；设置 {"model":"QFormerCross8SelectiveLoop","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Selective shared recurrence at increased width384
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.825441344；macroLogloss=0.055205329；is_click AUC/Logloss=0.665074960/0.182345863；cart AUC/Logloss=0.863184976/0.017943286；fav AUC/Logloss=0.878676326/0.013609606；buy AUC/Logloss=0.894829112/0.006922560。
- **开销**：参数 81616324；训练含验证 1850.6268046870828 秒；峰值CUDA 28366248960 bytes；更新 2193。
- **原记录差值**：+0.001386456；对照 `t2_cross8_w384_loop1-s20262027-20260907T092620130333Z`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":17.189215660095215,"mean_ms":17.21795525550842,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.001386456，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 round3](#phase-round3)。
- **后续假设**：控制尾部宽度、幅度和轮数；见round4。
- **证据**：`qformer-loop/taobao-transfer/round3/runs/t3_tail_half-s20262027-20260907T110200904859Z/manifest.json`（SHA256前缀 `6221ae55f4a2`）；`qformer-loop/taobao-transfer/round3/runs/t3_tail_half-s20262027-20260907T110200904859Z/result.json`（SHA256前缀 `e859e5fad2a5`）；`qformer-loop/taobao-transfer/round3/reports/final-audit.json`（SHA256前缀 `8138dc8ba582`）

<a id="t017"></a>
### T017 · `t3_post_full-s20262027-20260907T110201074860Z`

- **方法**：t3_post_full；设置 {"model":"QFormerCross8SelectiveLoop","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Selective shared recurrence at increased width384
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.823152578；macroLogloss=0.055513713；is_click AUC/Logloss=0.662372629/0.182884312；cart AUC/Logloss=0.862893222/0.018292599；fav AUC/Logloss=0.876561297/0.013737622；buy AUC/Logloss=0.890783163/0.007140320。
- **开销**：参数 81616324；训练含验证 1754.6101477518678 秒；峰值CUDA 24966265856 bytes；更新 2193。
- **原记录差值**：-0.000902309；对照 `t2_cross8_w384_loop1-s20262027-20260907T092620130333Z`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":15.384176254272461,"mean_ms":15.394256782531738,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000902309，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 round3](#phase-round3)。
- **后续假设**：控制尾部宽度、幅度和轮数；见round4。
- **证据**：`qformer-loop/taobao-transfer/round3/runs/t3_post_full-s20262027-20260907T110201074860Z/manifest.json`（SHA256前缀 `159657df81d5`）；`qformer-loop/taobao-transfer/round3/runs/t3_post_full-s20262027-20260907T110201074860Z/result.json`（SHA256前缀 `0722566c1921`）；`qformer-loop/taobao-transfer/round3/reports/final-audit.json`（SHA256前缀 `8138dc8ba582`）

<a id="t018"></a>
### T018 · `t3_post_half-s20262027-20260907T110201154133Z`

- **方法**：t3_post_half；设置 {"model":"QFormerCross8SelectiveLoop","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Selective shared recurrence at increased width384
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.824969763；macroLogloss=0.055336415；is_click AUC/Logloss=0.663222689/0.182465630；cart AUC/Logloss=0.863957549/0.018053475；fav AUC/Logloss=0.879058256/0.013725452；buy AUC/Logloss=0.893640558/0.007101101。
- **开销**：参数 81616324；训练含验证 1758.7010637447238 秒；峰值CUDA 24966265856 bytes；更新 2193。
- **原记录差值**：+0.000914876；对照 `t2_cross8_w384_loop1-s20262027-20260907T092620130333Z`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":15.414672374725342,"mean_ms":15.422980809211731,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.000914876，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 round3](#phase-round3)。
- **后续假设**：控制尾部宽度、幅度和轮数；见round4。
- **证据**：`qformer-loop/taobao-transfer/round3/runs/t3_post_half-s20262027-20260907T110201154133Z/manifest.json`（SHA256前缀 `5b78bfa2aa1f`）；`qformer-loop/taobao-transfer/round3/runs/t3_post_half-s20262027-20260907T110201154133Z/result.json`（SHA256前缀 `c809f4ce5db5`）；`qformer-loop/taobao-transfer/round3/reports/final-audit.json`（SHA256前缀 `8138dc8ba582`）

<a id="t019"></a>
### T019 · `taobao-round3-paired-bootstrap`

- **方法**：500 paired user resamples
- **原假设**：Quantify best local recurrence gain
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"bootstrap_seed":20260906,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
[
  {
    "reference": "t2_cross8_w384_loop1",
    "candidate": "t3_tail_half",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "delta_auc": {
      "is_click": 0.0005614286211309549,
      "cart": 0.0012787582242693718,
      "fav": 0.00035565970801165747,
      "buy": 0.003349978464221137
    },
    "ci95_auc": {
      "is_click": [
        3.0259479737140827e-05,
        0.001097100452470381
      ],
      "cart": [
        -0.000198371195024874,
        0.0027838480238445646
      ],
      "fav": [
        -0.0012137929465146247,
        0.0019221948659184551
      ],
      "buy": [
        0.0009160381717329497,
        0.0057023711557910774
      ]
    },
    "delta_macro_auc": 0.0013864562544083636,
    "ci95_macro_auc": [
      0.000520001541426901,
      0.0022455759554442325
    ],
    "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance or correct for candidate selection; correlated heads share bootstrap user weights"
  }
]
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 round3](#phase-round3)。
- **后续假设**：控制尾部宽度、幅度和轮数；见round4。
- **证据**：`qformer-loop/taobao-transfer/round3/reports/paired-bootstrap.json`（SHA256前缀 `f71fd013b57f`）

<a id="phase-round4"></a>
## 阶段 round4：尾部循环细化

**方法**：6个宽度/更新幅度/深度组合。

**归因与边界**：全部低于t3_tail_half；继续堆叠和增宽没有改善。

**历史后续假设**：固定LR/clip/结构，把AdamW换为PSGDKron。

<a id="t020"></a>
### T020 · `t4_tail_w384_a025_r1-s20262027-20260907T114731593518Z`

- **方法**：t4_tail_w384_a025_r1；设置 {"model":"QFormerCross8TailRefine","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Tail recurrence strength, depth and width
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.822215390；macroLogloss=0.055556747；is_click AUC/Logloss=0.662269068/0.183081353；cart AUC/Logloss=0.857948094/0.018365644；fav AUC/Logloss=0.877228330/0.013661494；buy AUC/Logloss=0.891416068/0.007118497。
- **开销**：参数 81616324；训练含验证 1846.0609805360436 秒；峰值CUDA 28367455232 bytes；更新 2193。
- **原记录差值**：-0.003225954；对照 `t3_tail_half-s20262027-20260907T110200904859Z`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":17.11604881286621,"mean_ms":17.122157764434814,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.003225954，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 round4](#phase-round4)。
- **后续假设**：固定LR/clip/结构，把AdamW换为PSGDKron。
- **证据**：`qformer-loop/taobao-transfer/round4/runs/t4_tail_w384_a025_r1-s20262027-20260907T114731593518Z/manifest.json`（SHA256前缀 `315e461fc2d3`）；`qformer-loop/taobao-transfer/round4/runs/t4_tail_w384_a025_r1-s20262027-20260907T114731593518Z/result.json`（SHA256前缀 `2da9e9593e67`）；`qformer-loop/taobao-transfer/round4/reports/final-audit.json`（SHA256前缀 `2c217471fff6`）

<a id="t021"></a>
### T021 · `t4_tail_w384_a075_r1-s20262027-20260907T114731600570Z`

- **方法**：t4_tail_w384_a075_r1；设置 {"model":"QFormerCross8TailRefine","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Tail recurrence strength, depth and width
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.823873194；macroLogloss=0.055418266；is_click AUC/Logloss=0.663162686/0.182695773；cart AUC/Logloss=0.862313792/0.018187637；fav AUC/Logloss=0.878264098/0.013686582；buy AUC/Logloss=0.891752199/0.007103071。
- **开销**：参数 81616324；训练含验证 1847.6770455911756 秒；峰值CUDA 28367455232 bytes；更新 2193。
- **原记录差值**：-0.001568150；对照 `t3_tail_half-s20262027-20260907T110200904859Z`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":17.137983322143555,"mean_ms":17.14423837661743,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001568150，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 round4](#phase-round4)。
- **后续假设**：固定LR/clip/结构，把AdamW换为PSGDKron。
- **证据**：`qformer-loop/taobao-transfer/round4/runs/t4_tail_w384_a075_r1-s20262027-20260907T114731600570Z/manifest.json`（SHA256前缀 `c669a34b417a`）；`qformer-loop/taobao-transfer/round4/runs/t4_tail_w384_a075_r1-s20262027-20260907T114731600570Z/result.json`（SHA256前缀 `e7ef37a0cca9`）；`qformer-loop/taobao-transfer/round4/reports/final-audit.json`（SHA256前缀 `2c217471fff6`）

<a id="t022"></a>
### T022 · `t4_tail_w384_a025_r2-s20262027-20260907T114731646036Z`

- **方法**：t4_tail_w384_a025_r2；设置 {"model":"QFormerCross8TailRefine","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Tail recurrence strength, depth and width
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.823071523；macroLogloss=0.055428432；is_click AUC/Logloss=0.660864411/0.182896944；cart AUC/Logloss=0.862230385/0.018141854；fav AUC/Logloss=0.877588407/0.013678457；buy AUC/Logloss=0.891602887/0.006996471。
- **开销**：参数 81616324；训练含验证 1981.0426245108247 秒；峰值CUDA 32630234112 bytes；更新 2193。
- **原记录差值**：-0.002369821；对照 `t3_tail_half-s20262027-20260907T110200904859Z`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":19.667312622070312,"mean_ms":19.67065758705139,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.002369821，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 round4](#phase-round4)。
- **后续假设**：固定LR/clip/结构，把AdamW换为PSGDKron。
- **证据**：`qformer-loop/taobao-transfer/round4/runs/t4_tail_w384_a025_r2-s20262027-20260907T114731646036Z/manifest.json`（SHA256前缀 `59f93f14b177`）；`qformer-loop/taobao-transfer/round4/runs/t4_tail_w384_a025_r2-s20262027-20260907T114731646036Z/result.json`（SHA256前缀 `5588ea60d498`）；`qformer-loop/taobao-transfer/round4/reports/final-audit.json`（SHA256前缀 `2c217471fff6`）

<a id="t023"></a>
### T023 · `t4_tail_w384_a050_r2-s20262027-20260907T114731708282Z`

- **方法**：t4_tail_w384_a050_r2；设置 {"model":"QFormerCross8TailRefine","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Tail recurrence strength, depth and width
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.821049470；macroLogloss=0.055546474；is_click AUC/Logloss=0.661485183/0.182745779；cart AUC/Logloss=0.857149481/0.018401278；fav AUC/Logloss=0.875273656/0.013785401；buy AUC/Logloss=0.890289559/0.007253438。
- **开销**：参数 81616324；训练含验证 1984.3778465017676 秒；峰值CUDA 32630234112 bytes；更新 2193。
- **原记录差值**：-0.004391874；对照 `t3_tail_half-s20262027-20260907T110200904859Z`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":19.646623611450195,"mean_ms":19.647151136398314,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.004391874，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 round4](#phase-round4)。
- **后续假设**：固定LR/clip/结构，把AdamW换为PSGDKron。
- **证据**：`qformer-loop/taobao-transfer/round4/runs/t4_tail_w384_a050_r2-s20262027-20260907T114731708282Z/manifest.json`（SHA256前缀 `5b16e040935c`）；`qformer-loop/taobao-transfer/round4/runs/t4_tail_w384_a050_r2-s20262027-20260907T114731708282Z/result.json`（SHA256前缀 `63cb25015f53`）；`qformer-loop/taobao-transfer/round4/reports/final-audit.json`（SHA256前缀 `2c217471fff6`）

<a id="t024"></a>
### T024 · `t4_tail_w256_a050_r1-s20262027-20260907T114731525940Z`

- **方法**：t4_tail_w256_a050_r1；设置 {"model":"QFormerCross8TailRefine","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Tail recurrence strength, depth and width
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.821633548；macroLogloss=0.055681106；is_click AUC/Logloss=0.661873762/0.182556258；cart AUC/Logloss=0.860102203/0.018724302；fav AUC/Logloss=0.876663618/0.013988804；buy AUC/Logloss=0.887894607/0.007455060。
- **开销**：参数 54763844；训练含验证 1587.9846074134111 秒；峰值CUDA 19222117888 bytes；更新 2193。
- **原记录差值**：-0.003807796；对照 `t3_tail_half-s20262027-20260907T110200904859Z`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":13.788095951080322,"mean_ms":13.864548778533935,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.003807796，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 round4](#phase-round4)。
- **后续假设**：固定LR/clip/结构，把AdamW换为PSGDKron。
- **证据**：`qformer-loop/taobao-transfer/round4/runs/t4_tail_w256_a050_r1-s20262027-20260907T114731525940Z/manifest.json`（SHA256前缀 `067cf568a76e`）；`qformer-loop/taobao-transfer/round4/runs/t4_tail_w256_a050_r1-s20262027-20260907T114731525940Z/result.json`（SHA256前缀 `1e5a335e6235`）；`qformer-loop/taobao-transfer/round4/reports/final-audit.json`（SHA256前缀 `2c217471fff6`）

<a id="t025"></a>
### T025 · `t4_tail_w512_a050_r1-s20262027-20260907T114731707152Z`

- **方法**：t4_tail_w512_a050_r1；设置 {"model":"QFormerCross8TailRefine","token_dim":512,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Tail recurrence strength, depth and width
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.822224673；macroLogloss=0.055414416；is_click AUC/Logloss=0.663543435/0.182337561；cart AUC/Logloss=0.858405137/0.018250787；fav AUC/Logloss=0.876414233/0.013720052；buy AUC/Logloss=0.890535885/0.007349264。
- **开销**：参数 118757956；训练含验证 2041.7142632380128 秒；峰值CUDA 37644490240 bytes；更新 2193。
- **原记录差值**：-0.003216671；对照 `t3_tail_half-s20262027-20260907T110200904859Z`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":20.8461275100708,"mean_ms":20.84546642303467,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.003216671，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 round4](#phase-round4)。
- **后续假设**：固定LR/clip/结构，把AdamW换为PSGDKron。
- **证据**：`qformer-loop/taobao-transfer/round4/runs/t4_tail_w512_a050_r1-s20262027-20260907T114731707152Z/manifest.json`（SHA256前缀 `79e5237b0719`）；`qformer-loop/taobao-transfer/round4/runs/t4_tail_w512_a050_r1-s20262027-20260907T114731707152Z/result.json`（SHA256前缀 `b249a2f42ffc`）；`qformer-loop/taobao-transfer/round4/reports/final-audit.json`（SHA256前缀 `2c217471fff6`）

<a id="phase-optimizer-psgd"></a>
## 阶段 optimizer-psgd：PSGDKron首个替换

**方法**：tail-half保持结构、学习率及clip1，替换稠密优化器。

**归因与边界**：优化器变化和结构变化分开归因；单组改进需同结构对照。

**历史后续假设**：在另外三种结构上并行比较优化器。

<a id="t026"></a>
### T026 · `psgd_t3_tail_half-s20262027-20260907T122243032651Z`

- **方法**：AdamW -> HeavyBall.PSGDKron 3.2.1; isolated dependency provenance recorded；设置 {"model":"QFormerCross8SelectiveLoop","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Change dense optimizer with fixed architecture, LR and clip1
- **状态／预算**：原记录 `keep_candidate`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.826217786；macroLogloss=0.054972332；is_click AUC/Logloss=0.665849302/0.182232776；cart AUC/Logloss=0.862986462/0.017659931；fav AUC/Logloss=0.879844636/0.013340346；buy AUC/Logloss=0.896190743/0.006656274。
- **开销**：参数 81616324；训练含验证 2114.3906337171793 秒；峰值CUDA 28375993856 bytes；更新 2193。
- **原记录差值**：+0.000776442；对照 `t3_tail_half-s20262027-20260907T110200904859Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计提高0.000776442，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 optimizer-psgd](#phase-optimizer-psgd)。
- **后续假设**：在另外三种结构上并行比较优化器。
- **证据**：`qformer-loop/taobao-transfer/optimizer-psgd/runs/psgd_t3_tail_half-s20262027-20260907T122243032651Z/manifest.json`（SHA256前缀 `2b0b688b037d`）；`qformer-loop/taobao-transfer/optimizer-psgd/runs/psgd_t3_tail_half-s20262027-20260907T122243032651Z/result.json`（SHA256前缀 `654ded5fef3f`）；`qformer-loop/taobao-transfer/optimizer-psgd/reports/final-audit.json`（SHA256前缀 `b77313d13276`）；`qformer-loop/taobao-transfer/optimizer-psgd/interpretation.md`（SHA256前缀 `41d206443847`）

<a id="t027"></a>
### T027 · `taobao-psgd-tail-half-paired-bootstrap`

- **方法**：500 paired user resamples, shared weights across heads
- **原假设**：Assess optimizer gain conditional on fixed checkpoints
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "t3_tail_half",
  "candidate": "psgd_t3_tail_half",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "delta_auc": {
    "is_click": 0.0007743420525723188,
    "cart": -0.0001985145936593824,
    "fav": 0.001168309898755382,
    "buy": 0.0013616314571823773
  },
  "ci95_auc": {
    "is_click": [
      0.00019222947346077268,
      0.0013849759930816133
    ],
    "cart": [
      -0.002101097739404539,
      0.0016405904251271392
    ],
    "fav": [
      -0.000677167720969482,
      0.0030036028449697705
    ],
    "buy": [
      -0.001120192164669387,
      0.0040782762635037636
    ]
  },
  "delta_macro_auc": 0.0007764422037126462,
  "ci95_macro_auc": [
    -0.0001912824473878249,
    0.0016948710897357222
  ],
  "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance or correct for candidate selection; correlated heads share bootstrap user weights"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 optimizer-psgd](#phase-optimizer-psgd)。
- **后续假设**：在另外三种结构上并行比较优化器。
- **证据**：`qformer-loop/taobao-transfer/optimizer-psgd/reports/paired-bootstrap.json`（SHA256前缀 `4893bf387693`）

<a id="phase-optimizer-psgd-parallel"></a>
## 阶段 optimizer-psgd-parallel：PSGDKron跨结构对照

**方法**：Cross8、384单轮、post-half与既有tail-half对照。

**归因与边界**：四结构点估计均受益；并非所有条件区间均为正。宽384单轮更好，不能认为循环与优化器收益相加。

**历史后续假设**：检查预条件更新频率和beta，固定其他预算。

<a id="t028"></a>
### T028 · `psgd_cross8_clip1-s20262027-20260907T123746713967Z`

- **方法**：AdamW -> HeavyBall.PSGDKron; clip1 and LR fixed；设置 {"model":"QFormerCross8","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Compare dense optimizers within unchanged architectures
- **状态／预算**：原记录 `keep_candidate`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.825737871；macroLogloss=0.055080607；is_click AUC/Logloss=0.665845869/0.182131004；cart AUC/Logloss=0.861640292/0.017959436；fav AUC/Logloss=0.880559219/0.013442276；buy AUC/Logloss=0.894906103/0.006789712。
- **开销**：参数 54763844；训练含验证 1752.1507428735495 秒；峰值CUDA 16407979520 bytes；更新 2193。
- **原记录差值**：+0.002688744；对照 `taobao_cross8_clip1-s20262027-20260907T084102654002Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计提高0.002688744，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 optimizer-psgd-parallel](#phase-optimizer-psgd-parallel)。
- **后续假设**：检查预条件更新频率和beta，固定其他预算。
- **证据**：`qformer-loop/taobao-transfer/optimizer-psgd-parallel/runs/psgd_cross8_clip1-s20262027-20260907T123746713967Z/manifest.json`（SHA256前缀 `af1241517419`）；`qformer-loop/taobao-transfer/optimizer-psgd-parallel/runs/psgd_cross8_clip1-s20262027-20260907T123746713967Z/result.json`（SHA256前缀 `c38f971e8b24`）；`qformer-loop/taobao-transfer/optimizer-psgd-parallel/reports/final-audit.json`（SHA256前缀 `32b079565025`）

<a id="t029"></a>
### T029 · `psgd_w384_single-s20262027-20260907T123813211655Z`

- **方法**：AdamW -> HeavyBall.PSGDKron; clip1 and LR fixed；设置 {"model":"QFormerCross8CapacityLoop","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Compare dense optimizers within unchanged architectures
- **状态／预算**：原记录 `keep_candidate`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.827466136；macroLogloss=0.054931224；is_click AUC/Logloss=0.666731695/0.182132962；cart AUC/Logloss=0.865257344/0.017700133；fav AUC/Logloss=0.880975091/0.013276315；buy AUC/Logloss=0.896900416/0.006615488。
- **开销**：参数 81616324；训练含验证 1972.8545798584819 秒；峰值CUDA 24111254016 bytes；更新 2193。
- **原记录差值**：+0.003411249；对照 `t2_cross8_w384_loop1-s20262027-20260907T092620130333Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计提高0.003411249，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 optimizer-psgd-parallel](#phase-optimizer-psgd-parallel)。
- **后续假设**：检查预条件更新频率和beta，固定其他预算。
- **证据**：`qformer-loop/taobao-transfer/optimizer-psgd-parallel/runs/psgd_w384_single-s20262027-20260907T123813211655Z/manifest.json`（SHA256前缀 `7d7aad0e6832`）；`qformer-loop/taobao-transfer/optimizer-psgd-parallel/runs/psgd_w384_single-s20262027-20260907T123813211655Z/result.json`（SHA256前缀 `f6c2176c9e8b`）；`qformer-loop/taobao-transfer/optimizer-psgd-parallel/reports/final-audit.json`（SHA256前缀 `32b079565025`）

<a id="t030"></a>
### T030 · `psgd_post_half-s20262027-20260907T123800399792Z`

- **方法**：AdamW -> HeavyBall.PSGDKron; clip1 and LR fixed；设置 {"model":"QFormerCross8SelectiveLoop","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Compare dense optimizers within unchanged architectures
- **状态／预算**：原记录 `keep_candidate`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.827087569；macroLogloss=0.054968432；is_click AUC/Logloss=0.666428843/0.182011463；cart AUC/Logloss=0.866191175/0.017735422；fav AUC/Logloss=0.880709072/0.013391914；buy AUC/Logloss=0.895021188/0.006734929。
- **开销**：参数 81616324；训练含验证 2031.1165935695171 秒；峰值CUDA 24974312960 bytes；更新 2193。
- **原记录差值**：+0.002117806；对照 `t3_post_half-s20262027-20260907T110201154133Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计提高0.002117806，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 optimizer-psgd-parallel](#phase-optimizer-psgd-parallel)。
- **后续假设**：检查预条件更新频率和beta，固定其他预算。
- **证据**：`qformer-loop/taobao-transfer/optimizer-psgd-parallel/runs/psgd_post_half-s20262027-20260907T123800399792Z/manifest.json`（SHA256前缀 `6466305218a4`）；`qformer-loop/taobao-transfer/optimizer-psgd-parallel/runs/psgd_post_half-s20262027-20260907T123800399792Z/result.json`（SHA256前缀 `d7370d929324`）；`qformer-loop/taobao-transfer/optimizer-psgd-parallel/reports/final-audit.json`（SHA256前缀 `32b079565025`）

<a id="t031"></a>
### T031 · `taobao-psgd_cross8_clip1-paired-bootstrap`

- **方法**：500 paired user resamples, shared weights across heads
- **原假设**：Assess optimizer gain conditional on checkpoints
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "taobao_cross8_clip1",
  "candidate": "psgd_cross8_clip1",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "delta_auc": {
    "is_click": 0.004408348391725769,
    "cart": 0.00018438756131011758,
    "fav": 0.0021971984741917527,
    "buy": 0.003965041464517927
  },
  "ci95_auc": {
    "is_click": [
      0.00376594391586938,
      0.004947953190160986
    ],
    "cart": [
      -0.0014783048672256966,
      0.0017515150199266026
    ],
    "fav": [
      0.0006366941388361886,
      0.0038074219747923853
    ],
    "buy": [
      0.0015276542570843655,
      0.00654668024596603
    ]
  },
  "delta_macro_auc": 0.0026887439729363916,
  "ci95_macro_auc": [
    0.0017247593871374401,
    0.0035902692496661676
  ],
  "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance or correct for candidate selection; correlated heads share bootstrap user weights"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 optimizer-psgd-parallel](#phase-optimizer-psgd-parallel)。
- **后续假设**：检查预条件更新频率和beta，固定其他预算。
- **证据**：`qformer-loop/taobao-transfer/optimizer-psgd-parallel/reports/paired-bootstrap.json`（SHA256前缀 `b3c5ab2ce106`）

<a id="t032"></a>
### T032 · `taobao-psgd_w384_single-paired-bootstrap`

- **方法**：500 paired user resamples, shared weights across heads
- **原假设**：Assess optimizer gain conditional on checkpoints
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "t2_cross8_w384_loop1",
  "candidate": "psgd_w384_single",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "delta_auc": {
    "is_click": 0.0022181628301505274,
    "cart": 0.00335112534664872,
    "fav": 0.0026544250002585867,
    "buy": 0.005421282619586187
  },
  "ci95_auc": {
    "is_click": [
      0.0016102084642900915,
      0.0028090792568192546
    ],
    "cart": [
      0.0015846838899466421,
      0.005313753160170014
    ],
    "fav": [
      0.0010137002986155496,
      0.0043928167711030424
    ],
    "buy": [
      0.0027282616115734347,
      0.007951301119111864
    ]
  },
  "delta_macro_auc": 0.0034112489491611164,
  "ci95_macro_auc": [
    0.0023696681044264984,
    0.004415754909658568
  ],
  "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance or correct for candidate selection; correlated heads share bootstrap user weights"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 optimizer-psgd-parallel](#phase-optimizer-psgd-parallel)。
- **后续假设**：检查预条件更新频率和beta，固定其他预算。
- **证据**：`qformer-loop/taobao-transfer/optimizer-psgd-parallel/reports/paired-bootstrap.json`（SHA256前缀 `b3c5ab2ce106`）

<a id="t033"></a>
### T033 · `taobao-psgd_post_half-paired-bootstrap`

- **方法**：500 paired user resamples, shared weights across heads
- **原假设**：Assess optimizer gain conditional on checkpoints
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "t3_post_half",
  "candidate": "psgd_post_half",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "delta_auc": {
    "is_click": 0.003206153359553765,
    "cart": 0.002233626089167662,
    "fav": 0.001650815548383866,
    "buy": 0.0013806296787379768
  },
  "ci95_auc": {
    "is_click": [
      0.0026327195377299994,
      0.0037860617777662805
    ],
    "cart": [
      0.00039134204694001864,
      0.004001230717938991
    ],
    "fav": [
      -0.00013621896204576154,
      0.0034057433243006694
    ],
    "buy": [
      -0.0010987393301721132,
      0.0040161086884229645
    ]
  },
  "delta_macro_auc": 0.002117806168960845,
  "ci95_macro_auc": [
    0.0011739746275845919,
    0.0030948610835725473
  ],
  "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance or correct for candidate selection; correlated heads share bootstrap user weights"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 optimizer-psgd-parallel](#phase-optimizer-psgd-parallel)。
- **后续假设**：检查预条件更新频率和beta，固定其他预算。
- **证据**：`qformer-loop/taobao-transfer/optimizer-psgd-parallel/reports/paired-bootstrap.json`（SHA256前缀 `b3c5ab2ce106`）

<a id="phase-optimizer-psgd-tune"></a>
## 阶段 optimizer-psgd-tune：预条件更新与beta

**方法**：PSGDKron beta0.95、每步预条件更新等3组。

**归因与边界**：psgd_precond_every_step macroAUC=0.828623533，为当前已归档最高点估计。

**历史后续假设**：在该优化器条件下继续研究循环职责和位置。

<a id="t034"></a>
### T034 · `psgd_beta095-s20262027-20260907T140713697041Z`

- **方法**：{"beta":0.95}；设置 {"model":"QFormerCross8CapacityLoop","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"beta":0.95},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Refine PSGDKron with fixed width384 single-pass architecture
- **状态／预算**：原记录 `keep_candidate`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.827993697；macroLogloss=0.054981736；is_click AUC/Logloss=0.666399676/0.182095874；cart AUC/Logloss=0.866655205/0.017768850；fav AUC/Logloss=0.883077393/0.013357420；buy AUC/Logloss=0.895842513/0.006704800。
- **开销**：参数 81616324；训练含验证 1965.6763686314225 秒；峰值CUDA 24111599104 bytes；更新 2193。
- **原记录差值**：+0.000527560；对照 `psgd_w384_single-s20262027-20260907T123813211655Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计提高0.000527560，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 optimizer-psgd-tune](#phase-optimizer-psgd-tune)。
- **后续假设**：在该优化器条件下继续研究循环职责和位置。
- **证据**：`qformer-loop/taobao-transfer/optimizer-psgd-tune/runs/psgd_beta095-s20262027-20260907T140713697041Z/manifest.json`（SHA256前缀 `246e8e29f038`）；`qformer-loop/taobao-transfer/optimizer-psgd-tune/runs/psgd_beta095-s20262027-20260907T140713697041Z/result.json`（SHA256前缀 `a2fbe876edd8`）；`qformer-loop/taobao-transfer/optimizer-psgd-tune/reports/final-audit.json`（SHA256前缀 `448f6442cfa3`）

<a id="t035"></a>
### T035 · `psgd_raw_precond-s20262027-20260907T140323549376Z`

- **方法**：{"momentum_into_precond_update":false}；设置 {"model":"QFormerCross8CapacityLoop","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"momentum_into_precond_update":false},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Refine PSGDKron with fixed width384 single-pass architecture
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.821978583；macroLogloss=0.055372352；is_click AUC/Logloss=0.663308680/0.182597108；cart AUC/Logloss=0.859770249/0.018172138；fav AUC/Logloss=0.876247797/0.013720424；buy AUC/Logloss=0.888587605/0.006999738。
- **开销**：参数 81616324；训练含验证 1957.0564484223723 秒；峰值CUDA 24113067520 bytes；更新 2193。
- **原记录差值**：-0.005487554；对照 `psgd_w384_single-s20262027-20260907T123813211655Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.005487554，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 optimizer-psgd-tune](#phase-optimizer-psgd-tune)。
- **后续假设**：在该优化器条件下继续研究循环职责和位置。
- **证据**：`qformer-loop/taobao-transfer/optimizer-psgd-tune/runs/psgd_raw_precond-s20262027-20260907T140323549376Z/manifest.json`（SHA256前缀 `93b7805b9fb9`）；`qformer-loop/taobao-transfer/optimizer-psgd-tune/runs/psgd_raw_precond-s20262027-20260907T140323549376Z/result.json`（SHA256前缀 `2983d4124f6e`）；`qformer-loop/taobao-transfer/optimizer-psgd-tune/reports/final-audit.json`（SHA256前缀 `448f6442cfa3`）

<a id="t036"></a>
### T036 · `psgd_precond_every_step-s20262027-20260907T140322750508Z`

- **方法**：{"preconditioner_update_probability":1.0}；设置 {"model":"QFormerCross8CapacityLoop","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Refine PSGDKron with fixed width384 single-pass architecture
- **状态／预算**：原记录 `keep_candidate`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.828623533；macroLogloss=0.054882804；is_click AUC/Logloss=0.667789778/0.182003985；cart AUC/Logloss=0.867019027/0.017635031；fav AUC/Logloss=0.882361229/0.013270322；buy AUC/Logloss=0.897324096/0.006621880。
- **开销**：参数 81616324；训练含验证 1979.6314656883478 秒；峰值CUDA 24111661568 bytes；更新 2193。
- **原记录差值**：+0.001157396；对照 `psgd_w384_single-s20262027-20260907T123813211655Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计提高0.001157396，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 optimizer-psgd-tune](#phase-optimizer-psgd-tune)。
- **后续假设**：在该优化器条件下继续研究循环职责和位置。
- **证据**：`qformer-loop/taobao-transfer/optimizer-psgd-tune/runs/psgd_precond_every_step-s20262027-20260907T140322750508Z/manifest.json`（SHA256前缀 `53977263de24`）；`qformer-loop/taobao-transfer/optimizer-psgd-tune/runs/psgd_precond_every_step-s20262027-20260907T140322750508Z/result.json`（SHA256前缀 `9ecf023b4dae`）；`qformer-loop/taobao-transfer/optimizer-psgd-tune/reports/final-audit.json`（SHA256前缀 `448f6442cfa3`）

<a id="t037"></a>
### T037 · `taobao-tune-psgd_beta095-paired-bootstrap`

- **方法**：500 paired user resamples
- **原假设**：Conditional optimizer gain
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "psgd_w384_single",
  "candidate": "psgd_beta095",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "delta_auc": {
    "is_click": -0.000332018275061019,
    "cart": 0.0013978611255235363,
    "fav": 0.0021023016109559256,
    "buy": -0.0010579026944415038
  },
  "ci95_auc": {
    "is_click": [
      -0.0008803886042271003,
      0.0001862609269602855
    ],
    "cart": [
      -0.0003016939272494079,
      0.0032683979221558718
    ],
    "fav": [
      0.0006534263253855727,
      0.0037003130942986546
    ],
    "buy": [
      -0.0032669844623983424,
      0.0011344916099492119
    ]
  },
  "delta_macro_auc": 0.0005275604417441793,
  "ci95_macro_auc": [
    -0.0002266175298421019,
    0.001436193043529698
  ],
  "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance or correct for candidate selection; correlated heads share bootstrap user weights"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 optimizer-psgd-tune](#phase-optimizer-psgd-tune)。
- **后续假设**：在该优化器条件下继续研究循环职责和位置。
- **证据**：`qformer-loop/taobao-transfer/optimizer-psgd-tune/reports/paired-bootstrap.json`（SHA256前缀 `9da73eb922c9`）

<a id="t038"></a>
### T038 · `taobao-tune-psgd_precond_every_step-paired-bootstrap`

- **方法**：500 paired user resamples
- **原假设**：Conditional optimizer gain
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "psgd_w384_single",
  "candidate": "psgd_precond_every_step",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "delta_auc": {
    "is_click": 0.0010580838164764295,
    "cart": 0.001761683024517624,
    "fav": 0.0013861377414385867,
    "buy": 0.00042368012181936976
  },
  "ci95_auc": {
    "is_click": [
      0.0004466133349234708,
      0.0016088516133680837
    ],
    "cart": [
      0.00023341649852549607,
      0.0034104389203389427
    ],
    "fav": [
      -4.787908942856519e-05,
      0.002860583318306281
    ],
    "buy": [
      -0.0015296894601300958,
      0.002513995762007355
    ]
  },
  "delta_macro_auc": 0.0011573961760629192,
  "ci95_macro_auc": [
    0.0003937883625616055,
    0.0019926658801359687
  ],
  "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance or correct for candidate selection; correlated heads share bootstrap user weights"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 optimizer-psgd-tune](#phase-optimizer-psgd-tune)。
- **后续假设**：在该优化器条件下继续研究循环职责和位置。
- **证据**：`qformer-loop/taobao-transfer/optimizer-psgd-tune/reports/paired-bootstrap.json`（SHA256前缀 `9da73eb922c9`）

<a id="phase-recurrence-middle"></a>
## 阶段 recurrence-middle：中间共享循环

**方法**：中间层两次调用，有无固定入口注入。

**归因与边界**：默认预条件调度下middle_r2接近单轮；anchor0.1退化。

**历史后续假设**：检验训练深度和中间监督，区分训练问题与结构问题。

<a id="t039"></a>
### T039 · `middle_r2-s20262027-20260907T162109565214Z`

- **方法**：{"anchor_scale":0.0,"middle_steps":2}；设置 {"model":"QFormerCross8MiddleLoop","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Shared middle recurrence with optional fixed entry injection
- **状态／预算**：原记录 `keep_candidate`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.827502882；macroLogloss=0.055026433；is_click AUC/Logloss=0.665850020/0.182236421；cart AUC/Logloss=0.865639189/0.017739375；fav AUC/Logloss=0.881919665/0.013373626；buy AUC/Logloss=0.896602655/0.006756310。
- **开销**：参数 81616324；训练含验证 2105.7653791829944 秒；峰值CUDA 28475930112 bytes；更新 2193。
- **原记录差值**：+0.000036746；对照 `psgd_w384_single-s20262027-20260907T123813211655Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计提高0.000036746，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-middle](#phase-recurrence-middle)。
- **后续假设**：检验训练深度和中间监督，区分训练问题与结构问题。
- **证据**：`qformer-loop/taobao-transfer/recurrence-middle/runs/middle_r2-s20262027-20260907T162109565214Z/manifest.json`（SHA256前缀 `e3646ed5f26f`）；`qformer-loop/taobao-transfer/recurrence-middle/runs/middle_r2-s20262027-20260907T162109565214Z/result.json`（SHA256前缀 `1f20d72688e6`）；`qformer-loop/taobao-transfer/recurrence-middle/reports/final-audit.json`（SHA256前缀 `19752ed04225`）

<a id="t040"></a>
### T040 · `middle_r2_anchor010-s20262027-20260907T161618129374Z`

- **方法**：{"anchor_scale":0.1,"middle_steps":2}；设置 {"model":"QFormerCross8MiddleLoop","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Shared middle recurrence with optional fixed entry injection
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.826446521；macroLogloss=0.055023275；is_click AUC/Logloss=0.664327018/0.182251682；cart AUC/Logloss=0.865675526/0.017663316；fav AUC/Logloss=0.878939084/0.013385310；buy AUC/Logloss=0.896844455/0.006792791。
- **开销**：参数 81616324；训练含验证 2088.269413098693 秒；峰值CUDA 28474819072 bytes；更新 2193。
- **原记录差值**：-0.001019616；对照 `psgd_w384_single-s20262027-20260907T123813211655Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.001019616，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-middle](#phase-recurrence-middle)。
- **后续假设**：检验训练深度和中间监督，区分训练问题与结构问题。
- **证据**：`qformer-loop/taobao-transfer/recurrence-middle/runs/middle_r2_anchor010-s20262027-20260907T161618129374Z/manifest.json`（SHA256前缀 `4c25f1237c47`）；`qformer-loop/taobao-transfer/recurrence-middle/runs/middle_r2_anchor010-s20262027-20260907T161618129374Z/result.json`（SHA256前缀 `a096dcf7847c`）；`qformer-loop/taobao-transfer/recurrence-middle/reports/final-audit.json`（SHA256前缀 `19752ed04225`）

<a id="t041"></a>
### T041 · `taobao-middle-middle_r2-paired-bootstrap`

- **方法**：500 paired user resamples
- **原假设**：Conditional middle recurrence gain
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "psgd_w384_single",
  "candidate": "middle_r2",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "delta_auc": {
    "is_click": -0.0008816746221884664,
    "cart": 0.00038184561785337134,
    "fav": 0.0009445740954674431,
    "buy": -0.00029776072824172495
  },
  "ci95_auc": {
    "is_click": [
      -0.0015185330275847618,
      -0.000262128733641268
    ],
    "cart": [
      -0.001412407359209264,
      0.002135253999489786
    ],
    "fav": [
      -0.0007669938266757158,
      0.0025276017763292435
    ],
    "buy": [
      -0.002441684202186553,
      0.0022847260022598485
    ]
  },
  "delta_macro_auc": 3.674609072257251e-05,
  "ci95_macro_auc": [
    -0.0007907269024882597,
    0.0010020121193010493
  ],
  "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance or correct for candidate selection; correlated heads share bootstrap user weights"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-middle](#phase-recurrence-middle)。
- **后续假设**：检验训练深度和中间监督，区分训练问题与结构问题。
- **证据**：`qformer-loop/taobao-transfer/recurrence-middle/reports/paired-bootstrap.json`（SHA256前缀 `b231c4d3c102`）

<a id="t042"></a>
### T042 · `taobao-middle-depth-middle-depth-curve`

- **方法**：Evaluate middle depth1/2/3/4
- **原假设**：Does additional recurrence improve one frozen checkpoint?
- **状态／预算**：原记录 `diagnostic`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "1": {
    "validation": {
      "is_click": {
        "AUC": 0.6597119815419278,
        "logloss": 0.18359743765503217
      },
      "cart": {
        "AUC": 0.86018674800029,
        "logloss": 0.01789054053240356
      },
      "fav": {
        "AUC": 0.8808096928396866,
        "logloss": 0.013460140410715676
      },
      "buy": {
        "AUC": 0.8953517080604937,
        "logloss": 0.006826552397128003
      }
    },
    "macro_AUC": 0.8240150326105995
  },
  "2": {
    "validation": {
      "is_click": {
        "AUC": 0.665850019952777,
        "logloss": 0.1822364208983367
      },
      "cart": {
        "AUC": 0.865639189234401,
        "logloss": 0.017739375238749012
      },
      "fav": {
        "AUC": 0.8819196652414157,
        "logloss": 0.01337362568854116
      },
      "buy": {
        "AUC": 0.8966026553087438,
        "logloss": 0.0067563097070809435
      }
    },
    "macro_AUC": 0.8275028824343343
  },
  "3": {
    "validation": {
      "is_click": {
        "AUC": 0.6648576661849764,
        "logloss": 0.18216563687006437
      },
      "cart": {
        "AUC": 0.8643506787619998,
        "logloss": 0.018146876498510718
      },
      "fav": {
        "AUC": 0.880493150765133,
        "logloss": 0.013501525219107894
      },
      "buy": {
        "AUC": 0.8932798807139176,
        "logloss": 0.007048671754391135
      }
    },
    "macro_AUC": 0.8257453441065067
  },
  "4": {
    "validation": {
      "is_click": {
        "AUC": 0.6605068677596102,
        "logloss": 0.18256670282353848
      },
      "cart": {
        "AUC": 0.8576264205430084,
        "logloss": 0.01913435711989578
      },
      "fav": {
        "AUC": 0.8778918187065845,
        "logloss": 0.014051977107172564
      },
      "buy": {
        "AUC": 0.8831093492934906,
        "logloss": 0.007888425006923667
      }
    },
    "macro_AUC": 0.8197836140756735
  }
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-middle](#phase-recurrence-middle)。
- **后续假设**：检验训练深度和中间监督，区分训练问题与结构问题。
- **证据**：`qformer-loop/taobao-transfer/recurrence-middle/reports/depth-middle/result.json`（SHA256前缀 `1bc49a4da484`）

<a id="t043"></a>
### T043 · `taobao-middle-depth-anchor010-depth-curve`

- **方法**：Evaluate middle depth1/2/3/4
- **原假设**：Does additional recurrence improve one frozen checkpoint?
- **状态／预算**：原记录 `diagnostic`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "1": {
    "validation": {
      "is_click": {
        "AUC": 0.6610125507459184,
        "logloss": 0.18305584457927632
      },
      "cart": {
        "AUC": 0.8625661133052164,
        "logloss": 0.017776426513920102
      },
      "fav": {
        "AUC": 0.8794428200711135,
        "logloss": 0.013399472581234423
      },
      "buy": {
        "AUC": 0.8957851799492597,
        "logloss": 0.006889105491798823
      }
    },
    "macro_AUC": 0.8247016660178769
  },
  "2": {
    "validation": {
      "is_click": {
        "AUC": 0.6643270175816082,
        "logloss": 0.182251681713331
      },
      "cart": {
        "AUC": 0.865675525598574,
        "logloss": 0.017663315656501342
      },
      "fav": {
        "AUC": 0.8789390840722765,
        "logloss": 0.01338531019841893
      },
      "buy": {
        "AUC": 0.8968444553996352,
        "logloss": 0.006792790688774889
      }
    },
    "macro_AUC": 0.8264465206630235
  },
  "3": {
    "validation": {
      "is_click": {
        "AUC": 0.6632867883324925,
        "logloss": 0.18226554665443448
      },
      "cart": {
        "AUC": 0.8651837925129034,
        "logloss": 0.017731096511784054
      },
      "fav": {
        "AUC": 0.8770308103310435,
        "logloss": 0.013501148523637372
      },
      "buy": {
        "AUC": 0.896148197581576,
        "logloss": 0.006868475749067778
      }
    },
    "macro_AUC": 0.8254123971895039
  },
  "4": {
    "validation": {
      "is_click": {
        "AUC": 0.6608846781108332,
        "logloss": 0.1825805805103598
      },
      "cart": {
        "AUC": 0.862310916006368,
        "logloss": 0.018097249262265518
      },
      "fav": {
        "AUC": 0.8742057545043856,
        "logloss": 0.013845232552363506
      },
      "buy": {
        "AUC": 0.8930639256035815,
        "logloss": 0.007150180062991446
      }
    },
    "macro_AUC": 0.822616318556292
  }
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-middle](#phase-recurrence-middle)。
- **后续假设**：检验训练深度和中间监督，区分训练问题与结构问题。
- **证据**：`qformer-loop/taobao-transfer/recurrence-middle/reports/depth-anchor010/result.json`（SHA256前缀 `9303f28af4f4`）

<a id="phase-recurrence-training"></a>
## 阶段 recurrence-training：循环训练方式

**方法**：mixed123训练深度或两轮中间监督，验证固定两轮。

**归因与边界**：两组均退化；推断深度更鲁棒不等于固定两轮AUC更高。

**历史后续假设**：改变共享核心更新或交互方式。

<a id="t044"></a>
### T044 · `taobao-middle-prior-history-bins`

- **方法**：Fixed bins 0-9,10-29,30-59,>=60 on saved validation predictions
- **原假设**：Recurrence benefits may depend on raw history length
- **状态／预算**：原记录 `No evidence that longest histories benefit most; descriptive only`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "delta_macro_auc_by_bin": [
    0.0032245900267859895,
    0.0004401339098651924,
    0.0004957415201107818,
    -0.0008169462917205772
  ]
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-training](#phase-recurrence-training)。
- **后续假设**：改变共享核心更新或交互方式。
- **证据**：`qformer-loop/taobao-transfer/recurrence-training/PRIOR_HISTORY.md`（SHA256前缀 `8fa65b8554e5`）；`qformer-loop/taobao-transfer/recurrence-training/reports/prior-history-diagnostic.json`（SHA256前缀 `c0e8b3ca7421`）

<a id="t045"></a>
### T045 · `middle_mixed123-s20262027-20260908T035111667976Z`

- **方法**：{"auxiliary_weight":0.25,"depth_training":"mixed123"}；设置 {"model":"QFormerCross8DepthTraining","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Mixed training depths or normalized intermediate supervision on shared middle recurrence
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.826472683；macroLogloss=0.054997064；is_click AUC/Logloss=0.666819042/0.182089407；cart AUC/Logloss=0.865984331/0.017753972；fav AUC/Logloss=0.880639377/0.013323337；buy AUC/Logloss=0.892447982/0.006821541。
- **开销**：参数 81616324；训练含验证 2088.7066301628947 秒；峰值CUDA 32736726528 bytes；更新 2193。
- **原记录差值**：-0.001030199；对照 `middle_r2-s20262027-20260907T162109565214Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.001030199，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-training](#phase-recurrence-training)。
- **后续假设**：改变共享核心更新或交互方式。
- **证据**：`qformer-loop/taobao-transfer/recurrence-training/runs/middle_mixed123-s20262027-20260908T035111667976Z/manifest.json`（SHA256前缀 `d34067fdf86f`）；`qformer-loop/taobao-transfer/recurrence-training/runs/middle_mixed123-s20262027-20260908T035111667976Z/result.json`（SHA256前缀 `b0f4cd7dc320`）；`qformer-loop/taobao-transfer/recurrence-training/reports/final-audit.json`（SHA256前缀 `cff6b0c8c036`）

<a id="t046"></a>
### T046 · `middle_supervised12-s20262027-20260908T035104357175Z`

- **方法**：{"auxiliary_weight":0.25,"depth_training":"supervised12"}；设置 {"model":"QFormerCross8DepthTraining","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Mixed training depths or normalized intermediate supervision on shared middle recurrence
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.825484529；macroLogloss=0.055050376；is_click AUC/Logloss=0.665805922/0.182196183；cart AUC/Logloss=0.862025774/0.017828435；fav AUC/Logloss=0.878272699/0.013411403；buy AUC/Logloss=0.895833720/0.006765484。
- **开销**：参数 81616324；训练含验证 3624.666365273297 秒；峰值CUDA 49095100928 bytes；更新 2193。
- **原记录差值**：-0.002018354；对照 `middle_r2-s20262027-20260907T162109565214Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.002018354，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-training](#phase-recurrence-training)。
- **后续假设**：改变共享核心更新或交互方式。
- **证据**：`qformer-loop/taobao-transfer/recurrence-training/runs/middle_supervised12-s20262027-20260908T035104357175Z/manifest.json`（SHA256前缀 `20577c4f3e0d`）；`qformer-loop/taobao-transfer/recurrence-training/runs/middle_supervised12-s20262027-20260908T035104357175Z/result.json`（SHA256前缀 `4f552e7c4083`）；`qformer-loop/taobao-transfer/recurrence-training/reports/final-audit.json`（SHA256前缀 `cff6b0c8c036`）

<a id="t047"></a>
### T047 · `taobao-depth-training-mixed123-depth-curve`

- **方法**：Frozen mixed123 checkpoint evaluated at depths1/2/3/4; only4 unseen in training
- **原假设**：Mixed depths reduce deterioration at greater recurrence depth
- **状态／预算**：原记录 `Improved depth robustness but lower fixed-R2 AUC`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "macro_auc_by_depth": {
    "1": 0.825977728690046,
    "2": 0.8264726829772377,
    "3": 0.8264130165451168,
    "4": 0.8259148748088244
  },
  "native_max_prediction_difference": 0
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-training](#phase-recurrence-training)。
- **后续假设**：改变共享核心更新或交互方式。
- **证据**：`qformer-loop/taobao-transfer/recurrence-training/reports/depth-mixed123/result.json`（SHA256前缀 `df63fa0413e0`）；`qformer-loop/taobao-transfer/recurrence-training/DEPTH.md`（SHA256前缀 `17644c08e647`）

<a id="t048"></a>
### T048 · `taobao-depth-training-supervised12-depth-curve`

- **方法**：Frozen supervised12 checkpoint at1/2/3/4;3/4 unseen
- **原假设**：Intermediate supervision reduces deterioration at greater recurrence depth
- **状态／预算**：原记录 `Less depth deterioration but lower peak AUC`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "macro_auc_by_depth": {
    "1": 0.8249836113180737,
    "2": 0.8254845285900987,
    "3": 0.8251059201766087,
    "4": 0.8235831741058937
  },
  "native_max_prediction_difference": 0
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-training](#phase-recurrence-training)。
- **后续假设**：改变共享核心更新或交互方式。
- **证据**：`qformer-loop/taobao-transfer/recurrence-training/reports/depth-supervised12/result.json`（SHA256前缀 `5dc6b75d9f30`）；`qformer-loop/taobao-transfer/recurrence-training/DEPTH.md`（SHA256前缀 `17644c08e647`）

<a id="phase-recurrence-core"></a>
## 阶段 recurrence-core：共享核心更新

**方法**：第二轮半幅更新或有界交互。

**归因与边界**：半幅明显退化；有界交互接近但未优于纯R2。

**历史后续假设**：将候选放到最佳每步预条件设置下做匹配控制。

<a id="t049"></a>
### T049 · `middle_repeat_half-s20262027-20260908T051342447366Z`

- **方法**：{"bounded_interaction":false,"repeat_update_scale":0.5}；设置 {"model":"QFormerCross8CoreUpdate","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Repeated-step damping or bounded query interaction in fixed two-step core
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.824856705；macroLogloss=0.055181374；is_click AUC/Logloss=0.662838510/0.182682744；cart AUC/Logloss=0.863611241/0.017835274；fav AUC/Logloss=0.878638014/0.013356714；buy AUC/Logloss=0.894339057/0.006850762。
- **开销**：参数 81616324；训练含验证 2085.4829764291644 秒；峰值CUDA 28474918400 bytes；更新 2193。
- **原记录差值**：-0.002646177；对照 `middle_r2-s20262027-20260907T162109565214Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.002646177，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-core](#phase-recurrence-core)。
- **后续假设**：将候选放到最佳每步预条件设置下做匹配控制。
- **证据**：`qformer-loop/taobao-transfer/recurrence-core/runs/middle_repeat_half-s20262027-20260908T051342447366Z/manifest.json`（SHA256前缀 `9614257a3850`）；`qformer-loop/taobao-transfer/recurrence-core/runs/middle_repeat_half-s20262027-20260908T051342447366Z/result.json`（SHA256前缀 `9ef0ba0a684a`）；`qformer-loop/taobao-transfer/recurrence-core/reports/final-audit.json`（SHA256前缀 `584e0ade17cf`）

<a id="t050"></a>
### T050 · `middle_bounded_interaction-s20262027-20260908T051339531098Z`

- **方法**：{"bounded_interaction":true,"repeat_update_scale":1.0}；设置 {"model":"QFormerCross8CoreUpdate","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Repeated-step damping or bounded query interaction in fixed two-step core
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.827147308；macroLogloss=0.054955822；is_click AUC/Logloss=0.667514724/0.182244930；cart AUC/Logloss=0.865709698/0.017654672；fav AUC/Logloss=0.880411711/0.013275898；buy AUC/Logloss=0.894953098/0.006647790。
- **开销**：参数 81616324；训练含验证 2079.4651391729712 秒；峰值CUDA 28476132864 bytes；更新 2193。
- **原记录差值**：-0.000355575；对照 `middle_r2-s20262027-20260907T162109565214Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.000355575，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-core](#phase-recurrence-core)。
- **后续假设**：将候选放到最佳每步预条件设置下做匹配控制。
- **证据**：`qformer-loop/taobao-transfer/recurrence-core/runs/middle_bounded_interaction-s20262027-20260908T051339531098Z/manifest.json`（SHA256前缀 `52da27854cdd`）；`qformer-loop/taobao-transfer/recurrence-core/runs/middle_bounded_interaction-s20262027-20260908T051339531098Z/result.json`（SHA256前缀 `4bea4d546e41`）；`qformer-loop/taobao-transfer/recurrence-core/reports/final-audit.json`（SHA256前缀 `584e0ade17cf`）

<a id="phase-recurrence-best"></a>
## 阶段 recurrence-best：最佳优化器下重测循环

**方法**：每步预条件更新下的纯R2及有界交互。

**归因与边界**：两组均低于最佳单轮；优化器与结构存在交互。

**历史后续假设**：只放松第二轮输出投影共享；见adapter。

<a id="t051"></a>
### T051 · `middle_r2_every_step-s20262027-20260908T055716005349Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"middle_steps":2,"model":"QFormerCross8MiddleLoop"}；设置 {"model":"QFormerCross8MiddleLoop","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Middle recurrence versus single pass under established every-step preconditioning
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.826190865；macroLogloss=0.055001378；is_click AUC/Logloss=0.666789383/0.181943096；cart AUC/Logloss=0.863317900/0.017842680；fav AUC/Logloss=0.879511895/0.013413537；buy AUC/Logloss=0.895144282/0.006806201。
- **开销**：参数 81616324；训练含验证 2093.758184760809 秒；峰值CUDA 28475769344 bytes；更新 2193。
- **原记录差值**：-0.002432667；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.002432667，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-best](#phase-recurrence-best)。
- **后续假设**：只放松第二轮输出投影共享；见adapter。
- **证据**：`qformer-loop/taobao-transfer/recurrence-best/runs/middle_r2_every_step-s20262027-20260908T055716005349Z/manifest.json`（SHA256前缀 `dc52f88a0bb9`）；`qformer-loop/taobao-transfer/recurrence-best/runs/middle_r2_every_step-s20262027-20260908T055716005349Z/result.json`（SHA256前缀 `7a76f699b139`）；`qformer-loop/taobao-transfer/recurrence-best/reports/final-audit.json`（SHA256前缀 `2b78826b1ac7`）

<a id="t052"></a>
### T052 · `middle_bounded_every_step-s20262027-20260908T055714438679Z`

- **方法**：{"bounded_interaction":true,"dense_optimizer_options":{"preconditioner_update_probability":1.0},"middle_steps":2,"model":"QFormerCross8CoreUpdate"}；设置 {"model":"QFormerCross8CoreUpdate","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Middle recurrence versus single pass under established every-step preconditioning
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.827117153；macroLogloss=0.054949321；is_click AUC/Logloss=0.666357406/0.182144022；cart AUC/Logloss=0.866058079/0.017667304；fav AUC/Logloss=0.879858227/0.013318793；buy AUC/Logloss=0.896194898/0.006667166。
- **开销**：参数 81616324；训练含验证 2092.084237039089 秒；峰值CUDA 28476132864 bytes；更新 2193。
- **原记录差值**：-0.001506380；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.001506380，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-best](#phase-recurrence-best)。
- **后续假设**：只放松第二轮输出投影共享；见adapter。
- **证据**：`qformer-loop/taobao-transfer/recurrence-best/runs/middle_bounded_every_step-s20262027-20260908T055714438679Z/manifest.json`（SHA256前缀 `b5a927460d15`）；`qformer-loop/taobao-transfer/recurrence-best/runs/middle_bounded_every_step-s20262027-20260908T055714438679Z/result.json`（SHA256前缀 `b6fe1280fca8`）；`qformer-loop/taobao-transfer/recurrence-best/reports/final-audit.json`（SHA256前缀 `2b78826b1ac7`）

<a id="phase-recurrence-adapter"></a>
## 阶段 recurrence-adapter：第二轮输出LoRA

**方法**：输出投影rank8 LoRA，增加6144参数。

**归因与边界**：较纯R2点估计恢复+0.000256，但低于最佳单轮且有任务/Logloss代价。

**历史后续假设**：将工作状态与预测状态分开。

<a id="t053"></a>
### T053 · `middle_r2_lora8_every_step-s20262027-20260908T082358293826Z`

- **方法**：{"adapter_rank":8,"dense_optimizer_options":{"preconditioner_update_probability":1.0},"middle_steps":2,"model":"QFormerCross8MiddleAdapter"}；设置 {"model":"QFormerCross8MiddleAdapter","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Second-iteration output-projection LoRA relaxes strict middle recurrence sharing
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.826446913；macroLogloss=0.055057659；is_click AUC/Logloss=0.664714846/0.182329114；cart AUC/Logloss=0.864781665/0.017749578；fav AUC/Logloss=0.880692415/0.013396808；buy AUC/Logloss=0.895598727/0.006755139。
- **开销**：参数 81622468；训练含验证 2103.2548705488443 秒；峰值CUDA 28377163264 bytes；更新 2193。
- **原记录差值**：-0.002176619；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.002176619，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-adapter](#phase-recurrence-adapter)。
- **后续假设**：将工作状态与预测状态分开。
- **证据**：`qformer-loop/taobao-transfer/recurrence-adapter/runs/middle_r2_lora8_every_step-s20262027-20260908T082358293826Z/manifest.json`（SHA256前缀 `03c1d004ca65`）；`qformer-loop/taobao-transfer/recurrence-adapter/runs/middle_r2_lora8_every_step-s20262027-20260908T082358293826Z/result.json`（SHA256前缀 `d6449f6913c2`）；`qformer-loop/taobao-transfer/recurrence-adapter/reports/final-audit.json`（SHA256前缀 `79b75375b4bb`）

<a id="phase-recurrence-dual-state"></a>
## 阶段 recurrence-dual-state：双状态一/二轮

**方法**：工作状态和预测状态分离；比较1/2轮。

**归因与边界**：R2−R1 macroAUC+0.000762715但95%区间跨0，加购收益较明确；任务间方向不一致。

**历史后续假设**：比较去反馈、GRU、归一化及共享范围。

<a id="t054"></a>
### T054 · `dual_state_r1-s20262027-20260908T090746489340Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"dual_steps":1,"model":"QFormerCross8DualState"}；设置 {"model":"QFormerCross8DualState","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Separate history working state and prediction state, then compare one versus two cycles
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.826631172；macroLogloss=0.055132852；is_click AUC/Logloss=0.665940189/0.182568477；cart AUC/Logloss=0.863395843/0.017832094；fav AUC/Logloss=0.881012038/0.013391776；buy AUC/Logloss=0.896176618/0.006739058。
- **开销**：参数 81616324；训练含验证 1870.6334405615926 秒；峰值CUDA 20207856128 bytes；更新 2193。
- **原记录差值**：-0.001992361；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.001992361，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-dual-state](#phase-recurrence-dual-state)。
- **后续假设**：比较去反馈、GRU、归一化及共享范围。
- **证据**：`qformer-loop/taobao-transfer/recurrence-dual-state/runs/dual_state_r1-s20262027-20260908T090746489340Z/manifest.json`（SHA256前缀 `38bca3a52a49`）；`qformer-loop/taobao-transfer/recurrence-dual-state/runs/dual_state_r1-s20262027-20260908T090746489340Z/result.json`（SHA256前缀 `ebf936590316`）；`qformer-loop/taobao-transfer/recurrence-dual-state/reports/final-audit.json`（SHA256前缀 `2b78826b1ac7`）

<a id="t055"></a>
### T055 · `dual_state_r2-s20262027-20260908T090747849085Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"dual_steps":2,"model":"QFormerCross8DualState"}；设置 {"model":"QFormerCross8DualState","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Separate history working state and prediction state, then compare one versus two cycles
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.827393887；macroLogloss=0.055054757；is_click AUC/Logloss=0.666252379/0.182351672；cart AUC/Logloss=0.867459564/0.017749009；fav AUC/Logloss=0.880795937/0.013408474；buy AUC/Logloss=0.895067670/0.006709872。
- **开销**：参数 81616324；训练含验证 2031.1431803107262 秒；峰值CUDA 25230854656 bytes；更新 2193。
- **原记录差值**：-0.001229645；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.001229645，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-dual-state](#phase-recurrence-dual-state)。
- **后续假设**：比较去反馈、GRU、归一化及共享范围。
- **证据**：`qformer-loop/taobao-transfer/recurrence-dual-state/runs/dual_state_r2-s20262027-20260908T090747849085Z/manifest.json`（SHA256前缀 `fac5def5c62a`）；`qformer-loop/taobao-transfer/recurrence-dual-state/runs/dual_state_r2-s20262027-20260908T090747849085Z/result.json`（SHA256前缀 `971de54103b1`）；`qformer-loop/taobao-transfer/recurrence-dual-state/reports/final-audit.json`（SHA256前缀 `2b78826b1ac7`）

<a id="t056"></a>
### T056 · `taobao-dual-state-dual_state_r1-dual_state_r2-paired-bootstrap`

- **方法**：500 paired user resamples
- **原假设**：Conditional core update gain
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "dual_state_r1",
  "candidate": "dual_state_r2",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "delta_auc": {
    "is_click": 0.00031218979397706903,
    "cart": 0.004063721006305143,
    "fav": -0.00021610097552882124,
    "buy": -0.0011089479004676717
  },
  "ci95_auc": {
    "is_click": [
      -0.00030860508748696276,
      0.0009073200560874734
    ],
    "cart": [
      0.002225742660435695,
      0.005830028510958245
    ],
    "fav": [
      -0.0020350644504237097,
      0.0018138984514218018
    ],
    "buy": [
      -0.00369910768687961,
      0.0016454626588159793
    ]
  },
  "delta_macro_auc": 0.0007627154810713188,
  "ci95_macro_auc": [
    -0.00022430915777606886,
    0.0017334739123308165
  ],
  "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance or correct for candidate selection; correlated heads share bootstrap user weights"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-dual-state](#phase-recurrence-dual-state)。
- **后续假设**：比较去反馈、GRU、归一化及共享范围。
- **证据**：`qformer-loop/taobao-transfer/recurrence-dual-state/reports/paired-bootstrap.json`（SHA256前缀 `b185ee98c70c`）

<a id="t057"></a>
### T057 · `taobao-dual-state-r1-depth-curve`

- **方法**：Evaluate one and two cycles on full validation; reproduce native predictions
- **原假设**：Does the trained recurrence help at fixed weights?
- **状态／预算**：原记录 `diagnostic`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "scores": {
    "1": {
      "validation": {
        "is_click": {
          "AUC": 0.6659401890142416,
          "logloss": 0.1825684773692603
        },
        "cart": {
          "AUC": 0.8633958433724217,
          "logloss": 0.017832093972120534
        },
        "fav": {
          "AUC": 0.8810120376512781,
          "logloss": 0.013391776227325186
        },
        "buy": {
          "AUC": 0.8961766175648992,
          "logloss": 0.006739058499943639
        }
      },
      "macro_AUC": 0.8266311719007102
    },
    "2": {
      "validation": {
        "is_click": {
          "AUC": 0.6556024746913354,
          "logloss": 0.1912323196693901
        },
        "cart": {
          "AUC": 0.8560118098478146,
          "logloss": 0.019462208975003708
        },
        "fav": {
          "AUC": 0.8588879408973733,
          "logloss": 0.01690342981149384
        },
        "buy": {
          "AUC": 0.8853757279551876,
          "logloss": 0.008548448091195803
        }
      },
      "macro_AUC": 0.8139694883479276
    }
  },
  "validation_rows": 2871767,
  "native_max_prediction_difference": 0.0,
  "training_updates": 0,
  "test_evaluated": false,
  "native_depth": 1,
  "scope": "Same frozen checkpoint at one/two cycles; off-native depth is an inference ablation, not independently trained performance."
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-dual-state](#phase-recurrence-dual-state)。
- **后续假设**：比较去反馈、GRU、归一化及共享范围。
- **证据**：`qformer-loop/taobao-transfer/recurrence-dual-state/reports/depth-r1/result.json`（SHA256前缀 `4c0e9aa3f7f6`）

<a id="t058"></a>
### T058 · `taobao-dual-state-r2-depth-curve`

- **方法**：Evaluate one and two cycles on full validation; reproduce native predictions
- **原假设**：Does the trained recurrence help at fixed weights?
- **状态／预算**：原记录 `diagnostic`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "scores": {
    "1": {
      "validation": {
        "is_click": {
          "AUC": 0.658321591209222,
          "logloss": 0.18495325167429774
        },
        "cart": {
          "AUC": 0.8574203971794438,
          "logloss": 0.018223126378836718
        },
        "fav": {
          "AUC": 0.8747163835839196,
          "logloss": 0.01375295232772944
        },
        "buy": {
          "AUC": 0.892727467949569,
          "logloss": 0.007446819523008563
        }
      },
      "macro_AUC": 0.8207964599805386
    },
    "2": {
      "validation": {
        "is_click": {
          "AUC": 0.6662523788082186,
          "logloss": 0.18235167227979568
        },
        "cart": {
          "AUC": 0.8674595643787268,
          "logloss": 0.01774900925277068
        },
        "fav": {
          "AUC": 0.8807959366757493,
          "logloss": 0.01340847389225111
        },
        "buy": {
          "AUC": 0.8950676696644315,
          "logloss": 0.0067098715273972405
        }
      },
      "macro_AUC": 0.8273938873817815
    }
  },
  "validation_rows": 2871767,
  "native_max_prediction_difference": 0.0,
  "training_updates": 0,
  "test_evaluated": false,
  "native_depth": 2,
  "scope": "Same frozen checkpoint at one/two cycles; off-native depth is an inference ablation, not independently trained performance."
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-dual-state](#phase-recurrence-dual-state)。
- **后续假设**：比较去反馈、GRU、归一化及共享范围。
- **证据**：`qformer-loop/taobao-transfer/recurrence-dual-state/reports/depth-r2/result.json`（SHA256前缀 `4cfc5add3625`）

<a id="phase-recurrence-explore"></a>
## 阶段 recurrence-explore：循环核心多结构探索

**方法**：repeat norm、GRU、entry residual、去预测反馈、独立层、双层core。

**归因与边界**：去反馈两轮0.828521213接近最佳单轮；相对有反馈提高，但输入混合权重也改变，不能把全部增量独归于删反馈。

**历史后续假设**：围绕去反馈设计控制轮数、宽度、入口权重。

<a id="t059"></a>
### T059 · `explore_repeat_norm-s20262027-20260908T092501210059Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"model":"QFormerCross8RecurrenceExplore","recurrence_mode":"repeat_norm"}；设置 {"model":"QFormerCross8RecurrenceExplore","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Explore six recurrence structures with fixed best optimizer and seed
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.826853770；macroLogloss=0.055021707；is_click AUC/Logloss=0.665150194/0.182298029；cart AUC/Logloss=0.864460893/0.017713431；fav AUC/Logloss=0.882582178/0.013316683；buy AUC/Logloss=0.895221814/0.006758683。
- **开销**：参数 81616324；训练含验证 2137.9383400455117 秒；峰值CUDA 28676765184 bytes；更新 2193。
- **原记录差值**：-0.001769763；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.001769763，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-explore](#phase-recurrence-explore)。
- **后续假设**：围绕去反馈设计控制轮数、宽度、入口权重。
- **证据**：`qformer-loop/taobao-transfer/recurrence-explore/runs/explore_repeat_norm-s20262027-20260908T092501210059Z/manifest.json`（SHA256前缀 `6759f0eb297a`）；`qformer-loop/taobao-transfer/recurrence-explore/runs/explore_repeat_norm-s20262027-20260908T092501210059Z/result.json`（SHA256前缀 `24310ac2ab06`）；`qformer-loop/taobao-transfer/recurrence-explore/reports/final-audit.json`（SHA256前缀 `bcdabcb1d71c`）

<a id="t060"></a>
### T060 · `explore_gru_repeat-s20262027-20260908T092103203475Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"model":"QFormerCross8RecurrenceExplore","recurrence_mode":"gru_repeat"}；设置 {"model":"QFormerCross8RecurrenceExplore","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Explore six recurrence structures with fixed best optimizer and seed
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.825971422；macroLogloss=0.055098392；is_click AUC/Logloss=0.663230555/0.182502973；cart AUC/Logloss=0.865622347/0.017788936；fav AUC/Logloss=0.880899330/0.013386319；buy AUC/Logloss=0.894133456/0.006715341。
- **开销**：参数 82503364；训练含验证 2146.437692180276 秒；峰值CUDA 29247138304 bytes；更新 2193。
- **原记录差值**：-0.002652111；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.002652111，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-explore](#phase-recurrence-explore)。
- **后续假设**：围绕去反馈设计控制轮数、宽度、入口权重。
- **证据**：`qformer-loop/taobao-transfer/recurrence-explore/runs/explore_gru_repeat-s20262027-20260908T092103203475Z/manifest.json`（SHA256前缀 `5a890d571df5`）；`qformer-loop/taobao-transfer/recurrence-explore/runs/explore_gru_repeat-s20262027-20260908T092103203475Z/result.json`（SHA256前缀 `7e71ced09963`）；`qformer-loop/taobao-transfer/recurrence-explore/reports/final-audit.json`（SHA256前缀 `bcdabcb1d71c`）

<a id="t061"></a>
### T061 · `explore_entry_residual-s20262027-20260908T092002151214Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"model":"QFormerCross8RecurrenceExplore","recurrence_mode":"entry_residual"}；设置 {"model":"QFormerCross8RecurrenceExplore","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Explore six recurrence structures with fixed best optimizer and seed
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.827319207；macroLogloss=0.055029758；is_click AUC/Logloss=0.666376447/0.182311251；cart AUC/Logloss=0.866123051/0.017707582；fav AUC/Logloss=0.882554945/0.013315860；buy AUC/Logloss=0.894222386/0.006784339。
- **开销**：参数 81616324；训练含验证 2116.8135821968317 秒；峰值CUDA 28677095936 bytes；更新 2193。
- **原记录差值**：-0.001304325；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.001304325，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-explore](#phase-recurrence-explore)。
- **后续假设**：围绕去反馈设计控制轮数、宽度、入口权重。
- **证据**：`qformer-loop/taobao-transfer/recurrence-explore/runs/explore_entry_residual-s20262027-20260908T092002151214Z/manifest.json`（SHA256前缀 `66cafc3fe346`）；`qformer-loop/taobao-transfer/recurrence-explore/runs/explore_entry_residual-s20262027-20260908T092002151214Z/result.json`（SHA256前缀 `bd210595f5cf`）；`qformer-loop/taobao-transfer/recurrence-explore/reports/final-audit.json`（SHA256前缀 `bcdabcb1d71c`）

<a id="t062"></a>
### T062 · `explore_dual_no_feedback-s20262027-20260908T091954887719Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"model":"QFormerCross8RecurrenceExplore","recurrence_mode":"dual_no_feedback"}；设置 {"model":"QFormerCross8RecurrenceExplore","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Explore six recurrence structures with fixed best optimizer and seed
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.828521213；macroLogloss=0.054888144；is_click AUC/Logloss=0.666869886/0.182044098；cart AUC/Logloss=0.867496836/0.017627952；fav AUC/Logloss=0.882010545/0.013276472；buy AUC/Logloss=0.897707586/0.006604053。
- **开销**：参数 81616324；训练含验证 2059.4449480995536 秒；峰值CUDA 25231673856 bytes；更新 2193。
- **原记录差值**：-0.000102319；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.000102319，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-explore](#phase-recurrence-explore)。
- **后续假设**：围绕去反馈设计控制轮数、宽度、入口权重。
- **证据**：`qformer-loop/taobao-transfer/recurrence-explore/runs/explore_dual_no_feedback-s20262027-20260908T091954887719Z/manifest.json`（SHA256前缀 `926952c6069a`）；`qformer-loop/taobao-transfer/recurrence-explore/runs/explore_dual_no_feedback-s20262027-20260908T091954887719Z/result.json`（SHA256前缀 `90c9883dbc48`）；`qformer-loop/taobao-transfer/recurrence-explore/reports/final-audit.json`（SHA256前缀 `bcdabcb1d71c`）

<a id="t063"></a>
### T063 · `explore_untied_r2-s20262027-20260908T094526671908Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"model":"QFormerCross8RecurrenceExplore","recurrence_mode":"untied_r2"}；设置 {"model":"QFormerCross8RecurrenceExplore","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Explore six recurrence structures with fixed best optimizer and seed
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.826954952；macroLogloss=0.054997472；is_click AUC/Logloss=0.665611710/0.182066868；cart AUC/Logloss=0.864377647/0.017796527；fav AUC/Logloss=0.881800661/0.013383868；buy AUC/Logloss=0.896029790/0.006742626。
- **开销**：参数 82503748；训练含验证 2147.9386936947703 秒；峰值CUDA 28589417984 bytes；更新 2193。
- **原记录差值**：-0.001668581；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.001668581，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-explore](#phase-recurrence-explore)。
- **后续假设**：围绕去反馈设计控制轮数、宽度、入口权重。
- **证据**：`qformer-loop/taobao-transfer/recurrence-explore/runs/explore_untied_r2-s20262027-20260908T094526671908Z/manifest.json`（SHA256前缀 `87166afb7d1d`）；`qformer-loop/taobao-transfer/recurrence-explore/runs/explore_untied_r2-s20262027-20260908T094526671908Z/result.json`（SHA256前缀 `ff0c95b0d4f4`）；`qformer-loop/taobao-transfer/recurrence-explore/reports/final-audit.json`（SHA256前缀 `bcdabcb1d71c`）

<a id="t064"></a>
### T064 · `explore_two_layer_core-s20262027-20260908T092455690446Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"model":"QFormerCross8RecurrenceExplore","recurrence_mode":"two_layer_core"}；设置 {"model":"QFormerCross8RecurrenceExplore","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Explore six recurrence structures with fixed best optimizer and seed
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.827946255；macroLogloss=0.054959282；is_click AUC/Logloss=0.666262666/0.182091975；cart AUC/Logloss=0.867224673/0.017673102；fav AUC/Logloss=0.881328971/0.013316212；buy AUC/Logloss=0.896968710/0.006755839。
- **开销**：参数 81616324；训练含验证 2274.3631464391947 秒；峰值CUDA 32837288448 bytes；更新 2193。
- **原记录差值**：-0.000677278；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.000677278，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-explore](#phase-recurrence-explore)。
- **后续假设**：围绕去反馈设计控制轮数、宽度、入口权重。
- **证据**：`qformer-loop/taobao-transfer/recurrence-explore/runs/explore_two_layer_core-s20262027-20260908T092455690446Z/manifest.json`（SHA256前缀 `78a7206b9f1d`）；`qformer-loop/taobao-transfer/recurrence-explore/runs/explore_two_layer_core-s20262027-20260908T092455690446Z/result.json`（SHA256前缀 `bb61787f1140`）；`qformer-loop/taobao-transfer/recurrence-explore/reports/final-audit.json`（SHA256前缀 `bcdabcb1d71c`）

<a id="t065"></a>
### T065 · `taobao-dual-feedback-removal-paired-bootstrap`

- **方法**：500 paired user resamples on frozen full-validation predictions
- **原假设**：Does no-feedback working-state update improve versus feedback R2?
- **状态／预算**：原记录 `retain_candidate`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "dual_state_r2",
  "candidate": "explore_dual_no_feedback",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "delta_auc": {
    "is_click": 0.0006175072169222195,
    "cart": 3.727132830300128e-05,
    "fav": 0.0012146088011447675,
    "buy": 0.0026399166234053473
  },
  "ci95_auc": {
    "is_click": [
      -9.45526074271341e-06,
      0.001305786526179649
    ],
    "cart": [
      -0.00150406237966815,
      0.0017769361724243604
    ],
    "fav": [
      -0.0008559802918301106,
      0.0029619419850794608
    ],
    "buy": [
      0.00016896425774538004,
      0.005223309885911338
    ]
  },
  "delta_macro_auc": 0.0011273259924439172,
  "ci95_macro_auc": [
    0.000220408837119241,
    0.0022419871031725317
  ],
  "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance or correct for candidate selection; correlated heads share bootstrap user weights"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-explore](#phase-recurrence-explore)。
- **后续假设**：围绕去反馈设计控制轮数、宽度、入口权重。
- **证据**：`qformer-loop/taobao-transfer/recurrence-explore/reports/feedback-paired-bootstrap.json`（SHA256前缀 `12c35f11a17a`）；`qformer-loop/taobao-transfer/recurrence-explore/feedback_bootstrap.py`

<a id="t066"></a>
### T066 · `taobao-no-feedback-r2-depth-curve`

- **方法**：Inference depths1..4
- **原假设**：Check whether extra inference recurrence improves the near-best no-feedback checkpoint
- **状态／预算**：原记录 `retain_trained_depth`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "scores": {
    "1": {
      "validation": {
        "is_click": {
          "AUC": 0.6560092315228884,
          "logloss": 0.18563793702135103
        },
        "cart": {
          "AUC": 0.8544316239973342,
          "logloss": 0.018731405426854423
        },
        "fav": {
          "AUC": 0.8699138373735673,
          "logloss": 0.014195489950957423
        },
        "buy": {
          "AUC": 0.8839537140287674,
          "logloss": 0.007589058549879338
        }
      },
      "macro_AUC": 0.8160771017306394
    },
    "2": {
      "validation": {
        "is_click": {
          "AUC": 0.6668698860251409,
          "logloss": 0.18204409784365805
        },
        "cart": {
          "AUC": 0.8674968357070298,
          "logloss": 0.017627952080585766
        },
        "fav": {
          "AUC": 0.8820105454768941,
          "logloss": 0.01327647198117377
        },
        "buy": {
          "AUC": 0.8977075862878369,
          "logloss": 0.0066040527588318325
        }
      },
      "macro_AUC": 0.8285212133742255
    },
    "3": {
      "validation": {
        "is_click": {
          "AUC": 0.6669189421962995,
          "logloss": 0.18194752628099067
        },
        "cart": {
          "AUC": 0.8662575111009897,
          "logloss": 0.01779011311760876
        },
        "fav": {
          "AUC": 0.8795540449164223,
          "logloss": 0.013399167859908119
        },
        "buy": {
          "AUC": 0.8955105654317489,
          "logloss": 0.006694782965582561
        }
      },
      "macro_AUC": 0.8270602659113651
    },
    "4": {
      "validation": {
        "is_click": {
          "AUC": 0.6612298553364696,
          "logloss": 0.18279976095521963
        },
        "cart": {
          "AUC": 0.8457442946808383,
          "logloss": 0.018593660030731506
        },
        "fav": {
          "AUC": 0.8567538546511069,
          "logloss": 0.014147991178199275
        },
        "buy": {
          "AUC": 0.8841449822121191,
          "logloss": 0.007032103548647657
        }
      },
      "macro_AUC": 0.8119682467201336
    }
  },
  "validation_rows": 2871767,
  "native_max_prediction_difference": 0.0,
  "training_updates": 0,
  "test_evaluated": false,
  "native_depth": 2,
  "scope": "Frozen no-feedback two-cycle checkpoint evaluated at depths1..4; depth1 ablates a trained cycle, depths3/4 are untrained extrapolation. Native two-cycle predictions must reproduce the original implementation."
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-explore](#phase-recurrence-explore)。
- **后续假设**：围绕去反馈设计控制轮数、宽度、入口权重。
- **证据**：`qformer-loop/taobao-transfer/recurrence-explore/reports/depth-no-feedback/result.json`（SHA256前缀 `7b9d597b1ad2`）；`qformer-loop/taobao-transfer/recurrence-explore/DEPTH.md`（SHA256前缀 `4645d5a1a66b`）

<a id="phase-recurrence-no-feedback"></a>
## 阶段 recurrence-no-feedback：去反馈参数控制

**方法**：3轮、256/512宽度、入口权重0.25/0.75。

**归因与边界**：五组均低于原去反馈两轮，继续加轮数/整体加宽优先级降低。

**历史后续假设**：研究工作/预测路径和每轮读什么。

<a id="t067"></a>
### T067 · `nf_r3-s20262027-20260908T102310408135Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"model":"QFormerCross8NoFeedback","token_dim":384,"work_steps":3}；设置 {"model":"QFormerCross8NoFeedback","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":3,"entry_weight":0.5}
- **原假设**：Vary one factor of the near-best no-feedback recurrence: depth, width or entry mixture
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.826571786；macroLogloss=0.054998236；is_click AUC/Logloss=0.665429353/0.182236783；cart AUC/Logloss=0.865506402/0.017670297；fav AUC/Logloss=0.880161573/0.013330737；buy AUC/Logloss=0.895189816/0.006755128。
- **开销**：参数 81616324；训练含验证 2224.4564643129706 秒；峰值CUDA 30254425600 bytes；更新 2193。
- **原记录差值**：-0.002051747；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.002051747，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-no-feedback](#phase-recurrence-no-feedback)。
- **后续假设**：研究工作/预测路径和每轮读什么。
- **证据**：`qformer-loop/taobao-transfer/recurrence-no-feedback/runs/nf_r3-s20262027-20260908T102310408135Z/manifest.json`（SHA256前缀 `0b0fb01edf1e`）；`qformer-loop/taobao-transfer/recurrence-no-feedback/runs/nf_r3-s20262027-20260908T102310408135Z/result.json`（SHA256前缀 `6cde98786180`）；`qformer-loop/taobao-transfer/recurrence-no-feedback/reports/final-audit.json`（SHA256前缀 `cfe64e8352f3`）

<a id="t068"></a>
### T068 · `nf_w256-s20262027-20260908T102239802613Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"model":"QFormerCross8NoFeedback","token_dim":256,"work_steps":2}；设置 {"model":"QFormerCross8NoFeedback","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Vary one factor of the near-best no-feedback recurrence: depth, width or entry mixture
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.826578748；macroLogloss=0.055117433；is_click AUC/Logloss=0.665190834/0.182356798；cart AUC/Logloss=0.864689033/0.017855681；fav AUC/Logloss=0.879801207/0.013445024；buy AUC/Logloss=0.896633917/0.006812228。
- **开销**：参数 54763844；训练含验证 1836.6374571099877 秒；峰值CUDA 17168791040 bytes；更新 2193。
- **原记录差值**：-0.002044785；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.002044785，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-no-feedback](#phase-recurrence-no-feedback)。
- **后续假设**：研究工作/预测路径和每轮读什么。
- **证据**：`qformer-loop/taobao-transfer/recurrence-no-feedback/runs/nf_w256-s20262027-20260908T102239802613Z/manifest.json`（SHA256前缀 `99228de7737f`）；`qformer-loop/taobao-transfer/recurrence-no-feedback/runs/nf_w256-s20262027-20260908T102239802613Z/result.json`（SHA256前缀 `0a3921c4efae`）；`qformer-loop/taobao-transfer/recurrence-no-feedback/reports/final-audit.json`（SHA256前缀 `cfe64e8352f3`）

<a id="t069"></a>
### T069 · `nf_w512-s20262027-20260908T102311202625Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"model":"QFormerCross8NoFeedback","token_dim":512,"work_steps":2}；设置 {"model":"QFormerCross8NoFeedback","token_dim":512,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Vary one factor of the near-best no-feedback recurrence: depth, width or entry mixture
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.827328026；macroLogloss=0.055086425；is_click AUC/Logloss=0.664334194/0.182562857；cart AUC/Logloss=0.866403014/0.017751240；fav AUC/Logloss=0.880386282/0.013401185；buy AUC/Logloss=0.898188615/0.006630416。
- **开销**：参数 118757956；训练含验证 2253.6116210222244 秒；峰值CUDA 33391995392 bytes；更新 2193。
- **原记录差值**：-0.001295506；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.001295506，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-no-feedback](#phase-recurrence-no-feedback)。
- **后续假设**：研究工作/预测路径和每轮读什么。
- **证据**：`qformer-loop/taobao-transfer/recurrence-no-feedback/runs/nf_w512-s20262027-20260908T102311202625Z/manifest.json`（SHA256前缀 `3676e5123a6b`）；`qformer-loop/taobao-transfer/recurrence-no-feedback/runs/nf_w512-s20262027-20260908T102311202625Z/result.json`（SHA256前缀 `391c1c7990e5`）；`qformer-loop/taobao-transfer/recurrence-no-feedback/reports/final-audit.json`（SHA256前缀 `cfe64e8352f3`）

<a id="t070"></a>
### T070 · `nf_entry025-s20262027-20260908T102300904513Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.25,"model":"QFormerCross8NoFeedback","token_dim":384,"work_steps":2}；设置 {"model":"QFormerCross8NoFeedback","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.25}
- **原假设**：Vary one factor of the near-best no-feedback recurrence: depth, width or entry mixture
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.825529326；macroLogloss=0.055075482；is_click AUC/Logloss=0.664197311/0.182571698；cart AUC/Logloss=0.865819573/0.017686000；fav AUC/Logloss=0.878653124/0.013333337；buy AUC/Logloss=0.893447297/0.006710895。
- **开销**：参数 81616324；训练含验证 2062.9815936908126 秒；峰值CUDA 25231051264 bytes；更新 2193。
- **原记录差值**：-0.003094206；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.003094206，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-no-feedback](#phase-recurrence-no-feedback)。
- **后续假设**：研究工作/预测路径和每轮读什么。
- **证据**：`qformer-loop/taobao-transfer/recurrence-no-feedback/runs/nf_entry025-s20262027-20260908T102300904513Z/manifest.json`（SHA256前缀 `ded373233687`）；`qformer-loop/taobao-transfer/recurrence-no-feedback/runs/nf_entry025-s20262027-20260908T102300904513Z/result.json`（SHA256前缀 `2c5f4be43119`）；`qformer-loop/taobao-transfer/recurrence-no-feedback/reports/final-audit.json`（SHA256前缀 `cfe64e8352f3`）

<a id="t071"></a>
### T071 · `nf_entry075-s20262027-20260908T102308673232Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.75,"model":"QFormerCross8NoFeedback","token_dim":384,"work_steps":2}；设置 {"model":"QFormerCross8NoFeedback","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.75}
- **原假设**：Vary one factor of the near-best no-feedback recurrence: depth, width or entry mixture
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.825908145；macroLogloss=0.055017752；is_click AUC/Logloss=0.665347385/0.182173322；cart AUC/Logloss=0.864079997/0.017844080；fav AUC/Logloss=0.881130214/0.013340699；buy AUC/Logloss=0.893074982/0.006712907。
- **开销**：参数 81616324；训练含验证 2055.7077881991863 秒；峰值CUDA 25231483392 bytes；更新 2193。
- **原记录差值**：-0.002715388；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.002715388，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-no-feedback](#phase-recurrence-no-feedback)。
- **后续假设**：研究工作/预测路径和每轮读什么。
- **证据**：`qformer-loop/taobao-transfer/recurrence-no-feedback/runs/nf_entry075-s20262027-20260908T102308673232Z/manifest.json`（SHA256前缀 `b01cb77f3f44`）；`qformer-loop/taobao-transfer/recurrence-no-feedback/runs/nf_entry075-s20262027-20260908T102308673232Z/result.json`（SHA256前缀 `8d64dcd5fe6c`）；`qformer-loop/taobao-transfer/recurrence-no-feedback/reports/final-audit.json`（SHA256前缀 `cfe64e8352f3`）

<a id="phase-recurrence-dual-width"></a>
## 阶段 recurrence-dual-width：有反馈双状态容量

**方法**：双状态R2宽度256/512。

**归因与边界**：宽度增大不保证优于最佳单轮；保留与同架构原宽度的配对，而不混入去反馈效应。

**历史后续假设**：固定合适宽度并研究记忆读取和状态路线。

<a id="t072"></a>
### T072 · `dual_r2_w256-s20262027-20260908T101307274817Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"dual_steps":2,"model":"QFormerCross8DualState","token_dim":256}；设置 {"model":"QFormerCross8DualState","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Scale width of the two-cycle dual-state architecture
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.826377025；macroLogloss=0.055137886；is_click AUC/Logloss=0.664971695/0.182338094；cart AUC/Logloss=0.865133534/0.017921437；fav AUC/Logloss=0.879333598/0.013466000；buy AUC/Logloss=0.896069274/0.006826011。
- **开销**：参数 54763844；训练含验证 1839.757906831801 秒；峰值CUDA 17171355136 bytes；更新 2193。
- **原记录差值**：-0.002246507；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.002246507，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-dual-width](#phase-recurrence-dual-width)。
- **后续假设**：固定合适宽度并研究记忆读取和状态路线。
- **证据**：`qformer-loop/taobao-transfer/recurrence-dual-width/runs/dual_r2_w256-s20262027-20260908T101307274817Z/manifest.json`（SHA256前缀 `2f930d5c816c`）；`qformer-loop/taobao-transfer/recurrence-dual-width/runs/dual_r2_w256-s20262027-20260908T101307274817Z/result.json`（SHA256前缀 `79ae80986ddc`）；`qformer-loop/taobao-transfer/recurrence-dual-width/reports/final-audit.json`（SHA256前缀 `4ecb5b4555cf`）

<a id="t073"></a>
### T073 · `dual_r2_w512-s20262027-20260908T101324145875Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"dual_steps":2,"model":"QFormerCross8DualState","token_dim":512}；设置 {"model":"QFormerCross8DualState","token_dim":512,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Scale width of the two-cycle dual-state architecture
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.827840344；macroLogloss=0.054983634；is_click AUC/Logloss=0.664640768/0.182490518；cart AUC/Logloss=0.866915311/0.017563012；fav AUC/Logloss=0.880644767/0.013260535；buy AUC/Logloss=0.899160529/0.006620473。
- **开销**：参数 118757956；训练含验证 2255.9636947140098 秒；峰值CUDA 33391531008 bytes；更新 2193。
- **原记录差值**：-0.000783189；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.000783189，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-dual-width](#phase-recurrence-dual-width)。
- **后续假设**：固定合适宽度并研究记忆读取和状态路线。
- **证据**：`qformer-loop/taobao-transfer/recurrence-dual-width/runs/dual_r2_w512-s20262027-20260908T101324145875Z/manifest.json`（SHA256前缀 `e7e11200e892`）；`qformer-loop/taobao-transfer/recurrence-dual-width/runs/dual_r2_w512-s20262027-20260908T101324145875Z/result.json`（SHA256前缀 `2a6b4efc9417`）；`qformer-loop/taobao-transfer/recurrence-dual-width/reports/final-audit.json`（SHA256前缀 `4ecb5b4555cf`）

<a id="t074"></a>
### T074 · `taobao-dual-width-dual_state_r2-dual_r2_w512-paired-bootstrap`

- **方法**：500 paired user resamples
- **原假设**：Conditional core update gain
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "dual_state_r2",
  "candidate": "dual_r2_w512",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "delta_auc": {
    "is_click": -0.0016116110351258595,
    "cart": -0.000544252883807661,
    "fav": -0.0001511695127153212,
    "buy": 0.004092859570019591
  },
  "ci95_auc": {
    "is_click": [
      -0.00223691657191237,
      -0.0009354073340982206
    ],
    "cart": [
      -0.0021766858516847286,
      0.0013475358010714273
    ],
    "fav": [
      -0.0021901036289165093,
      0.0016272742411746076
    ],
    "buy": [
      0.0015420543464195607,
      0.006600824689496385
    ]
  },
  "delta_macro_auc": 0.0004464565345926319,
  "ci95_macro_auc": [
    -0.00048120875815612914,
    0.0014366718223338686
  ],
  "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance or correct for candidate selection; correlated heads share bootstrap user weights"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-dual-width](#phase-recurrence-dual-width)。
- **后续假设**：固定合适宽度并研究记忆读取和状态路线。
- **证据**：`qformer-loop/taobao-transfer/recurrence-dual-width/reports/paired-bootstrap.json`（SHA256前缀 `5d312295cb2c`）

<a id="phase-recurrence-fixed-memory"></a>
## 阶段 recurrence-fixed-memory：固定工作记忆

**方法**：第二轮复用固定工作记忆。

**归因与边界**：移除动态第二次读取降低质量；表明当前模型依赖更新，不给出最优query公式。

**历史后续假设**：保留动态历史读取，分解输入条件。

<a id="t075"></a>
### T075 · `nf_fixed_memory-s20262027-20260908T115619206369Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":1.0,"model":"QFormerCross8NoFeedback","token_dim":384,"work_steps":2}；设置 {"model":"QFormerCross8NoFeedback","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":1.0}
- **原假设**：Set entry mixture to one: repeated prediction updates read identical working memory
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.825274532；macroLogloss=0.055060240；is_click AUC/Logloss=0.665954081/0.182081136；cart AUC/Logloss=0.863069980/0.017924337；fav AUC/Logloss=0.878686216/0.013494123；buy AUC/Logloss=0.893387851/0.006741364。
- **开销**：参数 81616324；训练含验证 2075.550017274916 秒；峰值CUDA 25231051264 bytes；更新 2193。
- **原记录差值**：-0.003349000；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.003349000，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-fixed-memory](#phase-recurrence-fixed-memory)。
- **后续假设**：保留动态历史读取，分解输入条件。
- **证据**：`qformer-loop/taobao-transfer/recurrence-fixed-memory/runs/nf_fixed_memory-s20262027-20260908T115619206369Z/manifest.json`（SHA256前缀 `6e2f489ec88e`）；`qformer-loop/taobao-transfer/recurrence-fixed-memory/runs/nf_fixed_memory-s20262027-20260908T115619206369Z/result.json`（SHA256前缀 `1b178dc3a556`）；`qformer-loop/taobao-transfer/recurrence-fixed-memory/reports/final-audit.json`（SHA256前缀 `fb7ba341f90e`）

<a id="phase-recurrence-state-routes"></a>
## 阶段 recurrence-state-routes：循环状态路线

**方法**：入口去除、工作/预测半幅、预测重置、norm、额外内循环。

**归因与边界**：7组均低于原去反馈；状态持久性和原始混合应作为后续对照。

**历史后续假设**：测试上下文回读、attention位置与查询槽数。

<a id="t076"></a>
### T076 · `route_anchor0-s20262027-20260908T120102731470Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"model":"QFormerCross8StateRoutes","route_mode":"anchor0","token_dim":384}；设置 {"model":"QFormerCross8StateRoutes","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Vary working/prediction recurrence routing relative to the near-best no-feedback model
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.825367030；macroLogloss=0.055140988；is_click AUC/Logloss=0.664528761/0.182446409；cart AUC/Logloss=0.863775779/0.017888498；fav AUC/Logloss=0.879271975/0.013421802；buy AUC/Logloss=0.893891606/0.006807242。
- **开销**：参数 81616324；训练含验证 2055.806269943714 秒；峰值CUDA 25231670784 bytes；更新 2193。
- **原记录差值**：-0.003256502；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.003256502，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-state-routes](#phase-recurrence-state-routes)。
- **后续假设**：测试上下文回读、attention位置与查询槽数。
- **证据**：`qformer-loop/taobao-transfer/recurrence-state-routes/runs/route_anchor0-s20262027-20260908T120102731470Z/manifest.json`（SHA256前缀 `bff4f1353ed0`）；`qformer-loop/taobao-transfer/recurrence-state-routes/runs/route_anchor0-s20262027-20260908T120102731470Z/result.json`（SHA256前缀 `5e94ee62e484`）；`qformer-loop/taobao-transfer/recurrence-state-routes/reports/final-audit.json`（SHA256前缀 `7c0fb1d9eeaa`）

<a id="t077"></a>
### T077 · `route_work_half-s20262027-20260908T120047806663Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"model":"QFormerCross8StateRoutes","route_mode":"work_half","token_dim":384}；设置 {"model":"QFormerCross8StateRoutes","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Vary working/prediction recurrence routing relative to the near-best no-feedback model
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.822044207；macroLogloss=0.055302534；is_click AUC/Logloss=0.659364834/0.183145198；cart AUC/Logloss=0.861859816/0.017907305；fav AUC/Logloss=0.876731956/0.013429502；buy AUC/Logloss=0.890220222/0.006728133。
- **开销**：参数 81616324；训练含验证 2064.35209569335 秒；峰值CUDA 25231673856 bytes；更新 2193。
- **原记录差值**：-0.006579326；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.006579326，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-state-routes](#phase-recurrence-state-routes)。
- **后续假设**：测试上下文回读、attention位置与查询槽数。
- **证据**：`qformer-loop/taobao-transfer/recurrence-state-routes/runs/route_work_half-s20262027-20260908T120047806663Z/manifest.json`（SHA256前缀 `543b229ec48e`）；`qformer-loop/taobao-transfer/recurrence-state-routes/runs/route_work_half-s20262027-20260908T120047806663Z/result.json`（SHA256前缀 `5f5e696f853e`）；`qformer-loop/taobao-transfer/recurrence-state-routes/reports/final-audit.json`（SHA256前缀 `7c0fb1d9eeaa`）

<a id="t078"></a>
### T078 · `route_prediction_half-s20262027-20260908T120058810641Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"model":"QFormerCross8StateRoutes","route_mode":"prediction_half","token_dim":384}；设置 {"model":"QFormerCross8StateRoutes","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Vary working/prediction recurrence routing relative to the near-best no-feedback model
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.820982528；macroLogloss=0.055246142；is_click AUC/Logloss=0.659054450/0.182776190；cart AUC/Logloss=0.856600514/0.017993000；fav AUC/Logloss=0.876522593/0.013425392；buy AUC/Logloss=0.891752554/0.006789985。
- **开销**：参数 81616324；训练含验证 2068.7867084518075 秒；峰值CUDA 25276833280 bytes；更新 2193。
- **原记录差值**：-0.007641005；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.007641005，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-state-routes](#phase-recurrence-state-routes)。
- **后续假设**：测试上下文回读、attention位置与查询槽数。
- **证据**：`qformer-loop/taobao-transfer/recurrence-state-routes/runs/route_prediction_half-s20262027-20260908T120058810641Z/manifest.json`（SHA256前缀 `5f72086c70cc`）；`qformer-loop/taobao-transfer/recurrence-state-routes/runs/route_prediction_half-s20262027-20260908T120058810641Z/result.json`（SHA256前缀 `63012b6fd7a4`）；`qformer-loop/taobao-transfer/recurrence-state-routes/reports/final-audit.json`（SHA256前缀 `7c0fb1d9eeaa`）

<a id="t079"></a>
### T079 · `route_prediction_reset-s20262027-20260908T120047717081Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"model":"QFormerCross8StateRoutes","route_mode":"prediction_reset","token_dim":384}；设置 {"model":"QFormerCross8StateRoutes","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Vary working/prediction recurrence routing relative to the near-best no-feedback model
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.827306938；macroLogloss=0.054974328；is_click AUC/Logloss=0.666567896/0.182124813；cart AUC/Logloss=0.864986645/0.017765067；fav AUC/Logloss=0.879771948/0.013334789；buy AUC/Logloss=0.897901264/0.006672645。
- **开销**：参数 81616324；训练含验证 2042.6154856160283 秒；峰值CUDA 25176163840 bytes；更新 2193。
- **原记录差值**：-0.001316594；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.001316594，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-state-routes](#phase-recurrence-state-routes)。
- **后续假设**：测试上下文回读、attention位置与查询槽数。
- **证据**：`qformer-loop/taobao-transfer/recurrence-state-routes/runs/route_prediction_reset-s20262027-20260908T120047717081Z/manifest.json`（SHA256前缀 `73cdc0d47b04`）；`qformer-loop/taobao-transfer/recurrence-state-routes/runs/route_prediction_reset-s20262027-20260908T120047717081Z/result.json`（SHA256前缀 `54fe73ce104e`）；`qformer-loop/taobao-transfer/recurrence-state-routes/reports/final-audit.json`（SHA256前缀 `7c0fb1d9eeaa`）

<a id="t080"></a>
### T080 · `route_work_norm-s20262027-20260908T120055362230Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"model":"QFormerCross8StateRoutes","route_mode":"work_norm","token_dim":384}；设置 {"model":"QFormerCross8StateRoutes","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Vary working/prediction recurrence routing relative to the near-best no-feedback model
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.826399454；macroLogloss=0.055108180；is_click AUC/Logloss=0.663863383/0.182568925；cart AUC/Logloss=0.866180612/0.017754447；fav AUC/Logloss=0.878491142/0.013413181；buy AUC/Logloss=0.897062678/0.006696167。
- **开销**：参数 81616324；训练含验证 2063.7955828756094 秒；峰值CUDA 25332435456 bytes；更新 2193。
- **原记录差值**：-0.002224079；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.002224079，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-state-routes](#phase-recurrence-state-routes)。
- **后续假设**：测试上下文回读、attention位置与查询槽数。
- **证据**：`qformer-loop/taobao-transfer/recurrence-state-routes/runs/route_work_norm-s20262027-20260908T120055362230Z/manifest.json`（SHA256前缀 `c8b9c234c5d8`）；`qformer-loop/taobao-transfer/recurrence-state-routes/runs/route_work_norm-s20262027-20260908T120055362230Z/result.json`（SHA256前缀 `ed6fd3cea281`）；`qformer-loop/taobao-transfer/recurrence-state-routes/reports/final-audit.json`（SHA256前缀 `7c0fb1d9eeaa`）

<a id="t081"></a>
### T081 · `route_prediction_norm-s20262027-20260908T120058680326Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"model":"QFormerCross8StateRoutes","route_mode":"prediction_norm","token_dim":384}；设置 {"model":"QFormerCross8StateRoutes","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Vary working/prediction recurrence routing relative to the near-best no-feedback model
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.825044663；macroLogloss=0.055080513；is_click AUC/Logloss=0.661552130/0.182716313；cart AUC/Logloss=0.866950349/0.017640664；fav AUC/Logloss=0.876750983/0.013342537；buy AUC/Logloss=0.894925190/0.006622538。
- **开销**：参数 81616324；训练含验证 2061.660775117576 秒；峰值CUDA 25433623040 bytes；更新 2193。
- **原记录差值**：-0.003578869；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.003578869，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-state-routes](#phase-recurrence-state-routes)。
- **后续假设**：测试上下文回读、attention位置与查询槽数。
- **证据**：`qformer-loop/taobao-transfer/recurrence-state-routes/runs/route_prediction_norm-s20262027-20260908T120058680326Z/manifest.json`（SHA256前缀 `4f1fbd53d072`）；`qformer-loop/taobao-transfer/recurrence-state-routes/runs/route_prediction_norm-s20262027-20260908T120058680326Z/result.json`（SHA256前缀 `828cbc089069`）；`qformer-loop/taobao-transfer/recurrence-state-routes/reports/final-audit.json`（SHA256前缀 `7c0fb1d9eeaa`）

<a id="t082"></a>
### T082 · `route_inner2-s20262027-20260908T120101271853Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"model":"QFormerCross8StateRoutes","route_mode":"inner2","token_dim":384}；设置 {"model":"QFormerCross8StateRoutes","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Vary working/prediction recurrence routing relative to the near-best no-feedback model
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.826667644；macroLogloss=0.055216942；is_click AUC/Logloss=0.664615745/0.182563278；cart AUC/Logloss=0.865297239/0.018021478；fav AUC/Logloss=0.880216668/0.013405648；buy AUC/Logloss=0.896540925/0.006877363。
- **开销**：参数 81616324；训练含验证 2324.169887021184 秒；峰值CUDA 33754466816 bytes；更新 2193。
- **原记录差值**：-0.001955888；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.001955888，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-state-routes](#phase-recurrence-state-routes)。
- **后续假设**：测试上下文回读、attention位置与查询槽数。
- **证据**：`qformer-loop/taobao-transfer/recurrence-state-routes/runs/route_inner2-s20262027-20260908T120101271853Z/manifest.json`（SHA256前缀 `3c4130d01f7d`）；`qformer-loop/taobao-transfer/recurrence-state-routes/runs/route_inner2-s20262027-20260908T120101271853Z/result.json`（SHA256前缀 `dd1f04387584`）；`qformer-loop/taobao-transfer/recurrence-state-routes/reports/final-audit.json`（SHA256前缀 `7c0fb1d9eeaa`）

<a id="phase-recurrence-context-read"></a>
## 阶段 recurrence-context-read：上下文回读、attention位置、Query容量

**方法**：中间context回读；attention中间/末尾/两处；q4/q6；入口1层。

**归因与边界**：额外回读与attention迁移未提升；ctx_q4宏AUC0.828472565、总参数51.724M，前向均值较q8少约9.8%。总参数下降不等于活跃计算同比下降。

**历史后续假设**：q4/q8固定两轮，检查局部容量及query信息来源。

<a id="t083"></a>
### T083 · `ctx_read-s20262027-20260908T133655778759Z`

- **方法**：{"attention_position":"end","context_reread":true,"dense_optimizer_options":{"preconditioner_update_probability":1.0},"ffn_ratio":4.0,"model":"QFormerCross8ContextRead","num_ns_layers":2,"num_queries":8,"token_dim":384}；设置 {"model":"QFormerCross8ContextRead","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Test context reread and attention placement without prediction-to-work feedback; retain small capacity comparisons
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.825594757；macroLogloss=0.055047130；is_click AUC/Logloss=0.662809620/0.182547381；cart AUC/Logloss=0.863367689/0.017724320；fav AUC/Logloss=0.880602381/0.013292756；buy AUC/Logloss=0.895599340/0.006624063。
- **开销**：参数 81616324；训练含验证 2107.154206946492 秒；峰值CUDA 26375536128 bytes；更新 2193。
- **原记录差值**：-0.003028775；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.003028775，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-context-read](#phase-recurrence-context-read)。
- **后续假设**：q4/q8固定两轮，检查局部容量及query信息来源。
- **证据**：`qformer-loop/taobao-transfer/recurrence-context-read/runs/ctx_read-s20262027-20260908T133655778759Z/manifest.json`（SHA256前缀 `72b4e89bcf78`）；`qformer-loop/taobao-transfer/recurrence-context-read/runs/ctx_read-s20262027-20260908T133655778759Z/result.json`（SHA256前缀 `72b8b74be4f5`）；`qformer-loop/taobao-transfer/recurrence-context-read/reports/final-audit.json`（SHA256前缀 `6f4c67919508`）

<a id="t084"></a>
### T084 · `ctx_middle-s20262027-20260908T133702910097Z`

- **方法**：{"attention_position":"middle","context_reread":false,"dense_optimizer_options":{"preconditioner_update_probability":1.0},"ffn_ratio":4.0,"model":"QFormerCross8ContextRead","num_ns_layers":2,"num_queries":8,"token_dim":384}；设置 {"model":"QFormerCross8ContextRead","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Test context reread and attention placement without prediction-to-work feedback; retain small capacity comparisons
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.826099221；macroLogloss=0.055068228；is_click AUC/Logloss=0.665654847/0.182195193；cart AUC/Logloss=0.862356852/0.017905368；fav AUC/Logloss=0.879816774/0.013420875；buy AUC/Logloss=0.896568410/0.006751477。
- **开销**：参数 81616324；训练含验证 2053.297335408628 秒；峰值CUDA 25887764992 bytes；更新 2193。
- **原记录差值**：-0.002524312；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.002524312，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-context-read](#phase-recurrence-context-read)。
- **后续假设**：q4/q8固定两轮，检查局部容量及query信息来源。
- **证据**：`qformer-loop/taobao-transfer/recurrence-context-read/runs/ctx_middle-s20262027-20260908T133702910097Z/manifest.json`（SHA256前缀 `fa02b5f3671e`）；`qformer-loop/taobao-transfer/recurrence-context-read/runs/ctx_middle-s20262027-20260908T133702910097Z/result.json`（SHA256前缀 `30bcbfff4a52`）；`qformer-loop/taobao-transfer/recurrence-context-read/reports/final-audit.json`（SHA256前缀 `6f4c67919508`）

<a id="t085"></a>
### T085 · `ctx_read_middle-s20262027-20260908T133653654700Z`

- **方法**：{"attention_position":"middle","context_reread":true,"dense_optimizer_options":{"preconditioner_update_probability":1.0},"ffn_ratio":4.0,"model":"QFormerCross8ContextRead","num_ns_layers":2,"num_queries":8,"token_dim":384}；设置 {"model":"QFormerCross8ContextRead","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Test context reread and attention placement without prediction-to-work feedback; retain small capacity comparisons
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.826907969；macroLogloss=0.055046720；is_click AUC/Logloss=0.666171981/0.182188498；cart AUC/Logloss=0.862806836/0.017892621；fav AUC/Logloss=0.881402795/0.013398504；buy AUC/Logloss=0.897250261/0.006707259。
- **开销**：参数 81616324；训练含验证 2115.550459511578 秒；峰值CUDA 27030620672 bytes；更新 2193。
- **原记录差值**：-0.001715564；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.001715564，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-context-read](#phase-recurrence-context-read)。
- **后续假设**：q4/q8固定两轮，检查局部容量及query信息来源。
- **证据**：`qformer-loop/taobao-transfer/recurrence-context-read/runs/ctx_read_middle-s20262027-20260908T133653654700Z/manifest.json`（SHA256前缀 `d481e307c946`）；`qformer-loop/taobao-transfer/recurrence-context-read/runs/ctx_read_middle-s20262027-20260908T133653654700Z/result.json`（SHA256前缀 `9ed89088105f`）；`qformer-loop/taobao-transfer/recurrence-context-read/reports/final-audit.json`（SHA256前缀 `6f4c67919508`）

<a id="t086"></a>
### T086 · `ctx_both-s20262027-20260908T133659906802Z`

- **方法**：{"attention_position":"both","context_reread":false,"dense_optimizer_options":{"preconditioner_update_probability":1.0},"ffn_ratio":4.0,"model":"QFormerCross8ContextRead","num_ns_layers":2,"num_queries":8,"token_dim":384}；设置 {"model":"QFormerCross8ContextRead","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Test context reread and attention placement without prediction-to-work feedback; retain small capacity comparisons
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.826098075；macroLogloss=0.054980245；is_click AUC/Logloss=0.665512085/0.182186188；cart AUC/Logloss=0.866544946/0.017652704；fav AUC/Logloss=0.879541420/0.013361699；buy AUC/Logloss=0.892793850/0.006720387。
- **开销**：参数 81616324；训练含验证 2092.8971737101674 秒；峰值CUDA 25941659136 bytes；更新 2193。
- **原记录差值**：-0.002525457；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.002525457，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-context-read](#phase-recurrence-context-read)。
- **后续假设**：q4/q8固定两轮，检查局部容量及query信息来源。
- **证据**：`qformer-loop/taobao-transfer/recurrence-context-read/runs/ctx_both-s20262027-20260908T133659906802Z/manifest.json`（SHA256前缀 `915788a9235b`）；`qformer-loop/taobao-transfer/recurrence-context-read/runs/ctx_both-s20262027-20260908T133659906802Z/result.json`（SHA256前缀 `5868060226a8`）；`qformer-loop/taobao-transfer/recurrence-context-read/reports/final-audit.json`（SHA256前缀 `6f4c67919508`）

<a id="t087"></a>
### T087 · `ctx_read_both-s20262027-20260908T133654667253Z`

- **方法**：{"attention_position":"both","context_reread":true,"dense_optimizer_options":{"preconditioner_update_probability":1.0},"ffn_ratio":4.0,"model":"QFormerCross8ContextRead","num_ns_layers":2,"num_queries":8,"token_dim":384}；设置 {"model":"QFormerCross8ContextRead","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Test context reread and attention placement without prediction-to-work feedback; retain small capacity comparisons
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.825209237；macroLogloss=0.055017431；is_click AUC/Logloss=0.664347987/0.182283774；cart AUC/Logloss=0.865006435/0.017712812；fav AUC/Logloss=0.878577058/0.013369072；buy AUC/Logloss=0.892905468/0.006704067。
- **开销**：参数 81616324；训练含验证 2136.2801009640098 秒；峰值CUDA 27084684800 bytes；更新 2193。
- **原记录差值**：-0.003414296；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.003414296，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-context-read](#phase-recurrence-context-read)。
- **后续假设**：q4/q8固定两轮，检查局部容量及query信息来源。
- **证据**：`qformer-loop/taobao-transfer/recurrence-context-read/runs/ctx_read_both-s20262027-20260908T133654667253Z/manifest.json`（SHA256前缀 `aad492dee39b`）；`qformer-loop/taobao-transfer/recurrence-context-read/runs/ctx_read_both-s20262027-20260908T133654667253Z/result.json`（SHA256前缀 `401d5a1f7261`）；`qformer-loop/taobao-transfer/recurrence-context-read/reports/final-audit.json`（SHA256前缀 `6f4c67919508`）

<a id="t088"></a>
### T088 · `ctx_q4-s20262027-20260908T133705147096Z`

- **方法**：{"attention_position":"end","context_reread":false,"dense_optimizer_options":{"preconditioner_update_probability":1.0},"ffn_ratio":4.0,"model":"QFormerCross8ContextRead","num_ns_layers":2,"num_queries":4,"token_dim":384}；设置 {"model":"QFormerCross8ContextRead","token_dim":384,"num_queries":4,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Test context reread and attention placement without prediction-to-work feedback; retain small capacity comparisons
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.828472565；macroLogloss=0.055039500；is_click AUC/Logloss=0.665974047/0.182313645；cart AUC/Logloss=0.866714978/0.017761683；fav AUC/Logloss=0.882673911/0.013366018；buy AUC/Logloss=0.898527326/0.006716654。
- **开销**：参数 51724228；训练含验证 1989.574068389833 秒；峰值CUDA 22593982976 bytes；更新 2193。
- **原记录差值**：-0.000150967；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.000150967，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-context-read](#phase-recurrence-context-read)。
- **后续假设**：q4/q8固定两轮，检查局部容量及query信息来源。
- **证据**：`qformer-loop/taobao-transfer/recurrence-context-read/runs/ctx_q4-s20262027-20260908T133705147096Z/manifest.json`（SHA256前缀 `487a2481d19b`）；`qformer-loop/taobao-transfer/recurrence-context-read/runs/ctx_q4-s20262027-20260908T133705147096Z/result.json`（SHA256前缀 `0b069c3006f9`）；`qformer-loop/taobao-transfer/recurrence-context-read/reports/final-audit.json`（SHA256前缀 `6f4c67919508`）

<a id="t089"></a>
### T089 · `ctx_q6-s20262027-20260908T133654922910Z`

- **方法**：{"attention_position":"end","context_reread":false,"dense_optimizer_options":{"preconditioner_update_probability":1.0},"ffn_ratio":4.0,"model":"QFormerCross8ContextRead","num_ns_layers":2,"num_queries":6,"token_dim":384}；设置 {"model":"QFormerCross8ContextRead","token_dim":384,"num_queries":6,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Test context reread and attention placement without prediction-to-work feedback; retain small capacity comparisons
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.826617707；macroLogloss=0.054950324；is_click AUC/Logloss=0.665229876/0.182195143；cart AUC/Logloss=0.867361121/0.017587326；fav AUC/Logloss=0.878367855/0.013356513；buy AUC/Logloss=0.895511976/0.006662314。
- **开销**：参数 64310980；训练含验证 2018.0737756341696 秒；峰值CUDA 23864946688 bytes；更新 2193。
- **原记录差值**：-0.002005825；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.002005825，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-context-read](#phase-recurrence-context-read)。
- **后续假设**：q4/q8固定两轮，检查局部容量及query信息来源。
- **证据**：`qformer-loop/taobao-transfer/recurrence-context-read/runs/ctx_q6-s20262027-20260908T133654922910Z/manifest.json`（SHA256前缀 `ffca638e38ba`）；`qformer-loop/taobao-transfer/recurrence-context-read/runs/ctx_q6-s20262027-20260908T133654922910Z/result.json`（SHA256前缀 `7384a5ef401a`）；`qformer-loop/taobao-transfer/recurrence-context-read/reports/final-audit.json`（SHA256前缀 `6f4c67919508`）

<a id="t090"></a>
### T090 · `ctx_ns1-s20262027-20260908T133646517199Z`

- **方法**：{"attention_position":"end","context_reread":false,"dense_optimizer_options":{"preconditioner_update_probability":1.0},"ffn_ratio":4.0,"model":"QFormerCross8ContextRead","num_ns_layers":1,"num_queries":8,"token_dim":384}；设置 {"model":"QFormerCross8ContextRead","token_dim":384,"num_queries":8,"num_ns_layers":1,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Test context reread and attention placement without prediction-to-work feedback; retain small capacity comparisons
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.825621844；macroLogloss=0.055238705；is_click AUC/Logloss=0.662712394/0.183170697；cart AUC/Logloss=0.865360488/0.017729448；fav AUC/Logloss=0.880232352/0.013334688；buy AUC/Logloss=0.894182140/0.006719988。
- **开销**：参数 80728900；训练含验证 2009.7439555302262 秒；峰值CUDA 24074465792 bytes；更新 2193。
- **原记录差值**：-0.003001689；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.003001689，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-context-read](#phase-recurrence-context-read)。
- **后续假设**：q4/q8固定两轮，检查局部容量及query信息来源。
- **证据**：`qformer-loop/taobao-transfer/recurrence-context-read/runs/ctx_ns1-s20262027-20260908T133646517199Z/manifest.json`（SHA256前缀 `18e2f180006b`）；`qformer-loop/taobao-transfer/recurrence-context-read/runs/ctx_ns1-s20262027-20260908T133646517199Z/result.json`（SHA256前缀 `e1b0163dba6a`）；`qformer-loop/taobao-transfer/recurrence-context-read/reports/final-audit.json`（SHA256前缀 `6f4c67919508`）

<a id="phase-recurrence-q4-capacity"></a>
## 阶段 recurrence-q4-capacity：小模型轮数与局部容量

**方法**：q4一轮、post FFN2/8、FFN2、入口3层及q8 FFN控制。

**归因与边界**：q4两轮比一轮+0.001960375；其余7个容量改动未超过各自父模型。

**历史后续假设**：优先改善第二轮条件形成，避免相邻容量标量扫参。

<a id="t091"></a>
### T091 · `taobao-q4-r1-r2-paired-bootstrap`

- **方法**：500 paired user resamples of matched R1 and R2 predictions
- **原假设**：Second shared work/prediction cycle benefits the four-query no-feedback candidate
- **状态／预算**：原记录 `supports_retaining_two_cycles_conditional_on_checkpoints`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "q4_r1",
  "candidate": "ctx_q4",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "delta_auc": {
    "is_click": 5.341478532294719e-05,
    "cart": 0.0027787112829202876,
    "fav": 0.0010524773672074828,
    "buy": 0.003956897505808188
  },
  "ci95_auc": {
    "is_click": [
      -0.0005595859446540397,
      0.0006640344557538225
    ],
    "cart": [
      0.0010717410411633988,
      0.004443218618955511
    ],
    "fav": [
      -0.0007056328773794834,
      0.0027809485381162236
    ],
    "buy": [
      0.0015420398270702019,
      0.006444152831939763
    ]
  },
  "delta_macro_auc": 0.001960375235314671,
  "ci95_macro_auc": [
    0.0009976644527789867,
    0.002777110216781185
  ],
  "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance or correct for candidate selection; correlated heads share bootstrap user weights"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-q4-capacity](#phase-recurrence-q4-capacity)。
- **后续假设**：优先改善第二轮条件形成，避免相邻容量标量扫参。
- **证据**：`qformer-loop/taobao-transfer/recurrence-q4-capacity/reports/paired-bootstrap-q4-depth.json`（SHA256前缀 `35e50dcdc377`）；`qformer-loop/taobao-transfer/recurrence-q4-capacity/reports/q4-depth-identities.json`（SHA256前缀 `668d285f7f93`）

<a id="t092"></a>
### T092 · `q4_r1-s20262027-20260908T150233962301Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"ffn_ratio":4.0,"model":"QFormerCross8NoFeedback","num_ns_layers":2,"num_queries":4,"token_dim":384,"work_steps":1}；设置 {"model":"QFormerCross8NoFeedback","token_dim":384,"num_queries":4,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":1,"entry_weight":0.5}
- **原假设**：Matched four-query single-cycle ablation and capacity allocation for two-cycle no-feedback recurrence
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.826512190；macroLogloss=0.055000786；is_click AUC/Logloss=0.665920632/0.182150275；cart AUC/Logloss=0.863936267/0.017758028；fav AUC/Logloss=0.881621434/0.013371428；buy AUC/Logloss=0.894570428/0.006723412。
- **开销**：参数 51724228；训练含验证 1849.4167940020561 秒；峰值CUDA 18028340224 bytes；更新 2193。
- **原记录差值**：-0.002111342；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.002111342，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-q4-capacity](#phase-recurrence-q4-capacity)。
- **后续假设**：优先改善第二轮条件形成，避免相邻容量标量扫参。
- **证据**：`qformer-loop/taobao-transfer/recurrence-q4-capacity/runs/q4_r1-s20262027-20260908T150233962301Z/manifest.json`（SHA256前缀 `d6ca7226fde8`）；`qformer-loop/taobao-transfer/recurrence-q4-capacity/runs/q4_r1-s20262027-20260908T150233962301Z/result.json`（SHA256前缀 `8e9cc239863c`）；`qformer-loop/taobao-transfer/recurrence-q4-capacity/reports/final-audit.json`（SHA256前缀 `6f4c67919508`）

<a id="t093"></a>
### T093 · `q4_postffn2-s20262027-20260908T150326065288Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"ffn_ratio":4.0,"model":"QFormerCross8NoFeedback","num_ns_layers":2,"num_queries":4,"token_dim":384,"unified_ffn_ratio":2.0,"work_steps":2}；设置 {"model":"QFormerCross8NoFeedback","token_dim":384,"num_queries":4,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Matched four-query single-cycle ablation and capacity allocation for two-cycle no-feedback recurrence
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.826287816；macroLogloss=0.055028482；is_click AUC/Logloss=0.665512915/0.182276116；cart AUC/Logloss=0.864858076/0.017778605；fav AUC/Logloss=0.879413134/0.013361835；buy AUC/Logloss=0.895367138/0.006697374。
- **开销**：参数 51133636；训练含验证 2028.115143969655 秒；峰值CUDA 22582057472 bytes；更新 2193。
- **原记录差值**：-0.002335717；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.002335717，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-q4-capacity](#phase-recurrence-q4-capacity)。
- **后续假设**：优先改善第二轮条件形成，避免相邻容量标量扫参。
- **证据**：`qformer-loop/taobao-transfer/recurrence-q4-capacity/runs/q4_postffn2-s20262027-20260908T150326065288Z/manifest.json`（SHA256前缀 `29068462c7b1`）；`qformer-loop/taobao-transfer/recurrence-q4-capacity/runs/q4_postffn2-s20262027-20260908T150326065288Z/result.json`（SHA256前缀 `965f9359f631`）；`qformer-loop/taobao-transfer/recurrence-q4-capacity/reports/final-audit.json`（SHA256前缀 `6f4c67919508`）

<a id="t094"></a>
### T094 · `q4_postffn8-s20262027-20260908T150329037662Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"ffn_ratio":4.0,"model":"QFormerCross8NoFeedback","num_ns_layers":2,"num_queries":4,"token_dim":384,"unified_ffn_ratio":8.0,"work_steps":2}；设置 {"model":"QFormerCross8NoFeedback","token_dim":384,"num_queries":4,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Matched four-query single-cycle ablation and capacity allocation for two-cycle no-feedback recurrence
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.826755327；macroLogloss=0.054985699；is_click AUC/Logloss=0.664896542/0.182275336；cart AUC/Logloss=0.866451023/0.017649115；fav AUC/Logloss=0.879416137/0.013358138；buy AUC/Logloss=0.896257606/0.006660209。
- **开销**：参数 52905412；训练含验证 2021.0460541918874 秒；峰值CUDA 22593117696 bytes；更新 2193。
- **原记录差值**：-0.001868206；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.001868206，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-q4-capacity](#phase-recurrence-q4-capacity)。
- **后续假设**：优先改善第二轮条件形成，避免相邻容量标量扫参。
- **证据**：`qformer-loop/taobao-transfer/recurrence-q4-capacity/runs/q4_postffn8-s20262027-20260908T150329037662Z/manifest.json`（SHA256前缀 `49670f7c345f`）；`qformer-loop/taobao-transfer/recurrence-q4-capacity/runs/q4_postffn8-s20262027-20260908T150329037662Z/result.json`（SHA256前缀 `4a95571be4dc`）；`qformer-loop/taobao-transfer/recurrence-q4-capacity/reports/final-audit.json`（SHA256前缀 `6f4c67919508`）

<a id="t095"></a>
### T095 · `q4_ffn2-s20262027-20260908T150239241335Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"ffn_ratio":2.0,"model":"QFormerCross8NoFeedback","num_ns_layers":2,"num_queries":4,"token_dim":384,"work_steps":2}；设置 {"model":"QFormerCross8NoFeedback","token_dim":384,"num_queries":4,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Matched four-query single-cycle ablation and capacity allocation for two-cycle no-feedback recurrence
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.826136360；macroLogloss=0.055029943；is_click AUC/Logloss=0.664489379/0.182281043；cart AUC/Logloss=0.865021340/0.017753689；fav AUC/Logloss=0.881273215/0.013348845；buy AUC/Logloss=0.893761504/0.006736194。
- **开销**：参数 50543044；训练含验证 1981.4318867474794 秒；峰值CUDA 22467556352 bytes；更新 2193。
- **原记录差值**：-0.002487173；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.002487173，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-q4-capacity](#phase-recurrence-q4-capacity)。
- **后续假设**：优先改善第二轮条件形成，避免相邻容量标量扫参。
- **证据**：`qformer-loop/taobao-transfer/recurrence-q4-capacity/runs/q4_ffn2-s20262027-20260908T150239241335Z/manifest.json`（SHA256前缀 `6b12fe24fa5f`）；`qformer-loop/taobao-transfer/recurrence-q4-capacity/runs/q4_ffn2-s20262027-20260908T150239241335Z/result.json`（SHA256前缀 `4103c299757f`）；`qformer-loop/taobao-transfer/recurrence-q4-capacity/reports/final-audit.json`（SHA256前缀 `6f4c67919508`）

<a id="t096"></a>
### T096 · `q4_ns3-s20262027-20260908T150248111717Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"ffn_ratio":4.0,"model":"QFormerCross8NoFeedback","num_ns_layers":3,"num_queries":4,"token_dim":384,"work_steps":2}；设置 {"model":"QFormerCross8NoFeedback","token_dim":384,"num_queries":4,"num_ns_layers":3,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Matched four-query single-cycle ablation and capacity allocation for two-cycle no-feedback recurrence
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.827051065；macroLogloss=0.054915510；is_click AUC/Logloss=0.667160485/0.181983172；cart AUC/Logloss=0.865200711/0.017662080；fav AUC/Logloss=0.880729533/0.013333110；buy AUC/Logloss=0.895113531/0.006683678。
- **开销**：参数 52611652；训练含验证 2055.0565112680197 秒；峰值CUDA 23522255872 bytes；更新 2193。
- **原记录差值**：-0.001572467；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.001572467，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-q4-capacity](#phase-recurrence-q4-capacity)。
- **后续假设**：优先改善第二轮条件形成，避免相邻容量标量扫参。
- **证据**：`qformer-loop/taobao-transfer/recurrence-q4-capacity/runs/q4_ns3-s20262027-20260908T150248111717Z/manifest.json`（SHA256前缀 `b5211aec312b`）；`qformer-loop/taobao-transfer/recurrence-q4-capacity/runs/q4_ns3-s20262027-20260908T150248111717Z/result.json`（SHA256前缀 `5f621911d897`）；`qformer-loop/taobao-transfer/recurrence-q4-capacity/reports/final-audit.json`（SHA256前缀 `6f4c67919508`）

<a id="t097"></a>
### T097 · `q8_postffn2-s20262027-20260908T150330839374Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"ffn_ratio":4.0,"model":"QFormerCross8NoFeedback","num_ns_layers":2,"num_queries":8,"token_dim":384,"unified_ffn_ratio":2.0,"work_steps":2}；设置 {"model":"QFormerCross8NoFeedback","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Matched four-query single-cycle ablation and capacity allocation for two-cycle no-feedback recurrence
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.826365356；macroLogloss=0.055167069；is_click AUC/Logloss=0.662569544/0.182934419；cart AUC/Logloss=0.866715536/0.017717451；fav AUC/Logloss=0.879619591/0.013349093；buy AUC/Logloss=0.896556751/0.006667312。
- **开销**：参数 81025732；训练含验证 2077.334398470819 秒；峰值CUDA 25166422528 bytes；更新 2193。
- **原记录差值**：-0.002258177；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.002258177，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-q4-capacity](#phase-recurrence-q4-capacity)。
- **后续假设**：优先改善第二轮条件形成，避免相邻容量标量扫参。
- **证据**：`qformer-loop/taobao-transfer/recurrence-q4-capacity/runs/q8_postffn2-s20262027-20260908T150330839374Z/manifest.json`（SHA256前缀 `481816dba43e`）；`qformer-loop/taobao-transfer/recurrence-q4-capacity/runs/q8_postffn2-s20262027-20260908T150330839374Z/result.json`（SHA256前缀 `fda2f4150bc1`）；`qformer-loop/taobao-transfer/recurrence-q4-capacity/reports/final-audit.json`（SHA256前缀 `6f4c67919508`）

<a id="t098"></a>
### T098 · `q8_postffn8-s20262027-20260908T150325024497Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"ffn_ratio":4.0,"model":"QFormerCross8NoFeedback","num_ns_layers":2,"num_queries":8,"token_dim":384,"unified_ffn_ratio":8.0,"work_steps":2}；设置 {"model":"QFormerCross8NoFeedback","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Matched four-query single-cycle ablation and capacity allocation for two-cycle no-feedback recurrence
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.825616116；macroLogloss=0.055096079；is_click AUC/Logloss=0.663593859/0.182448965；cart AUC/Logloss=0.866290846/0.017750910；fav AUC/Logloss=0.878883172/0.013431092；buy AUC/Logloss=0.893696586/0.006753350。
- **开销**：参数 82797508；训练含验证 2084.862995952368 秒；峰值CUDA 25635168256 bytes；更新 2193。
- **原记录差值**：-0.003007417；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.003007417，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-q4-capacity](#phase-recurrence-q4-capacity)。
- **后续假设**：优先改善第二轮条件形成，避免相邻容量标量扫参。
- **证据**：`qformer-loop/taobao-transfer/recurrence-q4-capacity/runs/q8_postffn8-s20262027-20260908T150325024497Z/manifest.json`（SHA256前缀 `cf613919bc62`）；`qformer-loop/taobao-transfer/recurrence-q4-capacity/runs/q8_postffn8-s20262027-20260908T150325024497Z/result.json`（SHA256前缀 `b7ea74fc398e`）；`qformer-loop/taobao-transfer/recurrence-q4-capacity/reports/final-audit.json`（SHA256前缀 `6f4c67919508`）

<a id="t099"></a>
### T099 · `q8_ffn2-s20262027-20260908T150232883029Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"ffn_ratio":2.0,"model":"QFormerCross8NoFeedback","num_ns_layers":2,"num_queries":8,"token_dim":384,"work_steps":2}；设置 {"model":"QFormerCross8NoFeedback","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Matched four-query single-cycle ablation and capacity allocation for two-cycle no-feedback recurrence
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.826450437；macroLogloss=0.055308669；is_click AUC/Logloss=0.664348236/0.183349519；cart AUC/Logloss=0.865317276/0.017747884；fav AUC/Logloss=0.880882327/0.013352511；buy AUC/Logloss=0.895253910/0.006784761。
- **开销**：参数 80435140；训练含验证 2045.3509408980608 秒；峰值CUDA 24951221760 bytes；更新 2193。
- **原记录差值**：-0.002173095；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.002173095，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-q4-capacity](#phase-recurrence-q4-capacity)。
- **后续假设**：优先改善第二轮条件形成，避免相邻容量标量扫参。
- **证据**：`qformer-loop/taobao-transfer/recurrence-q4-capacity/runs/q8_ffn2-s20262027-20260908T150232883029Z/manifest.json`（SHA256前缀 `b2616938b525`）；`qformer-loop/taobao-transfer/recurrence-q4-capacity/runs/q8_ffn2-s20262027-20260908T150232883029Z/result.json`（SHA256前缀 `c19d928b1490`）；`qformer-loop/taobao-transfer/recurrence-q4-capacity/reports/final-audit.json`（SHA256前缀 `6f4c67919508`）

<a id="phase-recurrence-memory-sources"></a>
## 阶段 recurrence-memory-sources：第二轮记忆来源拼接

**方法**：q4/q8分别拼接入口、历史、预测/工作状态等K/V来源。

**归因与边界**：8组均低于对应原循环；增加memory来源没有自动提升检索。

**历史后续假设**：保留原memory，检验入口/工作/均值/差值条件修正。

<a id="t102"></a>
### T102 · `memory_q4_work_entry-s20262027-20260908T155435293535Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"ffn_ratio":4.0,"memory_source":"work_entry","model":"QFormerCross8MemorySources","num_ns_layers":2,"num_queries":4,"token_dim":384,"work_steps":2}；设置 {"model":"QFormerCross8MemorySources","token_dim":384,"num_queries":4,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Change second-cycle memory sources while preserving no prediction feedback
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.825296054；macroLogloss=0.055115250；is_click AUC/Logloss=0.663699174/0.182482475；cart AUC/Logloss=0.863577461/0.017855484；fav AUC/Logloss=0.878871555/0.013382662；buy AUC/Logloss=0.895036026/0.006740379。
- **开销**：参数 51724228；训练含验证 1996.6715122014284 秒；峰值CUDA 24159483904 bytes；更新 2193。
- **原记录差值**：-0.003327478；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.003327478，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-memory-sources](#phase-recurrence-memory-sources)。
- **后续假设**：保留原memory，检验入口/工作/均值/差值条件修正。
- **证据**：`qformer-loop/taobao-transfer/recurrence-memory-sources/runs/memory_q4_work_entry-s20262027-20260908T155435293535Z/manifest.json`（SHA256前缀 `79d4333df3e2`）；`qformer-loop/taobao-transfer/recurrence-memory-sources/runs/memory_q4_work_entry-s20262027-20260908T155435293535Z/result.json`（SHA256前缀 `c4148123c224`）；`qformer-loop/taobao-transfer/recurrence-memory-sources/reports/final-audit.json`（SHA256前缀 `6f4c67919508`）

<a id="t103"></a>
### T103 · `memory_q4_prediction_history-s20262027-20260908T155442368353Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"ffn_ratio":4.0,"memory_source":"prediction_history","model":"QFormerCross8MemorySources","num_ns_layers":2,"num_queries":4,"token_dim":384,"work_steps":2}；设置 {"model":"QFormerCross8MemorySources","token_dim":384,"num_queries":4,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Change second-cycle memory sources while preserving no prediction feedback
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.827172217；macroLogloss=0.054954400；is_click AUC/Logloss=0.665831275/0.182070883；cart AUC/Logloss=0.866057826/0.017726815；fav AUC/Logloss=0.881329267/0.013323462；buy AUC/Logloss=0.895470498/0.006696441。
- **开销**：参数 51724228；训练含验证 2107.883416071534 秒；峰值CUDA 28192836608 bytes；更新 2193。
- **原记录差值**：-0.001451316；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.001451316，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-memory-sources](#phase-recurrence-memory-sources)。
- **后续假设**：保留原memory，检验入口/工作/均值/差值条件修正。
- **证据**：`qformer-loop/taobao-transfer/recurrence-memory-sources/runs/memory_q4_prediction_history-s20262027-20260908T155442368353Z/manifest.json`（SHA256前缀 `82ccbb0a3366`）；`qformer-loop/taobao-transfer/recurrence-memory-sources/runs/memory_q4_prediction_history-s20262027-20260908T155442368353Z/result.json`（SHA256前缀 `41ff32648baa`）；`qformer-loop/taobao-transfer/recurrence-memory-sources/reports/final-audit.json`（SHA256前缀 `6f4c67919508`）

<a id="t104"></a>
### T104 · `memory_q4_prediction_states-s20262027-20260908T155437780821Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"ffn_ratio":4.0,"memory_source":"prediction_states","model":"QFormerCross8MemorySources","num_ns_layers":2,"num_queries":4,"token_dim":384,"work_steps":2}；设置 {"model":"QFormerCross8MemorySources","token_dim":384,"num_queries":4,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Change second-cycle memory sources while preserving no prediction feedback
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.826790564；macroLogloss=0.055093456；is_click AUC/Logloss=0.663729468/0.182523313；cart AUC/Logloss=0.866686560/0.017746490；fav AUC/Logloss=0.878724575/0.013405768；buy AUC/Logloss=0.898021653/0.006698255。
- **开销**：参数 51724228；训练含验证 2002.945330105722 秒；峰值CUDA 22592967680 bytes；更新 2193。
- **原记录差值**：-0.001832968；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.001832968，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-memory-sources](#phase-recurrence-memory-sources)。
- **后续假设**：保留原memory，检验入口/工作/均值/差值条件修正。
- **证据**：`qformer-loop/taobao-transfer/recurrence-memory-sources/runs/memory_q4_prediction_states-s20262027-20260908T155437780821Z/manifest.json`（SHA256前缀 `20f6e65cbeb8`）；`qformer-loop/taobao-transfer/recurrence-memory-sources/runs/memory_q4_prediction_states-s20262027-20260908T155437780821Z/result.json`（SHA256前缀 `eb57a9d2de92`）；`qformer-loop/taobao-transfer/recurrence-memory-sources/reports/final-audit.json`（SHA256前缀 `6f4c67919508`）

<a id="t105"></a>
### T105 · `memory_q4_prediction_entry-s20262027-20260908T155428499084Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"ffn_ratio":4.0,"memory_source":"prediction_entry","model":"QFormerCross8MemorySources","num_ns_layers":2,"num_queries":4,"token_dim":384,"work_steps":2}；设置 {"model":"QFormerCross8MemorySources","token_dim":384,"num_queries":4,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Change second-cycle memory sources while preserving no prediction feedback
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.826079364；macroLogloss=0.054973886；is_click AUC/Logloss=0.666631104/0.182190618；cart AUC/Logloss=0.864492093/0.017682107；fav AUC/Logloss=0.879053389/0.013326627；buy AUC/Logloss=0.894140871/0.006696194。
- **开销**：参数 51724228；训练含验证 1992.5887530893087 秒；峰值CUDA 22594360832 bytes；更新 2193。
- **原记录差值**：-0.002544168；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.002544168，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-memory-sources](#phase-recurrence-memory-sources)。
- **后续假设**：保留原memory，检验入口/工作/均值/差值条件修正。
- **证据**：`qformer-loop/taobao-transfer/recurrence-memory-sources/runs/memory_q4_prediction_entry-s20262027-20260908T155428499084Z/manifest.json`（SHA256前缀 `c4900babe7b8`）；`qformer-loop/taobao-transfer/recurrence-memory-sources/runs/memory_q4_prediction_entry-s20262027-20260908T155428499084Z/result.json`（SHA256前缀 `75b70423942b`）；`qformer-loop/taobao-transfer/recurrence-memory-sources/reports/final-audit.json`（SHA256前缀 `6f4c67919508`）

<a id="t106"></a>
### T106 · `memory_q8_work_entry-s20262027-20260908T155431113128Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"ffn_ratio":4.0,"memory_source":"work_entry","model":"QFormerCross8MemorySources","num_ns_layers":2,"num_queries":8,"token_dim":384,"work_steps":2}；设置 {"model":"QFormerCross8MemorySources","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Change second-cycle memory sources while preserving no prediction feedback
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.824274325；macroLogloss=0.055208992；is_click AUC/Logloss=0.660746991/0.182806991；cart AUC/Logloss=0.864776035/0.017824144；fav AUC/Logloss=0.878914569/0.013415635；buy AUC/Logloss=0.892659706/0.006789198。
- **开销**：参数 81616324；训练含验证 2084.9270089119673 秒；峰值CUDA 27043084800 bytes；更新 2193。
- **原记录差值**：-0.004349208；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.004349208，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-memory-sources](#phase-recurrence-memory-sources)。
- **后续假设**：保留原memory，检验入口/工作/均值/差值条件修正。
- **证据**：`qformer-loop/taobao-transfer/recurrence-memory-sources/runs/memory_q8_work_entry-s20262027-20260908T155431113128Z/manifest.json`（SHA256前缀 `5c71eef2f44a`）；`qformer-loop/taobao-transfer/recurrence-memory-sources/runs/memory_q8_work_entry-s20262027-20260908T155431113128Z/result.json`（SHA256前缀 `b65ecb9dc8be`）；`qformer-loop/taobao-transfer/recurrence-memory-sources/reports/final-audit.json`（SHA256前缀 `6f4c67919508`）

<a id="t107"></a>
### T107 · `memory_q8_prediction_history-s20262027-20260908T155431290001Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"ffn_ratio":4.0,"memory_source":"prediction_history","model":"QFormerCross8MemorySources","num_ns_layers":2,"num_queries":8,"token_dim":384,"work_steps":2}；设置 {"model":"QFormerCross8MemorySources","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Change second-cycle memory sources while preserving no prediction feedback
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.824427801；macroLogloss=0.055207050；is_click AUC/Logloss=0.661518878/0.182646847；cart AUC/Logloss=0.862649866/0.017963957；fav AUC/Logloss=0.879563985/0.013411156；buy AUC/Logloss=0.893978475/0.006806241。
- **开销**：参数 81616324；训练含验证 2172.510194569826 秒；峰值CUDA 31305713152 bytes；更新 2193。
- **原记录差值**：-0.004195732；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.004195732，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-memory-sources](#phase-recurrence-memory-sources)。
- **后续假设**：保留原memory，检验入口/工作/均值/差值条件修正。
- **证据**：`qformer-loop/taobao-transfer/recurrence-memory-sources/runs/memory_q8_prediction_history-s20262027-20260908T155431290001Z/manifest.json`（SHA256前缀 `34269a9dddac`）；`qformer-loop/taobao-transfer/recurrence-memory-sources/runs/memory_q8_prediction_history-s20262027-20260908T155431290001Z/result.json`（SHA256前缀 `d45903550027`）；`qformer-loop/taobao-transfer/recurrence-memory-sources/reports/final-audit.json`（SHA256前缀 `6f4c67919508`）

<a id="t108"></a>
### T108 · `memory_q8_prediction_states-s20262027-20260908T155422308090Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"ffn_ratio":4.0,"memory_source":"prediction_states","model":"QFormerCross8MemorySources","num_ns_layers":2,"num_queries":8,"token_dim":384,"work_steps":2}；设置 {"model":"QFormerCross8MemorySources","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Change second-cycle memory sources while preserving no prediction feedback
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.822584359；macroLogloss=0.055107429；is_click AUC/Logloss=0.662922124/0.182557770；cart AUC/Logloss=0.861982902/0.017728860；fav AUC/Logloss=0.875974185/0.013401330；buy AUC/Logloss=0.889458225/0.006741756。
- **开销**：参数 81616324；训练含验证 2074.8612106218934 秒；峰值CUDA 25536383488 bytes；更新 2193。
- **原记录差值**：-0.006039174；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.006039174，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-memory-sources](#phase-recurrence-memory-sources)。
- **后续假设**：保留原memory，检验入口/工作/均值/差值条件修正。
- **证据**：`qformer-loop/taobao-transfer/recurrence-memory-sources/runs/memory_q8_prediction_states-s20262027-20260908T155422308090Z/manifest.json`（SHA256前缀 `7e1813f487ad`）；`qformer-loop/taobao-transfer/recurrence-memory-sources/runs/memory_q8_prediction_states-s20262027-20260908T155422308090Z/result.json`（SHA256前缀 `68237e0edf36`）；`qformer-loop/taobao-transfer/recurrence-memory-sources/reports/final-audit.json`（SHA256前缀 `6f4c67919508`）

<a id="t109"></a>
### T109 · `memory_q8_prediction_entry-s20262027-20260908T155429941953Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"ffn_ratio":4.0,"memory_source":"prediction_entry","model":"QFormerCross8MemorySources","num_ns_layers":2,"num_queries":8,"token_dim":384,"work_steps":2}；设置 {"model":"QFormerCross8MemorySources","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Change second-cycle memory sources while preserving no prediction feedback
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193}。
- **结果**：macroAUC=0.825530512；macroLogloss=0.055178623；is_click AUC/Logloss=0.664686312/0.182408268；cart AUC/Logloss=0.864630851/0.018174968；fav AUC/Logloss=0.878545541/0.013413066；buy AUC/Logloss=0.894259343/0.006718192。
- **开销**：参数 81616324；训练含验证 2074.6988025456667 秒；峰值CUDA 25536648704 bytes；更新 2193。
- **原记录差值**：-0.003093021；对照 `psgd_precond_every_step-s20262027-20260907T140322750508Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.003093021，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-memory-sources](#phase-recurrence-memory-sources)。
- **后续假设**：保留原memory，检验入口/工作/均值/差值条件修正。
- **证据**：`qformer-loop/taobao-transfer/recurrence-memory-sources/runs/memory_q8_prediction_entry-s20262027-20260908T155429941953Z/manifest.json`（SHA256前缀 `67c157cc90a1`）；`qformer-loop/taobao-transfer/recurrence-memory-sources/runs/memory_q8_prediction_entry-s20262027-20260908T155429941953Z/result.json`（SHA256前缀 `f5ed1b84d91d`）；`qformer-loop/taobao-transfer/recurrence-memory-sources/reports/final-audit.json`（SHA256前缀 `6f4c67919508`）

<a id="phase-recurrence-query-correction-v2"></a>
## 阶段 recurrence-query-correction-v2：第二轮输入低秩修正

**方法**：rank16加性修正，条件取entry/work/mean/difference；初始函数兼容。

**归因与边界**：q4 work接近原q4（差−0.000006789），q8 entry在新增候选中较好；8个完成变体均未超过原循环。修正同时影响检索、交互和residual，归因未分离。

**历史后续假设**：用固定checkpoint分路干预，再用单独路线重训验证。

<a id="t110"></a>
### T110 · `query_q4_entry-s20262027-20260908T171244424318Z`

- **方法**：{"correction_rank":16,"correction_source":"entry","dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"num_queries":4,"token_dim":384,"work_steps":2}；设置 {"model":"QFormerCross8QueryCorrection","token_dim":384,"num_queries":4,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Source-conditioned correction of the second work input; retrieval, interaction, and residual contributions remain unseparated
- **状态／预算**：原记录 `discard`；{"seed":20262027,"epochs":1,"optimizer_steps":2193,"additional_training_seeds":[]}。
- **结果**：macroAUC=0.826959156；macroLogloss=0.054960507；is_click AUC/Logloss=0.666961193/0.181992501；cart AUC/Logloss=0.864868012/0.017762973；fav AUC/Logloss=0.882078120/0.013343916；buy AUC/Logloss=0.893929298/0.006742637。
- **开销**：参数 51736516；训练含验证 2054.364055901766 秒；峰值CUDA 22622219264 bytes；更新 2193。
- **原记录差值**：-0.001513409；对照 `ctx_q4`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":14.199616432189941,"mean_ms":14.215755248069764,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001513409，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-query-correction-v2](#phase-recurrence-query-correction-v2)。
- **后续假设**：用固定checkpoint分路干预，再用单独路线重训验证。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-correction-v2/runs/query_q4_entry-s20262027-20260908T171244424318Z/manifest.json`（SHA256前缀 `b3a9efa3d82f`）；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/runs/query_q4_entry-s20262027-20260908T171244424318Z/result.json`（SHA256前缀 `bd3a104bcb7a`）；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/partial-audit.json`（SHA256前缀 `1be3ad69b28e`）；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/partial-paired-bootstrap.json`（SHA256前缀 `be9b264d56d0`）；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/latency/query_q4_entry-s20262027-20260908T171244424318Z.json`（SHA256前缀 `723e910375cf`）

<a id="t111"></a>
### T111 · `query_q4_work-s20262027-20260908T171234834290Z`

- **方法**：{"correction_rank":16,"correction_source":"work","dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"num_queries":4,"token_dim":384,"work_steps":2}；设置 {"model":"QFormerCross8QueryCorrection","token_dim":384,"num_queries":4,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Source-conditioned correction of the second work input; retrieval, interaction, and residual contributions remain unseparated
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"epochs":1,"optimizer_steps":2193,"additional_training_seeds":[]}。
- **结果**：macroAUC=0.828465777；macroLogloss=0.054936458；is_click AUC/Logloss=0.667076801/0.181950254；cart AUC/Logloss=0.866921325/0.017729525；fav AUC/Logloss=0.882559751/0.013341906；buy AUC/Logloss=0.897305230/0.006724147。
- **开销**：参数 51736516；训练含验证 2052.8850394040346 秒；峰值CUDA 22622383104 bytes；更新 2193。
- **原记录差值**：-0.000006789；对照 `ctx_q4`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":14.190639972686768,"mean_ms":14.19324083328247,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000006789，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-query-correction-v2](#phase-recurrence-query-correction-v2)。
- **后续假设**：用固定checkpoint分路干预，再用单独路线重训验证。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-correction-v2/runs/query_q4_work-s20262027-20260908T171234834290Z/manifest.json`（SHA256前缀 `c9a7859832a6`）；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/runs/query_q4_work-s20262027-20260908T171234834290Z/result.json`（SHA256前缀 `ee68bc7f758b`）；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/partial-audit.json`（SHA256前缀 `1be3ad69b28e`）；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/partial-paired-bootstrap.json`（SHA256前缀 `be9b264d56d0`）；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/latency/query_q4_work-s20262027-20260908T171234834290Z.json`（SHA256前缀 `2985d2704de6`）

<a id="t112"></a>
### T112 · `query_q4_mean-s20262027-20260908T171242116320Z`

- **方法**：{"correction_rank":16,"correction_source":"mean","dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"num_queries":4,"token_dim":384,"work_steps":2}；设置 {"model":"QFormerCross8QueryCorrection","token_dim":384,"num_queries":4,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Source-conditioned correction of the second work input; retrieval, interaction, and residual contributions remain unseparated
- **状态／预算**：原记录 `discard`；{"seed":20262027,"epochs":1,"optimizer_steps":2193,"additional_training_seeds":[]}。
- **结果**：macroAUC=0.826744379；macroLogloss=0.055079111；is_click AUC/Logloss=0.665675477/0.182359318；cart AUC/Logloss=0.865601046/0.017754834；fav AUC/Logloss=0.880625808/0.013385792；buy AUC/Logloss=0.895075187/0.006816500。
- **开销**：参数 51736516；训练含验证 2049.303800113499 秒；峰值CUDA 22672717824 bytes；更新 2193。
- **原记录差值**：-0.001728186；对照 `ctx_q4`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":14.188079833984375,"mean_ms":14.259200024604798,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001728186，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-query-correction-v2](#phase-recurrence-query-correction-v2)。
- **后续假设**：用固定checkpoint分路干预，再用单独路线重训验证。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-correction-v2/runs/query_q4_mean-s20262027-20260908T171242116320Z/manifest.json`（SHA256前缀 `c3ffea7d99c0`）；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/runs/query_q4_mean-s20262027-20260908T171242116320Z/result.json`（SHA256前缀 `c7d2a125d683`）；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/partial-audit.json`（SHA256前缀 `1be3ad69b28e`）；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/partial-paired-bootstrap.json`（SHA256前缀 `be9b264d56d0`）；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/latency/query_q4_mean-s20262027-20260908T171242116320Z.json`（SHA256前缀 `1176c2c74ed6`）

<a id="t113"></a>
### T113 · `query_q8_difference-s20262027-20260908T171234078895Z`

- **方法**：{"correction_rank":16,"correction_source":"difference","dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"num_queries":8,"token_dim":384,"work_steps":2}；设置 {"model":"QFormerCross8QueryCorrection","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Source-conditioned correction of the second work input; retrieval, interaction, and residual contributions remain unseparated
- **状态／预算**：原记录 `discard`；{"seed":20262027,"epochs":1,"optimizer_steps":2193,"additional_training_seeds":[]}。
- **结果**：macroAUC=0.824065983；macroLogloss=0.055085751；is_click AUC/Logloss=0.661885797/0.182596887；cart AUC/Logloss=0.863555822/0.017696034；fav AUC/Logloss=0.878400264/0.013341421；buy AUC/Logloss=0.892422048/0.006708662。
- **开销**：参数 81628612；训练含验证 2129.1541467905045 秒；峰值CUDA 25387389440 bytes；更新 2193。
- **原记录差值**：-0.004455230；对照 `explore_dual_no_feedback`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":15.777599811553955,"mean_ms":15.782925653457642,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.004455230，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-query-correction-v2](#phase-recurrence-query-correction-v2)。
- **后续假设**：用固定checkpoint分路干预，再用单独路线重训验证。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-correction-v2/runs/query_q8_difference-s20262027-20260908T171234078895Z/manifest.json`（SHA256前缀 `8c084e06ed91`）；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/runs/query_q8_difference-s20262027-20260908T171234078895Z/result.json`（SHA256前缀 `cfb37d276bc2`）；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/partial-audit.json`（SHA256前缀 `1be3ad69b28e`）；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/partial-paired-bootstrap.json`（SHA256前缀 `be9b264d56d0`）；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/latency/query_q8_difference-s20262027-20260908T171234078895Z.json`（SHA256前缀 `d51baf900543`）

<a id="t114"></a>
### T114 · `query_q8_entry-s20262027-20260908T171234804182Z`

- **方法**：{"correction_rank":16,"correction_source":"entry","dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"num_queries":8,"token_dim":384,"work_steps":2}；设置 {"model":"QFormerCross8QueryCorrection","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Source-conditioned correction of the second work input; retrieval, interaction, and residual contributions remain unseparated
- **状态／预算**：原记录 `discard`；{"seed":20262027,"epochs":1,"optimizer_steps":2193,"additional_training_seeds":[]}。
- **结果**：macroAUC=0.828221007；macroLogloss=0.054858056；is_click AUC/Logloss=0.667242811/0.182007035；cart AUC/Logloss=0.868362719/0.017548726；fav AUC/Logloss=0.881570991/0.013251830；buy AUC/Logloss=0.895707506/0.006624635。
- **开销**：参数 81628612；训练含验证 2127.3564210757613 秒；峰值CUDA 25387252224 bytes；更新 2193。
- **原记录差值**：-0.000300207；对照 `explore_dual_no_feedback`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":15.774560451507568,"mean_ms":15.807818531990051,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000300207，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-query-correction-v2](#phase-recurrence-query-correction-v2)。
- **后续假设**：用固定checkpoint分路干预，再用单独路线重训验证。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-correction-v2/runs/query_q8_entry-s20262027-20260908T171234804182Z/manifest.json`（SHA256前缀 `48ae7fc2c063`）；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/runs/query_q8_entry-s20262027-20260908T171234804182Z/result.json`（SHA256前缀 `b702777cdca6`）；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/partial-audit.json`（SHA256前缀 `1be3ad69b28e`）；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/partial-paired-bootstrap.json`（SHA256前缀 `be9b264d56d0`）；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/latency/query_q8_entry-s20262027-20260908T171234804182Z.json`（SHA256前缀 `324a9bdb323b`）

<a id="t115"></a>
### T115 · `query_q8_work-s20262027-20260908T171237243608Z`

- **方法**：{"correction_rank":16,"correction_source":"work","dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"num_queries":8,"token_dim":384,"work_steps":2}；设置 {"model":"QFormerCross8QueryCorrection","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Source-conditioned correction of the second work input; retrieval, interaction, and residual contributions remain unseparated
- **状态／预算**：原记录 `discard`；{"seed":20262027,"epochs":1,"optimizer_steps":2193,"additional_training_seeds":[]}。
- **结果**：macroAUC=0.825041226；macroLogloss=0.054976907；is_click AUC/Logloss=0.665300537/0.182195380；cart AUC/Logloss=0.863960466/0.017671388；fav AUC/Logloss=0.877673602/0.013370402；buy AUC/Logloss=0.893230299/0.006670456。
- **开销**：参数 81628612；训练含验证 2130.9525534808636 秒；峰值CUDA 25387386368 bytes；更新 2193。
- **原记录差值**：-0.003479987；对照 `explore_dual_no_feedback`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":15.74681568145752,"mean_ms":15.750293612480164,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.003479987，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-query-correction-v2](#phase-recurrence-query-correction-v2)。
- **后续假设**：用固定checkpoint分路干预，再用单独路线重训验证。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-correction-v2/runs/query_q8_work-s20262027-20260908T171237243608Z/manifest.json`（SHA256前缀 `a21f915fe1a4`）；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/runs/query_q8_work-s20262027-20260908T171237243608Z/result.json`（SHA256前缀 `5c8beb115bcb`）；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/partial-audit.json`（SHA256前缀 `1be3ad69b28e`）；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/partial-paired-bootstrap.json`（SHA256前缀 `be9b264d56d0`）；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/latency/query_q8_work-s20262027-20260908T171237243608Z.json`（SHA256前缀 `887e53d43b43`）

<a id="t116"></a>
### T116 · `query_q8_mean-s20262027-20260908T171226307157Z`

- **方法**：{"correction_rank":16,"correction_source":"mean","dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"num_queries":8,"token_dim":384,"work_steps":2}；设置 {"model":"QFormerCross8QueryCorrection","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Source-conditioned correction of the second work input; retrieval, interaction, and residual contributions remain unseparated
- **状态／预算**：原记录 `discard`；{"seed":20262027,"epochs":1,"optimizer_steps":2193,"additional_training_seeds":[]}。
- **结果**：macroAUC=0.825296760；macroLogloss=0.055023325；is_click AUC/Logloss=0.663142435/0.182461141；cart AUC/Logloss=0.864966150/0.017637812；fav AUC/Logloss=0.877957913/0.013337198；buy AUC/Logloss=0.895120543/0.006657147。
- **开销**：参数 81628612；训练含验证 2126.254412189126 秒；峰值CUDA 25387648512 bytes；更新 2193。
- **原记录差值**：-0.003224453；对照 `explore_dual_no_feedback`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":15.762815952301025,"mean_ms":15.778534436225891,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.003224453，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-query-correction-v2](#phase-recurrence-query-correction-v2)。
- **后续假设**：用固定checkpoint分路干预，再用单独路线重训验证。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-correction-v2/runs/query_q8_mean-s20262027-20260908T171226307157Z/manifest.json`（SHA256前缀 `2c00dbd581f0`）；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/runs/query_q8_mean-s20262027-20260908T171226307157Z/result.json`（SHA256前缀 `f3ba045b2899`）；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/partial-audit.json`（SHA256前缀 `1be3ad69b28e`）；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/partial-paired-bootstrap.json`（SHA256前缀 `be9b264d56d0`）；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/latency/query_q8_mean-s20262027-20260908T171226307157Z.json`（SHA256前缀 `5eb199219284`）

<a id="t117"></a>
### T117 · `taobao-query-correction-v2-query_q4_mean-query_q4_entry-paired-bootstrap`

- **方法**：500 paired user resamples; all five complete source-vs-mean pairs plus q4_work-vs-ctx_q4
- **原假设**：Conditional source and matched-reference AUC differences
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"bootstrap_seed":20260906,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "query_q4_mean",
  "candidate": "query_q4_entry",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "delta_auc": {
    "is_click": 0.0012857165312695207,
    "cart": -0.0007330336286652805,
    "fav": 0.001452312334141781,
    "buy": -0.001145888673227513
  },
  "ci95_auc": {
    "is_click": [
      0.000737043966803383,
      0.0018691352503661571
    ],
    "cart": [
      -0.002420758378441637,
      0.0009209566069001165
    ],
    "fav": [
      -0.00016775274109475933,
      0.002960641838954413
    ],
    "buy": [
      -0.003365740875313647,
      0.0010217808202261595
    ]
  },
  "delta_macro_auc": 0.00021477664087965476,
  "ci95_macro_auc": [
    -0.0005623838707544133,
    0.0009783773545280073
  ],
  "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance or correct for candidate selection; correlated heads share bootstrap user weights"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-query-correction-v2](#phase-recurrence-query-correction-v2)。
- **后续假设**：用固定checkpoint分路干预，再用单独路线重训验证。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/partial-paired-bootstrap.json`（SHA256前缀 `be9b264d56d0`）；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/partial-summary.json`（SHA256前缀 `2a5db473b048`）

<a id="t118"></a>
### T118 · `taobao-query-correction-v2-query_q4_mean-query_q4_work-paired-bootstrap`

- **方法**：500 paired user resamples; all five complete source-vs-mean pairs plus q4_work-vs-ctx_q4
- **原假设**：Conditional source and matched-reference AUC differences
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"bootstrap_seed":20260906,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "query_q4_mean",
  "candidate": "query_q4_work",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "delta_auc": {
    "is_click": 0.0014013237001535561,
    "cart": 0.0013202795631915842,
    "fav": 0.001933943103199831,
    "buy": 0.002230042684326139
  },
  "ci95_auc": {
    "is_click": [
      0.0008700551576478671,
      0.0019931821512234687
    ],
    "cart": [
      1.734806294756709e-05,
      0.00287747286338657
    ],
    "fav": [
      0.0004072628389658606,
      0.0037043213913475173
    ],
    "buy": [
      -0.0001436095969437611,
      0.004205596400103997
    ]
  },
  "delta_macro_auc": 0.0017213972627179164,
  "ci95_macro_auc": [
    0.0008525996105997538,
    0.0024866644337607356
  ],
  "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance or correct for candidate selection; correlated heads share bootstrap user weights"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-query-correction-v2](#phase-recurrence-query-correction-v2)。
- **后续假设**：用固定checkpoint分路干预，再用单独路线重训验证。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/partial-paired-bootstrap.json`（SHA256前缀 `be9b264d56d0`）；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/partial-summary.json`（SHA256前缀 `2a5db473b048`）

<a id="t119"></a>
### T119 · `taobao-query-correction-v2-query_q8_mean-query_q8_difference-paired-bootstrap`

- **方法**：500 paired user resamples; all five complete source-vs-mean pairs plus q4_work-vs-ctx_q4
- **原假设**：Conditional source and matched-reference AUC differences
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"bootstrap_seed":20260906,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "query_q8_mean",
  "candidate": "query_q8_difference",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "delta_auc": {
    "is_click": -0.0012566377620000235,
    "cart": -0.0014103274759088569,
    "fav": 0.0004423509547810678,
    "buy": -0.0026984945560012408
  },
  "ci95_auc": {
    "is_click": [
      -0.0018536652356977046,
      -0.0007269728896501824
    ],
    "cart": [
      -0.002952265736481302,
      3.4199695468764144e-05
    ],
    "fav": [
      -0.001159432599056123,
      0.0019341398672904906
    ],
    "buy": [
      -0.0048981254757008656,
      -0.0003924323233125234
    ]
  },
  "delta_macro_auc": -0.0012307772097823744,
  "ci95_macro_auc": [
    -0.0020491276973670864,
    -0.00042260563908793513
  ],
  "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance or correct for candidate selection; correlated heads share bootstrap user weights"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-query-correction-v2](#phase-recurrence-query-correction-v2)。
- **后续假设**：用固定checkpoint分路干预，再用单独路线重训验证。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/partial-paired-bootstrap.json`（SHA256前缀 `be9b264d56d0`）；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/partial-summary.json`（SHA256前缀 `2a5db473b048`）

<a id="t120"></a>
### T120 · `taobao-query-correction-v2-query_q8_mean-query_q8_entry-paired-bootstrap`

- **方法**：500 paired user resamples; all five complete source-vs-mean pairs plus q4_work-vs-ctx_q4
- **原假设**：Conditional source and matched-reference AUC differences
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"bootstrap_seed":20260906,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "query_q8_mean",
  "candidate": "query_q8_entry",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "delta_auc": {
    "is_click": 0.004100375630004427,
    "cart": 0.0033965689099487717,
    "fav": 0.003613077178069135,
    "buy": 0.0005869634721303862
  },
  "ci95_auc": {
    "is_click": [
      0.0035110922229819375,
      0.0046909147777043775
    ],
    "cart": [
      0.0019360865990669385,
      0.004938323840371894
    ],
    "fav": [
      0.002074383409158176,
      0.005270053456475373
    ],
    "buy": [
      -0.0018011947396040337,
      0.0029563215280712647
    ]
  },
  "delta_macro_auc": 0.00292424629753818,
  "ci95_macro_auc": [
    0.002119145044531312,
    0.003820982325002739
  ],
  "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance or correct for candidate selection; correlated heads share bootstrap user weights"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-query-correction-v2](#phase-recurrence-query-correction-v2)。
- **后续假设**：用固定checkpoint分路干预，再用单独路线重训验证。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/partial-paired-bootstrap.json`（SHA256前缀 `be9b264d56d0`）；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/partial-summary.json`（SHA256前缀 `2a5db473b048`）

<a id="t121"></a>
### T121 · `taobao-query-correction-v2-query_q8_mean-query_q8_work-paired-bootstrap`

- **方法**：500 paired user resamples; all five complete source-vs-mean pairs plus q4_work-vs-ctx_q4
- **原假设**：Conditional source and matched-reference AUC differences
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"bootstrap_seed":20260906,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "query_q8_mean",
  "candidate": "query_q8_work",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "delta_auc": {
    "is_click": 0.002158101721792982,
    "cart": -0.001005683161173776,
    "fav": -0.0002843113910905304,
    "buy": -0.0018902436653888133
  },
  "ci95_auc": {
    "is_click": [
      0.0016808721782525932,
      0.002664389425215474
    ],
    "cart": [
      -0.002443173179953803,
      0.0006064875405822386
    ],
    "fav": [
      -0.0018867565722344936,
      0.0013077762940695227
    ],
    "buy": [
      -0.003918165079762443,
      0.00040884172780958957
    ]
  },
  "delta_macro_auc": -0.00025553412396506214,
  "ci95_macro_auc": [
    -0.0010806146494780281,
    0.0005904822100743165
  ],
  "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance or correct for candidate selection; correlated heads share bootstrap user weights"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-query-correction-v2](#phase-recurrence-query-correction-v2)。
- **后续假设**：用固定checkpoint分路干预，再用单独路线重训验证。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/partial-paired-bootstrap.json`（SHA256前缀 `be9b264d56d0`）；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/partial-summary.json`（SHA256前缀 `2a5db473b048`）

<a id="t122"></a>
### T122 · `taobao-query-correction-v2-ctx_q4-query_q4_work-paired-bootstrap`

- **方法**：500 paired user resamples; all five complete source-vs-mean pairs plus q4_work-vs-ctx_q4
- **原假设**：Conditional source and matched-reference AUC differences
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"bootstrap_seed":20260906,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "ctx_q4",
  "candidate": "query_q4_work",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "delta_auc": {
    "is_click": 0.0011027535201866279,
    "cart": 0.00020634737945157688,
    "fav": -0.00011416036563804965,
    "buy": -0.001222095873015916
  },
  "ci95_auc": {
    "is_click": [
      0.0006414599064769434,
      0.0016085138821537086
    ],
    "cart": [
      -0.001475593284591059,
      0.0018732404727273067
    ],
    "fav": [
      -0.0016760500319913801,
      0.0014109457778608947
    ],
    "buy": [
      -0.0031892156682593624,
      0.00048077678255460297
    ]
  },
  "delta_macro_auc": -6.7888347538014315e-06,
  "ci95_macro_auc": [
    -0.0008123229695614021,
    0.0007222719546049674
  ],
  "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance or correct for candidate selection; correlated heads share bootstrap user weights"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-query-correction-v2](#phase-recurrence-query-correction-v2)。
- **后续假设**：用固定checkpoint分路干预，再用单独路线重训验证。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/partial-paired-bootstrap.json`（SHA256前缀 `be9b264d56d0`）；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/partial-summary.json`（SHA256前缀 `2a5db473b048`）

<a id="t123"></a>
### T123 · `query_q4_difference-s20262027-20260908T170705732331Z-smoke-failed`

- **方法**：Deferred zero A gradient once; second real-data PSGD step still fails
- **原假设**：Difference-conditioned second work input correction
- **状态／预算**：原记录 `crash`；{"seed":20262027,"training_updates":0,"smoke_optimizer_steps_completed":1,"additional_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "primary": {
    "name": "validation_macro_AUC"
  },
  "quality_available": false
}
```

</details>

- **归因**：这是执行或有效性问题；本条不能用作结构性能反证。修复后完整训练需用独立运行ID核对。 本阶段证据解释见[阶段 recurrence-query-correction-v2](#phase-recurrence-query-correction-v2)。
- **后续假设**：用固定checkpoint分路干预，再用单独路线重训验证。
- **失败后续门槛**：先复核同配置/初始函数/有效梯度及完整预算，再判断科学假设；不把同一次修复重复算成独立支持。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/query_q4_difference/smoke.log`；`qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/partial-audit.json`（SHA256前缀 `1be3ad69b28e`）

<a id="phase-recurrence-query-difference-retry"></a>
## 阶段 recurrence-query-difference-retry：差值条件重跑

**方法**：保留固定seed和参数，修复近零PSGD初始化后完成q4 difference训练。

**归因与边界**：必需差值重跑已完成：macroAUC0.825550580，低于mean、原q4和最佳单轮；初始化失败不是结构反证，完成结果才是。

**历史后续假设**：降低该差值加性设计优先级，保留更好对照作为路线研究起点。

<a id="t126"></a>
### T126 · `query_q4_difference-s20262027-20260908T184409455021Z`

- **方法**：Original q4 difference architecture plus scoped exact-gradient false-zero initialization compatibility; no optimizer setting or automatic scale formula change；设置 {"model":"QFormerCross8QueryCorrection","token_dim":384,"num_queries":4,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Complete the previously unmeasurable difference-conditioned second-work-input candidate under the fixed budget
- **状态／预算**：原记录 `discard`；{"seed":20262027,"epochs":1,"optimizer_steps":2193,"additional_training_seeds":[]}。
- **结果**：macroAUC=0.825550580；macroLogloss=0.055067740；is_click AUC/Logloss=0.665324689/0.182190655；cart AUC/Logloss=0.863106494/0.017837550；fav AUC/Logloss=0.879495594/0.013372587；buy AUC/Logloss=0.894275545/0.006870165。
- **开销**：参数 51736516；训练含验证 2061.4691116064787 秒；峰值CUDA 22672059392 bytes；更新 2193。
- **原记录差值**：-0.002921985；对照 `ctx_q4`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.002921985，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-query-difference-retry](#phase-recurrence-query-difference-retry)。
- **后续假设**：降低该差值加性设计优先级，保留更好对照作为路线研究起点。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-difference-retry/runs/query_q4_difference-s20262027-20260908T184409455021Z/manifest.json`（SHA256前缀 `1a225e3dcab6`）；`qformer-loop/taobao-transfer/recurrence-query-difference-retry/runs/query_q4_difference-s20262027-20260908T184409455021Z/result.json`（SHA256前缀 `be05c89b79b8`）；`qformer-loop/taobao-transfer/recurrence-query-difference-retry/reports/retry-audit.json`（SHA256前缀 `1fbdd7d62285`）；`qformer-loop/taobao-transfer/recurrence-query-difference-retry/reports/paired-bootstrap.json`（SHA256前缀 `4cbadb9b8aac`）；`qformer-loop/taobao-transfer/recurrence-query-difference-retry/reports/latency/query_q4_difference-s20262027-20260908T184409455021Z.json`（SHA256前缀 `294da653aa29`）

<a id="t127"></a>
### T127 · `taobao-query-difference-retry-query_q4_mean-query_q4_difference-paired-bootstrap`

- **方法**：500 paired user resamples; previous six bootstrap comparisons reused without re-appending
- **原假设**：Conditional AUC difference for the repaired q4 difference candidate
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"bootstrap_seed":20260906,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "query_q4_mean",
  "candidate": "query_q4_difference",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "delta_auc": {
    "is_click": -0.00035078811153266276,
    "cart": -0.00249455174625568,
    "fav": -0.0011302141469782256,
    "buy": -0.0007996417578494119
  },
  "ci95_auc": {
    "is_click": [
      -0.0009008415960933297,
      0.0002279069749941187
    ],
    "cart": [
      -0.004205905914404656,
      -0.0007017622641132436
    ],
    "fav": [
      -0.0025798012705587846,
      0.0003878690045789084
    ],
    "buy": [
      -0.003152447451549492,
      0.0012122667161488714
    ]
  },
  "delta_macro_auc": -0.0011937989406538563,
  "ci95_macro_auc": [
    -0.0020513313711588385,
    -0.00037679422165299794
  ],
  "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance or correct for candidate selection; correlated heads share bootstrap user weights; retry has the audited narrow initialization guard"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-query-difference-retry](#phase-recurrence-query-difference-retry)。
- **后续假设**：降低该差值加性设计优先级，保留更好对照作为路线研究起点。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-difference-retry/reports/paired-bootstrap.json`（SHA256前缀 `4cbadb9b8aac`）；`qformer-loop/taobao-transfer/recurrence-query-difference-retry/reports/paired-bootstrap-audit.json`（SHA256前缀 `d12004ff130a`）

<a id="t128"></a>
### T128 · `taobao-query-difference-retry-ctx_q4-query_q4_difference-paired-bootstrap`

- **方法**：500 paired user resamples; previous six bootstrap comparisons reused without re-appending
- **原假设**：Conditional AUC difference for the repaired q4 difference candidate
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"bootstrap_seed":20260906,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "ctx_q4",
  "candidate": "query_q4_difference",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "delta_auc": {
    "is_click": -0.000649358291499591,
    "cart": -0.0036084839299956872,
    "fav": -0.0031783176158161064,
    "buy": -0.004251780315191467
  },
  "ci95_auc": {
    "is_click": [
      -0.0011875563355810803,
      -8.60070613026067e-05
    ],
    "cart": [
      -0.005362704665546688,
      -0.001973386116724124
    ],
    "fav": [
      -0.004529010719271623,
      -0.001789183917923288
    ],
    "buy": [
      -0.0065061739691032435,
      -0.0020104083797157957
    ]
  },
  "delta_macro_auc": -0.002921985038125574,
  "ci95_macro_auc": [
    -0.003753409740340901,
    -0.0021092758058287773
  ],
  "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance or correct for candidate selection; correlated heads share bootstrap user weights; retry has the audited narrow initialization guard"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-query-difference-retry](#phase-recurrence-query-difference-retry)。
- **后续假设**：降低该差值加性设计优先级，保留更好对照作为路线研究起点。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-difference-retry/reports/paired-bootstrap.json`（SHA256前缀 `4cbadb9b8aac`）；`qformer-loop/taobao-transfer/recurrence-query-difference-retry/reports/paired-bootstrap-audit.json`（SHA256前缀 `d12004ff130a`）

<a id="t129"></a>
### T129 · `taobao-query-difference-retry-psgd_precond_every_step-query_q4_difference-paired-bootstrap`

- **方法**：500 paired user resamples; previous six bootstrap comparisons reused without re-appending
- **原假设**：Conditional AUC difference for the repaired q4 difference candidate
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"bootstrap_seed":20260906,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "psgd_precond_every_step",
  "candidate": "query_q4_difference",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "delta_auc": {
    "is_click": -0.002465089659410058,
    "cart": -0.003912532723411077,
    "fav": -0.002865635281870005,
    "buy": -0.0030485508615361034
  },
  "ci95_auc": {
    "is_click": [
      -0.0030891235326970775,
      -0.001737233078033338
    ],
    "cart": [
      -0.005738028271759915,
      -0.0021337252292220887
    ],
    "fav": [
      -0.004489896336556082,
      -0.0011344396593385825
    ],
    "buy": [
      -0.005630487389686914,
      -0.00035356983158002996
    ]
  },
  "delta_macro_auc": -0.003072952131556672,
  "ci95_macro_auc": [
    -0.004032821387491707,
    -0.002135923314224895
  ],
  "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance or correct for candidate selection; correlated heads share bootstrap user weights; retry has the audited narrow initialization guard"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-query-difference-retry](#phase-recurrence-query-difference-retry)。
- **后续假设**：降低该差值加性设计优先级，保留更好对照作为路线研究起点。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-difference-retry/reports/paired-bootstrap.json`（SHA256前缀 `4cbadb9b8aac`）；`qformer-loop/taobao-transfer/recurrence-query-difference-retry/reports/paired-bootstrap-audit.json`（SHA256前缀 `d12004ff130a`）

<a id="phase-recurrence-query-routes"></a>
## 阶段 recurrence-query-routes：输入修正路线隔离

**方法**：q4 work/q8 entry各只作用检索、交互、residual或attention。

**归因与边界**：8组均低于对应全路线及原循环点估计，按预定质量/成本规则不留新候选；同权重敏感性没有变成重训收益。

**历史后续假设**：改变更新形式，比较有界混合与同容量常量门。

<a id="t124"></a>
### T124 · `taobao-correction-three-routes-q4work-q8entry-5514`

- **方法**：Identity/zero/shuffle and correction-route ablations on two fixed checkpoints
- **原假设**：Separate retrieval, interaction and residual dependence of learned second-work input corrections
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "scope": "Fixed 5514-row stratified subset, 4647 selected users, CPU FP32, IPW BCE; paired bootstrap conditional on checkpoint and subset, not sampling-design uncertainty or training-seed variance. Multiple interventions and candidate selection uncorrected. No test evaluation.",
  "findings": [
    "q4_work: removing retrieval correction harms fav and buy while removing residual correction improves fav and buy, with conditional intervals excluding zero. Interaction removal remains inconclusive.",
    "q8_entry: removing retrieval correction improves buy, so q4 retrieval dependence does not replicate here; removing residual correction also improves buy.",
    "Zeroing the entire correction improves buy in both selected checkpoints. Shuffle intervals cross zero for all four tasks in both checkpoints. This does not support a general beneficial sample-specific correction claim."
  ]
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-query-routes](#phase-recurrence-query-routes)。
- **后续假设**：改变更新形式，比较有界混合与同容量常量门。
- **证据**：`qformer-loop/taobao-transfer/query-correction-diagnostics/reports/interpretation.json`（SHA256前缀 `c63c227c090e`）；`qformer-loop/taobao-transfer/query-correction-diagnostics/reports/q4-work/toolkit/perturbation_report.json`（SHA256前缀 `7e9442a5eff4`）；`qformer-loop/taobao-transfer/query-correction-diagnostics/reports/q8-entry/toolkit/perturbation_report.json`（SHA256前缀 `75e45ba25584`）

<a id="t130"></a>
### T130 · `route_q4_retrieval-s20262027-20260908T190038665636Z`

- **方法**：{"correction_rank":16,"correction_route":"retrieval_only","correction_source":"work","dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"model":"QFormerCross8QueryRoutes","num_queries":4,"work_steps":2}；设置 {"model":"QFormerCross8QueryRoutes","token_dim":384,"num_queries":4,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Test whether the learned second-query correction improves retrieval, explicit interaction, residual, or attention routes after retraining
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193,"new_training_seeds":[]}。
- **结果**：macroAUC=0.828008444；macroLogloss=0.055022604；is_click AUC/Logloss=0.666614526/0.182229665；cart AUC/Logloss=0.865355037/0.017781990；fav AUC/Logloss=0.881892581/0.013346377；buy AUC/Logloss=0.898171633/0.006732384。
- **开销**：参数 51736516；训练含验证 2057.9739333838224 秒；峰值CUDA 22696817152 bytes；更新 2193。
- **原记录差值**：-0.000457332；对照 `query_q4_work-s20262027-20260908T171234834290Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.000457332，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-query-routes](#phase-recurrence-query-routes)。
- **后续假设**：改变更新形式，比较有界混合与同容量常量门。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-routes/runs/route_q4_retrieval-s20262027-20260908T190038665636Z/manifest.json`（SHA256前缀 `c68917fb202b`）；`qformer-loop/taobao-transfer/recurrence-query-routes/runs/route_q4_retrieval-s20262027-20260908T190038665636Z/result.json`（SHA256前缀 `6f3b0c87fb31`）；`qformer-loop/taobao-transfer/recurrence-query-routes/reports/final-audit.json`（SHA256前缀 `6d34f910d60b`）；`qformer-loop/taobao-transfer/recurrence-query-routes/reports/summary.json`（SHA256前缀 `ea917954b3ed`）

<a id="t131"></a>
### T131 · `route_q4_interaction-s20262027-20260908T190043679512Z`

- **方法**：{"correction_rank":16,"correction_route":"interaction_only","correction_source":"work","dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"model":"QFormerCross8QueryRoutes","num_queries":4,"work_steps":2}；设置 {"model":"QFormerCross8QueryRoutes","token_dim":384,"num_queries":4,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Test whether the learned second-query correction improves retrieval, explicit interaction, residual, or attention routes after retraining
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193,"new_training_seeds":[]}。
- **结果**：macroAUC=0.827423329；macroLogloss=0.055002253；is_click AUC/Logloss=0.666379816/0.182023033；cart AUC/Logloss=0.865098227/0.017894353；fav AUC/Logloss=0.882035991/0.013336333；buy AUC/Logloss=0.896179281/0.006755295。
- **开销**：参数 51736516；训练含验证 2062.323479063809 秒；峰值CUDA 22697707008 bytes；更新 2193。
- **原记录差值**：-0.001042448；对照 `query_q4_work-s20262027-20260908T171234834290Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.001042448，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-query-routes](#phase-recurrence-query-routes)。
- **后续假设**：改变更新形式，比较有界混合与同容量常量门。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-routes/runs/route_q4_interaction-s20262027-20260908T190043679512Z/manifest.json`（SHA256前缀 `8a3f25cd1ff0`）；`qformer-loop/taobao-transfer/recurrence-query-routes/runs/route_q4_interaction-s20262027-20260908T190043679512Z/result.json`（SHA256前缀 `491b279715ca`）；`qformer-loop/taobao-transfer/recurrence-query-routes/reports/final-audit.json`（SHA256前缀 `6d34f910d60b`）；`qformer-loop/taobao-transfer/recurrence-query-routes/reports/summary.json`（SHA256前缀 `ea917954b3ed`）

<a id="t132"></a>
### T132 · `route_q4_residual-s20262027-20260908T190045767031Z`

- **方法**：{"correction_rank":16,"correction_route":"residual_only","correction_source":"work","dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"model":"QFormerCross8QueryRoutes","num_queries":4,"work_steps":2}；设置 {"model":"QFormerCross8QueryRoutes","token_dim":384,"num_queries":4,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Test whether the learned second-query correction improves retrieval, explicit interaction, residual, or attention routes after retraining
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193,"new_training_seeds":[]}。
- **结果**：macroAUC=0.825983751；macroLogloss=0.055000920；is_click AUC/Logloss=0.666892812/0.181992286；cart AUC/Logloss=0.863716137/0.017812280；fav AUC/Logloss=0.879450189/0.013417309；buy AUC/Logloss=0.893875868/0.006781805。
- **开销**：参数 51736516；训练含验证 2064.224739074707 秒；峰值CUDA 22697084928 bytes；更新 2193。
- **原记录差值**：-0.002482025；对照 `query_q4_work-s20262027-20260908T171234834290Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.002482025，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-query-routes](#phase-recurrence-query-routes)。
- **后续假设**：改变更新形式，比较有界混合与同容量常量门。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-routes/runs/route_q4_residual-s20262027-20260908T190045767031Z/manifest.json`（SHA256前缀 `80aee4b7c93d`）；`qformer-loop/taobao-transfer/recurrence-query-routes/runs/route_q4_residual-s20262027-20260908T190045767031Z/result.json`（SHA256前缀 `10a9f617f2f9`）；`qformer-loop/taobao-transfer/recurrence-query-routes/reports/final-audit.json`（SHA256前缀 `6d34f910d60b`）；`qformer-loop/taobao-transfer/recurrence-query-routes/reports/summary.json`（SHA256前缀 `ea917954b3ed`）

<a id="t133"></a>
### T133 · `route_q4_attention-s20262027-20260908T190044491955Z`

- **方法**：{"correction_rank":16,"correction_route":"attention_only","correction_source":"work","dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"model":"QFormerCross8QueryRoutes","num_queries":4,"work_steps":2}；设置 {"model":"QFormerCross8QueryRoutes","token_dim":384,"num_queries":4,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Test whether the learned second-query correction improves retrieval, explicit interaction, residual, or attention routes after retraining
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193,"new_training_seeds":[]}。
- **结果**：macroAUC=0.824700622；macroLogloss=0.055165965；is_click AUC/Logloss=0.660920225/0.182715296；cart AUC/Logloss=0.865079232/0.017764407；fav AUC/Logloss=0.880038757/0.013399856；buy AUC/Logloss=0.892764276/0.006784303。
- **开销**：参数 51736516；训练含验证 2064.3943313881755 秒；峰值CUDA 22696838144 bytes；更新 2193。
- **原记录差值**：-0.003765154；对照 `query_q4_work-s20262027-20260908T171234834290Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.003765154，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-query-routes](#phase-recurrence-query-routes)。
- **后续假设**：改变更新形式，比较有界混合与同容量常量门。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-routes/runs/route_q4_attention-s20262027-20260908T190044491955Z/manifest.json`（SHA256前缀 `792d1ce21d09`）；`qformer-loop/taobao-transfer/recurrence-query-routes/runs/route_q4_attention-s20262027-20260908T190044491955Z/result.json`（SHA256前缀 `de093db94d96`）；`qformer-loop/taobao-transfer/recurrence-query-routes/reports/final-audit.json`（SHA256前缀 `6d34f910d60b`）；`qformer-loop/taobao-transfer/recurrence-query-routes/reports/summary.json`（SHA256前缀 `ea917954b3ed`）

<a id="t134"></a>
### T134 · `route_q8_retrieval-s20262027-20260908T190022268468Z`

- **方法**：{"correction_rank":16,"correction_route":"retrieval_only","correction_source":"entry","dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"model":"QFormerCross8QueryRoutes","num_queries":8,"work_steps":2}；设置 {"model":"QFormerCross8QueryRoutes","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Test whether the learned second-query correction improves retrieval, explicit interaction, residual, or attention routes after retraining
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193,"new_training_seeds":[]}。
- **结果**：macroAUC=0.824528857；macroLogloss=0.054988426；is_click AUC/Logloss=0.663732641/0.182314383；cart AUC/Logloss=0.862899990/0.017679108；fav AUC/Logloss=0.877743411/0.013306721；buy AUC/Logloss=0.893739386/0.006653493。
- **开销**：参数 81628612；训练含验证 2134.450313255191 秒；峰值CUDA 25388402176 bytes；更新 2193。
- **原记录差值**：-0.003692150；对照 `query_q8_entry-s20262027-20260908T171234804182Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.003692150，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-query-routes](#phase-recurrence-query-routes)。
- **后续假设**：改变更新形式，比较有界混合与同容量常量门。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-routes/runs/route_q8_retrieval-s20262027-20260908T190022268468Z/manifest.json`（SHA256前缀 `2802d11e564c`）；`qformer-loop/taobao-transfer/recurrence-query-routes/runs/route_q8_retrieval-s20262027-20260908T190022268468Z/result.json`（SHA256前缀 `fbd0e6f1f505`）；`qformer-loop/taobao-transfer/recurrence-query-routes/reports/final-audit.json`（SHA256前缀 `6d34f910d60b`）；`qformer-loop/taobao-transfer/recurrence-query-routes/reports/summary.json`（SHA256前缀 `ea917954b3ed`）

<a id="t135"></a>
### T135 · `route_q8_interaction-s20262027-20260908T190028413046Z`

- **方法**：{"correction_rank":16,"correction_route":"interaction_only","correction_source":"entry","dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"model":"QFormerCross8QueryRoutes","num_queries":8,"work_steps":2}；设置 {"model":"QFormerCross8QueryRoutes","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Test whether the learned second-query correction improves retrieval, explicit interaction, residual, or attention routes after retraining
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193,"new_training_seeds":[]}。
- **结果**：macroAUC=0.825879782；macroLogloss=0.055031051；is_click AUC/Logloss=0.664854789/0.182282315；cart AUC/Logloss=0.863830011/0.017808109；fav AUC/Logloss=0.879495759/0.013316955；buy AUC/Logloss=0.895338571/0.006716825。
- **开销**：参数 81628612；训练含验证 2130.018462881446 秒；峰值CUDA 25387517440 bytes；更新 2193。
- **原记录差值**：-0.002341224；对照 `query_q8_entry-s20262027-20260908T171234804182Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.002341224，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-query-routes](#phase-recurrence-query-routes)。
- **后续假设**：改变更新形式，比较有界混合与同容量常量门。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-routes/runs/route_q8_interaction-s20262027-20260908T190028413046Z/manifest.json`（SHA256前缀 `44cda0bddcf9`）；`qformer-loop/taobao-transfer/recurrence-query-routes/runs/route_q8_interaction-s20262027-20260908T190028413046Z/result.json`（SHA256前缀 `bde30e5df63c`）；`qformer-loop/taobao-transfer/recurrence-query-routes/reports/final-audit.json`（SHA256前缀 `6d34f910d60b`）；`qformer-loop/taobao-transfer/recurrence-query-routes/reports/summary.json`（SHA256前缀 `ea917954b3ed`）

<a id="t136"></a>
### T136 · `route_q8_residual-s20262027-20260908T194602060181Z`

- **方法**：{"correction_rank":16,"correction_route":"residual_only","correction_source":"entry","dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"model":"QFormerCross8QueryRoutes","num_queries":8,"work_steps":2}；设置 {"model":"QFormerCross8QueryRoutes","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Test whether the learned second-query correction improves retrieval, explicit interaction, residual, or attention routes after retraining
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193,"new_training_seeds":[]}。
- **结果**：macroAUC=0.823672406；macroLogloss=0.055184573；is_click AUC/Logloss=0.662185726/0.182894437；cart AUC/Logloss=0.863943533/0.017714531；fav AUC/Logloss=0.877241549/0.013427485；buy AUC/Logloss=0.891318816/0.006701837。
- **开销**：参数 81628612；训练含验证 2115.9604328647256 秒；峰值CUDA 25387517440 bytes；更新 2193。
- **原记录差值**：-0.004548601；对照 `query_q8_entry-s20262027-20260908T171234804182Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.004548601，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-query-routes](#phase-recurrence-query-routes)。
- **后续假设**：改变更新形式，比较有界混合与同容量常量门。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-routes/runs/route_q8_residual-s20262027-20260908T194602060181Z/manifest.json`（SHA256前缀 `82b37db56083`）；`qformer-loop/taobao-transfer/recurrence-query-routes/runs/route_q8_residual-s20262027-20260908T194602060181Z/result.json`（SHA256前缀 `36bba694f1e8`）；`qformer-loop/taobao-transfer/recurrence-query-routes/reports/final-audit.json`（SHA256前缀 `6d34f910d60b`）；`qformer-loop/taobao-transfer/recurrence-query-routes/reports/summary.json`（SHA256前缀 `ea917954b3ed`）

<a id="t137"></a>
### T137 · `route_q8_attention-s20262027-20260908T190022893149Z`

- **方法**：{"correction_rank":16,"correction_route":"attention_only","correction_source":"entry","dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"model":"QFormerCross8QueryRoutes","num_queries":8,"work_steps":2}；设置 {"model":"QFormerCross8QueryRoutes","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Test whether the learned second-query correction improves retrieval, explicit interaction, residual, or attention routes after retraining
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193,"new_training_seeds":[]}。
- **结果**：macroAUC=0.825613451；macroLogloss=0.055006064；is_click AUC/Logloss=0.665303823/0.182241963；cart AUC/Logloss=0.863567293/0.017741749；fav AUC/Logloss=0.879294203/0.013377331；buy AUC/Logloss=0.894288486/0.006663212。
- **开销**：参数 81628612；训练含验证 2134.0981409549713 秒；峰值CUDA 25387124224 bytes；更新 2193。
- **原记录差值**：-0.002607555；对照 `query_q8_entry-s20262027-20260908T171234804182Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.002607555，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-query-routes](#phase-recurrence-query-routes)。
- **后续假设**：改变更新形式，比较有界混合与同容量常量门。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-routes/runs/route_q8_attention-s20262027-20260908T190022893149Z/manifest.json`（SHA256前缀 `9921f629ed40`）；`qformer-loop/taobao-transfer/recurrence-query-routes/runs/route_q8_attention-s20262027-20260908T190022893149Z/result.json`（SHA256前缀 `3ad9211559c3`）；`qformer-loop/taobao-transfer/recurrence-query-routes/reports/final-audit.json`（SHA256前缀 `6d34f910d60b`）；`qformer-loop/taobao-transfer/recurrence-query-routes/reports/summary.json`（SHA256前缀 `ea917954b3ed`）

<a id="t138"></a>
### T138 · `taobao-query-routes-query_q4_work-route_q4_retrieval-paired-bootstrap`

- **方法**：500 paired user resamples with shared task weights
- **原假设**：Quantify route-vs-all and matched baseline differences conditional on completed checkpoints
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "query_q4_work",
  "candidate": "route_q4_retrieval",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "training_updates": 0,
  "mandatory_all_route_contrast": true,
  "delta_auc": {
    "is_click": -0.0004622748162819157,
    "cart": -0.0015662881059558265,
    "fav": -0.0006671693585835747,
    "buy": 0.0008664028744558294
  },
  "ci95_auc": {
    "is_click": [
      -0.0009434891334504896,
      5.6222812030701e-05
    ],
    "cart": [
      -0.0032106638337418363,
      5.604299879227124e-05
    ],
    "fav": [
      -0.00217890406848098,
      0.0007680356772535146
    ],
    "buy": [
      -0.0009210465548232321,
      0.003152878766117981
    ]
  },
  "delta_macro_auc": -0.00045733235159151064,
  "ci95_macro_auc": [
    -0.0012380908214365501,
    0.00038367189431029393
  ],
  "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-query-routes](#phase-recurrence-query-routes)。
- **后续假设**：改变更新形式，比较有界混合与同容量常量门。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-routes/reports/paired-bootstrap.json`（SHA256前缀 `790a177fb101`）；`qformer-loop/taobao-transfer/recurrence-query-routes/reports/paired-bootstrap-identities.json`（SHA256前缀 `bbb80fbf1ba6`）

<a id="t139"></a>
### T139 · `taobao-query-routes-query_q4_work-route_q4_interaction-paired-bootstrap`

- **方法**：500 paired user resamples with shared task weights
- **原假设**：Quantify route-vs-all and matched baseline differences conditional on completed checkpoints
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "query_q4_work",
  "candidate": "route_q4_interaction",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "training_updates": 0,
  "mandatory_all_route_contrast": true,
  "delta_auc": {
    "is_click": -0.0006969844377778811,
    "cart": -0.0018230978597447711,
    "fav": -0.0005237597308971331,
    "buy": -0.0011259482668550813
  },
  "ci95_auc": {
    "is_click": [
      -0.0012221655063016535,
      -0.00018319823987017502
    ],
    "cart": [
      -0.003755365781442868,
      -0.00022164820755717453
    ],
    "fav": [
      -0.001970508552103814,
      0.0009667000274656992
    ],
    "buy": [
      -0.002913621835426747,
      0.0009297679440420327
    ]
  },
  "delta_macro_auc": -0.0010424475738187722,
  "ci95_macro_auc": [
    -0.0018051809894652972,
    -0.0003267512035853428
  ],
  "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-query-routes](#phase-recurrence-query-routes)。
- **后续假设**：改变更新形式，比较有界混合与同容量常量门。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-routes/reports/paired-bootstrap.json`（SHA256前缀 `790a177fb101`）；`qformer-loop/taobao-transfer/recurrence-query-routes/reports/paired-bootstrap-identities.json`（SHA256前缀 `bbb80fbf1ba6`）

<a id="t140"></a>
### T140 · `taobao-query-routes-query_q4_work-route_q4_residual-paired-bootstrap`

- **方法**：500 paired user resamples with shared task weights
- **原假设**：Quantify route-vs-all and matched baseline differences conditional on completed checkpoints
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "query_q4_work",
  "candidate": "route_q4_residual",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "training_updates": 0,
  "mandatory_all_route_contrast": true,
  "delta_auc": {
    "is_click": -0.00018398864549418192,
    "cart": -0.0032051886648571948,
    "fav": -0.003109561726142962,
    "buy": -0.003429361833394151
  },
  "ci95_auc": {
    "is_click": [
      -0.0006990199088502902,
      0.00031010286377100295
    ],
    "cart": [
      -0.004748143614072051,
      -0.001525712242504765
    ],
    "fav": [
      -0.004683282889441473,
      -0.0016483394934014412
    ],
    "buy": [
      -0.005645931523643777,
      -0.0013518331743412506
    ]
  },
  "delta_macro_auc": -0.0024820252174722057,
  "ci95_macro_auc": [
    -0.003322204222685762,
    -0.0016184830848244827
  ],
  "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-query-routes](#phase-recurrence-query-routes)。
- **后续假设**：改变更新形式，比较有界混合与同容量常量门。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-routes/reports/paired-bootstrap.json`（SHA256前缀 `790a177fb101`）；`qformer-loop/taobao-transfer/recurrence-query-routes/reports/paired-bootstrap-identities.json`（SHA256前缀 `bbb80fbf1ba6`）

<a id="t141"></a>
### T141 · `taobao-query-routes-query_q4_work-route_q4_attention-paired-bootstrap`

- **方法**：500 paired user resamples with shared task weights
- **原假设**：Quantify route-vs-all and matched baseline differences conditional on completed checkpoints
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "query_q4_work",
  "candidate": "route_q4_attention",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "training_updates": 0,
  "mandatory_all_route_contrast": true,
  "delta_auc": {
    "is_click": -0.00615657540659631,
    "cart": -0.001842093358151109,
    "fav": -0.002520994028748391,
    "buy": -0.004540954115300755
  },
  "ci95_auc": {
    "is_click": [
      -0.006702206593744872,
      -0.005637278903941437
    ],
    "cart": [
      -0.0035445002070031402,
      -0.0002729133207827435
    ],
    "fav": [
      -0.003916951512801156,
      -0.0010640527304267959
    ],
    "buy": [
      -0.00658802200234408,
      -0.0022281329838223563
    ]
  },
  "delta_macro_auc": -0.003765154227199252,
  "ci95_macro_auc": [
    -0.004567699385545,
    -0.0029791525489986643
  ],
  "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-query-routes](#phase-recurrence-query-routes)。
- **后续假设**：改变更新形式，比较有界混合与同容量常量门。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-routes/reports/paired-bootstrap.json`（SHA256前缀 `790a177fb101`）；`qformer-loop/taobao-transfer/recurrence-query-routes/reports/paired-bootstrap-identities.json`（SHA256前缀 `bbb80fbf1ba6`）

<a id="t142"></a>
### T142 · `taobao-query-routes-query_q8_entry-route_q8_retrieval-paired-bootstrap`

- **方法**：500 paired user resamples with shared task weights
- **原假设**：Quantify route-vs-all and matched baseline differences conditional on completed checkpoints
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "query_q8_entry",
  "candidate": "route_q8_retrieval",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "training_updates": 0,
  "mandatory_all_route_contrast": true,
  "delta_auc": {
    "is_click": -0.0035101699785703433,
    "cart": -0.005462728705010189,
    "fav": -0.0038275792846406143,
    "buy": -0.0019681202300498857
  },
  "ci95_auc": {
    "is_click": [
      -0.004106837279099121,
      -0.0029366203297663345
    ],
    "cart": [
      -0.0072662193722900035,
      -0.0039862639460661155
    ],
    "fav": [
      -0.0054317599367244285,
      -0.0023941700022121677
    ],
    "buy": [
      -0.0041958565219229835,
      0.0005787094301961365
    ]
  },
  "delta_macro_auc": -0.003692149549567758,
  "ci95_macro_auc": [
    -0.004526139306515168,
    -0.002840970183414894
  ],
  "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-query-routes](#phase-recurrence-query-routes)。
- **后续假设**：改变更新形式，比较有界混合与同容量常量门。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-routes/reports/paired-bootstrap.json`（SHA256前缀 `790a177fb101`）；`qformer-loop/taobao-transfer/recurrence-query-routes/reports/paired-bootstrap-identities.json`（SHA256前缀 `bbb80fbf1ba6`）

<a id="t143"></a>
### T143 · `taobao-query-routes-query_q8_entry-route_q8_interaction-paired-bootstrap`

- **方法**：500 paired user resamples with shared task weights
- **原假设**：Quantify route-vs-all and matched baseline differences conditional on completed checkpoints
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "query_q8_entry",
  "candidate": "route_q8_interaction",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "training_updates": 0,
  "mandatory_all_route_contrast": true,
  "delta_auc": {
    "is_click": -0.0023880215844530595,
    "cart": -0.004532707761128685,
    "fav": -0.0020752318926116065,
    "buy": -0.00036893529599957287
  },
  "ci95_auc": {
    "is_click": [
      -0.0029112057256314187,
      -0.001876213285541753
    ],
    "cart": [
      -0.006090808719728419,
      -0.0030547044557458465
    ],
    "fav": [
      -0.003871508110467586,
      -0.0006476439856097721
    ],
    "buy": [
      -0.002621559492969841,
      0.0020638901505910735
    ]
  },
  "delta_macro_auc": -0.002341224133548203,
  "ci95_macro_auc": [
    -0.003260880697020183,
    -0.00151798204853904
  ],
  "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-query-routes](#phase-recurrence-query-routes)。
- **后续假设**：改变更新形式，比较有界混合与同容量常量门。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-routes/reports/paired-bootstrap.json`（SHA256前缀 `790a177fb101`）；`qformer-loop/taobao-transfer/recurrence-query-routes/reports/paired-bootstrap-identities.json`（SHA256前缀 `bbb80fbf1ba6`）

<a id="t144"></a>
### T144 · `taobao-query-routes-query_q8_entry-route_q8_residual-paired-bootstrap`

- **方法**：500 paired user resamples with shared task weights
- **原假设**：Quantify route-vs-all and matched baseline differences conditional on completed checkpoints
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "query_q8_entry",
  "candidate": "route_q8_residual",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "training_updates": 0,
  "mandatory_all_route_contrast": true,
  "delta_auc": {
    "is_click": -0.005057084673297241,
    "cart": -0.004419185956446081,
    "fav": -0.004329441596192973,
    "buy": -0.004388690180176735
  },
  "ci95_auc": {
    "is_click": [
      -0.005668965304377477,
      -0.004459193940990561
    ],
    "cart": [
      -0.006110195812641622,
      -0.002981770901350861
    ],
    "fav": [
      -0.006056666536700475,
      -0.0026699342814827222
    ],
    "buy": [
      -0.006765286528549677,
      -0.0018239656984592215
    ]
  },
  "delta_macro_auc": -0.004548600601528285,
  "ci95_macro_auc": [
    -0.005454212685787415,
    -0.003618478271830941
  ],
  "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-query-routes](#phase-recurrence-query-routes)。
- **后续假设**：改变更新形式，比较有界混合与同容量常量门。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-routes/reports/paired-bootstrap.json`（SHA256前缀 `790a177fb101`）；`qformer-loop/taobao-transfer/recurrence-query-routes/reports/paired-bootstrap-identities.json`（SHA256前缀 `bbb80fbf1ba6`）

<a id="t145"></a>
### T145 · `taobao-query-routes-query_q8_entry-route_q8_attention-paired-bootstrap`

- **方法**：500 paired user resamples with shared task weights
- **原假设**：Quantify route-vs-all and matched baseline differences conditional on completed checkpoints
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "query_q8_entry",
  "candidate": "route_q8_attention",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "training_updates": 0,
  "mandatory_all_route_contrast": true,
  "delta_auc": {
    "is_click": -0.0019389878450730702,
    "cart": -0.004795425811231047,
    "fav": -0.002276787449856821,
    "buy": -0.0014190206212415513
  },
  "ci95_auc": {
    "is_click": [
      -0.0025169183563057166,
      -0.0014643022434374434
    ],
    "cart": [
      -0.006603904142379008,
      -0.003297326211658351
    ],
    "fav": [
      -0.003795467474348299,
      -0.0006384997869894798
    ],
    "buy": [
      -0.003908334348609812,
      0.0009401608190591796
    ]
  },
  "delta_macro_auc": -0.00260755543185065,
  "ci95_macro_auc": [
    -0.003517084885309231,
    -0.0017464702277697282
  ],
  "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-query-routes](#phase-recurrence-query-routes)。
- **后续假设**：改变更新形式，比较有界混合与同容量常量门。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-routes/reports/paired-bootstrap.json`（SHA256前缀 `790a177fb101`）；`qformer-loop/taobao-transfer/recurrence-query-routes/reports/paired-bootstrap-identities.json`（SHA256前缀 `bbb80fbf1ba6`）

<a id="t146"></a>
### T146 · `taobao-query-routes-final-interpretation`

- **方法**：Adjudicate the eight previously provisional training episodes against the prospective interpretation and latency policy; preserve original rows
- **原假设**：Route-restricted query correction should improve matching all-route/original recurrence if the diagnostic-guided architecture hypothesis transfers to retraining
- **状态／预算**：原记录 `discard`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "state": "completed_interpreted_and_consolidated",
  "new_training_runs": 8,
  "kept_new_candidates": [],
  "global_best_changed": false,
  "confidence": "One training seed; checkpoint-conditional user intervals, selection uncorrected; no equivalence or seed-robustness claim",
  "findings": [
    "All eight route-restricted retrainings have lower macroAUC point estimates than both matching all-route and original recurrence.",
    "Seven all-route conditional intervals are entirely negative; q4 retrieval-only crosses zero but misses the predeclared near-reference queue criterion. This is not proof of zero effect.",
    "Fixed-checkpoint removal of residual correction did not predict a retraining gain: attention-only retraining regressed in both selected source/query-count conditions.",
    "A second cycle remains supported by the earlier matched q4 R1/R2 result; these failures constrain rank16 additive correction sources and routes, not recurrence itself."
  ],
  "next_hypothesis": "Joint bounded mixing of entry and first-work states versus a parameter-matched sample-independent gate; channel and per-query granularity in q4/q8",
  "decisions": {
    "route_q4_retrieval": {
      "decision": "discard",
      "macro_auc": 0.8280084442398983,
      "original_ci95": null,
      "original_ci_note": "Not estimated by the frozen positive-candidate-only supplementary bootstrap policy",
      "all_route_ci95": [
        -0.0012380908214365501,
        0.00038367189431029393
      ],
      "matched_forward_mean_ms": 14.240333604812623,
      "forward_ratio_to_original": 1.0016971043170797,
      "near_reference_policy_passed": false,
      "versus_original": {
        "reference": "ctx_q4",
        "candidate": "route_q4_retrieval",
        "delta_macro_auc": -0.0004641211863453121
      },
      "versus_all_route": {
        "reference": "query_q4_work",
        "candidate": "route_q4_retrieval",
        "delta_macro_auc": -0.00045733235159151064
      },
      "versus_best_single": {
        "reference": "psgd_precond_every_step",
        "candidate": "route_q4_retrieval",
        "delta_macro_auc": -0.0006150882797764101
      }
    },
    "route_q4_interaction": {
      "decision": "discard",
      "macro_auc": 0.827423329017671,
      "original_ci95": null,
      "original_ci_note": "Not estimated by the frozen positive-candidate-only supplementary bootstrap policy",
      "all_route_ci95": [
        -0.0018051809894652972,
        -0.0003267512035853428
      ],
      "matched_forward_mean_ms": 14.237878370285035,
      "forward_ratio_to_original": 1.0015243975965102,
      "near_reference_policy_passed": false,
      "versus_original": {
        "reference": "ctx_q4",
        "candidate": "route_q4_interaction",
        "delta_macro_auc": -0.0010492364085725736
      },
      "versus_all_route": {
        "reference": "query_q4_work",
        "candidate": "route_q4_interaction",
        "delta_macro_auc": -0.0010424475738187722
      },
      "versus_best_single": {
        "reference": "psgd_precond_every_step",
        "candidate": "route_q4_interaction",
        "delta_macro_auc": -0.0012002035020036717
      }
    },
    "route_q4_residual": {
      "decision": "discard",
      "macro_auc": 0.8259837513740176,
      "original_ci95": null,
      "original_ci_note": "Not estimated by the frozen positive-candidate-only supplementary bootstrap policy",
      "all_route_ci95": [
        -0.003322204222685762,
        -0.0016184830848244827
      ],
      "matched_forward_mean_ms": 14.2431800365448,
      "forward_ratio_to_original": 1.0018973287291535,
      "near_reference_policy_passed": false,
      "versus_original": {
        "reference": "ctx_q4",
        "candidate": "route_q4_residual",
        "delta_macro_auc": -0.002488814052226007
      },
      "versus_all_route": {
        "reference": "query_q4_work",
        "candidate": "route_q4_residual",
        "delta_macro_auc": -0.0024820252174722057
      },
      "versus_best_single": {
        "reference": "psgd_precond_every_step",
        "candidate": "route_q4_residual",
        "delta_macro_auc": -0.002639781145657105
      }
    },
    "route_q4_attention": {
      "decision": "discard",
      "macro_auc": 0.8247006223642905,
      "original_ci95": null,
      "original_ci_note": "Not estimated by the frozen positive-candidate-only supplementary bootstrap policy",
      "all_route_ci95": [
        -0.004567699385545,
        -0.0029791525489986643
      ],
      "matched_forward_mean_ms": 14.2286936044693,
      "forward_ratio_to_original": 1.0008783205047247,
      "near_reference_policy_passed": false,
      "versus_original": {
        "reference": "ctx_q4",
        "candidate": "route_q4_attention",
        "delta_macro_auc": -0.0037719430619530536
      },
      "versus_all_route": {
        "reference": "query_q4_work",
        "candidate": "route_q4_attention",
        "delta_macro_auc": -0.003765154227199252
      },
      "versus_best_single": {
        "reference": "psgd_precond_every_step",
        "candidate": "route_q4_attention",
        "delta_macro_auc": -0.003922910155384152
      }
    },
    "route_q8_retrieval": {
      "decision": "discard",
      "macro_auc": 0.824528857028706,
      "original_ci95": null,
      "original_ci_note": "Not estimated by the frozen positive-candidate-only supplementary bootstrap policy",
      "all_route_ci95": [
        -0.004526139306515168,
        -0.002840970183414894
      ],
      "matched_forward_mean_ms": 15.793389582633973,
      "forward_ratio_to_original": 1.0027334521872724,
      "near_reference_policy_passed": false,
      "versus_original": {
        "reference": "explore_dual_no_feedback",
        "candidate": "route_q8_retrieval",
        "delta_macro_auc": -0.003992356345519421
      },
      "versus_all_route": {
        "reference": "query_q8_entry",
        "candidate": "route_q8_retrieval",
        "delta_macro_auc": -0.003692149549567758
      },
      "versus_best_single": {
        "reference": "psgd_precond_every_step",
        "candidate": "route_q8_retrieval",
        "delta_macro_auc": -0.004094675490968647
      }
    },
    "route_q8_interaction": {
      "decision": "discard",
      "macro_auc": 0.8258797824447256,
      "original_ci95": null,
      "original_ci_note": "Not estimated by the frozen positive-candidate-only supplementary bootstrap policy",
      "all_route_ci95": [
        -0.003260880697020183,
        -0.00151798204853904
      ],
      "matched_forward_mean_ms": 15.794897556304932,
      "forward_ratio_to_original": 1.0028291945000278,
      "near_reference_policy_passed": false,
      "versus_original": {
        "reference": "explore_dual_no_feedback",
        "candidate": "route_q8_interaction",
        "delta_macro_auc": -0.002641430929499866
      },
      "versus_all_route": {
        "reference": "query_q8_entry",
        "candidate": "route_q8_interaction",
        "delta_macro_auc": -0.002341224133548203
      },
      "versus_best_single": {
        "reference": "psgd_precond_every_step",
        "candidate": "route_q8_interaction",
        "delta_macro_auc": -0.0027437500749490917
      }
    },
    "route_q8_residual": {
      "decision": "discard",
      "macro_auc": 0.8236724059767455,
      "original_ci95": null,
      "original_ci_note": "Not estimated by the frozen positive-candidate-only supplementary bootstrap policy",
      "all_route_ci95": [
        -0.005454212685787415,
        -0.003618478271830941
      ],
      "matched_forward_mean_ms": 15.849574375152589,
      "forward_ratio_to_original": 1.0063006643216812,
      "near_reference_policy_passed": false,
      "versus_original": {
        "reference": "explore_dual_no_feedback",
        "candidate": "route_q8_residual",
        "delta_macro_auc": -0.004848807397479948
      },
      "versus_all_route": {
        "reference": "query_q8_entry",
        "candidate": "route_q8_residual",
        "delta_macro_auc": -0.004548600601528285
      },
      "versus_best_single": {
        "reference": "psgd_precond_every_step",
        "candidate": "route_q8_residual",
        "delta_macro_auc": -0.004951126542929174
      }
    },
    "route_q8_attention": {
      "decision": "discard",
      "macro_auc": 0.8256134511464231,
      "original_ci95": null,
      "original_ci_note": "Not estimated by the frozen positive-candidate-only supplementary bootstrap policy",
      "all_route_ci95": [
        -0.003517084885309231,
        -0.0017464702277697282
      ],
      "matched_forward_mean_ms": 15.837009596824647,
      "forward_ratio_to_original": 1.005502917676933,
      "near_reference_policy_passed": false,
      "versus_original": {
        "reference": "explore_dual_no_feedback",
        "candidate": "route_q8_attention",
        "delta_macro_auc": -0.002907762227802313
      },
      "versus_all_route": {
        "reference": "query_q8_entry",
        "candidate": "route_q8_attention",
        "delta_macro_auc": -0.00260755543185065
      },
      "versus_best_single": {
        "reference": "psgd_precond_every_step",
        "candidate": "route_q8_attention",
        "delta_macro_auc": -0.0030100813732515386
      }
    }
  },
  "summary_scope": "每个家族、裁决及关键对照均保留；完整逐任务训练指标见本明细的对应训练条目；其余原始统计见所列证据。"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-query-routes](#phase-recurrence-query-routes)。
- **后续假设**：改变更新形式，比较有界混合与同容量常量门。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-routes/reports/interpretation.json`（SHA256前缀 `949086b8aef3`）；`qformer-loop/taobao-transfer/recurrence-query-routes/reports/analysis-plan-review.json`（SHA256前缀 `6c104ab6e9c5`）

<a id="phase-recurrence-query-mixing"></a>
## 阶段 recurrence-query-mixing：有界门控混合

**方法**：rank16、bound0.25；q4/q8×channel/query×joint/constant。

**归因与边界**：8组均未通过预定候选规则；joint胜过弱constant不等于优于原循环；q4 fav零化/打乱门反而改善BCE。

**历史后续假设**：先用诊断定位第二次attention分配是否有独立价值，降低同类门扫参优先级。

<a id="t147"></a>
### T147 · `mix_q4_channel_joint-s20262027-20260909T035059317050Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"mixing_bound":0.25,"mixing_conditioning":"joint","mixing_granularity":"channel","mixing_rank":16,"model":"QFormerCross8QueryMixing","num_queries":4,"work_steps":2}；设置 {"model":"QFormerCross8QueryMixing","token_dim":384,"num_queries":4,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Test whether joint bounded entry/work mixing improves second-query formation beyond sample-independent mixing and the original recurrence
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193,"new_training_seeds":[]}。
- **结果**：macroAUC=0.828228691；macroLogloss=0.054967029；is_click AUC/Logloss=0.667021835/0.181978702；cart AUC/Logloss=0.866179251/0.017785876；fav AUC/Logloss=0.882540860/0.013340235；buy AUC/Logloss=0.897172818/0.006763305。
- **开销**：参数 51748804；训练含验证 2077.144857019186 秒；峰值CUDA 22827028992 bytes；更新 2193。
- **原记录差值**：-0.000243875；对照 `ctx_q4-s20262027-20260908T133705147096Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.000243875，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-query-mixing](#phase-recurrence-query-mixing)。
- **后续假设**：先用诊断定位第二次attention分配是否有独立价值，降低同类门扫参优先级。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-mixing/runs/mix_q4_channel_joint-s20262027-20260909T035059317050Z/manifest.json`（SHA256前缀 `b65a28cac3d8`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/runs/mix_q4_channel_joint-s20262027-20260909T035059317050Z/result.json`（SHA256前缀 `08e7cd5b0200`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/final-audit.json`（SHA256前缀 `2987d7a3f905`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/summary.json`（SHA256前缀 `859243590be6`）

<a id="t148"></a>
### T148 · `mix_q4_channel_constant-s20262027-20260909T035109726499Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"mixing_bound":0.25,"mixing_conditioning":"constant","mixing_granularity":"channel","mixing_rank":16,"model":"QFormerCross8QueryMixing","num_queries":4,"work_steps":2}；设置 {"model":"QFormerCross8QueryMixing","token_dim":384,"num_queries":4,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Test whether joint bounded entry/work mixing improves second-query formation beyond sample-independent mixing and the original recurrence
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193,"new_training_seeds":[]}。
- **结果**：macroAUC=0.827681921；macroLogloss=0.055043137；is_click AUC/Logloss=0.666082159/0.182177018；cart AUC/Logloss=0.866487044/0.017814803；fav AUC/Logloss=0.882211157/0.013379775；buy AUC/Logloss=0.895947326/0.006800952。
- **开销**：参数 51748804；训练含验证 2058.736142732203 秒；峰值CUDA 22776372736 bytes；更新 2193。
- **原记录差值**：-0.000790644；对照 `ctx_q4-s20262027-20260908T133705147096Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.000790644，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-query-mixing](#phase-recurrence-query-mixing)。
- **后续假设**：先用诊断定位第二次attention分配是否有独立价值，降低同类门扫参优先级。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-mixing/runs/mix_q4_channel_constant-s20262027-20260909T035109726499Z/manifest.json`（SHA256前缀 `f7182d070b06`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/runs/mix_q4_channel_constant-s20262027-20260909T035109726499Z/result.json`（SHA256前缀 `7caf1a409660`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/final-audit.json`（SHA256前缀 `2987d7a3f905`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/summary.json`（SHA256前缀 `859243590be6`）

<a id="t149"></a>
### T149 · `mix_q4_query_joint-s20262027-20260909T035110579796Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"mixing_bound":0.25,"mixing_conditioning":"joint","mixing_granularity":"query","mixing_rank":16,"model":"QFormerCross8QueryMixing","num_queries":4,"work_steps":2}；设置 {"model":"QFormerCross8QueryMixing","token_dim":384,"num_queries":4,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Test whether joint bounded entry/work mixing improves second-query formation beyond sample-independent mixing and the original recurrence
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193,"new_training_seeds":[]}。
- **结果**：macroAUC=0.828288863；macroLogloss=0.054940143；is_click AUC/Logloss=0.667175674/0.181897401；cart AUC/Logloss=0.865483602/0.017773827；fav AUC/Logloss=0.883711413/0.013359253；buy AUC/Logloss=0.896784762/0.006730089。
- **开销**：参数 51742676；训练含验证 2078.8124117031693 秒；峰值CUDA 22776266240 bytes；更新 2193。
- **原记录差值**：-0.000183703；对照 `ctx_q4-s20262027-20260908T133705147096Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.000183703，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-query-mixing](#phase-recurrence-query-mixing)。
- **后续假设**：先用诊断定位第二次attention分配是否有独立价值，降低同类门扫参优先级。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-mixing/runs/mix_q4_query_joint-s20262027-20260909T035110579796Z/manifest.json`（SHA256前缀 `411a2e31e9ee`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/runs/mix_q4_query_joint-s20262027-20260909T035110579796Z/result.json`（SHA256前缀 `a319bb78f1c1`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/final-audit.json`（SHA256前缀 `2987d7a3f905`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/summary.json`（SHA256前缀 `859243590be6`）

<a id="t150"></a>
### T150 · `mix_q4_query_constant-s20262027-20260909T035110802164Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"mixing_bound":0.25,"mixing_conditioning":"constant","mixing_granularity":"query","mixing_rank":16,"model":"QFormerCross8QueryMixing","num_queries":4,"work_steps":2}；设置 {"model":"QFormerCross8QueryMixing","token_dim":384,"num_queries":4,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Test whether joint bounded entry/work mixing improves second-query formation beyond sample-independent mixing and the original recurrence
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193,"new_training_seeds":[]}。
- **结果**：macroAUC=0.825584206；macroLogloss=0.055145818；is_click AUC/Logloss=0.663149372/0.182641625；cart AUC/Logloss=0.865145453/0.017768252；fav AUC/Logloss=0.879934982/0.013399975；buy AUC/Logloss=0.894107019/0.006773419。
- **开销**：参数 51742676；训练含验证 2069.3776799067855 秒；峰值CUDA 22725225984 bytes；更新 2193。
- **原记录差值**：-0.002888359；对照 `ctx_q4-s20262027-20260908T133705147096Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.002888359，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-query-mixing](#phase-recurrence-query-mixing)。
- **后续假设**：先用诊断定位第二次attention分配是否有独立价值，降低同类门扫参优先级。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-mixing/runs/mix_q4_query_constant-s20262027-20260909T035110802164Z/manifest.json`（SHA256前缀 `5d4a9f6b6d28`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/runs/mix_q4_query_constant-s20262027-20260909T035110802164Z/result.json`（SHA256前缀 `dda606af1532`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/final-audit.json`（SHA256前缀 `2987d7a3f905`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/summary.json`（SHA256前缀 `859243590be6`）

<a id="t151"></a>
### T151 · `mix_q8_channel_joint-s20262027-20260909T035053715005Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"mixing_bound":0.25,"mixing_conditioning":"joint","mixing_granularity":"channel","mixing_rank":16,"model":"QFormerCross8QueryMixing","num_queries":8,"work_steps":2}；设置 {"model":"QFormerCross8QueryMixing","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Test whether joint bounded entry/work mixing improves second-query formation beyond sample-independent mixing and the original recurrence
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193,"new_training_seeds":[]}。
- **结果**：macroAUC=0.823525267；macroLogloss=0.055235961；is_click AUC/Logloss=0.662203420/0.182720917；cart AUC/Logloss=0.860814065/0.017916883；fav AUC/Logloss=0.878204441/0.013458516；buy AUC/Logloss=0.892879143/0.006847529。
- **开销**：参数 81640900；训练含验证 2151.689044095576 秒；峰值CUDA 25895252992 bytes；更新 2193。
- **原记录差值**：-0.004995946；对照 `explore_dual_no_feedback-s20262027-20260908T091954887719Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.004995946，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-query-mixing](#phase-recurrence-query-mixing)。
- **后续假设**：先用诊断定位第二次attention分配是否有独立价值，降低同类门扫参优先级。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-mixing/runs/mix_q8_channel_joint-s20262027-20260909T035053715005Z/manifest.json`（SHA256前缀 `bcd56adb29aa`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/runs/mix_q8_channel_joint-s20262027-20260909T035053715005Z/result.json`（SHA256前缀 `38a5264d3b42`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/final-audit.json`（SHA256前缀 `2987d7a3f905`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/summary.json`（SHA256前缀 `859243590be6`）

<a id="t152"></a>
### T152 · `mix_q8_channel_constant-s20262027-20260909T035059340196Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"mixing_bound":0.25,"mixing_conditioning":"constant","mixing_granularity":"channel","mixing_rank":16,"model":"QFormerCross8QueryMixing","num_queries":8,"work_steps":2}；设置 {"model":"QFormerCross8QueryMixing","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Test whether joint bounded entry/work mixing improves second-query formation beyond sample-independent mixing and the original recurrence
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193,"new_training_seeds":[]}。
- **结果**：macroAUC=0.826065261；macroLogloss=0.055019656；is_click AUC/Logloss=0.665379655/0.182479162；cart AUC/Logloss=0.864444634/0.017671800；fav AUC/Logloss=0.879821806/0.013261160；buy AUC/Logloss=0.894614951/0.006666502。
- **开销**：参数 81640900；训练含验证 2135.066502198577 秒；峰值CUDA 25591297024 bytes；更新 2193。
- **原记录差值**：-0.002455952；对照 `explore_dual_no_feedback-s20262027-20260908T091954887719Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.002455952，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-query-mixing](#phase-recurrence-query-mixing)。
- **后续假设**：先用诊断定位第二次attention分配是否有独立价值，降低同类门扫参优先级。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-mixing/runs/mix_q8_channel_constant-s20262027-20260909T035059340196Z/manifest.json`（SHA256前缀 `958e95945eb3`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/runs/mix_q8_channel_constant-s20262027-20260909T035059340196Z/result.json`（SHA256前缀 `eb8372e4ab8d`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/final-audit.json`（SHA256前缀 `2987d7a3f905`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/summary.json`（SHA256前缀 `859243590be6`）

<a id="t153"></a>
### T153 · `mix_q8_query_joint-s20262027-20260909T035104953096Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"mixing_bound":0.25,"mixing_conditioning":"joint","mixing_granularity":"query","mixing_rank":16,"model":"QFormerCross8QueryMixing","num_queries":8,"work_steps":2}；设置 {"model":"QFormerCross8QueryMixing","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Test whether joint bounded entry/work mixing improves second-query formation beyond sample-independent mixing and the original recurrence
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193,"new_training_seeds":[]}。
- **结果**：macroAUC=0.825526622；macroLogloss=0.055110701；is_click AUC/Logloss=0.663682082/0.182689665；cart AUC/Logloss=0.864568845/0.017753993；fav AUC/Logloss=0.878697238/0.013331503；buy AUC/Logloss=0.895158324/0.006667644。
- **开销**：参数 81634772；训练含验证 2151.2610378563404 秒；峰值CUDA 25794298368 bytes；更新 2193。
- **原记录差值**：-0.002994591；对照 `explore_dual_no_feedback-s20262027-20260908T091954887719Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.002994591，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-query-mixing](#phase-recurrence-query-mixing)。
- **后续假设**：先用诊断定位第二次attention分配是否有独立价值，降低同类门扫参优先级。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-mixing/runs/mix_q8_query_joint-s20262027-20260909T035104953096Z/manifest.json`（SHA256前缀 `799c5274a55e`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/runs/mix_q8_query_joint-s20262027-20260909T035104953096Z/result.json`（SHA256前缀 `85227febbe30`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/final-audit.json`（SHA256前缀 `2987d7a3f905`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/summary.json`（SHA256前缀 `859243590be6`）

<a id="t154"></a>
### T154 · `mix_q8_query_constant-s20262027-20260909T035108993695Z`

- **方法**：{"dense_optimizer_options":{"preconditioner_update_probability":1.0},"entry_weight":0.5,"mixing_bound":0.25,"mixing_conditioning":"constant","mixing_granularity":"query","mixing_rank":16,"model":"QFormerCross8QueryMixing","num_queries":8,"work_steps":2}；设置 {"model":"QFormerCross8QueryMixing","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"work_steps":2,"entry_weight":0.5}
- **原假设**：Test whether joint bounded entry/work mixing improves second-query formation beyond sample-independent mixing and the original recurrence
- **状态／预算**：原记录 `provisional`；{"seed":20262027,"optimizer_steps":2193,"new_training_seeds":[]}。
- **结果**：macroAUC=0.827100568；macroLogloss=0.054981755；is_click AUC/Logloss=0.664734076/0.182425033；cart AUC/Logloss=0.865937694/0.017636449；fav AUC/Logloss=0.880649106/0.013285462；buy AUC/Logloss=0.897081398/0.006580077。
- **开销**：参数 81634772；训练含验证 2141.8167098760605 秒；峰值CUDA 25490670080 bytes；更新 2193。
- **原记录差值**：-0.001420645；对照 `explore_dual_no_feedback-s20262027-20260908T091954887719Z`。仅按该对照解释，不能当成相对当前最优的差。
- **归因**：相对上方记录对照的主指标点估计降低0.001420645，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-query-mixing](#phase-recurrence-query-mixing)。
- **后续假设**：先用诊断定位第二次attention分配是否有独立价值，降低同类门扫参优先级。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-mixing/runs/mix_q8_query_constant-s20262027-20260909T035108993695Z/manifest.json`（SHA256前缀 `ee72eff429db`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/runs/mix_q8_query_constant-s20262027-20260909T035108993695Z/result.json`（SHA256前缀 `cdd3b40f9083`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/final-audit.json`（SHA256前缀 `2987d7a3f905`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/summary.json`（SHA256前缀 `859243590be6`）

<a id="t155"></a>
### T155 · `taobao-query-mixing-mix_q4_channel_constant-mix_q4_channel_joint-paired-bootstrap`

- **方法**：500 paired user resamples with shared task weights
- **原假设**：Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "mix_q4_channel_constant",
  "candidate": "mix_q4_channel_joint",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "training_updates": 0,
  "reference_role": "conditioning_control",
  "mandatory": true,
  "mandatory_joint_constant_contrast": true,
  "mandatory_original_recurrence_contrast": false,
  "delta_auc": {
    "is_click": 0.0009396760089108724,
    "cart": -0.00030779292300375793,
    "fav": 0.0003297028138634639,
    "buy": 0.0012254920583231188
  },
  "ci95_auc": {
    "is_click": [
      0.0003943137650293805,
      0.0014970511368180475
    ],
    "cart": [
      -0.0018891050093504669,
      0.0012896822093320906
    ],
    "fav": [
      -0.0011219161088202034,
      0.0016688241167196354
    ],
    "buy": [
      -0.0011389509503691025,
      0.0036465499866002904
    ]
  },
  "delta_macro_auc": 0.0005467694895233688,
  "ci95_macro_auc": [
    -0.00034580346755074193,
    0.0014149764367677583
  ],
  "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-query-mixing](#phase-recurrence-query-mixing)。
- **后续假设**：先用诊断定位第二次attention分配是否有独立价值，降低同类门扫参优先级。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/paired-bootstrap.json`（SHA256前缀 `b29562a74050`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/paired-bootstrap-identities.json`（SHA256前缀 `38fd4c4d15e6`）

<a id="t156"></a>
### T156 · `taobao-query-mixing-mix_q4_query_constant-mix_q4_query_joint-paired-bootstrap`

- **方法**：500 paired user resamples with shared task weights
- **原假设**：Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "mix_q4_query_constant",
  "candidate": "mix_q4_query_joint",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "training_updates": 0,
  "reference_role": "conditioning_control",
  "mandatory": true,
  "mandatory_joint_constant_contrast": true,
  "mandatory_original_recurrence_contrast": false,
  "delta_auc": {
    "is_click": 0.004026301924688225,
    "cart": 0.00033814925358510717,
    "fav": 0.0037764315173665075,
    "buy": 0.0026777429462135904
  },
  "ci95_auc": {
    "is_click": [
      0.0033928000699526588,
      0.00464941256216283
    ],
    "cart": [
      -0.0012317892293798455,
      0.002091886799480741
    ],
    "fav": [
      0.002033056015334997,
      0.005640834004541323
    ],
    "buy": [
      0.0005798320885039538,
      0.004631683856751228
    ]
  },
  "delta_macro_auc": 0.0027046564104633575,
  "ci95_macro_auc": [
    0.0018877894099423621,
    0.0034858632233484375
  ],
  "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-query-mixing](#phase-recurrence-query-mixing)。
- **后续假设**：先用诊断定位第二次attention分配是否有独立价值，降低同类门扫参优先级。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/paired-bootstrap.json`（SHA256前缀 `b29562a74050`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/paired-bootstrap-identities.json`（SHA256前缀 `38fd4c4d15e6`）

<a id="t157"></a>
### T157 · `taobao-query-mixing-mix_q8_channel_constant-mix_q8_channel_joint-paired-bootstrap`

- **方法**：500 paired user resamples with shared task weights
- **原假设**：Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "mix_q8_channel_constant",
  "candidate": "mix_q8_channel_joint",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "training_updates": 0,
  "reference_role": "conditioning_control",
  "mandatory": true,
  "mandatory_joint_constant_contrast": true,
  "mandatory_original_recurrence_contrast": false,
  "delta_auc": {
    "is_click": -0.003176234324373972,
    "cart": -0.003630568236864251,
    "fav": -0.001617365217203015,
    "buy": -0.001735807866823147
  },
  "ci95_auc": {
    "is_click": [
      -0.003709854482947536,
      -0.002597738623642038
    ],
    "cart": [
      -0.005280279111062139,
      -0.002204917418873953
    ],
    "fav": [
      -0.0032190323571760714,
      9.241166552236895e-06
    ],
    "buy": [
      -0.004123537443484607,
      0.0008153443603363585
    ]
  },
  "delta_macro_auc": -0.0025399939113160963,
  "ci95_macro_auc": [
    -0.00339650489045982,
    -0.0017128938910583942
  ],
  "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-query-mixing](#phase-recurrence-query-mixing)。
- **后续假设**：先用诊断定位第二次attention分配是否有独立价值，降低同类门扫参优先级。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/paired-bootstrap.json`（SHA256前缀 `b29562a74050`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/paired-bootstrap-identities.json`（SHA256前缀 `38fd4c4d15e6`）

<a id="t158"></a>
### T158 · `taobao-query-mixing-mix_q8_query_constant-mix_q8_query_joint-paired-bootstrap`

- **方法**：500 paired user resamples with shared task weights
- **原假设**：Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "mix_q8_query_constant",
  "candidate": "mix_q8_query_joint",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "training_updates": 0,
  "reference_role": "conditioning_control",
  "mandatory": true,
  "mandatory_joint_constant_contrast": true,
  "mandatory_original_recurrence_contrast": false,
  "delta_auc": {
    "is_click": -0.001051994572609516,
    "cart": -0.0013688491357850552,
    "fav": -0.0019518679182036758,
    "buy": -0.001923074119085988
  },
  "ci95_auc": {
    "is_click": [
      -0.0016327013381994908,
      -0.0004522785970629773
    ],
    "cart": [
      -0.002993602416902938,
      0.00022663232321096734
    ],
    "fav": [
      -0.0034549692956826,
      -0.0005346780531825391
    ],
    "buy": [
      -0.0040314029894937455,
      0.00020800352740136724
    ]
  },
  "delta_macro_auc": -0.0015739464364210587,
  "ci95_macro_auc": [
    -0.002349156794962986,
    -0.0007747138269309635
  ],
  "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-query-mixing](#phase-recurrence-query-mixing)。
- **后续假设**：先用诊断定位第二次attention分配是否有独立价值，降低同类门扫参优先级。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/paired-bootstrap.json`（SHA256前缀 `b29562a74050`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/paired-bootstrap-identities.json`（SHA256前缀 `38fd4c4d15e6`）

<a id="t159"></a>
### T159 · `taobao-query-mixing-ctx_q4-mix_q4_channel_joint-paired-bootstrap`

- **方法**：500 paired user resamples with shared task weights
- **原假设**：Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "ctx_q4",
  "candidate": "mix_q4_channel_joint",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "training_updates": 0,
  "reference_role": "original_recurrence",
  "mandatory": true,
  "mandatory_joint_constant_contrast": false,
  "mandatory_original_recurrence_contrast": true,
  "delta_auc": {
    "is_click": 0.0010477879755581565,
    "cart": -0.0005357268052302855,
    "fav": -0.0001330516824029626,
    "buy": -0.0013545075722954936
  },
  "ci95_auc": {
    "is_click": [
      0.0004861313039931425,
      0.0015792725353308583
    ],
    "cart": [
      -0.002235592051036747,
      0.0010961163527792054
    ],
    "fav": [
      -0.0016314016624072946,
      0.001227642832564915
    ],
    "buy": [
      -0.0036294354134415283,
      0.0006782916088449606
    ]
  },
  "delta_macro_auc": -0.0002438745210926463,
  "ci95_macro_auc": [
    -0.001100299388456618,
    0.000557029552632559
  ],
  "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-query-mixing](#phase-recurrence-query-mixing)。
- **后续假设**：先用诊断定位第二次attention分配是否有独立价值，降低同类门扫参优先级。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/paired-bootstrap.json`（SHA256前缀 `b29562a74050`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/paired-bootstrap-identities.json`（SHA256前缀 `38fd4c4d15e6`）

<a id="t160"></a>
### T160 · `taobao-query-mixing-ctx_q4-mix_q4_channel_constant-paired-bootstrap`

- **方法**：500 paired user resamples with shared task weights
- **原假设**：Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "ctx_q4",
  "candidate": "mix_q4_channel_constant",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "training_updates": 0,
  "reference_role": "original_recurrence",
  "mandatory": true,
  "mandatory_joint_constant_contrast": false,
  "mandatory_original_recurrence_contrast": true,
  "delta_auc": {
    "is_click": 0.00010811196664728406,
    "cart": -0.00022793388222652755,
    "fav": -0.0004627544962664265,
    "buy": -0.0025799996306186124
  },
  "ci95_auc": {
    "is_click": [
      -0.00040503714291564575,
      0.0006541029600254017
    ],
    "cart": [
      -0.001821141295435813,
      0.0014881386366081685
    ],
    "fav": [
      -0.002042880667650912,
      0.0009823466517281938
    ],
    "buy": [
      -0.004687336377400162,
      -0.0004541224120995038
    ]
  },
  "delta_macro_auc": -0.0007906440106160151,
  "ci95_macro_auc": [
    -0.0015777067009283365,
    6.777985181484609e-05
  ],
  "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-query-mixing](#phase-recurrence-query-mixing)。
- **后续假设**：先用诊断定位第二次attention分配是否有独立价值，降低同类门扫参优先级。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/paired-bootstrap.json`（SHA256前缀 `b29562a74050`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/paired-bootstrap-identities.json`（SHA256前缀 `38fd4c4d15e6`）

<a id="t161"></a>
### T161 · `taobao-query-mixing-ctx_q4-mix_q4_query_joint-paired-bootstrap`

- **方法**：500 paired user resamples with shared task weights
- **原假设**：Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "ctx_q4",
  "candidate": "mix_q4_query_joint",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "training_updates": 0,
  "reference_role": "original_recurrence",
  "mandatory": true,
  "mandatory_joint_constant_contrast": false,
  "mandatory_original_recurrence_contrast": true,
  "delta_auc": {
    "is_click": 0.0012016268340766523,
    "cart": -0.0012313756430578637,
    "fav": 0.001037502030625248,
    "buy": -0.0017425635177172882
  },
  "ci95_auc": {
    "is_click": [
      0.0006715907249102065,
      0.0017842724149109783
    ],
    "cart": [
      -0.002744995310557466,
      0.00033933726732431605
    ],
    "fav": [
      -0.0005161908320194086,
      0.0024694815205208915
    ],
    "buy": [
      -0.003806432781218372,
      0.00020590958116525608
    ]
  },
  "delta_macro_auc": -0.00018370257401822965,
  "ci95_macro_auc": [
    -0.0009206318506348954,
    0.0005333416620977589
  ],
  "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-query-mixing](#phase-recurrence-query-mixing)。
- **后续假设**：先用诊断定位第二次attention分配是否有独立价值，降低同类门扫参优先级。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/paired-bootstrap.json`（SHA256前缀 `b29562a74050`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/paired-bootstrap-identities.json`（SHA256前缀 `38fd4c4d15e6`）

<a id="t162"></a>
### T162 · `taobao-query-mixing-ctx_q4-mix_q4_query_constant-paired-bootstrap`

- **方法**：500 paired user resamples with shared task weights
- **原假设**：Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "ctx_q4",
  "candidate": "mix_q4_query_constant",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "training_updates": 0,
  "reference_role": "original_recurrence",
  "mandatory": true,
  "mandatory_joint_constant_contrast": false,
  "mandatory_original_recurrence_contrast": true,
  "delta_auc": {
    "is_click": -0.0028246750906115725,
    "cart": -0.001569524896642971,
    "fav": -0.0027389294867412595,
    "buy": -0.004420306463930879
  },
  "ci95_auc": {
    "is_click": [
      -0.0034464809343528803,
      -0.0022335174474880544
    ],
    "cart": [
      -0.003158353599223049,
      4.170809789978412e-05
    ],
    "fav": [
      -0.004562041673711182,
      -0.0012425652951645967
    ],
    "buy": [
      -0.006733434564550341,
      -0.002332306765200038
    ]
  },
  "delta_macro_auc": -0.002888358984481587,
  "ci95_macro_auc": [
    -0.0037251126323888406,
    -0.0020567260238820934
  ],
  "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-query-mixing](#phase-recurrence-query-mixing)。
- **后续假设**：先用诊断定位第二次attention分配是否有独立价值，降低同类门扫参优先级。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/paired-bootstrap.json`（SHA256前缀 `b29562a74050`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/paired-bootstrap-identities.json`（SHA256前缀 `38fd4c4d15e6`）

<a id="t163"></a>
### T163 · `taobao-query-mixing-explore_dual_no_feedback-mix_q8_channel_joint-paired-bootstrap`

- **方法**：500 paired user resamples with shared task weights
- **原假设**：Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "explore_dual_no_feedback",
  "candidate": "mix_q8_channel_joint",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "training_updates": 0,
  "reference_role": "original_recurrence",
  "mandatory": true,
  "mandatory_joint_constant_contrast": false,
  "mandatory_original_recurrence_contrast": true,
  "delta_auc": {
    "is_click": -0.004666465801792641,
    "cart": -0.006682770277031458,
    "fav": -0.0038061049515943912,
    "buy": -0.004828442869057148
  },
  "ci95_auc": {
    "is_click": [
      -0.005303926320941235,
      -0.004020554237367991
    ],
    "cart": [
      -0.008424294821025326,
      -0.004947287797701411
    ],
    "fav": [
      -0.005325637926526164,
      -0.0021637805524471645
    ],
    "buy": [
      -0.007317971514083537,
      -0.002309674782114167
    ]
  },
  "delta_macro_auc": -0.004995945974868965,
  "ci95_macro_auc": [
    -0.005948306359540133,
    -0.004113655213096361
  ],
  "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-query-mixing](#phase-recurrence-query-mixing)。
- **后续假设**：先用诊断定位第二次attention分配是否有独立价值，降低同类门扫参优先级。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/paired-bootstrap.json`（SHA256前缀 `b29562a74050`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/paired-bootstrap-identities.json`（SHA256前缀 `38fd4c4d15e6`）

<a id="t164"></a>
### T164 · `taobao-query-mixing-explore_dual_no_feedback-mix_q8_channel_constant-paired-bootstrap`

- **方法**：500 paired user resamples with shared task weights
- **原假设**：Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "explore_dual_no_feedback",
  "candidate": "mix_q8_channel_constant",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "training_updates": 0,
  "reference_role": "original_recurrence",
  "mandatory": true,
  "mandatory_joint_constant_contrast": false,
  "mandatory_original_recurrence_contrast": true,
  "delta_auc": {
    "is_click": -0.0014902314774186687,
    "cart": -0.0030522020401672068,
    "fav": -0.002188739734391376,
    "buy": -0.003092635002234001
  },
  "ci95_auc": {
    "is_click": [
      -0.0021199715335199405,
      -0.00097348431138746
    ],
    "cart": [
      -0.0046886865729858646,
      -0.0015056325840211028
    ],
    "fav": [
      -0.0037316955363647596,
      -0.0005708480873126682
    ],
    "buy": [
      -0.0052659927506580025,
      -0.0010059651883430447
    ]
  },
  "delta_macro_auc": -0.0024559520635528687,
  "ci95_macro_auc": [
    -0.0032940635403033297,
    -0.0016133641263571176
  ],
  "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-query-mixing](#phase-recurrence-query-mixing)。
- **后续假设**：先用诊断定位第二次attention分配是否有独立价值，降低同类门扫参优先级。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/paired-bootstrap.json`（SHA256前缀 `b29562a74050`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/paired-bootstrap-identities.json`（SHA256前缀 `38fd4c4d15e6`）

<a id="t165"></a>
### T165 · `taobao-query-mixing-explore_dual_no_feedback-mix_q8_query_joint-paired-bootstrap`

- **方法**：500 paired user resamples with shared task weights
- **原假设**：Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "explore_dual_no_feedback",
  "candidate": "mix_q8_query_joint",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "training_updates": 0,
  "reference_role": "original_recurrence",
  "mandatory": true,
  "mandatory_joint_constant_contrast": false,
  "mandatory_original_recurrence_contrast": true,
  "delta_auc": {
    "is_click": -0.00318780413545261,
    "cart": -0.002927991196695623,
    "fav": -0.003313307731145132,
    "buy": -0.0025492624169622946
  },
  "ci95_auc": {
    "is_click": [
      -0.00377688877041345,
      -0.0026360426485845497
    ],
    "cart": [
      -0.004382965118201504,
      -0.0011609163136749258
    ],
    "fav": [
      -0.005025863481237708,
      -0.0016432432616736634
    ],
    "buy": [
      -0.0045612975915463125,
      -0.0002295001348551552
    ]
  },
  "delta_macro_auc": -0.0029945913700639704,
  "ci95_macro_auc": [
    -0.003824687661344195,
    -0.0021077088514912207
  ],
  "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-query-mixing](#phase-recurrence-query-mixing)。
- **后续假设**：先用诊断定位第二次attention分配是否有独立价值，降低同类门扫参优先级。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/paired-bootstrap.json`（SHA256前缀 `b29562a74050`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/paired-bootstrap-identities.json`（SHA256前缀 `38fd4c4d15e6`）

<a id="t166"></a>
### T166 · `taobao-query-mixing-explore_dual_no_feedback-mix_q8_query_constant-paired-bootstrap`

- **方法**：500 paired user resamples with shared task weights
- **原假设**：Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "reference": "explore_dual_no_feedback",
  "candidate": "mix_q8_query_constant",
  "users": 268201,
  "repetitions": 500,
  "bootstrap_seed": 20260906,
  "training_updates": 0,
  "reference_role": "original_recurrence",
  "mandatory": true,
  "mandatory_joint_constant_contrast": false,
  "mandatory_original_recurrence_contrast": true,
  "delta_auc": {
    "is_click": -0.002135809562843094,
    "cart": -0.0015591420609105677,
    "fav": -0.0013614398129414562,
    "buy": -0.0006261882978763067
  },
  "ci95_auc": {
    "is_click": [
      -0.002732984634094346,
      -0.0015873871743291219
    ],
    "cart": [
      -0.002929317925476585,
      -0.0001227159747359377
    ],
    "fav": [
      -0.0029469159526564954,
      0.00013214074761747896
    ],
    "buy": [
      -0.002721334952174195,
      0.0014771292637797944
    ]
  },
  "delta_macro_auc": -0.0014206449336429117,
  "ci95_macro_auc": [
    -0.0021973074694157215,
    -0.0006391588014134531
  ],
  "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-query-mixing](#phase-recurrence-query-mixing)。
- **后续假设**：先用诊断定位第二次attention分配是否有独立价值，降低同类门扫参优先级。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/paired-bootstrap.json`（SHA256前缀 `b29562a74050`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/paired-bootstrap-identities.json`（SHA256前缀 `38fd4c4d15e6`）

<a id="t167"></a>
### T167 · `taobao-query-mixing-final-interpretation`

- **方法**：Adjudicate eight completed training episodes under the prospective quality/compute policy; preserve historical rows and pending diagnostic evidence
- **原假设**：Joint bounded mixing should improve matched static controls and original recurrence if conditioning adds useful training benefit
- **状态／预算**：原记录 `discard`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "state": "completed_training_adjudication_diagnostics_separate",
  "new_training_runs": 0,
  "training_count": 109,
  "kept_new_candidates": [],
  "quality_compute_tradeoff_candidates": [],
  "all_below_original_and_outside_queue_rule": true,
  "confidence": [
    "One training seed20262027; no additional training seeds",
    "500 paired validation-user resamples; intervals conditional on checkpoints and released validation protocol",
    "Selection and multiple comparisons uncorrected; do not infer seed stability or statistical equivalence",
    "Diagnostic subset bootstrap conditions on frozen subset and ignores stratified sampling-design uncertainty",
    "Channel and query gates have different parameter counts; q4 and q8 interpreted separately",
    "Forward latency is one repeated real batch on matched GPU, not end-to-end serving latency; tiny mean differences are not reliable speedup claims"
  ],
  "retrieval_mechanism_claimed": false,
  "near_reference_rule": {
    "status": "Optimization queue policy; not an equivalence or noninferiority margin",
    "macro_auc_delta_to_same_q_original_minimum": -0.0001,
    "forward_mean_ratio_to_same_q_original_maximum": 1.05,
    "source": "qformer-loop/taobao-transfer/recurrence-query-routes/reports/analysis-plan.json",
    "requirements": [
      "Complete verified 2193-step training, all mandatory paired comparisons, and all 11 matched latency measurements",
      "Show all four task AUC/logloss and parameter/compute differences",
      "Any task harm remains an explicit tradeoff; never promote solely because it falls inside this queue threshold"
    ],
    "threshold_fixed_before_results": true
  },
  "diagnostic_integration": {
    "state": "pending_separate_zero_training_evidence_adjudication",
    "fixed_pair": [
      "mix_q4_channel_joint",
      "mix_q4_channel_constant"
    ],
    "diagnostic_queue_preserved": true,
    "note": "No diagnostic metric is consumed here; the gate diagnostic may be running or awaiting its separate evidence episode"
  },
  "decisions": {
    "mix_q4_channel_joint": {
      "decision": "discard",
      "retained_by_queue_policy": false,
      "macro_auc": 0.8282286909051509,
      "parameter_count": 51748804,
      "original_evidence": "inconclusive_crosses_zero",
      "forward_mean_ms": 14.495008063316345,
      "forward_ratio_to_original": 1.0074550684083148,
      "near_reference_policy_passed": false,
      "versus_original": {
        "reference": "ctx_q4",
        "candidate": "mix_q4_channel_joint",
        "delta_macro_auc": -0.0002438745210926463
      },
      "versus_best_single": {
        "reference": "psgd_precond_every_step",
        "candidate": "mix_q4_channel_joint",
        "delta_macro_auc": -0.00039484161452374433
      },
      "original_bootstrap": {
        "reference": "ctx_q4",
        "candidate": "mix_q4_channel_joint",
        "delta_macro_auc": -0.0002438745210926463,
        "ci95_macro_auc": [
          -0.001100299388456618,
          0.000557029552632559
        ]
      },
      "conditioning_evidence": {
        "point_case": "joint_better_than_constant_but_not_above_original",
        "joint_constant_evidence": "inconclusive_crosses_zero",
        "original_evidence": "inconclusive_crosses_zero",
        "both_conditional_intervals_positive": false,
        "scope": "Input-conditioned versus static parameterization under this fixed seed/budget; not isolated retrieval or training-seed robustness",
        "comparison": {
          "reference": "mix_q4_channel_constant",
          "delta_macro_auc": 0.0005467694895233688
        },
        "bootstrap": {
          "reference": "mix_q4_channel_constant",
          "delta_macro_auc": 0.0005467694895233688,
          "ci95_macro_auc": [
            -0.00034580346755074193,
            0.0014149764367677583
          ]
        }
      }
    },
    "mix_q4_channel_constant": {
      "decision": "discard",
      "retained_by_queue_policy": false,
      "macro_auc": 0.8276819214156276,
      "parameter_count": 51748804,
      "original_evidence": "inconclusive_crosses_zero",
      "forward_mean_ms": 14.351776790618896,
      "forward_ratio_to_original": 0.997499981042837,
      "near_reference_policy_passed": false,
      "versus_original": {
        "reference": "ctx_q4",
        "candidate": "mix_q4_channel_constant",
        "delta_macro_auc": -0.0007906440106160151
      },
      "versus_best_single": {
        "reference": "psgd_precond_every_step",
        "candidate": "mix_q4_channel_constant",
        "delta_macro_auc": -0.0009416111040471131
      },
      "original_bootstrap": {
        "reference": "ctx_q4",
        "candidate": "mix_q4_channel_constant",
        "delta_macro_auc": -0.0007906440106160151,
        "ci95_macro_auc": [
          -0.0015777067009283365,
          6.777985181484609e-05
        ]
      },
      "conditioning_evidence": {}
    },
    "mix_q4_query_joint": {
      "decision": "discard",
      "retained_by_queue_policy": false,
      "macro_auc": 0.8282888628522254,
      "parameter_count": 51742676,
      "original_evidence": "inconclusive_crosses_zero",
      "forward_mean_ms": 14.45967357158661,
      "forward_ratio_to_original": 1.0049991944531396,
      "near_reference_policy_passed": false,
      "versus_original": {
        "reference": "ctx_q4",
        "candidate": "mix_q4_query_joint",
        "delta_macro_auc": -0.00018370257401822965
      },
      "versus_best_single": {
        "reference": "psgd_precond_every_step",
        "candidate": "mix_q4_query_joint",
        "delta_macro_auc": -0.0003346696674493277
      },
      "original_bootstrap": {
        "reference": "ctx_q4",
        "candidate": "mix_q4_query_joint",
        "delta_macro_auc": -0.00018370257401822965,
        "ci95_macro_auc": [
          -0.0009206318506348954,
          0.0005333416620977589
        ]
      },
      "conditioning_evidence": {
        "point_case": "joint_better_than_constant_but_not_above_original",
        "joint_constant_evidence": "positive_conditional_interval",
        "original_evidence": "inconclusive_crosses_zero",
        "both_conditional_intervals_positive": false,
        "scope": "Input-conditioned versus static parameterization under this fixed seed/budget; not isolated retrieval or training-seed robustness",
        "comparison": {
          "reference": "mix_q4_query_constant",
          "delta_macro_auc": 0.0027046564104633575
        },
        "bootstrap": {
          "reference": "mix_q4_query_constant",
          "delta_macro_auc": 0.0027046564104633575,
          "ci95_macro_auc": [
            0.0018877894099423621,
            0.0034858632233484375
          ]
        }
      }
    },
    "mix_q4_query_constant": {
      "decision": "discard",
      "retained_by_queue_policy": false,
      "macro_auc": 0.825584206441762,
      "parameter_count": 51742676,
      "original_evidence": "negative_conditional_interval",
      "forward_mean_ms": 14.328207206726074,
      "forward_ratio_to_original": 0.9958618103947506,
      "near_reference_policy_passed": false,
      "versus_original": {
        "reference": "ctx_q4",
        "candidate": "mix_q4_query_constant",
        "delta_macro_auc": -0.002888358984481587
      },
      "versus_best_single": {
        "reference": "psgd_precond_every_step",
        "candidate": "mix_q4_query_constant",
        "delta_macro_auc": -0.003039326077912685
      },
      "original_bootstrap": {
        "reference": "ctx_q4",
        "candidate": "mix_q4_query_constant",
        "delta_macro_auc": -0.002888358984481587,
        "ci95_macro_auc": [
          -0.0037251126323888406,
          -0.0020567260238820934
        ]
      },
      "conditioning_evidence": {}
    },
    "mix_q8_channel_joint": {
      "decision": "discard",
      "retained_by_queue_policy": false,
      "macro_auc": 0.8235252673993565,
      "parameter_count": 81640900,
      "original_evidence": "negative_conditional_interval",
      "forward_mean_ms": 16.10703749656677,
      "forward_ratio_to_original": 1.020528206073035,
      "near_reference_policy_passed": false,
      "versus_original": {
        "reference": "explore_dual_no_feedback",
        "candidate": "mix_q8_channel_joint",
        "delta_macro_auc": -0.004995945974868965
      },
      "versus_best_single": {
        "reference": "psgd_precond_every_step",
        "candidate": "mix_q8_channel_joint",
        "delta_macro_auc": -0.005098265120318191
      },
      "original_bootstrap": {
        "reference": "explore_dual_no_feedback",
        "candidate": "mix_q8_channel_joint",
        "delta_macro_auc": -0.004995945974868965,
        "ci95_macro_auc": [
          -0.005948306359540133,
          -0.004113655213096361
        ]
      },
      "conditioning_evidence": {
        "point_case": "no_positive_joint_conditioning_point_advantage",
        "joint_constant_evidence": "negative_conditional_interval",
        "original_evidence": "negative_conditional_interval",
        "both_conditional_intervals_positive": false,
        "scope": "Input-conditioned versus static parameterization under this fixed seed/budget; not isolated retrieval or training-seed robustness",
        "comparison": {
          "reference": "mix_q8_channel_constant",
          "delta_macro_auc": -0.0025399939113160963
        },
        "bootstrap": {
          "reference": "mix_q8_channel_constant",
          "delta_macro_auc": -0.0025399939113160963,
          "ci95_macro_auc": [
            -0.00339650489045982,
            -0.0017128938910583942
          ]
        }
      }
    },
    "mix_q8_channel_constant": {
      "decision": "discard",
      "retained_by_queue_policy": false,
      "macro_auc": 0.8260652613106726,
      "parameter_count": 81640900,
      "original_evidence": "negative_conditional_interval",
      "forward_mean_ms": 15.946015238761902,
      "forward_ratio_to_original": 1.010325972674717,
      "near_reference_policy_passed": false,
      "versus_original": {
        "reference": "explore_dual_no_feedback",
        "candidate": "mix_q8_channel_constant",
        "delta_macro_auc": -0.0024559520635528687
      },
      "versus_best_single": {
        "reference": "psgd_precond_every_step",
        "candidate": "mix_q8_channel_constant",
        "delta_macro_auc": -0.0025582712090020943
      },
      "original_bootstrap": {
        "reference": "explore_dual_no_feedback",
        "candidate": "mix_q8_channel_constant",
        "delta_macro_auc": -0.0024559520635528687,
        "ci95_macro_auc": [
          -0.0032940635403033297,
          -0.0016133641263571176
        ]
      },
      "conditioning_evidence": {}
    },
    "mix_q8_query_joint": {
      "decision": "discard",
      "retained_by_queue_policy": false,
      "macro_auc": 0.8255266220041615,
      "parameter_count": 81634772,
      "original_evidence": "negative_conditional_interval",
      "forward_mean_ms": 16.05049443244934,
      "forward_ratio_to_original": 1.0169456855876922,
      "near_reference_policy_passed": false,
      "versus_original": {
        "reference": "explore_dual_no_feedback",
        "candidate": "mix_q8_query_joint",
        "delta_macro_auc": -0.0029945913700639704
      },
      "versus_best_single": {
        "reference": "psgd_precond_every_step",
        "candidate": "mix_q8_query_joint",
        "delta_macro_auc": -0.003096910515513196
      },
      "original_bootstrap": {
        "reference": "explore_dual_no_feedback",
        "candidate": "mix_q8_query_joint",
        "delta_macro_auc": -0.0029945913700639704,
        "ci95_macro_auc": [
          -0.003824687661344195,
          -0.0021077088514912207
        ]
      },
      "conditioning_evidence": {
        "point_case": "no_positive_joint_conditioning_point_advantage",
        "joint_constant_evidence": "negative_conditional_interval",
        "original_evidence": "negative_conditional_interval",
        "both_conditional_intervals_positive": false,
        "scope": "Input-conditioned versus static parameterization under this fixed seed/budget; not isolated retrieval or training-seed robustness",
        "comparison": {
          "reference": "mix_q8_query_constant",
          "delta_macro_auc": -0.0015739464364210587
        },
        "bootstrap": {
          "reference": "mix_q8_query_constant",
          "delta_macro_auc": -0.0015739464364210587,
          "ci95_macro_auc": [
            -0.002349156794962986,
            -0.0007747138269309635
          ]
        }
      }
    },
    "mix_q8_query_constant": {
      "decision": "discard",
      "retained_by_queue_policy": false,
      "macro_auc": 0.8271005684405826,
      "parameter_count": 81634772,
      "original_evidence": "negative_conditional_interval",
      "forward_mean_ms": 15.948627185821532,
      "forward_ratio_to_original": 1.0104914634204665,
      "near_reference_policy_passed": false,
      "versus_original": {
        "reference": "explore_dual_no_feedback",
        "candidate": "mix_q8_query_constant",
        "delta_macro_auc": -0.0014206449336429117
      },
      "versus_best_single": {
        "reference": "psgd_precond_every_step",
        "candidate": "mix_q8_query_constant",
        "delta_macro_auc": -0.0015229640790921373
      },
      "original_bootstrap": {
        "reference": "explore_dual_no_feedback",
        "candidate": "mix_q8_query_constant",
        "delta_macro_auc": -0.0014206449336429117,
        "ci95_macro_auc": [
          -0.0021973074694157215,
          -0.0006391588014134531
        ]
      },
      "conditioning_evidence": {}
    }
  },
  "summary_scope": "每个家族、裁决及关键对照均保留；完整逐任务训练指标见本明细的对应训练条目；其余原始统计见所列证据。"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-query-mixing](#phase-recurrence-query-mixing)。
- **后续假设**：先用诊断定位第二次attention分配是否有独立价值，降低同类门扫参优先级。
- **证据**：`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/interpretation.json`（SHA256前缀 `55f5f76b4a5c`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/reports/interpretation-plan.json`（SHA256前缀 `b4b6b540abe7`）；`qformer-loop/taobao-transfer/recurrence-query-mixing/INTERPRETATION.md`（SHA256前缀 `04bff4713b84`）

<a id="t168"></a>
### T168 · `taobao-query-mixing-q4-channel-gate-diagnostics-5514`

- **方法**：Native identity, zero-offset and gate-offset shuffle; fixed joint/constant q4 channel pair and shared donors
- **原假设**：Test sample-to-gate correspondence using matched offset shuffle while preserving each receiving sample entry/work states
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "schema_version": 1,
  "state": "both_fixed_gate_diagnostics_completed",
  "created_at_utc": "2026-09-09T04:35:36.929579+00:00",
  "training_updates": 0,
  "new_training_runs": 0,
  "test_evaluated": false,
  "families": {
    "mix_q4_channel_joint": {
      "perturbations": {
        "zero": {
          "is_click": {
            "original_ipw_bce": 0.18057706583378152,
            "perturbed_ipw_bce": 0.18058727059927848,
            "delta_ipw_bce": 1.0204765496962045e-05,
            "ci95": [
              -5.030836741134513e-06,
              2.4357800563652016e-05
            ],
            "interval_position": "crosses_zero",
            "users": 4647
          },
          "cart": {
            "original_ipw_bce": 0.017926234716385537,
            "perturbed_ipw_bce": 0.017925918869498632,
            "delta_ipw_bce": -3.15846886906102e-07,
            "ci95": [
              -1.7139434328638722e-06,
              1.4950080380488442e-06
            ],
            "interval_position": "crosses_zero",
            "users": 4647
          },
          "fav": {
            "original_ipw_bce": 0.013385327378269333,
            "perturbed_ipw_bce": 0.013382321502242674,
            "delta_ipw_bce": -3.0058760266606403e-06,
            "ci95": [
              -5.669880008535475e-06,
              -6.058644986344504e-07
            ],
            "interval_position": "negative",
            "users": 4647
          },
          "buy": {
            "original_ipw_bce": 0.006743103446140139,
            "perturbed_ipw_bce": 0.006742655654385224,
            "delta_ipw_bce": -4.4779175491530086e-07,
            "ci95": [
              -8.410827941407023e-07,
              4.0299075236706245e-08
            ],
            "interval_position": "crosses_zero",
            "users": 4647
          }
        },
        "shuffle": {
          "is_click": {
            "original_ipw_bce": 0.18057706583378152,
            "perturbed_ipw_bce": 0.18058423412489377,
            "delta_ipw_bce": 7.168291112259214e-06,
            "ci95": [
              -3.744813346687321e-06,
              1.9066880322491946e-05
            ],
            "interval_position": "crosses_zero",
            "users": 4647
          },
          "cart": {
            "original_ipw_bce": 0.017926234716385537,
            "perturbed_ipw_bce": 0.01792735132863872,
            "delta_ipw_bce": 1.116612253182249e-06,
            "ci95": [
              -1.0287024671596873e-06,
              3.6165133320320552e-06
            ],
            "interval_position": "crosses_zero",
            "users": 4647
          },
          "fav": {
            "original_ipw_bce": 0.013385327378269333,
            "perturbed_ipw_bce": 0.013381925219922163,
            "delta_ipw_bce": -3.4021583471712413e-06,
            "ci95": [
              -5.685328977264781e-06,
              -1.3059295141313788e-06
            ],
            "interval_position": "negative",
            "users": 4647
          },
          "buy": {
            "original_ipw_bce": 0.006743103446140139,
            "perturbed_ipw_bce": 0.006743161277632983,
            "delta_ipw_bce": 5.783149284375293e-08,
            "ci95": [
              -2.355855225839231e-07,
              3.7450404959052537e-07
            ],
            "interval_position": "crosses_zero",
            "users": 4647
          }
        }
      },
      "gate_statistics": {
        "shape": [
          5514,
          4,
          384
        ],
        "minimum": 0.43539464473724365,
        "maximum": 0.5725645273923874,
        "ipw_mean_gate": 0.4994308131008234,
        "ipw_mean_coordinate_variance": 3.4956581623400675e-05,
        "scope": "Gate variance across examples, separately per slot/channel; not a proof of useful conditioning or collapse"
      },
      "gate_maximum_absolute_difference_from_first_example": 0.07843221817165613,
      "delta_query_geometry": {
        "subset_unweighted": {
          "mean_coordinate_variance": 0.00023758786946701733,
          "total_centered_variance": 0.36493496750133864,
          "mean_correction_squared_norm": 0.5345606649075095,
          "centered_energy_fraction": 0.6826820442624228,
          "mean_delta_to_base_l2_ratio": 0.007789672406726995,
          "mean_delta_l2": 0.6728264032162928,
          "mean_base_l2": 86.6941317818739
        },
        "population_ipw": {
          "mean_coordinate_variance": 0.0002213574211582185,
          "total_centered_variance": 0.34000499889902364,
          "mean_correction_squared_norm": 0.6951809679490057,
          "centered_energy_fraction": 0.4890884741884423,
          "mean_delta_to_base_l2_ratio": 0.008893584399711461,
          "mean_delta_l2": 0.7694040423960876,
          "mean_base_l2": 86.98631554633411
        },
        "shape": [
          5514,
          4,
          384
        ],
        "all_finite": true,
        "variance_axis": "examples, independently per query slot/channel; population denominator",
        "zero_base_count": 0
      },
      "parameter_statistics": {
        "A": {
          "shape": [
            16,
            1152
          ],
          "finite": true,
          "frobenius_norm": 2.872248190162489,
          "max_abs": 0.06710272282361984,
          "nonzero_elements": 18432,
          "initial_frobenius_norm": 2.3100137794380755,
          "change_from_initial_frobenius_norm": 1.6916355340968345,
          "relative_change_from_initial": 0.7323053867273185,
          "exactly_equal_to_initial": false
        },
        "B": {
          "shape": [
            384,
            16
          ],
          "finite": true,
          "frobenius_norm": 1.0589213983282983,
          "max_abs": 0.057618334889411926,
          "nonzero_elements": 6144,
          "initial_frobenius_norm": 0.0,
          "change_from_initial_frobenius_norm": 1.0589213983282983,
          "exactly_equal_to_initial": false
        },
        "initialization_replay": "same training seed and constructor, clone before checkpoint load; CPU init"
      },
      "donor_statistics": {
        "samples": 5514,
        "different_user_count": 4927,
        "same_user_count_including_fixed_points": 587,
        "fixed_point_count": 379,
        "different_user_fraction": 0.8935437069278201,
        "same_user_fraction_including_fixed_points": 0.10645629307217991,
        "fixed_point_fraction": 0.06873413130214001
      },
      "identity_exact": true,
      "constant_shuffle_exact": false,
      "constant_shuffle_applicable": false,
      "export_seconds": 44.021969608962536,
      "export_audit_path": "qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/mix_q4_channel_joint/export/audit.json",
      "tool_report_path": "qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/mix_q4_channel_joint/toolkit/perturbation_report.json"
    },
    "mix_q4_channel_constant": {
      "perturbations": {
        "zero": {
          "is_click": {
            "original_ipw_bce": 0.18170596162670669,
            "perturbed_ipw_bce": 0.18170616448025279,
            "delta_ipw_bce": 2.0285354610037612e-07,
            "ci95": [
              5.164492231017958e-09,
              4.337338648627793e-07
            ],
            "interval_position": "positive",
            "users": 4647
          },
          "cart": {
            "original_ipw_bce": 0.019322289655011957,
            "perturbed_ipw_bce": 0.019322494706861894,
            "delta_ipw_bce": 2.0505184993905237e-07,
            "ci95": [
              -9.721513012866394e-09,
              6.071386973686327e-07
            ],
            "interval_position": "crosses_zero",
            "users": 4647
          },
          "fav": {
            "original_ipw_bce": 0.013259621660404867,
            "perturbed_ipw_bce": 0.013259634386427209,
            "delta_ipw_bce": 1.2726022342258106e-08,
            "ci95": [
              -1.701479670132352e-08,
              3.984729750412884e-08
            ],
            "interval_position": "crosses_zero",
            "users": 4647
          },
          "buy": {
            "original_ipw_bce": 0.006646721098703435,
            "perturbed_ipw_bce": 0.006646712927162072,
            "delta_ipw_bce": -8.171541362462717e-09,
            "ci95": [
              -1.7505650694848714e-08,
              5.546847299896314e-10
            ],
            "interval_position": "crosses_zero",
            "users": 4647
          }
        },
        "shuffle": {
          "is_click": {
            "original_ipw_bce": 0.18170596162670669,
            "perturbed_ipw_bce": 0.18170596162670669,
            "delta_ipw_bce": 0.0,
            "ci95": [
              0.0,
              0.0
            ],
            "interval_position": "exact_zero_implementation_control",
            "users": 4647
          },
          "cart": {
            "original_ipw_bce": 0.019322289655011957,
            "perturbed_ipw_bce": 0.019322289655011957,
            "delta_ipw_bce": 0.0,
            "ci95": [
              0.0,
              0.0
            ],
            "interval_position": "exact_zero_implementation_control",
            "users": 4647
          },
          "fav": {
            "original_ipw_bce": 0.013259621660404867,
            "perturbed_ipw_bce": 0.013259621660404867,
            "delta_ipw_bce": 0.0,
            "ci95": [
              0.0,
              0.0
            ],
            "interval_position": "exact_zero_implementation_control",
            "users": 4647
          },
          "buy": {
            "original_ipw_bce": 0.006646721098703435,
            "perturbed_ipw_bce": 0.006646721098703435,
            "delta_ipw_bce": 0.0,
            "ci95": [
              0.0,
              0.0
            ],
            "interval_position": "exact_zero_implementation_control",
            "users": 4647
          }
        }
      },
      "gate_statistics": {
        "shape": [
          5514,
          4,
          384
        ],
        "minimum": 0.4988792116055265,
        "maximum": 0.5009774029022083,
        "ipw_mean_gate": 0.4999334321039129,
        "ipw_mean_coordinate_variance": 2.1286829415063223e-28,
        "scope": "Gate variance across examples, separately per slot/channel; not a proof of useful conditioning or collapse"
      },
      "gate_maximum_absolute_difference_from_first_example": 0.0,
      "delta_query_geometry": {
        "subset_unweighted": {
          "mean_coordinate_variance": 1.8609946662058192e-07,
          "total_centered_variance": 0.0002858487807292138,
          "mean_correction_squared_norm": 0.0006303942045024302,
          "centered_energy_fraction": 0.4534444934417411,
          "mean_delta_to_base_l2_ratio": 0.0002774441075108076,
          "mean_delta_l2": 0.024643447225798296,
          "mean_base_l2": 88.1080416658009
        },
        "population_ipw": {
          "mean_coordinate_variance": 1.129726301556168e-07,
          "total_centered_variance": 0.0001735259599190274,
          "mean_correction_squared_norm": 0.0005742520097469398,
          "centered_energy_fraction": 0.30217736633694403,
          "mean_delta_to_base_l2_ratio": 0.00026829406804860395,
          "mean_delta_l2": 0.023677318100724126,
          "mean_base_l2": 87.77171490396262
        },
        "shape": [
          5514,
          4,
          384
        ],
        "all_finite": true,
        "variance_axis": "examples, independently per query slot/channel; population denominator",
        "zero_base_count": 0
      },
      "parameter_statistics": {
        "A": {
          "shape": [
            16,
            1152
          ],
          "finite": true,
          "frobenius_norm": 2.3116963634756327,
          "max_abs": 0.03416569158434868,
          "nonzero_elements": 18432,
          "initial_frobenius_norm": 2.3100137794380755,
          "change_from_initial_frobenius_norm": 0.08874210029880052,
          "relative_change_from_initial": 0.03841626447803595,
          "exactly_equal_to_initial": false
        },
        "B": {
          "shape": [
            384,
            16
          ],
          "finite": true,
          "frobenius_norm": 0.5876281880811561,
          "max_abs": 0.03346674516797066,
          "nonzero_elements": 6144,
          "initial_frobenius_norm": 0.0,
          "change_from_initial_frobenius_norm": 0.5876281880811561,
          "exactly_equal_to_initial": false
        },
        "initialization_replay": "same training seed and constructor, clone before checkpoint load; CPU init"
      },
      "donor_statistics": {
        "samples": 5514,
        "different_user_count": 4927,
        "same_user_count_including_fixed_points": 587,
        "fixed_point_count": 379,
        "different_user_fraction": 0.8935437069278201,
        "same_user_fraction_including_fixed_points": 0.10645629307217991,
        "fixed_point_fraction": 0.06873413130214001
      },
      "identity_exact": true,
      "constant_shuffle_exact": true,
      "constant_shuffle_applicable": true,
      "export_seconds": 39.322688706219196,
      "export_audit_path": "qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/mix_q4_channel_constant/export/audit.json",
      "tool_report_path": "qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/mix_q4_channel_constant/toolkit/perturbation_report.json"
    }
  },
  "findings": [
    {
      "id": "joint_fav",
      "claim": "Joint fav has negative zero and shuffle intervals. In this fixed checkpoint/subset, removing or misassigning the gate lowers fav BCE; this does not support useful sample-to-gate correspondence for fav.",
      "zero": {
        "original_ipw_bce": 0.013385327378269333,
        "perturbed_ipw_bce": 0.013382321502242674,
        "delta_ipw_bce": -3.0058760266606403e-06,
        "ci95": [
          -5.669880008535475e-06,
          -6.058644986344504e-07
        ],
        "interval_position": "negative",
        "users": 4647
      },
      "shuffle": {
        "original_ipw_bce": 0.013385327378269333,
        "perturbed_ipw_bce": 0.013381925219922163,
        "delta_ipw_bce": -3.4021583471712413e-06,
        "ci95": [
          -5.685328977264781e-06,
          -1.3059295141313788e-06
        ],
        "interval_position": "negative",
        "users": 4647
      }
    },
    {
      "id": "joint_other_tasks",
      "claim": "Joint is_click/cart/buy zero and shuffle intervals cross zero; effect direction is unresolved, not equivalence or evidence that inputs are unused."
    },
    {
      "id": "constant_zero_click",
      "claim": "Constant zero/is_click has a strictly positive but very small interval: the learned static mixture has a tiny conditional BCE advantage over zero-offset mean fusion on this subset. This is not evidence for useful input-conditioned mixing.",
      "evidence": {
        "original_ipw_bce": 0.18170596162670669,
        "perturbed_ipw_bce": 0.18170616448025279,
        "delta_ipw_bce": 2.0285354610037612e-07,
        "ci95": [
          5.164492231017958e-09,
          4.337338648627793e-07
        ],
        "interval_position": "positive",
        "users": 4647
      }
    },
    {
      "id": "constant_other_zero",
      "claim": "Constant zero cart/fav/buy intervals cross zero."
    },
    {
      "id": "constant_shuffle_control",
      "claim": "Constant shuffle is exactly invariant in probabilities and BCE for all four tasks. This confirms the expected implementation negative control, not that user/item/history data have no effect."
    },
    {
      "id": "gate_vs_delta",
      "claim": "Gate coefficient variance and delta=(gate-.5)*(work-entry) variance are different quantities. Constant gate is exactly identical across examples; its reported ~2e-28 variance is accumulation residue, while delta still varies because work-entry varies. Joint nonzero gate variance alone does not establish useful personalization."
    }
  ],
  "limitations": [
    "Conditional on two fixed checkpoints and one fixed stratified subset; no training-seed or sampling-design uncertainty.",
    "Stable-tool task intervals are not corrected for multiple interventions/tasks or model selection; do not synthesize an uncomputed macro interval.",
    "Within-minibatch donors include 379 fixed points and 587 same-user assignments; identical donor rows do not remove those limitations.",
    "Zero uses the trained checkpoint with mean fusion, not a retrained mean-fusion baseline; shuffle changes the complete second work query, not an isolated retrieval projection.",
    "No test data, new optimizer updates, new seed, or new model run was used. No memory was consolidated by this artifact."
  ],
  "memory_consolidated_at_analysis_creation": false
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-query-mixing](#phase-recurrence-query-mixing)。
- **后续假设**：先用诊断定位第二次attention分配是否有独立价值，降低同类门扫参优先级。
- **证据**：`qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/interpretation.json`（SHA256前缀 `d52d32aa2e94`）；`qformer-loop/taobao-transfer/query-mixing-diagnostics/RESULTS.md`（SHA256前缀 `3e0d69408300`）；`qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/memory-consolidation-status.json`（SHA256前缀 `8ec007620118`）

<a id="phase-recurrence-readout-retrieval"></a>
## 阶段 recurrence-readout-retrieval：任务读出与第二轮query专属参数

**方法**：q4/q8×共享读出/任务读出/query norm独立/query投影独立，8组固定预算。

**归因与边界**：8组完整训练macroAUC均低于同q原循环，且点差均超出near阈值；初始函数匹配并不保证优化轨迹一致。读出与检索共享需分别解释。

**历史后续假设**：保留全部10组用户配对和同卡延迟的后处理结果；task对shared的相对改善及固定权重诊断，不能替代对原循环的重训比较。

<a id="t100"></a>
### T100 · `taobao-q2-query-geometry-4096`

- **方法**：Export q1/q2 and run four stable rec-harness geometry analyses
- **原假设**：Second query may differ in geometry but causal utility is unknown
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "measurements": {
    "q1_mean": {
      "effective_rank": 52.5980855610032,
      "top1_variance_ratio": 0.0999913748786532,
      "centered_to_raw_energy_ratio": 0.20457801566366285,
      "random_pair_cosine": 0.7987646649124985
    },
    "q2_mean": {
      "effective_rank": 49.52891277685158,
      "top1_variance_ratio": 0.14986175833661217,
      "centered_to_raw_energy_ratio": 0.19691388471143081,
      "random_pair_cosine": 0.8090898524895148
    },
    "q1_slot0": {
      "effective_rank": 56.138983126626194,
      "top1_variance_ratio": 0.0993177037774296,
      "centered_to_raw_energy_ratio": 0.23590496987164464,
      "random_pair_cosine": 0.7647655216373164
    },
    "q2_slot0": {
      "effective_rank": 53.373652390051696,
      "top1_variance_ratio": 0.14683708584013083,
      "centered_to_raw_energy_ratio": 0.22785155500151172,
      "random_pair_cosine": 0.7777417796509472
    }
  },
  "scope": {
    "samples": 4096,
    "users": 375,
    "task_positives": {
      "is_click": 193,
      "cart": 11,
      "fav": 11,
      "buy": 4
    },
    "task_negatives": {
      "is_click": 3903,
      "cart": 4085,
      "fav": 4085,
      "buy": 4092
    },
    "query_shape": [
      4096,
      4,
      384
    ],
    "relative_change": 0.5330536365509033,
    "mean_paired_cosine": 0.9093810319900513,
    "finite": true,
    "subset": "first 4096 validation rows in released order; diagnostic only",
    "precision": "CPU float32; not comparable to BF16 full validation metrics",
    "training_updates": 0,
    "test_evaluated": false,
    "runtime_seconds": 9.34495746344328
  },
  "interpretation": "q2 changes but becomes slightly more concentrated; no demonstrated retrieval quality defect or causal bottleneck",
  "next_action": "stratified same-weight entry/work shuffle and zero at second query only",
  "task_sufficiency": "buy4/cart11/fav11 positives insufficient for sparse-task ranking"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-readout-retrieval](#phase-recurrence-readout-retrieval)。
- **后续假设**：保留全部10组用户配对和同卡延迟的后处理结果；task对shared的相对改善及固定权重诊断，不能替代对原循环的重训比较。
- **证据**：`qformer-loop/taobao-transfer/query-conditioning/reports/geometry-interpretation.json`（SHA256前缀 `259497f8f88e`）；`qformer-loop/taobao-transfer/query-conditioning/RESULTS.md`（SHA256前缀 `f12809256605`）

<a id="t101"></a>
### T101 · `taobao-query-parts-paired-5514-q4-q8`

- **方法**：Same-weight second-query-only Shuffle/Zero and exact identity control; rec-harness paired analyzer
- **原假设**：Entry and first-work sample-specific information both contribute to second query
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seeds":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "state": "completed_audited",
  "measurements": [
    {
      "candidate": "ctx_q4_entry_shuffle",
      "intervention": "entry_shuffle",
      "metric": "buy_weighted_bce",
      "delta_mean": 0.00010322366882255501,
      "block_bootstrap_ci95": [
        7.508807025863665e-05,
        0.00013478731834673464
      ]
    },
    {
      "candidate": "ctx_q4_entry_shuffle",
      "intervention": "entry_shuffle",
      "metric": "cart_weighted_bce",
      "delta_mean": -7.90021933219134e-05,
      "block_bootstrap_ci95": [
        -0.0009040931441227132,
        0.00033618987711953627
      ]
    },
    {
      "candidate": "ctx_q4_entry_shuffle",
      "intervention": "entry_shuffle",
      "metric": "fav_weighted_bce",
      "delta_mean": 0.00010225767597654927,
      "block_bootstrap_ci95": [
        4.203166645680115e-05,
        0.00016261947524946632
      ]
    },
    {
      "candidate": "ctx_q4_entry_shuffle",
      "intervention": "entry_shuffle",
      "metric": "is_click_weighted_bce",
      "delta_mean": 6.995295965039079e-05,
      "block_bootstrap_ci95": [
        -0.0007760466884264117,
        0.000780421539916113
      ]
    },
    {
      "candidate": "ctx_q4_entry_zero",
      "intervention": "entry_zero",
      "metric": "buy_weighted_bce",
      "delta_mean": 1.5233844713850135e-05,
      "block_bootstrap_ci95": [
        7.507840350677737e-06,
        2.2105633764975693e-05
      ]
    },
    {
      "candidate": "ctx_q4_entry_zero",
      "intervention": "entry_zero",
      "metric": "cart_weighted_bce",
      "delta_mean": -7.41995839757143e-05,
      "block_bootstrap_ci95": [
        -0.0003058863644775795,
        5.960912540771074e-05
      ]
    },
    {
      "candidate": "ctx_q4_entry_zero",
      "intervention": "entry_zero",
      "metric": "fav_weighted_bce",
      "delta_mean": 6.652570306871963e-05,
      "block_bootstrap_ci95": [
        3.319989464640376e-05,
        9.929339233691296e-05
      ]
    },
    {
      "candidate": "ctx_q4_entry_zero",
      "intervention": "entry_zero",
      "metric": "is_click_weighted_bce",
      "delta_mean": -0.00019721081513485828,
      "block_bootstrap_ci95": [
        -0.0006776490425721724,
        0.00023718459926114742
      ]
    },
    {
      "candidate": "ctx_q4_work_shuffle",
      "intervention": "work_shuffle",
      "metric": "buy_weighted_bce",
      "delta_mean": 0.0002368065095239089,
      "block_bootstrap_ci95": [
        0.00017972361730280394,
        0.0002927564655245339
      ]
    },
    {
      "candidate": "ctx_q4_work_shuffle",
      "intervention": "work_shuffle",
      "metric": "cart_weighted_bce",
      "delta_mean": 0.0001845591582754636,
      "block_bootstrap_ci95": [
        -0.0008353069689764805,
        0.0007853724668506576
      ]
    },
    {
      "candidate": "ctx_q4_work_shuffle",
      "intervention": "work_shuffle",
      "metric": "fav_weighted_bce",
      "delta_mean": 0.00019008358445869847,
      "block_bootstrap_ci95": [
        4.213911347613627e-05,
        0.00033114444827684003
      ]
    },
    {
      "candidate": "ctx_q4_work_shuffle",
      "intervention": "work_shuffle",
      "metric": "is_click_weighted_bce",
      "delta_mean": 0.0014694437878683168,
      "block_bootstrap_ci95": [
        6.21934410904698e-05,
        0.00293978685656828
      ]
    },
    {
      "candidate": "ctx_q4_work_zero",
      "intervention": "work_zero",
      "metric": "buy_weighted_bce",
      "delta_mean": 1.6203058887456045e-05,
      "block_bootstrap_ci95": [
        2.3576129412929878e-06,
        3.0149602649786403e-05
      ]
    },
    {
      "candidate": "ctx_q4_work_zero",
      "intervention": "work_zero",
      "metric": "cart_weighted_bce",
      "delta_mean": 0.00015465545200326475,
      "block_bootstrap_ci95": [
        -3.066116576941142e-05,
        0.00048413645943697986
      ]
    },
    {
      "candidate": "ctx_q4_work_zero",
      "intervention": "work_zero",
      "metric": "fav_weighted_bce",
      "delta_mean": 8.059902580084013e-05,
      "block_bootstrap_ci95": [
        -1.3053473344222491e-05,
        0.00018748527861611546
      ]
    },
    {
      "candidate": "ctx_q4_work_zero",
      "intervention": "work_zero",
      "metric": "is_click_weighted_bce",
      "delta_mean": 0.002028667005275599,
      "block_bootstrap_ci95": [
        0.0010638734406405836,
        0.003097040318449612
      ]
    },
    {
      "candidate": "explore_dual_no_feedback_entry_shuffle",
      "intervention": "entry_shuffle",
      "metric": "buy_weighted_bce",
      "delta_mean": 0.00018394704451901007,
      "block_bootstrap_ci95": [
        0.0001356681158315647,
        0.00023797622792157724
      ]
    },
    {
      "candidate": "explore_dual_no_feedback_entry_shuffle",
      "intervention": "entry_shuffle",
      "metric": "cart_weighted_bce",
      "delta_mean": 0.00024698168089309957,
      "block_bootstrap_ci95": [
        -9.189675200456036e-05,
        0.000504600459304319
      ]
    },
    {
      "candidate": "explore_dual_no_feedback_entry_shuffle",
      "intervention": "entry_shuffle",
      "metric": "fav_weighted_bce",
      "delta_mean": 0.00034123913003204954,
      "block_bootstrap_ci95": [
        0.0001407856293356885,
        0.00048086608158719017
      ]
    },
    {
      "candidate": "explore_dual_no_feedback_entry_shuffle",
      "intervention": "entry_shuffle",
      "metric": "is_click_weighted_bce",
      "delta_mean": -8.751440233958306e-06,
      "block_bootstrap_ci95": [
        -0.0006529853809771286,
        0.0006773977692427321
      ]
    },
    {
      "candidate": "explore_dual_no_feedback_entry_zero",
      "intervention": "entry_zero",
      "metric": "buy_weighted_bce",
      "delta_mean": 7.096335985582e-05,
      "block_bootstrap_ci95": [
        4.662615359888666e-05,
        0.00010168994404467057
      ]
    },
    {
      "candidate": "explore_dual_no_feedback_entry_zero",
      "intervention": "entry_zero",
      "metric": "cart_weighted_bce",
      "delta_mean": -9.861852690334089e-05,
      "block_bootstrap_ci95": [
        -0.0007958505244350993,
        0.00030972490816961204
      ]
    },
    {
      "candidate": "explore_dual_no_feedback_entry_zero",
      "intervention": "entry_zero",
      "metric": "fav_weighted_bce",
      "delta_mean": 0.00011084138559828756,
      "block_bootstrap_ci95": [
        2.7957482566930375e-05,
        0.00017063121837582472
      ]
    },
    {
      "candidate": "explore_dual_no_feedback_entry_zero",
      "intervention": "entry_zero",
      "metric": "is_click_weighted_bce",
      "delta_mean": -0.000177114150159786,
      "block_bootstrap_ci95": [
        -0.00047361428110452293,
        0.0001056542308729102
      ]
    },
    {
      "candidate": "explore_dual_no_feedback_work_shuffle",
      "intervention": "work_shuffle",
      "metric": "buy_weighted_bce",
      "delta_mean": 0.0007220228317005084,
      "block_bootstrap_ci95": [
        0.0006023942567998275,
        0.0008470590392322846
      ]
    },
    {
      "candidate": "explore_dual_no_feedback_work_shuffle",
      "intervention": "work_shuffle",
      "metric": "cart_weighted_bce",
      "delta_mean": 0.0008639300245093999,
      "block_bootstrap_ci95": [
        -0.00016524839852308508,
        0.0017081975512065505
      ]
    },
    {
      "candidate": "explore_dual_no_feedback_work_shuffle",
      "intervention": "work_shuffle",
      "metric": "fav_weighted_bce",
      "delta_mean": 0.0010238184330126816,
      "block_bootstrap_ci95": [
        0.0005848696097573456,
        0.001359723750542884
      ]
    },
    {
      "candidate": "explore_dual_no_feedback_work_shuffle",
      "intervention": "work_shuffle",
      "metric": "is_click_weighted_bce",
      "delta_mean": 0.0014421338985878896,
      "block_bootstrap_ci95": [
        0.0001673729240958238,
        0.0028393083445856296
      ]
    },
    {
      "candidate": "explore_dual_no_feedback_work_zero",
      "intervention": "work_zero",
      "metric": "buy_weighted_bce",
      "delta_mean": 0.00011410271891606406,
      "block_bootstrap_ci95": [
        8.973672189841942e-05,
        0.00014242424719643475
      ]
    },
    {
      "candidate": "explore_dual_no_feedback_work_zero",
      "intervention": "work_zero",
      "metric": "cart_weighted_bce",
      "delta_mean": 0.00015682880408400323,
      "block_bootstrap_ci95": [
        -0.00018028028421220962,
        0.0004380831744174761
      ]
    },
    {
      "candidate": "explore_dual_no_feedback_work_zero",
      "intervention": "work_zero",
      "metric": "fav_weighted_bce",
      "delta_mean": 0.00010652014145709216,
      "block_bootstrap_ci95": [
        -7.961267352722941e-05,
        0.0002223363988902816
      ]
    },
    {
      "candidate": "explore_dual_no_feedback_work_zero",
      "intervention": "work_zero",
      "metric": "is_click_weighted_bce",
      "delta_mean": 0.000990215260449554,
      "block_bootstrap_ci95": [
        0.00042895371833578456,
        0.0016041948034050233
      ]
    }
  ],
  "conclusion": "Preserve entry and first-work information; their example-specific contributions matter for fav/buy in both checkpoints; work shuffle additionally harms click. Cart remains inconclusive.",
  "scope": "Fixed stratified subset and checkpoints, CPU float32, user bootstrap conditional on selected subset, not full-population selection variance or training-seed variance",
  "training_updates": 0
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-readout-retrieval](#phase-recurrence-readout-retrieval)。
- **后续假设**：保留全部10组用户配对和同卡延迟的后处理结果；task对shared的相对改善及固定权重诊断，不能替代对原循环的重训比较。
- **证据**：`qformer-loop/taobao-transfer/query-conditioning/reports/paired-interpretation.json`（SHA256前缀 `8a3edfd91f49`）；`qformer-loop/taobao-transfer/query-conditioning/reports/paired-query-parts/toolkit/perturbation_report.json`（SHA256前缀 `fbfa78986085`）

<a id="t169"></a>
### T169 · `taobao-second-read-attention-originals-5514`

- **方法**：Original SDPA in native/identity/reuse; replace only second w_q input with first entry; original q2 interaction/residual preserved
- **原假设**：Determine whether second-round attention redistribution affects predictions independently of query interaction and residual
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seeds":[],"gpu_hours":0}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "samples_each": 5514,
  "users": 4647,
  "tasks": {
    "ctx_q4": {
      "is_click": {
        "native_ipw_bce": 0.18094589604724506,
        "identity_ipw_bce": 0.18094589604724506,
        "reuse_ipw_bce": 0.1811099522098907,
        "reuse_minus_identity": {
          "estimate": 0.00016405616264536792,
          "descriptive_ci95": [
            -8.458306429502778e-05,
            0.0004569532453446214
          ],
          "directional_ci99375": [
            -0.00016665589821951866,
            0.0005859963680992394
          ]
        },
        "reuse_minus_native": {
          "estimate": 0.00016405616264536792,
          "descriptive_ci95": [
            -8.458306429502778e-05,
            0.0004569532453446214
          ],
          "directional_ci99375": [
            -0.00016665589821951866,
            0.0005859963680992394
          ]
        },
        "epsilon": 1e-06,
        "ipw_mean_absolute_identity_drift": 0.0,
        "direction": "inconclusive",
        "action": "abstain_from_attention_specific_model_edits",
        "exact_row_bce_null": false,
        "directional_interval_within_epsilon": false,
        "native_consistency_checked": true
      },
      "cart": {
        "native_ipw_bce": 0.018065485271994586,
        "identity_ipw_bce": 0.018065485271994586,
        "reuse_ipw_bce": 0.018091870975892373,
        "reuse_minus_identity": {
          "estimate": 2.6385703897761454e-05,
          "descriptive_ci95": [
            -2.185444086053926e-05,
            6.294701653637133e-05
          ],
          "directional_ci99375": [
            -4.564488380140991e-05,
            7.452098516804465e-05
          ]
        },
        "reuse_minus_native": {
          "estimate": 2.6385703897761454e-05,
          "descriptive_ci95": [
            -2.185444086053926e-05,
            6.294701653637133e-05
          ],
          "directional_ci99375": [
            -4.564488380140991e-05,
            7.452098516804465e-05
          ]
        },
        "epsilon": 1e-06,
        "ipw_mean_absolute_identity_drift": 0.0,
        "direction": "inconclusive",
        "action": "abstain_from_attention_specific_model_edits",
        "exact_row_bce_null": false,
        "directional_interval_within_epsilon": false,
        "native_consistency_checked": true
      },
      "fav": {
        "native_ipw_bce": 0.013235783087343612,
        "identity_ipw_bce": 0.013235783087343612,
        "reuse_ipw_bce": 0.013362807415134967,
        "reuse_minus_identity": {
          "estimate": 0.00012702432779136035,
          "descriptive_ci95": [
            2.5421693398959408e-05,
            0.00025115045855211556
          ],
          "directional_ci99375": [
            -5.954712104536138e-06,
            0.00031352640927223953
          ]
        },
        "reuse_minus_native": {
          "estimate": 0.00012702432779136035,
          "descriptive_ci95": [
            2.5421693398959408e-05,
            0.00025115045855211556
          ],
          "directional_ci99375": [
            -5.954712104536138e-06,
            0.00031352640927223953
          ]
        },
        "epsilon": 1e-06,
        "ipw_mean_absolute_identity_drift": 0.0,
        "direction": "inconclusive",
        "action": "abstain_from_attention_specific_model_edits",
        "exact_row_bce_null": false,
        "directional_interval_within_epsilon": false,
        "native_consistency_checked": true
      },
      "buy": {
        "native_ipw_bce": 0.006609427434078904,
        "identity_ipw_bce": 0.006609427434078904,
        "reuse_ipw_bce": 0.0066237390903886315,
        "reuse_minus_identity": {
          "estimate": 1.4311656309733042e-05,
          "descriptive_ci95": [
            9.73809531246234e-06,
            1.9220549705229177e-05
          ],
          "directional_ci99375": [
            8.107827581993967e-06,
            2.1101592077506038e-05
          ]
        },
        "reuse_minus_native": {
          "estimate": 1.4311656309733042e-05,
          "descriptive_ci95": [
            9.73809531246234e-06,
            1.9220549705229177e-05
          ],
          "directional_ci99375": [
            8.107827581993967e-06,
            2.1101592077506038e-05
          ]
        },
        "epsilon": 1e-06,
        "ipw_mean_absolute_identity_drift": 0.0,
        "direction": "positive",
        "action": "preserve_second_redistribution_for_this_frozen_checkpoint_task",
        "exact_row_bce_null": false,
        "directional_interval_within_epsilon": false,
        "native_consistency_checked": true
      }
    },
    "explore_dual_no_feedback": {
      "is_click": {
        "native_ipw_bce": 0.18145874781177582,
        "identity_ipw_bce": 0.18145874781177582,
        "reuse_ipw_bce": 0.1817920781338053,
        "reuse_minus_identity": {
          "estimate": 0.0003333303220294364,
          "descriptive_ci95": [
            4.672020819842186e-05,
            0.0006292641523202294
          ],
          "directional_ci99375": [
            -5.125708461276574e-05,
            0.000760694329372942
          ]
        },
        "reuse_minus_native": {
          "estimate": 0.0003333303220294364,
          "descriptive_ci95": [
            4.672020819842186e-05,
            0.0006292641523202294
          ],
          "directional_ci99375": [
            -5.125708461276574e-05,
            0.000760694329372942
          ]
        },
        "epsilon": 1e-06,
        "ipw_mean_absolute_identity_drift": 0.0,
        "direction": "inconclusive",
        "action": "abstain_from_attention_specific_model_edits",
        "exact_row_bce_null": false,
        "directional_interval_within_epsilon": false,
        "native_consistency_checked": true
      },
      "cart": {
        "native_ipw_bce": 0.01801679791264649,
        "identity_ipw_bce": 0.01801679791264649,
        "reuse_ipw_bce": 0.018317192767206936,
        "reuse_minus_identity": {
          "estimate": 0.00030039485456040453,
          "descriptive_ci95": [
            -0.0001975344196785441,
            0.001081134147214237
          ],
          "directional_ci99375": [
            -0.00026331076419997495,
            0.0014940796627174852
          ]
        },
        "reuse_minus_native": {
          "estimate": 0.00030039485456040453,
          "descriptive_ci95": [
            -0.0001975344196785441,
            0.001081134147214237
          ],
          "directional_ci99375": [
            -0.00026331076419997495,
            0.0014940796627174852
          ]
        },
        "epsilon": 1e-06,
        "ipw_mean_absolute_identity_drift": 0.0,
        "direction": "inconclusive",
        "action": "abstain_from_attention_specific_model_edits",
        "exact_row_bce_null": false,
        "directional_interval_within_epsilon": false,
        "native_consistency_checked": true
      },
      "fav": {
        "native_ipw_bce": 0.01332012841318057,
        "identity_ipw_bce": 0.01332012841318057,
        "reuse_ipw_bce": 0.013267225495969303,
        "reuse_minus_identity": {
          "estimate": -5.290291721130152e-05,
          "descriptive_ci95": [
            -9.829220823064127e-05,
            -2.530767665451306e-06
          ],
          "directional_ci99375": [
            -0.000113471932076058,
            2.206656817021924e-05
          ]
        },
        "reuse_minus_native": {
          "estimate": -5.290291721130152e-05,
          "descriptive_ci95": [
            -9.829220823064127e-05,
            -2.530767665451306e-06
          ],
          "directional_ci99375": [
            -0.000113471932076058,
            2.206656817021924e-05
          ]
        },
        "epsilon": 1e-06,
        "ipw_mean_absolute_identity_drift": 0.0,
        "direction": "inconclusive",
        "action": "abstain_from_attention_specific_model_edits",
        "exact_row_bce_null": false,
        "directional_interval_within_epsilon": false,
        "native_consistency_checked": true
      },
      "buy": {
        "native_ipw_bce": 0.006566427378615031,
        "identity_ipw_bce": 0.006566427378615031,
        "reuse_ipw_bce": 0.0065513586568363745,
        "reuse_minus_identity": {
          "estimate": -1.5068721778670674e-05,
          "descriptive_ci95": [
            -3.23440543160087e-05,
            2.4604153342927415e-06
          ],
          "directional_ci99375": [
            -3.8730893810857566e-05,
            8.949244607792113e-06
          ]
        },
        "reuse_minus_native": {
          "estimate": -1.5068721778670674e-05,
          "descriptive_ci95": [
            -3.23440543160087e-05,
            2.4604153342927415e-06
          ],
          "directional_ci99375": [
            -3.8730893810857566e-05,
            8.949244607792113e-06
          ]
        },
        "epsilon": 1e-06,
        "ipw_mean_absolute_identity_drift": 0.0,
        "direction": "inconclusive",
        "action": "abstain_from_attention_specific_model_edits",
        "exact_row_bce_null": false,
        "directional_interval_within_epsilon": false,
        "native_consistency_checked": true
      }
    }
  },
  "bootstrap_replicates": 10000,
  "directional_ci": 0.99375,
  "epsilon": 1e-06,
  "identity_exact": true,
  "actual_allpadding_identity_exact": true,
  "tool_six_gates_passed": true
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-readout-retrieval](#phase-recurrence-readout-retrieval)。
- **后续假设**：保留全部10组用户配对和同卡延迟的后处理结果；task对shared的相对改善及固定权重诊断，不能替代对原循环的重训比较。
- **证据**：`qformer-loop/taobao-transfer/memory/tool-evolution/candidates/taobao-second-read-attention-v1/artifacts/outcome.json`（SHA256前缀 `f822a782617a`）；`qformer-loop/taobao-transfer/memory/tool-evolution/candidates/taobao-second-read-attention-v1/artifacts/shadow/aggregate.json`（SHA256前缀 `5e0cc948bfb3`）；`qformer-loop/taobao-transfer/memory/tool-evolution/candidates/taobao-second-read-attention-v1/artifacts/shadow-validation.json`（SHA256前缀 `f87d94308869`）；`qformer-loop/taobao-transfer/memory/tool-evolution/candidates/taobao-second-read-attention-v1/artifacts/memory-consolidation-status.json`（SHA256前缀 `4886ff2422bf`）

<a id="t170"></a>
### T170 · `taobao-recurrence-task-effects-saved-validation`

- **方法**：Replay historical q4 trained-depth 500 user draws and add q8 frozen-depth paired intervals; retain all four AUC/BCE effects and all six task contrasts.
- **原假设**：Predictive benefit of a second cycle may differ across task heads; distinguish differences in magnitude from exclusive task utility.
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_runs":0,"new_training_seeds":[],"model_forwards":0,"gpu_hours":0,"bootstrap_repetitions_each":500,"bootstrap_seed":20260906}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "q4_delta_auc": {
    "is_click": 5.341478532294719e-05,
    "cart": 0.0027787112829202876,
    "fav": 0.0010524773672074828,
    "buy": 0.003956897505808188
  },
  "q8_delta_auc": {
    "is_click": 0.010860654502252487,
    "cart": 0.013065211709695657,
    "fav": 0.012096708103326748,
    "buy": 0.013753872259069433
  },
  "q4_click_delta_bce": 0.000163370153419059,
  "q4_click_bce_ci95": [
    0.00010765296501773477,
    0.00022282304819187275
  ],
  "direct_auc_contrasts": {
    "q4_trained": [
      {
        "first_task": "is_click",
        "second_task": "cart",
        "sign": "first effect minus second effect",
        "delta_auc": -0.0027252964975973404,
        "ci95_auc": [
          -0.0045223999597569595,
          -0.0010514665313744466
        ],
        "delta_bce": 0.00015971645345358635,
        "ci95_bce": [
          6.68007504797972e-05,
          0.0002502592073984823
        ]
      },
      {
        "first_task": "is_click",
        "second_task": "buy",
        "sign": "first effect minus second effect",
        "delta_auc": -0.0039034827204852407,
        "ci95_auc": [
          -0.006529910768944316,
          -0.0015277135121195885
        ],
        "delta_bce": 0.00017012846290499225,
        "ci95_bce": [
          9.470652524180969e-05,
          0.0002429032208488879
        ]
      }
    ],
    "q8_frozen": [
      {
        "first_task": "is_click",
        "second_task": "cart",
        "sign": "first effect minus second effect",
        "delta_auc": -0.0022045572074431696,
        "ci95_auc": [
          -0.003986746174442055,
          -0.0006855924601310963
        ],
        "delta_bce": -0.002490392766784844,
        "ci95_bce": [
          -0.0026361725597461643,
          -0.002333991413218153
        ]
      },
      {
        "first_task": "is_click",
        "second_task": "buy",
        "sign": "first effect minus second effect",
        "delta_auc": -0.002893217756816946,
        "ci95_auc": [
          -0.005305512896996029,
          -0.0009480822150964473
        ],
        "delta_bce": -0.0026088349479567556,
        "ci95_bce": [
          -0.0027389334566999624,
          -0.0024464906463447854
        ]
      }
    ]
  },
  "all_other_auc_contrast_intervals": "cross_zero_in_both_families",
  "q8_all_four_auc_positive_bce_negative_ci95": true,
  "q4_legacy_auc_intervals_exact": true,
  "measured_cost": {
    "wall_seconds": 199.51010938733816,
    "cpu_seconds": 565.0630770559999,
    "peak_rss_bytes": 1059528704
  }
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 recurrence-readout-retrieval](#phase-recurrence-readout-retrieval)。
- **后续假设**：保留全部10组用户配对和同卡延迟的后处理结果；task对shared的相对改善及固定权重诊断，不能替代对原循环的重训比较。
- **证据**：`qformer-loop/taobao-transfer/recurrence-task-effects/RESULTS.md`（SHA256前缀 `4801871121d8`）；`qformer-loop/taobao-transfer/recurrence-task-effects/reports/interpretation.json`（SHA256前缀 `f8c1a4808873`）；`qformer-loop/taobao-transfer/recurrence-task-effects/result-review.json`（SHA256前缀 `5981f5cc870f`）；`qformer-loop/taobao-transfer/recurrence-task-effects/memory-consolidation-status.json`（SHA256前缀 `cb02ab0d9557`）

<a id="t171"></a>
### T171 · `role_q4_query_norm-s20262027-20260909T063952376231Z`

- **方法**：本次快照发现的未归档正式运行，配置来自manifest；不改写主线程memory。；设置 {"model":"QFormerCross8RetrievalUntied","token_dim":384,"num_queries":4,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"retrieval_untie":"q_norm"}
- **原假设**：q4/q8×共享读出/任务读出/query norm独立/query投影独立，8组固定预算。
- **状态／预算**：原记录 `unconsolidated`；{}。
- **结果**：macroAUC=0.827361315；macroLogloss=0.054950230；is_click AUC/Logloss=0.665808518/0.182087872；cart AUC/Logloss=0.865759894/0.017668768；fav AUC/Logloss=0.880814630/0.013352376；buy AUC/Logloss=0.897062219/0.006691903。
- **开销**：参数 51724420；训练含验证 2002.548456221819 秒；峰值CUDA 22593746944 bytes；更新 2193。
- **最新状态**：已保存完整2193步结果；worker退出码 `0`；整轮配对/延迟/诊断若未完成，本条不宣称已最终审计或已成为新最佳。
- **归因**：这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-readout-retrieval](#phase-recurrence-readout-retrieval)。
- **后续假设**：保留全部10组用户配对和同卡延迟的后处理结果；task对shared的相对改善及固定权重诊断，不能替代对原循环的重训比较。
- **证据**：`qformer-loop/taobao-transfer/recurrence-readout-retrieval/runs/role_q4_query_norm-s20262027-20260909T063952376231Z/manifest.json`（SHA256前缀 `a881b9f8f9fb`）；`qformer-loop/taobao-transfer/recurrence-readout-retrieval/runs/role_q4_query_norm-s20262027-20260909T063952376231Z/result.json`（SHA256前缀 `25a2dbd257ca`）

<a id="t172"></a>
### T172 · `role_q4_query_projection-s20262027-20260909T063952342021Z`

- **方法**：本次快照发现的未归档正式运行，配置来自manifest；不改写主线程memory。；设置 {"model":"QFormerCross8RetrievalUntied","token_dim":384,"num_queries":4,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"retrieval_untie":"w_q"}
- **原假设**：q4/q8×共享读出/任务读出/query norm独立/query投影独立，8组固定预算。
- **状态／预算**：原记录 `unconsolidated`；{}。
- **结果**：macroAUC=0.826303350；macroLogloss=0.055005456；is_click AUC/Logloss=0.664452061/0.182330172；cart AUC/Logloss=0.864694079/0.017680958；fav AUC/Logloss=0.880271713/0.013322615；buy AUC/Logloss=0.895795546/0.006688081。
- **开销**：参数 51872068；训练含验证 1996.9452750608325 秒；峰值CUDA 22595128320 bytes；更新 2193。
- **最新状态**：已保存完整2193步结果；worker退出码 `0`；整轮配对/延迟/诊断若未完成，本条不宣称已最终审计或已成为新最佳。
- **归因**：这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-readout-retrieval](#phase-recurrence-readout-retrieval)。
- **后续假设**：保留全部10组用户配对和同卡延迟的后处理结果；task对shared的相对改善及固定权重诊断，不能替代对原循环的重训比较。
- **证据**：`qformer-loop/taobao-transfer/recurrence-readout-retrieval/runs/role_q4_query_projection-s20262027-20260909T063952342021Z/manifest.json`（SHA256前缀 `114dad14cc4f`）；`qformer-loop/taobao-transfer/recurrence-readout-retrieval/runs/role_q4_query_projection-s20262027-20260909T063952342021Z/result.json`（SHA256前缀 `c549c3ddb84f`）

<a id="t173"></a>
### T173 · `role_q4_read_shared-s20262027-20260909T063952359077Z`

- **方法**：本次快照发现的未归档正式运行，配置来自manifest；不改写主线程memory。；设置 {"model":"QFormerCross8DepthReadout","token_dim":384,"num_queries":4,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"depth_readout_mode":"shared"}
- **原假设**：q4/q8×共享读出/任务读出/query norm独立/query投影独立，8组固定预算。
- **状态／预算**：原记录 `unconsolidated`；{}。
- **结果**：macroAUC=0.826371326；macroLogloss=0.055005125；is_click AUC/Logloss=0.666479743/0.182023461；cart AUC/Logloss=0.862244926/0.017848891；fav AUC/Logloss=0.882142123/0.013360310；buy AUC/Logloss=0.894618512/0.006787836。
- **开销**：参数 51724229；训练含验证 2018.7086542248726 秒；峰值CUDA 22593282560 bytes；更新 2193。
- **最新状态**：已保存完整2193步结果；worker退出码 `0`；整轮配对/延迟/诊断若未完成，本条不宣称已最终审计或已成为新最佳。
- **归因**：这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-readout-retrieval](#phase-recurrence-readout-retrieval)。
- **后续假设**：保留全部10组用户配对和同卡延迟的后处理结果；task对shared的相对改善及固定权重诊断，不能替代对原循环的重训比较。
- **证据**：`qformer-loop/taobao-transfer/recurrence-readout-retrieval/runs/role_q4_read_shared-s20262027-20260909T063952359077Z/manifest.json`（SHA256前缀 `a7fd55495ee2`）；`qformer-loop/taobao-transfer/recurrence-readout-retrieval/runs/role_q4_read_shared-s20262027-20260909T063952359077Z/result.json`（SHA256前缀 `1df0ec422138`）

<a id="t174"></a>
### T174 · `role_q4_read_task-s20262027-20260909T063952341877Z`

- **方法**：本次快照发现的未归档正式运行，配置来自manifest；不改写主线程memory。；设置 {"model":"QFormerCross8DepthReadout","token_dim":384,"num_queries":4,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"depth_readout_mode":"task"}
- **原假设**：q4/q8×共享读出/任务读出/query norm独立/query投影独立，8组固定预算。
- **状态／预算**：原记录 `unconsolidated`；{}。
- **结果**：macroAUC=0.826192700；macroLogloss=0.055115317；is_click AUC/Logloss=0.664218981/0.182458808；cart AUC/Logloss=0.863259567/0.017874488；fav AUC/Logloss=0.880714709/0.013357703；buy AUC/Logloss=0.896577542/0.006770268。
- **开销**：参数 51724232；训练含验证 2028.1428565904498 秒；峰值CUDA 22593515008 bytes；更新 2193。
- **最新状态**：已保存完整2193步结果；worker退出码 `0`；整轮配对/延迟/诊断若未完成，本条不宣称已最终审计或已成为新最佳。
- **归因**：这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-readout-retrieval](#phase-recurrence-readout-retrieval)。
- **后续假设**：保留全部10组用户配对和同卡延迟的后处理结果；task对shared的相对改善及固定权重诊断，不能替代对原循环的重训比较。
- **证据**：`qformer-loop/taobao-transfer/recurrence-readout-retrieval/runs/role_q4_read_task-s20262027-20260909T063952341877Z/manifest.json`（SHA256前缀 `d830327bfadc`）；`qformer-loop/taobao-transfer/recurrence-readout-retrieval/runs/role_q4_read_task-s20262027-20260909T063952341877Z/result.json`（SHA256前缀 `0bdd26511d7d`）

<a id="t175"></a>
### T175 · `role_q8_query_norm-s20262027-20260909T063952342021Z`

- **方法**：本次快照发现的未归档正式运行，配置来自manifest；不改写主线程memory。；设置 {"model":"QFormerCross8RetrievalUntied","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"retrieval_untie":"q_norm"}
- **原假设**：q4/q8×共享读出/任务读出/query norm独立/query投影独立，8组固定预算。
- **状态／预算**：原记录 `unconsolidated`；{}。
- **结果**：macroAUC=0.824924790；macroLogloss=0.055020655；is_click AUC/Logloss=0.661277722/0.182503701；cart AUC/Logloss=0.865101287/0.017603831；fav AUC/Logloss=0.878193462/0.013330348；buy AUC/Logloss=0.895126689/0.006644742。
- **开销**：参数 81616516；训练含验证 2068.572003260255 秒；峰值CUDA 25231252992 bytes；更新 2193。
- **最新状态**：已保存完整2193步结果；worker退出码 `0`；整轮配对/延迟/诊断若未完成，本条不宣称已最终审计或已成为新最佳。
- **归因**：这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-readout-retrieval](#phase-recurrence-readout-retrieval)。
- **后续假设**：保留全部10组用户配对和同卡延迟的后处理结果；task对shared的相对改善及固定权重诊断，不能替代对原循环的重训比较。
- **证据**：`qformer-loop/taobao-transfer/recurrence-readout-retrieval/runs/role_q8_query_norm-s20262027-20260909T063952342021Z/manifest.json`（SHA256前缀 `3dc43253d18a`）；`qformer-loop/taobao-transfer/recurrence-readout-retrieval/runs/role_q8_query_norm-s20262027-20260909T063952342021Z/result.json`（SHA256前缀 `189508d033af`）

<a id="t176"></a>
### T176 · `role_q8_query_projection-s20262027-20260909T063952355394Z`

- **方法**：本次快照发现的未归档正式运行，配置来自manifest；不改写主线程memory。；设置 {"model":"QFormerCross8RetrievalUntied","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"retrieval_untie":"w_q"}
- **原假设**：q4/q8×共享读出/任务读出/query norm独立/query投影独立，8组固定预算。
- **状态／预算**：原记录 `unconsolidated`；{}。
- **结果**：macroAUC=0.826579499；macroLogloss=0.054980444；is_click AUC/Logloss=0.666398604/0.182107476；cart AUC/Logloss=0.864931376/0.017742203；fav AUC/Logloss=0.880058502/0.013363903；buy AUC/Logloss=0.894929512/0.006708194。
- **开销**：参数 81764164；训练含验证 2067.8856638893485 秒；峰值CUDA 25232643584 bytes；更新 2193。
- **最新状态**：已保存完整2193步结果；worker退出码 `0`；整轮配对/延迟/诊断若未完成，本条不宣称已最终审计或已成为新最佳。
- **归因**：这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-readout-retrieval](#phase-recurrence-readout-retrieval)。
- **后续假设**：保留全部10组用户配对和同卡延迟的后处理结果；task对shared的相对改善及固定权重诊断，不能替代对原循环的重训比较。
- **证据**：`qformer-loop/taobao-transfer/recurrence-readout-retrieval/runs/role_q8_query_projection-s20262027-20260909T063952355394Z/manifest.json`（SHA256前缀 `4a4a02139aae`）；`qformer-loop/taobao-transfer/recurrence-readout-retrieval/runs/role_q8_query_projection-s20262027-20260909T063952355394Z/result.json`（SHA256前缀 `702c76c8d870`）

<a id="t177"></a>
### T177 · `role_q8_read_shared-s20262027-20260909T063952341972Z`

- **方法**：本次快照发现的未归档正式运行，配置来自manifest；不改写主线程memory。；设置 {"model":"QFormerCross8DepthReadout","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"depth_readout_mode":"shared"}
- **原假设**：q4/q8×共享读出/任务读出/query norm独立/query投影独立，8组固定预算。
- **状态／预算**：原记录 `unconsolidated`；{}。
- **结果**：macroAUC=0.821507140；macroLogloss=0.055301022；is_click AUC/Logloss=0.656986888/0.183528831；cart AUC/Logloss=0.861232609/0.017693754；fav AUC/Logloss=0.875557431/0.013336397；buy AUC/Logloss=0.892251632/0.006645107。
- **开销**：参数 81616325；训练含验证 2106.740642309189 秒；峰值CUDA 26863696896 bytes；更新 2193。
- **最新状态**：已保存完整2193步结果；worker退出码 `0`；整轮配对/延迟/诊断若未完成，本条不宣称已最终审计或已成为新最佳。
- **归因**：这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-readout-retrieval](#phase-recurrence-readout-retrieval)。
- **后续假设**：保留全部10组用户配对和同卡延迟的后处理结果；task对shared的相对改善及固定权重诊断，不能替代对原循环的重训比较。
- **证据**：`qformer-loop/taobao-transfer/recurrence-readout-retrieval/runs/role_q8_read_shared-s20262027-20260909T063952341972Z/manifest.json`（SHA256前缀 `782122fbcde4`）；`qformer-loop/taobao-transfer/recurrence-readout-retrieval/runs/role_q8_read_shared-s20262027-20260909T063952341972Z/result.json`（SHA256前缀 `ef53e775133e`）

<a id="t178"></a>
### T178 · `role_q8_read_task-s20262027-20260909T063952422184Z`

- **方法**：本次快照发现的未归档正式运行，配置来自manifest；不改写主线程memory。；设置 {"model":"QFormerCross8DepthReadout","token_dim":384,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"HeavyBall.PSGDKron","dense_optimizer_options":{"preconditioner_update_probability":1.0},"dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1,"depth_readout_mode":"task"}
- **原假设**：q4/q8×共享读出/任务读出/query norm独立/query投影独立，8组固定预算。
- **状态／预算**：原记录 `unconsolidated`；{}。
- **结果**：macroAUC=0.826662645；macroLogloss=0.055047103；is_click AUC/Logloss=0.664134201/0.182476860；cart AUC/Logloss=0.865666102/0.017717601；fav AUC/Logloss=0.880921936/0.013312326；buy AUC/Logloss=0.895928342/0.006681625。
- **开销**：参数 81616328；训练含验证 2111.6860541403294 秒；峰值CUDA 26863729664 bytes；更新 2193。
- **最新状态**：已保存完整2193步结果；worker退出码 `0`；整轮配对/延迟/诊断若未完成，本条不宣称已最终审计或已成为新最佳。
- **归因**：这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 recurrence-readout-retrieval](#phase-recurrence-readout-retrieval)。
- **后续假设**：保留全部10组用户配对和同卡延迟的后处理结果；task对shared的相对改善及固定权重诊断，不能替代对原循环的重训比较。
- **证据**：`qformer-loop/taobao-transfer/recurrence-readout-retrieval/runs/role_q8_read_task-s20262027-20260909T063952422184Z/manifest.json`（SHA256前缀 `77e04b020dab`）；`qformer-loop/taobao-transfer/recurrence-readout-retrieval/runs/role_q8_read_task-s20262027-20260909T063952422184Z/result.json`（SHA256前缀 `396dd4e39deb`）

## 跨实验归因与可证伪条件

以下保留已有lesson的限定范围与状态；`supported`是当时收录状态，不自动等于本报告完成了独立确认。历史要求增加seed的设想不作为当前执行授权；当前仍遵守不增加训练seed的约束。

| Lesson ID | 归因记录 | 状态 | 可证伪条件／后续假设 |
|---|---|---|---|
| taobao-transfer-r7-not-winner | KuaiRec r7 winner did not beat Cross8 on four-task Taobao; Cross8clip1 is current strongest reference, r7 retains a modest inference speed tradeoff. | provisional | Capacity/recurrence changes under the same Taobao budget may change the ranking |
| taobao-round2-width-not-whole-loop | Width384 single pass wins macroAUC; whole-history recurrence lowers macroAUC at widths256/384/512. Click supplies76.5% of best macro gain; latency rises23.5%. | provisional | 按本阶段固定对照检验；原lesson未另写可证伪条件。 |
| taobao-round3-local-damped-loop-wins | Tail half update increases all4taskAUC at unchanged parameters; macro delta0.001386 CI[0.000520,0.002246], inference+17.2%. Entire stack recurrence loses. | provisional | 按本阶段固定对照检验；原lesson未另写可证伪条件。 |
| taobao-round4-tail-sweep-no-gain | All6width/strength/depth candidates below t3_tail_half across4AUCs; extra passes and width512 increase latency. Keep previous winner and focus requested optimizer substitution. | provisional | 按本阶段固定对照检验；原lesson未另写可证伪条件。 |
| taobao-psgdkron-four-structures | PSGDKron improves macro AUC within all four structures. Width384 single pass is highest observed at 0.827466136; extra tail/post recurrence does not improve its point estimate. Within-architecture optimizer gain user-bootstrap intervals are positive for Cross8, width384 single and post-half, but cross zero for tail-half. | provisional | 按本阶段固定对照检验；原lesson未另写可证伪条件。 |
| taobao-q4-two-cycle-benefit | Matched q4 R2 exceeds R1 by 0.001960375; paired-user CI excludes zero. Keep R2 for this candidate; this does not show recurrence beats the best single-pass architecture. | provisional | A matched query-formation retraining control erases the benefit; bootstrap does not establish seed robustness |
| taobao-local-capacity-sweep-negative | Seven tested FFN/entry capacity variants failed to exceed their matching q4/q8 two-cycle references; lower priority of more nearby FFN scalar sweeps. | supported | A controlled capacity reallocation exceeds the matched baseline under unchanged protocol |
| taobao-query-information-not-yet-diagnosed | Removing dynamic second-work retrieval and tested context reread both lose quality; these outcomes do not identify an optimal second query. Measure first-work information contribution before further formula sweeps. | provisional | Same-weight paired interventions show no sample-specific dependence on first_work or entry |
| second-query-preserve-both-sources | Preserve entry and first-work information; their example-specific contributions matter for fav/buy in both checkpoints; work shuffle additionally harms click. Cart remains inconclusive. | supported | Repetition under full validation or independent subset reverses paired dependence; this lesson does not predict adapter gain |
| taobao-second-cycle-memory-concatenation-negative | All eight tested second-cycle memory concatenations lose against matching q4/q8 references; retain original memory paths for query-correction tests. | supported | A controlled different routing design exceeds its matched reference; these outcomes do not establish query correction will help. |
| taobao-query-correction-source-depends-on-query-count | Among completed candidates, work has the highest q4 point estimate while entry has the highest q8 point estimate. Their respective improvements over same-size mean controls have positive conditional user-bootstrap intervals; this does not establish a transferable source ranking or a gain over the original recurrence. | provisional | A controlled query-count/source comparison reverses the observed source ranking; current evidence does not identify a transferable optimal source. |
| taobao-query-q4-work-near-reference-without-demonstrated-gain | Retain q4 work correction as a near-reference optimization candidate: macro AUC delta -0.000006788835 versus ctx_q4, conditional CI [-0.000812322970, 0.000722271955]. All eight completed added-capacity variants do not improve their matched recurrence point estimates; extra capacity alone does not guarantee a gain. | provisional | A later audited matched result establishes a material improvement or regression; current evidence neither promotes the incumbent nor identifies the retrieval mechanism. |
| correction-route-dependence-is-checkpoint-specific | Fixed-checkpoint q4 retrieval correction helps fav/buy while residual correction hurts them; q8 retrieval does not repeat the same benefit. Subsequent matched route-restricted retraining did not improve all-route or original recurrence. Preserve the diagnostic observation, lower the associated architecture priority. | provisional | The fixed-checkpoint intervention fails to reproduce under identical checkpoint/subset/protocol; do not interpret it as a prediction of retraining gain. |
| taobao-difference-only-additive-query-correction-negative | Difference-conditioned rank16 whole-input additive correction is below the corresponding mean-conditioned control in q4 and q8; both fixed-checkpoint macroAUC intervals are negative. Lower this specific source/offset design's priority. | supported | A matched difference-only additive correction exceeds its mean-conditioned control under the same budget |
| taobao-query-route-restriction-no-retraining-gain | All eight tested route restrictions are below matching all-route and original-recurrence point estimates; seven all-route intervals are negative, q4 retrieval crosses zero. No new candidate meets the predeclared quality/compute queue rule. | supported | A controlled route-restricted retraining demonstrates useful matched quality/compute benefit |
| taobao-query-mixing-bounded-gate-outcomes | Eight bounded rank16 mixing outcomes: 0 pass the predeclared near-reference quality/compute queue rule; 0 positive-quality candidates exceed its compute limit. This queue policy is not equivalence. | provisional | Later matched training and compute evidence changes the observed quality/compute conclusions |
| taobao-query-mixing-conditioning-versus-static-control | Of four matched joint-minus-constant pairs, 0 also have positive lower bounds against the original recurrence. All four pairs and intervals are retained; fixed-checkpoint gate diagnostics are separate evidence. | provisional | Matched training reverses the conditional comparisons; shuffle sensitivity alone cannot establish training benefit |
| taobao-q4-channel-gate-correspondence-not-demonstrated-useful | On the prospectively fixed q4 channel joint checkpoint, zeroing and shuffling the learned gate both reduce fav IPW BCE with negative conditional intervals; click/cart/buy intervals cross zero. Constant shuffle is exact, and constant zero/click shows only a tiny positive conditional effect. Nonzero gate variance does not establish useful sample conditioning; preserve full-training comparisons separately. | provisional | A separately validated matched checkpoint/subset intervention changes these task-specific findings; training gain must be established by training comparisons |
| taobao-q4-buy-second-attention-redistribution | For original ctx_q4 on the fixed validation subset, replacing only second-round P2 by P1 increases buy IPW BCE by 1.4311656309733042e-5; the precommitted 99.375% interval remains above epsilon1e-6. Preserve second attention redistribution in this checkpoint/task; no reuse-training branch is supported. | provisional | A separately validated matching probability-only intervention changes the supported task direction; new architecture gains require the fixed training and full-validation comparison |
| taobao-recurrence-benefit-depends-on-task-and-comparison | In each fixed q4 trained-depth and q8 frozen-depth comparison, cart/buy AUC effects exceed click at exploratory 95% paired task-contrast intervals. Yet the q8 checkpoint benefits from its whole second cycle on all four AUC/BCE heads, while q4 click BCE worsens. Task/metric-specific benefit is a useful research clue; only-buy or task-label-only rules are unsupported. | provisional | A matched task-effect analysis changes these fixed-scope contrasts; any task-readout gain must beat the shared-mixture and original readout controls under the existing training and compute policy. |

## 失败登记补充

以下是failures.jsonl的独立避免清单，可能与上方episode指向同一事件；不能相加作为独立实验次数。

### `alternative-missing-accumulation`

- **方法／触发**：model_contract_error
- **结果／归因／后续避免**：

```json
{
  "failure_id": "alternative-missing-accumulation",
  "kind": "model_contract_error",
  "training_updates": 0,
  "artifact": "round2/reports/alternative-cpu-initial-failure.txt",
  "repair": "Added accumulation_steps, CPU5 subsequently passed"
}
```

### `alternative-poststep-gradient-fixture`

- **方法／触发**：test_fixture_error
- **结果／归因／后续避免**：

```json
{
  "failure_id": "alternative-poststep-gradient-fixture",
  "kind": "test_fixture_error",
  "training_updates": 0,
  "artifact": "round2/reports/alternative-cpu-gradient-fixture-failure.txt",
  "repair": "Inspect gradients after explicit backward, not after optimizer clears gradients"
}
```

### `capacity-four-task-fixture`

- **方法／触发**：test_fixture_error
- **结果／归因／后续避免**：

```json
{
  "failure_id": "capacity-four-task-fixture",
  "kind": "test_fixture_error",
  "training_updates": 0,
  "artifact": "round2/reports/cpu-task-fixture-failure.txt",
  "repair": "Set scalar binary_classification task in four-task fixture; CPU/GPU5 passed"
}
```

### `query-parts-item-index-contract`

- **方法／触发**：diagnostic_export_contract
- **结果／归因／后续避免**：

```json
{
  "failure_id": "query-parts-item-index-contract",
  "kind": "diagnostic_export_contract",
  "training_updates": 0,
  "signature": "KeyError: item_id",
  "repair": "Use batch item_index, matching frozen training prediction exporter; retain item_id as NPZ archive key",
  "artifact_path": "qformer-loop/taobao-transfer/query-conditioning/reports/attempts/query-parts-item-key"
}
```

### `query-correction-zero-a-psgd-init`

- **方法／触发**：optimizer_initialization_contract
- **结果／归因／后续避免**：

```json
{
  "failure_id": "query-correction-zero-a-psgd-init",
  "kind": "optimizer_initialization_contract",
  "formal_training_updates": 0,
  "partial_smoke_steps_may_have_updated_other_parameters": true,
  "affected_families": [
    "query_q4_difference",
    "query_q4_entry",
    "query_q4_mean",
    "query_q4_work",
    "query_q8_difference",
    "query_q8_entry",
    "query_q8_mean",
    "query_q8_work"
  ],
  "artifact": "qformer-loop/taobao-transfer/recurrence-query-correction/reports",
  "repair": "Defer A optimizer initialization until first nonzero gradient; preserve optimizer options and exact model initialization; validate two real-data steps."
}
```

### `query-route-export-global-ceiling-budget`

- **方法／触发**：diagnostic_contract
- **结果／归因／后续避免**：

```json
{
  "failure_id": "query-route-export-global-ceiling-budget",
  "kind": "diagnostic_contract",
  "training_updates": 0,
  "repair": "Use audited per-payload release budget 2193 steps and 17822296 rows; global ceil is invalid for this loader.",
  "artifact": "qformer-loop/taobao-transfer/query-correction-diagnostics/reports/attempts/global-ceiling-budget"
}
```

### `query-route-export-symlink-root`

- **方法／触发**：diagnostic_contract
- **结果／归因／后续避免**：

```json
{
  "failure_id": "query-route-export-symlink-root",
  "kind": "diagnostic_contract",
  "training_updates": 0,
  "repair": "Resolve metadata from data_root/dataset_id and valid input containment from resolved valid symlink target.",
  "artifact": "qformer-loop/taobao-transfer/query-correction-diagnostics/reports/attempts/symlink-data-root"
}
```

### `query-correction-q4-difference-second-psgd-init`

- **方法／触发**：optimizer_initialization_contract
- **结果／归因／后续避免**：

```json
{
  "failure_id": "query-correction-q4-difference-second-psgd-init",
  "kind": "optimizer_initialization_contract",
  "formal_training_updates": 0,
  "smoke_first_step_completed": true,
  "family": "query_q4_difference",
  "artifact": "qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/query_q4_difference/smoke.log",
  "repair": "Pending instrumented reproduction; preserve locked optimizer settings and completed seven candidates."
}
```
