# 成熟度与历史版本联合结果草稿

八个固定 checkpoint 的已保存结果；保留全部预定比较。本草稿不决定 keep、晋升或修改 incumbent。
A=mature_r2；DD=H2/H2；SD=H1/H2；DS=H2/H1；X=cross8_mature。

| 模型 | click AUC | cart AUC | fav AUC | buy AUC | macro AUC | 参数 | 前向均值 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| released_r1 | 0.65205176 | 0.84276352 | 0.86158708 | 0.87412462 | 0.80763174 | 51724260 | 11.439 |
| released_r2 | 0.65174199 | 0.83963468 | 0.85720077 | 0.87344109 | 0.80550463 | 51724260 | 14.120 |
| mature_r1 | 0.64926817 | 0.79281518 | 0.79941222 | 0.82325339 | 0.76618724 | 51724260 | 11.373 |
| mature_r2 | 0.64993629 | 0.79243427 | 0.80246349 | 0.82354575 | 0.76709495 | 51724260 | 14.177 |
| hist_deep_deep | 0.64974575 | 0.79069748 | 0.78372356 | 0.82251761 | 0.76167110 | 51955556 | 20.788 |
| hist_shallow_deep | 0.65326132 | 0.79065795 | 0.80665651 | 0.81896120 | 0.76738424 | 51955556 | 20.791 |
| hist_deep_shallow | 0.64895960 | 0.79168534 | 0.79294555 | 0.81710001 | 0.76267263 | 51955556 | 20.749 |
| cross8_mature | 0.64993677 | 0.78920674 | 0.79650201 | 0.80198016 | 0.75940642 | 81616356 | 14.526 |

macro 为四任务 AUC 等权平均；不新增 macro BCE 选择目标。

**B：成熟度 × 训练深度的三个效果**（AUC 点差与 95% 区间）

| 效果 | macro AUC：点差 [下界, 上界] |
| --- | --- |
| D_released | -0.00212711 [-0.00341444, -0.00069451] |
| D_mature | +0.00090771 [-0.00185968, +0.00371336] |
| I | +0.00303483 [+0.00004703, +0.00629221] |

D_released、D_mature 分别是同输入模式 R2−R1；I=D_mature−D_released。正 D_mature 可与负 I 并存。I 不是实际泄漏率或旧 AUC 的因果归因。B 不套用 C 的候选阈值。

**C：十个预定比较**（AUC 点差与 95% 区间）

| 比较 | macro AUC：点差 [下界, 上界] |
| --- | --- |
| hist_deep_deep__minus__mature_r2 | -0.00542385 [-0.00841250, -0.00264542] |
| hist_shallow_deep__minus__mature_r2 | +0.00028929 [-0.00264109, +0.00308386] |
| hist_deep_shallow__minus__mature_r2 | -0.00442233 [-0.00785002, -0.00071060] |
| hist_shallow_deep__minus__hist_deep_deep | +0.00571314 [+0.00321678, +0.00805310] |
| hist_deep_shallow__minus__hist_deep_deep | +0.00100153 [-0.00191672, +0.00372585] |
| hist_shallow_deep__minus__hist_deep_shallow | +0.00471162 [+0.00193607, +0.00765648] |
| mature_r2__minus__cross8_mature | +0.00768853 [+0.00500841, +0.01091708] |
| hist_deep_deep__minus__cross8_mature | +0.00226468 [-0.00084104, +0.00499677] |
| hist_shallow_deep__minus__cross8_mature | +0.00797782 [+0.00509936, +0.01099315] |
| hist_deep_shallow__minus__cross8_mature | +0.00326621 [+0.00021567, +0.00638613] |

**任务效果与直接任务差**：完整 [CSV](results-all-effects.csv) 保留 273 行，分别标记 13 个 macro、52 个任务 AUC、52 个任务 BCE、78 个直接任务 AUC 差和 78 个直接任务 BCE 差；没有省略负值或跨零区间。

AUC 正差、BCE 负差有利。异质性看同效果内的六个直接任务差及其区间；不能从“一项显著、另一项不显著”推出任务差异。I 和任务差均在同 draw 内计算，未相减区间端点。跨零表示方向未确定，不表示等效。

**C 候选条件事实，分别报告**

| 模型 | Δmacro vs A | 点差>0 | 点差≥-0.0001 | 前向延迟/A | 延迟比≤1.05 |
| --- | --- | --- | --- | --- | --- |
| hist_deep_deep | -0.00542385 | 否 | 否 | 1.4663 | 否 |
| hist_shallow_deep | +0.00028929 | 是 | 是 | 1.4665 | 否 |
| hist_deep_shallow | -0.00442233 | 否 | 否 | 1.4635 | 否 |

以上是队列政策的分项事实，不合成新的布尔保留规则。X 是匹配输入的基准。最终取舍由根代理结合全部任务、区间和成本说明；新 mature 或严格 released 分数均不直接晋升旧发布协议 incumbent。

**结构解释边界**：DD−A 同时增加历史编码容量与计算，不能单独证明循环或顺序收益；SD−DD 改第一轮版本，DS−DD 改第二轮版本，SD−DS 同时交换两处。缺少 H1/H1，不能完整识别版本交互。H1/H2 在工作循环前预计算，未检验 z1 驱动的历史写回。与 X 比较还包含 q4/q8 的结构和容量差异。

**成本**

| 模型 | 正式 worker 墙钟 s | 正式 CPU s | fit含验证 s | 导出 s | smoke墙钟 s | smoke CPU s | 计时峰值 GiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| released_r1 | 2387.31 | 2889.46 | 2265.41 | 112.82 | 145.53 | 158.67 | 1.175 |
| released_r2 | 2526.18 | 3035.02 | 2399.30 | 117.78 | 145.53 | 159.35 | 1.193 |
| mature_r1 | 2390.00 | 2898.50 | 2268.60 | 112.52 | 192.48 | 317.18 | 1.175 |
| mature_r2 | 2534.68 | 3041.43 | 2408.73 | 117.28 | 188.15 | 315.24 | 1.193 |
| hist_deep_deep | 2885.27 | 3398.60 | 2742.33 | 133.70 | 390.07 | 773.60 | 1.487 |
| hist_shallow_deep | 2898.32 | 3421.14 | 2753.96 | 135.69 | 382.39 | 746.00 | 1.487 |
| hist_deep_shallow | 2879.74 | 3399.23 | 2737.74 | 133.20 | 387.72 | 776.26 | 1.487 |
| cross8_mature | 2522.78 | 3028.68 | 2390.84 | 121.88 | 145.57 | 159.26 | 1.309 |

八个模型各 2193 次正式更新，另各 2 次真实 smoke 更新；训练 seed 固定 20262027。fit/导出/runner 成本是 worker 内部子项，不能与 worker 总量再次相加；并行 worker 墙钟之和不是总历时。RSS 峰值不可相加。

延迟为同一 GPU、BF16、相同 1024 原始行；两种转换输入各自绑定，同模式复用。8 个唯一模型各 10 预热＋40 计时，共 400 次前向，A 只测一次。CUDA Event 不包含读盘、时间查找、成熟编码、搬运、模型构造或 checkpoint 加载；下列原始成本同时保留这些阶段。相同参数量/模块次数不保证同延迟或 PSGD 轨迹。

<details><summary>完整已记录成本与范围（原值）</summary>

```json
{
  "models": {
    "released_r1": {
      "formal_worker": {
        "wall_seconds": 2387.3118413984776,
        "cpu_seconds": 2889.456151949,
        "peak_rss_bytes": 17057886208,
        "gpu_peak_bytes": 23166189568,
        "samples": 2192
      },
      "formal_runner": {
        "wall_seconds": 2383.0501463487744,
        "cpu_seconds": 2782.182926046,
        "peak_process_rss_bytes": 5225852928
      },
      "fit_including_validation": {
        "wall_seconds": 2265.406079865992,
        "cpu_seconds": 2616.654641489,
        "peak_process_rss_bytes": 5225852928
      },
      "prediction_export": {
        "wall_seconds": 112.8208601102233,
        "cpu_seconds": 160.06504317100007,
        "peak_process_rss_bytes": 5225852928
      },
      "smoke_worker": {
        "wall_seconds": 145.52870550751686,
        "cpu_seconds": 158.667184139,
        "peak_rss_bytes": 16678748160,
        "gpu_peak_bytes": 24386732032,
        "samples": 134
      },
      "smoke_runner": {
        "wall_seconds": 140.71798942238092,
        "cpu_seconds": 138.45715398200002,
        "peak_process_rss_bytes": 3212541952
      },
      "latency_constructor": {
        "wall_seconds": 0.7635872140526772,
        "cpu_seconds": 0.833082364,
        "peak_rss_bytes": 1567404032
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.14913657307624817,
        "cpu_seconds": 0.14848171199999882,
        "peak_rss_bytes": 1567404032
      },
      "latency_peak_cuda_bytes": 1261748224,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "released_r2": {
      "formal_worker": {
        "wall_seconds": 2526.1776882708073,
        "cpu_seconds": 3035.017940275,
        "peak_rss_bytes": 17027457024,
        "gpu_peak_bytes": 28367126528,
        "samples": 2318
      },
      "formal_runner": {
        "wall_seconds": 2521.8999362289906,
        "cpu_seconds": 2921.364642386,
        "peak_process_rss_bytes": 5197889536
      },
      "fit_including_validation": {
        "wall_seconds": 2399.2993115633726,
        "cpu_seconds": 2749.798049814,
        "peak_process_rss_bytes": 5197889536
      },
      "prediction_export": {
        "wall_seconds": 117.78031320124865,
        "cpu_seconds": 166.1182913030002,
        "peak_process_rss_bytes": 5197889536
      },
      "smoke_worker": {
        "wall_seconds": 145.52918346226215,
        "cpu_seconds": 159.349367189,
        "peak_rss_bytes": 16883384320,
        "gpu_peak_bytes": 29570891776,
        "samples": 134
      },
      "smoke_runner": {
        "wall_seconds": 141.09737965464592,
        "cpu_seconds": 139.346163969,
        "peak_process_rss_bytes": 3189407744
      },
      "latency_constructor": {
        "wall_seconds": 0.7765096426010132,
        "cpu_seconds": 0.8460525220000008,
        "peak_rss_bytes": 2000564224
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.14918746799230576,
        "cpu_seconds": 0.14865729799999983,
        "peak_rss_bytes": 2000564224
      },
      "latency_peak_cuda_bytes": 1280950272,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "mature_r1": {
      "formal_worker": {
        "wall_seconds": 2390.003037005663,
        "cpu_seconds": 2898.495570814,
        "peak_rss_bytes": 17044246528,
        "gpu_peak_bytes": 23166189568,
        "samples": 2192
      },
      "formal_runner": {
        "wall_seconds": 2385.9209888502955,
        "cpu_seconds": 2788.019562719,
        "peak_process_rss_bytes": 5187223552
      },
      "fit_including_validation": {
        "wall_seconds": 2268.599045239389,
        "cpu_seconds": 2622.833993405,
        "peak_process_rss_bytes": 5187223552
      },
      "prediction_export": {
        "wall_seconds": 112.52327074855566,
        "cpu_seconds": 159.7889330820003,
        "peak_process_rss_bytes": 5187223552
      },
      "smoke_worker": {
        "wall_seconds": 192.47689237445593,
        "cpu_seconds": 317.184706596,
        "peak_rss_bytes": 20020105216,
        "gpu_peak_bytes": 24470618112,
        "samples": 177
      },
      "smoke_runner": {
        "wall_seconds": 187.93101455271244,
        "cpu_seconds": 168.78829639100002,
        "peak_process_rss_bytes": 3282354176
      },
      "latency_constructor": {
        "wall_seconds": 0.764507070183754,
        "cpu_seconds": 0.8373208879999989,
        "peak_rss_bytes": 2019135488
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.13596045225858688,
        "cpu_seconds": 0.1354720020000002,
        "peak_rss_bytes": 2019135488
      },
      "latency_peak_cuda_bytes": 1261617152,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "mature_r2": {
      "formal_worker": {
        "wall_seconds": 2534.6803352087736,
        "cpu_seconds": 3041.4287067490004,
        "peak_rss_bytes": 17010442240,
        "gpu_peak_bytes": 28369223680,
        "samples": 2324
      },
      "formal_runner": {
        "wall_seconds": 2530.8240097388625,
        "cpu_seconds": 2924.3865203349997,
        "peak_process_rss_bytes": 5197381632
      },
      "fit_including_validation": {
        "wall_seconds": 2408.725445777178,
        "cpu_seconds": 2754.180437591,
        "peak_process_rss_bytes": 5197381632
      },
      "prediction_export": {
        "wall_seconds": 117.28338450938463,
        "cpu_seconds": 164.79081047,
        "peak_process_rss_bytes": 5197381632
      },
      "smoke_worker": {
        "wall_seconds": 188.1520602926612,
        "cpu_seconds": 315.23697800900004,
        "peak_rss_bytes": 20016791552,
        "gpu_peak_bytes": 29654777856,
        "samples": 173
      },
      "smoke_runner": {
        "wall_seconds": 184.04285093396902,
        "cpu_seconds": 165.61778548,
        "peak_process_rss_bytes": 3281965056
      },
      "latency_constructor": {
        "wall_seconds": 0.7444503977894783,
        "cpu_seconds": 0.8169494050000008,
        "peak_rss_bytes": 2019135488
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.09087871760129929,
        "cpu_seconds": 0.09054152500000079,
        "peak_rss_bytes": 2019135488
      },
      "latency_peak_cuda_bytes": 1280950272,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "hist_deep_deep": {
      "formal_worker": {
        "wall_seconds": 2885.2715956643224,
        "cpu_seconds": 3398.598635535,
        "peak_rss_bytes": 17073332224,
        "gpu_peak_bytes": 37873516544,
        "samples": 2651
      },
      "formal_runner": {
        "wall_seconds": 2880.8002161830664,
        "cpu_seconds": 3269.8243722119996,
        "peak_process_rss_bytes": 5321830400
      },
      "fit_including_validation": {
        "wall_seconds": 2742.3323982432485,
        "cpu_seconds": 3084.707496516,
        "peak_process_rss_bytes": 5321830400
      },
      "prediction_export": {
        "wall_seconds": 133.7023969218135,
        "cpu_seconds": 179.6792724080001,
        "peak_process_rss_bytes": 5321830400
      },
      "smoke_worker": {
        "wall_seconds": 390.07422936707735,
        "cpu_seconds": 773.6041447599999,
        "peak_rss_bytes": 21456265216,
        "gpu_peak_bytes": 39586889728,
        "samples": 358
      },
      "smoke_runner": {
        "wall_seconds": 383.3361012637615,
        "cpu_seconds": 312.98804439,
        "peak_process_rss_bytes": 4063207424
      },
      "latency_constructor": {
        "wall_seconds": 0.7472046092152596,
        "cpu_seconds": 0.8193787999999991,
        "peak_rss_bytes": 2019459072
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.1395946741104126,
        "cpu_seconds": 0.1389796299999997,
        "peak_rss_bytes": 2019459072
      },
      "latency_peak_cuda_bytes": 1596911104,
      "latency_peak_cuda_reserved_bytes": 1885339648
    },
    "hist_shallow_deep": {
      "formal_worker": {
        "wall_seconds": 2898.3198798447847,
        "cpu_seconds": 3421.1364017429996,
        "peak_rss_bytes": 17097760768,
        "gpu_peak_bytes": 37873516544,
        "samples": 2662
      },
      "formal_runner": {
        "wall_seconds": 2894.301814697683,
        "cpu_seconds": 3290.927189509,
        "peak_process_rss_bytes": 5375836160
      },
      "fit_including_validation": {
        "wall_seconds": 2753.957258604467,
        "cpu_seconds": 3103.5553596490004,
        "peak_process_rss_bytes": 5375836160
      },
      "prediction_export": {
        "wall_seconds": 135.6942832544446,
        "cpu_seconds": 182.0496151100001,
        "peak_process_rss_bytes": 5375836160
      },
      "smoke_worker": {
        "wall_seconds": 382.3934711366892,
        "cpu_seconds": 745.996829154,
        "peak_rss_bytes": 21443477504,
        "gpu_peak_bytes": 39586889728,
        "samples": 351
      },
      "smoke_runner": {
        "wall_seconds": 376.3137733414769,
        "cpu_seconds": 307.61251734300004,
        "peak_process_rss_bytes": 4063899648
      },
      "latency_constructor": {
        "wall_seconds": 0.7764620780944824,
        "cpu_seconds": 0.8493744810000017,
        "peak_rss_bytes": 2179629056
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.14760559052228928,
        "cpu_seconds": 0.14702845600000103,
        "peak_rss_bytes": 2179629056
      },
      "latency_peak_cuda_bytes": 1596911104,
      "latency_peak_cuda_reserved_bytes": 1885339648
    },
    "hist_deep_shallow": {
      "formal_worker": {
        "wall_seconds": 2879.7387092188,
        "cpu_seconds": 3399.227580377,
        "peak_rss_bytes": 17094365184,
        "gpu_peak_bytes": 37875613696,
        "samples": 2641
      },
      "formal_runner": {
        "wall_seconds": 2875.5411411970854,
        "cpu_seconds": 3266.9059760309997,
        "peak_process_rss_bytes": 5337481216
      },
      "fit_including_validation": {
        "wall_seconds": 2737.7414908483624,
        "cpu_seconds": 3082.45303297,
        "peak_process_rss_bytes": 5337481216
      },
      "prediction_export": {
        "wall_seconds": 133.2015559002757,
        "cpu_seconds": 179.19545729999982,
        "peak_process_rss_bytes": 5337481216
      },
      "smoke_worker": {
        "wall_seconds": 387.7233123779297,
        "cpu_seconds": 776.25840938,
        "peak_rss_bytes": 21437984768,
        "gpu_peak_bytes": 39586889728,
        "samples": 355
      },
      "smoke_runner": {
        "wall_seconds": 382.11504117399454,
        "cpu_seconds": 311.32617109800003,
        "peak_process_rss_bytes": 4048855040
      },
      "latency_constructor": {
        "wall_seconds": 0.7496187686920166,
        "cpu_seconds": 0.8225819120000004,
        "peak_rss_bytes": 2179629056
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.14736035466194153,
        "cpu_seconds": 0.14678289000000078,
        "peak_rss_bytes": 2179629056
      },
      "latency_peak_cuda_bytes": 1596911104,
      "latency_peak_cuda_reserved_bytes": 1885339648
    },
    "cross8_mature": {
      "formal_worker": {
        "wall_seconds": 2522.7814552783966,
        "cpu_seconds": 3028.683010009,
        "peak_rss_bytes": 16983564288,
        "gpu_peak_bytes": 30013390848,
        "samples": 2316
      },
      "formal_runner": {
        "wall_seconds": 2517.830292247236,
        "cpu_seconds": 2915.230754392,
        "peak_process_rss_bytes": 5166972928
      },
      "fit_including_validation": {
        "wall_seconds": 2390.841024532914,
        "cpu_seconds": 2738.14177712,
        "peak_process_rss_bytes": 5166972928
      },
      "prediction_export": {
        "wall_seconds": 121.87652725726366,
        "cpu_seconds": 171.29153207000036,
        "peak_process_rss_bytes": 5166972928
      },
      "smoke_worker": {
        "wall_seconds": 145.57136395573616,
        "cpu_seconds": 159.25709092199997,
        "peak_rss_bytes": 16900792320,
        "gpu_peak_bytes": 31368151040,
        "samples": 134
      },
      "smoke_runner": {
        "wall_seconds": 141.30999380350113,
        "cpu_seconds": 138.768773325,
        "peak_process_rss_bytes": 3169501184
      },
      "latency_constructor": {
        "wall_seconds": 1.0492680445313454,
        "cpu_seconds": 1.1287579450000003,
        "peak_rss_bytes": 2298408960
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.22298692911863327,
        "cpu_seconds": 0.22225663400000073,
        "peak_rss_bytes": 2298408960
      },
      "latency_peak_cuda_bytes": 1405035008,
      "latency_peak_cuda_reserved_bytes": 1713373184
    }
  },
  "quality_audit": {
    "wall_seconds": 0.5633813515305519,
    "cpu_seconds": 0.559921634,
    "peak_rss_bytes": 29446144,
    "scope": "This stdlib audit process only; includes reused small-file worker checks and current stat checks."
  },
  "cohort_launchers": {
    "b": {
      "wall_seconds": 2535.080653525889
    },
    "c": {
      "wall_seconds": 2899.219774633646
    }
  },
  "paired": {
    "wall_seconds": 391.456679277122,
    "cpu_seconds": 579.057779404,
    "peak_rss_bytes": 3401994240,
    "artifact_bytes": 1336123
  },
  "paired_scope": "Claim through all analysis artifacts; excludes final completion serialization",
  "latency": {
    "wall_seconds": 24.955616541206837,
    "cpu_seconds": 25.184595643,
    "peak_rss_bytes": 2298408960,
    "artifact_bytes": 168865,
    "peak_cuda_bytes": 1596911104
  },
  "latency_scope": "Explicit --run entry through all model artifacts, excluding read-only preflight and final completion serialization; peak RSS is whole process high-water mark.",
  "latency_input_preparation": {
    "released": {
      "input_batch_sha256": "920c74e1f08cb400f6c7633d2534a6c667d3c2302a3c80f2cd5f3a41e32291c0",
      "input_tensor_bytes": 5513216,
      "input_leaf_schema": [
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024,
            101
          ],
          "dtype": "torch.int32",
          "stride": [
            101,
            1
          ],
          "bytes": 413696
        },
        {
          "shape": [
            1024,
            101
          ],
          "dtype": "torch.int32",
          "stride": [
            101,
            1
          ],
          "bytes": 413696
        },
        {
          "shape": [
            1024,
            101
          ],
          "dtype": "torch.int32",
          "stride": [
            101,
            1
          ],
          "bytes": 413696
        },
        {
          "shape": [
            1024,
            101
          ],
          "dtype": "torch.int32",
          "stride": [
            101,
            1
          ],
          "bytes": 413696
        },
        {
          "shape": [
            1024,
            101
          ],
          "dtype": "torch.int32",
          "stride": [
            101,
            1
          ],
          "bytes": 413696
        },
        {
          "shape": [
            1024,
            101
          ],
          "dtype": "torch.int32",
          "stride": [
            101,
            1
          ],
          "bytes": 413696
        },
        {
          "shape": [
            1024,
            101
          ],
          "dtype": "torch.int64",
          "stride": [
            101,
            1
          ],
          "bytes": 827392
        },
        {
          "shape": [
            1024,
            100
          ],
          "dtype": "torch.float32",
          "stride": [
            100,
            1
          ],
          "bytes": 409600
        },
        {
          "shape": [
            1024,
            100
          ],
          "dtype": "torch.float32",
          "stride": [
            400,
            4
          ],
          "bytes": 409600
        },
        {
          "shape": [
            1024,
            100
          ],
          "dtype": "torch.float32",
          "stride": [
            400,
            4
          ],
          "bytes": 409600
        },
        {
          "shape": [
            1024,
            100
          ],
          "dtype": "torch.float32",
          "stride": [
            400,
            4
          ],
          "bytes": 409600
        },
        {
          "shape": [
            1024,
            100
          ],
          "dtype": "torch.float32",
          "stride": [
            400,
            4
          ],
          "bytes": 409600
        }
      ],
      "constructor_cost": {
        "wall_seconds": 0.014224834740161896,
        "cpu_seconds": 0.015669722000000164,
        "peak_rss_bytes": 980643840
      },
      "first_batch_cost": {
        "wall_seconds": 0.2517428696155548,
        "cpu_seconds": 0.4861607440000002,
        "peak_rss_bytes": 1274376192
      },
      "input_hash_cost": {
        "wall_seconds": 0.006296247243881226,
        "cpu_seconds": 0.021544027999999216,
        "peak_rss_bytes": 1274376192
      },
      "loader_audit": {
        "schema_version": 1,
        "maturity_mode": "released",
        "counts": {
          "target_rows": 1024,
          "original_item_positive_tokens": 69230,
          "strict_history_tokens": 66412,
          "removed_same_timestamp_tokens": 2818,
          "removed_future_timestamp_tokens": 0,
          "empty_strict_history_rows": 29,
          "immature_window_tokens": 28108,
          "exact_window_boundary_tokens": 0,
          "full_mature_tokens": 38304,
          "unknown_action_tokens": 0,
          "unknown_click_zero_tokens": 0,
          "unknown_click_one_tokens": 0,
          "immature_is_click_positive_tokens": 959,
          "immature_is_click_zero_tokens": 27149,
          "immature_cart_positive_tokens": 197,
          "immature_cart_zero_tokens": 27911,
          "immature_fav_positive_tokens": 45,
          "immature_fav_zero_tokens": 28063,
          "immature_buy_positive_tokens": 0,
          "immature_buy_zero_tokens": 28108,
          "batches": 1
        },
        "ordered_source_rows_sha256": "d7277e88f80c2e29e8f4f9e7621791398190a732470d18d2a137a45d36012b79",
        "ordered_source_batches_sha256": "f622457434efeadf2fd34b2ad82d9bb84f1b2ae0b22777e9193ffa2e21fa2ca5",
        "converted_action_and_strict_mask_sha256": "36df0d82946e640210317a6849bb0f651cfe5cf5b47aa5b7fdb08fec55a82933",
        "common_strict_history_mask_sha256": "8f92810d001b2562f98353169a61409bcad6a06b292d18ac64097f13f27d975c",
        "block_side_loads": 1,
        "block_side_cache_hits": 0,
        "hash_wall_seconds": 0.001495547592639923,
        "hash_cpu_seconds": 0.0044800110000000615,
        "conversion_wall_seconds": 0.018705658614635468,
        "conversion_cpu_seconds": 0.03780923899999955,
        "row_identity_scope": "Ordered original payload bytes, canonical data-file paths and part IDs; no label-based selection or added RNG.",
        "timestamp_scope": "Time inferred at full_timestamp_seq[seq_len]; no independent target-time check. Full side sequences physically decoded, including later values; only retained prefix and target used.",
        "click_availability_assumed": true,
        "unknown_behavior_is_known_zero": false,
        "task_mask_scope": "Unknown behavior task masks are zero because no positive behavior is observed; this does not mean known negative. Actual action inputs carry categories 17/18.",
        "window_boundary": "Strictly mature only when th+86400<t0; equality remains unknown.",
        "population_inference": false
      },
      "all_nested_parts_preserved": true,
      "transformed_batch_instances": 1,
      "transfer_cost": {
        "wall_seconds": 0.009733714163303375,
        "cpu_seconds": 0.026125029000000133,
        "peak_rss_bytes": 1326243840
      },
      "device_transfer_exact": true
    },
    "mature": {
      "input_batch_sha256": "9ba7300679dcc598c9e2cb121bacc5907e2f602c8ba3b047a3cc73d27429f397",
      "input_tensor_bytes": 5513216,
      "input_leaf_schema": [
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024
          ],
          "dtype": "torch.float64",
          "stride": [
            19
          ],
          "bytes": 8192
        },
        {
          "shape": [
            1024,
            101
          ],
          "dtype": "torch.int32",
          "stride": [
            101,
            1
          ],
          "bytes": 413696
        },
        {
          "shape": [
            1024,
            101
          ],
          "dtype": "torch.int32",
          "stride": [
            101,
            1
          ],
          "bytes": 413696
        },
        {
          "shape": [
            1024,
            101
          ],
          "dtype": "torch.int32",
          "stride": [
            101,
            1
          ],
          "bytes": 413696
        },
        {
          "shape": [
            1024,
            101
          ],
          "dtype": "torch.int32",
          "stride": [
            101,
            1
          ],
          "bytes": 413696
        },
        {
          "shape": [
            1024,
            101
          ],
          "dtype": "torch.int32",
          "stride": [
            101,
            1
          ],
          "bytes": 413696
        },
        {
          "shape": [
            1024,
            101
          ],
          "dtype": "torch.int32",
          "stride": [
            101,
            1
          ],
          "bytes": 413696
        },
        {
          "shape": [
            1024,
            101
          ],
          "dtype": "torch.int64",
          "stride": [
            101,
            1
          ],
          "bytes": 827392
        },
        {
          "shape": [
            1024,
            100
          ],
          "dtype": "torch.float32",
          "stride": [
            100,
            1
          ],
          "bytes": 409600
        },
        {
          "shape": [
            1024,
            100
          ],
          "dtype": "torch.float32",
          "stride": [
            400,
            4
          ],
          "bytes": 409600
        },
        {
          "shape": [
            1024,
            100
          ],
          "dtype": "torch.float32",
          "stride": [
            400,
            4
          ],
          "bytes": 409600
        },
        {
          "shape": [
            1024,
            100
          ],
          "dtype": "torch.float32",
          "stride": [
            400,
            4
          ],
          "bytes": 409600
        },
        {
          "shape": [
            1024,
            100
          ],
          "dtype": "torch.float32",
          "stride": [
            400,
            4
          ],
          "bytes": 409600
        }
      ],
      "constructor_cost": {
        "wall_seconds": 0.009883932769298553,
        "cpu_seconds": 0.010901051000000272,
        "peak_rss_bytes": 1274376192
      },
      "first_batch_cost": {
        "wall_seconds": 0.1947249323129654,
        "cpu_seconds": 0.38093226400000013,
        "peak_rss_bytes": 1326243840
      },
      "input_hash_cost": {
        "wall_seconds": 0.00629868358373642,
        "cpu_seconds": 0.02113031799999998,
        "peak_rss_bytes": 1326243840
      },
      "loader_audit": {
        "schema_version": 1,
        "maturity_mode": "mature",
        "counts": {
          "target_rows": 1024,
          "original_item_positive_tokens": 69230,
          "strict_history_tokens": 66412,
          "removed_same_timestamp_tokens": 2818,
          "removed_future_timestamp_tokens": 0,
          "empty_strict_history_rows": 29,
          "immature_window_tokens": 28108,
          "exact_window_boundary_tokens": 0,
          "full_mature_tokens": 38304,
          "unknown_action_tokens": 28108,
          "unknown_click_zero_tokens": 27149,
          "unknown_click_one_tokens": 959,
          "immature_is_click_positive_tokens": 959,
          "immature_is_click_zero_tokens": 27149,
          "immature_cart_positive_tokens": 197,
          "immature_cart_zero_tokens": 27911,
          "immature_fav_positive_tokens": 45,
          "immature_fav_zero_tokens": 28063,
          "immature_buy_positive_tokens": 0,
          "immature_buy_zero_tokens": 28108,
          "batches": 1
        },
        "ordered_source_rows_sha256": "d7277e88f80c2e29e8f4f9e7621791398190a732470d18d2a137a45d36012b79",
        "ordered_source_batches_sha256": "f622457434efeadf2fd34b2ad82d9bb84f1b2ae0b22777e9193ffa2e21fa2ca5",
        "converted_action_and_strict_mask_sha256": "c3560cba247ca60db9161939ddc3c6cd4b6b9fa566c47c7f057ec4d7bad4ec49",
        "common_strict_history_mask_sha256": "8f92810d001b2562f98353169a61409bcad6a06b292d18ac64097f13f27d975c",
        "block_side_loads": 1,
        "block_side_cache_hits": 0,
        "hash_wall_seconds": 0.0015103146433830261,
        "hash_cpu_seconds": 0.007204775000000829,
        "conversion_wall_seconds": 0.0185694620013237,
        "conversion_cpu_seconds": 0.03440674399999999,
        "row_identity_scope": "Ordered original payload bytes, canonical data-file paths and part IDs; no label-based selection or added RNG.",
        "timestamp_scope": "Time inferred at full_timestamp_seq[seq_len]; no independent target-time check. Full side sequences physically decoded, including later values; only retained prefix and target used.",
        "click_availability_assumed": true,
        "unknown_behavior_is_known_zero": false,
        "task_mask_scope": "Unknown behavior task masks are zero because no positive behavior is observed; this does not mean known negative. Actual action inputs carry categories 17/18.",
        "window_boundary": "Strictly mature only when th+86400<t0; equality remains unknown.",
        "population_inference": false
      },
      "all_nested_parts_preserved": true,
      "transformed_batch_instances": 1,
      "transfer_cost": {
        "wall_seconds": 0.009983889758586884,
        "cpu_seconds": 0.026969077000000397,
        "peak_rss_bytes": 2000564224
      },
      "device_transfer_exact": true
    }
  }
}
```

这些是保存的阶段/worker 计量；不包含未在消费工件中记录的早期合成检查成本，也不将 worker 占用墙钟冒充 GPU kernel 活跃时间。
</details>

**统计与输入身份范围**：2,871,767 行、268,201 用户、一次共同 500 draw，bootstrap seed 20260906。95% 条件区间未包含训练 seed 方差、多重或选择校正。BCE 采用源预测 dtype 的 epsilon 裁剪、float64 log，分母为每个 draw 实际重采样行数。

Original N audit supplies the 122 full input-content SHA256 values. Production B/C runners verified input stat snapshots before and after their actual traversals, and both original metadata hashes before those traversals; this audit now verifies the 122 current stat records and two original metadata hashes. No Parquet, checkpoint or prediction array is opened or rehashed here. Model-output SHA256 values are inherited from successful production completions and matched worker/launcher receipts. Stat continuity cannot independently rule out content changes that preserve stat fields.

成熟窗口/目标时间由现有时间戳推定，click 可知性仍是假设；同一输入身份不证明线上可用时间。未知行为类不同于成熟真零。

[判读规则](../INTERPRETATION_GUIDE.md)；本脚本只读取小 JSON/源码，不读取 NPZ、checkpoint 或原数据，不产生新的抽样或模型前向。

<details><summary>报告消费身份</summary>

```json
{
  "input_sha256": {
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/render_joint_results.py": "8c3a062fc431d9756efa89cd28a00ea164b6253f231c39380a9e2086b27771ce",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/protocol.json": "ec34db634b4c3e276f9269bc94dcb2fb7fbb61a4491bab5fb47d2f1d618a0d8a",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/protocol.json": "002a874de807cc1ebb997a86874c957e96d1bd923c02421b4c1a6f09164d2c6b",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/configs.json": "3c83c83720783e7ab531e62dc9739f0e164161f42043825520260e07859b8f45",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/configs.json": "cedc3fcaebc0ce4f1a4347d9d8e69a902c2d8b3fe03fc6c9495132da52b1ac56",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/INTERPRETATION_GUIDE.md": "e6c2fcbc2e4c1049fcf2368330250e21d0820862b6747edd56ac82205bbca503",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/audit_joint_quality.py": "7fc78a81d0dd437513ae2b9e39df9f3655a019dd3a03ae86c3348411adb2cdde",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/paired_analysis.py": "4aaca6f86e07306cf923a6c62e45fb618fe1d6eff3fc7b6edaabc3c8db3f15c9",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/joint_effects.py": "ebe696d8062c7d7c18701bd8203caac64083433a307ed367f95ccf535e143729",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/benchmark_latency.py": "30b3ceaf66ab7985445121dc196bb1037551177d1c665d368c31ea52787c822a",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/joint-quality-summary.json": "4ab0782f1064b0f78f4b9c6a6282a4a2b3d2c5ec23c722ca9c6de233ef279e51",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/joint-paired/completion.json": "fdda304429a00938111cffc91b3a29c3d994decb556cb4856f0881fdb4360321",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/latency/completion.json": "90bf87c3c19f22b25a068f4c7fb2bf24d774e07dc080e91291c29ceabd36be7b",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/joint-paired/claim.json": "34da8d14ab4ebde6879d0f79e38544e4027325aa953fce2927eccb59cbb30e1a",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/joint-paired/result.json": "e971ac6fd2ba8874ccc4a0070f4b86bf91b0f329623919917304e6ef8e6bfc0a",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/joint-paired/cost.json": "76e917ae8a1019b13cdb00230d009475f4ce3ca9a0b4fc04ad9e041ab01f8449",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/latency/batch.json": "672edf3f3b5afc23572a6a0e118d672e2e0d5232fd5ac81963e324a5a52c6148",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/latency/claim.json": "d6db1611efe6b8489edee798bb402cc07b74592ce600d04d93ddb5e35e17c7e4",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/latency/context-binding.json": "b7e1fea5748f601897faf7f191deb2585d0d9859820e183ff689b8f7566cbdf3",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/latency/cross8_mature.json": "437267e8c01d98c5fe91e6e9e19c35cebf9f539229e3f2fc8a7ad27d1d7bc82e",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/latency/hist_deep_deep.json": "85e4ca21e00e918aba52b2535431b08cf81e913757ab050cdf3889c3461cc6bf",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/latency/hist_deep_shallow.json": "2511a115e9e41657d6b0a29d688ad1cb26f5b137e89c377747b44be775cd38bc",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/latency/hist_shallow_deep.json": "b546732ebf07f2612adfe6a2cc1a5d6caba06e150f8b72a654d816fcc4e7ae0c",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/latency/mature_r1.json": "007eeff40e49b431a7c14a66fe1b6bd5fde400eb8b0c42fd22b92501e4c0a1a0",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/latency/mature_r2.json": "15e26291e6142eb016322ae85d81c6f54c0bb67aa739376060e411f596022050",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/latency/plan.json": "95d4efefbd91c181f7429d8268c76d8c990b2c986852d9a7d4d492b4b7974610",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/latency/released_r1.json": "30ed4b37351839ff3f4de88ba23d4b4707974a9fac7cd33957aca00514006395",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/latency/released_r2.json": "60170a7faa75e0697d3fede51a6a9caf357e80ca9dd8f3e0e66e8a441836da44",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/released_r1/result.json": "388a958bb69de83038489a4fd94ad4fb42b3f0a1b0db6cbd147bbb532b2b5c69",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/released_r1/manifest.json": "9c308e243f9ab08b5090ba293947c94e3162d822c5af6dbe689c5a1470d80df6",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/reports/released_r1/formal/exit.json": "5f79706b8713b81fbda5d6d716a02c266eed38fbb40bbbf273f13121d4579f57",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/reports/released_r1/smoke/exit.json": "17a664f1aae44e9c2f8b5a38a3baf39dc898cfb425bfcd1d3e5ed0b5440210a4",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/smoke-runs/released_r1/smoke.json": "cce3c5823dae39267764fd8d6b3a7a6177587ba58db9bcc94f2d4ead8f74a306",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/released_r2/result.json": "55575cad30b2b41e3c4be6ad42b750c9165a6528dfce19c63ec827320ec21a3d",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/released_r2/manifest.json": "8277d9582ccb4302e340493826c44181f6b64ae6bd843122b43190f8ee17d823",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/reports/released_r2/formal/exit.json": "3e0b10b0d519c6f857ad3ac8abe20b153a4da65a85fd23188a00ec7cac9271f1",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/reports/released_r2/smoke/exit.json": "850c5fcb91c4f686d6129f6c2dd83fd4ddee8b444d11d65f800cbc38fbfb39f6",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/smoke-runs/released_r2/smoke.json": "ced4c0e8d3e1c17578f92b624316d30e2447ebe36c18f9aa4af4cf32e4635ea7",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r1/result.json": "f20cc0055017503b428368f2db8ab676ada98fca7f5f6f91dbe99cd480b26f85",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r1/manifest.json": "b0cd61cf86b51afa3dcf9d3bde93a987e464abf1ebde4a1577714dac8c67b2f6",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/reports/mature_r1/formal/exit.json": "d48519ec0ce0c67569e935b5c6da8b2ad823211633e8ba91864a08b661bdf14e",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/reports/mature_r1/smoke/exit.json": "80d8cc450274c97ccdb3c1f0d6b2169822f3f8b64865e06cf396d88ea0f703d8",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/smoke-runs/mature_r1/smoke.json": "62fd1381e35c0dac3f34889553ac98952bc7bf296c949d0aca53e62420565117",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2/result.json": "15c18a7aee32a1dd3015c9a8c9613fe532887416cd49e5a063943f3eee4d246b",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2/manifest.json": "a8439ee9ab331449eb1625f5822deb9a2ad15bf4ad5be8fefa451e78a17949de",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/reports/mature_r2/formal/exit.json": "0a814029b8cedcea62429041dacff3eda1fa875e93151ed300e2151e3456cae7",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/reports/mature_r2/smoke/exit.json": "dcae789a6a0c64dd982291e62eaab85cfce32f5758cb74fb0fabce223047cbdc",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/smoke-runs/mature_r2/smoke.json": "d034bd61e75061a211157047d5c4eac9d94559470ef60f348bb94f09223f81f9",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/hist_deep_deep/result.json": "beef2eece0bd568ecaf13be7b0448c380647b1e1e664ea38c6e57f030f1c6430",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/hist_deep_deep/manifest.json": "d802e843a901c9fc96e995d5e64cf8b121363e2d5b2eb038a46c1c0213fd4de5",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/hist_deep_deep/formal/exit.json": "e63f0bf685f0d240db783053a2b729b2f0ba3557a465c2d2310c23ec504d9d59",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/hist_deep_deep/smoke/exit.json": "0040d78937542f33d6db6c1c25d446cc3b1d1d485fe02442fc6da3b416f5b19a",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/smoke-runs/hist_deep_deep/smoke.json": "80927de110ca9a81872696bfa10476dd44bef70829b3c463b2c16afe5b02baba",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/hist_shallow_deep/result.json": "8b5385dff3136403baa61cbc8550f453f2cd12a446d122be6d1995d7f5248aa7",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/hist_shallow_deep/manifest.json": "18df9929a44abebfb9a83f5888db43c9b25f659c2dc46d4bd4102cf911332cce",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/hist_shallow_deep/formal/exit.json": "b9c4e7a9ae2d05c03910204fcccd92be90b17d194304d6b906f0cef1c69ff2c4",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/hist_shallow_deep/smoke/exit.json": "1beb1a0516083e8807aecd38def528acf6b55c033f53361157861ee3d14abcae",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/smoke-runs/hist_shallow_deep/smoke.json": "bab16ce4df0eac5a8f6628bc57eac7392452114fbd51adaf1672d3bbbe85073e",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/hist_deep_shallow/result.json": "70715ebad43b2a86605bfe2a1a76c9ea5fa3e61cf3b7e7b7f742a9a160cb3dd2",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/hist_deep_shallow/manifest.json": "29d4d701fb97e12a478024218ac2cf9be2d7cb11efdc28170eed37635982696b",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/hist_deep_shallow/formal/exit.json": "5f3488c30b65360a60a5a9cfbaf488812f39fb3f244fe6a08393c19c529ee9cf",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/hist_deep_shallow/smoke/exit.json": "50affea3d878282c92ff8b49e1b587c4c70b6d504fc9f69d6a15ff1054623de7",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/smoke-runs/hist_deep_shallow/smoke.json": "2c86ade17332574f4bc332943be8fb6dba811e666f94cd368cda123385d65d34",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature/result.json": "e301e2959cd9d8aa5443bdc8ee9bdee02576bc62946241c34996435bf9ef6103",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature/manifest.json": "f1a54a6ebd39c772cb1b858b69cb5a132d3b2773da264276c402eb689567e50f",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/cross8_mature/formal/exit.json": "e726a6438fa2c9958595b167f608b369ab559c027705a2e8709f01d3e1b17fc7",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/cross8_mature/smoke/exit.json": "e2b92d7e21d9f0f2faacae0ddfb40ae2031ecf4981e15c147b0a172497b7b468",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/smoke-runs/cross8_mature/smoke.json": "b63ff5e265f171d54a6865ee40a926cd63435dd7aff5e971c89e0be591434ba3"
  },
  "csv_sha256": "aedc0519b2ea4f8df67a8286412a3a736e722fe3590ebff1ac3af3a317ddef70",
  "recorded_shared_draw_sha256": {
    "model-draws.npz": "b82c358c630a57a7013c82e83963524be7c818d04e11f4f001752cb4a59e085a",
    "effect-draws.npz": "0b9c20d4e25d56ac09a27d960524acff704f2809e5eb1b7a28a00c9ed861eb79"
  },
  "resample_indices_sha256": "d696a8abcd6826b74cffc9de67e66d23abf714f38cac73c3ccc8b46acd8625f2",
  "row_identity_sha256": {
    "user_id": "812c391af58f68aa0792501bf0ed78cf59b3b162f5cdbd7c633e1d9128efd503",
    "item_id": "97377c9da45a8117fe934f1a0fe9c69e01e83005a8cfed6583dfbccbb6370239",
    "seq_len": "692c8962c3e5ff5f6f2e39cf1af4692065a623f577049338c5f0faaf3d1737be",
    "label_is_click": "392d77d38d338df4c9cdcf4886d751c36f362f111a5596e83c5ea6079f1d3a67",
    "label_cart": "0dee52990f0e511dba957874f4eef779941364b22d1ecf04367b3c33d9e428cd",
    "label_fav": "a1e5664b8d41941b24ae769a158c66172695b7955b862b91046b94393b491469",
    "label_buy": "f8bedbd95bfe55b054703fa73d904aa47498deabc3c4f9e26f224a5251125222"
  },
  "sorted_user_ids_sha256": "8432eca55d86b67685d476a611b55733a5b515bab0f1a4ea93f24617f8a0ea7f",
  "draw_artifact_bytes_read": false
}
```

</details>
