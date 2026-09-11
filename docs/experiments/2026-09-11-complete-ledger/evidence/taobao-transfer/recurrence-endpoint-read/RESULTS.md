# 末条事件保留：一个候选，尚未确认原循环提升

三组固定 seed 的正式训练和14个配对对照已完成。even_last macro AUC为0.768530297，相对mature_r2 +0.001435345，95%探索性区间[−0.001703625,+0.004117848]；相对此前first_odd仅+0.000197426，区间[−0.002918969,+0.002967312]。保留研究候选，不替换原循环或旧发布协议基线。

点击AUC +0.001031731，区间[+0.000314855,+0.001770698]，点击BCE改善；购买BCE反而+0.000204581，区间[+0.000120889,+0.000296989]。其他任务AUC区间跨零。不能将宏观正点估计解释为所有任务获益。延迟14.499318ms，相对同期原循环+0.6071%，通过既定候选成本门槛。

odd_last与odd_no_last相对原循环macro分别−0.004784711和−0.007211877，两者区间全负，discard。even_last相对复用的first_even +0.006110436，区间[+0.003000971,+0.008898164]；odd_last相对odd_no_last +0.002427166，区间[−0.000831778,+0.005225740]。两种取样下末条保留效应的直接交互为−0.003683270，区间[−0.008154073,+0.000770318]：不能由一组显著、另一组不显著推断取样方式调节效应。

末条指存储顺序的最后有效位置，并非已验证的时间最近事件；入口层仍读取完整历史。此结果不能证明全模型获得新信息，也不能隔离第二轮查询中介。当前另行进行的轮次区分累积更新四组对照和样本/槽诊断，才针对新的用户问题。

执行记录：三组正式训练2193步均成功，三个2步smoke均成功。原延迟测量在完成四个参考200次前向后因遗漏模块导入失败；恢复只测三个新模型150次，同GPU与输入但分属两个窗口。原配对分析被失败管线终止，已完成draw数未知且未保存完整结果；独立恢复使用同seed、同数学完成500次重采样和294项区间。原失败记录和成本保留，未重训。区间条件于单seed训练出的checkpoint，不含训练seed、多重比较或模型选择校正。

# Equal-count history support and work-round placement: actual results

三个新模型与四个已完成参考；固定 seed20262027、两轮、PSGDKron 每步预条件、LR1e-4、clip1。
14 个预定比较 × 21 项指标 = 294 个探索性区间。全部区间仅条件于保存模型，不含训练 seed 方差、多重比较或选择校正。

| model | macro AUC | parameters | forward ms | ratio to A |
| --- | --- | --- | --- | --- |
| mature_r2 | 0.767094951 | 51724260 | 14.4118 | 1.0000 |
| cross8_mature | 0.759406420 | 81616356 | 14.7400 | 1.0228 |
| first_even | 0.762419860 | 51724260 | 14.6045 | 1.0134 |
| first_odd | 0.768332871 | 51724260 | 14.6147 | 1.0141 |
| even_last | 0.768530297 | 51724260 | 14.4993 | 1.0061 |
| odd_last | 0.762310240 | 51724260 | 14.4924 | 1.0056 |
| odd_no_last | 0.759883074 | 51724260 | 14.5305 | 1.0082 |

| comparison | macro delta | 95% interval |
| --- | --- | --- |
| even_last__minus__mature_r2 | +0.001435345 | [-0.001703624913421238, 0.004117847779228911] |
| even_last__minus__cross8_mature | +0.009123877 | [0.006325664326927619, 0.011991797300099279] |
| even_last__minus__first_odd | +0.000197426 | [-0.00291896933166691, 0.002967311915196923] |
| odd_last__minus__mature_r2 | -0.004784711 | [-0.007743902623144495, -0.00213613676149459] |
| odd_last__minus__cross8_mature | +0.002903820 | [3.991417333172186e-05, 0.005648346734759868] |
| odd_last__minus__first_odd | -0.006022630 | [-0.00889742853069146, -0.0033833486818336195] |
| odd_no_last__minus__mature_r2 | -0.007211877 | [-0.010450704717793483, -0.004335497843915446] |
| odd_no_last__minus__cross8_mature | +0.000476654 | [-0.002152493483931351, 0.0036434595326973736] |
| odd_no_last__minus__first_odd | -0.008449796 | [-0.011282326392177782, -0.005765063190823054] |
| even_last__minus__first_even | +0.006110436 | [0.0030009711184742992, 0.008898163943809555] |
| odd_last__minus__odd_no_last | +0.002427166 | [-0.0008317778533958271, 0.005225739664836559] |
| odd_no_last__minus__first_even | -0.002536786 | [-0.005345358582909904, 0.0006210780299661046] |
| odd_last__minus__even_last | -0.006220056 | [-0.009240939085681788, -0.0031071687483180525] |
| endpoint_effect_odd_minus_even | -0.003683270 | [-0.008154072547029424, 0.0007703183683111161] |

[全部294项结果 CSV](reports/results-all-effects.csv)：包含每任务 AUC/BCE 和直接任务效果差。负 BCE 差有利；任务差不是整体收益。

归因边界：加权方差与均值比较统计量，与均匀方差比较第一轮寻址的作用，与常量比较样本条件信息。辅助 FP32 softmax 不保证逐位等于 fused BF16 注意力。方差不是校准不确定性；相同参数量不代表相同有效容量或优化器轨迹。只有内部对照收益不足以证明优于原循环。

三个正式拟合共6579次更新，三个真实 smoke 共6次更新；四个参考不重训。延迟原执行完成四参考200前向后因缺少模块导入失败；恢复只执行三新模型150前向，同GPU0与相同输入，但分属两个执行窗口。原配对分析被管线终止，未留下完整结果，恢复为同seed同数学的一次500重采样；原尝试已执行draw数未知。

全部训练成本、测量成本与身份见下方；未重读原始大文件计算SHA，输入内容身份沿用既有审计并检查文件连续性。

```json
{
  "costs": {
    "mature_r2": {
      "training_reused_reference": true,
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
        "wall_seconds": 0.7678022533655167,
        "cpu_seconds": 0.8406405979999994,
        "peak_rss_bytes": 1528922112
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.16279436647891998,
        "cpu_seconds": 0.16222851299999963,
        "peak_rss_bytes": 1528922112
      },
      "latency_peak_cuda_bytes": 1280622592,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "cross8_mature": {
      "training_reused_reference": true,
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
        "wall_seconds": 1.1203361079096794,
        "cpu_seconds": 1.1991298350000008,
        "peak_rss_bytes": 2094125056
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.2482910081744194,
        "cpu_seconds": 0.24687274300000084,
        "peak_rss_bytes": 2094125056
      },
      "latency_peak_cuda_bytes": 1405035008,
      "latency_peak_cuda_reserved_bytes": 1713373184
    },
    "first_even": {
      "training_reused_reference": true,
      "formal_worker": {
        "wall_seconds": 2515.3534044995904,
        "cpu_seconds": 3025.922370799,
        "peak_rss_bytes": 16983592960,
        "gpu_peak_bytes": 28371320832,
        "samples": 2306
      },
      "formal_runner": {
        "wall_seconds": 2510.978076197207,
        "cpu_seconds": 2911.186048419,
        "peak_process_rss_bytes": 5240745984
      },
      "fit_including_validation": {
        "wall_seconds": 2387.611212775111,
        "cpu_seconds": 2739.738395661,
        "peak_process_rss_bytes": 5240745984
      },
      "prediction_export": {
        "wall_seconds": 117.95157391577959,
        "cpu_seconds": 165.77488881099998,
        "peak_process_rss_bytes": 5240745984
      },
      "smoke_worker": {
        "wall_seconds": 147.4100959300995,
        "cpu_seconds": 161.93760790800002,
        "peak_rss_bytes": 16945926144,
        "gpu_peak_bytes": 29570891776,
        "samples": 135
      },
      "smoke_runner": {
        "wall_seconds": 141.91200194507837,
        "cpu_seconds": 139.593388292,
        "peak_process_rss_bytes": 3239772160
      },
      "latency_constructor": {
        "wall_seconds": 0.9842376708984375,
        "cpu_seconds": 1.1027963119999988,
        "peak_rss_bytes": 2094125056
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.13933110237121582,
        "cpu_seconds": 0.1387346650000012,
        "peak_rss_bytes": 2094125056
      },
      "latency_peak_cuda_bytes": 1281155072,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "first_odd": {
      "training_reused_reference": true,
      "formal_worker": {
        "wall_seconds": 2528.382029980421,
        "cpu_seconds": 3044.418955067,
        "peak_rss_bytes": 17071022080,
        "gpu_peak_bytes": 28371320832,
        "samples": 2317
      },
      "formal_runner": {
        "wall_seconds": 2524.141234844923,
        "cpu_seconds": 2927.1683170829997,
        "peak_process_rss_bytes": 5244911616
      },
      "fit_including_validation": {
        "wall_seconds": 2397.915154516697,
        "cpu_seconds": 2747.219195707,
        "peak_process_rss_bytes": 5244911616
      },
      "prediction_export": {
        "wall_seconds": 120.8867464736104,
        "cpu_seconds": 174.2454546419999,
        "peak_process_rss_bytes": 5244911616
      },
      "smoke_worker": {
        "wall_seconds": 152.04752030968666,
        "cpu_seconds": 166.190266503,
        "peak_rss_bytes": 16907530240,
        "gpu_peak_bytes": 29570891776,
        "samples": 140
      },
      "smoke_runner": {
        "wall_seconds": 148.8715279698372,
        "cpu_seconds": 145.14428338000002,
        "peak_process_rss_bytes": 3207770112
      },
      "latency_constructor": {
        "wall_seconds": 0.8305772542953491,
        "cpu_seconds": 0.9118659719999993,
        "peak_rss_bytes": 2101911552
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.15558020770549774,
        "cpu_seconds": 0.15493358900000054,
        "peak_rss_bytes": 2101911552
      },
      "latency_peak_cuda_bytes": 1281155072,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "even_last": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2504.268677316606,
        "cpu_seconds": 3004.7624286259997,
        "peak_rss_bytes": 17043628032,
        "gpu_peak_bytes": 28369223680,
        "samples": 2296
      },
      "formal_runner": {
        "wall_seconds": 2500.447254151106,
        "cpu_seconds": 2895.7986620089996,
        "peak_process_rss_bytes": 5262491648
      },
      "fit_including_validation": {
        "wall_seconds": 2376.759330958128,
        "cpu_seconds": 2722.683524909,
        "peak_process_rss_bytes": 5262491648
      },
      "prediction_export": {
        "wall_seconds": 118.8226051479578,
        "cpu_seconds": 167.5417705169998,
        "peak_process_rss_bytes": 5262491648
      },
      "smoke_worker": {
        "wall_seconds": 142.17819248884916,
        "cpu_seconds": 155.180190424,
        "peak_rss_bytes": 16914710528,
        "gpu_peak_bytes": 29572988928,
        "samples": 131
      },
      "smoke_runner": {
        "wall_seconds": 138.36269782483578,
        "cpu_seconds": 137.21933768600002,
        "peak_process_rss_bytes": 3236663296
      },
      "latency_constructor": {
        "wall_seconds": 0.7623036354780197,
        "cpu_seconds": 0.8318260120000005,
        "peak_rss_bytes": 1530429440
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.16031786799430847,
        "cpu_seconds": 0.1597225790000003,
        "peak_rss_bytes": 1530429440
      },
      "latency_peak_cuda_bytes": 1280827392,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "odd_last": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2506.589051552117,
        "cpu_seconds": 3006.224509484,
        "peak_rss_bytes": 17033830400,
        "gpu_peak_bytes": 28369223680,
        "samples": 2298
      },
      "formal_runner": {
        "wall_seconds": 2501.779577061534,
        "cpu_seconds": 2896.588143339,
        "peak_process_rss_bytes": 5275492352
      },
      "fit_including_validation": {
        "wall_seconds": 2376.681172825396,
        "cpu_seconds": 2717.655542041,
        "peak_process_rss_bytes": 5275492352
      },
      "prediction_export": {
        "wall_seconds": 120.26605863124132,
        "cpu_seconds": 173.62243787099987,
        "peak_process_rss_bytes": 5275492352
      },
      "smoke_worker": {
        "wall_seconds": 143.2635766044259,
        "cpu_seconds": 156.45439083600002,
        "peak_rss_bytes": 16941613056,
        "gpu_peak_bytes": 29570891776,
        "samples": 132
      },
      "smoke_runner": {
        "wall_seconds": 138.54176331311464,
        "cpu_seconds": 137.051349051,
        "peak_process_rss_bytes": 3235897344
      },
      "latency_constructor": {
        "wall_seconds": 0.7668727189302444,
        "cpu_seconds": 0.8368182060000002,
        "peak_rss_bytes": 2098741248
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.16916535794734955,
        "cpu_seconds": 0.16852593200000143,
        "peak_rss_bytes": 2098741248
      },
      "latency_peak_cuda_bytes": 1281155072,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "odd_no_last": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2506.5823511853814,
        "cpu_seconds": 3007.4703000850004,
        "peak_rss_bytes": 17018593280,
        "gpu_peak_bytes": 28369223680,
        "samples": 2297
      },
      "formal_runner": {
        "wall_seconds": 2501.7509671002626,
        "cpu_seconds": 2896.83181013,
        "peak_process_rss_bytes": 5272592384
      },
      "fit_including_validation": {
        "wall_seconds": 2378.765804104507,
        "cpu_seconds": 2723.707955877,
        "peak_process_rss_bytes": 5272592384
      },
      "prediction_export": {
        "wall_seconds": 118.09135127812624,
        "cpu_seconds": 167.66163491099996,
        "peak_process_rss_bytes": 5272592384
      },
      "smoke_worker": {
        "wall_seconds": 139.8507951721549,
        "cpu_seconds": 153.768312719,
        "peak_rss_bytes": 16946548736,
        "gpu_peak_bytes": 29570891776,
        "samples": 129
      },
      "smoke_runner": {
        "wall_seconds": 136.00064123421907,
        "cpu_seconds": 135.380392657,
        "peak_process_rss_bytes": 3245948928
      },
      "latency_constructor": {
        "wall_seconds": 0.7393206655979156,
        "cpu_seconds": 0.8097832609999998,
        "peak_rss_bytes": 2099417088
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.1104494109749794,
        "cpu_seconds": 0.11000088800000007,
        "peak_rss_bytes": 2099417088
      },
      "latency_peak_cuda_bytes": 1281155072,
      "latency_peak_cuda_reserved_bytes": 1568669696
    }
  },
  "paired_cost": {
    "wall_seconds": 365.5903059914708,
    "cpu_seconds": 554.590713591,
    "peak_rss_bytes": 3001974784,
    "artifact_bytes": 1405095
  },
  "latency_cost": {
    "wall_seconds": 13.549801900982857,
    "cpu_seconds": 13.911672735,
    "peak_rss_bytes": 2099417088,
    "artifact_bytes": 148394,
    "peak_cuda_bytes": 1281155072
  },
  "prior_failed_latency_cost": {
    "wall_seconds": 15.293759860098362,
    "cpu_seconds": 15.528417671,
    "peak_rss_bytes": 2101911552
  },
  "csv_sha256": "30423987f42bba03a63cd2af5c051f468d45e705982479c90e3523d82d3be5c6",
  "input_sha256": {
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/render_results_recovery.py": "c6feb7600239ba70e4be88c2d33d3ae5b3fee5385402e5c3d8701ebc80e6928e",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/analysis-protocol.json": "df8003f8ffa2d411a6e5f50f9daceee0cee36ffdc479e704265371a2dbe1c318",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/configs.json": "6b20d66b6d840d105cddca37b73b21d87c3d0587c9e282ed67a6df8d913497e5",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/configs.json": "3c83c83720783e7ab531e62dc9739f0e164161f42043825520260e07859b8f45",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/configs.json": "cedc3fcaebc0ce4f1a4347d9d8e69a902c2d8b3fe03fc6c9495132da52b1ac56",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/paired_analysis_recovery.py": "eb9bd4fadf374ca144e22e44909c53362b71fc605bff373fc18e1656db6c4e02",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/single_effects.py": "c8d673ccb03399670876c434868995ecf91a6d381a2329974998ea66ba151d3a",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/benchmark_latency_recovery.py": "a0524be859adff5d00de76a370dc091527f62202edde0888833e3aa38bd52e0c",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/quality-summary.json": "11f342302065c15d672387e761cee5e107fd20cd9632fc547a22270f390cc2b2",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/paired-analysis-recovery/completion.json": "4a37782006080bbc623b6cd920b470f4a1446d4f8313e9bfa243c0afc004893e",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/latency-recovery/completion.json": "e255d284cf4cdc93e9d2841e9d822d6950771806d1855a83c998033c5f488583",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/paired-analysis-recovery/claim.json": "6989c6b76d184286c6fd115a4514b14e962f2a38776f04022de4eea0ee58a8fa",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/paired-analysis-recovery/result.json": "64c6d0d8db29da73d5e89f889a643ec5f5aa3f3a25c3eb8517db2d304fcdfdd1",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/paired-analysis-recovery/cost.json": "7512d021962846a26936437b857b26195d78b7b7a3b0a709e2eb018439c6f4c2",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/latency-recovery/batch.json": "a9af626aa008ec145ebbf8842eebd359be13124fa7098172ec9e8c2ce8673b6f",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/latency-recovery/claim.json": "18b6b07ce5af052359ede81b319ada45adfe79ce7f6393657ca2e1fcfc426242",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/latency-recovery/context-binding.json": "90a828bb611e181b9e507174543cfc73a874cda1acf2384bc431c53ae4476382",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/latency-recovery/even_last.json": "566661e2eca7163ad6ecb257c2d054a041365665cfff456266216ab4ed33d7af",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/latency-recovery/odd_last.json": "03fe783ca9fe1bf9d5486a495d2db803df788545d40d6db0a5eb65341126c69b",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/latency-recovery/odd_no_last.json": "c4d2bf47d009a0f1ee8b90d95685aca673e57fca854db515174a6e5c03bb7259",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/latency-recovery/plan.json": "f9f738bff6af54b4c8c06ec43aa6a636386d2c81c0bbf2162bb4a1db002c3fc4",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/latency/failure.json": "f2693b36d0fd1683a0aa4374d8a158c1c1ae8dabc1d43ab022fd30de824abf1e",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/measurement-pipeline/failure.json": "dabf8f634318242003b4663e45e4a4956bb057776cdaafab46e3b57bd8a7a218",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/paired-recovery-review.json": "5279c3d7f2bb81932c1cade6d1850c48eb198b7912a1d1413125566cab70b18b",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/latency/mature_r2.json": "65161a5d5310e7d8f547ef800d14822d4a906c8b4d743954d7c648a8a8774c65",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/latency/cross8_mature.json": "92c200025d2b6ad5f17985752cb21da1e25bbd336299f1cb0b0bf13ead753183",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/latency/first_even.json": "f9c898bc0e9dd1876e1cb69232ec2f2ad92a47809a4994570cd7bfb0d98fbfe6",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/latency/first_odd.json": "678fc3ab32a8236a07e9eae3215805455244aab6fd8a751dd8b01de3387c1760",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2/result.json": "15c18a7aee32a1dd3015c9a8c9613fe532887416cd49e5a063943f3eee4d246b",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2/manifest.json": "a8439ee9ab331449eb1625f5822deb9a2ad15bf4ad5be8fefa451e78a17949de",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/reports/mature_r2/formal/exit.json": "0a814029b8cedcea62429041dacff3eda1fa875e93151ed300e2151e3456cae7",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/reports/mature_r2/smoke/exit.json": "dcae789a6a0c64dd982291e62eaab85cfce32f5758cb74fb0fabce223047cbdc",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/smoke-runs/mature_r2/smoke.json": "d034bd61e75061a211157047d5c4eac9d94559470ef60f348bb94f09223f81f9",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature/result.json": "e301e2959cd9d8aa5443bdc8ee9bdee02576bc62946241c34996435bf9ef6103",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature/manifest.json": "f1a54a6ebd39c772cb1b858b69cb5a132d3b2773da264276c402eb689567e50f",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/cross8_mature/formal/exit.json": "e726a6438fa2c9958595b167f608b369ab559c027705a2e8709f01d3e1b17fc7",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/cross8_mature/smoke/exit.json": "e2b92d7e21d9f0f2faacae0ddfb40ae2031ecf4981e15c147b0a172497b7b468",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/smoke-runs/cross8_mature/smoke.json": "b63ff5e265f171d54a6865ee40a926cd63435dd7aff5e971c89e0be591434ba3",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_even/result.json": "a6955c66720f6698a85c6b1b48224c385fb161428fdc39d5158b6393ed185145",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_even/manifest.json": "4a7e620352c18b79d89cf49cec6363400a000943f290d698a8af60cb812ec59a",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/first_even/formal/exit.json": "875ea534b3fdc2cd7e99939c6672bd2c96e4d7050b12b6ce9725e9d281d49038",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/first_even/smoke/exit.json": "f1ff58f5be4b9f5eec70861ca8d7d43720bcf8f505c785f34f0fd24ec516a229",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/smoke-runs/first_even/smoke.json": "e456922f23eed006b4ffa0d17be5e4c44c8cb8b275a899cec44933df4f4474f4",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_odd/result.json": "9ca8426ec0ef9fc2f65fd48208f7869c55048dc044b94c3ce6c7d01e9c83c3de",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_odd/manifest.json": "62896430d504d095bd7c3bd08c54ce9da5935d8a5aa0e2b589f9709d2a4984ee",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/first_odd/formal/exit.json": "2830a5f1142efad9bf7d2870ec6ad384ae3468104a15e0f50dc0eada0146d589",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/first_odd/smoke/exit.json": "cd8760337a84b26c3958e18e48cc8cdc0d34cf93643b4166741ff28428c3ac17",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/smoke-runs/first_odd/smoke.json": "4388338f9a9faab0035065a911a6a6ee5ecf8f5d85d6dd629d133929509bc8a5",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/runs/even_last/result.json": "369dd225e41698d50a0ec225357d01f4c330497aae2a1917c89f2d7192ed3d69",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/runs/even_last/manifest.json": "7825c5671ac2a0e9b3371633aa68b11b62861d19e95f4e8ab6399e80d432f672",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/even_last/formal/exit.json": "42ec66c3cf5be09d64964b46f48b417146a9c546991339f3f9b37229cdcc152d",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/even_last/smoke/exit.json": "eae68917586899089e8f3e3e48ddfeb38c804c35518294fff9d01cb4f70e160d",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/smoke-runs/even_last/smoke.json": "984cf0d6551b9af491ad6a1b8f0082edf7820d5d69b5b9d6c4460a82106c69ed",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/runs/odd_last/result.json": "6fe5a68fce701f7226f1b96d66c16bc4b1627b61d29303d2d48f758bb2e3c7fa",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/runs/odd_last/manifest.json": "742c70a8f5dfc1af89e24bc1ff4153854be43facb60e286e3ea0d7d8538b85a6",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/odd_last/formal/exit.json": "681b95cf3e65aa1b591cb92612c06ec9a1c0ec5a2e864918f425d5ac0d7e4fb0",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/odd_last/smoke/exit.json": "b6a96504f1051e93896f70bd4d98f212ffc826d7f7bed3b3879c2e0949b548ee",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/smoke-runs/odd_last/smoke.json": "beee0c23f418c33e8ea7a14269d574297013f9ce1c9c3a4828c26f5b2461983f",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/runs/odd_no_last/result.json": "b5490a2193cb61d68bc9d67b4a26fffc4d788e03763efd2da15c325d54b0282c",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/runs/odd_no_last/manifest.json": "0f417cfd48fef3f15ba8a6e6a6797722c4ea344fd61b11c1af1ade87612fb2b9",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/odd_no_last/formal/exit.json": "6143947e9693d9060ac0be95b7a7984bca8a7af90bf2b07ecf8d9c858da8c47b",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/odd_no_last/smoke/exit.json": "bf99f2e5f55f46aecb9787f6781206ec11f0afbc9b124a2216cf60c01a27c765",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/smoke-runs/odd_no_last/smoke.json": "9cd07123800627ca03ebce89ecd44d5f2992b7d45cb0d0af6f83625455e4e7da"
  }
}
```
