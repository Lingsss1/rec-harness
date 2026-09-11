# First-read moments for second-query addressing: actual results

四个新模型与两个已完成参考；固定 seed20262027、两轮、PSGDKron 每步预条件、LR1e-4、clip1。
11 个预定比较 × 21 项指标 = 231 个探索性区间。全部区间仅条件于保存模型，不含训练 seed 方差、多重比较或选择校正。

| model | macro AUC | parameters | forward ms | ratio to A |
| --- | --- | --- | --- | --- |
| mature_r2 | 0.767094951 | 51724260 | 14.3620 | 1.0000 |
| cross8_mature | 0.759406420 | 81616356 | 14.7526 | 1.0272 |
| weighted_variance | 0.764097142 | 51736548 | 16.7221 | 1.1643 |
| weighted_mean | 0.761793527 | 51736548 | 16.6768 | 1.1612 |
| uniform_variance | 0.761079140 | 51736548 | 16.6933 | 1.1623 |
| constant | 0.763089527 | 51736548 | 16.6879 | 1.1619 |

| comparison | macro delta | 95% interval |
| --- | --- | --- |
| weighted_variance__minus__mature_r2 | -0.002997810 | [-0.005467256714010241, -0.0008875458971996674] |
| weighted_variance__minus__cross8_mature | +0.004690722 | [0.0021007615184228745, 0.007722091482036069] |
| weighted_mean__minus__mature_r2 | -0.005301424 | [-0.008202795852167826, -0.0027167669438501324] |
| weighted_mean__minus__cross8_mature | +0.002387107 | [-0.00012144941345763804, 0.00551680614691667] |
| uniform_variance__minus__mature_r2 | -0.006015812 | [-0.008745644480710763, -0.003450261280446835] |
| uniform_variance__minus__cross8_mature | +0.001672719 | [-0.0014210356992132992, 0.00511525888900908] |
| constant__minus__mature_r2 | -0.004005424 | [-0.0072340066829167556, -0.0012263822228989322] |
| constant__minus__cross8_mature | +0.003683107 | [0.0009564965195302034, 0.006603019450163549] |
| weighted_variance__minus__weighted_mean | +0.002303615 | [-0.0004429570498071285, 0.0048404831683537] |
| weighted_variance__minus__uniform_variance | +0.003018002 | [9.167537280184164e-05, 0.005520583184496593] |
| weighted_variance__minus__constant | +0.001007615 | [-0.001312000048539222, 0.003495990945661] |

[全部231项结果 CSV](reports/results-all-effects.csv)：包含每任务 AUC/BCE 和直接任务效果差。负 BCE 差有利；任务差不是整体收益。

归因边界：加权方差与均值比较统计量，与均匀方差比较第一轮寻址的作用，与常量比较样本条件信息。辅助 FP32 softmax 不保证逐位等于 fused BF16 注意力。方差不是校准不确定性；相同参数量不代表相同有效容量或优化器轨迹。只有内部对照收益不足以证明优于原循环。

四个正式拟合共8772次更新，四个真实 smoke 共8次更新；旧两个参考不重训。延迟在同一GPU0和同一1024行成熟输入上各10次预热+40次计时，共300次前向。

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
        "wall_seconds": 0.8115947172045708,
        "cpu_seconds": 0.8803018960000006,
        "peak_rss_bytes": 1525379072
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.18509206175804138,
        "cpu_seconds": 0.18345838099999945,
        "peak_rss_bytes": 1525379072
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
        "wall_seconds": 1.1673275455832481,
        "cpu_seconds": 1.2430763230000004,
        "peak_rss_bytes": 2091470848
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.2336268648505211,
        "cpu_seconds": 0.23148751899999986,
        "peak_rss_bytes": 2091470848
      },
      "latency_peak_cuda_bytes": 1405035008,
      "latency_peak_cuda_reserved_bytes": 1713373184
    },
    "weighted_variance": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2648.5044471919537,
        "cpu_seconds": 3150.032396146,
        "peak_rss_bytes": 17106907136,
        "gpu_peak_bytes": 32622247936,
        "samples": 2426
      },
      "formal_runner": {
        "wall_seconds": 2644.3102850094438,
        "cpu_seconds": 3032.661496179,
        "peak_process_rss_bytes": 5248606208
      },
      "fit_including_validation": {
        "wall_seconds": 2514.1464553028345,
        "cpu_seconds": 2855.036611121,
        "peak_process_rss_bytes": 5248606208
      },
      "prediction_export": {
        "wall_seconds": 125.24761318415403,
        "cpu_seconds": 172.05325436300018,
        "peak_process_rss_bytes": 5248606208
      },
      "smoke_worker": {
        "wall_seconds": 164.33842162787914,
        "cpu_seconds": 221.305303578,
        "peak_rss_bytes": 19607048192,
        "gpu_peak_bytes": 33826013184,
        "samples": 151
      },
      "smoke_runner": {
        "wall_seconds": 159.66691584140062,
        "cpu_seconds": 152.100370713,
        "peak_process_rss_bytes": 3300425728
      },
      "latency_constructor": {
        "wall_seconds": 0.7710365206003189,
        "cpu_seconds": 0.840356689,
        "peak_rss_bytes": 2091470848
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.12224019318819046,
        "cpu_seconds": 0.12112652300000093,
        "peak_rss_bytes": 2091470848
      },
      "latency_peak_cuda_bytes": 1369385472,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "weighted_mean": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2613.6563271060586,
        "cpu_seconds": 3119.7526819329996,
        "peak_rss_bytes": 17078816768,
        "gpu_peak_bytes": 31366053888,
        "samples": 2394
      },
      "formal_runner": {
        "wall_seconds": 2609.928161472082,
        "cpu_seconds": 3003.2489274480004,
        "peak_process_rss_bytes": 5271633920
      },
      "fit_including_validation": {
        "wall_seconds": 2480.7927304655313,
        "cpu_seconds": 2826.817824467,
        "peak_process_rss_bytes": 5271633920
      },
      "prediction_export": {
        "wall_seconds": 124.16637706756592,
        "cpu_seconds": 170.98404432200005,
        "peak_process_rss_bytes": 5271633920
      },
      "smoke_worker": {
        "wall_seconds": 162.1670394986868,
        "cpu_seconds": 220.020367819,
        "peak_rss_bytes": 19715153920,
        "gpu_peak_bytes": 33826013184,
        "samples": 149
      },
      "smoke_runner": {
        "wall_seconds": 158.17813896387815,
        "cpu_seconds": 150.808072362,
        "peak_process_rss_bytes": 3284299776
      },
      "latency_constructor": {
        "wall_seconds": 0.7936275526881218,
        "cpu_seconds": 0.8601520479999998,
        "peak_rss_bytes": 2275430400
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.1169351115822792,
        "cpu_seconds": 0.11588948600000037,
        "peak_rss_bytes": 2275430400
      },
      "latency_peak_cuda_bytes": 1369385472,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "uniform_variance": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2609.309548191726,
        "cpu_seconds": 3118.178352815,
        "peak_rss_bytes": 17156976640,
        "gpu_peak_bytes": 31359762432,
        "samples": 2390
      },
      "formal_runner": {
        "wall_seconds": 2605.421052493155,
        "cpu_seconds": 3001.9935053020004,
        "peak_process_rss_bytes": 5270626304
      },
      "fit_including_validation": {
        "wall_seconds": 2475.459663592279,
        "cpu_seconds": 2823.857328107,
        "peak_process_rss_bytes": 5270626304
      },
      "prediction_export": {
        "wall_seconds": 124.97570534050465,
        "cpu_seconds": 172.6486653840002,
        "peak_process_rss_bytes": 5270626304
      },
      "smoke_worker": {
        "wall_seconds": 167.59183104336262,
        "cpu_seconds": 225.642715105,
        "peak_rss_bytes": 19711717376,
        "gpu_peak_bytes": 33832304640,
        "samples": 154
      },
      "smoke_runner": {
        "wall_seconds": 163.55575590580702,
        "cpu_seconds": 156.311133073,
        "peak_process_rss_bytes": 3294953472
      },
      "latency_constructor": {
        "wall_seconds": 0.8055965378880501,
        "cpu_seconds": 0.8719005580000001,
        "peak_rss_bytes": 2276122624
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.1287439838051796,
        "cpu_seconds": 0.1276145660000001,
        "peak_rss_bytes": 2276122624
      },
      "latency_peak_cuda_bytes": 1369385472,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "constant": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2573.4254642799497,
        "cpu_seconds": 3082.337381397,
        "peak_rss_bytes": 17102426112,
        "gpu_peak_bytes": 31370248192,
        "samples": 2357
      },
      "formal_runner": {
        "wall_seconds": 2569.155454210937,
        "cpu_seconds": 2966.0550361769997,
        "peak_process_rss_bytes": 5273583616
      },
      "fit_including_validation": {
        "wall_seconds": 2441.0259727463126,
        "cpu_seconds": 2790.806207023,
        "peak_process_rss_bytes": 5273583616
      },
      "prediction_export": {
        "wall_seconds": 123.1439737379551,
        "cpu_seconds": 169.75532905,
        "peak_process_rss_bytes": 5273583616
      },
      "smoke_worker": {
        "wall_seconds": 168.6941207125783,
        "cpu_seconds": 228.416749538,
        "peak_rss_bytes": 19714613248,
        "gpu_peak_bytes": 33830207488,
        "samples": 155
      },
      "smoke_runner": {
        "wall_seconds": 164.32935041934252,
        "cpu_seconds": 156.152255692,
        "peak_process_rss_bytes": 3274907648
      },
      "latency_constructor": {
        "wall_seconds": 0.7705128565430641,
        "cpu_seconds": 0.8377152730000006,
        "peak_rss_bytes": 2276122624
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.15250489860773087,
        "cpu_seconds": 0.15127142099999702,
        "peak_rss_bytes": 2276122624
      },
      "latency_peak_cuda_bytes": 1369385472,
      "latency_peak_cuda_reserved_bytes": 1568669696
    }
  },
  "paired_cost": {
    "wall_seconds": 317.1779059842229,
    "cpu_seconds": 506.578713311,
    "peak_rss_bytes": 2603044864,
    "artifact_bytes": 1145816
  },
  "latency_cost": {
    "wall_seconds": 20.192166693508625,
    "cpu_seconds": 20.218717209,
    "peak_rss_bytes": 2276122624,
    "artifact_bytes": 172123,
    "peak_cuda_bytes": 1405035008
  },
  "csv_sha256": "258421591f8ad189cc6efc7af45b51c5d5c641c05d1af50528799bf463ba7d34",
  "input_sha256": {
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/render_results.py": "2c29ab04b98354ba9da5f1d5603970208a83eeee670954fbb32263b0198447cc",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/analysis-protocol.json": "7a65f216824a8f79783fffbf3838df7783b5ed85dd4a44f993b8d6db7860a96d",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/configs.json": "b4015e071154cf3adea5d9c98f8181d190b93bb9d2b885b6671bfe8c6f066a2a",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/configs.json": "3c83c83720783e7ab531e62dc9739f0e164161f42043825520260e07859b8f45",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/configs.json": "cedc3fcaebc0ce4f1a4347d9d8e69a902c2d8b3fe03fc6c9495132da52b1ac56",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/paired_analysis.py": "58ca48176815ec7296b094f9d7e91a23403a5d00f9f2904dc47d996bfa894d5b",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/single_effects.py": "a49c5fd1328f89e8c94a93fc2ebde12d74e94b291b6400d14a33af700e0017a1",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/benchmark_latency.py": "a510914c11f46e19555e01084dc6c1b45f1eaa1f0482792f742b44494c6dabf2",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/quality-summary.json": "b12d1f2b7d8dc6f25bf488448e75a47554d202c08c6b8680bb3caa0156afb914",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/paired-analysis/completion.json": "056187c3f2ada75b36f926432171a451e22df3259fa01900a177dd04ac864400",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/latency/completion.json": "b477e9a36b428512520246245b0a9ed4236f2d0ec26fda27fcc19d03a70d6c57",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/paired-analysis/claim.json": "98aae7a52d822f65ff1cd0936cda9c86105bbae6380796ca07bce95ed46bea7d",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/paired-analysis/result.json": "08941f86271d2a26068c4799dd967ca4ac4885c1df08cf9285749d5381610e9f",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/paired-analysis/cost.json": "1f5a14c15bf509c70832bfca4c40a55e3e5f9ecade0c52a08f8f9829e4d34399",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/latency/batch.json": "8f4a3b3ba5958d4945706151c793f7cf071a58152f3e6a5388c3972231244fe0",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/latency/claim.json": "37246409f80fa419cf5759b15a3a2467c8f24bd2ddc1be295e9b3bc2aff7557c",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/latency/constant.json": "ccf0feec83ec4e98f406318fdb0a2e37014643097bbc16f152d202fdd3b817ba",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/latency/context-binding.json": "1382160216db8c2efabd63a789297508fc71b1cc1cc34c426f916ecd5d452fad",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/latency/cross8_mature.json": "d62a25a47408d956c5e57e771d6c608263c75cc560831438d345f1eba8be20b9",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/latency/mature_r2.json": "96744868577e50f7b683ea4ec5d88bdd12a60d06eaf38f80b6d2901c6b1b119b",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/latency/plan.json": "7a10ed5230773e0de446cb1616be978ec7d8e6b6442c9bbc65c1d6d89d5c3c85",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/latency/uniform_variance.json": "356fe66e706706f5eb768e168e9b4c0d53fb180cce746cc89b81442cf7277160",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/latency/weighted_mean.json": "b099c6a83611a5a851d3e7176b4994eba80f25cb6c704b6265c6dbefca1c145f",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/latency/weighted_variance.json": "44213f6fbf059aa288aa7de18cf4dbe5bea06ce6845249ad6e08b3a01b8df588",
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
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/weighted_variance/result.json": "3fe81cc4deb4139c08d48c09c015a6e357c11749c29e6bab58fc3ee108e2ad61",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/weighted_variance/manifest.json": "c0ec969e86c38efaf31a5a34e15d930c6059b64ef19743babe7c2dc8060f73af",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/weighted_variance/formal/exit.json": "2f5b3381ce92cf4d3d03d4ad71dd1050a9d1d063d5656b91f36f54bd369bf7f3",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/weighted_variance/smoke/exit.json": "ae013a78e7f90b70642b3efc0d675700bf0dabc63c1e45c6da5db8d8d4b67c85",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/smoke-runs/weighted_variance/smoke.json": "5cbbe496f681b84c5a87b3a5900f197ae41ab3cbe63b7813102a21e685974a73",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/weighted_mean/result.json": "e8ff5d9fd7eead51e82bb1fc25a80fe75e5fa63b9bf4ce8badd2bfefe939d399",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/weighted_mean/manifest.json": "159d62f8dd0f111d55563d67ae6367b0381d867bf72e2f54fe87e557a4804491",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/weighted_mean/formal/exit.json": "61031aa7a4c3c0ef9aab1eea86a8e47673da3a2e66b2048c05d186f2d5a5380a",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/weighted_mean/smoke/exit.json": "460682496b568521beb34a018a30b1d218515724bec73df8bada0fdedab32708",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/smoke-runs/weighted_mean/smoke.json": "babd0b159b062824b78192191fa3a7391a619b20abfae920cfda9f888a43365b",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/uniform_variance/result.json": "d8e5acff92f0bef90a529d03d8770029595c375a16fde9c04cdf0ec82b6d4e61",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/uniform_variance/manifest.json": "2a3234f31faa4fc0478d69143b8c9c8a2fbedd034f839705cf987ef37108e6d5",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/uniform_variance/formal/exit.json": "1ffd0b6aca646508f81a0751cc32c62ca24ce0ad8f884da70e78275240670ce0",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/uniform_variance/smoke/exit.json": "a50d9d08e9f70cf619e9cf0e25bdbf62fe2c54e15d5afa5a1b91b7d1de71a06a",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/smoke-runs/uniform_variance/smoke.json": "bfcf87d55c39bb2f28b09685dff2c8ebffa6d014eccf38bf8c62efb68a7c9dcb",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/constant/result.json": "9271a9d486b88c82c70e0c3ef6ee9b93f186993217947104cc14843318d7bd43",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/runs/constant/manifest.json": "f1ddfbaf8e3afcd4aedd33bf569bded0f40cfdf1e920f7fc028a85b89b6d7e71",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/constant/formal/exit.json": "d3bffb246042f8605ff8164b7186b1027ed606461e5ff4d0fdc60db6ffd89c21",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/constant/smoke/exit.json": "9d4e9313c6b17255eb056114ee86ca019ea8c64ab90c4044a713c1757dcded4c",
    "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/smoke-runs/constant/smoke.json": "7dcc790e7d09fa12d75d02e8aba8e6f1def5377e894c89d27bd9c3ea684d0069"
  }
}
```
