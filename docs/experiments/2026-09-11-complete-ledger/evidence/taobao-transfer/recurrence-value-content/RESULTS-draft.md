# Second-work value content: actual results

六个新模型与两个已完成参考；固定 seed20262027、两轮、PSGDKron 每步预条件、LR1e-4、clip1。
21 个预定比较 × 21 项指标 = 441 个探索性区间。全部区间仅条件于保存模型，不含训练 seed 方差、多重比较或选择校正。

| model | macro AUC | parameters | forward ms | ratio to A |
| --- | --- | --- | --- | --- |
| mature_r2 | 0.767094951 | 51724260 | 14.2767 | 1.0000 |
| cross8_mature | 0.759406420 | 81616356 | 14.5326 | 1.0179 |
| event_entry | 0.762755591 | 51742692 | 15.9023 | 1.1139 |
| event_work | 0.764189742 | 51742692 | 15.9448 | 1.1168 |
| event_constant | 0.762619645 | 51742692 | 15.9093 | 1.1144 |
| pooled_entry | 0.763743324 | 51742692 | 14.2662 | 0.9993 |
| pooled_work | 0.759877762 | 51742692 | 14.3092 | 1.0023 |
| pooled_constant | 0.762871450 | 51742692 | 14.2654 | 0.9992 |

| comparison | macro delta | 95% interval |
| --- | --- | --- |
| event_entry__minus__mature_r2 | -0.004339360 | [-0.00789869835970372, -0.0008016800889363466] |
| event_work__minus__mature_r2 | -0.002905209 | [-0.005597721227477079, -0.00040624993050394914] |
| event_constant__minus__mature_r2 | -0.004475306 | [-0.007605460367732112, -0.0014428959967006203] |
| pooled_entry__minus__mature_r2 | -0.003351628 | [-0.00633145636213481, -0.0003471266979163905] |
| pooled_work__minus__mature_r2 | -0.007217189 | [-0.010118841791676636, -0.0042218699572851605] |
| pooled_constant__minus__mature_r2 | -0.004223502 | [-0.007324589522790518, -0.000923203284268388] |
| event_entry__minus__cross8_mature | +0.003349171 | [2.1065200761098056e-07, 0.006480661449644977] |
| event_work__minus__cross8_mature | +0.004783322 | [0.0019252730358451549, 0.007803504812156308] |
| event_constant__minus__cross8_mature | +0.003213225 | [7.870933662974407e-05, 0.006063728247578188] |
| pooled_entry__minus__cross8_mature | +0.004336904 | [0.0012580335331483553, 0.007782453669999475] |
| pooled_work__minus__cross8_mature | +0.000471342 | [-0.002503317264701223, 0.0036486390863802232] |
| pooled_constant__minus__cross8_mature | +0.003465029 | [0.0004733433606766717, 0.005931692134220944] |
| event_work__minus__event_entry | +0.001434151 | [-0.0017163203768568059, 0.004531709042086111] |
| event_work__minus__event_constant | +0.001570097 | [-0.0012037464265058318, 0.004583983959190252] |
| event_entry__minus__event_constant | +0.000135946 | [-0.0031499835696204397, 0.003429056158559577] |
| pooled_work__minus__pooled_entry | -0.003865562 | [-0.006989559898956577, -0.0008467517321487585] |
| pooled_work__minus__pooled_constant | -0.002993688 | [-0.006090106449406745, -0.0002223709344236027] |
| pooled_entry__minus__pooled_constant | +0.000871874 | [-0.002541345630540502, 0.00416568162071926] |
| event_entry__minus__pooled_entry | -0.000987732 | [-0.004773053872229927, 0.002503981295255304] |
| event_work__minus__pooled_work | +0.004311980 | [0.001515423766723957, 0.007267848372803551] |
| event_constant__minus__pooled_constant | -0.000251804 | [-0.0029954776395498214, 0.0024113440355742212] |

[全部441项结果 CSV](reports/results-all-effects.csv)：包含每任务 AUC/BCE 和直接任务效果差。负 BCE 差有利；任务差不是整体收益。

归因边界：event 与 pooled 比较交互位置及计算；work 与 entry/constant 比较条件来源。固定常量控制有效自由度较少。新增训练梯度可改变共享 Q/K，不能把局部保留寻址代码解释为训练后注意力不变。非零初始化不与原模型初始函数等价。

六个正式拟合共13158次更新，六个真实 smoke 共12次更新；旧两个参考不重训。延迟在同一GPU0和同一1024行成熟输入上各10次预热+40次计时，共400次前向。

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
        "wall_seconds": 0.7606646195054054,
        "cpu_seconds": 0.8327150350000005,
        "peak_rss_bytes": 1514426368
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.16021567583084106,
        "cpu_seconds": 0.15965750099999987,
        "peak_rss_bytes": 1514426368
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
        "wall_seconds": 1.1079973876476288,
        "cpu_seconds": 1.1970980079999993,
        "peak_rss_bytes": 2080583680
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.2774854749441147,
        "cpu_seconds": 0.2765909369999999,
        "peak_rss_bytes": 2080583680
      },
      "latency_peak_cuda_bytes": 1405035008,
      "latency_peak_cuda_reserved_bytes": 1713373184
    },
    "event_entry": {
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
    "event_work": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2633.404682211578,
        "cpu_seconds": 3135.145446079,
        "peak_rss_bytes": 17063841792,
        "gpu_peak_bytes": 30262951936,
        "samples": 2416
      },
      "formal_runner": {
        "wall_seconds": 2629.4034451395273,
        "cpu_seconds": 3017.2109455540003,
        "peak_process_rss_bytes": 5271363584
      },
      "fit_including_validation": {
        "wall_seconds": 2500.870648562908,
        "cpu_seconds": 2841.480256417,
        "peak_process_rss_bytes": 5271363584
      },
      "prediction_export": {
        "wall_seconds": 123.4742039591074,
        "cpu_seconds": 170.3087260780003,
        "peak_process_rss_bytes": 5271363584
      },
      "smoke_worker": {
        "wall_seconds": 200.153367318213,
        "cpu_seconds": 254.368102502,
        "peak_rss_bytes": 19768623104,
        "gpu_peak_bytes": 32792117248,
        "samples": 183
      },
      "smoke_runner": {
        "wall_seconds": 195.74498146772385,
        "cpu_seconds": 178.15051898000002,
        "peak_process_rss_bytes": 3311620096
      },
      "latency_constructor": {
        "wall_seconds": 0.7709200605750084,
        "cpu_seconds": 0.8482239109999998,
        "peak_rss_bytes": 2134388736
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.14211653172969818,
        "cpu_seconds": 0.14165325799999984,
        "peak_rss_bytes": 2134388736
      },
      "latency_peak_cuda_bytes": 1284208128,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "event_constant": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2627.9050848707557,
        "cpu_seconds": 3127.0364961640003,
        "peak_rss_bytes": 17082970112,
        "gpu_peak_bytes": 30260854784,
        "samples": 2412
      },
      "formal_runner": {
        "wall_seconds": 2624.0310389623046,
        "cpu_seconds": 3009.6277336079997,
        "peak_process_rss_bytes": 5293260800
      },
      "fit_including_validation": {
        "wall_seconds": 2495.8989638984203,
        "cpu_seconds": 2834.030955709,
        "peak_process_rss_bytes": 5293260800
      },
      "prediction_export": {
        "wall_seconds": 123.08756574243307,
        "cpu_seconds": 170.1786524529998,
        "peak_process_rss_bytes": 5293260800
      },
      "smoke_worker": {
        "wall_seconds": 196.88931387662888,
        "cpu_seconds": 252.435959932,
        "peak_rss_bytes": 19725701120,
        "gpu_peak_bytes": 32792117248,
        "samples": 180
      },
      "smoke_runner": {
        "wall_seconds": 193.41917557269335,
        "cpu_seconds": 176.10555503199998,
        "peak_process_rss_bytes": 3283312640
      },
      "latency_constructor": {
        "wall_seconds": 0.7622329965233803,
        "cpu_seconds": 0.8374763460000008,
        "peak_rss_bytes": 2134814720
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.1455247551202774,
        "cpu_seconds": 0.14504931600000148,
        "peak_rss_bytes": 2134814720
      },
      "latency_peak_cuda_bytes": 1284208128,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "pooled_entry": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2543.965931020677,
        "cpu_seconds": 3047.307200656,
        "peak_rss_bytes": 17067089920,
        "gpu_peak_bytes": 28375515136,
        "samples": 2334
      },
      "formal_runner": {
        "wall_seconds": 2539.445438310504,
        "cpu_seconds": 2930.838562725,
        "peak_process_rss_bytes": 5303574528
      },
      "fit_including_validation": {
        "wall_seconds": 2415.562258735299,
        "cpu_seconds": 2758.594179099,
        "peak_process_rss_bytes": 5303574528
      },
      "prediction_export": {
        "wall_seconds": 118.96000053733587,
        "cpu_seconds": 166.62952875300016,
        "peak_process_rss_bytes": 5303574528
      },
      "smoke_worker": {
        "wall_seconds": 209.93510429561138,
        "cpu_seconds": 269.716068123,
        "peak_rss_bytes": 19876069376,
        "gpu_peak_bytes": 30000807936,
        "samples": 192
      },
      "smoke_runner": {
        "wall_seconds": 205.9832237586379,
        "cpu_seconds": 186.734289816,
        "peak_process_rss_bytes": 3371372544
      },
      "latency_constructor": {
        "wall_seconds": 0.7465212643146515,
        "cpu_seconds": 0.823265211999999,
        "peak_rss_bytes": 2134851584
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.14822092652320862,
        "cpu_seconds": 0.14773312300000185,
        "peak_rss_bytes": 2134851584
      },
      "latency_peak_cuda_bytes": 1284208128,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "pooled_work": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2530.5203538760543,
        "cpu_seconds": 3032.308274989,
        "peak_rss_bytes": 17086054400,
        "gpu_peak_bytes": 28371320832,
        "samples": 2322
      },
      "formal_runner": {
        "wall_seconds": 2526.054277524352,
        "cpu_seconds": 2917.193958029,
        "peak_process_rss_bytes": 5221584896
      },
      "fit_including_validation": {
        "wall_seconds": 2403.0723568201065,
        "cpu_seconds": 2746.774595737,
        "peak_process_rss_bytes": 5221584896
      },
      "prediction_export": {
        "wall_seconds": 117.90648832172155,
        "cpu_seconds": 164.90178650299958,
        "peak_process_rss_bytes": 5221584896
      },
      "smoke_worker": {
        "wall_seconds": 201.25274242460728,
        "cpu_seconds": 258.87551321999996,
        "peak_rss_bytes": 19847925760,
        "gpu_peak_bytes": 30000807936,
        "samples": 184
      },
      "smoke_runner": {
        "wall_seconds": 197.00315845757723,
        "cpu_seconds": 179.777079433,
        "peak_process_rss_bytes": 3362832384
      },
      "latency_constructor": {
        "wall_seconds": 0.7421007975935936,
        "cpu_seconds": 0.8177351739999992,
        "peak_rss_bytes": 2134851584
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.13428709656000137,
        "cpu_seconds": 0.13383748300000065,
        "peak_rss_bytes": 2134851584
      },
      "latency_peak_cuda_bytes": 1284208128,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "pooled_constant": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2552.4282142072916,
        "cpu_seconds": 3065.293133013,
        "peak_rss_bytes": 17054146560,
        "gpu_peak_bytes": 28369223680,
        "samples": 2343
      },
      "formal_runner": {
        "wall_seconds": 2548.5179096236825,
        "cpu_seconds": 2950.633148782,
        "peak_process_rss_bytes": 5300428800
      },
      "fit_including_validation": {
        "wall_seconds": 2424.5024347752333,
        "cpu_seconds": 2778.4093952010003,
        "peak_process_rss_bytes": 5300428800
      },
      "prediction_export": {
        "wall_seconds": 118.95606475323439,
        "cpu_seconds": 166.78306442300027,
        "peak_process_rss_bytes": 5300428800
      },
      "smoke_worker": {
        "wall_seconds": 204.48706128448248,
        "cpu_seconds": 261.991356354,
        "peak_rss_bytes": 19870900224,
        "gpu_peak_bytes": 30002905088,
        "samples": 187
      },
      "smoke_runner": {
        "wall_seconds": 200.56655617058277,
        "cpu_seconds": 182.56273697100002,
        "peak_process_rss_bytes": 3390656512
      },
      "latency_constructor": {
        "wall_seconds": 0.7515730932354927,
        "cpu_seconds": 0.8265921199999973,
        "peak_rss_bytes": 2135166976
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.13166256248950958,
        "cpu_seconds": 0.13121063000000177,
        "peak_rss_bytes": 2135166976
      },
      "latency_peak_cuda_bytes": 1284208128,
      "latency_peak_cuda_reserved_bytes": 1568669696
    }
  },
  "paired_cost": {
    "wall_seconds": 394.45777340978384,
    "cpu_seconds": 582.454817481,
    "peak_rss_bytes": 3401244672,
    "artifact_bytes": 1989009
  },
  "latency_cost": {
    "wall_seconds": 24.64083832502365,
    "cpu_seconds": 24.596968536,
    "peak_rss_bytes": 2135166976,
    "artifact_bytes": 163634,
    "peak_cuda_bytes": 1405035008
  },
  "csv_sha256": "49e1da4cbe78762e725abf56173688a5257aab63697cb04a18bdce5bfcf19d9b",
  "input_sha256": {
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/render_results.py": "536229675900768f99430d8646ead8897f3ae01fd59ff52f5989adc6f009faf2",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/protocol.json": "b39ead6427743f067fa0b17e5c7bd8d043563078ff50a3949f03bb155be9d52a",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/configs.json": "48fab6d4321a7c8b4de0abc18192687b95c1331dd87f4f35cac4daf6ce65a672",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/configs.json": "3c83c83720783e7ab531e62dc9739f0e164161f42043825520260e07859b8f45",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/configs.json": "cedc3fcaebc0ce4f1a4347d9d8e69a902c2d8b3fe03fc6c9495132da52b1ac56",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/paired_analysis.py": "68649b58954c8b74f9ebf906f24151d46bf8db8e50ee610aae682e3615d3de6a",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/single_effects.py": "83892a00f920a95d3019641104e884649f90030ef81bdd14c90072315ad48250",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/benchmark_latency.py": "5fe16a619e91647cc6f3f439d5c6a886244828714eb9837dd9957e628a0ee6df",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/quality-summary.json": "684a56619b7010149e6995fae22158aa8803b9ccb94d3c655d49a4c83d87d700",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/paired-analysis/completion.json": "be16e05f9daaae3830b4ef8d22cadf8511a6589a0516872b36ca1272fa66f12f",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/latency/completion.json": "1b796dbbf2c849cb6aa1f7aa53e7eb0c9dce0fdb226f08d372d48765beca84e7",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/paired-analysis/claim.json": "eb950a4797d078c4d8e21aec2ae0060c3b7eea25b57e9504b82a4c694cf761e1",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/paired-analysis/result.json": "1cf90bed3f2a5a542664f541d022fc03cb67f7bc2ab84700927e6dd2809b1949",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/paired-analysis/cost.json": "f70acec3bc2c2ef389c474b12421e288a1c245f2278ac37f87925b00363c3de4",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/latency/batch.json": "00460e7387a427a5cb5a5c0cc7bcd14a108a1b3c5cd6b8e53e38e84d4e0af1d8",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/latency/claim.json": "ee2a4da379f8d44203a6764e58de0985a946635614e50ac37e1dbe897251b41a",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/latency/context-binding.json": "11db11d1c075a3bbdfb1b1105321f5b19ac8911bea386b97a2a3e32c7b01e5b8",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/latency/cross8_mature.json": "9f36ef287ed8057eb334eba334bb5e79398f7cba7e97a6db4314015fea653dd0",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/latency/event_constant.json": "1d67869332d6e388ee44fe8d7757b968d4e1b940c3742c0c01fc351d02e2f9a6",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/latency/event_entry.json": "3af9a99fb2b0393d75fff3e13bd6b0d0fcfdf59f5f06bbfec1658cba9d48b4ec",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/latency/event_work.json": "fd18428a5e59aad2c3c4a3cecd91b3021f6bb2dd9febeba1e3bb81718ee0a270",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/latency/mature_r2.json": "3cfbd90f1a15266e995f5955f291dc465eea1f26a1ff85297ce3e69be3a137bb",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/latency/plan.json": "4359e7040339f107b32bd8704017433d266f1e28163512939c79f9602e6719a2",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/latency/pooled_constant.json": "af1b47e66499ce87fdc9ddfe12aaafc488af61f869940a7ad7e7e4c05afab3c7",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/latency/pooled_entry.json": "ad11bed4398947fec6688a07fc5f4c3dc8c7e73732287f633c3b9aeacf4f7600",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/latency/pooled_work.json": "1b37dca1755d7a0fee6ad090c62f1b96e4663b09a11a753bf7878cc458f7de07",
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
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/event_entry/result.json": "ac13577abdfedb68d2fcf706a127c80c40f3b92ffa7045fd9d973c2ba7914ace",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/event_entry/manifest.json": "df317fb4aa68b7a502e133629b1a5d793381947c293ef1267167206a21368b47",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/event_entry/formal/exit.json": "20efedf725750f34b2a94ae9bb01fd2508459c5e9f7db4e4ba6f3ea2bcc3078b",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/event_entry/smoke/exit.json": "b1ed9d8ead749c94bef663953fe7892a800c9ea0843db2323ac5684d94be0ee1",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/smoke-runs/event_entry/smoke.json": "12313dc0967bb78a7cb76259027bf159406fd6cac0cbd2c36bcbfbc84382862c",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/event_work/result.json": "65b206717b57488d8bc1b60dde847b5e5688d51f877e2aec753ef444724c33e8",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/event_work/manifest.json": "7b1b6da6d2d924f7504024cd77896b5c19b74f08c5e1e2f93c9957584153ce71",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/event_work/formal/exit.json": "05bf986ff374a8a816864cd94ce82d8374a1e3525c4c8876a809fe2f1c5736af",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/event_work/smoke/exit.json": "82c2d25850d7c8d66f9a34f69e96294ccd932936c4e3d214e929130d1c737c86",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/smoke-runs/event_work/smoke.json": "4b091ebe14f3bd82f03b9784c0deedd798e11c8ec9bbb506048713f50120c4e7",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/event_constant/result.json": "00600836e75b40c5bd45a68aa2a868a7b161e656f8902dc1c7c1a4e913ef4c83",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/event_constant/manifest.json": "a5a88e37afc2a87bb1a20dfd2425cfb7f03d5ea06a51de88951d4c1fb0ca5568",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/event_constant/formal/exit.json": "1d9745e4084f45922c18c5afc7abe4286bc9a73fe5883d2b1029a20c8677213c",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/event_constant/smoke/exit.json": "ebe64a7827d2e4a3942d3584828cf913b1f6c74d560353efa119e0748c361d84",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/smoke-runs/event_constant/smoke.json": "afbd950c5b83a93e703dfc292fb132dea7c56e8d77fb896832e9a3efb2910126",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/pooled_entry/result.json": "2a8756d1983d4d9de3291952463837a2dd0fdcc06b44b76516ac1bb721cf9b73",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/pooled_entry/manifest.json": "d6c274f63292488dd3d1550b38956a6837c8ad71fee589e800d378a7e7ea4ebd",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/pooled_entry/formal/exit.json": "90f24de815fe16362e5295c32418c396022d56caeae6ef12cc32dee95b880809",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/pooled_entry/smoke/exit.json": "bdf2438620514692b5fa2e5cc6600e20612e3db15f0c7ff8a83c436f9bde688b",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/smoke-runs/pooled_entry/smoke.json": "2beeee97289c21aa6f2001030e9b9a495d43c5c23eb66f9155b1b370590e9c00",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/pooled_work/result.json": "c88b7a68e53c64c355bc1a09b75174a04ae1cf63ecc889d3ccf7aed2ef5c14db",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/pooled_work/manifest.json": "127a4953dd524c52352ee47bf9accc3c2aa56640deabd13c26517f26c78c7b14",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/pooled_work/formal/exit.json": "f3309da03685cb13803ea533803ebe59f6b5f38702ff97785f47ccbc291b2d32",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/pooled_work/smoke/exit.json": "df714ad723a17fc218d30442e0685efcccd0fb6c25ea7b49b2e1608a84c63012",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/smoke-runs/pooled_work/smoke.json": "68d7631f5dc15273e2d968aec35520312e35a63ef35a4ff4ed56eac1d0d61f03",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/pooled_constant/result.json": "b37f43f7bec6a68d26cd3662f64543df5c1d9992d6b42031457ea3dec21a6d3d",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/runs/pooled_constant/manifest.json": "db42b383fc3e34fbf9544d875707834809a335190773f47433d3f061b4f8ded0",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/pooled_constant/formal/exit.json": "d73703b5143207c3827e5836ef3d0afabee1311141c5d9dbe8d3d188b72865ac",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/pooled_constant/smoke/exit.json": "50b87bbd2dad0ead34d9f30e083059ada14b7582cace76a4630db48c1fb1646d",
    "workspace/qformer-loop/taobao-transfer/recurrence-value-content/smoke-runs/pooled_constant/smoke.json": "5a6a9bb1ab122fdbe6fd321678f5e7b421e568480a210f46c97b4736353a2903"
  }
}
```
