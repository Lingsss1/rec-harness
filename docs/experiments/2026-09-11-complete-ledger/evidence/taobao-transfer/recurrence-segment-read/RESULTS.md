# 分段读取：保留一个候选，尚未确认检索改善

八组中仅 first_odd 满足既定正macro点估计与成本门槛，保留研究候选而非确认改进。相对原循环+0.001237919，区间跨零，点击AUC/BCE退步；相对第二轮同样筛选+0.005011363且区间全正，主要体现收藏收益。其余七组未达到正点估计或近似持平门槛，discard。未证实纯第二轮查询中介、时间近期性或跨seed稳定收益。

first_odd 相对 mature_r2 的 macro AUC 95% 探索性区间为 [−0.001372965,+0.003909335]；相对 Cross8 +0.008926451，区间 [+0.006004513,+0.012429006]。延迟14.313994ms，相对原循环+1.1287%，参数不增加。保留仅用于进一步结构消融，不替换原循环或旧发布协议 incumbent。

相对原循环，点击AUC−0.000875547，区间[−0.001735897,−0.000018985]；点击BCE+0.000251948，区间全正。加购、收藏、购买AUC点差均正，但各自区间跨零，不能宣称收藏或购买单任务改善已被证实。全部直接任务效果差见完整CSV。

first_odd−second_odd 的 macro 差+0.005011363，区间[+0.002250476,+0.007689342]；收藏AUC+0.019410865，区间全正，点击/加购/购买AUC区间跨零。prefix限制放第一轮也优于放第二轮；suffix和even的放置比较区间跨零，不能推广为“任何首轮稀疏化都有效”。第一轮偶数位筛选本身显著低于原循环，说明不能从一个正候选推出普遍的稀疏读取收益。

入口层一直读取完整历史；位置保留且不重排。因此该对照不是全模型新信息注入，也不是纯Q2中介实验。候选使用有效历史秩1,3,...，并非已验证的真实时间近期事件。偶数长度时odd包含最后有效事件而even不包含；奇数长度>1时二者均舍弃最后一项。这是后续应区分的结构因素，尚不是收益机制结论。优先检查最后事件保留与全序列覆盖的影响，保持相同seed，不加seed重复训练。

执行限制：原启动器和等待管线在会话中断后缺失终态；八个真实worker均保存成功退出/清理/预算记录，OS残留为exit0僵尸。独立核验全部worker与runner产物后继续首次统计和延迟测量，未重训。未补造启动器成功、总耗时或总清理记录。恢复证据见 reports/recovered-worker-terminal.json 和 recover_quality.py。配对500次/588区间及延迟十模型500前向均真实完成；区间不包含训练seed、多重比较或候选选择不确定性。

# Equal-count history support and work-round placement: actual results

八个新模型与两个已完成参考；固定 seed20262027、两轮、PSGDKron 每步预条件、LR1e-4、clip1。
28 个预定比较 × 21 项指标 = 588 个探索性区间。全部区间仅条件于保存模型，不含训练 seed 方差、多重比较或选择校正。

| model | macro AUC | parameters | forward ms | ratio to A |
| --- | --- | --- | --- | --- |
| mature_r2 | 0.767094951 | 51724260 | 14.1542 | 1.0000 |
| cross8_mature | 0.759406420 | 81616356 | 14.5267 | 1.0263 |
| first_prefix | 0.763802123 | 51724260 | 14.2322 | 1.0055 |
| first_suffix | 0.762223109 | 51724260 | 14.2487 | 1.0067 |
| first_even | 0.762419860 | 51724260 | 14.3490 | 1.0138 |
| first_odd | 0.768332871 | 51724260 | 14.3140 | 1.0113 |
| second_prefix | 0.754238908 | 51724260 | 14.2420 | 1.0062 |
| second_suffix | 0.764886352 | 51724260 | 14.2539 | 1.0070 |
| second_even | 0.764987189 | 51724260 | 14.3262 | 1.0121 |
| second_odd | 0.763321508 | 51724260 | 14.3440 | 1.0134 |

| comparison | macro delta | 95% interval |
| --- | --- | --- |
| first_prefix__minus__mature_r2 | -0.003292829 | [-0.006288188901790596, -0.0005409899799124383] |
| first_prefix__minus__cross8_mature | +0.004395702 | [0.001024663269957546, 0.007560991693365318] |
| first_suffix__minus__mature_r2 | -0.004871842 | [-0.007814593774694572, -0.0022479055763103472] |
| first_suffix__minus__cross8_mature | +0.002816689 | [-8.375742450305935e-05, 0.0056259086933696375] |
| first_even__minus__mature_r2 | -0.004675091 | [-0.007934792701201314, -0.0017019098487541366] |
| first_even__minus__cross8_mature | +0.003013440 | [-0.00012574522496085116, 0.006000488575488294] |
| first_odd__minus__mature_r2 | +0.001237919 | [-0.0013729652000838489, 0.003909334895993671] |
| first_odd__minus__cross8_mature | +0.008926451 | [0.006004513438290563, 0.012429005603353194] |
| second_prefix__minus__mature_r2 | -0.012856043 | [-0.017204751230466074, -0.00938755160395674] |
| second_prefix__minus__cross8_mature | -0.005167512 | [-0.008388430236713419, -0.002434450735359026] |
| second_suffix__minus__mature_r2 | -0.002208599 | [-0.005286403531037203, 0.0005428426213684278] |
| second_suffix__minus__cross8_mature | +0.005479932 | [0.00267475260652567, 0.008257294156149446] |
| second_even__minus__mature_r2 | -0.002107763 | [-0.005397456379620315, 0.0012215223930179182] |
| second_even__minus__cross8_mature | +0.005580769 | [0.0027951218077896605, 0.009058572060600572] |
| second_odd__minus__mature_r2 | -0.003773443 | [-0.0063889316344838635, -0.0012820873871544369] |
| second_odd__minus__cross8_mature | +0.003915088 | [0.0013812840607195912, 0.006997224457180522] |
| first_prefix__minus__second_prefix | +0.009563214 | [0.006487903188882395, 0.012791765389221813] |
| first_suffix__minus__second_suffix | -0.002663243 | [-0.005277334359706552, 8.642408780872287e-07] |
| first_even__minus__second_even | -0.002567328 | [-0.005659813985604775, 0.0002909665130584482] |
| first_odd__minus__second_odd | +0.005011363 | [0.0022504757905253397, 0.007689341673881042] |
| first_prefix__minus__first_even | +0.001382262 | [-0.0010283945236185838, 0.003997605283987673] |
| first_prefix__minus__first_odd | -0.004530748 | [-0.0073558261815326546, -0.0018547853348243565] |
| first_suffix__minus__first_even | -0.000196751 | [-0.0029292809844941895, 0.0025546791977236467] |
| first_suffix__minus__first_odd | -0.006109762 | [-0.009019188728277084, -0.0035315702830934657] |
| second_prefix__minus__second_even | -0.010748280 | [-0.014002541293038531, -0.008163577204027885] |
| second_prefix__minus__second_odd | -0.009082600 | [-0.012232931472353294, -0.00643034544077727] |
| second_suffix__minus__second_even | -0.000100836 | [-0.003079744334956494, 0.002707492664425175] |
| second_suffix__minus__second_odd | +0.001564844 | [-0.0009105712924647728, 0.004029780782590069] |

[全部588项结果 CSV](reports/results-all-effects.csv)：包含每任务 AUC/BCE 和直接任务效果差。负 BCE 差有利；任务差不是整体收益。

归因边界：加权方差与均值比较统计量，与均匀方差比较第一轮寻址的作用，与常量比较样本条件信息。辅助 FP32 softmax 不保证逐位等于 fused BF16 注意力。方差不是校准不确定性；相同参数量不代表相同有效容量或优化器轨迹。只有内部对照收益不足以证明优于原循环。

八个正式拟合共17544次更新，八个真实 smoke 共16次更新；旧两个参考不重训。延迟在同一GPU0和同一1024行成熟输入上各10次预热+40次计时，共500次前向。

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
        "wall_seconds": 0.7671919912099838,
        "cpu_seconds": 0.8355612949999998,
        "peak_rss_bytes": 1524813824
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.18212484568357468,
        "cpu_seconds": 0.1806749389999993,
        "peak_rss_bytes": 1524813824
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
        "wall_seconds": 1.0957261770963669,
        "cpu_seconds": 1.1800403950000007,
        "peak_rss_bytes": 2092183552
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.23573662340641022,
        "cpu_seconds": 0.23443466900000054,
        "peak_rss_bytes": 2092183552
      },
      "latency_peak_cuda_bytes": 1405035008,
      "latency_peak_cuda_reserved_bytes": 1713373184
    },
    "first_prefix": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2515.3734920471907,
        "cpu_seconds": 3027.6236540930004,
        "peak_rss_bytes": 17074065408,
        "gpu_peak_bytes": 28369223680,
        "samples": 2307
      },
      "formal_runner": {
        "wall_seconds": 2511.4031151235104,
        "cpu_seconds": 2913.355100181,
        "peak_process_rss_bytes": 5250736128
      },
      "fit_including_validation": {
        "wall_seconds": 2386.5013393461704,
        "cpu_seconds": 2735.4487980989998,
        "peak_process_rss_bytes": 5250736128
      },
      "prediction_export": {
        "wall_seconds": 119.57211645692587,
        "cpu_seconds": 172.42904857799977,
        "peak_process_rss_bytes": 5250736128
      },
      "smoke_worker": {
        "wall_seconds": 140.26956152170897,
        "cpu_seconds": 153.914924593,
        "peak_rss_bytes": 16939466752,
        "gpu_peak_bytes": 29570891776,
        "samples": 129
      },
      "smoke_runner": {
        "wall_seconds": 135.68229084461927,
        "cpu_seconds": 134.544787276,
        "peak_process_rss_bytes": 3234557952
      },
      "latency_constructor": {
        "wall_seconds": 0.7381048575043678,
        "cpu_seconds": 0.8098489739999994,
        "peak_rss_bytes": 2092183552
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.13329923152923584,
        "cpu_seconds": 0.1324712789999989,
        "peak_rss_bytes": 2092183552
      },
      "latency_peak_cuda_bytes": 1281155072,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "first_suffix": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2515.9312758892775,
        "cpu_seconds": 3025.838640644,
        "peak_rss_bytes": 16970461184,
        "gpu_peak_bytes": 28369223680,
        "samples": 2308
      },
      "formal_runner": {
        "wall_seconds": 2512.0273484811187,
        "cpu_seconds": 2911.2831492200003,
        "peak_process_rss_bytes": 5224308736
      },
      "fit_including_validation": {
        "wall_seconds": 2386.3059087842703,
        "cpu_seconds": 2737.348164384,
        "peak_process_rss_bytes": 5224308736
      },
      "prediction_export": {
        "wall_seconds": 120.38198284059763,
        "cpu_seconds": 168.29038888500008,
        "peak_process_rss_bytes": 5224308736
      },
      "smoke_worker": {
        "wall_seconds": 142.06238438934088,
        "cpu_seconds": 157.28573032900002,
        "peak_rss_bytes": 16923885568,
        "gpu_peak_bytes": 29572988928,
        "samples": 130
      },
      "smoke_runner": {
        "wall_seconds": 138.9994521290064,
        "cpu_seconds": 137.70408842400002,
        "peak_process_rss_bytes": 3218051072
      },
      "latency_constructor": {
        "wall_seconds": 0.7834699973464012,
        "cpu_seconds": 0.8559188819999992,
        "peak_rss_bytes": 2092183552
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.156565360724926,
        "cpu_seconds": 0.1553112210000016,
        "peak_rss_bytes": 2092183552
      },
      "latency_peak_cuda_bytes": 1281155072,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "first_even": {
      "training_reused_reference": false,
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
        "wall_seconds": 0.7686734199523926,
        "cpu_seconds": 0.8429877159999997,
        "peak_rss_bytes": 2092183552
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.1590116173028946,
        "cpu_seconds": 0.15740078299999993,
        "peak_rss_bytes": 2092183552
      },
      "latency_peak_cuda_bytes": 1281155072,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "first_odd": {
      "training_reused_reference": false,
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
        "wall_seconds": 0.7723697051405907,
        "cpu_seconds": 0.8444687860000002,
        "peak_rss_bytes": 2092183552
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.12406931072473526,
        "cpu_seconds": 0.1233411509999982,
        "peak_rss_bytes": 2092183552
      },
      "latency_peak_cuda_bytes": 1281155072,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "second_prefix": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2519.057643711567,
        "cpu_seconds": 3030.651342093,
        "peak_rss_bytes": 17107640320,
        "gpu_peak_bytes": 28373417984,
        "samples": 2309
      },
      "formal_runner": {
        "wall_seconds": 2514.669528260827,
        "cpu_seconds": 2913.995312215,
        "peak_process_rss_bytes": 5292318720
      },
      "fit_including_validation": {
        "wall_seconds": 2389.772987000644,
        "cpu_seconds": 2740.770105793,
        "peak_process_rss_bytes": 5292318720
      },
      "prediction_export": {
        "wall_seconds": 119.00944044440985,
        "cpu_seconds": 167.03438005099997,
        "peak_process_rss_bytes": 5292318720
      },
      "smoke_worker": {
        "wall_seconds": 146.30966036766768,
        "cpu_seconds": 160.27972168399998,
        "peak_rss_bytes": 16942546944,
        "gpu_peak_bytes": 29572988928,
        "samples": 134
      },
      "smoke_runner": {
        "wall_seconds": 142.10140262544155,
        "cpu_seconds": 139.846790512,
        "peak_process_rss_bytes": 3240267776
      },
      "latency_constructor": {
        "wall_seconds": 0.7278218269348145,
        "cpu_seconds": 0.8003103280000019,
        "peak_rss_bytes": 2092183552
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.12073562294244766,
        "cpu_seconds": 0.11998605400000173,
        "peak_rss_bytes": 2092183552
      },
      "latency_peak_cuda_bytes": 1281155072,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "second_suffix": {
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
    "second_even": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2531.232098005712,
        "cpu_seconds": 3039.49522237,
        "peak_rss_bytes": 17119875072,
        "gpu_peak_bytes": 28371320832,
        "samples": 2322
      },
      "formal_runner": {
        "wall_seconds": 2527.1809554323554,
        "cpu_seconds": 2924.2699800670002,
        "peak_process_rss_bytes": 5262262272
      },
      "fit_including_validation": {
        "wall_seconds": 2401.580759651959,
        "cpu_seconds": 2750.4913705550002,
        "peak_process_rss_bytes": 5262262272
      },
      "prediction_export": {
        "wall_seconds": 120.27337439358234,
        "cpu_seconds": 168.22248308000007,
        "peak_process_rss_bytes": 5262262272
      },
      "smoke_worker": {
        "wall_seconds": 157.3881392478943,
        "cpu_seconds": 173.41980454999998,
        "peak_rss_bytes": 17164509184,
        "gpu_peak_bytes": 29996613632,
        "samples": 144
      },
      "smoke_runner": {
        "wall_seconds": 153.03456237912178,
        "cpu_seconds": 149.510633044,
        "peak_process_rss_bytes": 3293913088
      },
      "latency_constructor": {
        "wall_seconds": 0.7560282275080681,
        "cpu_seconds": 0.8299144679999984,
        "peak_rss_bytes": 2092183552
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.14342691749334335,
        "cpu_seconds": 0.142560256000003,
        "peak_rss_bytes": 2092183552
      },
      "latency_peak_cuda_bytes": 1281155072,
      "latency_peak_cuda_reserved_bytes": 1568669696
    },
    "second_odd": {
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
    }
  },
  "paired_cost": {
    "wall_seconds": 517.0917917117476,
    "cpu_seconds": 706.770203436,
    "peak_rss_bytes": 4199694336,
    "artifact_bytes": 2653828
  },
  "latency_cost": {
    "wall_seconds": 28.969647467136383,
    "cpu_seconds": 28.748373272,
    "peak_rss_bytes": 2092183552,
    "artifact_bytes": 212712,
    "peak_cuda_bytes": 1405035008
  },
  "csv_sha256": "b5dc8fc5936ac4e231f26a6f5d1ffac4bcc2d84f72ee0f5e2a741186fbb6a519",
  "input_sha256": {
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/render_results.py": "51790f127bc808aeb5fec2f35159fb287574e469dadb8a545cf7adc6b1c2f0ce",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/analysis-protocol.json": "06e681c72719d6308e5d43858b37422fe7995f40502eb7d564789fe8e2ca8c4e",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/configs.json": "946ebad146177d8e36c010166e891949d5cbaf519e72eafd880605456938dcba",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/configs.json": "3c83c83720783e7ab531e62dc9739f0e164161f42043825520260e07859b8f45",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/configs.json": "cedc3fcaebc0ce4f1a4347d9d8e69a902c2d8b3fe03fc6c9495132da52b1ac56",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/paired_analysis.py": "f25a77ef5cd8358e06dd09b128687235738cdde4f81aa06de8ad6ddf2624ca37",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/single_effects.py": "f77bfae80dc4fdabdc69ec54ca3bcced06eeefcde5ac614d486c76d05ee88d7a",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/benchmark_latency.py": "1f1d6993b0720614e49ef0106a9a8dcc56efe56672e92cb83c34f4d1ac279df4",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/quality-summary.json": "8012e1ea78f95e8d9e1fae8703bb4293ceccdb501b4b3995808698433a18abbd",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/paired-analysis/completion.json": "af1ea803460caa43b57f93ceb6a8c38dd7f9726fb5417af5087c184c10f4d357",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/latency/completion.json": "9ec8673fe2de190eb57199cd81a11d4ecca8c8f91014eb4d0920be104bb923ee",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/paired-analysis/claim.json": "7da1d09de66f8f2385bc8840721e58c44674ad9ea6f3569f595f6f32da6efdd3",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/paired-analysis/result.json": "355aeed2737d2b2ab8e3f1774c5ee81be23f7b580461e5cfdcf7f41fec3024af",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/paired-analysis/cost.json": "81a17f04e777a982e404220ab216ade0a5add50f8cf5bd32024c48b5ef19b0c2",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/latency/batch.json": "8e62973c2c1487e9530fe772f7a2fa87263da01f6d38af508f85384d1a061b28",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/latency/claim.json": "4a382690b5360a30122694792566c7b173de3c7238619d8a502adc8d69773161",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/latency/context-binding.json": "ca137fe196266ab8d9ebbf03b6a76b5b5c9a96bb6f887f6399231effb215e726",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/latency/cross8_mature.json": "06cb2ea35e8df303697215bae2433d15dcdf0b4d7ffa7c8316e98d2e8c81b3cc",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/latency/first_even.json": "d2d43b98d904922b3df8f5f0d31a59cf266d33b0b71ba320dbb9e7cb4d5e6e19",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/latency/first_odd.json": "57ab68648d920ec3b312c302ef6c567e7f121ca1dcbdd04ed83e85471c2ae473",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/latency/first_prefix.json": "0c055fa2c7dd774e97b94ae5abc7661715d8b53877958481c50eedbef4b0695b",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/latency/first_suffix.json": "f27d0f8dc96bf757c1acbafe5c88e6b838a2646163f5196833d189f651f9052a",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/latency/mature_r2.json": "b84e5dcffe5748c2f457d57f4cbfa6ec07bd75774599e1fde834d1d43c47a27e",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/latency/plan.json": "15cf36996574cfcb455ac27af2b6fe1ae0211f03df5b1ff14c8de64c09a2651c",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/latency/second_even.json": "5edc596d2f40004db09d1e414e30a80e920c47a983513c55b9b5edb5a4b08e46",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/latency/second_odd.json": "27d574a396d082787d651c198e69534055376095a012ad6c053da9be8f0c0623",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/latency/second_prefix.json": "894a02ab5886b61934f3fa289961cfa0481b49f646b7f66e84db592396c737eb",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/latency/second_suffix.json": "89989708858129d3f0619bf2c579835160b24d5bbe98a4c5db3b6a8b286b5761",
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
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_prefix/result.json": "233d9271226e854abc905c37ba8ab23eae8c21001e35a6124a1e0e3ef01be8d3",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_prefix/manifest.json": "8ccd149a5dfe675c91f264895f0e3432a328cf0062ba2cf2ae0f326cbbe32a83",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/first_prefix/formal/exit.json": "2526ff2779aad554630d46ad282ff4a806b6f3f942deb03c26d525aeaba9258b",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/first_prefix/smoke/exit.json": "a9ff781918121837c326a66695562e7a237aea6ac78ad17d279de3a47474231c",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/smoke-runs/first_prefix/smoke.json": "73ae019e798422660df2080694bd2ea6a00ca626a504bcc1b04a70d734858706",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_suffix/result.json": "b1449afb1b9b5ae124a3336bb03477ff93c7e815d83d0dc954f45f5eafcdda9c",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/first_suffix/manifest.json": "1c1d7a1b1b50f064b4434b81fa61cdca12c75c2c08d9bb73df2e748af91d174e",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/first_suffix/formal/exit.json": "a3d8b290ddd57f680d86cf30292ba300d753dc6528e63991ef5f6e7a772500a1",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/first_suffix/smoke/exit.json": "e7dfd635c478206ef555d92a0bb861e968c730adfde36379bae8e4b52e1d4375",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/smoke-runs/first_suffix/smoke.json": "679cb74ba61679cfece190623ac0fb2e4da174518a77c324aeb5dd18a95d99a0",
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
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_prefix/result.json": "cb5ac16cae01ec500e55e43976530414b4b8fe4863f39046eda353ac8c92fcfc",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_prefix/manifest.json": "e8875b5e68c8e6c46ac40a236a466f8b4a7a845967d895a85ed17edce7cbcce8",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/second_prefix/formal/exit.json": "b47d78ff97e9ced6cb9ed8478f92a58aad68190a6cf736b3d2326bb00142f835",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/second_prefix/smoke/exit.json": "b60159c3cc4a27c53ac836bafe9fd1976d4b663c7830fe9d11bfe063190b0b33",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/smoke-runs/second_prefix/smoke.json": "83ea593a25eb841b718d7cb68abc9f07c9237d592eec226daeec40353579c090",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_suffix/result.json": "2784289062f3197dc82eaa446a39c5db355939b25b78d62a4021de5430901b8a",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_suffix/manifest.json": "8892ecd991f995a539b1cf1a214f36c125166622f5d5f2fcd5695443d980db47",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/second_suffix/formal/exit.json": "7d9a4238a75670743c86fd959abd33efc5381cd2bf16d6f3e8381eb9ad1c8859",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/second_suffix/smoke/exit.json": "468c082e9fb9ebb89d82e367b53dedafcd3b8cb9232ba0515df4a39d111e74d1",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/smoke-runs/second_suffix/smoke.json": "0e4126e2fdf2da7a3a2b0c7c86755077eaffff9db94fc8161a2f7caa64fa5ff8",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_even/result.json": "75935c3862faaede9ac4bc6ae383589d520c0bbb6563b940b7e72959bd662e05",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_even/manifest.json": "6437b589340aca8da768e42b7850194dc11e164259ad5ec86943bafe90c9a85e",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/second_even/formal/exit.json": "82585ca0edcca576ce87309c55004be0082731c52445f39daf6b6540f9ef4e31",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/second_even/smoke/exit.json": "6bb329620ffd532013599f57e399549b2c7d15701b39f0bde185234977ed5c72",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/smoke-runs/second_even/smoke.json": "02af5b9ad08e1c26113ba4d46548d00e530590859bb51e0122c992e5269571e2",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_odd/result.json": "2dd97782b3d5fd821f0da8c9b86c8667be4398361c818a8e1dfb5e37e6265f53",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/runs/second_odd/manifest.json": "f68199c4064e85d6fa7b93d5594b5aa3f2057d50f52c0d7b5d07898ed4b58de5",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/second_odd/formal/exit.json": "62060b9ae8a4d10e03ce151e4f1765b7b9eae10b84b5c200a864d18c12f5e54e",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/second_odd/smoke/exit.json": "f81c39ff676a962ac42632df0332e61524db07c01a20d9d96b0fcd6def2ccc96",
    "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/smoke-runs/second_odd/smoke.json": "8be09a794fad0e7c4b6c5cfe60545fa24ad9d18c1e0980cb858f91267ed15e50"
  }
}
```
