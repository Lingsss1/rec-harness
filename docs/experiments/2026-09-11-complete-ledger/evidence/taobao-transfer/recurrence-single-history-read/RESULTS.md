# 单次历史编码与第二轮条件读取：最终结果

本轮八组均未成为优化候选：相对匹配的成熟输入两轮基线 A，所有新模型的 macro AUC 点估计更低，前向延迟更高。按预定规则全部记为 discard，保留模型、预测和全部统计结果用于研究；继续使用 A，旧 released 协议 incumbent 不变。这不等于已证明每个模型都显著退步。

本轮最佳 `mlp_10`（首轮读编码历史、次轮读原始历史）macro AUC 为0.76618163，A为0.76709495；差值−0.00091332，95%用户重采样区间[−0.00361977,+0.00169206]。它慢17.25%（16.65665ms 对14.20594ms），未达到“点差≥−0.0001且延迟比≤1.05”的近候选条件。它比匹配Cross8的macro高0.00677521，区间[+0.00411302,+0.00965496]，但不足以替换已更强、更快的A。八个模型与Cross8的完整比较均在下表中。

工作状态门控尚无优于入口门控的macro证据：`mlp_gate_work−mlp_gate_entry`为−0.00212721，区间[−0.00507221,+0.00115429]。四个任务存在取舍：click/cart/fav的AUC差值区间均负；buy点差+0.00631241，但区间[−0.00190335,+0.01318033]跨零，不能称购买排序收益已确认。buy BCE则下降0.000131614，区间[−0.000218715,−0.0000413125]，其余三任务BCE区间均正。直接的cart−buy、fav−buy AUC效果差区间为[−0.02398221,−0.00288219]、[−0.02250147,−0.00238297]，支持这个门控对照中的任务效果差异；不能推广为“只有购买会从循环中得到信息”。

读取位置的结果依赖编码器。MLP01−MLP10的macro为−0.00691809，区间[−0.01039986,−0.00377544]；MLP11−MLP10为−0.00482566，区间[−0.00783992,−0.00210665]。这支持在本轮MLP家族中第二轮保留原始历史的相对优势，仍没有得到优于A的模型。三种attention读取顺序之间的macro区间均跨零；不能把MLP结果推广成普遍的循环读取规律。SA01相对A的macro上界仅−0.00000191，是贴近零的探索性结果，也不宜夸大。

全部新模型的buy AUC点值均低于A，但不声称八个buy区间均为负。MLP10相对A的fav AUC差值+0.00586328、区间[+0.00024996,+0.01193470]，同时buy AUC为−0.00681727、区间[−0.01430454,+0.00012076]；buy BCE则有正向损失差，区间[+0.00015412,+0.00031095]。AUC与BCE的范围应分别解释。

本轮用一次共同的500次用户重采样计算29个预定对比、609个区间；没有增加训练seed或重复旧差值训练。所有区间条件于单个训练seed和当前checkpoints，没有训练随机性、多重比较或选择校正。这里得到的是完整保存的负向候选结果和局部结构取舍，尚未证明更好的第二轮检索条件。下一步可审查保留原Q/K和原始读取，仅添加工作条件化读值的结构；本轮结果不能证明失败原因必然是K的变化。

以下为已核对的全部模型、任务、对比和成本；原始渲染草稿另存于 `RESULTS-draft.md`。

**实际模型质量与前向成本**

| 模型 | 训练来源 | click AUC | cart AUC | fav AUC | buy AUC | macro AUC | 实际参数 | 前向 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mature_r2 | 历史参考 | 0.64993629 | 0.79243427 | 0.80246349 | 0.82354575 | 0.76709495 | 51724260 | 14.206 |
| cross8_mature | 历史参考 | 0.64993677 | 0.78920674 | 0.79650201 | 0.80198016 | 0.75940642 | 81616356 | 14.601 |
| hist_shallow_deep | 历史参考 | 0.65326132 | 0.79065795 | 0.80665651 | 0.81896120 | 0.76738424 | 51955556 | 20.827 |
| sa_01 | 本轮新训练 | 0.65072595 | 0.79390494 | 0.80613342 | 0.80779281 | 0.76463928 | 51955556 | 17.593 |
| sa_11 | 本轮新训练 | 0.65146926 | 0.78940072 | 0.79893192 | 0.82001337 | 0.76495382 | 51955556 | 17.524 |
| sa_10 | 本轮新训练 | 0.65155504 | 0.79372941 | 0.79776892 | 0.81755408 | 0.76515187 | 51955556 | 17.624 |
| mlp_01 | 本轮新训练 | 0.65045396 | 0.78877477 | 0.78932385 | 0.80850156 | 0.75926354 | 51889252 | 16.665 |
| mlp_11 | 本轮新训练 | 0.64905092 | 0.78792353 | 0.79724922 | 0.81120021 | 0.76135597 | 51889252 | 16.666 |
| mlp_10 | 本轮新训练 | 0.64943366 | 0.79023760 | 0.80832678 | 0.81672848 | 0.76618163 | 51889252 | 16.657 |
| mlp_gate_entry | 本轮新训练 | 0.65042096 | 0.79648450 | 0.79653182 | 0.80982572 | 0.76331575 | 51915364 | 18.889 |
| mlp_gate_work | 本轮新训练 | 0.64937548 | 0.78902672 | 0.79021384 | 0.81613814 | 0.76118854 | 51915364 | 18.906 |

macro 为四任务 AUC 等权平均。全部模型均采用成熟输入；原始 released incumbent 的 raw AUC 不能与这些 mature 分数直接排名。

**四任务 BCE 与源 logloss**（每格为配对分析重算 BCE / 训练结果保存的 logloss）

| 模型 | click BCE / logloss | cart BCE / logloss | fav BCE / logloss | buy BCE / logloss |
| --- | --- | --- | --- | --- |
| mature_r2 | 0.184391646 / 0.184391646 | 0.0218279544 / 0.0218279543 | 0.0167981577 / 0.0167981577 | 0.00964380194 / 0.00964380193 |
| cross8_mature | 0.184317938 / 0.184317938 | 0.0218344116 / 0.0218344116 | 0.0169733792 / 0.0169733792 | 0.0100073903 / 0.0100073903 |
| hist_shallow_deep | 0.18363845 / 0.18363845 | 0.021811822 / 0.021811822 | 0.0168238156 / 0.0168238156 | 0.00971450595 / 0.00971450592 |
| sa_01 | 0.184037737 / 0.184037737 | 0.0218363924 / 0.021836391 | 0.0167792904 / 0.0167792904 | 0.00984264687 / 0.0098426468 |
| sa_11 | 0.183901063 / 0.183901063 | 0.0219010095 / 0.0219010095 | 0.0169921625 / 0.0169921624 | 0.0098953725 / 0.0098953725 |
| sa_10 | 0.183875538 / 0.183875538 | 0.0217918384 / 0.0217918384 | 0.0171711484 / 0.0171711484 | 0.00997989559 / 0.00997989554 |
| mlp_01 | 0.184258541 / 0.184258541 | 0.0218984232 / 0.0218984229 | 0.0170516184 / 0.0170516184 | 0.00998579834 / 0.00998579834 |
| mlp_11 | 0.18467221 / 0.18467221 | 0.0219561967 / 0.0219561967 | 0.0169676163 / 0.0169676163 | 0.00984007312 / 0.00984007311 |
| mlp_10 | 0.184088627 / 0.184088627 | 0.0218883352 / 0.0218885269 | 0.0166501121 / 0.0166501116 | 0.00987779039 / 0.00987779034 |
| mlp_gate_entry | 0.18403209 / 0.18403209 | 0.0217739357 / 0.0217739356 | 0.0169465669 / 0.0169465668 | 0.00998645178 / 0.00998645178 |
| mlp_gate_work | 0.184225451 / 0.184225451 | 0.0220050311 / 0.0220050311 | 0.0172715962 / 0.0172715961 | 0.009854838 / 0.009854838 |

两种损失口径均原样保留。配对 BCE 使用源预测 dtype 的 epsilon 裁剪和 float64 log；不要求与原 sklearn logloss 逐位相等，不新增 macro BCE 选择目标。

**全部 29 个预定比较**（候选减参考；macro AUC 点差与 95% 区间）

| 比较 | macro AUC：点差 [下界, 上界] |
| --- | --- |
| sa_01__minus__mature_r2 | -0.00245567 [-0.00498069, -0.00000191] |
| sa_11__minus__mature_r2 | -0.00214114 [-0.00474877, +0.00039578] |
| sa_10__minus__mature_r2 | -0.00194309 [-0.00499568, +0.00060710] |
| mlp_01__minus__mature_r2 | -0.00783142 [-0.01137708, -0.00409488] |
| mlp_11__minus__mature_r2 | -0.00573898 [-0.00935142, -0.00235989] |
| mlp_10__minus__mature_r2 | -0.00091332 [-0.00361977, +0.00169206] |
| mlp_gate_entry__minus__mature_r2 | -0.00377920 [-0.00735817, -0.00072761] |
| mlp_gate_work__minus__mature_r2 | -0.00590641 [-0.00892655, -0.00283662] |
| sa_01__minus__cross8_mature | +0.00523286 [+0.00227100, +0.00842403] |
| sa_11__minus__cross8_mature | +0.00554739 [+0.00246015, +0.00881592] |
| sa_10__minus__cross8_mature | +0.00574545 [+0.00280011, +0.00891876] |
| mlp_01__minus__cross8_mature | -0.00014288 [-0.00300795, +0.00288076] |
| mlp_11__minus__cross8_mature | +0.00194955 [-0.00083730, +0.00517174] |
| mlp_10__minus__cross8_mature | +0.00677521 [+0.00411302, +0.00965496] |
| mlp_gate_entry__minus__cross8_mature | +0.00390933 [+0.00132172, +0.00710019] |
| mlp_gate_work__minus__cross8_mature | +0.00178212 [-0.00122230, +0.00494047] |
| sa_01__minus__sa_11 | -0.00031454 [-0.00323375, +0.00229906] |
| sa_01__minus__sa_10 | -0.00051259 [-0.00360411, +0.00250608] |
| sa_11__minus__sa_10 | -0.00019805 [-0.00299565, +0.00248094] |
| mlp_01__minus__mlp_11 | -0.00209243 [-0.00491503, +0.00095179] |
| mlp_01__minus__mlp_10 | -0.00691809 [-0.01039986, -0.00377544] |
| mlp_11__minus__mlp_10 | -0.00482566 [-0.00783992, -0.00210665] |
| mlp_01__minus__sa_01 | -0.00537574 [-0.00878432, -0.00188846] |
| mlp_11__minus__sa_11 | -0.00359784 [-0.00705149, -0.00047725] |
| mlp_10__minus__sa_10 | +0.00102977 [-0.00181405, +0.00427903] |
| mlp_gate_work__minus__mlp_gate_entry | -0.00212721 [-0.00507221, +0.00115429] |
| mlp_gate_entry__minus__mlp_01 | +0.00405222 [+0.00070626, +0.00766386] |
| mlp_gate_work__minus__mlp_01 | +0.00192501 [-0.00123708, +0.00524804] |
| sa_01__minus__hist_shallow_deep | -0.00274497 [-0.00582678, +0.00043294] |

完整 [609 行 CSV](reports/results-all-effects.csv) 包含 29 个 macro AUC、116 个任务 AUC、116 个任务 BCE、174 个直接任务 AUC 差和 174 个直接任务 BCE 差；保留所有正值、负值及跨零区间。
AUC 正差、BCE 负差有利。六个直接任务差均在同一比较、同一用户 draw 内计算；不能将“一项显著、另一项不显著”当成任务间差异，也不能据此称只有某任务拥有循环信息。跨零表示方向未确定，不表示等效。

**预定候选条件：事实与最终决策分开**

| 模型 | Δmacro/A | 质量点差>0 | 点差≥-0.0001 | 延迟/A | 延迟≤1.05 | 近候选双门槛 | 延迟/X | 延迟/SD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sa_01 | -0.00245567 | 否 | 否 | 1.2384 | 否 | 否 | 1.2050 | 0.8447 |
| sa_11 | -0.00214114 | 否 | 否 | 1.2336 | 否 | 否 | 1.2002 | 0.8414 |
| sa_10 | -0.00194309 | 否 | 否 | 1.2406 | 否 | 否 | 1.2071 | 0.8462 |
| mlp_01 | -0.00783142 | 否 | 否 | 1.1731 | 否 | 否 | 1.1414 | 0.8002 |
| mlp_11 | -0.00573898 | 否 | 否 | 1.1732 | 否 | 否 | 1.1415 | 0.8002 |
| mlp_10 | -0.00091332 | 否 | 否 | 1.1725 | 否 | 否 | 1.1408 | 0.7998 |
| mlp_gate_entry | -0.00377920 | 否 | 否 | 1.3297 | 否 | 否 | 1.2937 | 0.9069 |
| mlp_gate_work | -0.00590641 | 否 | 否 | 1.3309 | 否 | 否 | 1.2949 | 0.9078 |

质量候选以对 A 的正 macro 点差定义；近候选同时要求 Δmacro≥−0.0001 与前向延迟比≤1.05。这些是既有队列条件，不是等效性检验或自动保留；最终取舍须结合全部任务、区间和实际成本。

**条件来源与结构解释边界**

- mlp_gate_work−mlp_gate_entry 比较门控条件 z1 与 e；两者按协议匹配门控参数、构造初始化和调用次数。差值包含可微训练路径与优化器轨迹的改变，不能解释为固定权重下抽取了多少首轮信息。
- 两个 gate 模型与 mlp_01 的比较同时增加门控容量与调制；不能将收益单独归因于第一轮工作信息。
- 同顺序 MLP−SA 同时改变历史编码器的交互方式、容量和计算；它不是只关闭 self-attention 的等容量对照。
- 同编码器内 01−11 改第一读，11−10 改第二读，01−10 交换两次读取的版本。与 A 比较增加编码结构；与 X 比较还包含 q4/q8 容量差异，均不能单独证明循环因果收益。
- sa_01−旧 SD 同时将两次 G 改为一次，并把 H1/H2 改为 H0/H1；不能称为仅省去一次相同历史编码而保持检索条件不变。
- 所有新模型固定两轮、无 prediction→work 反馈、原四任务读出；只有预定初始化组内部承诺构造和静态布局匹配，不声称跨组或跨训练全程优化器动态相同。

**训练与计时成本**

| 模型 | 训练成本归属 | 正式 worker s | 正式 CPU s | fit含验证 s | 导出 s | smoke s | smoke CPU s | 计时峰值 GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mature_r2 | 历史已发生 | 2534.68 | 3041.43 | 2408.73 | 117.28 | 188.15 | 315.24 | 1.193 |
| cross8_mature | 历史已发生 | 2522.78 | 3028.68 | 2390.84 | 121.88 | 145.57 | 159.26 | 1.309 |
| hist_shallow_deep | 历史已发生 | 2898.32 | 3421.14 | 2753.96 | 135.69 | 382.39 | 746.00 | 1.487 |
| sa_01 | 本轮 | 2735.90 | 3266.09 | 2599.00 | 126.96 | 214.02 | 242.12 | 1.341 |
| sa_11 | 本轮 | 2729.33 | 3254.00 | 2593.09 | 126.72 | 220.35 | 247.29 | 1.341 |
| sa_10 | 本轮 | 2742.39 | 3266.39 | 2604.69 | 128.03 | 220.26 | 248.91 | 1.341 |
| mlp_01 | 本轮 | 2693.62 | 3207.07 | 2559.07 | 124.26 | 363.31 | 813.67 | 1.414 |
| mlp_11 | 本轮 | 2688.21 | 3214.71 | 2552.61 | 125.25 | 330.04 | 648.56 | 1.414 |
| mlp_10 | 本轮 | 2686.68 | 3200.81 | 2550.92 | 125.67 | 331.71 | 640.18 | 1.414 |
| mlp_gate_entry | 本轮 | 2797.81 | 3322.49 | 2655.06 | 132.64 | 363.31 | 732.63 | 1.560 |
| mlp_gate_work | 本轮 | 2795.63 | 3308.54 | 2655.28 | 131.23 | 372.19 | 748.02 | 1.560 |

本轮新增 8×2193=17544 次正式更新、8×2=16 次真实 smoke 更新；seed 固定 20262027。旧 A、X、SD 的训练和 smoke 成本仅作历史参考，不计作本轮重训。
fit、导出、runner 是 worker 内部子项，不能与 worker 总成本重复相加；并行 worker 墙钟之和不是批次总历时，RSS 峰值不能相加。表内计时峰值为前向事件测量的 CUDA allocated 峰值，训练和 smoke 的实际资源原值见下列附录。

11 个模型在同一 GPU、BF16、同一个成熟输入 1024 行完整 batch 上各执行 10 次预热＋40 次计时，共 550 次前向。只有一次输入转换及一次 batch 搬运，旧三个 checkpoint 本轮重新计时；这不是重新训练。
CUDA Event 的模型前向延迟不含数据读取、时间查找、成熟转换、搬运、模型构造和 checkpoint 加载；总进程成本包含这些阶段。模型之间的比较是这一个固定 batch 上的实测，不代表完整线上请求分布。

<details><summary>完整已保存成本与范围（原值）</summary>

```json
{
  "models": {
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
        "wall_seconds": 0.7728349268436432,
        "cpu_seconds": 0.8449352790000004,
        "peak_rss_bytes": 1531215872
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.1827087178826332,
        "cpu_seconds": 0.1819560229999997,
        "peak_rss_bytes": 1531215872
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
        "wall_seconds": 1.0972465574741364,
        "cpu_seconds": 1.1762366010000012,
        "peak_rss_bytes": 2085142528
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.213261179625988,
        "cpu_seconds": 0.21245469399999983,
        "peak_rss_bytes": 2085142528
      },
      "latency_peak_cuda_bytes": 1405035008,
      "latency_peak_cuda_reserved_bytes": 1713373184
    },
    "hist_shallow_deep": {
      "training_reused_reference": true,
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
        "wall_seconds": 0.7399371191859245,
        "cpu_seconds": 0.8203377210000014,
        "peak_rss_bytes": 2085142528
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.1145358607172966,
        "cpu_seconds": 0.1140523120000001,
        "peak_rss_bytes": 2085142528
      },
      "latency_peak_cuda_bytes": 1596911104,
      "latency_peak_cuda_reserved_bytes": 1885339648
    },
    "sa_01": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2735.8964571282268,
        "cpu_seconds": 3266.0868918600004,
        "peak_rss_bytes": 17174323200,
        "gpu_peak_bytes": 32846643200,
        "samples": 2508
      },
      "formal_runner": {
        "wall_seconds": 2731.286681301892,
        "cpu_seconds": 3137.908126878,
        "peak_process_rss_bytes": 5363322880
      },
      "fit_including_validation": {
        "wall_seconds": 2598.997734941542,
        "cpu_seconds": 2958.509525001,
        "peak_process_rss_bytes": 5363322880
      },
      "prediction_export": {
        "wall_seconds": 126.9584673345089,
        "cpu_seconds": 173.91830803700032,
        "peak_process_rss_bytes": 5363322880
      },
      "smoke_worker": {
        "wall_seconds": 214.01883608847857,
        "cpu_seconds": 242.118427847,
        "peak_rss_bytes": 18256908288,
        "gpu_peak_bytes": 34518073344,
        "samples": 195
      },
      "smoke_runner": {
        "wall_seconds": 209.05336381495,
        "cpu_seconds": 201.525273956,
        "peak_process_rss_bytes": 3588407296
      },
      "latency_constructor": {
        "wall_seconds": 0.7607376798987389,
        "cpu_seconds": 0.8288122829999995,
        "peak_rss_bytes": 2127491072
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.1378684788942337,
        "cpu_seconds": 0.13734789999999997,
        "peak_rss_bytes": 2127491072
      },
      "latency_peak_cuda_bytes": 1439624704,
      "latency_peak_cuda_reserved_bytes": 1728053248
    },
    "sa_11": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2729.3299817740917,
        "cpu_seconds": 3254.001422899,
        "peak_rss_bytes": 17137303552,
        "gpu_peak_bytes": 32838254592,
        "samples": 2501
      },
      "formal_runner": {
        "wall_seconds": 2725.120545618236,
        "cpu_seconds": 3126.140330797,
        "peak_process_rss_bytes": 5348847616
      },
      "fit_including_validation": {
        "wall_seconds": 2593.0873567461967,
        "cpu_seconds": 2946.3063217139998,
        "peak_process_rss_bytes": 5348847616
      },
      "prediction_export": {
        "wall_seconds": 126.71776077896357,
        "cpu_seconds": 174.3501731309998,
        "peak_process_rss_bytes": 5348847616
      },
      "smoke_worker": {
        "wall_seconds": 220.34590608626604,
        "cpu_seconds": 247.291189484,
        "peak_rss_bytes": 18178170880,
        "gpu_peak_bytes": 34518073344,
        "samples": 201
      },
      "smoke_runner": {
        "wall_seconds": 215.0680452734232,
        "cpu_seconds": 206.74445259200002,
        "peak_process_rss_bytes": 3583287296
      },
      "latency_constructor": {
        "wall_seconds": 0.7746501937508583,
        "cpu_seconds": 0.845911490999999,
        "peak_rss_bytes": 2127491072
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.1796056553721428,
        "cpu_seconds": 0.17891683599999908,
        "peak_rss_bytes": 2127491072
      },
      "latency_peak_cuda_bytes": 1439624704,
      "latency_peak_cuda_reserved_bytes": 1728053248
    },
    "sa_10": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2742.3903021663427,
        "cpu_seconds": 3266.386917536,
        "peak_rss_bytes": 17142169600,
        "gpu_peak_bytes": 32842448896,
        "samples": 2515
      },
      "formal_runner": {
        "wall_seconds": 2737.9571212530136,
        "cpu_seconds": 3139.25531185,
        "peak_process_rss_bytes": 5309968384
      },
      "fit_including_validation": {
        "wall_seconds": 2604.688973426819,
        "cpu_seconds": 2958.6417768439997,
        "peak_process_rss_bytes": 5309968384
      },
      "prediction_export": {
        "wall_seconds": 128.02782264351845,
        "cpu_seconds": 174.90762048299985,
        "peak_process_rss_bytes": 5309968384
      },
      "smoke_worker": {
        "wall_seconds": 220.2592827528715,
        "cpu_seconds": 248.913234207,
        "peak_rss_bytes": 18462494720,
        "gpu_peak_bytes": 34518073344,
        "samples": 201
      },
      "smoke_runner": {
        "wall_seconds": 215.4159775301814,
        "cpu_seconds": 206.778589205,
        "peak_process_rss_bytes": 3581857792
      },
      "latency_constructor": {
        "wall_seconds": 0.7632900476455688,
        "cpu_seconds": 0.8349890309999992,
        "peak_rss_bytes": 2127785984
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.16074050217866898,
        "cpu_seconds": 0.16013383599999997,
        "peak_rss_bytes": 2127785984
      },
      "latency_peak_cuda_bytes": 1439624704,
      "latency_peak_cuda_reserved_bytes": 1728053248
    },
    "mlp_01": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2693.6236927732825,
        "cpu_seconds": 3207.0729744530004,
        "peak_rss_bytes": 17061519360,
        "gpu_peak_bytes": 31582060544,
        "samples": 2470
      },
      "formal_runner": {
        "wall_seconds": 2688.7218604534864,
        "cpu_seconds": 3078.203309263,
        "peak_process_rss_bytes": 5295271936
      },
      "fit_including_validation": {
        "wall_seconds": 2559.0737301260233,
        "cpu_seconds": 2901.330851231,
        "peak_process_rss_bytes": 5295271936
      },
      "prediction_export": {
        "wall_seconds": 124.26068166643381,
        "cpu_seconds": 171.2875121269999,
        "peak_process_rss_bytes": 5295271936
      },
      "smoke_worker": {
        "wall_seconds": 363.31496554613113,
        "cpu_seconds": 813.674330908,
        "peak_rss_bytes": 21506883584,
        "gpu_peak_bytes": 34522267648,
        "samples": 332
      },
      "smoke_runner": {
        "wall_seconds": 358.1419350877404,
        "cpu_seconds": 288.676639968,
        "peak_process_rss_bytes": 3944697856
      },
      "latency_constructor": {
        "wall_seconds": 0.7607448697090149,
        "cpu_seconds": 0.8318885960000024,
        "peak_rss_bytes": 2127785984
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.14177099615335464,
        "cpu_seconds": 0.14119830499999964,
        "peak_rss_bytes": 2127785984
      },
      "latency_peak_cuda_bytes": 1517870080,
      "latency_peak_cuda_reserved_bytes": 1885339648
    },
    "mlp_11": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2688.214631251991,
        "cpu_seconds": 3214.7056538939996,
        "peak_rss_bytes": 17146290176,
        "gpu_peak_bytes": 31579963392,
        "samples": 2463
      },
      "formal_runner": {
        "wall_seconds": 2683.2708602547646,
        "cpu_seconds": 3086.7875545469997,
        "peak_process_rss_bytes": 5415366656
      },
      "fit_including_validation": {
        "wall_seconds": 2552.6138797700405,
        "cpu_seconds": 2907.9745994659997,
        "peak_process_rss_bytes": 5415366656
      },
      "prediction_export": {
        "wall_seconds": 125.24789342284203,
        "cpu_seconds": 173.1782290279998,
        "peak_process_rss_bytes": 5415366656
      },
      "smoke_worker": {
        "wall_seconds": 330.0408606007695,
        "cpu_seconds": 648.563692039,
        "peak_rss_bytes": 21133541376,
        "gpu_peak_bytes": 34520170496,
        "samples": 302
      },
      "smoke_runner": {
        "wall_seconds": 324.9010729715228,
        "cpu_seconds": 273.638816288,
        "peak_process_rss_bytes": 3967148032
      },
      "latency_constructor": {
        "wall_seconds": 0.757634311914444,
        "cpu_seconds": 0.828613777000001,
        "peak_rss_bytes": 2127785984
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.15426567941904068,
        "cpu_seconds": 0.1536313800000002,
        "peak_rss_bytes": 2127785984
      },
      "latency_peak_cuda_bytes": 1517870080,
      "latency_peak_cuda_reserved_bytes": 1885339648
    },
    "mlp_10": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2686.6765558496118,
        "cpu_seconds": 3200.8075648920003,
        "peak_rss_bytes": 17117454336,
        "gpu_peak_bytes": 31582060544,
        "samples": 2465
      },
      "formal_runner": {
        "wall_seconds": 2681.9731072708964,
        "cpu_seconds": 3074.678935057,
        "peak_process_rss_bytes": 5262561280
      },
      "fit_including_validation": {
        "wall_seconds": 2550.924990721047,
        "cpu_seconds": 2895.549258471,
        "peak_process_rss_bytes": 5262561280
      },
      "prediction_export": {
        "wall_seconds": 125.67101988941431,
        "cpu_seconds": 173.53620460400043,
        "peak_process_rss_bytes": 5262561280
      },
      "smoke_worker": {
        "wall_seconds": 331.70812125504017,
        "cpu_seconds": 640.181904754,
        "peak_rss_bytes": 21112852480,
        "gpu_peak_bytes": 34520170496,
        "samples": 303
      },
      "smoke_runner": {
        "wall_seconds": 326.35462345182896,
        "cpu_seconds": 275.634385866,
        "peak_process_rss_bytes": 3954733056
      },
      "latency_constructor": {
        "wall_seconds": 0.7541157677769661,
        "cpu_seconds": 0.8249897829999995,
        "peak_rss_bytes": 2127785984
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.14292560517787933,
        "cpu_seconds": 0.14234471199999987,
        "peak_rss_bytes": 2127785984
      },
      "latency_peak_cuda_bytes": 1517870080,
      "latency_peak_cuda_reserved_bytes": 1885339648
    },
    "mlp_gate_entry": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2797.814953006804,
        "cpu_seconds": 3322.488007626,
        "peak_rss_bytes": 17079107584,
        "gpu_peak_bytes": 35361128448,
        "samples": 2567
      },
      "formal_runner": {
        "wall_seconds": 2793.009986616671,
        "cpu_seconds": 3192.404120614,
        "peak_process_rss_bytes": 5424553984
      },
      "fit_including_validation": {
        "wall_seconds": 2655.057724826038,
        "cpu_seconds": 3006.824734512,
        "peak_process_rss_bytes": 5424553984
      },
      "prediction_export": {
        "wall_seconds": 132.64384725689888,
        "cpu_seconds": 180.05422202399996,
        "peak_process_rss_bytes": 5424553984
      },
      "smoke_worker": {
        "wall_seconds": 363.3127840310335,
        "cpu_seconds": 732.6299671910001,
        "peak_rss_bytes": 21325135872,
        "gpu_peak_bytes": 38303432704,
        "samples": 333
      },
      "smoke_runner": {
        "wall_seconds": 358.2261445298791,
        "cpu_seconds": 297.617862696,
        "peak_process_rss_bytes": 4024729600
      },
      "latency_constructor": {
        "wall_seconds": 0.7524328827857971,
        "cpu_seconds": 0.8231612879999979,
        "peak_rss_bytes": 2127785984
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.1393095999956131,
        "cpu_seconds": 0.1387498100000002,
        "peak_rss_bytes": 2127785984
      },
      "latency_peak_cuda_bytes": 1675310080,
      "latency_peak_cuda_reserved_bytes": 2042626048
    },
    "mlp_gate_work": {
      "training_reused_reference": false,
      "formal_worker": {
        "wall_seconds": 2795.627824872732,
        "cpu_seconds": 3308.538712535,
        "peak_rss_bytes": 17096470528,
        "gpu_peak_bytes": 35356934144,
        "samples": 2565
      },
      "formal_runner": {
        "wall_seconds": 2791.9022495672107,
        "cpu_seconds": 3179.3194538790003,
        "peak_process_rss_bytes": 5362212864
      },
      "fit_including_validation": {
        "wall_seconds": 2655.282621279359,
        "cpu_seconds": 2996.456239095,
        "peak_process_rss_bytes": 5362212864
      },
      "prediction_export": {
        "wall_seconds": 131.2333941012621,
        "cpu_seconds": 177.27610152299985,
        "peak_process_rss_bytes": 5362212864
      },
      "smoke_worker": {
        "wall_seconds": 372.1879269257188,
        "cpu_seconds": 748.0175769309999,
        "peak_rss_bytes": 21358002176,
        "gpu_peak_bytes": 38303432704,
        "samples": 340
      },
      "smoke_runner": {
        "wall_seconds": 366.3057048469782,
        "cpu_seconds": 302.939835425,
        "peak_process_rss_bytes": 4047511552
      },
      "latency_constructor": {
        "wall_seconds": 0.7362759038805962,
        "cpu_seconds": 0.8097608479999998,
        "peak_rss_bytes": 2165723136
      },
      "latency_checkpoint_load": {
        "wall_seconds": 0.11422152817249298,
        "cpu_seconds": 0.11374941200000066,
        "peak_rss_bytes": 2165723136
      },
      "latency_peak_cuda_bytes": 1675310080,
      "latency_peak_cuda_reserved_bytes": 2042626048
    }
  },
  "new_training_worker_sums": {
    "formal_worker": {
      "wall_seconds": 21869.574398823082,
      "cpu_seconds": 26040.088145695
    },
    "smoke_worker": {
      "wall_seconds": 2415.1886832863092,
      "cpu_seconds": 4321.390323361
    }
  },
  "quality_audit": {
    "wall_seconds": 0.682443156838417,
    "cpu_seconds": 0.679510248,
    "peak_rss_bytes": 32382976,
    "scope": "This stdlib collector process only."
  },
  "recorded_new_terminal_evidence": {
    "completion_path": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/launch-formal/completion.json",
    "completion": {
      "state": "completed",
      "phase": "formal",
      "launcher_exit_code": 0,
      "worker_exit_codes": {
        "sa_01": 0,
        "sa_11": 0,
        "sa_10": 0,
        "mlp_01": 0,
        "mlp_11": 0,
        "mlp_10": 0,
        "mlp_gate_entry": 0,
        "mlp_gate_work": 0
      },
      "authorization_sha256": "c716eb20655e6fcf06d8f6cd02a6cb57f5ec4fea485910512f7ca9b21285a7d9",
      "smoke_review_sha256": "4c73083f05f4da8c68a870c064a0067ca0dfee1b4fd1a07d2be3388f85e46004",
      "worker_receipt_paths": {
        "sa_01": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/sa_01/formal/exit.json",
        "sa_11": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/sa_11/formal/exit.json",
        "sa_10": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/sa_10/formal/exit.json",
        "mlp_01": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/mlp_01/formal/exit.json",
        "mlp_11": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/mlp_11/formal/exit.json",
        "mlp_10": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/mlp_10/formal/exit.json",
        "mlp_gate_entry": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/mlp_gate_entry/formal/exit.json",
        "mlp_gate_work": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/mlp_gate_work/formal/exit.json"
      },
      "worker_receipt_sha256": {
        "sa_01": "61f2107a47778b40daf51e700449caf58186f57a44dfe47c5761780c5cdd4094",
        "sa_11": "412a7b1e77a81cff801b451b93808b66bb94dde59eb117869d51d44c69b8f56a",
        "sa_10": "4bbcbd12c64106d731d122dd71d3ff880bc3ad22e873bd4a5b21c47a33616bbb",
        "mlp_01": "351a1374c68f03e9596da7b3dcc5e74455065062721cf48d7d7b1250bbbca9c7",
        "mlp_11": "12c58ad3458d4b1cdcd76e7618dbe9adb76efd4990dcece075c43aa6fcdfedb6",
        "mlp_10": "328ca75f857eb62fe3a75461e996c47900db8e41e89e534c8c9d82ee8de1f39a",
        "mlp_gate_entry": "bfd3da4c01a473c973407a6f0c56e92a47a7e0c29153e154ac67e9761bc25989",
        "mlp_gate_work": "fe96e702c57a78844672fedac5cabf39002775df2739374d1db81113436a0fdf"
      },
      "error": null,
      "receipt_errors": [],
      "deadline_exceeded": false,
      "signals": [],
      "cleanup": {
        "completed": true,
        "signals": [],
        "reaped_adoptees": [],
        "remaining": [],
        "wall_seconds": 0.01011514663696289,
        "subreaper": true,
        "pidfd_identity_checked": false,
        "signalling_methods": [],
        "fallback_scope": "ENOSYS only: fresh owned PPID chain and start-time checks before os.kill; these checks and kill are not atomic like pidfd."
      },
      "wall_seconds": 2799.2349936515093,
      "launcher_wall_limit_seconds": 7320,
      "cpu_seconds_including_reaped_children": 26045.705150776004,
      "launcher_peak_rss_bytes": 218849280,
      "worker_costs": {
        "sa_01": {
          "wall_seconds": 2735.8964571282268,
          "cpu_seconds": 3266.0868918600004,
          "peak_rss_bytes": 17174323200,
          "gpu_peak_bytes": 32846643200,
          "samples": 2508
        },
        "sa_11": {
          "wall_seconds": 2729.3299817740917,
          "cpu_seconds": 3254.001422899,
          "peak_rss_bytes": 17137303552,
          "gpu_peak_bytes": 32838254592,
          "samples": 2501
        },
        "sa_10": {
          "wall_seconds": 2742.3903021663427,
          "cpu_seconds": 3266.386917536,
          "peak_rss_bytes": 17142169600,
          "gpu_peak_bytes": 32842448896,
          "samples": 2515
        },
        "mlp_01": {
          "wall_seconds": 2693.6236927732825,
          "cpu_seconds": 3207.0729744530004,
          "peak_rss_bytes": 17061519360,
          "gpu_peak_bytes": 31582060544,
          "samples": 2470
        },
        "mlp_11": {
          "wall_seconds": 2688.214631251991,
          "cpu_seconds": 3214.7056538939996,
          "peak_rss_bytes": 17146290176,
          "gpu_peak_bytes": 31579963392,
          "samples": 2463
        },
        "mlp_10": {
          "wall_seconds": 2686.6765558496118,
          "cpu_seconds": 3200.8075648920003,
          "peak_rss_bytes": 17117454336,
          "gpu_peak_bytes": 31582060544,
          "samples": 2465
        },
        "mlp_gate_entry": {
          "wall_seconds": 2797.814953006804,
          "cpu_seconds": 3322.488007626,
          "peak_rss_bytes": 17079107584,
          "gpu_peak_bytes": 35361128448,
          "samples": 2567
        },
        "mlp_gate_work": {
          "wall_seconds": 2795.627824872732,
          "cpu_seconds": 3308.538712535,
          "peak_rss_bytes": 17096470528,
          "gpu_peak_bytes": 35356934144,
          "samples": 2565
        }
      },
      "automatic_retry": false,
      "partial_preserved": true,
      "missing_workers": []
    },
    "worker_costs": {
      "sa_01": {
        "wall_seconds": 2735.8964571282268,
        "cpu_seconds": 3266.0868918600004,
        "peak_rss_bytes": 17174323200,
        "gpu_peak_bytes": 32846643200,
        "samples": 2508
      },
      "sa_11": {
        "wall_seconds": 2729.3299817740917,
        "cpu_seconds": 3254.001422899,
        "peak_rss_bytes": 17137303552,
        "gpu_peak_bytes": 32838254592,
        "samples": 2501
      },
      "sa_10": {
        "wall_seconds": 2742.3903021663427,
        "cpu_seconds": 3266.386917536,
        "peak_rss_bytes": 17142169600,
        "gpu_peak_bytes": 32842448896,
        "samples": 2515
      },
      "mlp_01": {
        "wall_seconds": 2693.6236927732825,
        "cpu_seconds": 3207.0729744530004,
        "peak_rss_bytes": 17061519360,
        "gpu_peak_bytes": 31582060544,
        "samples": 2470
      },
      "mlp_11": {
        "wall_seconds": 2688.214631251991,
        "cpu_seconds": 3214.7056538939996,
        "peak_rss_bytes": 17146290176,
        "gpu_peak_bytes": 31579963392,
        "samples": 2463
      },
      "mlp_10": {
        "wall_seconds": 2686.6765558496118,
        "cpu_seconds": 3200.8075648920003,
        "peak_rss_bytes": 17117454336,
        "gpu_peak_bytes": 31582060544,
        "samples": 2465
      },
      "mlp_gate_entry": {
        "wall_seconds": 2797.814953006804,
        "cpu_seconds": 3322.488007626,
        "peak_rss_bytes": 17079107584,
        "gpu_peak_bytes": 35361128448,
        "samples": 2567
      },
      "mlp_gate_work": {
        "wall_seconds": 2795.627824872732,
        "cpu_seconds": 3308.538712535,
        "peak_rss_bytes": 17096470528,
        "gpu_peak_bytes": 35356934144,
        "samples": 2565
      }
    }
  },
  "paired": {
    "wall_seconds": 561.0275173485279,
    "cpu_seconds": 754.971574594,
    "peak_rss_bytes": 4598923264,
    "artifact_bytes": 2740946
  },
  "paired_scope": "Claim through all analysis artifacts; excludes final completion serialization",
  "latency": {
    "wall_seconds": 32.85884192585945,
    "cpu_seconds": 32.533738053,
    "peak_rss_bytes": 2165723136,
    "artifact_bytes": 214822,
    "peak_cuda_bytes": 1675310080
  },
  "latency_scope": "Explicit --run entry through all model artifacts, excluding read-only preflight and final completion serialization; peak RSS is whole process high-water mark.",
  "latency_input_preparation": {
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
        "wall_seconds": 0.0151248499751091,
        "cpu_seconds": 0.016503498000000505,
        "peak_rss_bytes": 979345408
      },
      "first_batch_cost": {
        "wall_seconds": 0.25664014369249344,
        "cpu_seconds": 0.4958036099999994,
        "peak_rss_bytes": 1278668800
      },
      "input_hash_cost": {
        "wall_seconds": 0.006438195705413818,
        "cpu_seconds": 0.017508230999999874,
        "peak_rss_bytes": 1278668800
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
        "hash_wall_seconds": 0.001734025776386261,
        "hash_cpu_seconds": 0.00471562400000014,
        "conversion_wall_seconds": 0.018928326666355133,
        "conversion_cpu_seconds": 0.03714878599999949,
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
        "wall_seconds": 0.01066318154335022,
        "cpu_seconds": 0.026514596999999362,
        "peak_rss_bytes": 1278668800
      },
      "device_transfer_exact": true
    }
  }
}
```

未在消费工件中记录的早期合成检查或失败准备成本不推定为零，也不将 worker 占用时间称为 GPU kernel 活跃时间。
</details>

**统计与输入身份范围**：2,871,767 行、268,201 用户、同一组 500 次以用户为单位的重采样，bootstrap seed 20260906。全部 95% 区间仅条件于这些 checkpoint，未包含训练 seed 方差、多重比较或选择校正。BCE 分母是各 draw 实际重采样行数。

Original N full-input content SHA256 values are inherited through completed C quality and E authorization. Production runners checked input stats before/after their traversals and two original metadata hashes before them. This collector now checks 122 current stats and both original metadata hashes; it does not rehash Parquet or open checkpoint/prediction arrays. New E worker/smoke/effective-optimizer gates are evaluated from actual small receipts. Three references retain exact C quality records and producer completion hashes, with small source/artifact identity checks. Stat continuity cannot independently exclude content changes that preserve stat fields.

目标时间和成熟窗口来自现有时间戳推定，click 可知性仍是假设；这不是完整线上可用性或去泄漏证明。未知行为类别不等于成熟真零。
[冻结协议](protocol.json)；本渲染器只读小 JSON/源码，不读取保存 draw 数组、预测、checkpoint 或原始数据，不重复任何 bootstrap 或模型前向。

<details><summary>报告消费身份</summary>

```json
{
  "input_sha256": {
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/render_results.py": "e9a6b6242f7408d9da91569d60db71c711b92ca86a085a03a1bb37bfcd36dbc5",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/protocol.json": "82c32ac9a05cedfdc7b72ab1e76e6846e8c381f8f1c2112ed2514c1224e8596c",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/configs.json": "46a9e858dc9019a39cc2785b93e445edf843f65f7c2a4deeb98105e6243ecd5f",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/configs.json": "3c83c83720783e7ab531e62dc9739f0e164161f42043825520260e07859b8f45",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/configs.json": "cedc3fcaebc0ce4f1a4347d9d8e69a902c2d8b3fe03fc6c9495132da52b1ac56",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/paired_analysis.py": "f379c615e8cc3e699312da9809e50ddf52069ca4eae2a9c89269591851270c98",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/single_effects.py": "7a8cb047b38064737411b9d9411a21bfe9c23d49a92210eefbad8947c4e0f834",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/benchmark_latency.py": "403f56368b2ee4b6c730f8f3bbb9e10c29e09689bc97e9e86013680a25aebdd2",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/quality-summary.json": "3b25e49579cdd65c3b2cfc448e683dca5f298c74f1ad14f64ba3ef9d3e4a952b",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/paired-analysis/completion.json": "3b713560261f0d50cafc917f23afc456051e511f1e1792562df7bd9d7e455aff",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/latency/completion.json": "a5d3e717847b0bcfc431126b4c1f8bbd693d4c87dd25c088207134b9c5bd11e1",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/paired-analysis/claim.json": "07120698ad311a2b6190d3d44afb96ca972d8f49147401364af5b175b9717656",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/paired-analysis/result.json": "aeefa5c13cafe4ad1c75ce9505c346624f0be2368f625809e42388e577a48609",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/paired-analysis/cost.json": "f4fbee13d84e36425d765e91ffb6a9e872059d00001ec4b212f96d3d41d148c6",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/latency/batch.json": "d0a125845c0c6c3444aee00b3db46819e8f56057ba117f144da49486799eee6f",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/latency/claim.json": "f2d4390ec23da6bb4be06b0b382d6fc8b96df7a88a8b7728d7e6b59ce3c6e174",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/latency/context-binding.json": "a20d247ea655ccc7797579cb3169466bb9778869e7c910d3b67266dc6c2709eb",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/latency/cross8_mature.json": "a26dddab05cc61e2adbe3379fae137875f2002dd46236b65a5f4a49b79f361d2",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/latency/hist_shallow_deep.json": "a62792c4e360407af2f0e4da332d9f39a9d52c3edd33f3645a67d596ef4fc8e7",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/latency/mature_r2.json": "23695f60e960257f8b5a9201c30f7d7c74dc17593906d89d4f3de9dfcd0f9946",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/latency/mlp_01.json": "3eaaf98c1ce6687f90db2b73a4fcfbcfa6172265a9819d09b87ae974bd5843f5",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/latency/mlp_10.json": "f4fc6bded26543354e65996cacc21a59b6c0f7768a788bb959de9a6c8288d163",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/latency/mlp_11.json": "941f71d3a6ce7c665929ce232aed150dbcce8625e61d765fd9ae0344c96211cc",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/latency/mlp_gate_entry.json": "8d673cfb3dbdcd1c3dc93ab329df55683a310ecb40c87b6d743e5d68309a5541",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/latency/mlp_gate_work.json": "79fea12fc48f88ad6788a436188b9c7105ebbe7c38cc0450ff5828523efe2ea6",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/latency/plan.json": "294a9c8112829e0b06d8d2e91b5efcd1d1a62f888aca085a24f7edb120716d1e",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/latency/sa_01.json": "74ea2645c9103af236d51ec6e9464fa36b008ca712cd5861b1bbd50632dfd792",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/latency/sa_10.json": "aaf67974abcb39de2e8dda689f154bc3c457965fe465f871e5bbee1b48829d47",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/latency/sa_11.json": "f5d9c9c5258d636c6b1ca02f313ab9473ba3dab2fc279330525773b3e9029a36",
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
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/hist_shallow_deep/result.json": "8b5385dff3136403baa61cbc8550f453f2cd12a446d122be6d1995d7f5248aa7",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/hist_shallow_deep/manifest.json": "18df9929a44abebfb9a83f5888db43c9b25f659c2dc46d4bd4102cf911332cce",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/hist_shallow_deep/formal/exit.json": "b9c4e7a9ae2d05c03910204fcccd92be90b17d194304d6b906f0cef1c69ff2c4",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/hist_shallow_deep/smoke/exit.json": "1beb1a0516083e8807aecd38def528acf6b55c033f53361157861ee3d14abcae",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/smoke-runs/hist_shallow_deep/smoke.json": "bab16ce4df0eac5a8f6628bc57eac7392452114fbd51adaf1672d3bbbe85073e",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/sa_01/result.json": "db332d367e9e986085b519d592cc1b26b97003e2f043a0952504d90959300095",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/sa_01/manifest.json": "e2c740a9395a36c098f6b4e3bc918d2ab42f27c33ac7cbbb268cf7f21fcb5472",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/sa_01/formal/exit.json": "61f2107a47778b40daf51e700449caf58186f57a44dfe47c5761780c5cdd4094",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/sa_01/smoke/exit.json": "6d4e752c2c38b9576de4148347e817f1524f6fbc121332cb5545320fe1149d18",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/smoke-runs/sa_01/smoke.json": "399f942f0130bfffefd53ee686b2853b97da2827a804adeb31e7b02358cecf0f",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/sa_11/result.json": "0b1781fe1ff9f34d32330c04c4056ac27af276a68bc61f7c305417aaadb20761",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/sa_11/manifest.json": "6a32be1ebc5f489454f2bd2365208092978805263f7220f734117eb2b9948849",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/sa_11/formal/exit.json": "412a7b1e77a81cff801b451b93808b66bb94dde59eb117869d51d44c69b8f56a",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/sa_11/smoke/exit.json": "0737fbe17d8e9a1b54e6a83e892830056ee141b3c131e56cfeb595eb3fc8b9b9",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/smoke-runs/sa_11/smoke.json": "8178682b5f1f3719b2527b2227cd98d0858df9b4b97eaeed12f73992be710e01",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/sa_10/result.json": "be7453154da0c7b3dc778ed637739891b1a5e18831a925470f00b57a2ec3768b",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/sa_10/manifest.json": "4060ba58315c4887ee8a02ec6101bee2e6f35c02535c23b7d71c53a3c64fde80",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/sa_10/formal/exit.json": "4bbcbd12c64106d731d122dd71d3ff880bc3ad22e873bd4a5b21c47a33616bbb",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/sa_10/smoke/exit.json": "b2e0bbbcc0c997d2bc2c3510f5caf6e7070dc8c0d83c06b1bf20401f185daac1",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/smoke-runs/sa_10/smoke.json": "87e070d8db5859fc553114e99228898106eeda4d5535de95b7d581ade8dc195d",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_01/result.json": "424b57640df4685750bc8ce328aacd3666cdf129446346f265b6d5c45667aedb",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_01/manifest.json": "3c2f232c399365efc54d7380bb027d9b3307173ac6115e77eaa1469d4b6652be",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/mlp_01/formal/exit.json": "351a1374c68f03e9596da7b3dcc5e74455065062721cf48d7d7b1250bbbca9c7",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/mlp_01/smoke/exit.json": "4fb1f3bcabe80f50cde472ab3720bda187d0fa5869f6bbedcf9e16715d0a025e",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/smoke-runs/mlp_01/smoke.json": "ebed51f3e2e2e153d950d522bc2bdf419db4f2622be6dfc6f7bef2bff591c8eb",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_11/result.json": "f7051c61f2a06393d1a619eb07a33c98b2901831245ed6c254aafbdfa3f0b8fc",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_11/manifest.json": "17ba6cfcb9e6357e819a6325cefd99324660fbd4ff8433c27e05b4f826e053fe",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/mlp_11/formal/exit.json": "12c58ad3458d4b1cdcd76e7618dbe9adb76efd4990dcece075c43aa6fcdfedb6",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/mlp_11/smoke/exit.json": "82f4efd156491c3bd9c0c04d469f9e835adfb9574710a4194d2a3337fce2a171",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/smoke-runs/mlp_11/smoke.json": "69d4909385c6cd9319de441bf4f5521daa7ab7145fbb1d61947c2127d4d45ba1",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_10/result.json": "864fbf969dd257c882ad1b0077dc28da62fec41ff6f1dc362a1edb659c0b91c9",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_10/manifest.json": "0f21c1acb407f2364d5537753a3deb5993f60b4e826465119e81ace6a5bfb191",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/mlp_10/formal/exit.json": "328ca75f857eb62fe3a75461e996c47900db8e41e89e534c8c9d82ee8de1f39a",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/mlp_10/smoke/exit.json": "26020ce61d7454f0758a0818568dc60373aeb7b555157ac81d3bf68fc803843a",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/smoke-runs/mlp_10/smoke.json": "556edda07fb862e3297392d985afde355d4ca996257a12f74e42404ad808e1d9",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_gate_entry/result.json": "4c6a94218554a83da08adbe0de768bafadff9b85104f41664a0521caaabede81",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_gate_entry/manifest.json": "b47f41677a605c2bcee8fc864aa9e6fc0345ced3bf70cbbafb4985f6e759f8de",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/mlp_gate_entry/formal/exit.json": "bfd3da4c01a473c973407a6f0c56e92a47a7e0c29153e154ac67e9761bc25989",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/mlp_gate_entry/smoke/exit.json": "92c35c02df7e4ed659e7c93f3f6e9137836a7fd3075cdb51bf2422004c1e1ff5",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/smoke-runs/mlp_gate_entry/smoke.json": "d603336532285d59737395346d265c3e56391a2efb7ea5243f06660debd20ab2",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_gate_work/result.json": "420968af1fe679dc2ca9c66216f5237d165985fd12e4c7f490bd0d9960dff356",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/runs/mlp_gate_work/manifest.json": "5f093ceb3227bd5c3e908fb67359dd94243d8bb5c9cf1e702a622ca4911e5e5d",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/mlp_gate_work/formal/exit.json": "fe96e702c57a78844672fedac5cabf39002775df2739374d1db81113436a0fdf",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/mlp_gate_work/smoke/exit.json": "85b6dd3d45037d62cd3d5ccb3ed195f425c5d5d6c8980fa09f5b55dc27c3e8c9",
    "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/smoke-runs/mlp_gate_work/smoke.json": "c9983cf21ce5965278ff89e2cda71985dfc05b4ace12795e9f88778b8b427e8f"
  },
  "csv_sha256": "adc36458f49075618f4160e2af4951157e1006e14d59847dafed79fdc3663d25",
  "recorded_shared_draw_sha256": {
    "model-draws.npz": "eb75d2ca342246eafecec952b1231232d4de8411816b7bbcd79baea550dec5d4",
    "effect-draws.npz": "78f0c4dab185f5a79b38e76d08697bd7129cee6cec42792a27fa2fb1819a5012"
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
