第二轮查询的联合有界融合

q4/q8 分开比较 channel/query 两种门粒度；每种粒度都有 joint 与固定 RMS1 输入的 constant 配对。仅改变第二轮 q2 对入口 e 和首轮工作状态 z1 的融合，保留两轮及无预测状态反馈；门不读取标签或验证统计。

| 候选 | Macro AUC | Δ 同q原循环 | Δ 最佳单轮 | 参数量 |
|---|---:|---:|---:|---:|
| mix_q4_channel_joint | 0.828229 | -0.000244 | -0.000395 | 51,748,804 |
| mix_q4_channel_constant | 0.827682 | -0.000791 | -0.000942 | 51,748,804 |
| mix_q4_query_joint | 0.828289 | -0.000184 | -0.000335 | 51,742,676 |
| mix_q4_query_constant | 0.825584 | -0.002888 | -0.003039 | 51,742,676 |
| mix_q8_channel_joint | 0.823525 | -0.004996 | -0.005098 | 81,640,900 |
| mix_q8_channel_constant | 0.826065 | -0.002456 | -0.002558 | 81,640,900 |
| mix_q8_query_joint | 0.825527 | -0.002995 | -0.003097 | 81,634,772 |
| mix_q8_query_constant | 0.827101 | -0.001421 | -0.001523 | 81,634,772 |

| Joint候选 | Constant对照 | Δ AUC joint−constant | Δ logloss | 参数比 |
|---|---|---:|---:|---:|
| mix_q4_channel_joint | mix_q4_channel_constant | +0.000547 | -0.00007611 | 1.000000 |
| mix_q4_query_joint | mix_q4_query_constant | +0.002705 | -0.00020568 | 1.000000 |
| mix_q8_channel_joint | mix_q8_channel_constant | -0.002540 | +0.00021631 | 1.000000 |
| mix_q8_query_joint | mix_q8_query_constant | -0.001574 | +0.00012895 | 1.000000 |

| 候选 | click AUC | cart AUC | fav AUC | buy AUC |
|---|---:|---:|---:|---:|
| mix_q4_channel_joint | 0.667022 | 0.866179 | 0.882541 | 0.897173 |
| mix_q4_channel_constant | 0.666082 | 0.866487 | 0.882211 | 0.895947 |
| mix_q4_query_joint | 0.667176 | 0.865484 | 0.883711 | 0.896785 |
| mix_q4_query_constant | 0.663149 | 0.865145 | 0.879935 | 0.894107 |
| mix_q8_channel_joint | 0.662203 | 0.860814 | 0.878204 | 0.892879 |
| mix_q8_channel_constant | 0.665380 | 0.864445 | 0.879822 | 0.894615 |
| mix_q8_query_joint | 0.663682 | 0.864569 | 0.878697 | 0.895158 |
| mix_q8_query_constant | 0.664734 | 0.865938 | 0.880649 | 0.897081 |

| 候选 | click logloss | cart logloss | fav logloss | buy logloss |
|---|---:|---:|---:|---:|
| mix_q4_channel_joint | 0.18197870 | 0.01778588 | 0.01334024 | 0.00676330 |
| mix_q4_channel_constant | 0.18217702 | 0.01781480 | 0.01337977 | 0.00680095 |
| mix_q4_query_joint | 0.18189740 | 0.01777383 | 0.01335925 | 0.00673009 |
| mix_q4_query_constant | 0.18264162 | 0.01776825 | 0.01339998 | 0.00677342 |
| mix_q8_channel_joint | 0.18272092 | 0.01791688 | 0.01345852 | 0.00684753 |
| mix_q8_channel_constant | 0.18247916 | 0.01767180 | 0.01326116 | 0.00666650 |
| mix_q8_query_joint | 0.18268966 | 0.01775399 | 0.01333150 | 0.00666764 |
| mix_q8_query_constant | 0.18242503 | 0.01763645 | 0.01328546 | 0.00658008 |

| 候选 | 参考 | Macro AUC Δ | 用户配对95% CI | 必做 |
|---|---|---:|---|---|
| mix_q4_channel_joint | mix_q4_channel_constant | +0.000547 | [-0.000346, +0.001415] | 是 |
| mix_q4_query_joint | mix_q4_query_constant | +0.002705 | [+0.001888, +0.003486] | 是 |
| mix_q8_channel_joint | mix_q8_channel_constant | -0.002540 | [-0.003397, -0.001713] | 是 |
| mix_q8_query_joint | mix_q8_query_constant | -0.001574 | [-0.002349, -0.000775] | 是 |
| mix_q4_channel_joint | ctx_q4 | -0.000244 | [-0.001100, +0.000557] | 是 |
| mix_q4_channel_constant | ctx_q4 | -0.000791 | [-0.001578, +0.000068] | 是 |
| mix_q4_query_joint | ctx_q4 | -0.000184 | [-0.000921, +0.000533] | 是 |
| mix_q4_query_constant | ctx_q4 | -0.002888 | [-0.003725, -0.002057] | 是 |
| mix_q8_channel_joint | explore_dual_no_feedback | -0.004996 | [-0.005948, -0.004114] | 是 |
| mix_q8_channel_constant | explore_dual_no_feedback | -0.002456 | [-0.003294, -0.001613] | 是 |
| mix_q8_query_joint | explore_dual_no_feedback | -0.002995 | [-0.003825, -0.002108] | 是 |
| mix_q8_query_constant | explore_dual_no_feedback | -0.001421 | [-0.002197, -0.000639] | 是 |

| 候选 | mean ms | p50 ms | p95 ms | 相对同q原循环 | 相对Constant |
|---|---:|---:|---:|---:|---:|
| mix_q4_channel_joint | 14.4950 | 14.4063 | 14.6469 | 1.0075 | 1.0100 |
| mix_q4_channel_constant | 14.3518 | 14.3515 | 14.4010 | 0.9975 | 1.0000 |
| mix_q4_query_joint | 14.4597 | 14.4081 | 14.4738 | 1.0050 | 1.0092 |
| mix_q4_query_constant | 14.3282 | 14.3351 | 14.3544 | 0.9959 | 1.0000 |
| mix_q8_channel_joint | 16.1070 | 16.1097 | 16.1613 | 1.0205 | 1.0101 |
| mix_q8_channel_constant | 15.9460 | 15.9169 | 15.9645 | 1.0103 | 1.0000 |
| mix_q8_query_joint | 16.0505 | 16.0432 | 16.0837 | 1.0169 | 1.0064 |
| mix_q8_query_constant | 15.9486 | 15.9085 | 15.9452 | 1.0105 | 1.0000 |

耗时为同一 GPU、同一真实验证 batch1024、BF16 的重复 forward，排除数据读取与搬运；11组记录包含实际GPU编号、完整batch哈希及checkpoint哈希。

Channel新增24,576参数，query新增18,448参数；同粒度 joint/constant 参数相同。Constant省去三次LayerNorm，所以不声称完全相同FLOPs；固定输入为交替±1、RMS1，样本及槽间共享。

必做12个配对用户bootstrap：4个joint−constant及8个候选−同q原循环，无论点估计正负。最佳单轮的区间仅对正向候选补充。固定500次重采样、bootstrap seed20260906，四任务共享用户权重；这不是新增训练seed。区间条件于固定checkpoint和本次验证集，不包含训练seed波动，未校正候选选择，跨零不构成持平证明。

Joint相对原循环的变化同时包含容量和融合机制。只有同时优于同参数constant和原固定均值，且任务/耗时取舍可用，才支持条件化结构候选；门随输入变化或非零梯度本身不证明有效个性化。channel/query参数量不同，q4/q8分开判读。

兼容处理仅允许第二步有限且与 A[16,1152] 梯度 exact 匹配的初始化恢复，返回原函数已算出的自动scale原值；没有扩大跳过规则。各phase触发记录见final-audit.json。完整各任务指标、显存及训练时间见 [summary](reports/summary.json)，质量输入固定保存在 [quality-summary](reports/quality-summary.json)。未评估test。
