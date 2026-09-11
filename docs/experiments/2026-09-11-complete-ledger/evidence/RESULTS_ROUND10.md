# KuaiRec 第十轮：循环 block 的选择性子层

八组均在已有 seed=20262027 从头完成一轮训练，没有新增 seed、复测或测试集评估。主基线 Cross8，优化参考 Cross11 全 Query。

保留父模型 r7_small_attention_first 的 N(全部字段) → S(历史) → A → N(上下文) → S(历史) 顺序。首次 N/S 完整执行，仅第二次及以后的 N/S 选择子层。额外 A 仍只在第一次 S 后执行。

## 冻结结构

| 模型 | 第二次 N | 第二次 S |
|---|---|---|
| r10_field_cross | CrossValue | self-attention + CrossValue + FFN |
| r10_field_cross_ffn | CrossValue + FFN | self-attention + CrossValue + FFN |
| r10_field_self_cross | self-attention + CrossValue | self-attention + CrossValue + FFN |
| r10_history_cross | self-attention + CrossValue + FFN | CrossValue |
| r10_history_cross_ffn | self-attention + CrossValue + FFN | CrossValue + FFN |
| r10_history_self_cross | self-attention + CrossValue + FFN | self-attention + CrossValue |
| r10_both_cross_ffn | CrossValue + FFN | CrossValue + FFN |
| r10_both_cross | CrossValue | CrossValue |

跳过 self-attention 或 FFN 时同时跳过该子层的 residual LayerNorm，因此是子层结构对照，不是 norm 次数匹配对照。字段最终 norm、序列末尾 norm 和全部历史源保持原样。
所有新模型参数均为 8,115,473，参数键、对象与初始值沿用父模型。被第二次跳过的模块仍在第一次完整读取时接受训练；变化主要在重复执行的计算量。

## 同 seed 完整验证

| 模型 | AUC ↑ | Δ父模型 | ΔCross8 | ΔCross11 | Logloss ↓ | UAUC ↑ | NDCG@10 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| cross8_original | 0.869547 | -0.001687 | +0.000000 | -0.000426 | 0.189132 | 0.788347 | 0.404062 |
| baseline_all | 0.869973 | -0.001261 | +0.000426 | +0.000000 | 0.189433 | 0.786362 | 0.402129 |
| r7_small_attention_first | 0.871234 | +0.000000 | +0.001687 | +0.001261 | 0.188067 | 0.790837 | 0.409856 |
| r8_first_loop1 | 0.870096 | -0.001138 | +0.000549 | +0.000122 | 0.188919 | 0.792501 | 0.412650 |
| r9_feedback_omit | 0.869712 | -0.001522 | +0.000165 | -0.000262 | 0.188932 | 0.788871 | 0.405604 |
| r10_history_cross_ffn | 0.869990 | -0.001244 | +0.000443 | +0.000017 | 0.188682 | 0.789353 | 0.407261 |
| r10_history_self_cross | 0.869942 | -0.001292 | +0.000395 | -0.000032 | 0.189067 | 0.789530 | 0.409631 |
| r10_field_self_cross | 0.869917 | -0.001318 | +0.000369 | -0.000057 | 0.188756 | 0.791310 | 0.408904 |
| r10_field_cross_ffn | 0.869621 | -0.001613 | +0.000074 | -0.000352 | 0.189162 | 0.788684 | 0.404826 |
| r10_both_cross | 0.869313 | -0.001921 | -0.000234 | -0.000661 | 0.189217 | 0.787471 | 0.404955 |
| r10_history_cross | 0.869270 | -0.001964 | -0.000277 | -0.000703 | 0.189320 | 0.787078 | 0.405668 |
| r10_field_cross | 0.868952 | -0.002282 | -0.000595 | -0.001022 | 0.189350 | 0.790092 | 0.408171 |
| r10_both_cross_ffn | 0.868908 | -0.002326 | -0.000639 | -0.001066 | 0.189370 | 0.789571 | 0.406448 |

同一 kuairec-loop-v1 输入哈希、1 epoch/1240 更新、batch8192、AdamW1e-4/Adagrad.05、BF16、clip1、全 Query。原始 Cross8 保留 clip10。

## 同卡前向与训练成本

| 模型 | 参数/M | p50/ms | p95/ms | 训练含验证/s | 峰值显存/GiB |
|---|---:|---:|---:|---:|---:|
| cross8_original | 9.232 | 13.70 | 13.92 | 448.7 | 16.67 |
| baseline_all | 11.408 | 15.13 | 15.48 | 495.1 | 18.88 |
| r7_small_attention_first | 8.115 | 12.70 | 12.80 | 441.6 | 15.74 |
| r8_first_loop1 | 8.115 | 8.66 | 8.73 | 337.0 | 9.81 |
| r9_feedback_omit | 8.115 | 11.20 | 11.31 | 396.1 | 13.70 |
| r10_history_cross_ffn | 8.115 | 12.21 | 12.25 | 422.8 | 15.08 |
| r10_history_self_cross | 8.115 | 12.54 | 12.56 | 437.1 | 15.61 |
| r10_field_self_cross | 8.115 | 12.55 | 12.59 | 433.1 | 15.27 |
| r10_field_cross_ffn | 8.115 | 12.26 | 12.35 | 425.8 | 15.17 |
| r10_both_cross | 8.115 | 11.47 | 11.58 | 400.0 | 14.29 |
| r10_history_cross | 8.115 | 12.07 | 12.12 | 425.6 | 15.08 |
| r10_field_cross | 8.115 | 12.07 | 12.17 | 421.4 | 14.83 |
| r10_both_cross_ffn | 8.115 | 11.77 | 11.94 | 407.0 | 14.63 |

同一 A800、BF16、batch1024，10 次预热、40 次固定真实批量前向；排除 IO，不代表服务端到端延迟。训练墙钟受同时运行的任务影响。

## 正确性与复现

CPU/GPU 各 5 项测试通过：默认初始化/输出/梯度精确兼容，实际子层和 norm 调用顺序，源张量/遮罩不变，全部参数的梯度与优化器覆盖，首次调用不受选择影响，空历史、三次诊断循环和真实最终输出干预。
[计划](reports/round10/PLAN.md) · [配置](reports/round10/configs.json) · [模型](model_zoo/QFormerCross11Selective.py) · [结果](reports/round10/summary.json) · [训练状态](reports/round10/training/status.json)

## 本轮判断

本轮八组均未刷新最高 AUC，继续保留 `r7_small_attention_first`（0.871234）。累计完整训练 111 次，本轮无训练失败，无新增 seed，测试集未评估。

新结构最高 AUC 为 `r10_history_cross_ffn` 的 0.869990，距父模型 -0.001244。它保留第二次历史 CrossValue 和 FFN，只跳过内部 self-attention 及其 norm；同卡 p50 从 12.70 降至 12.21 ms，减少 3.9%。AUC 接近 Cross11 全 Query（0.869973），且参数少 28.9%、前向耗时少 19.3%，按候选策略继续保存，但不是当前质量首选。

`r10_field_self_cross` 在第二次字段块中去掉 FFN 及其 norm，AUC=0.869917，UAUC=0.791310，比父模型 0.790837 略高；Logloss/NDCG 则较差。它保留为用户指标取舍候选。

最快的 `r10_both_cross` 第二次字段/历史都仅保留 CrossValue，p50=11.47 ms，较完整父模型减少 9.7%，AUC=0.869313。已有一次循环模型 AUC=0.870096、p50=8.66 ms，并有更高的 UAUC/NDCG，因此低成本首选仍是已有一次循环。

这些实验支持保留当前完整 block 作为搜索起点，但删除子层也同时删除了其 norm，不能据此证明每个 attention/FFN 运算本身都不可替代。近三轮 24 个新结构均未超过父模型 AUC，下一方向改为循环轮次条件化：保留完整共享计算，通过零初始化的轮次向量或输入 Query 决定的更新门控，让第二次读取适应不同职责。该方向尚未运行，初始函数兼容和门控梯度将先验证，再进行完整训练。


## rec-harness 诊断

2 个候选导出完成，18 个工具调用成功；JSON 与 Markdown 均已读取。每个候选使用固定 57,408 行验证子集，表示/梯度为其中 7,232 行。checkpoint 未变化，未使用测试集。

| 模型 | 子集原始 AUC | 关闭额外 A | 推理一次循环 | 推理三次循环 | 最终 Q0 有效秩 |
|---|---:|---:|---:|---:|---:|
| r10_history_cross_ffn | 0.860856 | 0.857851 | 0.832164 | 0.845673 | 11.27 |
| r10_field_self_cross | 0.860715 | 0.859648 | 0.801696 | 0.844351 | 11.43 |

- `r10_history_cross_ffn` 关闭 A 的分块平均 Logloss 增量 +0.002274，95% block CI [+0.001777, +0.002759]；八个 Query 中 8 个 Shuffle 后子集 AUC 下降。最终 Q0 的工具标记为 medium，有效秩为 11.27/256。
- `r10_field_self_cross` 关闭 A 的分块平均 Logloss 增量 +0.001086，95% block CI [+0.000611, +0.001624]；八个 Query 中 8 个 Shuffle 后子集 AUC 下降。最终 Q0 的工具标记为 medium，有效秩为 11.43/256。

推理深度扰动只反映固定权重对训练结构的依赖；与本轮从头训练的结构对照分开解释。有效秩和局部梯度是机制线索，不能直接证明坍缩或保证重训收益。区间条件于固定 checkpoint，沿用既有工具 bootstrap 设置，不涉及新增训练 seed。
[完整诊断摘要](reports/round10/diagnostic-evidence.json) · [工具状态](reports/round10/toolkit-candidates-status.json)


保留接近 Cross8/Cross11 或存在成本/排序取舍的候选，同时保存较差对照。固定 seed 的筛选结果不等于已证明稳定泛化收益。
