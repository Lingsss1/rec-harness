## rec-harness 诊断

2 个候选导出完成，17 个工具调用成功；JSON 与 Markdown 均已读取。每个候选使用固定 57,408 行验证子集，表示/梯度为其中 7,232 行。checkpoint 未变化，未使用测试集。

| 模型 | 子集原始 AUC | 关闭额外 A | 推理一次循环 | 推理三次循环 | 最终 Q0 有效秩 |
|---|---:|---:|---:|---:|---:|
| r8_first_loop1 | 0.862323 | 0.856407 | 0.862323 | 0.756627 | 9.77 |
| r8_first_midnorm | 0.861455 | 0.858364 | 0.811636 | 0.806962 | 11.01 |

- `r8_first_loop1` 关闭 A 的分块平均 Logloss 增量 +0.003699，95% block CI [+0.002741, +0.004619]；八个 Query 中 8 个 Shuffle 后子集 AUC 下降。最终 Q0 的工具标记为 medium，有效秩为 9.77/256。
- `r8_first_midnorm` 关闭 A 的分块平均 Logloss 增量 +0.002302，95% block CI [+0.001756, +0.002891]；八个 Query 中 8 个 Shuffle 后子集 AUC 下降。最终 Q0 的工具标记为 medium，有效秩为 11.01/256。

推理深度扰动只反映固定权重对训练结构的依赖；与上面的从头训练深度对照分开解释。有效秩和局部梯度是机制线索，不能直接证明坍缩或保证重训收益。区间条件于固定 checkpoint，沿用既有工具 bootstrap 设置，不涉及新增训练 seed。
[完整诊断摘要](reports/round8/diagnostic-evidence.json) · [工具状态](reports/round8/toolkit-candidates-status.json)
