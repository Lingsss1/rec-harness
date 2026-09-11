## rec-harness 诊断

3 个候选导出完成，27 个工具调用成功；JSON 与 Markdown 均已读取。每个候选使用固定 57,408 行验证子集，表示/梯度为其中 7,232 行。checkpoint 未变化，未使用测试集。

| 模型 | 子集原始 AUC | 关闭额外 A | 推理一次循环 | 推理三次循环 | 最终 Q0 有效秩 |
|---|---:|---:|---:|---:|---:|
| r16_base_mixed1232 | 0.860182 | 0.859258 | 0.858538 | 0.859782 | 5.00 |
| r16_field512_curriculum13 | 0.857607 | 0.851826 | 0.855732 | 0.860969 | 5.37 |
| r16_base_curriculum13 | 0.856458 | 0.856021 | 0.852551 | 0.860973 | 3.55 |

- `r16_base_mixed1232` 关闭 A 的分块平均 Logloss 增量 +0.000722，95% block CI [+0.000448, +0.001007]；八个 Query 中 8 个 Shuffle 后子集 AUC 下降。最终 Q0 的工具标记为 medium，有效秩为 5.00/256。
- `r16_field512_curriculum13` 关闭 A 的分块平均 Logloss 增量 +0.006981，95% block CI [+0.006272, +0.007695]；八个 Query 中 8 个 Shuffle 后子集 AUC 下降。最终 Q0 的工具标记为 medium，有效秩为 5.37/256。
- `r16_base_curriculum13` 关闭 A 的分块平均 Logloss 增量 +0.000689，95% block CI [-0.000286, +0.001737]；八个 Query 中 8 个 Shuffle 后子集 AUC 下降。最终 Q0 的工具标记为 medium，有效秩为 3.55/256。

推理深度扰动只反映固定权重对训练结构的依赖；与本轮从头训练的结构对照分开解释。有效秩和局部梯度是机制线索，不能直接证明坍缩或保证重训收益。区间条件于固定 checkpoint，沿用既有工具 bootstrap 设置，不涉及新增训练 seed。
[完整诊断摘要](reports/round16/diagnostic-evidence.json) · [工具状态](reports/round16/toolkit-candidates-status.json)
