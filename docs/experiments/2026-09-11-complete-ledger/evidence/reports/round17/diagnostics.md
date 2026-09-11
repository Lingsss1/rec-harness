## rec-harness 诊断

2 个候选导出完成，18 个工具调用成功；JSON 与 Markdown 均已读取。每个候选使用固定 57,408 行验证子集，表示/梯度为其中 7,232 行。checkpoint 未变化，未使用测试集。

| 模型 | 子集原始 AUC | 关闭额外 A | 推理一次循环 | 推理三次循环 | 最终 Q0 有效秩 |
|---|---:|---:|---:|---:|---:|
| r17_base_phase132 | 0.860228 | 0.855910 | 0.853104 | 0.851440 | 5.62 |
| r17_field512_phase132 | 0.861007 | 0.856985 | 0.854452 | 0.852919 | 7.04 |

- `r17_base_phase132` 关闭 A 的分块平均 Logloss 增量 +0.004434，95% block CI [+0.003571, +0.005250]；八个 Query 中 8 个 Shuffle 后子集 AUC 下降。最终 Q0 的工具标记为 medium，有效秩为 5.62/256。
- `r17_field512_phase132` 关闭 A 的分块平均 Logloss 增量 +0.007778，95% block CI [+0.006732, +0.008994]；八个 Query 中 8 个 Shuffle 后子集 AUC 下降。最终 Q0 的工具标记为 medium，有效秩为 7.04/256。

推理深度扰动只反映固定权重对训练结构的依赖；与本轮从头训练的结构对照分开解释。有效秩和局部梯度是机制线索，不能直接证明坍缩或保证重训收益。区间条件于固定 checkpoint，沿用既有工具 bootstrap 设置，不涉及新增训练 seed。
[完整诊断摘要](reports/round17/diagnostic-evidence.json) · [工具状态](reports/round17/toolkit-candidates-status.json)
