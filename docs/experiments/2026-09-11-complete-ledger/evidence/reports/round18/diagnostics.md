## rec-harness 诊断

2 个候选导出完成，18 个工具调用成功；JSON 与 Markdown 均已读取。每个候选使用固定 57,408 行验证子集，表示/梯度为其中 7,232 行。checkpoint 未变化，未使用测试集。

| 模型 | 子集原始 AUC | 关闭额外 A | 推理一次循环 | 推理三次循环 | 最终 Q0 有效秩 |
|---|---:|---:|---:|---:|---:|
| r18_base_norm_phase132 | 0.862352 | 0.856072 | 0.851137 | 0.852311 | 6.91 |
| r18_field512_norm_phase132 | 0.859739 | 0.855442 | 0.856494 | 0.850617 | 7.48 |

- `r18_base_norm_phase132` 关闭 A 的分块平均 Logloss 增量 +0.004774，95% block CI [+0.004011, +0.005506]；八个 Query 中 8 个 Shuffle 后子集 AUC 下降。最终 Q0 的工具标记为 medium，有效秩为 6.91/256。
- `r18_field512_norm_phase132` 关闭 A 的分块平均 Logloss 增量 +0.006089，95% block CI [+0.005217, +0.006944]；八个 Query 中 8 个 Shuffle 后子集 AUC 下降。最终 Q0 的工具标记为 medium，有效秩为 7.48/256。

本轮推理深度扰动同时改变循环次数和选中的已训练norm，反映二者联合依赖，不能归因于循环次数单独变化；与本轮从头训练的结构对照分开解释。有效秩和局部梯度是机制线索，不能直接证明坍缩或保证重训收益。区间条件于固定 checkpoint，沿用既有工具 bootstrap 设置，不涉及新增训练 seed。
[完整诊断摘要](reports/round18/diagnostic-evidence.json) · [工具状态](reports/round18/toolkit-candidates-status.json)
