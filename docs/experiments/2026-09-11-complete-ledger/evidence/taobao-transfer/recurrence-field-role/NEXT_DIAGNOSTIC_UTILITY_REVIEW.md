# 首轮更新样本对应诊断：暂不执行

2026-09-09，F 已完成训练、配对、组合量、延迟、结果审查和 memory 入账。根代理结合 query_design_audit 的只读独立审查，决定暂不执行 [此前提出的 w_q-only donor 诊断](NEXT_QUERY_CONDITION_AUDIT.md)。这是一项研究优先级决定，没有新的模型测量或收益结论。

拟议干预只替换原父模型首轮更新 `r=z1-e` 在第二次 `w_q` 输入中的样本对应关系，保留交互及残差路径。它与此前替换整个 `z1` 的 query-parts、或新增 adapter correction 的分路诊断不完全相同；但当前缺少一个能被其结果直接选择或淘汰的新结构。

- 若打乱后变差，只能补充指定 checkpoint 对匹配更新的条件依赖证据，也包含 donor 更新与当前历史不匹配的扰动。它不能选出优于已有失败门控、修正、投影或方向方案的 Q2 公式。
- 若结果跨零或改善，也不能推出删除首轮更新、替换为固定更新或重新训练会有收益。
- 只有未来明确提出“第二检索使用与样本无关的更新”这一简化候选时，才重新评估所需对照。届时应直接对应候选；donor 打乱本身不是固定更新的直接对照。

用户的任务差异问题已有更直接的 [任务证据](../recurrence-research/TASKWISE_RECURRENCE_EVIDENCE.md)：q4 分别训练的两轮相对一轮，加购和购买的 AUC 增益高于点击，直接任务效果差区间为正；q8 同权重删除整轮则四任务都受损。新诊断即使仅 buy 排除零，也不能证明只有 buy 从循环获益，或取代直接任务比较。

根代理接受“暂不执行”的建议。队列记录为低决策价值的条件性搁置，不视为技术阻塞或科学反证；不新增训练、seed、forward、统计抽样、工具候选或 handoff，也不触发 optimizer shadow。后续仍需一个与已完成实验不同、结果能改变结构选择的明确假设。研究目标保持 active；本轮关闭不等于已得到更好的 Q2。

F 入账记录：[memory-consolidation.json](reports/memory-consolidation.json)，SHA256 `61c46f81b4952c8a817fa1c98ec3513a3caf0d5fe6565caccb6f5f72a7f8d1af`。入账后 141 次完成训练、211 条 episode、211 条 TSV、26 条 lesson；新增八个训练结果均 discard，最佳单模型及原 q4/q8 候选不变。
