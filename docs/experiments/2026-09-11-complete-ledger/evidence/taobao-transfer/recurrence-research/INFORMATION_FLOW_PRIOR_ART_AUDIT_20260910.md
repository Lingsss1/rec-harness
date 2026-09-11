# 循环信息流的本地查重（2026-09-10）

结论：**更新历史 token 后再读历史**、**Attention/FFN 全程的多残差流 HCR**，本次定点检查未见已完成的对应训练；**预测查询每轮重置**、**把出口放在中间循环外**、**共享头深度监督**则已有相近实跑，不能再次统称“未试”。最值得区分的是循环状态集合，而非继续把查询混合改一个名字。

范围：只读当前 153 次完成训练的台账以及下列 20 个具体小源码/结果/摘要；未读取数据、checkpoint、预测数组或编译缓存，未执行模型、测试、统计或训练。关于 LoopCTR 的比较，以根代理提供的 v1 原文核对为前提：[原文](https://arxiv.org/abs/2604.19550v1)。本笔记未另做网页核查。这里“未见”是该限定审查的结果，不是整个仓库所有历史版本的穷尽证明，也不预报新设计收益。

## 原模型实际上循环什么

[q4 ContextRead](../recurrence-context-read/QFormerCross8ContextRead.py:12)、[q8 Explore](../recurrence-explore/QFormerCross8RecurrenceExplore.py:33) 和 [NoFeedback](../recurrence-no-feedback/QFormerCross8NoFeedback.py:15) 的主路径为：

```
e = L0(Q, H)
z1 = L1(e, H)              y1 = L2(e, z1)
z2 = L1((e+z1)/2, H)       y2 = L2(y1, z2)
output = original_post(y2) -> original four task towers
```

H 在模型入口由字段投影、位置项和 LN 构造一次；循环没有覆盖 H。z、y 都是 S 个查询槽的潜在表示，并非历史序列 token，也不是任务概率。实际四任务头位于循环之后；不能把 y1→y2 描述为“上一轮预测概率回流”。[历史构造](../recurrence-work-relay/source-snapshot/QFormerCross8.py:1083)、[任务头位置](../recurrence-context-read/QFormerCross8ContextRead.py:185)。

每次 L1 调用**已经重新执行 Wk(H)、Wfi(H)**，但共享同一层、同一输入 H；缺的是历史状态更新，不是遗漏投影计算。L0 与 L1 是不同层，也不能声称所有层共用相同 K/V。[原 attention](../recurrence-work-relay/source-snapshot/QFormerCross8.py:178)。

## 三项查重

| 所问机制 | 本地实跑与具体证据 | 能判定什么 |
|---|---|---|
| 历史自身更新，下一轮使用更新后的 K/V | NoFeedback / ContextRead / Explore / MiddleLoop 都仅更新查询，H 不变。MemorySources 的八组只是把 e、z1 或 H 拼接到第二轮工作/预测读取来源；[代码](../recurrence-memory-sources/QFormerCross8MemorySources.py:16)、[结果](../recurrence-memory-sources/RESULTS.md:7)，台账 102–109。 | **未见对应的循环 H 更新训练。** 新的历史自注意力状态更新不等于已失败的 memory 拼接、field-role 切换或 query relay。 |
| 预测状态每轮从 entry 重置，不保留 y1→y2 | `route_prediction_reset` 明确 `query=e`，每轮 `L2(e,z)`；[代码 21–24 行](../recurrence-state-routes/QFormerCross8StateRoutes.py:21)。台账 79，2193 更新、seed 20262027、无 test；[真实结果](../recurrence-state-routes/runs/route_prediction_reset-s20262027-20260908T120047717081Z/result.json)。macro 0.8273069383473249，比原 q8 nofeedback 0.8285212133742255 低 0.0012142750269006。 | **重置本身已试且点值不佳。** 这组仅末轮损失；无 prediction→work 反馈时，首轮 L2 输出被后续重置丢弃，不参与最终输出。因此不等于带每层出口损失的完整实现。 |
| 出口在循环外、共享头深度监督 | [MiddleLoop 15–25 行](../recurrence-middle/QFormerCross8MiddleLoop.py:15) 只循环 L1，最后一次 L2 仍读取 H。`middle_supervised12` 用同一个模型/头分别完整前向 R2 与 R1，[loss](../recurrence-training/QFormerCross8DepthTraining.py:30) 为 `(loss_R2+.25*loss_R1)/1.25`。台账 46；[真实结果](../recurrence-training/runs/middle_supervised12-s20262027-20260908T035104357175Z/result.json) macro 0.8254845285900987，较原 middle_r2 0.8275028824343343 低 0.0020183538442356。另 mixed123 为 0.8264726829772377。 | **这些组成部分已有实跑。** 不含 depth0、不等权平均、不基于当前 y/z 的每层 entry-reset 出口，且辅助前向重算前缀、计算不匹配。不能声称已否定 LoopCTR 的联合信息流，也不能忽略这项负证据。 |
| Attention 和 FFN 的真正多残差流 HCR | [QueryMixing 50–78 行](../recurrence-query-mixing/QFormerCross8QueryMixing.py:50) 仅在两轮之间门控混合 e/z1。其修正送入整个 L1，**同时影响 w_q、w_qi 和外残差**，并不是仅 w_q；[原残差定义](../recurrence-work-relay/source-snapshot/QFormerCross8.py:236)。[DualState](../recurrence-dual-state/QFormerCross8DualState.py:18) 的两种潜在角色状态，也不是历史/全局双 token 流。 | **未见完整 HCR 对应训练。** QueryMixing 失败不能当成 HCR 消融；反过来 HCR 文献收益也不能替这些失败的门控背书。原模型只有末尾统一 post FFN，把 HCR 同时移植到 Attention/FFN 会涉及更广的架构改变。 |

[DepthReadout](../recurrence-readout-retrieval/QFormerCross8DepthReadout.py:38) 是 `h2+a_t(h1-h2)` 后取原任务头，仍只计算最终任务损失；它不是逐层深度监督。共享工作/预测两个角色状态、四/八查询槽、任务特定读出，也都不能仅凭名称当成多残差流实现。

上面两项精确失败数值来自保存的 result.json；其旧报告对照口径不同：reset 的旧表默认对最佳单轮，supervised12 的旧表对 middle_r2。本笔记显式区分，未新增 CI 或显著性判断。[reset 结果说明](../recurrence-state-routes/RESULTS.md:15)、[辅助监督说明](../recurrence-training/RESULTS.md:10)。

## 最多两条后续信息价值判断（不是执行授权）

1. **优先检查“历史表示是否必须成为循环状态”。** 限定 H 更新只看历史、全局/查询可读历史，避免同时恢复 prediction→work 反馈。必要对照使用相同历史编码器、相同调用次数：一侧把全部历史编码提前到第一次读取之前，再固定读取；另一侧把指定更新放到两次读取之间，使第二轮读取更新后的历史。这样比较的是更新与读取的时序，避免一般序列编码增益被解释为循环增益。该匹配不意味着优化器轨迹或实际梯度相同。长度 L 的历史更新可能引入 L² attention，而现有工作读取主要为 S×L；参数复用也不会使额外计算、激活和梯度代价消失。该机制未被现有查询侧负结果直接覆盖。
2. **有条件考虑“无持续 y 状态的共享出口 + 同权深度监督”，优先级低于第 1 项。** 精确待验的是联合设计及其梯度路径，不是再次运行 reset 或换一个辅助权重。至少先明确同一出口定义下末轮损失与逐层损失的匹配对照；要把收益归因于取消 y 累积，还需相应信息流对照。已有 reset 和 .25 辅助监督两项负证据使无对照的整体拼装信息价值很低。本笔记不建议先上完整 HCR：它尚未被直接否定，但会同时改动多个残差位置，现有证据不足以优先选择它。

所有新想法须继续区分原发布输入协议与行为可用性问题，不能用历史时间在前替代标签窗口已成熟。B 的匹配深度控制仍是待定源准备；文献启发不自动授权新训练，也不改变已完成 153 次训练的结果。以上亦不能回答“只有 buy 才有循环信息增量”：任务差异需要同一对照里的直接任务效应差，不能由不同干预的显著/不显著拼接推出。

## 本次 20 个定点输入

1. `memory/strategy.json`（当前完成数、best、exhausted/queue）
2. `recurrence-memory-sources/QFormerCross8MemorySources.py`
3. `recurrence-explore/QFormerCross8RecurrenceExplore.py`
4. `recurrence-training/QFormerCross8DepthTraining.py`
5. `recurrence-dual-state/QFormerCross8DualState.py`
6. `memory/episodes.jsonl`（按 family/行号读取小字段，不读取产物数组）
7. `recurrence-middle/QFormerCross8MiddleLoop.py`
8. `recurrence-query-mixing/QFormerCross8QueryMixing.py`
9. `recurrence-readout-retrieval/QFormerCross8DepthReadout.py`
10. `recurrence-state-routes/QFormerCross8StateRoutes.py`
11. `recurrence-state-routes/RESULTS.md`
12. `recurrence-training/RESULTS.md`
13. `recurrence-context-read/QFormerCross8ContextRead.py`
14. `recurrence-no-feedback/QFormerCross8NoFeedback.py`
15. `recurrence-fixed-memory/QFormerCross8NoFeedback.py`
16. `recurrence-work-relay/source-snapshot/QFormerCross8.py`（仅相关段落）
17. `recurrence-state-routes/runs/route_prediction_reset-s20262027-20260908T120047717081Z/result.json`
18. `recurrence-training/runs/middle_supervised12-s20262027-20260908T035104357175Z/result.json`
19. `recurrence-explore/runs/explore_dual_no_feedback-s20262027-20260908T091954887719Z/result.json`
20. `recurrence-memory-sources/RESULTS.md`
