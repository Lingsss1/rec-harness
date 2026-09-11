# 下一步 Q2 条件形成：查重与决策审计

本次只推荐一个条件性机制诊断，不推荐立即开始新的结构训练。先完成 F 的 16 项配对、组合量 I、结果解释及入账；下述建议既不是已验证工具，也不是调度指令。

## 当前状态与四种不同问题

本次读取时，F 八组正式训练和 quality audit 已完成，11 模型 latency 已完成。根节点已核：八组对各自原父的 macro AUC 点差均低于 −0.0001，延迟比均大于 1.05，按预定点值/成本规则没有保留候选。配对任务仍在执行、I 尚未计算；本笔记没有读取未完成配对产物，不能写成八个区间都负、角色差值显著或组合非加性已成立。质量文件 SHA256 为 `03f49184cbdfde17bf200dae10caa28b5a2619c9d9aa9ba88bd2daf526927af5`，latency completion 为 `7d258b93cc8c05057a517a2641b5d508452598b576a74b1dafd0e0032672dce6`。

| 问题 | 本地含义与不能混用的证据 |
|---|---|
| 改善原 Q2 条件 | 原父 `q2=(e+z1)/2`，再经 `w_q` 与逐 head `q_norm` 形成检索 Q。问题是第一轮结果怎样参与后续寻址；改整个 q2 还会同时改变 `w_qi` 和 residual。 |
| 改 K/V 供给 | F 保持原 q2 公式，只限制第二轮历史 K/V 的 item/action 字段；coverage 改 score，memory-sources 改可读 token。它们的失败不能替代对 Q2 信息来源的检验。 |
| 改读出 | `h2+a_t(h1-h2)` 改任务头使用哪轮状态，不等于改善第二轮检索；现有任务差异不自动要求独立读出。 |
| 改优化器轨迹 | 新参数可改变 clip、预条件器分桶初始化或随机探针；零新参数的结构改动仍会改变真实梯度。初始参数和成员布局一致均不保证训练动态一致。 |

当前 memory 账本为 133 次完成训练，F 的八次已完成但尚未入账；本笔记不提前修改为 141。`strategy.pending_field_role` / `next_queue[0]` 的 `paired_and_latency_running` 是较早状态摘要，已保存的 latency completion 已完成；保留 F 配对/I 尚待收尾这一边界，不反过来把该摘要当作 latency 未完成的证据。

## 已经做过什么，不能换名字重跑什么

查重依据为当前 [strategy](../memory/strategy.json)、[lessons](../memory/lessons.json) 的已完成/失败 families，以下实际源码和已完成说明，以及 [C 的下一结构审查](../recurrence-coverage-retrieval/NEXT_STRUCTURE_AUDIT.md)、[F 文献对照](LITERATURE_CONTEXT.md)。C 文档中两个“未实现”假设是历史时点：其中字段角色分工现已成为 F；逐事件 V 条件残差仍只是另一个供给假设，不能改称 Q2 方案。

| 已有近似变体或反例 | 本次排除的重复建议 |
|---|---|
| [StateRoutes](../recurrence-state-routes/QFormerCross8StateRoutes.py) 已做 `anchor0`（第二轮只取 z）、work/prediction 缩放或归一化和更多 inner 更新；上下文回读及 memory 拼接也有负结果。 | 不再提去掉 e、简单 LN、增加内循环或拼回入口 token，称为首次改善条件。 |
| 八种 rank16 加性 source 修正已完成，含被要求补跑的 difference。q4 work 仅是近原父候选（Δmacro −0.000006789），没有证明质量提升；difference-only 在两尺寸均低于 mean 控制。 | 不重跑 `B GELU(A LN(z1-e))` 或把修正非零当作有效信息。 |
| [QueryMixing](../recurrence-query-mixing/QFormerCross8QueryMixing.py) 已做 `base + gate(e,z1,z1-e)*(z1-e)`，channel/query × joint/constant；没有保留候选。随后固定 q4 channel gate 的 zero/shuffle 也未证明有益条件化。 | 不把“学习第二轮两源比例”、同式固定门控或小幅改 rank/bound 作为独立机制。 |
| [QueryRoutes](../recurrence-query-routes/RESULTS.md) 已把学得 correction 限到 retrieval/interaction/residual 等路径，八组均未进入候选。第二 `q_norm` / 完整 `W_q` 解共享也已失败。 | 不能仅凭“这次只改 w_q”或轮次参数克隆宣称未做过；诊断上的分路依赖没有转化为训练收益。 |
| [方向改造](../recurrence-directional-retrieval-v2/RESULTS.md) 的外推及三类 key 反馈、[coverage](../recurrence-coverage-retrieval/RESULTS.md) 的 mean/LOO 正负偏置均未产生候选；F 又检验固定字段供给。 | 不追加几何强度、coverage 强度或字段方向小扫；原 P2 / 方向必要性不证明放大、抑制或另换来源会更好。 |

这些结果否定的是指定结构、固定 seed/预算下的候选资格，不是所有循环或所有条件化。尤其是“只有 q4.buy 的某条干预路径达到门槛”，不能变成把 buy 作为新主目标的依据。

## 唯一建议：先隔离原生首轮更新的样本对应关系

**问题：原父模型第一轮工作更新与当前样本的对应关系，是否对第二次 `w_q` 检索有条件性价值？** 这是诊断问题，不是假定能提高 AUC 的训练公式。

令 `e` 为入口、`z1` 为首轮 work 输出，记实际张量差 `r=z1-e`。实数下原 `q2=e+r/2`。只在原 q4/q8 checkpoint 第二次 `w_q` 输入处，比较 native 与一个预先固定的跨样本 donor 干预：

```text
q2_retrieval_i = q2_native_i + 0.5 * (r_donor(i) - r_i)
q2_interaction_i = q2_residual_i = q2_native_i
```

保留原 H/K/V/mask、第一轮、原 prediction/no-feedback 与后续读出。公式写成相对原 q2 的改变量，是为了不将整个 native 路径重算为 `e+(z1-e)/2`；两者在浮点下不保证逐位相同。native/identity 应直接走原运算，donor 路径的减加舍入属于实际干预，必须记录。`r` 是包含原 cross-value/affine 更新作用的实现差量，不是已定义的信息量；此替换也会影响最终 Q 的方向和范数，不能当作纯方向诊断。

未见严格相同既往执行，依据是：

- [原 query-parts](../query-conditioning/probe_query_parts.py) 的 `work_shuffle` 在整个 L2 prehook 替换 `z1`，并非只替换 retrieval。由于 `z1` 自身含入口残差，它还带入 donor 的入口；旧“work 有用”不能独立归因首轮新增更新的检索作用。
- [旧 route probe](../query-correction-diagnostics/route_probe.py) 只允许 `QueryCorrectionStage`，干预的是新增 adapter 的 correction，不是原父第一轮 r；其失败训练反证须保留，不能拿其调用结果冒充原父的测量。
- 已完成 P2-reuse / direction-norm 证明指定第二查询替换会伤害 q4.buy，但没有区分“当前样本的首轮更新”与“同分布 donor 更新”是否等价。这里问样本对应关系，不再重复测 P2 与 P1 的整体差异。

若之后确有决策价值，可复用既有 5514 行/4647 用户/IPW 子集及既有统计随机流，只研究这一个干预；不重新选数据、挑任务、扫描比例或扩大 checkpoint 集合。身份对照须 bitwise、两次 work / `w_q` 调用准确，`w_qi` 和 residual 仍取原样本 native q2，所有模型参数/buffer/RNG 与临时 hooks 正常及异常恢复。两 checkpoint 的全部四任务必须报告；若要讨论任务差异，必须预声明直接任务效果差，不能按各任务区间是否跨零分类。现有固定子集存在条件推断边界，donor 中相同用户、固定点和批次组成也应如实记录。

**决策边界：** native 更好只能支持保留该样本对应关系，仍不产生一个已证明更优的 Q2 公式；打乱后变差也可能包含训练分布外扰动，不能量化独立“信息增量”。结果不明确则不优先继续针对 r 的新结构训练；干预改善也只说明该固定 checkpoint 下的敏感性，不授权重训或替换 incumbent。这项诊断的价值是限制接下来的条件来源主张，不能为了产生一个正结果而把它自动转成模型候选。

本次不实现、不启动该诊断，不创建 tool candidate/handoff。旧 probe 不能直接放宽结构 guard 来运行它；需要根节点在 F 闭环后先确认是否值得单独冻结这项窄问题的实现、身份与总预算。没有该决策便不执行。

## native optimizer layout shadow 当前没有被触发

已核 `next_queue[1]` 原文：`Before claiming a new-parameter variant preserves native optimizer dynamics`，状态 `conditional_not_dispatched`。F 零新增参数，smoke 只验证原初始化/RNG/静态布局及真实更新，没有声称改变结构后的全程梯度或优化器动态不变。因此 **F 负结果不满足这项触发条件**，本次建议的同权重推断诊断也不插入训练参数或执行 optimizer step。

[已有重放结果](../optimizer-layout-replay/reports/interpretation.json) 仅是 synthetic CPU eager 一步实现证据；原 [structural-trigger](../optimizer-layout-replay/structural-trigger.json) 中 `synthetic_execution_started=false` 是注册时快照，不否定后来完成的重放，也不表示原生 shadow 已完成。[method card](../memory/tool-evolution/candidates/probe-302bf4e60eb7a4e126b2d1b1/method-card.md) 所要求的独立 native compiled 两条件及 utility gate 仍 pending，不能把 synthetic_pass 当作可自动消费的修复工具。

只有以后真的决定引入新参数、并需要声称或构造与原 optimizer 动态匹配的对照时，才重新核该触发条件及独立预算；这仍不是自动派发。无论重放结果如何，都不能据一两步更新差异解释 Taobao AUC 下跌量或挽救已有负训练。

## 保持的边界与记录

任何后续训练仍保留原两轮、无 prediction→work 反馈、训练 seed20262027、PSGDKron 每步预条件更新、原学习率、clip1、2193 updates；本次没有建议改变它们。优化器设置保持与真实优化器轨迹相同是两回事。当前 F 收尾优先，宏目标、近候选阈值和 best single 参考不改，配对/I 的正负结果最终全部保留。

本次只读小型源码/元数据/既有说明并新增本文件，未浏览、加载数组或 checkpoint、执行测试/forward/bootstrap/GPU/训练或写全局 memory。读取时 strategy SHA256 为 `d0b38b36732d26c0cbfc349732888b54c4e7cae9920edefb9d4bc66c9aaa33cb`，lessons SHA256 为 `709387b84e9deb7a9414d47a943a29bf2336a076fa6ff1dc23417e9562550aea`；它们是本次查重快照，不阻止根节点正常完成后续入账。
