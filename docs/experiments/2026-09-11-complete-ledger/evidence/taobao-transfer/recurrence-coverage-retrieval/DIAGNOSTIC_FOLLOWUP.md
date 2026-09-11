本轮结果未出；以下是条件决策，不预设 coverage 有收益，也不构成新实验启动。检查时 memory 仍登记八组正式训练进行中；本说明未读取新训练指标或真实数组，只读既有协议、源码、诊断结果与工具状态。

先完成既定的质量审核、16 项固定配对（原循环 8 对、正负号 4 对、mean/LOO 4 对）、预定最多 2 项对最佳单轮的条件配对，以及 11 模型成本测量。使用本轮四任务和六项直接任务差异，不能因一个任务显著、另一个不显著就宣称任务间差异。具体来源为 [protocol.json](protocol.json)、`reports/quality-summary.json`、`reports/paired-analysis/` 和 `reports/latency/`；后两项是待完成证据，不是本说明已观察到的结果。

| 完成后的分支 | 先用现有证据解决什么 | 只有仍影响决策时才需要的最小新诊断 |
| --- | --- | --- |
| 有质量候选：相对同 q 原循环 macro AUC 点差为正 | 先核四任务代价、配对区间与延迟；正差只是协议定义的候选，不能称已证明赢过原循环或最佳单轮。按原定规则读取 eligible 的最佳单轮对照。正负号、mean/LOO 的训练对照用于评估本轮固定训练条件的结构差异，不能替代同权重机制归因。 | 若要决定保留其推理时 coverage 偏置，验证“固定该 checkpoint，只去掉第二轮实际 additive bias 是否变差”。优先一个按既定结果规则选出的候选；工具适用性验证需要额外条件时必须单独冻结范围，不能自动全八组展开。 |
| 只有接近候选 | 按原协议检查相对同 q 原循环点差 ≥ −0.0001 且实测 forward 延迟比 ≤ 1.05；这是队列规则，不是等效检验。先用已有参数量、延迟及任务折中判断是否值得保留，不为接近结果自动再做诊断。 | 仅当“偏置在当前权重下有用”与“偏置可关掉以简化”会改变保留决策时，做同一个实际 bias 干预；敏感性不能把一个不达质量/成本门槛的模型升格为赢家。 |
| 全部负，且无满足上述接近规则者 | 保留原循环和最佳单轮，归档完整配对与延迟结果。若全负但有合格接近候选，归上一行。先看正负号是否都落后原模型、mean/LOO 是否都落后；候选之间较好不能抵消对原模型的退步。 | 默认不启动新 probe。只有一个具体决策尚未解决，例如“推理偏置伤害了当前 checkpoint，还是训练后权重本身已变差”，才值得同权重 zero-bias 诊断。它最多区分当前依赖/伤害，不能解释全部训练差距，更不能直接支持重新训练或放大/缩小 strength。 |

当前可复用的是统计边界与旧证据，不是可直接覆盖新模型的 attention 适配器：

- [probe-coverage.json](reports/probe-coverage.json) 已明确缺口：旧 Q/K 重建未包含新增第二轮 additive bias。原 [SecondReadAttentionProbe](../memory/tool-evolution/candidates/taobao-second-read-attention-v1/src/attention_probe.py) 通过第二次 `w_q` 输入替换实现原模型 P2→P1；其 descriptor 只算 `softmax(QKᵀ*scale + padding)`。在 coverage 模型中，即使替换 Q2 为 Q1，实际概率仍含 coverage bias，不能称复用第一轮概率；旧类还绑定原源文件路径，不能绕过 guard 强行套用。
- 原工具 [candidate.json](../memory/tool-evolution/candidates/taobao-second-read-attention-v1/candidate.json) 为 `accepted`，六 gate 已通过；但仅为已验证的 repo-local 原模型工具，registry 无 promoted 事件。`probe-842422a498d9bcfd4105a273` 的 `idea/pending` 是一对一来源别名，不是另一份待执行 shadow。实际 `memory/tool-evolution/handoffs/` 不存在，无 ready edit。优化器布局工具 `probe-302bf4e60eb7a4e126b2d1b1` 仅 `synthetic_pass`，原生 shadow/utility 未通过，不能用于解释本轮 AUC。
- 原 [attention outcome](../memory/tool-evolution/candidates/taobao-second-read-attention-v1/artifacts/outcome.json) 只有 q4.buy 的复用损害通过预定方向门槛：BCE +1.43117e−5，99.375% 区间 [8.10783e−6, 2.11016e−5]；另七项 inconclusive。已有方向/范数分解进一步约束这一固定 checkpoint 的 query 方向，不能外推新 coverage 权重，也不能推出“重复历史无用”。这两项均已完成，不重跑。
- [lessons.json](../memory/lessons.json) 已记录任务收益依赖比较范围、四个 early/final readout 和四个第二轮 untie 均未超过原模型、八个方向/Key-feedback 变体全部不达队列。此前 learned gate 非零方差也未证明有用的样本对应。保留这些反证；不重复 readout、方向/范数、query gate 的同一诊断，不扩展标量扫描。

若上述分支确实触发新诊断，最小可证伪问题是：**在冻结 coverage checkpoint 上，保持第一轮、原 Q2/K2/V2、`w_qi(q2)` 投影和残差输入，只把有效历史位上的实际 coverage bias 置零，四任务配对 BCE 怎样变化？** 使用 `zero − native` 的明确符号：正值表示当前权重依赖这条偏置，负值表示去掉后当前预测改善；跨零仍 inconclusive。它不是训练收益估计，也不证明历史语义相关性更高。

建议届时先冻结薄适配器的三个条件：原 native、同形同 dtype additive-mask 的 `format_zero`、直接原路径的 `identity`（去 bias 并绕过 coverage）。最后一个只用于量化 mask/实现格式差异，不假称等同于训练前原 baseline 权重。`format_zero` 对比能减少格式变化混杂；与原路径的差异若不可忽略，就必须保留实质格式对照，不能用数学上的零偏置等价代替数值校准。推理的 eval/no_grad 校准也不能证明训练时 requires-grad mask 的 backend/梯度等价。

适配器应捕获模型真正传给第二次 SDPA 的有效 bias，或以捕获的原 Q1/K1、mask、scale 复算后逐元素核对；不能仅修改旧 descriptor 的名称。FP32 reference P1 是定义 coverage 的参考分布，仍不是 fused kernel 暴露的内部概率。归零只针对有效位，padding 保持 −inf；全 padding 的描述量 N/A，原 affine bias 行为保留。第一轮、raw Q2、K/V、投影输入、残差、输入数组、state/RNG、调用次数和异常恢复必须有证据；attended value 改变会自然改变后续乘积，不能声称整个 interaction 输出不变。

未来验证先做可注入的有用/有害/零效应 toy、mask 孔洞/全 padding/单 key/相同 V、identity 和恢复检查，再按单独固定的预算在当前 checkpoint 做 shadow。可沿用现有 5514 行、4647 用户、16 strata 的验证子集及已接受 IPW/user-cluster 统计接口，绝不重新选子集或按结果选 donor。统计调用、条件数和区间门槛须在新测量前固定，不沿用旧工具的“已接受”作为新 tensor 契约的免检通行。现有 `analyze_perturbations.py` 可消费配对表；它解决统计接口，不能保证 exporter 的路径隔离正确。

[诊断 toolkit](../../../rec-harness/model-diagnostic-toolkit/SKILL.md) 与 [evolver 流程](../../../rec-harness/model-diagnostic-tool-evolver/references/co-evolution-loop.md) 要求先说明决策价值，再验证改变过的 tensor 契约；null/冲突证据不产生 edit handoff。新 adapter 若有必要，应在隔离 candidate 下验证，不能写进 stable toolkit 或覆盖当前冻结源。本说明未创建 candidate/handoff、未运行训练、forward、bootstrap 或 GPU，亦未修改全局 memory。

状态读取时间：2026-09-09T11:26:17.827504+00:00。状态摘要 SHA-256（只读快照，后续 memory 合法推进不改变本说明的条件性质）：

- `recurrence-coverage-retrieval/reports/probe-coverage.json`: `a56c75210ea9658718b15e01be1e027ecebebb5ac4115c615774ff25ea406ed6`。
- `memory/strategy.json`: `f217097196341cb96e85f7a6c3bcef2c858d5582a04cee846582b5a5d06a39a9`。
- `memory/lessons.json`: `ea96fccca8964b0f584a3b4eacd6a4a89f6726599f0cfefeab543867e223b670`。
- `memory/tool-evolution/registry.jsonl`: `8fa11e00c858ae0239e585203b4aa5c2f9df750cca6673cff8348d284412be17`。
