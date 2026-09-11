# 两轮共享检索：是否只拆 q_norm 或 W_q

研究日期：2026-09-09。状态：备选的独立结构假设，尚未实现、未安排训练。当前研究优先级是先整理已有逐任务 R1/R2 证据，回答用户提出的任务异质性问题；本文不越过该步骤。本文只核查既有实验和原始论文；没有修改模型、冻结工具或全局 memory。

**最小未测假设是：保持两个完整检索及共享主干，仅让第二次 L2 使用独立的 query-head LayerNorm 仿射参数。** 当前 `q_norm=LayerNorm(96)` 的 γ、β 跨 head／slot 共用，复制一套只增加 **192 参数**。文献为研究共享约束提供动机，同时有明确负结果；没有发现与此位置完全一致、可直接外推收益的论文消融。

本轮 probe 给出的约束是保留原第二检索：q4.buy 的 `BCE(reuse P1)−BCE(original P2)` 为 +0.000014312，预定 99.375% 区间 [+0.000008108,+0.000021102]，高于 ε=0.000001。其余七个 checkpoint×task 组合均不确定。它不证明新增参数会改善，也不支持缓存 P1、增加 novelty penalty 或主动放大两轮差异。[固定 checkpoint 配对结果](../memory/tool-evolution/candidates/taobao-second-read-attention-v1/artifacts/shadow/aggregate.json)

## 本地查重：哪些已做过

核查范围为读取时 Taobao 的 168 条 episode／109 条实际训练记录，以及 KuaiRec 的 233 条 episode／167 条实际训练记录；同时核对对应配置与模型源码。数字只表示这次读取时的账本快照，之后的零训练归档可继续增加 episode。

| 已有项目 | 实际改变及结果 | 与本假设的关系 |
|---|---|---|
| Taobao `middle_r2_lora8_every_step` | 第二次 middle 调用只替换 `w_o.weight` 为 `W_o+BA`，rank8，增加 6,144 参数；macro AUC 0.826446913，比最佳单轮低 0.002176619，比匹配纯 R2 高 0.000256048，但 click 与 logloss 存在取舍。 | 是输出投影 LoRA，且基于纯 middle R2，不能当作 query-only LoRA 或当前 no-feedback 的等价对照；也没有理由原样重跑。[源码](workspace/qformer-loop/taobao-transfer/recurrence-adapter/QFormerCross8MiddleAdapter.py:33)、[结果](../recurrence-adapter/RESULTS.md) |
| Taobao `explore_untied_r2` | 复制完整中间层的全部参数供第二次调用；增加 887,424 参数，macro AUC 0.826954952。 | 同时放松检索、交互和输出等约束，并非只拆 `W_q`；在旧纯 R2 家族中完成，不能当作当前 no-feedback 下的匹配消融。保留其负证据，不重跑整个解共享层。[源码](workspace/qformer-loop/taobao-transfer/recurrence-explore/QFormerCross8RecurrenceExplore.py:23)、[结果](../recurrence-explore/RESULTS.md) |
| Taobao `route_q4_retrieval`／`route_q8_retrieval` | 只向第二轮 `w_q` 输入加 `B GELU(A LN(source))`；q4 source=work，q8 source=entry，rank16。`q2` 的交互／残差分支保持原公式。macro AUC 分别 0.828008444／0.824528857，相对同 q 原循环约 −0.000464121／−0.003992356。 | **已经严格做过只改变检索路径的训练**，不能宣称这是首次。但它适配的是非线性输入、保留共享 `w_q/q_norm`，并非轮次专属 `W_q` 权重或 `q_norm`。[分路源码](workspace/qformer-loop/taobao-transfer/recurrence-query-routes/QFormerCross8QueryRoutes.py:37)、[修正公式](workspace/qformer-loop/taobao-transfer/recurrence-query-routes/QFormerCross8QueryRoutes.py:115) |
| Taobao `explore_repeat_norm` | 第二次 middle 之前对整个 work state 做 functional LN；macro AUC 0.826853770。 | 改变所有 query 分支，且没有新增轮次专属仿射参数，不是 `q_norm` 消融。[源码](workspace/qformer-loop/taobao-transfer/recurrence-explore/QFormerCross8RecurrenceExplore.py:49) |
| KuaiRec `r9_field/history/both_splitnorm` | 分别拆字段／历史的接口 `final_norm`；相对位置匹配父模型的 AUC 增量为 −0.000369925／−0.000923987／−0.000114717。 | 轮次接口归一化已测且未胜出；作用于整个状态，既非 attention 内 `q_norm`，也非淘宝 L2 的检索专属路径。[源码](workspace/qformer-loop/model_zoo/QFormerCross11Disentangle.py:28)、[账本](workspace/qformer-loop/memory/episodes.jsonl:132) |
| KuaiRec `r9_untied_clone` | 同初始函数复制完整 history block；AUC 比共享父模型低 0.000460628，增加 1,185,536 参数。 | 完整解共享的匹配负证据；不等于 query-only 解共享的否定，但足以降低整体解共享的优先级。[源码](workspace/qformer-loop/model_zoo/QFormerCross11Disentangle.py:32)、[记录](workspace/qformer-loop/memory/episodes.jsonl:135) |
| KuaiRec round18 `DepthNorm` | 按总执行深度 1／2／3 拆最终 `final_norm`，配合深度训练安排；四组均未超过固定两轮父模型，只有部分匹配安排有所恢复。 | 属于最终读出与训练深度适配，不是同一 forward 两轮 L2 的 `q_norm`。[源码](workspace/qformer-loop/model_zoo/QFormerCross11DepthNorm.py:27)、[条件结论](workspace/qformer-loop/memory/lessons.json:745) |
| KuaiRec `QFormerCross11PassNorm` | 已有源码和 5 项 CPU fixture；strategy 明确 `training_started=false`。它拆 `self_attn_norm/cross_attn_norm/ffn_norm` 三个残差归一化。 | **不能记为已训练失败**；即使以后执行，也不是仅拆 retrieval `q_norm`。[源码](workspace/qformer-loop/model_zoo/QFormerCross11PassNorm.py:7)、[延后状态](workspace/qformer-loop/memory/strategy.json:1962) |

因此，查重结论是“尚未测过严格第二 L2 的 `q_norm` 独立仿射参数或仅 `W_q` 的线性轮次权重增量”，而非“轮次归一化／只改检索从未试过”。所有上述 AUC 都是既有固定 seed、固定训练预算结果；不同父模型的数值不能作为同构结构消融相减。

## 四篇原始论文：支持理由与反证

以下原文均在 2026-09-09 查阅；版本固定以便复核。没有下载或移植外部代码。

| 原始研究 | 原文实际证据 | 对本地假设的限定意义 |
|---|---|---|
| Bae et al., **Relaxed Recursive Transformers**, ICLR 2025，arXiv v3（2025-02-28），§2.2–2.3 | 共享主干可以配少量轮次专属 LoRA，使共享层保留不同深度的自由度。但原文连 RMSNorm 也共享；LoRA 加到全部共享线性矩阵，并利用预训练模型转换和残差 SVD 初始化。 | 支持“小幅放松共享”这一形式；没有验证独立 `q_norm` 或 query-only 收益。Norm 初始化消融不能改写成 Norm 共享／独立消融，也不能用其收益推翻本地输出 LoRA 的负结果。[原文](https://arxiv.org/html/2410.20672v3#S2.SS3) |
| Bae et al., **Mixture-of-Recursions**, arXiv v3（2025-10-25），Appendix H.2、Table 16 | 从头训练共享循环模型，试 Q/V rank64 LoRA、DoRA、轮次输出乘性编码等；作者报告没有实质性性能改善。对应 few-shot 均值：基线 40.2、LoRA 39.7、DoRA 39.2、轮次编码 40.2。 | 是“少量轮次参数自然有收益”的直接反证，降低继续扫 LoRA 的优先级；Q+V 仍不等于 Q-only。论文 KV cache 复用分支也不能替代本地固定外部 H/K/V 的实验。[原文](https://arxiv.org/html/2507.10524v3#A8.SS2) |
| Csordás et al., **MoEUT**, NeurIPS 2024，arXiv v2（2024-10-13），§2.4、§3/Figure 8 | 分析共享层面对随深度变化的 residual 尺度时的困难；peri-LN 区分用于选择的归一化路径与内容传播。归一化位置消融显示收益依赖规模，719M 的 peri/post PPL 仅 11.29/11.32。 | 为“单独审视检索打分的归一化”提供理由；它未测试轮次专属 Norm，归一化位置也与本地 post-projection head LN 不同。不能据此增加残差缩放或重排整层 LN。[原文](https://arxiv.org/html/2405.16039v2#S2.SS4) |
| Chen et al., **LoopMoE**, arXiv v2（2026-08-27），§3.3、§5.3/Table 3 | 独立比较 loop base、仅轮次 AdaLN、轮次＋token IterAdaLN，五任务均值为 42.26、41.47、42.38；仅轮次条件在五项上均下降。 | 给出归一化条件化的直接先例，也说明轮次身份本身不足。其 attention/FFN 归一化及 residual gate 远大于本地 `q_norm` 改动，不能宣称其正负结果直接复现于淘宝，也不由此扩展 token-conditioned 新网络。[原文](https://arxiv.org/html/2606.04438v2#S5.SS3) |

共同启发是：共享变换可以服务不同轮次，但放松共享的位置、参数化和训练起点决定结果。RRT 的成功与 MoR／LoopMoE 的相关负结果应一起保留。本地两个输入 `e` 与 `(e+z1)/2` 承担不同读取职责，只构成一个可证伪的设计理由；目前没有证据证明共享 affine 参数形成了实际瓶颈。

## 两个不同的最小结构假设

固定 `e=L1(q,H)`、`z1=L2(e,H)`、`q2=(e+z1)/2`。第二次读取仍然重新计算 `P(q2,H)`；保留 `w_qi(q2)`、原 residual `q2`、共享 K/V/mask、预测状态不反馈、原末尾 post attention/FFN。原位置见 [Q/K 归一化与投影](workspace/qformer-loop/upstream/model_zoo/QFormerCross8.py:164) 和 [no-feedback 两轮](workspace/qformer-loop/taobao-transfer/recurrence-context-read/QFormerCross8ContextRead.py:13)。

**优先级 1：仅拆第二轮 q_norm 的 γ₂、β₂。**

```text
U1 = heads(Wq e  + bq)        Q1 = LN(U1; γ1, β1)
U2 = heads(Wq q2 + bq)        Q2 = LN(U2; γ2, β2)
K, V、softmax mask/scale、I2=w_qi(q2)、residual 与其余共享权重保持原设计
初始化 (γ2,β2) = clone(γ1,β1)，其后两套 affine 参数独立训练
```

LN 的计算轴仍为每个 head 的 96 维；不是把 norm 扩为 384 维，也不是每 head／slot 各加参数。新增 192 参数：q4 预计总计 **51,724,420**，q8 **81,616,516**，最终须由模型实例审计。没有新增 attention 或 LN 调用，理论算子数不变；不能在实际计时前声称同延迟。

这个自由度允许第二轮对规范化后的 query 坐标作不同缩放／平移，随后通过同一个 K 改变打分。它不要求熵增加、重叠减少或范数变大；也没有新信息源。是否改善应由独立训练后的四任务指标判断。

**优先级 2：仅第二轮 W_q 的线性 rank8 增量。**

```text
Q1 = q_norm(heads(Wq e + bq))
Q2 = q_norm(heads(Wq q2 + bq + B(A q2)))
A: 8×384，B: 384×8；仅第二次 w_q 使用，q_norm 仍共享
```

新增 6,144 参数；q4/q8 预计总计 51,730,372／81,622,468。它与旧输入修正 `Wq·B GELU(A LN(source))` 的结构和 source 不同，也与旧 `w_o` LoRA 不同。该候选目前只有形式上的研究理由，受 MoR 及本地 retrieval-only 负结果约束，优先级低于 192 参数方案；不提出 rank 扫描或与独立 q_norm 同时增加。若未来另行实现，零初始化的两矩阵参数必须通过既有真实 PSGDKron 激活路径审计，不能擅改 optimizer、LR 或 clip 来绕过异常。

“保留 q2／交互／残差”指计算公式与连接保持，并非声称分别重新训练后的隐藏值和共享权重仍与原 checkpoint 逐元素相等。额外第二轮参数也会通过反向传播改变共享参数的学习过程。收益即使出现，也首先是这个训练结构对照的证据，不能直接归因于某一统计尺度或某个 head。

## 证据边界与当前优先级

当前先整理已有逐任务 R1/R2 结果，不启动本笔记中的结构。用户提出“是否只有某类任务从循环获得信息增量”是合理待检验问题；已观察到的 q4.buy 证据仅隔离第二轮概率重分配，既不等于整轮 R2 相对 R1 的贡献，也不代表 q8.buy。固定 H 的第二轮还包含状态计算与交互，不能把它泛称为新增外部信息。

本文不设置训练矩阵、追加 seed 或新的训练预算。备选问题只是“仅放松检索归一化的轮次共享是否值得验证”。任何未来结构结论都需要同 q 原循环的可比结果、完整四任务 AUC/logloss 及配对区间，不能被 macro、单个任务、head 或历史长度分桶掩盖；参数和实际延迟也是判断候选价值的组成部分。已有近候选政策不等于统计等价。

克隆 γ/β 可以保持起始函数相同，但之后参数分离、attention 改变或 perturbation 敏感都不是训练收益。少量参数的优势是让假设更局部、代价更低，不是保证收益。只有新的结构证据才能提高该备选假设的优先级；本轮工具结果不构成 cache、novelty、输出 LoRA 或整层解共享的重启理由。
