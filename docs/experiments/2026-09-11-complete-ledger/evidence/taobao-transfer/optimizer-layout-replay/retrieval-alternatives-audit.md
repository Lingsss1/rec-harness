**查重结论：在当前淘宝 117 次已完成训练、179 条 episode 及 `recurrence-*` 人工源码中，两项精确结构均未发现已实现或已训练的记录。** 这是本地已归档范围内的结论，不是整个领域的新颖性声明。此次只读代码和记录，未读取训练数据、运行模型或修改全局 memory。

| 精确结构 | 本地状态 | 与旧实验的区别 |
| --- | --- | --- |
| 第二次历史 attention 的 logits 加固定正系数 `α log P1` | 未做 | 旧 probe 是把第二检索整体替换为第一检索；没有把原第二检索与第一检索相加。 |
| 上述先验取固定负系数，抑制第一轮偏好的位置 | 未做 | 没有已训 coverage、log-probability prior 或这种负向 logits 修正；旧记录中的 novelty 是未执行建议或明确限制。 |
| 第二轮投影查询向量减去其在第一轮投影查询方向上的分量 | 未做 | 旧 `query_projection` 是完整 Wq 参数克隆，不是向量投影消除；旧 difference correction 也不是正交化。 |

本地两轮确实共享历史 K、mask 和 scale：[NoFeedbackStage](../recurrence-no-feedback/QFormerCross8NoFeedback.py:15) 第 18–27 行把相同历史和 mask 交给同一 L2；[原 CrossValue](../recurrence-readout-retrieval/source-snapshot/QFormerCross8.py:178) 第 184–205 行用共享 `w_k/k_norm`，SDPA dropout 为 0。已完成的 [attention probe](../memory/tool-evolution/candidates/taobao-second-read-attention-v1/src/attention_probe.py:290) 第 291–298 行逐前向 exact 核验 K/V/H/mask 和 scale；两原 checkpoint 的现成 `artifacts/shadow/{ctx_q4,explore_dual_no_feedback}/audit.json` 均记录 `shared_memory_mask_K_V=true`。

令 `Q1,Q2` 为 **q_norm 后、逐 head/slot** 的查询向量，`S_i=s Q_i Kᵀ`，`P1=softmax_valid(S1)`。对非空有效支持集：

`softmax_valid(S2 + α log P1) = softmax_valid(S2 + α S1) = softmax_valid(s (Q2 + α Q1) Kᵀ)`。

原因是 `α logsumexp_valid(S1)` 对每条 query 都是常数。因此“log 概率先验”在本地条件下就是归一化后查询方向的线性组合，不能作为另一个独立概率机制计数。该式为实数数学等价，不承诺 BF16/不同计算图逐位相等；`detach`、概率裁剪或额外归一化会改变梯度或破坏等价。特别是负 α 不能直接乘 padding 的 `log 0=-inf`；运算仅针对有效支持，padding 仍屏蔽，全 padding 沿用原处理。

方向消除若定义在同一 q_norm 后空间，则 `Q2' = Q2 − (<Q2,Q1>/<Q1,Q1>) Q1`（非零 Q1）是逐样本/head/slot 系数的上述线性组合；它与固定 α 不同，也不保证两轮概率低重叠。若在原始 `Wq(query)` 上先消除方向再经过 LayerNorm，已经是另一种运算，不能声称实际打分向量仍正交。零方向的处理及归一化位置尚未定为训练协议；本文件不扩展 sweep。

| 已完成的近邻 | 具体实现 / episode | 为什么不算做过 |
| --- | --- | --- |
| 不增参数的入口/工作态固定混合 | [NoFeedbackStage](../recurrence-no-feedback/QFormerCross8NoFeedback.py:21)；`nf_entry025-s20262027-20260908T102300904513Z`、`nf_entry075-s20262027-20260908T102308673232Z`、`nf_fixed_memory-s20262027-20260908T115619206369Z`（[episodes](../memory/episodes.jsonl:70)，71、75 行） | 改的是 q_norm 前整个工作输入，连同交互与 residual 一起变；不是 q_norm 后只改检索方向。`explore_repeat_norm`（59 行）也只是对第一工作态先做 functional LayerNorm。 |
| 第二轮 retrieval-only rank16 修正 | [QueryRoutes](../recurrence-query-routes/QFormerCross8QueryRoutes.py:37)，第 109–120 行新增 A/B；`route_q4_retrieval-s20262027-20260908T190038665636Z`、`route_q8_retrieval-s20262027-20260908T190022268468Z`（[episodes](../memory/episodes.jsonl:130)，134 行） | 修正 Wq 的输入，再经过原 q_norm；不引用第一轮 Q 或 P。八条 route 已由 `taobao-query-routes-final-interpretation` 判为无入队候选。 |
| 第二轮 q_norm / 全 Wq 克隆 | [RetrievalUntied](../recurrence-readout-retrieval/QFormerCross8RetrievalUntied.py:27)，第 40–49 行替换参数；四条 `role_q{4,8}_query_{norm,projection}`（[episodes](../memory/episodes.jsonl:173)，174、177、178 行） | 新增参数后解共享，不做方向消除；四组均低于对应原模型，已 discard。 |
| difference additive / bounded mixing | [QueryCorrection](../recurrence-query-correction-v2/QFormerCross8QueryCorrection.py)、[QueryMixing](../recurrence-query-mixing/QFormerCross8QueryMixing.py:50)；`query_q4_difference-s20262027-20260908T184409455021Z`、`query_q8_difference-s20262027-20260908T171234078895Z`，以及 `taobao-query-mixing-final-interpretation` | 输入空间新增低秩网络，影响检索、交互和 residual；并非对 Q1 方向的解析投影，也非 logits 先验。差值补跑已完成，不能重复列待跑。 |
| 输出 LoRA / 全 L2 解共享 | [MiddleAdapter](../recurrence-adapter/QFormerCross8MiddleAdapter.py:33)、[RecurrenceExplore](../recurrence-explore/QFormerCross8RecurrenceExplore.py:23)；`middle_r2_lora8_every_step-s20262027-20260908T082358293826Z`、`explore_untied_r2-s20262027-20260908T094526671908Z` | 分别改 `w_o` 和整层参数；不是保持原参数布局的第二检索变换。 |
| 固定 checkpoint 的 P2←P1 | [probe](../memory/tool-evolution/candidates/taobao-second-read-attention-v1/src/attention_probe.py:248)；`taobao-second-read-attention-originals-5514`（[episodes](../memory/episodes.jsonl:169)） | 已完成零训练诊断，仅支持原 q4.buy 保留第二次重分配；其余七项不确定。不证明正先验、负先验或正交化有训练收益。 |

直接的函数式查询/打分变换可以不注册新参数，从而保留原参数名称、形状和顺序；这仍需将来实现核验，不能推导优化器更新或训练轨迹不变。当前只能把它们记作**未做过的局部结构假设**。最近 [lessons](../memory/lessons.json) 中第二检索解共享的负结果和 P2 必要性的限定仍有效；`recurrence-research/INSIGHTS.md` 第 98–104 行“尚未诊断”是历史状态，不能覆盖已完成 probe。未提出或启动任何新训练。
