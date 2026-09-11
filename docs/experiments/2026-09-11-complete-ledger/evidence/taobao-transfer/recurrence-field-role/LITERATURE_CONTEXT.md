# 第二轮 K/V 字段分工：原始文献与本轮解释边界

核读日期：2026-09-09。本说明在 F 正式训练进行期间编写，未读取本轮训练结果；只补充解释背景，不改变冻结协议、候选政策或训练安排。结论是：文献给出了“寻址所需特征”和“读回内容”可以不同的设计理由，但没有证明淘宝应采用 item 作 K、action 作 V，更没有证明只改第二轮就会提升。

## 本地查重与阅读范围

先检查了 `recurrence-research` 的研究笔记、`memory/strategy.json`、`memory/lessons.json` 和已接受 attention 诊断的主要说明文件。本次检索范围内未见以下三篇的专门原文核读；既有 [轮次专属检索笔记](../recurrence-research/ROUND_SPECIFIC_RETRIEVAL_NOTE.md)、[coverage 审查](../recurrence-research/COVERAGE_RETRIEVAL_AUDIT.md) 和 [任务证据](../recurrence-research/TASKWISE_RECURRENCE_EVIDENCE.md) 已覆盖其他问题，故不重做 q_norm、LoRA 或循环深度综述。早期本地文档的“尚未运行”等字段保留其历史时点，不能据此否定后来完成的诊断。

外部共执行两个搜索查询：`"Key-Value Memory Networks" Miller 2016 paper`、`"disentangling" "search" "retrieval" attention paper`；精读下列三篇第一手论文。没有使用二手综述作为证据，没有运行模型、真实数组、bootstrap 或 GPU，也未修改全局 memory。

## 三篇原文实际支持什么

**Miller 等，Key-Value Memory Networks for Directly Reading Documents，EMNLP 2016。** 文中先按 key 寻址，再加权读取 value，并以 `q_next = R_j(q + o)` 更新下一跳查询。WikiMovies 文档任务 test Hits@1 从 MemNN 的 69.9 到 KV-MemNN 的 76.2；但 dev 表的窗口、title、center encoding 改动涉及表示、词典规模或额外 memory slots，不能视为严格同容量的 V-only 消融。WikiQA 的 value 又使用完整句子，说明角色分工依赖任务。它支持不同 K/V 表示的可能价值，不直接支持 F 的字段方向或 R2-only 位置。[原文 §3、§5.1 Tables 2–3、§5.2](https://aclanthology.org/D16-1147.pdf)

**Mittal 等，Compositional Attention: Disentangling Search and Retrieval，ICLR 2022。** 方法把 S 个 search 和 R 个 value 映射交叉组合，再用额外的条件化注意力选择返回内容。Sort-of-CLEVR 的 binary 准确率中，4-search/3-retrieval 为 89.8±1.1，原 Transformer 为 84.4±5.3；但 4/4 的 ternary 为 64.8±4.1，对照为 64.9±3.3，并非各配置都获益。论文采用参数匹配比较，但增加了组合/选择计算。它支持 search 与 retrieval 不必固定一一配对；F 没有这套动态选择器，固定字段拆分不是该方法的直接复现。[原文 §3、§4.2 Table 1、Appendix B.5](https://arxiv.org/pdf/2110.09419)

**Sukhbaatar 等，End-To-End Memory Networks，2015。** 同一条记忆已经分别编码为 input memory 和 output memory；共享跨跳编码器时，以 `u_next = H u + o` 更新查询即可让后续读取具有不同条件。共享层的 PTB 设置中，2、3、7 跳 test perplexity 分别为 121、122、114：总体更深可以有益，但增加一跳不保证改善。其 QA 部分还采用多次初始化选择，不能与 F 单 seed 数值直接对齐。它支持“固定记忆、更新读取状态”的多跳机制，不证明逐跳改字段是必要条件。[原文 §2.1–2.2、§4.2、§5 Table 2](https://arxiv.org/pdf/1503.08895)

## 与本地 F 的逐项对照

原父 attention 本来就有不同的 `w_k` 与 `w_fi`。本轮新增的问题是第二次读取的**原始历史字段供给是否应分工**，不是首次引入 K/V 参数分工。以下公式来自本地实现，不是上述论文的实验结论：

```text
H_full   = mask * LN(W [x_item; x_action] + b + position)
H_item   = mask * LN(W_item x_item + b + position)
H_action = mask * LN(W_action x_action + b + position)
q2       = (entry + first_work) / 2
K2       = k_norm(w_k(H_key))
V2       = w_fi(H_value)
```

| 本地实现约束 | 对解释的影响 |
|---|---|
| 六个 item 字段与历史 action 使用原投影的对应列；full 仍走原一次完整投影 | 是零新增参数的字段限制；不是学习新的独立编码器，也不是把 full 表示线性拆成两个 LN 后分量。 |
| 备用历史共享原 bias、位置编码和同一个 LN 的 affine/eps；最后乘原 mask | action 路径仍有位置及共享表示变换，不能叫“纯动作向量”。重复使用共享参数会改变梯度累积；LN 使两分支一般不可相加还原 full。 |
| 所有字段先去掉最后目标槽；历史 mask 不重建 | 不引入当前四任务标签。这里的 action 是历史事件字段，query slots 也不是任务标签。 |
| 第一轮仍读 full；原 q2 公式、`w_q/q_norm`、`w_qi` 直接输入及 residual 公式不改 | 第二轮仍可通过 q2 携带第一轮全部字段的信息；item/action 分配不是全模型删除某类信息。“Q2 保持”指计算定义，分别训练后的实际权重与 Q2 数值并不保持相同。 |
| 只在第二 work attention 替换 K/V 输入，原 SDPA/mask/scale 复用；prediction 不反馈到 work，post 一次 | 与论文逐跳查询更新或全层动态组合的修改范围不同。原 no-feedback 边界继续成立；没有加入论文中的读出到查询更新公式。 |
| 无新参数、无新增 attention 次数，但多构造一条或两条备用历史 | 保持初始参数/优化器布局，不代表梯度、clip、PSGDKron 状态、激活内存或时延相同，必须使用本轮真实成本。 |

四个 K/V 分配 `item/full`、`full/action`、`item/action`、`action/item` 在 q4/q8 分别对原 `full/full` 比较。两个单改和组合可回答“这套训练路径的字段限制是否有用”；正反分配可回答方向是否影响结果。它们不能单独证明被限制字段是噪声、item 天然只适合寻址、action 天然只适合返回，也不能把单一任务显著而另一任务不显著当作任务差异。任务差异应读预定的直接效果差，而非按显著性标签分组。

## 最多两条条件性解释线索

1. **若某一角色分配满足既定候选规则，先用已冻结对照定位收益范围。** 单改对父、组合对两个单改以及反向对组合可以区分整体路径表现；已有共同 draws 的组合量 `I = D_combo - D_key - D_value` 只表示指定训练结果的非加性。I 为正或组合胜过单改，都不能替代组合对原父的质量/开销要求，也不证明同权重下的“信息互补”。无需为此新增试验。
2. **若固定字段分配失败，结论只及于这套固定角色与共享变换。** Compositional Attention 的条件化选择和 KV-MemNN 的任务定制表示都超出了 F 范围；失败不否定所有 K/V 分工，论文成功也不能据此授权直接追加动态路由。只有本轮完整结果及资源证据能决定是否值得另立假设；当前不安排后续训练或诊断。

本轮仍遵守原 macro 目标和同父候选政策，保留全部正负结果。已有第二检索必要性、方向依赖或任务异质性证据只提供解释约束，不能预报 F 收益；文献也不提供“只有 buy 才从循环获得信息”的结论。

本地依据：[冻结协议](protocol.json)、[模型](QFormerCross8FieldRoleRetrieval.py)、[设计与公式审查](DESIGN_AUDIT.md)。本次记录的 protocol SHA256 为 `dac1d1dbb9aa0d145235b0946303d94f8417f0c69c06a0ce0748606226d3415d`，model SHA256 为 `98a93587e2f39109b5d6799a6ad9bc3dea1e423a9dcaf98de6812965f7142283`。只新增本说明，不修改任何冻结来源。
