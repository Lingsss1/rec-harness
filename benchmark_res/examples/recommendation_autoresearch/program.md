# 推荐模型 autoresearch 指令

本指令适配 Karpathy autoresearch 的单编码 Agent 循环。执行器通过 `request.json` 提供当前阶段和上下文；运行方法见 [README](README.md)，具体方法见 [适配方案](../../docs/karpathy-recommendation-autoresearch-v0.md)。训练和独立评分由主机执行，Agent 不自行启动训练或下载数据。真实服务器验收与研究效用仍需单独验证。

`phase=query` 时先阅读当前 `candidate.py`、模型事实、诊断、全部历史结果及分析（包括 discard/crash）。只写 `retrieval-plan.json`，不编辑模型、不训练。每轮重新生成 1–3 个检索问题；不能把固定问题机械沿用。问题必须来自当前证据，且至少有一个反证问题。输出格式：

```json
{
  "queries": [{
    "text": "当前随机负样本区分较好，但全候选排名较差；难负采样何时因假负例而失败？",
    "intent": "counterevidence",
    "keywords": ["hard negative", "false negative"],
    "conditions": [{"field": "model.negatives", "op": "lt", "value": 32}],
    "rationale": "解释观察到的目标不匹配，同时寻找竞争解释；这里只是格式示例",
    "evidence_refs": ["model.negatives", "diagnostic.mean_target_rank", "trial:t0000"]
  }]
}
```

`intent` 可取 mechanism/method/counterevidence。引用必须来自当前请求；整体计划要引用模型事实、已有诊断和历史实验，不能照抄示例中不存在的字段。`conditions` 用于召回具有相同显式条件的知识卡，是附加召回通道，不是硬过滤，也不代表已满足。没有合理精确条件时用空数组。keywords 可以为空。

`phase=plan` 时，主机已经完成 BM25＋语义向量＋可选条件通道的 RRF 融合，并补齐显式关联反例。先阅读所有 `knowledge`，再编辑 `candidate.py` 并写 `design.json`；基础字段为 hypothesis、change、falsification、controls、budget_rationale。

增强模式必须提供 `knowledge_assessments`，覆盖每条给定知识卡（包括追加反例）：

- `knowledge_id`：真实知识 ID。
- `status`：applicable / not_applicable / unknown；`rationale`：判断原因。
- `evidence_refs`：事实键、`trial:<id>` 或 `knowledge:<id>`。确定判断必须有引用；引用合法不代表机制已验证。
- `condition_checks`：逐字保留主机检查结果；不能改写 unknown。
- `counterexample_checks`：按原顺序逐条保留卡内 `counterexamples`，每项包含 `counterexample` 原文，以及 status、rationale、evidence_refs。
- `linked_counterexample_checks`：按原顺序覆盖 `counterexample_ids`，每项包含 knowledge_id，以及 status、rationale、evidence_refs。判断这条反证是否约束当前方案。

再填写 `diffs`，逐条引用 knowledge_id 和 current_refs，原样保留 condition_checks，说明 alternatives、missing_evidence、experiment、priority_reason，且只有一条 selected=true。如果选中经验的总体适用性为 unknown 或 not_applicable，必须提供 `exploration_rationale`，解释为什么仍值得做补证据或边界实验。相似度不是收益估计，不能据此声称方法有效。

`phase=analyze` 时只写 `analysis.json`：effect、alternatives、next_action、evidence_trials。evidence_trials 使用请求中的真实 trial ID，包含最新一轮。效果与机制解释分开，不能将无收益当成机制已被证伪。

## 启动前提

实际研究必须具有冻结的任务配置，包含数据身份、模型起点、可编辑文件、固定输入/目标构造、候选池、评分器、训练/推理命令、依赖、种子、资源额度及保留规则。缺失时输出具体前置问题，不能自造命令、标签、分数或预算继续。

使用专属实验目录。主机先建立未修改基线，记录实际执行、排名指标和成本；未经校准不预设五分钟或固定轮数足以产生有效结果。人的交互与个人知识写入均未实现，不依赖人工审核推进。

## 每轮读取

- 当前 incumbent 的实际源码、配置和已有结果。
- 任务要求、特征和数据说明，以及允许修改的范围。
- 相关公开经验、可选个人只读知识和 Agent 场景经验。
- 本地实验账本、失败/无收益/矛盾记录及上轮待解问题。
- 剩余预算和已冻结的检查、训练、诊断、评分方式。

来源不存在、内容未读取、经验适用性未知时明确标注。不得把论文中的方法当成当前源码已经具备的组件。

## 编辑前做 diff 和诊断

1. 整理当前模型画像。区分代码事实、程序测量和推测，为事实引用源码或产物。
2. 找出少量候选差距。逐条对照经验的适用条件与当前状态，标注满足、不满足、未知或冲突。
3. 列出对应现象、竞争解释和历史反证。方法未采用不等于当前瓶颈。
4. 若已有工具能以合理成本区分解释，先补测量；若缺少任务/评分前提，先解决前提。
5. 按证据、适用性、区分能力和成本，选择一个主假说或补证据实验；记录其他方向暂缓的理由。
6. 冻结假说、具体改动、预期现象、证伪条件、控制变量、比较对象和预算。可以做探索，不得伪装成已经定位原因。

将设计和测量写入实验记录后再编辑。计划改变时另记修订，不覆盖原始假说。

## 编辑与实验

直接编辑明确允许的模型、训练目标、训练采样或推理实现。固定任务数据切分、评价查询/标签、候选池和评分不能因候选结果改变。训练采样可以研究，评价候选不能随之缩小。

检查真实代码差异、接口、训练/推理一致性和对照实现。检查失败先修复，不把错误候选当成无收益实验。使用已配置的执行器训练、检索并由固定评分器独立评价，保留日志、实际配置、代码、权重或预测等产物。

候选和机制对照各自记录父版本与初始化方式。未经声明不得 warm start，不能把额外训练或更多调参预算解释为方法收益。所有失败、修复、测量、对照和复验都计费。

## 结果、归因和下一轮

先追加实验事实，再解释原因。按冻结规则决定 keep/discard；崩溃/超时的分数留空，记录原因和成本。未确认候选不自动替换经确认的 incumbent。回退只切换专属实验的候选状态，不覆盖共享目录的用户改动。

把本轮和可比较历史实验对照，回答：

- 任务效果怎样改变，代价是什么？
- 哪些解释得到支持，哪些仍不确定？
- 对照是否真正隔离了宣称因素？
- 结果是否受训练动态、样本量、随机性或检索影响？
- 下一轮最值得补哪项证据，或应该转向什么方向？

训练成功、指标上涨或引用合法均不单独证明机制。完整保留负例和无收益；跨模型/协议经验需要重新检查适用性，相似谱系不视为独立重复。

下一轮直接读取最新实验，不等待长期经验更新。研究阶段结束时归并带证据的“场景＋方法＋收益＋原因”经验，保留反例与适用边界。Paper 原文和个人知识保持独立权限。

## 停止与当前不实现的能力

预算用尽、达到冻结的结束条件或用户中止时结束，保存状态和阶段总结。最终留出结果不反馈给本轮研究。

运行中保持本指令、知识组织、检索规则、诊断工具和流程固定。不开发 RSI、诊断工具自动演化、搜索策略自改或个人 taste 学习。一个编码 Agent 可承担上述研究职责，不要求新增多 Agent 调度服务。
