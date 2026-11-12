# 腾讯广告赛题 autoresearch

实现 Karpathy 式单编码 Agent 循环：编辑候选、真实训练、全候选检索、独立评分、保留/回退。基础模式先跑通任务；`enhanced: true` 增加知识检索、适用条件对照、diff、反馈诊断及实验后分析。

这是可信本地研究运行器。Agent 请求不包含 gate/final 标签或分数；本版没有容器隔离，不能用来防御同一用户下的恶意 Agent 或候选代码。最终留出分数在研究关闭后才产生。

## 安装和任务准备

在仓库根目录，使用 Python 3.10+：

```sh
uv sync --extra taac
uv run --extra taac python -m recevobench.taac --help
```

准备官方 TencentGR-1M 本地发布文件：`seq/`、`item_feat/`、`user_feat/`、`candidate/`、`indexer.pkl`、`mm_emb/emb_81_32_parquet/`。代码不自动下载全量数据。

编辑 [prepare.example.json](prepare.example.json)，填写绝对路径、准确 release 和可信官方 `indexer.pkl` 的 SHA256。pickle 有可执行语义，只读取自己核验的发布文件；摘要用于固定同一文件，不替代来源核验。

```sh
uv run --extra taac python -m recevobench.taac prepare \
  --config examples/recommendation_autoresearch/prepare.example.json \
  --output /absolute/path/prepared-taac
uv run --extra taac python -m recevobench.taac verify --data /absolute/path/prepared-taac
```

适配明确区分历史 RID、候选 OID、retrieval ID 和多模态 OID；处理 `cold_start/feature_value`，保留 ID/稀疏属性/用户数组/多模态特征。历史特征字典同时提供 `history_action` 和 `history_timestamp`，供候选编码器使用；不提供目标行为或未来时间。生产数据要求完整 660000 候选。

训练保留公开代码的 next-event 监督定义；它不是额外声称的纯点击监督。每用户因果训练，用户按固定哈希分为 train/feedback/gate/final。留出评价采用本地保守重建：切点后第一个点击必须有唯一可归属且位于切点后的曝光，输入严格早于该曝光；缺失、歧义及不在候选池内的目标排除并计数。先确认这个协议的覆盖和业务含义，再冻结真实验收。它不是原赛事隐藏标签或榜单复现。

## 基础研究循环

编辑 [study.example.json](study.example.json)。相对路径基于配置文件所在目录解析。`limits` 和 `min_delta` 是示例值，服务器基线校准后再确定；`max_trials` 包含基线，训练、诊断、Agent 调用、失败与对照均消耗总墙钟预算。独立初始化每个候选和 seed；不会沿用父代权重。

```sh
uv run --extra taac python -m recevobench.taac init \
  --config examples/recommendation_autoresearch/study.example.json \
  --output /absolute/path/taac-study
uv run --extra taac python -m recevobench.taac baseline --study /absolute/path/taac-study
uv run --extra taac python -m recevobench.taac context --study /absolute/path/taac-study
```

未配置 Agent 时可以先手工验证执行链：复制 incumbent 的 `candidate.py`，修改模型或字面量 `SETTINGS`，编写设计，再提交：

```sh
uv run --extra taac python -m recevobench.taac trial --study /absolute/path/taac-study \
  --candidate /absolute/path/candidate.py --design /absolute/path/design.json
```

最少设计内容：

```json
{
  "hypothesis": "明确可检验的假说",
  "change": "相对父版本的具体改动",
  "falsification": "什么结果将不支持这项假说",
  "controls": ["同一模型、数据、种子与训练设置；需要时另提交对照候选"],
  "budget_rationale": "预估训练与必要复验为何值得"
}
```

主分数固定为 `0.31*HitRate@10 + 0.69*NDCG@10`，固定 runner 精确遍历完整候选池并输出十个 OID，独立评分器校验用户集合、重复和候选 ID。未超过 incumbent 加 `min_delta` 则 discard；崩溃/超时分数为空。keep 是开发选择，机制解释仍待验证。原版“同分但更简单”的裁量未加入自动保留。

## 接入编码 Agent

将配置中的 `agent` 设为执行本机编码 Agent 的 argv 数组，例如：

```json
{"argv": ["/absolute/path/your-agent-wrapper", "{request}"]}
```

这是进程协议示例，不是内置模型服务。wrapper 在当前工作目录读取 `request.json`、`program.md` 和 `candidate.py`，然后用自己已配置的编码 Agent 完成文件编辑。`{workspace}`、`{request}` 支持按参数替换，不通过 shell 展开，不在配置中写密钥。只读取 stdout 不能代替产出文件。

- `phase=query`：增强模式先写 `retrieval-plan.json`，动态生成问题、精确术语、条件与证据引用；至少含一个反证问题。
- `phase=plan`：编辑 `candidate.py` 并写 `design.json`，不自行启动训练。
- `phase=analyze`：增强模式下写 `analysis.json`，包含 `effect`、`alternatives`、`next_action`、`evidence_trials`；必须引用本轮真实 trial ID。

```sh
uv run --extra taac python -m recevobench.taac run --study /absolute/path/taac-study --rounds 2
uv run --extra taac python -m recevobench.taac status --study /absolute/path/taac-study
```

`run` 在尚无试验时先建立基线，然后执行指定数量的候选轮次。每轮输入保留历史假说、结果、失败与分析。候选快照独立保存，discard 切回 incumbent，不执行共享工作区 Git reset。中断后用 `recover` 结算未完成工作：有终态回执则不重跑；否则保留全额预约预算并记录 interrupted。

## 知识、diff 与诊断

在初始化前设置 `enhanced: true`，指定 [knowledge.jsonl](knowledge.jsonl)，可选提供 `personal` 只读 JSON，以及 `agent_memory` 指向前一研究阶段的 `stage-memory.json`。旧阶段正负结果保留为待重新核验的经验，不继承机制结论。运行时冻结这些输入；不实现个人审核或 taste 学习。

每条经验包含场景、方法、收益、原因、来源、条件与反例。来源仍标记外部待验证，不把搜索引用当成正文核验。程序按固定字段比较 `eq/ne/lt/le/gt/ge`；缺少字段返回 unknown。Agent 必须保留这些状态，不能把未知改成满足。

增强模式另需安装语义编码依赖并准备本地 `intfloat/multilingual-e5-small` 模型目录（config、tokenizer 和 safetensors）：

```sh
uv sync --extra taac --extra taac-semantic
```

在 `retrieval.encoder.model_path` 中填写本地路径。运行时不联网下载；模型文件、编码配置、依赖版本在初始化时冻结，缺失或失败直接报错，不退化为关键词检索。基础模式无需配置编码器。

每轮执行：`query → 混合召回 → 关联反例补齐 → plan（适用性判断＋diff＋代码修改）→ 训练/评分/诊断 → analyze`。query 读取当前模型、诊断及全部历史分析，包括失败和 discard；下一轮重新生成。本轮历史也转换为 `trial-<id>` 经验卡参与混合召回，保留状态和 provisional 解释；不会因 discard 删除。BM25 使用英文词及中文字符/二元词，语义通道使用 E5 前缀、掩码均值池化和归一化余弦；各 query 的通道以 RRF（常数 60）融合。明确条件匹配是可选附加召回通道，条件未知/不满足不会硬过滤。首轮默认 top_k=4，不代表经过效果调优。

知识卡可带 `counterexample_ids` 指向同一语料中的其他卡。主机从 top-k 开始补齐显式关联，处理循环并去重；这些反例不受 top-k 截断，因此最终上下文可能超过 k。缺失关联 ID 会报错。卡内 `counterexamples` 原文始终保留。示例中 ranking-negatives 关联 feature-fusion，表达“输入问题也可能解释排序差”，是竞争解释而非已观测的负实验。没有 GraphRAG、关系自动推断或图数据库。

Agent 必须用 `knowledge_assessments` 逐条覆盖检索结果与关联反例，再生成 diff。结构与约束见 [program.md](program.md)。主机校验覆盖、引用和条件一致性；科学解释是否正确仍需实验验证。未知/不适用但值得探索时需明确探索理由。

`context` 是只读查看，不调用 Agent。增强模式手工提交前，运行下面命令生成当前证据对应的检索上下文（需要已配置 Agent）：

```sh
uv run --extra taac --extra taac-semantic python -m recevobench.taac research-context \
  --study /absolute/path/taac-study
```

训练或分析改变历史后，旧检索上下文自动失效，必须重新生成；自动 `run` 会处理。query、检索和失败均计入总预算；`limits.retrieval_seconds` 独立限制检索进程，未配置时沿用 agent_seconds。模型当前按每次检索加载，尚未加入常驻服务或向量缓存。

检索接口可以替换：默认 `backend=hybrid`；`backend=command` 接受 `argv` 和 `artifacts`（需要冻结摘要的本地实现/模型文件）。argv 支持 `{request}`、`{output}`、`{workspace}`，不经过 shell。输入 schema 为 `taac-retrieval-request-v1`，含 plan、cards、model_state、config、evidence_digest；输出必须为：

```json
{"schema":"taac-retrieval-ranking-v1","request_digest":"输入 JSON 按 common.digest 计算的摘要","ranking":[{"id":"ranking-negatives","score":1.0,"trace":[]}]}
```

排名须按有限分数降序，ID 不得重复或超出输入语料。后端只输出排名；主机始终从原始卡取回正文、检查条件并补齐关联反例，因此替换后端不会绕过这些约束。自定义命令环境由调用方准备，artifacts 应完整列出所依赖的本地文件；未声明的外部服务版本无法由主机冻结。进程内扩展可实现 `RetrievalBackend.rank(cards, plan)` 或 `Encoder.encode(texts, query=...)`。

增强设计在基础字段之外增加 `knowledge_assessments`（必填，格式见 program.md）和 `diffs`：

```json
{
  "diffs": [{
    "knowledge_id": "ranking-negatives",
    "current_refs": ["model.negatives", "task.candidate_count"],
    "condition_checks": ["原样复制 request 中该卡的 condition_checks 对象数组"],
    "alternatives": ["额外训练、随机性或检索问题"],
    "missing_evidence": ["需要区分负样本数量和目标形式"],
    "experiment": "本轮最小实验或补测量",
    "priority_reason": "为什么本轮选它，哪些方向暂缓",
    "selected": true
  }]
}
```

这里 `condition_checks` 的字符串是说明占位，实际必须复制上下文中的对象，不能直接提交这个片段。允许条件不满足时做探索，但必须解释原因。

每次成功训练后，固定诊断在首个声明 seed 上选取最多 128 个确定性反馈查询，测量目标全候选排名、与随机非目标广告的分差/胜率及历史长度组样本量；候选池不缩小。诊断是描述性证据，未观察广告不是已确认负反馈。当前检索为精确检索，尚未实现 ANN，因此没有 ANN/精确差异诊断。

## 阶段结束与验收

```sh
uv run --extra taac python -m recevobench.taac finalize --study /absolute/path/taac-study
uv run --extra taac python -m recevobench.taac consolidate --study /absolute/path/taac-study
```

`finalize` 先关闭研究并冻结 feedback 选出的模型，再对基线与所选模型做 gate/final 留出评价；gate 当前用于结束审计，不参与逐轮确认。此后禁止继续研究，失败也不自动重试隐藏评价。`consolidate` 输出阶段经验，保留负例、父版本、来源、成本与 provisional 解释；当前是可追溯经验集合，没有自动确认因果或跨场景泛化。

检索证据保存在 `agents/*-query/retrieval-plan.json`、`retrieval/*/request.json`、`ranking.json`、`context.json`；每个 trial 的 context.json 也保存当轮计划、排名及完整证据。analyze 读取该实验的原始检索结果，下一轮 query 再读取更新后的事实。

结果保存在 `state.json`、`results.tsv`、`trials/<id>/`、`agents/`、`final.json` 和 `stage-memory.json`。训练产生实际 checkpoint，评分由真实预测计算。不会以脚本 Agent 的输出代替模型训练。

本地集成测试：

```sh
uv run --extra taac --extra taac-semantic --extra dev python -m pytest tests/taac -q
```

测试使用明确标注的原生 Parquet 合成数据和脚本 Agent，实际执行 PyTorch CPU 训练、检索和独立评分，验证连续两轮及失败恢复；不调用付费模型，不构成真实腾讯数据/GPU 或真实 LLM 研究效用验收。

## 来源

研究组织参考 [Karpathy autoresearch](https://github.com/karpathy/autoresearch) 的单 Agent 循环；腾讯模型起点取自官方 baseline commit `c4481fb40b2583687a7040939ee1a63f035028df`。模型源及其修改说明见 [PROVENANCE.json](baseline/PROVENANCE.json)。该上游模型使用 **CC BY-NC 4.0**，完整许可保留在 [LICENSE.upstream](baseline/LICENSE.upstream)，不由本仓库的 MIT 标识替代。
