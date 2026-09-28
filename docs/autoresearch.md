# 单入口自动研究

`autoresearch` 负责创建或恢复冻结 campaign，连续运行 evidence → hypothesis → experiment → 主机训练/评分 → evaluation → lead → knowledge，直到轮次或预算停止。角色直接调用已配置的 Codex CLI。模型有无提升由实际结果决定，运行完成不等于研究假设成立。

## 小型示例

准备 Codex jail 和 CPU runtime recipe 后，配置以下环境变量（准备方式见 [运行指南](runtime.md) 的隔离执行章节）：

```bash
export REC_HARNESS_CODEX_JAIL=/path/to/prepared-codex-jail
export REC_HARNESS_AGENT_RUNTIME=/path/to/private-agent-runtimes
export REC_HARNESS_CPU_RECIPE=/path/to/cpu-recipe.json
export REC_HARNESS_CPU_PYTHON=/path/to/torch-python
export REC_HARNESS_CPU_UIDS=/path/to/uid-registry
python3 -m rec_harness autoresearch /path/to/campaign --config examples/toy/autoresearch.json
# 中断后恢复或读取已停止的结果：
python3 -m rec_harness autoresearch /path/to/campaign
```

隔离启动器需要有 chroot、降权、设备节点创建权限的 Linux 执行环境。CPU recipe 声明解释器与公开依赖，不应包含宿主数据或研究归档。模型进程收到训练数据和验证特征；验证标签由主机保存，评分由主机计算。每个 seed 在两个新进程中训练，比较可复现性。序列化系数数量仅限制返回的系数，不代表任意模型的完整内部复杂度。

示例冻结 24 BCU、最多 3 轮。基线成本 2，每轮 6 个角色加 2 个 seed，并为知识修复预留 1 BCU，通常完成 2 轮后自动停机。BCU 是配置的工作成本，不是美元或 GPU 秒。调用超时、非法提案和失败训练保留成本与证据；知识格式错误可自动修复一次。控制器中断的未完成通用轮次保留为失败轨迹，随后开始新轮；不会把未完成实验冒充成功。`max_rounds` 是整个 campaign 的上限。

隔离小型任务可通过 `protocol.synthetic_task` 冻结任务类型、样本规模及噪声，支持双线性、分段和周期交互；省略时保留原任务。配置与边界见[合成任务合同](implementation/synthetic-task-contract.md)。

## 显式研究 Skill

可在通用模型配置的 `codex.role_skills` 中按角色提供冻结指令；Amazon 和工具/Harness 演化使用 `agent_backend.role_skills`。前缀验证套件也使用其 `agent_backend.role_skills`。例如：

```json
{
  "role_skills": {
    "hypothesis": [
      {"name": "competing-explanations", "content": "List competing explanations and a discriminating experiment before proposing the change."}
    ]
  }
}
```

这里只接受显式正文，不读取主机路径或自动展开 Skill 的附属文件。名称在同一角色内唯一，版本由正文内容 hash 标识；正文随 campaign 配置冻结，并完整进入实际提示词和请求归档。它们是从属于冻结协议的研究指令，不能改变评分、权限和主机准入规则。需要修改正文时创建新 campaign。

优化器轨迹核对请求、响应 schema、提示词与 Skill 内容的一致性，记录显式 Skill 版本。未配置时保留原指令收据和旧轨迹语义。此记录证明传入的内容，不证明模型逐条执行；Codex 自动发现的 Skill 和执行情况仍标为未知。提供的内容应是本任务允许发送的公开研究指令，不放入凭据或私有数据。

## Amazon NextItNet

该适配器在已初始化、冻结协议并有完整 incumbent 配对结果的 Amazon study 上运行。创建 study 和历史成本接续见 [运行指南](runtime.md)。

```bash
export REC_HARNESS_AMAZON_STUDY=/path/to/active-amazon-study
# 其余变量与上例相同；CPU recipe 用于独立 checkpoint 复评分。
python3 -m rec_harness autoresearch /path/to/amazon-campaign --config examples/amazon/autoresearch.json
python3 -m rec_harness autoresearch /path/to/amazon-campaign
```

控制器自动冻结研究提案、注册候选、检查模型源码、运行 CPU 数值契约及隔离的三步合成梯度检查，再调度完整开发集训练和独立 CPU 复评分。梯度检查要求学习后仍满足因果性、padding、尾部表示读出，以及 forward 与编码表示一致性；它不改变真实训练的优化器、协议或 checkpoint。只有通过开发集门槛并获 lead 保留建议才释放确认 seed。新任务只需先完成开发 seed 的 incumbent 对照；释放后，控制器按同一 execution_id 运行或复用对照确认训练并独立复评分，再运行候选确认阶段，不为淘汰的候选提前训练确认对照。确认结果只在主机侧判定晋升，不发送至 Codex。当前确认仍使用相同验证用户的新初始化，不是最终测试。

源码修改范围是 `NextItNetModel` / `NextItNetBlock` 内已声明方法，允许的张量算子和模块可在 `model_source_contract.py` 查看。训练、数据处理、导入和方法签名冻结。新增模型结构不需要修改逐候选字符串模板；超出能力范围的代码被拒绝。这个受限源码契约不是任意 Python 的安全证明。Amazon 配置可通过 `execution_id` 绑定已注册的冻结隔离运行环境；基线、候选和确认阶段必须使用相同执行版本。未指定时仍沿用 reviewed-source 路径。隔离路径已通过实际基线训练与独立复评分，不代表任意 GPU 代码或驱动层安全证明。

每轮复用已审计 incumbent 作为配对对照；可选 `control_model_ids` 冻结额外已注册的对照模型，例如 `{"control_model_ids":["legacy_no_prefix"]}`。这些模型必须已在同一 study 注册，并具有匹配当前 execution_id 的完整开发结果；在生成设计输入前，还须逐类别/seed 核对与 incumbent 相同的候选集合和数据哈希。控制器审计后才向研究角色提供其开发指标，并在评审前独立复评分。提案必须原样引用 incumbent 与所有指定对照。确认释放后，主机按需训练或复用全部对照的确认结果并独立评分；候选须在每个确认 seed 上超过 incumbent 的 minimum_effect，同时严格优于每个额外对照，才能晋升。

可选 `generated_control_count`（非负整数，默认 0）指定本轮新增机制对照数量。框架按 proposal ID 分配唯一模型 ID 和 `control-1.py` 等文件名；hypothesis 在提案中解释这些对照，experiment 在同一次 Codex 调用中输出完整候选模块 `model.py` 及所有对照模块。`changed_files` 必须严格匹配预先分配的文件列表；模型外代码保持冻结。底层 `admit_plan` 支持冻结提案中的新对照源码。它先对所有新源码做 AST 预检，再分别执行 CPU 数值和合成训练检查，全部成功后用一个事件注册全部新版本；任何失败均不留下部分注册，失败检查产物保留。已完成准入的恢复不重复执行检查。新对照与候选一起进入冻结 development/confirmation 矩阵，主机独立评分所有新版本；只有已注册对照走复用路径。角色恢复、完成事件及周期归档均绑定每份生成源码，拒绝事后篡改。评估和负责人还接收每份新对照自身的准入、源码合同与开发执行摘要（`generated_control_evidence`），并按不同源码身份与候选区分；这些检查不替代机制归因。 新准入记录另含 `structural_changes`：基线与候选 AST 身份、实际变化的方法，以及变化前后写入的模型字段。评估和负责人据此核对提案中的插入位置；每个对照使用自己的结构差异。注释和排版不构成结构变化，旧记录缺少此字段时不会补造历史检查。该证据不验证自然语言语义合同或完整计算图。模拟回归覆盖确认与中断恢复；[真实合成 E2E](experiments/generated-control-e2e-v2.md) 已完成六次 Codex 调用、候选及对照生成、三次训练/独立评分和零新增执行恢复。真实开发结果相同，正确保留基线，未进入确认；这不证明机制归因。既有对照结果属于此前已暴露证据；支持多对照执行不等于已验证某个机制。这不会增加默认角色调用数量，但会增加必要的训练/评分工作；所有成本仍保留。角色调用上限和 study GPU 秒分别计账；失败计费、历史成本接续及每个训练任务的预算检查保持生效。`minimum_round_gpu_seconds` 是启动下限，不是整轮成本预测；每个提案仍受自己的冻结预算和 study 总预算限制。

主机确认晋升要求每个确认 seed 的类别平均增益都严格超过冻结的 `minimum_effect`。独立配对报告中的旧字段 `effect_gate_passed` 仍仅表示跨 seed 平均增益达标；新增 `per_seed_effect_gate_passed` 和 `all_seed_effect_gates_passed` 明确逐 seed 判断，摘要与 Markdown 报告保留这一差异。缺失配对时不生成这些门槛结论，旧报告未提供逐 seed 判断时显示“未提供”。配对报告本身不释放确认、不执行晋升，也不建立机制结论。

Amazon 角色有终态响应时恢复会复用，候选源码和上下文有哈希绑定；成功传输但不合法的响应最多尝试两次；传输失败立即停止，普通恢复不自动重发。已完成训练由 study 的审计与恢复机制复用；存在冻结计划的 CPU 复评分按原收据恢复，不移走目录后重复推理。失败轮写入 `failure.json` 并继续下一轮。已选中的 candidate 成为下一轮 incumbent。下一轮同时获得此前设计、开发集决策与 provisional knowledge；确认结果和确认阶段失败详情始终保留在主机侧。

按轮次检查角色提交、完成、拒绝、待完成和复用数量：

```bash
python3 -m rec_harness amazon-call-report CAMPAIGN --format markdown
# 默认 JSON 还包含逐次提交时间、账本状态和源事件头。
python3 -m rec_harness amazon-call-report CAMPAIGN
```

`submission_cadence` 另给出首末提交跨度、相邻提交的最短/中位间隔，以及任意闭区间 60 秒窗口内的最大提交数。它只统计本控制器记录的提交，不把继承费用或角色复用虚构为带时间的请求；不能据此判断服务端 API 限流或并发。少于两次提交时不推算间隔，时间戳缺失、缺时区或发生时钟倒退时报告 unknown。

报告使用只读 SQLite 快照核验事件与费用投影，可以在控制器运行期间读取，不恢复或重发任务。继承费用单列；扣费但未记录提交的中断列为 unattributed，不伪装成模型调用。角色提交发生在传输之前，可能在 CLI 启动前失败，所以提交次数不等于 CLI 启动数或内部 API 请求数。内部 API 请求数明确为 null。复用角色校验原完成事件，但不增加本轮调用；完整训练与继承来源审计仍使用 verify。若仅训练对照、尚未初始化角色控制器，报告会明确拒绝而不是返回零费用。该入口已包含在通过验收的 v8 安装包中，早先 v7 包不包含它。

开发记忆导出也以只读方式打开源控制器账本，并在同一 SQLite 快照中校验和读取事件；不会顺便修复中断的 events.jsonl 投影。确认已执行不妨碍导出已有开发研究，但私有确认文件和终态中的确认结果不进入记忆。真实 v2 周期的双次导出与原记忆逐字段一致，源事件头和投影哈希不变，证据为 `read-only-memory-acceptance-v1/acceptance.json`；该验收没有新增角色、训练或推理。

Amazon 也支持下述 `diagnostic_registry` 与 `measurement_requirements`。诊断只接收已审计开发 seed 的品类聚合指标，包含 dataset 标签；不传入原始交互、checkpoint 或确认结果。启用工具时须指定独立 `diagnostic_budget`，费用记录在 diagnostic-budget，不能混算为 Codex 调用次数。必需测量缺失会生成 MeasurementGap，以 measurement_required 停止，保留 incumbent；此时不会生成模型提案或启动训练。工具外循环的 measurement_source 可直接引用该 Amazon campaign。新研究回合可冻结晋升后的工具并复测；项目总调度可自动创建 Amazon 后继回合，继续使用同一冻结 study 和共享 GPU 账本。

工具计划显式选用 `public-observation-measurement-v2` 后，还可读取当前开发模型已提交的激活/block-score聚合观测；来源、范围、输入归档与恢复均受主机校验。旧合同输入不变，独立验证仍须覆盖新输入。见[公开观测工具合同](implementation/observation-tool-contract.md)。

小型在线验收可在协议中启用 `output_activation_observations: true`，由主机从已有预测收据生成当前模型的标量输出统计量，为v2工具提供逐轮观测；三组对照保持相同基础观测，额外工具调用单独计费。见[在线输出观测](implementation/online-output-observations.md)，该功能不代表Amazon隐藏层诊断在线效用已获验证。

`verify` 和 `recover` 已识别 Amazon campaign，无需初始化 Codex 服务。verify 核对控制器/绑定 study 事件链、角色内容与绑定、已提交开发周期及 CAS，并单列旧角色记录缺少完整事件绑定的情况；它不替代逐 GPU 作业或私有确认结果审计。恢复在控制器锁下检查角色和诊断工作进程，保留未知实际成本的预留费用。

可选 `history_diagnostic: true` 在假设生成前提供基线开发历史长度分布，并在候选开发完成后提供配对分桶结果。仅使用已审计开发 seed，事件/CAS 绑定输入并支持精确恢复；不会新增模型推理。具体字段边界和真实分布见 [开发历史诊断](experiments/amazon-baseline-history.md)。

## 独立最终评估

研究结束后，先封存比较模型和验证集选出的 checkpoint，再执行最终测试。第一个 `--model` 是报告中的 comparator。每个模型必须已有全部类别、全部确认初始化的可审计 checkpoint，且执行版本和类别词表一致。

```bash
python3 -m rec_harness amazon-final-seal /path/to/study /path/to/private-final \
  --model legacy_incumbent --model selected-candidate \
  --test-directory /path/to/test --eval-seed 20260926 --eval-users 1000
python3 -m rec_harness amazon-final-run /path/to/private-final \
  --recipe /path/to/cpu-recipe.json --uid-registry /path/to/final-uids
```

`amazon-final-seal` 不读取测试集；它冻结源码、checkpoint、评估代码和评分口径，并关闭该 study 的后续研究及预算接续。输出目录必须在 study 之外。`amazon-final-run` 首次打开 `<category>_test.txt`，用训练加验证历史预测第一条测试交互；固定采样至多 `eval_users` 个用户，每个用户使用一个正例和 99 个唯一负例。所有模型及初始化共用同一类别请求。缺失历史、未知物品或负例不足会拒绝，不能静默筛除用户。评分沿用严格大于正例分数的负例计数，并报告平分用户数。

私有用户与正例键保留在最终评估目录，隔离进程只收到模型、checkpoint、匿名历史和打乱的候选集。主机将每个初始化的完整类别均值与配对差值写入 `final-report.json`，结果不写回研究记忆，也不用于自动选择新模型。这是采样候选排名测试，不是全物品排名。恢复会重新验证数据、评分产物和报告，已完成推理不重跑；遗留的未完成推理目录需先审计，当前入口不会自动丢弃后重试。

已有两模型×两初始化、4 个真实隔离 CPU 进程的合成验收，精确恢复通过。模型和验证准入收据是合成夹具，因此该验收只证明最终执行链路，不证明真实模型收益或真实最终测试已完成。

## 工具与 Harness 外循环

工具的 correctness、synthetic、robustness 阶段可使用主机参考验证器：

```bash
python3 -m rec_harness.tool_validation --suite /path/to/reference-suite.json \
  --candidate /path/to/candidate --output /path/to/stage/result.json --stage correctness
```

suite 含 identity、tool_files（必须含 diagnostic.py）、budget、case_cost、timeout、CPU execution，以及 cases。每个 case 包含唯一 id、stage、input、expected 数值字典和 absolute_tolerance。参考值与容差始终留在主机，仅 input、候选源码和固定 worker 进入 CPU jail；主机核验输出格式并计算误差。每个已完成案例写入事件链，失败也保留，恢复不重跑案例以取得通过结果。修改 suite、源码或 worker 必须使用新验证目录。

报告的 measurement_accuracy 是通过案例比例；兼容验证阶梯的 gain 字段也表示此比例，不能解释为模型增益。online 固定为 false，因此不能作为 utility 阶段的在线证据。当前接口只支持 JSON 聚合测量，不覆盖原始张量、非有限 JSON 值或自动构造参考答案。影子与三组在线效用使用下文的独立验证器，不复用参考值准确率作为在线收益。

同一入口接受 `adapter: research-evolution-v1`。配置提供 `candidate_id`、`kind`（`tool` 或 `harness`）、公开 `context`、冻结 `plan` 和 `agent_backend`。工具可提供 `source` 与 `model_files`，初始化时归档；后续恢复不再读取源工作区。Codex 分别承担 tool 或 optimizer 角色。

新建工具演化任务默认通过 Codex 内置工具进行论文搜索，并在配置中冻结完整策略。可用 `agent_backend.web_search: {"tool":"disabled"}` 明确禁用，或设置 `cached`；旧冻结任务不改变默认行为。来源 URL 和版本/章节写入 `diagnostic.py` 模块说明，来源局限写入 `limitations`。生成请求、日志、搜索收据与进程收据在成功和失败时均进入事件/CAS 档案；归档与搜索发生不代表论文内容已经验证，详见[生成档案合同](implementation/evolution-generation-archive.md)。

`plan` 保留现有 evolve 验证合同：工具依次经过 correctness、synthetic、robustness、shadow、utility、cost_regression；Harness 依次经过 prefix_audit、development、confirmation。每阶段使用冻结的独立 validator 命令及验收阈值。内置 `-m rec_harness...` 验证器还冻结整个包的 Python 源码；外部模块须以阶段的 `validator_files` 列出本地依赖。验证前校验文件哈希，变更会在执行和扣费前被拒绝。这不替代解释器、第三方依赖与系统环境锁。验证程序由操作者提供，候选不能修改；声明 `online: true` 本身不能证明真实在线收益，必须由验证程序实际执行独立实验并保留证据。下述在线配对与三组工具验证器支持隔离 toy 任务；任意任务适配仍待扩展。

Harness 的开发/确认阶段已有隔离 toy 在线配对验证器。先创建规格文件：`cases` 列表中的每项含唯一 `id`、`stage`（development 或 confirmation）和模型研究 `config` 路径，另可指定 `baseline_policy`。任务必须使用 `isolated-toy-v1`、`jailed_codex` 和显式 `codex.model`，各 case 的数据/seed 条件不能重复，包括训练 seed 与另一 case 验证 seed 的重叠；改写 data_version 标签不会创建独立数据。验证器冻结模型及评估器源码，计算 base_llm/tools/start/graph/budget/protocol 配对身份。开发与确认使用同一冻结 suite 的不同条件。

```bash
python3 -m rec_harness.online_validation freeze /path/to/spec.json /path/to/suite
python3 -m rec_harness.online_validation run --suite /path/to/suite/suite.json \
  --candidate /path/to/candidate --output /path/to/development/result.json --stage development
```

候选目录须含已生成的 `policy.json`。将 run 命令填入对应验证阶段的 `command`，分别使用 `{candidate}`、`{output}` 占位符；`plan.identity` 使用冻结 suite 的 identity。每组执行完整研究角色与实际隔离训练，分别建立事件/记忆目录。报告的 gain 是配对终态 incumbent 分数差的平均值，regression 是最大退化，total_cost 汇总两组实际账本费用，阈值仍由外层冻结计划判定。`passed` 仅表示验证执行有效，不能单独作为晋升条件。测试注入 Agent 时强制 `online: false`。该实现覆盖单个合成任务族的独立数据/seed 条件，尚不支持 Amazon 在线 Harness 验证、多任务族或统计置信门槛。

Harness 外循环可配置 `trace_source`（campaign、可选 roles 和 cutoff_seq）代替手写 context。输入来自已提交角色调用：当时可见的上下文、调用前源码快照、检索节点、输出、费用和事件身份。新运行保留精确源码快照；旧档案缺少快照时明确标注，不从调用后的代码倒推。优化器输出的 trace_refs 必须引用所提供的轨迹。源 task_id 必须列入 plan.development_ids，从独立验证中排除。当前轨迹接口不会把后续实验结果补入原始决策输入。

```bash
python3 -m rec_harness autoresearch /path/to/evolution --config /path/to/evolution.json
python3 -m rec_harness autoresearch /path/to/evolution
python3 -m rec_harness verify /path/to/evolution
python3 -m rec_harness evolve /path/to/evolution rollback --kind harness
```

中断后复用已提交的实现和验证结果。已拒绝的科学验证为终态；修改方案需新 candidate。验证投影丢失可以由事件记录恢复，投影被改写则拒绝。发布幂等，首次发布也能回退至内置默认策略；正在运行的 campaign 始终使用自己的冻结版本。

新模型 campaign 可用 `harness_registry` 和匹配的 `harness_identity` 启用已验证策略。提示词补丁传入隔离 Codex；Amazon 也支持此入口，`retrieval_limit` 限制发送的历史、研究记忆及执行失败条目数量。

通用模型入口可配置 `diagnostic_registry` 对象，包含 `registry`、匹配的 `identity`、`execution`（CPU recipe、python、uid_registry）、`cost` 与 `timeout`。工具的验证计划须冻结 `tool_contract: public-aggregate-measurement-v1`，实现提供 `diagnostic.py` 的 `measure(request)`。输入仅为当前模型的公开逐 seed 聚合指标；输出严格为 `measurements` 数值字典与 `limitations` 字符串列表。新 campaign 复制已验证源码至自身 CAS，并在 evidence 角色之前通过 CPU jail 运行，输出与 ToolVersion 通过 MEASURED_BY 关系关联。相同输入和工具版本复用结果，失败保留为无效观测。工具不能通过返回 passed 或 selection 字段控制晋升。这一接口尚不支持原始激活张量；Amazon 的输入限定为开发集品类聚合。

通用模型配置可冻结 `measurement_requirements` 列表，每项包含 `measurement`、`blocked_decision`、至少两个 `competing_explanations`、`cheapest_test`、`development_budget` 和 `acceptance_contract`。缺少规定输出时，控制器将覆盖不足写为独立 MeasurementGap，关联暴露问题的 Experiment，并用 BLOCKS 关系阻止继续生成模型改造。轮次以 provisional 归档，campaign 以 `measurement_required` 停止；原 incumbent 保留，恢复不会重复调用。工具执行失败或基线数据无效只记为前置条件不足，不据此建立工具能力不足结论。已提供规定测量时正常进入模型研究；输出存在本身不代表已证明因果解释。

工具外循环可用 `measurement_source: {"campaign": "/path/to/model-campaign", "gap_id": "measurement-gap-..."}` 替代手写 `context`。初始化从已提交触发事件冻结 GapCard 及当时图版本中的公开实验聚合值；不携带整份归档、原始样本或未来结果。`plan.development_ids` 必须包含这些开发实验 ID，后续 shadow/utility 不能复用它们。候选通过验证后，新模型 campaign 通过 `diagnostic_registry` 冻结新工具并复测。项目总调度器已支持从模型停止状态创建后继 campaign；工具验证任务仍由冻结计划指定，尚未自动设计。

自主测量缺口的工具转交还保留已归档的审查输入（`measurement_review_input`），包括当时确实可见的激活/block-score 聚合及其限制；核对审查事件、内容身份和触发前图快照。开发证据不扩大工具运行时的数据合同，详见[转交证据](implementation/measurement-review-handoff.md)。

## 开发经验续接

```bash
python3 -m rec_harness amazon-memory-export /path/to/amazon-campaign /path/to/development-memory.json
```

下一 Amazon 配置通过 `development_memory_files` 导入。导出包括设计、开发集读数、开发集决策、临时知识和非确认阶段执行失败；不导出确认结果。导入核验协议、包和读数哈希、开发 seed 范围。这是经验传递接口；跨任务迁移收益仍需独立实验。

通用模型入口使用 `initial_memory_files`。初始化将经验证的记忆包复制至当前 CAS，在第一次模型评估前导入独立目标图；来源结论保留为待目标验证的类比。重复恢复不重复导入，也不把来源证据当作目标实验。

普通通用图导出为 `memory-policy-v3`；包含方法来源元数据或 Method→Source 关系时导出 `memory-policy-v4`，读取端兼容 v2/v3/v4。v4 保留原始引用及未核验状态，不将来源升级为目标证据。v3 将公开 Proposal
节点及 `USES` 关系纳入证据闭包，避免实际研究中 Claim 引用 Proposal、Proposal
连接 Gap/Method 时被整组排除。字段白名单仍排除源码、权重和原始用户数据；
目标图中的 Proposal 降为 Source，不能直接执行或作为已证实的目标证据。
容量不足时仍整组舍弃支持与反证，不能单独保留成功结论。

## 项目总调度

新建自主工具任务和项目默认冻结 `"tool_method_contract": "diagnostic-method-v1"`，要求生成角色提交独立方法卡，区分旧工具优化、论文适配和自主设计，并将测量定义与来源对应关系归档到注册信息。字段、兼容性和验证边界见[工具方法卡](implementation/tool-method-card.md)。未启用的既有冻结计划不追写卡片，旧项目尚未创建的子任务也沿用父项目合同。低层 Evolution API 保留显式计划行为。

工具配置还可声明 `tool_history_sources: [{"campaign": "../old-tool-campaign"}]`，冻结已有和失败工具的方法卡及验证状态，供作者先比较历史再选择方法。历史评价条件自动加入新验证的排除集合；不会传入分数或验证样本。截止点、候选筛选和缺失记录处理见[工具历史输入](implementation/tool-history.md)。

这是模型、工具和 Harness 三个循环的联合入口。`examples/toy/autoresearch.json` 只启动模型循环；完整项目需要另外提供工具和 Harness 的冻结验证配置。不会用固定返回 `passed` 的演示验证器替代真实验证。

项目配置结构如下；路径指向已准备好的子配置，工具运行环境与小型示例相同：

```json
{
  "adapter": "research-project-v1",
  "task_id": "my-research-project",
  "max_model_episodes": 3,
  "model_config": "model.json",
  "tool_config": "tool.json",
  "harness_config": "harness.json",
  "max_tool_updates": 1,
  "max_harness_updates": 1,
  "tool_runtime": {
    "execution": {
      "recipe_file": "${REC_HARNESS_CPU_RECIPE}",
      "python": "${REC_HARNESS_CPU_PYTHON}",
      "uid_registry": "${REC_HARNESS_CPU_UIDS}"
    },
    "cost": 1,
    "timeout": 120
  }
}
```

`model.json` 可使用前述小型示例配置；其源码路径按该文件所在目录解析。 通过 `model_config`、`tool_config`、`harness_config` 引入的子配置各自解析其目录下的 agent_backend.template/runtime_root、UID registry、validator_files、直接 Python 验证脚本及内置验证命令的 `--suite` 路径。命令占位符与自定义验证器普通参数不改写；裸可执行程序名仍按 PATH 查找。`validation_preflight` CLI 对演化配置采用相同解析规则，对独立 plan 使用 plan 文件目录。已冻结套件内部绑定的绝对路径不随搬迁重写，搬迁这些依赖仍须重新冻结套件。`tool.json`、`harness.json` 使用上一节的 `research-evolution-v1` 合同，分别配置六阶段工具验证与三阶段 Harness 验证。工具阶段可接入下文的 `tool_pipeline`，Harness 阶段使用前缀和独立在线验证。三轮是最大模型回合数，真实分支由缺口和验证结果决定，不保证每次都触发或晋升工具及 Harness。

新项目还会对启用的工具/Harness 验证槽执行只读预检：内置验证器核对冻结套件、绑定输入与计划身份，工具在线套件核对当前 diagnostic worker。缺失或不兼容时在创建项目和调用角色之前拒绝。禁用槽不要求其依赖就绪。可单独检查演化配置或 plan JSON：

```bash
python3 -m rec_harness.validation_preflight /path/to/evolution.json
```

预检通过只代表已检查项在检查时一致，不执行候选，不证明主机权限、服务连通性、任意工具语义或科学效用。自定义验证器只做文件依赖检查；验证阶段仍须执行原有完整门槛。旧冻结套件不自动升级，工作进程不兼容时应创建新套件。

新项目在创建目录和发起研究调用前检查工具执行设置：可激活的工具槽必须提供 `tool_runtime`，费用须为非负有限数，超时须为正有限数，执行设置须含 recipe、解释器和 UID registry。这是配置检查；实际主机权限、recipe 内容和进程执行仍由隔离运行器验证。恢复旧冻结项目不修改其配置。

`adapter: research-project-v1` 将模型、工具和 Harness 的入口连接为连续项目。配置含 `task_id`、`max_model_episodes`、`model_config`，以及可选 `tool_config`、`harness_config`（均是配置文件路径）。`max_tool_updates`、`max_harness_updates` 默认各为 1；工具更新还需 `tool_runtime`，其字段为 execution、cost、timeout，与 diagnostic_registry 的执行配置一致。两者的 execution 都可用 recipe_file 读取 JSON；相对路径以所属配置文件目录为准，不得同时提供 recipe 和 recipe_file。各子配置继续使用前述冻结协议和验证计划，项目调度器不能降低验收门槛。

```bash
python3 -m rec_harness autoresearch /path/to/project --config /path/to/project.json
python3 -m rec_harness autoresearch /path/to/project
python3 -m rec_harness verify /path/to/project
```

工具计划可声明 `measurement_scope.supported_outputs`，例如 `{"metric_spread":"maximum minus minimum over valid rows"}`。新项目默认冻结 `enforce_tool_measurement_scope: true`：自主或优化器发现的测量名称不在声明范围时，以 `measurement_contract_unsupported` 停止并保留缺口，不调用工具生成。这只检查名称，不代替定义和效用验证。旧冻结项目缺少该字段时保持原逻辑；未声明范围的计划沿用原验证路径，不推断为任意测量均可支持。详见[工具范围准入](implementation/tool-measurement-scope.md)。

模型回合因测量覆盖不足停止时，调度器从已提交 MeasurementGap 创建工具候选，并将触发实验加入开发排除列表。工具通过全部阶段后才注册到新模型回合，保留测量要求并重新测量；失败候选使项目以 tool_rejected 停止。正常模型回合结束后，如配置了 Harness 优化，则从真实决策轨迹生成候选，通过验证后在新回合启用。旧回合不热更新。

新项目默认 `continue_after_harness_rejection: true`：Harness 的合法独立报告未过数值门槛时，消耗一次 `max_harness_updates` 额度，沿用原有策略进入下一模型回合。执行或报告合同失败仍停止；工具缺口也不会被忽略。设为 `false` 可遇拒即停，缺少该字段的旧冻结项目保持旧行为。详见[淘汰后的续接与恢复](implementation/harness-rejection-continuation.md)。

后继回合继承当前 incumbent 源码、原冻结评估器，以及投影后的来源经验包，使用独立目标图。每次子回合结果与状态迁移一并提交到项目事件链；中断发生在子回合完成后、项目提交前时，恢复复用子回合结果。费用保留在各子 campaign 的账本中，外层验证成本与其内部验证回合不能重复求和。项目主目录的预算账本不代表子回合总成本。

新项目还默认冻结三个连续演化设置：`measurement_lineage: true` 保留原测量缺口 ID，并把后继实际输出关联到已提交诊断；`inherit_tool_source: true` 让第二次及后续工具开发继承当前已验证源码；`inherit_harness_policy: true` 将最小 Harness 补丁应用到当前策略，保留未修改字段与其他角色提示词。旧项目缺少这些字段时仍按原冻结配置恢复，不能通过修改 project.json 就地改变历史语义。

同时配置工具和 Harness 时，两者依次在模型回合边界生效。例如模型→工具→模型→Harness→模型，最后一个模型回合同时保留新工具、测量要求和新策略，并继承上一模型的 incumbent。该五阶段组合已通过集成测试，包括 Harness 晋升提交后中断、恢复不重复工具/优化器调用、末轮实际复测及五个子回合完整性检查。它使用脚本角色和固定验证报告，证明联合调度与恢复；真实三循环收益仍须独立实验。

新项目默认启用 `optimizer_measurement_handoff: true`，在原优化器调用位置使用一次 `optimizer_router` 调用，选择增量 Harness 补丁或测量缺口审查；不会无条件追加调用。主机接受测量转交后，跳过 Harness 验证/晋升，进入独立工具候选验证及模型复测。无可用工具槽或额度则保留缺口并停止；数据、调用等前置条件不足则拒绝转交。独立演化配置可用 `optimizer_handoff_budget` 开启此选择合同，工具配置通过 `optimizer_measurement_source` 引用已提交转交的 Harness campaign。详见[转交合同与恢复](implementation/optimizer-measurement-handoff.md)。 优化器转交保留引用轨迹中当时可见的公开激活/block-score 观测及限制，拒绝重复提议已有观测量；这些开发证据不扩大工具执行时的数据合同。

当前总调度支持通用模型及 Amazon 适配器。Amazon 后继回合继承 incumbent ID 和经过验证的开发记忆包；study 配置与初始记忆包在项目创建时冻结，后续 GPU 成本继续进入同一 study 账本。Amazon 新角色调用也保留调用前上下文/源码快照及完成事件，可直接作为 Harness trace_source；旧档案没有这些绑定时不会倒推补造。工具验证任务的自动设计、任意未预声明测量缺口发现、多任务族在线验证，以及三循环联动的真实收益验收尚未完成。控制器测试中的固定阶段报告仅验证交接与恢复，不是在线收益证据。

## 结果与边界

`run-report.json` 包含终态、停止原因、预算和事件链校验；`state.json` 记录轮次和 incumbent。每轮保留源代码、角色请求/响应、失败及独立评估证据。恢复已达到停止条件的 campaign 不额外调用角色或训练。要改变冻结配置、追加轮次或预算，创建新 campaign；Amazon GPU 预算仍以底层 study 的授权接续为准。

查询当前工作流状态使用 `python3 -m rec_harness status /path/to/campaign`。命令以只读事务读取账本，不恢复任务、不重写事件导出。Amazon 轮次及项目阶段优先采用已提交状态事件；状态文件缺失或与事件快照不同会显式报告。既有 Amazon 控制器在最后一个轮次事件后发布的 stopped/stop_reason 保留为明确标注的文件层终态，不当作事件事实。`process_status: not_inspected` 表示该命令没有核验进程存活；不能仅据 ready/stopped 文件判断是否应重启。通用旧控制器没有适配的状态事件时仍返回 state.json，并标记 not_event_bound。

Amazon 角色传输故障会以 `role_transport_failure` 停止整个 campaign，重复 `autoresearch` 不会通过新一轮重复调用故障服务。若失败尚未进入准入，修复主机/服务后使用 `amazon-retry-source <campaign> --reason "具体修复"` 再恢复；重开事件绑定原文件，首次恢复时将失败角色移入 `*-transport-archive-*`，保留已成功角色及全部费用，归档前拒绝文件变更。准入后的评审故障仍使用 `amazon-review-development` 的既有专门恢复流程，不允许回退重做来源设计。项目子 campaign 遇到此故障时不会转入工具/Harness；先显式修复重开子 campaign，再恢复原项目。项目 `status` 的 attention 给出待处理子目录，状态查询不执行修复。

本入口实现顺序模型研究闭环、预声明测量覆盖要求触发、工具/Harness 候选生命周期与通用/Amazon 项目总调度。自主发现缺口的真实正例和在线收益、自主工具候选的多条件在线收益验收、三循环真实联动收益验收、经验迁移收益实验、论文引用的内容级验证、任意任务适配和多 worker 分布式调度仍未完成。

### 工具影子验证与三组在线效用

`python3 -m rec_harness.tool_shadow freeze SPEC DIRECTORY` 冻结 Amazon 开发聚合条件；
`python3 -m rec_harness.tool_shadow run --suite DIRECTORY/suite.json --candidate TOOL --output REPORT`
在每个条件上运行两个全新 CPU 隔离进程，检查预声明数值范围及重复性。至少两个不同源码/seed条件；确认seed、同条件别名和变更后的聚合输入被拒绝。该报告始终 `online:false`。

`python3 -m rec_harness.tool_online freeze SPEC DIRECTORY` 冻结三组研究实验。
SPEC 沿用 `online_validation` 的 `cases`（development/confirmation）和可选 `baseline_policy`，
增加 `existing_tool`（相对 SPEC 的目录）、`tool_files`（必须包含 `diagnostic.py`）及
`tool_runtime: {execution: {recipe, python, uid_registry}, cost, timeout}`。
所有研究配置必须指定明确的 `codex.model`、隔离 Codex 与隔离 toy 执行器。

运行 `python3 -m rec_harness.tool_online run --suite DIRECTORY/suite.json --candidate TOOL --output REPORT --stage utility`。
三组为 `no_tool`、`existing_tool`、`candidate_tool`，使用独立研究图、相同初始模型/评估器/策略/协议/总预算。
工具只能产生测量，最终 incumbent 分数由主机评估。工具成本占用所在研究预算；总成本包含三组全部运行。
收益取候选相对两个对照的平均增益中的较小值；回退量取所有条件和两个对照中的最大退步。
工具未执行或测量失败不能通过。候选工具仅在标记 `validation_only` 的研究中试用，不产生注册表发布。
`confirmation` 使用冻结的独立确认条件；与开发无交叠检查在套件冻结时执行。
报告适配 ToolEvolver 的 utility 阶段。注入测试 agent 的运行始终 `online:false`，不构成真实在线效用证据。
当前宿主支持一个合成任务族；已完成18次真实 Codex 调用的三组验收，增益为0，候选未通过门槛。完整自主工具演化及多条件收益验收仍待完成，详见验收记录。

`python3 -m rec_harness.tool_exposure CAMPAIGN OUTPUT` 只读审计工具结果的送达链。 支持通用模型与 Amazon 控制器，含测量审查的精简摘要；Amazon 同时检查 study 冻结身份及开发条件，见[适配器审计](implementation/amazon-tool-exposure.md)。
审计核对工具源码/worker CAS、收集的测量输出、事件先后、协议及当前 baseline 输入摘要，
并从提交时保存的角色可见输入证明结果已送达；单独标记该角色传输是否完成。
它不把工具送达等同于角色采纳、机制证据或决策收益。

工具成本验收：

```sh
python3 -m rec_harness.tool_cost --suite suite/suite.json --utility utility/result.json \
  --candidate candidate --output cost-regression/result.json \
  --development-ledger completed-development/memory
```

`--development-ledger` 可重复指定已结束、采用相同 BCU 单位的开发账本；没有自动发现或估算未记录成本。
验收绑定冻结套件的全部条件，核对各组初始源码、评估器、策略、工具版本及运行配置，并从事件重新计算所有预留/结算，与数据库投影、终止事件、incumbent分数、实际隔离Codex调用收据和测量送达链核对。
未知实际成本继续保留预留值。重复账本、尚在运行的预留、被改写的分数/成本和不同候选源码均被拒绝。
输出保留双对照收益/最坏退步，将显式开发成本与三组全部在线成本相加，供 ToolEvolver 的 `cost_regression` 门槛检查。
演化控制器在每个验证阶段启动前写入 `controller-cost.json`，提供 `{cost_context}` 命令占位符。
将 `--evolution-context {cost_context}` 加入 `cost_regression` 阶段命令，可把控制器已发生的设计/验证费用和本次验证预留额计入总成本。
该快照按事件截止点验证，后续结算不会改变原报告；同一账本同时作为开发账本传入会被拒绝。
控制器预留开销与外部验证实际成本是分别记账的项目；配置时应明确计费口径，不能将同一费用同时登记在两处。

### ToolEvolver 六阶段统一入口

`tool_pipeline` 连接已实现的参考、影子、在线效用和成本验证器。先提供规格：

```json
{
  "reference": "reference-suite.json",
  "shadow": "shadow/suite.json",
  "online": "online/suite.json"
}
```

三类套件必须预先冻结相同的 identity 和 tool_files；参考套件须包含 correctness、synthetic、robustness 的独立案例。

```sh
python3 -m rec_harness.tool_pipeline freeze pipeline-spec.json pipeline
python3 -m rec_harness.tool_pipeline run --suite pipeline/suite.json \
  --stage correctness --candidate candidate --output correctness/result.json
```

在 ToolEvolver 的六个阶段 command 中使用此 run 命令，分别填入 stage，并使用 `{candidate}`、`{output}`；
`cost_regression` 另外传入 `--cost-context {cost_context}`。原始阶段报告的身份、收益和在线标记保持不变，晋升门槛仍由演化控制器执行。
该入口要求标准的演化 stage 输出目录，以读取已接受的前序报告。
最后阶段校验候选源码与前序报告一致，自动汇总三个参考阶段、影子阶段、三组在线过程以及控制器成本快照。
已停止的历史 validation-attempts 也纳入成本；实际费用未知时保留原预留额。
统一入口不自动发明参考答案、不选择研究假设，也不会把离线正确率换算成在线收益。


## Codex 内置论文检索

论文查询直接使用 Codex CLI 的内置工具。假设生成角色默认传入
`-c 'web_search="live"'`；其他角色默认 `disabled`。通用配置放在
`codex.web_search`，Amazon 和演化配置放在 `agent_backend.web_search`。
值可以是 `"disabled"` 等统一模式，也可以是角色到模式的映射，例如：

```json
{"hypothesis": "live", "tool": "live", "optimizer": "disabled"}
```

隔离和普通 CLI 共用这一策略。检索要求优先使用论文原文及作者来源，并在
`method.sources` 保留 URL、版本或章节。每次调用生成 `literature-search.json`，
保存模式、实际 CLI 搜索事件、原始日志哈希及角色报告的引用。启用工具不等于
实际检索；引用仍是未独立核验的外部论文主张，不能据此升级本地机制结论。
没有额外 arXiv API 客户端或检索服务依赖。

历史前缀验证器 `python -m rec_harness.harness_prefix` 只回放冻结输入，强制
关闭内置搜索，并将传输实现哈希纳入绑定。它检查角色合同，不验证科学收益；
关闭搜索工具也不等同于整个联网 CLI 的系统级网络隔离。

配置依据：[Codex web search mode](https://developers.openai.com/codex/config-basic/#web-search-mode)。

## 自主测量缺口评审

带 `tool` 槽且未声明 `measurement_requirements` 的新 `research-project-v1`
项目会自动开启测量评审。单独的通用或 Amazon 模型 campaign 也可以设置：

```json
{"measurement_discovery": {"development_budget": 50}}
```

`development_budget` 是工具开发 BCU 上限，不是模型训练 GPU 秒数。
证据角色完成后，控制器调用独立 `measurement` 角色，输出以下三种结果之一：

- `sufficient`：现有测量足以评价下一次模型实验，继续提出模型方案。
- `measurement_gap`：提出一个会影响具体决策的缺失测量，记录竞争解释、最小试验和独立验收条件，再交给工具演化。
- `prerequisite_unresolved`：现有数据、调用、实现或适用性尚未明确，先记录问题并停止本轮；不据此宣称工具或模型无效。

控制器先检查基线有效性和诊断执行状态，再核对引用、四项前提状态、测量名称及开发额度。
缺口记录标为 `agent_hypothesis`，评审输入和响应进入 CAS，已提交评审可以复用。
预先声明的缺失测量仍优先处理。工具只能经过原有独立验证阶梯晋升；后继模型
重新测量后再次评审，不能将提案本身当作已补足测量的证据。
项目保留所有已晋升工具承诺解决的测量要求，并将它们冻结到后继模型配置。
即使工具通过了验证，只要实际输出缺失目标测量，后继研究仍停在
`measurement_required`；达到项目回合上限时报告 `measurement_unresolved`，
不会把未解决的问题归为普通回合结束。

候选尚未训练、性能尚未知、训练费用待主机估计，均不构成测量前提不足。
角色调用单独计入原研究账本；启用评审的新轮次预留额外一次调用。

## Harness 统一验证入口

`python -m rec_harness.harness_pipeline` 将 `prefix_audit`、`development`、
`confirmation` 接入同一冻结验证链。先分别冻结在线 suite 和历史 prefix suite，
二者必须使用相同的 `identity`、基线策略和显式 Codex 模型；prefix 的 identity
取自在线 suite。pipeline spec 只引用这两份冻结文件：

```json
{"prefix": "prefix/suite.json", "online": "online/suite.json"}
```

```bash
python -m rec_harness.harness_pipeline freeze pipeline-spec.json pipeline
```

演化计划各阶段命令使用以下参数（`STAGE` 替换为该阶段名）：

```text
{python} -m rec_harness.harness_pipeline run --suite /absolute/pipeline/suite.json --stage STAGE --candidate {candidate} --output {output} --cost-context {cost_context}
```

控制器提供当前阶段的成本截止快照。报告同时保留本阶段成本、原始报告哈希、
各子任务账本以及累计成本；累计值包括生成/阶段开销、此前阶段和中断尝试。
各阶段的 `maximum_total_cost` 因此是累计上限。前缀回放始终标记 `online=false`；
它通过后仍须运行独立开发与确认实验，不能代替在线效用。

冻结时检查历史来源与在线任务身份、seed 及生成器验证 seed 的重叠；运行时
检查前序已接受报告对应同一候选和 pipeline。当前适配器覆盖合成 toy 任务及
四个只读前缀角色，不代表多任务族或完整外部信息隔离验收。

## 本地激活与残差旁路诊断

```bash
rec-harness activation-probe probe-config.json probe-run
```

配置要求 `source`（`path`、`sha256`）、`settings`（`HIDDEN_UNITS`、
`NUM_BLOCKS`、`KERNEL_SIZE`、`DROPOUT_RATE`、`MAXLEN`）、`execution`
（隔离 `recipe`、PyTorch `python`、`uid_registry`）、`budget`、`cost`、`timeout`。
默认使用长度 16 的五条合成历史，`MAXLEN` 至少为 16；费用为声明的 CPU
探针 BCU，超时上限 120 秒。省略 checkpoint 配置时使用随机初始化。

可选的已训练模型输入使用注册研究身份，而不是直接传任意权重文件：

```json
{"checkpoint_source": {"study": "/absolute/study", "model_version": "legacy_incumbent", "dataset": "Industrial_and_Scientific", "seed": 42}}
```

主机先核对开发 seed、注册源码和训练档案，拒绝确认 seed；权重只进入禁止
网络的本地 CPU worker，使用 `weights_only=True` 和严格 state_dict 匹配。
输出为 masked 激活 RMS、残差 RMS、零值比例以及长短历史的旁路表示变化。
主机检查来源、计数、有限值及聚合算术，仅公开固定数值字段，不转发 worker
自由文本。结果不是准确率或因果机制结论，旁路干预可能偏离训练分布。

已有开发 checkpoint 时，可增加 `"history_sampling": {"sample_size": 32, "seed": 13}`。
主机从同一 study、同一 dataset 的冻结 `train` 分割确定性采样，核对读取前后
数据哈希，截断到模型 MAXLEN 后右对齐。最多 128 条、MAXLEN 最多 512；不足
指定样本数则拒绝。仅把采样序列写入私有本地输入，不附带用户标识、验证目标
或逐用户排名。主机独立计算长短分组及有效位置计数，并核对 worker 输入哈希。
输出明确标为训练历史上的表示敏感性，不代表验证集效果或未见用户泛化。

重复运行相同配置会核验并复用结果；中断和失败保留成本及输入/进程档案，
不会悄悄重试成成功。改变配置或实现需要新运行目录。该入口目前支持独立
本地诊断和以下可选研究上下文接入，尚未完成诊断增量效用实验。

Amazon campaign 可选配置 `activation_diagnostic`，字段为 `settings`、`execution`、
`budget`、`cost`、`timeout`，以及可选的 `datasets`、`seeds`、`history_sampling`。它对当前 incumbent
的所选开发 checkpoint 自动运行本地诊断，在证据上下文中提供
`activation_diagnostics` 聚合列表，并写入带来源标识的 Observation。
不将权重、私有档案引用或本地输入路径放入角色上下文。失败会阻止本轮模型
提案；同一模型和协议复用已核验结果。`budget` 是整个 campaign 的探针额度，
`activation_costs` 汇总子账本，不能再与子账本重复相加。

这项配置会让外部 Codex 角色读取模型内部响应的聚合量，应由操作者明确
授权相应出站范围；仅需本地诊断时使用 `activation-probe` 命令。当前真实
角色发送验收因自动审批拒绝而等待用户补充授权，本地生成和恢复已通过。

## 同信息图/平面前缀对照

已冻结的历史前缀 suite 可通过同一入口运行呈现对照：

```bash
python3 -m rec_harness.harness_prefix run --suite prefix/suite.json \
  --candidate unchanged-policy --output memory-comparison/report.json \
  --memory-context-bytes 131072
```

`unchanged-policy/policy.json` 必须与 suite.baseline_policy 完全一致。每个历史
上下文仅切换 evidence 的 graph/flat 呈现，其余输入、源码及提示策略相同，
外部搜索关闭。两侧规范 JSON 均须在同一字节上限内；超限或策略变化在任何
调用前拒绝。报告提供完整上下文的归一化哈希、证据信息哈希、两侧大小、
具体输入归档和恢复账本。baseline 对应 graph，candidate 对应 flat。

该报告的 gain 是合同通过率差，不是模型指标收益；online 始终为 false。
统一字节上限不等于 token 消耗相等，实际 token 用量保留在调用收据中。
要主张图组织提高研究效率，仍需独立在线决策/模型评估与完整成本对照。

研究配置可设置正整数 `context_byte_limit`，限制每次角色实际接收的规范
JSON 上下文字节数。检查发生在 graph/flat 转换之后、费用预留和启动 Codex
之前；不裁剪证据，不将字节数当作 token 数。

独立 campaign 的输入可用以下命令审计：

```bash
python3 -m rec_harness.memory_comparison graph-campaign flat-campaign audit.json
```

两侧配置除 `memory_presentation` 外必须完全相同，并声明同一
`context_byte_limit`。审计验证事件链和归档输入，按角色及出现次序配对，
还原 evidence 后比较完整上下文和源码快照。运行收据、源码或角色数量差异
均保留；`all_inputs_equal=false` 表示不能将该配对视为同信息呈现证据。
即使为 true，也不代表决策质量或在线指标改善。该命令只审计已有记录，
不启动研究或补跑角色。

`outcomes` 从已提交事件和 CAS 评估归档读取每轮有效性、分数、选择及逐 seed
失败原因，同时列出提交后未完成的角色和中断预留。两组终态分数相等不代表
候选都得到有效评估；这些执行差异必须与指标一起解释。审计期间事件头变化
会拒绝输出混合时刻的结果。

## 共享基线的在线记忆对照

```bash
python3 -m rec_harness.memory_online --suite online/suite.json \
  --output memory-online/report.json --stage development \
  --context-byte-limit 262144
```

入口使用已冻结的 isolated toy suite，先完成一次基线评估，在任何角色调用
之前锁定并验证其事件链、源码和收据。随后通过 SQLite backup 创建 graph/
flat 两个独立账本，保留共同历史，并追加分叉来源事件；不复制临时 jail。
两组使用相同提示、模型、工具、字节上限且关闭外部搜索，首个 evidence
输入必须通过精确同信息审计。后续研究允许产生不同提案、结果和知识，
报告保留这些差异，不要求所有后续输入仍相同。

每组预算保留共享历史费用；总执行成本只计一次共享基线，再加两组新增
费用。报告列出每组终态分数、图组减平面组的指标差、运行次序和完整输入
审计。不同 case 交替执行顺序；单个 case 仍不足以控制服务时序与随机性。
实现源码哈希与 suite 一起绑定，改变实现必须使用新的验收目录。重复执行
同一终态入口只验证既有结果。注入测试 agent 的结果始终 `online=false`。

知识角色调用前，控制器将已完成的评估和负责人决策写入 CAS 检查点并提交
事件。该位置发生中断时，恢复会保留决策和候选源码、保留中断费用，不重复
训练；未完成的知识回写记为 deferred，可用 `consolidate` 单独重试。
归档实现用于恢复时必须保留完整的 `rec_harness/` 包目录及原生资源，确保
清理环境后的 CPU 子进程也能从包父目录导入启动器。

## 来源经验与 Fresh 的目标任务对照

```bash
python3 -m rec_harness.memory_transfer freeze source-campaign target/suite.json transfer-study
python3 -m rec_harness.memory_transfer run --study transfer-study/study.json \
  --output transfer-result/report.json --stage development
```

来源必须是已结束的 isolated toy 研究，使用固定基础工具且没有导入其他任务
经验。冻结入口锁定来源事件截止点、公开经验包及源阶段成本，并复制目标
suite 的输入对象。来源与目标任务 ID、生成数据条件（seed 及 seed+10000）
不能重叠；基础 Codex 配置、工具和 Harness 策略保持一致。

Fresh 和 Experienced 各自初始化、评估同一目标模型并在独立图中研究。
Experienced 仅增加冻结经验包，在第一次目标评估前导入；两组使用相同
上下文字节上限并关闭外部搜索。报告核对首个证据角色实际可见的来源节点，
同时列出候选有效性、未完成角色、目标指标差和负迁移比例。初始信息本来
不同，不能将该实验称为图/平面的同信息对照。

`source_cost` 与 `target_cost` 分列，`total_attributable_cost` 为二者之和。
来源阶段可以复用已有研究，不代表冻结时重新消耗了这笔费用；重复运行同一
终态不会重新调用角色或训练。注入测试 agent 或来源缺少真实 Codex 请求时，
报告不标为在线证据。目前仅支持单一合成任务族，跨业务迁移另需验证。

## 固定权重的开发集块旁路评分

```bash
python3 -m rec_harness amazon-block-score /path/to/study legacy_incumbent \
  Industrial_and_Scientific 42 /path/to/new-output \
  --execution /path/to/cpu-execution.json --bypass-block 6
```

execution 文件提供 `recipe` 和 `uid_registry`。省略 `--bypass-block` 得到参考
重评分；指定从 0 开始的块号时，以该块输入替换输出。入口只允许 development
seed，先审计已登记源码和 checkpoint，再在隔离 CPU 进程推理。主机保留正例
位置密钥，计算排名、指标及与原记录的差异。旁路钩子结束时移除，不改权重。

每次输出使用新目录，并记录推理墙钟时间。正式诊断需先验证参考排名一致，
再完整报告计划内条件及费用。这是固定权重的开发集敏感性，不能当作删除后
重训收益、机制确认或最终测试。示例四类全块验收另用独立账本计 36 BCU。

## 恢复训练产物归档

冻结隔离任务的图证据通过 `isolated/runtime-lock.json` 引用运行库，避免为每个实验重复展开数万条路径。导出 Experiment 的 `attrs.artifact_refs` 为一个 JSON 文件后，可执行：

```bash
python3 -m rec_harness amazon-restore-archive snapshot.json /path/to/artifacts/objects /path/to/new-output
```

恢复前核验全部引用、环境锁及请求/完成收据绑定，兼容旧的平面文件映射。锁内与平面映射冲突、缺失对象、已有输出目录均会拒绝。恢复运行库的只读/可执行权限；冻结环境身份仍按现有 runtime validator 校验。该命令只恢复已归档文件，不重建设备节点，不启动训练，也不把恢复目录自动认定为可执行研究环境。

真实 Electronics 已完成任务的恢复验收还原了 67,967 个文件，其中 67,909 个环境文件的字节和权限逐项一致，checkpoint/排名归档一致，源研究图未变。验收未新增训练或 Codex 调用。


新初始化的 Amazon campaign 默认启用 `structured_insertion_contract`。hypothesis 的 `Proposal.insertion_contract` 仍是字符串字段，但内容须为 JSON：

```json
{"schema":"nextitnet-method-scope-v1","allowed_methods":["NextItNetModel.__init__","NextItNetModel.encode_sequence"],"rationale":"说明为何这些方法承载候选与机制对照的改造"}
```

上下文提供当前源码的实际方法目录。合同在 hypothesis 阶段检查并冻结，candidate 和所有新生成对照共享这个改动范围；源码 AST 的实际变化必须非空且位于声明方法内。experiment 校验和独立准入预检均执行这一约束；准入在启动任何 CPU 探针前检查全部新模型，不能只检查候选而漏掉对照。每个版本的范围核验结果进入其评审证据。

已有 campaign 不回填配置或改写合同。显式配置 `structured_insertion_contract: false` 保留自由文本兼容入口，但如果主动提交该 JSON 合同，仍须遵守范围约束。方法范围正确不证明自然语言机制成立；原因果、评分、批次独立性和梯度检查仍执行，对照实验仍承担效果和归因判断。

通用 campaign（含小型示例）的 Codex 传输失败现在同样停止整轮调度。失败调用收据及成本保留，重复 `autoresearch` 不会自动重试；知识角色失败时保留此前已经提交的模型选择。服务修复后，使用：

```bash
python3 -m rec_harness retry-transport /path/to/campaign --reason "说明已完成的服务修复"
python3 -m rec_harness autoresearch /path/to/campaign
```

此入口保留失败轮次，按原冻结上限进入下一轮；不重放已经完成的角色，也不增加预算或轮次上限。达到原轮次上限时需另建 campaign。知识写入可另用既有 `consolidate` 显式补做，无须重训。Amazon campaign 仍使用 `amazon-retry-source` 或准入后的独立评审恢复入口。项目在子 campaign 传输停止期间保持原模型阶段，子任务修复重开后再恢复项目。

查看已提交轮次的实际 incumbent 开发成绩及成本边界：

```bash
python3 -m rec_harness amazon-progress /path/to/amazon-campaign
```

`rows` 按完成事件顺序保留每次尝试（包括失败及重开前的尝试）。成绩取当轮实际 incumbent 的开发结果，按数据集与开发 seeds 聚合；不以事后最高候选代替当时选择，不读取确认或最终测试成绩。失败轮次没有已提交开发读出时成绩为 null。

新轮次在完成事件中绑定角色账本和 study GPU 账本的成本前缀，查询时重新核验。两种单位分开列出，均保留继承成本与失败支出；这不是含外部诊断/演化账本的项目总费用。历史角色费用可以按其事件前缀重建，但缺少轮末 GPU 前缀时明确标为 `historical_boundary_unavailable`，不会拿当前总费用回填历史点。查询为只读，不启动服务、训练或恢复账本。

每行的 `completion_sequence` 和 `event_hash` 标识具体完成事件；显式重开后同一 round 可有多行，原失败点仍保留。`role_pending_reservations` 和 `study_pending_reservations` 是该历史边界仍在运行的预留数，不是当前待结算数。之后的结算不会重写过去的预留金额。缺失历史 GPU 边界时，其金额和预留数均为 null，不能解释为零支出。

新项目的后继记忆默认启用来源一致性检查：恢复时从已提交前驱重建导出并核对缓存，拒绝替换包与提交后改变的前驱事件链，详见[后继记忆绑定](implementation/successor-memory-binding.md)。

项目运行期间或结束后，可用 `python3 -m rec_harness project-report /path/to/project --format markdown` 查看各回合及嵌套验证的角色调用、费用和待结算项；[报表范围与真实核验](implementation/project-report.md)。此命令只读，不触发恢复或额外调用。

Harness 演化可使用 `trace_sources` 收集多个 campaign 的独立冻结前缀，避免手工拼接及本地角色 ID 冲突。所有来源任务必须排除在独立验证之外；项目模板和测量转交保留完整来源集合，详见[多来源轨迹入口](implementation/multi-trace-sources.md)。
