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

## Amazon NextItNet

该适配器在已初始化、冻结协议并有完整 incumbent 配对结果的 Amazon study 上运行。创建 study 和历史成本接续见 [运行指南](runtime.md)。

```bash
export REC_HARNESS_AMAZON_STUDY=/path/to/active-amazon-study
# 其余变量与上例相同；CPU recipe 用于独立 checkpoint 复评分。
python3 -m rec_harness autoresearch /path/to/amazon-campaign --config examples/amazon/autoresearch.json
python3 -m rec_harness autoresearch /path/to/amazon-campaign
```

控制器自动冻结研究提案、注册候选、检查模型源码、运行 CPU 数值契约及隔离的三步合成梯度检查，再调度完整开发集训练和独立 CPU 复评分。梯度检查要求学习后仍满足因果性、padding、尾部表示读出，以及 forward 与编码表示一致性；它不改变真实训练的优化器、协议或 checkpoint。只有通过开发集门槛并获 lead 保留建议才释放确认 seed。确认结果只在主机侧判定晋升，不发送至 Codex。当前确认仍使用相同验证用户的新初始化，不是最终测试。

源码修改范围是 `NextItNetModel` / `NextItNetBlock` 内已声明方法，允许的张量算子和模块可在 `model_source_contract.py` 查看。训练、数据处理、导入和方法签名冻结。新增模型结构不需要修改逐候选字符串模板；超出能力范围的代码被拒绝。这个受限源码契约不是任意 Python 的安全证明。现有 Amazon 训练仍使用与历史 comparator 匹配的 reviewed-source 执行路径，不应把此适配器作为通用不可信 GPU 代码执行器。

每轮复用已审计 incumbent 作为唯一配对对照。另设机制消融仍需扩展适配器；不能据此声称已完成任意多对照研究。角色调用上限和 study GPU 秒分别计账；失败计费、历史成本接续及每个训练任务的预算检查保持生效。`minimum_round_gpu_seconds` 是启动下限，不是整轮成本预测；每个提案仍受自己的冻结预算和 study 总预算限制。

Amazon 角色有终态响应时恢复会复用，候选源码和上下文有哈希绑定；不合法响应最多尝试两次。已完成训练由 study 的审计与恢复机制复用；不完整 CPU 复评分另存后重试。失败轮写入 `failure.json` 并继续下一轮。已选中的 candidate 成为下一轮 incumbent。下一轮同时获得此前设计、开发集决策与 provisional knowledge；确认结果和确认阶段失败详情始终保留在主机侧。

Amazon 也支持下述 `diagnostic_registry` 与 `measurement_requirements`。诊断只接收已审计开发 seed 的品类聚合指标，包含 dataset 标签；不传入原始交互、checkpoint 或确认结果。启用工具时须指定独立 `diagnostic_budget`，费用记录在 diagnostic-budget，不能混算为 Codex 调用次数。必需测量缺失会生成 MeasurementGap，以 measurement_required 停止，保留 incumbent；此时不会生成模型提案或启动训练。工具外循环的 measurement_source 可直接引用该 Amazon campaign。新研究回合可冻结晋升后的工具并复测；项目总调度可自动创建 Amazon 后继回合，继续使用同一冻结 study 和共享 GPU 账本。

`verify` 和 `recover` 已识别 Amazon campaign，无需初始化 Codex 服务。verify 核对控制器/绑定 study 事件链、角色内容与绑定、已提交开发周期及 CAS，并单列旧角色记录缺少完整事件绑定的情况；它不替代逐 GPU 作业或私有确认结果审计。恢复在控制器锁下检查角色和诊断工作进程，保留未知实际成本的预留费用。

## 工具与 Harness 外循环

工具的 correctness、synthetic、robustness 阶段可使用主机参考验证器：

```bash
python3 -m rec_harness.tool_validation --suite /path/to/reference-suite.json \
  --candidate /path/to/candidate --output /path/to/stage/result.json --stage correctness
```

suite 含 identity、tool_files（必须含 diagnostic.py）、budget、case_cost、timeout、CPU execution，以及 cases。每个 case 包含唯一 id、stage、input、expected 数值字典和 absolute_tolerance。参考值与容差始终留在主机，仅 input、候选源码和固定 worker 进入 CPU jail；主机核验输出格式并计算误差。每个已完成案例写入事件链，失败也保留，恢复不重跑案例以取得通过结果。修改 suite、源码或 worker 必须使用新验证目录。

报告的 measurement_accuracy 是通过案例比例；兼容验证阶梯的 gain 字段也表示此比例，不能解释为模型增益。online 固定为 false，因此不能作为 utility 阶段的在线证据。当前接口只支持 JSON 聚合测量，不覆盖原始张量、非有限 JSON 值或自动构造参考答案。影子与三组在线效用使用下文的独立验证器，不复用参考值准确率作为在线收益。

同一入口接受 `adapter: research-evolution-v1`。配置提供 `candidate_id`、`kind`（`tool` 或 `harness`）、公开 `context`、冻结 `plan` 和 `agent_backend`。工具可提供 `source` 与 `model_files`，初始化时归档；后续恢复不再读取源工作区。Codex 分别承担 tool 或 optimizer 角色。

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

## 开发经验续接

```bash
python3 -m rec_harness amazon-memory-export /path/to/amazon-campaign /path/to/development-memory.json
```

下一 Amazon 配置通过 `development_memory_files` 导入。导出包括设计、开发集读数、开发集决策、临时知识和非确认阶段执行失败；不导出确认结果。导入核验协议、包和读数哈希、开发 seed 范围。这是经验传递接口；跨任务迁移收益仍需独立实验。

通用模型入口使用 `initial_memory_files`。初始化将经验证的记忆包复制至当前 CAS，在第一次模型评估前导入独立目标图；来源结论保留为待目标验证的类比。重复恢复不重复导入，也不把来源证据当作目标实验。

当前通用导出格式为 `memory-policy-v3`，读取端兼容 v2。v3 将公开 Proposal
节点及 `USES` 关系纳入证据闭包，避免实际研究中 Claim 引用 Proposal、Proposal
连接 Gap/Method 时被整组排除。字段白名单仍排除源码、权重和原始用户数据；
目标图中的 Proposal 降为 Source，不能直接执行或作为已证实的目标证据。
容量不足时仍整组舍弃支持与反证，不能单独保留成功结论。

## 项目总调度

`adapter: research-project-v1` 将模型、工具和 Harness 的入口连接为连续项目。配置含 `task_id`、`max_model_episodes`、`model_config`，以及可选 `tool_config`、`harness_config`（均是配置文件路径）。`max_tool_updates`、`max_harness_updates` 默认各为 1；工具更新还需 `tool_runtime`，其字段为 execution、cost、timeout，与 diagnostic_registry 的执行配置一致。两者的 execution 都可用 recipe_file 读取 JSON；相对路径以所属配置文件目录为准，不得同时提供 recipe 和 recipe_file。各子配置继续使用前述冻结协议和验证计划，项目调度器不能降低验收门槛。

```bash
python3 -m rec_harness autoresearch /path/to/project --config /path/to/project.json
python3 -m rec_harness autoresearch /path/to/project
python3 -m rec_harness verify /path/to/project
```

模型回合因测量覆盖不足停止时，调度器从已提交 MeasurementGap 创建工具候选，并将触发实验加入开发排除列表。工具通过全部阶段后才注册到新模型回合，保留测量要求并重新测量；失败候选使项目以 tool_rejected 停止。正常模型回合结束后，如配置了 Harness 优化，则从真实决策轨迹生成候选，通过验证后在新回合启用。旧回合不热更新。

后继回合继承当前 incumbent 源码、原冻结评估器，以及投影后的来源经验包，使用独立目标图。每次子回合结果与状态迁移一并提交到项目事件链；中断发生在子回合完成后、项目提交前时，恢复复用子回合结果。费用保留在各子 campaign 的账本中，外层验证成本与其内部验证回合不能重复求和。项目主目录的预算账本不代表子回合总成本。

当前总调度支持通用模型及 Amazon 适配器。Amazon 后继回合继承 incumbent ID 和经过验证的开发记忆包；study 配置与初始记忆包在项目创建时冻结，后续 GPU 成本继续进入同一 study 账本。Amazon 新角色调用也保留调用前上下文/源码快照及完成事件，可直接作为 Harness trace_source；旧档案没有这些绑定时不会倒推补造。工具验证任务的自动设计、任意未预声明测量缺口发现、多任务族在线验证，以及三循环联动的真实收益验收尚未完成。控制器测试中的固定阶段报告仅验证交接与恢复，不是在线收益证据。

## 结果与边界

`run-report.json` 包含终态、停止原因、预算和事件链校验；`state.json` 记录轮次和 incumbent。每轮保留源代码、角色请求/响应、失败及独立评估证据。恢复已达到停止条件的 campaign 不额外调用角色或训练。要改变冻结配置、追加轮次或预算，创建新 campaign；Amazon GPU 预算仍以底层 study 的授权接续为准。

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

`python3 -m rec_harness.tool_exposure CAMPAIGN OUTPUT` 只读审计工具结果的送达链。
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
