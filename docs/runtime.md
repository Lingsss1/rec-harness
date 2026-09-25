# Auto Research v5 运行指南

实现入口为 `rec_harness/`，Python 3.11+、无运行时第三方依赖。可从仓库运行 `python3 -m rec_harness`，或 `pip install -e .` 后使用 `rec-harness`。模型训练由用户提供的命令适配，不绑定某个推荐框架。

连续多轮的统一入口、隔离示例及 Amazon 自动研究配置见 [单入口自动研究](autoresearch.md)。

## 最小验收

```bash
bash scripts/validate-core.sh
python3 -m rec_harness codex-smoke /tmp/rec-codex-smoke
python3 -m rec_harness init examples/toy/campaign.json /tmp/rec-demo
python3 -m rec_harness run /tmp/rec-demo --rounds 1
python3 -m rec_harness status /tmp/rec-demo
python3 -m rec_harness verify /tmp/rec-demo
```

`codex-smoke` 和 `run` 实际调用已安装、已登录的 Codex CLI，会消耗该配置的模型额度。测试套件使用明确标注的模拟 Agent，并执行真实 CPU 训练；它不冒充真实模型调用。真实 Codex 的测试结果单独记录。

示例是用合成用户/物品特征预测偏好的回归模型。初始模型只有线性特征，允许 Agent 提出交互或训练方法改造。每个候选两个 seed，独立初始化；冻结评估器计算 `quality=1/(1+MSE)`、MSE 和参数数目。它验证框架流程，不支持真实推荐收益或机制因果的结论。

## 配置与执行

`campaign.json` 的 `source` 相对配置文件目录解析。`model_files` 是公开模型源文件白名单，`evaluator_files` 是控制器持有的公开检查/评分脚本，二者分别内容寻址归档。拒绝路径穿越、符号链接和显式 private/Hidden 字段。

`protocol` 冻结数据版本、切分、seeds、指标方向、最小效应、guardrails、检查/评估命令及成本。命令是 argv 数组，不经 shell。可用占位符为 `{python}`、`{model}`、`{output}`、`{seed}`、`{evaluator}`。每个命令必须使用独立输出路径，评分 JSON 必须包含主指标和所有 guardrail 字段。guardrail 当前为“数值不超过上限”。外部命令需要将数据切分与实际资源配置纳入自己的冻结实现。

角色按依赖顺序调用：

1. evidence：GapCard；区分观察、竞争解释和最小验证。
2. hypothesis：MethodCard、Proposal；固定模型、协议和 seeds。
3. experiment：只在独立 `model/` 修改候选代码，返回相对该目录的文件名。
4. 控制器：运行独立检查器，再重新训练与公开评分。
5. evaluation + lead：给出选模与科学结论建议。
6. knowledge：提出局部 GraphPatch，由控制器校验后提交。

当前执行策略为单 worker 顺序模式。Agent 并行和分布式训练调度没有实现。训练命令的资源隔离由执行环境负责。所有角色使用同一可配置基础 Codex 模型，非写代码角色使用 `read-only`，模型/工具开发角色使用 `workspace-write`。

`codex` 支持 `executable`、`model`、`timeout`。没有指定 model 时沿用本机 CLI 配置。调用采用 `exec --json --ephemeral --output-schema --output-last-message`；请求、实际上下文、JSONL 事件、最终响应、stderr、退出状态及 token usage 保留。使用官方 [Codex 非交互模式](https://learn.chatgpt.com/docs/non-interactive-mode) 的参数，并以本机 `codex exec --help` 核验。

`budget` 为预注册 BCU 总额；`agent_cost`、`evaluation_cost` 是每次调用/seed 的 BCU，而非美元、tokens 或 GPU 秒。BCU 不等价于真实物理资源硬限制。实际 token、墙钟单独记录，未采集的 GPU 秒/内存为 null。超时终止进程组；失败不退款。恢复无法确定实际消耗的运行时保留 reservation，实际值仍为 null。每次重试使用新任务 ID。

知识补丁被拒绝后，可执行 `consolidate DIRECTORY ATTEMPT_ID` 单独重试知识角色；新调用单独记账，原训练和旧补丁保留。

`--rounds N` 表示本次最多执行 N 轮，同时受 campaign 累计 `max_rounds` 上限约束。预算不足则停止。`recover` 只在取得控制器锁后将遗留运行标为 interrupted；`run` 同样自动恢复。配置和评估器身份变动会被拒绝。修改协议请创建新的 campaign。

选模要求所有 seed 通过冻结检查和 guardrail，聚合公开收益严格大于最小效应，且负责人建议 keep。协议不匹配和非有限分数拒绝进入比较。模型 keep 仍只产生 provisional 机制结论；Agent 建议不会绕过控制门槛。未经执行的提案对照仅作为待验证事项，当前适配器只自动运行候选和 incumbent 的配对评估。需要机制对照时，应在外部实验适配器中显式执行并以新证据提交。

## 事件、图与产物

NextItNet兼容模型可使用安装PyTorch的解释器运行本地结构探针：

```bash
python -m rec_harness.adapters.model_contract \
  --source /path/to/reviewed-model.py --expected-hash SOURCE_SHA256 \
  --model-id candidate-v1 --output /tmp/model-probe-v1/report.json
```

输入必须是已审查源码；该命令会执行源码导入，本身不提供任意Python代码隔离。它使用CPU合成输入，不读取训练数据，固定种子在源模块导入后设置，避免旧源码的导入副作用覆盖探针seed。输出目录保留内容hash命名的模型与探针源码，已有report拒绝覆盖。

检查包含多处因果扰动、前缀对未来embedding的梯度、所有可训练参数的BPR梯度连通与有限性、padding梯度、空历史形状和尾位置候选读出。追加 `--checkpoint CHECKPOINT.pt --checkpoint-hash SHA256` 可检查真实训练权重：weights_only加载、全权重有限值、严格state_dict匹配，并从embedding确定词表大小。加载前后及探针结束时核对checkpoint hash。默认不传这两个参数时仍只检查随机初始化。

`model_graph.register_contract`可将核验后的报告和源码CAS引用导入ModelVersion/Component/Observation；训练后报告必须提供 `checkpoint_path=`，模型身份包含权重hash，权重同时进入CAS。`compare_sites`报告跨模型同位置的类型、形状与参数布局对应，始终要求语义复查。模块复用时按调用序号区分组件身份。只记录直接张量身份连接，不虚构函数操作的完整数据流；探针通过只说明有限合成输入下的合同成立，不证明性能收益。Amazon适配器的新作业已接入训练前源码探针和训练后checkpoint探针，两个CPU子进程与训练共用作业deadline；任一失败均使结果无效，前置失败不启动训练，槽位占用期间的时间仍计费。请求标记pre-and-post-v1并归档探针源码/hash、报告及进程日志，比较报告重新核验门控结果。原有完成收据不回写为通过新门控；旧36份checkpoint复测证据单独保留。该入口目前仍只允许已审查模型，尚不构成任意Codex生成源码的GPU隔离准入。

```text
campaign/
  campaign.json                    # 冻结配置
  state.json                       # 当前投影，可从完成事件恢复
  memory/memory.sqlite             # 事务事件源、版本图、预算
  memory/events.jsonl              # 可重建的只读事件导出
  memory/exports/                  # 兼容 graph/lessons/results 视图
  artifacts/objects/<sha256>       # 无损内容寻址产物
  tasks/<task-id>/                 # 各 Codex 角色的独立工作目录
  evaluator/                      # 冻结公开评估实现
  research_archive/manifest.json
  research_archive/decisions/      # 当时可见状态、上下文与动作
  research_archive/attempts/       # 代码前后快照、各 seed、观测、选模
```

SQLite events 是唯一可写事实源，JSONL/图表/导出均是投影。hash 链提供完整性审计，不是针对恶意系统管理员的签名。`verify` 校验事件、图快照和引用产物。`graph rebuild` 从事件重建全部图版本；回退创建新版本，旧事实/快照不删除。

```bash
python3 -m rec_harness graph /tmp/rec-demo snapshot
python3 -m rec_harness graph /tmp/rec-demo bundle --task toy-preference-interaction --protocol HASH --role evidence --query interaction
python3 -m rec_harness graph /tmp/rec-demo patch --file patch.json
python3 -m rec_harness graph /tmp/rec-demo rollback --version 1
python3 -m rec_harness graph /tmp/rec-demo rebuild
```

GraphPatch 为 `{base_version, nodes, edges}`；节点 `{id,type,attrs}`，attrs 至少包含 `task_id,protocol_hash,origin`。关系 `{id,source,target,relation,attrs}`，attrs 包含 `scope,evidence_refs,origin,status,created_at,revision_id`。支持/反证和对照须同任务同协议，引用必须存在。事实节点不可覆盖。

Claim 需要 `status,scope,evidence_refs`。supported 至少两个不同 independent_run_id 的有效、可比、通过预注册 effect_gate 的本地 Experiment；显式标记为 development 的实验不能充当确认依据，legacy_unverified 或 agent_hypothesis 来源也不能升级结论。confirmed_in_scope 另需同样通过上述资格门槛的机制对照，失败或开发阶段的对照不能满足该条件。兼容未声明 partition 的旧本地协议，其独立运行合同仍有效；这不代表其具备独立确认集。运行时不会把一次模型胜出自动标为满足这些条件。字段校验只能证明结构与声明满足门槛，不能替代可信实验器和语义复查。

检索先执行任务、协议和可见前缀过滤，再按词项匹配与角色类型优先级排序，保留反证配额并展开两跳关系，最后才用其他节点补足容量。诊断优先观察，假设设计优先缺口和方法，实施优先提案和组件，评价优先实验；具体优先级冻结在 retrieval.py。没有向量 embedding 服务。EvidenceBundle 带图版本/hash、来源 ID、检索方法版本、角色优先级、逐节点入选原因和被截断反证数。该行为已有合同测试，但尚无等信息记忆消融的收益证据。旧模型缺口晋升后记录 recheck 事件，不能自动升格到新模型的科学事实。

配置顶层 `memory_presentation` 可选 `graph`（默认）或 `flat`，随 campaign manifest 冻结。控制器内部保留原始图，在每次 agent 调用前转换呈现并归档实际输入；各研究角色分别执行对应检索。flat 将节点与关系转成统一记录列表，保留每条关系、来源、反证和检索元数据。两种形式都带 `information_hash`，`memory_view.restore` 可无损还原并校验。`python3 -m rec_harness graph DIRECTORY bundle-pair --task T --protocol P --role evaluation --version V` 可从同一次冻结检索导出两个视图，用于相同证据前缀的配对决策实验。它只控制信息与呈现，不保证字节数/token 数相等；完整在线运行分支后也不保证证据继续相同，不能直接宣称等信息收益。

旧记忆通过 `migrate DIRECTORY LEGACY_DIRECTORY --task T --protocol P` 幂等导入。五种旧文件的原文保留为LegacyRecord，精确文件字节（包括lessons envelope及文件修订）进入CAS：results.tsv和episodes.jsonl生成Experiment标识与Observation摘要，failures.jsonl生成Failure，lessons.json生成Claim，strategy.json生成非历史证据的Context。支持lessons列表及`{"lessons": [...]}`封装。

同任务/协议下相同显式experiment_id或run_id共用一个实验标识，各来源摘要独立保留，不以重复日志增加独立样本。源文件没有实验ID时仅建立明确标为unidentified的记录标识。所有旧实验均`valid=false`、`effect_gate_passed=false`、`independent_run_id=null`；原来声称的validity、seed列表、confidence和confirmed状态仅作为reported字段保存，不继承科学资格。lesson的支持和反证关系保留，缺失引用写为unresolved而不生成假实验；后续导入实际源记录可补齐引用，旧图版本和原文不变。父实验存在时保留报告的设计血缘，失败只作执行证据。

源文件不删除。原记录中的外部日志/权重路径保留为原文，不自动跟随并声称已归档；仅上述五种实际读取文件有可恢复CAS字节。单独旧血缘图的其他schema尚需显式适配，未验证来源仍不能导出为合格迁移经验。

`memory DIRECTORY export PACKAGE --task A --protocol P --cutoff-seq N --version V --limit 100 --max-bytes 65536` 导出 `memory-policy-v2`。截止点绑定来源事件序号、hash和时间，拒绝截止点之后的图版本；省略截止点时冻结当前事件末尾。容量同时限制节点数和nodes/edges规范JSON的UTF-8字节数，元数据另计。相同快照、截止点和容量下导出内容hash稳定。

包只保留白名单中的条件性结论、方法、来源、聚合实验证据与组件/模型身份；权重、可执行工具、原始用户列表、预测数组及任意产物引用不在允许字段中。类型化字段检查不能判断自由文本是否真实或被人刻意误标为公开，因此仍须使用合格源图。结论的图证据引用及相关支持/反证按完整关联组导出，不能为满足容量而单独丢掉反证；遇到跨任务、缺引用或不合格节点则省略整组并记录数量。回放attempt依赖标签作为来源元数据保留，不误当作图节点ID。

`memory NEW_DIRECTORY import PACKAGE --task B --protocol Q` 校验schema、hash、容量和引用闭合后写入独立目标图。源实验变为Source资料，源Claim降为GapHypothesis；目标的有效证据引用为空，相关关系标为历史来源关系，反证仍有检索配额。迁移必须在目标研究开始前冻结，与源图分库；持有目标控制器锁时拒绝导入。重复导入幂等，导入知识不能再作为新源经验反向导出。v1包需从原始源图重新导出，不静默把缺少截止点和容量的旧包当成v2。尚无自动语义组件对应或A→B收益证据。

## 历史回放

已审查GPU训练代码可通过独立入口 `python -m rec_harness.gpu_launcher --root JAIL --script /inputs/worker.py --expected-hash SHA256 --device 0 --uid UID -- ARGS...` 验证隔离兼容性。JAIL须由isolation.prepare以显式gpu_devices和匹配UID准备；脚本需root所有且不可被组/其他用户写入。启动器核验入口身份与单卡设备路径，清空环境，拒绝额外继承句柄，预初始化CUDA后仅保留指定设备及配对pipe/eventfd，关闭socket和其他句柄，启用TSYNC过滤后chroot降权，以runpy执行入口，避免exec丢失CUDA内部状态。外部控制器仍负责超时、预算预留、独立作业暂存、只读输入、输出归档与清理。本入口已通过真实NextItNet缩短协议作业，尚未作为任意生成代码的默认执行器，也不提供完整驱动安全证明或校验器与模型的进程分离。

`gpu_job.run_reviewed_job`封装每作业新目录、注册表内不重复UID、指定GPU节点、输入hash校验与独立只读副本、启动器调用和终态输出收集。调用方必须先预留预算；暂存/运行/收集共用计时并返回allocated_gpu_seconds，异常也必须结算或保留未知成本。产物由受限收集器进入CAS后恢复到控制器所有的results目录。沙箱保留以便审计，不自动重用或删除。运行时recipe来自操作者审查的目录/文件清单，recipe hash只证明清单身份，尚不代表所有依赖包字节的完整环境锁。该执行器已经过独立UID的真实NextItNet小作业验收，Amazon正式调度切换仍需显式记录执行版本，不能静默改写历史收据。

Amazon调度已支持显式版本接入：`python3 -m rec_harness.adapters.amazon_study register-execution STUDY recipe.json`追加注册事件；recipe只有绝对路径列表runtime_dirs/runtime_files。注册同时记录执行器、调度器和合同检查代码hash。随后 `python3 -m rec_harness.adapters.amazon_study run STUDY --execution EXECUTION_ID --model-version VERSION --seed SEED`使用独立沙箱执行，前后合同门控仍生效。代码变化要求注册新版本；已完成job若属于其他版本则拒绝覆盖，需新study。没有--execution时保留已审查源码的旧直接执行路径，旧收据兼容标为direct-reviewed-v1。配对报告拒绝混用执行版本，聚合也只包含所选执行版本。隔离路径仍只适用于已审查模型，未取消冻结模型注册限制，也未绕过真实Codex上下文出站审批。

执行注册还将整个rec_harness Python控制器包存入CAS，涵盖预算、恢复及间接导入，避免只记录hash而丢失代码字节。注册时对照归档hash与即时hash，解析版本时重新核验每份CAS产物；缺失、损坏或当前代码变化均拒绝使用。旧的仅hash注册记录保留原样，不能伪称已补全历史源代码。该归档仍不包含Python/PyTorch/驱动等完整环境字节，不能据此宣称环境锁已完成。

`runtime_archive.freeze_runtime(template, artifacts, output)`可进一步将已准备模板中声明的Python/依赖文件与加载器位置存入CAS，记录文件模式、recipe hash及主机内核/驱动版本指纹。排除inputs/output/workspace/tmp/dev/proc；`restore_runtime`从CAS恢复到新目录、校验hash与只读模式，并拒绝主机指纹不匹配。它归档用户态文件，不归档内核或GPU状态。当前尚未将该归档强制绑定到GPU预初始化前的宿主解释器/模块加载，因而不能把“有归档”当成“每个执行进程已使用归档环境”；完整执行环境锁仍待接入。

独立策略进程入口为 `replay-policy`。Linux 主机需要 root 启动权限、系统 `/usr/bin/python3` 和 libseccomp；执行策略前会 chroot、切换独立非特权 UID、关闭额外文件描述符，并禁用网络、fork/clone、io_uring、ptrace 等调用。模板仅包含冻结并校验 hash 的系统 Python 运行时，不包含完整 world、历史图、项目数据或 Codex 凭据。GPU 不挂入此环境。该边界不防御内核漏洞。

```bash
/usr/bin/python3 -m rec_harness.policy_sandbox prepare /tmp/replay-runtime
python3 -m rec_harness world-from-campaign /tmp/rec-demo /tmp/world.json
python3 -m rec_harness replay-policy /tmp/world.json examples/toy/replay_policy.py /tmp/policy-report \
  --runtime /tmp/replay-runtime --jails /tmp/policy-jails --budget 24 --seed 42
```

每次决策启动新进程，只在 stdin 提供 `view/previous_outcome/memory/seed/step/direction/worker_limit`。stdout 必须是一个 JSON 对象，字段为 `parents/action_kind/select/memory/rationale`；`select` 只能引用此前可见成功节点，`memory` 用于显式跨决策状态。策略不得假定已知未揭示的后继。修改代码、工具或提示的动作返回 unsupported，不能从档案获得虚构分数。

策略输出及状态限制为64 KiB，进程有CPU、内存、输出文件和墙钟上限。失败保留日志并终止该回放，不伪造动作。轨迹记录实际可见输入、选择、结果与成本；策略实际执行墙钟单独保存，不能当作历史训练成本。示例策略在固定 world/seed 下通过两次独立进程回放一致性测试；自定义随机策略应显式使用所提供 seed，不承诺任意依赖时钟或系统随机源的程序可复现。真实OS回归可用 `REC_HARNESS_OS_TESTS=1 python3 -m unittest discover -s tests -p test_replay_policy.py -v` 执行；普通测试环境明确跳过需要权限的集成测试。

```bash
python3 -m rec_harness world-from-campaign /tmp/rec-demo /tmp/world.json
# 动作为 parent ID 批次数组，例如 [["root"], ["attempt-1"], []]
python3 -m rec_harness replay /tmp/world.json actions.json --budget 12
```

`world-build NODES_JSON WORLD_JSON` 也可摄取外部适配节点。缺请求、快照、依赖或完整性检查时降为 partial_replay；一般 DAG 标 audit_only，不能静默转换为分支链。没有执行结果的提案不能成为有分数节点。

根按创建顺序展开，非根仅允许当前前沿。同步批次只使用批次开始时已揭示依赖，统一完成后揭示；未知成本、未揭示依赖、新 prompt/代码/工具返回 unsupported。耗尽历史返回 exhausted，不能解释为没有改进空间。失败照常记历史消耗，缺失分数不补零；选定质量和 best-so-far 分开报告。

从本运行时转换时，采用所有先前尝试作为保守上下文依赖，可能限制更多动作，但不会悄悄删除跨分支依赖。回放在可信模拟器中持有完整 world；给策略的只有 `Replay.view()`/`step()`。不应将 world 文件直接交给待评估策略。Replay 类本身不隔离调用者；执行外部策略须使用上述 replay-policy 隔离入口。BCU 回放和真实墙钟不同，首版不提供并行提速推断或官方 Dream-RSI 格式兼容声明。

转换器保留所有已启动尝试。缺少最终档案时，节点状态为 incomplete、结果为空，并将整个world标为audit_only；Replay拒绝执行该世界，不能通过跳过失败生成更好看的策略结果。resource_ledger保留每笔预留、实付和状态；无法归属某轮的支出（包括研究开始前失败的基线）计入initial_cost，未知实付按预留保留。单个已归属尝试若实付未知，其cost仍为null，不能按零成本展开。转换前后核验源事件head，期间发生写入则拒绝混合快照。

真实 Amazon 固定训练档案使用 `python3 -m rec_harness amazon-archive STUDY OUTPUT.json` 导出。转换器独立复核成功训练的rank指标、候选/数据/源码/检查点身份，将收据成本与事件预算流水对账，保留没有成功收据的前序、中止和未知成本，并以源事件head和内容hash冻结输出。不同类别的得分不能直接当作模型优劣排序。当前36次GPU训练没有研究决策上下文、模型设计父节点与同步研究动作，因此该输出明确为 `audit_only`，不是Replay输入；不补造父子链或据此评估调度策略收益。重试旧版本的详细收据仍保留于原始attempt-history，导出资源流水涵盖其费用。

## 诊断工具与 Harness 外部优化

```bash
python3 -m rec_harness evolve /tmp/evolution propose --id candidate-1 --kind harness --context context.json --plan plan.json
python3 -m rec_harness evolve /tmp/evolution validate --id candidate-1 --stage prefix_audit
python3 -m rec_harness evolve /tmp/evolution validate --id candidate-1 --stage development
python3 -m rec_harness evolve /tmp/evolution validate --id candidate-1 --stage confirmation
python3 -m rec_harness evolve /tmp/evolution promote --id candidate-1
python3 -m rec_harness evolve /tmp/evolution rollback --kind harness
```

`propose` 实际调用 Codex，在独立目录开发，保存失败成本。context 应提供 MeasurementGap，或历史决策可见前缀及相关失败，不提供未来结果与 Hidden。工具验证阶梯为 correctness/synthetic/robustness/shadow/utility/cost_regression；Harness 为 prefix_audit/development/confirmation。

plan 必含 `budget,agent_cost,development_ids,identity,rollback_version,stages`。identity 冻结 `base_llm,tools,start,graph,budget,protocol`。每个 stage 包含 argv `command`、`timeout,cost,minimum_gain,maximum_regression,maximum_total_cost`。验证命令使用 `{python},{candidate},{output}`；必须包含至少一个本地验证文件，其内容 hash 在开发前冻结。

验证器由操作者独立提供，不由候选自己生成或评分。输出 JSON 字段为 `passed,gain,regression,total_cost,evaluation_ids,identity,comparison_groups,online`，旧 `arms` 字段仅兼容读取。报告中的成本需包含开发与诊断。工具 utility 必须比较 no_tool/existing_tool/candidate_tool；Harness 比较 baseline/candidate。这些是固定实验对照组，不是多臂搜索调度。shadow 至少两个条件；utility/development/confirmation 必须 online=true，且不与开发数据重合；confirmation 不能重用 development 任务。相同数据血缘的独立性由验证器保证，框架只校验显式 ID 和身份。

候选代码、验证器与验收计划冻结，验证阶段不能跳过，晋升必须对应账本内控制器已提交的通过记录。未满足阶梯就保留未晋升候选。真实效用验证未运行时不能宣称工具或 Harness 已改善研究。

Harness 补丁采用声明式研究策略 JSON：`prompts`、`retrieval_limit`、固定合法 `routing`。稳定内核不在 editable scope。新campaign配置可指定 `harness_registry`（相对配置文件目录解析）和 `harness_identity`（必须与冻结验证plan.identity完全一致）；init从注册表读取已提交活动版本，核对晋升/回退链、全部验证报告和CAS策略文件，再将策略与来源event head、candidate/plan/policy hash冻结到campaign。不能同时传非空research_policy覆盖；注册表后续晋升或回退不会改变已有campaign。harness_identity是操作者对适用条件的声明，框架的相等检查不替代验证器对真实模型、任务及预算身份的审计。该接入已通过合成验证器的晋升、下一campaign读取、回退与篡改拒绝测试，尚无真实Harness收益证据。当前不支持自动重写已安装Skill、运行中热更新或任意内核代码补丁。工具晋升保存版本化代码与回退点；接入实际诊断命令仍由训练适配器配置。

## 部署边界与尚未验收项

目录隔离和 JSON 的 public 标签不等于系统级数据权限。Codex read-only 允许读取运行账户可见文件；训练候选也是可执行代码。**研究执行账户/容器不得访问 Hidden、凭证目录或私有数据；终评必须由独立身份/服务完成。** 控制器/评估器宜以独立身份运行，候选仅获公开输入和写入自身输出的权限。框架的 hash 检查可以发现改动，不替代 OS 权限。

本交付实现本地控制流程及测试，不包含训练平台、GPU 调度、分布式服务、向量数据库、可信 Hidden 服务、官方 Dream-RSI 适配或真实业务效果验收。现有三个 Skill 的诊断/记忆协议继续可用；运行时只接收可信适配器的公开证据，不自动执行任意 Skill 文本。

`isolation.prepare(..., runtime_archive=report, runtime_artifacts=artifacts)` 和
`gpu_job.run_reviewed_job(..., runtime_archive=report)` 支持从已冻结的 userspace
CAS 归档恢复作业目录。准备阶段核验归档摘要、宿主机内核/驱动身份和请求的
runtime recipe；归档不匹配即失败，不回退到宿主机复制。请求、jail manifest
和完成回执记录 `runtime_archive_hash`。归档对象应与作业使用同一 artifact store。
此选项只固定 jail 内的运行文件；launcher 在 chroot 前启动的解释器和 CUDA
预初始化仍使用调用方指定的宿主运行环境，因此尚不构成完整执行环境锁。

启用 runtime_archive 的 GPU 启动器在 CUDA 预初始化后、执行作业前，调用
`runtime_loaded.audit_loaded_runtime`。它逐项核对当前 `/proc/self/maps` 中的
文件与归档路径、设备/inode、大小及 SHA256，拒绝已删除、未归档或不一致映射，
并将审计结果写入进程日志。该快照不覆盖后续 lazy load、Python 源码或驱动内存。
当前执行环境的 `stat` 与 proc 映射身份不同，且 `/proc/self/map_files` 读取被拒绝；
真实 CPU 导入校验因此未通过，启用该归档选项不能据此声称已可运行。旧的未启用
归档的 reviewed-code 路径仍保持原有边界声明。

Amazon 执行注册支持 `register-execution STUDY recipe.json --runtime-template PREPARED_JAIL`。
模板必须包含与 recipe 一致的 `jail-manifest.json` 和实际运行文件。注册将运行文件
存入该 study 的 CAS，并把归档及其摘要绑定到新的 execution_id；输入数据、输出和
内核设备不进入归档。每次解析执行版本会验证归档对象以及宿主内核/驱动指纹，
训练入口将同一归档交给 GPU launcher，不会静默退回读取实时运行库。
不传该参数时保持历史注册格式。注册/恢复/篡改拒绝及训练参数传递已通过测试；
以上真实加载映射审计限制仍然存在，尚未完成启用归档的真实 GPU 训练验收。

存在待运行作业且执行版本绑定环境归档时，Amazon 调度在任何新 GPU 预留前
启动指定 Python 的 CPU 预检，清空 CUDA 可见设备并导入 PyTorch，再核对当前
文件映射。每次预检保存进程、结果 CAS 和 `amazon_runtime_preflight` 事件；失败
直接停止，历史无归档路径不变。预检成功不替代 GPU launcher 在 CUDA 初始化后
进行的第二次审计。`runtime-preflight-acceptance-v1` 的真实 CPU 子进程拒绝了
本机不一致的 Python 映射，保留一条失败事件，GPU 账本变化为 0；这是失败路径
验收，不是环境锁成功或 GPU 运行验收。

`adapters.isolated_contract.run_isolated_contract(...)` 在独立 UID 的 CPU jail 中
运行 NextItNet 合成输入检查；仅装入运行库、冻结的模型与检查器，不提供训练数据、
GPU 节点或凭据。启动器清空环境、关闭宿主文件描述符，使用 TSYNC seccomp 禁止
网络和进程创建，并设置 CPU 时间、地址空间及文件大小限制。检查使用单线程，
避免进入边界后 OpenMP 再创建线程。输出在进程终止后按文件类型和大小核验并收集
到 CAS；进程退出成功但缺失报告仍返回 invalid。

这不是正式候选准入：候选与检查器同进程，仍可能修改 Python 状态或伪造输出。
回执始终标记 `qualification=untrusted_probe_output`、`candidate_admitted=false`。
不能用它代替独立评估、语义审查或系统级资源配额；运行库目前仍按 reviewed recipe
准备，未宣称完整环境锁。已完成的 CPU 边界实测覆盖无 GPU、宿主 study/proc 隐藏、
网络/fork/提权/设备创建/源码写入被拒绝；不代表对内核漏洞的防护证明。

生成当前 Amazon 验证读数：

```bash
python -m rec_harness amazon-summary STUDY_ROOT NEW_REPORT_DIRECTORY
```

此命令对 baseline/incumbent 及固定前缀消融分别重新执行配对审计，拒绝缺失配对、
排名与指标不一致、源码变化和期间事件链更新；输出 report.json、README.md 及
两份配对报告。报告包含逐类别确认均值、所有负向种子切片和完整预算账本读数，
不把共享验证用户上的确认种子当作隐藏测试，也不宣称新的 Codex 候选收益。
输出目录必须为新目录，失败留下的部分材料不能作为完整报告。

真实任务研究交付预注册：

```bash
python -m rec_harness amazon-plan STUDY_ROOT HANDOFF_JSON
```

HANDOFF_JSON 包含 `cards`（gap/method/proposal，遵循 Codex 角色输出 schema）和
`versions`。versions 的 candidate 为 `{model_version, source: {path, sha256}}`，
controls 为同结构的非空数组；Proposal.controls 必须与 control model_version 顺序
一致。源码只被读取和归档，不导入执行。Gap/Proposal 必须绑定已注册基准与原协议，
Proposal.seeds 必须按原 development_seeds + confirmation_seeds 给出，预算不得
超过当前账本余量。开发与确认的具体类别/种子矩阵由控制器生成。

注册在 controller.lock 下追加 amazon_research_plan_registered 事件，将源码存入
CAS；不改冻结 study.json、不加入可执行 model_versions、不保留 GPU 预算。相同
proposal 的重复请求幂等，改写同名 proposal 或占用其他计划的模型名称被拒绝；
通过源码 hash 查找 jobs/attempt-history，防止结果产生后换名伪装为预注册。

这一步只完成设计交付，qualification 为 preregistered_design_only，
candidate_admitted=false。正式代码准入、开发执行及开发后冻结选择的确认释放仍须
接入；phases.release_condition 是交付契约，不意味着已有自动释放实现。现有审阅
模型执行入口不会因为计划注册而接受新模型。科学陈述仍受共享验证用户限制。

独立排名接口 `adapters.independent_scoring` 将每个用户的候选项随机置换，仅将
历史、置换后的候选项和不含用户身份的行标识交给模型。正样本位置及用户 ID 保留
在控制器私有 key 中。`score_worker.py` 在 CPU jail 中加载模型/checkpoint 并返回
分数，控制器拒绝缺失、重复、非有限数值或夹带自报指标的响应，独立计算排名。
原冻结协议按严格大于正样本的负项数处理同分；该接口保留此规则并单独报告同分
用户数。全等分数会暴露该旧规则的偏置，不能因此准入候选或默改历史评估协议。
实际验收已重算 Movies/legacy_incumbent/seed42 的 1000 用户：所有排名与原归档
一致，正样本同分用户为 0。其余版本/类别仍需独立复核，这不是完整候选真实性证明。

`adapters.amazon_prefix_candidate.admit_plan` 目前只允许两个明确的算术改动：
15%→20% 前缀混合，以及 `(16/17) * incumbent_blend` 对照。先验证注册计划中的
CAS 源码与精确单行替换相同，再运行 CPU 合同；任何其他代码变化在执行前拒绝。
这不是任意生成模型的通用准入。新版本训练必须按一个注册计划单独提交，确认
种子需要显式释放事件；训练费用含失败或未知的保留成本，同时受提案和研究预算
限制。未释放的确认种子在训练调度前被拒绝。

研究计划的开发读数与确认释放：

```bash
python -m rec_harness amazon-development STUDY_ROOT PROPOSAL_ID NEW_READOUT_JSON
python -m rec_harness amazon-confirm STUDY_ROOT PROPOSAL_ID LEAD_DECISION_JSON --readout-hash SHA256
```

开发读数逐一审计完整的类别/seed/版本矩阵，并要求基准、候选和对照的执行、数据、
候选集身份匹配。负责人决定必须引用该读数摘要，且仅为 provisional/unknown 的
keep；控制器还要求候选相对基准超过冻结 minimum_effect、相对每个对照为正。
开发证据变化、缺失配对、提前运行确认或改写冻结选择均不能释放确认。适配器的
开发释放门控是在首轮开发训练期间实现，报告明确记录这一时点，不将它表述为
新的预注册确认标准。原提案的确认标准与开发选择标准仍分开保存。

`amazon_research_graph.ingest_development` 将预注册 Gap/Method/Proposal 连接到
已审计开发比较及原训练实验，保留各类别负向切片；开发汇总不能作为独立类别证据
或确认性 Claim。生成的研究子图仅包含本轮开发上下文，便于后续知识角色处理。

对照复用支持已注册、源码身份相同的既有模型。计划在 reused_control_versions
中显式列出这些版本，并保留其既有结果暴露范围；候选仍必须是新版本，旧结果不能
换名伪装成新候选。开发/确认矩阵去重，避免把同一个既有基准兼对照重复计数。
历史训练成本继续留在研究总账，不重复扣入新提案额度；复用仍须逐项通过执行、
协议、数据与候选集身份审计。

受限 prefix 准入新增 linear_recency 家族：只允许将统一前缀均值的两行替换为
三行 `weights = valid * valid.cumsum(1)` 加权累计和，保留 0.15 混合及全部其他
源码。`adapters.recency_math.inspect_recency` 从已核对源码抽取实际聚合语句，
以显式循环、手算标量、解析导数和 double gradcheck 验证数学合同，并测试内部
padding、零/一个前驱的等价性、未来扰动与填充输出。此证明不等于性能收益。

按预注册阶段运行，自动排除需复用的既有对照：

```bash
python -m rec_harness amazon-run-phase STUDY_ROOT PROPOSAL_ID development
```

该入口先检查所有候选已准入、源码匹配，并审计既有基准/对照在该阶段的全部配对
结果；执行版本不一致时拒绝复用。只调度新模型版本，因此既有基准兼对照不会
重复训练。使用 confirmation 参数仍须先通过冻结选择释放；入口不会自动解锁。

研究闭环的知识补丁、DecisionTrace 与 amazon_research_cycle_completed 事件现由
同一个 SQLite 事务提交。完成事件写入前中断会连同图版本回滚；SQL 提交后 JSONL
投影失败时，重开 Store 从已提交事件恢复投影，图与完成记录保持一致。磁盘中的
cycle.json 是可重建输出，不是提交依据。故障注入覆盖上述两个边界；此保证仅限
本次图及完成事件提交，不代表所有文件系统、训练和远程服务操作均具跨系统事务。

授权追加预算通过 amazon_campaign.continue_campaign 创建新研究，并在新账本一次
继承前驱全部已收费/预留成本。复用的 incumbent 结果先审计、复制后保留原凭据；
它们不是新实验。旧调度器拒绝继续开支，新研究在前驱封存并完成激活前不可运行。
activate_campaign 可恢复封存后/激活前的中断；同一前驱禁止重复分叉，避免余额
重复使用。历史未采用此续接协议的前驱研究保持原读取兼容。

明确授权后可使用命令行续接：

```bash
python -m rec_harness amazon-continue NEW_STUDY PREDECESSOR PROJECT --additional-budget 4000 --authorization-file AUTHORIZATION.txt
python -m rec_harness amazon-activate NEW_STUDY
```

第二条仅用于激活恢复，不会追加预算。授权文件记录用户已经给出的授权；命令本身
不会替用户授予追加资源权限。预算累计上限等于旧上限加本次追加量，历史费用一次
继承，旧研究配置与旧提案均不改写。
