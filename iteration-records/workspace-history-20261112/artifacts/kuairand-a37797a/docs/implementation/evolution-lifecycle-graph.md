# Evolution 生命周期图

自动工具/Harness 演化在终态将已提交事件投影到自身 memory 图，覆盖候选、实现、阶段验证、晋升、版本替代及回滚。工具使用 ToolCandidate、ToolValidation、ToolVersion；Harness 使用 HarnessPatch、HarnessValidation、HarnessVersion。结构化 MeasurementGap 通过 ADDRESSES 连接候选，跨研究引用保留为外部引用，不伪造本地图节点。

投影携带源事件 hash、冻结计划和实现身份；所有节点明确不是独立实验单位，不生成模型效果或机制 Claim。公共图只记录验证阶段与通过/拒绝状态，不复制确认分数、评估任务 ID 或自由文本失败内容。旧格式缺口缺少必要字段时标记 `legacy_unstructured`，不补造缺口身份。方案在 proposal 提交前失败时，仅保留原停止事件，不虚构候选。

新运行记录图投影请求。终态提交后、图写入前中断，恢复只补图，不重复 Codex 生成或验证；重复同步不新增事件，已有投影被修改则拒绝覆盖。旧终态恢复保持原行为，可显式执行 `rec-harness evolution-sync-graph DIRECTORY` 补图。

Harness 新候选另将完整已接受提案写入 CAS，由 `candidate_implemented.proposal_ref` 绑定。执行策略与提案分别保存，失败诊断、因果假设、引用轨迹、风险、预期变化和验证计划不再只依赖可修改的开发目录。验证检查 CAS 完整性及候选投影与事件的一致性，提交后恢复也校验提案归档。

提案生成 provisional DecisionFailureHypothesis，HarnessPatch 通过 ADDRESSES 关联它；假设通过 SUPPORTED_BY 关联冻结上下文中实际存在的 DecisionTrace。该支持关系标记为 agent_hypothesis，不表示主机验证过因果解释。轨迹只保存外部身份、内容 hash 和上下文 hash，不复制完整可见图；源轨迹若记录 Harness 版本，则通过 GENERATED_BY 关联一个明确标为源轨迹版本的节点，不冒充本地晋升。没有不可变提案收据的历史候选不从可变 response.json 补造这条链。

此补充全量回归 406 项通过（21 项环境相关跳过）；最后补充源 Harness 版本关联后，Evolution 定向 17 项全部通过。测试覆盖实际自动演化入口中的拒绝候选保留假设、冻结轨迹关联、无递归上下文复制，以及归档篡改拒绝。此处是工程集成验证，没有新增真实 Codex 调用或 Harness 效用实验。

验证：全量 404 项回归通过，其中 21 项环境相关跳过；随后新增提前拒绝边界测试，Evolution 定向 16 项全部通过。项目集成覆盖模型缺口→工具候选→结构化图关联，工具完整六阶段、晋升、回滚和投影中断恢复均有覆盖。

真实验收位于 `/home/liuzhishan/lhr/rec-harness-runs/evolution-graph-acceptance-v1/acceptance.json`：只读原 `tool-evolver-acceptance-v1/campaign`，在独立图副本导入 8 个节点和 7 条边。保留实际拒绝结果，没有 ToolVersion 或模型 Claim；原事件、配置和报告哈希未变，无新 Codex、验证或训练执行。本项验证图集成，不增加工具效用证据。此改动尚未进入 v13 安装包。
