# 优化器到 Tool Evolver 的测量转交

对应方案 5.5D。新项目默认冻结 `optimizer_measurement_handoff: true`；独立 Harness 演化可显式配置 `optimizer_handoff_budget`。该模式以一次 `optimizer_router` 调用替代一次旧 optimizer 调用，并不额外增加评审调用。输出选择 `harness_patch` 或 `measurement_gap`，`payload_json` 必须符合输入提供的相应 schema。旧项目缺少该配置时沿用原 HarnessPatch 合同。

主机从冻结决策轨迹构造测量请求，只接收成功角色调用中的有效公开基线数据；诊断执行失败不当作测量不足。通用来源保留有效 baseline runs，Amazon 来源要求开发基线及成功执行收据。输入绑定轨迹 hash、允许引用、测量开发预算和当前诊断，压缩轨迹先无损还原再检查。

测量分支复用现有 `validate_review`：必须引用给定证据，四类前置条件符合规则，仅提出一个有区分能力的测量，不能超出开发额度或重复已有输出。通过后，主机提交 `measurement_handoff_recorded`，将审查归档到 CAS 并创建 provisional MeasurementGap。引用的是轨迹而非伪造实验，因此 `experiment_refs` 保持空列表，实际源轨迹 hash 单独保留。

该分支不生成 policy.json，不运行 Harness 验证器，不晋升 Harness。项目将测量缺口转交独立 Tool Evolver，源模型任务加入独立验证的排除列表；工具仍须通过原六阶段门槛，晋升后才启用并复测。没有 tool 槽、运行环境、更新额度或后继模型回合时，项目保留缺口并以 measurement_unresolved 停止。前置条件不成立则拒绝该优化器结果，不绕到工具生成。

转交消耗一次 Harness 尝试额度，避免同一项目无限转交；费用仍留在原 Harness 子账本。项目迁移、已提交测量收据和图投影均可恢复，中断后不重复调用优化器。转交记录不能经 promote 当作可执行策略发布，投影修改和归档损坏会被拒绝。

工程验收包含模型→优化器转交→工具验证→后继模型复测、项目提交后中断、测量收据提交但候选文件未写入时中断、无工具槽停止，以及数据条件不足拒绝。测试使用脚本角色、测量替身和固定验证报告，不构成真实自主缺口发现或工具效用证据。此项晚于 v15 安装包，尚未用新路由角色执行真实 Codex 验收。

后续[真实选择合同验收](../experiments/harness-router-codex-v1.md)已完成一次 Codex 调用。角色自行选择 Harness 补丁，schema/主机范围校验、归档、图记录与无重复调用恢复通过。真实测量转交正例仍未验收，不因返回补丁而重试强求测量缺口。

全量回归 419 项通过（21 项环境相关跳过）；随后补充的合同/恢复加强经转交定向 4 项及项目定向 13 项检查通过。日志为 `/tmp/rec-harness-optimizer-handoff-full-tests.log`、`/tmp/rec-harness-optimizer-handoff-final-targeted.log` 和 `/tmp/rec-harness-optimizer-handoff-project-final.log`。
