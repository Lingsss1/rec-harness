# 生成工具读取已提交的公开观测

新工具计划可以显式选择：

```json
{"tool_contract": "public-observation-measurement-v2"}
```

工具仍实现 `diagnostic.py:measure(request)`，返回有限数值的 `measurements` 与 `limitations`。主机把完整接口说明冻结到工具生成角色的 `diagnostic_contract` 中，注册信息保留所选合同。默认 `public-aggregate-measurement-v1` 的请求结构和历史任务不变。

v2 请求在已有 `task_id`、`model_version`、`protocol_hash`、`runs` 之外，增加：

```json
{
  "observations": {
    "activation_diagnostics": [],
    "block_score_diagnostics": []
  }
}
```

每个观测必须恰好包含 `observation_id`、`model_version`、`dataset`、`seed`、`measurements`、`limitations`、`scope`。主机先核对它是图中已提交、有效、公开、同任务和协议的本地 Observation，逐字段一致；模型必须是当前 incumbent，dataset/seed 必须属于有效开发基线。重复观测、额外字段、非有限值、未提交或过期观测在扣费和启动进程前拒绝。

Amazon v2 控制器先运行已配置的激活/块级诊断公开投影，再执行生成工具。投影失败则保留测量失败状态，不运行依赖它的工具。工具只获得这些聚合量，不获得 checkpoint、用户、排名、张量或任意模型访问能力。没有配置观测时对应列表为空；工具必须明确处理不支持的输入，不能据此推测数值。

完整输入存入 CAS，与工具和 worker 一起绑定测量身份。恢复核对原输入并复用已完成进程；角色送达审计从实际角色输入重建同一请求，覆盖完整 evidence 输入和精简 measurement 审查输入。范围和限制不被合并成一个无条件结论。

验证专用工具可在 `tool_runtime` 指定 `contract: public-observation-measurement-v2`，由已有冻结机制传入。六阶段门槛没有放宽；验证案例必须提供与候选声明相符的输入。离线参考验证器能接收这一请求结构；内置 Amazon shadow 套件已支持[冻结公开观测来源](observation-shadow.md)，并核对多阶段合同一致性。合成在线套件可显式启用[模型标量输出观测](online-output-observations.md)，仍不等于产生Amazon内部观测。依赖新观测的真实工具仍需要相应的独立在线验收输入，不能用旧套件或本次固定报告测试声称效用已通过。

7项新增工程测试覆盖严格输入边界、CAS与缓存、进程完成后的中断恢复、旧合同兼容、Amazon项目生成/注册/复测/角色送达，以及采集失败短路。Amazon测试通过实际 worker 函数运行固定工具代码，CPU进程和角色由夹具替代，不是新的真实Codex、GPU或CPU jail运行。
