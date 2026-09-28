# Amazon工具接入六阶段验证

现有 tool_pipeline 按online套件类型分派：online-tool-suite-v1沿用合成任务；amazon-online-tool-suite-v1调用Amazon三组研究及专用费用审计。correctness、synthetic、robustness、shadow仍复用既有验证器，utility运行Amazon研究，cost_regression重新核对真实证据与累计费用。未改动晋升门槛，也未把零收益候选变成通过。

使用[组合配置示例](../../examples/amazon-tool-online/pipeline.json)指定reference、shadow和Amazon online套件，再运行原入口：

```bash
python -m rec_harness.tool_pipeline freeze /path/to/pipeline.json /path/to/frozen-pipeline
```

三个套件必须具有完全相同的冻结identity、候选文件集合和输入合同。后续在工具计划中配置原tool_pipeline六阶段命令即可；没有新增模型调用入口。运行前预检会递归检查Amazon源码/数据身份及条件ID，正确区分utility与confirmation条件，并核对开发/历史排除集。组合配置不是已经准备好离线/影子实验的证据。

费用审计重新核对每组初始study、控制器默认值、GPU执行版本、工具源码、启动标记、完成记录与实际结果。逐个研究回合必须具有六种角色的完成事件，以及指定模型的成功隔离Codex传输收据；工具组还必须有成功测量送达完成角色。它拒绝只有online=true标签、但没有对应收据的报告。传输检查证明调用和输入送达，不证明角色实际采纳了工具建议。

费用阶段从角色调用、GPU秒、诊断三个账本轴复算在线报告，按原套件权重折算；离线/影子开发费用和外层Evolution费用已经是BCU，按原数值加入。旧验证尝试中的Amazon账本根据相邻冻结配置辨认轴，GPU秒不会直接作为BCU相加。所有费用来源查重，未结算或配置/结果/聚合不一致则拒绝。CPU准入和独立复算的wall time仍未计价，不能将该数解释为全部硬件费用。

恢复沿用保守规则：Amazon直接在线入口和pipeline utility只有在所有已初始化组均提交完成记录、且证据复核一致时，才继续其余组。任何组结果未知都会停止，不自动重发研究调用或训练。费用阶段自身中断仍不在自动重启白名单中；已提交外层结果继续沿用原CAS恢复合同。

新增6项Amazon费用/恢复测试和1项pipeline路由测试；合计43项相关测试通过，7.044秒，覆盖旧合成费用、pipeline、预检、Amazon对照、角色送达及外层恢复。测试中的训练结果、角色传输和部分审计是明确夹具；不是实际Codex/GPU验收。本轮没有真实角色调用、训练或晋升。证据位于rec-harness-runs/amazon-tool-pipeline-v1。后续已补[激活观测及采集费用](amazon-online-activation.md)；真实Amazon完整六阶段运行和后续安装包验收仍未完成。
