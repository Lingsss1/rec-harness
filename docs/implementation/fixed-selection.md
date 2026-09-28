# 固定候选选择执行器

`python -m rec_harness.fixed_selection` 在冻结候选池上比较无工具、现有工具、候选工具三组选择。正式入口直接调用隔离的 Codex CLI selection 角色；每个条件三次调用，不生成新候选、不自动重试。所有组使用相同模型、候选描述和角色指令，仅观测不同。论文检索不参与这一封闭信息实验，搜索固定禁用。

先复制并修改 [示例配置](../../examples/fixed-selection/spec.json) 的模型、隔离模板、运行目录及实际条件，再执行：

```bash
python -m rec_harness.fixed_selection freeze /path/to/spec.json /path/to/frozen-suite
python -m rec_harness.fixed_selection run --suite /path/to/frozen-suite/suite.json --stage development --output /path/to/development/report.json
python -m rec_harness.fixed_selection run --suite /path/to/frozen-suite/suite.json --stage confirmation --output /path/to/confirmation/report.json
```

开发和确认必须使用不同输出目录。示例的两个条件只演示格式，不能作为独立科学条件。配置提供每个候选的公开描述、各组观测和主机评分表；调用方负责确保描述与观测不泄露评分，并证明观测来源及条件独立性。执行器不运行诊断工具或训练模型，不自动验证这些科学前提。

冻结时校验候选与观测覆盖、有限分数、两阶段条件ID唯一、总调用额度，绑定角色合同和包内Python源码。每次角色只收到本条件候选与本组观测，评分表、其他组观测、既往选择和确认条件均不进入请求。主机在选择提交后计算分数、相对两组对照的有符号平均收益、最大单条件回退和最优候选遗憾值。最小化目标自动反转收益方向。非法候选不获分；报告passed仅表示选择合同有效，不表示产生正收益。

请求、响应、传输收据进入事件链及CAS。已提交响应在恢复时直接复用；未知调用结果停止，不能因恢复重复发送。结算中断可恢复且不重复扣费。报告类型为fixed_candidate_selection，online始终为false；live_decisions仅表示使用正式后端而非测试注入，不把预先提供的评分当作新模型实验。该入口独立于端到端候选生成执行器，不能代替要求在线效用的晋升阶段。

8项新增测试覆盖评分、方向、三组请求边界、正式后端接线、非法选择、合同变化和恢复；与搜索、隔离环境及评价类型相关共18项测试通过。正式后端接线测试使用mock，本次没有实际Codex调用或训练。变更晚于v27安装包。

完整源码回归：593项中572项通过、21项因当前解释器依赖或OS测试条件跳过，163.048秒。公开树边界检查及CLI冻结通过；示例两阶段均完成脚本选择、评分和终态恢复。证据位于rec-harness-runs/fixed-selection-v1，不能据此宣称真实选择效用。
