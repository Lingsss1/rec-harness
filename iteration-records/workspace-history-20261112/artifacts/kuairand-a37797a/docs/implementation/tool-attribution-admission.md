# 工具按归因增量启用

新工具计划可以设置 `validation_basis: attribution_increment`，`stages` 只包含 `attribution`。不再要求独立任务、在线模型收益、三组对照或六阶段晋升实验；允许使用产生缺口的原始轨迹与已有工件，不需要训练。已冻结的旧计划继续按其原规则恢复，不自动改写历史。

归因增量定义：实际诊断输出使至少一个竞争解释的判断发生变化，并改变下一步研究决策。变化既可以是支持/排除一个解释，也可以撤销原先没有根据的判断。仅新增数字、重复现有统计、写出诊断计划或改写结论措辞不构成科学上的归因增量。

## 计划和报告

计划保留 budget、agent_cost、development_ids（来源追踪，不作独立性排除）、identity、rollback_version。attribution 阶段规则需要 command（argv）、timeout、cost、maximum_total_cost，不要求 minimum_gain 或 maximum_regression。

验证器在恢复出的候选代码上运行诊断。报告包含 passed、implementation_hash（候选实现绑定）、evaluation_ids（允许与 development_ids 重叠）、identity、total_cost 和 attribution：

- gap_id：对应原测量缺口；存在来源缺口时必须匹配。
- before / after：相同的至少两个竞争解释 ID 到 unresolved / supported / refuted 的映射。
- observations：非空列表；每项含 evaluation_id、measurement、有限数值 value、explanation_ids、interpretation。改变判断的解释必须有测量支撑。
- controls：基本正确性检查，包含 name、passed、observed、expected。
- decision_before、decision_after、rationale：说明测量如何改变下一步行动。
- limitations：非空限制列表，不能将局部归因增量表述为独立因果证明。

报告由现有验证进程执行和归档路径接收，检查通过后自动登记可用版本，后续研究回合可加载。内部仍沿用 candidate_promoted 事件和 promoted 状态以保持注册、继承、回滚及恢复兼容；它在该模式下表示“局部归因验收后启用”，不表示通过独立效用实验。无需额外人工晋升。

工具角色会收到该验收标准，注册信息记录 validation_basis。Harness 的原验证要求不受影响。

## 边界与验证

宿主检查报告结构、实现绑定、来源、测量及控制结果、前后判断和决策变化；科学解释的正确性仍依赖具体验证器/审稿，结构校验不能防止貌似合理的错误解释。当前不提供适用于所有模型的通用归因判官，也不把现有六阶段验证器自动转换成归因验证器：新任务需要提供能实际运行对应诊断的 attribution command。

测试覆盖真实执行一个固定输入上的合成诊断、复用开发来源、仅单阶段自动启用、运行时注册、恢复幂等，以及拒绝缺少测量、实现错配、无判断增量、控制失败的报告。这些是流程测试，不是 GR 归因收益证据。
