# 决策轨迹中的主机后续反馈

通用研究控制器的新 `agent_submitted` 事件显式记录 `episode_id`。轨迹提取依据该身份查找冻结截止点以内的 `round_completed`，把原主机 decision、事件序号及 hash 放入 `host_feedback`。没有对应已提交结果时，明确为 `not_committed_in_prefix`。

反馈与 `visible_state` 分离：后者仍是该角色当时真正收到的输入，input hash 不变。离线前缀验证器只重放原 visible_state，不将后续反馈放回角色输入。不同回合的结果不会串接；同回合多个角色引用同一主机结果，不成为多个独立实验证据。

旧提交没有显式 episode_id 时不根据任务名称猜测归属，保持旧轨迹结构。Amazon 分支不导出 auto_round_completed 的结果，因为其中可能包含主机确认/选型信息，不能直接复用通用结果出口。

测试核验主机事件与反馈相同、截止点前结果不可见、跨回合不混淆、原角色输入完全不变；轨迹 2 项、前缀 9 项、优化器转交 6 项均通过。没有新增实际 Codex 调用或训练。

Amazon 后续实现：新任务冻结 `trace_development_feedback: true`，角色提交显式记录回合 ID。开发训练及独立复评分完成后、评审与确认释放前，记录 `development_feedback_committed`，绑定开发读出 CAS、协议及开发 seeds。先核验读出 schema/hash、字段和开发分区，再提交；恢复只接受完全一致的原记录。

Amazon 轨迹按截止点和回合 ID 读取这条开发事件，并重新核验归档及分区，输出 `partition: development`。不读取最终选择、确认指标或 private-confirmation 文件，原 visible_state 不变。旧冻结任务缺少开关时不追加这种反馈，也不回填历史身份。review-only 旧资料不会被伪装成新的已提交开发事件。

Amazon 相关 161 项回归：157 通过、4 项 Torch 环境跳过。覆盖实际初始化/流水线接入、确认与对照恢复，以及开发反馈截止点、重复提交、确认哨兵排除、非开发 seed/未知字段拒绝。旧流水线替身已补齐真实开发读出的 schema 和来源字段，校验门槛未放宽。本轮没有真实 Codex、训练或确认调用。

后续[真实 Amazon 读出只读验收](../experiments/amazon-feedback-readonly-v1.md)通过：已完成研究的 8 行 seed 42 开发结果可在独立夹具账本完整导出，确认哨兵不进入反馈，非开发 seed 拒绝，原研究不变。这不替代新在线控制器验收。
