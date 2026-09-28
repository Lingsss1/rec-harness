# 连续 Harness 补丁

新项目冻结 `inherit_harness_policy: true`，从生成诊断轨迹的模型 campaign 读取当前 `research_policy`，作为后继 Harness 候选的 `base_policy`。该策略及合并规则进入冻结上下文，和原轨迹一起交给优化器。旧项目没有此配置时继续使用原冻结输入语义。

`patch_diff` 仍只包含优化器声明范围内的修改，先按原规则校验修改范围，再应用到基底策略：给出的顶层字段替换旧值；`prompts` 按角色合并；省略的字段或角色保留。合并后的完整策略重新校验，只允许已有提示词、检索量和固定路由范围，不开放稳定控制内核修改。

不可变提案保留原始最小补丁，`policy.json` 保存合并后的实际策略。前缀、开发和确认验证器以及后继模型使用的是这份完整策略。第二个只改负责人提示词的补丁不会让第一个补丁的检索量或既有 evidence 提示词恢复默认。

五阶段项目测试覆盖模型→Harness 1→模型→Harness 2→模型：逐次核对优化器收到的基底、最终策略和归档补丁，检查五个子阶段的完整性及终态恢复不重复调用。测试使用脚本角色和固定验证报告，证明工程合并/调度行为，不构成新的 Harness 效用证据。本项晚于 v14 安装包。

项目定向 9 项通过；完整源码回归 408 项通过（21 项环境相关跳过），日志为 `/tmp/rec-harness-successive-harness-tests.log` 和 `/tmp/rec-harness-successive-harness-full-tests.log`。
