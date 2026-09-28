# 两轮生成对照端到端验收 v4

本轮验证连续 autoresearch 的轮次衔接及最近新增证据字段，不评价真实推荐收益。目录为 `/home/liuzhishan/lhr/rec-harness-runs/generated-control-e2e-v4`，冻结源码 `27857270ec01448cd930fc0f8e605f252bf661d8`。

沿用 v3 合成数据和基线，执行配置重新冻结。共两轮，每轮由 Codex 生成候选及一个机制对照，训练/独立复评分后完成 evaluation、lead、knowledge。正常十二次串行角色调用；完整保留重试及失败费用。基线一次、每轮两份新模型，共五个开发训练任务；只有通过开发门槛才按原规则释放确认任务。继承 v3 已发生 GPU 成本，旧档案不改写。

终态验收脚本检查两轮角色与周期归档、实际角色收到的训练前后合同、长期图中的非独立合同观测、第二轮的上一轮记忆/缺口复查、由第一轮决定的 incumbent、独立排名一致性，以及禁止新增调用/训练/评分后的精确恢复。

当前状态：已启动。控制器 PID 40976、start_ticks 5892124878；观察器 PID 41036、start_ticks 5892128180。两者 boot_id 为 `17e9114f-f060-4b1d-b1dc-db95f3d07e7d`，启动时身份已核验。当前进度文件为 freeze_execution，尚无完整训练、角色周期或终态通过证据。终态以 `controller-result.json`、`watch-result.json` 和 `terminal-acceptance.json` 为准，不能以启动记录认定完成。
