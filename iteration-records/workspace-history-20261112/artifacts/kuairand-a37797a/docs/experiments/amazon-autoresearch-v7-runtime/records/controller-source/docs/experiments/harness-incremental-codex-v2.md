# 真实 Codex 共享轨迹输入验收

冻结源码 `4a9944cf3357889c970ba22541ab9e658c9e12ad`。运行目录 `/home/liuzhishan/lhr/rec-harness-runs/harness-incremental-codex-v2`；`acceptance.json`、`trace-identity-audit.json` 和 `receipt-manifest.json` 分别保存合同结果、还原后轨迹身份核验和绑定收据。

沿用 v1 的公开历史轨迹及测试基底策略，在发送前确认共享格式完整还原后与 v1 上下文完全相等。此次新增一次隔离 Codex optimizer 调用，实际收到 6 个共享字段及逐轨迹引用。角色产生一个 evidence 提示词补丁，要求后续实验必须另行提案，不能把条件性后续动作算作当前实验的自动延续。

补丁通过已有 schema、轨迹引用及编辑范围合同，合并后保留 knowledge 提示词和检索量 25。不可变提案、完整策略及 6 节点/6 关系图归档通过；只读核验确认图中的引用轨迹 hash 对应完整还原后的原输入。禁止代理调用后重复 propose/sync，7 条事件、图及费用不变。事件头为 `d8fdbf84aa6365f5b344385586c3f3dc250a1f117ae16af71a4a10e4074f9a2e`。

本次角色收据约 72.36 秒，记账 1 BCU；CLI 报告 input_tokens 169241（cached_input_tokens 139008），output_tokens 1291。该数字不等同于底层 API 请求次数或独立计费。两个验收各一次新增角色调用，均无训练、效用验证或晋升。

本次提出的补丁与 v1 不同，不能据此声称压缩前后决策相同或研究收益等价。此次结果证明真实角色可使用共享表示并完成增量补丁、归档、图身份和恢复合同。原始轨迹与研究结果未改写。
