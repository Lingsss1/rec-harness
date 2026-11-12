# 真实 Codex 增量 Harness 提案验收

冻结源码 `f4ae86a1d38317e5c42decd76e7e885877388d5b`。运行目录 `/home/liuzhishan/lhr/rec-harness-runs/harness-incremental-codex-v1`，`acceptance.json` 保存结果，`receipt-manifest.json` 绑定源码身份、输入配置、脚本、角色收据、提案、实际策略和事件导出。

复用此前真实 Harness 开发所用的公开可见状态轨迹，仅新增一次隔离 Codex optimizer 角色调用。输入明确说明当前基底是验收用策略，而非此前晋升版本：检索量 25，knowledge 提示词要求保留临时结论和反证。没有传递确认/最终分数、原始交互或 checkpoint。

Codex 生成只修改 evaluation 提示词的最小补丁，要求区分“本次评审未见源码”和“全局没有源码”，不能把 artifact hash 或提案文字当成已检查实现。此诊断仍是待验证假设；本验收不判定其科学正确或效用。

主机按声明范围验证补丁，在基底上合并，实际 policy.json 保留 knowledge 提示词及检索量 25。提案进入 CAS；事件图包含 6 个节点、6 条边，包括失败假设及冻结轨迹关联。随后禁止代理调用并重复 propose/sync，结果、7 条事件、图版本和成本均不变。事件头 `80e2cb7e143564b82ba807c05844a34655a5742080310ce14deaaac34f27539b`。

实际新增角色调用 1 次，收据耗时约 71.81 秒；控制器记账 1 BCU。CLI 报告 input_tokens 184487，其中 cached_input_tokens 148480，output_tokens 1127；这些是该 CLI 收据值，不换算服务费用，也不等同于底层 API 请求次数。

没有运行前缀、在线开发、确认验证器，没有模型训练或策略晋升。结果证明真实角色遵守增量补丁交付、归档、图投影和恢复合同，不证明 Harness 收益。原始历史研究未继续执行。
