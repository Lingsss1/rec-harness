# 优化器轨迹上下文无损共享

新 Harness 演化配置默认冻结 `compact_optimizer_traces: true`。当多条决策轨迹的 visible_state 顶层字段完全相同且足够大时，只在 `shared_visible_values` 保存一份，轨迹以 `visible_state_shared_fields` 指向内容 hash。内联字段、引用字段、角色输出及其他上下文都保留；上下文带显式还原说明。不共享不同内容，不摘要或截断证据，不改变每条轨迹原来可见的字段集合。共享后不更小时返回原格式。

初始化通过 `expand_traces` 校验逐字段还原与原上下文一致。共享值 hash、字段冲突、缺失引用、额外值以及整体原始 hash 都被检查。已有冻结 campaign 直接恢复原输入，不重写旧角色请求。原上下文使用保留字段名时不启用压缩。图中的轨迹 hash 仍基于还原后的完整轨迹。

真实只读验收使用上一轮 `harness-incremental-codex-v1` 的实际输入：原上下文 210,942 字节，共享 6 个重复字段后 168,201 字节，减少 42,741 字节（约 20.3%）；完整还原与原值相等，原文件 hash 未变。结果在 `/home/liuzhishan/lhr/rec-harness-runs/optimizer-trace-sharing-v1/acceptance.json`。

本轮没有新增 Codex 调用。该结果是 JSON UTF-8 字节减少，不是实际 token、计费或研究收益改善；尚未据此宣称真实优化器使用共享表示的决策等价性。此项晚于 v14 安装包。

后续[真实 Codex v2](../experiments/harness-incremental-codex-v2.md)已用一次新角色调用验证共享表示的实际交付、提案/策略合并、归档图身份及零新增调用恢复。补丁内容与未共享版本不同，因此仍不宣称决策等价或研究效用等价。

全量回归 413 项通过（21 项环境相关跳过）。随后对图中的还原后轨迹身份补充检查，共享表示定向 4 项及项目定向 11 项均通过，覆盖无损、篡改/缺失引用拒绝和连续演化入口。
