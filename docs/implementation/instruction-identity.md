# 角色实际指令的版本身份

新 CodexAgent 与 JailedCodexAgent 请求均在 `request.json` 记录 `role-instruction-identity-v1`：实际发送的完整 prompt 和响应 schema 的 hash。hash 使用项目统一 canonical JSON digest，因此是字符串/对象身份，不是 `prompt.txt` 文件字节 hash。完整提示包含该角色的冻结覆盖提示和搜索约束。

决策轨迹只在归档请求存在这份记录时增加 `instruction_identity`。提取器验证请求和 schema 的 CAS 身份，重新计算摘要，并附上原归档引用；内容不匹配或缺少必要归档时拒绝。旧请求不补造记录，保持旧轨迹结构及其已有 hash。

这不是 Skill 执行追踪。两个传输器没有显式向任务提供 Skill 文件；CLI 自动发现或执行哪些 Skill 仍未记录。不能因为仓库包含 Skill 就声称调用使用了它，也不能从提示 hash 推断服务端精确模型版本。方案要求的完整 Skill 身份追踪仍保留未验收状态。

验证包含实际传输参数与归档提示一致、覆盖提示参与摘要、改写提示而保留旧摘要被拒绝、旧请求不补写，以及完整小型研究的六条轨迹携带可验证的原始指令引用。所有角色执行在本轮测试中使用替身，没有新 Codex 调用。

后续一次[真实隔离 tool 调用](../experiments/tool-authoring-codex-v1.md)已核验实际请求中的完整 prompt/schema 身份及原档案。它补充传输证据，不提供 CLI 自动 Skill 执行追踪。

后续[三组真实在线研究](../experiments/tool-authoring-online-v1.md)的 18 条实际轨迹也已核验指令身份、归档引用及主机反馈；终态精确恢复未追加调用或费用。
