# Codex 输出 schema 传输兼容性

2026-10-04 修复：`efecae1` 的首个真实 researcher 请求被服务以
`400 invalid_json_schema` 拒绝。本地 JSON 校验通过并不能证明服务支持该 schema。

`bound_schema` 继续保存主机的精确合同。新增 `service_schema` 将整数组 enum
转换为元素 enum 和长度上下界，将 exclusive 数值界限转换为非 exclusive 界限。
Codex 只接收兼容投影；响应返回后仍按原合同验证数组顺序、完整取值及严格预算
范围。不会重写角色返回的证据或把服务投影当作全部业务验收。

另一个独立错误是基础 schema 字段共享可变对象：绑定 researcher 的图引用时，
约束会污染 affected_components、alternative_explanations 等自由文本字段。
绑定前通过 JSON 往返复制切断别名；回归测试覆盖引用约束隔离及原 schema 不变。
归档 schema 和 instruction identity 均绑定实际发送的服务投影。

使用 gpt-6-sol / medium，将 researcher、experiment、reviewer、analyst 的合成
schema 合并为一次真实请求，服务返回 turn.completed。该检查只证明接口接受
这组 schema；合成响应的零预算等内容仍不能通过真实研究合同，不能作为研究证据。
没有逐项远端消融，因此不单独断言整数组 enum 或 exclusive 界限中的哪一个是
400 的唯一原因。

已有 v7 任务的修复采用显式注入的 schema 传输适配器，记录适配器与修复源码哈希，
保留原冻结训练/评分包与所有失败费用；通过原控制器 pre-admission reopen 流程
恢复第一轮。新的源码 checkout 直接使用修复后的 Codex agent 实现。
