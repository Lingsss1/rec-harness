# 图记忆转移保留论文来源

通用图记忆导出先前只保留部分关系和属性，因此新的 Method→Source 关联会丢失，Source 的未核验状态与版本信息也会被过滤。下一轮虽可能见到引用字符串，却无法完整追溯其来源关系和限制。

`memory-policy-v4` 保留 Method→Source 的 DERIVED_FROM 关系，以及 citation_hash、verification_status、source_version、qualification 和明确为空的 retrieved_content_ref。它不打包论文正文或外部 CAS 文档，也不导出不可解析的正文对象引用。

只有导出图涉及这类关系或来源元数据时才使用 v4；普通图继续生成 v3 格式。v2/v3 包继续按原有字段与关系范围校验，不能通过修改 schema 标签夹带 v4 字段。旧已冻结包不会被重写。

来源与方法进入同一闭包，容量不足时整组省略。目标图保留原节点及来源限制，关系标为 source_relation_only；目标状态仍为 unknown/analogy_requires_target_validation。论文引用不是目标实验，不能由转移提高 Claim 的证据等级。重复导入不追加节点或事件。

该变更适用于通用 `export_memory/import_memory` 图包。Amazon 的开发记忆采用独立格式，仍保留设计卡中的原始引用；此处不宣称已扩展该格式的图来源元数据。

13 项相关测试通过，覆盖来源状态/关系的导出导入、目标降级、重复导入、容量闭包和旧格式兼容。使用合成图与虚构引用，不证明论文内容或经验迁移效用；没有新增真实 Codex 调用或训练。

完整核心检查通过：公开边界、语法及 450 项测试（429 项通过、21 项环境条件跳过）。源码变更尚未纳入 v18 安装包。
