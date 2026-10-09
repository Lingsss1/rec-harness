# Amazon autoresearch v7：运行记录归档

本归档对应 `amazon-autoresearch-next-v7` 的第 1–10 轮，包含首次 schema 传输失败、全部角色调用与重试，以及控制器 schema 修复记录。仅归档已有记录，没有重新调用 Codex 或训练模型。

## 内容与核验

- 4,142 个原始记录/导出文件，合计 81,958,004 字节。
- `records/`：原目录结构下的控制器、配置、运行状态、角色输入输出、开发及确认阶段聚合结果、训练/审计日志、候选和对照源码。
- `records/controller-source/`：运行时冻结的源码快照；其中历史文档是当时的版本。
- `schema-repair/`：传输修复代码、锁文件、服务探测及修复前记录。
- 每个调用目录的 `request.json` 保存角色 prompt、argv、context 和指令身份；`context.json` 是角色收到的上下文文件；`schema.json` 是输出约束；`stdout.jsonl` 保存 CLI 事件、工具输出及可用的用量计数。
- `memory.sqlite.sql` 是 SQLite 数据库的完整逻辑导出，可用 `sqlite3 memory.sqlite < memory.sqlite.sql` 恢复；不保证恢复后的数据库二进制哈希相同。
- [manifest.json](manifest.json) 记录每个文件的来源、大小、SHA-256 和排除项。标记 `byte-exact` 的文件逐字节保持原样。

## 发布边界

这是一份完整的可发布运行记录，不是可直接恢复所有训练环境的磁盘镜像。凭据、逐用户交互/候选/排名/分数、checkpoint、二进制对象仓库、隔离运行环境及缓存未上传，具体范围见 manifest.exclusions。上下文中的聚合指标、原始路径与内容哈希予以保留；部分路径和哈希因此只在原机器可解析。CLI 服务端隐藏提示词、未记录的内部请求不在本地归档中。

上传前对文本进行凭据模式和敏感字段检查；并验证所有已归档文件的哈希。

## 各轮调用目录

| 轮次 | 调用（含失败/重试） |
|---|---|
| round-1 | [experiment](records/research/rounds/round-1/experiment/request.json) · [researcher](records/research/rounds/round-1/researcher/request.json) · [researcher-transport-archive-5bc774ca24bee75d](records/research/rounds/round-1/researcher-transport-archive-5bc774ca24bee75d/request.json) · [reviewer](records/research/rounds/round-1/reviewer/request.json) · [reviewer-rejected-1](records/research/rounds/round-1/reviewer-rejected-1/request.json) |
| round-2 | [experiment](records/research/rounds/round-2/experiment/request.json) · [experiment-rejected-1](records/research/rounds/round-2/experiment-rejected-1/request.json) · [researcher](records/research/rounds/round-2/researcher/request.json) · [reviewer](records/research/rounds/round-2/reviewer/request.json) · [reviewer-rejected-1](records/research/rounds/round-2/reviewer-rejected-1/request.json) |
| round-3 | [experiment](records/research/rounds/round-3/experiment/request.json) · [researcher](records/research/rounds/round-3/researcher/request.json) · [reviewer](records/research/rounds/round-3/reviewer/request.json) · [reviewer-rejected-1](records/research/rounds/round-3/reviewer-rejected-1/request.json) |
| round-4 | [experiment](records/research/rounds/round-4/experiment/request.json) · [researcher](records/research/rounds/round-4/researcher/request.json) · [reviewer](records/research/rounds/round-4/reviewer/request.json) |
| round-5 | [analyst](records/research/rounds/round-5/analyst/request.json) · [analyst-rejected-1](records/research/rounds/round-5/analyst-rejected-1/request.json) · [experiment](records/research/rounds/round-5/experiment/request.json) · [researcher](records/research/rounds/round-5/researcher/request.json) · [reviewer](records/research/rounds/round-5/reviewer/request.json) · [reviewer-rejected-1](records/research/rounds/round-5/reviewer-rejected-1/request.json) |
| round-6 | [experiment](records/research/rounds/round-6/experiment/request.json) · [researcher](records/research/rounds/round-6/researcher/request.json) · [reviewer](records/research/rounds/round-6/reviewer/request.json) · [reviewer-rejected-1](records/research/rounds/round-6/reviewer-rejected-1/request.json) |
| round-7 | [experiment](records/research/rounds/round-7/experiment/request.json) · [researcher](records/research/rounds/round-7/researcher/request.json) · [reviewer](records/research/rounds/round-7/reviewer/request.json) |
| round-8 | [experiment](records/research/rounds/round-8/experiment/request.json) · [experiment-rejected-1](records/research/rounds/round-8/experiment-rejected-1/request.json) · [researcher](records/research/rounds/round-8/researcher/request.json) · [reviewer](records/research/rounds/round-8/reviewer/request.json) · [reviewer-rejected-1](records/research/rounds/round-8/reviewer-rejected-1/request.json) |
| round-9 | [experiment](records/research/rounds/round-9/experiment/request.json) · [researcher](records/research/rounds/round-9/researcher/request.json) · [reviewer](records/research/rounds/round-9/reviewer/request.json) · [reviewer-rejected-1](records/research/rounds/round-9/reviewer-rejected-1/request.json) |
| round-10 | [analyst](records/research/rounds/round-10/analyst/request.json) · [analyst-rejected-1](records/research/rounds/round-10/analyst-rejected-1/request.json) · [experiment](records/research/rounds/round-10/experiment/request.json) · [researcher](records/research/rounds/round-10/researcher/request.json) · [reviewer](records/research/rounds/round-10/reviewer/request.json) · [reviewer-rejected-1](records/research/rounds/round-10/reviewer-rejected-1/request.json) |

共 45 个角色调用目录，包括传输失败归档。

## 汇总入口

- [运行结果](records/result.json)
- [进度](records/progress.json)
- [调用 token 用量审计](records/token-usage-audit.json)
- [修复后的控制器日志](records/controller-schema-repair.log)
- [原控制器日志](records/controller.log)
