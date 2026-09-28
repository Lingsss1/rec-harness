# Amazon 实验文件清单移出图快照

新导入的 Amazon Experiment 不再把完整文件映射直接写入 `artifact_refs`。完整映射以 `amazon-job-files-v1` 清单保存在同一 CAS，图节点只保留 `amazon-job-archive-v1` 描述符：清单引用及文件数。原始文件仍全部归档，checkpoint 的直接引用保持不变。

这修复了未配置 runtime lock 时的实际性能问题：隔离环境的数万条文件引用曾进入每个实验节点，并在后续图快照中反复复制。运行时锁的既有间接引用仍保留；锁外的输入、输出、日志和意外文件也不丢弃。已有事件和图快照不改写，因此该变更防止新节点继续膨胀，不会自动缩小旧数据库。

## 读取与恢复

`resolve_job_archive(artifact_refs, artifacts)` 返回原始完整文件映射，同时支持旧的平铺格式。`restore_job_archive` 和已有 CLI 自动识别新描述符，校验清单 CAS、版本、文件数、路径、所有文件对象以及运行时锁/收据绑定；失败时不会先创建目标目录。

将 Experiment 的 `attrs.artifact_refs` 保存为 JSON 后，使用原 study 的 CAS：

```bash
python -m rec_harness amazon-restore-archive /path/to/artifact-refs.json /path/to/study/artifacts/objects /path/to/new-directory
```

完整恢复仍需原始 CAS；单独复制描述符或清单不构成完整实验包。恢复只还原归档文件，既有规则仍不创建 GPU 设备节点或授予执行资格。

## 验收

- 新增 401 文件的无运行时锁案例，逐文件恢复内容一致；图快照小于 5 KB，重复导入不改事件或图。
- 新旧格式、运行时锁展开、文件数/版本错误、重复路径、目录越界和清单损坏检查通过。
- 归档、运行时、研究循环、合同图、记忆和回放共 41 项相关测试全部通过，零跳过，0.443 秒。
- 真实运行的一个已归档实验有 68,304 条引用，其中 68,262 条为 jail 文件。完整清单往返完全相同，其他属性不变；节点从 17,136,247 字节降到 1,299 字节，约缩小 13,192 倍。证据为 `rec-harness-runs/amazon-compact-archive-v1/acceptance.json`。

真实审计只读源数据库，并在独立目录保存清单，没有复制或重新还原整套运行时。它证明映射无损和节点尺寸变化，不是全流程耗时基准。原工具 v2 继续使用冻结旧代码；本改动没有新增 Codex 调用或训练，晚于 v29 安装包。
