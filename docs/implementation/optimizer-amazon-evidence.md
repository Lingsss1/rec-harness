# Amazon 优化器转交的证据准入

Amazon 的 `baseline_execution_evidence` 来自主机协议核验，状态为 `verified`，不是原始训练作业的 `success`。原转交入口误按 success 过滤，导致有效 Amazon 输入不能进入允许引用集合；现已使用实际摘要合同。

开发指标与核验摘要按 `(dataset, seed)` 一一匹配，不依赖列表顺序。拒绝缺失、重复或错位条件、协议不一致、未经核验的状态，以及缺少合法 receipt_hash/manifest_hash 的摘要。旧上下文的 baseline 若为列表，不再被当作通用 runs 字典访问；仍须有完整 Amazon 核验摘要，不能凭旧指标列表推定有效。

只读真实检查保留源事件、配置和状态 hash：

| 来源 | 决策轨迹 | 可引用测量轨迹 |
|---|---:|---:|
| generated-control-e2e-v4：真实执行的 Amazon 适配器、小型合成数据 | 12 | 10 |
| amazon-autoresearch-next-v3：已完成的真实 Amazon NextItNet 研究 | 7 | 5 |

结果在 `/home/liuzhishan/lhr/rec-harness-runs/optimizer-amazon-evidence-v1/acceptance.json` 与 `amazon-real-acceptance.json`。未修改原研究，也没有发送新上下文至 Codex、重训或执行最终测试。这是转交请求准备的实际证据准入验收，不是优化器判断或测量缺口正例。

全量回归 422 项通过（21 项环境相关跳过）；随后补充旧列表 baseline 兼容分支，Amazon 门控定向 4 项全部通过，覆盖乱序合法匹配、错位/重复/无核验拒绝和旧格式兼容。全量日志 `/tmp/rec-harness-amazon-handoff-evidence-tests.log`。
