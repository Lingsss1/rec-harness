# 评审可见的实现验证证据

位置幅度参数实验的 lead 报告称 CPU 正确性结果未提供；主机实际上已完成源码准入和 CPU 检查，但评审上下文没有这些证据。后续 evaluation/lead 上下文新增 `implementation_evidence`，由主机从唯一的 `amazon_model_candidate_admitted` 事件生成。

生成前核验研究事件链、协议、候选模型 ID、当前候选源码与归档源码的一致性，以及数值检查和可用的合成训练检查归档。公开摘要只含模型/协议身份、检查状态和证据哈希，不传入主机路径或原始检查报告。缺失旧准入事件时明确返回 unavailable；冲突、源码或报告篡改、失败检查均拒绝构造成功摘要。

这份证据证明既有实现检查完成，不证明机制、推荐收益、训练协议覆盖或最终测试性能。评审仍需根据完整开发指标决定是否推进。evaluation 不获得模型文件；lead 获得候选文件。归档哈希是证据引用，不假装对应文件已挂载给角色。

真实离线验收位于 `/home/liuzhishan/lhr/rec-harness-runs/review-evidence-acceptance-v1/`，使用完成的 `amazon-autoresearch-next-v1` 候选。重复提取相同、源事件头不变；没有新增 Codex 调用、训练或推理。它不构成评审决策效用提升的实测证据。原冻结周期与角色上下文保持原样。

回归覆盖：稳定提取、字段边界、源码变更、协议/模型身份冲突、归档损坏、失败数值门控、旧任务证据缺失，以及自主流程的评审接入和精确恢复。

后续基线上下文另提供 `baseline_execution_evidence`：直接投影已经通过 `audit_job` 的开发收据与 manifest，列出实际 epochs/batches/eval_users/eval_seed/patience/eval_every、fresh 初始化、worker/协议/执行身份和收据哈希。没有完整 manifest 时标注 unavailable；协议、worker 或开发 seed 不匹配时拒绝。该字段仅证明基线主机审计结果，不把它当作尚未执行候选的合规证明，也不传递原始用户、排名或 checkpoint。

真实离线验收 `protocol-context-acceptance-v1/acceptance.json` 覆盖 v2 四类对照，均为每类 1000 验证用户，未新增调用或推理。已运行的冻结 v2 不热更新，后续 campaign 才使用该公开投影。
