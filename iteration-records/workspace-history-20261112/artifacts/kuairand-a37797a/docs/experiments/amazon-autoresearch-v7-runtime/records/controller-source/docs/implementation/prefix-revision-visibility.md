# 历史前缀中的知识修订提示

离线 Harness 前缀重建现与实时检索共用 `revision_visibility`：撤回状态、已选替代节点和因限额未展示的替代节点数量均按冻结图计算。否则，近期包含修订提示的真实 EvidenceBundle 会在冻结验证套件时因字段缺失无法精确重建。

改变检索限额时重算该提示，但不修改原 Claim 状态，不把 SUPERSEDES 关系当作新结论已获证实，也不读取冻结版本之后的图。冻结套件的源图查询使用只读 Store。

集成回归覆盖：含旧结论及替代节点的实时 bundle 精确重建；增大限额后显示冻结时已有的替代节点；后来添加的修订不进入旧前缀；实际 freeze_suite 入口成功且源事件不写入。

另只读重建此前真实 `harness-evolver-acceptance-v1/prefix/suite.json` 的 4 个角色案例，结果与原 bundle 一致，源文件 hash 未变。收据：`/home/liuzhishan/lhr/rec-harness-runs/prefix-revision-compatibility-v1/acceptance.json`。没有新增 Codex 调用、前缀角色重跑或模型训练。旧格式若无法按现行检索规则精确重建，仍拒绝而不改写历史。

前缀定向 9 项通过；全量 415 项回归通过（21 项环境相关跳过），日志 `/tmp/rec-harness-prefix-revisions-full-tests.log`。
