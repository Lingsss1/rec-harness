# 结构化插入合同端到端验收 v3

状态：已启动，尚未完成终态验收。运行目录 `/home/liuzhishan/lhr/rec-harness-runs/generated-control-e2e-v3`，冻结源码 `896a8e5`。使用与 v2 相同的 32 用户合成数据和 2 epoch × 2 batch 训练协议，继承 v2 全部 326.55892564356327 GPU 秒历史成本。

本轮新增验收新 campaign 默认启用的方法范围合同：真实 hypothesis 在生成代码之前声明允许改动的方法，experiment 同一次调用生成候选和新对照，主机在角色输出检查与原子准入中分别检查每份源码没有越界改动。评估和负责人必须收到两份模型各自的方法范围核验、操作图摘要和整段序列语义检查。预期仍为六角色调用，无新增评审角色。

终态审计保留 v2 的三源码身份、开发对比、独立评分、归档/CAS 与零新增执行恢复检查，并新增上述合同和角色实际可见证据字段断言。结果以 `controller-result.json`、`terminal-acceptance.json`、`watch-result.json` 为准；启动不构成通过，合成结果不证明真实推荐机制或性能收益。未改写此前 campaign 或最终测试配置。
